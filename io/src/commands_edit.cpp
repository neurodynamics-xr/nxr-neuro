// `add dictionary`, `edit dictionary`, `edit manifold` and `edit operator` (a projector's status) (docs/specs/2026-11-20-data-layer-queries.md, 3.5 and 3.6): a dictionary of codes made from a list of entries, an entry or the whole list
// changed, a manifold's label, description and primary flag changed. Each is ONE unit (task::Unit): everything it changes, or nothing. An edit changes a node's description, never its arrays.
#include <algorithm>
#include <cctype>
#include <array>
#include <fstream>
#include <set>
#include <sstream>

#include "commands_open.h"
#include "nxr/io/errors.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

namespace nxr::io::task {

using namespace commands_detail;

namespace {

// a dataset opened, its unit open on `subject` (named, or in a one-database dataset any one subject: a dataset's own dictionary belongs to none, the unit needs a subject to write through),
// committed when `body` returns
template <class F>
CommandResult in_unit_of(const std::string& dataset, const std::optional<std::string>& subject, const std::vector<std::string>& templates, const std::string& created_by, const Json& producer,
                         const Progress& progress, F&& body) {
  DatasetWriter ds = open_dataset(dataset, subject, OpenOptions{}, templates, progress);
  std::string name;
  if (subject) {
    name = *subject;
  } else {
    const auto r = ds.db().one("SELECT name FROM subject ORDER BY name LIMIT 1");
    if (!r) throw Error(code::no_such_row, "the dataset holds no subject");
    name = text(*r, "name");
  }
  OpenSubjectOptions oo;
  oo.created_by = created_by;
  SubjectWriter sub = SubjectWriter::open(ds, name, oo);
  const Subject ts = Subject::of(sub);
  CommandResult r;
  {
    Unit u = ts.unit(By{created_by, producer});
    r.result = body(ts, u, sub);
    u.commit();
  }
  return r;
}

// a colour as #rrggbb: [r, g, b]
std::array<int, 3> colour_of(const std::string& text, const std::string& where) {
  const bool ok = text.size() == 7 && text[0] == '#' && std::all_of(text.begin() + 1, text.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
  if (!ok) throw Refused(where + ": the colour '" + text + "' is not #rrggbb (six hexadecimal digits)");
  return {std::stoi(text.substr(1, 2), nullptr, 16), std::stoi(text.substr(3, 2), nullptr, 16), std::stoi(text.substr(5, 2), nullptr, 16)};
}
std::string colour_text(const std::array<int, 3>& c) {
  static const char* hex = "0123456789abcdef";
  std::string t = "#";
  for (const int x : c) { t += hex[(x >> 4) & 15]; t += hex[x & 15]; }
  return t;
}

// the entries as given: JSON text (a list), or the path of a JSON file
Json entries_json(const std::string& given, const std::string& option) {
  std::string text = given;
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos || (text[first] != '[' && text[first] != '{')) {
    std::ifstream in(given, std::ios::binary);
    if (!in) throw Refused(option + " " + given + ": not a JSON list, and no such file");
    std::stringstream ss;
    ss << in.rdbuf();
    text = ss.str();
  }
  Json j = Json::parse(text, nullptr, false);
  if (j.is_discarded()) throw Refused(option + " " + given + ": not JSON");
  if (j.is_object() && j.contains("entries")) j = j["entries"];
  if (!j.is_array() || j.empty()) throw Refused(option + ": a non-empty list of entries, each {code, name, label, colour, description, params}");
  return j;
}

std::vector<DictionaryOut::Entry> entries_of(const Json& list, const std::string& option) {
  std::vector<DictionaryOut::Entry> out;
  std::set<std::int64_t> codes;
  std::size_t at = 0;
  for (const Json& e : list) {
    const std::string where = option + " entry " + std::to_string(at++);
    if (!e.is_object()) throw Refused(where + ": an entry is an object {code, name, label, colour, description, params}");
    for (const auto& [k, v] : e.items()) {
      (void)v;
      if (k != "code" && k != "name" && k != "label" && k != "colour" && k != "description" && k != "params") throw Refused(where + ": '" + k + "' is not a field of an entry (code, name, label, colour, description, params)");
    }
    DictionaryOut::Entry x;
    if (!e.contains("code") || !e["code"].is_number_integer()) throw Refused(where + ": a code, a whole number");
    x.code = e["code"].get<std::int64_t>();
    if (!e.contains("name") || !e["name"].is_string() || e["name"].get<std::string>().empty()) throw Refused(where + ": a name, the text the code stands for");
    x.name = e["name"].get<std::string>();
    for (const char* k : {"label", "description", "colour"})
      if (e.contains(k) && !e[k].is_null() && !e[k].is_string()) throw Refused(where + ": " + k + " is text");
    if (e.contains("label") && e["label"].is_string()) x.label = e["label"].get<std::string>();
    if (e.contains("description") && e["description"].is_string()) x.description = e["description"].get<std::string>();
    if (e.contains("colour") && e["colour"].is_string()) x.colour = colour_of(e["colour"].get<std::string>(), where);
    if (e.contains("params") && !e["params"].is_null()) x.params = e["params"];
    if (!codes.insert(x.code).second) throw Refused(where + ": the code " + std::to_string(x.code) + " is used by an earlier entry (codes are unique)");
    out.push_back(std::move(x));
  }
  return out;
}

Json producer_for(const std::string& command, const std::string& created_by) { return Json{{"command", command}, {"program", created_by}}; }

}  // namespace

CommandResult add_dictionary(const AddDictionaryRequest& rq, const Progress& progress) {
  if (rq.name.empty()) throw Refused("--name: the name the dictionary is stored under");
  if (rq.scope != "dataset" && rq.scope != "subject") throw Refused("--scope " + rq.scope + " (dataset or subject)");
  if (rq.scope == "subject" && !rq.subject) throw Refused("--scope subject names the subject the dictionary belongs to: --subject S");
  DictionaryOut d;
  d.name = rq.name;
  d.scope = rq.scope;
  d.description = rq.description;
  d.entries = entries_of(entries_json(rq.entries, "--entries"), "--entries");
  return in_unit_of(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_for("add dictionary", rq.created_by), progress, [&](const Subject&, Unit& u, SubjectWriter&) {
    const std::string id = u.dictionary(d);
    Json entries = Json::array();
    for (const auto& e : d.entries) entries.push_back(Json{{"code", e.code}, {"name", e.name}});
    return Json{{"class", "dictionary"}, {"name", d.name}, {"id", id}, {"scope", d.scope}, {"entries", entries}};
  });
}

CommandResult edit_dictionary(const EditDictionaryRequest& rq, const Progress& progress) {
  const bool one = rq.entry.has_value();
  const bool edits_entry = rq.name_of || rq.label || rq.colour || rq.description || rq.params;
  if (rq.entries) {
    if (one || edits_entry) throw Refused("--entries replaces the whole list: leave out --entry and the entry's own options");
    if (!rq.replace) throw Refused("--entries with --replace: the list given replaces every entry (the codes a selection reads must stay)");
  } else {
    if (rq.replace) throw Refused("--replace goes with --entries");
    if (!one) throw Refused("name the entry to change (--entry CODE with --name-of, --label, --colour, --description or --params), or give --entries FILE.json --replace");
    if (!edits_entry) throw Refused("--entry " + std::to_string(*rq.entry) + ": nothing to change (--name-of, --label, --colour, --description, --params)");
  }
  Unit::EntryEdit edit;
  edit.name = rq.name_of;
  edit.label = rq.label;
  edit.description = rq.description;
  if (rq.colour) edit.colour = colour_of(*rq.colour, "--colour");
  if (rq.params) {
    edit.params = Json::parse(*rq.params, nullptr, false);
    if (edit.params.is_discarded()) throw Refused("--params: not JSON (" + rq.params->substr(0, 40) + ")");
  }
  std::vector<DictionaryOut::Entry> entries;
  if (rq.entries) entries = entries_of(entries_json(*rq.entries, "--entries"), "--entries");
  return in_unit_of(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_for("edit dictionary", rq.created_by), progress, [&](const Subject&, Unit& u, SubjectWriter& sub) {
    Database& db = sub.db();
    // the subject's own dictionary of that name first, else the dataset's; the app's are the program's
    std::optional<Row> have;
    if (rq.subject) have = db.one("SELECT id, scope FROM dictionary WHERE scope = 'subject' AND subject_id = ? AND name = ?", {V(sub.id()), V(rq.name)});
    if (!have) have = db.one("SELECT id, scope FROM dictionary WHERE scope = 'dataset' AND dataset_id = ? AND name = ?", {V(sub.dataset().id()), V(rq.name)});
    if (!have) {
      if (db.one("SELECT id FROM dictionary WHERE scope IN ('app', 'user') AND name = ?", {V(rq.name)}))
        throw Refused("'" + rq.name + "' is an app dictionary, the program's own: make a dictionary of the dataset or the subject (add dictionary) to change names or colours");
      throw Error(code::no_such_row, "no dictionary named '" + rq.name + "' in the subject or the dataset");
    }
    const std::string id = text(*have, "id");
    if (rq.entries) {
      u.replace_dictionary_entries(id, entries);
    } else {
      u.edit_dictionary_entry(id, *rq.entry, edit);
    }
    Json out = Json{{"class", "dictionary"}, {"name", rq.name}, {"id", id}, {"scope", text(*have, "scope")}};
    Json shown = Json::array();
    for (const Row& e : db.all("SELECT code, name, color_json FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal, code", {V(id)})) {
      Json x = Json{{"code", integer(e, "code")}, {"name", text(e, "name")}};
      if (const auto c = opt_text(e, "color_json")) {
        const Json cj = Json::parse(*c, nullptr, false);
        if (cj.is_array() && cj.size() == 3) x["colour"] = colour_text({cj[0].get<int>(), cj[1].get<int>(), cj[2].get<int>()});
      }
      shown.push_back(std::move(x));
    }
    out["entries"] = shown;
    return out;
  });
}

CommandResult edit_manifold(const EditManifoldRequest& rq, const Progress& progress) {
  std::optional<bool> primary;
  if (rq.primary) {
    if (*rq.primary == "yes") primary = true;
    else if (*rq.primary == "no") primary = false;
    else throw Refused("--primary " + *rq.primary + " (yes or no)");
  }
  if (!primary && !rq.label && !rq.description) throw Refused("nothing to change: give --primary yes|no, --label or --description");
  return in_unit_of(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_for("edit manifold", rq.created_by), progress, [&](const Subject& ts, Unit& u, SubjectWriter& sub) {
    const auto m = ts.by_id_or_name(Cls::manifold, rq.name);
    if (!m) throw Error(code::no_such_row, "the subject holds no manifold named '" + rq.name + "'");
    if (primary && *primary && m->type != std::optional<std::string>("surface")) throw Refused("'" + m->name + "' is a " + m->type.value_or("manifold") + ": only a surface is primary");
    u.edit_manifold(m->id, primary, rq.label, rq.description);
    const auto row = sub.db().one("SELECT is_primary, label, comment FROM manifold WHERE id = ?", {V(m->id)});
    Json out = Json{{"class", "manifold"}, {"name", m->name}, {"id", m->id}, {"primary", row && integer(*row, "is_primary") != 0}};
    if (row && !is_null(*row, "label")) out["label"] = text(*row, "label");
    if (row && !is_null(*row, "comment")) out["description"] = text(*row, "comment");
    return out;
  });
}

CommandResult edit_operator(const EditOperatorRequest& rq, const Progress& progress) {
  if (!rq.status) throw Refused("nothing to change: give --status 0|1 (a projector's status: 0 inactive, 1 active)");
  if (*rq.status != "0" && *rq.status != "1") throw Refused("--status " + *rq.status + " (0 inactive or 1 active)");
  const int status = *rq.status == "1" ? 1 : 0;
  return in_unit_of(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_for("edit operator", rq.created_by), progress, [&](const Subject& ts, Unit& u, SubjectWriter&) {
    const auto o = ts.by_id_or_name(Cls::operator_, rq.name);
    if (!o) throw Error(code::no_such_row, "the subject holds no operator named '" + rq.name + "'");
    if (o->kind != std::optional<std::string>("projector"))
      throw Refused("'" + o->name + "' is an operator of kind " + o->kind.value_or("?") + ": --status applies to projector operators only (the SSP and ICA projectors on a recording's channels)");
    const Json entry = u.edit_projector_status(o->id, status, rq.note, rq.by);
    return Json{{"class", "operator"}, {"name", o->name}, {"id", o->id}, {"kind", "projector"}, {"status", status}, {"changed", !entry.is_null()}, {"edit", entry}};
  });
}

}  // namespace nxr::io::task
