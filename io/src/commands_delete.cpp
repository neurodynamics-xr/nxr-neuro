// `delete subject --node CLASS:NAME` (nxr/io/task/commands.h; docs/specs/2026-11-20-data-layer-queries.md, 3.7): one node of a subject removed with what it owns, refused while other nodes depend on it.
//
// A node is named by its class and its name (names repeat across classes). What it OWNS goes with it: a manifold's geometry and topology, a field's arrays, a selection's members and weights array,
// the measurements made of a field, an operator's by-products (its eigenvalues); the arrays go through io's own removal (writer.h `remove_node`: the paths only this row named). What DEPENDS on it
// (another node made from it, lying on it, or reading it) is never removed with it: the delete is refused, naming those nodes. The operator a deleted field was made by goes too, when nothing
// else uses it.
#include <algorithm>
#include <set>

#include "commands_open.h"
#include "nxr/io/errors.h"
#include "nxr/io/task/dataset.h"
#include "nxr/io/task/handle.h"
#include "nxr/io/writer.h"

namespace nxr::io::task {

using namespace commands_detail;

namespace {

const std::vector<std::string> kClasses = {"manifold", "field", "selection", "operator", "statistic", "dictionary"};

bool has_table(const Database& db, const std::string& t) { return db.one("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", {V(t)}).has_value(); }

// the nodes that depend on the node `id` of `cls`, each "class NAME" (qualified by its subject when it is another's)
std::vector<std::string> dependants(const Database& db, const std::string& cls, const std::string& id, const std::string& subject_id) {
  std::vector<std::string> out;
  std::set<std::string> seen;
  auto add = [&](const std::string& label, const std::string& sql, int binds) {
    const std::vector<Value> params(static_cast<std::size_t>(binds), V(id));
    for (const Row& r : db.all(sql, params)) {
      std::string n = label + " " + text(r, "name");
      if (r.count("subject") && text(r, "subject_id") != subject_id) n += " (subject " + text(r, "subject") + ")";
      if (seen.insert(n).second) out.push_back(n);
    }
  };
  const std::string axes_manifold = " AND EXISTS (SELECT 1 FROM json_each(x.axes_json) a WHERE json_extract(a.value, '$.manifold_id') = ?)";
  const std::string axes_selection = " EXISTS (SELECT 1 FROM json_each(x.axes_json) a WHERE json_extract(a.value, '$.selection_id') = ?)";
  const bool stats = has_table(db, "statistic");
  if (cls == "manifold") {
    add("field", "SELECT x.name AS name FROM field x WHERE x.manifold_id = ?", 1);
    add("selection", "SELECT x.name AS name FROM selection x WHERE x.manifold_id = ? AND x.member_of_id IS NULL", 1);
    add("operator", "SELECT x.name AS name FROM operator x WHERE x.from_manifold_id = ? OR x.to_manifold_id = ?", 2);
    if (stats) add("statistic", "SELECT x.name AS name FROM statistic x WHERE 1 = 1" + axes_manifold, 1);
    add("manifold", "SELECT x.name AS name FROM manifold x WHERE x.derived_from_id = ? OR x.component_of_id = ?", 2);
    // a product of it (a line over time, a surface over time), and what lies on a product
    add("manifold", "SELECT p.name AS name FROM manifold p JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE d.manifold_id = ? AND p.id <> ?", 2);
    // (the second binding of the next two is the same id)
    add("field", "SELECT f.name AS name FROM field f JOIN manifold p ON p.id = f.manifold_id JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE d.manifold_id = ? AND p.id <> ?", 2);
    add("selection", "SELECT s.name AS name FROM selection s JOIN manifold p ON p.id = s.manifold_id JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE d.manifold_id = ? AND p.id <> ? AND s.member_of_id IS NULL", 2);
  } else if (cls == "field") {
    add("field", "SELECT x.name AS name FROM field x WHERE x.derived_from_id = ? OR x.of_field_id = ?", 2);
    add("selection", "SELECT x.name AS name FROM selection x WHERE x.of_field_id = ? AND x.member_of_id IS NULL", 1);
    if (stats) add("statistic", "SELECT x.name AS name FROM statistic x WHERE x.of_field_id = ?", 1);
    add("field", "SELECT x.name AS name, s.name AS subject, x.subject_id AS subject_id FROM field_contribution c JOIN field x ON x.id = c.field_id JOIN subject s ON s.id = x.subject_id WHERE c.source_field_id = ?", 1);
  } else if (cls == "selection") {
    add("operator", "SELECT x.name AS name FROM operator x WHERE x.from_selection_id = ? OR x.to_selection_id = ?", 2);
    add("operator", "SELECT x.name AS name FROM operator_factor f JOIN operator x ON x.id = f.operator_id WHERE f.factor_selection_id = ?", 1);
    if (stats) add("statistic", "SELECT x.name AS name FROM statistic x WHERE x.partition_selection_id = ? OR" + axes_selection, 2);
    add("selection", "SELECT x.name AS name FROM selection_factor f JOIN selection x ON x.id = f.selection_id WHERE f.factor_selection_id = ?", 1);
  } else if (cls == "operator") {
    add("field", "SELECT x.name AS name FROM field x WHERE x.by_operator_id = ?", 1);
    add("operator", "SELECT x.name AS name FROM operator x WHERE x.derived_from_id = ? OR x.inverts_id = ?", 2);
    add("operator", "SELECT x.name AS name FROM operator_factor f JOIN operator x ON x.id = f.operator_id WHERE f.factor_operator_id = ?", 1);
  } else if (cls == "statistic") {
    add("statistic", "SELECT x.name AS name FROM statistic x WHERE x.of_statistic_id = ?", 1);
    add("operator", "SELECT x.name AS name FROM operator x WHERE x.of_statistic_id = ?", 1);
  } else if (cls == "dictionary") {
    add("selection", "SELECT x.name AS name FROM selection x WHERE x.dictionary_id = ?", 1);
  }
  return out;
}

struct Target {
  std::string cls, id, name;
  bool by_name = false;
};

std::string list_of(const std::vector<std::string>& v) {
  std::string t;
  for (const auto& x : v) t += (t.empty() ? "" : ", ") + x;
  return t;
}

// the node `NAME` of class `cls` in the subject (a dictionary: the subject's own, by name)
std::optional<Target> by_class_name(SubjectWriter& sub, const std::string& cls, const std::string& name) {
  if (cls == "dictionary") {
    if (const auto r = sub.db().one("SELECT id, name FROM dictionary WHERE scope = 'subject' AND subject_id = ? AND name = ?", {V(sub.id()), V(name)})) return Target{cls, text(*r, "id"), text(*r, "name"), true};
    if (const auto r = sub.db().one("SELECT scope FROM dictionary WHERE name = ? AND scope <> 'subject'", {V(name)}))
      throw Refused("dictionary " + name + " is " + text(*r, "scope") + "-scoped: it is not a node of subject " + sub.name() + " (a subject's own dictionaries are deleted here)");
    return std::nullopt;
  }
  if (const auto r = sub.by_name(cls, name)) return Target{cls, text(*r, "id"), text(*r, "name"), true};
  return std::nullopt;
}

Cls cls_of(const std::string& cls) {
  if (cls == "manifold") return Cls::manifold;
  if (cls == "field") return Cls::field;
  if (cls == "operator") return Cls::operator_;
  if (cls == "selection") return Cls::selection;
  return Cls::statistic;
}

}  // namespace

CommandResult delete_node(DatasetWriter& ds, const std::string& subject, const std::string& node, const std::string& created_by, const std::optional<std::string>& by_author) {
  OpenSubjectOptions oo;
  oo.created_by = created_by;
  SubjectWriter sub = SubjectWriter::open(ds, subject, oo);
  const Dataset td = Dataset::of(ds);
  Database& db = sub.db();
  std::optional<Target> t;
  std::string by;
  const auto colon = node.find(':');
  if (colon != std::string::npos && std::find(kClasses.begin(), kClasses.end(), node.substr(0, colon)) != kClasses.end()) {
    const std::string cls = node.substr(0, colon), name = node.substr(colon + 1);
    t = by_class_name(sub, cls, name);
    if (!t) throw Error(code::no_such_row, node + ": subject " + subject + " holds no " + cls + " named '" + name + "'");
    by = "class";
  } else {
    // a store path first (unique by construction), then a name; a name several classes hold is refused, naming the CLASS:NAME forms
    for (const char* c : {"manifold", "field", "operator", "selection"})
      if (const auto row = sub.find(c, node)) { t = Target{c, text(*row, "id"), text(*row, "name"), false}; by = "path"; break; }
    if (!t) {
      std::vector<Target> hits;
      for (const auto& c : kClasses) if (const auto h = by_class_name(sub, c, node)) hits.push_back(*h);
      if (hits.size() > 1) {
        std::string forms;
        for (const auto& h : hits) forms += (forms.empty() ? "" : ", ") + h.cls + ":" + node;
        throw Error(code::ambiguous, node + ": subject " + subject + " holds several nodes with this name; name the class: " + forms);
      }
      if (hits.empty()) throw Error(code::no_such_row, node + ": no node of " + subject + " by CLASS:NAME, store path or name");
      t = hits[0];
      by = "name";
    }
  }
  if (by_author) {   // only what this author saved: the node's own `created_by` (`add <class> --by`)
    const auto row = db.one("SELECT created_by FROM " + t->cls + " WHERE id = ?", {V(t->id)});
    const std::string author = row && !is_null(*row, "created_by") ? text(*row, "created_by") : std::string();
    if (author != *by_author)
      throw Error(code::invalid_argument, t->cls + " " + t->name + " was saved by " + (author.empty() ? std::string("no recorded author") : "'" + author + "'") + ", not '" + *by_author +
                                              "': --by deletes only the nodes that author saved (" + t->cls + " " + t->name + " was not deleted)");
  }
  const std::vector<std::string> deps = dependants(db, t->cls, t->id, sub.id());
  if (!deps.empty())
    throw Error(code::invalid_argument, t->cls + " " + t->name + " is used by " + list_of(deps) + ": delete those first (" + t->cls + " " + t->name + " was not deleted)");
  if (t->cls == "dictionary") {
    db.tx([&] { delete_row(db, "dictionary", t->id); });
  } else {
    td.remove(cls_of(t->cls), t->id);
  }
  CommandResult r;
  r.result = Json{{"removed", t->name}, {"class", t->cls}, {"table", t->cls}, {"by", by}};
  // the operator a field was made by (an inverse kernel, a forward model) is never removed with it: it is a stored node of its own, deleted only when named
  return r;
}

}  // namespace nxr::io::task
