#include "nxr/io/participants.h"

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

#include "nxr/io/catalogue.h"
#include "nxr/io/database.h"
#include "nxr/io/writer_lock.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {
const char* const kParticipantTables =
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;\n"
    "CREATE TABLE IF NOT EXISTS variable_dictionary (\n"
    "  id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, name TEXT NOT NULL, type TEXT NOT NULL CHECK (type IN ('number', 'string', 'boolean')),\n"
    "  unit TEXT, description TEXT, levels_json TEXT, UNIQUE (dataset_id, name)\n"
    ") STRICT;\n"
    "CREATE TABLE IF NOT EXISTS subject_variable (\n"
    "  subject_id TEXT NOT NULL, variable_dictionary_id TEXT NOT NULL REFERENCES variable_dictionary (id) ON DELETE CASCADE,\n"
    "  value_num REAL, value_text TEXT, PRIMARY KEY (subject_id, variable_dictionary_id), CHECK ((value_num IS NULL) <> (value_text IS NULL))\n"
    ") STRICT;\n";

Error refuse(const std::string& m) { return Error(code::invalid_argument, m); }

VariableType type_of(const std::string& t) { return t == "number" ? VariableType::Number : t == "boolean" ? VariableType::Boolean : VariableType::String; }

std::vector<std::string> split_tab(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) { if (c == '\t') { out.push_back(cur); cur.clear(); } else cur += c; }
  out.push_back(cur);
  return out;
}

bool parse_number(const std::string& s, double& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end && *end == '\0';
}

bool is_missing(const std::string& s) { return s.empty() || s == "n/a" || s == "N/A"; }
}  // namespace

std::vector<VariableInfo> participant_variables(const fs::path& folder) {
  std::vector<VariableInfo> out;
  if (!fs::is_regular_file(folder / kParticipantsFile)) return out;
  Database db = Database::open_plain(folder / kParticipantsFile, false, true);
  for (const auto& r : db.all("SELECT * FROM variable_dictionary ORDER BY rowid")) {
    VariableInfo v;
    v.name = text(r, "name");
    v.type = type_of(text(r, "type"));
    v.unit = opt_text(r, "unit");
    v.description = opt_text(r, "description");
    if (const auto l = opt_text(r, "levels_json")) v.levels = Json::parse(*l);
    out.push_back(std::move(v));
  }
  return out;
}

std::map<std::string, ParticipantValues> participant_values(const fs::path& folder) {
  std::map<std::string, ParticipantValues> out;
  if (!fs::is_regular_file(folder / kParticipantsFile)) return out;
  Database db = Database::open_plain(folder / kParticipantsFile, false, true);
  for (const auto& r : db.all("SELECT v.subject_id, d.name, d.type, v.value_num, v.value_text FROM subject_variable v JOIN variable_dictionary d ON d.id = v.variable_dictionary_id")) {
    const VariableType t = type_of(text(r, "type"));
    ParticipantValue val;
    if (t == VariableType::String) val = text(r, "value_text");
    else if (t == VariableType::Boolean) val = integer(r, "value_num") != 0;
    else val = std::holds_alternative<double>(r.at("value_num")) ? std::get<double>(r.at("value_num")) : static_cast<double>(integer(r, "value_num"));
    out[text(r, "subject_id")][text(r, "name")] = val;
  }
  return out;
}

ParticipantsImportReport participants_import(const fs::path& folder_in, const fs::path& tsv, const std::optional<fs::path>& sidecar) {
  Catalogue cat = Catalogue::open(folder_in);
  const fs::path folder = cat.folder();
  std::ifstream in(tsv);
  if (!in) throw Error(code::not_found, tsv.string() + ": cannot be read");
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    rows.push_back(split_tab(line));
  }
  if (rows.empty()) throw refuse(tsv.string() + ": empty");
  const auto& header = rows[0];
  std::size_t id_col = header.size();
  for (std::size_t i = 0; i < header.size(); ++i) if (header[i] == "participant_id") id_col = i;
  if (id_col == header.size()) throw refuse(tsv.string() + ": no participant_id column");
  Json side = Json::object();
  if (sidecar) {
    std::ifstream sj(*sidecar);
    if (!sj) throw Error(code::not_found, sidecar->string() + ": cannot be read");
    std::stringstream ss;
    ss << sj.rdbuf();
    try { side = Json::parse(ss.str()); } catch (const std::exception& e) { throw refuse(sidecar->string() + ": " + e.what()); }
  }

  // participants -> subject ids; an unknown one is refused by name and nothing is written
  std::map<std::string, std::string> id_of;
  for (const auto& e : cat.entries()) if (e.status != "replaced") id_of[e.name] = e.subject_id;
  std::string unknown;
  for (std::size_t r = 1; r < rows.size(); ++r) {
    if (rows[r].size() != header.size()) throw refuse(tsv.string() + ": row " + std::to_string(r + 1) + " has " + std::to_string(rows[r].size()) + " columns, the header " + std::to_string(header.size()));
    if (!id_of.count(rows[r][id_col])) unknown += (unknown.empty() ? "" : ", ") + rows[r][id_col];
  }
  if (!unknown.empty()) throw refuse(tsv.string() + ": participants this dataset does not hold: " + unknown);

  struct Var { std::string name; VariableType type; Json levels; std::string unit, description; bool has_unit = false, has_desc = false, has_levels = false; std::size_t col; };
  std::vector<Var> vars;
  for (std::size_t c = 0; c < header.size(); ++c) {
    if (c == id_col) continue;
    Var v;
    v.name = header[c]; v.col = c;
    const Json meta = side.contains(v.name) && side[v.name].is_object() ? side[v.name] : Json::object();
    if (meta.contains("Type") && meta["Type"].is_string()) {
      const std::string t = meta["Type"].get<std::string>();
      if (t != "number" && t != "string" && t != "boolean") throw refuse(v.name + ": Type must be number, string or boolean, not '" + t + "'");
      v.type = type_of(t);
    } else {
      bool all_numbers = true, any = false;
      for (std::size_t r = 1; r < rows.size(); ++r) {
        if (is_missing(rows[r][c])) continue;
        any = true;
        double d;
        if (!parse_number(rows[r][c], d)) all_numbers = false;
      }
      v.type = any && all_numbers ? VariableType::Number : VariableType::String;
    }
    if (meta.contains("Units") && meta["Units"].is_string()) { v.unit = meta["Units"].get<std::string>(); v.has_unit = true; }
    if (meta.contains("Description") && meta["Description"].is_string()) { v.description = meta["Description"].get<std::string>(); v.has_desc = true; }
    if (meta.contains("Levels") && meta["Levels"].is_object()) {
      v.levels = Json::array();
      for (auto it = meta["Levels"].begin(); it != meta["Levels"].end(); ++it) {
        double d;
        if (v.type == VariableType::Number && parse_number(it.key(), d)) v.levels.push_back(d);
        else v.levels.push_back(it.key());
      }
      v.has_levels = true;
    }
    vars.push_back(std::move(v));
  }

  // typed values, checked before anything is written
  struct Cell { std::string subject_id, variable; std::optional<double> num; std::optional<std::string> str; };
  std::vector<Cell> cells;
  for (const auto& v : vars) {
    for (std::size_t r = 1; r < rows.size(); ++r) {
      const std::string& raw = rows[r][v.col];
      if (is_missing(raw)) continue;
      Cell c;
      c.subject_id = id_of[rows[r][id_col]];
      c.variable = v.name;
      const std::string where = tsv.string() + " row " + std::to_string(r + 1) + ", " + v.name + ": ";
      if (v.type == VariableType::Number) {
        double d;
        if (!parse_number(raw, d)) throw refuse(where + "'" + raw + "' is not a number");
        c.num = d;
      } else if (v.type == VariableType::Boolean) {
        if (raw == "true" || raw == "1") c.num = 1; else if (raw == "false" || raw == "0") c.num = 0;
        else throw refuse(where + "'" + raw + "' is not a boolean (true, false, 1 or 0)");
      } else {
        c.str = raw;
      }
      if (v.has_levels) {
        bool ok = false;
        for (const auto& lv : v.levels) {
          if (c.str && lv.is_string() && lv.get<std::string>() == *c.str) ok = true;
          if (c.num && lv.is_number() && lv.get<double>() == *c.num) ok = true;
        }
        if (!ok) throw refuse(where + "'" + raw + "' is not one of its levels");
      }
      cells.push_back(std::move(c));
    }
  }

  WriterLock lock = acquire_writer_lock(folder, "nxr-io");
  const bool fresh = !fs::exists(folder / kParticipantsFile);
  Database db = Database::open_plain(folder / kParticipantsFile, true);
  db.exec_script(kParticipantTables);
  const std::string dataset_id = text(cat.dataset_row(), "id");
  ParticipantsImportReport rep;
  std::set<std::string> with_values;
  db.tx([&] {
    if (fresh) db.execute("INSERT INTO meta (key, value) VALUES ('participants_version', '1')");
    std::map<std::string, std::string> var_id;
    for (const auto& v : vars) {
      std::string id = uuid7();
      if (const auto old = db.one("SELECT id FROM variable_dictionary WHERE dataset_id = ? AND name = ?", {dataset_id, v.name})) {
        id = text(*old, "id");
        db.execute("DELETE FROM variable_dictionary WHERE id = ?", {id});   // its values go with it
      }
      db.insert("variable_dictionary", {{"id", id}, {"dataset_id", dataset_id}, {"name", v.name}, {"type", std::string(variable_type_name(v.type))},
                                        {"unit", v.has_unit ? Value(v.unit) : Value(std::monostate{})}, {"description", v.has_desc ? Value(v.description) : Value(std::monostate{})},
                                        {"levels_json", v.has_levels ? Value(jdump(v.levels)) : Value(std::monostate{})}});
      var_id[v.name] = id;
      rep.variables.push_back(v.name);
    }
    for (const auto& c : cells) {
      db.insert("subject_variable", {{"subject_id", c.subject_id}, {"variable_dictionary_id", var_id[c.variable]}, {"value_num", c.num ? Value(*c.num) : Value(std::monostate{})},
                                     {"value_text", c.str ? Value(*c.str) : Value(std::monostate{})}});
      with_values.insert(c.subject_id);
      ++rep.n_values;
    }
  });
  rep.n_subjects = static_cast<std::int64_t>(with_values.size());
  return rep;
}

void participants_export(const fs::path& folder_in, const fs::path& tsv, const std::optional<fs::path>& sidecar) {
  Catalogue cat = Catalogue::open(folder_in);
  if (!cat.has_participants()) throw Error(code::not_found, cat.folder().string() + ": this dataset has no participants table");
  const auto vars = participant_variables(cat.folder());
  const auto values = participant_values(cat.folder());
  std::ofstream out(tsv, std::ios::binary | std::ios::trunc);
  if (!out) throw Error(code::io, tsv.string() + ": cannot be written");
  out << "participant_id";
  for (const auto& v : vars) out << '\t' << v.name;
  out << '\n';
  for (const auto& e : cat.entries()) {
    const auto it = values.find(e.subject_id);
    if (e.status == "replaced" || it == values.end()) continue;
    out << e.name;
    for (const auto& v : vars) {
      out << '\t';
      const auto f = it->second.find(v.name);
      if (f == it->second.end()) out << "n/a";
      else if (auto* d = std::get_if<double>(&f->second)) out << js_number(*d);
      else if (auto* b = std::get_if<bool>(&f->second)) out << (*b ? "true" : "false");
      else out << std::get<std::string>(f->second);
    }
    out << '\n';
  }
  if (sidecar) {
    Json j = Json::object();
    for (const auto& v : vars) {
      Json m = Json::object();
      if (v.description) m["Description"] = *v.description;
      if (v.unit) m["Units"] = *v.unit;
      if (v.levels) { Json lv = Json::object(); for (const auto& l : *v.levels) { const std::string k = l.is_string() ? l.get<std::string>() : js_number(l.get<double>()); lv[k] = k; } m["Levels"] = lv; }
      m["Type"] = variable_type_name(v.type);
      j[v.name] = m;
    }
    std::ofstream o(*sidecar, std::ios::binary | std::ios::trunc);
    if (!o) throw Error(code::io, sidecar->string() + ": cannot be written");
    o << j.dump(2) << '\n';
  }
}

}  // namespace nxr::io
