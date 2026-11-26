#include "nxr/io/layout.h"

#include <algorithm>

namespace nxr::io {

namespace {
const char* const kHead[] = {"head", "scalp", "outer", "inner", "skull"};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}
bool is_head(const std::string& name) {
  const std::string n = lower(name);
  return std::any_of(std::begin(kHead), std::end(kHead), [&](const char* w) { return n.find(w) != std::string::npos; });
}
bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// a line's role (SPEC §6 — derived from its coordinate system): time · spectral · eigen · none
std::string role_of(const Database& db, const Row& m) {
  const auto row = db.one("SELECT c.name AS cs, c.unit AS cs_unit, d.unit AS unit FROM manifold m LEFT JOIN geometry g ON g.id = m.geometry_id "
                          "LEFT JOIN coordinate_system c ON c.id = g.coordinate_system_id "
                          "LEFT JOIN geometry_dimension d ON d.geometry_id = m.geometry_id AND d.ordinal = 0 WHERE m.id = ?",
                          {text(m, "id")});
  if (!row) return "";
  const std::string name = lower(text(*row, "cs"));
  const std::string unit = !is_null(*row, "unit") && !text(*row, "unit").empty() ? text(*row, "unit") : text(*row, "cs_unit");
  if (name == "clock" || unit == "s") return "time";
  if (starts(name, "eigenvalue")) return "eigen";
  if (unit == "Hz" || starts(name, "hz") || starts(name, "frequency")) return "spectral";
  return "";
}

[[noreturn]] void nofolder(const std::string& name, const std::string& type) {
  throw Error(code::invalid_argument, "manifold '" + name + "' of type '" + type + "': no folder in nxr.subject@6.0");
}
}  // namespace

std::string line_role(const Database& db, const std::string& manifold_id) {
  const auto m = db.one("SELECT * FROM manifold WHERE id = ?", {Value(manifold_id)});
  return m ? role_of(db, *m) : std::string();
}

std::string folder_of(const Database& db, const std::string& manifold_id) {
  const auto m = db.one("SELECT * FROM manifold WHERE id = ?", {manifold_id});
  if (!m) throw Error(code::invalid_argument, "manifold " + manifold_id + ": no such row");
  const std::string t = text(*m, "type");
  if (t == "product") {
    const auto dims = db.all("SELECT d.manifold_id FROM geometry_dimension d WHERE d.geometry_id = ? ORDER BY d.ordinal", {V(opt_text(*m, "geometry_id"))});
    if (dims.empty()) throw Error(code::invalid_argument, "product '" + text(*m, "name") + "': no factors");
    std::string out;
    for (const auto& d : dims) out += (out.empty() ? "" : "-") + folder_of(db, text(d, "manifold_id"));
    return out;
  }
  if (t == "volume") return "mri";
  if (t == "surface") return is_head(text(*m, "name")) ? "mri" : "source";
  if (t == "line") {
    const std::string role = role_of(db, *m);
    if (role == "time") return "time";
    if (role == "spectral") return "frequency";
    if (role == "eigen") return "eigenvalue";
    throw Error(code::invalid_argument, "line '" + text(*m, "name") + "': neither a time, a frequency nor a spectrum line — no folder in nxr.subject@6.0");
  }
  if (t == "points") return !is_null(*m, "operator") && !text(*m, "operator").empty() ? "eigenvalue" : "sensor";
  if (t == "curve") return "fibre";
  if (t == "graph") return "fibre";                                       // a graph: the fibre network on a surface's tiles, beside the bundles it is made from
  nofolder(text(*m, "name"), t);
}

std::string node_path(const Database& db, const std::string& manifold_id, const std::string& name) { return folder_of(db, manifold_id) + "/" + name; }

std::string folder_for(const std::string& type, const std::string& name, const std::optional<std::string>& role, const std::optional<std::string>& op) {
  if (type == "volume") return "mri";
  if (type == "surface") return is_head(name) ? "mri" : "source";
  if (type == "line") {
    if (role == "time") return "time";
    if (role == "spectral") return "frequency";
    if (role == "eigen") return "eigenvalue";
    throw Error(code::invalid_argument, "line '" + name + "': give its role (time · spectral · eigen)");
  }
  if (type == "points") return op && !op->empty() ? "eigenvalue" : "sensor";
  if (type == "curve") return "fibre";
  if (type == "graph") return "fibre";
  nofolder(name, type);
}

std::string safe_name(const std::string& s) {
  std::string out;
  bool run = false;
  for (unsigned char c : s) {
    const bool ok = std::isalnum(c) && c < 128 ? true : (c == '_' || c == '.' || c == '-');
    if (ok) { out += static_cast<char>(c); run = false; }
    else if (!run) { out += '_'; run = true; }
  }
  const auto a = out.find_first_not_of('_');
  if (a == std::string::npos) return "";
  return out.substr(a, out.find_last_not_of('_') - a + 1);
}

std::string atlas_node(const Database& db, const std::string& manifold_id, const std::string& name) {
  return node_path(db, manifold_id, safe_name(name) + "_atlas");
}

}  // namespace nxr::io
