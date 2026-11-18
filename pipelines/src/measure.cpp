// nxr-pipelines — measure compute: stored Fields measured over a partition's members (measure.h). The arithmetic is nxr::partition (the Studio's
// measureOverPartition); the rows are the Studio's (`selection_measurement`, written through nxr::io::write_measurements, the port of its repository);
// the recipe is an entry of the dataset's `recipes` dictionary as the Studio's `recipes.save` writes one (adapters/nxr/recipes.ts, recipe.ts entryOfRecipe).
#include "nxr/pipelines/measure.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/spectral_record.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <regex>
#include <set>

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/io/measures.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }


// the Studio's stableJson / Python's json.dumps(sort_keys=True, separators=(',', ':')): keys sorted at every depth



// the unit of a measure of a field in `u` (the Studio's measureUnit and the registry: a square for mean_square, the weight's unit times u for a sum)
}  // namespace


// ── THE EXPORT ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
namespace {
std::string csv_cell(const std::string& v) {
  if (v.find_first_of(",\"\n\r") == std::string::npos) return v;
  std::string q = "\"";
  for (char c : v) { if (c == '"') q += '"'; q += c; }
  return q + "\"";
}
std::string csv_row(const std::vector<std::string>& cells) {
  std::string s;
  for (size_t i = 0; i < cells.size(); ++i) s += (i ? "," : "") + csv_cell(cells[i]);
  return s + "\n";
}
std::vector<std::vector<std::string>> csv_read(const fs::path& file) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream in(file);
  if (!in) return rows;
  std::stringstream buf; buf << in.rdbuf();
  const std::string t = buf.str();
  std::vector<std::string> row; std::string cell; bool quoted = false;
  for (size_t i = 0; i < t.size(); ++i) {
    const char c = t[i];
    if (quoted) { if (c == '"') { if (i + 1 < t.size() && t[i + 1] == '"') { cell += '"'; ++i; } else quoted = false; } else cell += c; continue; }
    if (c == '"') quoted = true;
    else if (c == ',') { row.push_back(cell); cell.clear(); }
    else if (c == '\n') { row.push_back(cell); cell.clear(); rows.push_back(row); row.clear(); }
    else if (c != '\r') cell += c;
  }
  if (!cell.empty() || !row.empty()) { row.push_back(cell); rows.push_back(row); }
  return rows;
}
// a column name part: letters, digits and '_' (a study release's variables are identifiers)
std::string ident(const std::string& s) {
  std::string o;
  for (char c : s) o += (std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  std::string r;
  for (char c : o) if (!(c == '_' && !r.empty() && r.back() == '_')) r += c;
  while (!r.empty() && r.back() == '_') r.pop_back();
  while (!r.empty() && r.front() == '_') r.erase(r.begin());
  return r;
}
std::string number_text(double v) { char b[40]; std::snprintf(b, sizeof b, "%.17g", v); return b; }
bool geometric(const std::string& st) { return st == "area" || st == "n_elements"; }
// the stored ingredients a statistic is read from (its unit and recipe are theirs)
bool gradient_readout(const std::string& st) { return st == "gradient_strength" || st == "gradient_coherence" || st == "gradient_direction_deg"; }
bool tensor_readout(const std::string& st) { return st == "normal_share" || st == "anisotropy" || st == "circularity" || st == "axis_deg" || st == "power_density"; }
std::string ingredient(const std::string& st) {
  if (tensor_readout(st)) return "tensor_total";
  if (gradient_readout(st)) return "gradient_magnitude";
  if (st == "sd" || st == "rms") return st == "sd" ? "mean" : "mean_square";
  if (st == "variance") return "mean_square";
  return st;
}
std::optional<std::string> unit_text(const std::string& st, const std::optional<std::string>& stored) {
  if (!stored) return std::nullopt;
  if (tensor_readout(st) || gradient_readout(st)) return stored;   // set by the readout
  if (st == "sd") return stored;                                   // read from mean's unit
  if (st == "rms") { const std::string& u = *stored; if (u.size() > 3 && u.front() == '(' && u.rfind(")\xc2\xb2") == u.size() - 3) return u.substr(1, u.size() - 4); return stored; }
  return stored;                                                   // variance: mean_square's unit, (u)²
}
}  // namespace

Json measure_export(const std::string& dataset_folder, const MeasureExportOptions& opt) {
  if (opt.labelling.empty()) invalid("--labelling names the partition");
  if (opt.fields.empty()) invalid("--fields is a pattern over the Field names (its first capture group names the condition)");
  if (opt.table.empty() || ident(opt.table) != opt.table) invalid("--table is the table's name: letters, digits and '_'");
  if (opt.out.empty()) invalid("--out is the folder the table is written to");
  if (opt.statistics.empty()) invalid("--statistics names at least one");
  if (opt.layout != "split" && opt.layout != "wide") invalid("--layout is split or wide");
  static const std::set<std::string> known = {"mean", "sd", "variance", "rms", "mean_square", "max", "min", "abs_max", "sum", "area", "n_elements",
                                             "tensor_11", "tensor_22", "tensor_nn", "tensor_12_re", "tensor_12_im", "tensor_1n_re", "tensor_1n_im", "tensor_2n_re", "tensor_2n_im", "tensor_total",
                                             "normal_share", "anisotropy", "circularity", "axis_deg", "power_density",
                                             "gradient_weight", "gradient_1", "gradient_2", "gradient_n", "gradient_11", "gradient_22", "gradient_12", "gradient_magnitude",
                                             "gradient_strength", "gradient_coherence", "gradient_direction_deg"};
  for (const auto& st : opt.statistics) if (!known.count(st)) invalid("--statistics: " + st + " is not one of mean, sd, variance, rms, mean_square, max, min, abs_max, sum, area, n_elements, "
                                                                     "tensor_* (a hermitian3 Field's sums), normal_share, anisotropy, circularity, axis_deg, power_density (read from them), "
                                                                     "gradient_* (a gradient's sums), gradient_strength, gradient_coherence, gradient_direction_deg (read from them)");
  nxr::signal::spectral_scaling::Target target;
  try { target = nxr::signal::spectral_scaling::parseTarget(opt.scaling); } catch (const std::invalid_argument& e) { invalid(std::string("--scaling ") + e.what()); }
  // how a statistic of a Field follows its values: 1 (a factor: mean, max, …, sd), 2 (the factor squared: mean_square, variance), 0 (a ratio, unchanged)
  auto power_of = [](const std::string& st) -> int {
    static const std::set<std::string> one = {"mean", "max", "min", "sum", "abs_max", "sd", "rms", "tensor_11", "tensor_22", "tensor_nn", "tensor_12_re", "tensor_12_im",
                                              "tensor_1n_re", "tensor_1n_im", "tensor_2n_re", "tensor_2n_im", "tensor_total", "power_density"};
    static const std::set<std::string> two = {"mean_square", "variance"};
    static const std::set<std::string> zero = {"normal_share", "anisotropy", "circularity", "axis_deg"};
    return one.count(st) ? 1 : two.count(st) ? 2 : zero.count(st) ? 0 : -1;
  };
  if (target != nxr::signal::spectral_scaling::Target::Stored)
    for (const auto& st : opt.statistics) {
      if (st == "area" || st == "n_elements") continue;
      if (power_of(st) < 0) invalid("--scaling " + opt.scaling + ": the statistic " + st + " is not converted (a gradient's sums): export it with --scaling stored");
      if (!nxr::signal::spectral_scaling::isLinear(target) && st != "mean" && st != "max" && st != "min" && power_of(st) != 0)
        invalid("--scaling " + opt.scaling + " is non-linear: it applies to mean, max and min (of the stored power), not to " + st);
    }
  std::regex re;
  try { re = std::regex(opt.fields, std::regex::ECMAScript); } catch (const std::regex_error& e) { invalid("--fields " + opt.fields + ": not a regular expression (" + e.what() + ")"); }
  const fs::path folder(dataset_folder);
  const Database db = Database::open(folder / "dataset.sqlite", false, "nxr-neuro", true);

  std::vector<Row> subjects;
  if (opt.subjects.empty()) subjects = complete_subjects(db);
  else for (const auto& n : opt.subjects) {
    const auto r = find_subject(db, n);
    if (!r) throw Error(code::no_such_row, "no subject " + n);
    subjects.push_back(*r);
  }
  if (subjects.empty()) throw Error(code::no_such_row, "the dataset has no subjects");

  struct Col { std::string variable, condition, member, statistic, kind, unit, recipe; std::int64_t field_order = 0, member_order = 0; int stat_order = 0; };
  std::map<std::string, Col> cols;                                 // by variable
  std::map<std::string, std::map<std::string, double>> values;     // subject → variable → value
  std::map<std::string, std::int64_t> cond_order, member_order;
  Json recipes = Json::object(), per_subject = Json::array();
  auto recipe_of = [&](const std::string& by) -> std::string {     // "recipe <name>@<v> · …" → name@v
    if (by.rfind("recipe ", 0) != 0) return by.empty() ? "(none)" : by;
    const std::string rest = by.substr(7);
    return rest.substr(0, rest.find(' '));
  };
  auto wanted = [&](const std::string& rv) {
    if (!opt.recipe) return true;
    return rv == *opt.recipe || rv.rfind(*opt.recipe + "@", 0) == 0;
  };

  for (const Row& subj : subjects) {
    const std::string sname = text(subj, "name"), sid_subj = text(subj, "id");
    const auto sel = nxr::io::selection::labelling_row(db, sid_subj, opt.labelling);
    if (!sel) { per_subject.push_back(Json{{"subject", sname}, {"missing", "no partition " + opt.labelling}}); continue; }
    const std::string sid = text(*sel, "id");
    std::map<std::int64_t, std::string> names;
    if (!is_null(*sel, "dictionary_id")) names = nxr::io::selection::lookup_names(db, text(*sel, "dictionary_id"));
    auto member_name = [&](std::int64_t c) { const auto it = names.find(c); return it != names.end() ? it->second : "member " + std::to_string(c); };
    // the Fields: one per condition
    std::map<std::string, Row> by_cond;
    for (const Row& f : fields_on_manifold(db, sid_subj, text(*sel, "manifold_id"))) {
      std::smatch m;
      const std::string fname = text(f, "name");
      if (!std::regex_match(fname, m, re)) continue;
      const std::string cond = m.size() > 1 && m[1].matched ? m[1].str() : opt_text(f, "kind").value_or("field");
      if (by_cond.count(cond)) invalid(sname + ": two Fields of condition '" + cond + "' match " + opt.fields + " (" + text(by_cond[cond], "name") + ", " + fname + "): narrow the pattern");
      by_cond[cond] = f;
      if (!cond_order.count(cond)) { const std::int64_t k = static_cast<std::int64_t>(cond_order.size()); cond_order[cond] = k; }
    }
    // the maps of statistics (2026-11-13): a band of a map statistic, `<statistic>_<band>`, matched and read as a Field would be; its rows name the slice
    std::map<std::string, std::int64_t> slice_index;
    for (const MapSlice& ms : map_slices(db, sid_subj, text(*sel, "manifold_id"))) {
      std::smatch m;
      if (!std::regex_match(ms.name, m, re)) continue;
      const std::string cond = m.size() > 1 && m[1].matched ? m[1].str() : text(ms.statistic, "measure");
      if (by_cond.count(cond)) invalid(sname + ": two maps of condition '" + cond + "' match " + opt.fields + " (" + text(by_cond[cond], "name") + ", " + ms.name + "): narrow the pattern");
      Row r = ms.statistic;
      r["name"] = ms.name;
      by_cond[cond] = r;
      slice_index[cond] = ms.index;
      if (!cond_order.count(cond)) { const std::int64_t k = static_cast<std::int64_t>(cond_order.size()); cond_order[cond] = k; }
    }
    Json subj_fields = Json::object();
    auto take = [&](const std::optional<Row>& f, const std::string& cond, const std::string& st, int st_order) {
      const std::string ing = ingredient(st);
      Json q = {{"selectionId", sid}, {"measure", st}};
      nxr::io::selection::MeasuredOf of_what;
      if (f && slice_index.count(cond)) {                       // a statistic's map: its rows name the statistic and the slice
        q["ofStatisticId"] = text(*f, "id"); q["ofIndex"] = slice_index.at(cond);
        of_what.statistic_id = text(*f, "id"); of_what.index = slice_index.at(cond);
      } else {
        q["ofFieldId"] = f ? Json(text(*f, "id")) : Json(nullptr);
        if (f) of_what.field_id = text(*f, "id");
      }
      const auto meta = nxr::io::selection::computed_by_and_unit(db, sid, ing, of_what);
      if (meta.empty()) return;
      std::set<std::string> rvs;
      std::optional<std::string> unit;
      for (const Row& r : meta) { rvs.insert(recipe_of(opt_text(r, "computed_by").value_or(""))); if (!unit) unit = opt_text(r, "unit"); }
      for (const auto& rv : rvs) if (!wanted(rv)) invalid(sname + ": the " + ing + " of " + (f ? text(*f, "name") : std::string("the members")) + " on " + opt.labelling + " was made by " + rv + ", not " + *opt.recipe);
      const std::string rv = *rvs.begin();
      if (!recipes.contains(rv)) {
        const auto e = nxr::io::selection::lookup_entry_named(db, "recipes", rv);
        recipes[rv] = e ? Json::parse(opt_text(*e, "params_json").value_or("null"), nullptr, false) : Json(nullptr);
      }
      Json got;
      if (tensor_readout(st)) {
        // a TENSOR READOUT, from the member's stored tensor_* sums (a ratio of sums, never a mean of per-vertex ratios): the tangential part
        // P = [[11, 12], [12*, 22]] in the canonical frames gives the anisotropy |S1 + i S2| / S0, the circularity S3 / S0 (> 0 turning from north
        // towards west), the axis 1/2 arg(S1 + i S2) from north (gauge-dependent); nn over the frame trace the normal share; total over the area
        // the power density
        std::map<std::string, std::map<std::int64_t, std::pair<double, double>>> c;
        for (const char* m : {"tensor_11", "tensor_22", "tensor_nn", "tensor_12_re", "tensor_12_im", "tensor_total"}) {
          Json qq = q; qq["measure"] = m;
          for (const auto& r : query_measurements(db, qq)) c[m][r["code"].get<std::int64_t>()] = {r["value"].is_null() ? std::nan("") : r["value"].get<double>(), r["weight"].is_null() ? 1.0 : r["weight"].get<double>()};
        }
        got = Json::array();
        for (const auto& [code, tot] : c["tensor_total"]) {
          const double p11 = c["tensor_11"][code].first, p22 = c["tensor_22"][code].first, pnn = c["tensor_nn"][code].first, re12 = c["tensor_12_re"][code].first, im12 = c["tensor_12_im"][code].first;
          const double S0 = p11 + p22, S1 = p11 - p22, S2 = 2 * re12, S3 = 2 * im12;
          double v = std::nan("");
          if (st == "normal_share") v = pnn / (S0 + pnn);
          else if (st == "anisotropy") v = std::hypot(S1, S2) / S0;
          else if (st == "circularity") v = S3 / S0;
          else if (st == "axis_deg") v = 0.5 * std::atan2(S2, S1) * 180.0 / M_PI;
          else if (st == "power_density") v = tot.first / tot.second;
          got.push_back(Json{{"code", code}, {"value", std::isfinite(v) ? Json(v) : Json(nullptr)}});
        }
        if (st == "normal_share" || st == "anisotropy" || st == "circularity") unit = std::nullopt;
        else if (st == "axis_deg") unit = std::string("deg");
        else if (unit && unit->size() > 5 && unit->rfind("\xc2\xb7m\xc2\xb2") == unit->size() - 5) unit = unit->substr(0, unit->size() - 5);   // the sum's u·m² over the area: u
      } else if (gradient_readout(st)) {
        // a GRADIENT READOUT, from the member's gradient_* sums: the strength Σa|∇x| / Σa (gauge-free), the coherence |Σa ∇x| / Σa|∇x| (1: aligned over
        // the member, 0: no common direction; gauge-free only as far as the frame turns little over the member), the direction atan2(Σ c2, Σ c1) from
        // north towards west (gauge-dependent)
        std::map<std::string, std::map<std::int64_t, double>> c;
        for (const char* m : {"gradient_weight", "gradient_1", "gradient_2", "gradient_magnitude"}) {
          Json qq = q; qq["measure"] = m;
          for (const auto& r : query_measurements(db, qq)) c[m][r["code"].get<std::int64_t>()] = r["value"].is_null() ? std::nan("") : r["value"].get<double>();
        }
        got = Json::array();
        for (const auto& [code, mag] : c["gradient_magnitude"]) {
          const double w = c["gradient_weight"][code], g1 = c["gradient_1"][code], g2 = c["gradient_2"][code];
          double v = std::nan("");
          if (st == "gradient_strength") v = mag / w;
          else if (st == "gradient_coherence") v = std::hypot(g1, g2) / mag;
          else if (st == "gradient_direction_deg") v = std::atan2(g2, g1) * 180.0 / M_PI;
          got.push_back(Json{{"code", code}, {"value", std::isfinite(v) ? Json(v) : Json(nullptr)}});
        }
        if (st == "gradient_coherence") unit = std::nullopt;
        else if (st == "gradient_direction_deg") unit = std::string("deg");
        else if (unit && unit->size() > 5 && unit->rfind("\xc2\xb7m\xc2\xb2") == unit->size() - 5) unit = unit->substr(0, unit->size() - 5);   // Σa|∇x| over Σa: (u/m)
      } else {
        got = query_measurements(db, q);
      }
      // the scaling chosen at read, from the Field's producer record
      double factor = 1.0;
      std::optional<nxr::signal::spectral_scaling::Scaling> conv;
      if (f && target != nxr::signal::spectral_scaling::Target::Stored && power_of(st) != 0) {
        const auto rec = spectral_record(Json::parse(opt_text(*f, "producer_json").value_or("{}"), nullptr, false));
        if (!rec) invalid(sname + ": " + text(*f, "name") + " has no spectral scaling record (not a spectrum, a relative power, or computed before 2026-11-13): export it with --scaling stored");
        try {
          if (nxr::signal::spectral_scaling::isLinear(target)) factor = std::pow(nxr::signal::spectral_scaling::linearFactor(rec->scaling, target), power_of(st));
          else conv = rec->scaling;
        } catch (const std::invalid_argument& e) { invalid(sname + ": " + text(*f, "name") + ": " + e.what()); }
        const std::string base = nxr::signal::spectral_scaling::unitOf(rec->power_unit, rec->scaling, target);
        unit = power_of(st) == 2 ? "(" + base + ")\xc2\xb2" : base;
      }
      for (const auto& r : got) {
        const std::int64_t code = r["code"].get<std::int64_t>();
        const std::string member = member_name(code);
        if (std::find(opt.exclude.begin(), opt.exclude.end(), member) != opt.exclude.end()) continue;
        if (!member_order.count(member)) member_order[member] = code;
        const std::string var = f ? ident(cond) + "_" + ident(member) + "_" + st : ident(member) + "_" + st;
        Col& c = cols[var];
        if (!c.recipe.empty() && c.recipe != rv) invalid(var + ": subjects measured by different recipes (" + c.recipe + ", " + rv + " in " + sname + "): one column is one method; measure them alike, or name one with --recipe");
        c.variable = var; c.condition = f ? cond : ""; c.member = member; c.statistic = st; c.kind = f ? opt_text(*f, "kind").value_or("") : "";
        c.unit = unit_text(st, unit).value_or(""); c.recipe = rv; c.field_order = f ? cond_order[cond] : -1; c.member_order = member_order[member]; c.stat_order = st_order;
        if (!r["value"].is_null()) {
          const double v = r["value"].get<double>();
          values[sname][var] = conv ? nxr::signal::spectral_scaling::convert(v, *conv, target) : v * factor;
        }
      }
    };
    for (int i = 0; i < static_cast<int>(opt.statistics.size()); ++i) {
      const auto& st = opt.statistics[static_cast<size_t>(i)];
      if (geometric(st)) take(std::nullopt, "", st, i);
      else for (const auto& [cond, f] : by_cond) take(f, cond, st, i);
    }
    for (const auto& [cond, f] : by_cond) {
      subj_fields[cond] = Json{{"id", text(f, "id")}, {"name", text(f, "name")}};
      if (slice_index.count(cond)) { subj_fields[cond]["statistic"] = text(f, "id"); subj_fields[cond]["index"] = slice_index.at(cond); }
    }
    per_subject.push_back(Json{{"subject", sname}, {"labelling", sid}, {"fields", subj_fields}});
  }
  if (cols.empty()) throw Error(code::no_such_row, "no measurements of " + opt.fields + " on " + opt.labelling + " in any subject (measure compute makes them)");

  // the columns: condition (as first met), member (by code), statistic (as asked); the members' own after the conditions
  std::vector<Col> order;
  for (auto& [k, c] : cols) order.push_back(c);
  std::sort(order.begin(), order.end(), [](const Col& a, const Col& b) {
    if (a.field_order != b.field_order) return (a.field_order < 0 ? 1LL << 40 : a.field_order) < (b.field_order < 0 ? 1LL << 40 : b.field_order);
    if (a.member_order != b.member_order) return a.member_order < b.member_order;
    return a.stat_order < b.stat_order;
  });
  const fs::path out(opt.out);
  fs::create_directories(out);
  // THE TABLES: split (the default, the release's layout) — one a condition × statistic, a column a member (`<member>_<statistic>`), and one a
  // statistic of the members' own (area, n_elements); or wide — one table, a column a condition × member × statistic
  struct Table { std::string name; std::vector<std::pair<Col, std::string>> cols; };   // each column: its Col and its name in the table
  std::vector<Table> tables;
  if (opt.layout == "wide") {
    Table t{opt.table, {}};
    for (const auto& c : order) t.cols.emplace_back(c, c.variable);
    tables.push_back(std::move(t));
  } else {
    std::map<std::string, size_t> at;
    for (const auto& c : order) {
      const std::string name = opt.table + (c.condition.empty() ? "" : "_" + ident(c.condition)) + "_" + c.statistic;
      if (!at.count(name)) { at[name] = tables.size(); tables.push_back(Table{name, {}}); }
      tables[at[name]].cols.emplace_back(c, ident(c.member) + "_" + c.statistic);
    }
  }
  for (const Table& tb : tables) {
    std::ofstream t(out / (tb.name + ".csv"));
    std::vector<std::string> head = {"subject"};
    for (const auto& [c, name] : tb.cols) head.push_back(name);
    t << csv_row(head);
    for (const Row& subj : subjects) {
      const std::string sname = text(subj, "name");
      std::vector<std::string> cells = {sname};
      const auto vs = values.find(sname);
      for (const auto& [c, name] : tb.cols) {
        if (vs == values.end()) { cells.emplace_back(); continue; }
        const auto it = vs->second.find(c.variable);
        cells.push_back(it == vs->second.end() ? "" : number_text(it->second));
      }
      t << csv_row(cells);
    }
  }
  // the dictionary: the rows of this request's tables (now, and as its last export named them) replaced, every other table's kept
  std::set<std::string> mine;
  for (const Table& tb : tables) mine.insert(tb.name);
  {
    std::ifstream pj(out / (opt.table + ".json"));
    if (pj) { const Json was = Json::parse(pj, nullptr, false); if (was.is_object()) { mine.insert(was.value("table", std::string())); if (was.contains("tables")) for (const auto& t : was["tables"]) mine.insert(t.value("name", std::string())); } }
  }
  for (const auto& name : mine) {                                            // a table this request made before and no longer makes: its file goes too
    bool now = false; for (const Table& tb : tables) now |= tb.name == name;
    if (!now && !name.empty()) fs::remove(out / (name + ".csv"));
  }
  const std::vector<std::string> dhead = {"index", "data_table", "column_number", "variable", "description", "units", "possible_data_values", "value_type", "notes",
                                          "labelling", "member", "condition", "statistic", "field_kind", "recipe"};
  std::vector<std::vector<std::string>> kept;
  {
    const auto old = csv_read(out / "dictionary.csv");
    for (size_t i = 1; i < old.size(); ++i) if (old[i].size() > 1 && !mine.count(old[i][1])) kept.push_back(old[i]);
  }
  auto describe = [&](const Col& c) {
    const std::string what = c.statistic == "area" ? "area" : c.statistic == "n_elements" ? "number of vertices" : c.statistic;
    if (c.condition.empty()) return "the " + what + " of " + opt.labelling + " member '" + c.member + "'";
    return "the " + what + " of " + (c.kind.empty() ? std::string("") : c.kind + " ") + c.condition + " over " + opt.labelling + " member '" + c.member + "'";
  };
  auto notes = [&](const Col& c) {
    std::string w;
    if (recipes.contains(c.recipe) && recipes[c.recipe].is_object() && recipes[c.recipe].contains("measure")) w = recipes[c.recipe]["measure"].value("weight", std::string());
    std::string n = "made by recipe " + c.recipe + (w.empty() ? "" : " (weights: " + w + ")");
    if (c.statistic == "sd" || c.statistic == "variance" || c.statistic == "rms") n += "; derived at read from the stored mean and mean_square";
    if (tensor_readout(c.statistic)) n += "; read from the member's stored tensor_* sums (a ratio of sums)" + std::string(c.statistic == "axis_deg" ? "; measured from north (the canonical frames): depends on the gauge" : "");
    if (gradient_readout(c.statistic)) n += "; read from the member's gradient_* sums (the gradient of the Field)" + std::string(c.statistic == "gradient_direction_deg" ? "; from north towards west (the canonical frames): depends on the gauge" : "");
    return n;
  };
  Json tables_out = Json::array();
  {
    std::ofstream d(out / "dictionary.csv");
    d << csv_row(dhead);
    std::int64_t idx = 1;
    for (auto r : kept) { if (!r.empty()) r[0] = std::to_string(idx++); d << csv_row(r); }
    for (const Table& tb : tables) {
      d << csv_row({std::to_string(idx++), tb.name, "1", "subject", "the subject, by its name in the dataset", "", "", "string", "", "", "", "", "", "", ""});
      int col = 2;
      for (const auto& [c, name] : tb.cols)
        d << csv_row({std::to_string(idx++), tb.name, std::to_string(col++), name, describe(c), c.unit, "", c.statistic == "n_elements" ? "integer" : "decimal", notes(c),
                      opt.labelling, c.member, c.condition, c.statistic, c.kind, c.recipe});
      tables_out.push_back(Json{{"name", tb.name}, {"columns", tb.cols.size() + 1}});
    }
  }
  const auto dsrow = dataset_row(db);
  Json prov = {{"table", opt.table}, {"generated_utc", now_utc()}, {"generated_by", std::string("nxr-neuro ") + NXR_NEURO_VERSION + " measure export"},
               {"dataset", Json{{"path", fs::absolute(folder).string()}, {"id", dsrow ? text(*dsrow, "id") : ""}, {"name", dsrow ? text(*dsrow, "name") : ""}}},
               {"request", Json{{"labelling", opt.labelling}, {"fields", opt.fields}, {"statistics", opt.statistics}, {"recipe", opt.recipe ? Json(*opt.recipe) : Json(nullptr)}, {"exclude", opt.exclude}, {"scaling", opt.scaling}}},
               {"recipes", recipes}, {"subjects", per_subject}, {"rows", subjects.size()}, {"layout", opt.layout}, {"tables", tables_out}};
  { std::ofstream j(out / (opt.table + ".json")); j << prov.dump(1) << "\n"; }
  return Json{{"stage", "result"}, {"folder", out.string()}, {"tables", tables_out}, {"dictionary", (out / "dictionary.csv").string()}, {"provenance", (out / (opt.table + ".json")).string()},
              {"subjects", subjects.size()}, {"columns", order.size()}, {"recipes", recipes.size()}};
}

}  // namespace nxr::pipelines
