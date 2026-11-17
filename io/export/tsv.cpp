// Export of stored measurements and statistics as BIDS-style TSV: see nxr/exporters/tsv.h.
#include "nxr/exporters/tsv.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "nxr/formats/tsv.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/errors.h"
#include "nxr/io/measures.h"

namespace nxr::exporters {

namespace tsv = nxr::formats::tsv;
namespace code = nxr::io::code;
using nxr::io::ArrayReader;
using nxr::io::Error;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }

Json json_of(const Json& j) {
  if (j.is_string()) { const Json p = Json::parse(j.get<std::string>(), nullptr, false); return p.is_discarded() ? Json() : p; }
  return j;
}

std::string cell(double v) {
  if (std::isnan(v)) return "n/a";
  if (std::isinf(v)) refuse("a value is infinite; BIDS tables have no infinity");
  return tsv::format_number(v);
}

struct Axis {
  std::string name;
  std::int64_t size = 0;
  std::vector<std::string> columns;                                  // the column names this axis contributes
  std::vector<std::vector<std::string>> cells;                       // cells[i] = this axis's cells at index i
  Json column_meta = Json::object();                                 // per column: Description, Units, Levels
};

// the coordinates of one axis of a statistic, as the columns it adds
Axis axis_of(const nxr::io::SubjectPackage& pkg, const task::Subject& subject, const Json& a, std::int64_t position) {
  Axis x;
  x.name = a.contains("name") && a["name"].is_string() ? a["name"].get<std::string>() : "axis" + std::to_string(position);
  x.size = a.contains("size") && a["size"].is_number() ? a["size"].get<std::int64_t>() : 0;
  const std::string unit = a.contains("unit") && a["unit"].is_string() ? a["unit"].get<std::string>() : "";
  auto numeric = [&](const std::string& col, const std::vector<double>& v, const std::string& u, const std::string& what) {
    x.columns.push_back(col);
    for (const double d : v) x.cells.push_back({cell(d)});
    Json m = Json{{"Description", what}};
    if (!u.empty()) m["Units"] = u;
    x.column_meta[col] = m;
  };
  if (a.contains("edges") && a["edges"].is_array()) {
    const auto& e = a["edges"];
    if (static_cast<std::int64_t>(e.size()) != x.size + 1) refuse("the axis '" + x.name + "' has " + std::to_string(x.size) + " bins but " + std::to_string(e.size()) + " edges");
    x.columns = {x.name + "_low", x.name + "_high"};
    for (std::int64_t i = 0; i < x.size; ++i) x.cells.push_back({cell(e[i].get<double>()), cell(e[i + 1].get<double>())});
    for (const auto& c : x.columns) {
      Json m = Json{{"Description", std::string("a bin edge of the axis ") + x.name + (c.size() > 4 && c.compare(c.size() - 4, 4, "_low") == 0 ? " (lower, inclusive)" : " (upper)")}};
      if (!unit.empty()) m["Units"] = unit;
      x.column_meta[c] = m;
    }
  } else if (a.contains("values_path") && a["values_path"].is_string()) {
    const ArrayReader r = ArrayReader::open(pkg.store(), a["values_path"].get<std::string>());
    const std::vector<double> v = widen(r.read(), r.dtype());
    if (static_cast<std::int64_t>(v.size()) != x.size) refuse("the axis '" + x.name + "' has " + std::to_string(x.size) + " entries but its values hold " + std::to_string(v.size()));
    numeric(x.name, v, unit, "the value of the axis " + x.name + " at each entry");
  } else if (a.contains("manifold_id") && a["manifold_id"].is_string()) {
    const task::Node m = subject.require_id(task::Cls::manifold, a["manifold_id"].get<std::string>());
    const task::Geometry g = subject.geometry(m.id);
    if (m.type && *m.type == "line" && !g.axes.empty() && g.axes[0].spacing) {
      std::vector<double> v(static_cast<std::size_t>(x.size));
      for (std::int64_t i = 0; i < x.size; ++i) v[static_cast<std::size_t>(i)] = g.axes[0].origin.value_or(0.0) + static_cast<double>(i) * *g.axes[0].spacing;
      numeric(x.name, v, !unit.empty() ? unit : g.axes[0].unit.value_or(""), "the coordinate of the axis " + x.name + " (" + m.name + ")");
    } else {
      x.columns = {x.name};
      for (std::int64_t i = 0; i < x.size; ++i) x.cells.push_back({std::to_string(i)});
      x.column_meta[x.name] = Json{{"Description", "the index along " + m.name + " (0 is its first element)"}};
    }
  } else if (a.contains("selection_id") && a["selection_id"].is_string()) {
    const task::Node s = subject.require_id(task::Cls::selection, a["selection_id"].get<std::string>());
    std::map<std::int64_t, std::string> names;
    if (s.dictionary_id) for (const auto& e : subject.dictionary_entries(*s.dictionary_id)) names[e.code] = e.name;
    x.columns = {x.name};
    Json levels = Json::object();
    for (std::int64_t i = 0; i < x.size; ++i) {
      const auto it = names.find(i);
      x.cells.push_back({it == names.end() ? std::to_string(i) : it->second});
      if (it != names.end()) levels[std::to_string(i)] = it->second;
    }
    x.column_meta[x.name] = Json{{"Description", "the member of " + s.name}};
  } else {
    x.columns = {x.name};
    for (std::int64_t i = 0; i < x.size; ++i) x.cells.push_back({std::to_string(i)});
    x.column_meta[x.name] = Json{{"Description", "the index along the axis " + x.name}};
  }
  if (static_cast<std::int64_t>(x.cells.size()) != x.size) refuse("the axis '" + x.name + "' states " + std::to_string(x.size) + " entries");
  return x;
}

}  // namespace

Json export_tsv(const std::filesystem::path& package, const TsvExport& o) {
  const bool stat = !o.statistic.empty();
  const bool meas = !o.partition.empty() || !o.field.empty() || !o.measures.empty();
  if (stat == meas) refuse("name either a statistic, or a partition, a field and measures");
  if (o.out.empty()) refuse("an export names the path stem it writes under");
  const nxr::io::SubjectPackage pkg = nxr::io::SubjectPackage::open(package);
  const task::Subject subject = task::Subject::of(pkg);
  tsv::Table table;
  Json columns = Json::object();
  Json extra = Json::object();
  std::vector<task::Node> inputs;

  if (meas) {
    if (o.partition.empty() || o.field.empty() || o.measures.empty()) refuse("measurements need a partition (--partition), a field (--field) and at least one measure (--measures)");
    const task::Node part = subject.require({.cls = task::Cls::selection, .name = o.partition}, "the partition '" + o.partition + "'");
    const task::Node field = subject.require({.cls = task::Cls::field, .name = o.field}, "the field '" + o.field + "'");
    inputs = {field, part};
    std::map<std::int64_t, std::string> names;
    Json dictionary = Json::array();
    if (part.dictionary_id)
      for (const auto& e : subject.dictionary_entries(*part.dictionary_id)) {
        names[e.code] = e.name;
        dictionary.push_back(Json{{"code", e.code}, {"name", e.name}});
      }
    const auto kinds = subject.measured_kinds(part.id);
    // one query per measure; the members are the union of the codes the queries return
    std::map<std::int64_t, std::map<std::string, std::string>> values;
    Json measure_meta = Json::object();
    for (const std::string& name : o.measures) {
      Json q = Json{{"selectionId", part.id}, {"measure", name}, {"ofFieldId", field.id}};
      Json rows;
      try {
        rows = subject.read_measurements(q);
      } catch (const Error& e) {
        throw Error(e.code(), "the measure '" + name + "' of the field '" + o.field + "' on '" + o.partition + "': " + e.what());
      }
      if (rows.empty()) refuse("'" + o.partition + "' holds no measurement '" + name + "' of the field '" + o.field + "'");
      for (const auto& r : rows) values[r["code"].get<std::int64_t>()][name] = r["value"].is_null() ? "n/a" : cell(r["value"].get<double>());
      const bool derived = nxr::io::measure_is_derived(name);
      Json m = Json{{"Description", std::string("the measure ") + name + " of " + o.field + " over each member of " + o.partition + (derived ? ", derived at read from its stored ingredients" : ", as stored")},
                    {"Derived", derived}};
      std::string unit;
      for (const auto& k : kinds) if (k["measure"] == name && k.contains("unit") && k["unit"].is_string()) unit = k["unit"].get<std::string>();
      if (unit.empty()) if (const auto spec = nxr::io::measure_spec(name)) if (spec->contains("unit") && (*spec)["unit"].is_string()) unit = (*spec)["unit"].get<std::string>();
      if (!unit.empty()) m["Units"] = unit;
      else if (field.unit) m["FieldUnits"] = *field.unit;
      columns[name] = m;
    }
    table.columns = {"member", "code"};
    for (const auto& n : o.measures) table.columns.push_back(n);
    for (const auto& [c, vs] : values) {
      std::vector<std::string> row = {names.count(c) ? names[c] : "code " + std::to_string(c), std::to_string(c)};
      for (const auto& n : o.measures) row.push_back(vs.count(n) ? vs.at(n) : "n/a");
      table.rows.push_back(std::move(row));
    }
    Json levels = Json::object();
    for (const auto& [c, n] : names) levels[std::to_string(c)] = n;
    columns["member"] = Json{{"Description", "the member of the partition " + o.partition + " (its dictionary's name)"}, {"Levels", levels}};
    columns["code"] = Json{{"Description", "the code of the member in the partition " + o.partition}};
    extra["Partition"] = Json{{"name", o.partition}, {"n_elements", part.n_elements ? Json(*part.n_elements) : Json()}, {"dictionary", dictionary}};
    extra["Field"] = Json{{"name", o.field}, {"unit", field.unit ? Json(*field.unit) : Json()}};
  } else {
    const task::Node st = subject.require({.cls = task::Cls::statistic, .name = o.statistic}, "the statistic '" + o.statistic + "'");
    if (!st.path) refuse("the statistic '" + o.statistic + "' stores no values");
    inputs = {st};
    const Json axes = json_of(st.fact("axes_json"));
    if (!axes.is_array() || axes.empty()) refuse("the statistic '" + o.statistic + "' records no axes");
    const Json comps = json_of(st.fact("components_json"));
    const ArrayReader ar = ArrayReader::open(pkg.store(), *st.path);
    std::vector<Axis> xs;
    std::int64_t cells = 1;
    for (std::size_t i = 0; i < axes.size(); ++i) {
      xs.push_back(axis_of(pkg, subject, axes[i], static_cast<std::int64_t>(i)));
      cells *= xs.back().size;
    }
    const bool has_components = ar.shape().size() == axes.size() + 1;
    const std::int64_t ncomp = has_components ? ar.shape().back() : 1;
    if (!(ar.shape().size() == axes.size() || has_components)) refuse("the statistic '" + o.statistic + "' has a stored shape of rank " + std::to_string(ar.shape().size()) + " for " + std::to_string(axes.size()) + " axes");
    for (std::size_t i = 0; i < xs.size(); ++i) if (ar.shape()[i] != xs[i].size) refuse("the axis '" + xs[i].name + "' states " + std::to_string(xs[i].size) + " entries but the array has " + std::to_string(ar.shape()[i]));
    std::vector<std::string> comp_names;
    if (has_components) {
      for (std::int64_t k = 0; k < ncomp; ++k) comp_names.push_back(comps.is_array() && static_cast<std::int64_t>(comps.size()) == ncomp && comps[k].is_string() ? comps[k].get<std::string>() : std::to_string(k));
    }
    const std::vector<double> v = widen(ar.read(), ar.dtype());
    for (const auto& x : xs) for (const auto& c : x.columns) { table.columns.push_back(c); columns[c] = x.column_meta[c]; }
    if (has_components) {
      table.columns.push_back("component");
      columns["component"] = Json{{"Description", "the component of the value"}, {"Levels", comp_names}};
    }
    table.columns.push_back("value");
    const Json measure = st.fact("measure");
    Json vm = Json{{"Description", "the " + (measure.is_string() ? measure.get<std::string>() : std::string("statistic")) + " " + o.statistic}};
    if (st.unit) vm["Units"] = *st.unit;
    columns["value"] = vm;
    std::vector<std::int64_t> idx(xs.size(), 0);
    std::size_t at = 0;
    for (std::int64_t flat = 0; flat < cells; ++flat) {
      std::vector<std::string> head;
      for (std::size_t i = 0; i < xs.size(); ++i) for (const auto& c : xs[i].cells[static_cast<std::size_t>(idx[i])]) head.push_back(c);
      for (std::int64_t k = 0; k < ncomp; ++k) {
        std::vector<std::string> row = head;
        if (has_components) row.push_back(comp_names[static_cast<std::size_t>(k)]);
        row.push_back(cell(v[at++]));
        table.rows.push_back(std::move(row));
      }
      for (std::size_t i = xs.size(); i-- > 0;) if (++idx[i] < xs[i].size) break; else idx[i] = 0;
    }
    extra["Statistic"] = Json{{"name", o.statistic}, {"measure", measure}, {"method", st.method ? Json(*st.method) : Json()}, {"n_members", st.n_members ? Json(*st.n_members) : Json()},
                              {"value_type", st.value_type ? Json(*st.value_type) : Json()}, {"scaling", json_of(st.fact("scaling_json"))}, {"span", json_of(st.fact("span_json"))}};
  }

  const std::string tsv_path = o.out + ".tsv";
  tsv::write_file(tsv_path, table);
  const std::string sidecar_path = o.out + ".json";
  Json prov = provenance(pkg, subject, o.ctx, inputs, Json(), extra);
  Json side = Json::object();
  for (auto it = columns.begin(); it != columns.end(); ++it) side[it.key()] = it.value();
  for (auto it = extra.begin(); it != extra.end(); ++it) side[it.key()] = it.value();
  Json outs = Json::array({Json{{"file", tsv_path.substr(tsv_path.find_last_of("/\\") == std::string::npos ? 0 : tsv_path.find_last_of("/\\") + 1)}, {"sha256", file_sha256(tsv_path)}}});
  prov["outputs"] = outs;
  side["GeneratedBy"] = Json::array({prov});
  write_json(sidecar_path, side);
  return Json{{"files", Json::array({tsv_path})}, {"sidecar", sidecar_path}, {"rows", table.rows.size()}, {"columns", table.columns}};
}

}  // namespace nxr::exporters
