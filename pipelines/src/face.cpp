// nxr-pipelines — what a face asks (face.h). The command line's own calls (runner.h run, listing.h, io's catalogue and task API) made from a JSON
// description of the arguments, so the Python wheel and the MATLAB binding hold no logic of their own.
#include "nxr/io/subject_api.h"
#include "nxr/pipelines/face.h"

#include <algorithm>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>

#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/errors.h"
#include "nxr/io/finder.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/dataset.h"
#include "nxr/io/task/handle.h"
#include "nxr/io/task/task.h"
#include "nxr/io/default_subject.h"
#include "nxr/io/selection_store.h"
#include "nxr/measurement/roll_up.h"
#include "nxr/measurement/measures.h"
#include "nxr/pipelines/selection_bridge.h"
#include "nxr/quantity/quantity.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/listing.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/reading.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/step_nodes.h"
#include "nxr/graph/type_json.h"
#include "nxr/signal/spectral.h"

namespace nxr::pipelines::face {

using namespace nxr::io;
namespace fs = std::filesystem;
namespace g = nxr::pipelines::graph;

std::string stamp() { return std::string("nxr-neuro ") + NXR_NEURO_VERSION + " / C++17"; }

namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }



// the dataset a run works on, as the command line opens it (cli.h open_dataset): a catalogue's subject package, or a one-database dataset
DatasetWriter open_for(const fs::path& folder, const std::string& subject, bool lock, const std::vector<std::string>& templates = {}) {
  OpenOptions o;
  o.lock = lock;
  TemplateSearch search;
  for (const auto& t : templates) search.command_paths.push_back(t);
  if (is_catalogue_dataset(folder)) return open_subject_package(folder, subject, o, search);
  return open_writer(folder, o);
}

double seconds_since(const std::chrono::steady_clock::time_point& t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

}  // namespace

Json pipelines() {
  Json out = Json::array();
  std::vector<Json> all;
  for (const std::string& n : g::available()) {
    const g::Definition d = g::load(n);
    Json params = Json::array();
    for (const auto& [k, v] : d.parameters.items()) {
      const Json doc = d.parameter_docs.contains(k) ? d.parameter_docs[k] : Json::object();
      params.push_back(Json{{"name", k}, {"default", v}, {"meaning", doc.contains("meaning") ? doc["meaning"] : Json("")}, {"unit", doc.contains("unit") ? doc["unit"] : Json("-")}, {"chart", doc.contains("chart") ? doc["chart"] : Json("")},
                            {"choices", doc.contains("choices") ? doc["choices"] : Json::array()}});
    }
    std::string first = d.description.substr(0, d.description.find_first_of(".\n"));
    Json presets = Json::array();
    for (const auto& [pk, pv] : d.presets.items()) { (void)pv; presets.push_back(pk); }
    all.push_back(Json{{"name", n}, {"form", "definition"}, {"kind", d.kind}, {"scope", d.scope}, {"presets", presets}, {"summary", first.empty() ? "Run the pipeline definition " + n + "." : first + "."}, {"parameters", params}});
  }
  std::sort(all.begin(), all.end(), [](const Json& a, const Json& b) { return a["name"].get<std::string>() < b["name"].get<std::string>(); });
  for (auto& j : all) out.push_back(std::move(j));
  return out;
}

Json run(const fs::path& dataset, const std::string& subject, const RunRequest& rq, const Progress& progress) {
  const int given = (rq.from || rq.to ? 1 : 0) + (rq.all ? 1 : 0);
  const bool streams = g::streams_recording(g::load(rq.pipeline));         // a definition that reads no recording takes no time range
  if (given == 0 && streams) refuse("a time range is required: give from and to (seconds), or all for the whole recording");
  if (given > 0 && !streams) refuse(rq.pipeline + " reads no recording: it takes no time range (give neither from/to nor all)");
  if (given > 1) refuse("give one time range: from and to, or all");
  if ((rq.from.has_value()) != (rq.to.has_value())) refuse("from and to go together: give both");
  if (!rq.parameters.is_object()) refuse("the parameters are a set of names and values");
  const auto t0 = std::chrono::steady_clock::now();
  const Progress say = progress ? progress : Progress([](const Json&) {});
  DatasetWriter ds = open_for(dataset, subject, true);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, subject, oo);
  Json runs = Json::array(), stored = Json::array();
  bool all_hit = true;
  {
    const g::Definition d = g::load(rq.pipeline);
    // a face that carries parameters only (the Python and MATLAB `run`) names a preset as the parameter `preset`: it is lifted out here, as the command line's --set NAME is
    Json params = rq.parameters;
    std::string preset = rq.preset;
    if (params.is_object() && params.contains("preset") && !d.parameters.contains("preset")) { preset = params["preset"].get<std::string>(); params.erase("preset"); }
    // likewise the run-wide `projectors` (the command line's --projectors): active | none | the projectors' names, a list or comma-separated text
    std::optional<std::string> projectors;
    if (params.is_object() && params.contains("projectors") && !d.parameters.contains("projectors")) {
      const Json& pj = params["projectors"];
      std::string t;
      if (pj.is_array()) { for (const auto& x : pj) { if (!x.is_string()) refuse("projectors: a list of the projectors' names (text)"); t += (t.empty() ? "" : ",") + x.get<std::string>(); } }
      else if (pj.is_string()) t = pj.get<std::string>();
      else refuse("projectors is active, none, or the projectors' names (a list or comma-separated text)");
      projectors = t;
      params.erase("projectors");
    }
    for (auto it = params.begin(); it != params.end(); ++it)
      if (!d.parameters.contains(it.key())) {
        std::string all;
        for (const auto& [k, _] : d.parameters.items()) all += (all.empty() ? "" : ", ") + k;
        refuse(d.name + " has no parameter '" + it.key() + "' (its parameters: " + (all.empty() ? "none" : all) + ")");
      }
    g::RunOptions o;
    o.requested_by = stamp();
    if (rq.from) { o.start = *rq.from; o.stop = *rq.to; }
    if (rq.all) o.chunks = "all";
    o.replace = rq.replace;
    o.preset = preset;
    if (projectors) o.projectors = projectors_from(*projectors);
    for (const Json& r : g::run(ds, sub, d, params, o, rq.dry_run, say)) {
      const bool hit = r.value("hit", false);
      all_hit = all_hit && hit;
      runs.push_back(Json{{"cache", rq.dry_run ? "planned" : hit ? "hit" : "computed"}, {"info", r}});
      if (r.contains("stored"))
        for (const Json& s : r["stored"]) {   // the stored thing's own name: a statistic, a selection or a field by its record; a measurement by what it is over
          std::string name = s.value("statistic", s.value("selection", s.value("name", std::string())));
          if (name.empty()) name = s.value("output", "");
          stored.push_back(Json{{"output", s.value("output", "")}, {"name", name}, {"id", s.value("id", "")}, {"path", s.value("path", "")}});
        }
    }
  }
  return Json{{"pipeline", rq.pipeline}, {"subject", subject}, {"cache", all_hit ? "hit" : "computed"}, {"stored", stored}, {"runs", runs}, {"seconds", seconds_since(t0)}};
}

Json run_group(const fs::path& dataset, const GroupRunRequest& rq, const Progress& progress) {
  if (rq.subjects.empty()) refuse("a group run states its subjects (a list of names): there is no default group");
  if (!rq.parameters.is_object()) refuse("the parameters are a set of names and values");
  const auto t0 = std::chrono::steady_clock::now();
  const Progress say = progress ? progress : Progress([](const Json&) {});
  const g::Definition d = g::load(rq.pipeline);
  if (d.scope != "group") refuse(d.name + " is not a group definition (scope " + d.scope + "): `run` takes it on one subject");
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  // the members are opened to be read, the group's default subject to be written; a catalogue dataset: each member's package, and the group package the caller names (`into`); a one-database dataset: its one writer
  std::vector<std::unique_ptr<DatasetWriter>> writers;
  std::vector<std::unique_ptr<SubjectWriter>> subjects;
  std::vector<g::GroupMember> members;
  DatasetWriter* group_ds = nullptr;
  std::string group_name;
  if (is_catalogue_dataset(dataset)) {
    if (rq.into.empty()) refuse("a group run in a catalogue dataset names the group package its results go to (into)");
    writers.push_back(std::make_unique<DatasetWriter>(open_for(dataset, rq.into, true)));
    group_ds = writers.back().get();
    group_name = rq.into;
    for (const auto& name : rq.subjects) {
      writers.push_back(std::make_unique<DatasetWriter>(open_for(dataset, name, false)));
      subjects.push_back(std::make_unique<SubjectWriter>(SubjectWriter::open(*writers.back(), name)));
      members.push_back(g::GroupMember{name, writers.back().get(), subjects.back().get()});
    }
  } else {
    writers.push_back(std::make_unique<DatasetWriter>(open_for(dataset, rq.subjects.front(), true)));
    group_ds = writers.back().get();
    const auto tpl = group_ds->template_();
    if (!tpl) refuse("the dataset has no default subject: a group's results are stored in it");
    group_name = text(*tpl, "name");
    for (const auto& name : rq.subjects) {
      subjects.push_back(std::make_unique<SubjectWriter>(SubjectWriter::open(*group_ds, name)));
      members.push_back(g::GroupMember{name, group_ds, subjects.back().get()});
    }
  }
  SubjectWriter group_sub = SubjectWriter::open(*group_ds, group_name, oo);
  g::RunOptions o;
  o.requested_by = stamp();
  o.replace = rq.replace;
  o.preset = rq.preset;
  Json runs = Json::array(), stored = Json::array();
  bool all_hit = true;
  for (const Json& r : g::run_group(*group_ds, group_sub, members, d, rq.parameters, o, rq.dry_run, say)) {
    const bool hit = r.value("hit", false);
    all_hit = all_hit && hit;
    runs.push_back(Json{{"cache", rq.dry_run ? "planned" : hit ? "hit" : "computed"}, {"info", r}});
    if (r.contains("stored"))
      for (const Json& s : r["stored"]) {   // the stored thing's own name: a statistic, a selection or a field by its record; a measurement by what it is over
          std::string name = s.value("statistic", s.value("selection", s.value("name", std::string())));
          if (name.empty()) name = s.value("output", "");
          stored.push_back(Json{{"output", s.value("output", "")}, {"name", name}, {"id", s.value("id", "")}, {"path", s.value("path", "")}});
        }
  }
  return Json{{"pipeline", rq.pipeline}, {"subjects", rq.subjects}, {"into", group_name}, {"cache", all_hit ? "hit" : "computed"}, {"stored", stored}, {"runs", runs}, {"seconds", seconds_since(t0)}};
}

namespace {
// ── the measurement query's compute side ──
// ADAPTER (the one place that meets the other batches' names, written to the consolidation spec: io's `measurement ingredients` call and compute's `nxr::measurement`): io hands the FINEST stored
// ingredients of a measurement and the grid's closed form; compute's `roll_up` and `value` do the arithmetic (the merge rule of each measure is the registry's data, the same io applies in SQL).
//   io:      Subject::measurement_ingredients({selectionId, ofFieldId?, measure}) -> {rows, members, unit, ingredients: {count, weight, sum, sum_sq, max, min, abs_max: [rows x members]},
//            grid: {kind: dyadic | time | product, depth, level, factors?: [{name, kind, depth}], levels: [{level, diameter_mm?, area_m2?, seconds?, weyl_band_per_m2?}]}}
//   compute: nxr::measurement::Ingredients {rows, members, count, weight, sum, sum_sq, max, min, abs_max}; Request {measures, weighting}; coarser_groups(grid, from, to); roll_up; across; value
nxr::measurement::Ingredients ingredients_of(const Json& fine) {
  nxr::measurement::Ingredients ing;
  ing.rows = fine.at("rows").get<std::int64_t>();
  ing.members = fine.at("members").get<std::int64_t>();
  const Json& v = fine.at("ingredients");
  auto take = [&](const char* name, std::vector<double>& into) { if (v.contains(name)) into = v[name].get<std::vector<double>>(); };
  if (v.contains("count")) ing.count = v["count"].get<std::vector<std::int64_t>>();
  take("weight", ing.weight); take("sum", ing.sum); take("sum_sq", ing.sum_sq); take("max", ing.max); take("min", ing.min); take("abs_max", ing.abs_max);
  return ing;
}
// The finest stored ingredients of a measure over a selection, in the shape `ingredients_of` reads: each ingredient (count, weight, sum, sum_square, max, min, abs_max) as io's `measurement_values`
// gives it, [rows x members]; an ingredient the selection does not hold is left out. `grid` carries the selection's own description only: io's level dictionary of a grid (tile diameters, durations,
// Weyl bands) is io's `Subject::grid_levels`, written by the grid makers; a selection of no grid has none, and a roll-up `to` a physical size then refuses (see level_for).
Json measurement_ingredients_of(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& field_id, const std::string& measure) {
  Json out = Json{{"rows", 0}, {"members", 0}, {"ingredients", Json::object()}, {"unit", Json()}};
  std::vector<std::string> names{"count", "weight", "sum", "sum_square", "max", "min", "abs_max"};
  (void)measure;
  for (const auto& n : names) {
    try {
      const MeasurementValues mv = measurement_values(sub, selection_id, field_id, std::nullopt, std::nullopt, n);
      out["rows"] = mv.rows; out["members"] = mv.members;
      if (!out["unit"].is_string() && mv.unit && n != "count" && n != "weight") out["unit"] = *mv.unit;
      const std::string key = n == "sum_square" ? "sum_sq" : n;
      if (n == "count") { std::vector<std::int64_t> c(mv.values.begin(), mv.values.end()); out["ingredients"][key] = c; }
      else out["ingredients"][key] = mv.values;
    } catch (const Error&) {}
  }
  if (out["rows"].get<std::int64_t>() == 0) throw Error(code::no_such_row, "the selection holds none of the ingredients of '" + measure + "' (count, weight, sum, sum_square, max, min, abs_max)");
  // the grid's LEVEL DICTIONARY from io (the makers write it: a dyadic tile's diameter in mm and its Weyl band, a constant-Q level's window in s, a time grid's 2^k samples): what a roll-up `to` a physical size reads
  Json levels = Json::array();
  for (const Json& e : task::Subject::of(sub).grid_levels(selection_id)) {
    Json L = Json{{"level", e["level"]}};
    const Json& at = e["attributes"];
    if (e.contains("measurement") && e.value("unit", std::string()) == "mm") L["diameter_mm"] = e["measurement"];
    else if (at.contains("diameter_mm")) L["diameter_mm"] = at["diameter_mm"];
    if (e.contains("measurement") && e.value("unit", std::string()) == "s") L["seconds"] = e["measurement"];
    else if (at.contains("window_s")) L["seconds"] = at["window_s"];
    if (at.contains("weyl_band_per_m2")) L["weyl_band_per_m2"] = at["weyl_band_per_m2"];
    levels.push_back(std::move(L));
  }
  out["grid"] = Json{{"kind", "dyadic"}, {"levels", levels}};
  return out;
}
// the level a stated quantity means on a grid, from the level attributes io returned (consolidation spec 3.3): a tile diameter = the coarsest level whose mean diameter is at most it; a duration = the
// first level whose tile lasts at least it; a wavelength = the Weyl level holding it; `whole` = -1. The snapped quantity (what the level really is) is returned with it.
std::pair<int, Json> level_for(const std::string& to, const Json& grid) {
  if (to == "whole") return {-1, Json{{"whole", true}}};
  const nxr::quantity::Quantity q = nxr::quantity::parse(to);
  const Json levels = grid.value("levels", Json::array());
  for (const Json& L : levels) {
    const int l = L.at("level").get<int>();
    if ((q.unit == "mm" && q.chart != "wavelength") && L.contains("diameter_mm") && L["diameter_mm"].get<double>() <= q.hi) return {l, Json{{"diameter_mm", L["diameter_mm"]}}};
    if (q.unit == "s" && L.contains("seconds") && L["seconds"].get<double>() >= q.lo) return {l, Json{{"seconds", L["seconds"]}}};
    if (q.chart == "wavelength" && L.contains("weyl_band_per_m2") && L["weyl_band_per_m2"].is_array() && L["weyl_band_per_m2"].size() == 2) {
      const double lam_per_m2 = std::pow(2.0 * M_PI / (q.lo * 1e-3), 2.0);                 // 1/m2 from a wavelength in mm
      if (lam_per_m2 >= L["weyl_band_per_m2"][0].get<double>() && lam_per_m2 < L["weyl_band_per_m2"][1].get<double>()) return {l, Json{{"weyl_band_per_m2", L["weyl_band_per_m2"]}}};
    }
  }
  refuse("measurement query: '" + to + "' names no level of this grid (levels: " + std::to_string(levels.size()) + "; give a tile diameter in mm, a duration in s, a wavelength in mm or whole)");
}
}  // namespace

Json measurement_query(const fs::path& dataset, const std::string& subject, const MeasurementQuery& q) {
  if (q.where != "io" && q.where != "compute") refuse("measurement query: where is io (the SQL query) or compute (in memory)");
  if (q.selection.empty() || q.measure.empty()) refuse("measurement query: name the selection and the measure");
  task::Lease lease = task::open_reader(dataset.string(), subject);
  DatasetWriter& ds = *lease;
  SubjectWriter sub = SubjectWriter::open(ds, subject);
  const task::Subject ts = task::Subject::of(sub);
  const auto sel = ts.by_id_or_name(task::Cls::selection, q.selection);
  if (!sel) throw Error(code::no_such_row, "no selection named '" + q.selection + "' in subject " + subject);
  std::string field_id;
  if (q.of_field) {
    const auto f = ts.by_id_or_name(task::Cls::field, *q.of_field);
    if (!f) throw Error(code::no_such_row, "no field named '" + *q.of_field + "' in subject " + subject);
    field_id = f->id;
  }
  Json jq = Json{{"selectionId", sel->id}, {"measure", q.measure}};
  if (!field_id.empty()) jq["ofFieldId"] = field_id;
  const bool filtered = q.min_value || q.max_value || q.limit > 0 || !q.order.empty();
  if (filtered && q.where != "io") throw Error(code::invalid_argument, "measurement query: --min-value, --max-value, --limit and --order are filters of io's query: use --where io");
  if (q.where == "io" && filtered) {                                        // the value filters, the order and the limit are io's SQL (query::members), the level through the grid's closed-form shift
    nxr::io::query::Query iq;
    iq.measure = q.measure;
    iq.filter.selections = {sel->id};
    if (!field_id.empty()) iq.filter.of_field = field_id;
    iq.filter.min_value = q.min_value;
    iq.filter.max_value = q.max_value;
    iq.filter.subjects = q.subjects.empty() ? std::vector<std::string>{subject} : q.subjects;
    iq.order = q.order;
    iq.limit = q.limit;
    if (!q.order.empty() && q.order != "asc" && q.order != "desc") throw Error(code::invalid_argument, "measurement query: --order is asc or desc");
    if (q.to) {
      Json grid = Json{{"levels", Json::array()}};
      for (const Json& e : ts.grid_levels(sel->id)) {
        Json L = Json{{"level", e["level"]}};
        if (e.contains("measurement") && e.value("unit", std::string()) == "mm") L["diameter_mm"] = e["measurement"];
        if (e.contains("measurement") && e.value("unit", std::string()) == "s") L["seconds"] = e["measurement"];
        if (e["attributes"].contains("weyl_band_per_m2")) L["weyl_band_per_m2"] = e["attributes"]["weyl_band_per_m2"];
        grid["levels"].push_back(L);
      }
      const auto [target, snapped] = level_for(*q.to, grid);
      if (target < 0) iq.roll.whole = true;
      else iq.roll.shift = std::max(0, static_cast<int>(sel->member_ordinal.value_or(grid["levels"].size() - 1)) - target);   // coarser than the stored level by the difference of the levels
    }
    if (!q.subjects.empty() && q.subjects.size() > 1) iq.roll.across_subjects = true;
    return Json{{"measure", q.measure}, {"selection", q.selection}, {"where", "io"}, {"rows", ts.query_members(iq)}};
  }
  if (q.where == "io") {                                                    // the SQL query: the roll-up and the derived measures in io's query, across subjects when asked
    if (q.to) jq["to"] = *q.to;
    if (q.factor) jq["factor"] = *q.factor;
    if (!q.subjects.empty()) jq["subjects"] = q.subjects;
    return Json{{"measure", q.measure}, {"selection", q.selection}, {"where", "io"}, {"rows", ts.read_measurements(jq)}};
  }
  // a selection holding no ingredients but the measure itself (or the stored sums a derived measure is a ratio of): the values member by member, as stored or derived at read
  {
    bool has_ingredients = true;
    try { (void)measurement_ingredients_of(sub, sel->id, field_id.empty() ? std::nullopt : std::optional<std::string>(field_id), q.measure); }
    catch (const Error& e) { if (e.code() != code::no_such_row) throw; has_ingredients = false; }
    if (!has_ingredients && q.subjects.size() <= 1 && !q.to) {
      const MeasurementValues mv = member_measure_values(sub, sel->id, q.measure);
      if (mv.rows != 1) throw Error(code::invalid_argument, "measurement query: '" + q.measure + "' holds " + std::to_string(mv.rows) + " values a member (a map): query a measure of one value a member");
      Json rows = Json::array();
      for (std::int64_t m = 0; m < mv.members; ++m) rows.push_back(Json{{"code", m}, {"value", mv.values[static_cast<std::size_t>(m)]}});
      return Json{{"measure", q.measure}, {"selection", q.selection}, {"where", "compute"}, {"unit", mv.unit ? Json(*mv.unit) : Json()}, {"rows", rows}};
    }
  }
  // compute: the finest ingredients (of each subject asked), summed code for code across subjects, rolled up through the grid's closed form, the measure derived from them
  std::vector<Json> fines{measurement_ingredients_of(sub, sel->id, field_id.empty() ? std::nullopt : std::optional<std::string>(field_id), q.measure)};
  for (const auto& other : q.subjects) {
    if (other == subject) continue;
    task::Lease olease = task::open_reader(dataset.string(), other);
    DatasetWriter& ods = *olease;
    SubjectWriter osub = SubjectWriter::open(ods, other);
    const task::Subject ots = task::Subject::of(osub);
    const auto osel = ots.by_id_or_name(task::Cls::selection, q.selection);
    if (!osel) throw Error(code::no_such_row, "no selection named '" + q.selection + "' in subject " + other);
    Json oq = jq;
    oq["selectionId"] = osel->id;
    if (q.of_field) { const auto f = ots.by_id_or_name(task::Cls::field, *q.of_field); if (!f) throw Error(code::no_such_row, "no field named '" + *q.of_field + "' in subject " + other); oq["ofFieldId"] = f->id; }
    fines.push_back(measurement_ingredients_of(osub, osel->id, oq.contains("ofFieldId") ? std::optional<std::string>(oq["ofFieldId"].get<std::string>()) : std::nullopt, q.measure));
  }
  nxr::measurement::Request request;
  request.measures = {q.measure};
  std::vector<nxr::measurement::Ingredients> all;
  for (const Json& f : fines) all.push_back(ingredients_of(f));
  nxr::measurement::Ingredients ing = all.size() == 1 ? all.front() : [&] { std::vector<const nxr::measurement::Ingredients*> ptrs; for (const auto& a : all) ptrs.push_back(&a); return nxr::measurement::across(ptrs, request); }();
  const Json& grid = fines.front().at("grid");
  Json level_json = Json::object();
  if (q.to) {
    const auto [to_level, snapped] = level_for(*q.to, grid);
    const int from_level = grid.value("level", grid.value("depth", 0));
    const nxr::selection::Selection grid_selection = nxr::pipelines::made(nxr::io::selection::load(sub, sel->id).description);
    const nxr::selection::Elements groups = grid.value("kind", std::string()) == "product" && q.factor
                                                ? nxr::measurement::coarser_groups(grid_selection, [&] { const Json fs = grid.value("factors", Json::array()); for (std::size_t i = 0; i < fs.size(); ++i) if (fs[i].value("name", std::string()) == *q.factor) return i; refuse("measurement query: the product has no factor '" + *q.factor + "'"); }(), from_level, to_level)
                                                : nxr::measurement::coarser_groups(grid_selection, from_level, to_level);
    ing = nxr::measurement::roll_up(ing, groups, request);
    level_json = Json{{"level", to_level}, {"snapped", snapped}};
  }
  std::vector<double> values;
  for (std::int64_t r = 0; r < ing.rows; ++r)
    for (std::int64_t m = 0; m < ing.members; ++m) values.push_back(nxr::measurement::value(ing, r, m, q.measure));
  return Json{{"measure", q.measure}, {"selection", q.selection}, {"where", "compute"}, {"subjects", q.subjects.empty() ? Json::array({subject}) : Json(q.subjects)}, {"to", q.to ? Json(*q.to) : Json()},
              {"level", level_json}, {"rows", ing.rows}, {"members", ing.members}, {"values", Json{{"shape", Json::array({ing.rows, ing.members})}, {"data", values}}}, {"unit", fines.front().value("unit", Json())}};
}

Json datasets(const fs::path& datastore) {
  const Json cat = list_datasets(datastore);
  return Json{{"id", cat.contains("id") ? cat["id"] : Json()}, {"datasets", cat.contains("datasets") ? cat["datasets"] : Json::array()}};
}

Json subjects(const fs::path& dataset, const std::vector<std::string>& names, const std::optional<std::string>& where) {
  Json rows = Json::array();
  if (is_catalogue_dataset(dataset)) {
    const Catalogue cat = Catalogue::open(dataset);
    if (!names.empty() || where) {
      SubjectSelection sel;
      sel.subjects = names;
      sel.where = where;
      const SelectedSubjects chosen = select_subjects(dataset, sel);
      for (const auto& m : chosen.members) rows.push_back(m.to_json());
      return Json{{"dataset", text(cat.dataset_row(), "name")}, {"subjects", rows}, {"selection", chosen.provenance()}};
    }
    for (const auto& e : cat.entries()) rows.push_back(e.to_json());
    return Json{{"dataset", text(cat.dataset_row(), "name")}, {"subjects", rows}};
  }
  if (!names.empty() || where) refuse("subjects: names and where need a catalogue dataset; " + dataset.string() + " is a one-database dataset");
  task::Lease lease = task::open_reader(dataset.string(), std::nullopt);
  DatasetWriter& ds = *lease;
  for (const auto& r : task::Dataset::of(ds).subjects()) rows.push_back(Json{{"name", r.name}, {"kind", r.kind}, {"status", r.status}, {"path", r.path}});
  return Json{{"dataset", text(ds.row(), "name")}, {"subjects", rows}};
}

Json results(const fs::path& dataset, const std::string& subject, const std::string& what) {
  task::Lease lease = task::open_reader(dataset.string(), subject);
  DatasetWriter& ds = *lease;
  SubjectWriter sub = SubjectWriter::open(ds, subject);
  return list_subject_nodes(ds, sub, what);
}

namespace {

// an array, or the arrays inside a group (a sparse operator is a group of data, indices and indptr), read whole from the subject's store
bool is_array_key(const Store& store, const std::string& key) {
  const std::string t = store.get_text(join_key(key, "zarr.json")).value_or("");
  return t.find("\"node_type\":\"array\"") != std::string::npos || t.find("\"node_type\": \"array\"") != std::string::npos;
}
// the arrays under a key (one array, or a group's), each with the name the read gives it; nothing is read
void collect_keys(const Store& store, const std::string& key, const std::string& name, std::vector<std::pair<std::string, std::string>>& out) {
  if (!store.exists(join_key(key, "zarr.json"))) return;
  if (is_array_key(store, key)) { out.emplace_back(key, name); return; }
  for (const auto& k : store.list(key).prefixes) collect_keys(store, join_key(key, k), name.empty() ? k : name + "/" + k, out);
}

std::string size_text(double bytes) {
  char t[64];
  if (bytes >= 1e9) std::snprintf(t, sizeof t, "%.0f bytes (%.2f GB)", bytes, bytes / 1e9);
  else if (bytes >= 1e6) std::snprintf(t, sizeof t, "%.0f bytes (%.1f MB)", bytes, bytes / 1e6);
  else std::snprintf(t, sizeof t, "%.0f bytes", bytes);
  return t;
}
std::string dims_text(const std::vector<std::int64_t>& shape) {
  std::string t;
  for (const auto d : shape) t += (t.empty() ? "" : " x ") + std::to_string(d);
  return "[" + t + "]";
}

// THE PART OF A NODE A READ GIVES, AND ITS BUDGET: `region` is [start, stop) per leading axis of each array (a stop past an array's extent is cut to it); before any array is read, the
// bytes of every array the read gives are added up, and a read above `budget` is refused, naming its size and the region that would fit. `regions` records what each array read.
struct Window {
  std::vector<std::array<std::int64_t, 2>> region;
  double budget = 1e9;
  std::string node;
  Json regions = Json::object();

  void check_region() const {
    for (const auto& r : region)
      if (r[0] < 0 || r[1] < r[0]) throw task::Refused("--region " + std::to_string(r[0]) + ":" + std::to_string(r[1]) + ": START:STOP with 0 <= START <= STOP, one a leading axis");
  }
  // the [start, stop) of each axis of an array of `shape` this read takes
  std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> part(const std::string& name, const std::vector<std::int64_t>& shape) const {
    if (region.size() > shape.size())
      throw task::Refused("--region names " + std::to_string(region.size()) + " axes; '" + node + "' " + (name.empty() ? std::string("values") : name) + " is " + dims_text(shape) + " (" +
                          std::to_string(shape.size()) + " axes)");
    std::vector<std::int64_t> lo(shape.size(), 0), hi = shape;
    for (std::size_t d = 0; d < region.size(); ++d) {
      lo[d] = std::min(region[d][0], shape[d]);
      hi[d] = std::min(region[d][1], shape[d]);
    }
    return {lo, hi};
  }
  void record(const std::string& name, const std::vector<std::int64_t>& lo, const std::vector<std::int64_t>& hi) {
    if (region.empty()) return;
    Json r = Json::array();
    for (std::size_t d = 0; d < lo.size(); ++d) r.push_back(Json::array({lo[d], hi[d]}));
    regions[name] = r;
  }
  // refused when `bytes` (the read's arrays, `what` each) is above the budget; `row_bytes`: the bytes of one index of the first axis of the largest array (for the region suggested)
  void need(double bytes, const std::string& what, double row_bytes, bool regions_apply = true) const {
    if (bytes <= budget) return;
    std::string msg = "'" + node + "' as asked is " + size_text(bytes) + " (" + what + "), above the read's budget of " + size_text(budget) + ": nothing was read; ";
    if (regions_apply && row_bytes > 0) {
      const auto fit = static_cast<std::int64_t>(std::floor(budget / row_bytes));
      msg += "read a part with --region START:STOP (along the first axis, e.g. --region 0:" + std::to_string(std::max<std::int64_t>(fit, 1)) + "), or raise --budget";
    } else {
      msg += "raise --budget";
    }
    throw task::Refused(msg);
  }
};

// The arrays at `keys`, each read over the window's region, after their total size is checked against its budget (nothing is read when it is above)
void read_keys(const Store& store, const std::vector<std::pair<std::string, std::string>>& keys, Window& w, std::vector<StoredArray>& out) {
  struct Part { const ArrayReader* a; std::string name; std::vector<std::int64_t> lo, hi; };
  std::vector<ArrayReader> readers;
  readers.reserve(keys.size());
  for (const auto& k : keys) readers.push_back(ArrayReader::open(store, k.first));
  std::vector<Part> parts;
  double total = 0, row = 0, largest = -1;
  std::string what;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const ArrayReader& a = readers[i];
    auto [lo, hi] = w.part(keys[i].second, a.shape());
    double n = static_cast<double>(a.itemsize());
    for (std::size_t d = 0; d < lo.size(); ++d) n *= static_cast<double>(hi[d] - lo[d]);
    total += n;
    what += (what.empty() ? "" : ", ") + (keys[i].second.empty() ? std::string("values") : keys[i].second) + " " + dims_text(a.shape()) + " " + dtype_to_string(a.dtype());
    if (n > largest) {
      largest = n;
      row = static_cast<double>(a.itemsize());
      for (std::size_t d = 1; d < a.shape().size(); ++d) row *= static_cast<double>(a.shape()[d]);
    }
    parts.push_back(Part{&a, keys[i].second, lo, hi});
  }
  w.need(total, what, row);
  for (const Part& p : parts) {
    StoredArray s;
    s.name = p.name;
    s.dtype = dtype_to_string(p.a->dtype());
    s.shape.resize(p.lo.size());
    for (std::size_t d = 0; d < p.lo.size(); ++d) s.shape[d] = p.hi[d] - p.lo[d];
    const bool whole = s.shape == p.a->shape();
    s.bytes = whole ? p.a->read() : p.a->read(p.lo, p.hi);
    w.record(p.name.empty() ? "values" : p.name, p.lo, p.hi);
    out.push_back(std::move(s));
  }
}

// An array the read made (a selection's spans, a labelling's codes, a measurement table): the window's region of it along its one axis, then the budget
void window_made(std::vector<StoredArray>& arrays, Window& w) {
  double total = 0;
  std::string what;
  for (auto& a : arrays) {
    auto [lo, hi] = w.part(a.name, a.shape);
    if (!w.region.empty() && a.shape.size() == 1) {
      const std::size_t item = a.shape[0] ? a.bytes.size() / static_cast<std::size_t>(a.shape[0]) : 0;
      std::vector<std::uint8_t> b(a.bytes.begin() + static_cast<std::ptrdiff_t>(lo[0] * item), a.bytes.begin() + static_cast<std::ptrdiff_t>(hi[0] * item));
      a.bytes = std::move(b);
      a.shape[0] = hi[0] - lo[0];
      w.record(a.name, lo, hi);
    } else if (!w.region.empty()) {
      throw task::Refused("--region: '" + w.node + "' " + a.name + " is " + dims_text(a.shape) + ", read whole");
    }
    total += static_cast<double>(a.bytes.size());
    what += (what.empty() ? "" : ", ") + a.name + " " + dims_text(a.shape) + " " + a.dtype;
  }
  w.need(total, what, 0, false);
}

StoredArray doubles(const std::string& name, const std::vector<double>& v) {
  StoredArray s;
  s.name = name;
  s.dtype = "float64";
  s.shape = {static_cast<std::int64_t>(v.size())};
  s.bytes.resize(v.size() * sizeof(double));
  if (!v.empty()) std::memcpy(s.bytes.data(), v.data(), s.bytes.size());
  return s;
}

StoredArray int64s(const std::string& name, const std::vector<std::int64_t>& v) {
  StoredArray s;
  s.name = name;
  s.dtype = "int64";
  s.shape = {static_cast<std::int64_t>(v.size())};
  s.bytes.resize(v.size() * sizeof(std::int64_t));
  if (!v.empty()) std::memcpy(s.bytes.data(), v.data(), s.bytes.size());
  return s;
}

Json dictionary_json(const std::vector<task::DictionaryLevel>& levels) {
  Json out = Json::array();
  for (const auto& e : levels) out.push_back(Json{{"code", e.code}, {"name", e.name}});
  return out;
}

Window window_of(const ReadRequest& rq, const std::string& node) {
  Window w;
  w.region = rq.region;
  w.budget = rq.budget;
  w.node = node;
  w.check_region();
  if (!(w.budget > 0)) throw task::Refused("--budget is the bytes a read may hold, above 0");
  return w;
}

// A TIME-RESOLVED FIELD read through the chunk service (the one door, reading.h): the rows and samples asked, chunk by chunk, into one array in the field's stored precision
Stored read_series(SubjectWriter& sub, const task::Subject& ts, const task::Node& n, const finder::Value& v, const ReadRequest& rq) {
  if (rq.reading != "samples" && rq.reading != "shape") refuse("read: reading is samples or shape (got '" + rq.reading + "')");
  if (rq.start.has_value() != rq.stop.has_value()) refuse("read: a span of samples gives both start and stop, or neither (the whole field, read chunk by chunk)");
  if (rq.from.has_value() != rq.to.has_value()) throw task::Refused("--from and --to go together: give both");
  const int ranges = (rq.start ? 1 : 0) + (rq.from ? 1 : 0) + (rq.all ? 1 : 0);
  if (ranges > 1) throw task::Refused("give one time range: --from and --to (seconds), or --all");
  if (ranges == 0 && rq.range_required && rq.reading != "shape")
    throw task::Refused("'" + n.name + "' is a time-resolved field: a time range is required: give --from SECONDS --to SECONDS, or --all for the whole field (read one chunk at a time)");
  if (!rq.region.empty()) throw task::Refused("'" + n.name + "' is a time-resolved field: its part is named by --rows and --from/--to, not --region");
  const Projectors projectors = projectors_from(rq.projectors);
  ChunkSource src = nxr::pipelines::source_of(sub, n, v.rate, v.origin, rq.rows, std::nullopt, projectors);
  for (const std::int32_t r : rq.rows)
    if (r < 0 || r >= src.n_channels()) refuse("read: row " + std::to_string(r) + " is not one of the field's " + std::to_string(src.n_channels()) + " rows");
  const io::Span span = rq.from ? nxr::pipelines::span_of(n, src.n_samples(), v.rate, v.origin, std::nullopt, rq.from, rq.to)
                                : nxr::pipelines::sample_span(src, rq.start.value_or(0), rq.stop.value_or(src.n_samples()));
  std::string dtype = "float32";   // a recording read in place from its file is float32 (LinkedBlocks); a stored one keeps its array's
  if (n.path && sub.store().exists(join_key(*n.path, "zarr.json")) && is_array_key(sub.store(), *n.path)) dtype = dtype_to_string(ArrayReader::open(sub.store(), *n.path).dtype());
  const std::int64_t rows = src.rows() ? src.rows() : src.n_channels(), width = span.s1 - span.s0;
  Json row_list = Json::array();
  if (rq.rows.empty()) for (std::int64_t r = 0; r < rows; ++r) row_list.push_back(r);
  else for (const auto r : rq.rows) row_list.push_back(r);
  const auto factors = ts.factors(n.manifold_id.value_or(""));
  std::optional<std::string> channels;
  if (!factors.empty() && factors.front() != *v.time)
    if (const auto m = ts.by_id(task::Cls::manifold, factors.front())) channels = m->name;
  Stored out;
  out.meta = Json{{"class", "field"}, {"name", n.name}, {"id", n.id}, {"kind", n.kind ? Json(*n.kind) : Json()}, {"unit", n.unit ? Json(*n.unit) : Json()},
                  {"value_type", n.value_type ? Json(*n.value_type) : Json()}, {"producer", n.producer}, {"params", n.params}, {"rate", v.rate}, {"origin", v.origin},
                  {"n_samples", src.n_samples()}, {"n_channels", src.n_channels()}, {"rows", row_list}, {"channels", channels ? Json(*channels) : Json()},
                  {"dtype", dtype}, {"span", span_record(span, v.rate, v.origin)}, {"projectors", rq.projectors}, {"projection", src.projection_record()}, {"reading", rq.reading},
                  {"manifold", Json()}, {"manifold_id", n.manifold_id ? Json(*n.manifold_id) : Json()}, {"cell", n.fact("cell").is_null() ? Json("0") : n.fact("cell")}};
  if (n.manifold_id) if (const auto m = ts.by_id(task::Cls::manifold, *n.manifold_id)) out.meta["manifold"] = m->name;
  if (const auto line = ts.by_id(task::Cls::manifold, *v.time)) out.meta["time"] = line->name;
  if (rq.reading == "shape") return out;
  const bool single = dtype == "float32";
  {   // the budget, before the array is made: the samples asked of the rows asked
    const double bytes = static_cast<double>(rows) * static_cast<double>(width) * (single ? 4.0 : 8.0);
    if (bytes > rq.budget) {
      const double per_second = static_cast<double>(rows) * (single ? 4.0 : 8.0) * v.rate;
      char fit[64];
      std::snprintf(fit, sizeof fit, "%.3g", per_second > 0 ? rq.budget / per_second : 0.0);
      throw task::Refused("'" + n.name + "' as asked is " + size_text(bytes) + " (" + std::to_string(rows) + " rows x " + std::to_string(width) + " samples, " + dtype + "), above the read's budget of " +
                          size_text(rq.budget) + ": nothing was read; read a shorter time range (at most " + fit + " s of these rows), fewer --rows, or raise --budget");
    }
  }
  StoredArray a;
  a.name = "values";
  a.dtype = dtype;
  a.shape = {rows, width};
  a.bytes.resize(static_cast<std::size_t>(rows * width) * (single ? sizeof(float) : sizeof(double)));
  if (width > 0)
    chunk_loop(src, span, [&](const Chunk& c) {
      const std::int64_t at = c.first - span.s0;
      for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t t = 0; t < c.n; ++t) {
          const double x = c.data[static_cast<std::size_t>(r * c.n + t)];
          const std::size_t i = static_cast<std::size_t>(r * width + at + t);
          if (single) reinterpret_cast<float*>(a.bytes.data())[i] = static_cast<float>(x);
          else reinterpret_cast<double*>(a.bytes.data())[i] = x;
        }
    });
  out.arrays.push_back(std::move(a));
  return out;
}

// A MANIFOLD: its positions and cells as stored, its axes, its element names; or one labelling family of it (codes per element and the dictionary)
Stored read_manifold(SubjectWriter& sub, const task::Subject& ts, const task::Node& m, const ReadRequest& rq) {
  Stored out;
  if (rq.family) {
    const auto row = nxr::io::selection::labelling_row(sub, *rq.family, m.id);
    if (!row) throw Error(code::no_such_row, "manifold '" + m.name + "' has no labelling '" + *rq.family + "' (its selections: read the manifold without a family)");
    const std::string id = text(*row, "id");
    const auto members = ts.element_members(id);
    out.arrays.push_back(int64s("codes", members.element_labels()));
    const auto node = ts.by_id(task::Cls::selection, id);
    Json dict = Json::array();
    if (node && node->dictionary_id) dict = dictionary_json(ts.dictionary_entries(*node->dictionary_id));
    else for (const auto& [code, nm] : members.names) dict.push_back(Json{{"code", code}, {"name", nm}});
    out.meta = Json{{"class", "labelling"}, {"name", text(*row, "name")}, {"id", id}, {"family", *rq.family}, {"manifold", m.name}, {"manifold_id", m.id},
                    {"n", static_cast<std::int64_t>(members.member_of.size())}, {"dictionary", dict}};
    Window w = window_of(rq, m.name);
    window_made(out.arrays, w);
    if (!rq.region.empty()) out.meta["regions"] = w.regions;
    return out;
  }
  const task::Geometry g = ts.geometry(m.id);
  std::vector<std::pair<std::string, std::string>> keys;
  if (g.positions_path) collect_keys(sub.store(), *g.positions_path, "positions", keys);
  if (m.producer.is_object() && m.producer.contains("kinds") && m.producer["kinds"].is_string()) collect_keys(sub.store(), m.producer["kinds"].get<std::string>(), "kinds", keys);   // a stored line's code a point
  for (const auto& [rank, path] : g.cells) collect_keys(sub.store(), path, "cells_" + std::to_string(rank), keys);
  Window w = window_of(rq, m.name);
  read_keys(sub.store(), keys, w, out.arrays);
  Json axes = Json::array();
  for (const auto& a : g.axes)
    axes.push_back(Json{{"name", a.name ? Json(*a.name) : Json()}, {"n", a.n_vertices}, {"spacing", a.spacing ? Json(*a.spacing) : Json()}, {"origin", a.origin ? Json(*a.origin) : Json()},
                        {"unit", a.unit ? Json(*a.unit) : Json()}});
  Json names;   // the element names: the manifold's `<name>_names` labelling, when stored
  if (const auto row = nxr::io::selection::labelling_row(sub, m.name + "_names", m.id)) {
    const auto members = ts.element_members(text(*row, "id"));
    names = Json::array();
    for (const auto label : members.element_labels()) names.push_back(label < 0 ? std::string() : members.name_of(label));
  }
  out.meta = Json{{"class", "manifold"}, {"name", m.name}, {"id", m.id}, {"type", m.type ? Json(*m.type) : Json()}, {"n", m.n_vertices ? Json(*m.n_vertices) : Json()},
                  {"coordinate_unit", g.coordinate_unit ? Json(*g.coordinate_unit) : Json()},
                  {"coordinate_system", g.coordinate_system ? Json{{"name", *g.coordinate_system}, {"unit", g.coordinate_unit ? Json(*g.coordinate_unit) : Json()}, {"space", g.space ? Json(*g.space) : Json()}} : Json()},
                  {"axes", axes}, {"names", names},
                  {"selections", nxr::io::selection_names_on(sub, m.id)}, {"producer", m.producer}};
  if (!rq.region.empty()) out.meta["regions"] = w.regions;
  return out;
}

}  // namespace

Stored read(const fs::path& dataset, const std::string& subject, const std::string& name, const std::optional<std::string>& measure, const std::optional<std::string>& of_field) {
  ReadRequest rq;
  rq.measure = measure;
  rq.of_field = of_field;
  return read(dataset, subject, name, rq);
}

Stored read(const fs::path& dataset, const std::string& subject, const std::string& name, const ReadRequest& rq) {
  if (!rq.cls.empty() && rq.cls != "field" && rq.cls != "statistic" && rq.cls != "operator" && rq.cls != "selection" && rq.cls != "manifold")
    refuse("read: class '" + rq.cls + "' (field, statistic, operator, selection or manifold)");
  task::Lease lease = task::open_reader(dataset.string(), subject, rq.templates);
  DatasetWriter& ds = *lease;
  SubjectWriter sub = SubjectWriter::open(ds, subject);
  const task::Subject ts = task::Subject::of(sub);
  if (rq.onto) {
    // CARRIED ONTO THE GLUED CORTEX: the node read as stored, then its rows selected by the glue's rule (input.h carried_onto_glued): a selection, no maths
    if (!rq.cls.empty() && rq.cls != "field" && rq.cls != "operator") refuse("read: onto carries the rows of an operator or a field onto a glued cortex (read operator or read field)");
    if (!rq.region.empty() || !rq.rows.empty()) refuse("read: onto selects the rows itself: not with a region or rows");
    std::optional<task::Node> n;
    if (rq.cls.empty() || rq.cls == "field") n = ts.by_id_or_name(task::Cls::field, name);
    if (!n && (rq.cls.empty() || rq.cls == "operator")) n = ts.by_id_or_name(task::Cls::operator_, name);
    if (!n) throw Error(code::no_such_row, "subject " + subject + " holds no " + (rq.cls.empty() ? std::string("field or operator") : rq.cls) + " named '" + name + "'");
    const auto glued = ts.by_id_or_name(task::Cls::manifold, *rq.onto);
    if (!glued) throw Error(code::no_such_row, "subject " + subject + " holds no manifold named '" + *rq.onto + "' (onto names a glued cortex)");
    const auto map = glue_map_of(sub, glued->id);
    if (!map) refuse("read: '" + glued->name + "' is not a glued cortex (a surface made by `run glue`): nothing to carry onto");
    const bool is_op = n->cls == task::Cls::operator_;
    const std::optional<std::string> rows_on = is_op ? n->to_manifold_id : n->manifold_id;
    const auto original = ts.by_id(task::Cls::manifold, map->original_id);
    if (!original || !original->n_vertices) throw Error(code::no_such_row, glued->name + ": its original manifold is not stored");
    if (rows_on != std::optional<std::string>(map->original_id))
      refuse("read: the rows of '" + n->name + "' are not the vertices of " + original->name + ", the surface " + glued->name + " is glued from" + (is_op ? " (an operator's rows are its target's: to_manifold)" : ""));
    ReadRequest inner = rq;
    inner.onto.reset();
    inner.cls = is_op ? "operator" : "field";
    Stored st = read(dataset, subject, n->name, inner);
    if (st.arrays.size() != 1) refuse("read: '" + n->name + "' is stored sparse (data, indices, indptr): onto carries a dense operator or field");
    StoredArray& a = st.arrays[0];
    if (a.shape.empty() || a.shape[0] % *original->n_vertices != 0)
      refuse("read: '" + n->name + "' has " + (a.shape.empty() ? std::string("no rows") : std::to_string(a.shape[0]) + " rows") + ", not k rows for each of the " + std::to_string(*original->n_vertices) + " vertices of " + original->name);
    const std::int64_t k = a.shape[0] / *original->n_vertices;
    a.bytes = carried_onto_glued(a.bytes, a.shape[0], *map, *original->n_vertices);
    a.shape[0] = k * map->n_glued();
    const Json rel = glued->producer.is_object() && glued->producer.contains("glue") ? glued->producer["glue"] : Json();
    st.meta["onto"] = Json{{"glued", glued->name}, {"glued_id", glued->id}, {"original", original->name}, {"rows_per_vertex", k}, {"glued_vertices", map->n_glued()},
                           {"bridge_rows", "zero"}, {"rule", rel.is_object() && rel.contains("rule") ? rel["rule"] : Json("each glued vertex the rows of its original vertex (the original's vertices off the medial wall, ascending), the bridge's rows zero")}};
    return st;
  }
  const std::optional<std::string>& measure = rq.measure;
  const std::optional<std::string>& of_field = rq.of_field;
  const bool series_options = rq.start || rq.stop || rq.from || rq.to || rq.all || !rq.rows.empty() || rq.reading != "samples" || rq.projectors != "active";
  auto no_series = [&](const std::string& what) {
    if (series_options) refuse("read: '" + name + "' is " + what + ", not a time-resolved field: a time range, rows, projectors and reading do not apply");
  };
  auto wanted = [&](const char* c) { return rq.cls.empty() || rq.cls == c; };
  if (rq.family && !wanted("manifold")) refuse("read: family names a labelling of a manifold (read manifold)");
  if (measure && !wanted("selection")) refuse("read: measure names a measurement table of a selection (read selection)");
  if (rq.family) {
    no_series("a manifold's labelling");
    if (measure) refuse("read: family names a labelling of a manifold; measure names a measurement of a selection: not both");
    const auto m = ts.by_id_or_name(task::Cls::manifold, name);
    if (!m) throw Error(code::no_such_row, "subject " + subject + " holds no manifold named '" + name + "' (family names a labelling of a manifold)");
    return read_manifold(sub, ts, *m, rq);
  }
  Stored out;
  if (measure) {
    no_series("a selection's measurement table");
    const auto sel = ts.by_id_or_name(task::Cls::selection, name);
    if (!sel) throw Error(code::no_such_row, "no selection named '" + name + "' in subject " + subject);
    Json q = Json{{"selectionId", sel->id}, {"measure", *measure}};
    if (of_field) {
      const auto f = ts.by_id_or_name(task::Cls::field, *of_field);
      if (!f) throw Error(code::no_such_row, "no field named '" + *of_field + "' in subject " + subject);
      q["ofFieldId"] = f->id;
    }
    const Json rows = ts.read_measurements(q);
    std::vector<double> code_v, value, weight, members;
    for (const Json& r : rows) {
      code_v.push_back(r.value("code", 0.0));
      value.push_back(r.contains("value") && r["value"].is_number() ? r["value"].get<double>() : std::nan(""));
      weight.push_back(r.contains("weight") && r["weight"].is_number() ? r["weight"].get<double>() : std::nan(""));
      members.push_back(r.contains("members") && r["members"].is_number() ? r["members"].get<double>() : std::nan(""));
    }
    out.arrays = {doubles("code", code_v), doubles("value", value), doubles("weight", weight), doubles("members", members)};
    out.meta = Json{{"class", "measurements"}, {"name", name}, {"selection", sel->id}, {"measure", *measure}, {"rows", rows.size()}};
    Window w = window_of(rq, name);
    window_made(out.arrays, w);
    if (!rq.region.empty()) out.meta["regions"] = w.regions;
    return out;
  }
  if (const auto st = wanted("statistic") ? ts.by_id_or_name(task::Cls::statistic, name) : std::nullopt) {
    no_series("a statistic");
    const StoredStatistic s = read_statistic(sub, st->name, false);   // its record; the values are read here, in the stored precision, over the region asked
    Window w = window_of(rq, st->name);
    if (st->path && is_array_key(sub.store(), *st->path)) {
      const ArrayReader ar = ArrayReader::open(sub.store(), *st->path);
      if (ar.dtype() != DType::Float32 && ar.dtype() != DType::Float64) throw Error(code::invalid_argument, st->name + ": a statistic's values are float32 or float64, not " + dtype_to_string(ar.dtype()));
      read_keys(sub.store(), {{*st->path, "values"}}, w, out.arrays);
    } else {
      StoredArray a;
      a.name = "values";
      a.dtype = "float64";
      a.shape = s.shape;
      out.arrays.push_back(std::move(a));
    }
    out.meta = Json{{"class", "statistic"}, {"name", st->name}, {"id", st->id}, {"measure", st->row.value("measure", "")}, {"method", st->row.value("method", "")}, {"unit", st->unit ? Json(*st->unit) : Json()},
                    {"axes", s.axes}, {"components", s.components}, {"span", s.span}, {"options", s.options}, {"producer", s.producer}, {"scaling", s.scaling}};
    if (!rq.region.empty()) out.meta["regions"] = w.regions;
    return out;
  }
  for (const task::Cls c : {task::Cls::field, task::Cls::operator_, task::Cls::selection}) {
    if (!wanted(task::to_string(c))) continue;
    const auto n = ts.by_id_or_name(c, name);
    if (!n) continue;
    if (c == task::Cls::field) {
      const finder::Value v = ts.describe(*n);
      if (v.time && v.rate > 0) return read_series(sub, ts, *n, v, rq);
    }
    no_series(std::string("a ") + task::to_string(c) + (c == task::Cls::field ? " with no time line" : ""));
    const std::optional<std::string> where = c == task::Cls::selection ? n->array_path : n->path;
    if (!where && c == task::Cls::selection && n->type == std::optional<std::string>("spans")) {   // a selection of SPANS (events, bad segments): its intervals, codes and dictionary
      std::vector<double> start, stop;
      std::vector<std::int64_t> codes;
      for (const auto& sp : ts.spans(*n)) {
        start.push_back(sp.start);
        stop.push_back(sp.stop);
        codes.push_back(sp.code.value_or(-1));
      }
      out.arrays = {doubles("start", start), doubles("stop", stop), int64s("code", codes)};
      const auto line = n->manifold_id ? ts.by_id(task::Cls::manifold, *n->manifold_id) : std::nullopt;
      out.meta = Json{{"class", "selection"}, {"name", n->name}, {"id", n->id}, {"type", "spans"}, {"unit", "s"}, {"line", line ? Json(line->name) : Json()},
                      {"dictionary", n->dictionary_id ? dictionary_json(ts.dictionary_entries(*n->dictionary_id)) : Json::array()}, {"producer", n->producer}, {"params", n->params}};
      Window w = window_of(rq, n->name);
      if (n->params.is_object() && n->params.contains("weights") && n->params["weights"].is_string()) {
        std::vector<std::pair<std::string, std::string>> keys;
        collect_keys(sub.store(), n->params["weights"].get<std::string>(), "weights", keys);
        std::vector<StoredArray> weights;
        read_keys(sub.store(), keys, w, weights);
        window_made(out.arrays, w);
        for (auto& x : weights) out.arrays.push_back(std::move(x));
      } else {
        window_made(out.arrays, w);
      }
      if (!rq.region.empty()) out.meta["regions"] = w.regions;
      return out;
    }
    if (!where) throw Error(code::unsupported, "'" + name + "' has no stored array (it is virtual: an analytic operator, a product, a tiling, or a set with nothing stored per element)");
    std::vector<std::pair<std::string, std::string>> keys;
    collect_keys(sub.store(), *where, "", keys);
    if (keys.size() > 1 && !rq.region.empty())   // a sparse operator or field: its data, indices and indptr only make sense together
      throw task::Refused("'" + n->name + "' is stored sparse (data, indices, indptr): it is read whole, not by --region");
    for (auto& k : keys) if (k.second.empty()) k.second = "values";
    if (c == task::Cls::selection && n->params.is_object() && n->params.contains("weights") && n->params["weights"].is_string())   // one weight a member, stored beside it
      collect_keys(sub.store(), n->params["weights"].get<std::string>(), "weights", keys);
    Window w = window_of(rq, n->name);
    read_keys(sub.store(), keys, w, out.arrays);
    out.meta = Json{{"class", task::to_string(c)}, {"name", n->name}, {"id", n->id}, {"kind", n->kind ? Json(*n->kind) : Json()}, {"unit", n->unit ? Json(*n->unit) : Json()},
                    {"value_type", n->value_type ? Json(*n->value_type) : Json()}, {"producer", n->producer}, {"params", n->params}};
    auto manifold_name = [&](const std::optional<std::string>& id) -> Json {
      if (!id) return Json();
      const auto m = ts.by_id(task::Cls::manifold, *id);
      return m ? Json(m->name) : Json();
    };
    if (c == task::Cls::field) {                    // the manifold it lies on and the cells its values sit on
      out.meta["manifold"] = manifold_name(n->manifold_id);
      out.meta["manifold_id"] = n->manifold_id ? Json(*n->manifold_id) : Json();
      out.meta["cell"] = n->fact("cell").is_null() ? Json("0") : n->fact("cell");
    } else if (c == task::Cls::operator_) {         // the manifolds it maps between
      out.meta["from_manifold"] = manifold_name(n->from_manifold_id);
      out.meta["to_manifold"] = manifold_name(n->to_manifold_id);
      out.meta["layout"] = n->fact("layout");
    } else {
      out.meta["type"] = n->type ? Json(*n->type) : Json();
      out.meta["manifold"] = manifold_name(n->manifold_id);
      if (n->dictionary_id) out.meta["dictionary"] = dictionary_json(ts.dictionary_entries(*n->dictionary_id));
    }
    if (!rq.region.empty()) out.meta["regions"] = w.regions;
    return out;
  }
  if (const auto m = wanted("manifold") ? ts.by_id_or_name(task::Cls::manifold, name) : std::nullopt) {
    no_series("a manifold");
    return read_manifold(sub, ts, *m, rq);
  }
  if (!rq.cls.empty())                              // named in another class: say which, and the command that reads it
    for (const task::Cls c : {task::Cls::statistic, task::Cls::field, task::Cls::operator_, task::Cls::selection, task::Cls::manifold})
      if (ts.by_id_or_name(c, name)) {
        const std::string other = task::to_string(c);
        throw Error(code::no_such_row, "subject " + subject + " holds no " + rq.cls + " named '" + name + "': it is " + (std::string("aeiou").find(other.front()) != std::string::npos ? "an " : "a ") + other + " (read " + other + ")");
      }
  throw Error(code::no_such_row, "subject " + subject + " holds no " + (rq.cls.empty() ? std::string("statistic, field, operator, selection or manifold") : rq.cls) + " named '" + name + "'");
}

namespace {
// `read <class>`: the face's read restricted to the class, each option refused where it does not apply; the arrays handed back, and written as .npy files on request
CommandResult read_class(const std::string& cls, const ReadNodeRequest& r) {
  ReadRequest rq;
  rq.cls = cls;
  rq.templates = r.templates;
  rq.region = r.region;
  rq.budget = r.budget;
  const bool series = r.from || r.to || r.all || !r.rows.empty() || r.projectors != "active" || r.shape_only;
  if (cls == "field") {
    rq.from = r.from;
    rq.to = r.to;
    rq.all = r.all;
    rq.rows = r.rows;
    rq.projectors = r.projectors;
    rq.reading = r.shape_only ? "shape" : "samples";
    rq.range_required = true;
  } else if (series) {
    throw task::Refused("a time range, rows, projectors and shape-only are options of `read field`");
  }
  if (cls == "selection") {
    if (r.field && !r.measure) throw task::Refused("--field names the field a measurement is of: give --measure too");
    rq.measure = r.measure;
    rq.of_field = r.field;
  } else if (r.measure || r.field) {
    throw task::Refused("--measure and --field read a selection's measurement table: options of `read selection`");
  }
  if (cls == "field" || cls == "operator") rq.onto = r.onto;
  else if (r.onto) throw task::Refused("--onto carries the rows of an operator or a field onto a glued cortex: an option of `read operator` and `read field`");
  if (cls == "manifold") rq.family = r.family;
  else if (r.family) throw task::Refused("--family reads a labelling of a manifold: an option of `read manifold`");
  const Stored st = read(r.dataset, r.subject, r.name, rq);
  CommandResult out;
  out.result = st.meta;
  Json arrays = Json::array();
  for (const StoredArray& a : st.arrays) {
    arrays.push_back(Json{{"name", a.name}, {"dtype", a.dtype}, {"shape", a.shape}});
    out.arrays.push_back(task::CommandArray{a.name, a.dtype, a.shape, a.bytes});
  }
  out.result["arrays"] = arrays;
  if (r.out) {
    if (out.arrays.empty()) throw task::Refused("'" + r.name + "' read as asked has no array to write (--out)");
    // one array: the file named; several: <stem>.<array>.npy beside it (a sparse operator's data, indices, indptr; a manifold's positions and cells; spans' start, stop, code)
    fs::path stem = *r.out;
    if (stem.extension() == ".npy") stem.replace_extension();
    Json files = Json::object();
    for (const auto& a : out.arrays) {
      std::string tag = a.name;
      std::replace(tag.begin(), tag.end(), '/', '.');
      const fs::path file = out.arrays.size() == 1 ? fs::path(*r.out) : fs::path(stem.string() + "." + tag + ".npy");
      write_npy(file, a.dtype, a.shape, a.bytes);
      files[a.name] = file.string();
    }
    out.result["files"] = files;
  }
  return out;
}
}  // namespace

CommandResult read_field(const ReadNodeRequest& request) { return read_class("field", request); }
CommandResult read_statistic(const ReadNodeRequest& request) { return read_class("statistic", request); }
CommandResult read_operator(const ReadNodeRequest& request) { return read_class("operator", request); }
CommandResult read_selection(const ReadNodeRequest& request) { return read_class("selection", request); }
CommandResult read_manifold(const ReadNodeRequest& request) { return read_class("manifold", request); }

fs::path synthetic_dataset(const fs::path& datastore, const std::string& name, const std::string& subject, const std::string& recording, int channels, double sfreq, double seconds, unsigned seed) {
  if (channels < 2) refuse("synthetic_dataset: at least two channels");
  const std::int64_t M = channels, T = static_cast<std::int64_t>(seconds * sfreq);
  const double bad_start = 0.3 * seconds, bad_stop = 0.375 * seconds;
  // known signals (an LCG and Box-Muller: the same samples on every platform): channel 0 an 8 Hz sinusoid with a little noise on an offset, the others noise, the last scaled by 40 (the bad one)
  std::uint64_t state = seed;
  auto uniform = [&] { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return ((state >> 11) + 0.5) / 9007199254740992.0; };
  auto normal = [&] { return std::sqrt(-2.0 * std::log(uniform())) * std::cos(2.0 * M_PI * uniform()); };
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (std::int64_t i = 0; i < M; ++i)
    for (std::int64_t t = 0; t < T; ++t) {
      double v = i == 0 ? 2.0 * std::sin(2 * M_PI * 8.0 * static_cast<double>(t) / sfreq) + 0.2 * normal() + 5.0 : (i == M - 1 ? 40.0 : 1.0) * normal();
      xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(v);
    }
  DatasetWriter ds = nxr::io::create_dataset(datastore, name);
  {
    SubjectWriter sub = SubjectWriter::create(ds, subject);
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M * 3));
      for (auto& p : pos) p = static_cast<float>(normal() * 0.1);
      ManifoldSpec m;
      m.id = cid;
      m.name = "sensors";
      m.path = "sensor/sensors";
      m.type = "points";
      m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs;
      gs.form = "stored";
      gs.cs_key = "scs";
      gs.positions_path = "sensor/sensors";
      m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      std::vector<std::int64_t> number(static_cast<std::size_t>(M));
      std::vector<std::string> channel_names;
      for (std::int64_t i = 0; i < M; ++i) {
        number[static_cast<std::size_t>(i)] = i;
        channel_names.push_back("C" + std::to_string(i));
      }
      nxr::bst::LabellingOptions names;   // the channel names, one level a channel (as an importer stores a montage's)
      names.manifold_id = cid;
      names.dictionary = "sensors_names";
      names.description = "the channel names";
      nxr::bst::labelling(sub, "sensor/sensors_names", number, channel_names, names);
      std::vector<std::int64_t> family(static_cast<std::size_t>(M), 0), quality(static_cast<std::size_t>(M), 0);
      quality.back() = 1;
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", family, {"MEG MAG"}, {}}, {"unit", family, {"T"}, {}}, {"quality", quality, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      ManifoldSpec tm;
      tm.name = recording + "_time";
      tm.path = "time/" + recording + "_time";
      tm.type = "line";
      tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      const auto lc = lattice_counts({T});
      if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
      GeometrySpec tg;
      tg.form = "closed_form";
      tg.cs_key = "clock";
      DimensionSpec d;
      d.name = "time";
      d.n_vertices = T;
      d.spacing = 1.0 / sfreq;
      d.origin = 0.0;
      d.unit = "s";
      tg.dimensions.push_back(d);
      tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      FieldSpec rec;
      rec.name = recording;
      rec.path = nxr::io::task::Subject::of(sub).node_path(prod, recording);
      rec.kind = "recording";
      rec.manifold_id = prod;
      rec.data = view(xf, {M, T});
      rec.array.chunks = Shape{M, 700};
      rec.array.compress = false;
      rec.cols.set("unit", "T");
      rec.cols.json("producer_json", Json{{"kind", "recording"}});
      const std::string rid = sub.field(std::move(rec));
      nxr::bst::TimeSpansOptions so;
      so.line_id = tid;
      so.of_field_id = rid;
      so.description = "bad segments";
      nxr::bst::time_spans(sub, "time/" + recording + "_bad_segments", {{bad_start, bad_stop, 0}}, {"BAD"}, so);
    });
  }
  const fs::path folder = ds.folder();
  ds.close();
  return folder;
}

// `read subject` and `list <class>`: io reads the rows and pages them; the type of each node is the type system's, built the way the runner types a stored input (`port_of` over io's
// description, then the JSON `compile` prints): one conversion, not a second one.
namespace {
task::NodeTyper node_typer() {
  return [](const std::string& cls, const nxr::io::finder::Value& v) -> Json {
    nxr::io::finder::Value x = v;
    if (cls == "field" && x.time && x.rate > 0) x.chunked = true;   // a time-resolved field is read as a stream, as the runner's input over the chunk loop is
    const nxr::graph::PortType t = nxr::pipelines::graph::port_of(x);
    if (cls == "manifold" && !t.domain.factors.empty()) return nxr::graph::to_json(t.domain.factors.front());   // a manifold's type is its term
    return nxr::graph::to_json(t);
  };
}
}  // namespace
CommandResult read_subject(const task::ReadSubjectRequest& request, const Progress& progress) {
  task::ReadSubjectRequest q = request;
  q.typer = node_typer();
  return task::read_subject(q, progress);
}
CommandResult list_class(const task::ListClassRequest& request, const Progress& progress) {
  task::ListClassRequest q = request;
  q.typer = node_typer();
  return task::list_class(q, progress);
}

}  // namespace nxr::pipelines::face
