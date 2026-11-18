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
#include "nxr/io/task/dataset.h"
#include "nxr/io/task/task.h"
#include "nxr/io/default_subject.h"
#include "nxr/io/selection_store.h"
#include "nxr/measurement/roll_up.h"
#include "nxr/measurement/measures.h"
#include "nxr/pipelines/selection_bridge.h"
#include "nxr/quantity/quantity.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/listing.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/runner.h"
#include "nxr/signal/spectral.h"

namespace nxr::pipelines::face {

using namespace nxr::io;
namespace fs = std::filesystem;
namespace g = nxr::pipelines::graph;

std::string stamp() { return std::string("nxr-neuro ") + NXR_NEURO_VERSION + " / C++17"; }

namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }



// the dataset a run works on, as the command line opens it (cli.h open_dataset): a catalogue's subject package, or a one-database dataset
DatasetWriter open_for(const fs::path& folder, const std::string& subject, bool lock) {
  OpenOptions o;
  o.lock = lock;
  if (is_catalogue_dataset(folder)) return open_subject_package(folder, subject, o, {});
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
  if (given == 0) refuse("a time range is required: give from and to (seconds), or all for the whole recording");
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
    for (auto it = params.begin(); it != params.end(); ++it)
      if (!d.parameters.contains(it.key())) {
        std::string all;
        for (const auto& [k, _] : d.parameters.items()) all += (all.empty() ? "" : ", ") + k;
        refuse(d.name + " has no parameter '" + it.key() + "' (its parameters: " + (all.empty() ? "none" : all) + ")");
      }
    g::RunOptions o;
    if (rq.from) { o.start = *rq.from; o.stop = *rq.to; }
    if (rq.all) o.chunks = "all";
    o.replace = rq.replace;
    o.preset = preset;
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
  DatasetWriter ds = open_for(dataset, subject, false);
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
  // compute: the finest ingredients (of each subject asked), summed code for code across subjects, rolled up through the grid's closed form, the measure derived from them
  std::vector<Json> fines{measurement_ingredients_of(sub, sel->id, field_id.empty() ? std::nullopt : std::optional<std::string>(field_id), q.measure)};
  for (const auto& other : q.subjects) {
    if (other == subject) continue;
    DatasetWriter ods = open_for(dataset, other, false);
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
  DatasetWriter ds = open_writer(dataset, OpenOptions{false, std::nullopt});
  for (const auto& r : task::Dataset::of(ds).subjects()) rows.push_back(Json{{"name", r.name}, {"kind", r.kind}, {"status", r.status}, {"path", r.path}});
  return Json{{"dataset", text(ds.row(), "name")}, {"subjects", rows}};
}

Json results(const fs::path& dataset, const std::string& subject, const std::string& what) {
  DatasetWriter ds = open_for(dataset, subject, false);
  SubjectWriter sub = SubjectWriter::open(ds, subject);
  return list_subject_nodes(ds, sub, what);
}

namespace {

// an array, or the arrays inside a group (a sparse operator is a group of data, indices and indptr), read whole from the subject's store
bool is_array_key(const Store& store, const std::string& key) {
  const std::string t = store.get_text(join_key(key, "zarr.json")).value_or("");
  return t.find("\"node_type\":\"array\"") != std::string::npos || t.find("\"node_type\": \"array\"") != std::string::npos;
}
void collect_arrays(const Store& store, const std::string& key, const std::string& name, std::vector<StoredArray>& out) {
  if (!store.exists(join_key(key, "zarr.json"))) return;
  if (is_array_key(store, key)) {
    const ArrayReader a = ArrayReader::open(store, key);
    StoredArray s;
    s.name = name;
    s.dtype = dtype_to_string(a.dtype());
    s.shape = a.shape();
    s.bytes = a.read();
    out.push_back(std::move(s));
    return;
  }
  for (const auto& k : store.list(key).prefixes) collect_arrays(store, join_key(key, k), name.empty() ? k : name + "/" + k, out);
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

}  // namespace

Stored read(const fs::path& dataset, const std::string& subject, const std::string& name, const std::optional<std::string>& measure, const std::optional<std::string>& of_field) {
  DatasetWriter ds = open_for(dataset, subject, false);
  SubjectWriter sub = SubjectWriter::open(ds, subject);
  const task::Subject ts = task::Subject::of(sub);
  Stored out;
  if (measure) {
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
    return out;
  }
  if (const auto st = ts.by_id_or_name(task::Cls::statistic, name)) {
    const StoredStatistic s = read_statistic(sub, st->name);
    StoredArray a;
    a.name = "values";
    a.dtype = "float64";
    a.shape = s.shape;
    a.bytes.resize(s.values.size() * sizeof(double));
    if (!s.values.empty()) std::memcpy(a.bytes.data(), s.values.data(), a.bytes.size());
    out.arrays.push_back(std::move(a));
    out.meta = Json{{"class", "statistic"}, {"name", st->name}, {"id", st->id}, {"measure", st->row.value("measure", "")}, {"method", st->row.value("method", "")}, {"unit", st->unit ? Json(*st->unit) : Json()},
                    {"axes", s.axes}, {"components", s.components}, {"span", s.span}, {"options", s.options}, {"producer", s.producer}, {"scaling", s.scaling}};
    return out;
  }
  for (const task::Cls c : {task::Cls::field, task::Cls::operator_, task::Cls::selection}) {
    const auto n = ts.by_id_or_name(c, name);
    if (!n) continue;
    const std::optional<std::string> where = c == task::Cls::selection ? n->array_path : n->path;
    if (!where) throw Error(code::unsupported, "'" + name + "' has no stored array (it is virtual: an analytic operator, a product, or a selection of spans)");
    collect_arrays(sub.store(), *where, "", out.arrays);
    for (auto& a : out.arrays) if (a.name.empty()) a.name = "values";
    out.meta = Json{{"class", task::to_string(c)}, {"name", n->name}, {"id", n->id}, {"kind", n->kind ? Json(*n->kind) : Json()}, {"unit", n->unit ? Json(*n->unit) : Json()},
                    {"value_type", n->value_type ? Json(*n->value_type) : Json()}, {"producer", n->producer}, {"params", n->params}};
    return out;
  }
  throw Error(code::no_such_row, "subject " + subject + " holds no statistic, field, operator or selection named '" + name + "'");
}

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
  DatasetWriter ds = create_dataset(datastore, name);
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

}  // namespace nxr::pipelines::face
