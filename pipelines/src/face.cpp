// nxr-pipelines — what a face asks (face.h). The command line's own calls (pipelines/meeg.h psd_compute, runner.h run, listing.h, io's catalogue and task API) made from a JSON
// description of the arguments, so the Python wheel and the MATLAB binding hold no logic of their own.
#include "nxr/pipelines/face.h"

#include <algorithm>
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
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/listing.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/runner.h"
#include "nxr/welch.h"

namespace nxr::pipelines::face {

using namespace nxr::io;
namespace fs = std::filesystem;
namespace g = nxr::pipelines::graph;

std::string stamp() { return std::string("nxr-neuro ") + NXR_NEURO_VERSION + " / C++17"; }

namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }

// ── the built-in pipeline `psd`: its parameters are PsdOptions' (pipelines/meeg.h), the command line's `run psd` options with the dashes as underscores ──
struct ParamDoc { const char* name; const char* def; const char* meaning; const char* unit; std::vector<const char*> choices; };
const std::vector<ParamDoc>& psd_parameters() {
  static const std::vector<ParamDoc> p = {
      {"recording", "", "the recording (a Field of kind recording) by name; required", "-", {}},
      {"name", "", "the stored statistic's name; default <recording>_psd_welch", "-", {}},
      {"set", "tiles", "the named set of Welch options the other options start from (recorded with the result)", "-", {"tiles", "brainstorm", "scipy", "nxr-chain"}},
      {"segment_seconds", "4.0", "the segment length in seconds (an alternative to segment_samples and segment_level)", "s", {}},
      {"segment_samples", "", "the segment length in samples (an alternative to segment_seconds and segment_level)", "samples", {}},
      {"segment_level", "", "under the set tiles: the segment is 2^segment_level samples (an alternative to the other two)", "-", {}},
      {"overlap", "", "the fraction of a segment shared with the next, in [0, 1); default the set's", "-", {}},
      {"window", "", "the taper; default the set's", "-", {"hann", "hamming", "blackman", "rectangular"}},
      {"periodic", "", "the DFT-even form of the window (true) or the symmetric form (false); default the set's", "-", {}},
      {"detrend", "", "removed from each segment before the window; default the set's", "-", {"none", "constant", "linear"}},
      {"scaling", "", "the density's normalisation; default the set's", "-", {"density", "spectrum", "chain"}},
      {"exclude_bad_segments", "true", "a segment touching one of the recording's bad segments is left out", "-", {}},
      {"through", "", "the name of a stored rotation-plane Operator: the two-sided (rotary) PSD of its z = fx.X + i fy.X instead of the sensors' PSD", "-", {}},
  };
  return p;
}

nxr::pipelines::PsdOptions psd_options(const Json& p) {
  using namespace nxr::welch;
  std::set<std::string> known;
  for (const auto& d : psd_parameters()) known.insert(d.name);
  for (auto it = p.begin(); it != p.end(); ++it)
    if (!known.count(it.key())) {
      std::string all;
      for (const auto& d : psd_parameters()) all += (all.empty() ? "" : ", ") + std::string(d.name);
      refuse("psd has no parameter '" + it.key() + "' (its parameters: " + all + ")");
    }
  auto str = [&](const char* k) -> std::optional<std::string> {
    if (!p.contains(k) || p[k].is_null()) return std::nullopt;
    if (!p[k].is_string()) refuse(std::string(k) + " must be text");
    return p[k].get<std::string>();
  };
  auto num = [&](const char* k) -> std::optional<double> {
    if (!p.contains(k) || p[k].is_null()) return std::nullopt;
    if (!p[k].is_number()) refuse(std::string(k) + " must be a number");
    return p[k].get<double>();
  };
  auto whole = [&](const char* k) -> std::optional<std::int64_t> {
    const auto v = num(k);
    if (!v) return std::nullopt;
    if (*v != std::floor(*v)) refuse(std::string(k) + " must be a whole number");
    return static_cast<std::int64_t>(*v);
  };
  nxr::pipelines::PsdOptions o;
  const auto rec = str("recording");
  if (!rec) refuse("psd needs the recording: recording=NAME");
  o.recording = *rec;
  o.name = str("name");
  o.through = str("through");
  const std::string set = str("set").value_or("tiles");
  o.option_set = set;
  if (set == "tiles" || set == "brainstorm") o.welch = brainstormPsdOptions(1.0);
  else if (set == "scipy") o.welch = scipyWelchOptions(1.0);
  else if (set == "nxr-chain") o.welch = nxrChainWelchOptions(1.0, 1.0);
  else refuse("set must be tiles, brainstorm, scipy or nxr-chain");
  if (const auto v = num("segment_seconds")) o.segment_seconds = *v;
  if (const auto v = whole("segment_samples")) o.segment_samples = *v;
  if (const auto v = whole("segment_level")) o.segment_level = static_cast<int>(*v);
  if ((p.contains("segment_seconds") ? 1 : 0) + (p.contains("segment_samples") ? 1 : 0) + (p.contains("segment_level") ? 1 : 0) > 1)
    refuse("segment_seconds, segment_samples and segment_level are alternatives");
  if (const auto v = num("overlap")) o.welch.overlap = *v;
  if (const auto v = str("window")) {
    if (*v == "hann") o.welch.window = Window::Hann;
    else if (*v == "hamming") o.welch.window = Window::Hamming;
    else if (*v == "blackman") o.welch.window = Window::Blackman;
    else if (*v == "rectangular") o.welch.window = Window::Rectangular;
    else refuse("window must be hann, hamming, blackman or rectangular");
  }
  if (p.contains("periodic") && !p["periodic"].is_null()) {
    if (!p["periodic"].is_boolean()) refuse("periodic must be true or false");
    o.welch.periodic = p["periodic"].get<bool>();
  }
  if (const auto v = str("detrend")) {
    if (*v == "none") o.welch.detrend = Detrend::None;
    else if (*v == "constant") o.welch.detrend = Detrend::Constant;
    else if (*v == "linear") o.welch.detrend = Detrend::Linear;
    else refuse("detrend must be none, constant or linear");
  }
  if (const auto v = str("scaling")) {
    if (*v == "density") o.welch.scaling = Scaling::Density;
    else if (*v == "spectrum") o.welch.scaling = Scaling::Spectrum;
    else if (*v == "chain") o.welch.scaling = Scaling::Chain;
    else refuse("scaling must be density, spectrum or chain");
  }
  if (p.contains("exclude_bad_segments") && !p["exclude_bad_segments"].is_null()) {
    if (!p["exclude_bad_segments"].is_boolean()) refuse("exclude_bad_segments must be true or false");
    o.exclude_bad_segments = p["exclude_bad_segments"].get<bool>();
  }
  return o;
}

Json doc_of(const ParamDoc& d) {
  Json c = Json::array();
  for (const char* x : d.choices) c.push_back(x);
  Json def = nullptr;
  const std::string s = d.def;
  if (!s.empty()) {
    try { def = Json::parse(s); } catch (...) { def = s; }
  }
  return Json{{"name", d.name}, {"default", def}, {"meaning", d.meaning}, {"unit", d.unit}, {"choices", c}};
}

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
  Json psd_params = Json::array();
  for (const auto& d : psd_parameters()) psd_params.push_back(doc_of(d));
  std::vector<Json> all;
  all.push_back(Json{{"name", "psd"}, {"form", "built-in"}, {"summary", "The Welch power spectral density of a recording, bad segments left out."}, {"parameters", psd_params}});
  for (const std::string& n : g::available()) {
    const g::Definition d = g::load(n);
    Json params = Json::array();
    for (const auto& [k, v] : d.parameters.items()) {
      const Json doc = d.parameter_docs.contains(k) ? d.parameter_docs[k] : Json::object();
      params.push_back(Json{{"name", k}, {"default", v}, {"meaning", doc.contains("meaning") ? doc["meaning"] : Json("")}, {"unit", doc.contains("unit") ? doc["unit"] : Json("-")},
                            {"choices", doc.contains("choices") ? doc["choices"] : Json::array()}});
    }
    std::string first = d.description.substr(0, d.description.find_first_of(".\n"));
    all.push_back(Json{{"name", n}, {"form", "definition"}, {"summary", first.empty() ? "Run the pipeline definition " + n + "." : first + "."}, {"parameters", params}});
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
  if (rq.pipeline == "psd") {
    nxr::pipelines::PsdOptions o = psd_options(rq.parameters);
    if (rq.from) { o.start = *rq.from; o.stop = *rq.to; }
    if (rq.all) o.chunks = "all";
    o.replace = rq.replace;
    const auto r = nxr::pipelines::psd_compute(ds, sub, o, say);
    Json info = r.info;
    info["seconds"] = r.seconds;
    all_hit = r.hit;
    runs.push_back(Json{{"cache", r.hit ? "hit" : "computed"}, {"info", info}});
    stored.push_back(Json{{"output", "psd"}, {"name", info.value("name", "")}, {"id", info.value("id", "")}, {"path", info.value("path", "")}});
  } else {
    const g::Definition d = g::load(rq.pipeline);
    for (auto it = rq.parameters.begin(); it != rq.parameters.end(); ++it)
      if (!d.parameters.contains(it.key())) {
        std::string all;
        for (const auto& [k, _] : d.parameters.items()) all += (all.empty() ? "" : ", ") + k;
        refuse(d.name + " has no parameter '" + it.key() + "' (its parameters: " + (all.empty() ? "none" : all) + ")");
      }
    g::RunOptions o;
    if (rq.from) { o.start = *rq.from; o.stop = *rq.to; }
    if (rq.all) o.chunks = "all";
    o.replace = rq.replace;
    for (const Json& r : g::run(ds, sub, d, rq.parameters, o, false, say)) {
      const bool hit = r.value("hit", false);
      all_hit = all_hit && hit;
      runs.push_back(Json{{"cache", hit ? "hit" : "computed"}, {"info", r}});
      if (r.contains("stored"))
        for (const Json& s : r["stored"]) stored.push_back(Json{{"output", s.value("output", "")}, {"name", s.value("output", "")}, {"id", s.value("id", "")}, {"path", s.value("path", "")}});
    }
  }
  return Json{{"pipeline", rq.pipeline}, {"subject", subject}, {"cache", all_hit ? "hit" : "computed"}, {"stored", stored}, {"runs", runs}, {"seconds", seconds_since(t0)}};
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
