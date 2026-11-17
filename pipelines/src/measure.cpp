// nxr-pipelines — measure compute: stored Fields measured over a partition's members (measure.h). The arithmetic is nxr::partition (the Studio's
// measureOverPartition); the rows are the Studio's (`selection_measurement`, written through nxr::io::write_measurements, the port of its repository);
// the recipe is an entry of the dataset's `recipes` dictionary as the Studio's `recipes.save` writes one (adapters/nxr/recipes.ts, recipe.ts entryOfRecipe).
#include "nxr/pipelines/measure.h"
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

#include "nxr/delaunay.h"
#include "nxr/io/measures.h"
#include "nxr/ladder.h"
#include "nxr/partition.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }

const char* kRecipes = "recipes";
const char* kRecipesDescription = "the recipes of this dataset (nxr.recipe): one entry a version, the recipe in params_json";

// the Studio's stableJson / Python's json.dumps(sort_keys=True, separators=(',', ':')): keys sorted at every depth
Json sorted(const Json& v) {
  if (v.is_object()) {
    std::map<std::string, Json> m;
    for (auto it = v.begin(); it != v.end(); ++it) m[it.key()] = sorted(it.value());
    Json o = Json::object();
    for (auto& [k, x] : m) o[k] = x;
    return o;
  }
  if (v.is_array()) { Json a = Json::array(); for (const auto& x : v) a.push_back(sorted(x)); return a; }
  return v;
}

std::vector<double> read_doubles(SubjectWriter& sub, const std::string& path, std::int64_t expect, const std::string& what) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  std::int64_t n = 1;
  for (auto s : r.shape()) n *= s;
  if (r.shape().size() != 1 || n != expect) invalid(what + ": a value per vertex is [" + std::to_string(expect) + "], this array has another shape");
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<double> out(static_cast<size_t>(n));
  if (r.dtype() == DType::Float64) std::memcpy(out.data(), raw.data(), out.size() * 8);
  else if (r.dtype() == DType::Float32) { std::vector<float> f(out.size()); std::memcpy(f.data(), raw.data(), f.size() * 4); std::copy(f.begin(), f.end(), out.begin()); }
  else invalid(what + ": a scalar Field is float32 or float64");
  return out;
}

std::vector<std::int32_t> read_codes(SubjectWriter& sub, const std::string& path, std::int64_t expect, const std::string& what) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  std::int64_t n = 1;
  for (auto s : r.shape()) n *= s;
  if (n != expect) invalid(what + ": a partition has a code per vertex (" + std::to_string(expect) + "), this one has " + std::to_string(n));
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<std::int32_t> out(static_cast<size_t>(n));
  if (r.dtype() == DType::Int32) std::memcpy(out.data(), raw.data(), out.size() * 4);
  else if (r.dtype() == DType::Int64) { std::vector<std::int64_t> v(out.size()); std::memcpy(v.data(), raw.data(), v.size() * 8); std::transform(v.begin(), v.end(), out.begin(), [](std::int64_t x) { return static_cast<std::int32_t>(x); }); }
  else if (r.dtype() == DType::Int16) { std::vector<std::int16_t> v(out.size()); std::memcpy(v.data(), raw.data(), v.size() * 2); std::copy(v.begin(), v.end(), out.begin()); }
  else invalid(what + ": a partition's codes are integers");
  return out;
}

// the unit of a measure of a field in `u` (the Studio's measureUnit and the registry: a square for mean_square, the weight's unit times u for a sum)
std::optional<std::string> unit_of(const std::string& measure, const std::optional<std::string>& u, bool areas) {
  if (measure == "area") return std::string("m²");
  if (measure == "n_elements") return std::nullopt;
  if (!u || u->empty()) return std::nullopt;
  if (measure == "mean_square") return "(" + *u + ")²";
  if (measure == "sum") return areas ? *u + "·m²" : *u;
  return u;
}
}  // namespace

MeasureComputeResult measure_compute(DatasetWriter& ds, SubjectWriter& sub, const MeasureComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  if (opt.weight != "none" && opt.weight != "barycentric" && opt.weight != "voronoi") invalid("--weight is one of none, barycentric, voronoi (it changes the numbers, so it is always stated)");
  if (opt.labelling.empty()) invalid("--labelling names the partition (a set Selection with a code per vertex)");
  if (opt.fields.empty()) invalid("--fields is a pattern over the Field names to measure");
  std::vector<std::string> measures = opt.measures.empty() ? nxr::partition::field_measures() : opt.measures;
  for (const auto& m : measures)
    if (m != "median" && (!nxr::partition::known_measure(m) || m == "area" || m == "n_elements")) invalid("--measures: " + m + " is not one this producer writes per Field (mean, mean_square, max, min, abs_max, sum, median)");
  // the median is exact and unweighted (nxr::partition::medians;); a weighted median is not built
  const bool want_median = std::find(measures.begin(), measures.end(), "median") != measures.end();
  if (want_median && opt.weight != "none") invalid("--measures median: a weighted median is not built; the median is exact and unweighted, so it needs --weight none");
  std::regex re;
  try { re = std::regex(opt.fields, std::regex::ECMAScript); } catch (const std::regex_error& e) { invalid("--fields " + opt.fields + ": not a regular expression (" + e.what() + ")"); }

  // THE PARTITION and its surface
  const auto sels = sub.db().all("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'set' AND array_path IS NOT NULL AND status = 'complete'", {V(sub.id()), V(opt.labelling)});
  if (sels.empty()) throw Error(code::no_such_row, sub.name() + ": no partition '" + opt.labelling + "' (a set Selection with a code per element)");
  if (sels.size() > 1) invalid(sub.name() + ": " + std::to_string(sels.size()) + " partitions are called '" + opt.labelling + "'");
  const Row& sel = sels[0];
  if (is_null(sel, "manifold_id")) invalid(opt.labelling + ": a partition on a space has no vertices to measure");
  const auto man = sub.db().read("manifold", text(sel, "manifold_id"));
  if (!man || text(*man, "type") != "surface") invalid(opt.labelling + ": its manifold is not a surface");
  const SurfaceInput S = read_surface(sub, text(*man, "name"), true);
  const std::int64_t nV = S.n_vertices();
  const std::vector<std::int32_t> codes = read_codes(sub, text(sel, "array_path"), nV, opt.labelling);
  const int members = codes.empty() ? 0 : *std::max_element(codes.begin(), codes.end()) + 1;

  // THE WEIGHTS
  std::vector<double> w;
  if (opt.weight == "barycentric") {
    nxr::ladder::Mesh m; m.positions = S.vertices; m.faces = S.faces; m.n = static_cast<int>(nV);
    w = nxr::ladder::vertexAreas(m);
  } else if (opt.weight == "voronoi") {
    namespace dl = nxr::manifold::delaunay;
    dl::Mesh m = dl::prepare(S.vertices.data(), static_cast<int>(nV), S.faces.data(), static_cast<int>(S.n_faces()));
    w = m.mass.empty() ? dl::mixedVoronoiMass(m) : m.mass;
  }
  const bool areas = !w.empty();
  const double* wp = areas ? w.data() : nullptr;

  // THE RECIPES: every choice that changes a number (not the inputs: the labelling and the Fields are the rows' keys). One recipe a FAMILY of
  // measurements, so a choice of one family never makes another's rows "another recipe's": `<name>` the scalar measures and the members' sizes,
  // `<name>-gradient` a scalar Field's gradient, `<name>-tensor` a hermitian3 Field's tensor_* sums; each a new version when its own choices change.
  struct Recipe { std::string name; int version = 1; std::string tag, by; };
  auto recipe_for = [&](const std::string& rn, const Json& choice_in, const std::string& description) {
    const Json choice = sorted(choice_in);
    Recipe r; r.name = rn;
    const auto dict = sub.db().one("SELECT id FROM dictionary WHERE scope = 'dataset' AND dataset_id = ? AND name = ?", {V(ds.id()), V(std::string(kRecipes))});
    if (dict) {
      int top = 0;
      bool found = false;
      for (const Row& e : sub.db().all("SELECT * FROM dictionary_entry WHERE dictionary_id = ?", {V(text(*dict, "id"))})) {
        const Json a = Json::parse(opt_text(e, "attributes_json").value_or("{}"), nullptr, false);
        if (!a.is_object() || a.value("recipe", std::string()) != rn) continue;
        const int v = a.value("version", 0);
        top = std::max(top, v);
        const Json pp = Json::parse(opt_text(e, "params_json").value_or("{}"), nullptr, false);
        if (pp.is_object() && pp.contains("measure") && sorted(pp["measure"]) == choice) { r.version = v; found = true; }
      }
      if (!found) r.version = top + 1;
    }
    const Json recipe = sorted(Json{{"format", 1}, {"name", rn}, {"version", r.version}, {"measure", choice}, {"description", description}, {"provenance", Json{{"made_by", "nxr-neuro"}}}});
    DictionaryEntry e;
    e.name = rn + "@" + std::to_string(r.version);
    e.cols.set("function", std::string("reduction"));
    e.cols.set("params_json", jdump(recipe));
    e.cols.set("attributes_json", jdump(Json{{"recipe", rn}, {"version", r.version}, {"format", 1}}));
    DictionarySpec spec;
    spec.name = kRecipes; spec.scope = "dataset"; spec.description = kRecipesDescription; spec.entries = std::vector<DictionaryEntry>{e};
    sub.dictionary(spec);
    r.tag = "recipe " + rn + "@" + std::to_string(r.version);
    r.by = r.tag + " \xc2\xb7 nxr-neuro " + NXR_NEURO_VERSION;
    return r;
  };
  const std::string rname = opt.recipe.value_or("partition-" + opt.weight);
  const Recipe base = recipe_for(rname, Json{{"kind", "reduction"}, {"method", "partition"}, {"weight", opt.weight}, {"measures", measures}},
                                 "a Field measured over a partition's members (nxr-neuro measure compute; the Studio's measureOverPartition), weights: " + opt.weight);
  std::optional<Recipe> grad_recipe, tensor_recipe;
  auto gradient_recipe = [&]() -> const Recipe& {
    if (!grad_recipe) grad_recipe = recipe_for(rname + "-gradient", Json{{"kind", "reduction"}, {"method", "partition-gradient"}, {"weight", opt.weight}, {"gauge", "canonical frames"},
        {"gradient", "nxr measure.vectors.vertex_gradient: the area-weighted face gradients, projected on the vertex's tangent plane"}, {"measures", "the registry's gradient_*, over the frames' regular vertices"}},
        "the gradient of a scalar Field measured over a partition's members: the registry's gradient_* sums in the canonical frames, weights: " + opt.weight);
    return *grad_recipe;
  };
  auto tensor_recipe_of = [&]() -> const Recipe& {
    if (!tensor_recipe) tensor_recipe = recipe_for(rname + "-tensor", Json{{"kind", "reduction"}, {"method", "partition-tensor"}, {"weight", opt.weight}, {"gauge", "canonical frames"},
        {"measures", "the registry's tensor_*: area-weighted sums of the tensor's components in (e1 north, e2 west, n); total over every vertex, the components over the frames' regular ones"}},
        "a cross-spectral tensor measured over a partition's members: the registry's tensor_* sums in the canonical frames, weights: " + opt.weight);
    return *tensor_recipe;
  };
  const std::string tag = base.tag, by = base.by;
  const int version = base.version;
  const std::string stamp = now_utc();
  const std::string sid = text(sel, "id");

  // one recipe's rows at a time per (labelling, Field, measure): the same recipe is a cache hit, another is refused unless replace
  // a map of a statistic (2026-11-13): its rows name the statistic and the slice (of_statistic_id, of_index) where a Field's name the Field
  const bool stat_cols = has_column(sub.db(), "selection_measurement", "of_statistic_id");
  std::optional<std::pair<std::string, std::int64_t>> slice;      // set while a map slice is measured
  auto existing = [&](const std::optional<std::string>& fid, const std::vector<std::string>& ms) {
    std::string q = "SELECT computed_by, COUNT(*) AS n FROM selection_measurement WHERE selection_id = ? AND ";
    std::vector<Value> a = {V(sid)};
    if (slice) { q += "of_statistic_id = ? AND of_index = ?"; a.push_back(V(slice->first)); a.push_back(V(slice->second)); }
    else { q += fid ? "of_field_id = ?" : "of_field_id IS NULL"; if (fid) a.push_back(V(*fid)); if (stat_cols) q += " AND of_statistic_id IS NULL"; }
    q += " AND measure IN (";
    for (size_t i = 0; i < ms.size(); ++i) { q += (i ? ",?" : "?"); a.push_back(V(ms[i])); }
    return sub.db().all(q + ") GROUP BY computed_by", a);
  };
  auto settle = [&](const std::optional<std::string>& fid, const std::string& what, const std::vector<std::string>& ms, const Recipe& rec) -> bool {   // true: write
    const std::string& tag = rec.tag;
    const auto have = existing(fid, ms);
    if (have.empty()) return true;
    bool ours = true;
    for (const Row& r : have) if (opt_text(r, "computed_by").value_or("").rfind(tag + " ", 0) != 0) ours = false;
    if (ours && !opt.replace) return false;
    if (!fid && !slice && !opt.replace) {   // the members' sizes depend on the weight rule alone: another recipe's, made with the same weights, are these
      bool same = true;
      for (const Row& r : have) {
        const std::string other = opt_text(r, "computed_by").value_or("");
        const std::string rv = other.rfind("recipe ", 0) == 0 ? other.substr(7, other.find(' ', 7) - 7) : "";
        const auto e = rv.empty() ? std::nullopt : sub.db().one("SELECT params_json FROM dictionary_entry WHERE name = ? AND dictionary_id IN (SELECT id FROM dictionary WHERE name = 'recipes')", {V(rv)});
        const Json pj = e ? Json::parse(opt_text(*e, "params_json").value_or("{}"), nullptr, false) : Json();
        if (!pj.is_object() || !pj.contains("measure") || pj["measure"].value("weight", std::string()) != opt.weight) same = false;
      }
      if (same) return false;
    }
    if (!ours && !opt.replace) {
      std::string other;
      for (const Row& r : have) if (opt_text(r, "computed_by").value_or("").rfind(tag + " ", 0) != 0) { other = opt_text(r, "computed_by").value_or("(no recipe)"); break; }
      throw Error(code::invalid_argument, what + " on " + opt.labelling + " already has measurements by " + other + "; --replace replaces them with " + tag);
    }
    for (const auto& m : ms) {
      Json where = {{"selectionId", sid}, {"measure", m}};
      if (slice) { delete_measurements(sub.db(), Json{{"selectionId", sid}, {"measure", m}, {"ofStatisticId", slice->first}, {"ofIndex", slice->second}}); continue; }
      where["ofFieldId"] = fid ? Json(*fid) : Json(nullptr);
      if (fid) delete_measurements(sub.db(), where);
      else sub.db().execute("DELETE FROM selection_measurement WHERE selection_id = ? AND of_field_id IS NULL AND measure = ?", {V(sid), V(m)});
    }
    return true;
  };

  Json warnings = Json::array(), fields_out = Json::array();
  std::int64_t written = 0;
  // THE MEMBERS themselves (geometric: no Field)
  const nxr::partition::Sums geo = nxr::partition::accumulate(codes.data(), codes.size(), members, nullptr, wp);
  if (geo.unassigned) warnings.push_back(std::to_string(geo.unassigned) + " vertices have a negative code: in no member, left out");
  std::vector<std::string> geo_ms = {"n_elements"};
  if (areas) geo_ms.push_back("area");
  // both size measures are one recipe's: a recipe without areas (none) also takes the place of another's `area`
  if (settle(std::nullopt, "the members' sizes", {"n_elements", "area"}, base)) {
    Json rows = Json::array();
    for (int k = 0; k < members; ++k)
      for (const auto& m : geo_ms) {
        const double v = nxr::partition::measure(geo, k, m);
        if (std::isnan(v)) continue;
        rows.push_back(Json{{"selection_id", sid}, {"code", k}, {"of_field_id", nullptr}, {"measure", m}, {"value", v}, {"weight", geo.weight[static_cast<size_t>(k)]},
                            {"unit", unit_of(m, std::nullopt, areas) ? Json(*unit_of(m, std::nullopt, areas)) : Json(nullptr)}, {"computed_utc", stamp}, {"computed_by", by}});
      }
    sub.db().tx([&] { written += write_measurements(sub.db(), rows, by); });
  }
  std::int64_t empty = 0;
  for (int k = 0; k < members; ++k) if (geo.count[static_cast<size_t>(k)] == 0) ++empty;
  if (empty) warnings.push_back(std::to_string(empty) + " codes below " + std::to_string(members) + " have no vertex: their means are written as 0, as the reference does");

  // A TENSOR Field (hermitian3, [nV x 3 x 3 x 2]): the registry's tensor_* — per member, the area-weighted sums of its components in the canonical
  // frames, P = E T E^T with E's rows e1 (north), e2 (west), n = e1 x e2 (nxr measure.vectors.band_tensor's layout): 11 22 nn, 12 1n 2n (real and
  // imaginary), and the total (the trace, over EVERY vertex of the member; the components over the frames' regular vertices only, as nxr's).
  static const std::vector<std::string> kTensor = {"tensor_11", "tensor_22", "tensor_nn", "tensor_12_re", "tensor_12_im", "tensor_1n_re", "tensor_1n_im", "tensor_2n_re", "tensor_2n_im", "tensor_total"};
  std::optional<StoredFrames> frames;
  // a hermitian3 map (a Field, or a band of a statistic: `slice` set), its values [nV x 18] loaded by `load`
  auto measure_tensor = [&](const std::string& fname, const std::optional<std::string>& fid_opt, const std::optional<std::string>& funit, const std::function<std::vector<float>()>& load) {
    const std::string fid = fid_opt.value_or("");
    const Recipe& tr = tensor_recipe_of();
    const std::string& by = tr.by;
    if (!settle(fid_opt, fname, kTensor, tr)) { fields_out.push_back(Json{{"name", fname}, {"cache", "hit"}}); return; }
    if (!frames) {
      frames = read_stored_frames(sub, S.name, nV);
      if (!frames->canonical) invalid(S.name + " has no canonical frames: a tensor is measured in them (`nxr-neuro frames compute` first)");
    }
    const std::vector<float> T = load();
    const std::int64_t cnt = static_cast<std::int64_t>(T.size());
    if (cnt != nV * 18) invalid(fname + ": a hermitian3 map is [" + std::to_string(nV) + " x 3 x 3 x 2]");
    std::vector<std::array<double, 10>> sums(static_cast<size_t>(members));
    for (auto& a : sums) a.fill(0.0);
    std::vector<double> mw(static_cast<size_t>(members), 0.0);
    std::int64_t singular = 0;
    for (std::int64_t v = 0; v < nV; ++v) {
      const int k = codes[static_cast<size_t>(v)];
      if (k < 0) continue;
      const double w = wp ? wp[v] : 1.0;
      auto re = [&](int i, int j) { return static_cast<double>(T[static_cast<size_t>(18 * v + 2 * (3 * i + j))]); };
      auto im = [&](int i, int j) { return static_cast<double>(T[static_cast<size_t>(18 * v + 2 * (3 * i + j) + 1)]); };
      mw[static_cast<size_t>(k)] += w;
      sums[static_cast<size_t>(k)][9] += w * (re(0, 0) + re(1, 1) + re(2, 2));
      const Eigen::Vector3d e1(frames->e1(v, 0), frames->e1(v, 1), frames->e1(v, 2)), e2(frames->e2(v, 0), frames->e2(v, 1), frames->e2(v, 2));
      if (!e1.allFinite() || !e2.allFinite()) { ++singular; continue; }
      const Eigen::Vector3d n = e1.cross(e2);
      const Eigen::Vector3d E[3] = {e1, e2, n};
      auto P = [&](int a, int b, bool imag) { double x = 0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) x += E[a][i] * (imag ? im(i, j) : re(i, j)) * E[b][j]; return x; };
      const double c[9] = {P(0, 0, false), P(1, 1, false), P(2, 2, false), P(0, 1, false), P(0, 1, true), P(0, 2, false), P(0, 2, true), P(1, 2, false), P(1, 2, true)};
      for (int i = 0; i < 9; ++i) sums[static_cast<size_t>(k)][static_cast<size_t>(i)] += w * c[i];
    }
    if (singular) warnings.push_back(fname + ": " + std::to_string(singular) + " vertices where the canonical frames are singular: in tensor_total, not in the components");
    const auto u = unit_of("sum", funit, areas);
    Json rows = Json::array();
    for (int k = 0; k < members; ++k) {
      if (mw[static_cast<size_t>(k)] <= 0) continue;
      for (std::size_t i = 0; i < kTensor.size(); ++i) {
        Json row{{"selection_id", sid}, {"code", k}, {"measure", kTensor[i]}, {"value", sums[static_cast<size_t>(k)][i]}, {"weight", mw[static_cast<size_t>(k)]},
                 {"unit", u ? Json(*u) : Json(nullptr)}, {"computed_utc", stamp}, {"computed_by", by}};
        if (slice) { row["of_statistic_id"] = slice->first; row["of_index"] = slice->second; } else row["of_field_id"] = fid;
        rows.push_back(row);
      }
    }
    std::int64_t n = 0;
    sub.db().tx([&] { n = write_measurements(sub.db(), rows, by); });
    written += n;
    fields_out.push_back(Json{{"name", fname}, {"cache", "miss"}, {"rows", n}, {"kind", "tensor"}});
    if (progress) progress(Json{{"stage", "field"}, {"name", fname}, {"rows", n}});
  };

  // THE GRADIENT of a scalar Field (nxr's vertex_gradient and vector_stats): per member, over the frames' regular vertices, the registry's
  // gradient_weight Σa, gradient_1 · gradient_2 · gradient_n Σa·(∇x)_c, gradient_11 · gradient_22 · gradient_12 Σa·(∇x)_c (∇x)_c', gradient_magnitude Σa·|∇x|_tangent
  static const std::vector<std::string> kGrad = {"gradient_weight", "gradient_1", "gradient_2", "gradient_n", "gradient_11", "gradient_22", "gradient_12", "gradient_magnitude"};
  auto measure_gradient = [&](const std::string& fid, const std::string& fname, const std::vector<double>& x, const std::optional<std::string>& funit) {
    const Recipe& gr = gradient_recipe();
    const std::string& by = gr.by;
    if (!settle(fid, fname + " (its gradient)", kGrad, gr)) { fields_out.push_back(Json{{"name", fname}, {"gradient", "hit"}}); return; }
    if (!frames) {
      frames = read_stored_frames(sub, S.name, nV);
      if (!frames->canonical) invalid(S.name + " has no canonical frames: a gradient is measured in them (`nxr-neuro frames compute` first)");
    }
    // the face gradients, area-weighted onto the vertices
    std::vector<Eigen::Vector3d> acc(static_cast<size_t>(nV), Eigen::Vector3d::Zero());
    std::vector<double> wsum(static_cast<size_t>(nV), 0.0);
    auto pos = [&](std::int32_t v) { return Eigen::Vector3d(S.vertices[static_cast<size_t>(3 * v)], S.vertices[static_cast<size_t>(3 * v + 1)], S.vertices[static_cast<size_t>(3 * v + 2)]); };
    for (std::int64_t f = 0; f < S.n_faces(); ++f) {
      const std::int32_t i[3] = {S.faces[static_cast<size_t>(3 * f)], S.faces[static_cast<size_t>(3 * f + 1)], S.faces[static_cast<size_t>(3 * f + 2)]};
      const Eigen::Vector3d p0 = pos(i[0]), p1 = pos(i[1]), p2 = pos(i[2]);
      const Eigen::Vector3d nn = (p1 - p0).cross(p2 - p0);
      const double A2 = nn.norm();
      if (!(A2 > 0)) continue;
      const Eigen::Vector3d nh = nn / A2;
      // ∇φ_k = n × (the edge opposite k) / (2A)
      const Eigen::Vector3d g = (nh.cross(p2 - p1) * x[static_cast<size_t>(i[0])] + nh.cross(p0 - p2) * x[static_cast<size_t>(i[1])] + nh.cross(p1 - p0) * x[static_cast<size_t>(i[2])]) / A2;
      for (int k = 0; k < 3; ++k) { acc[static_cast<size_t>(i[k])] += g * (A2 / 2); wsum[static_cast<size_t>(i[k])] += A2 / 2; }
    }
    std::vector<std::array<double, 8>> sums(static_cast<size_t>(members));
    for (auto& a : sums) a.fill(0.0);
    std::int64_t singular = 0;
    for (std::int64_t v = 0; v < nV; ++v) {
      const int k = codes[static_cast<size_t>(v)];
      if (k < 0) continue;
      const Eigen::Vector3d e1(frames->e1(v, 0), frames->e1(v, 1), frames->e1(v, 2)), e2(frames->e2(v, 0), frames->e2(v, 1), frames->e2(v, 2));
      if (!e1.allFinite() || !e2.allFinite() || !(wsum[static_cast<size_t>(v)] > 0)) { ++singular; continue; }
      const Eigen::Vector3d n = e1.cross(e2);
      Eigen::Vector3d gv = acc[static_cast<size_t>(v)] / wsum[static_cast<size_t>(v)];
      gv -= gv.dot(n) * n;                                                 // on the vertex's tangent plane
      const double c1 = gv.dot(e1), c2 = gv.dot(e2), cn = gv.dot(n), a = wp ? wp[v] : 1.0;
      auto& s = sums[static_cast<size_t>(k)];
      s[0] += a; s[1] += a * c1; s[2] += a * c2; s[3] += a * cn; s[4] += a * c1 * c1; s[5] += a * c2 * c2; s[6] += a * c1 * c2; s[7] += a * std::hypot(c1, c2);
    }
    if (singular) warnings.push_back(fname + " (its gradient): " + std::to_string(singular) + " vertices where the canonical frames are singular, left out");
    const std::string u = funit.value_or("");
    auto unit_g = [&](std::size_t i) -> Json {
      if (i == 0) return areas ? Json("m\xc2\xb2") : Json(nullptr);
      if (u.empty()) return Json(nullptr);
      if (i >= 4 && i <= 6) return Json("(" + u + "/m)\xc2\xb2" + (areas ? "\xc2\xb7m\xc2\xb2" : ""));
      return Json("(" + u + "/m)" + (areas ? "\xc2\xb7m\xc2\xb2" : ""));
    };
    Json rows = Json::array();
    for (int k = 0; k < members; ++k) {
      if (!(sums[static_cast<size_t>(k)][0] > 0)) continue;
      for (std::size_t i = 0; i < kGrad.size(); ++i)
        rows.push_back(Json{{"selection_id", sid}, {"code", k}, {"of_field_id", fid}, {"measure", kGrad[i]}, {"value", sums[static_cast<size_t>(k)][i]}, {"weight", sums[static_cast<size_t>(k)][0]},
                            {"unit", unit_g(i)}, {"computed_utc", stamp}, {"computed_by", by}});
    }
    std::int64_t n = 0;
    sub.db().tx([&] { n = write_measurements(sub.db(), rows, by); });
    written += n;
    fields_out.push_back(Json{{"name", fname}, {"gradient", "miss"}, {"rows", n}});
  };

  // THE FIELDS
  const auto frows = sub.db().all("SELECT * FROM field WHERE subject_id = ? AND manifold_id = ? AND status = 'complete' AND path IS NOT NULL ORDER BY name", {V(sub.id()), V(S.id)});
  int matched = 0;
  for (const Row& f : frows) {
    const std::string fname = text(f, "name");
    if (!std::regex_match(fname, re)) continue;
    const std::string vt = opt_text(f, "value_type").value_or("scalar");
    if (vt == "hermitian3") {
      ++matched;
      measure_tensor(fname, text(f, "id"), opt_text(f, "unit"), [&] {
        ArrayReader r = ArrayReader::open(sub.store(), text(f, "path"));
        if (r.dtype() != DType::Float32) invalid(fname + ": a hermitian3 Field is float32");
        const std::vector<std::uint8_t> raw = r.read();
        std::vector<float> T(raw.size() / 4); std::memcpy(T.data(), raw.data(), raw.size());
        return T;
      });
      continue;
    }
    if (vt != "scalar") { warnings.push_back(fname + ": a " + vt + " Field is not measured (scalar and hermitian3 are)"); continue; }
    ++matched;
    const std::string fid = text(f, "id");
    const std::optional<std::string> funit = opt_text(f, "unit");
    if (!settle(fid, fname, measures, base)) {
      fields_out.push_back(Json{{"name", fname}, {"cache", "hit"}});
      if (opt.gradient) measure_gradient(fid, fname, read_doubles(sub, text(f, "path"), nV, fname), funit);
      continue;
    }
    const std::vector<double> x = read_doubles(sub, text(f, "path"), nV, fname);
    if (opt.gradient) measure_gradient(fid, fname, x, funit);
    const nxr::partition::Sums s = nxr::partition::accumulate(codes.data(), codes.size(), members, x.data(), wp);
    const std::vector<double> med = want_median ? nxr::partition::medians(codes.data(), codes.size(), members, x.data()) : std::vector<double>();
    std::int64_t bad = 0;
    for (auto c : s.nonfinite) bad += c;
    if (bad) warnings.push_back(fname + ": " + std::to_string(bad) + " non-finite values; the members holding them are not written");
    Json rows = Json::array();
    for (int k = 0; k < members; ++k)
      for (const auto& m : measures) {
        const double v = m == "median" ? med[static_cast<size_t>(k)] : nxr::partition::measure(s, k, m);
        if (std::isnan(v)) continue;
        const auto u = unit_of(m, funit, areas);
        rows.push_back(Json{{"selection_id", sid}, {"code", k}, {"of_field_id", fid}, {"measure", m}, {"value", v}, {"weight", s.weight[static_cast<size_t>(k)]},
                            {"unit", u ? Json(*u) : Json(nullptr)}, {"computed_utc", stamp}, {"computed_by", by}});
      }
    std::int64_t n = 0;
    sub.db().tx([&] { n = write_measurements(sub.db(), rows, by); });
    written += n;
    fields_out.push_back(Json{{"name", fname}, {"cache", "miss"}, {"rows", n}});
    if (progress) progress(Json{{"stage", "field"}, {"name", fname}, {"rows", n}});
  }
  // THE MAPS OF STATISTICS on the surface (2026-11-13: a band of a map statistic, `<statistic>_<band>`, read as a map; its rows name the statistic and the band)
  for (const MapSlice& ms : map_slices(sub.db(), sub.id(), S.id)) {
    if (!std::regex_match(ms.name, re)) continue;
    if (ms.value_type != "scalar" && ms.value_type != "hermitian3") { warnings.push_back(ms.name + ": a " + ms.value_type + " map is not measured (scalar and hermitian3 are)"); continue; }
    ++matched;
    slice = std::make_pair(text(ms.statistic, "id"), ms.index);
    if (ms.value_type == "hermitian3") {                        // a band of a tensor statistic: the registry's tensor_* sums, as a hermitian3 Field's
      measure_tensor(ms.name, std::nullopt, ms.unit, [&] { const auto d = read_map_slice(sub, ms); return std::vector<float>(d.begin(), d.end()); });
      slice.reset();
      continue;
    }
    if (opt.gradient) warnings.push_back(ms.name + ": the gradient of a statistic's map is not measured (Fields only)");
    if (!settle(std::nullopt, ms.name, measures, base)) { fields_out.push_back(Json{{"name", ms.name}, {"statistic", text(ms.statistic, "name")}, {"cache", "hit"}}); slice.reset(); continue; }
    if (ms.n != nV) throw Error(code::invalid_argument, ms.name + ": " + std::to_string(ms.n) + " values for the " + std::to_string(nV) + " vertices of " + S.name);
    const std::vector<double> x = read_map_slice(sub, ms);
    const nxr::partition::Sums sm = nxr::partition::accumulate(codes.data(), codes.size(), members, x.data(), wp);
    const std::vector<double> med = want_median ? nxr::partition::medians(codes.data(), codes.size(), members, x.data()) : std::vector<double>();
    std::int64_t bad = 0;
    for (auto c : sm.nonfinite) bad += c;
    if (bad) warnings.push_back(ms.name + ": " + std::to_string(bad) + " non-finite values; the members holding them are not written");
    Json rows = Json::array();
    for (int k = 0; k < members; ++k)
      for (const auto& m : measures) {
        const double v = m == "median" ? med[static_cast<size_t>(k)] : nxr::partition::measure(sm, k, m);
        if (std::isnan(v)) continue;
        const auto u = unit_of(m, ms.unit, areas);
        rows.push_back(Json{{"selection_id", sid}, {"code", k}, {"of_statistic_id", text(ms.statistic, "id")}, {"of_index", ms.index}, {"measure", m}, {"value", v},
                            {"weight", sm.weight[static_cast<size_t>(k)]}, {"unit", u ? Json(*u) : Json(nullptr)}, {"computed_utc", stamp}, {"computed_by", by}});
      }
    std::int64_t n = 0;
    sub.db().tx([&] { n = write_measurements(sub.db(), rows, by); });
    written += n;
    fields_out.push_back(Json{{"name", ms.name}, {"statistic", text(ms.statistic, "name")}, {"cache", "miss"}, {"rows", n}});
    if (progress) progress(Json{{"stage", "field"}, {"name", ms.name}, {"rows", n}});
    slice.reset();
  }
  if (!matched) throw Error(code::no_such_row, sub.name() + ": no scalar or hermitian3 Field, nor a map of a statistic, on " + S.name + " matches '" + opt.fields + "'");
  drain(ds.db(), ds.node_store());

  MeasureComputeResult out;
  out.info = Json{{"stage", "result"}, {"labelling", opt.labelling}, {"surface", S.name}, {"members", members}, {"weight", opt.weight}, {"measures", measures},
                  {"recipe", Json{{"name", rname}, {"version", version}}}, {"computed_by", by},
                  {"recipes", [&] { Json r = Json::array({base.name + "@" + std::to_string(base.version)}); if (grad_recipe) r.push_back(grad_recipe->name + "@" + std::to_string(grad_recipe->version));
                                     if (tensor_recipe) r.push_back(tensor_recipe->name + "@" + std::to_string(tensor_recipe->version)); return r; }()}, {"fields", fields_out}, {"rows", written}, {"warnings", warnings}};
  return out;
}

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
  nxr::spectral::Target target;
  try { target = nxr::spectral::parseTarget(opt.scaling); } catch (const std::invalid_argument& e) { invalid(std::string("--scaling ") + e.what()); }
  // how a statistic of a Field follows its values: 1 (a factor: mean, max, …, sd), 2 (the factor squared: mean_square, variance), 0 (a ratio, unchanged)
  auto power_of = [](const std::string& st) -> int {
    static const std::set<std::string> one = {"mean", "max", "min", "sum", "abs_max", "sd", "rms", "tensor_11", "tensor_22", "tensor_nn", "tensor_12_re", "tensor_12_im",
                                              "tensor_1n_re", "tensor_1n_im", "tensor_2n_re", "tensor_2n_im", "tensor_total", "power_density"};
    static const std::set<std::string> two = {"mean_square", "variance"};
    static const std::set<std::string> zero = {"normal_share", "anisotropy", "circularity", "axis_deg"};
    return one.count(st) ? 1 : two.count(st) ? 2 : zero.count(st) ? 0 : -1;
  };
  if (target != nxr::spectral::Target::Stored)
    for (const auto& st : opt.statistics) {
      if (st == "area" || st == "n_elements") continue;
      if (power_of(st) < 0) invalid("--scaling " + opt.scaling + ": the statistic " + st + " is not converted (a gradient's sums): export it with --scaling stored");
      if (!nxr::spectral::isLinear(target) && st != "mean" && st != "max" && st != "min" && power_of(st) != 0)
        invalid("--scaling " + opt.scaling + " is non-linear: it applies to mean, max and min (of the stored power), not to " + st);
    }
  std::regex re;
  try { re = std::regex(opt.fields, std::regex::ECMAScript); } catch (const std::regex_error& e) { invalid("--fields " + opt.fields + ": not a regular expression (" + e.what() + ")"); }
  const fs::path folder(dataset_folder);
  const Database db = Database::open(folder / "dataset.sqlite", false, "nxr-neuro", true);

  std::vector<Row> subjects;
  if (opt.subjects.empty()) subjects = db.all("SELECT id, name FROM subject WHERE kind = 'subject' AND status = 'complete' ORDER BY name");
  else for (const auto& n : opt.subjects) {
    const auto r = db.one("SELECT id, name FROM subject WHERE name = ? AND status = 'complete'", {V(n)});
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
    const auto sel = db.one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'set' AND array_path IS NOT NULL AND status = 'complete'", {V(sid_subj), V(opt.labelling)});
    if (!sel) { per_subject.push_back(Json{{"subject", sname}, {"missing", "no partition " + opt.labelling}}); continue; }
    const std::string sid = text(*sel, "id");
    std::map<std::int64_t, std::string> names;
    if (!is_null(*sel, "dictionary_id"))
      for (const Row& e : db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", {V(text(*sel, "dictionary_id"))})) names[integer(e, "code")] = text(e, "name");
    auto member_name = [&](std::int64_t c) { const auto it = names.find(c); return it != names.end() ? it->second : "member " + std::to_string(c); };
    // the Fields: one per condition
    std::map<std::string, Row> by_cond;
    for (const Row& f : db.all("SELECT * FROM field WHERE subject_id = ? AND manifold_id = ? AND status = 'complete' ORDER BY rowid", {V(sid_subj), V(text(*sel, "manifold_id"))})) {
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
      std::vector<Value> a = {V(sid), V(ing)};
      std::string w = "selection_id = ? AND measure = ? AND ";
      if (f && slice_index.count(cond)) {                       // a statistic's map: its rows name the statistic and the slice
        q["ofStatisticId"] = text(*f, "id"); q["ofIndex"] = slice_index.at(cond);
        w += "of_statistic_id = ? AND of_index = ?"; a.push_back(V(text(*f, "id"))); a.push_back(V(slice_index.at(cond)));
      } else {
        q["ofFieldId"] = f ? Json(text(*f, "id")) : Json(nullptr);
        if (f) { w += "of_field_id = ?"; a.push_back(V(text(*f, "id"))); } else w += "of_field_id IS NULL";
        if (has_column(db, "selection_measurement", "of_statistic_id")) w += " AND of_statistic_id IS NULL";
      }
      const auto meta = db.all("SELECT DISTINCT computed_by, unit FROM selection_measurement WHERE " + w, a);
      if (meta.empty()) return;
      std::set<std::string> rvs;
      std::optional<std::string> unit;
      for (const Row& r : meta) { rvs.insert(recipe_of(opt_text(r, "computed_by").value_or(""))); if (!unit) unit = opt_text(r, "unit"); }
      for (const auto& rv : rvs) if (!wanted(rv)) invalid(sname + ": the " + ing + " of " + (f ? text(*f, "name") : std::string("the members")) + " on " + opt.labelling + " was made by " + rv + ", not " + *opt.recipe);
      const std::string rv = *rvs.begin();
      if (!recipes.contains(rv)) {
        const auto e = db.one("SELECT params_json FROM dictionary_entry WHERE name = ? AND dictionary_id IN (SELECT id FROM dictionary WHERE name = 'recipes')", {V(rv)});
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
      std::optional<nxr::spectral::Scaling> conv;
      if (f && target != nxr::spectral::Target::Stored && power_of(st) != 0) {
        const auto rec = spectral_record(Json::parse(opt_text(*f, "producer_json").value_or("{}"), nullptr, false));
        if (!rec) invalid(sname + ": " + text(*f, "name") + " has no spectral scaling record (not a spectrum, a relative power, or computed before 2026-11-13): export it with --scaling stored");
        try {
          if (nxr::spectral::isLinear(target)) factor = std::pow(nxr::spectral::linearFactor(rec->scaling, target), power_of(st));
          else conv = rec->scaling;
        } catch (const std::invalid_argument& e) { invalid(sname + ": " + text(*f, "name") + ": " + e.what()); }
        const std::string base = nxr::spectral::unitOf(rec->power_unit, rec->scaling, target);
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
          values[sname][var] = conv ? nxr::spectral::convert(v, *conv, target) : v * factor;
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
  const auto dsrow = db.one("SELECT id, name FROM dataset");
  Json prov = {{"table", opt.table}, {"generated_utc", now_utc()}, {"generated_by", std::string("nxr-neuro ") + NXR_NEURO_VERSION + " measure export"},
               {"dataset", Json{{"path", fs::absolute(folder).string()}, {"id", dsrow ? text(*dsrow, "id") : ""}, {"name", dsrow ? text(*dsrow, "name") : ""}}},
               {"request", Json{{"labelling", opt.labelling}, {"fields", opt.fields}, {"statistics", opt.statistics}, {"recipe", opt.recipe ? Json(*opt.recipe) : Json(nullptr)}, {"exclude", opt.exclude}, {"scaling", opt.scaling}}},
               {"recipes", recipes}, {"subjects", per_subject}, {"rows", subjects.size()}, {"layout", opt.layout}, {"tables", tables_out}};
  { std::ofstream j(out / (opt.table + ".json")); j << prov.dump(1) << "\n"; }
  return Json{{"stage", "result"}, {"folder", out.string()}, {"tables", tables_out}, {"dictionary", (out / "dictionary.csv").string()}, {"provenance", (out / (opt.table + ".json")).string()},
              {"subjects", subjects.size()}, {"columns", order.size()}, {"recipes", recipes.size()}};
}

}  // namespace nxr::pipelines
