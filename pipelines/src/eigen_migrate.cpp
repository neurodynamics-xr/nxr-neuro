// nxr-pipelines — `nxr-neuro migrate eigensolve`: a stored Hodge–Dirac eigensolve of the earlier form converted IN PLACE to the four classes (eigen.h), with no solve.
//
// The earlier form (what `run eigenmodes` stored before): the eigenvalue line; a kind FIELD on it; three operators from it, `_eigenvectors` (-> the surface, cell 0: the vertex potentials), `_forms1`
// (-> the triangulation's edges; absent in a solve made before the 1-form parts were stored) and `_forms2` (-> its faces); the triangulation with its own copy of the positions, d0 and d1 as stored
// csr operators and the stars and signposts as fields. The new form: ONE eigenvectors operator (line -> triangulation, cell key '0+1+2', rows 0-forms · 1-forms · 2-forms, the same float32 numbers),
// the kinds as a LABELLING selection of the line, the eigenvalues as a field of the new operator, no support (the glue's correspondence is derived), the triangulation sharing the surface's positions, d0, d1 and the stars as ANALYTIC rows (method
// 'dec'), the lengths and signposts kept as fields. Absent 1-form parts are made from the stored parts and operators (`signal::oneFormFromParts`, the solve's own expression: no solve).
// One unit a stem (all of it or nothing); the replaced items are moved aside by io (their arrays kept in a `_replaced-*` folder, never deleted). Each new item records `migrated_from` (the old ids and
// names, the date). Idempotent: a stem already in the new form is reported and left. A form it does not know (a missing part, shapes that do not fit, anything else lying on the triangulation) is refused.
#include <algorithm>
#include <cstring>
#include <set>

#include <Eigen/Sparse>

#include "nxr/io/array_reader.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/task/task.h"
#include "nxr/io/subject_api.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/eigen_migrate.h"
#include "nxr/pipelines/input.h"
#include "nxr/selection/selection.h"
#include "nxr/signal/intrinsic_forms.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace tk = nxr::io::task;

namespace {

[[noreturn]] void unknown(const std::string& stem, const std::string& why) {
  throw Error(code::invalid_argument, "migrate eigensolve " + stem + ": not a stored eigensolve of a form this migration knows (" + why + "); nothing changed");
}

template <class T>
std::vector<T> read_as(const Store& store, const std::string& key, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(store, key);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  const DType t = r.dtype();
  std::vector<T> out;
  auto conv = [&](auto tag) {
    using S = decltype(tag);
    std::vector<S> v(raw.size() / sizeof(S));
    std::memcpy(v.data(), raw.data(), v.size() * sizeof(S));
    out.assign(v.begin(), v.end());
  };
  switch (t) {
    case DType::Float32: conv(float{}); break;
    case DType::Float64: conv(double{}); break;
    case DType::Int32: conv(std::int32_t{}); break;
    case DType::Int64: conv(std::int64_t{}); break;
    case DType::UInt8: conv(std::uint8_t{}); break;
    default: throw Error(code::unsupported, key + ": an array of a type the migration does not read");
  }
  return out;
}

// a node's columns (io's task API: the record as JSON, never the database)
std::string js(const Json& r, const char* k) { return r.contains(k) && r[k].is_string() ? r[k].get<std::string>() : std::string(); }
std::optional<std::string> jo(const Json& r, const char* k) { if (r.contains(k) && r[k].is_string()) return r[k].get<std::string>(); return std::nullopt; }
std::int64_t ji(const Json& r, const char* k) { return r.contains(k) && r[k].is_number() ? r[k].get<std::int64_t>() : 0; }
// a JSON column of a record (stored as text, or already parsed): null when absent
Json json_column(const Json& r, const char* k) {
  if (!r.contains(k) || r[k].is_null()) return Json();
  return r[k].is_string() ? Json::parse(r[k].get<std::string>(), nullptr, false) : r[k];
}
Json producer_of_row(const Json& r) {
  if (!r.contains("producer_json")) return Json::object();
  const Json& p = r["producer_json"];
  const Json j = p.is_string() ? Json::parse(p.get<std::string>(), nullptr, false) : p;
  return j.is_object() ? j : Json::object();
}

tk::Cls cls_of(const std::string& table) { return table == "operator" ? tk::Cls::operator_ : table == "field" ? tk::Cls::field : table == "selection" ? tk::Cls::selection : tk::Cls::manifold; }
// a complete node of the subject by name, as its record
std::optional<Json> row_named(SubjectWriter& sub, const std::string& table, const std::string& name) {
  tk::Want w{.cls = cls_of(table)};
  w.name = name;
  const auto n = tk::Subject::of(sub).find(w);
  if (!n) return std::nullopt;
  return n->row;
}
std::optional<Json> row_by_id(SubjectWriter& sub, const std::string& table, const std::string& id) {
  const auto n = tk::Subject::of(sub).by_id(cls_of(table), id);
  if (!n) return std::nullopt;
  return n->row;
}
std::vector<Json> rows_of(SubjectWriter& sub, const tk::Want& w) {
  std::vector<Json> out;
  for (const auto& n : tk::Subject::of(sub).find_all(w)) out.push_back(n.row);
  return out;
}

Eigen::SparseMatrix<double> csr_of(const Store& store, const Json& op) {
  const std::string p = js(op, "path");
  const auto ip = read_as<std::int64_t>(store, p + "/indptr"), ix = read_as<std::int64_t>(store, p + "/indices");
  const auto dv = read_as<double>(store, p + "/data");
  const std::int64_t rows = ji(op, "n_rows"), cols = ji(op, "n_cols");
  std::vector<Eigen::Triplet<double>> t;
  for (std::int64_t i = 0; i < rows; ++i) for (std::int64_t k = ip[static_cast<std::size_t>(i)]; k < ip[static_cast<std::size_t>(i + 1)]; ++k) t.emplace_back(i, static_cast<int>(ix[static_cast<std::size_t>(k)]), dv[static_cast<std::size_t>(k)]);
  Eigen::SparseMatrix<double> M(rows, cols);
  M.setFromTriplets(t.begin(), t.end());
  return M;
}

// the rows [r0, r1) of every column of a stored 2-D float32 array, into `out` (row-major [rows x cols]) at row r0: one block of rows at a time (`block`), never the raw bytes of the whole array
void read_rows_into(const Store& store, const std::string& key, std::int64_t r0, std::int64_t r1, std::int64_t cols, std::int64_t block, std::vector<float>& out) {
  ArrayReader r = ArrayReader::open(store, key);
  if (r.dtype() != DType::Float32) throw Error(code::unsupported, key + ": not float32");
  for (std::int64_t a = r0; a < r1; a += block) {
    const std::int64_t b = std::min(r1, a + block);
    const std::vector<std::uint8_t> raw = r.read({a, 0}, {b, cols});
    std::memcpy(out.data() + a * cols, raw.data(), raw.size());
  }
}

// A SOLVE ALREADY IN THE FOUR CLASSES, made by an earlier version, brought to the current form in place (no solve), each part only when it differs:
//   - a coexact mode's vertex rows held its stream function (the face-to-vertex mean of *2 chi/sqrt2), not its 0-form part: set to 0 (the eigenvectors and the eigenvalues, a field OF them, written again);
//     The kind of each mode is read from its PARTS: a mode whose 2-form rows (the faces) are all 0 is exact, one with a non-zero 2-form part coexact (an exact mode's face rows are stored as exact
//     zeros); the stored labelling must agree, or the stem is refused;
//   - the support selection of an earlier form moved aside (the glue's correspondence is derived).
// Memory: the eigenvectors, (V + E + F) x K float32, read one chunk row (V rows) at a time.
Json migrate_current(SubjectWriter& sub, const std::string& stem, const Json& ev, const std::optional<Json>& support, const std::function<void(const Json&)>& say) {
  const StoredEigensolve e = stored_eigensolve(sub, stem);
  const Store& store = sub.store();
  const std::int64_t V = e.V, E = e.E, F = e.F, K = e.K, N = V + E + F;
  const std::string path = js(ev, "path");
  std::vector<float> stacked(static_cast<std::size_t>(N * K));
  read_rows_into(store, path, 0, N, K, V, stacked);
  std::vector<std::int32_t> kind(static_cast<std::size_t>(K), 0);
  std::int64_t zeroed = 0;
  for (std::int64_t k = 0; k < K; ++k) {
    bool face = false, vertex = false;
    for (std::int64_t f = 0; f < F && !face; ++f) face = stacked[static_cast<std::size_t>((V + E + f) * K + k)] != 0.0f;
    for (std::int64_t v = 0; v < V && !vertex; ++v) vertex = stacked[static_cast<std::size_t>(v * K + k)] != 0.0f;
    kind[static_cast<std::size_t>(k)] = face ? 2 : 0;
    if (!face && !vertex) unknown(stem, "mode " + std::to_string(k) + " has neither a 0-form nor a 2-form part");
    if (kind[static_cast<std::size_t>(k)] != e.kind[static_cast<std::size_t>(k)])
      unknown(stem, "mode " + std::to_string(k) + " is labelled " + std::to_string(e.kind[static_cast<std::size_t>(k)]) + " but its parts make it " + (face ? "coexact (a 2-form part)" : "exact (no 2-form part)"));
    if (face && vertex) {
      ++zeroed;
      for (std::int64_t v = 0; v < V; ++v) stacked[static_cast<std::size_t>(v * K + k)] = 0.0f;
    }
  }
  if (!zeroed && !support) return Json{{"stem", stem}, {"status", "already in the new form"}};
  Json from = Json{{"date", now_utc(false)}, {"from", "the four classes of an earlier version"}, {"items", Json::object()}};
  if (zeroed) from["vertex_rows"] = "the vertex rows of " + std::to_string(zeroed) + " coexact modes held the stream function; set to 0 (the 0-form part)";
  const auto lamf = row_named(sub, "field", stem + "_eigenvalues");
  if (zeroed) { from["items"][stem + "_eigenvectors"] = Json{{"class", "operator"}, {"id", js(ev, "id")}}; if (lamf) from["items"][stem + "_eigenvalues"] = Json{{"class", "field"}, {"id", js(*lamf, "id")}}; }
  if (support) from["items"][stem + "_support"] = Json{{"class", "selection"}, {"id", js(*support, "id")}};
  if (say) say(Json{{"stage", "migrate"}, {"stem", stem}, {"coexact_vertex_rows_zeroed", zeroed}, {"support_removed", static_cast<bool>(support)}});
  std::vector<double> lam;
  if (zeroed && lamf) lam = read_as<double>(store, js(*lamf, "path"));
  const tk::Subject ts = tk::Subject::of(sub);
  tk::Unit u = ts.unit(tk::By{"nxr-neuro migrate eigensolve", Json{{"migrated_from", from}}});
  const std::string what = "the eigensolve " + stem + " of an earlier version";
  Json out = Json{{"stem", stem}, {"status", "migrated"}, {"coexact_vertex_rows_zeroed", zeroed}};
  if (zeroed) {
    if (lamf) u.replace(tk::Cls::field, js(*lamf, "path"), true, what);
    u.replace(tk::Cls::operator_, path, true, what);
    tk::OperatorOut o;
    o.name = stem + "_eigenvectors"; o.output = "eigenvectors"; o.kind = "eigenvectors"; o.path = path;
    o.from_manifold_id = js(ev, "from_manifold_id"); o.to_manifold_id = js(ev, "to_manifold_id"); o.from_cell = "0"; o.to_cell = "0+1+2";
    o.chunk_shape = {V, 1}; o.shard_shape = {V, std::min<std::int64_t>(K, 64)};
    o.description = "Hodge-Dirac modes as mixed forms on " + text(e.triangulation, "name") + ": rows 0-forms (the vertices), 1-forms (the edges), 2-forms (the faces), one column a mode";
    o.params = json_column(ev, "params_json");
    if (o.params.is_object() && o.params.contains("forms"))
      o.params["forms"] = "each mode's mixed form, orthonormal under the stars *0, *1, *2: the 0-form part psi/sqrt2 (0 on a coexact mode); the 1-form part d psi/(mu sqrt2) or *1^-1 d1^T *2 chi/(mu sqrt2) (*1-orthogonal, energy 1/2); the 2-form part chi/sqrt2 (0 on an exact mode). A coexact mode's stream function at the vertices is derived from its 2-form part (each vertex the mean over its faces of *2 chi/sqrt2), never stored";
    o.producer = producer_of_row(ev);
    o.data.shape = {N, K};
    o.data.f32 = std::move(stacked);
    const std::string oid = u.operator_(o);
    if (lamf) {
      tk::FieldOut lf;
      lf.name = stem + "_eigenvalues"; lf.output = "eigenvalues"; lf.kind = "eigenvalue"; lf.manifold_id = text(e.line, "id"); lf.of_operator_id = oid;
      lf.unit = jo(*lamf, "unit").value_or("1/m^2");
      lf.description = jo(*lamf, "description").value_or("per mode: the eigenvalue lambda = mu^2 of the squared operator");
      lf.producer = producer_of_row(*lamf);
      lf.data.shape = {K}; lf.data.f64 = std::move(lam);
      u.field(lf);
    }
    out["eigenvectors"] = oid;
  }
  if (support) { u.replace(tk::Cls::selection, js(*support, "path"), true, "the support of the eigensolve " + stem + ": derived from the glued cortex's hemispheres labelling and the medial wall"); out["removed"] = stem + "_support"; }
  u.commit();
  return out;
}

Json migrate_one(SubjectWriter& sub, const std::string& stem, const std::function<void(const Json&)>& say) {
  const auto ev = row_named(sub, "operator", stem + "_eigenvectors");
  if (!ev) unknown(stem, stem + "_eigenvectors is not stored");
  const std::string to_cell = jo(*ev, "to_cell").value_or("0");
  const auto support = row_named(sub, "selection", stem + "_support");   // the earlier forms' support: the glue's correspondence, derived now from the glued cortex's labelling and the medial wall
  if (to_cell == "0+1+2" && row_named(sub, "selection", stem + "_kinds")) return migrate_current(sub, stem, *ev, support, say);
  if (to_cell != "0") unknown(stem, stem + "_eigenvectors maps to cell '" + to_cell + "'");
  const auto f2 = row_named(sub, "operator", stem + "_forms2");
  if (!f2) unknown(stem, stem + "_forms2 (the 2-form parts) is not stored");
  const auto f1 = row_named(sub, "operator", stem + "_forms1");   // absent in a solve made before the 1-form parts were stored: made here
  const auto line = row_by_id(sub, "manifold", js(*ev, "from_manifold_id"));
  const auto surf = row_by_id(sub, "manifold", js(*ev, "to_manifold_id"));
  const auto tri = row_by_id(sub, "manifold", js(*f2, "to_manifold_id"));
  const auto kindf = row_named(sub, "field", stem + "_kind");
  if (!line || !surf || !tri) unknown(stem, "its line, its surface or its triangulation is not stored");
  if (!kindf || js(*kindf, "manifold_id") != js(*line, "id")) unknown(stem, stem + "_kind is not a field of its line");
  if (jo(*tri, "derived_from_id") != std::optional<std::string>(js(*surf, "id"))) unknown(stem, js(*tri, "name") + " is not made from " + js(*surf, "name"));
  if (js(*f2, "from_manifold_id") != js(*line, "id") || (f1 && (js(*f1, "from_manifold_id") != js(*line, "id") || js(*f1, "to_manifold_id") != js(*tri, "id"))))
    unknown(stem, "its 1- or 2-form parts are not of its line and its triangulation");
  const std::string tname = js(*tri, "name");
  const std::int64_t nV = ji(*tri, "n_vertices"), nE = ji(*tri, "n_edges"), nF = ji(*tri, "n_faces");
  if (ji(*surf, "n_vertices") != nV) unknown(stem, tname + " and " + js(*surf, "name") + " differ in vertices");
  if (js(*surf, "geometry_id").empty()) unknown(stem, js(*surf, "name") + " has no geometry");

  // everything lying on the triangulation must be the earlier form's own items
  const std::set<std::string> known_fields = {tname + "_length", tname + "_star0", tname + "_star1", tname + "_star2", tname + "_angle_tail", tname + "_angle_tip", tname + "_angle_sum", tname + "_reference"};
  const std::set<std::string> known_ops = {tname + "_d0", tname + "_d1", stem + "_forms1", stem + "_forms2"};
  std::map<std::string, Json> tfields, tops;
  const std::string tri_id = js(*tri, "id");
  tk::Want on_tri{.cls = tk::Cls::field}; on_tri.manifold_id = tri_id;
  tk::Want from_tri{.cls = tk::Cls::operator_}; from_tri.from_manifold_id = tri_id;
  tk::Want to_tri{.cls = tk::Cls::operator_}; to_tri.to_manifold_id = tri_id;
  tk::Want sel_tri{.cls = tk::Cls::selection}; sel_tri.manifold_id = tri_id;
  for (const Json& r : rows_of(sub, on_tri)) {
    if (!known_fields.count(js(r, "name"))) unknown(stem, "the field " + js(r, "name") + " lies on " + tname);
    tfields.emplace(js(r, "name"), r);
  }
  for (const tk::Want& w : {from_tri, to_tri})
    for (const Json& r : rows_of(sub, w)) {
      if (!known_ops.count(js(r, "name"))) unknown(stem, "the operator " + js(r, "name") + " lies on " + tname);
      tops.emplace(js(r, "name"), r);
    }
  if (!rows_of(sub, sel_tri).empty()) unknown(stem, "a selection lies on " + tname);

  // READ (before anything is moved aside): mu, the kinds, the eigenvalues, the three parts, the triangulation's cells and fields
  const Store& store = sub.store();
  const std::vector<double> mu = read_as<double>(store, js(*line, "path"));
  const std::int64_t K = static_cast<std::int64_t>(mu.size());
  const std::vector<std::int32_t> kind = read_as<std::int32_t>(store, js(*kindf, "path"));
  if (static_cast<std::int64_t>(kind.size()) != K) unknown(stem, stem + "_kind holds " + std::to_string(kind.size()) + " values for " + std::to_string(K) + " modes");
  for (const auto k : kind) if (k != 0 && k != 2) unknown(stem, stem + "_kind holds a kind other than 0 (exact) and 2 (coexact)");
  const auto lamf = row_named(sub, "field", stem + "_eigenvalues");
  std::vector<double> lam;
  if (lamf) lam = read_as<double>(store, js(*lamf, "path"));
  else for (const double m : mu) lam.push_back(m * m);
  const std::int64_t N = nV + nE + nF;
  std::vector<std::int64_t> shape;
  std::vector<float> stacked = read_as<float>(store, js(*ev, "path"), &shape);              // [nV x K] first: the 0-form rows
  if (shape != std::vector<std::int64_t>{nV, K}) unknown(stem, stem + "_eigenvectors is not [" + std::to_string(nV) + " x " + std::to_string(K) + "]");
  stacked.resize(static_cast<std::size_t>(N * K), 0.0f);
  {
    const std::vector<float> x2 = read_as<float>(store, js(*f2, "path"), &shape);
    if (shape != std::vector<std::int64_t>{nF, K}) unknown(stem, stem + "_forms2 is not [" + std::to_string(nF) + " x " + std::to_string(K) + "]");
    std::copy(x2.begin(), x2.end(), stacked.begin() + (nV + nE) * K);
  }
  std::string forms1_made = "stored";
  if (f1) {
    const std::vector<float> x1 = read_as<float>(store, js(*f1, "path"), &shape);
    if (shape != std::vector<std::int64_t>{nE, K}) unknown(stem, stem + "_forms1 is not [" + std::to_string(nE) + " x " + std::to_string(K) + "]");
    std::copy(x1.begin(), x1.end(), stacked.begin() + nV * K);
  } else {
    // the 1-form parts from the stored parts and the stored operators (no solve): dψ/(μ√2) = d0 (ψ/√2)/μ, ⋆1⁻¹ d1ᵀ ⋆2 (χ/√2)/μ; one mode at a time
    if (!tops.count(tname + "_d0") || !tops.count(tname + "_d1") || !tfields.count(tname + "_star1") || !tfields.count(tname + "_star2")) unknown(stem, "no 1-form parts and no stored d0, d1, star1, star2 to make them from");
    const Eigen::SparseMatrix<double> d0 = csr_of(store, tops.at(tname + "_d0")), d1 = csr_of(store, tops.at(tname + "_d1"));
    const std::vector<double> s1 = read_as<double>(store, js(tfields.at(tname + "_star1"), "path")), s2 = read_as<double>(store, js(tfields.at(tname + "_star2"), "path"));
    if (d0.rows() != nE || d0.cols() != nV || d1.rows() != nF || d1.cols() != nE || static_cast<std::int64_t>(s1.size()) != nE || static_cast<std::int64_t>(s2.size()) != nF) unknown(stem, "the stored d0, d1 or stars do not fit the triangulation");
    const Eigen::Map<const Eigen::VectorXd> star1(s1.data(), nE), star2(s2.data(), nF);
    for (std::int64_t k = 0; k < K; ++k) {
      Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(nV, 1), two = Eigen::MatrixXd::Zero(nF, 1);
      if (kind[static_cast<std::size_t>(k)] == 0) for (std::int64_t v = 0; v < nV; ++v) zero(v, 0) = stacked[static_cast<std::size_t>(v * K + k)] / mu[static_cast<std::size_t>(k)];
      else for (std::int64_t f = 0; f < nF; ++f) two(f, 0) = stacked[static_cast<std::size_t>((nV + nE + f) * K + k)] / mu[static_cast<std::size_t>(k)];
      const Eigen::MatrixXd part = nxr::signal::oneFormFromParts(d0, d1, star1, star2, zero, two);
      for (std::int64_t e = 0; e < nE; ++e) stacked[static_cast<std::size_t>((nV + e) * K + k)] = static_cast<float>(part(e, 0));
    }
    forms1_made = "made from the stored 0- and 2-form parts with the stored d0, d1, star1, star2 (signal::oneFormFromParts; no solve)";
  }
  // a coexact mode's vertex rows held its stream function (the earlier form's vertex potential): its 0-form part is 0
  for (std::int64_t k = 0; k < K; ++k)
    if (kind[static_cast<std::size_t>(k)] == 2) for (std::int64_t v = 0; v < nV; ++v) stacked[static_cast<std::size_t>(v * K + k)] = 0.0f;
  const std::vector<std::int32_t> edges = read_as<std::int32_t>(store, js(*tri, "path") + "/edges"), faces = read_as<std::int32_t>(store, js(*tri, "path") + "/faces");
  if (static_cast<std::int64_t>(edges.size()) != 2 * nE || static_cast<std::int64_t>(faces.size()) != 3 * nF) unknown(stem, tname + "'s stored cells do not fit its counts");
  struct Kept { std::string name, cell, unit, description; std::vector<double> f64; std::vector<std::int32_t> i32; };
  std::vector<Kept> kept_fields;
  for (const char* suffix : {"_length", "_angle_tail", "_angle_tip", "_angle_sum", "_reference"}) {
    const auto it = tfields.find(tname + suffix);
    if (it == tfields.end()) continue;
    Kept k{js(it->second, "name"), jo(it->second, "cell").value_or("0"), jo(it->second, "unit").value_or(""), jo(it->second, "description").value_or(""), {}, {}};
    if (std::string(suffix) == "_reference") k.i32 = read_as<std::int32_t>(store, js(it->second, "path")); else k.f64 = read_as<double>(store, js(it->second, "path"));
    kept_fields.push_back(std::move(k));
  }
  const Json tri_prod = producer_of_row(*tri);
  const Json delaunay = tri_prod.contains("delaunay") ? tri_prod["delaunay"] : Json();
  const std::string cs = coordinate_system_name_of_geometry(sub, js(*surf, "geometry_id")).value_or("scs");

  // the record every new item carries
  Json from = Json{{"date", now_utc(false)}, {"from", "the earlier form: _eigenvectors, _forms1, _forms2, a kind field, stored d0/d1/stars"}, {"forms1", forms1_made},
                   {"items", Json::object()}};
  auto note = [&](const Json& r, const char* cls) { from["items"][js(r, "name")] = Json{{"class", cls}, {"id", js(r, "id")}}; };
  note(*ev, "operator"); note(*f2, "operator"); if (f1) note(*f1, "operator"); note(*kindf, "field"); if (lamf) note(*lamf, "field"); note(*tri, "manifold");
  for (const auto& [n, r] : tfields) note(r, "field");
  for (const auto& [n, r] : tops) note(r, "operator");
  if (support) note(*support, "selection");
  if (say) say(Json{{"stage", "migrate"}, {"stem", stem}, {"V", nV}, {"E", nE}, {"F", nF}, {"modes", K}, {"forms1", f1 ? "stored" : "made"}});

  const tk::Subject ts = tk::Subject::of(sub);
  tk::Unit u = ts.unit(tk::By{"nxr-neuro migrate eigensolve", Json{{"migrated_from", from}}});
  const std::string what = "the earlier form of the eigensolve " + stem;
  // MOVE ASIDE the earlier form: what lies on the triangulation first, then the operators, the line's fields, the triangulation
  for (const auto& [n, r] : tfields) u.replace(tk::Cls::field, js(r, "path"), true, what);
  for (const auto& [n, r] : tops) u.replace(tk::Cls::operator_, js(r, "path"), true, what);
  if (lamf) u.replace(tk::Cls::field, js(*lamf, "path"), true, what);
  u.replace(tk::Cls::field, js(*kindf, "path"), true, what);
  u.replace(tk::Cls::operator_, js(*ev, "path"), true, what);
  u.replace(tk::Cls::manifold, js(*tri, "path"), true, what);
  if (support) { u.replace(tk::Cls::selection, js(*support, "path"), true, what); }

  // THE INTRINSIC TRIANGULATION: the surface's positions, its own cells
  tk::ComplexOut c;
  c.name = tname; c.output = "triangulation"; c.path = js(*tri, "path"); c.cs_key = cs;
  c.derived_from_id = js(*surf, "id"); c.geometry_of = js(*surf, "id");
  c.edges = edges; c.faces = faces;
  c.description = "the intrinsic (mollified) Delaunay triangulation of " + js(*surf, "name") + ": its positions, its own edges (lengths along the surface) and faces";
  c.producer = Json{{"intrinsic_of", js(*surf, "name")}, {"delaunay", delaunay}};
  if (tri_prod.contains("mesh")) c.producer["mesh"] = tri_prod["mesh"];
  const std::string tid = u.cell_complex(c);
  const double mollify = delaunay.is_object() ? delaunay.value("mollify", 1e-6) : 1e-6, flip_eps = delaunay.is_object() ? delaunay.value("flip_eps", 1e-6) : 1e-6;
  auto analytic = [&](const char* op, const char* kind_name, std::int64_t rows, std::int64_t cols, const char* fc, const char* tcell) {
    tk::OperatorOut o;
    o.name = tname + "_" + op; o.output = op; o.kind = kind_name; o.from_manifold_id = tid; o.to_manifold_id = tid; o.from_cell = fc; o.to_cell = tcell;
    o.analytic = tk::OperatorOut::Analytic{"dec", rows, cols};
    o.params = Json{{"operator", op}, {"mollify", mollify}, {"flip_eps", flip_eps}, {"evaluated", "from the triangulation (its cells, its lengths field) as the Studio's hodgeDirac.ts dec(m) and nxr-compute hodgedirac::assemble build it"}};
    u.operator_(o);
  };
  analytic("d0", "exterior derivative", nE, nV, "0", "1");
  analytic("d1", "exterior derivative", nF, nE, "1", "2");
  analytic("star0", "hodge star", nV, nV, "0", "0*");
  analytic("star1", "hodge star", nE, nE, "1", "1*");
  analytic("star2", "hodge star", nF, nF, "2", "2*");
  for (Kept& k : kept_fields) {
    tk::FieldOut f;
    f.name = k.name; f.output = k.name.substr(tname.size() + 1); f.kind = "intrinsic triangulation"; f.manifold_id = tid; f.cell = k.cell;
    if (!k.unit.empty()) f.unit = k.unit;
    if (!k.description.empty()) f.description = k.description;
    const std::int64_t n = static_cast<std::int64_t>(k.f64.empty() ? k.i32.size() : k.f64.size());
    f.data.shape = {n};
    if (k.f64.empty()) { f.value_type = "integer"; f.data.i32 = std::move(k.i32); } else f.data.f64 = std::move(k.f64);
    u.field(f);
  }
  // THE EIGENVECTORS: one operator, line -> triangulation '0+1+2'
  tk::OperatorOut o;
  o.name = stem + "_eigenvectors"; o.output = "eigenvectors"; o.kind = "eigenvectors"; o.path = js(*ev, "path");
  o.from_manifold_id = js(*line, "id"); o.to_manifold_id = tid; o.from_cell = "0"; o.to_cell = "0+1+2";
  o.chunk_shape = {nV, 1}; o.shard_shape = {nV, std::min<std::int64_t>(K, 64)};
  o.description = "Hodge-Dirac modes as mixed forms on " + tname + ": rows 0-forms (the vertices), 1-forms (the edges), 2-forms (the faces), one column a mode";
  o.params = Json{{"rows", "0-forms [0, " + std::to_string(nV) + ") · 1-forms [" + std::to_string(nV) + ", " + std::to_string(nV + nE) + ") · 2-forms [" + std::to_string(nV + nE) + ", " + std::to_string(N) + ")"},
                  {"analysis", "the line <- triangulation map is this operator's D106 pair (its M-adjoint), not stored"}};
  o.producer = producer_of_row(*ev);                                       // the solve's own record (its options, its mesh, its triangulation's construction)
  o.data.shape = {N, K};
  o.data.f32 = std::move(stacked);
  const std::string oid = u.operator_(o);
  tk::FieldOut lf;
  lf.name = stem + "_eigenvalues"; lf.output = "eigenvalues"; lf.kind = "eigenvalue"; lf.manifold_id = js(*line, "id"); lf.of_operator_id = oid; lf.unit = "1/m^2";
  lf.description = "per mode: the eigenvalue lambda = mu^2 of the squared operator";
  lf.data.shape = {K}; lf.data.f64 = lam;
  u.field(lf);
  tk::SelectionOut ks;
  ks.name = stem + "_kinds"; ks.output = "kinds"; ks.type = "partition"; ks.manifold_id = js(*line, "id");
  ks.entries = {"exact", "harmonic", "coexact"};
  ks.description = "the kind of each mode: exact (label 0, a 0-form potential) or coexact (label 2, a 2-form potential); no mode is harmonic on a closed surface of genus 0";
  {
    nxr::selection::Lookup names;
    for (std::size_t k = 0; k < ks.entries.size(); ++k) names.push_back(nxr::selection::Entry{static_cast<std::int64_t>(k), ks.entries[k], std::nullopt});
    const nxr::selection::Selection sel = nxr::selection::from_labels(kind, names);
    ks.member_of = sel.member_of; ks.member_codes = sel.member_codes;
  }
  u.selection(ks);
  u.commit();
  return Json{{"stem", stem}, {"status", "migrated"}, {"eigenvectors", oid}, {"triangulation", tid}, {"rows", N}, {"modes", K}, {"forms1", forms1_made}};
}

// A GLUED CORTEX of an earlier form -> the hemispheres labelling of the glued cortex alone. Two earlier forms:
//   - the field `<glued>_map` (glued vertex -> original vertex, -1 on the bridge) and the indices `<glued>_bridge`: the map is checked against the glue's rule (the kept vertices first, ascending, then
//     the bridge) and the bridge selection, the original's hemisphere labelling against the original surface; the labelling is made, the map and the bridge moved aside;
//   - the labelling and the original's kept vertices `<glued>_kept`: the kept vertices are redundant (the glue removes exactly the medial wall) and are moved aside.
// Either way the glue's kept vertices must be EXACTLY the original's vertices off the medial wall (the Desikan-Killiany `unassigned`): a glue whose cut removed more (an earlier glue widened its cut
// to one simple loop) or kept wall is refused, nothing changed: re-run glue. One unit a glued cortex.
Json migrate_glue_one(SubjectWriter& sub, const tk::Node& glued, const std::string& hemispheres, const std::string& left, const std::string& right, const std::function<void(const Json&)>& say) {
  const tk::Subject ts = tk::Subject::of(sub);
  const std::string gname = glued.name;
  auto unknown_glue = [&](const std::string& why) -> void { throw Error(code::invalid_argument, "migrate glue " + gname + ": not a stored glued cortex of a form this migration knows (" + why + "); nothing changed"); };
  const Json& g = glued.producer["glue"];
  if (!g.is_object() || !g.contains("original") || !g["original"].is_string()) unknown_glue("its glue record names no original");
  const std::string original = g["original"].get<std::string>();
  tk::Want wk{.cls = tk::Cls::selection}; wk.name = gname + "_kept"; wk.manifold_id = original;
  tk::Want wl{.cls = tk::Cls::selection}; wl.name = gname + "_hemispheres"; wl.manifold_id = glued.id;
  const auto keptn = ts.find(wk), labelsn = ts.find(wl);
  if (!keptn && labelsn) return Json{{"glued", gname}, {"status", "already in the new form"}};
  const auto originaln = ts.by_id(tk::Cls::manifold, original);
  if (!originaln || !originaln->n_vertices) unknown_glue("its original manifold is not stored");
  std::vector<std::int32_t> off_wall;
  { const std::vector<std::uint8_t> wall = medial_wall_of(sub, original, originaln->name, *originaln->n_vertices);
    for (std::size_t v = 0; v < wall.size(); ++v) if (!wall[v]) off_wall.push_back(static_cast<std::int32_t>(v)); }
  auto exactly_the_wall = [&](const std::vector<std::int32_t>& kept) {
    if (kept != off_wall)
      throw Error(code::invalid_argument, "migrate glue " + gname + ": its " + std::to_string(kept.size()) + " kept vertices are not the " + std::to_string(off_wall.size()) + " vertices of " + originaln->name +
                                          " off the medial wall: the stored glue removed more than the medial wall; re-run glue (nothing changed)");
  };
  if (keptn && labelsn) {                                                   // the labelling and the kept vertices: the kept vertices checked, then moved aside
    if (!keptn->array_path || !keptn->path) unknown_glue(gname + "_kept holds no array");
    exactly_the_wall(read_as<std::int32_t>(sub.store(), *keptn->array_path));
    if (say) say(Json{{"stage", "migrate"}, {"glued", gname}, {"kept", static_cast<std::int64_t>(off_wall.size())}});
    tk::Unit u = ts.unit(tk::By{"nxr-neuro migrate glue", Json{{"migrated_from", Json{{"date", now_utc(false)}, {"from", "the labelling and the kept vertices (redundant: the glue removes exactly the medial wall)"},
                                                                                       {"items", Json{{gname + "_kept", Json{{"class", "selection"}, {"id", keptn->id}}}}}}}}});
    u.replace(tk::Cls::selection, *keptn->path, true, "the kept vertices of the glued cortex " + gname + ": redundant, the vertices off the medial wall");
    u.commit();
    return Json{{"glued", gname}, {"status", "migrated"}, {"kept", static_cast<std::int64_t>(off_wall.size())}, {"removed", gname + "_kept"}};
  }
  tk::Want wm{.cls = tk::Cls::field}; wm.name = gname + "_map"; wm.manifold_id = glued.id;
  tk::Want wb{.cls = tk::Cls::selection}; wb.name = gname + "_bridge"; wb.manifold_id = glued.id;
  const auto mapn = ts.find(wm), bridgen = ts.find(wb);
  if (!mapn || !mapn->path) unknown_glue(gname + "_map is not stored");
  if (!bridgen || !bridgen->array_path || !bridgen->path) unknown_glue(gname + "_bridge is not stored");
  const std::vector<std::int32_t> map = read_as<std::int32_t>(sub.store(), *mapn->path);
  const std::vector<std::int32_t> br = read_as<std::int32_t>(sub.store(), *bridgen->array_path);
  const std::int64_t n = static_cast<std::int64_t>(map.size());
  if (!glued.n_vertices || *glued.n_vertices != n) unknown_glue(gname + "_map holds " + std::to_string(n) + " values for the glued surface's vertices");
  std::int64_t m = 0;
  while (m < n && map[static_cast<std::size_t>(m)] >= 0) ++m;
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int32_t o = map[static_cast<std::size_t>(i)];
    if ((i < m) != (o >= 0) || (i > 0 && i < m && o <= map[static_cast<std::size_t>(i - 1)])) unknown_glue(gname + "_map breaks the glue's rule at glued vertex " + std::to_string(i) + " (the kept vertices first, ascending, then the bridge)");
  }
  if (static_cast<std::int64_t>(br.size()) != n - m) unknown_glue(gname + "_bridge holds " + std::to_string(br.size()) + " vertices, the map " + std::to_string(n - m));
  for (std::size_t i = 0; i < br.size(); ++i) if (br[i] != m + static_cast<std::int64_t>(i)) unknown_glue(gname + "_bridge is not the glued vertices after the kept ones");
  // the original's hemisphere labelling, checked against the original surface
  const SurfaceInput O = read_surface(sub, ts.by_id(tk::Cls::manifold, original)->name);
  tk::Want wh{.cls = tk::Cls::selection}; wh.name = hemispheres; wh.manifold_id = original;
  const auto hn = ts.find(wh);
  if (!hn) unknown_glue("the original " + O.name + " has no hemisphere labelling named " + hemispheres);
  const StoredLabelling H = stored_labelling(sub, hn->id);
  if (static_cast<std::int64_t>(H.labels.size()) != O.n_vertices()) unknown_glue(hemispheres + " does not label every vertex of " + O.name);
  if (const std::string why = hemispheres_disagree(O.faces, H, left, right); !why.empty()) unknown_glue(why);
  for (std::int64_t i = 0; i < m; ++i) if (map[static_cast<std::size_t>(i)] >= O.n_vertices()) unknown_glue(gname + "_map names a vertex outside " + O.name);
  exactly_the_wall(std::vector<std::int32_t>(map.begin(), map.begin() + m));
  if (say) say(Json{{"stage", "migrate"}, {"glued", gname}, {"kept", m}, {"bridge", n - m}});

  tk::Unit u = ts.unit(tk::By{"nxr-neuro migrate glue", Json{{"migrated_from", Json{{"date", now_utc(false)}, {"from", "the earlier form: a map field and a bridge selection"},
                                                                                     {"items", Json{{gname + "_map", Json{{"class", "field"}, {"id", mapn->id}}}, {gname + "_bridge", Json{{"class", "selection"}, {"id", bridgen->id}}}}}}}}});
  u.replace(tk::Cls::field, *mapn->path, true, "the earlier form of the glued cortex " + gname);
  u.replace(tk::Cls::selection, *bridgen->path, true, "the earlier form of the glued cortex " + gname);
  std::int64_t bcode = 0;
  tk::SelectionOut ls;
  ls.name = gname + "_hemispheres"; ls.output = "labels"; ls.type = "partition"; ls.manifold_id = glued.id;
  nxr::selection::Lookup names;
  for (const auto& e : H.entries) { bcode = std::max(bcode, e.code + 1); ls.lookup.push_back({e.code, e.name, e.color}); names.push_back(nxr::selection::Entry{e.code, e.name, e.color}); }
  ls.lookup.push_back({bcode, "bridge", std::array<int, 3>{128, 128, 128}});
  names.push_back(nxr::selection::Entry{bcode, "bridge", std::array<int, 3>{128, 128, 128}});
  std::vector<std::int32_t> labels(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) labels[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i < m ? H.labels[static_cast<std::size_t>(map[static_cast<std::size_t>(i)])] : bcode);
  const nxr::selection::Selection sel = nxr::selection::from_labels(labels, names);
  ls.member_of = sel.member_of; ls.member_codes = sel.member_codes;
  ls.description = "the hemispheres of the glued cortex: " + H.name + " of " + O.name + " carried over (its entries kept), and the bridge (not cortex: no sources, a forward zero there)";
  u.selection(ls);
  u.commit();
  return Json{{"glued", gname}, {"status", "migrated"}, {"kept", m}, {"bridge", n - m}, {"hemispheres", H.name}};
}

}  // namespace

Json migrate_glue(SubjectWriter& sub, const std::optional<std::string>& glued_name, const std::string& hemispheres, const std::string& left, const std::string& right,
                  const std::function<void(const Json&)>& progress) {
  tk::Want w{.cls = tk::Cls::manifold};
  w.type = "surface";
  if (glued_name) w.name = *glued_name;
  Json out = Json::array();
  for (const auto& n : tk::Subject::of(sub).find_all(w))
    if (n.producer.is_object() && n.producer.contains("glue")) out.push_back(migrate_glue_one(sub, n, hemispheres, left, right, progress));
  if (glued_name && out.empty()) throw Error(code::no_such_row, sub.name() + ": no glued cortex named " + *glued_name);
  return Json{{"subject", sub.name()}, {"glued", out}};
}

Json migrate_eigensolve(SubjectWriter& sub, const std::optional<std::string>& stem, const std::function<void(const Json&)>& progress) {
  std::vector<std::string> stems;
  if (stem) stems.push_back(*stem);
  else
    for (const auto& node : tk::Subject::of(sub).find_all([] { tk::Want w{.cls = tk::Cls::operator_}; w.kind = "eigenvectors"; return w; }())) {
      const std::string n = node.name, suf = "_eigenvectors";
      if (n.size() > suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0) stems.push_back(n.substr(0, n.size() - suf.size()));
    }
  Json out = Json::array();
  for (const auto& s : stems) out.push_back(migrate_one(sub, s, progress));
  return Json{{"subject", sub.name()}, {"stems", out}};
}

}  // namespace nxr::pipelines
