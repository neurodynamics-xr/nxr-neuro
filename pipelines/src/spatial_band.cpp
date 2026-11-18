// nxr-pipelines — the spatial band of a tile level (spatial_band.h).
#include "nxr/pipelines/spatial_band.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/io/array_reader.h"
#include "nxr/select/dyadic.h"
#include "nxr/signal/graphwavelet.h"
#include "nxr/pipelines/task.h"
#include "nxr/signal/riesz.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/meeg_internal.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace fb = nxr::signal::graphfilterbank;
namespace lad = nxr::select;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }
template <class T> std::vector<T> read_array(SubjectWriter& sub, const std::string& path) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}
}  // namespace

SurfaceLevel read_surface_level(SubjectWriter& sub, const SurfaceInput& S, int l) {
  SurfaceLevel L;
  L.surface = S.name; L.l = l; L.member = l;
  const std::int64_t nV = S.n_vertices();
  const task::Subject subj = task::Subject::of(sub);
  const auto ladder = subj.find({.cls = task::Cls::selection, .name = S.name + " tiles", .type = "ladder", .manifold_id = S.id});
  if (!ladder) missing(sub.name() + ": " + S.name + " has no tile ladder; run `nxr-neuro run select_dyadic_surface` first");
  L.ladder = *ladder;
  L.tiling = S.name + " tiles L" + std::to_string(l);
  const auto lvl = subj.find({.cls = task::Cls::selection, .name = L.tiling, .member_of_id = ladder->id});
  if (!lvl) invalid(sub.name() + ": the ladder of " + S.name + " has no level " + std::to_string(l) + " (its levels are 0 … " + std::to_string(ladder->n_members.value_or(0) - 1) + ")");
  L.level = *lvl;
  L.record = lvl->params;
  const Json gp = ladder->params;
  if (!L.record.is_object() || !L.record.contains("node") || !L.record.contains("areas") || !gp.is_object() || !gp.contains("area_m2") || !gp.contains("pieces"))
    invalid(L.tiling + ": its record lacks the node of its arrays or the surface's area (a ladder made by another producer)");
  L.nodes = lvl->n_members.value_or(0);
  L.tile = read_array<std::int32_t>(sub, lvl->array_path.value_or(std::string()));
  if (static_cast<std::int64_t>(L.tile.size()) != nV) invalid(L.tiling + ": its codes are for " + std::to_string(L.tile.size()) + " vertices, the surface has " + std::to_string(nV));
  L.tile_area = read_array<double>(sub, L.record["areas"].get<std::string>());
  if (static_cast<std::int64_t>(L.tile_area.size()) != L.nodes) invalid(L.tiling + ": " + std::to_string(L.tile_area.size()) + " tile areas for " + std::to_string(L.nodes) + " tiles");
  // the tile graph: the level's stored adjacency (a < b), read by its name; a ladder without it is read from the mesh
  const std::string apath = L.record["node"].get<std::string>() + "/lc/L" + std::to_string(l) + "/adjacency";
  if (sub.store().exists_prefix(apath)) {
    const std::vector<std::int32_t> a = read_array<std::int32_t>(sub, apath);
    for (std::size_t i = 0; i + 1 < a.size(); i += 2) L.pairs.emplace_back(a[i], a[i + 1]);
  } else {
    for (const auto& e : lad::tileAdjacency(S.faces.data(), static_cast<int>(S.n_faces()), L.tile)) L.pairs.emplace_back(e.first, e.second);
  }
  L.piece_area = gp["area_m2"].get<double>() / std::max(1, gp["pieces"].get<int>());
  // the Weyl pairing: member l of the bank whose edges are the ladder's lambda(2^j, mean piece area) IS level l's band; checked against the level's own record
  L.bank = fb::weyl(L.piece_area, std::max(1, l + 1));
  const fb::Member& m = L.bank.members[static_cast<std::size_t>(l)];
  auto close = [](double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(std::abs(a), std::abs(b)); };
  const double lmax_l = L.record.value("lambda_max", std::nan("")), lmin_l = L.record.value("lambda_min", std::nan(""));
  if (!close(m.hi, lmax_l) || !close(m.lo, lmin_l))
    invalid(L.tiling + ": its band (" + std::to_string(lmin_l) + " … " + std::to_string(lmax_l) + " 1/m²) is not Weyl member " + std::to_string(l) + " of the surface's bank (" + std::to_string(m.lo) + " … " + std::to_string(m.hi) + "): the ladder was made with another area");
  return L;
}

namespace {
// chebyshev | exact: the transforms of the input mesh itself, needing no subject
// the mesh gradient of the Riesz part (nxr::signal::vertexGradient: the P1 faces' gradients averaged on the vertices), on the faces whose three vertices are all `kept` (empty: every face): a vertex
// at the edge of the cut medial wall has a one-sided gradient, never one across the wall's zero
nxr::signal::Gradient riesz_gradient(const SurfaceInput& S, const std::vector<std::int32_t>& kept, std::int64_t* faces_used = nullptr) {
  const std::int64_t nV = S.n_vertices();
  std::vector<std::uint8_t> keep(static_cast<std::size_t>(nV), kept.empty() ? 1 : 0);
  for (const auto v : kept) keep[static_cast<std::size_t>(v)] = 1;
  std::vector<std::int32_t> faces;
  for (std::int64_t f = 0; f < S.n_faces(); ++f) {
    const std::int32_t* t = &S.faces[static_cast<std::size_t>(3 * f)];
    if (keep[static_cast<std::size_t>(t[0])] && keep[static_cast<std::size_t>(t[1])] && keep[static_cast<std::size_t>(t[2])]) faces.insert(faces.end(), t, t + 3);
  }
  Eigen::MatrixXd P(nV, 3);
  for (std::int64_t i = 0; i < nV; ++i) for (int d = 0; d < 3; ++d) P(i, d) = S.vertices[static_cast<std::size_t>(3 * i + d)];
  Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> F(static_cast<Eigen::Index>(faces.size() / 3), 3);
  for (std::size_t f = 0; f < faces.size() / 3; ++f) for (int s = 0; s < 3; ++s) F(static_cast<Eigen::Index>(f), s) = faces[3 * f + static_cast<std::size_t>(s)];
  if (faces_used) *faces_used = static_cast<std::int64_t>(faces.size() / 3);
  return nxr::signal::vertexGradient(P, F);
}

constexpr double kHatFloor = 1e-3;          // the hat's tail the eigen route may drop: the modes must reach where the gain falls under this
void check_shape(const SpatialBandOptions& o) {
  if (o.spatial_shape != "meyer" && o.spatial_shape != "mexhat") invalid("spatial band: spatial_shape " + o.spatial_shape + ": mexhat (the level's Mexican hat) or meyer (the level's Weyl band)");
}
bool is_hat(const SpatialBandOptions& o) { return o.spatial_shape == "mexhat"; }
// the hat's centre: the GEOMETRIC centre of the level's Weyl band [lambda_min, lambda_max] (nxr hat_operator: 4π·2^(l−0.5)/A); level 0's band starts at 0: hi/√2 (as if it began one octave below)
double hat_centre(const SurfaceLevel& L) {
  const fb::Member& m = L.bank.members[static_cast<std::size_t>(L.l)];
  return m.lo > 0 ? std::sqrt(m.lo * m.hi) : m.hi / std::sqrt(2.0);
}
// the level's gain: its Weyl member's (Meyer) or its Mexican hat
fb::Gain level_gain(const SurfaceLevel& L, const SpatialBandOptions& o) {
  if (!is_hat(o)) return L.bank.members[static_cast<std::size_t>(L.l)].gain;
  const double c = hat_centre(L);
  return [c](double lam) { return fb::mexicanHat(lam, c); };
}
// how far the modes must reach: the band's top with its Meyer edge (the gain is 0 above hi * 2^halfWidth), or where the hat is under kHatFloor
double band_reach(const SurfaceLevel& L, const SpatialBandOptions& o) {
  const fb::Member& m = L.bank.members[static_cast<std::size_t>(L.l)];
  if (is_hat(o)) return fb::mexicanHatReach(hat_centre(L), kHatFloor);
  const double h = std::isfinite(L.bank.halfWidth) ? L.bank.halfWidth : 1.0 / 3.0;
  return m.hi * std::pow(2.0, h);
}
// what the stored result records of the shape
Json shape_info(const SurfaceLevel& L, const SpatialBandOptions& o) {
  return is_hat(o) ? Json{{"spatial_shape", "mexhat"}, {"hat_centre_lambda", hat_centre(L)}, {"hat_gain_floor", kHatFloor}, {"hat_reach_lambda", band_reach(L, o)}} : Json{{"spatial_shape", "meyer"}};
}
SpatialBand build_mesh_band(const SurfaceInput& S, const SurfaceLevel& L, const SpatialBandOptions& opt) {
  check_shape(opt);
  const fb::Gain gain = level_gain(L, opt);
  if (opt.order < 0) invalid("spatial band: the order is at least 0 (0: chosen from the tolerance)");
  if (!(opt.tolerance > 0)) invalid("spatial band: the tolerance is above 0");
  const std::int64_t nV = S.n_vertices();
  const int l = L.l;
  const fb::Member& member = L.bank.members[static_cast<std::size_t>(l)];
  nxr::geometry::IntrinsicSurface isurf(S.vertices.data(), static_cast<int>(nV), S.faces.data(), static_cast<int>(S.n_faces()));
  const Eigen::SparseMatrix<double> Lap = isurf.cotanLaplacian();          // on geometry-central's intrinsic Delaunay triangulation
  const std::vector<double> mass = isurf.inputMixedVoronoiMass();
  Eigen::SparseMatrix<double> M(static_cast<Eigen::Index>(nV), static_cast<Eigen::Index>(nV));
  {
    std::vector<Eigen::Triplet<double>> t;
    for (std::int64_t i = 0; i < nV; ++i) t.emplace_back(static_cast<int>(i), static_cast<int>(i), mass[static_cast<std::size_t>(i)]);
    M.setFromTriplets(t.begin(), t.end());
  }
  SpatialBand out;
  out.info = Json{{"level", l}, {"member", l}, {"role", member.role == fb::Role::Scaling ? "scaling" : "wavelet"}, {"lambda_min", member.lo}, {"lambda_max", member.hi},
                  {"wavelength_m", L.record.value("wavelength_m", std::nan(""))}, {"half_width_octaves", 1.0 / 3.0}, {"weyl_area_m2", L.piece_area}, {"transform", opt.transform}};
  { const Json si = shape_info(L, opt); for (const auto& [k, v] : si.items()) out.info[k] = v; }
  const fb::GraphFilterbank bank = L.bank;
  if (opt.transform == "exact") {
    if (nV > 4096) invalid("spatial band: the exact transform is the dense eigen-decomposition: at most 4096 vertices, this surface has " + std::to_string(nV) + "; use chebyshev");
    Eigen::MatrixXd Ld(Lap), Md = Eigen::MatrixXd::Zero(nV, nV);
    for (std::int64_t i = 0; i < nV; ++i) Md(i, i) = mass[static_cast<std::size_t>(i)];
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Ld, Md);
    auto T = std::make_shared<fb::EigenTransform>();
    T->phi = es.eigenvectors(); T->lambda = es.eigenvalues().cwiseMax(0.0); T->mass = M;
    out.info["modes"] = static_cast<std::int64_t>(T->lambda.size());
    if (is_hat(opt)) {
      auto gv0 = std::make_shared<Eigen::VectorXd>(T->lambda.size());
      for (Eigen::Index k = 0; k < gv0->size(); ++k) (*gv0)(k) = gain(T->lambda(k));
      out.apply = [T, gv0](const Eigen::MatrixXd& F) { return Eigen::MatrixXd(T->phi * (gv0->asDiagonal() * (T->phi.transpose() * (T->mass * F)))); };
    } else {
      out.apply = [bank, l, T](const Eigen::MatrixXd& F) { return nxr::signal::gwt(bank, l, *T, F); };
    }
    if (opt.riesz) {
      const auto G = std::make_shared<nxr::signal::Gradient>(riesz_gradient(S, {}));
      const auto gR = nxr::signal::rieszGain(gain);
      auto gv = std::make_shared<Eigen::VectorXd>(T->lambda.size());
      for (Eigen::Index k = 0; k < gv->size(); ++k) (*gv)(k) = gR(T->lambda(k));
      out.apply_riesz = [T, gv, G](const Eigen::MatrixXd& F) { return nxr::signal::rieszTransform(*G, Eigen::MatrixXd(T->phi * (gv->asDiagonal() * (T->phi.transpose() * (T->mass * F)))), 1); };
      out.info["riesz"] = Json{{"gain", "g/sqrt(lambda), 0 at the constant mode"}, {"gradient", "P1 faces averaged on the vertices (nxr::signal::vertexGradient), the input mesh"}};
    }
    return out;
  }
  const fb::ChebyshevOperator probe(Lap, M, 0, 40);          // the spectral bound, once
  const double lmax = probe.lmax();
  int order = opt.order;
  double err = 0;
  if (order > 0) {
    err = fb::chebyshevError(gain, lmax, order);
    if (err > opt.tolerance) invalid("spatial band: the Chebyshev series of degree " + std::to_string(order) + " is off the level " + std::to_string(l) + " band by up to " + std::to_string(err) + " (tolerance " + std::to_string(opt.tolerance) + "): raise --order, --order 0 to choose it, or use --transform exact");
  } else {
    for (int o = 40; o <= 5120; o *= 2) { err = fb::chebyshevError(gain, lmax, o); if (err <= opt.tolerance) { order = o; break; } }
    if (order == 0) invalid("spatial band: no Chebyshev series up to degree 5120 follows the level " + std::to_string(l) + " band to within " + std::to_string(opt.tolerance) + " (error " + std::to_string(err) + " at 5120) on this surface (lambda_max " + std::to_string(lmax) + "): raise the tolerance or use the exact transform");
  }
  auto op = std::make_shared<fb::ChebyshevOperator>(Lap, M, lmax, order);
  out.info["order"] = order; out.info["series_error"] = err; out.info["lambda_top"] = lmax;
  if (is_hat(opt)) out.apply = [gain, op](const Eigen::MatrixXd& F) { return op->apply(gain, F); };
  else out.apply = [bank, l, op](const Eigen::MatrixXd& F) { return nxr::signal::gwt(bank, l, *op, F); };
  if (opt.riesz) {
    // the series of g/√λ: it must follow ITS gain within the tolerance (a scaling member's gain is not 0 at λ = 0: g/√λ is singular and no series follows it, refused)
    const fb::Gain gR = nxr::signal::rieszGain(gain);
    double peak = 0;
    for (int i = 0; i <= 4096; ++i) peak = std::max(peak, std::abs(gR(lmax * i / 4096)));
    int oR = order;
    double eR = 0;                                                  // relative to the peak of g/√λ (it is in metres, the band's gain is not)
    for (; oR <= 5120; oR *= 2) { eR = peak > 0 ? fb::chebyshevError(gR, lmax, oR) / peak : 0.0; if (eR <= opt.tolerance) break; }
    if (oR > 5120) invalid("spatial band: no Chebyshev series up to degree 5120 follows the Riesz gain g/sqrt(lambda) of level " + std::to_string(l) + " to within " + std::to_string(opt.tolerance) + " of its peak (error " + std::to_string(eR) + "): raise the tolerance, or use eigenmodes (a scaling level's gain is not 0 at lambda = 0)");
    const auto opR = oR == order ? op : std::make_shared<fb::ChebyshevOperator>(Lap, M, lmax, oR);
    const auto G = std::make_shared<nxr::signal::Gradient>(riesz_gradient(S, {}));
    out.apply_riesz = [gR, opR, G](const Eigen::MatrixXd& F) { return nxr::signal::rieszTransform(*G, opR->apply(gR, F), 1); };
    out.info["riesz"] = Json{{"gain", "g/sqrt(lambda), 0 at the constant mode"}, {"order", oR}, {"series_error", eR}, {"gradient", "P1 faces averaged on the vertices (nxr::signal::vertexGradient), the input mesh"}};
  }
  return out;
}

// the band on the GLUED surface: `inner` is a band of the glued surface's own vertices (the dense or Chebyshev one); the native columns are placed on the kept vertices (the bridge is 0), the band
// fills the bridge, and `apply` reads the kept vertices back
void on_glued(SpatialBand& b, const internal::GluedDomain& dom, std::int64_t nV) {
  const auto inner = std::make_shared<std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)>>(b.apply);
  const auto support = std::make_shared<std::vector<std::int32_t>>(dom.support);
  const std::int64_t Vg = dom.surface.n_vertices();
  b.apply_glued = [inner, support, nV, Vg](const Eigen::MatrixXd& F) {
    if (F.rows() != nV) invalid("spatial band: the field has " + std::to_string(F.rows()) + " rows, the surface " + std::to_string(nV) + " vertices");
    Eigen::MatrixXd Xg = Eigen::MatrixXd::Zero(Vg, F.cols());
    for (std::size_t i = 0; i < support->size(); ++i) Xg.row(static_cast<Eigen::Index>(i)) = F.row((*support)[i]);
    return (*inner)(Xg);
  };
  const auto glued = b.apply_glued;
  b.apply = [glued, support, nV](const Eigen::MatrixXd& F) {
    const Eigen::MatrixXd Y = glued(F);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(nV, F.cols());
    for (std::size_t i = 0; i < support->size(); ++i) R.row((*support)[i]) = Y.row(static_cast<Eigen::Index>(i));
    return R;
  };
  b.glued_surface = dom.surface.name;
  b.glued_vertices = Vg;
  b.info["glued"] = Json{{"surface", dom.surface.name}, {"vertices", Vg}, {"kept_vertices", static_cast<std::int64_t>(dom.support.size())}, {"bridge_vertices", Vg - static_cast<std::int64_t>(dom.support.size())},
                         {"domain", "the band is evaluated on every vertex of the glued surface: the field is zero on the bridge before it, the band fills the bridge"}};
}

const double kFour_pi = 4 * M_PI;
Json series_key(const SpatialBandOptions& o) { return Json{{"transform", "chebyshev"}, {"order", o.order}, {"tolerance", o.tolerance}}; }
std::string fmt(double v) { char b[32]; std::snprintf(b, sizeof b, "%.6g", v); return b; }
}  // namespace

SpatialBandPlan plan_spatial_band(SubjectWriter& sub, const SurfaceInput& S, const SurfaceLevel& L, const SpatialBandOptions& opt) {
  const std::string& t = opt.transform;
  if (t != "auto" && t != "eigen" && t != "chebyshev" && t != "exact") invalid("spatial band: the transform is auto, eigen, chebyshev or exact");
  if (opt.order < 0) invalid("spatial band: the order is at least 0 (0: chosen from the tolerance)");
  if (!(opt.tolerance > 0)) invalid("spatial band: the tolerance is above 0");
  if (opt.modes < 0 || opt.max_modes < 2) invalid("spatial band: modes is at least 0 (0: the standard size), max modes at least 2");
  check_shape(opt);
  SpatialBandPlan P;
  P.reach = band_reach(L, opt);
  // the Meyer keys are unchanged (a stored result stays a hit); the hat's carry the shape and its centre
  auto keyed = [&](Json k) { if (is_hat(opt)) { k["spatial_shape"] = "mexhat"; k["hat_centre_lambda"] = hat_centre(L); } if (opt.glued) k["glued"] = true; return k; };
  if (t == "chebyshev") { P.transform = t; P.key = keyed(series_key(opt)); return P; }
  if (t == "exact") { P.transform = t; P.key = keyed(Json{{"transform", "exact"}}); return P; }
  // eigenmode first. Why the series is used, when it is, goes into `reason`
  const std::string who = "level " + std::to_string(L.l) + " of " + S.name;
  const std::string stem = hodge_package_stem(S);
  std::string reason;
  const internal::StoredLaplace st = internal::stored_laplace_info(sub, S);
  if (st.found) {
    if (P.reach <= st.lambda_max * (1 + 1e-9)) {
      P.transform = "eigen"; P.source = "stored"; P.package = st.package; P.key = keyed(Json{{"transform", "eigen"}, {"package", st.package}});
      return P;
    }
    reason = "the stored eigenmodes (" + st.package + ": " + std::to_string(st.n_exact) + " Laplace-Beltrami modes, largest eigenvalue " + fmt(st.lambda_max) + " 1/m^2) do not reach " + who + ": its band tops at " +
             fmt(L.bank.members[static_cast<std::size_t>(L.l)].hi) + " 1/m^2 (" + fmt(P.reach) + (is_hat(opt) ? ", where the Mexican hat's gain falls under 1e-3)" : " with the Meyer edge)");
  } else {
    // none stored: can they be computed? one closed piece, or two with the medial wall; enough modes by Weyl's law; not beyond the cap
    const int pieces = connected_pieces(S.faces, S.n_vertices());
    const Json gp = L.ladder.params;
    const double area = L.piece_area * std::max(1, gp.is_object() ? gp.value("pieces", 1) : 1);
    const std::int64_t need = static_cast<std::int64_t>(std::ceil(1.25 * area * P.reach / kFour_pi)) + 16;       // exact modes up to the band's top, with a margin
    const std::int64_t request = opt.modes > 0 ? opt.modes : std::max<std::int64_t>(3000, 2 * need);              // half exact, half coexact
    if (pieces > 2) reason = S.name + " is " + std::to_string(pieces) + " pieces (the eigen solve takes one closed piece, or two hemispheres)";
    else if (pieces == 2) { try { (void)medial_wall(sub, S); } catch (const Error& e) { reason = e.what(); } }
    if (reason.empty() && request / 2 < need) reason = "a package of " + std::to_string(request) + " modes holds about " + std::to_string(request / 2) + " exact modes, " + who + " needs about " + std::to_string(need) + " (Weyl's law)";
    if (reason.empty() && request > opt.max_modes) reason = who + " needs a package of about " + std::to_string(request) + " modes, above the cap of " + std::to_string(opt.max_modes) + " (the solve and its storage grow with it)";
    if (reason.empty()) {
      P.transform = "eigen"; P.source = "computed"; P.package = stem; P.compute_modes = request; P.key = keyed(Json{{"transform", "eigen"}, {"package", stem}});
      P.messages.push_back("no stored eigenmodes for " + S.name + ": " + std::to_string(request) + " modes will be computed and stored (" + who + " needs about " + std::to_string(need) + " exact modes up to " + fmt(P.reach) + " 1/m^2)");
      return P;
    }
    reason = "no stored eigenmodes for " + S.name + " and they cannot be computed: " + reason;
  }
  if (t == "eigen") invalid("spatial band: --transform eigen: " + reason + "; use --transform chebyshev");
  P.transform = "chebyshev"; P.key = keyed(series_key(opt));
  P.warnings.push_back("spatial band of " + who + ": CHEBYSHEV fallback, not eigenmodes: " + reason);
  return P;
}

SpatialBand make_spatial_band(const SurfaceInput& S, const SurfaceLevel& L, const SpatialBandOptions& opt) {
  if (opt.transform != "chebyshev" && opt.transform != "exact") invalid("spatial band: without a subject the transform is chebyshev or exact (auto and eigen read, and may write, the subject's eigenmodes)");
  if (opt.glued) invalid("spatial band: the glued surface is the subject's (its stored package): build the band with the subject");
  SpatialBand b = build_mesh_band(S, L, opt);
  b.info["source"] = "input mesh";
  return b;
}

Json make_eigenmodes(DatasetWriter& ds, SubjectWriter& sub, const std::string& surface, const SpatialBandPlan& P, const std::function<void(const Json&)>& progress) {
  EigenSolveOptions eo;
  eo.modes = static_cast<int>(P.compute_modes);
  eo.surface = surface;
  const EigenSolveResult r = eigen_solve(ds, sub, eo, progress);   // the `eigenmodes` pipeline: its own run, its own provenance and cache key
  return Json{{"package", P.package}, {"modes", r.modes.n_modes()}, {"exact", r.modes.n_exact}, {"coexact", r.modes.n_coexact}, {"seconds", r.seconds}, {"hit", r.hit}};
}

SpatialBand make_spatial_band(DatasetWriter& ds, SubjectWriter& sub, const SurfaceInput& S, const SurfaceLevel& L, const SpatialBandOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.glued && opt.riesz) invalid("spatial band: the Riesz part is not built on the glued surface");
  SpatialBandPlan P = plan_spatial_band(sub, S, L, opt);
  SpatialBand out;
  if (P.transform != "eigen") {
    SpatialBandOptions o = opt;
    o.transform = P.transform;
    if (opt.glued) {
      // the fallback is the series (or the dense modes) of the GLUED mesh, never of the native one (it has no bridge)
      const internal::GluedDomain dom = internal::glued_domain(sub, S);
      if (!dom.found) invalid("spatial band: glued: " + dom.why);
      out = build_mesh_band(dom.surface, L, o);
      on_glued(out, dom, S.n_vertices());
    } else {
      out = build_mesh_band(S, L, o);
    }
    // a fallback names the series it fell back on (the degree resolved from the tolerance)
    if (P.transform == "chebyshev")
      for (auto& w : P.warnings) w += "; the series used has degree " + std::to_string(out.info.value("order", 0)) + " (its largest |gain - series| " + fmt(out.info.value("series_error", 0.0)) + ")";
    for (const auto& m : P.warnings) say(Json{{"stage", "spatial_band"}, {"warning", m}});
    out.info["source"] = "input mesh";
    out.info["requested"] = opt.transform;
    out.warnings = P.warnings;
    if (!P.warnings.empty()) out.info["warnings"] = P.warnings;
    return out;
  }
  const int l = L.l;
  const fb::Member& member = L.bank.members[static_cast<std::size_t>(l)];
  const fb::Gain gain = level_gain(L, opt);
  const std::string who = "level " + std::to_string(l) + " of " + S.name;
  Json computed = nullptr;
  if (P.source == "computed") {
    for (const auto& m : P.messages) say(Json{{"stage", "spatial_band"}, {"message", m}});
    computed = make_eigenmodes(ds, sub, S.name, P, progress);
  }
  const internal::StoredLaplace st = internal::stored_laplace_info(sub, S);
  if (!st.found || P.reach > st.lambda_max * (1 + 1e-9))
    invalid("spatial band: " + who + ": the " + std::string(P.source == "computed" ? "computed" : "stored") + " eigenmodes " + (st.found ? "(" + st.package + ": largest eigenvalue " + fmt(st.lambda_max) + " 1/m^2) do not reach the band's top " + fmt(P.reach) + " 1/m^2"
            : "are not there") + "; use --transform chebyshev");
  if (P.source == "computed") {
    const std::string m = "eigenmodes computed and stored: " + std::to_string(st.modes_total) + " modes (" + std::to_string(st.n_exact) + " exact), largest exact eigenvalue " + fmt(st.lambda_max) + " 1/m^2, name " + st.package;
    P.messages.push_back(m);
    say(Json{{"stage", "spatial_band"}, {"message", m}});
  }
  const double reach = P.reach;
  const auto modes = std::make_shared<internal::LaplaceModes>(internal::read_laplace_modes(sub, S, [&](double lam) { return std::abs(gain(lam)) > 1e-12 && lam <= reach * (1 + 1e-9); }));
  const Eigen::Index k = modes->lambda.size();
  auto g = std::make_shared<Eigen::VectorXd>(k);
  for (Eigen::Index j = 0; j < k; ++j) (*g)(j) = gain(modes->lambda(j));
  const std::int64_t nV = S.n_vertices(), kept = static_cast<std::int64_t>(modes->support.size()), Vg = modes->phi.rows();
  out.info = Json{{"level", l}, {"member", l}, {"role", member.role == fb::Role::Scaling ? "scaling" : "wavelet"}, {"lambda_min", member.lo}, {"lambda_max", member.hi},
                  {"wavelength_m", L.record.value("wavelength_m", std::nan(""))}, {"half_width_octaves", 1.0 / 3.0}, {"weyl_area_m2", L.piece_area}, {"transform", "eigen"},
                  {"requested", opt.transform}, {"source", P.source}, {"modes", static_cast<std::int64_t>(k)},
                  {"package", Json{{"name", st.package}, {"glued_surface", st.glued_surface}, {"modes", st.modes_total}, {"exact", st.n_exact}, {"lambda_max", st.lambda_max}, {"lambda_top_band", reach},
                                   {"glued_vertices", Vg}, {"kept_vertices", kept}, {"bridge_vertices", Vg - kept}, {"computed", computed}}},
                  {"domain", "the package's glued surface: the field is zero on the bridge and the cut medial wall, the band is read back on the kept vertices (the wall's get 0)"}};
  { const Json si = shape_info(L, opt); for (const auto& [k, v] : si.items()) out.info[k] = v; }
  out.messages = P.messages;
  out.warnings = P.warnings;
  if (!P.messages.empty()) out.info["messages"] = P.messages;
  out.apply = [modes, g, nV](const Eigen::MatrixXd& F) {
    if (F.rows() != nV) invalid("spatial band: the field has " + std::to_string(F.rows()) + " rows, the surface " + std::to_string(nV) + " vertices");
    const Eigen::Index Vg = modes->phi.rows(), n = F.cols();
    Eigen::MatrixXd Xg = Eigen::MatrixXd::Zero(Vg, n);
    for (std::size_t i = 0; i < modes->support.size(); ++i) Xg.row(static_cast<Eigen::Index>(i)) = F.row(modes->support[i]);
    Eigen::MatrixXd C = modes->phi.transpose() * modes->mass.asDiagonal() * Xg;     // Φᵀ ⋆0 x
    C = g->asDiagonal() * C;
    const Eigen::MatrixXd Y = modes->phi * C;
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(nV, n);
    for (std::size_t i = 0; i < modes->support.size(); ++i) R.row(modes->support[i]) = Y.row(static_cast<Eigen::Index>(i));
    return R;
  };
  if (opt.glued) {
    out.apply_glued = [modes, g, nV](const Eigen::MatrixXd& F) {
      if (F.rows() != nV) invalid("spatial band: the field has " + std::to_string(F.rows()) + " rows, the surface " + std::to_string(nV) + " vertices");
      const Eigen::Index Vg = modes->phi.rows(), n = F.cols();
      Eigen::MatrixXd Xg = Eigen::MatrixXd::Zero(Vg, n);
      for (std::size_t i = 0; i < modes->support.size(); ++i) Xg.row(static_cast<Eigen::Index>(i)) = F.row(modes->support[i]);
      return Eigen::MatrixXd(modes->phi * (g->asDiagonal() * (modes->phi.transpose() * modes->mass.asDiagonal() * Xg)));    // Φ g Φᵀ ⋆0 x on EVERY glued vertex
    };
    out.glued_surface = st.glued_surface;
    out.glued_vertices = Vg;
    out.info["glued"] = Json{{"surface", st.glued_surface}, {"vertices", Vg}, {"kept_vertices", kept}, {"bridge_vertices", Vg - kept},
                             {"domain", "the band is evaluated on every vertex of the glued surface: the field is zero on the bridge before it, the band fills the bridge"}};
  }
  if (opt.riesz) {
    // R_l = G Φ diag(g/√λ) Φᵀ ⋆0 on the glued surface's kept vertices, the gradient on the input mesh's faces whose vertices are all kept (the wall's and the bridge's are not)
    std::int64_t faces_used = 0;
    const auto G = std::make_shared<nxr::signal::Gradient>(riesz_gradient(S, modes->support, &faces_used));
    const auto gRfn = nxr::signal::rieszGain(gain);
    auto gR = std::make_shared<Eigen::VectorXd>(k);
    for (Eigen::Index j = 0; j < k; ++j) (*gR)(j) = gRfn(modes->lambda(j));
    out.apply_riesz = [modes, gR, G, nV](const Eigen::MatrixXd& F) {
      if (F.rows() != nV) invalid("spatial band: the field has " + std::to_string(F.rows()) + " rows, the surface " + std::to_string(nV) + " vertices");
      const Eigen::Index Vg = modes->phi.rows(), n = F.cols();
      Eigen::MatrixXd Xg = Eigen::MatrixXd::Zero(Vg, n);
      for (std::size_t i = 0; i < modes->support.size(); ++i) Xg.row(static_cast<Eigen::Index>(i)) = F.row(modes->support[i]);
      const Eigen::MatrixXd Y = modes->phi * (gR->asDiagonal() * (modes->phi.transpose() * modes->mass.asDiagonal() * Xg));      // Δ^{-1/2} F_l f on the glued surface
      Eigen::MatrixXd Sx = Eigen::MatrixXd::Zero(nV, n);
      for (std::size_t i = 0; i < modes->support.size(); ++i) Sx.row(modes->support[i]) = Y.row(static_cast<Eigen::Index>(i));
      return nxr::signal::rieszTransform(*G, Sx, 1);
    };
    out.info["riesz"] = Json{{"gain", "g/sqrt(lambda), 0 at the constant mode"}, {"gradient", "P1 faces averaged on the vertices (nxr::signal::vertexGradient), the input mesh's faces whose three vertices are kept"}, {"faces_used", faces_used}};
  }
  return out;
}

}  // namespace nxr::pipelines
