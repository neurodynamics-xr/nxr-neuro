// nxr::operators for Python: the operators of a surface (made by geometry.surfaceArrays), the Hodge–Dirac pencil and its modes, the two Hodge ladders and
// the Poisson solve. A face: arguments and results converted (dense as NumPy, sparse as SciPy CSC), the library called, no maths.
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <optional>

#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/factorization.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/phase_gradient.h"
#include "surface.h"

namespace nxrpy {

namespace ops = nxr::operators;
using Sparse = Eigen::SparseMatrix<double>;

namespace {

nb::dict manifold_operators(const nb::dict& surface, const std::string& mass) {
  const geom::SurfaceArrays s = surface_in(surface);
  const ops::MassMatrixVariant variant = ops::parseMassMatrixVariant(mass);
  std::optional<ops::ManifoldOperators> o;
  {
    nb::gil_scoped_release release;
    o.emplace(ops::assembleManifoldOperators(s, variant));
  }
  nb::dict out;
  out["cotanLaplacian"] = o->cotanLaplacian;
  out["mass"] = o->mass;
  out["massVariant"] = mass;
  out["vertexDualAreas"] = o->vertexDualAreas;
  out["vertexNormals"] = Eigen::MatrixXd(o->vertexNormals);
  out["totalArea"] = o->totalArea;
  return out;
}

nb::dict dec_operators(const nb::dict& surface) {
  const geom::SurfaceArrays s = surface_in(surface);
  std::optional<ops::DECOperators> d;
  {
    nb::gil_scoped_release release;
    d.emplace(ops::assembleDECOperators(s));
  }
  nb::dict out;
  out["d0"] = d->d0; out["d1"] = d->d1;
  out["hodge0"] = d->hodge0; out["hodge1"] = d->hodge1; out["hodge2"] = d->hodge2; out["hodge1Inverse"] = d->hodge1Inverse;
  return out;
}

nb::dict complex_out(geom::IntrinsicSurface::Complex c) {
  nb::dict out;
  out["faces"] = rows_out(std::move(c.faces), 3);
  out["edges"] = rows_out(std::move(c.edges), 2);
  out["faceEdges"] = rows_out(std::move(c.faceEdges), 3);
  out["length"] = vec_out(std::move(c.length));
  out["angle"] = rows_out(std::move(c.angle), 3);
  out["angleSum"] = vec_out(std::move(c.angleSum));
  out["reference"] = vec_out(std::move(c.reference));
  return out;
}

nb::dict hodge_dirac_assemble(const nb::dict& surface, double star1_floor, double mollify, double flip_eps) {
  const geom::SurfaceArrays s = surface_in(surface);
  ops::hodgedirac::HodgeDirac h;
  {
    nb::gil_scoped_release release;
    h = ops::hodgedirac::assemble(s, star1_floor, mollify, flip_eps);
  }
  nb::dict out;
  out["K"] = h.K; out["d0"] = h.d0; out["d1"] = h.d1;
  out["star0"] = h.star0; out["star1"] = h.star1; out["star2"] = h.star2; out["M"] = h.M;
  out["nV"] = h.nV; out["nE"] = h.nE; out["nF"] = h.nF;
  out["star1Floor"] = h.star1Floor; out["star1Clamped"] = h.star1Clamped; out["star1Negative"] = h.star1Negative;
  out["eulerCharacteristic"] = h.eulerCharacteristic; out["boundaryLoops"] = h.boundaryLoops; out["genus"] = h.genus;
  out["mollifyEps"] = h.mollifyEps;
  out["faces"] = Eigen::MatrixXi(h.faces);
  out["record"] = record_out(h.record);
  out["complex"] = complex_out(std::move(h.complex));
  return out;
}

/** The pencil dict back to the library's HodgeDirac: the fields a mode solve and its checks read. */
ops::hodgedirac::HodgeDirac hodge_dirac_in(const nb::dict& d) {
  ops::hodgedirac::HodgeDirac h;
  h.d0 = nb::cast<Sparse>(field(d, "d0"));
  h.d1 = nb::cast<Sparse>(field(d, "d1"));
  h.star0 = nb::cast<Eigen::VectorXd>(field(d, "star0"));
  h.star1 = nb::cast<Eigen::VectorXd>(field(d, "star1"));
  h.star2 = nb::cast<Eigen::VectorXd>(field(d, "star2"));
  h.nV = nb::cast<int>(field(d, "nV")); h.nE = nb::cast<int>(field(d, "nE")); h.nF = nb::cast<int>(field(d, "nF"));
  h.star1Floor = nb::cast<double>(field(d, "star1Floor"));
  h.star1Clamped = nb::cast<int>(field(d, "star1Clamped"));
  h.star1Negative = nb::cast<int>(field(d, "star1Negative"));
  h.eulerCharacteristic = nb::cast<int>(field(d, "eulerCharacteristic"));
  h.boundaryLoops = nb::cast<int>(field(d, "boundaryLoops"));
  h.genus = nb::cast<int>(field(d, "genus"));
  if (h.d0.rows() != h.nE || h.d0.cols() != h.nV || h.d1.rows() != h.nF || h.d1.cols() != h.nE || h.star0.size() != h.nV || h.star1.size() != h.nE ||
      h.star2.size() != h.nF)
    throw std::invalid_argument("the Hodge–Dirac's d0, d1 and stars do not match its nV, nE, nF: pass the dict made by operators.hodgedirac.assemble");
  return h;
}

ops::HodgeShift shift_of(const std::string& s) {
  if (s == "absolute") return ops::HodgeShift::Absolute;
  if (s == "relative_ladder") return ops::HodgeShift::RelativeLadder;
  throw std::invalid_argument("shift '" + s + "' is not one of absolute, relative_ladder");
}

ops::HodgeSolveOptions solve_options(const nb::dict& o) {
  ops::HodgeSolveOptions opt;
  if (o.contains("shift")) opt.shift = shift_of(nb::cast<std::string>(o["shift"]));
  if (o.contains("sigma")) opt.sigma = nb::cast<double>(o["sigma"]);
  if (o.contains("slice")) opt.slice = nb::cast<int>(o["slice"]);
  if (o.contains("normalize")) opt.normalize = nb::cast<bool>(o["normalize"]);
  if (o.contains("tol")) opt.tol = nb::cast<double>(o["tol"]);
  if (o.contains("ncv")) opt.ncv = nb::cast<int>(o["ncv"]);
  if (o.contains("maxit")) opt.maxit = nb::cast<int>(o["maxit"]);
  if (o.contains("limit")) opt.limit = nb::cast<int>(o["limit"]);
  return opt;
}

nb::dict eigen_out(ops::EigenResult&& r) {
  nb::dict out;
  out["eigenvalues"] = std::move(r.eigenvalues);
  out["eigenvectors"] = std::move(r.eigenvectors);
  out["k"] = r.k;
  out["nConverged"] = r.nConverged;
  if (r.kind.size() > 0) out["kind"] = std::move(r.kind);
  return out;
}

nb::dict hodge_dirac_modes(const nb::dict& pencil, int count, bool both_signs, const nb::dict& options) {
  const ops::hodgedirac::HodgeDirac h = hodge_dirac_in(pencil);
  const ops::HodgeSolveOptions opt = solve_options(options);
  ops::EigenResult r;
  {
    nb::gil_scoped_release release;
    r = ops::hodgeDiracModes(h, count, both_signs, opt);
  }
  return eigen_out(std::move(r));
}

nb::dict hodge_ladders(const Sparse& L0, const Sparse& M0, const Sparse& S2, const Sparse& M2, int count, const nb::dict& options) {
  const ops::HodgeSolveOptions opt = solve_options(options);
  ops::HodgeLadders lad;
  {
    nb::gil_scoped_release release;
    lad = ops::hodgeLadders(L0, M0, S2, M2, count, opt);
  }
  Eigen::VectorXd mu(lad.modes.size());
  Eigen::VectorXi kind(lad.modes.size()), col(lad.modes.size());
  for (size_t i = 0; i < lad.modes.size(); ++i) {
    mu(i) = lad.modes[i].mu; kind(i) = lad.modes[i].kind; col(i) = lad.modes[i].col;
  }
  nb::dict out;
  out["exact"] = eigen_out(std::move(lad.exact));
  out["coexact"] = eigen_out(std::move(lad.coexact));
  out["top"] = lad.top;
  out["mu"] = mu; out["kind"] = kind; out["col"] = col;
  out["nExact"] = lad.nExact; out["nCoexact"] = lad.nCoexact;
  return out;
}

Eigen::VectorXd poisson(const Sparse& K, const Sparse& M, const std::vector<int>& vertices, const std::vector<double>& values) {
  if (vertices.size() != values.size()) throw std::invalid_argument("density_vertices and density_values have different lengths");
  std::map<int, double> density;
  for (size_t i = 0; i < vertices.size(); ++i) density[vertices[i]] = values[i];
  nb::gil_scoped_release release;
  const auto factor = ops::laplacianFactor(K);
  return ops::poisson(K, M, *factor, density);
}

// the intrinsic complex (nxr::signal::IntrinsicComplex) <-> a dict, made once by signal.intrinsicComplex(surface) and passed to every form function. The signposts'
// fans (per vertex its input edges' angles and directions, ragged) cross as fanStart [nV + 1], fanAngle [n], fanDirection [n, 3].
nb::dict intrinsic_complex_out(nxr::signal::IntrinsicComplex&& c) {
  nb::dict out;
  out["nV"] = c.nV; out["nE"] = c.nE; out["nF"] = c.nF;
  out["edges"] = rows_out(std::move(c.edges), 2);
  out["length"] = std::move(c.length);
  out["d0"] = std::move(c.d0); out["d1"] = std::move(c.d1);
  out["star0"] = std::move(c.star0); out["star1"] = std::move(c.star1); out["star2"] = std::move(c.star2);
  out["angleTail"] = std::move(c.angleTail); out["angleTip"] = std::move(c.angleTip); out["coneAngle"] = std::move(c.coneAngle);
  out["reference"] = vec_out(std::move(c.reference));
  out["positions"] = std::move(c.positions); out["normals"] = std::move(c.normals);
  std::vector<int64_t> start{0};
  std::vector<double> angle, direction;
  for (const auto& fan : c.fans) {
    for (const auto& [a, d] : fan) { angle.push_back(a); direction.insert(direction.end(), {d.x(), d.y(), d.z()}); }
    start.push_back(static_cast<int64_t>(angle.size()));
  }
  out["fanStart"] = vec_out(std::move(start));
  out["fanAngle"] = vec_out(std::move(angle));
  out["fanDirection"] = rows_out(std::move(direction), 3);
  return out;
}

nxr::signal::IntrinsicComplex intrinsic_complex_in(const nb::dict& d) {
  if (!d.contains("fanStart") || !d.contains("star1"))
    throw std::invalid_argument("expected the intrinsic complex made by signal.intrinsicComplex(surface)");
  nxr::signal::IntrinsicComplex c;
  c.nV = nb::cast<int>(field(d, "nV")); c.nE = nb::cast<int>(field(d, "nE")); c.nF = nb::cast<int>(field(d, "nF"));
  const auto e = nb::cast<Array2<int32_t>>(field(d, "edges"));
  c.edges.assign(e.data(), e.data() + e.shape(0) * e.shape(1));
  c.length = nb::cast<Eigen::VectorXd>(field(d, "length"));
  c.d0 = nb::cast<Sparse>(field(d, "d0")); c.d1 = nb::cast<Sparse>(field(d, "d1"));
  c.star0 = nb::cast<Eigen::VectorXd>(field(d, "star0")); c.star1 = nb::cast<Eigen::VectorXd>(field(d, "star1")); c.star2 = nb::cast<Eigen::VectorXd>(field(d, "star2"));
  c.angleTail = nb::cast<Eigen::VectorXd>(field(d, "angleTail")); c.angleTip = nb::cast<Eigen::VectorXd>(field(d, "angleTip"));
  c.coneAngle = nb::cast<Eigen::VectorXd>(field(d, "coneAngle"));
  c.reference = vec_in<int32_t>(d, "reference");
  c.positions = nb::cast<Eigen::MatrixXd>(field(d, "positions")); c.normals = nb::cast<Eigen::MatrixXd>(field(d, "normals"));
  const std::vector<int64_t> start = vec_in<int64_t>(d, "fanStart");
  const std::vector<double> angle = vec_in<double>(d, "fanAngle");
  const auto dir = nb::cast<Array2<double>>(field(d, "fanDirection"));
  if (static_cast<int>(start.size()) != c.nV + 1 || start.back() != static_cast<int64_t>(angle.size()) || dir.shape(0) != angle.size() || dir.shape(1) != 3 ||
      static_cast<int>(c.edges.size()) != 2 * c.nE || c.star1.size() != c.nE || c.star0.size() != c.nV || c.star2.size() != c.nF)
    throw std::invalid_argument("the intrinsic complex's arrays do not match its nV, nE, nF: pass the dict made by signal.intrinsicComplex(surface)");
  c.fans.assign(static_cast<size_t>(c.nV), {});
  for (int v = 0; v < c.nV; ++v)
    for (int64_t i = start[static_cast<size_t>(v)]; i < start[static_cast<size_t>(v) + 1]; ++i)
      c.fans[static_cast<size_t>(v)].push_back({angle[static_cast<size_t>(i)], Eigen::Vector3d(dir.data()[3 * i], dir.data()[3 * i + 1], dir.data()[3 * i + 2])});
  return c;
}

nb::dict intrinsic_complex(const nb::dict& surface, double star1_floor) {
  const geom::SurfaceArrays s = surface_in(surface);
  nxr::signal::IntrinsicComplex c;
  {
    nb::gil_scoped_release release;
    c = nxr::signal::intrinsicComplex(s, star1_floor);
  }
  return intrinsic_complex_out(std::move(c));
}

// the 1-forms of the intrinsic complex (nxr/signal/intrinsic_forms.h) <-> tangent vectors
Eigen::MatrixXd sharp(const nb::dict& complex, const Eigen::MatrixXd& w) {
  const nxr::signal::IntrinsicComplex c = intrinsic_complex_in(complex);
  if (w.rows() != c.nE) throw std::invalid_argument("the 1-form has " + std::to_string(w.rows()) + " rows; the intrinsic complex has " + std::to_string(c.nE) + " edges");
  nb::gil_scoped_release release;
  return nxr::signal::sharp(c, w);
}

Eigen::MatrixXd flat(const nb::dict& complex, const Eigen::MatrixXd& J) {
  const nxr::signal::IntrinsicComplex c = intrinsic_complex_in(complex);
  if (J.rows() != 3 * c.nV) throw std::invalid_argument("the vector field has " + std::to_string(J.rows()) + " rows; expected 3 per vertex (" + std::to_string(3 * c.nV) + ")");
  nb::gil_scoped_release release;
  return nxr::signal::flat(c, J);
}

// the modes of the intrinsic forms (nxr::signal::Modes) <-> a dict: P [V, K], X2 [F, K], mu [K], kind [K]
nb::dict modes_out(nxr::signal::Modes&& m) {
  nb::dict out;
  out["P"] = std::move(m.P); out["X2"] = std::move(m.X2); out["mu"] = std::move(m.mu);
  out["kind"] = Eigen::VectorXi(Eigen::Map<const Eigen::VectorXi>(m.kind.data(), static_cast<Eigen::Index>(m.kind.size())));
  return out;
}

nxr::signal::Modes modes_in(const nb::dict& d) {
  nxr::signal::Modes m;
  m.P = nb::cast<Eigen::MatrixXd>(field(d, "P"));
  m.X2 = nb::cast<Eigen::MatrixXd>(field(d, "X2"));
  m.mu = nb::cast<Eigen::VectorXd>(field(d, "mu"));
  const Eigen::VectorXi kind = nb::cast<Eigen::VectorXi>(field(d, "kind"));
  m.kind.assign(kind.data(), kind.data() + kind.size());
  return m;
}

nb::dict modes_of(const Eigen::MatrixXd& eigenvectors, const Eigen::VectorXd& eigenvalues, const Eigen::VectorXi& kind, int nV, int nE, int nF) {
  ops::EigenResult r;
  r.eigenvectors = eigenvectors; r.eigenvalues = eigenvalues; r.kind = kind;
  r.k = static_cast<int>(eigenvalues.size()); r.nConverged = r.k;
  return modes_out(nxr::signal::modesOf(r, nV, nE, nF));
}

// each form function takes the intrinsic complex made once by signal.intrinsicComplex(surface)
template <class F>
Eigen::MatrixXd on_complex(const nb::dict& complex, F&& f) {
  const nxr::signal::IntrinsicComplex c = intrinsic_complex_in(complex);
  nb::gil_scoped_release release;
  return f(c);
}

Eigen::MatrixXd project_one_form(const nb::dict& complex, const nb::dict& modes, const Eigen::MatrixXd& w) {
  const nxr::signal::Modes m = modes_in(modes);
  return on_complex(complex, [&](const nxr::signal::IntrinsicComplex& c) { return nxr::signal::projectOneForm(c, m, w); });
}
Eigen::MatrixXd synthesize_one_form(const nb::dict& complex, const nb::dict& modes, const Eigen::MatrixXd& coef) {
  const nxr::signal::Modes m = modes_in(modes);
  return on_complex(complex, [&](const nxr::signal::IntrinsicComplex& c) { return nxr::signal::synthesizeOneForm(c, m, coef); });
}
Eigen::MatrixXd project_two_form(const nb::dict& complex, const nb::dict& modes, const Eigen::MatrixXd& sigma) {
  const nxr::signal::Modes m = modes_in(modes);
  return on_complex(complex, [&](const nxr::signal::IntrinsicComplex& c) { return nxr::signal::projectTwoForm(c, m, sigma); });
}
Eigen::MatrixXd synthesize_two_form(const nb::dict& complex, const nb::dict& modes, const Eigen::MatrixXd& coef) {
  const nxr::signal::Modes m = modes_in(modes);
  return on_complex(complex, [&](const nxr::signal::IntrinsicComplex& c) { return nxr::signal::synthesizeTwoForm(c, m, coef); });
}
Eigen::MatrixXd coexact_stream(const nb::dict& complex, const nb::dict& modes, const Eigen::MatrixXd& coef) {
  const nxr::signal::Modes m = modes_in(modes);
  return on_complex(complex, [&](const nxr::signal::IntrinsicComplex& c) { return nxr::signal::coexactStream(c, m, coef); });
}
Eigen::MatrixXd exact_potential(const nb::dict& modes, const Eigen::MatrixXd& coef) {
  const nxr::signal::Modes m = modes_in(modes);
  nb::gil_scoped_release release;
  return nxr::signal::exactPotential(m, coef);
}
Eigen::MatrixXd phase_current(const nb::dict& complex, const Eigen::MatrixXcd& z) {
  return on_complex(complex, [&](const nxr::signal::IntrinsicComplex& c) { return nxr::signal::phaseCurrent(c, z); });
}
Eigen::MatrixXd phase_gradient_edges(const nb::dict& complex, const Eigen::MatrixXcd& x) {
  return on_complex(complex, [&](const nxr::signal::IntrinsicComplex& c) { return nxr::signal::phaseGradient(c, x); });
}

// the vertex discretisation (nxr/signal/phase_gradient.h, the reference): the mesh's vertex gradient, the current and the gradient as vectors at the vertices
nb::tuple vertex_gradient(const Array2<double>& positions, const Array2<int32_t>& faces) {
  if (positions.shape(1) != 3 || faces.shape(1) != 3) throw std::invalid_argument("positions and faces are [n, 3]");
  const Eigen::MatrixXd P = to_matrix<double>(positions);
  const Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> F =
      Eigen::Map<const Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>>(faces.data(), static_cast<Eigen::Index>(faces.shape(0)), 3);
  nxr::signal::Gradient G;
  {
    nb::gil_scoped_release release;
    G = nxr::signal::vertexGradient(P, F);
  }
  return nb::make_tuple(G[0], G[1], G[2]);
}

Eigen::MatrixXd phase_current_vertices(const Eigen::MatrixXcd& Z, int components, const Sparse& Gx, const Sparse& Gy, const Sparse& Gz, const Eigen::MatrixXd& N) {
  const nxr::signal::Gradient G{Gx, Gy, Gz};
  nb::gil_scoped_release release;
  return nxr::signal::phaseCurrent(Z, components, G, N);
}

Eigen::MatrixXd phase_gradient_vertices(const Eigen::MatrixXd& j, const Eigen::MatrixXd& p) {
  nb::gil_scoped_release release;
  return nxr::signal::phaseGradient(j, p);
}

nb::dict split_phase(const nb::dict& complex, const nb::dict& modes, const Eigen::VectorXd& w) {
  const nxr::signal::Modes m = modes_in(modes);
  const nxr::signal::IntrinsicComplex c = intrinsic_complex_in(complex);
  nxr::signal::PhaseSplit p;
  {
    nb::gil_scoped_release release;
    if (w.size() != c.nE) throw std::invalid_argument("splitPhase: the 1-form has " + std::to_string(w.size()) + " values; the intrinsic complex has " + std::to_string(c.nE) + " edges");
    p = nxr::signal::splitPhase(c, m, w);
  }
  nb::dict out;
  out["coef"] = p.coef; out["exact"] = p.exact; out["coexact"] = p.coexact;
  out["shareTiming"] = p.shareTiming; out["shareRotation"] = p.shareRotation; out["shareResidual"] = p.shareResidual;
  out["potential"] = p.potential; out["stream"] = p.stream; out["inconsistency"] = p.inconsistency;
  out["propagation"] = p.propagation;   // [3V, 1], x y z a vertex
  return out;
}

}  // namespace

void bind_operators(nb::module_& m) {
  using nb::arg;
  m.def("operators_assembleManifoldOperators", &manifold_operators, arg("surface"), arg("mass"));
  m.def("operators_assembleDECOperators", &dec_operators, arg("surface"));
  // operators_assemble = nxr::operators::hodgedirac::assemble (the native names are <toolbox>_<word>; the package calls it operators.hodgedirac.assemble)
  m.def("operators_assemble", &hodge_dirac_assemble, arg("surface"), arg("star1_floor"), arg("mollify"), arg("flip_eps"));
  m.def("operators_hodgeDiracModes", &hodge_dirac_modes, arg("pencil"), arg("count"), arg("both_signs"), arg("options"));
  m.def("operators_hodgeLadders", &hodge_ladders, arg("L0"), arg("M0"), arg("S2"), arg("M2"), arg("count"), arg("options"));
  m.def("operators_poisson", &poisson, arg("K"), arg("M"), arg("density_vertices"), arg("density_values"));
  m.def("signal_intrinsicComplex", &intrinsic_complex, arg("surface"), arg("star1_floor"));
  m.def("signal_sharp", &sharp, arg("complex"), arg("w"));
  m.def("signal_flat", &flat, arg("complex"), arg("J"));
  m.def("signal_modesOf", &modes_of, arg("eigenvectors"), arg("eigenvalues"), arg("kind"), arg("nV"), arg("nE"), arg("nF"));
  m.def("signal_projectOneForm", &project_one_form, arg("complex"), arg("modes"), arg("w"));
  m.def("signal_synthesizeOneForm", &synthesize_one_form, arg("complex"), arg("modes"), arg("coef"));
  m.def("signal_projectTwoForm", &project_two_form, arg("complex"), arg("modes"), arg("sigma"));
  m.def("signal_synthesizeTwoForm", &synthesize_two_form, arg("complex"), arg("modes"), arg("coef"));
  m.def("signal_exactPotential", &exact_potential, arg("modes"), arg("coef"));
  m.def("signal_coexactStream", &coexact_stream, arg("complex"), arg("modes"), arg("coef"));
  m.def("signal_phaseCurrent", &phase_current, arg("complex"), arg("z"));
  m.def("signal_phaseCurrent", &phase_current_vertices, arg("Z"), arg("components"), arg("Gx"), arg("Gy"), arg("Gz"), arg("N"));
  m.def("signal_phaseGradient", &phase_gradient_edges, arg("complex"), arg("x"));
  m.def("signal_phaseGradient", &phase_gradient_vertices, arg("j"), arg("p"));
  m.def("signal_vertexGradient", &vertex_gradient, arg("positions"), arg("faces"));
  m.def("signal_splitPhase", &split_phase, arg("complex"), arg("modes"), arg("w"));
}

}  // namespace nxrpy
