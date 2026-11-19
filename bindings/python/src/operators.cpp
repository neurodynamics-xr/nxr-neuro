// nxr::operators for Python: the operators of a surface (made by geometry.surfaceArrays), the Hodge–Dirac pencil and its modes, the two Hodge ladders and
// the Poisson solve. A face: arguments and results converted (dense as NumPy, sparse as SciPy CSC), the library called, no maths.
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <optional>

#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/factorization.h"
#include "nxr/signal/intrinsic_forms.h"
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

// the 1-forms of the intrinsic complex (nxr/signal/intrinsic_forms.h), built from the surface's arrays by complexOf
Eigen::MatrixXd sharp(const nb::dict& surface, const Eigen::MatrixXd& w) {
  const geom::SurfaceArrays s = surface_in(surface);
  nb::gil_scoped_release release;
  const nxr::signal::Complex c = nxr::signal::complexOf(s);
  if (w.rows() != c.nE) throw std::invalid_argument("the 1-form has " + std::to_string(w.rows()) + " rows; the intrinsic complex has " + std::to_string(c.nE) + " edges");
  return nxr::signal::sharp(c, w);
}

Eigen::MatrixXd flat(const nb::dict& surface, const Eigen::MatrixXd& J) {
  const geom::SurfaceArrays s = surface_in(surface);
  nb::gil_scoped_release release;
  const nxr::signal::Complex c = nxr::signal::complexOf(s);
  if (J.rows() != 3 * c.nV) throw std::invalid_argument("the vector field has " + std::to_string(J.rows()) + " rows; expected 3 per vertex (" + std::to_string(3 * c.nV) + ")");
  return nxr::signal::flat(c, J);
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
  m.def("signal_sharp", &sharp, arg("surface"), arg("w"));
  m.def("signal_flat", &flat, arg("surface"), arg("J"));
}

}  // namespace nxrpy
