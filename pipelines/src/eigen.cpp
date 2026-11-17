#include "nxr/pipelines/eigen.h"
#include "nxr/io/subject_reads.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/run_shipped.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <chrono>
#include <cstring>
#include <numeric>

#include "nxr/compute.h"

#ifndef NXR_COMPUTE_VERSION
#define NXR_COMPUTE_VERSION "unknown"
#endif
#ifndef NXR_IO_VERSION
#define NXR_IO_VERSION "0.0.0"
#endif

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {

template <class T>
std::vector<T> read_vector(const nxr::io::Store& store, const std::string& key) {
  ArrayReader r = ArrayReader::open(store, key);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
  return v;
}

}  // namespace

HodgeDirac solve_hodge_dirac(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, int modes, const SolveMethod& method) {
  const auto t0 = std::chrono::steady_clock::now();
  const int nV = static_cast<int>(vertices.size() / 3), nF = static_cast<int>(faces.size() / 3);
  // the Hodge–Dirac's triangulation: THE one Delaunay construction (mollify, the mixed-Voronoi mass of the mollified input recorded, flip to Delaunay) —
  // exactly as nxr and the Studio prepare a surface; `mollify` and `flip_eps` are the named options
  nxr::manifold::DelaunayOptions dopt;
  dopt.mollify = method.mollify; dopt.flipEps = method.flip_eps;
  nxr::manifold::Manifold m(vertices.data(), nV, faces.data(), nF, /*intrinsicDelaunay=*/true, dopt);
  const nxr::manifold::ops::hodgedirac::HodgeDirac hd =
      nxr::manifold::ops::hodgedirac::assemble(m, method.star1_floor.value_or(0.0), method.mollify, method.flip_eps);
  const int nE = hd.nE;
  const std::int64_t clamped = hd.star1Clamped;

  // *1 = the cotan weights (nxr: none floored; Rheome: floor 1e-6; here the option `star1_floor`, unset = none), and its inverse
  const Eigen::VectorXd& s1 = hd.star1;
  const Eigen::VectorXd s1_inv = s1.cwiseInverse();

  // the exact ladder: (d0^T *1 d0, *0 = the input's mixed Voronoi mass)
  Eigen::SparseMatrix<double> M0(nV, nV);
  for (int i = 0; i < nV; ++i) M0.insert(i, i) = hd.star0(i);
  M0.makeCompressed();
  const Eigen::SparseMatrix<double> L0 = hd.d0.transpose() * s1.asDiagonal() * hd.d0;
  // the coexact ladder: (*2 d1 *1^-1 d1^T *2, *2)
  Eigen::SparseMatrix<double> star2(nF, nF);
  for (int f = 0; f < nF; ++f) star2.insert(f, f) = hd.star2(f);
  star2.makeCompressed();
  const Eigen::SparseMatrix<double> B2 = star2 * hd.d1;
  const Eigen::SparseMatrix<double> S2 = B2 * s1_inv.asDiagonal() * B2.transpose();

  // both ladders, the constants out, ONE lambda cut, merged by mu: nxr-compute's hodgeLadders (the reference rule)
  const nxr::manifold::solve::HodgeLadders lad = nxr::manifold::solve::hodgeLadders(L0, M0, S2, star2, modes, method.solve);
  const Eigen::MatrixXd& psi = lad.exact.eigenvectors;
  const Eigen::MatrixXd& chi = lad.coexact.eigenvectors;

  // W: <phi2, w_v> = the sum over the faces round v of *2 phi2, divided by their count (nxr.pipelines.hodge._W)
  std::vector<int> count(static_cast<std::size_t>(nV), 0);
  std::vector<std::array<int, 3>> fv(static_cast<std::size_t>(nF));
  for (int f = 0; f < nF; ++f)
    for (int k = 0; k < 3; ++k) {
      fv[static_cast<std::size_t>(f)][static_cast<std::size_t>(k)] = hd.faces(f, k);
      ++count[static_cast<std::size_t>(hd.faces(f, k))];
    }
  const double r2 = std::sqrt(2.0);                                  // divided by, as nxr does (not multiplied by 1/sqrt 2)
  const std::size_t K = lad.modes.size(), V = static_cast<std::size_t>(nV);
  HodgeDirac out;
  out.n_vertices = nV;
  out.gauge = Json{{"gauge", hd.record.gauge}, {"gauge_record", Json{{"mollify", hd.record.mollify}, {"epsilon", hd.record.epsilon}, {"flipEps", hd.record.flipEps}, {"mass", hd.record.mass},
                                                                      {"boundary_vertices", hd.record.boundaryVertices}, {"fingerprint", hd.record.fingerprint}}}};
  out.mu.resize(K);
  out.kind.resize(K);
  out.eigenvectors.assign(V * K, 0.0f);
  out.top = lad.top;
  out.star1_clamped = clamped;
  out.mollify_epsilon = hd.record.epsilon;
  out.flips = hd.record.flips;
  std::vector<double> acc(V);
  for (std::size_t j = 0; j < K; ++j) {
    const nxr::manifold::solve::HodgeLadders::Mode& md = lad.modes[j];
    out.mu[j] = md.mu;
    out.kind[j] = static_cast<std::uint8_t>(md.kind);
    if (md.kind == 0) {
      ++out.n_exact;
      for (std::size_t v = 0; v < V; ++v) out.eigenvectors[v * K + j] = static_cast<float>(psi(static_cast<Eigen::Index>(v), md.col) / r2);
    } else {
      ++out.n_coexact;
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int f = 0; f < nF; ++f) {
        const double c = hd.star2(f) * chi(f, md.col);
        for (int k = 0; k < 3; ++k) acc[static_cast<std::size_t>(fv[static_cast<std::size_t>(f)][static_cast<std::size_t>(k)])] += c;
      }
      for (std::size_t v = 0; v < V; ++v) out.eigenvectors[v * K + j] = static_cast<float>(acc[v] / std::max(1, count[v]) / r2);
    }
  }
  // THE FULL FORMS: the intrinsic complex (the operator mesh the pencil was built on) and each coexact mode's 2-form χ/√2
  {
    const auto& cx = hd.complex;   // the intrinsic complex of geometry-central's triangulation, in the stored layout
    if (static_cast<int>(cx.faces.size()) != 3 * nF || static_cast<int>(cx.length.size()) != nE) throw nxr::io::Error(nxr::io::code::invalid_argument, "solve_hodge_dirac: the operator mesh and the pencil disagree in size");
    out.edges = cx.edges;
    out.lengths = cx.length;
    out.faces = cx.faces;
    out.face_edges = cx.faceEdges;
    // the stored orientation must be the pencil's: d0 (−1 tail, +1 tip) and d1 (the face's signed edges)
    {
      const Eigen::SparseMatrix<double, Eigen::RowMajor> D0(hd.d0), D1(hd.d1);
      for (int e = 0; e < nE; ++e) {
        double tail = 0, tip = 0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(D0, e); it; ++it) {
          if (it.col() == out.edges[2 * static_cast<std::size_t>(e)]) tail = it.value();
          if (it.col() == out.edges[2 * static_cast<std::size_t>(e) + 1]) tip = it.value();
        }
        if (tail != -1.0 || tip != 1.0) throw nxr::io::Error(nxr::io::code::invalid_argument, "solve_hodge_dirac: d0's orientation is not the operator mesh's at edge " + std::to_string(e));
      }
      for (int f = 0; f < nF; ++f) {
        double sum = 0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(D1, f); it; ++it) {
          double want = 0;
          for (int k = 0; k < 3; ++k) { const int se = out.face_edges[3 * static_cast<std::size_t>(f) + static_cast<std::size_t>(k)]; if (std::abs(se) - 1 == it.col()) want += se > 0 ? 1.0 : -1.0; }
          sum += std::abs(it.value() - want);
        }
        if (sum > 0) throw nxr::io::Error(nxr::io::code::invalid_argument, "solve_hodge_dirac: d1's signs are not the operator mesh's at face " + std::to_string(f));
      }
    }
    out.star0.assign(hd.star0.data(), hd.star0.data() + hd.star0.size());
    out.star1.assign(hd.star1.data(), hd.star1.data() + hd.star1.size());
    out.star2.assign(hd.star2.data(), hd.star2.data() + hd.star2.size());
    out.angle = cx.angle; out.angle_sum = cx.angleSum; out.reference = cx.reference;
    const std::size_t F_ = static_cast<std::size_t>(nF);
    out.forms2.assign(F_ * K, 0.0f);
    for (std::size_t j = 0; j < K; ++j) {
      const nxr::manifold::solve::HodgeLadders::Mode& md = lad.modes[j];
      if (md.kind != 2) continue;
      for (std::size_t f = 0; f < F_; ++f) out.forms2[f * K + j] = static_cast<float>(chi(static_cast<Eigen::Index>(f), md.col) / r2);
    }
  }
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

// THE VERB IS A DEFINITION NOW (pipelines/definitions/eigenmodes.yaml, the step `hodge_dirac` in eigen_steps.cpp): this is the library's thin caller of it (the spatial bands make missing
// eigenmodes, `dataset create --modes` makes the default subject's): it picks `glued` from the surface's pieces, runs the definition as `nxr-neuro run eigenmodes` does (the same plan, run,
// store, cache and --replace) and reads the stored package back for its caller.
EigenSolveResult eigen_solve(DatasetWriter& ds, SubjectWriter& sub, const EigenSolveOptions& opt, const std::function<void(const Json&)>& progress) {
  if (opt.modes < 2) throw Error(code::invalid_argument, "eigen_solve: modes must be at least 2");
  const auto t0 = std::chrono::steady_clock::now();
  const SurfaceInput in = read_surface(sub, opt.surface);
  const std::int64_t V = in.n_vertices();
  const int pieces = connected_pieces(in.faces, V);
  if (pieces > 2)
    throw Error(code::unsupported, in.name + " is " + std::to_string(pieces) + " pieces: the Hodge–Dirac solve takes a surface of one closed piece, or two (a cortex's hemispheres, glued by anatomy)");
  const bool glue = pieces == 2;
  const SolveMethod& sm = opt.method;
  const nxr::manifold::solve::HodgeSolveOptions dflt;
  Json set = Json{{"surface", in.name}, {"modes", opt.modes}, {"glued", glue}, {"mollify", sm.mollify}, {"flip_eps", sm.flip_eps},
                  {"shift", sm.solve.shift == nxr::manifold::solve::HodgeShift::Absolute ? "absolute" : "relative"}, {"slice", sm.solve.slice}, {"normalize", sm.solve.normalize}};
  if (sm.solve.sigma != dflt.sigma) set["sigma"] = sm.solve.sigma;       // a default stays unset, as on the command line: the same run, the same cache key
  if (sm.star1_floor) set["star1_floor"] = *sm.star1_floor;
  const std::vector<Json> reports = run_shipped(ds, sub, "eigenmodes", set, opt.replace, progress);
  if (reports.empty()) throw Error(code::no_such_row, "eigen_solve: the run made no report");
  const Json& rep = reports.front();

  // the stored package, by name
  const std::string stem = in.primary ? "hodgeDirac" : "hodgeDirac__" + in.name;
  const nxr::io::task::Subject sj = nxr::io::task::Subject::of(sub);
  auto node = [&](nxr::io::task::Cls c, const std::string& name) {
    nxr::io::task::Want w{.cls = c};
    w.name = name;
    const auto n = sj.find(w);
    if (!n || !n->path) throw Error(code::no_such_row, in.name + ": the stored Hodge–Dirac package is incomplete (" + name + " is missing)");
    return *n;
  };
  const auto axis = node(nxr::io::task::Cls::manifold, stem), kind = node(nxr::io::task::Cls::field, stem + "_kind"), vecs = node(nxr::io::task::Cls::operator_, stem + "_eigenvectors");
  EigenSolveResult r;
  r.hit = rep.value("hit", false);
  HodgeDirac& h = r.modes;
  h.mu = read_vector<double>(sub.store(), *axis.path);
  h.kind = read_vector<std::uint8_t>(sub.store(), *kind.path);
  h.eigenvectors = read_vector<float>(sub.store(), *vecs.path);
  for (auto k : h.kind) (k == 0 ? h.n_exact : h.n_coexact)++;
  const std::int64_t K = h.n_modes();
  if (K < 1 || static_cast<std::int64_t>(h.eigenvectors.size()) % K != 0) throw Error(code::invalid_argument, in.name + ": the stored eigenvectors do not fit the modes");
  h.n_vertices = static_cast<std::int64_t>(h.eigenvectors.size()) / K;
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  r.info = Json{{"axis", *axis.path}, {"surface", in.name}, {"glued", glue}, {"rows", h.n_vertices}, {"moved_aside", Json()}, {"n_modes", K}, {"n_exact", h.n_exact}, {"n_coexact", h.n_coexact},
                {"mu", Json::array({h.mu.front(), h.mu.back()})}, {"of", V}, {"replaced", opt.replace && !r.hit}, {"operator_id", vecs.id}, {"cache", r.hit ? "hit" : "miss"}, {"run", rep}};
  return r;
}

}  // namespace nxr::pipelines
