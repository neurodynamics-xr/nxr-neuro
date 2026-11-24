// nxr-pipelines — the intrinsic forms as graph nodes (docs/specs/2026-11-19-intrinsic-form-nodes.md).
//
// THE GRAPH. x (a complex scalar field on the surface a Hodge-Dirac package is solved on, streamed) --[signal.phase_split]--> per sample, at the vertices: the exact part's potential,
// the coexact part's stream function, the propagation (−∇ potential) and the ⋆1 energies of the 1-form, of its exact part, of its coexact part and of what the exact part leaves.
// The 1-form is the parameter `one_form`: phase_gradient (the reference: the phase step on the intrinsic edges) or phase_current (Im(conj(z_tail) z_tip)).
// The 1-forms live only inside the node: its ports carry vertex fields. The package's complex and its modes (the eigenvectors and the
// 2-form parts, which the runner binds in preparation: `reads_package_forms`) are bound once. Every output is pointwise; the shares of the energy are measurements: each energy
// reduced by `measurement.reduce` (measure sum under its name) over any partition, the shares derived at read (exact_share, coexact_share, residual_share, exact_misfit_share).
#include "nxr/pipelines/form_steps.h"

#include <cmath>
#include <complex>
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/geometry/surface_arrays.h"
#include "nxr/graph/node.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/mode_spectrum.h"
#include <tuple>

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {

const char* const kWho = "signal.phase_split";
const char* const kScalarOutputs[] = {"potential", "stream", "one_form_energy", "exact_energy", "coexact_energy", "exact_misfit_energy"};

struct SplitPrepared {
  std::shared_ptr<const nxr::signal::IntrinsicComplex> complex;
  std::shared_ptr<const nxr::signal::Modes> modes;
  bool phase_current = false;   // the parameter `one_form`: phase_current (else phase_gradient)
  std::string boundary;         // a fact of the method: none (a closed surface) | neumann_potential_dirichlet_stream (an open one)
};

// the package constant as the runner binds it (runner.cpp: the eigenvectors [K x V] mode-major, mu, kind, the glued surface's positions and faces; with `reads_package_forms`
// the 2-form parts [nF x K] and the complex's options) → the intrinsic complex the modes were solved on, and the modes' full forms
// `who` names the node in refusals; `use` the first modes read (0: all). The package's stored 1-form parts (`forms1`), when it has them and `stored_one_form_parts` asks, become the modes' X1.
SplitPrepared bind_package(const ng::Array& H, const char* who = kWho, std::int64_t use = 0, bool stored_one_form_parts = true) {
  const std::string me = who;
  if (!H.meta.value("hodge", false)) ng::refuse(me + ": package is not a Hodge-Dirac package (an operator of kind eigenvectors, <stem>_eigenvectors)");
  if (!H.meta.contains("forms2") || !H.meta.contains("complex")) ng::refuse(me + ": the package's 2-form parts were not bound (the node reads the package's full forms)");
  const std::int64_t Kall = H.rows, V = H.cols, K = use > 0 ? std::min(use, Kall) : Kall;
  const std::vector<double> pos = H.meta.at("positions").get<std::vector<double>>();
  const std::vector<std::int32_t> faces = H.meta.at("faces").get<std::vector<std::int32_t>>();
  if (static_cast<std::int64_t>(pos.size()) != 3 * V) ng::refuse(me + ": the package's surface has " + std::to_string(pos.size() / 3) + " vertices, its eigenvectors " + std::to_string(V));
  const Json& o = H.meta.at("complex");
  nxr::geometry::IntrinsicOptions io;
  io.mollify = o.value("mollify", io.mollify);
  io.flipEps = o.value("flip_eps", io.flipEps);
  const nxr::geometry::SurfaceArrays arrays = nxr::geometry::surfaceArrays(pos.data(), static_cast<int>(V), faces.data(), static_cast<int>(faces.size() / 3), io);
  auto c = std::make_shared<nxr::signal::IntrinsicComplex>(nxr::signal::intrinsicComplex(arrays, o.value("star1_floor", 0.0)));   // ⋆1 the pencil's (raiseStar1, the same floor)
  const std::vector<std::int64_t> shape = H.meta.at("forms2_shape").get<std::vector<std::int64_t>>();
  const auto& bin = H.meta.at("forms2").get_binary();
  if (shape.size() != 2 || shape[1] != Kall || shape[0] != c->nF || bin.size() != static_cast<std::size_t>(shape[0] * shape[1]) * sizeof(float))
    ng::refuse(me + ": the package's 2-form parts are not [" + std::to_string(c->nF) + " faces x " + std::to_string(K) + " modes] of the complex rebuilt from its surface (another triangulation)");
  auto m = std::make_shared<nxr::signal::Modes>();
  m->P.resize(V, K);
  for (std::int64_t k = 0; k < K; ++k) for (std::int64_t v = 0; v < V; ++v) m->P(v, k) = H.data[static_cast<std::size_t>(k * V + v)];
  m->X2.resize(c->nF, K);
  std::vector<float> x2(static_cast<std::size_t>(shape[0] * shape[1]));
  std::memcpy(x2.data(), bin.data(), bin.size());
  for (std::int64_t f = 0; f < c->nF; ++f) for (std::int64_t k = 0; k < K; ++k) m->X2(f, k) = x2[static_cast<std::size_t>(f * Kall + k)];
  if (stored_one_form_parts && H.meta.contains("forms1")) {          // the stored 1-form parts: read as they are, no rebuilding
    const std::vector<std::int64_t> s1 = H.meta.at("forms1_shape").get<std::vector<std::int64_t>>();
    const auto& b1 = H.meta.at("forms1").get_binary();
    if (s1.size() != 2 || s1[1] != Kall || s1[0] != c->nE || b1.size() != static_cast<std::size_t>(s1[0] * s1[1]) * sizeof(float))
      ng::refuse(me + ": the package's 1-form parts are not [" + std::to_string(c->nE) + " edges x " + std::to_string(Kall) + " modes] of the complex rebuilt from its surface (another triangulation)");
    std::vector<float> x1(static_cast<std::size_t>(s1[0] * s1[1]));
    std::memcpy(x1.data(), b1.data(), b1.size());
    m->X1.resize(c->nE, K);
    for (std::int64_t e = 0; e < c->nE; ++e) for (std::int64_t k = 0; k < K; ++k) m->X1(e, k) = x1[static_cast<std::size_t>(e * Kall + k)];
  }
  const std::vector<double> mu = H.meta.at("mu").get<std::vector<double>>();
  const std::vector<int> kind = H.meta.at("kind").get<std::vector<int>>();
  m->mu = Eigen::Map<const Eigen::VectorXd>(mu.data(), K);
  m->kind.assign(kind.begin(), kind.begin() + K);
  for (Eigen::Index k = 0; k < m->mu.size(); ++k)
    if (!(m->mu(k) > 0) || (m->kind[static_cast<std::size_t>(k)] != 0 && m->kind[static_cast<std::size_t>(k)] != 2))
      ng::refuse(me + ": mode " + std::to_string(k) + " is not an exact or coexact mode with a positive eigenvalue (a harmonic or negative mode)");
  // the boundary loops of the complex: an edge with one face (a column of d1 with one entry) lies on the boundary
  int boundary_edges = 0;
  {
    const Eigen::SparseMatrix<double> d1 = c->d1;
    for (int e = 0; e < d1.outerSize(); ++e) if (d1.col(e).nonZeros() == 1) ++boundary_edges;
  }
  return SplitPrepared{c, m, false, boundary_edges == 0 ? "none" : "neumann_potential_dirichlet_stream"};
}

ng::Block block_rows(const Eigen::MatrixXd& A, std::int64_t left, std::int64_t right) {
  ng::Block b;
  b.rows = A.rows(); b.width = A.cols(); b.left = left; b.right = right;
  b.re.resize(static_cast<std::size_t>(b.rows * b.width));
  for (Eigen::Index r = 0; r < A.rows(); ++r)
    for (Eigen::Index t = 0; t < A.cols(); ++t) b.re[static_cast<std::size_t>(r * b.width + t)] = static_cast<float>(A(r, t));
  return b;
}

ng::NodeSpec phase_split() {
  ng::NodeSpec n;
  n.name = kWho;
  n.kind = ng::Kind::chunk;
  n.reads_package_forms = true;
  n.inputs = {{"x", "a complex scalar field on the surface the package's modes are solved on (the glued surface of a cortex: `meeg.apply_kernel` with `glued`), streamed", false},
              {"package", "the Hodge-Dirac package (its eigenvectors operator); the runner binds its 2-form parts and its complex's options with it", false}};
  n.outputs = {{"potential", "the exact part's potential φ at the vertices (rad): d0 φ is the exact (curl-free) part of the 1-form", false},
               {"stream", "the coexact part's stream function at the vertices (rad)", false},
               {"propagation", "−∇φ at the vertices, x y z a vertex (rad per length unit, sharpened from the exact part)", false},
               {"one_form_energy", "the ⋆1 energy of the 1-form at the vertices (each edge's ⋆1 w² half at each end; rad²)", false},
               {"exact_energy", "the ⋆1 energy of its exact part at the vertices (rad²)", false},
               {"coexact_energy", "the ⋆1 energy of its coexact part at the vertices (rad²)", false},
               {"exact_misfit_energy", "the ⋆1 energy of w − the exact part at the vertices (rad²)", false}};
  n.parameters = {{"one_form", Json("phase_gradient"), "the 1-form that is split: phase_gradient (the reference: the phase step arg(x_tip conj(x_tail)) on the intrinsic edges) | phase_current (Im(conj(x_tail) x_tip), the amplitude-weighted phase step)", "-", {"phase_gradient", "phase_current"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "each sample on its own: the 1-form on the intrinsic edges (the phase step or the phase current), their split by inner products with the modes, the pointwise parts and energies; double, float32 result for float32 input");
  n.matches = "compute nxr::signal::phaseGradient(complex, x) or phaseCurrent(complex, x), splitPhase, vertexEnergy (the intrinsic forms; the Hodge-Dirac modes' full forms)";
  n.description = "a 1-form of a complex field (its phase step, or its phase current) split by the Hodge-Dirac modes per sample into its exact and coexact parts: the exact part's potential, the coexact part's stream function, the propagation and the ⋆1 energies at the vertices "
                  "(the shares are measurements: each energy summed over a partition, the ratios derived at read)";
  n.plan = [](const ng::Types& in, const Json&) {
    const auto it = in.find("x");
    if (it == in.end()) ng::refuse(std::string(kWho) + ": x is not given");
    const ng::PortType& x = it->second;
    if (!x.streamed()) ng::refuse(std::string(kWho) + ": x is a streamed field (samples or frames)");
    if (x.value != ng::ValueType::complex) ng::refuse(std::string(kWho) + ": x must be a complex scalar field (the analytic signal on the vertices): its phase is split");
    if (!in.count("package")) ng::refuse(std::string(kWho) + ": package is not given");
    // each output lies where x does (its value as the runner describes it: same manifolds and time line), real, made by a non-linear node
    auto made = [&x](ng::ValueType value) {
      ng::PortType y = x;
      y.value = value; y.complex_parts = false; y.components = value == ng::ValueType::vector3 ? 3 : 1; y.nonlinear = true; y.quantity.clear();
      if (y.rows > 0 && value == ng::ValueType::vector3) y.rows *= 3;
      if (y.facts.contains("value") && y.facts["value"].is_object()) {
        Json& v = y.facts["value"];
        v["complex"] = false;
        v["value_type"] = value == ng::ValueType::vector3 ? "vector3" : "scalar";
        v["facts"] = Json{{"nonlinear", true}, {"made_by", kWho}};
        v["unit"] = nullptr;
      }
      return y;
    };
    ng::Planned out;
    for (const char* o : kScalarOutputs) out.out[o] = made(ng::ValueType::real);
    out.out["propagation"] = made(ng::ValueType::vector3);
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& params, ng::Shared&) {
    const std::string form = params.value("one_form", std::string("phase_gradient"));
    if (form != "phase_gradient" && form != "phase_current") ng::refuse(std::string(kWho) + ": one_form is phase_gradient (the reference) or phase_current, not '" + form + "'");
    const auto it = c.find("package");
    if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(kWho) + ": package was not handed over");
    SplitPrepared s = bind_package(it->second);
    s.phase_current = form == "phase_current";
    return std::any(s);
  };
  n.record = [](const std::any& prepared) {
    const SplitPrepared& s = std::any_cast<const SplitPrepared&>(prepared);
    return Json{{"complex", Json{{"vertices", s.complex->nV}, {"edges", s.complex->nE}, {"faces", s.complex->nF}}}, {"modes", s.modes->mu.size()}, {"one_form", s.phase_current ? "phase_current" : "phase_gradient"}, {"boundary", s.boundary}};
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t keep, const ng::Place&) {
    const SplitPrepared& s = std::any_cast<const SplitPrepared&>(prepared);
    const nxr::signal::IntrinsicComplex& c = *s.complex;
    const ng::Block& b = in.at("x");
    if (b.rows != c.nV) ng::refuse(std::string(kWho) + ": x has " + std::to_string(b.rows) + " rows; the package's surface has " + std::to_string(c.nV) + " vertices (a field on that surface)");
    const std::int64_t L = std::min(keep, b.left), R = std::min(keep, b.right), first = b.left - L, m = b.own() + L + R;
    Eigen::MatrixXcd X(b.rows, m);
    for (std::int64_t r = 0; r < b.rows; ++r)
      for (std::int64_t t = 0; t < m; ++t) {
        const std::size_t at = static_cast<std::size_t>(r * b.width + first + t);
        X(r, t) = std::complex<double>(b.re[at], b.complex() ? b.im[at] : 0.0);
      }
    const Eigen::MatrixXd W = s.phase_current ? nxr::signal::phaseCurrent(c, X) : nxr::signal::phaseGradient(c, X);   // the 1-form on the intrinsic edges [nE x m]: inside the node only
    Eigen::MatrixXd pot(c.nV, m), str(c.nV, m), prop(3 * c.nV, m), parts(c.nE, 4 * m);
    for (std::int64_t t = 0; t < m; ++t) {
      const nxr::signal::PhaseSplit sp = nxr::signal::splitPhase(c, *s.modes, W.col(t));
      pot.col(t) = sp.potential; str.col(t) = sp.stream; prop.col(t) = sp.propagation.col(0);
      parts.col(t) = W.col(t); parts.col(m + t) = sp.exact; parts.col(2 * m + t) = sp.coexact; parts.col(3 * m + t) = W.col(t) - sp.exact;
    }
    const Eigen::MatrixXd E = nxr::signal::vertexEnergy(c, parts);       // [V x 4m]: w, exact, coexact, w - exact
    ng::Blocks out;
    out["potential"] = block_rows(pot, L, R);
    out["stream"] = block_rows(str, L, R);
    out["propagation"] = block_rows(prop, L, R);
    const char* const energies[] = {"one_form_energy", "exact_energy", "coexact_energy", "exact_misfit_energy"};
    for (int k = 0; k < 4; ++k) out[energies[k]] = block_rows(E.middleCols(k * m, m), L, R);
    return out;
  };
  return n;
}

// ── operators.hodge_one_form_parts ───────────────────────────────────────────────────────────────────────────────────
// THE MODES' 1-FORM PARTS of a stored Hodge-Dirac package, as an operator axis -> the complex's edges [nE x K]: the recipe `signal::oneFormParts` (dψ/(μ√2) exact, ⋆1⁻¹ d1ᵀ ⋆2 χ/(μ√2) coexact), run on
// the package's stored 0- and 2-form parts and its rebuilt complex. The eigensolve stores the same numbers (`<stem>_forms1`); this node makes them for a package solved before they were stored (the
// definition `hodgeforms1`), without solving again. Made a block of modes at a time, written straight into the output (no [nE x K] temporary).
ng::NodeSpec hodge_one_form_parts() {
  ng::NodeSpec n;
  n.name = "operators.hodge_one_form_parts";
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"package", "a Hodge-Dirac package (its eigenvectors operator); the runner binds its 2-form parts and its complex's options with it", false}};
  n.outputs = {{"forms1", "the modes' 1-form parts on the complex's edges [nE x K] (an operator from the mode axis, cell 0, to the complex, cell 1)", false}};
  n.parameters = {{"modes", Json(), "the first K modes of the package; null: all", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "made in double, once a run, a block of modes at a time; stored as float32 like the 2-form parts");
  n.matches = "compute nxr::signal::oneFormParts (the eigensolve's own 1-form rows: nxr::operators::hodgeDiracModes)";
  n.description = "the 1-form parts of a Hodge-Dirac package's modes on its complex's edges: dpsi/(mu sqrt2) for an exact mode, *1^-1 d1^T *2 chi/(mu sqrt2) for a coexact one";
  n.plan = [](const ng::Types& in, const Json&) {
    if (!in.count("package")) ng::refuse("operators.hodge_one_form_parts: package is not given");
    ng::Planned out;
    ng::PortType t;
    t.cls = ng::PortClass::operator_;
    t.facts["cls"] = "operator";
    out.out["forms1"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    const auto it = c.find("package");
    if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse("operators.hodge_one_form_parts: package was not handed over");
    const std::int64_t use = p.contains("modes") && !p["modes"].is_null() ? p["modes"].get<std::int64_t>() : 0;
    if (p.contains("modes") && !p["modes"].is_null() && use < 1) ng::refuse("operators.hodge_one_form_parts: modes is at least 1");
    SplitPrepared s = bind_package(it->second, "operators.hodge_one_form_parts", use, false);
    return std::any(std::make_pair(std::move(s), it->second.meta.value("name", std::string())));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& pr = std::any_cast<const std::pair<SplitPrepared, std::string>&>(prepared);
    const nxr::signal::IntrinsicComplex& c = *pr.first.complex;
    const nxr::signal::Modes& m = *pr.first.modes;
    const Eigen::Index K = m.mu.size(), block = 128;
    ng::Array a;
    a.rows = c.nE; a.cols = K;
    a.data.resize(static_cast<std::size_t>(a.rows * a.cols));
    for (Eigen::Index k0 = 0; k0 < K; k0 += block) {
      const Eigen::Index w = std::min(block, K - k0);
      nxr::signal::Modes part;
      part.P = m.P.middleCols(k0, w); part.X2 = m.X2.middleCols(k0, w); part.mu = m.mu.segment(k0, w);
      part.kind.assign(m.kind.begin() + k0, m.kind.begin() + k0 + w);
      const Eigen::MatrixXd X = nxr::signal::oneFormParts(c, part);
      for (Eigen::Index e = 0; e < c.nE; ++e) for (Eigen::Index j = 0; j < w; ++j) a.data[static_cast<std::size_t>(e * K + k0 + j)] = X(e, j);
    }
    const std::string stem = pr.second;
    a.meta["describe"] = Json{{"precision", "float32"}, {"chunks", Json::array({c.nE, 1})}, {"shards", Json::array({c.nE, std::min<std::int64_t>(K, 64)})},
                              {"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"from_cell", "0"}, {"to_cell", "1"},
                              {"producer", Json{{"package", stem}, {"made_by", "operators.hodge_one_form_parts"}}},
                              {"params", Json{{"forms", "1-form part of each Dirac eigenvector: d psi/(mu sqrt2) (exact), *1^-1 d1^T *2 chi/(mu sqrt2) (coexact); *1-orthogonal, sum *1 phi^2 = 1/2"}}},
                              {"description", "Hodge-Dirac modes' 1-form parts on the intrinsic edges; with " + stem + "_eigenvectors and " + stem + "_forms2 the full forms"}};
    return ng::Arrays{{"forms1", std::move(a)}};
  };
  return n;
}

// THE WHOLE-RUN FLOW MAPS of the cells of a joint filterbank, read from the stored mode-space Fourier coefficients A [tiles x bins x rows] (`spectral_coefficients`), the cells being the
// members of the PRODUCT of two stored kernel sets: frequency members (kernels on the coefficients' frequency line) x scale members (kernels on the package's mode axis). For each cell:
// the member's cross-spectrum over the tiles in its low-rank form Y = A sqrt(g_f^2) (rows x tiles.bins; the quadratic field takes g^2), each Cartesian component's field on the
// surface z = Phi diag(g_s) Y (Phi the kept modes' 0-form parts, normalised as the coefficients were), then pointwise: the POWER per vertex sum |z|^2 (0-form) and the PHASE CURRENT
// Im(conj(z_tail) z_tip) per intrinsic edge summed over the columns and components (1-form), and the phase current's Hodge split by the package's modes. Nothing reduces across space;
// across time each map is the sum over the coefficients' tiles and the member's bins (the mean is sum / tiles). Every value is ONE MEASUREMENT per cell: found by its member.
double kernel_gain(const Json& k, double x) {
  const std::string fam = k.at("family"), chart = k.at("chart");
  const double c = k.at("start").get<double>(), fw = k.at("params").at("fwhm").get<double>(), g = k.value("gain", 1.0);
  if (chart != "linear" && !(x > 0)) return 0.0;
  const double u = chart == "log2" ? std::log2(x) - std::log2(c) : x - c;
  if (fam == "gaussian") { const double sd = fw / (2.0 * std::sqrt(2.0 * std::log(2.0))); return g * std::exp(-u * u / (2.0 * sd * sd)); }
  if (fam == "box") return std::abs(u) <= fw / 2.0 ? g : 0.0;
  ng::refuse("signal.flow_map: a kernel's family is gaussian or box, not " + fam);
}
std::vector<Json> kernel_members(const ng::Array& sel, const char* port) {
  const Json p = sel.meta.value("params", Json::object());
  if (!p.contains("kernels") || !p["kernels"].contains("members")) ng::refuse(std::string("signal.flow_map: ") + port + " is not a kernel set (`nxr-neuro run select_kernels`)");
  return p["kernels"]["members"].get<std::vector<Json>>();
}
ng::NodeSpec flow_map() {
  ng::NodeSpec n;
  n.name = "signal.flow_map";
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"coefficients", "the stored mode-space Fourier coefficients (`spectral_coefficients`): [tiles x bins x rows] complex, on the package's mode axis", false},
              {"package", "the Hodge-Dirac package the coefficients are on (its eigenvectors operator); the runner binds its form parts and its complex's options with it", false},
              {"frequency", "the frequency kernel set (stored: `select_kernels` on the coefficients' frequency line): the product's first factor", false},
              {"scale", "the scale kernel set (stored: `select_kernels` on the package's mode axis): the product's second factor", false},
              {"regions", "a stored partition of the package's surface (the caller's: the whole surface as one member, lobes, an atlas): the regions each cell's totals are summed over", false}};
  n.outputs = {{"totals", "per member of frequency x scale x regions (one value a member): the sums of power, one_form_energy, exact_energy, coexact_energy over the region's vertices, and its vertex count (the shares derived at read)", false},
               {"maps", "per cell (a member of frequency x scale, the frequency member fastest), per vertex: power, exact_potential, coexact_stream, one_form_energy, exact_energy, coexact_energy", false},
               {"edges", "per cell, per intrinsic edge: the phase current (a 1-form)", false}};
  n.parameters = {};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run: per frequency member the low-rank cross-spectrum of the stored tiles, per cell the fields of its columns synthesised in blocks, pointwise maps in double");
  n.matches = "the nxr flow scripts' whole-recording flow map per cell (power, phase current, Hodge split); compute nxr::signal::phaseCurrent, splitPhase, vertexEnergy; nxr::selection's kernel members";
  n.description = "the whole-run flow maps of every cell of a frequency kernel set x a scale kernel set, from stored mode-space Fourier coefficients: per cell, power per vertex, phase current per edge, "
                  "its Hodge split; each a measurement over the product's members";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"coefficients", "package", "frequency", "scale", "regions"}) if (!in.count(k)) ng::refuse(std::string("signal.flow_map: ") + k + " is not given");
    ng::Planned out;
    for (const char* o : {"totals", "maps", "edges"}) { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "measurement"; out.out[o] = t; }
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json&, ng::Shared&) {
    for (const char* k : {"coefficients", "package", "frequency", "scale", "regions"})
      if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("signal.flow_map: ") + k + " was not handed over");
    const ng::Array& G = c.at("regions");
    if (G.meta.value("type", std::string()) != "set" || G.data.empty()) ng::refuse("signal.flow_map: regions is a stored partition of the surface (a member number per vertex)");
    std::vector<std::int32_t> region(G.data.begin(), G.data.end());
    return std::any(std::make_tuple(bind_package(c.at("package"), "signal.flow_map"), c.at("package"), c.at("coefficients"), kernel_members(c.at("frequency"), "frequency"),
                                    kernel_members(c.at("scale"), "scale"), region, G.meta.value("n_members", std::int64_t(0))));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [sp, H, A, fk, sk, region, NRm] = std::any_cast<const std::tuple<SplitPrepared, ng::Array, ng::Array, std::vector<Json>, std::vector<Json>, std::vector<std::int32_t>, std::int64_t>&>(prepared);
    const nxr::signal::IntrinsicComplex& c = *sp.complex;
    const nxr::signal::Modes& m = *sp.modes;
    const Json pr = A.meta.value("producer", Json::object());
    const std::vector<std::int64_t> shape = A.meta.at("shape").get<std::vector<std::int64_t>>();
    if (shape.size() != 4 || shape[3] != 2) ng::refuse("signal.flow_map: coefficients are not [tiles x bins x rows x 2] (complex)");
    const std::int64_t T = shape[0], B = shape[1], R = shape[2];
    const std::vector<std::int64_t> modes = pr.at("modes").get<std::vector<std::int64_t>>();
    const int comps = pr.value("components", 1);
    const std::int64_t K = static_cast<std::int64_t>(modes.size()), V = m.P.rows();
    if (K * comps != R) ng::refuse("signal.flow_map: the coefficients' rows are not their modes x components");
    if (static_cast<std::int64_t>(region.size()) != m.P.rows()) ng::refuse("signal.flow_map: regions has " + std::to_string(region.size()) + " elements, the package's surface " + std::to_string(m.P.rows()) + " vertices");
    std::int64_t NR = NRm;
    for (const auto r : region) NR = std::max<std::int64_t>(NR, r + 1);
    const double fs = pr.at("sfreq").get<double>(), nt = pr.at("tile_samples").get<double>();
    const std::int64_t first = pr.at("first_bin").get<std::int64_t>();
    const std::vector<double> mass = H.meta.at("mass").get<std::vector<double>>();
    const bool unit_mass = pr.value("normalisation", Json("unit_mass")) == Json("unit_mass");
    Eigen::MatrixXd Phi(V, K);
    for (std::int64_t j = 0; j < K; ++j) {
      Phi.col(j) = m.P.col(modes[static_cast<std::size_t>(j)]);
      if (unit_mass) { double s2 = 0; for (Eigen::Index v = 0; v < V; ++v) s2 += mass[static_cast<std::size_t>(v)] * Phi(v, j) * Phi(v, j); Phi.col(j) /= std::sqrt(s2); }
    }
    const std::int64_t NF = static_cast<std::int64_t>(fk.size()), NS = static_cast<std::int64_t>(sk.size()), M = NF * NS;
    // the scale members' amplitudes on the kept modes (anchors on the mode axis's coordinate mu)
    Eigen::MatrixXd Hs(K, NS);
    for (std::int64_t s2 = 0; s2 < NS; ++s2) for (std::int64_t j = 0; j < K; ++j) Hs(j, s2) = kernel_gain(sk[static_cast<std::size_t>(s2)], m.mu[modes[static_cast<std::size_t>(j)]]);
    const int NV = 6;                                                            // power, exact_potential, coexact_stream, one_form_energy, exact_energy, coexact_energy
    ng::Array maps, edges, totals;
    const int NT = 5;                                                            // per region: power, one_form_energy, exact_energy, coexact_energy (sums over its vertices), count
    totals.rows = NR * M; totals.cols = NT; totals.data.assign(static_cast<std::size_t>(NR * M * NT), 0.0);
    maps.rows = V * M; maps.cols = NV; maps.data.assign(static_cast<std::size_t>(V * M * NV), 0.0);
    edges.rows = c.nE * M; edges.cols = 1; edges.data.assign(static_cast<std::size_t>(c.nE * M), 0.0);
    Json dof = Json::array();
    for (std::int64_t f = 0; f < NF; ++f) {
      // the frequency member's bins and their weight g^2 (a quadratic field); its low-rank cross-spectrum Y [rows x tiles.bins]
      std::vector<std::int64_t> bins; std::vector<double> w;
      double wmax = 0;
      for (std::int64_t k = 0; k < B; ++k) { const double g = kernel_gain(fk[static_cast<std::size_t>(f)], static_cast<double>(first + k) * fs / nt); wmax = std::max(wmax, g * g); }
      for (std::int64_t k = 0; k < B; ++k) { const double g = kernel_gain(fk[static_cast<std::size_t>(f)], static_cast<double>(first + k) * fs / nt); if (g * g > 1e-6 * wmax && g > 0) { bins.push_back(k); w.push_back(g * g); } }
      double sw = 0, sw2 = 0; for (double x : w) { sw += x; sw2 += x * x; }
      dof.push_back(Json{{"member", f}, {"bins", bins.size()}, {"weighted_bins", sw2 > 0 ? sw * sw / sw2 : 0.0}, {"tiles", T}});
      if (bins.empty()) continue;
      const Eigen::Index cols = T * static_cast<Eigen::Index>(bins.size());
      Eigen::MatrixXcd Y(R, cols);
      for (std::int64_t t = 0; t < T; ++t) for (std::size_t q = 0; q < bins.size(); ++q) {
        const double sq = std::sqrt(w[q]);
        for (std::int64_t r = 0; r < R; ++r) {
          const std::size_t at = static_cast<std::size_t>(((t * B + bins[q]) * R + r) * 2);
          Y(r, t * static_cast<Eigen::Index>(bins.size()) + static_cast<Eigen::Index>(q)) = sq * std::complex<double>(A.at(at), A.at(at + 1));
        }
      }
      for (std::int64_t s2 = 0; s2 < NS; ++s2) {
        const std::int64_t cell = f + NF * s2;                                   // the product's member: the first factor (frequency) fastest
        Eigen::VectorXd power = Eigen::VectorXd::Zero(V), J = Eigen::VectorXd::Zero(c.nE);
        for (int cc = 0; cc < comps; ++cc) {
          const Eigen::MatrixXcd Yc = Hs.col(s2).cast<std::complex<double>>().asDiagonal() * Y.middleRows(cc * K, K);
          for (Eigen::Index c0 = 0; c0 < cols; c0 += 128) {                       // the synthesis in column blocks: at most V x 128 complex at a time
            const Eigen::Index wdt = std::min<Eigen::Index>(128, cols - c0);
            const Eigen::MatrixXcd Z = Phi * Yc.middleCols(c0, wdt);
            power += Z.cwiseAbs2().rowwise().sum();
            J += nxr::signal::phaseCurrent(c, Z).rowwise().sum();
          }
        }
        const nxr::signal::PhaseSplit split = nxr::signal::splitPhase(c, m, J);
        Eigen::MatrixXd parts(c.nE, 3);
        parts.col(0) = J; parts.col(1) = split.exact; parts.col(2) = split.coexact;
        const Eigen::MatrixXd E = nxr::signal::vertexEnergy(c, parts);
        for (std::int64_t v = 0; v < V; ++v) {
          const std::size_t r = static_cast<std::size_t>((v * M + cell) * NV);
          maps.data[r] = power[v]; maps.data[r + 1] = split.potential[v]; maps.data[r + 2] = split.stream[v];
          maps.data[r + 3] = E(v, 0); maps.data[r + 4] = E(v, 1); maps.data[r + 5] = E(v, 2);
        }
        for (std::int64_t e = 0; e < c.nE; ++e) edges.data[static_cast<std::size_t>(e * M + cell)] = J[e];
        for (std::int64_t v = 0; v < V; ++v) {                                   // the totals: a reduction across space is a measurement over a partition (the caller's regions)
          const std::int32_t g = region[static_cast<std::size_t>(v)];
          if (g < 0 || !std::isfinite(power[v]) || !std::isfinite(E(v, 0))) continue;   // a vertex in no region, or of no value (a medial wall the glue cut)
          const std::size_t r = static_cast<std::size_t>((g * M + cell) * NT);
          totals.data[r] += power[v]; totals.data[r + 1] += E(v, 0); totals.data[r + 2] += E(v, 1); totals.data[r + 3] += E(v, 2); totals.data[r + 4] += 1.0;
        }
      }
    }
    const Json attrs = Json{{"rows", "the vertices of the package's surface"}, {"members", "the cells of frequency x scale, the frequency member fastest"}, {"frequency_members", dof},
                            {"sum_over", "the coefficients' tiles and the member's bins (the mean is sum / tiles)"}};
    maps.meta["describe"] = Json{{"measures", Json::array({Json{{"measure", "power"}}, Json{{"measure", "exact_potential"}}, Json{{"measure", "coexact_stream"}}, Json{{"measure", "one_form_energy"}},
                                                         Json{{"measure", "exact_energy"}}, Json{{"measure", "coexact_energy"}}})},
                                 {"values_shape", Json::array({V, M})}, {"attributes", attrs}};
    Json eattrs = attrs; eattrs["rows"] = "the edges of the package's intrinsic complex (tail -> tip)";
    Json tattrs = attrs; tattrs["members"] = "the cells of frequency x scale x regions, the frequency member fastest, then the scale member, then the region"; tattrs.erase("rows");
    totals.meta["describe"] = Json{{"measures", Json::array({Json{{"measure", "power"}}, Json{{"measure", "one_form_energy"}}, Json{{"measure", "exact_energy"}}, Json{{"measure", "coexact_energy"}},
                                                           Json{{"measure", "count"}}})},
                                   {"values_shape", Json::array({NR * M})}, {"attributes", tattrs}};
    edges.meta["describe"] = Json{{"measures", Json::array({Json{{"measure", "phase_current"}}})}, {"values_shape", Json::array({c.nE, M})}, {"attributes", eattrs}};
    return ng::Arrays{{"totals", std::move(totals)}, {"maps", std::move(maps)}, {"edges", std::move(edges)}};
  };
  return n;
}
}  // namespace

void register_form_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {phase_split(), hodge_one_form_parts(), flow_map()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
