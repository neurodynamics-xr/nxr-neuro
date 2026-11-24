// The node `signal.phase_split` (pipelines/src/form_steps.cpp) through the graph's dispatch point, on a synthetic sphere whose Hodge-Dirac modes are solved here (never a real dataset):
//   - the package constant as the runner binds it (eigenvectors mode-major, mu, kind, the surface; the 2-form parts and the complex's options: `reads_package_forms`) is read back into the
//     same intrinsic complex and modes: every output equals the compute functions it wraps (phaseGradient on the edges, splitPhase, vertexEnergy), per sample, to float32;
//   - the outputs are pointwise: the result does not depend on how the samples are cut into blocks, and the halo kept is the caller's;
//   - the shares are measurements: the energies summed over the members of a partition (two hemispheres) give, through the registry's derived measures, the ratios of the sums; over the
//     whole sphere a travelling (exact) phase is nearly all timing;
//   - the node `operators.hodge_one_form_parts` makes the package's 1-form parts (= compute's `oneFormParts`); a package WITH stored `forms1` is read by the node without rebuilding them (same outputs);
//   - refusals: no `one_form` (the definition states it), a real x, a field on another surface, a package without its 2-form parts, 2-form parts of another triangulation.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"
#include "nxr/measurement/measures.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/pipelines/form_steps.h"
#include "nxr/signal/intrinsic_forms.h"

namespace ng = nxr::graph;
using ng::Json;

namespace {
template <class F> bool refused(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
}  // namespace

int main() {
  nxr::pipelines::graph::register_form_nodes();
  const ng::NodeSpec& node = ng::registry().at("signal.phase_split");
  CHECK(node.traits.declared() && node.reads_package_forms);

  // the sphere and its modes (as `eigen solve` makes them)
  std::vector<double> V; std::vector<int32_t> F;
  icosphere(2, V, F, 1.0);
  const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
  const auto s = nxr::geometry::surfaceArrays(V.data(), nV, F.data(), nF);
  const auto h = nxr::operators::hodgedirac::assemble(s);
  nxr::operators::HodgeSolveOptions o; o.limit = 30;
  const auto r = nxr::operators::hodgeDiracModes(h, 40, false, o);
  const nxr::signal::Modes md = nxr::signal::modesOf(r, h.nV, h.nE, h.nF);
  const nxr::signal::IntrinsicComplex cx = nxr::signal::intrinsicComplex(s);
  const int K = static_cast<int>(md.mu.size());

  // the package constant as the runner binds it: eigenvectors [K x V] mode-major (float32 as stored), the 2-form parts as float32 bytes [nF x K]
  auto package = [&](const nxr::signal::Modes& m, std::int64_t faces) {
    ng::Array H;
    H.rows = K; H.cols = nV;
    H.data.resize(static_cast<std::size_t>(K * nV));
    for (int k = 0; k < K; ++k) for (int v = 0; v < nV; ++v) H.data[static_cast<std::size_t>(k * nV + v)] = static_cast<float>(m.P(v, k));
    std::vector<double> mu(m.mu.data(), m.mu.data() + K);
    H.meta["hodge"] = true; H.meta["mu"] = mu; H.meta["kind"] = m.kind; H.meta["positions"] = V; H.meta["faces"] = F;
    std::vector<float> x2(static_cast<std::size_t>(faces * K), 0.0f);
    for (std::int64_t f = 0; f < std::min<std::int64_t>(faces, m.X2.rows()); ++f) for (int k = 0; k < K; ++k) x2[static_cast<std::size_t>(f * K + k)] = static_cast<float>(m.X2(f, k));
    const auto* b = reinterpret_cast<const std::uint8_t*>(x2.data());
    H.meta["forms2"] = Json::binary(std::vector<std::uint8_t>(b, b + x2.size() * sizeof(float)));
    H.meta["forms2_shape"] = Json::array({faces, K});
    H.meta["complex"] = Json{{"mollify", 1e-6}, {"flip_eps", 1e-6}, {"star1_floor", 0.0}};
    return H;
  };
  // the modes as the node sees them (float32 storage), for the direct computation
  nxr::signal::Modes m32 = md;
  m32.P = md.P.cast<float>().cast<double>();
  m32.X2 = md.X2.cast<float>().cast<double>();

  ng::Types in;
  ng::PortType x; x.cls = ng::PortClass::field; x.value = ng::ValueType::complex; x.axis = ng::Axis::samples; x.sfreq = 100.0; x.rows = nV;
  in["x"] = x;
  ng::PortType pk; pk.cls = ng::PortClass::operator_;
  in["package"] = pk;
  const ng::Planned planned = node.plan(in, Json::object());
  CHECK(planned.out.size() == 7 && planned.out.at("propagation").value == ng::ValueType::vector3 && planned.out.at("exact_energy").value == ng::ValueType::real);
  ng::Arrays constants{{"package", package(md, cx.nF)}};
  ng::Shared shared;
  const Json no_params = Json{{"one_form", "phase_gradient"}};   // the call holds pointers: named values outlive it. The definition states the 1-form (no default)
  {   // the 1-form is never defaulted: without `one_form` the node is refused, naming it
    const Json unstated = Json::object();
    std::string why;
    try { ng::NodeCall p = ng::NodeCall::prepare(in, constants, unstated, shared); ng::dispatch(node, p); } catch (const std::exception& e) { why = e.what(); }
    CHECK(why.find("one_form") != std::string::npos);
  }
  ng::NodeCall prep = ng::NodeCall::prepare(in, constants, no_params, shared);
  ng::dispatch(node, prep);

  // a travelling wave round the z axis with a slow drift, plus a rotating component: T samples
  const int T = 7;
  Eigen::MatrixXcd Z(nV, T);
  for (int v = 0; v < nV; ++v)
    for (int t = 0; t < T; ++t) {
      const double px = V[3 * v], py = V[3 * v + 1], pz = V[3 * v + 2];
      Z(v, t) = std::polar(1.0 + 0.3 * pz, 2.0 * pz - 0.4 * t + 0.3 * std::sin(3.0 * px + py + 0.2 * t));
    }
  Eigen::MatrixXd pot(nV, T), str(nV, T), prop(3 * nV, T), en(nV, 4 * T);   // the direct computation (below, on the float32 samples the node reads)
  const char* const names[] = {"potential", "stream", "propagation", "one_form_energy", "exact_energy", "coexact_energy", "exact_misfit_energy"};
  auto direct = [&](int o) -> Eigen::MatrixXd { return o == 0 ? pot : o == 1 ? str : o == 2 ? prop : Eigen::MatrixXd(en.middleCols((o - 3) * T, T)); };

  // run the node on blocks of `own` samples with one sample of halo each side, keep 0 (the halo is dropped) and gather the own columns
  auto run = [&](int own, std::int64_t keep) {
    std::vector<Eigen::MatrixXd> got(7);
    for (int o = 0; o < 7; ++o) got[static_cast<std::size_t>(o)] = Eigen::MatrixXd::Zero(o == 2 ? 3 * nV : nV, T);
    for (int first = 0; first < T; first += own) {
      const int n = std::min(own, T - first), left = first > 0 ? 1 : 0, right = first + n < T ? 1 : 0, width = left + n + right;
      ng::Block b; b.rows = nV; b.width = width; b.left = left; b.right = right;
      b.re.resize(static_cast<std::size_t>(nV * width)); b.im.resize(b.re.size());
      for (int v = 0; v < nV; ++v) for (int c = 0; c < width; ++c) {
        const auto z = Z(v, first - left + c);
        b.re[static_cast<std::size_t>(v * width + c)] = static_cast<float>(z.real()); b.im[static_cast<std::size_t>(v * width + c)] = static_cast<float>(z.imag());
      }
      const ng::Blocks blocks{{"x", b}};                                 // the call holds pointers: named values outlive it
      const ng::Place place{first, n, nullptr};
      ng::NodeCall call = ng::NodeCall::chunk(prep.made, blocks, keep, place);
      ng::dispatch(node, call);
      for (int o = 0; o < 7; ++o) {
        const ng::Block& y = call.made_blocks.at(names[o]);
        CHECK(y.left == std::min<std::int64_t>(keep, left) && y.right == std::min<std::int64_t>(keep, right) && y.own() == n && !y.complex());
        for (std::int64_t rr = 0; rr < y.rows; ++rr) for (int t = 0; t < n; ++t) got[static_cast<std::size_t>(o)](rr, first + t) = y.re[static_cast<std::size_t>(rr * y.width + y.left + t)];
      }
    }
    return got;
  };
  // the direct computation reads the same float32 samples the node does
  {
    Eigen::MatrixXcd Z32 = Z;
    for (int v = 0; v < nV; ++v) for (int t = 0; t < T; ++t) Z32(v, t) = {static_cast<float>(Z(v, t).real()), static_cast<float>(Z(v, t).imag())};
    Z = Z32;
  }
  {
    const Eigen::MatrixXd W32 = nxr::signal::phaseGradient(cx, Z);
    Eigen::MatrixXd parts(cx.nE, 4 * T);
    for (int t = 0; t < T; ++t) {
      const auto sp = nxr::signal::splitPhase(cx, m32, W32.col(t));
      pot.col(t) = sp.potential; str.col(t) = sp.stream; prop.col(t) = sp.propagation.col(0);
      parts.col(t) = W32.col(t); parts.col(T + t) = sp.exact; parts.col(2 * T + t) = sp.coexact; parts.col(3 * T + t) = W32.col(t) - sp.exact;
    }
    en = nxr::signal::vertexEnergy(cx, parts);
  }
  const auto whole = run(T, 0), a = run(2, 0), b3 = run(3, 1);
  double worst = 0, chunked = 0;
  for (int o = 0; o < 7; ++o) {
    const Eigen::MatrixXd d = direct(o);
    const double scale = std::max(1e-12, d.cwiseAbs().maxCoeff());
    worst = std::max(worst, (whole[static_cast<std::size_t>(o)] - d).cwiseAbs().maxCoeff() / scale);
    chunked = std::max(chunked, std::max((a[static_cast<std::size_t>(o)] - whole[static_cast<std::size_t>(o)]).cwiseAbs().maxCoeff(), (b3[static_cast<std::size_t>(o)] - whole[static_cast<std::size_t>(o)]).cwiseAbs().maxCoeff()) / scale);
  }
  std::printf("  node vs compute: %.2e (relative to each output's peak); blocks of 2 and 3 vs one block: %.2e\n", worst, chunked);
  CHECK(worst < 1e-5);
  CHECK(chunked == 0.0);

  // the shares are measurements: the energies summed over each hemisphere (a partition of two members), the ratios derived from the sums
  {
    const auto& reg = nxr::measurement::Registry::standard();
    const Eigen::MatrixXd& E0 = whole[3]; const Eigen::MatrixXd& Ex = whole[4]; const Eigen::MatrixXd& Ec = whole[5]; const Eigen::MatrixXd& Em = whole[6];
    double total[2] = {0, 0}, tim[2] = {0, 0}, rot[2] = {0, 0}, mis[2] = {0, 0}, all = 0, all_t = 0;
    const int t = 3;
    for (int v = 0; v < nV; ++v) {
      const int member = V[3 * v + 2] >= 0 ? 0 : 1;
      total[member] += E0(v, t); tim[member] += Ex(v, t); rot[member] += Ec(v, t); mis[member] += Em(v, t);
      all += E0(v, t); all_t += Ex(v, t);
    }
    for (int k = 0; k < 2; ++k) {
      auto g = [&](const std::string& n) { return n == "one_form_energy" ? total[k] : n == "exact_energy" ? tim[k] : n == "coexact_energy" ? rot[k] : mis[k]; };
      const double ts = reg.derive("exact_share", g), ti = reg.derive("exact_misfit_share", g);
      CHECK(std::abs(ts - tim[k] / total[k]) < 1e-12 && std::abs(ti - mis[k] / total[k]) < 1e-12 && ts > 0);   // over a tile the parts are not orthogonal (their cross terms do not cancel): a share may pass 1 there
    }
    const double whole_share = all_t / all;
    std::printf("  timing share over the sphere %.4f; per hemisphere %.4f %.4f\n", whole_share, tim[0] / total[0], tim[1] / total[1]);
    CHECK(whole_share > 0.5 && whole_share <= 1.0 + 1e-6);
  }

  // the parameter one_form: phase_current splits Im(conj(z_tail) z_tip) instead of the phase step (the node equals the compute functions); the record holds the choice and the boundary
  {
    Json params = Json::object(); params["one_form"] = "phase_current";
    ng::NodeCall pc = ng::NodeCall::prepare(in, constants, params, shared);
    ng::dispatch(node, pc);
    ng::NodeCall rc = ng::NodeCall::record(pc.made), rg = ng::NodeCall::record(prep.made);
    ng::dispatch(node, rc); ng::dispatch(node, rg);
    const Json rec = rc.made_json;
    CHECK(rec.value("one_form", "") == "phase_current" && rec.value("boundary", "") == "none");
    CHECK(rg.made_json.value("one_form", "") == "phase_gradient");
    ng::Block b; b.rows = nV; b.width = T; b.re.resize(static_cast<std::size_t>(nV * T)); b.im.resize(b.re.size());
    for (int v = 0; v < nV; ++v) for (int c = 0; c < T; ++c) { b.re[static_cast<std::size_t>(v * T + c)] = static_cast<float>(Z(v, c).real()); b.im[static_cast<std::size_t>(v * T + c)] = static_cast<float>(Z(v, c).imag()); }
    const ng::Blocks blocks{{"x", b}};
    const ng::Place place{0, T, nullptr};
    ng::NodeCall call = ng::NodeCall::chunk(pc.made, blocks, 0, place);
    ng::dispatch(node, call);
    const Eigen::MatrixXd J = nxr::signal::phaseCurrent(cx, Z);
    const Eigen::MatrixXd Ej = nxr::signal::vertexEnergy(cx, J);
    const ng::Block& y = call.made_blocks.at("one_form_energy");
    double worst_j = 0, apart = 0;
    for (int v = 0; v < nV; ++v) for (int t = 0; t < T; ++t) {
      worst_j = std::max(worst_j, std::abs(y.re[static_cast<std::size_t>(v * y.width + y.left + t)] - Ej(v, t)) / std::max(1e-12, Ej.cwiseAbs().maxCoeff()));
      apart = std::max(apart, std::abs(Ej(v, t) - whole[3](v, t)));
    }
    std::printf("  phase_current: node vs compute %.2e; the energy differs from the phase step's by %.2e\n", worst_j, apart);
    CHECK(worst_j < 1e-5 && apart > 1e-6);
    Json bad_form = Json::object(); bad_form["one_form"] = "phase_lag";
    CHECK(refused([&] { ng::NodeCall p = ng::NodeCall::prepare(in, constants, bad_form, shared); ng::dispatch(node, p); }));
  }

  // the node `operators.hodge_one_form_parts`: the package's 1-form parts from its stored 0- and 2-form parts = compute's oneFormParts (to float32), for all modes and for the first few
  const Eigen::MatrixXd X1 = nxr::signal::oneFormParts(cx, m32);
  {
    const ng::NodeSpec& pn = ng::registry().at("operators.hodge_one_form_parts");
    CHECK(pn.traits.declared() && pn.reads_package_forms);
    ng::Types pin; pin["package"] = pk;
    CHECK(pn.plan(pin, Json::object()).out.count("forms1") == 1);
    for (const std::int64_t use : {std::int64_t(0), std::int64_t(7)}) {
      Json pp = Json::object(); pp["modes"] = use > 0 ? Json(use) : Json();
      ng::NodeCall pprep = ng::NodeCall::prepare(pin, constants, pp, shared);
      ng::dispatch(pn, pprep);
      ng::NodeCall pcall = ng::NodeCall::run(pprep.made, ng::Arrays{});
      ng::dispatch(pn, pcall);
      const ng::Array& A = pcall.made_arrays.at("forms1");
      const int Ku = use > 0 ? static_cast<int>(use) : K;
      CHECK(A.rows == cx.nE && A.cols == Ku && A.meta["describe"]["to_cell"] == "1" && A.meta["describe"]["from_cell"] == "0");
      double e = 0;
      for (int ed = 0; ed < cx.nE; ++ed) for (int k = 0; k < Ku; ++k) e = std::max(e, std::abs(A.data[static_cast<std::size_t>(ed * Ku + k)] - X1(ed, k)));
      std::printf("  hodge_one_form_parts (%d modes) vs compute: %.2e\n", Ku, e);
      CHECK(e < 1e-12 * std::max(1.0, X1.cwiseAbs().maxCoeff()));
    }
  }
  // the package WITH stored 1-form parts: the node reads them (a scaled copy changes the outputs: they are the ones used), and the true ones give the outputs the rebuilding path gives
  {
    auto with_forms1 = [&](double scale) {
      ng::Array H = package(md, cx.nF);
      std::vector<float> x1(static_cast<std::size_t>(cx.nE * K));
      for (int e = 0; e < cx.nE; ++e) for (int k = 0; k < K; ++k) x1[static_cast<std::size_t>(e * K + k)] = static_cast<float>(scale * X1(e, k));
      const auto* bytes = reinterpret_cast<const std::uint8_t*>(x1.data());
      H.meta["forms1"] = Json::binary(std::vector<std::uint8_t>(bytes, bytes + x1.size() * sizeof(float)));
      H.meta["forms1_shape"] = Json::array({cx.nE, K});
      return H;
    };
    ng::Block bz; bz.rows = nV; bz.width = T; bz.re.resize(static_cast<std::size_t>(nV * T)); bz.im.resize(bz.re.size());
    for (int v = 0; v < nV; ++v) for (int c = 0; c < T; ++c) { bz.re[static_cast<std::size_t>(v * T + c)] = static_cast<float>(Z(v, c).real()); bz.im[static_cast<std::size_t>(v * T + c)] = static_cast<float>(Z(v, c).imag()); }
    const ng::Blocks bl{{"x", bz}};
    const ng::Place pl{0, T, nullptr};
    auto energy = [&](double scale) {
      ng::Arrays cs{{"package", with_forms1(scale)}};
      ng::NodeCall p = ng::NodeCall::prepare(in, cs, no_params, shared);
      ng::dispatch(node, p);
      ng::NodeCall c = ng::NodeCall::chunk(p.made, bl, 0, pl);
      ng::dispatch(node, c);
      return c.made_blocks.at("exact_energy");
    };
    const ng::Block yt = energy(1.0), ys = energy(1.5);
    double same = 0, differs = 0, peak = 1e-12;
    for (int v = 0; v < nV; ++v) for (int t = 0; t < T; ++t) {
      const std::size_t at = static_cast<std::size_t>(v * yt.width + yt.left + t);
      same = std::max(same, std::abs(yt.re[at] - whole[4](v, t))); differs = std::max(differs, std::abs(ys.re[at] - whole[4](v, t))); peak = std::max(peak, std::abs(whole[4](v, t)));
    }
    std::printf("  stored 1-form parts: outputs vs the rebuilt ones %.2e; with the parts scaled 1.5 they differ by %.2e (relative to the peak)\n", same / peak, differs / peak);
    CHECK(same < 1e-5 * peak && differs > 1e-2 * peak);
    ng::Arrays wrong{{"package", with_forms1(1.0)}};
    wrong["package"].meta["forms1_shape"] = Json::array({cx.nE + 1, K});
    CHECK(refused([&] { ng::NodeCall p = ng::NodeCall::prepare(in, wrong, no_params, shared); ng::dispatch(node, p); }));
  }

  // refusals
  {
    ng::Types bad = in; bad["x"].value = ng::ValueType::real;
    CHECK(refused([&] { node.plan(bad, Json::object()); }));
    ng::Arrays no_forms{{"package", package(md, cx.nF)}};
    no_forms["package"].meta.erase("forms2");
    CHECK(refused([&] { ng::NodeCall p = ng::NodeCall::prepare(in, no_forms, no_params, shared); ng::dispatch(node, p); }));
    ng::Arrays other{{"package", package(md, cx.nF + 3)}};
    CHECK(refused([&] { ng::NodeCall p = ng::NodeCall::prepare(in, other, no_params, shared); ng::dispatch(node, p); }));
    ng::Block b; b.rows = nV - 1; b.width = 1; b.re.assign(static_cast<std::size_t>(nV - 1), 1.0f); b.im = b.re;
    const ng::Blocks blocks{{"x", b}};
    const ng::Place place{0, 1, nullptr};
    CHECK(refused([&] { ng::NodeCall c = ng::NodeCall::chunk(prep.made, blocks, 0, place); ng::dispatch(node, c); }));
  }
  return nxrtest::finish("phase_split");
}
