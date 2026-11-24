// The node `signal.flow_map` (pipelines/src/form_steps.cpp) through the graph's dispatch point, on a synthetic sphere whose Hodge-Dirac modes are solved here (never a real dataset):
//   coefficients made from a known complex field on the sphere (the same field every tile and bin, read through the exact modes' analysis rows), one frequency member, one scale member
//   holding every mode, the sphere's two hemispheres as the regions:
//   - a TRAVELLING phase (exp(i c z): the phase grows from pole to pole) gives a phase current that is mostly exact (travel), and a ROTATING one (x + i y: the phase turns about the
//     z axis) mostly coexact (rotation), whole sphere and per hemisphere;
//   - the per-region totals add up to the whole sphere's, member by member; the power equals |z|^2 x (the frequency member's g^2 summed over its bins) x tiles;
//   - refusals: regions that are not a partition of the surface, a scale set that is not a kernel set.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/pipelines/form_steps.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/mode_spectrum.h"

namespace ng = nxr::graph;
using ng::Json;

namespace {
template <class F> bool refused(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
}  // namespace

int main() {
  nxr::pipelines::graph::register_form_nodes();
  const ng::NodeSpec& node = ng::registry().at("signal.flow_map");
  CHECK(node.traits.declared() && node.reads_package_forms);

  std::vector<double> V; std::vector<int32_t> F;
  icosphere(3, V, F, 1.0);
  const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
  const auto s = nxr::geometry::surfaceArrays(V.data(), nV, F.data(), nF);
  const auto h = nxr::operators::hodgedirac::assemble(s);
  nxr::operators::HodgeSolveOptions so; so.limit = 60;
  const auto r = nxr::operators::hodgeDiracModes(h, 120, false, so);
  const nxr::signal::Modes md = nxr::signal::modesOf(r, h.nV, h.nE, h.nF);
  const nxr::signal::IntrinsicComplex cx = nxr::signal::intrinsicComplex(s);
  const int K = static_cast<int>(md.mu.size());

  ng::Array H;
  H.rows = K; H.cols = nV;
  for (int k = 0; k < K; ++k) for (int v = 0; v < nV; ++v) H.data.push_back(static_cast<float>(md.P(v, k)));
  H.meta["hodge"] = true; H.meta["mu"] = std::vector<double>(md.mu.data(), md.mu.data() + K); H.meta["kind"] = md.kind; H.meta["positions"] = V; H.meta["faces"] = F;
  H.meta["mass"] = std::vector<double>(cx.star0.data(), cx.star0.data() + nV);
  std::vector<float> x2(static_cast<std::size_t>(cx.nF * K));
  for (int f = 0; f < cx.nF; ++f) for (int k = 0; k < K; ++k) x2[static_cast<std::size_t>(f * K + k)] = static_cast<float>(md.X2(f, k));
  const auto* b = reinterpret_cast<const std::uint8_t*>(x2.data());
  H.meta["forms2"] = Json::binary(std::vector<std::uint8_t>(b, b + x2.size() * sizeof(float)));
  H.meta["forms2_shape"] = Json::array({cx.nF, K});
  H.meta["complex"] = Json{{"mollify", 1e-6}, {"flip_eps", 1e-6}, {"star1_floor", 0.0}};

  // the exact modes' analysis rows under unit_mass (as `spectral_coefficients` composes them)
  const nxr::signal::ModeAnalysis ma = nxr::signal::modeAnalysis(md.P, cx.star0, md.kind, nxr::signal::ModeKinds::exact, nxr::signal::ModeNormalisation::unit_mass);
  const std::int64_t R = static_cast<std::int64_t>(ma.modes.size()), T = 3, B = 9;
  const double fs = 100.0, nt = 64.0;                                       // bins k fs / nt: 0, 1.5625, ... Hz
  auto coefficients = [&](const Eigen::VectorXcd& z) {
    const Eigen::VectorXcd a = ma.rows.cast<std::complex<double>>() * z;
    ng::Array A;
    A.rows = T; A.cols = B * R * 2;
    for (std::int64_t t = 0; t < T; ++t) for (std::int64_t k = 0; k < B; ++k) for (std::int64_t j = 0; j < R; ++j) { A.single.push_back(static_cast<float>(a[j].real())); A.single.push_back(static_cast<float>(a[j].imag())); }
    A.meta["shape"] = Json::array({T, B, R, 2});
    A.meta["producer"] = Json{{"sfreq", fs}, {"tile_samples", nt}, {"first_bin", 0}, {"bins", B}, {"modes", ma.modes}, {"components", 1}, {"normalisation", "unit_mass"}};
    return A;
  };
  auto kernel_set = [](const Json& members) { ng::Array a; a.meta["params"] = Json{{"kernels", Json{{"members", members}}}}; a.meta["type"] = "set"; return a; };
  const Json fmem = Json::array({Json{{"start", 6.25}, {"stop", 6.25}, {"code", 0}, {"family", "gaussian"}, {"chart", "log2"}, {"gain", 1.0}, {"params", Json{{"fwhm", 0.5}}}}});
  const Json smem = Json::array({Json{{"start", 1.0}, {"stop", 1.0}, {"code", 0}, {"family", "box"}, {"chart", "log2"}, {"gain", 1.0}, {"params", Json{{"fwhm", 100.0}}}}});
  ng::Array regions; regions.meta["type"] = "set"; regions.meta["n_members"] = 2;
  for (int v = 0; v < nV; ++v) regions.data.push_back(V[static_cast<std::size_t>(3 * v + 2)] >= 0 ? 0.0 : 1.0);

  double g2 = 0;
  for (std::int64_t k = 0; k < B; ++k) { const double f = k * fs / nt; if (f > 0) { const double u = std::log2(f) - std::log2(6.25), sd = 0.5 / (2 * std::sqrt(2 * std::log(2.0))); const double g = std::exp(-u * u / (2 * sd * sd)); if (g * g > 1e-6) g2 += g * g; } }

  auto run = [&](const Eigen::VectorXcd& z, ng::Array sc, ng::Array rg) {
    const ng::Types none;
    ng::Shared shared;
    const Json no_params = Json::object();
    ng::Arrays c{{"coefficients", coefficients(z)}, {"package", H}, {"frequency", kernel_set(fmem)}, {"scale", sc}, {"regions", rg}};
    ng::NodeCall prep = ng::NodeCall::prepare(none, c, no_params, shared);
    ng::dispatch(node, prep);
    ng::NodeCall go = ng::NodeCall::run(prep.made, ng::Arrays{});
    ng::dispatch(node, go);
    return go.made_arrays;
  };
  Eigen::VectorXcd travel(nV), rotate(nV);
  for (int v = 0; v < nV; ++v) {
    const double x = V[3 * v], y = V[3 * v + 1], zc = V[3 * v + 2];
    travel[v] = std::exp(std::complex<double>(0, 2.0 * zc));
    rotate[v] = std::complex<double>(x, y);
  }
  for (int pass = 0; pass < 2; ++pass) {
    const ng::Arrays out = run(pass == 0 ? travel : rotate, kernel_set(smem), regions);
    const ng::Array& tot = out.at("totals");                                // [members = 1 x 1 x 2 regions][power, one_form, exact, coexact, count]
    CHECK(tot.rows == 2 && tot.cols == 5);
    const double one = tot.data[1] + tot.data[6], ex = tot.data[2] + tot.data[7], co = tot.data[3] + tot.data[8];
    std::printf("  %s: exact share %.3f, coexact share %.3f (whole sphere)\n", pass == 0 ? "travelling" : "rotating", ex / one, co / one);
    if (pass == 0) CHECK(ex / one > 0.9); else CHECK(co / one > 0.9);
    CHECK(tot.data[4] + tot.data[9] == nV);                                 // the regions hold every vertex once
    const ng::Array& maps = out.at("maps");                                 // [V x 1 member][6 measures]
    double power = 0, power_maps = 0;
    for (int v = 0; v < nV; ++v) power_maps += maps.data[static_cast<std::size_t>(v * 6)];
    power = tot.data[0] + tot.data[5];
    CHECK(std::abs(power - power_maps) < 1e-9 * power);                     // the totals are the maps summed over the regions
    // the power per vertex: |z_kept|^2 (the field read back through the kept modes) x g^2 summed x tiles
    const Eigen::VectorXcd zk = ma.synthesis.cast<std::complex<double>>() * (ma.rows.cast<std::complex<double>>() * (pass == 0 ? travel : rotate));
    double worst = 0, scale = 0;
    for (int v = 0; v < nV; ++v) { const double want = std::norm(zk[v]) * g2 * T; worst = std::max(worst, std::abs(maps.data[static_cast<std::size_t>(v * 6)] - want)); scale = std::max(scale, want); }
    CHECK(worst < 1e-4 * scale);
  }
  ng::Array not_partition; not_partition.meta["type"] = "indices"; not_partition.data = {0, 1};
  CHECK(refused([&] { run(travel, kernel_set(smem), not_partition); }));
  ng::Array not_kernels; not_kernels.meta["type"] = "set";
  CHECK(refused([&] { run(travel, not_kernels, regions); }));
  return nxrtest::finish("test_flow_map");
}
