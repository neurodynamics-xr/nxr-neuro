// The bispectrum and flow-across-frequencies nodes (pipelines/src/bispectrum_steps.cpp) through the graph's dispatch point, on a synthetic sphere whose Hodge-Dirac modes are solved
// here (never a real dataset). Each recording is served by signal.overlapped_spectrum and gathered by signal.chunk_spectra, as the sensor flow pipeline does:
//   signal.bispectrum on a rising sawtooth at 8 Hz whose amplitude varies smoothly over the sphere: the harmonics triple of the 8 Hz atom has a biphase near +90 degrees (slow
//     rise, fast decay) on the constant coordinate and at every vertex of the surface; the triples listed are the readings asked; with two bands, one member per triple and band;
//   signal.flow_across_frequencies on waves travelling at 8 and 16 Hz: with the same wavelength their flows agree (edge cosine near 1) and the speed slope is near 1; with the same
//     speed the slope is near 0; refusals with their reasons.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <vector>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/pipelines/bispectrum_steps.h"
#include "nxr/pipelines/sensor_flow_steps.h"
#include "nxr/pipelines/spectral_steps.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/mode_spectrum.h"

namespace ng = nxr::graph;
using ng::Json;

namespace {
template <class F> bool refused(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
ng::Arrays run_array(const ng::NodeSpec& node, const ng::Arrays& c, const Json& p) {
  ng::Shared shared;
  const ng::Types types;
  const ng::Arrays none;
  ng::NodeCall prep = ng::NodeCall::prepare(types, c, p, shared);
  ng::dispatch(node, prep);
  ng::NodeCall go = ng::NodeCall::run(prep.made, c);   // as the executor does: run is handed the arrays prepare was
  ng::dispatch(node, go);
  return go.made_arrays;
}
ng::Array kernel_set(const Json& members) { ng::Array a; a.meta["params"] = Json{{"kernels", Json{{"members", members}}}}; a.meta["type"] = "set"; return a; }
Json atoms_at(const std::vector<double>& centres) {
  Json atoms = Json::array();
  for (const double c : centres)   // standard deviation centre / 6
    atoms.push_back(Json{{"start", c}, {"stop", c}, {"family", "gaussian"}, {"chart", "linear"}, {"gain", 2.0}, {"params", Json{{"fwhm_per_centre", 2.0 * std::sqrt(2.0 * std::log(2.0)) / 6.0}}}});
  return atoms;
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main() {
  nxr::pipelines::graph::register_sensor_flow_nodes();
  nxr::pipelines::graph::register_spectral_nodes();
  nxr::pipelines::graph::register_bispectrum_nodes();
  for (const char* name : {"signal.bispectrum", "signal.flow_across_frequencies"}) CHECK(ng::registry().at(name).traits.declared());

  // ── the sphere, its modes, as a package is handed over ──
  std::vector<double> V; std::vector<int32_t> F;
  icosphere(2, V, F, 1.0);
  const int nV = static_cast<int>(V.size() / 3), nF = static_cast<int>(F.size() / 3);
  const auto surface = nxr::geometry::surfaceArrays(V.data(), nV, F.data(), nF);
  const auto h = nxr::operators::hodgedirac::assemble(surface);
  nxr::operators::HodgeSolveOptions so; so.limit = 40;
  const auto r = nxr::operators::hodgeDiracModes(h, 80, false, so);
  const nxr::signal::Modes md = nxr::signal::modesOf(r, h.nV, h.nE, h.nF);
  const nxr::signal::IntrinsicComplex cx = nxr::signal::intrinsicComplex(surface);
  const int K = static_cast<int>(md.mu.size());
  ng::Array H;
  H.rows = K; H.cols = nV;
  for (int k = 0; k < K; ++k) for (int v = 0; v < nV; ++v) H.data.push_back(md.P(v, k));
  H.meta["hodge"] = true; H.meta["mu"] = std::vector<double>(md.mu.data(), md.mu.data() + K); H.meta["kind"] = md.kind; H.meta["positions"] = V; H.meta["faces"] = F;
  H.meta["mass"] = std::vector<double>(cx.star0.data(), cx.star0.data() + nV);
  std::vector<std::int64_t> keep(static_cast<std::size_t>(nV));
  for (int v = 0; v < nV; ++v) keep[static_cast<std::size_t>(v)] = v;
  H.meta["keep"] = keep;
  std::vector<float> x2(static_cast<std::size_t>(cx.nF * K));
  for (int f = 0; f < cx.nF; ++f) for (int k = 0; k < K; ++k) x2[static_cast<std::size_t>(f * K + k)] = static_cast<float>(md.X2(f, k));
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(x2.data());
  H.meta["forms2"] = Json::binary(std::vector<std::uint8_t>(bytes, bytes + x2.size() * sizeof(float)));
  H.meta["forms2_shape"] = Json::array({cx.nF, K});
  H.meta["complex"] = Json{{"mollify", 1e-6}, {"flip_eps", 1e-6}, {"star1_floor", 0.0}};

  // ── the serving operator (the constant and the exact modes) and the serving of a recording into chunk spectra ──
  ng::Array map;
  map.rows = nV; map.cols = 1;
  for (int v = 0; v < nV; ++v) map.data.push_back(v);
  ng::Types ptypes;
  ng::PortType pk; ptypes["package"] = pk; ptypes["channel_map"] = pk;
  ng::Shared shared;
  const ng::Arrays none;
  const ng::Arrays analysisInputs{{"package", H}, {"channel_map", map}};           // a call holds its inputs by reference: named, never temporaries
  const Json analysisParameters{{"modes", "exact_and_constant"}, {"normalisation", "unit_mass"}};
  ng::NodeCall aprep = ng::NodeCall::prepare(ptypes, analysisInputs, analysisParameters, shared);
  ng::dispatch(ng::registry().at("signal.mode_analysis"), aprep);
  ng::NodeCall arun = ng::NodeCall::run(aprep.made, none);
  ng::dispatch(ng::registry().at("signal.mode_analysis"), arun);
  const ng::Array O = arun.made_arrays.at("operator");
  const double fs = 128.0;
  const std::int64_t chunk = 256, overlap = 128, transform = 512, chunks = 8, total = chunks * chunk;
  const Json sp = Json{{"overlap_samples", overlap}, {"transform_samples", transform}, {"frequency_span", Json::array({0.0, 40.0})}};
  auto serve = [&](const std::function<double(int, double)>& make) {
    ng::Types stypes, gtypes;
    ng::PortType xt; xt.sfreq = fs; xt.axis = ng::Axis::samples; stypes["x"] = xt; stypes["operator"] = pk;
    ng::PortType st; st.sfreq = fs; for (const auto& [k, v] : sp.items()) st.facts[k] = v; gtypes["spectrum"] = st;   // the spectrum carries how it was made (overlapped_spectrum's plan)
    gtypes["operator"] = pk;
    const ng::NodeSpec& s = ng::registry().at("signal.overlapped_spectrum");
    const ng::NodeSpec& g = ng::registry().at("signal.chunk_spectra");
    const ng::Arrays operatorInput{{"operator", O}};
    ng::NodeCall sprep = ng::NodeCall::prepare(stypes, operatorInput, sp, shared);
    ng::dispatch(s, sprep);
    ng::NodeCall gprep = ng::NodeCall::prepare(gtypes, operatorInput, sp, shared);
    ng::dispatch(g, gprep);
    ng::NodeCall gs = ng::NodeCall::start(gprep.made);
    ng::dispatch(g, gs);
    std::any state = std::move(gs.made);
    for (std::int64_t k = 0; k < chunks; ++k) {
      const std::int64_t left = std::min(overlap, k * chunk), right = std::min(overlap, total - (k + 1) * chunk);
      ng::Block b;
      b.rows = nV; b.width = left + chunk + right; b.left = left; b.right = right;
      for (int v = 0; v < nV; ++v) for (std::int64_t t = 0; t < b.width; ++t) b.re.push_back(static_cast<float>(make(v, static_cast<double>(k * chunk - left + t) / fs)));
      std::vector<std::uint8_t> good(static_cast<std::size_t>(chunk), 1);
      ng::Place at; at.first = k * chunk; at.n = chunk; at.good = &good;
      const ng::Blocks serveBlocks{{"x", b}};
      ng::NodeCall sc = ng::NodeCall::chunk(sprep.made, serveBlocks, 0, at);
      ng::dispatch(s, sc);
      const ng::Blocks gatherBlocks{{"spectrum", sc.made_blocks.at("spectrum")}};
      ng::NodeCall gst = ng::NodeCall::step(gprep.made, std::move(state), gatherBlocks, 0, at);
      ng::dispatch(g, gst);
      state = std::move(gst.made_state);
    }
    ng::NodeCall fin = ng::NodeCall::finish(gprep.made, std::move(state));
    ng::dispatch(g, fin);
    return fin.made_arrays.at("spectra");
  };
  std::srand(7);
  auto noise = [] { return 0.05 * (std::rand() / double(RAND_MAX) - 0.5); };

  // ── the bispectrum of a rising sawtooth (slow rise, fast decay) at 8 Hz ──
  auto frac = [](double v) { return v - std::floor(v); };
  const ng::Array saw = serve([&](int v, double t) { return (1.0 + 0.5 * V[3 * v + 2]) * (2.0 * frac(8.0 * t) - 1.0) + noise(); });
  const Json bp = Json{{"readings", Json::array({"phase_amplitude_coupling", "harmonics"})}, {"normalisation", "threenorm"}, {"space", "coordinates"},
                       {"frequency_normalisation", "sum_of_squared_gains"}, {"support_floor", 1e-3}};
  const ng::Arrays coords = run_array(ng::registry().at("signal.bispectrum"), ng::Arrays{{"spectra", saw}, {"frequency", kernel_set(atoms_at({4.0, 8.0, 16.0}))}}, bp);
  const ng::Array& bc = coords.at("bispectrum");
  const Json triples = bc.meta.at("describe").at("attributes").at("triples");
  CHECK(triples.size() == 3 + 3);                                       // three coupling pairs (4-8, 4-16, 8-16), three harmonics
  CHECK(bc.cols == 5 && bc.rows == O.rows * 6);
  std::int64_t harmonic8 = -1;
  for (std::size_t t = 0; t < triples.size(); ++t) if (triples[t]["reading"] == "harmonics" && triples[t]["first"] == 1) harmonic8 = static_cast<std::int64_t>(t);
  CHECK(harmonic8 >= 0);
  auto biphase = [](const ng::Array& a, std::int64_t row, std::int64_t member, std::int64_t members) {
    const double* x = &a.data[static_cast<std::size_t>((row * members + member) * a.cols)];
    return std::atan2(x[1], x[0]) * 180.0 / M_PI;
  };
  const double constant = biphase(bc, 0, harmonic8, 6);                  // row 0: the constant coordinate
  std::printf("  sawtooth, 8 Hz harmonics, constant coordinate: biphase %.1f degrees\n", constant);
  CHECK(std::abs(constant - 90.0) < 10.0);
  const ng::Arrays onSurface = run_array(ng::registry().at("signal.bispectrum"), ng::Arrays{{"spectra", saw}, {"frequency", kernel_set(atoms_at({4.0, 8.0, 16.0}))}, {"package", H}},
                                         Json{{"readings", Json::array({"harmonics"})}, {"normalisation", "kim_powers"}, {"space", "surface"}, {"frequency_normalisation", "none"}});
  const ng::Array& bs = onSurface.at("bispectrum");
  CHECK(bs.rows == nV * 3 && bs.cols == 4);
  std::vector<double> off;
  for (int v = 0; v < nV; ++v) off.push_back(std::abs(biphase(bs, v, 1, 3) - 90.0));
  std::printf("  sawtooth on the surface: the largest distance of a vertex's biphase from 90 degrees %.1f\n", *std::max_element(off.begin(), off.end()));
  CHECK(*std::max_element(off.begin(), off.end()) < 10.0);
  Json bands = Json::array();
  for (const double c : {1.0, 4.0}) bands.push_back(Json{{"start", c}, {"stop", c}, {"family", "cos2_partition"}, {"chart", "log2"}, {"gain", 1.0}});
  const ng::Arrays banded = run_array(ng::registry().at("signal.bispectrum"),
                                      ng::Arrays{{"spectra", saw}, {"frequency", kernel_set(atoms_at({4.0, 8.0, 16.0}))}, {"package", H}, {"scale", kernel_set(bands)}},
                                      Json{{"readings", Json::array({"harmonics"})}, {"normalisation", "none"}, {"space", "surface"}, {"frequency_normalisation", "none"}});
  CHECK(banded.at("bispectrum").rows == nV * 3 * 2 && banded.at("bispectrum").cols == 2);

  // ── coupling across spatial scales: a 6 Hz rhythm on a coarse pattern whose phase sets the amplitude of a 30 Hz rhythm on a fine pattern ──
  {
    auto fine = [&](int v) { const double z = V[3 * v + 2]; return (35 * z * z * z * z - 30 * z * z + 3) / 8.0; };   // a degree-4 zonal pattern (mu about 4.5)
    const ng::Array planted = serve([&](int v, double t) {
      const double low = std::cos(2 * M_PI * 6 * t);
      return V[3 * v + 2] * low + fine(v) * (1 + 0.8 * low) * std::cos(2 * M_PI * 30 * t) + noise(); });
    const ng::Arrays triples3 = run_array(ng::registry().at("signal.bispectrum"),
                                          ng::Arrays{{"spectra", planted}, {"frequency", kernel_set(atoms_at({6.0, 30.0}))}, {"package", H}, {"scale", kernel_set(bands)}},
                                          Json{{"readings", Json::array({"phase_amplitude_coupling"})}, {"normalisation", "threenorm"}, {"space", "band_triples"},
                                               {"frequency_normalisation", "none"}});
    const ng::Array& bt = triples3.at("bispectrum");
    CHECK(bt.rows == nV * 8 && bt.cols == 5);                                    // one coupling triple x 8 band triples; B and the three cube sums
    std::vector<double> total(8, 0.0);
    for (int v = 0; v < nV; ++v)
      for (int m = 0; m < 8; ++m) {
        const double* x = &bt.data[static_cast<std::size_t>((v * 8 + m) * bt.cols)];
        total[static_cast<std::size_t>(m)] += std::hypot(x[0], x[1]);
      }
    const int strongest = static_cast<int>(std::max_element(total.begin(), total.end()) - total.begin());
    std::printf("  planted coarse-phase / fine-amplitude coupling: the strongest band triple %d (want 3 = coarse, fine, fine)\n", strongest);
    CHECK(strongest == 3);                                                       // (p, q, r) = (0, 1, 1): index (0 x 2 + 1) x 2 + 1
  }

  // ── the flow across frequencies: waves travelling at 8 and 16 Hz ──
  const Json fp = Json{{"frequency_weighting", "kernels"}, {"frequency_normalisation", "sum_of_squared_gains"}, {"support_floor", 1e-3}};
  const ng::Array atoms = kernel_set(atoms_at({8.0, 16.0}));
  auto across = [&](double secondWavenumber) {
    const ng::Array spectra = serve([&](int v, double t) { const double z = V[3 * v + 2]; return std::cos(2 * M_PI * 8 * t - 2.0 * z) + std::cos(2 * M_PI * 16 * t - secondWavenumber * z) + noise(); });
    const ng::Arrays flow = run_array(ng::registry().at("signal.spectral_flow"), ng::Arrays{{"spectra", spectra}, {"package", H}, {"frequency", atoms}}, fp);
    return run_array(ng::registry().at("signal.flow_across_frequencies"),
                     ng::Arrays{{"edges", flow.at("whole_edges")}, {"modes", flow.at("whole_modes")}, {"vertices", flow.at("whole_vertices")}, {"package", H}, {"frequency", atoms}}, Json::object());
  };
  auto slopes = [&](const ng::Array& vertices) { std::vector<double> s; for (int v = 0; v < nV; ++v) { const double x = vertices.data[static_cast<std::size_t>(v * 4 + 1)]; if (std::isfinite(x)) s.push_back(x); } return s; };
  const ng::Arrays sameWavelength = across(2.0), sameSpeed = across(4.0);
  const double cosine = sameWavelength.at("atom_pairs").data[static_cast<std::size_t>((0 * 2 + 1) * 3 + 0)];
  const double slopeWavelength = median(slopes(sameWavelength.at("dispersion"))), slopeSpeed = median(slopes(sameSpeed.at("dispersion")));
  std::printf("  waves at 8 and 16 Hz: edge cosine %.3f; median speed slope, same wavelength %.3f, same speed %.3f\n", cosine, slopeWavelength, slopeSpeed);
  CHECK(cosine > 0.9);
  CHECK(std::abs(slopeWavelength - 1.0) < 0.2 && std::abs(slopeSpeed) < 0.2);

  // ── refusals ──
  CHECK(refused([&] { run_array(ng::registry().at("signal.bispectrum"), ng::Arrays{{"spectra", saw}, {"frequency", kernel_set(atoms_at({8.0}))}}, Json{{"readings", Json::array({"harmonics"})}}); }));
  ng::Types t2; t2["spectra"] = pk; t2["frequency"] = pk;
  CHECK(refused([&] { ng::registry().at("signal.bispectrum").plan(t2, Json{{"readings", Json::array({"harmonics"})}, {"normalisation", "threenorm"}, {"space", "surface"}, {"frequency_normalisation", "none"}}); }));
  CHECK(refused([&] { ng::registry().at("signal.bispectrum").plan(t2, Json{{"readings", Json::array({"skewness"})}, {"normalisation", "threenorm"}, {"space", "coordinates"}, {"frequency_normalisation", "none"}}); }));
  return nxrtest::finish("test_sensor_bispectrum");
}
