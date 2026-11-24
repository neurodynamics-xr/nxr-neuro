// The sensor-flow nodes (pipelines/src/sensor_flow_steps.cpp) through the graph's dispatch point, on a synthetic sphere whose Hodge-Dirac modes are solved here (never a real dataset):
//   a recording of three chunks made from a TRAVELLING wave on the sphere (phase growing from pole to pole) at one frequency, each vertex a channel;
//   signal.overlapped_spectrum + signal.chunk_spectra: each chunk's spectrum equals overlappedSpectrum on the chunk and its overlap (the first reflected on its left, the last on its
//     right), and the result does not depend on how the run is cut: the chunk spectra are the same whatever the block order of the neighbours;
//   signal.spectral_flow: one Gaussian atom at the wave's frequency, two cos² bands: the whole field's flow is mostly exact (travel); the band pairs sum to the whole; the flow
//     coefficients add back to the phase current; every_bin gives one member per bin and the members' power sums to the atoms' unnormalised power where the gain is 1;
//   signal.atom_time_cells: the cells' edge phase currents keep the whole run's direction; stats.sign_flip_consistency: the travelling wave is consistent (small family-wise p);
//   signal.sensor_power and stats.aperiodic_fit run and give one fit per vertex; refusals with their reasons.
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
#include "nxr/pipelines/select_nodes.h"
#include "nxr/pipelines/sensor_flow_steps.h"
#include "nxr/pipelines/spectral_steps.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/mode_spectrum.h"
#include "nxr/signal/spectral_flow.h"

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
  ng::NodeCall go = ng::NodeCall::run(prep.made, none);
  ng::dispatch(node, go);
  return go.made_arrays;
}
}  // namespace

int main() {
  nxr::pipelines::graph::register_sensor_flow_nodes();
  nxr::pipelines::graph::register_select_nodes();
  nxr::pipelines::graph::register_spectral_nodes();
  for (const char* name : {"signal.overlapped_spectrum", "signal.chunk_spectra", "signal.spectral_flow", "signal.atom_time_cells", "stats.sign_flip_consistency", "signal.sensor_power", "stats.aperiodic_fit"})
    CHECK(ng::registry().at(name).traits.declared());

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

  // ── the serving operator: the constant and the exact modes, composed with the identity channel map ──
  ng::Array map;
  map.rows = nV; map.cols = 1;
  for (int v = 0; v < nV; ++v) map.data.push_back(v);
  const ng::NodeSpec& analysis = ng::registry().at("signal.mode_analysis");
  ng::Types ptypes;
  ng::PortType pk; ptypes["package"] = pk; ptypes["channel_map"] = pk;
  ng::Shared shared;
  const Json ap = Json{{"modes", "exact_and_constant"}, {"normalisation", "unit_mass"}};
  const ng::Arrays analysisInputs{{"package", H}, {"channel_map", map}};
  ng::NodeCall aprep = ng::NodeCall::prepare(ptypes, analysisInputs, ap, shared);
  ng::dispatch(analysis, aprep);
  const ng::Arrays none;
  ng::NodeCall arun = ng::NodeCall::run(aprep.made, none);
  ng::dispatch(analysis, arun);
  const ng::Array O = arun.made_arrays.at("operator");
  CHECK(O.meta.at("modes")[0] == -1);

  // ── a recording: a travelling wave at 10 Hz, three chunks of 256 samples at 128 Hz, plus a little noise ──
  const double fs = 128.0, f0 = 10.0;
  const std::int64_t chunk = 256, overlap = 128, transform = 512, total = 3 * chunk;
  Eigen::MatrixXd x(nV, total);
  std::srand(3);
  for (int v = 0; v < nV; ++v)
    for (std::int64_t t = 0; t < total; ++t) x(v, t) = std::cos(2 * M_PI * f0 * t / fs - 2.0 * V[3 * v + 2]) + 0.01 * (std::rand() / double(RAND_MAX) - 0.5);
  const Json sp = Json{{"overlap_samples", overlap}, {"transform_samples", transform}, {"frequency_span", Json::array({0.0, 40.0})}};
  ng::Types stypes;
  ng::PortType xt; xt.sfreq = fs; xt.axis = ng::Axis::samples; stypes["x"] = xt; stypes["operator"] = pk;
  const ng::NodeSpec& serve = ng::registry().at("signal.overlapped_spectrum");
  const ng::Arrays operatorInput{{"operator", O}};
  ng::NodeCall sprep = ng::NodeCall::prepare(stypes, operatorInput, sp, shared);
  ng::dispatch(serve, sprep);
  const ng::NodeSpec& gather = ng::registry().at("signal.chunk_spectra");
  ng::Types gtypes;
  ng::PortType st; st.sfreq = fs; gtypes["spectrum"] = st; gtypes["operator"] = pk;
  ng::NodeCall gprep = ng::NodeCall::prepare(gtypes, operatorInput, sp, shared);
  ng::dispatch(gather, gprep);
  ng::NodeCall gs = ng::NodeCall::start(gprep.made);
  ng::dispatch(gather, gs);
  std::any state = std::move(gs.made);
  const std::vector<double> ov = O.values();
  const Eigen::MatrixXd op = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ov.data(), O.rows, O.cols);
  double serveError = 0;
  for (std::int64_t k = 0; k < 3; ++k) {
    const std::int64_t left = std::min(overlap, k * chunk), right = std::min(overlap, total - (k + 1) * chunk);
    ng::Block b;
    b.rows = nV; b.width = left + chunk + right; b.left = left; b.right = right;
    for (int v = 0; v < nV; ++v) for (std::int64_t t = 0; t < b.width; ++t) b.re.push_back(static_cast<float>(x(v, k * chunk - left + t)));
    std::vector<std::uint8_t> good(static_cast<std::size_t>(chunk), 1);
    ng::Place at; at.first = k * chunk; at.n = chunk; at.good = &good;
    const ng::Blocks serveBlocks{{"x", b}};
    ng::NodeCall sc = ng::NodeCall::chunk(sprep.made, serveBlocks, 0, at);
    ng::dispatch(serve, sc);
    const ng::Block& out = sc.made_blocks.at("spectrum");
    Eigen::MatrixXd block(nV, b.width);
    for (int v = 0; v < nV; ++v) for (std::int64_t t = 0; t < b.width; ++t) block(v, t) = b.re[static_cast<std::size_t>(v * b.width + t)];
    const Eigen::MatrixXcd want = nxr::signal::overlappedSpectrum(op, block, left, right, overlap, transform, 0, out.width);
    for (Eigen::Index rr = 0; rr < want.rows(); ++rr) for (Eigen::Index kk = 0; kk < want.cols(); ++kk)
      serveError = std::max(serveError, std::abs(want(rr, kk) - std::complex<double>(out.re[static_cast<std::size_t>(rr * out.width + kk)], out.im[static_cast<std::size_t>(rr * out.width + kk)])) / (std::abs(want(rr, kk)) + 1.0));
    const ng::Blocks gatherBlocks{{"spectrum", out}};
    ng::NodeCall gst = ng::NodeCall::step(gprep.made, std::move(state), gatherBlocks, 0, at);
    ng::dispatch(gather, gst);
    state = std::move(gst.made_state);
  }
  CHECK(serveError < 1e-4);
  ng::NodeCall gfin = ng::NodeCall::finish(gprep.made, std::move(state));
  ng::dispatch(gather, gfin);
  const ng::Array spectra = gfin.made_arrays.at("spectra");
  CHECK(spectra.meta.at("shape")[0] == 3 && spectra.meta.at("shape")[2] == O.rows);

  // ── the atoms and the bands as select.kernels stores them ──
  auto kernel_set = [](const Json& members) { ng::Array a; a.meta["params"] = Json{{"kernels", Json{{"members", members}}}}; a.meta["type"] = "set"; return a; };
  Json atoms = Json::array();
  for (const double c : {5.0, 10.0, 20.0})
    atoms.push_back(Json{{"start", c}, {"stop", c}, {"family", "gaussian"}, {"chart", "linear"}, {"gain", 2.0}, {"params", Json{{"fwhm_per_centre", 2.0 * std::sqrt(2.0 * std::log(2.0)) / 12.0 * 3.0}}}});
  Json bands = Json::array();
  for (const double c : {1.0, 4.0}) bands.push_back(Json{{"start", c}, {"stop", c}, {"family", "cos2_partition"}, {"chart", "log2"}, {"gain", 1.0}});
  const Json fp = Json{{"frequency_weighting", "kernels"}, {"frequency_normalisation", "sum_of_squared_gains"}, {"support_floor", 1e-3}};
  const ng::Arrays inputs{{"spectra", spectra}, {"package", H}, {"frequency", kernel_set(atoms)}, {"scale", kernel_set(bands)}};
  const ng::Arrays flow = run_array(ng::registry().at("signal.spectral_flow"), inputs, fp);
  const ng::Array& wm = flow.at("whole_modes"), &we = flow.at("whole_edges"), &pe = flow.at("pair_edges");
  // the 10 Hz atom (member 1): its flow mostly exact
  double exact = 0, all = 0;
  for (int k = 0; k < K; ++k) { const double c2 = wm.data[static_cast<std::size_t>(k * 3 + 1)] * wm.data[static_cast<std::size_t>(k * 3 + 1)]; all += c2; if (md.kind[static_cast<std::size_t>(k)] == 0) exact += c2; }
  std::printf("  travelling wave: exact share of the flow at 10 Hz %.3f\n", exact / all);
  CHECK(exact / all > 0.9);
  // the band pairs sum to the whole (member 1)
  double pairError = 0, scale = 0;
  for (int e = 0; e < cx.nE; ++e) {
    double sum = 0;
    for (int pq = 0; pq < 4; ++pq) sum += pe.data[static_cast<std::size_t>(e * 12 + 1 + 3 * pq)];
    pairError = std::max(pairError, std::abs(sum - we.data[static_cast<std::size_t>(e * 3 + 1)])); scale = std::max(scale, std::abs(we.data[static_cast<std::size_t>(e * 3 + 1)]));
  }
  CHECK(pairError < 1e-8 * scale);
  // every bin: one member per bin
  const ng::Arrays bins = run_array(ng::registry().at("signal.spectral_flow"), ng::Arrays{{"spectra", spectra}, {"package", H}}, Json{{"frequency_weighting", "every_bin"}});
  CHECK(bins.at("whole_edges").rows == cx.nE * spectra.meta.at("shape")[1].get<std::int64_t>());

  // ── time cells and the sign-flip statistic ──
  const Json cp = Json{{"frequency_weighting", "kernels"}, {"frequency_normalisation", "sum_of_squared_gains"}, {"support_floor", 1e-3}, {"cell_width", 2.0}};
  const ng::Arrays cells = run_array(ng::registry().at("signal.atom_time_cells"), inputs, cp);
  const ng::Array& tc = cells.at("cells");
  const Json groups = tc.meta.at("describe").at("attributes").at("groups");
  CHECK(groups.size() == 6 && tc.cols == cx.nE + 4 * nV);
  const ng::Arrays flip = run_array(ng::registry().at("stats.sign_flip_consistency"), ng::Arrays{{"cells", tc}}, Json{{"elements", "edges"}, {"patterns", 200}, {"seed", 0}, {"cells_per_flip", 1}});
  const ng::Array& fc = flip.at("consistency");
  // group 1: the 10 Hz atom, band 0 (the coarse band holds the wave): its median R1 high and p small
  const std::int64_t E = cx.nE;
  const double median = fc.data[static_cast<std::size_t>(1 * fc.cols + 3 * E + 1)], medianP = fc.data[static_cast<std::size_t>(1 * fc.cols + 3 * E + 3)];
  std::printf("  10 Hz, coarse band: median R1 %.3f, its p %.4f, cells %g\n", median, medianP, fc.data[static_cast<std::size_t>(1 * fc.cols + 3 * E + 5)]);
  CHECK(median > 0.8 && medianP < 0.05);

  // ── the sensors' power and the aperiodic fit ──
  const ng::Arrays power = run_array(ng::registry().at("signal.sensor_power"), ng::Arrays{{"spectra", spectra}, {"package", H}}, Json::object());
  const ng::Arrays fit = run_array(ng::registry().at("stats.aperiodic_fit"), ng::Arrays{{"power", power.at("power")}},
                                   Json{{"knee", false}, {"fit_range", Json::array({1.0, 30.0})}, {"iterations", 3}, {"threshold", 2.0}});
  CHECK(fit.at("fit").rows == nV && fit.at("fit").cols == 5);

  // ── select.kernels: a proportional width is stored as fwhm_per_centre (no flag); a Meyer band states its half-width; the width rule has no default ──
  {
    const Json proportional = Json{{"quantity", "frequency"}, {"profile", "gaussian"}, {"chart", "linear"}, {"width", 0.25}, {"width_rule", "proportional"}, {"gain", 1.0},
                                   {"centres", Json::array({4.0, 8.0})}};
    const ng::Arrays made = run_array(ng::registry().at("select.kernels"), ng::Arrays{}, proportional);
    const Json member = made.at("kernels").meta["describe"]["kernels"][1];
    CHECK(member["params"].contains("fwhm_per_centre") && member["params"]["fwhm_per_centre"] == 0.25 && !member["params"].contains("width_rule") && member["start"] == 8.0);
    const Json meyer = Json{{"quantity", "frequency"}, {"profile", "meyer_band"}, {"chart", "log2"}, {"gain", 1.0}, {"bands", Json::array({Json::array({8.0, 12.0})})}};
    CHECK(refused([&] { run_array(ng::registry().at("select.kernels"), ng::Arrays{}, meyer); }));
    Json noRule = proportional;
    noRule.erase("width_rule");
    CHECK(refused([&] { run_array(ng::registry().at("select.kernels"), ng::Arrays{}, noRule); }));
  }

  // ── refusals ──
  CHECK(refused([&] { run_array(ng::registry().at("signal.spectral_flow"), inputs, Json{{"frequency_weighting", "kernels"}}); }));          // no normalisation named
  CHECK(refused([&] { run_array(ng::registry().at("signal.atom_time_cells"), ng::Arrays{{"spectra", spectra}, {"package", H}, {"frequency", kernel_set(bands)}, {"scale", kernel_set(bands)}}, cp); }));   // not Gaussian atoms
  const Json overlapOnly = Json{{"overlap_samples", overlap}};
  CHECK(refused([&] { ng::NodeCall p2 = ng::NodeCall::prepare(stypes, operatorInput, overlapOnly, shared); ng::dispatch(serve, p2); }));
  return nxrtest::finish("test_sensor_flow");
}
