// The flow through the fibre network (pipelines/src/connectome_flow_steps.cpp) through the graph's dispatch point, on hand-made arrays (never a dataset): four tiles on two pieces,
// one vertex each; the kernel the identity on twelve channels (three components a tile); the spectra served through a random complete serving operator; tile 1 follows tile 0
// by a planted lag. Checked: the pairs (counts, medians, across), the network as a graph (its edges, its metric: tile areas and conductance) and its Hodge-Dirac modes (meeg.graph_hodge_dirac:
// the constants out, orthonormal under the stars), the tile operator (the served coordinates back to the tiles), the
// whole run's tile cross-spectrum (the planted lag back, the reference's sign), the segments and the reference's sign-flip test and control on them (the lagged pair consistent),
// the network's divergence (out of tile 0, into tile 1) and the flow's coefficients on the graph's STORED modes (a forest: all gradient); refusals with their reasons.
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "check.h"
#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"
#include "nxr/meeg/tile_network.h"
#include "nxr/pipelines/connectome_flow_steps.h"
#include "nxr/pipelines/sensor_flow_steps.h"
#include "nxr/signal/fourier.h"

namespace ng = nxr::graph;
using ng::Json;

namespace {
template <class F> bool refused(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
ng::Arrays run_array(const char* name, const ng::Arrays& c, const Json& p) {
  const ng::NodeSpec& node = ng::registry().at(name);
  ng::Shared shared;
  const ng::Types types;
  const ng::Arrays none;
  ng::NodeCall prep = ng::NodeCall::prepare(types, c, p, shared);
  ng::dispatch(node, prep);
  ng::NodeCall go = ng::NodeCall::run(prep.made, c);   // as the executor does: run is handed the arrays prepare was
  ng::dispatch(node, go);
  return go.made_arrays;
}
ng::Array column(const std::vector<double>& v) { ng::Array a; a.rows = static_cast<std::int64_t>(v.size()); a.cols = 1; a.data = v; return a; }
double at(const ng::Array& a, std::int64_t r, std::int64_t c) { return a.at(static_cast<std::size_t>(r * a.cols + c)); }
}  // namespace

int main() {
  nxr::pipelines::graph::register_sensor_flow_nodes();
  nxr::pipelines::graph::register_connectome_flow_nodes();
  for (const char* name : {"stats.network_sign_flip", "stats.run_replication", "meeg.fiber_lengths", "meeg.tile_pairs", "meeg.graph_hodge_dirac", "signal.tile_operator", "signal.tile_flow", "signal.tile_time_cells", "meeg.network_flow"})
    CHECK(ng::registry().at(name).traits.declared());

  // ── the surface (four vertices, two triangles), its grid (two pieces, depth 1: codes 2 piece + tile), the fibres ──
  ng::Array surface; surface.rows = 4; surface.cols = 3; surface.data = {0, 0, 0, 0.03, 0.01, 0, 0.1, 0, 0, 0.13, 0.01, 0};
  surface.meta["faces"] = std::vector<std::int32_t>{0, 1, 2, 1, 3, 2}; surface.meta["unit"] = "m";
  ng::Array grid = column({0, 1, 2, 3}); grid.meta["depth"] = 1; grid.meta["slots"] = 2;
  std::vector<double> first, second, lengths;
  auto join = [&](int a, int b, int n, double length) { for (int i = 0; i < n; ++i) { first.push_back(a); second.push_back(b); lengths.push_back(length + 0.001 * i); } };
  join(0, 1, 12, 0.05); join(2, 3, 11, 0.04); join(0, 2, 3, 0.12);
  std::vector<double> ends = first; ends.insert(ends.end(), second.begin(), second.end());
  // a fibre's points: two points 1 apart then a third 2 further: length 3
  ng::Array fibres; fibres.rows = 6; fibres.cols = 3; fibres.data = {0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 3}; fibres.meta["points_per_fiber"] = 3;
  const auto len = run_array("meeg.fiber_lengths", {{"fibers", fibres}}, Json{{"points_per_fiber", 0}});
  CHECK(len.at("lengths").rows == 2 && std::abs(len.at("lengths").data[0] - 3) < 1e-12 && std::abs(len.at("lengths").data[1] - 7) < 1e-12);

  const Json tp = Json{{"scale", "parts 2"}, {"min_streamlines", 10}};
  const auto net = run_array("meeg.tile_pairs", {{"end_vertex", column(ends)}, {"lengths", column(lengths)}, {"tiles", grid}, {"surface", surface}}, tp);
  const ng::Array& pairs = net.at("pairs");
  CHECK(pairs.rows == 6 && pairs.cols == 6);
  CHECK(at(pairs, nxr::meeg::pairIndex(0, 1, 4), 2) == 12 && at(pairs, nxr::meeg::pairIndex(2, 3, 4), 2) == 11 && at(pairs, nxr::meeg::pairIndex(0, 2, 4), 2) == 3);
  CHECK(std::abs(at(pairs, nxr::meeg::pairIndex(0, 1, 4), 3) - 0.0555) < 1e-9);      // numpy median of 0.050 .. 0.061
  CHECK(at(pairs, nxr::meeg::pairIndex(1, 2, 4), 5) == 1 && at(pairs, nxr::meeg::pairIndex(0, 1, 4), 5) == 0);
  CHECK(refused([&] { run_array("meeg.tile_pairs", {{"end_vertex", column(ends)}, {"tiles", grid}, {"surface", surface}}, Json{{"scale", "parts 3"}, {"min_streamlines", 10}}); }));   // not a power of two
  CHECK(refused([&] { run_array("meeg.tile_pairs", {{"end_vertex", column(ends)}, {"tiles", grid}, {"surface", surface}}, Json{{"scale", "parts 4"}, {"min_streamlines", 10}}); }));   // finer than the grid
  CHECK(refused([&] { run_array("meeg.tile_pairs", {{"end_vertex", column(ends)}, {"tiles", grid}, {"surface", surface}}, Json{{"scale", 2}, {"min_streamlines", 10}}); }));          // a bare number: not a quantity

  // the network as a GRAPH: the edges 0-1 and 2-3 (0-2 has 3 < 10 streamlines), two components; the metric: each tile's barycentric area, conductance = count / median length
  const ng::Array& g = net.at("graph");
  CHECK(g.rows == 4 && g.cols == 3 && (g.meta["describe"]["edges"] == Json::array({0, 1, 2, 3})) && g.meta["describe"]["components"] == 2);
  const ng::Array& conductance = net.at("conductance");
  CHECK(conductance.rows == 2 && std::abs(conductance.data[0] - 12 / 0.0555) < 1e-6);
  double areas = 0;
  for (const double a : net.at("area").data) areas += a;
  CHECK(std::abs(areas - 0.5 * 0.1 * 0.01 - 0.5 * 0.1 * 0.01) < 1e-12);                     // the two triangles' areas, shared out to the tiles
  ng::Array stored = g;                                                                         // as the runner hands a stored graph back
  stored.meta = Json{{"type", "graph"}, {"name", "net"}, {"edges", g.meta["describe"]["edges"]}, {"components", 2}, {"producer", g.meta["describe"]["producer"]}};
  const auto modes = run_array("meeg.graph_hodge_dirac", {{"graph", stored}, {"area", net.at("area")}, {"conductance", conductance}}, Json{{"null_tolerance", 1e-10}});
  const ng::Array& Phi = modes.at("eigenvectors");
  CHECK(Phi.rows == 6 && Phi.cols == 2 && modes.at("modes").rows == 2);                       // V - components = 2 exact modes, no cycle (a forest)
  {
    Eigen::VectorXd stars(6);
    for (int i = 0; i < 4; ++i) stars(i) = net.at("area").data[static_cast<std::size_t>(i)];
    for (int e = 0; e < 2; ++e) stars(4 + e) = conductance.data[static_cast<std::size_t>(e)];
    Eigen::MatrixXd m(6, 2);
    for (int i = 0; i < 12; ++i) m(i / 2, i % 2) = Phi.data[static_cast<std::size_t>(i)];
    CHECK((m.transpose() * stars.asDiagonal() * m - Eigen::MatrixXd::Identity(2, 2)).norm() < 1e-9);   // orthonormal under the stars
  }
  ng::Array zeroArea = net.at("area"); zeroArea.data[0] = 0;
  CHECK(refused([&] { run_array("meeg.graph_hodge_dirac", {{"graph", stored}, {"area", zeroArea}, {"conductance", conductance}}, Json{{"null_tolerance", 1e-10}}); }));   // a weight that is not positive

  // ── the recording: twelve channels (tile t component c = channel 3 t + c), tile 1's x follows tile 0's x by tau; served through a random complete operator ──
  const int C = 12;
  const double fs = 200, tau = 0.012;
  const std::int64_t N = 4096, overlap = 1024, own = 2048;
  const double f = fs * 205.0 / static_cast<double>(N);
  std::mt19937 rng(5);
  std::normal_distribution<double> normal(0, 1);
  Eigen::MatrixXd x(C, N);
  for (std::int64_t t = 0; t < N; ++t) {
    const double s = static_cast<double>(t) / fs;
    for (int r = 0; r < C; ++r) x(r, t) = 0.05 * normal(rng);
    x(0, t) += std::cos(2 * M_PI * f * s); x(3, t) += std::cos(2 * M_PI * f * (s - tau));
  }
  Eigen::MatrixXd serving(C, C);
  for (int i = 0; i < C * C; ++i) serving(i / C, i % C) = normal(rng);
  const Eigen::MatrixXcd Y = serving.cast<std::complex<double>>() * nxr::signal::rfft(x, 1);   // [K x bins]
  const std::int64_t B = Y.cols();
  ng::Array spectra;
  spectra.rows = 1; spectra.cols = B * C * 2;
  for (std::int64_t k = 0; k < B; ++k) for (int r = 0; r < C; ++r) { spectra.single.push_back(static_cast<float>(Y(r, k).real())); spectra.single.push_back(static_cast<float>(Y(r, k).imag())); }
  Json modeList = Json::array();
  for (int r = 0; r < C; ++r) modeList.push_back(r);
  spectra.meta["shape"] = Json::array({1, B, C, 2});
  spectra.meta["producer"] = Json{{"sfreq", fs}, {"transform_samples", N}, {"first_bin", 0}, {"modes", modeList}, {"overlap_samples", overlap}, {"chunks", Json::array({Json{{"samples", own}}})}};
  ng::Array S; S.rows = C; S.cols = C;
  for (int r = 0; r < C; ++r) for (int c = 0; c < C; ++c) S.data.push_back(serving(r, c));
  std::vector<std::int32_t> channels;
  for (int c = 0; c < C; ++c) channels.push_back(c + 5);                                    // the recording's channel numbers
  S.meta["channels"] = channels; S.meta["modes"] = modeList;
  ng::Array K; K.rows = C; K.cols = C;
  for (int r = 0; r < C; ++r) for (int c = 0; c < C; ++c) K.data.push_back(r == c ? 1.0 : 0.0);
  std::vector<std::int32_t> kernelChannels(channels.rbegin(), channels.rend());              // the kernel's columns in another order: matched by channel
  ng::Array Kr = K;
  for (int r = 0; r < C; ++r) for (int c = 0; c < C; ++c) Kr.data[static_cast<std::size_t>(r * C + c)] = K.data[static_cast<std::size_t>(r * C + (C - 1 - c))];
  Kr.meta["channels"] = kernelChannels; Kr.meta["name"] = "identity";
  const auto op = run_array("signal.tile_operator", {{"kernel", Kr}, {"tiles", grid}, {"serving", S}}, tp);
  const ng::Array& M = op.at("operator");
  CHECK(M.rows == 12 && M.cols == 12 && M.meta["synthesis_rank"] == 12);
  {
    Eigen::MatrixXd m(12, 12);
    for (int i = 0; i < 144; ++i) m(i / 12, i % 12) = M.data[static_cast<std::size_t>(i)];
    CHECK((m * serving - Eigen::MatrixXd::Identity(12, 12)).norm() < 1e-8);             // the served coordinates back to the tiles' currents
  }
  ng::Array wrongKernel = Kr; wrongKernel.meta["channels"] = std::vector<std::int32_t>(12, 99);
  CHECK(refused([&] { run_array("signal.tile_operator", {{"kernel", wrongKernel}, {"tiles", grid}, {"serving", S}}, tp); }));

  ng::Array atoms;
  const double fwhm = 2.0 * std::sqrt(2.0 * std::log(2.0)) / 12.0;                           // standard deviation centre / 12
  atoms.meta["params"] = Json{{"kernels", Json{{"members", Json::array({Json{{"start", f}, {"stop", f}, {"family", "gaussian"}, {"chart", "linear"}, {"gain", 2.0}, {"params", Json{{"fwhm_per_centre", fwhm}}}}})}}}};
  const Json fp = Json{{"frequency_normalisation", "none"}};
  const auto flow = run_array("signal.tile_flow", {{"spectra", spectra}, {"operator", M}, {"frequency", atoms}}, fp);
  // measurements over the atoms: [pairs x atoms x (Re, Im)] and [tiles x atoms x power], value (row atoms + atom) measures + measure
  const ng::Array& F = flow.at("pair_flow");
  const ng::Array& Fp = flow.at("tile_power");
  CHECK(F.rows == 6 && F.cols == 2 && Fp.rows == 4 && Fp.cols == 1);
  const std::int64_t q01 = nxr::meeg::pairIndex(0, 1, 4);
  const double lag = -std::atan2(at(F, q01, 1), at(F, q01, 0)) / (2 * M_PI * f);
  std::printf("  planted lag %.4f s, read %.4f s\n", tau, lag);
  CHECK(std::abs(lag + tau) < 1e-3);
  CHECK(at(F, q01, 1) / std::sqrt(at(Fp, 0, 0) * at(Fp, 1, 0)) > 0.3);                     // the edge flow: tile 1 lags tile 0
  ng::Array wrongSpectra = spectra; wrongSpectra.meta["producer"]["modes"] = Json::array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99});
  CHECK(refused([&] { run_array("signal.tile_flow", {{"spectra", wrongSpectra}, {"operator", M}, {"frequency", atoms}}, fp); }));

  const auto cells = run_array("signal.tile_time_cells", {{"spectra", spectra}, {"operator", M}, {"frequency", atoms}}, Json{{"frequency_normalisation", "none"}, {"cells", "segments"}, {"segment_seconds", 1.0}});
  const ng::Array& Z = cells.at("cells");
  const Json groups = Z.meta["producer"]["groups"];
  CHECK(groups.size() == 1 && Z.cols == 2 * 6 + 4 && Z.rows == 10 && groups[0]["cell_samples"] == 200);   // 2048 own samples: ten whole segments of 200 (the eleventh cut off, as S = N // L)
  bool positive = true;
  for (std::int64_t r = 0; r < Z.rows; ++r) positive = positive && at(Z, r, 6 + q01) > 0;
  CHECK(positive);
  const auto width = run_array("signal.tile_time_cells", {{"spectra", spectra}, {"operator", M}, {"frequency", atoms}}, Json{{"frequency_normalisation", "none"}, {"cells", "atom_width"}, {"cell_width", 2.0}});
  CHECK(width.at("cells").rows == 2048 / width.at("cells").meta["producer"]["groups"][0]["cell_samples"].get<std::int64_t>());
  const Json flipParameters = Json{{"patterns", 500}, {"seed", 0}, {"min_streamlines", 10}, {"distance_bin_edges", Json::array({0, 0.03, 0.045, 0.06, 0.08, 0.1, 0.13, 0.17, 0.25})}};
  const auto flip = run_array("stats.network_sign_flip", {{"cells", Z}, {"pairs", pairs}}, flipParameters);
  const ng::Array& R = flip.at("consistency");
  CHECK(R.rows == 6 && R.cols == 5);                                                          // [pairs x 1 atom x (mean flow, p, family-wise p, control, phase)]
  CHECK(at(R, q01, 0) > 0.3 && at(R, q01, 1) < 0.01 && at(R, q01, 2) < 0.05);               // the lagged pair: a positive mean flow, p and family-wise p small
  CHECK(std::abs(-at(R, q01, 4) / (2 * M_PI * f) + tau) < 1e-3);                            // the phase of the cells' summed cross-spectrum gives the lag back
  // the control: the joined pairs (0-1 at 0.03, 2-3 at 0.03) occupy one distance bin; the not-joined pairs in it: none here (0-2 0.1, 0-3 0.13, 1-2 0.07, 1-3 0.1)
  double controls = 0;
  for (int q = 0; q < 6; ++q) controls += at(R, q, 3);
  CHECK(controls == 0);
  const auto again = run_array("stats.network_sign_flip", {{"cells", Z}, {"pairs", pairs}}, flipParameters);
  CHECK(again.at("consistency").data == R.data);                                            // one seed, one result
  const auto replication = run_array("stats.run_replication", {{"first", R}, {"second", R}, {"pairs", pairs}}, Json{{"min_streamlines", 1}});
  CHECK(std::abs(at(replication.at("replication"), 0, 0) - 1) < 1e-12);

  const ng::Arrays graphIn = {{"pairs", pairs}, {"network", stored}, {"conductance", conductance}, {"modes", Phi}, {"eigenvalues", modes.at("eigenvalues")}, {"flow", R}};
  const auto network = run_array("meeg.network_flow", graphIn, Json::object());
  const ng::Array& W = network.at("network_tiles");
  const ng::Array& Wt = network.at("network_totals");
  CHECK(W.rows == 4 && W.cols == 2 && network.at("network_modes").rows == 2 && Wt.rows == 1 && Wt.cols == 3);
  CHECK(at(W, 0, 0) < 0 && at(W, 1, 0) > 0 && std::abs(at(W, 0, 0) + at(W, 1, 0)) < 1e-12);  // the divergence: out of tile 0, into tile 1
  CHECK(std::abs(at(Wt, 0, 0) / at(Wt, 0, 2) - 1) < 1e-9 && std::abs(at(Wt, 0, 1)) < 1e-12);  // a forest: every flow is a gradient, no cycle part
  ng::Array fewer = Phi; fewer.rows = 5; fewer.data.resize(10);
  ng::Arrays otherGraph = graphIn; otherGraph["modes"] = fewer;
  CHECK(refused([&] { run_array("meeg.network_flow", otherGraph, Json::object()); }));         // modes solved on another graph
  ng::Arrays notMean = graphIn; notMean["flow"] = F;
  CHECK(refused([&] { run_array("meeg.network_flow", notMean, Json::object()); }));           // a flow whose first measure is not the mean edge flow
  return nxrtest::finish("test_connectome_flow");
}
