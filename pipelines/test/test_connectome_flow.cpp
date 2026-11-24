// The flow through the fibre network (pipelines/src/connectome_flow_steps.cpp) through the graph's dispatch point, on hand-made arrays (never a dataset): four tiles on two pieces,
// one vertex each; the kernel the identity on twelve channels (three components a tile); the spectra served through a random complete serving operator; tile 1 follows tile 0
// by a planted lag. Checked: the pairs (counts, medians, across), the connectome eigenmodes (null modes), the tile operator (the served coordinates back to the tiles), the
// whole run's tile cross-spectrum (the planted lag back, the reference's sign), the time cells and the sensor chain's sign-flip test on them (the lagged pair consistent),
// the network's divergence (out of tile 0, into tile 1); refusals with their reasons.
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
  ng::NodeCall go = ng::NodeCall::run(prep.made, none);
  ng::dispatch(node, go);
  return go.made_arrays;
}
ng::Array column(const std::vector<double>& v) { ng::Array a; a.rows = static_cast<std::int64_t>(v.size()); a.cols = 1; a.data = v; return a; }
double at(const ng::Array& a, std::int64_t r, std::int64_t c) { return a.at(static_cast<std::size_t>(r * a.cols + c)); }
}  // namespace

int main() {
  nxr::pipelines::graph::register_sensor_flow_nodes();
  nxr::pipelines::graph::register_connectome_flow_nodes();
  for (const char* name : {"meeg.fiber_lengths", "connectome.tile_pairs", "connectome.laplacian_eigenmodes", "connectome.tile_operator", "connectome.tile_flow", "connectome.tile_time_cells", "connectome.network_flow"})
    CHECK(ng::registry().at(name).traits.declared());

  // ── the surface (four vertices), its grid (two pieces, depth 1: codes 2 piece + tile), the fibres ──
  ng::Array surface; surface.rows = 4; surface.cols = 3; surface.data = {0, 0, 0, 0.03, 0, 0, 0.1, 0, 0, 0.13, 0, 0};
  ng::Array grid = column({0, 1, 2, 3}); grid.meta["depth"] = 1; grid.meta["slots"] = 2;
  std::vector<double> first, second, lengths;
  auto join = [&](int a, int b, int n, double length) { for (int i = 0; i < n; ++i) { first.push_back(a); second.push_back(b); lengths.push_back(length + 0.001 * i); } };
  join(0, 1, 12, 0.05); join(2, 3, 11, 0.04); join(0, 2, 3, 0.12);
  std::vector<double> ends = first; ends.insert(ends.end(), second.begin(), second.end());
  // a fibre's points: two points 1 apart then a third 2 further: length 3
  ng::Array fibres; fibres.rows = 6; fibres.cols = 3; fibres.data = {0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 3}; fibres.meta["points_per_fiber"] = 3;
  const auto len = run_array("meeg.fiber_lengths", {{"fibers", fibres}}, Json{{"points_per_fiber", 0}});
  CHECK(len.at("lengths").rows == 2 && std::abs(len.at("lengths").data[0] - 3) < 1e-12 && std::abs(len.at("lengths").data[1] - 7) < 1e-12);

  const Json tp = Json{{"tiles_per_piece", 2}};
  const auto net = run_array("connectome.tile_pairs", {{"end_vertex", column(ends)}, {"lengths", column(lengths)}, {"tiles", grid}, {"surface", surface}}, tp);
  const ng::Array& pairs = net.at("pairs");
  CHECK(pairs.rows == 6 && pairs.cols == 6);
  CHECK(at(pairs, nxr::meeg::pairIndex(0, 1, 4), 2) == 12 && at(pairs, nxr::meeg::pairIndex(2, 3, 4), 2) == 11 && at(pairs, nxr::meeg::pairIndex(0, 2, 4), 2) == 3);
  CHECK(std::abs(at(pairs, nxr::meeg::pairIndex(0, 1, 4), 3) - 0.0555) < 1e-9);      // numpy median of 0.050 .. 0.061
  CHECK(at(pairs, nxr::meeg::pairIndex(1, 2, 4), 5) == 1 && at(pairs, nxr::meeg::pairIndex(0, 1, 4), 5) == 0);
  CHECK(refused([&] { run_array("connectome.tile_pairs", {{"end_vertex", column(ends)}, {"tiles", grid}, {"surface", surface}}, Json{{"tiles_per_piece", 3}}); }));   // not a power of two
  CHECK(refused([&] { run_array("connectome.tile_pairs", {{"end_vertex", column(ends)}, {"tiles", grid}, {"surface", surface}}, Json{{"tiles_per_piece", 4}}); }));   // finer than the grid

  const Json eig = Json{{"min_streamlines", 10}, {"null_tolerance", 1e-10}};
  const auto modes = run_array("connectome.laplacian_eigenmodes", {{"pairs", pairs}}, eig);
  CHECK(modes.at("eigenmodes").rows == 4 && modes.at("eigenmodes").meta["producer"]["null_modes"] == 2);   // the edges 0-1 and 2-3: two components (0-2 has 3 < 10)
  CHECK(std::abs(at(modes.at("eigenmodes"), 3, 0) - 24) < 1e-9);                                            // the largest: 2 x 12

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
  const auto op = run_array("connectome.tile_operator", {{"kernel", Kr}, {"tiles", grid}, {"serving", S}}, tp);
  const ng::Array& M = op.at("operator");
  CHECK(M.rows == 12 && M.cols == 12 && M.meta["synthesis_rank"] == 12);
  {
    Eigen::MatrixXd m(12, 12);
    for (int i = 0; i < 144; ++i) m(i / 12, i % 12) = M.data[static_cast<std::size_t>(i)];
    CHECK((m * serving - Eigen::MatrixXd::Identity(12, 12)).norm() < 1e-8);             // the served coordinates back to the tiles' currents
  }
  ng::Array wrongKernel = Kr; wrongKernel.meta["channels"] = std::vector<std::int32_t>(12, 99);
  CHECK(refused([&] { run_array("connectome.tile_operator", {{"kernel", wrongKernel}, {"tiles", grid}, {"serving", S}}, tp); }));

  ng::Array atoms;
  const double fwhm = 2.0 * std::sqrt(2.0 * std::log(2.0)) / 12.0;                           // standard deviation centre / 12
  atoms.meta["params"] = Json{{"kernels", Json{{"members", Json::array({Json{{"start", f}, {"stop", f}, {"family", "gaussian"}, {"chart", "linear"}, {"gain", 2.0}, {"params", Json{{"fwhm", fwhm}, {"width_rule", "proportional"}}}}})}}}};
  const Json fp = Json{{"frequency_normalisation", "none"}};
  const auto flow = run_array("connectome.tile_flow", {{"spectra", spectra}, {"operator", M}, {"frequency", atoms}}, fp);
  const ng::Array& F = flow.at("flow");
  CHECK(F.rows == 1 && F.cols == 2 * 6 + 4);
  const std::int64_t q01 = nxr::meeg::pairIndex(0, 1, 4);
  const double lag = -std::atan2(at(F, 0, 6 + q01), at(F, 0, q01)) / (2 * M_PI * f);
  std::printf("  planted lag %.4f s, read %.4f s\n", tau, lag);
  CHECK(std::abs(lag + tau) < 1e-3);
  CHECK(at(F, 0, 6 + q01) / std::sqrt(at(F, 0, 12) * at(F, 0, 13)) > 0.3);                 // the edge flow: tile 1 lags tile 0
  ng::Array wrongSpectra = spectra; wrongSpectra.meta["producer"]["modes"] = Json::array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99});
  CHECK(refused([&] { run_array("connectome.tile_flow", {{"spectra", wrongSpectra}, {"operator", M}, {"frequency", atoms}}, fp); }));

  const auto cells = run_array("connectome.tile_time_cells", {{"spectra", spectra}, {"operator", M}, {"frequency", atoms}}, Json{{"frequency_normalisation", "none"}, {"cell_width", 2.0}});
  const ng::Array& Z = cells.at("cells");
  const Json groups = Z.meta["describe"]["attributes"]["groups"];
  CHECK(groups.size() == 1 && Z.cols == 6 + 4 && Z.rows == groups[0]["cells"].get<std::int64_t>() && Z.rows >= 8);
  bool positive = true;
  for (std::int64_t r = 0; r < Z.rows; ++r) positive = positive && at(Z, r, q01) > 0;
  CHECK(positive);
  const auto flip = run_array("stats.sign_flip_consistency", {{"cells", Z}}, Json{{"elements", "edges"}, {"patterns", 500}, {"seed", 0}, {"cells_per_flip", 1}});
  const ng::Array& R = flip.at("consistency");
  CHECK(R.rows == 1 && std::abs(at(R, 0, q01) - 1) < 1e-12 && at(R, 0, 6 + q01) < 0.01);   // R1 = 1 and p small for the lagged pair

  const auto network = run_array("connectome.network_flow", {{"pairs", pairs}, {"eigenmodes", modes.at("eigenmodes")}, {"flow", F}}, Json{{"min_streamlines", 10}});
  const ng::Array& W = network.at("network");
  CHECK(W.rows == 1 && W.cols == 3 * 4 + 2);
  CHECK(at(W, 0, 0) < 0 && at(W, 0, 1) > 0 && std::abs(at(W, 0, 0) + at(W, 0, 1)) < 1e-12);  // out of tile 0, into tile 1
  CHECK(std::abs(at(W, 0, 12) / at(W, 0, 13) - 1) < 1e-9);                                  // a tree: every flow is a gradient
  CHECK(refused([&] { run_array("connectome.network_flow", {{"pairs", pairs}, {"eigenmodes", modes.at("eigenmodes")}, {"flow", F}}, Json{{"min_streamlines", 2}}); }));   // eigenmodes made with 10
  return nxrtest::finish("test_connectome_flow");
}
