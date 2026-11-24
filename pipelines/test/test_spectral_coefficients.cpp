// The nodes `signal.mode_analysis` and `signal.spectral_coefficients` (pipelines/src/spectral_steps.cpp) through the graph's dispatch point, on synthetic data (never a real dataset):
//   a small sphere standing for a sensor surface (each vertex a channel through a channel map), modes made ⋆0-orthonormal here and stored as a package stores them (ψ/√2);
//   mode_analysis: the exact modes' rows under unit_mass, its channels the map's; refused without a kernel or a map;
//   spectral_coefficients: a recording of three tiles fed in blocks that do not fall on the tiles, one bad sample in the middle tile: the first and last tiles' values equal
//   spectralCoefficients on the tiles, the middle tile is left out and listed; a span ending inside a tile is refused.
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"
#include "nxr/pipelines/spectral_steps.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/mode_spectrum.h"

namespace ng = nxr::graph;
using ng::Json;

namespace {
template <class F> bool refused(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
}  // namespace

int main() {
  nxr::pipelines::graph::register_spectral_nodes();
  const ng::NodeSpec& analysis = ng::registry().at("signal.mode_analysis");
  const ng::NodeSpec& coefficients = ng::registry().at("signal.spectral_coefficients");
  CHECK(analysis.traits.declared() && coefficients.traits.declared());

  std::vector<double> V; std::vector<int32_t> F;
  icosphere(1, V, F, 1.0);
  const int nV = static_cast<int>(V.size() / 3), K = 10, C = nV + 3;
  const nxr::signal::IntrinsicComplex cx = nxr::signal::intrinsicComplex(nxr::geometry::surfaceArrays(V.data(), nV, F.data(), static_cast<int>(F.size() / 3)));
  const Eigen::VectorXd mass = cx.star0;
  std::mt19937 rng(3);
  std::normal_distribution<double> nd(0, 1);
  Eigen::MatrixXd Q(nV, K);
  for (int v = 0; v < nV; ++v) for (int k = 0; k < K; ++k) Q(v, k) = nd(rng);
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < k; ++j) Q.col(k) -= Q.col(j).cwiseProduct(mass).dot(Q.col(k)) * Q.col(j);
    Q.col(k) /= std::sqrt(Q.col(k).cwiseAbs2().dot(mass));
  }
  std::vector<int> kind(K);
  std::vector<double> mu(K);
  for (int k = 0; k < K; ++k) { kind[static_cast<std::size_t>(k)] = k % 3 == 2 ? 2 : 0; mu[static_cast<std::size_t>(k)] = 1.0 + k; }
  // the package constant as the runner binds it: [K x V] mode-major, ψ/√2
  ng::Array H;
  H.rows = K; H.cols = nV;
  for (int k = 0; k < K; ++k) for (int v = 0; v < nV; ++v) H.data.push_back(Q(v, k) / std::sqrt(2.0));
  std::vector<std::int64_t> keep(static_cast<std::size_t>(nV));
  for (int v = 0; v < nV; ++v) keep[static_cast<std::size_t>(v)] = v;
  H.meta["hodge"] = true; H.meta["mu"] = mu; H.meta["kind"] = kind; H.meta["positions"] = V; H.meta["faces"] = F; H.meta["keep"] = keep; H.meta["input_positions"] = V;
  H.meta["mass"] = std::vector<double>(mass.data(), mass.data() + nV);
  H.meta["name"] = "sphere modes";
  ng::Array map;
  map.rows = nV; map.cols = 1;
  for (int v = 0; v < nV; ++v) map.data.push_back(v + 2);                  // vertex v is the recording's channel v + 2

  // mode_analysis
  const ng::Types no_types;
  ng::Shared shared;
  const Json ap = Json{{"modes", "exact"}, {"normalisation", "unit_mass"}};
  const ng::Arrays aconst{{"package", H}, {"channel_map", map}}, none, only_package{{"package", H}};
  ng::NodeCall prep = ng::NodeCall::prepare(no_types, aconst, ap, shared);
  ng::dispatch(analysis, prep);
  ng::NodeCall run = ng::NodeCall::run(prep.made, none);
  ng::dispatch(analysis, run);
  const ng::Array O = run.made_arrays.at("operator");
  const nxr::signal::ModeAnalysis ma = nxr::signal::modeAnalysis(Q / std::sqrt(2.0), mass, kind, nxr::signal::ModeKinds::exact, nxr::signal::ModeNormalisation::unit_mass);
  const Eigen::MatrixXd Om = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(O.data.data(), O.rows, O.cols);
  CHECK(O.rows == 7 && O.cols == nV && (Om - ma.rows).cwiseAbs().maxCoeff() < 1e-12);
  CHECK(O.meta.at("channels").get<std::vector<std::int32_t>>().front() == 2 && O.meta.at("components").get<int>() == 1);
  CHECK((Om * ma.synthesis - Eigen::MatrixXd::Identity(7, 7)).cwiseAbs().maxCoeff() < 1e-10);
  CHECK(refused([&] { ng::NodeCall p2 = ng::NodeCall::prepare(no_types, only_package, ap, shared); ng::dispatch(analysis, p2); }));

  // spectral_coefficients: three tiles of 64 samples, blocks of 50, a bad sample in the middle tile
  const double fs = 256;
  const std::int64_t tile = 64, total = 3 * tile;
  Eigen::MatrixXf x(C, total);
  for (int ch = 0; ch < C; ++ch) for (std::int64_t t = 0; t < total; ++t) x(ch, t) = static_cast<float>(std::cos(2 * M_PI * (5 + ch) * t / fs) + 0.3 * nd(rng));
  ng::Types types;
  ng::PortType xt;
  xt.sfreq = fs;
  types["x"] = xt;
  const Json cp = Json{{"tile_samples", tile}, {"frequency_span", Json::array({0.0, 64.0})}, {"bad_segments", "exclude"}};
  const ng::Arrays cconst{{"operator", O}};
  ng::NodeCall cprep = ng::NodeCall::prepare(types, cconst, cp, shared);
  ng::dispatch(coefficients, cprep);
  auto feed = [&](std::int64_t end) {
    ng::NodeCall st = ng::NodeCall::start(cprep.made);
    ng::dispatch(coefficients, st);
    std::any state = std::move(st.made);
    for (std::int64_t first = 0; first < end; first += 50) {
      const std::int64_t own = std::min<std::int64_t>(50, end - first);
      ng::Block b;
      b.rows = C; b.width = own;
      for (int ch = 0; ch < C; ++ch) for (std::int64_t t = 0; t < own; ++t) b.re.push_back(x(ch, first + t));
      std::vector<std::uint8_t> good(static_cast<std::size_t>(own), 1);
      if (first <= 100 && 100 < first + own) good[static_cast<std::size_t>(100 - first)] = 0;
      ng::Place at; at.first = first; at.n = own; at.good = &good;
      const ng::Blocks blocks{{"x", b}};
      ng::NodeCall s = ng::NodeCall::step(cprep.made, std::move(state), blocks, 0, at);
      ng::dispatch(coefficients, s);
      state = std::move(s.made_state);
    }
    ng::NodeCall fin = ng::NodeCall::finish(cprep.made, std::move(state));
    ng::dispatch(coefficients, fin);
    return fin.made_arrays.at("coefficients");
  };
  const ng::Array A = feed(total);
  const Json& d = A.meta.at("describe");
  // a statistic of the run on three declared axes: the tiles kept (positions on the time line), the bins kept of the frequency line, the modes kept of the package's mode axis
  const std::int64_t bins = d["axes"][1]["size"].get<std::int64_t>(), R = O.rows;
  CHECK(bins == 16 && d["axes"][1]["members"].size() == 16 && d["axes"][1]["line"]["fft"]["n"] == tile);
  CHECK(d["axes"][0]["size"] == 2 && d["axes"][0]["values"][0]["member"] == 0 && d["axes"][0]["values"][1]["member"] == 2 && d["options"]["tiles_left_out"] == Json::array({1}));
  CHECK(d["axes"][2]["size"] == R && d["axes"][2]["members"].size() == static_cast<std::size_t>(R) && d["value_type"] == "complex");
  CHECK(A.rows == 2 && A.cols == 2 * bins * R && d["shape"] == Json::array({2, bins, R, 2}));
  double worst = 0, scale = 0;
  for (int i = 0; i < 2; ++i) {
    const std::int64_t t0 = i == 0 ? 0 : 2 * tile;
    Eigen::MatrixXd tileX(nV, tile);
    for (int v = 0; v < nV; ++v) tileX.row(v) = x.row(v + 2).segment(t0, tile).cast<double>();
    const Eigen::MatrixXcd ref = nxr::signal::spectralCoefficients(ma.rows, tileX, 0, bins);
    for (std::int64_t k = 0; k < bins; ++k)
      for (std::int64_t r = 0; r < R; ++r) {
        const std::size_t at = static_cast<std::size_t>(((i * bins + k) * R + r) * 2);
        worst = std::max(worst, std::abs(std::complex<double>(A.at(at), A.at(at + 1)) - ref(r, k)));
        scale = std::max(scale, std::abs(ref(r, k)));
      }
  }
  std::printf("  tiles 0 and 2: worst |difference| %.3g of %.3g (float32 values)\n", worst, scale);
  CHECK(worst < 1e-5 * scale);
  CHECK(refused([&] { feed(total - 10); }));
  std::printf("test_spectral_coefficients: OK\n");
  return 0;
}
