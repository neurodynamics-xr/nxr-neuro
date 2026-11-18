// nxr-pipelines — THE QUADRATURE NODE (the hand-wired `quadrature compute` it replaces): what no zero-delay mixing can make of the sources' band cross-spectrum.
//
//   quadrature.source_pairs  array  spectrum (the sensors' band cross-spectrum, welch.cross_spectrum) + kernel (an inverse kernel on its channels)
//                                   ->  energy (per source and band: the quadrature energy, with the band's quadrature fraction and the sensors' rank),
//                                       pairs (per source: the strongest quadrature pairs p -> q of each band, with sigma, share and rotary sense)
//
// THE GRAPH (written first; checked against the Studio's chain and Rheome's):
//   recording (channels x samples, streamed)
//     --[welch.cross_spectrum, accumulate: the Welch segments are the partition; the kernel's channels]--> spectrum [bands x M x M, complex]
//     --[quadrature.source_pairs, array: F = K L with S = L L^H; Im(F F^H) read in factored form, never an N x N matrix]--> energy, pairs
// The recording is read by the chunk service only; the segments are a Rule on the time line (the cross-spectrum node's `align`); the node after the loop reads the accumulated
// cross-spectrum and the kernel, never a recording. Every reduction over time is the cross-spectrum's (a statistic over the segment partition); this node is a pointwise function
// of matrices (a factorisation and Gram products), with no neighbourhood: the kernel's rows are the source axis, one member each.
// Studio: the rows' linear map reads the chain BEFORE any non-linear transform (D106): here the kernel is applied to the cross-spectrum's factor, never to a squared quantity.
// Reference: nxr.spectral.lagged (compute nxr/quadrature.h).
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/pipelines/meeg_steps.h"
#include "nxr/core/algebra.h"
#include "nxr/measurement/congruence_norms.h"
#include "nxr/geometry/mesh.h"
#include "nxr/signal/complex_signal.h"
#include "nxr/signal/crossspectral.h"
#include "nxr/signal/crossspectral_decomposition.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/stats/circular.h"
#include "nxr/measurement/congruence_norms.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {

ng::PortType statistic_out() {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = "statistic";
  return t;
}

struct PairsPrepared { ng::Array spectrum, kernel; int pairs = 3; };

ng::NodeSpec source_pairs() {
  ng::NodeSpec n;
  n.name = "signal.source_planes";
  n.kind = ng::Kind::array;
  n.inputs = {{"spectrum", "the sensors' band cross-spectrum (signal.xpsd): bands x channels x channels, complex, on the kernel's channels", false},
              {"kernel", "an inverse kernel (free or fixed orientation) on the cross-spectrum's channels", false}};
  n.outputs = {{"energy", "per source and band, the quadrature energy sum_j (Im C_ij)^2, with each band's quadrature fraction and the sensors' rank (a statistic over the segment partition)", false},
               {"pairs", "per source, the strongest quadrature pairs p -> q of each band (vector3 for a free kernel), with sigma, share and rotary sense (a statistic over the segment partition)", false}};
  n.parameters = {{"pairs", Json(3), "the k strongest quadrature pairs per band", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "from an accumulated cross-spectrum, in double");
  n.matches = "nxr.spectral.lagged (im_fraction, im_rows, planes); compute nxr::signal";
  n.description = "the quadrature spectrum of the sources: per band the quadrature fraction, the energy per source and the strongest quadrature pairs, from the sensors' cross-spectrum and a kernel";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectrum", "kernel"}) if (!in.count(k)) ng::refuse(std::string("signal.source_planes: ") + k + " is not given");
    if (p.value("pairs", 3) < 1) ng::refuse("signal.source_planes: at least one pair");
    ng::Planned out;
    out.out["energy"] = statistic_out();
    out.out["pairs"] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    PairsPrepared pr;
    for (const char* k : {"spectrum", "kernel"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("signal.source_planes: ") + k + " was not handed over");
    pr.spectrum = c.at("spectrum");
    pr.kernel = c.at("kernel");
    pr.pairs = p.value("pairs", 3);
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const PairsPrepared& pr = std::any_cast<const PairsPrepared&>(prepared);
    const ng::Array& KA = pr.kernel;
    const ng::Array& SA = pr.spectrum;
    const Json d = SA.meta.value("describe", Json::object());
    const std::vector<std::int32_t> chans = KA.meta.value("channels", std::vector<std::int32_t>());
    const std::int64_t M = static_cast<std::int64_t>(chans.size());
    const int orientations = KA.meta.value("to_value_type", std::string("vector3")) == "scalar" ? 1 : 3;
    if (M == 0 || KA.cols != M) ng::refuse("signal.source_planes: the kernel's array is not [rows x its " + std::to_string(M) + " channels]");
    if (KA.rows % orientations != 0) ng::refuse("signal.source_planes: the kernel's rows are not a multiple of " + std::to_string(orientations));
    const std::int64_t N = KA.rows / orientations;
    const Json bands_in = d.contains("axes") && d["axes"].is_array() && !d["axes"].empty() ? d["axes"][0].value("values", Json::array()) : Json::array();
    const std::int64_t nb = static_cast<std::int64_t>(bands_in.size());
    if (nb < 1 || SA.rows != nb || SA.cols != M * M * 2) ng::refuse("signal.source_planes: the cross-spectrum is not bands x " + std::to_string(M) + " x " + std::to_string(M) + " (complex) on the kernel's channels");
    const bool density = d.value("measure", std::string()) == "cross_spectral_density";
    using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::MatrixXcd Kc = Eigen::Map<const MatRM>(KA.data.data(), KA.rows, M).cast<std::complex<double>>();

    std::vector<double> fraction(static_cast<std::size_t>(nb)), rank(static_cast<std::size_t>(nb)), cumulative(static_cast<std::size_t>(nb), 0.0);
    Eigen::MatrixXd energy = Eigen::MatrixXd::Zero(N, nb);
    std::vector<std::vector<nxr::signal::QuadraturePair>> pairs(static_cast<std::size_t>(nb));
    std::vector<std::vector<double>> share(static_cast<std::size_t>(nb));
    std::vector<std::vector<nxr::signal::RotaryPower>> rotary(static_cast<std::size_t>(nb));
    for (std::int64_t b = 0; b < nb; ++b) {
      Eigen::MatrixXcd Sb(M, M);
      for (std::int64_t i = 0; i < M; ++i)
        for (std::int64_t j = 0; j < M; ++j) {
          const std::size_t at = static_cast<std::size_t>(((b * M + i) * M + j) * 2);
          Sb(i, j) = std::complex<double>(SA.data[at], SA.data[at + 1]);
        }
      const Eigen::MatrixXcd L = nxr::core::hermitianFactor(Sb);
      rank[static_cast<std::size_t>(b)] = static_cast<double>(L.cols());
      const Eigen::MatrixXcd F = Kc * L;                                          // [(oN) x r]
      fraction[static_cast<std::size_t>(b)] = nxr::signal::quadratureFractionFromFactor(F);
      const Eigen::VectorXd rows = nxr::measurement::quadratureEnergy(F, F);
      for (std::int64_t i = 0; i < N; ++i) for (int o = 0; o < orientations; ++o) energy(i, b) += rows[orientations * i + o];
      const nxr::core::QuadratureFactor qf = nxr::core::quadratureFactor(F);
      pairs[static_cast<std::size_t>(b)] = nxr::signal::planes(qf, pr.pairs);
      for (const auto& q : pairs[static_cast<std::size_t>(b)]) {
        share[static_cast<std::size_t>(b)].push_back(nxr::signal::lagged_share(qf, {q}));
        rotary[static_cast<std::size_t>(b)].push_back(nxr::signal::rotaryPower(F, q.p, q.q));
      }
      cumulative[static_cast<std::size_t>(b)] = nxr::signal::lagged_share(qf, pairs[static_cast<std::size_t>(b)]);
    }
    std::int64_t total_pairs = 0;
    for (const auto& v : pairs) total_pairs += static_cast<std::int64_t>(v.size());
    if (total_pairs == 0) ng::refuse("signal.source_planes: the sensors' cross-spectrum has no quadrature pair in any band (no imaginary part beyond rounding): nothing travels or turns in it");

    // the statistics ride on the cross-spectrum's own partition (its Welch segments) and options
    Json bands = Json::array();
    for (std::int64_t b = 0; b < nb; ++b) {
      Json e = bands_in[static_cast<std::size_t>(b)];
      e["quadrature_fraction"] = fraction[static_cast<std::size_t>(b)];
      e["sensor_rank"] = rank[static_cast<std::size_t>(b)];
      e.erase("spectral_scaling");                                                  // the sensors' scaling is of C, these are of (Im C)^2
      bands.push_back(std::move(e));
    }
    Json producer = d.value("producer", Json::object());
    producer["stage"] = "quadrature";
    producer["kernel"] = KA.meta.value("name", std::string());
    producer["orientations"] = orientations;
    producer["orientation_combination"] = orientations == 3 ? "quadrature taken over the 3N rows (x, y, z of each source); the energy of a source is the sum of its three rows' energies; a pair is one pattern over the 3N rows, stored as its x, y, z components"
                                                            : "fixed orientation";
    producer["quadrature_fraction_formula"] = "||Im C||_F / ||C||_F, C = F F^H the band's cross-spectrum of the sources, F = K L, S_b = L L^H the band's sensor cross-spectrum";
    producer["pairs"] = pr.pairs;
    Json options = d.value("options", Json::object());
    options["pairs"] = pr.pairs;
    options["kernel"] = KA.meta.value("name", std::string());
    auto common = [&](const char* measure, const std::string& value_type, const std::vector<std::int64_t>& shape, const Json& unit, const Json& axes, const Json& extra, const std::string& what) {
      Json pj = producer;
      for (auto it = extra.begin(); it != extra.end(); ++it) pj[it.key()] = it.value();
      return Json{{"measure", measure}, {"method", "welch"}, {"label", "welch"}, {"precision", "float32"}, {"value_type", value_type}, {"shape", shape}, {"unit", unit}, {"axes", axes},
                  {"tiling", d.value("tiling", Json())}, {"n_members", d.value("n_members", Json())}, {"members_skipped", d.value("members_skipped", Json())},
                  {"producer", pj}, {"options", options}, {"bad_samples", d.value("bad_samples", Json())},
                  {"description", what + " of the sources through " + KA.meta.value("name", std::string())}};
    };
    ng::Array E;
    E.rows = N; E.cols = nb;
    E.data.resize(static_cast<std::size_t>(N * nb));
    for (std::int64_t i = 0; i < N; ++i) for (std::int64_t b = 0; b < nb; ++b) E.data[static_cast<std::size_t>(i * nb + b)] = energy(i, b);
    const Json eunit = density ? Json(std::string("A\xe2\x81\xb4\xc2\xb7m\xe2\x81\xb4/Hz\xc2\xb2")) : Json(std::string("A\xe2\x81\xb4\xc2\xb7m\xe2\x81\xb4"));
    E.meta["describe"] = common("quadrature_energy", "scalar", {N, nb}, eunit, Json::array({Json{{"name", "vertex"}, {"size", N}}, Json{{"name", "band"}, {"size", nb}, {"values", bands}}}),
                                Json{{"energy_formula", "per source, sum_j (Im C_ij)^2"}}, "the quadrature energy (per source, the sum over the other sources of (Im C)^2), per band");

    const std::int64_t C = orientations == 3 ? 3 : 1;
    Json entries = Json::array();
    std::vector<const Eigen::VectorXd*> cols;
    for (std::int64_t b = 0; b < nb; ++b)
      for (std::size_t k = 0; k < pairs[static_cast<std::size_t>(b)].size(); ++k) {
        const auto& qp = pairs[static_cast<std::size_t>(b)][k];
        for (char which : {'p', 'q'}) {
          cols.push_back(which == 'p' ? &qp.p : &qp.q);
          entries.push_back(Json{{"name", "pair" + std::to_string(k + 1) + "_" + which + "_" + bands_in[static_cast<std::size_t>(b)].value("name", std::string())}, {"band", bands_in[static_cast<std::size_t>(b)].value("name", std::string())},
                                 {"pair", static_cast<int>(k) + 1}, {"pattern", std::string(1, which)}, {"sigma", qp.sigma}, {"share", share[static_cast<std::size_t>(b)][k]},
                                 {"cumulative_share", cumulative[static_cast<std::size_t>(b)]}, {"rotary_sense", rotary[static_cast<std::size_t>(b)][k].sense},
                                 {"rotary_power_positive", rotary[static_cast<std::size_t>(b)][k].positive}, {"rotary_power_negative", rotary[static_cast<std::size_t>(b)][k].negative}});
        }
      }
    const std::int64_t Ecount = static_cast<std::int64_t>(cols.size());
    ng::Array P;
    P.rows = N; P.cols = Ecount * C;
    P.data.resize(static_cast<std::size_t>(N * Ecount * C));
    for (std::int64_t e = 0; e < Ecount; ++e)
      for (std::int64_t i = 0; i < N; ++i)
        for (std::int64_t c = 0; c < C; ++c) P.data[static_cast<std::size_t>((i * Ecount + e) * C + c)] = (*cols[static_cast<std::size_t>(e)])[i * C + c];
    const std::vector<std::int64_t> pshape = C == 3 ? std::vector<std::int64_t>{N, Ecount, 3} : std::vector<std::int64_t>{N, Ecount};
    P.meta["describe"] = common("quadrature_pattern", C == 3 ? "vector3" : "scalar", pshape, Json(),
                                Json::array({Json{{"name", "vertex"}, {"size", N}}, Json{{"name", "pattern"}, {"size", Ecount}, {"values", entries}}}),
                                Json{{"sigma_unit", density ? "A^2 m^2 / Hz" : "A^2 m^2"}, {"share_formula", "2 sigma^2 / ||Im C||_F^2"},
                                     {"pair_formula", "Im C contains sigma (p q^T - q p^T): p leads q by a quarter cycle; rotary sense +1 = p -> q counter-clockwise (power at +f)"},
                                     {"gauge", "the row where p^2 + q^2 is largest has q = 0 and p > 0"}, {"unit_norm", "over all rows of the kernel (3N for a free kernel)"}},
                                "the quadrature pairs (p leads q by a quarter cycle)");
    return ng::Arrays{{"energy", std::move(E)}, {"pairs", std::move(P)}};
  };
  return n;
}

}  // namespace

void register_quadrature_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {source_pairs()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
