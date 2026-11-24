// The source-flow nodes (pipelines/src/source_flow_steps.cpp) through the graph's dispatch point, on synthetic data only (never a real dataset):
//   a "cortex" of two spheres stored as one surface (each sphere a hemisphere, a component with its vertex range), each with its Hodge-Dirac modes solved here;
//   a sensor sheet of 42 channels with a complete, mass-orthonormal set of field coordinates (and a truncated one);
//   a free-orientation kernel whose x current on the left hemisphere is cos(2 z) from channel 0 and sin(2 z) from channel 1 (small random rows elsewhere);
//   a recording whose channels 0 and 1 carry cos(w t) and sin(w t), so the left hemisphere's current is cos(w t - 2 z): a wave TRAVELLING towards +z.
// Checked: signal.mode_analysis selects each hemisphere's rows of the cortex kernel (and a glued-style package's bridge rows are zero); signal.source_mode_operator option B is
// the mode operator itself and option A equals option B when the sensor basis is complete (nothing left out; the cortical flow identical), with a truncated basis what option A
// leaves out is stated; signal.cortical_flow: the travelling source maps to an exact (travelling) flow with a small remainder, the band pairs add back to the whole field and the
// complement band holds the kernel's content outside the kept modes; signal.cortical_time_cells + stats.sign_flip_consistency: the travelling wave is consistent over the cells.
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/pipelines/sensor_flow_steps.h"
#include "nxr/pipelines/source_flow_steps.h"
#include "nxr/pipelines/spectral_steps.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/spectral_flow.h"

namespace ng = nxr::graph;
using ng::Json;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
Eigen::MatrixXd matrix_of(const ng::Array& a) { const std::vector<double> v = a.values(); return Eigen::Map<const RowMajor>(v.data(), a.rows, a.cols); }
double relative(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { d = std::max(d, static_cast<double>(std::abs(a[i] - b[i]))); s = std::max(s, static_cast<double>(std::abs(b[i]))); }
  return a.size() == b.size() ? d / std::max(s, 1e-300) : 1e9;
}
// a sphere's Hodge-Dirac package as the runner hands it over
struct Sphere { std::vector<double> positions; std::vector<std::int32_t> faces; nxr::signal::Modes modes; nxr::signal::IntrinsicComplex complex; ng::Array package; };
Sphere sphere(int level, int modeCount, double xOffset) {
  Sphere s;
  icosphere(level, s.positions, s.faces, 1.0);
  for (std::size_t i = 0; i < s.positions.size(); i += 3) s.positions[i] += xOffset;     // the right hemisphere moved along x
  const int nV = static_cast<int>(s.positions.size() / 3), nF = static_cast<int>(s.faces.size() / 3);
  const auto surface = nxr::geometry::surfaceArrays(s.positions.data(), nV, s.faces.data(), nF);
  const auto h = nxr::operators::hodgedirac::assemble(surface);
  nxr::operators::HodgeSolveOptions so; so.limit = 40;
  s.modes = nxr::signal::modesOf(nxr::operators::hodgeDiracModes(h, modeCount, false, so), h.nV, h.nE, h.nF);
  s.complex = nxr::signal::intrinsicComplex(surface);
  const int K = static_cast<int>(s.modes.mu.size());
  ng::Array& H = s.package;
  H.rows = K; H.cols = nV;
  for (int k = 0; k < K; ++k) for (int v = 0; v < nV; ++v) H.data.push_back(s.modes.P(v, k));
  H.meta["hodge"] = true; H.meta["mu"] = std::vector<double>(s.modes.mu.data(), s.modes.mu.data() + K); H.meta["kind"] = s.modes.kind; H.meta["positions"] = s.positions; H.meta["faces"] = s.faces;
  H.meta["mass"] = std::vector<double>(s.complex.star0.data(), s.complex.star0.data() + nV);
  std::vector<std::int64_t> keep(static_cast<std::size_t>(nV));
  for (int v = 0; v < nV; ++v) keep[static_cast<std::size_t>(v)] = v;
  H.meta["keep"] = keep;
  H.meta["input_positions"] = s.positions;
  std::vector<float> x2(static_cast<std::size_t>(s.complex.nF * K));
  for (int f = 0; f < s.complex.nF; ++f) for (int k = 0; k < K; ++k) x2[static_cast<std::size_t>(f * K + k)] = static_cast<float>(s.modes.X2(f, k));
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(x2.data());
  H.meta["forms2"] = Json::binary(std::vector<std::uint8_t>(bytes, bytes + x2.size() * sizeof(float)));
  H.meta["forms2_shape"] = Json::array({s.complex.nF, K});
  H.meta["complex"] = Json{{"mollify", 1e-6}, {"flip_eps", 1e-6}, {"star1_floor", 0.0}};
  return s;
}
}  // namespace

int main() {
  nxr::pipelines::graph::register_sensor_flow_nodes();
  nxr::pipelines::graph::register_spectral_nodes();
  nxr::pipelines::graph::register_source_flow_nodes();
  for (const char* name : {"signal.source_mode_operator", "signal.cortical_flow", "signal.cortical_time_cells"}) CHECK(ng::registry().at(name).traits.declared());
  std::mt19937 rng(17);
  std::normal_distribution<double> normal(0, 1);

  // ── the cortex: two spheres as hemispheres of one surface ──
  Sphere left = sphere(2, 80, 0.0), right = sphere(2, 80, 3.0);
  const int nV = static_cast<int>(left.positions.size() / 3);
  left.package.meta["input_component_of"] = "cortex"; left.package.meta["input_from_index"] = 0; left.package.meta["name"] = "left";
  right.package.meta["input_component_of"] = "cortex"; right.package.meta["input_from_index"] = nV; right.package.meta["name"] = "right";

  // ── the sensor sheet: 42 channels, a complete mass-orthonormal coordinate set (synthesis Psi = mass^-1/2 x an orthogonal matrix), the constant first ──
  const int channels = 42;
  Eigen::VectorXd sensorMass(channels);
  for (int c = 0; c < channels; ++c) sensorMass(c) = 0.5 + 0.01 * c;
  Eigen::MatrixXd orth = Eigen::MatrixXd::Zero(channels, channels);
  orth.col(0) = sensorMass.cwiseSqrt() / std::sqrt(sensorMass.sum());
  for (int j = 1; j < channels; ++j) for (int c = 0; c < channels; ++c) orth(c, j) = normal(rng);
  const Eigen::HouseholderQR<Eigen::MatrixXd> qr(orth);
  Eigen::MatrixXd orthonormal = qr.householderQ() * Eigen::MatrixXd::Identity(channels, channels);
  if (orthonormal.col(0).dot(orth.col(0)) < 0) orthonormal.col(0) *= -1;
  const Eigen::MatrixXd sensorSynthesis = sensorMass.cwiseSqrt().cwiseInverse().asDiagonal() * orthonormal;   // [channels x K_s]
  auto sensorOperator = [&](int coordinates) {
    const Eigen::MatrixXd analysis = sensorSynthesis.leftCols(coordinates).transpose() * sensorMass.asDiagonal();
    ng::Array a;
    a.rows = analysis.rows(); a.cols = analysis.cols();
    for (Eigen::Index r = 0; r < analysis.rows(); ++r) for (Eigen::Index c = 0; c < analysis.cols(); ++c) a.data.push_back(analysis(r, c));
    std::vector<std::int32_t> ch(channels); std::vector<std::int64_t> modes;
    for (int c = 0; c < channels; ++c) ch[static_cast<std::size_t>(c)] = channels - 1 - c;       // the sheet's vertices list the channels backwards: the composition matches by channel
    for (int k = 0; k < coordinates; ++k) modes.push_back(k - 1);
    a.meta["channels"] = ch; a.meta["modes"] = modes; a.meta["mu"] = std::vector<double>(static_cast<std::size_t>(coordinates), 1.0);
    a.meta["rows"] = Json{{"normalisation", "unit_mass"}};
    return a;
  };
  // the sheet's vertex v is channel channels - 1 - v: the operator's columns are vertices; reorder the synthesis rows accordingly
  ng::Array sensorPackage;
  sensorPackage.meta["mass"] = std::vector<double>(sensorMass.data(), sensorMass.data() + channels);

  // ── the kernel: 3 x (2 nV) rows, the left hemisphere's x current cos(2 z) / sin(2 z) from channels 0 / 1 ──
  Eigen::MatrixXd W(3 * 2 * nV, channels);
  for (Eigen::Index i = 0; i < W.size(); ++i) W.data()[i] = 1e-3 * normal(rng);
  for (int v = 0; v < nV; ++v) { const double z = left.positions[static_cast<std::size_t>(3 * v + 2)]; W(3 * v, 0) = std::cos(2 * z); W(3 * v, 1) = std::sin(2 * z); }
  ng::Array kernel;
  kernel.rows = W.rows(); kernel.cols = W.cols();
  for (Eigen::Index r = 0; r < W.rows(); ++r) for (Eigen::Index c = 0; c < W.cols(); ++c) kernel.data.push_back(W(r, c));
  std::vector<std::int32_t> kernelChannels(channels);
  for (int c = 0; c < channels; ++c) kernelChannels[static_cast<std::size_t>(c)] = c;
  kernel.meta["channels"] = kernelChannels; kernel.meta["to_manifold_id"] = "cortex"; kernel.meta["to_value_type"] = "vector3"; kernel.meta["name"] = "synthetic_kernel";

  // ── signal.mode_analysis: each hemisphere's rows of the cortex kernel ──
  const Json modeChoice = Json{{"modes", "exact_and_constant"}, {"normalisation", "unit_mass"}};
  const ng::Array cortexLeft = run_array("signal.mode_analysis", ng::Arrays{{"package", left.package}, {"kernel", kernel}}, modeChoice).at("operator");
  const ng::Array cortexRight = run_array("signal.mode_analysis", ng::Arrays{{"package", right.package}, {"kernel", kernel}}, modeChoice).at("operator");
  {
    // the right hemisphere's operator from its own rows: the mode analysis of rows [3 nV, 6 nV) alone
    ng::Array alone = kernel;
    alone.rows = 3 * nV; alone.data.assign(kernel.data.begin() + 3 * nV * channels, kernel.data.end());
    alone.meta.erase("to_manifold_id");
    ng::Array own = right.package; own.meta.erase("input_from_index");
    const ng::Array direct = run_array("signal.mode_analysis", ng::Arrays{{"package", own}, {"kernel", alone}}, modeChoice).at("operator");
    CHECK((matrix_of(direct) - matrix_of(cortexRight)).cwiseAbs().maxCoeff() == 0);
    // a glued-style package: its last 5 vertices are a bridge (keep -1), its input surface has 3 more (wall) vertices: the bridge rows are zero, never blended
    ng::Array glued = left.package;
    std::vector<std::int64_t> keep = glued.meta["keep"].get<std::vector<std::int64_t>>();
    for (int i = nV - 5; i < nV; ++i) keep[static_cast<std::size_t>(i)] = -1;
    glued.meta["keep"] = keep; glued.meta.erase("input_from_index");
    std::vector<double> inputPositions(left.positions.begin(), left.positions.begin() + 3 * (nV - 2));
    glued.meta["input_positions"] = inputPositions;
    ng::Array wall = kernel; wall.rows = 3 * (nV - 2); wall.data.assign(kernel.data.begin(), kernel.data.begin() + 3 * (nV - 2) * channels); wall.meta.erase("to_manifold_id");
    const ng::Array g = run_array("signal.mode_analysis", ng::Arrays{{"package", glued}, {"kernel", wall}}, modeChoice).at("operator");
    // its operator equals the analysis of the kernel with the bridge rows zero: compare with a kernel whose bridge-source rows (the wall's) are 1e6 — nothing changes
    ng::Array loud = wall;
    for (std::size_t i = static_cast<std::size_t>(3 * (nV - 5) * channels); i < loud.data.size(); ++i) loud.data[i] = 1e6;
    const ng::Array g2 = run_array("signal.mode_analysis", ng::Arrays{{"package", glued}, {"kernel", loud}}, modeChoice).at("operator");
    CHECK((matrix_of(g) - matrix_of(g2)).cwiseAbs().maxCoeff() == 0);
    std::printf("  hemispheres selected by vertex range; a glued package's bridge rows zero (a wall at 1e6 changes nothing)\n");
    CHECK(refused([&] { ng::Array bad = right.package; bad.meta["input_component_of"] = "other"; run_array("signal.mode_analysis", ng::Arrays{{"package", bad}, {"kernel", kernel}}, modeChoice); }));
  }

  // ── signal.source_mode_operator: options B and A ──
  auto sourceOperator = [&](const ng::Array& cortexOperator, const ng::Array& package, const char* columns, int coordinates) {
    ng::Arrays in{{"cortex_operator", cortexOperator}, {"kernel", kernel}, {"package", package}};
    if (std::string(columns) == "sensor_coordinates") {
      // the sheet's operator columns are its vertices (channel channels-1-v): reorder the analysis columns to match
      ng::Array op = sensorOperator(coordinates);
      const Eigen::MatrixXd byChannel = matrix_of(op);
      op.data.clear();
      for (Eigen::Index r = 0; r < byChannel.rows(); ++r) for (int v = 0; v < channels; ++v) op.data.push_back(byChannel(r, channels - 1 - v));
      in["sensor_operator"] = op;
      ng::Array mass = sensorPackage;
      std::vector<double> mv(channels);
      for (int v = 0; v < channels; ++v) mv[static_cast<std::size_t>(v)] = sensorMass(channels - 1 - v);
      mass.meta["mass"] = mv;
      in["sensor_package"] = mass;
    }
    return run_array("signal.source_mode_operator", in, Json{{"columns", columns}});
  };
  const ng::Arrays optionB = sourceOperator(cortexLeft, left.package, "channels", channels);
  const ng::Arrays optionA = sourceOperator(cortexLeft, left.package, "sensor_coordinates", channels);
  const ng::Arrays optionATruncated = sourceOperator(cortexLeft, left.package, "sensor_coordinates", channels - 6);
  CHECK((matrix_of(optionB.at("operator")) - matrix_of(cortexLeft)).cwiseAbs().maxCoeff() == 0);
  const double leftOutComplete = optionA.at("operator").meta.at("option_a_left_out").at("left_out_frobenius_share").get<double>();
  const double leftOutTruncated = optionATruncated.at("operator").meta.at("option_a_left_out").at("left_out_frobenius_share").get<double>();
  std::printf("  option A left out: complete sensor basis %.2e, truncated by 6 coordinates %.3f; operator %lld x %lld, rank %lld\n", leftOutComplete, leftOutTruncated,
              static_cast<long long>(optionA.at("operator").rows), static_cast<long long>(optionA.at("operator").cols), static_cast<long long>(optionA.at("operator").meta.at("rank").get<std::int64_t>()));
  CHECK(leftOutComplete < 1e-10 && leftOutTruncated > 1e-3);
  CHECK(refused([&] { run_array("signal.source_mode_operator", ng::Arrays{{"cortex_operator", cortexLeft}, {"kernel", kernel}, {"package", left.package}}, Json{{"columns", "sensor_coordinates"}}); }));

  // ── the recording: channels 0 and 1 carry cos(w t) and sin(w t) (+ noise), three chunks; the chunk spectra through each option's serving operator ──
  const double fs = 128.0, f0 = 10.0;
  const std::int64_t chunk = 256, overlap = 128, transform = 512, total = 3 * chunk;
  Eigen::MatrixXd x(channels, total);
  for (int c = 0; c < channels; ++c) for (std::int64_t t = 0; t < total; ++t) x(c, t) = 0.01 * normal(rng);
  for (std::int64_t t = 0; t < total; ++t) { x(0, t) += std::cos(2 * M_PI * f0 * t / fs); x(1, t) += std::sin(2 * M_PI * f0 * t / fs); }
  const std::int64_t bins = static_cast<std::int64_t>(40.0 / (fs / transform));
  auto spectra_through = [&](const ng::Array& serving) {
    const Eigen::MatrixXd op = matrix_of(serving);
    const std::vector<std::int32_t> ch = serving.meta.at("channels").get<std::vector<std::int32_t>>();
    Eigen::MatrixXd byColumn(op.cols(), total);
    for (std::size_t i = 0; i < ch.size(); ++i) byColumn.row(static_cast<Eigen::Index>(i)) = x.row(ch[i]);
    ng::Array a;
    Json chunks = Json::array();
    for (std::int64_t k = 0; k < 3; ++k) {
      const std::int64_t leftSide = std::min(overlap, k * chunk), rightSide = std::min(overlap, total - (k + 1) * chunk);
      const Eigen::MatrixXcd X = nxr::signal::overlappedSpectrum(op, byColumn.middleCols(k * chunk - leftSide, leftSide + chunk + rightSide), leftSide, rightSide, overlap, transform, 0, bins);
      for (std::int64_t b = 0; b < bins; ++b) for (Eigen::Index r = 0; r < X.rows(); ++r) { a.single.push_back(static_cast<float>(X(r, b).real())); a.single.push_back(static_cast<float>(X(r, b).imag())); }
      chunks.push_back(Json{{"first", k * chunk}, {"samples", chunk}});
    }
    a.rows = 3; a.cols = bins * op.rows() * 2;
    a.meta["shape"] = Json::array({3, bins, op.rows(), 2});
    a.meta["producer"] = Json{{"sfreq", fs}, {"transform_samples", transform}, {"overlap_samples", overlap}, {"first_bin", 0}, {"bins", bins}, {"modes", serving.meta.at("modes")}, {"chunks", chunks}};
    return a;
  };
  const ng::Array spectraB = spectra_through(optionB.at("serving")), spectraA = spectra_through(optionA.at("serving"));

  // ── the atoms and the cortical bands as select.kernels stores them ──
  auto kernel_set = [](const Json& members) { ng::Array a; a.meta["params"] = Json{{"kernels", Json{{"members", members}}}}; a.meta["type"] = "set"; return a; };
  Json atoms = Json::array();
  for (const double c : {5.0, 10.0, 20.0})
    atoms.push_back(Json{{"start", c}, {"stop", c}, {"family", "gaussian"}, {"chart", "linear"}, {"gain", 2.0}, {"params", Json{{"fwhm_per_centre", 2.0 * std::sqrt(2.0 * std::log(2.0)) / 12.0 * 3.0}}}});
  Json bands = Json::array();
  for (const double c : {1.5, 3.0, 5.0}) bands.push_back(Json{{"start", c}, {"stop", c}, {"family", "cos2_partition"}, {"chart", "log2"}, {"gain", 1.0}});
  const Json fp = Json{{"frequency_weighting", "kernels"}, {"frequency_normalisation", "sum_of_squared_gains"}, {"support_floor", 1e-3}};
  auto flow_of = [&](const ng::Arrays& source, const ng::Array& spectra) {
    return run_array("signal.cortical_flow", ng::Arrays{{"spectra", spectra}, {"package", left.package}, {"operator", source.at("operator")}, {"kernel_columns", source.at("kernel_columns")},
                                                         {"frequency", kernel_set(atoms)}, {"scale", kernel_set(bands)}}, fp);
  };
  const ng::Arrays flowB = flow_of(optionB, spectraB), flowA = flow_of(optionA, spectraA);
  // option A = option B with a complete sensor basis
  double worst = 0;
  for (const char* o : {"whole_edges", "whole_vertices", "pair_edges"}) worst = std::max(worst, relative(flowA.at(o).single, flowB.at(o).single));
  std::printf("  option A vs option B, complete sensor basis: %.2e\n", worst);
  CHECK(worst < 1e-4);
  // the travelling source: at 10 Hz (member 1) the whole field's flow is mostly exact, and the remainder beyond the kept modes small
  const ng::Array& wm = flowB.at("whole_modes");
  const std::int64_t M = static_cast<std::int64_t>(left.modes.mu.size());
  double exact = 0, all = 0;
  for (std::int64_t k = 0; k < M; ++k) { const double c2 = std::pow(wm.single[static_cast<std::size_t>(k * 3 + 1)], 2); all += c2; if (left.modes.kind[static_cast<std::size_t>(k)] == 0) exact += c2; }
  const double energy = wm.single[static_cast<std::size_t>(M * 3 + 1)], remainder = wm.single[static_cast<std::size_t>((M + 1) * 3 + 1)];
  std::printf("  travelling source at 10 Hz: exact share of the flow %.3f, the remainder's share of the energy %.3f\n", exact / all, remainder / energy);
  CHECK(exact / all > 0.9 && remainder / energy < 0.2);
  // the travel direction: the exact potential (travel runs down it... J < 0 along travel) correlates with z: the phase current along +z edges is negative
  {
    const ng::Array& we = flowB.at("whole_edges");
    double along = 0;
    for (int e = 0; e < left.complex.nE; ++e) {
      const int tail = left.complex.edges[static_cast<std::size_t>(2 * e)], tip = left.complex.edges[static_cast<std::size_t>(2 * e + 1)];
      along += we.single[static_cast<std::size_t>(e * 3 + 1)] * (left.positions[static_cast<std::size_t>(3 * tip + 2)] - left.positions[static_cast<std::size_t>(3 * tail + 2)]);
    }
    std::printf("  phase current along +z: %.3g (negative: travel towards +z)\n", along);
    CHECK(along < 0);
  }
  // the band pairs add back to the whole field
  {
    const ng::Array& we = flowB.at("whole_edges"), &pe = flowB.at("pair_edges");
    double err = 0, scale = 0;
    for (int e = 0; e < left.complex.nE; ++e) {
      double sum = 0;
      for (int pq = 0; pq < 9; ++pq) sum += pe.single[static_cast<std::size_t>(e * 27 + 1 + 3 * pq)];
      err = std::max(err, std::abs(sum - we.single[static_cast<std::size_t>(e * 3 + 1)])); scale = std::max(scale, std::abs(static_cast<double>(we.single[static_cast<std::size_t>(e * 3 + 1)])));
    }
    std::printf("  band pairs vs whole field: %.2e\n", err / scale);
    CHECK(err < 1e-4 * scale);
    const double outside = flowB.at("whole_edges").meta.at("describe").at("attributes").at("source").at("kernel_outside_modes_energy_share").get<double>();
    std::printf("  the kernel's energy outside the kept modes (in the complement band): %.3f\n", outside);
    CHECK(outside > 0 && outside < 1);
  }

  // ── time cells and the sign-flip test ──
  {
    const ng::Arrays cells = run_array("signal.cortical_time_cells", ng::Arrays{{"spectra", spectraB}, {"package", left.package}, {"operator", optionB.at("operator")}, {"kernel_columns", optionB.at("kernel_columns")},
                                                                            {"frequency", kernel_set(atoms)}, {"scale", kernel_set(bands)}},
                                       Json{{"frequency_weighting", "kernels"}, {"frequency_normalisation", "sum_of_squared_gains"}, {"support_floor", 1e-3}, {"cell_width", 2.0}});
    const ng::Array& tc = cells.at("cells");
    const Json groups = tc.meta.at("describe").at("attributes").at("groups");
    CHECK(groups.size() == 12 && tc.cols == left.complex.nE + 4 * nV);
    const ng::Array fc = run_array("stats.sign_flip_consistency", ng::Arrays{{"cells", tc}}, Json{{"elements", "edges"}, {"patterns", 200}, {"seed", 0}, {"cells_per_flip", 1}}).at("consistency");
    const std::int64_t E = left.complex.nE;
    const double median = fc.data[static_cast<std::size_t>(1 * fc.cols + 3 * E + 1)], medianP = fc.data[static_cast<std::size_t>(1 * fc.cols + 3 * E + 3)];
    std::printf("  10 Hz, whole field: median R1 %.3f, its p %.4f\n", median, medianP);
    CHECK(median > 0.8 && medianP < 0.05);
  }

  // ── refusals ──
  CHECK(refused([&] { flow_of(optionB, spectraA); }));                                        // the spectra are not the operator's columns
  return nxrtest::finish("test_source_flow_nodes");
}
