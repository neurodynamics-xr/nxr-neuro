// nxr-pipelines — THE SOURCE MAPPING OF THE SENSOR-SPACE FLOW as graph nodes (spec docs/specs/2026-11-24-source-flow.md; definitions `source_flow`, `source_flow_bins`).
// The source step is an existing minimum-norm kernel projected onto the cortex's eigenmodes, nothing else: no inverse is derived here, no prior over modes is added.
//
// THE GRAPH. before: cortex eigenvectors + kernel --[signal.mode_analysis, exact_and_constant, unit_mass]--> the kernel in the cortex's mode coordinates [3 K_c x channels] (each
//         Cartesian component, no orientation assumed) --[signal.source_mode_operator]--> option A (columns: the sensor surface's field coordinates, composed with their synthesis
//         back to the channels) or option B (columns: the channels), the kernel's rows selected onto the cortex in the same columns (for the complement band), and the serving
//         operator of the chunk spectra in those columns (option A: the sensor surface's; option B: the identity on the kernel's channels).
// chunks: the sensor chain's signal.overlapped_spectrum and signal.chunk_spectra, unchanged.
// after:  chunk spectra + cortex eigenvectors + the operator + the frequency weighting + the cortical eigenvalue bands --[signal.cortical_flow]--> per frequency member the whole
//         field's and every band pair's power, phase current per edge and per vertex, flow coefficients on the cortex's stored edge rows, potential and stream, and the remainder of
//         the flow beyond the kept modes split by sparse Poisson solves; --[signal.cortical_time_cells]--> the same sums per time cell; the sensor chain's
//         stats.sign_flip_consistency, unchanged.
// Every node is a function of arrays: no file, no subject, no loop over a recording.
#include "nxr/pipelines/source_flow_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "chunk_spectra_readers.h"
#include "nxr/graph/node.h"
#include "nxr/pipelines/form_steps.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/flow_readout.h"
#include "nxr/signal/source_flow.h"
#include "nxr/signal/spectral_flow.h"
#include "source_flow_readers.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
namespace readers = spectra_readers;

KernelSelection kernel_rows_for_package(const ng::Array& kernel, const ng::Array& package, const char* who) {
  const std::string name = kernel.meta.value("name", std::string());
  const std::vector<std::int32_t> channels = kernel.meta.value("channels", std::vector<std::int32_t>());
  if (kernel.meta.value("to_value_type", std::string("vector3")) == "scalar" || kernel.rows % 3 != 0) ng::refuse(std::string(who) + ": " + name + " is not a free-orientation kernel (x y z rows per vertex)");
  if (static_cast<std::int64_t>(channels.size()) != kernel.cols) ng::refuse(std::string(who) + ": " + name + "'s columns are not its channels");
  const std::int64_t kernelVertices = kernel.rows / 3;
  const std::int64_t inputVertices = static_cast<std::int64_t>(package.meta.at("input_positions").size() / 3);
  const std::vector<std::int64_t> keep = package.meta.at("keep").get<std::vector<std::int64_t>>();
  KernelSelection s;
  std::int64_t first = 0;
  if (kernelVertices == inputVertices) s.description = "the package's input surface";
  else if (package.meta.contains("input_from_index") && kernel.meta.value("to_manifold_id", std::string()) == package.meta.value("input_component_of", std::string()) &&
           package.meta["input_from_index"].get<std::int64_t>() + inputVertices <= kernelVertices) {
    first = package.meta["input_from_index"].get<std::int64_t>();
    s.description = "a component of the kernel's surface (a hemisphere): its vertices [" + std::to_string(first) + ", " + std::to_string(first + inputVertices) + ")";
  } else
    ng::refuse(std::string(who) + ": " + name + " (" + std::to_string(kernelVertices) + " vertices) is on neither the package's input surface (" + std::to_string(inputVertices) +
               " vertices) nor the surface that surface is a component of");
  for (const std::int64_t v : keep) {
    if (v >= inputVertices) ng::refuse(std::string(who) + ": the package keeps a vertex outside its input surface");
    s.source_vertex.push_back(v < 0 ? -1 : first + v);
    s.bridge_vertices += v < 0;
  }
  if (s.bridge_vertices > 0) s.description += "; the glued surface: the kept vertices' rows selected, the " + std::to_string(s.bridge_vertices) + " bridge vertices' rows zero";
  return s;
}

namespace {

const ng::Array& constant_input(const ng::Arrays& c, const char* port, const char* who) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(who) + ": " + port + " was not handed over");
  return it->second;
}
bool handed(const ng::Arrays& c, const char* port) { const auto it = c.find(port); return it != c.end() && !it->second.meta.value("placeholder", false); }
std::string text_param(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_string()) ng::refuse(std::string(who) + ": " + key + " has no default: name it");
  return p[key].get<std::string>();
}
Eigen::MatrixXd matrix_of(const ng::Array& a) { const std::vector<double> v = a.values(); return Eigen::Map<const RowMajor>(v.data(), a.rows, a.cols); }
ng::Array array_of(const Eigen::MatrixXd& m) {
  ng::Array a;
  a.rows = m.rows(); a.cols = m.cols();
  a.data.resize(static_cast<std::size_t>(a.rows * a.cols));
  Eigen::Map<RowMajor>(a.data.data(), a.rows, a.cols) = m;
  return a;
}
Json indexed_axis(const char* name, std::int64_t size, const Json& values = Json()) {
  Json v = values;
  if (v.is_null()) { v = Json::array(); for (std::int64_t i = 0; i < size; ++i) v.push_back(i); }
  return Json{{"name", name}, {"size", size}, {"values", v}};
}
ng::PortType operator_port() { ng::PortType t; t.cls = ng::PortClass::operator_; t.facts["cls"] = "operator"; return t; }
ng::PortType statistic_port() { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "statistic"; return t; }
// the numerical rank of a matrix with few columns: the eigenvalues of Mᵀ M above 1e-20 of the largest (singular values above 1e-10 of the largest)
std::int64_t rank_of(const Eigen::MatrixXd& m) {
  const Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(m.transpose() * m, Eigen::EigenvaluesOnly).eigenvalues();
  const double top = ev.cwiseAbs().maxCoeff();
  std::int64_t r = 0;
  for (Eigen::Index i = 0; i < ev.size(); ++i) r += ev(i) > 1e-20 * top;
  return r;
}

// ── signal.source_mode_operator ──────────────────────────────────────────────────────────────────────────────────
const char* const kOperator = "signal.source_mode_operator";
struct OperatorPrepared { ng::Array cortex, kernel, package, sensor_operator, sensor_package; bool option_a = false; };

ng::NodeSpec source_mode_operator() {
  ng::NodeSpec n;
  n.name = kOperator;
  n.kind = ng::Kind::array;
  n.inputs = {{"cortex_operator", "the kernel in the cortex's mode coordinates (signal.mode_analysis with the kernel: [3 K_c x the kernel's channels], component-major)", false},
              {"kernel", "the same free-orientation kernel as stored (its rows are selected onto the cortex for the complement band)", false},
              {"package", "the cortex's Hodge-Dirac package the cortex operator was made on (a hemisphere, or the glued cortex)", false},
              {"sensor_operator", "option A: the sensor chain's serving operator (signal.mode_analysis with the channel map: the sensor surface's field coordinates from its vertices)", true},
              {"sensor_package", "option A: the sensor surface's Hodge-Dirac package (its vertex mass: the coordinates' synthesis back to the channels)", true}};
  n.outputs = {{"operator", "the cortex mode operator [3 K_c x columns]: option A (columns the sensor field coordinates) = the cortex operator times the coordinates' synthesis back to the channels; option B (columns the channels) = the cortex operator", false},
               {"kernel_columns", "the kernel's rows selected onto the cortex package's vertices, in the same columns [3 V x columns] (vertex-major): the readout of the whole field and of the complement band", false},
               {"serving", "the operator from the recording's channels to the columns, for signal.overlapped_spectrum: option A the sensor serving operator, option B the identity on the kernel's channels", false}};
  n.parameters = {{"columns", Json(), "sensor_coordinates (option A: the operator acts on the served sensor field coordinates) | channels (option B: it acts on the channels) (no default)", "-",
                   {"sensor_coordinates", "channels"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "made in double, once a run, from the operators handed over (a change of coordinates, composed once)");
  n.linear_map = true;
  n.matches = "the reference's band kernels k_b = Phi g_b Phiᵀ star0 k start from this operator (Phiᵀ star0 k); compute nxr::signal::synthesisFromAnalysis, selectVertexRows";
  n.description = "the minimum-norm kernel written in the cortex's eigenmode coordinates, acting on the sensor field coordinates (option A) or on the channels (option B); composed once";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"cortex_operator", "kernel", "package"}) if (!in.count(k)) ng::refuse(std::string(kOperator) + ": " + k + " is not given");
    const std::string columns = text_param(p, "columns", kOperator);
    if (columns != "sensor_coordinates" && columns != "channels") ng::refuse(std::string(kOperator) + ": columns is sensor_coordinates or channels");
    if (columns == "sensor_coordinates" && (!in.count("sensor_operator") || !in.count("sensor_package")))
      ng::refuse(std::string(kOperator) + ": columns sensor_coordinates (option A) needs sensor_operator and sensor_package");
    ng::Planned out;
    for (const char* o : {"operator", "kernel_columns", "serving"}) out.out[o] = operator_port();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    OperatorPrepared pr;
    pr.cortex = constant_input(c, "cortex_operator", kOperator);
    pr.kernel = constant_input(c, "kernel", kOperator);
    pr.package = constant_input(c, "package", kOperator);
    pr.option_a = p["columns"].get<std::string>() == "sensor_coordinates";
    if (pr.option_a) { pr.sensor_operator = constant_input(c, "sensor_operator", kOperator); pr.sensor_package = constant_input(c, "sensor_package", kOperator); }
    if (pr.cortex.meta.value("components", 1) != 3) ng::refuse(std::string(kOperator) + ": cortex_operator is not a kernel's mode operator (three components)");
    if (pr.cortex.meta.value("channels", std::vector<std::int32_t>()) != pr.kernel.meta.value("channels", std::vector<std::int32_t>()))
      ng::refuse(std::string(kOperator) + ": cortex_operator was not made from this kernel (their channels differ)");
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const OperatorPrepared& pr = std::any_cast<const OperatorPrepared&>(prepared);
    const Eigen::MatrixXd cortexOperator = matrix_of(pr.cortex);                                 // [3 K_c x kernel channels]
    const KernelSelection selection = kernel_rows_for_package(pr.kernel, pr.package, kOperator);
    const Eigen::MatrixXd kernelRows = nxr::signal::selectVertexRows(matrix_of(pr.kernel), selection.source_vertex, 3);   // [3 V x kernel channels]
    const std::vector<std::int32_t> kernelChannels = pr.kernel.meta.at("channels").get<std::vector<std::int32_t>>();
    Eigen::MatrixXd composed, kernelColumns;
    ng::Array serving;
    Json columns, difference;
    if (pr.option_a) {
      const ng::Array& S = pr.sensor_operator;
      if (S.meta.value("rows", Json::object()).value("normalisation", std::string()) != "unit_mass")
        ng::refuse(std::string(kOperator) + ": the sensor operator's coordinates are not at unit mass norm (signal.mode_analysis normalisation unit_mass): their synthesis is not the analysis rows' inverse");
      const std::vector<double> mass = pr.sensor_package.meta.at("mass").get<std::vector<double>>();
      if (static_cast<std::int64_t>(mass.size()) != S.cols) ng::refuse(std::string(kOperator) + ": the sensor package's vertices are not the sensor operator's columns");
      const std::vector<std::int32_t> sensorChannels = S.meta.at("channels").get<std::vector<std::int32_t>>();
      std::map<std::int32_t, Eigen::Index> kernelColumnOf;
      for (std::size_t i = 0; i < kernelChannels.size(); ++i) kernelColumnOf[kernelChannels[i]] = static_cast<Eigen::Index>(i);
      if (sensorChannels.size() != kernelChannels.size()) ng::refuse(std::string(kOperator) + ": the sensor surface has " + std::to_string(sensorChannels.size()) + " channels, the kernel " + std::to_string(kernelChannels.size()) + ": they must be the same channels");
      const Eigen::MatrixXd sensorAnalysis = matrix_of(S);                                    // [K_s x sensor vertices]
      const Eigen::MatrixXd synthesis = nxr::signal::synthesisFromAnalysis(sensorAnalysis, Eigen::Map<const Eigen::VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size())));   // [vertices x K_s]
      Eigen::MatrixXd synthesisByKernelColumn = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(kernelChannels.size()), synthesis.cols());
      Eigen::MatrixXd analysisByKernelColumn = Eigen::MatrixXd::Zero(synthesis.cols(), static_cast<Eigen::Index>(kernelChannels.size()));
      for (std::size_t v = 0; v < sensorChannels.size(); ++v) {
        const auto it = kernelColumnOf.find(sensorChannels[v]);
        if (it == kernelColumnOf.end()) ng::refuse(std::string(kOperator) + ": the sensor surface's channel " + std::to_string(sensorChannels[v]) + " is not one of the kernel's");
        synthesisByKernelColumn.row(it->second) = synthesis.row(static_cast<Eigen::Index>(v));
        analysisByKernelColumn.col(it->second) = sensorAnalysis.col(static_cast<Eigen::Index>(v));
      }
      // compose first: (I ⊗ Phiᵀ M) W was made at its smallest size; one product with the synthesis [channels x K_s]; the kernel's rows likewise (for the complement)
      composed = cortexOperator * synthesisByKernelColumn;
      kernelColumns = kernelRows * synthesisByKernelColumn;
      serving = S;
      columns = Json{{"are", "the sensor surface's field coordinates"}, {"members", S.meta.value("modes", Json::array())}, {"mu", S.meta.value("mu", Json::array())}};
      // what option A leaves out: the part of the field in the sensor coordinates the solve did not make, B (I - Q P), stated as its share of B's Frobenius norm
      const Eigen::MatrixXd leftOut = cortexOperator - composed * analysisByKernelColumn;
      difference = Json{{"left_out_frobenius_share", leftOut.norm() / std::max(1e-300, cortexOperator.norm())}, {"sensor_coordinates", synthesis.cols()}, {"channels", static_cast<std::int64_t>(kernelChannels.size())},
                        {"meaning", "option A equals option B on the projection of the channels onto the sensor coordinates; B (I - Q P) is what it leaves out (zero when the sensor basis is complete)"}};
    } else {
      composed = cortexOperator;
      kernelColumns = kernelRows;
      serving = array_of(Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(kernelChannels.size()), static_cast<Eigen::Index>(kernelChannels.size())));
      serving.meta["channels"] = kernelChannels;
      Json members = Json::array();
      for (std::size_t i = 0; i < kernelChannels.size(); ++i) members.push_back(static_cast<std::int64_t>(i));
      serving.meta["modes"] = members;
      serving.meta["mu"] = std::vector<double>(kernelChannels.size(), 0.0);
      serving.meta["rows"] = Json{{"order", "the kernel's channels in its column order (the identity: option B reads the channels)"}, {"normalisation", "as_stored"}};
      serving.meta["name"] = pr.kernel.meta.value("name", std::string()) + "_channels";
      columns = Json{{"are", "the kernel's channels"}, {"members", members}};
    }
    const std::int64_t rank = rank_of(composed);
    ng::Array op = array_of(composed);
    op.meta = pr.cortex.meta;
    op.meta["columns"] = columns;
    op.meta["option"] = pr.option_a ? "A" : "B";
    op.meta["rank"] = rank;
    op.meta["kernel_selection"] = selection.description;
    if (!difference.is_null()) op.meta["option_a_left_out"] = difference;
    op.meta["unit"] = "mode coefficient (the kernel's unit) per column unit";
    ng::Array kc = array_of(kernelColumns);
    kc.meta = Json{{"columns", columns}, {"rows", "vertex-major: 3 v + component (x y z) of the cortex package's vertices"}, {"kernel_selection", selection.description},
                   {"bridge_vertices", selection.bridge_vertices}, {"name", pr.kernel.meta.value("name", std::string())}};
    return ng::Arrays{{"operator", std::move(op)}, {"kernel_columns", std::move(kc)}, {"serving", std::move(serving)}};
  };
  n.record = [](const std::any& prepared) {
    const OperatorPrepared& pr = std::any_cast<const OperatorPrepared&>(prepared);
    return Json{{"columns", pr.option_a ? "sensor_coordinates" : "channels"}, {"kernel", pr.kernel.meta.value("name", std::string())}, {"package", pr.package.meta.value("name", std::string())}};
  };
  return n;
}

// ── what the cortical readers share ──────────────────────────────────────────────────────────────────────────────
struct CorticalPrepared {
  BoundPackage package;
  ng::Array spectra;
  Eigen::MatrixXd frequency;                 // [members x bins]
  std::vector<double> standardDeviations;    // the atoms' (time cells only)
  std::vector<Eigen::MatrixXd> readouts;     // the whole field's readout first, then each band's (component-major [3V x columns])
  Eigen::Index bands = 0;
  Json operatorMeta, p;
  double outsideModesShare = 0;              // the kernel's mass-weighted energy outside the kept cortical modes, as a share of its whole
};

CorticalPrepared prepare_cortical(const ng::Arrays& c, const Json& p, const char* who, bool needAtoms) {
  CorticalPrepared pr;
  pr.package = bind_hodge_package(constant_input(c, "package", who), who);
  pr.spectra = constant_input(c, "spectra", who);
  const readers::Spectra s = readers::spectra_of(pr.spectra, who);
  pr.frequency = readers::frequency_gains(s, c, p, who, needAtoms ? &pr.standardDeviations : nullptr);
  const ng::Array& O = constant_input(c, "operator", who);
  const ng::Array& KC = constant_input(c, "kernel_columns", who);
  pr.operatorMeta = O.meta;
  // type checks before any data: the spectra's coordinates are the operator's columns, the kernel's rows are the package's vertices
  const Json columns = O.meta.value("columns", Json::object());
  if (O.cols != s.rows || columns.value("members", Json::array()) != Json(s.modes))
    ng::refuse(std::string(who) + ": the operator's columns (" + columns.value("are", std::string("?")) + ", " + std::to_string(O.cols) + ") are not the spectra's coordinates (" + std::to_string(s.rows) + ")");
  const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
  if (KC.rows != 3 * cx.nV || KC.cols != O.cols) ng::refuse(std::string(who) + ": kernel_columns is not [3 V x columns] of the package's vertices");
  // the cortex's coordinates (the constant and the exact modes, as the operator's rows name them) at the vertices
  readers::Spectra rowsOfOperator;
  rowsOfOperator.modes = O.meta.at("modes").get<std::vector<std::int64_t>>();
  rowsOfOperator.unit_mass = O.meta.value("rows", Json::object()).value("normalisation", std::string("unit_mass")) != "as_stored";
  const auto [modeSynthesis, mu] = readers::synthesis_of(rowsOfOperator, *pr.package.modes, cx.star0, who);
  const Eigen::Index K = modeSynthesis.cols();
  if (O.rows != 3 * K) ng::refuse(std::string(who) + ": the operator's rows are not three components of its modes");
  const Eigen::MatrixXd modeOperator = matrix_of(O), kernelColumns = matrix_of(KC);
  const Eigen::MatrixXd whole = nxr::signal::componentMajor(kernelColumns, 3);
  // the content outside the kept modes (it lies in the complement band): its mass-weighted energy share
  double outside = 0, all = 0;
  for (int comp = 0; comp < 3; ++comp) {
    const Eigen::MatrixXd field = whole.middleRows(comp * cx.nV, cx.nV), residual = field - modeSynthesis * modeOperator.middleRows(comp * K, K);
    outside += (residual.cwiseAbs2().transpose() * cx.star0).sum(); all += (field.cwiseAbs2().transpose() * cx.star0).sum();
  }
  pr.outsideModesShare = all > 0 ? outside / all : 0;
  pr.readouts.push_back(whole);
  if (handed(c, "scale")) {
    Eigen::MatrixXd gains = readers::kernel_gains_of(c.at("scale"), mu, who, "scale").gains;
    for (Eigen::Index j = 0; j < mu.size(); ++j) if (!(mu(j) > 0) && gains.col(j).sum() == 0) gains(0, j) = 1.0;   // the constant (mu 0) lies in the coarsest band of a partition
    const std::vector<Eigen::MatrixXd> bands = nxr::signal::bandReadouts(modeOperator, modeSynthesis, gains, kernelColumns, 3);
    pr.bands = static_cast<Eigen::Index>(bands.size());
    pr.readouts.insert(pr.readouts.end(), bands.begin(), bands.end());
  }
  pr.p = p;
  return pr;
}

// ── signal.cortical_flow ─────────────────────────────────────────────────────────────────────────────────────────
const char* const kFlow = "signal.cortical_flow";
const char* const kVertexMeasures[] = {"power", "exact_potential", "coexact_stream", "vertex_phase_current_x", "vertex_phase_current_y", "vertex_phase_current_z", "remainder_potential", "remainder_stream"};
constexpr int kVertexMeasureCount = 8;

ng::NodeSpec cortical_flow() {
  ng::NodeSpec n;
  n.name = kFlow;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in the operator's columns (signal.chunk_spectra, served through signal.source_mode_operator's serving operator)", false},
              {"package", "the cortex's Hodge-Dirac package (a hemisphere, or the glued cortex), with its stored 1-form parts", false},
              {"operator", "the cortex mode operator (signal.source_mode_operator)", false},
              {"kernel_columns", "the kernel's rows on the cortex in the operator's columns (signal.source_mode_operator)", false},
              {"frequency", "the frequency weighting's kernel set (the atoms). Absent with frequency_weighting every_bin", true},
              {"scale", "the cortical eigenvalue bands, a kernel set on the cortex's eigenvalue line (an amplitude partition: cos2_partition; the last member is the complement). Absent: the whole field only", true}};
  n.outputs = {{"whole_vertices", "per frequency member, per vertex: power, exact_potential, coexact_stream, vertex_phase_current x y z, remainder_potential, remainder_stream (the whole field)", false},
               {"whole_edges", "per frequency member, per intrinsic edge of the cortex: the phase current (a 1-form, the three components summed)", false},
               {"whole_modes", "per frequency member: the phase current's coefficient on each Hodge-Dirac mode of the cortex (exact: travel; coexact: rotation), then its star1 energy and the remainder's", false},
               {"pair_vertices", "per frequency member x band x band (the member fastest): the eight vertex measures of the band pair", false},
               {"pair_edges", "per frequency member x band x band: the band pair's phase current per edge", false},
               {"pair_modes", "per frequency member x band x band: the band pair's flow coefficients, its star1 energy and the remainder's", false}};
  n.parameters = {
      {"frequency_weighting", Json(), "kernels (the atoms) | every_bin (each bin alone, weight 1) (no default)", "-", {"kernels", "every_bin"}, false},
      {"frequency_normalisation", Json(), "sum_of_squared_gains | none: as signal.spectral_flow's (no default with kernels)", "-", {"sum_of_squared_gains", "none"}, true},
      {"support_floor", Json(), "with sum_of_squared_gains: as signal.spectral_flow's (no default)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "once a run: per frequency member one weighted sum of the chunk spectra (in sensor coordinates), per band the readout's right factor, per pair only the entries read; float32 out for float32 spectra");
  n.matches = "the reference's flow identity per vertex P = sum over components of grad k Im(C) kᵀ and |w|^2 = sum k Re(C) kᵀ with the band kernels k_b = Phi g_b Phiᵀ star0 k (last band the complement); "
              "the split from the stored edge rows; compute nxr::signal::bandReadouts, pairFlow, flowSplit, OneFormRemainderSplit";
  n.description = "the cortical flow of a weighted cross-spectrum read through the minimum-norm kernel in eigenmode coordinates: per frequency member the whole field's and every cortical band pair's power, "
                  "phase current per edge and per vertex, flow coefficients, potential and stream, and the remainder beyond the kept modes";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "package", "operator", "kernel_columns"}) if (!in.count(k)) ng::refuse(std::string(kFlow) + ": " + k + " is not given");
    const std::string w = text_param(p, "frequency_weighting", kFlow);
    if (w == "kernels" && !in.count("frequency")) ng::refuse(std::string(kFlow) + ": frequency_weighting kernels needs the frequency kernel set");
    if (w == "kernels") text_param(p, "frequency_normalisation", kFlow);
    ng::Planned out;
    for (const char* o : {"whole_vertices", "whole_edges", "whole_modes", "pair_vertices", "pair_edges", "pair_modes"}) out.out[o] = statistic_port();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_cortical(c, p, kFlow, false)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const CorticalPrepared& pr = std::any_cast<const CorticalPrepared&>(prepared);
    const readers::Spectra s = readers::spectra_of(pr.spectra, kFlow);
    const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
    const nxr::signal::Modes& m = *pr.package.modes;
    const Eigen::SparseMatrix<double> sharpWeights = nxr::signal::sharpOperator(cx);
    const nxr::signal::OneFormRemainderSplit splitter(cx);
    const Eigen::Index V = cx.nV, E = cx.nE, M = m.mu.size(), A = pr.frequency.rows(), Q = pr.bands, pairs = Q * Q;
    auto measured = [](Eigen::Index rows, Eigen::Index cells, Eigen::Index measures) { ng::Array a; a.rows = rows * cells; a.cols = measures; a.single.assign(static_cast<std::size_t>(rows * cells * measures), 0.0f); return a; };
    ng::Array wv = measured(V, A, kVertexMeasureCount), we = measured(E, A, 1), wm = measured(M + 2, A, 1);
    ng::Array pv = measured(V, A * pairs, kVertexMeasureCount), pe = measured(E, A * pairs, 1), pm = measured(M + 2, A * pairs, 1);
    double remainderEnergy = 0, flowEnergy = 0;
    // the readings of one frequency member's columns (whole field: one column; pairs: Q² columns), stored at their cells
    auto store = [&](ng::Array& vertices, ng::Array& edges, ng::Array& modes, Eigen::Index cells, const std::vector<Eigen::Index>& cellOf, const nxr::signal::PairFlow& f) {
      const nxr::signal::FlowSplit split = nxr::signal::flowSplit(cx, m, f.pairImaginary, &splitter);        // the stored edge rows, then the remainder beyond the kept modes
      for (Eigen::Index j = 0; j < f.pairImaginary.cols(); ++j) {
        const Eigen::Index cell = cellOf[static_cast<std::size_t>(j)];
        flowEnergy += split.flowEnergy(j); remainderEnergy += split.remainderEnergy(j);
        for (Eigen::Index v = 0; v < V; ++v) {
          float* r = vertices.single.data() + (v * cells + cell) * kVertexMeasureCount;
          r[0] = static_cast<float>(f.power(v, j)); r[1] = static_cast<float>(split.potential(v, j)); r[2] = static_cast<float>(split.stream(v, j));
          for (int comp = 0; comp < 3; ++comp) r[3 + comp] = static_cast<float>(f.vertexCurrent(3 * v + comp, j));
          r[6] = static_cast<float>(split.remainderPotential(v, j)); r[7] = static_cast<float>(split.remainderStream(v, j));
        }
        for (Eigen::Index e = 0; e < E; ++e) edges.single[static_cast<std::size_t>(e * cells + cell)] = static_cast<float>(f.pairImaginary(e, j));
        for (Eigen::Index k = 0; k < M; ++k) modes.single[static_cast<std::size_t>(k * cells + cell)] = static_cast<float>(split.coefficients(k, j));
        modes.single[static_cast<std::size_t>(M * cells + cell)] = static_cast<float>(split.flowEnergy(j));
        modes.single[static_cast<std::size_t>((M + 1) * cells + cell)] = static_cast<float>(split.remainderEnergy(j));
      }
    };
    const std::vector<Eigen::MatrixXd> whole(pr.readouts.begin(), pr.readouts.begin() + 1), bands(pr.readouts.begin() + 1, pr.readouts.end());
    Json members = Json::array();
    for (Eigen::Index a = 0; a < A; ++a) {
      const Eigen::MatrixXcd Y = readers::weighted_columns(s, pr.frequency.row(a));
      members.push_back(Json{{"member", a}, {"columns", Y.cols()}, {"order", 2 * Y.cols() < Y.rows() ? "factored" : "dense"}});
      if (Y.cols() == 0) continue;
      store(wv, we, wm, A, {a}, nxr::signal::pairFlow(Y, whole, 3, cx.edges, &sharpWeights));
      if (Q == 0) continue;
      std::vector<Eigen::Index> cells;
      for (Eigen::Index pq = 0; pq < pairs; ++pq) cells.push_back(a + A * pq);
      store(pv, pe, pm, A * pairs, cells, nxr::signal::pairFlow(Y, bands, 3, cx.edges, &sharpWeights));
    }
    Json vertexMeasures = Json::array();
    for (const char* name : kVertexMeasures) vertexMeasures.push_back(Json{{"measure", name}});
    Json modeMeasures = Json::array({Json{{"measure", "flow_coefficient"}}});
    const Json attrs = Json{{"members", "the frequency weighting's members"}, {"frequency_members", members}, {"boundary", pr.package.boundary},
                            {"frequency_weighting", pr.p.value("frequency_weighting", Json())}, {"frequency_normalisation", pr.p.value("frequency_normalisation", Json())},
                            {"sum_over", "the chunks and the member's bins (each sample counted twice with an overlap of half a chunk)"},
                            {"source", Json{{"option", pr.operatorMeta.value("option", Json())}, {"columns", pr.operatorMeta.value("columns", Json::object()).value("are", Json())},
                                            {"rank", pr.operatorMeta.value("rank", Json())}, {"kernel_selection", pr.operatorMeta.value("kernel_selection", Json())},
                                            {"option_a_left_out", pr.operatorMeta.value("option_a_left_out", Json())},
                                            {"kernel_outside_modes_energy_share", pr.outsideModesShare}, {"cortical_modes", pr.operatorMeta.value("modes", Json::array()).size()}}},
                            {"bands", Q}, {"complement", "the last band is the kernel minus every other band: its own gain and the kernel's content outside the kept modes"},
                            {"components", "the three Cartesian components of the current, each read as a scalar field, summed (no orientation assumed)"},
                            {"split", "the flow coefficients on the cortex's stored Hodge-Dirac edge rows, potential sum (coefficient / mu) x mode, stream likewise; the remainder beyond the kept modes split by sparse Poisson solves (Neumann potential, Dirichlet stream on an open surface)"},
                            {"remainder_energy_share", flowEnergy > 0 ? remainderEnergy / flowEnergy : 0.0},
                            {"flow_modes", M}, {"sign", "phase current J = Im(conj(z_tail) z_tip): travel from tail to tip gives J < 0"}};
    auto describe = [&](ng::Array& a, const Json& measures, Eigen::Index rows, Eigen::Index cells, const char* rowsAre, const char* membersAre) {
      Json at = attrs; at["rows"] = rowsAre; if (membersAre) at["members"] = membersAre;
      Json names = Json::array();
      for (const auto& x : measures) names.push_back(x["measure"]);
      a.meta["describe"] = Json{{"method", "cortical_flow"}, {"measure", "flow"}, {"shape", Json::array({rows, cells, measures.size()})}, {"precision", "float32"},
                                {"axes", Json::array({indexed_axis("element", rows), indexed_axis("member", cells), indexed_axis("measure", static_cast<std::int64_t>(measures.size()), names)})},
                                {"measures", measures}, {"values_shape", Json::array({rows, cells})}, {"attributes", at}};
    };
    const char* pairsAre = "frequency member x band p x band q, the frequency member fastest (the phase current of the pair: conj(z_p) at the tail, z_q at the tip)";
    const char* modeRows = "the cortex's Hodge-Dirac modes (exact: travel; coexact: rotation), then the phase current's star1 energy, then the remainder's (beyond the kept modes)";
    describe(wv, vertexMeasures, V, A, "the vertices of the cortex package", nullptr);
    describe(we, Json::array({Json{{"measure", "phase_current"}}}), E, A, "the edges of the cortex's intrinsic complex (tail -> tip)", nullptr);
    describe(wm, modeMeasures, M + 2, A, modeRows, nullptr);
    describe(pv, vertexMeasures, V, A * pairs, "the vertices of the cortex package", pairsAre);
    describe(pe, Json::array({Json{{"measure", "phase_current"}}}), E, A * pairs, "the edges of the cortex's intrinsic complex (tail -> tip)", pairsAre);
    describe(pm, modeMeasures, M + 2, A * pairs, modeRows, pairsAre);
    return ng::Arrays{{"whole_vertices", std::move(wv)}, {"whole_edges", std::move(we)}, {"whole_modes", std::move(wm)},
                      {"pair_vertices", std::move(pv)}, {"pair_edges", std::move(pe)}, {"pair_modes", std::move(pm)}};
  };
  n.record = [](const std::any& prepared) {
    const CorticalPrepared& pr = std::any_cast<const CorticalPrepared&>(prepared);
    return Json{{"boundary", pr.package.boundary}, {"frequency_members", pr.frequency.rows()}, {"bands", pr.bands}, {"option", pr.operatorMeta.value("option", Json())},
                {"kernel_outside_modes_energy_share", pr.outsideModesShare}};
  };
  return n;
}

// ── signal.cortical_time_cells ───────────────────────────────────────────────────────────────────────────────────
const char* const kCells = "signal.cortical_time_cells";
ng::NodeSpec cortical_time_cells() {
  ng::NodeSpec n;
  n.name = kCells;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in the operator's columns (signal.chunk_spectra)", false},
              {"package", "the cortex's Hodge-Dirac package", false},
              {"operator", "the cortex mode operator (signal.source_mode_operator)", false},
              {"kernel_columns", "the kernel's rows on the cortex in the operator's columns (signal.source_mode_operator)", false},
              {"frequency", "the atoms: a kernel set of Gaussians on the spectra's frequency line (their time width sets the cells)", false},
              {"scale", "the cortical eigenvalue bands; each band's field is read on its own, after the whole field. Absent: the whole field only", true}};
  n.outputs = {{"cells", "per atom x readout (the whole field, then each band; the atom fastest), per time cell of the atom's own width: the edge phase currents, the vertex phase currents (x y z), the power per vertex (sums over the cell, the components summed)", false}};
  n.parameters = {
      {"frequency_weighting", Json("kernels"), "kernels (the only weighting with atoms: the time cells are the atoms')", "-", {"kernels"}, false},
      {"frequency_normalisation", Json(), "sum_of_squared_gains | none: as signal.spectral_flow's (no default)", "-", {"sum_of_squared_gains", "none"}, false},
      {"support_floor", Json(), "with sum_of_squared_gains: as signal.spectral_flow's (no default)", "-", {}, true},
      {"cell_width", Json(), "a time cell's width in the atom's time standard deviations (1 / (2 pi sigma_f)), snapped to the nearest power of two of samples (no default)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "once a run, chunk by chunk of the stored spectra: the atom's spectrum back in time on the columns' side (one inverse transform of the columns), then the readout per cell and its sums");
  n.matches = "signal.atom_time_cells with the cortical readout in place of the sensor synthesis; compute nxr::signal::timeCellPairFlow, takeWholeCells";
  n.description = "the cortical flow of each atom in the atom's own time cells, for the whole field and each band: per cell the sums of the edge and vertex phase currents and of the power";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "package", "operator", "kernel_columns", "frequency"}) if (!in.count(k)) ng::refuse(std::string(kCells) + ": " + k + " is not given");
    text_param(p, "frequency_normalisation", kCells);
    if (!p.contains("cell_width") || !p["cell_width"].is_number() || !(p["cell_width"].get<double>() > 0)) ng::refuse(std::string(kCells) + ": cell_width has no default: name it (> 0)");
    ng::Planned out;
    out.out["cells"] = statistic_port();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_cortical(c, p, kCells, true)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const CorticalPrepared& pr = std::any_cast<const CorticalPrepared&>(prepared);
    const readers::Spectra s = readers::spectra_of(pr.spectra, kCells);
    const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
    const Eigen::SparseMatrix<double> sharpWeights = nxr::signal::sharpOperator(cx);
    const Eigen::Index V = cx.nV, E = cx.nE, A = pr.frequency.rows(), R = static_cast<Eigen::Index>(pr.readouts.size());
    if (static_cast<std::int64_t>(s.owns.size()) != s.chunks || static_cast<std::int64_t>(s.firsts.size()) != s.chunks) ng::refuse(std::string(kCells) + ": the spectra do not record each chunk's samples");
    const double width = pr.p["cell_width"].get<double>();
    ng::Array out;
    out.cols = E + 4 * V;
    Json groups = Json::array();
    std::int64_t row = 0;
    for (Eigen::Index r = 0; r < R; ++r)
      for (Eigen::Index a = 0; a < A; ++a) {
        const double sigmaT = 1.0 / (2.0 * M_PI * pr.standardDeviations[static_cast<std::size_t>(a)]);
        const std::int64_t cell = std::int64_t(1) << static_cast<int>(std::lround(std::log2(width * sigmaT * s.sfreq)));
        std::int64_t cells = 0;
        nxr::signal::TimeCellPairFlow carried;                                               // a cell cut by a chunk's end, completed from the next chunk
        for (std::int64_t t = 0; t < s.chunks; ++t) {
          Eigen::MatrixXcd X(s.rows, s.bins);
          for (std::int64_t k = 0; k < s.bins; ++k) for (std::int64_t c = 0; c < s.rows; ++c) X(c, k) = pr.frequency(a, k) * s.at(t, k, c);
          const nxr::signal::TimeCellPairFlow part = nxr::signal::timeCellPairFlow(X, s.first, s.transform, s.overlap, s.owns[static_cast<std::size_t>(t)], s.firsts[static_cast<std::size_t>(t)],
                                                                                   cell, pr.readouts[static_cast<std::size_t>(r)], 3, cx.edges, &sharpWeights);
          const nxr::signal::TimeCellPairFlow f = nxr::signal::takeWholeCells(carried, part, cell);
          for (Eigen::Index j = 0; j < f.power.cols(); ++j) {
            for (Eigen::Index e = 0; e < E; ++e) out.single.push_back(static_cast<float>(f.pairImaginary(e, j)));
            for (Eigen::Index i = 0; i < 3 * V; ++i) out.single.push_back(static_cast<float>(f.vertexCurrent(i, j)));
            for (Eigen::Index v = 0; v < V; ++v) out.single.push_back(static_cast<float>(f.power(v, j)));
          }
          cells += f.power.cols();
        }
        groups.push_back(Json{{"atom", a}, {"band", r - 1}, {"first_row", row}, {"cells", cells}, {"cell_samples", cell}, {"cell_seconds", static_cast<double>(cell) / s.sfreq}});
        row += cells;
      }
    out.rows = row;
    out.meta["describe"] = Json{{"method", "cortical_time_cells"}, {"measure", "time_cell_flow"}, {"shape", Json::array({row, E + 4 * V})}, {"axes", Json::array({indexed_axis("cell", row), indexed_axis("column", E + 4 * V)})},
                                {"measures", Json::array({Json{{"measure", "phase_current"}, {"columns", Json::array({0, E})}}, Json{{"measure", "vertex_phase_current"}, {"columns", Json::array({E, E + 3 * V})}},
                                                         Json{{"measure", "power"}, {"columns", Json::array({E + 3 * V, E + 4 * V})}}})},
                                {"values_shape", Json::array({row, E + 4 * V})}, {"precision", "float32"},
                                {"attributes", Json{{"groups", groups}, {"rows", "the time cells of each atom x readout (band -1: the whole field), in the groups' order; whole cells anchored at the recording's first sample, a cell cut by a chunk's edge completed from the next chunk, a part cell at the recording's ends left out"},
                                                    {"columns", "the edges' phase currents, then each vertex's x y z phase current, then each vertex's power (the three current components summed)"},
                                                    {"cell_width", width}, {"sum_over", "the cell's samples (each sample once)"}}}};
    return ng::Arrays{{"cells", std::move(out)}};
  };
  return n;
}

}  // namespace

void register_source_flow_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {source_mode_operator(), cortical_flow(), cortical_time_cells()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
