// nxr-pipelines — the mode analysis as a graph node (signal.mode_analysis): a surface's modes, composed with a free kernel or the sensor surface's
// vertex-to-channel map, the operator that serves a recording in those coordinates (the flow definitions).
#include "nxr/pipelines/spectral_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/mode_spectrum.h"
#include "nxr/signal/source_flow.h"
#include "source_flow_readers.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

ng::PortType constant_of(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}
std::string text_param(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_string()) ng::refuse(std::string(who) + ": " + key + " has no default: name it");
  return p[key].get<std::string>();
}

// ── signal.mode_analysis ────────────────────────────────────────────────────────────────────────────────────────
const char* const kAnalysis = "signal.mode_analysis";
struct AnalysisPrepared { ng::Array package, kernel, channel_map; bool has_kernel = false, has_map = false; Json p; };

nxr::operators::Basis basis_from(const ng::Array& H) {
  nxr::operators::Basis b;
  const std::int64_t K = H.rows, V = H.cols;
  const std::vector<double> P = H.values();
  b.P = Eigen::Map<const RowMajor>(P.data(), K, V);
  const std::vector<double> mu = H.meta.at("mu").get<std::vector<double>>();
  b.mu = Eigen::Map<const Eigen::VectorXd>(mu.data(), K);
  b.kind = H.meta.at("kind").get<std::vector<int>>();
  const std::vector<double> mass = H.meta.at("mass").get<std::vector<double>>();
  b.mass = Eigen::Map<const Eigen::VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size()));
  const std::vector<double> pos = H.meta.at("positions").get<std::vector<double>>();
  b.positions = Eigen::Map<const nxr::operators::RowMat3>(pos.data(), V, 3);
  const std::vector<std::int32_t> faces = H.meta.at("faces").get<std::vector<std::int32_t>>();
  b.faces = Eigen::Map<const nxr::operators::FaceMat>(faces.data(), static_cast<Eigen::Index>(faces.size() / 3), 3);
  b.keep = H.meta.at("keep").get<std::vector<std::int64_t>>();
  return b;
}

ng::NodeSpec mode_analysis() {
  ng::NodeSpec n;
  n.name = kAnalysis;
  n.kind = ng::Kind::array;
  n.inputs = {{"package", "a Hodge-Dirac package (its eigenvectors, as the runner hands them over): the modes the coefficients are on", false},
              {"kernel", "a free-orientation inverse kernel (channels -> vertices, x y z) on the package's input surface or on the cortex a hemisphere's package is a part of (its rows selected): the cortex. Exactly one of kernel, channel_map", true},
              {"channel_map", "the sensor surface's vertex-to-channel map (`<surface>_map`): the helmet, whose vertices are channels. Exactly one of kernel, channel_map", true}};
  n.outputs = {{"operator", "the composed spatial operator, rows x the channels it reads: the modes' analysis rows (the selected modes' 0-form parts, normalised as stated, times the vertex mass), "
                            "composed with the kernel carried onto the package's surface (one block of rows per Cartesian component) or with the vertex-to-channel map", false}};
  n.parameters = {
      {"modes", Json(), "which modes: exact | coexact | all | exact_and_constant (the constant, index -1 and mu 0, then the exact modes: a field's coordinates on the surface) (no default: the caller names it; the nxr flow scripts expand on the exact modes)", "-", {"exact", "coexact", "all", "exact_and_constant"}, false},
      {"normalisation", Json(), "unit_mass (each mode rescaled to unit vertex-mass norm: rows x synthesis = I, the nxr flow scripts) | as_stored (the package's parts as stored) (no default)", "-", {"unit_mass", "as_stored"}, false},
      {"mode_count", Json(), "the first modes of the package read; null: all", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "made in double, once a run, from the package and the kernel");
  n.matches = "the nxr flow scripts' expansion of each Cartesian current component on the exact modes; compute nxr::signal::modeAnalysis, modeAnalysisOfComponents, selectVertexRows (the kernel's rows selected onto the package's surface: a hemisphere's range, or the glued surface with the bridge zero)";
  n.description = "the composed spatial operator of the mode-space coefficients: the selected modes' analysis rows composed with a free kernel (each component) or with the sensor surface's channel map";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("package")) ng::refuse(std::string(kAnalysis) + ": package is not given");
    if (in.count("kernel") == in.count("channel_map")) ng::refuse(std::string(kAnalysis) + ": exactly one of kernel (the cortex) and channel_map (the sensor surface)");
    text_param(p, "modes", kAnalysis);
    text_param(p, "normalisation", kAnalysis);
    ng::Planned out;
    ng::PortType t;
    t.cls = ng::PortClass::operator_;
    t.facts["cls"] = "operator";
    // its TARGET is the package's mode axis (the eigenvectors' source manifold): the rows it gives are modes of that stored axis (a subset: the coefficients name the members)
    if (const auto pk = in.find("package"); pk != in.end() && !pk->second.domain.factors.empty()) { t.typed = true; t.target = pk->second.domain; }
    out.out["operator"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    AnalysisPrepared a;
    if (!c.count("package") || c.at("package").meta.value("placeholder", false)) ng::refuse(std::string(kAnalysis) + ": package was not handed over");
    a.package = c.at("package");
    if (!a.package.meta.value("hodge", false)) ng::refuse(std::string(kAnalysis) + ": package is not a Hodge-Dirac package (an operator of kind eigenvectors, <stem>_eigenvectors)");
    if (c.count("kernel") && !c.at("kernel").meta.value("placeholder", false)) { a.kernel = c.at("kernel"); a.has_kernel = true; }
    if (c.count("channel_map") && !c.at("channel_map").meta.value("placeholder", false)) { a.channel_map = c.at("channel_map"); a.has_map = true; }
    if (a.has_kernel == a.has_map) ng::refuse(std::string(kAnalysis) + ": exactly one of kernel and channel_map was handed over");
    a.p = p;
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const AnalysisPrepared& a = std::any_cast<const AnalysisPrepared&>(prepared);
    const std::string modes = a.p["modes"].get<std::string>(), norm = a.p["normalisation"].get<std::string>();
    nxr::operators::Basis basis = basis_from(a.package);
    std::int64_t K = basis.P.rows();
    if (a.p.contains("mode_count") && !a.p["mode_count"].is_null()) K = std::min<std::int64_t>(K, a.p["mode_count"].get<std::int64_t>());
    const std::vector<int> kind(basis.kind.begin(), basis.kind.begin() + K);
    const nxr::signal::ModeAnalysis ma = nxr::signal::modeAnalysis(basis.P.topRows(K).transpose(), basis.mass, kind,
                                                                   modes == "exact" ? nxr::signal::ModeKinds::exact : modes == "coexact" ? nxr::signal::ModeKinds::coexact
                                                                   : modes == "exact_and_constant" ? nxr::signal::ModeKinds::exact_and_constant : nxr::signal::ModeKinds::all,
                                                                   norm == "unit_mass" ? nxr::signal::ModeNormalisation::unit_mass : nxr::signal::ModeNormalisation::as_stored);
    Eigen::MatrixXd O;
    std::vector<std::int32_t> channels;
    int components = 1;
    if (a.has_kernel) {
      const ng::Array& KA = a.kernel;
      channels = KA.meta.value("channels", std::vector<std::int32_t>());
      // the kernel's rows SELECTED onto the package's surface (the glue is a selection): the input surface's own rows, a hemisphere's vertex range of the cortex, or the glued
      // surface's kept vertices with the bridge rows zero; never a blend
      const KernelSelection selection = kernel_rows_for_package(KA, a.package, kAnalysis);
      const std::vector<double> kv = KA.values();
      const Eigen::MatrixXd W = Eigen::Map<const RowMajor>(kv.data(), KA.rows, KA.cols);
      O = nxr::signal::modeAnalysisOfComponents(ma.rows, nxr::signal::selectVertexRows(W, selection.source_vertex, 3), 3);
      components = 3;
    } else {
      const std::vector<double> mv = a.channel_map.values();
      const std::int64_t V = basis.P.cols();
      if (static_cast<std::int64_t>(mv.size()) != V) ng::refuse(std::string(kAnalysis) + ": the channel map has " + std::to_string(mv.size()) + " vertices, the package's surface " + std::to_string(V));
      for (std::int64_t v = 0; v < V; ++v) {
        if (basis.keep[static_cast<std::size_t>(v)] != v) ng::refuse(std::string(kAnalysis) + ": the package's surface is not the channel map's (a glued or re-ordered surface)");
        if (!(mv[static_cast<std::size_t>(v)] >= 0)) ng::refuse(std::string(kAnalysis) + ": vertex " + std::to_string(v) + " of the channel map has no channel");
        channels.push_back(static_cast<std::int32_t>(mv[static_cast<std::size_t>(v)]));
      }
      O = ma.rows;
    }
    ng::Array out;
    out.rows = O.rows(); out.cols = O.cols();
    out.data.resize(static_cast<std::size_t>(out.rows * out.cols));
    Eigen::Map<RowMajor>(out.data.data(), out.rows, out.cols) = O;
    out.meta["channels"] = channels;
    out.meta["components"] = components;
    out.meta["modes"] = ma.modes;
    std::vector<double> mu;
    for (const auto k : ma.modes) mu.push_back(k < 0 ? 0.0 : basis.mu(k));
    out.meta["mu"] = mu;
    out.meta["name"] = a.package.meta.value("name", std::string()) + (a.has_kernel ? "+" + a.kernel.meta.value("name", std::string()) : std::string("+channels"));
    out.meta["axis_id"] = a.package.meta.value("axis_id", std::string());
    out.meta["to_value_type"] = "scalar";
    out.meta["unit"] = "mode coefficient";
    out.meta["rows"] = Json{{"order", components == 3 ? "component-major: rows c*K .. (c+1)*K - 1 are Cartesian component c (x, y, z) of mode modes[k]" : "mode modes[k]"},
                            {"selection", modes}, {"normalisation", norm}};
    return ng::Arrays{{"operator", std::move(out)}};
  };
  n.record = [](const std::any& prepared) {
    const AnalysisPrepared& a = std::any_cast<const AnalysisPrepared&>(prepared);
    return Json{{"modes", a.p.value("modes", Json())}, {"normalisation", a.p.value("normalisation", Json())}, {"source", a.has_kernel ? "kernel" : "channel_map"}};
  };
  return n;
}

}  // namespace

void register_spectral_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {mode_analysis()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
