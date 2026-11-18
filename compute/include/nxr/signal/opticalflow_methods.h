#pragma once
// nxr::flow — ALTERNATIVE METHODS for the optical flow, selectable by name, to be compared on real data. `pcg` is the default; `direct` is the one-solve-per-frame form (opticalflow.h). Every method yields the velocity [3V] per frame pair on the input surface (and the Hodge–Dirac flow weights where it has them).
//
//   direct    (smoothness `regulariser`, data `datafit`: connection / gauge by default, the Cortical Flow Studio's; ambient / triangle is Lefevre and Baillet 2008) sparse LDLT refactored every frame.
//   pcg       the SAME equation by preconditioned conjugate gradients, warm-started from the previous frame's [a; b]. Preconditioner built once from the first frame:
//             the LDLT of alpha K (+) alpha K + c M (+) M (c the mean data weight; `ldlt`, the default) or Eigen's IncompleteCholesky of it (`ic`). Options: relative residual
//             tolerance (default 1e-8), maximum iterations. Iterations per frame are reported.
//   perscale  a MEASUREMENT, not a solver: the flow of each band-pass member of a graph filterbank (family required) applied to the map, each solved by `pcg` (a member's map is scaled
//             to its own unit amplitude over the window). One series per member.
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "nxr/signal/opticalflow.h"

namespace nxr::signal {

enum class FlowMethod { Direct, Pcg, Perscale };
Regulariser parseRegulariser(const std::string& name);   // "connection", "ambient"; throws std::invalid_argument
const char* regulariserName(Regulariser r);
DataFit parseDataFit(const std::string& name);           // "gauge", "triangle", "auto"; throws std::invalid_argument
const char* dataFitName(DataFit d);
/** Auto -> the regulariser's own data term (connection: gauge; ambient: triangle); throws std::invalid_argument when the pair does not go together (gauge is the connection's only). */
DataFit resolveDataFit(Regulariser r, DataFit d);
FlowMethod parseFlowMethod(const std::string& name);   // throws std::invalid_argument on an unknown name
const char* flowMethodName(FlowMethod method);

// THE DEFAULT SMOOTHNESS of direct / pcg / perscale: the Cortical Flow Studio's functional, `connection` with its
// own `gauge` data term (opticalflow_connection.h). One constant, flipped here (and in meeg.h's `regulariser` string) if the maintainers rule another. The others stay options:
//   ambient    the CORRECT Lefevre and Baillet 2008 form (Brainstorm's implementation under-counts its right-hand side: docs/audits/2026-11-11-opticalflow-smoothness.md); data `triangle`
inline constexpr Regulariser kDefaultRegulariser = Regulariser::Connection;
inline constexpr DataFit kDefaultDataFit = DataFit::Auto;   // the regulariser's own: gauge for connection, triangle for ambient

struct FlowMethodOptions {
    FlowMethod method = FlowMethod::Pcg;
    // direct, pcg, perscale: the functional (opticalflow.h, FlowForm). Defaults: the Studio's connection Laplacian with its gauge data term; ambient / triangle is Lefevre and Baillet's
    Regulariser regulariser = kDefaultRegulariser;
    DataFit dataFit = kDefaultDataFit;
    // pcg, perscale
    double tolerance = 1e-8;
    int maxIterations = 500;
    PcgOptions::Preconditioner preconditioner = PcgOptions::Preconditioner::Ldlt;
    // perscale: the graph filterbank ("mexhat", "itersine" or "logitersine"; its scaling and top members are low- and high-pass and are left out)
    std::string scaleFamily;
    int scaleVoices = 3;
    int scaleFilters = 0;                              // 0: derived from voices
};

struct FlowStep {
    std::vector<Eigen::VectorXd> velocity;             // per series [3V]
    std::vector<int> iterations;                       // per series (pcg / perscale); empty for direct
    double seconds = 0;                                // the step's wall time
};

class FlowEngine {
public:
    FlowEngine(const FlowMethodOptions& options, const RowMat3& positions, const operators::FaceMat& faces, const RowMat3& normals, double alpha);
    ~FlowEngine();
    FlowEngine(const FlowEngine&) = delete;
    FlowEngine& operator=(const FlowEngine&) = delete;
    int series() const;                                // 1, but the filterbank's band-pass members for perscale
    std::vector<std::string> seriesNames() const;      // "" for one series; "m00", "m01", ... for perscale
    std::vector<double> seriesCentres() const;         // perscale: each member's centre wavenumber (1/length); empty otherwise
    /** Pass 1: every map of the window, once (the unit amplitude is over the whole window). */
    void observe(const Eigen::VectorXd& map);
    double maxAbs() const;                             // the raw map's max |A| seen
    /** Pass 2: the velocity between two consecutive maps. */
    FlowStep step(const Eigen::VectorXd& previous, const Eigen::VectorXd& next);
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace nxr::signal
