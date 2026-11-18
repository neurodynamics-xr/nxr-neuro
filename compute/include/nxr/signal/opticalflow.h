#pragma once
// nxr::flow — the apparent motion of an activation pattern on the cortex, and its Hodge–Dirac reading. A line port of Rheome's
// +dynamics/opticalflow_scalar, tangent_basis, flow_readout and +flow/apparent, activation (same maths, same names; Rheome is MIT-licensed MATLAB code).
//
//   tangentBasis(normals)                  -> e1, e2 [V x 3]: an orthonormal frame of each vertex's tangent plane (Brainstorm's optical-flow basis); e2 = n x e1.
//   opticalflowScalar(A, ...)              -> the velocity [3V x (T-1)] (rows x, y, z per vertex, tangent) of a scalar activity map A [V x T], by manifold Horn–Schunck:
//                                             per frame min_v sum_v M_v (dI/dt + v . grad I)^2 + alpha (a'Ka + b'Kb), v = a e1 + b e2; K the embedded cotan Laplacian,
//                                             M the lumped (row sum of the Galerkin) mass, grad the ambient hat-function face gradient averaged to the vertices
//                                             (FaceOperators with double positions). I = A / max|A|. One sparse 2V solve per frame.
//   flowReadout(v, ...)                    -> divergence, vorticity (curl . n, counter-clockwise positive from outside) and speed |v| of a velocity field.
//   activation(J, ...)                     -> the scalar activation [V x T] of a current J [3V x T]: the analytic envelope (default; Hilbert along time, then |J|) or |J|.
//   apparent(A, ...)                       -> the pipeline: activation map -> velocity -> divergence, vorticity, speed (per frame; x rate for m/s).
//   flowCoefficients(basis, v)             -> the Hodge–Dirac reading of a velocity field v [3V x frames] on the basis' (glued) surface: the flow weights b [K x frames]
//                                             (hodgeWeights is linear in its input). Exact modes: the divergence potential's coefficients (sources and sinks);
//                                             coexact modes: the stream function's (rotation). With hodgeSynthesis: potentials A and S, v = -grad A - n x grad S.
//
// WARNINGS (Rheome's, kept):
//  * KINEMATIC, NOT PHYSICAL. The flow is the APPARENT velocity of the amplitude pattern (as a stadium wave). Cortical activity is generated locally, it does not advect.
//    Where brightness constancy breaks (activity created or destroyed, not moved) the residual appears as div v: a source or sink here is either genuine propagation
//    converging or an amplitude change the model cannot express. It differs from the curl and divergence of the CURRENT J (an instantaneous property of J at one time).
//  * APERTURE-LIMITED. Horn–Schunck recovers only the velocity along the intensity gradient; the rest comes from the smoothness prior, so alpha sets how much of v is
//    data and how much regularisation. Vary it and report the range.
//  * PER FRAME, NOT PER SECOND. Adjacent frames are differenced, so |v| is length per FRAME; apparent().speed = speedPerFrame x rate (length per second). Without a rate,
//    speed is empty.
//  * THE INSTANTANEOUS NORM |J| IS THE WRONG MAP for a band-limited signal (it pulses at 2f and touches zero twice per cycle; the flow then reports the carrier). Use the
//    analytic envelope; band-limit first. Hilbert belongs at the sensors when you can put it there (the kernel is linear): activation accepts a complex J.
//  * COST. Each frame solves a 2V x 2V sparse system: work on one hemisphere.
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <complex>
#include <limits>
#include <memory>
#include <vector>

#include "nxr/hodge_weights.h"

namespace nxr::signal {

namespace connection { class ConnectionFlow; }
using nxr::hodge::FaceMat;
using nxr::hodge::RowMat3;

struct TangentBasis {
    RowMat3 e1, e2, normal;     // [V x 3]
};
/** Rheome's tangent_basis: from the vertex normals (normalised here). e1 is proportional to axis x n (Rheome's axis is (1,1,1)); any other axis gives another, equally valid frame. */
TangentBasis tangentBasis(const RowMat3& vertexNormals, const Eigen::Vector3d& axis = Eigen::Vector3d::Ones());

/** The smoothness the flow is charged, and the data term's integration. The velocity is v = a e1 + b e2 per vertex (Connection: z = (a, b) in the signpost gauge, embedded).
 *    Connection / Gauge  THE DEFAULT: the Cortical Flow Studio's optical flow (opticalflow_connection.h): alpha v'Cv with C the connection Laplacian of the intrinsic
 *                     Delaunay triangulation (gauge-free: it compares a vector with its neighbour's PARALLEL-TRANSPORTED one), the data term lumped at the vertices with the gradient
 *                     averaged at the corners in the gauge. A line port of the Studio's flow/opticalFlow.ts; nxr and the Studio are one pipeline.
 *    (Removed 2026-11-11: Rheome's Frame / Vertex form, the Laplacian of the frame COORDINATES; it ignores how the frames turn, so it depends on the frame choice: docs/audits/2026-11-11-opticalflow-smoothness.md.)
 *    Ambient / Triangle  THE CORRECT Lefevre and Baillet 2008 form (Brainstorm's implementation under-counts its right-hand side: docs/audits/2026-11-11-opticalflow-smoothness.md). Lefevre and Baillet 2008 (IEEE TPAMI 30(6), "Optical flow and advection on 2-Riemannian manifolds"): v = sum_i (a_i e1_i + b_i e2_i) w_i on P1 hat functions w_i;
 *                     the smoothness is the Dirichlet energy of the AMBIENT 3-vector field, alpha int sum_{d=x,y,z} |grad V_d|^2 dA = alpha sum_{ij} K_ij (u_i . u_j), u_i = a_i e1_i + b_i e2_i
 *                     (frame-independent: the frames enter only through the dot products e_k(i) . e_l(j)); the data term int (dI/dt + V . grad I)^2 dA per triangle with the triangle's own
 *                     gradient of I and the interpolated V and dI/dt (finite elements, not vertex-lumped). The ambient energy is the covariant (tangent) Dirichlet energy plus a bending
 *                     term of the shape operator; by Bochner the covariant energy is the Hodge energy minus a Gaussian curvature term. */
enum class Regulariser { Ambient, Connection };
enum class DataFit { Triangle, Gauge, Auto };
struct FlowForm {
    Regulariser regulariser = Regulariser::Connection;  // the Studio's connection Laplacian
    DataFit dataFit = DataFit::Auto;                    // Auto: Gauge for Connection, Triangle for Ambient
    Eigen::Vector3d frameAxis = Eigen::Vector3d::Ones();   // the tangent frames' reference axis (see tangentBasis)
};
/** The area-weighted unit vertex normals of a mesh (double positions). */
RowMat3 vertexNormals(const RowMat3& positions, const FaceMat& faces);

/** The embedded cotan Laplacian K (symmetrised, PSD) and the lumped Galerkin mass (row sums) of a mesh: the smoothness and the data weights of the optical flow. */
void embeddedLaplacian(const RowMat3& positions, const FaceMat& faces, Eigen::SparseMatrix<double>& K, Eigen::VectorXd& mass);

/** Rheome's opticalflow_scalar. A [V x T] -> [3V x (T-1)]. `alpha` the smoothness weight (Rheome's default 1). */
Eigen::MatrixXd opticalflowScalar(const Eigen::MatrixXd& activityMap, const RowMat3& positions, const FaceMat& faces, const RowMat3& vertexNormals, double alpha = 1.0, const FlowForm& form = {});

/** Rheome's opticalflow_scalar one frame pair at a time (the streaming form). Everything that does not change between frames is built once: the face operators, the
 *  embedded Laplacian and mass, the tangent basis, the system's sparsity pattern with the smoothness block already in it, and the symbolic analysis of its factorisation.
 *  `maxAbs` is the window's max |A| (the unit amplitude is taken over the whole window, so it is needed before the first step). Results equal opticalflowScalar's. */
/** The conjugate-gradient solve of the same per-frame system (opticalflow_methods.h: method `pcg`). Relative residual |r|/|rhs| <= tolerance, at most maxIterations. */
struct PcgOptions {
    enum class Preconditioner { Ldlt, IncompleteCholesky };
    double tolerance = 1e-8;
    int maxIterations = 500;
    Preconditioner preconditioner = Preconditioner::Ldlt;   // Ldlt: the exact factor of alpha K (+) alpha K + c M (+) M, built once; IncompleteCholesky: Eigen's, of the same matrix
};
struct PcgInfo { int iterations = 0; double residual = 0; bool converged = true; };

class OpticalflowScalar {
public:
    OpticalflowScalar(const RowMat3& positions, const FaceMat& faces, const RowMat3& vertexNormals, double alpha, double maxAbs, const FlowForm& form = {});
    ~OpticalflowScalar();
    /** The velocity [3V] (x, y, z per vertex, tangent) between two raw activation maps [V]. */
    Eigen::VectorXd step(const Eigen::VectorXd& previous, const Eigen::VectorXd& next);
    /** The same velocity by preconditioned conjugate gradients, warm-started from `ab` [2V] (the previous solution; any other size starts from zero) and left holding the new [a; b].
     *  `scale` replaces the unit amplitude given at construction when it is not NaN. The preconditioner is built once, on the first call (or by buildPreconditioner). */
    Eigen::VectorXd stepPcg(const Eigen::VectorXd& previous, const Eigen::VectorXd& next, Eigen::VectorXd& ab, const PcgOptions& options, PcgInfo* info = nullptr, double scale = std::numeric_limits<double>::quiet_NaN());
    /** Builds the preconditioner from a map's gradient (the data term's typical size c), if not built yet. */
    void buildPreconditioner(const Eigen::VectorXd& map, const PcgOptions& options, double scale = std::numeric_limits<double>::quiet_NaN());
private:
    void assemble(const Eigen::VectorXd& previous, const Eigen::VectorXd& next, double scale, Eigen::VectorXd& rhs, double* dataScale = nullptr);
    Eigen::VectorXd toVelocity(const Eigen::VectorXd& ab) const;
    void makePreconditioner(const PcgOptions& options, double c);
    Eigen::Index nV_;
    FlowForm form_;
    FaceMat faces_;
    Eigen::VectorXd area_;                                     // [nF] (triangle data term)
    std::vector<Eigen::Index> faceSlot_;                       // [nF x 36] where triangle f's (corner p, corner q, k, l) entry sits in system_.valuePtr()
    double scale_;
    nxr::hodge::FaceOperators fg_;
    Eigen::VectorXd mass_;
    TangentBasis tb_;
    Eigen::SparseMatrix<double> system_;                       // S + the data's four diagonals (values rewritten each frame)
    std::vector<double> smoothValues_;                         // system_'s values with the data terms zero
    std::vector<Eigen::Index> slot_;                           // [4 x V] where the data terms sit in system_.valuePtr()
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
    bool analysed_ = false;
    Eigen::SparseMatrix<double> pre_;                          // alpha K (+) alpha K + c M (+) M
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> preLdlt_;
    Eigen::IncompleteCholesky<double> preIc_;
    PcgOptions::Preconditioner preKind_ = PcgOptions::Preconditioner::Ldlt;
    bool preBuilt_ = false;
    std::unique_ptr<connection::ConnectionFlow> conn_;         // the Studio's connection-Laplacian flow (form.regulariser == Connection); the members above are then unused
};

/** flowReadout for one velocity [3V] at a time, with the operators built once. */
class FlowReader {
public:
    FlowReader(const RowMat3& positions, const FaceMat& faces);
    /** Divergence of the field as given, vorticity of its tangent part, speed |v| ([V] each). */
    void read(const Eigen::VectorXd& velocity, Eigen::VectorXd& divergence, Eigen::VectorXd& vorticity, Eigen::VectorXd& speed) const;
private:
    nxr::hodge::FaceOperators fg_;
};

struct FlowReadout {
    Eigen::MatrixXd divergence, vorticity, speed;   // [V x frames]
};
/** Rheome's flow_readout: the divergence is that of the field as given, the vorticity that of its tangent part. */
FlowReadout flowReadout(const Eigen::MatrixXd& velocity, const RowMat3& positions, const FaceMat& faces);

enum class ActivationMethod { Analytic, Norm };
/** Rheome's activation: J [3V x T] real (the Hilbert transform is taken here) or complex (already analytic) -> [V x T]. Analytic needs T >= 8. */
Eigen::MatrixXd activation(const Eigen::MatrixXcd& J, ActivationMethod method = ActivationMethod::Analytic, bool alreadyAnalytic = true);
/** The analytic signal along the columns' time axis (MATLAB hilbert on each row of X [rows x T]). */
Eigen::MatrixXcd hilbertRows(const Eigen::MatrixXd& X);
/** THE EXPLICIT NEIGHBOURHOOD of the FFT Hilbert transform (spec 2026-11-17-selection-module, 7.3 and 8: a halo is a planner fact the node declares, never a selection). hilbertRows takes the whole block it is
 *  handed, so its result at a sample depends on where the pipeline cut its chunks. hilbertRowsOwn takes a block that carries `left` samples of halo before its own columns and whatever halo follows, and returns the
 *  analytic signal of the own columns [left, left + own) only: the neighbourhood is the halo the caller read, stated, not the block's accident. hilbertHalo is the halo a node declares for a band whose lowest
 *  frequency is `lowestHz`: `cycles` cycles of it on each side, in samples (the Studio's `supportOf` / `fitLevelOf` use 4 cycles); a signal band-limited above `lowestHz` has its Hilbert response inside that
 *  support up to the band's own tail, so the own columns do not depend on the chunk size beyond it. Refuses a block shorter than left + own. */
std::int64_t hilbertHalo(double sfreq, double lowestHz, double cycles = 4.0);
Eigen::MatrixXcd hilbertRowsOwn(const Eigen::MatrixXd& block, Eigen::Index left, Eigen::Index own);

struct ApparentOptions {
    double rate = std::numeric_limits<double>::quiet_NaN();   // Hz; required for .speed in length per second
    double alpha = 1.0;                                         // Horn–Schunck smoothness weight
    FlowForm form;                                              // the Studio's connection flow by default
};
struct Apparent {
    Eigen::MatrixXd velocity;       // [3V x T-1]
    Eigen::MatrixXd divergence;     // [V x T-1]
    Eigen::MatrixXd vorticity;      // [V x T-1]
    Eigen::MatrixXd speedPerFrame;  // [V x T-1]
    Eigen::MatrixXd speed;          // [V x T-1] x rate; empty without a rate
};
/** Rheome's apparent: A [V x T] real, non-negative; at least two frames. */
Apparent apparent(const Eigen::MatrixXd& A, const RowMat3& positions, const FaceMat& faces, const ApparentOptions& options = {});

/** The flow weights b [K x frames] of a velocity field [3V x frames] on the basis' glued surface. */
Eigen::MatrixXd flowCoefficients(const nxr::hodge::Basis& basis, const Eigen::MatrixXd& velocity);

}  // namespace nxr::signal
