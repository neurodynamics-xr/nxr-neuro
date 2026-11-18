#pragma once
// nxr::hodge — THE FUSED INVERSE: an imaging kernel read into the Hodge–Dirac modes. PURE MATHS (arrays in, arrays out; the kernel is just a matrix).
// A line port of nxr's `nxr.inverse.fused` (glued_kernel, FaceOperators, fused_weights, synthesis; Rheome's hodge_weights / hodge_synthesis; the Studio's
// hodgeSynthesisWeights): the same operators, the same names where they exist.
//
//   hodgeWeights(P, mu, kind, mass, positions, faces, W)  ->  B [2K x C]: rows k < K the normal-current weight a_k (coexact modes; 0 on exact ones), rows K + k the
//                                                             flow weight b_k. A sensor column x gives the mode coefficients w = B x; an FFT column X(f) complex ones.
//   hodgeSynthesis(...)  ->  w [2K x m] back to the current J [3V x m] on the glued surface and its forms: A = sum_exact b P, S = sum_coexact b P (the flow's potentials),
//                            normal = sum_coexact a P; J = -grad A - n x grad S with its normal part set to `normal`.
//   gluedKernel(...)     ->  a kernel [3 V_input x C] carried onto the glued surface [3 V_glued x C]: a kept vertex its own rows, a bridge vertex the inverse-distance blend of
//                            its 3 nearest cut-out (wall) vertices, in input positions (rounded to single first).
//   modePower(B, S)      ->  [2K x bands] diag(B S_b B^T), S_b a band's (real part of the) sensor cross-spectrum (nxr.spectral.chain.mode_power).
//
// The stored Hodge–Dirac package supplies P [K x V_glued] (the eigenvectors operator transposed: vertex potentials, mode-major), mu, kind (0 exact, 2 coexact), the glued
// positions [V_glued x 3] and faces [nF x 3], and the vertex mass (mixed Voronoi of the glued mesh as the solve took it: nxr.operators.intrinsic_delaunay.lumped_mass).
// Face operators: Brainstorm's ambient hat-function gradient (grad phi_i = N x opposite edge / 2A) — not the face-native dual calculus of ops::facegrad.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <functional>
#include <vector>

namespace nxr::operators {

using RowMat3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using FaceMat = Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

/** The package's side of a solve (nxr's `Solve`): everything the weights need. */
struct Basis {
    Eigen::MatrixXd P;                 // [K x V_glued] vertex potentials, mode-major
    Eigen::VectorXd mu;                // [K]
    std::vector<int> kind;             // [K] 0 exact · 2 coexact
    Eigen::VectorXd mass;              // [V_glued] the 0-form (vertex) mass
    RowMat3 positions;                 // [V_glued x 3]
    FaceMat faces;                     // [nF x 3] the glued faces
    std::vector<std::int64_t> keep;    // [V_glued] the input vertex, -1 on the bridge (kept first)
    std::int64_t kept() const;         // the number of kept vertices
};

/** The face gradient of hat functions on a mesh (faceGradient.ts `assembleFaceGradient`) as sparse matrices. */
class FaceOperators {
public:
    /** `roundToSingle`: the Studio's positions are float32 (default); false keeps double (Rheome's face_gradient on double vertices: the optical flow). */
    FaceOperators(const RowMat3& positions, const FaceMat& faces, bool roundToSingle = true);
    int nV = 0, nF = 0;
    Eigen::SparseMatrix<double> grad[3], curl[3];   // [nF x nV] per xyz
    Eigen::SparseMatrix<double> toVertex;           // [nV x nF] area-weighted face -> vertex mean
    RowMat3 normals;                                // [nV x 3] area-weighted vertex normals
    /** divergence and curl of the TANGENT part of J (three [V x m] blocks: x, y, z) at the vertices. */
    void divCurl(const Eigen::MatrixXd J[3], Eigen::MatrixXd& div, Eigen::MatrixXd& curl) const;
    /** the area-weighted mean of the face gradients of x [V x m]: three [V x m] blocks. */
    void vertexGradient(const Eigen::MatrixXd& x, Eigen::MatrixXd out[3]) const;
};

/** A kernel [3 V_input x C] (rows x, y, z per vertex) on the glued surface [3 V_glued x C]. */
Eigen::MatrixXd gluedKernel(const Eigen::MatrixXd& W, const Basis& basis, const RowMat3& inputPositions);

/** A kernel carried onto the glued surface, with what does not change built once (the bridge vertices' three nearest cut vertices and weights): `apply` is gluedKernel for one
 *  kernel or one velocity column at a time. */
class GluedCarry {
public:
    GluedCarry(const Basis& basis, const RowMat3& inputPositions);
    Eigen::MatrixXd apply(const Eigen::MatrixXd& W) const;      // [3 V_input x C] -> [3 V_glued x C]
private:
    Eigen::Index vInput_ = 0, kept_ = 0, vGlued_ = 0;
    std::vector<std::int64_t> keep_;                  // [kept] the input vertex of each kept glued vertex
    std::vector<int> wallVertex_;                     // [3 x bridges] the cut vertices blended
    std::vector<double> wallWeight_;                  // [3 x bridges]
};

/** The reading of velocity fields (or kernels) in the modes, with everything that does not depend on the field built once: the face operators, the mode matrices, the
 *  normal-current Gram factor and the LU of the flow system (the [K x K] system of the modes' own flows). `weights` is hodgeWeights; `flow` reads only the flow weights b. */
class WeightsReader {
public:
    WeightsReader(const Basis& basis, int block = 256, const std::function<void(std::int64_t, std::int64_t)>& progress = {});
    Eigen::MatrixXd weights(const Eigen::MatrixXd& W) const;    // [2K x C] of W [3 V_glued x C]
    Eigen::MatrixXd flow(const Eigen::MatrixXd& W) const;       // [K x C]: the flow weights b only (rows K.. of weights)
private:
    Eigen::MatrixXd flowSolve(const Eigen::MatrixXd& W, Eigen::MatrixXd* normalReading) const;
    FaceOperators ops_;
    Eigen::Index V_ = 0, K_ = 0, ne_ = 0;
    std::vector<int> co_, ex_, order_;
    Eigen::MatrixXd PcoM_, PexM_;                     // P rows (coexact / exact) times the mass: [.. x V]
    Eigen::LLT<Eigen::MatrixXd> gram_;
    Eigen::PartialPivLU<Eigen::MatrixXd> flowLu_;
};

/** The fused weights [2K x C] of the kernel W [3 V_glued x C] on the basis' glued surface. `progress(done, of)` per block of flow-system columns. */
Eigen::MatrixXd hodgeWeights(const Basis& basis, const Eigen::MatrixXd& W, int block = 256, const std::function<void(std::int64_t, std::int64_t)>& progress = {});

struct Synthesis {
    Eigen::MatrixXd J;        // [3 V x m] the current (rows x, y, z per vertex)
    Eigen::MatrixXd A;        // [V x m] exact flow potential
    Eigen::MatrixXd S;        // [V x m] coexact flow potential
    Eigen::MatrixXd normal;   // [V x m] the normal current (s2)
};
Synthesis hodgeSynthesis(const Basis& basis, const Eigen::MatrixXd& weights, const FaceOperators* ops = nullptr);


}  // namespace nxr::hodge
