#pragma once
// nxr::flow — the Cortical Flow Studio's optical flow: MANIFOLD HORN–SCHUNCK WITH THE CONNECTION LAPLACIAN (regulariser `connection`, the default;).
//
// A line port of the Studio's `flow/opticalFlow.ts` (nxr-cortical-flow studio/src/engine/toolbox/flow/opticalFlow.ts), with `differential/intrinsicGauge.ts` (corners, intrinsicGradient,
// cornerAt, embedAt), `differential/faceGradient.ts` (assembleFaceGradient) and `differential/connectionLaplacian.ts` (tangentBasis' normals): same maths, same names, same constants.
//
// Per frame, with I the map normalised to unit peak, İ = I(t+1) − I(t), ∇I the vertex gradient and v a tangent vector at a vertex written z = (a, b) in the vertex's SIGNPOST GAUGE:
//     min over v   sum_v M_v (İ + v . ∇I)^2  +  alpha v' C v
// C the connection Laplacian of the intrinsic Delaunay triangulation (cotan weights, each edge carrying a vector across by its parallel transport), M the lumped (mixed Voronoi) mass of
// the input triangulation. The normal equations are the 2n SPD system ( alpha C + blockdiag( M [g1^2 g1g2; g1g2 g2^2] ) ) z = − M İ (g1, g2), g the gradient in C's gauge.
// The gauge: vectors go into and out of C's gauge FACE BY FACE through the intrinsic corner angles (`Corners`); nothing of a vertex's projected plane measures an angle, so the result does
// not depend on the reference edges (gauge-free), unlike `frame` (Rheome's a'Ka + b'Kb on the frame coefficients).
// DATA TERM: the Studio's own (`DataFit::Gauge`): the gradient of I averaged AT THE CORNERS in C's gauge (each incident face's gradient at its corner's intrinsic angle, by area), the
// mass-lumped square. It is neither `vertex` (the ambient 3-vector averaged, projected on the frames) nor `triangle` (per-triangle finite elements).
// SOLVER: sparse LDLT of the 2n system (the Studio's, the parity reference), a ridge of 1e-9 alpha mean(C_ii) on the diagonal; `pcg` as the other methods.
// RETURNS: the EMBEDDED velocity (3 a vertex, ambient 3-vectors tangent to the surface, the positions' unit PER FRAME), and the gauge components z (2 a vertex, a_i at 2i, b_i at 2i+1).
// DIFFERENCES FROM THE STUDIO (none of the maths): the input map and the gradient operator are double here, with the gradient operator's float32 rounding KEPT (positions rounded to float32,
// the 9 gradient numbers a face stored as float32, the vertex normals as float32) so the numbers equal the Studio's; the output is double (the Studio rounds it to float32); the
// factorisation's ordering is Eigen's AMD (the Studio's nested dissection): the same solution.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "nxr/intrinsic_surface.h"
#include "nxr/opticalflow.h"

namespace nxr::signal::connection {
using DelaunayOptions = nxr::manifold::DelaunayOptions;

/** `Corners` (intrinsicGauge.ts): C's gauge, face by face. Per halfedge of the component's INPUT faces: its signpost psi0 (rescaled), its corner's width in psi, its edge's unit vector and the in-plane perpendicular. */
struct Corners {
    std::vector<double> psi0, width, edge, perp;   // [nH], [nH], [3 nH], [3 nH]
    std::vector<double> scale;                     // [n] the rescaling (psi = theta * scale)
    std::vector<int32_t> ptr, order;               // [n+1], [nH]: the corners of a vertex in fan order (CSR over halfedges)
    std::vector<int32_t> faces;                    // [3 nF] the input faces (local)
    std::vector<double> area;                      // [nF]
};
/** `assembleFaceGradient` (faceGradient.ts), the part the flow reads: per face the three hat functions' gradients, stored as float32 as the Studio stores them. */
struct FaceGradient { int nF = 0; std::vector<float> grads; };   // [9 nF]

/** `cornersOf`: the corners of the input faces (their signposts and angles, before any flip: `IntrinsicSurface::inputCorners`); P [3n] the input positions, F [3 nF] the input faces. */
Corners cornersOf(const nxr::manifold::IntrinsicSurface::InputCorners& m, const double* P, const int32_t* F);
/** `assembleFaceGradient` from float32 positions. */
FaceGradient assembleFaceGradient(const float* position, const int32_t* index, int nV, int nF);
/** `intrinsicGradient`: a scalar's gradient at every vertex in C's gauge (2n, a at 2v, b at 2v+1): each face's gradient at its corners' intrinsic angles, averaged by area. */
Eigen::VectorXd intrinsicGradient(const Corners& c, const FaceGradient& grad, const double* s);
/** `cornerAt`: the corner (halfedge) at vertex v whose wedge holds the intrinsic angle psi, and psi's angle theta inside that face from its edge (h = −1: a vertex in no face). */
void cornerAt(const Corners& c, int v, double psi, int& h, double& theta);
/** `embedAt`: z = (a, b) in C's gauge at vertex v, EMBEDDED: unscaled into its corner's face, projected on the tangent plane `normal` (3 floats' values), at length |z|. out[0..2]. */
void embedAt(const Corners& c, const double* normal, int v, double a, double b, double* out);

/** `componentPencil`: a connected component's pencil from its own faces (vertices numbered from 0). */
struct Pencil {
    Eigen::SparseMatrix<double> C;   // the connection Laplacian, real 2n x 2n (vertex i: rows 2i, 2i+1)
    Eigen::VectorXd M;               // the lumped (mixed Voronoi) mass [n]
    std::vector<double> normals;     // [3n] the area-weighted unit vertex normals, float32 values
    Corners corners;
    FaceGradient grad;
};
Pencil componentPencil(const double* positions, int n, const std::vector<int32_t>& faces, const DelaunayOptions& o = {});
/** The connection Laplacian of the intrinsic surface as the real symmetric 2n x 2n matrix (complex entry a+ib -> [[a, −b], [b, a]]). */
Eigen::SparseMatrix<double> realConnectionLaplacian(nxr::manifold::IntrinsicSurface& s);

/** The flow itself: built once per mesh (components found, pencils, the system's pattern and its symbolic factorisation). */
class ConnectionFlow {
public:
    /** `maxAbs` the window's peak (the unit amplitude); `positions` [V x 3], `faces` [F x 3] (any number of connected pieces, solved independently). */
    ConnectionFlow(const RowMat3& positions, const FaceMat& faces, double alpha, double maxAbs, const DelaunayOptions& o = {});
    ~ConnectionFlow();
    ConnectionFlow(const ConnectionFlow&) = delete;
    ConnectionFlow& operator=(const ConnectionFlow&) = delete;
    /** The embedded velocity [3V] between two raw maps [V] (LDLT); the gauge solution z [2V] is `gauge()`. */
    Eigen::VectorXd step(const Eigen::VectorXd& previous, const Eigen::VectorXd& next);
    /** The same by preconditioned CG warm-started from `z` [2V] (left holding the new z). `scale` replaces the construction's unit amplitude when not NaN. */
    Eigen::VectorXd stepPcg(const Eigen::VectorXd& previous, const Eigen::VectorXd& next, Eigen::VectorXd& z, const PcgOptions& options, PcgInfo* info = nullptr, double scale = std::numeric_limits<double>::quiet_NaN());
    void buildPreconditioner(const Eigen::VectorXd& map, const PcgOptions& options, double scale = std::numeric_limits<double>::quiet_NaN());
    const Eigen::VectorXd& gauge() const;                 // the last LDLT solution z [2V]
    const Eigen::VectorXd& mass() const;                  // M [V]
    /** The system's smoothness part alpha C (2V x 2V, global numbering) */
    const Eigen::SparseMatrix<double>& smoothness() const;
    /** The gradient of a raw map in C's gauge (2V) with the unit amplitude applied. */
    Eigen::VectorXd gaugeGradient(const Eigen::VectorXd& map, double scale = std::numeric_limits<double>::quiet_NaN()) const;
    Eigen::VectorXd embed(const Eigen::VectorXd& z) const;   // z [2V] -> [3V]
    /** The inverse reading: a tangent vector field [3V] (embedded) in C's gauge (z [2V]): each incident face's angle of the vertex's vector at its corner, averaged by area (as intrinsicGradient does a gradient). */
    Eigen::VectorXd gaugeOf(const Eigen::VectorXd& field) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace nxr::signal::connection
