#pragma once
// The light halfedge bundles of a mesh (topology and per-element geometry), the connection choice, and the extrinsic Delaunay repair.
#include <complex>
#include <cstdint>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/core/errors.h"
#include "nxr/core/cancellation.h"
#include "nxr/core/progress.h"

namespace geometrycentral { namespace surface {
class ManifoldSurfaceMesh;
class VertexPositionGeometry;
class IntrinsicGeometryInterface;
} }

namespace nxr::geometry {

// Cross-cutting types live in nxr::core; bring them into manifold scope
// so unqualified uses below (Error, CancellationToken, ProgressObserver)
// resolve. See <nxr/core/errors.h>, <nxr/core/cancellation.h>, <nxr/core/progress.h>.
using core::Error;
using core::ErrorCode;
using core::CancellationToken;
using core::ProgressObserver;

struct MeshGeometry; struct MeshTopology;

enum class ConnectionType { Euclidean, LeviCivita, Trivial };   // the choice of connection (G7: formerly GaugeType; "gauge" names only the tangent-space convention)

// There is no surface object: a surface is its arrays (nxr/geometry/surface_arrays.h, the mesh and its intrinsic triangulation), and each function builds
// geometry-central's objects from them inside (IntrinsicSurface(const SurfaceArrays&)). The connection a function uses is its argument.

/** χ = nV − nE + nF of the input mesh. */
int eulerCharacteristic(const SurfaceArrays& surface);
/** The trivial connection's singularities must be given and sum to χ (Gauss–Bonnet); any other connection takes none. Throws Error(InvalidInput). */
void checkSingularities(const SurfaceArrays& surface, ConnectionType type, const std::map<int,double>& singularities);

// Extrinsic Delaunay edge-flip repair (geometry-central fixDelaunay).
// Flips edges in place — same vertices (positions/count/indices), new faces.
// Best-effort PSD (reduces obtuse/negative cotan weights), not a certificate;
// the intrinsic-Delaunay certificate is a separate facility. Input must be a
// valid manifold (the underlying ManifoldSurfaceMesh throws otherwise).
struct DelaunayNormalization {
    Eigen::MatrixXi faces;  // [nF, 3] 0-based, re-triangulated; same vertex indices
    int flips = 0;          // edge flips performed (0 => already Delaunay)
};
DelaunayNormalization fixDelaunay(
    const double* vertices, int nV, const int32_t* faces, int nF);
} // namespace nxr::geometry

namespace nxr::geometry {
// ── Halfedge mesh topology ────────────────────────────────────
//
// Flat struct-of-arrays representation of the halfedge combinatorics,
// using 0-based indices matching geometry-central's internal storage.
// Bindings that want 1-based MATLAB indices convert at the binding edge
// (see indexVectorToMx1Based in marshal.h).
// A value of -1 (INVALID) means "none / boundary" for .face and .corner
// on exterior halfedges; all other entries are non-negative for a valid
// closed manifold mesh.
//
//   nH == 2 * nE   (every edge has exactly 2 halfedges)
//   nC == 3 * nF   (every interior face has 3 corners)

struct MeshTopology {
    // Element counts.
    int nV, nE, nF, nH, nC;

    // Reverse lookups: one representative halfedge per element (0-based).
    std::vector<long> vertexHalfedge;   // [nV]  he that "belongs to" vertex
    std::vector<long> edgeHalfedge;     // [nE]
    std::vector<long> faceHalfedge;     // [nF]
    std::vector<long> cornerHalfedge;   // [nC]

    // Halfedge adjacency table (0-based; -1 = none / exterior).
    std::vector<long> heTwin;      // [nH]
    std::vector<long> heNext;      // [nH]
    std::vector<long> heVertex;    // [nH]  tail vertex
    std::vector<long> heEdge;      // [nH]
    std::vector<long> heFace;      // [nH]  -1 for exterior halfedges
    std::vector<long> heCorner;    // [nH]  -1 for exterior halfedges

    // Boolean attributes (stored as char: 1 = true, 0 = false).
    std::vector<char> heOrientation;  // [nH]  true iff he == he.edge().halfedge()
    std::vector<char> heIsInterior;   // [nH]
};

/** Extract the halfedge mesh topology into a plain struct-of-arrays.
 *  All indices are 0-based (geometry-central convention). -1 encodes
 *  geometry-central's INVALID_IND for "none / boundary" slots. */
MeshTopology meshTopology(const SurfaceArrays& surface);

// ── Light per-element geometry bundle ────────────────────────
//
// All light O(V/E/F/H/C) per-element quantities, element-grouped.
// Heavy sparse operators are intentionally excluded (on-demand
// surface, separate). Frames are the complex `grid` (c = e1+i·e2);
// normals derive as real×imag. Curvature is the 2-RoSy deviatoric
// + mean (see vertexCurvature). See the bundle design spec §4.2.
//
// Corner-array sizing note: cornerAngles / cornerScaledAngles are
// sized to nH (not nC) for safety on boundary meshes, where
// geometry-central corner ids are halfedge-based and may be sparse.
// On a closed mesh nH == nC so the size is identical.
struct MeshGeometry {
    double totalArea = 0.0;
    // vertex
    Eigen::VectorXd  vertexDualAreas;           // [nV]
    Eigen::VectorXd  vertexAngleSums;           // [nV]
    Eigen::VectorXcd vertexCurvatureDeviatoric; // [nV] q (2-RoSy)
    Eigen::VectorXd  vertexMeanCurvature;       // [nV] H
    Eigen::MatrixXcd vertexGrid;                // [nV,3] c = e1+i·e2
    // edge
    Eigen::VectorXd  edgeLengths;              // [nE]
    Eigen::VectorXd  edgeCotanWeights;         // [nE]
    Eigen::VectorXd  edgeDihedralAngles;       // [nE]
    // face
    Eigen::VectorXd  faceAreas;                // [nF]
    Eigen::MatrixXd  faceCentroids;            // [nF,3]
    Eigen::MatrixXcd faceGrid;                 // [nF,3]
    // halfedge
    Eigen::VectorXd  halfedgeCotanWeights;     // [nH]
    Eigen::VectorXcd halfedgeVectorsInVertex;  // [nH]
    Eigen::VectorXcd halfedgeVectorsInFace;    // [nH]
    Eigen::VectorXcd halfedgeTransportAlong;   // [nH]
    Eigen::VectorXcd halfedgeTransportAcross;  // [nH]
    // corner (sized nH; on closed mesh nH==nC)
    Eigen::VectorXd  cornerAngles;             // [nH]
    Eigen::VectorXd  cornerScaledAngles;       // [nH]
};

MeshGeometry meshGeometry(const SurfaceArrays& surface);

/** Area [nF] of each face of a mesh (positions [nV x 3], faces [nF x 3]). */
Eigen::VectorXd faceAreas(const RowMat3& positions, const FaceMat& faces);

} // namespace nxr::geometry

