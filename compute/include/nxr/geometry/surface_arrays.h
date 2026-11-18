// A surface as ARRAYS: the input mesh (positions, faces) and the intrinsic Delaunay triangulation made from it, exported as geometry-central's own arrays.
//
// There is no surface object in compute. A function that needs geometry-central builds its objects inside, from these arrays
// (`geometry::IntrinsicSurface(const SurfaceArrays&)`), and drops them on return: no cache is held anywhere in compute.
// The intrinsic triangulation is an expensive DATA PRODUCT (mollification and edge flips): it is made once, by `surfaceArrays`, and handed in.
// Rebuilt from its arrays, the triangulation is the same mesh with the same element numbering, edge lengths and gauge as the one exported, so every
// operator built on it is bit-identical to the one built on the original (test_surface_arrays).
#pragma once

#include "nxr/geometry/intrinsic_surface.h"

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace nxr::geometry {

using RowMat3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using FaceMat = Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

/** The intrinsic triangulation of a mesh, as arrays. With `options.intrinsic == false` (the named option `mesh = embedded`) the operator mesh is the input
 *  itself and only `options`, `record` and `mass` are filled. */
struct IntrinsicTriangulation {
    IntrinsicOptions options;                 // what the construction was told
    IntrinsicRecord  record;                  // what it did
    std::vector<double> mass;                 // [nV] the lumped mass the operators use (input mixed-Voronoi); empty: geometry-central's own (barycentric)
    std::vector<double> inputEdgeLength;      // [nE of the input] the mollified input's edge lengths, by the input mesh's edge index
    // The operator mesh, in geometry-central's internal layout (implicit twins: twin(h) = h xor 1, edge(h) = h / 2; boundary loops at the end of `faceHalfedge`).
    std::vector<int64_t> halfedgeNext, halfedgeVertex, halfedgeFace;   // [nH]
    std::vector<int64_t> vertexHalfedge;      // [nV]
    std::vector<int64_t> faceHalfedge;        // [nF + boundary loops]
    int64_t boundaryLoops = 0;
    std::vector<double> edgeLength;           // [nE] the intrinsic edge lengths
    std::vector<double> signpost;             // [nH] the raw signpost angle of each halfedge (from the input's reference halfedge)
    std::vector<double> vertexAngleSum;       // [nV] the cone angle each vertex has in the triangulation (as the triangulation holds it)
    std::vector<double> halfedgeVectorInVertex;   // [2 nH] (x, y) the halfedge in its tail vertex's tangent plane: the gauge
    bool intrinsic() const { return options.intrinsic; }
};

/** The mesh and its intrinsic triangulation: everything a geometry or operator function needs, as plain data. */
struct SurfaceArrays {
    RowMat3 positions;                        // [nV x 3]
    FaceMat faces;                            // [nF x 3], 0-based, oriented
    IntrinsicTriangulation triangulation;
    int nV() const { return static_cast<int>(positions.rows()); }
    int nF() const { return static_cast<int>(faces.rows()); }
};

/** Make the intrinsic triangulation of a mesh (the expensive step, done once): positions [3 nV] row-major, faces [3 nF]. */
IntrinsicTriangulation intrinsicTriangulation(const double* positions, int nV, const int32_t* faces, int nF, const IntrinsicOptions& options = {});

/** The surface arrays of a mesh: its positions and faces, and its intrinsic triangulation made with `options`. */
SurfaceArrays surfaceArrays(const double* positions, int nV, const int32_t* faces, int nF, const IntrinsicOptions& options = {});
SurfaceArrays surfaceArrays(const RowMat3& positions, const FaceMat& faces, const IntrinsicOptions& options = {});

/** The mesh option and the intrinsic Delaunay options as the construction's options (the mass is the input mixed-Voronoi, as every operator uses). */
IntrinsicOptions intrinsicOptions(bool intrinsicDelaunay = true, const DelaunayOptions& delaunay = {});

}  // namespace nxr::geometry
