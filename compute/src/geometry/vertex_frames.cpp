#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/facets.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>

namespace nxr::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// See VertexFrames in compute.h: geometry-central's gauge of the (mollified) input mesh, realised by the IntrinsicSurface (equal to `vertexTangentBasis`
// when nothing was mollified), so it matches the tangent space the connection Laplacian and the transport use.
VertexFrames vertexFrames(Manifold& m) {
    const TangentBasis B = m.intrinsicSurface().tangentBasis();
    const int nV = m.nV();
    VertexFrames out;
    out.e1.resize(nV, 3); out.e2.resize(nV, 3); out.normals.resize(nV, 3);
    for (int v = 0; v < nV; ++v)
        for (int c = 0; c < 3; ++c) {
            out.e1(v, c) = B.e1[static_cast<size_t>(3 * v + c)];
            out.e2(v, c) = B.e2[static_cast<size_t>(3 * v + c)];
            out.normals(v, c) = B.n[static_cast<size_t>(3 * v + c)];
        }
    return out;
}

// DEPRECATED alias of vertexFrames (G7 removes it): there is one gauge now.

} // namespace nxr::geometry
