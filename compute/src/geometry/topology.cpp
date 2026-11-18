#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/utilities/utilities.h"

namespace nxr::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

static long idxOrNone(size_t i) {
    return (i == INVALID_IND) ? -1L : static_cast<long>(i);
}

MeshTopology meshTopology(const SurfaceArrays& surface) {
    geometry::IntrinsicSurface gs(surface);
    auto& mesh = gs.inputMesh();

    const int nV = surface.nV(), nE = static_cast<int>(gs.inputMesh().nEdges()), nF = surface.nF();
    const int nH = static_cast<int>(mesh.nHalfedges());
    const int nC = static_cast<int>(mesh.nCorners());

    MeshTopology t;
    t.nV = nV; t.nE = nE; t.nF = nF; t.nH = nH; t.nC = nC;

    t.vertexHalfedge.resize(nV);
    t.edgeHalfedge.resize(nE);
    t.faceHalfedge.resize(nF);
    // sized nH (== nCorners on a closed mesh); corner ids are halfedge-based
    // in geometry-central, so on a boundary mesh indices are sparse.
    t.cornerHalfedge.assign(nH, -1L);

    for (Vertex v : mesh.vertices())  t.vertexHalfedge[v.getIndex()] = idxOrNone(v.halfedge().getIndex());
    for (Edge e : mesh.edges())       t.edgeHalfedge  [e.getIndex()] = idxOrNone(e.halfedge().getIndex());
    for (Face f : mesh.faces())       t.faceHalfedge  [f.getIndex()] = idxOrNone(f.halfedge().getIndex());
    for (Corner c : mesh.corners())   t.cornerHalfedge[c.getIndex()] = idxOrNone(c.halfedge().getIndex());

    t.heTwin.resize(nH);      t.heNext.resize(nH);
    t.heVertex.resize(nH);    t.heEdge.resize(nH);
    t.heFace.resize(nH);      t.heCorner.resize(nH);
    t.heOrientation.resize(nH); t.heIsInterior.resize(nH);

    for (Halfedge he : mesh.halfedges()) {
        size_t i = he.getIndex();
        t.heTwin[i]       = idxOrNone(he.twin().getIndex());
        t.heNext[i]       = idxOrNone(he.next().getIndex());
        t.heVertex[i]     = idxOrNone(he.vertex().getIndex());
        t.heEdge[i]       = idxOrNone(he.edge().getIndex());
        t.heFace[i]       = he.isInterior() ? idxOrNone(he.face().getIndex())   : -1L;
        t.heCorner[i]     = he.isInterior() ? idxOrNone(he.corner().getIndex()) : -1L;
        t.heOrientation[i]= (he == he.edge().halfedge()) ? 1 : 0;
        t.heIsInterior[i] = he.isInterior() ? 1 : 0;
    }

    return t;
}

} // namespace nxr::geometry

namespace nxr::geometry {

int eulerCharacteristic(const SurfaceArrays& surface) {
    IntrinsicSurface gs(surface);
    return static_cast<int>(gs.inputMesh().nVertices()) - static_cast<int>(gs.inputMesh().nEdges()) + static_cast<int>(gs.inputMesh().nFaces());
}

void checkSingularities(const SurfaceArrays& surface, ConnectionType type, const std::map<int,double>& s) {
    if (type != ConnectionType::Trivial) {
        // Fail loud rather than silently discard a caller's singularities.
        if (!s.empty())
            throw Error(ErrorCode::InvalidInput, "singularities are only meaningful for the trivial connection.",
                        "Levi-Civita / Euclidean connections take no singularities — pass an empty map.");
        return;
    }
    const int chi = eulerCharacteristic(surface);
    if (s.empty())
        throw Error(ErrorCode::InvalidInput,
            "Trivial connection requires a singularity map; Levi-Civita needs none.",
            "Pass a {vertexIndex -> index} map whose values sum to the Euler characteristic chi = " + std::to_string(chi) + ".");
    double sum = 0.0;
    for (auto& kv : s) sum += kv.second;
    if (std::abs(sum - static_cast<double>(chi)) > 1e-9)
        throw Error(ErrorCode::InvalidInput,
            "Gauss-Bonnet violated: singularity indices sum to " + std::to_string(sum) +
            " but must equal the Euler characteristic chi = " + std::to_string(chi) + ".",
            "A closed genus-0 surface has chi=2 (e.g. two +1 singularities); a disk chi=1.");
}

}  // namespace nxr::geometry
