#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>

namespace nxr::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── geometry::Manifold implementation ────────────────────────────

geometry::Manifold::Manifold(const double* vertices, int nV, const int32_t* faces, int nF, bool intrinsicDelaunay,
                   const geometry::DelaunayOptions& delaunayOptions)
    : geometry::Manifold(vertices, nV, faces, nF,
               geometry::ManifoldOptions{intrinsicDelaunay ? geometry::MeshKind::Intrinsic : geometry::MeshKind::Embedded, delaunayOptions}) {}

geometry::Manifold::Manifold(const double* vertices, int nV, const int32_t* faces, int nF, const geometry::ManifoldOptions& options)
    : options_(options) {
    // Build polygon list from flat face array
    std::vector<std::vector<size_t>> polygons(nF);
    for (int i = 0; i < nF; i++) {
        polygons[i] = {
            static_cast<size_t>(faces[3 * i]),
            static_cast<size_t>(faces[3 * i + 1]),
            static_cast<size_t>(faces[3 * i + 2])
        };
    }

    mesh_ = std::make_unique<ManifoldSurfaceMesh>(polygons);

    VertexData<Vector3> positions(*mesh_);
    for (size_t i = 0; i < static_cast<size_t>(nV); i++) {
        positions[mesh_->vertex(i)] = Vector3{
            vertices[3*i], vertices[3*i+1], vertices[3*i+2]
        };
    }
    geometry_ = std::make_unique<VertexPositionGeometry>(*mesh_, positions);

    // Retain raw input positions for vertexPositions() accessor.
    vertexPositions_.resize(nV, 3);
    for (int i = 0; i < nV; ++i) {
        vertexPositions_(i, 0) = vertices[3 * i + 0];
        vertexPositions_(i, 1) = vertices[3 * i + 1];
        vertexPositions_(i, 2) = vertices[3 * i + 2];
    }

    faces_.resize(nF, 3);
    for (int i = 0; i < nF; ++i) {
        faces_(i, 0) = faces[3 * i + 0];
        faces_(i, 1) = faces[3 * i + 1];
        faces_(i, 2) = faces[3 * i + 2];
    }

    // THE intrinsic surface: geometry-central's signpost intrinsic Delaunay triangulation of the mollified input (or the input as it is).
    geometry::IntrinsicOptions io;
    io.intrinsic = options_.mesh == geometry::MeshKind::Intrinsic;
    io.mollify   = options_.delaunay.mollify;
    io.flipEps   = options_.delaunay.flipEps;
    io.mass      = geometry::MassKind::InputMixedVoronoi;
    surface_ = std::make_unique<geometry::IntrinsicSurface>(vertices, nV, faces, nF, io);
}

// Singularity-aware constructor (pattern 2): delegates to primary ctor,
// then validates Gauss-Bonnet and sets the active gauge to Trivial.
geometry::Manifold::Manifold(const double* vertices, int nV, const int32_t* faces, int nF,
                   const std::map<int,double>& singularities, bool intrinsicDelaunay,
                   const geometry::DelaunayOptions& delaunayOptions)
    : geometry::Manifold(vertices, nV, faces, nF, intrinsicDelaunay, delaunayOptions) {
    validateSingularities_(singularities);
    activeConnectionType_     = geometry::ConnectionType::Trivial;
    activeSingularities_ = singularities;
}

geometry::Manifold::~Manifold() = default;

int geometry::Manifold::nV() const { return static_cast<int>(mesh_->nVertices()); }
int geometry::Manifold::nE() const { return static_cast<int>(mesh_->nEdges()); }
int geometry::Manifold::nF() const { return static_cast<int>(mesh_->nFaces()); }

ManifoldSurfaceMesh& geometry::Manifold::mesh() { return *mesh_; }
VertexPositionGeometry& geometry::Manifold::geometry() { return *geometry_; }

bool geometry::Manifold::isIntrinsicDelaunay() const { return options_.mesh == geometry::MeshKind::Intrinsic; }
geometry::IntrinsicSurface& geometry::Manifold::intrinsicSurface() { return *surface_; }
const geometry::ManifoldOptions& geometry::Manifold::options() const { return options_; }
} // namespace nxr::geometry
