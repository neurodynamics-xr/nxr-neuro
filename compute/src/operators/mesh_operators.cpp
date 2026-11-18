#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/operators/facets.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>
#include <iostream>

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
const Eigen::SparseMatrix<double>& geometry::Manifold::operatorCotan() {
    if (!operatorCotan_) operatorCotan_ = std::make_unique<Eigen::SparseMatrix<double>>(surface_->cotanLaplacian());
    return *operatorCotan_;
}

const nxr::operators::DecMatrices& geometry::Manifold::operatorDecMatrices() {
    if (!operatorDec_) {
        // geometry-central's DEC of the intrinsic triangulation: d0, d1, ⋆1 (the cotan weights), ⋆2 (1/area) and ⋆0 = vertexDualAreas, which the
        // geometry::IntrinsicSurface serves as the input's mixed-Voronoi mass. Edge e is `e.halfedge()` (tail → tip); the vertices are the input's.
        auto& g = surface_->geometry();
        g.requireDECOperators();
        auto dm = std::make_unique<nxr::operators::DecMatrices>();
        dm->d0 = g.d0; dm->d1 = g.d1;
        dm->hodge0 = g.hodge0; dm->hodge1 = g.hodge1; dm->hodge2 = g.hodge2; dm->hodge1Inverse = g.hodge1Inverse;
        g.unrequireDECOperators();
        operatorDec_ = std::move(dm);
    }
    return *operatorDec_;
}

const Eigen::VectorXd& geometry::Manifold::operatorDualAreas() {
    if (!operatorDualAreas_) {
        const std::vector<double> b = surface_->barycentricMass();
        operatorDualAreas_ = std::make_unique<Eigen::VectorXd>(Eigen::Map<const Eigen::VectorXd>(b.data(), static_cast<Eigen::Index>(b.size())));
    }
    return *operatorDualAreas_;
}

} // namespace nxr::geometry

namespace nxr::operators {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Mass-matrix variant dispatch ─────────────────────────────
//
//   MixedVoronoi — diagonal; the recorded mixed Voronoi mass of the mollified INPUT (default).
//   Galerkin     — sparse; the L² integrals of P1 hat functions on the operator mesh as it is.
//   Barycentric  — diagonal; A/3 per vertex on the operator mesh as it is (formerly "Lumped").
// All conserve total area and are symmetric. All are built from the intrinsic surface (nxr/intrinsic_surface.h).

static Eigen::SparseMatrix<double> diagonalMatrix(const std::vector<double>& d) {
    const int n = static_cast<int>(d.size());
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(d.size());
    for (int i = 0; i < n; ++i) t.emplace_back(i, i, d[static_cast<size_t>(i)]);
    Eigen::SparseMatrix<double> M(n, n);
    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();
    return M;
}

Eigen::SparseMatrix<double> vertexMass(geometry::Manifold& m, MassMatrixVariant variant) {
    geometry::IntrinsicSurface& surf = m.intrinsicSurface();
    switch (variant) {
        case MassMatrixVariant::MixedVoronoi: return diagonalMatrix(surf.massDiagonal());
        case MassMatrixVariant::Barycentric:  return diagonalMatrix(surf.barycentricMass());
        case MassMatrixVariant::Galerkin:     return surf.galerkinMassMatrix();
    }
    throw Error(ErrorCode::InvalidInput, "vertexMass: unknown MassMatrixVariant");
}

// ── Operator Assembly ────────────────────────────────────────

ManifoldOperators assembleManifoldOperators(geometry::Manifold& m,
                                    MassMatrixVariant variant) {
    auto& mesh = m.mesh();
    // embGeom — always the embedded VertexPositionGeometry. Used for vertex normals (extrinsic quantity).
    auto& embGeom = m.geometry();

    int nV = m.nV();
    int nE = m.nE();
    int nF = m.nF();

    const Eigen::SparseMatrix<double>& cotan = m.operatorCotan();

    Eigen::SparseMatrix<double> mass = vertexMass(m, variant);
    const double totalArea = (variant == MassMatrixVariant::Galerkin) ? mass.sum() : mass.diagonal().sum();

    // vertexDualAreas reports the per-vertex A/3 dual area regardless of the chosen mass variant — downstream consumers (curvature,
    // gradients, diffusion) want a per-vertex weight even when the L² inner product uses another M.
    const std::vector<double> bary = m.intrinsicSurface().barycentricMass();
    Eigen::VectorXd vertexDualAreas(nV);
    for (int i = 0; i < nV; ++i) vertexDualAreas(i) = bary[static_cast<size_t>(i)];

    // Vertex normals — extrinsic quantity; sourced from the embedded VertexPositionGeometry. V×3, stored row-major (VertexNormalsMatrix) so the
    // output buffer at the binding edge is a direct memcpy.
    embGeom.requireVertexNormals();
    VertexNormalsMatrix vertexNormals(nV, 3);
    for (Vertex v : mesh.vertices()) {
        int idx = static_cast<int>(v.getIndex());
        Vector3 n = embGeom.vertexNormals[v];
        vertexNormals(idx, 0) = n.x;
        vertexNormals(idx, 1) = n.y;
        vertexNormals(idx, 2) = n.z;
    }

    const char* variantName =
        variant == MassMatrixVariant::MixedVoronoi ? "mixedVoronoi" :
        variant == MassMatrixVariant::Barycentric  ? "barycentric"  :
        variant == MassMatrixVariant::Galerkin     ? "galerkin"     : "?";
    std::cout << "[mesh_operators] Assembled: "
              << nV << " V, " << nE << " E, " << nF << " F "
              << "(total area = " << totalArea
              << ", mass = " << variantName
              << ", mass nnz = " << mass.nonZeros() << ")" << std::endl;

    return ManifoldOperators(
        cotan,
        std::move(mass),
        variant,
        std::move(vertexDualAreas),
        std::move(vertexNormals),
        totalArea
    );
}

MassMatrixVariant parseMassMatrixVariant(const std::string& name) {
    if (name == "mixedVoronoi") return MassMatrixVariant::MixedVoronoi;
    if (name == "galerkin")     return MassMatrixVariant::Galerkin;
    if (name == "barycentric")  return MassMatrixVariant::Barycentric;
    throw Error(ErrorCode::InvalidInput,
        "Unknown mass-matrix variant '" + name +
        "' — expected 'mixedVoronoi' (the default), 'galerkin' or 'barycentric' ('lumped' is retired).");
}

// ── DEC Operators ────────────────────────────────────────────

DECOperators assembleDECOperators(geometry::Manifold& m) {
    const DecMatrices& d = m.operatorDecMatrices();
    return DECOperators(d.d0, d.d1, d.hodge0, d.hodge1, d.hodge2, d.hodge1Inverse);
}

// ── Zero-copy passthrough accessors ──────────────────────────
// Bit-identical to the matching field on the struct returned by assembleDECOperators.

const Eigen::SparseMatrix<double>& d0(geometry::Manifold& m)            { return m.operatorDecMatrices().d0; }
const Eigen::SparseMatrix<double>& d1(geometry::Manifold& m)            { return m.operatorDecMatrices().d1; }
const Eigen::SparseMatrix<double>& hodge0(geometry::Manifold& m)        { return m.operatorDecMatrices().hodge0; }
const Eigen::SparseMatrix<double>& hodge1(geometry::Manifold& m)        { return m.operatorDecMatrices().hodge1; }
const Eigen::SparseMatrix<double>& hodge2(geometry::Manifold& m)        { return m.operatorDecMatrices().hodge2; }
const Eigen::SparseMatrix<double>& hodge1Inverse(geometry::Manifold& m) { return m.operatorDecMatrices().hodge1Inverse; }
const Eigen::VectorXd&             vertexDualAreas(geometry::Manifold& m) { return m.operatorDualAreas(); }

} // namespace nxr::operators