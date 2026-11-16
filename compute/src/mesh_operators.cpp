#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>
#include <iostream>

namespace nxr::manifold {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Manifold implementation ────────────────────────────

Manifold::Manifold(const double* vertices, int nV, const int32_t* faces, int nF, bool intrinsicDelaunay,
                   const delaunay::DelaunayOptions& delaunayOptions)
    : Manifold(vertices, nV, faces, nF,
               ManifoldOptions{intrinsicDelaunay ? MeshKind::Intrinsic : MeshKind::Embedded, delaunayOptions}) {}

Manifold::Manifold(const double* vertices, int nV, const int32_t* faces, int nF, const ManifoldOptions& options)
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

    // THE operator mesh: the one Delaunay construction (mollify, signposts, the input's mixed-Voronoi mass, flips) — or the input as it is.
    operatorMesh_ = std::make_unique<delaunay::Mesh>(
        delaunay::prepare(vertices, nV, faces, nF, options_.delaunay, options_.mesh == MeshKind::Intrinsic));
}

// Singularity-aware constructor (pattern 2): delegates to primary ctor,
// then validates Gauss-Bonnet and sets the active gauge to Trivial.
Manifold::Manifold(const double* vertices, int nV, const int32_t* faces, int nF,
                   const std::map<int,double>& singularities, bool intrinsicDelaunay,
                   const delaunay::DelaunayOptions& delaunayOptions)
    : Manifold(vertices, nV, faces, nF, intrinsicDelaunay, delaunayOptions) {
    validateSingularities_(singularities);
    activeGaugeType_     = GaugeType::Trivial;
    activeSingularities_ = singularities;
}

Manifold::~Manifold() = default;

int Manifold::nV() const { return static_cast<int>(mesh_->nVertices()); }
int Manifold::nE() const { return static_cast<int>(mesh_->nEdges()); }
int Manifold::nF() const { return static_cast<int>(mesh_->nFaces()); }

ManifoldSurfaceMesh& Manifold::mesh() { return *mesh_; }
VertexPositionGeometry& Manifold::geometry() { return *geometry_; }

bool Manifold::isIntrinsicDelaunay() const { return options_.mesh == MeshKind::Intrinsic; }
const delaunay::Mesh& Manifold::operatorMesh() const { return *operatorMesh_; }
const ManifoldOptions& Manifold::options() const { return options_; }
const Eigen::SparseMatrix<double>& Manifold::operatorCotan() {
    if (!operatorCotan_) operatorCotan_ = std::make_unique<Eigen::SparseMatrix<double>>(delaunay::cotanLaplacian(*operatorMesh_));
    return *operatorCotan_;
}

const ops::DecMatrices& Manifold::operatorDecMatrices() {
    if (!operatorDec_) {
        const delaunay::Mesh& om = *operatorMesh_;
        const delaunay::Dec d = delaunay::dec(om);
        auto diag = [](const Eigen::VectorXd& v) {
            const int n = static_cast<int>(v.size());
            std::vector<Eigen::Triplet<double>> t;
            t.reserve(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) t.emplace_back(i, i, v(i));
            Eigen::SparseMatrix<double> M(n, n);
            M.setFromTriplets(t.begin(), t.end());
            M.makeCompressed();
            return M;
        };
        auto dm = std::make_unique<ops::DecMatrices>();
        dm->d0 = d.d0; dm->d1 = d.d1;
        dm->hodge0 = diag(Eigen::Map<const Eigen::VectorXd>(om.mass.data(), static_cast<Eigen::Index>(om.mass.size())));
        dm->hodge1 = diag(d.star1);
        dm->hodge2 = diag(d.star2);
        dm->hodge1Inverse = diag(d.star1.cwiseInverse());
        operatorDec_ = std::move(dm);
    }
    return *operatorDec_;
}

const Eigen::VectorXd& Manifold::operatorDualAreas() {
    if (!operatorDualAreas_) {
        const std::vector<double> b = delaunay::barycentricMass(*operatorMesh_);
        operatorDualAreas_ = std::make_unique<Eigen::VectorXd>(Eigen::Map<const Eigen::VectorXd>(b.data(), static_cast<Eigen::Index>(b.size())));
    }
    return *operatorDualAreas_;
}

} // namespace nxr::manifold

namespace nxr::manifold::ops {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Mass-matrix variant dispatch ─────────────────────────────
//
//   MixedVoronoi — diagonal; the recorded mixed Voronoi mass of the mollified INPUT (default).
//   Galerkin     — sparse; the L² integrals of P1 hat functions on the operator mesh as it is.
//   Barycentric  — diagonal; A/3 per vertex on the operator mesh as it is (formerly "Lumped").
// All conserve total area and are symmetric. All are built from the operator mesh (nxr/delaunay.h).

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

Eigen::SparseMatrix<double> vertexMass(Manifold& m, MassMatrixVariant variant) {
    const delaunay::Mesh& mesh = m.operatorMesh();
    switch (variant) {
        case MassMatrixVariant::MixedVoronoi: return diagonalMatrix(mesh.mass);
        case MassMatrixVariant::Barycentric:  return diagonalMatrix(delaunay::barycentricMass(mesh));
        case MassMatrixVariant::Galerkin:     return delaunay::galerkinMass(mesh);
    }
    throw Error(ErrorCode::InvalidInput, "vertexMass: unknown MassMatrixVariant");
}

// ── Operator Assembly ────────────────────────────────────────

ManifoldOperators assembleManifoldOperators(Manifold& m,
                                    MassMatrixVariant variant) {
    auto& mesh = m.mesh();
    // The operator mesh: the one Delaunay construction (intrinsic mollified Delaunay by default, the input with MeshKind::Embedded).
    const delaunay::Mesh& opMesh = m.operatorMesh();
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
    const std::vector<double> bary = delaunay::barycentricMass(opMesh);
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

DECOperators assembleDECOperators(Manifold& m) {
    const DecMatrices& d = m.operatorDecMatrices();
    return DECOperators(d.d0, d.d1, d.hodge0, d.hodge1, d.hodge2, d.hodge1Inverse);
}

// ── Zero-copy passthrough accessors ──────────────────────────
// Bit-identical to the matching field on the struct returned by assembleDECOperators.

const Eigen::SparseMatrix<double>& d0(Manifold& m)            { return m.operatorDecMatrices().d0; }
const Eigen::SparseMatrix<double>& d1(Manifold& m)            { return m.operatorDecMatrices().d1; }
const Eigen::SparseMatrix<double>& hodge0(Manifold& m)        { return m.operatorDecMatrices().hodge0; }
const Eigen::SparseMatrix<double>& hodge1(Manifold& m)        { return m.operatorDecMatrices().hodge1; }
const Eigen::SparseMatrix<double>& hodge2(Manifold& m)        { return m.operatorDecMatrices().hodge2; }
const Eigen::SparseMatrix<double>& hodge1Inverse(Manifold& m) { return m.operatorDecMatrices().hodge1Inverse; }
const Eigen::VectorXd&             vertexDualAreas(Manifold& m) { return m.operatorDualAreas(); }

} // namespace nxr::manifold::ops