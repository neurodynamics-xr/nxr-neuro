#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/operators/dec.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <vector>
#include <iostream>

namespace nxr::operators {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Mass-matrix variant dispatch ─────────────────────────────
//
//   MixedVoronoi — diagonal; the recorded mixed Voronoi mass of the mollified INPUT (default).
//   Galerkin     — sparse; the L² integrals of P1 hat functions on the operator mesh as it is.
//   Barycentric  — diagonal; A/3 per vertex on the operator mesh as it is (formerly "Lumped").
// All conserve total area and are symmetric. All are built from the intrinsic surface (nxr/geometry/intrinsic_surface.h).

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

    Eigen::SparseMatrix<double> cotan = cotanLaplacian(m);

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

Eigen::SparseMatrix<double> cotanLaplacian(geometry::Manifold& m) { return m.intrinsicSurface().cotanLaplacian(); }

// geometry-central's DEC of the intrinsic triangulation: d0, d1, ⋆1 (the cotan weights), ⋆2 (1/area) and ⋆0 = vertexDualAreas, which the
// geometry::IntrinsicSurface serves as the input's mixed-Voronoi mass. Edge e is `e.halfedge()` (tail → tip); the vertices are the input's.
DecMatrices decMatrices(geometry::Manifold& m) {
    auto& g = m.intrinsicSurface().geometry();
    g.requireDECOperators();
    DecMatrices dm;
    dm.d0 = g.d0; dm.d1 = g.d1;
    dm.hodge0 = g.hodge0; dm.hodge1 = g.hodge1; dm.hodge2 = g.hodge2; dm.hodge1Inverse = g.hodge1Inverse;
    g.unrequireDECOperators();
    return dm;
}

DECOperators assembleDECOperators(geometry::Manifold& m) {
    const DecMatrices d = decMatrices(m);
    return DECOperators(d.d0, d.d1, d.hodge0, d.hodge1, d.hodge2, d.hodge1Inverse);
}

// The DEC of the INPUT mesh (geometry-central's, embedded): edges and faces are the input's.
DECOperators embeddedDEC(geometry::Manifold& m) {
    auto& g = m.geometry();
    g.requireDECOperators();
    return DECOperators(g.d0, g.d1, g.hodge0, g.hodge1, g.hodge2, g.hodge1Inverse);
}

Eigen::SparseMatrix<double> d0(geometry::Manifold& m)            { return decMatrices(m).d0; }
Eigen::SparseMatrix<double> d1(geometry::Manifold& m)            { return decMatrices(m).d1; }
Eigen::SparseMatrix<double> hodge0(geometry::Manifold& m)        { return decMatrices(m).hodge0; }
Eigen::SparseMatrix<double> hodge1(geometry::Manifold& m)        { return decMatrices(m).hodge1; }
Eigen::SparseMatrix<double> hodge2(geometry::Manifold& m)        { return decMatrices(m).hodge2; }
Eigen::SparseMatrix<double> hodge1Inverse(geometry::Manifold& m) { return decMatrices(m).hodge1Inverse; }
Eigen::VectorXd vertexDualAreas(geometry::Manifold& m) {
    const std::vector<double> b = m.intrinsicSurface().barycentricMass();
    return Eigen::Map<const Eigen::VectorXd>(b.data(), static_cast<Eigen::Index>(b.size()));
}

} // namespace nxr::operators
