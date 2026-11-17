#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/heat_method_distance.h"

#include <iostream>

namespace nxr::manifold::solve {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// ── Solver impl (PIMPL) ───────────────────────────────────
//
// Owns the geometry-central HeatMethodDistanceSolver plus a
// reference to the Manifold. The solver factorizes both
// Cholesky systems (M + tA, and A) at construction; subsequent
// computeDistance() calls back-substitute only. Mirrors the
// VectorHeatSolver / SignedHeatSolver pattern.

class HeatGeodesicSolverImpl {
public:
    HeatGeodesicSolverImpl(Manifold& c, double tCoef, HeatMesh mesh) : m(c) {
        // geometry-central's heat method on the intrinsic Delaunay triangulation (default; it reads the surface's mass, the input's mixed Voronoi) or on
        // the input mesh (HeatMesh::Input). Vertex indices are the input's in both.
        if (mesh == HeatMesh::Intrinsic) intrinsic = std::make_unique<HeatMethodDistanceSolver>(c.intrinsicSurface().geometry(), tCoef);
        else input = std::make_unique<HeatMethodDistanceSolver>(c.geometry(), tCoef);
    }
    Manifold& m;
    std::unique_ptr<HeatMethodDistanceSolver> intrinsic;   // on the intrinsic Delaunay triangulation (default)
    std::unique_ptr<HeatMethodDistanceSolver> input;       // on the input mesh (HeatMesh::Input)
};

HeatGeodesicSolver::HeatGeodesicSolver(Manifold& m, double tCoef, HeatMesh mesh)
    : impl_(std::make_unique<HeatGeodesicSolverImpl>(m, tCoef, mesh)) {}

HeatGeodesicSolver::~HeatGeodesicSolver() = default;

HeatGeodesicSolverImpl& HeatGeodesicSolver::impl() { return *impl_; }

// ── Free function ─────────────────────────────────────────

Eigen::VectorXd heat(
    HeatGeodesicSolver& solver,
    const std::vector<int>& sourceVertices
) {
    auto& s = solver.impl();
    auto& mesh = s.m.mesh();
    int nV = s.m.nV();

    // Source vertices on the mesh the solver lives on (the intrinsic triangulation's vertices carry the input's indices)
    auto& solverMesh = s.intrinsic ? s.m.intrinsicSurface().mesh() : mesh;
    std::vector<Vertex> sources;
    sources.reserve(sourceVertices.size());
    for (int idx : sourceVertices) {
        sources.push_back(solverMesh.vertex(static_cast<size_t>(idx)));
    }

    // Solve via the cached heat-method solver — back-substitution only
    // on subsequent calls; no per-call factorization.
    VertexData<double> distances = (s.intrinsic ? s.intrinsic : s.input)->computeDistance(sources);

    // Convert to Eigen::VectorXd indexed by vertex index
    Eigen::VectorXd result(nV);
    for (size_t i = 0; i < static_cast<size_t>(nV); ++i) result(static_cast<int>(i)) = distances[i];

    std::cout << "[geodesic] Computed distances from " << sourceVertices.size()
              << " sources. Max distance: " << result.maxCoeff() << std::endl;

    return result;
}

} // namespace nxr::manifold::solve