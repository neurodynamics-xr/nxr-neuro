#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/transport.h"
#include "nxr/operators/factorization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"

#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/vector2.h"

#include <Eigen/SparseCore>

#include <complex>
#include <vector>

namespace nxr::operators::laplacian::connection {

using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace {

using ComplexTriplet = Eigen::Triplet<std::complex<double>>;

// Vertex connection Laplacian on the n-RoSy bundle: the geometry::IntrinsicSurface's (geometry-central's `vertexConnectionLaplacian` for n = 1; for n > 1 its
// transport raised to the n-th power with its edge cotangent weights), on the geometry::Manifold's intrinsic triangulation — in geometry-central's gauge. The
// connection lifts the scalar Laplacian's cotangent weight to the complex tangent bundle.
void assembleVertexCL(geometry::IntrinsicSurface& surface, int nSym, std::vector<ComplexTriplet>& out) {
    const Eigen::SparseMatrix<std::complex<double>> K = surface.connectionLaplacian(nSym);
    out.reserve(static_cast<size_t>(K.nonZeros()));
    for (int c = 0; c < K.outerSize(); ++c)
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(K, c); it; ++it)
            out.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
}

// Face connection Laplacian on the n-RoSy bundle.
//
// Mirrors the file-static `computeFaceConnectionLaplacian` in
// `geometry-central/src/surface/direction_fields.cpp:52`. Inherits
// geometry-central's FIXME on face weights (uniform 1.0). The
// numerical answer is still well-defined for nRoSy energy
// minimization; revisit if a more principled face weight emerges
// upstream.
void assembleFaceCL(VertexPositionGeometry& geometry,
                    int nSym,
                    std::vector<ComplexTriplet>& out) {
    SurfaceMesh& mesh = geometry.mesh;

    geometry.requireFaceIndices();
    geometry.requireTransportVectorsAcrossHalfedge();

    out.reserve(4 * mesh.nFaces());
    for (Face f : mesh.faces()) {
        const size_t i = geometry.faceIndices[f];

        std::complex<double> weightDiagSum(0.0, 0.0);
        for (Halfedge he : f.adjacentHalfedges()) {
            if (!he.twin().isInterior()) continue;

            const Face   neigh = he.twin().face();
            const size_t j     = geometry.faceIndices[neigh];

            const Vector2 rot = geometry.transportVectorsAcrossHalfedge[he.twin()].pow(nSym);

            const double weight = 1.0;
            out.emplace_back(i, j, std::complex<double>(-weight * rot.x,
                                                        -weight * rot.y));
            weightDiagSum += weight;
        }

        out.emplace_back(i, i, weightDiagSum);
    }
}

// Crouzeix-Raviart connection Laplacian on the n-RoSy bundle.
//
// Forks `IntrinsicGeometryInterface::computeCrouzeixRaviartConnectionLaplacian`
// at `geometry-central/src/surface/intrinsic_geometry_interface.cpp:639`,
// raising the signed connection rotation `s_ij · rot_ij` to the
// nSym power. For nSym=1 the matrix matches geometry-central's
// `crouzeixRaviartConnectionLaplacian` property exactly; for even
// nSym the orientation sign drops out (line / cross fields are
// orientation-agnostic on the n-RoSy bundle).
//
// Triangle-only — geometry-central's own assertion.
void assembleEdgeCRCL(VertexPositionGeometry& geometry,
                      int nSym,
                      std::vector<ComplexTriplet>& out) {
    SurfaceMesh& mesh = geometry.mesh;

    geometry.requireEdgeIndices();
    geometry.requireHalfedgeCotanWeights();
    geometry.requireFaceAreas();
    geometry.requireEdgeLengths();

    out.reserve(12 * mesh.nFaces());
    for (Face f : mesh.faces()) {
        for (Halfedge he : f.adjacentHalfedges()) {
            const Halfedge heA = he.next();
            const Halfedge heB = heA.next();
            if (heB.next() != he) {
                throw Error(ErrorCode::InvalidInput,
                            "Connection Laplacian (Crouzeix-Raviart) requires a triangle mesh",
                            "Face " + std::to_string(f.getIndex()) +
                            " is not a triangle. Use the vertex or face domain instead.");
            }

            const size_t iE_i = geometry.edgeIndices[heA.edge()];
            const size_t iE_j = geometry.edgeIndices[heB.edge()];

            const double weight =  4.0 * geometry.halfedgeCotanWeights[he];
            const double s_ij   = (heA.orientation() == heB.orientation()) ? 1.0 : -1.0;
            const double lOpp   = geometry.edgeLengths[he.edge()];
            const double lA     = geometry.edgeLengths[heB.edge()];
            const double lB     = geometry.edgeLengths[heA.edge()];
            const double cosT   = (lA * lA + lB * lB - lOpp * lOpp) / (2.0 * lA * lB);
            const double sinT   = 2.0 * geometry.faceAreas[f] / (lA * lB);

            // signed_rot = s_ij · (-cosT + i·sinT) — unit complex on the
            // n-RoSy bundle. Raising to nSym applies the connection rotation
            // n times in the line / cross / k-RoSy field.
            const std::complex<double> signed_rot(s_ij * (-cosT), s_ij * sinT);
            const std::complex<double> rot_n = std::pow(signed_rot, nSym);

            out.emplace_back(iE_i, iE_i, std::complex<double>(weight, 0.0));
            out.emplace_back(iE_j, iE_j, std::complex<double>(weight, 0.0));
            out.emplace_back(iE_i, iE_j, -weight * rot_n);
            out.emplace_back(iE_j, iE_i, -weight * std::conj(rot_n));
        }
    }
}

// Lower a complex Hermitian matrix to its 2N × 2N symmetric real
// representation: each entry a + bi maps to the 2 × 2 block
// [[a, -b], [b, a]] placed at (i, j), (i, j+N), (i+N, j), (i+N, j+N).
// For a Hermitian K, the result is symmetric real and positive
// (semi-)definite iff K is.
Eigen::SparseMatrix<double> lowerToReal2N(
    const Eigen::SparseMatrix<std::complex<double>>& K_complex,
    int N
) {
    std::vector<Eigen::Triplet<double>> real_triplets;
    real_triplets.reserve(4 * K_complex.nonZeros());

    for (int k = 0; k < K_complex.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(K_complex, k);
             it; ++it) {
            const int    i = static_cast<int>(it.row());
            const int    j = static_cast<int>(it.col());
            const double a = it.value().real();
            const double b = it.value().imag();
            real_triplets.emplace_back(i,     j,     a);
            real_triplets.emplace_back(i + N, j + N, a);
            real_triplets.emplace_back(i,     j + N, -b);
            real_triplets.emplace_back(i + N, j,     b);
        }
    }

    Eigen::SparseMatrix<double> K_real(2 * N, 2 * N);
    K_real.setFromTriplets(real_triplets.begin(), real_triplets.end());
    return K_real;
}

} // namespace

ConnectionLaplacian assembleConnectionLaplacian(
    const geometry::SurfaceArrays& surface,
    const ConnectionLaplacianOptions& opts
) {
    geometry::IntrinsicSurface gs(surface);
    if (opts.nSym <= 0) {
        throw Error(ErrorCode::InvalidInput,
                    "Connection Laplacian: nSym must be positive (got " +
                    std::to_string(opts.nSym) + ")",
                    "Common values: 1 (vector field), 2 (line field), 4 (cross field).");
    }

    auto& geometry = gs.embedded();

    std::vector<ComplexTriplet> triplets;
    int N = 0;
    switch (opts.domain) {
        case ConnectionDomain::Vertex:
            // The intrinsic surface's cotan weights and transport (the signpost intrinsic Delaunay triangulation by default; geometry::MeshKind::Embedded: the input).
            N = surface.nV();
            assembleVertexCL(gs, opts.nSym, triplets);
            break;
        case ConnectionDomain::Face:
            N = surface.nF();
            assembleFaceCL(geometry, opts.nSym, triplets);
            break;
        case ConnectionDomain::EdgeCrouzeixRaviart:
            N = static_cast<int>(gs.inputMesh().nEdges());
            assembleEdgeCRCL(geometry, opts.nSym, triplets);
            break;
    }

    // Add ε·I diagonal regularization.
    if (opts.regularization != 0.0) {
        for (int i = 0; i < N; ++i) {
            triplets.emplace_back(i, i, std::complex<double>(opts.regularization, 0.0));
        }
    }

    Eigen::SparseMatrix<std::complex<double>> K_complex(N, N);
    K_complex.setFromTriplets(triplets.begin(), triplets.end());

    ConnectionLaplacian result;
    result.baseDim         = N;
    result.domain          = opts.domain;
    result.nSym            = opts.nSym;
    result.regularization  = opts.regularization;
    result.format          = opts.format;

    // Populate tangent frame (gauge) used to build this Laplacian.
    // Eigenvectors U[:,k] satisfy: field at v = real(U[v,k])*e1(v) + imag(U[v,k])*e2(v).
    //
    // The vertex frame is geometry-central's gauge (geometry::IntrinsicSurface::tangentBasis): n the corner-angle-weighted normal, e1 the average of the outgoing
    // edges each rotated back by its own rescaled angle (angle 0 = the input's v.halfedge()), e2 = n × e1.
    if (opts.domain == ConnectionDomain::Vertex) {
        const geometry::VertexFrames vf = geometry::vertexFrames(surface);
        result.frameE1 = vf.e1;
        result.frameE2 = vf.e2;
    } else if (opts.domain == ConnectionDomain::Face) {
        geometry.requireFaceTangentBasis();
        geometry.requireFaceIndices();
        SurfaceMesh& mesh = geometry.mesh;
        result.frameE1.resize(N, 3);
        result.frameE2.resize(N, 3);
        for (Face f : mesh.faces()) {
            int fi = static_cast<int>(geometry.faceIndices[f]);
            auto b0 = geometry.faceTangentBasis[f][0];
            auto b1 = geometry.faceTangentBasis[f][1];
            result.frameE1(fi, 0) = b0.x; result.frameE1(fi, 1) = b0.y; result.frameE1(fi, 2) = b0.z;
            result.frameE2(fi, 0) = b1.x; result.frameE2(fi, 1) = b1.y; result.frameE2(fi, 2) = b1.z;
        }
    }
    // EdgeCrouzeixRaviart: no standard tangent basis — frameE1/E2 remain empty (0 rows).

    if (opts.format == ConnectionLaplacianFormat::Real2N) {
        result.K_real    = lowerToReal2N(K_complex, N);
        result.outputDim = 2 * N;
    } else {
        result.K_complex = std::move(K_complex);
        result.outputDim = N;
    }
    return result;
}

ConnectionDomain parseConnectionDomain(const std::string& name) {
    if (name == "vertex") return ConnectionDomain::Vertex;
    if (name == "face")   return ConnectionDomain::Face;
    if (name == "edge" || name == "crouzeix-raviart" || name == "cr")
        return ConnectionDomain::EdgeCrouzeixRaviart;
    throw Error(ErrorCode::InvalidInput,
                "Unknown connection-Laplacian domain '" + name +
                "' — expected 'vertex', 'face', or 'edge'.");
}

ConnectionLaplacianFormat parseConnectionLaplacianFormat(const std::string& name) {
    if (name == "real2N" || name == "real2n" || name == "real")
        return ConnectionLaplacianFormat::Real2N;
    if (name == "complex")
        return ConnectionLaplacianFormat::Complex;
    throw Error(ErrorCode::InvalidInput,
                "Unknown connection-Laplacian format '" + name +
                "' — expected 'real2N' or 'complex'.");
}

ConnectionLaplacian assembleTrivialConnectionLaplacian(
    const geometry::SurfaceArrays& surface,
    const std::map<int, double>& singularityMap,
    const nxr::operators::DECOperators& dec,
    const nxr::operators::CholeskyFactor& exactFactor,
    const ConnectionLaplacianOptions& opts
) {
    geometry::IntrinsicSurface gs(surface);
    if (opts.domain != ConnectionDomain::Vertex) {
        throw Error(ErrorCode::InvalidInput,
            "assembleTrivialConnectionLaplacian: only ConnectionDomain::Vertex "
            "is currently supported.",
            "Face and EdgeCrouzeixRaviart trivial-connection Laplacians are "
            "not yet implemented. Use domain='vertex'.");
    }
    if (opts.nSym <= 0) {
        throw Error(ErrorCode::InvalidInput,
            "assembleTrivialConnectionLaplacian: nSym must be positive (got " +
            std::to_string(opts.nSym) + ")",
            "Common values: 1 (vector field), 2 (line field), 4 (cross field).");
    }

    Eigen::VectorXd phi =
        nxr::operators::connection::computeTrivialConnection(surface, dec, exactFactor, singularityMap);

    auto& geometry = gs.embedded();
    SurfaceMesh& mesh = gs.inputMesh();
    int N = surface.nV();

    geometry.requireVertexIndices();
    geometry.requireEdgeCotanWeights();
    geometry.requireTransportVectorsAlongHalfedge();
    geometry.requireEdgeIndices();

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(2 * mesh.nHalfedges());

    for (Halfedge he : mesh.halfedges()) {
        const size_t iTail  = geometry.vertexIndices[he.vertex()];
        const size_t iTip   = geometry.vertexIndices[he.next().vertex()];
        const double weight = geometry.edgeCotanWeights[he.edge()];
        const int    eIdx   = static_cast<int>(geometry.edgeIndices[he.edge()]);

        const double signTwin = (he.twin() == he.edge().halfedge()) ? 1.0 : -1.0;
        const Vector2 lcRotBase = geometry.transportVectorsAlongHalfedge[he.twin()];
        const std::complex<double> lcRotC(lcRotBase.x, lcRotBase.y);
        const std::complex<double> correction = std::polar(1.0, signTwin * phi(eIdx));
        const std::complex<double> tcRotC =
            std::pow(lcRotC * correction, static_cast<double>(opts.nSym));

        triplets.emplace_back(iTail, iTail, std::complex<double>(weight, 0.0));
        triplets.emplace_back(iTail, iTip,
            std::complex<double>(-weight * tcRotC.real(), -weight * tcRotC.imag()));
    }

    if (opts.regularization != 0.0) {
        for (int i = 0; i < N; ++i)
            triplets.emplace_back(i, i, std::complex<double>(opts.regularization, 0.0));
    }

    Eigen::SparseMatrix<std::complex<double>> K_complex(N, N);
    K_complex.setFromTriplets(triplets.begin(), triplets.end());

    ConnectionLaplacian result;
    result.baseDim        = N;
    result.domain         = opts.domain;
    result.nSym           = opts.nSym;
    result.regularization = opts.regularization;
    result.format         = opts.format;

    // Populate tangent frame (gauge). Trivial connection Laplacian is vertex-only.
    geometry.requireVertexTangentBasis();
    result.frameE1.resize(N, 3);
    result.frameE2.resize(N, 3);
    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(geometry.vertexIndices[v]);
        auto b0 = geometry.vertexTangentBasis[v][0];
        auto b1 = geometry.vertexTangentBasis[v][1];
        result.frameE1(vi, 0) = b0.x; result.frameE1(vi, 1) = b0.y; result.frameE1(vi, 2) = b0.z;
        result.frameE2(vi, 0) = b1.x; result.frameE2(vi, 1) = b1.y; result.frameE2(vi, 2) = b1.z;
    }

    if (opts.format == ConnectionLaplacianFormat::Real2N) {
        result.K_real    = lowerToReal2N(K_complex, N);
        result.outputDim = 2 * N;
    } else {
        result.K_complex = std::move(K_complex);
        result.outputDim = N;
    }
    return result;
}

Eigen::SparseMatrix<std::complex<double>>
assembleConnectionGradient(const geometry::SurfaceArrays& surface, int nSym) {
    geometry::IntrinsicSurface gs(surface);
    if (nSym <= 0)
        throw Error(ErrorCode::InvalidInput,
                    "assembleConnectionGradient: nSym must be > 0",
                    "Common values: 1 (vector field), 2 (line field).");

    // v1: Levi-Civita gauge only. The intrinsic surface's transport does not vary with the active gauge, so this gradient is gauge-independent and
    // squares to leviCivitaConnectionLaplacian.
    auto& g = gs.geometry();
    const int E = static_cast<int>(g.mesh.nEdges());
    const int V = surface.nV();
    g.requireVertexIndices();
    g.requireTransportVectorsAlongHalfedge();

    // Per edge e = (i→j) via its halfedge he = e.halfedge():
    //   D[e, j] = +1
    //   D[e, i] = −ρ   where ρ = conj(r^n), r = the unit transport along the twin halfedge (the one the vertex connection Laplacian's (i, j) entry uses)
    // Identity: (Dᴴ ⋆₁ D)[i,j] = w · conj(−ρ) = −w r^n = the vertex connection Laplacian's entry.
    std::vector<Eigen::Triplet<std::complex<double>>> T;
    T.reserve(2 * static_cast<size_t>(E));

    for (Edge e : g.mesh.edges()) {
        const Halfedge he = e.halfedge();
        const int i = static_cast<int>(g.vertexIndices[he.vertex()]), j = static_cast<int>(g.vertexIndices[he.next().vertex()]);
        const Vector2 r = g.transportVectorsAlongHalfedge[he.twin()];
        std::complex<double> rn(1.0, 0.0);
        for (int k = 0; k < nSym; ++k) rn *= std::complex<double>(r.x, r.y);
        T.emplace_back(static_cast<int>(e.getIndex()), j, std::complex<double>(1.0, 0.0));
        T.emplace_back(static_cast<int>(e.getIndex()), i, -std::conj(rn));
    }
    g.unrequireTransportVectorsAlongHalfedge();
    g.unrequireVertexIndices();

    Eigen::SparseMatrix<std::complex<double>> D(E, V);
    D.setFromTriplets(T.begin(), T.end());
    D.makeCompressed();
    return D;
}

} // namespace nxr::operators::laplacian::connection