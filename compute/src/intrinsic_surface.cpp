// nxr::manifold::IntrinsicSurface: see nxr/intrinsic_surface.h and docs/specs/2026-11-17-geometry-central-switch.md.
#include "nxr/intrinsic_surface.h"

#include "nxr/errors.h"

#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/intrinsic_mollification.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/signpost_intrinsic_triangulation.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifndef NXR_GEOMETRY_CENTRAL_COMMIT
#define NXR_GEOMETRY_CENTRAL_COMMIT "unknown"
#endif

namespace nxr::manifold {

namespace gcs = geometrycentral::surface;
namespace gc = geometrycentral;
using core::Error;
using core::ErrorCode;

const char* geometryCentralVersion() { return NXR_GEOMETRY_CENTRAL_COMMIT; }

namespace {

// geometry-central's extension point: the dual area of each vertex is virtual, so the chosen mass reaches `vertexDualAreas`, `vertexLumpedMassMatrix`, `hodge0`
// and the heat solvers with no other change. `mass == nullptr` leaves geometry-central's own (barycentric).
template <class Base>
class MassServing : public Base {
public:
    template <class... A>
    explicit MassServing(const std::vector<double>* mass, A&&... a) : Base(std::forward<A>(a)...), mass_(mass) {}
protected:
    void computeVertexDualAreas() override {
        if (mass_ == nullptr || mass_->empty()) { Base::computeVertexDualAreas(); return; }
        this->vertexDualAreas = gcs::VertexData<double>(this->mesh);
        for (size_t i = 0; i < mass_->size(); ++i) this->vertexDualAreas[i] = (*mass_)[i];
    }
private:
    const std::vector<double>* mass_;
};

using Embedded = MassServing<gcs::VertexPositionGeometry>;
using Signpost = MassServing<gcs::SignpostIntrinsicTriangulation>;

const char* massName(MassKind k) {
    switch (k) {
        case MassKind::InputMixedVoronoi: return "input-mixed-voronoi";
        case MassKind::Barycentric:       return "barycentric";
        case MassKind::Galerkin:          return "galerkin";
    }
    return "";
}

std::string hex8(uint32_t x) {
    char b[9];
    std::snprintf(b, sizeof b, "%08x", x);
    return b;
}

}  // namespace

struct IntrinsicSurface::Impl {
    IntrinsicOptions opt;
    IntrinsicRecord rec;
    std::unique_ptr<gcs::ManifoldSurfaceMesh> mesh;
    std::vector<double> mass;                                  // the input mixed-Voronoi mass (empty unless that option)
    std::unique_ptr<Embedded> embedded;
    std::unique_ptr<gcs::EdgeLengthGeometry> inputGeom;        // the mollified input (intrinsic mode only)
    std::unique_ptr<Signpost> tri;                             // intrinsic mode only
    int nV = 0, nF = 0;

    gcs::IntrinsicGeometryInterface& input() { return inputGeom ? static_cast<gcs::IntrinsicGeometryInterface&>(*inputGeom) : *embedded; }
    gcs::IntrinsicGeometryInterface& geom() { return tri ? static_cast<gcs::IntrinsicGeometryInterface&>(*tri) : *embedded; }
};

namespace {

std::vector<double> mixedVoronoi(gcs::ManifoldSurfaceMesh& mesh, gcs::IntrinsicGeometryInterface& g) {
    g.requireEdgeLengths();
    g.requireFaceAreas();
    g.requireCornerAngles();
    std::vector<double> M(mesh.nVertices(), 0.0);
    for (gcs::Face f : mesh.faces()) {
        gcs::Halfedge h[3] = {f.halfedge(), f.halfedge().next(), f.halfedge().next().next()};
        double l[3], a[3];
        for (int k = 0; k < 3; ++k) { l[k] = g.edgeLengths[h[k].edge()]; a[k] = g.cornerAngles[h[k].corner()]; }
        const double A = g.faceAreas[f];
        int obtuse = -1;
        for (int k = 0; k < 3; ++k) if (a[k] > M_PI / 2) { obtuse = k; break; }
        if (obtuse >= 0) {
            for (int k = 0; k < 3; ++k) M[h[k].vertex().getIndex()] += (k == obtuse ? A / 2 : A / 4);
            continue;
        }
        double cot[3];
        for (int k = 0; k < 3; ++k) cot[k] = std::cos(a[k]) / std::max(1e-150, std::sin(a[k]));
        // h[k] runs from corner k to corner k+1 and is opposite corner k+2: corner k's Voronoi part is its two edges, each |e|^2 cot(opposite angle) / 8
        for (int k = 0; k < 3; ++k)
            M[h[k].vertex().getIndex()] += (l[k] * l[k] * cot[(k + 2) % 3] + l[(k + 2) % 3] * l[(k + 2) % 3] * cot[(k + 1) % 3]) / 8;
    }
    return M;
}

}  // namespace

IntrinsicSurface::IntrinsicSurface(const double* positions, int nV, const int32_t* faces, int nF, const IntrinsicOptions& o) : p_(new Impl) {
    Impl& P = *p_;
    P.opt = o;
    P.nV = nV; P.nF = nF;
    if (nV < 0 || nF < 0) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface: negative size");
    std::vector<std::vector<size_t>> polys(static_cast<size_t>(nF));
    for (int f = 0; f < nF; ++f) {
        for (int k = 0; k < 3; ++k) {
            const int32_t v = faces[3 * f + k];
            if (v < 0 || v >= nV) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface: a face index is outside [0, nV)");
            polys[static_cast<size_t>(f)].push_back(static_cast<size_t>(v));
        }
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> pos(nV, 3);
    for (int v = 0; v < nV; ++v) for (int c = 0; c < 3; ++c) pos(v, c) = positions[3 * v + c];
    try {
        P.mesh.reset(new gcs::ManifoldSurfaceMesh(polys));
    } catch (const std::exception& e) {
        throw Error(ErrorCode::InvalidInput, std::string("IntrinsicSurface: the input is not an oriented manifold triangle mesh: ") + e.what());
    }
    if (static_cast<int>(P.mesh->nVertices()) != nV) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface: a vertex belongs to no face");

    const bool wantInputMass = (o.mass == MassKind::InputMixedVoronoi);
    // 1. the input's embedded geometry (not mollified): positions, normals
    P.embedded.reset(new Embedded(nullptr, *P.mesh, pos));
    gcs::EdgeData<double> len(*P.mesh);
    P.embedded->requireEdgeLengths();
    for (gcs::Edge e : P.mesh->edges()) {
        const double l = P.embedded->edgeLengths[e];
        if (!std::isfinite(l) || !(l > 0)) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface: an input edge length is not finite and positive");
        len[e] = l;
    }

    P.rec.mesh = o.intrinsic ? "intrinsic" : "embedded";
    P.rec.mass = massName(o.mass);
    P.rec.geometryCentral = geometryCentralVersion();
    for (gcs::Vertex v : P.mesh->vertices()) if (v.isBoundary()) ++P.rec.boundaryVertices;

    if (!o.intrinsic) {
        // mesh = embedded: every operator is the input geometry's
        if (wantInputMass) { P.mass = mixedVoronoi(*P.mesh, *P.embedded); P.embedded.reset(new Embedded(&P.mass, *P.mesh, pos)); }
        P.rec.flipEps = 0.0; P.rec.mollify = 0.0;
    } else {
        // 2. mollify the INPUT lengths, before the signposts are laid out
        gcs::EdgeData<double> before = len;
        gcs::mollifyIntrinsic(*P.mesh, len, o.mollify);
        double eps = 0.0;
        for (gcs::Edge e : P.mesh->edges()) eps = std::max(eps, len[e] - before[e]);
        P.rec.mollify = o.mollify; P.rec.epsilon = eps;
        P.inputGeom.reset(new gcs::EdgeLengthGeometry(*P.mesh, len));
        // 3. the mass of the mollified input, before any flip
        if (wantInputMass) P.mass = mixedVoronoi(*P.mesh, *P.inputGeom);
        // 4. the signposts, laid out from the mollified input
        P.tri.reset(new Signpost(wantInputMass ? &P.mass : nullptr, *P.mesh, *P.inputGeom));
        // 5. the flips
        P.tri->triangleTestEPS = o.flipEps;
        int flips = 0;
        P.tri->edgeFlipCallbackList.push_back([&flips](gcs::Edge) { ++flips; });
        P.tri->flipToDelaunay();
        P.tri->edgeFlipCallbackList.pop_back();
        P.rec.flipEps = o.flipEps; P.rec.flips = flips;
        P.rec.delaunay = P.tri->isDelaunay();
        // every length stays positive and finite (geometry-central refuses only a non-finite one)
        P.tri->requireEdgeLengths();
        for (gcs::Edge e : P.tri->intrinsicMesh->edges()) {
            const double l = P.tri->edgeLengths[e];
            if (!std::isfinite(l) || !(l > 0)) throw Error(ErrorCode::GeometryDegenerate, "IntrinsicSurface: a flip produced an edge length that is not finite and positive");
        }
    }

    // 6. the gauge identity: a hash of each vertex's reference neighbour and ε
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t x) { for (int i = 0; i < 8; ++i) { h ^= (x >> (8 * i)) & 0xff; h *= 1099511628211ull; } };
    for (gcs::Vertex v : P.mesh->vertices()) mix(v.halfedge().tipVertex().getIndex());
    uint64_t bits; std::memcpy(&bits, &P.rec.epsilon, sizeof bits); mix(bits);
    P.rec.fingerprint = hex8(static_cast<uint32_t>(h ^ (h >> 32)));
    P.rec.gauge = std::string("geometry-central@") + P.rec.geometryCentral + "/vertex";
    P.rec.gaugeId = P.rec.gauge + ":" + o.surface + ":" + P.rec.fingerprint;
}

IntrinsicSurface::~IntrinsicSurface() = default;

int IntrinsicSurface::nV() const { return p_->nV; }
int IntrinsicSurface::nF() const { return p_->nF; }
const IntrinsicOptions& IntrinsicSurface::options() const { return p_->opt; }
const IntrinsicRecord& IntrinsicSurface::record() const { return p_->rec; }
gcs::ManifoldSurfaceMesh& IntrinsicSurface::inputMesh() { return *p_->mesh; }
gcs::VertexPositionGeometry& IntrinsicSurface::embedded() { return *p_->embedded; }
gcs::IntrinsicGeometryInterface& IntrinsicSurface::inputGeometry() { return p_->input(); }
gcs::IntrinsicGeometryInterface& IntrinsicSurface::geometry() { return p_->geom(); }
gcs::ManifoldSurfaceMesh& IntrinsicSurface::mesh() { return p_->tri ? *p_->tri->intrinsicMesh : *p_->mesh; }

Eigen::SparseMatrix<double> IntrinsicSurface::cotanLaplacian() {
    auto& g = geometry();
    g.requireCotanLaplacian();
    Eigen::SparseMatrix<double> L = g.cotanLaplacian;
    g.unrequireCotanLaplacian();
    return L;
}

std::vector<double> IntrinsicSurface::massDiagonal() {
    if (p_->opt.mass == MassKind::Galerkin) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface::massDiagonal: the galerkin mass is a sparse matrix, not a diagonal; use massMatrix()");
    auto& g = geometry();
    g.requireVertexDualAreas();
    std::vector<double> M(static_cast<size_t>(p_->nV));
    for (int v = 0; v < p_->nV; ++v) M[static_cast<size_t>(v)] = g.vertexDualAreas[static_cast<size_t>(v)];
    g.unrequireVertexDualAreas();
    return M;
}

Eigen::SparseMatrix<double> IntrinsicSurface::massMatrix() {
    auto& g = geometry();
    if (p_->opt.mass == MassKind::Galerkin) {
        g.requireVertexGalerkinMassMatrix();
        Eigen::SparseMatrix<double> M = g.vertexGalerkinMassMatrix;
        g.unrequireVertexGalerkinMassMatrix();
        return M;
    }
    const std::vector<double> d = massDiagonal();
    Eigen::SparseMatrix<double> M(p_->nV, p_->nV);
    std::vector<Eigen::Triplet<double>> t;
    for (int v = 0; v < p_->nV; ++v) t.emplace_back(v, v, d[static_cast<size_t>(v)]);
    M.setFromTriplets(t.begin(), t.end());
    return M;
}

Eigen::SparseMatrix<std::complex<double>> IntrinsicSurface::connectionLaplacian(int nSym) {
    if (nSym < 1) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface::connectionLaplacian: the symmetry n must be >= 1");
    auto& g = geometry();
    if (nSym == 1) {
        g.requireVertexConnectionLaplacian();
        Eigen::SparseMatrix<std::complex<double>> K = g.vertexConnectionLaplacian;
        g.unrequireVertexConnectionLaplacian();
        return K;
    }
    g.requireVertexIndices();
    g.requireEdgeCotanWeights();
    g.requireTransportVectorsAlongHalfedge();
    std::vector<Eigen::Triplet<std::complex<double>>> t;
    for (gcs::Halfedge he : g.mesh.halfedges()) {
        const size_t i = g.vertexIndices[he.vertex()], j = g.vertexIndices[he.next().vertex()];
        const double w = g.edgeCotanWeights[he.edge()];
        const gc::Vector2 r = g.transportVectorsAlongHalfedge[he.twin()];
        std::complex<double> rn(1.0, 0.0);
        for (int k = 0; k < nSym; ++k) rn *= std::complex<double>(r.x, r.y);
        t.emplace_back(i, i, w);
        t.emplace_back(i, j, -w * rn);
    }
    Eigen::SparseMatrix<std::complex<double>> K(p_->nV, p_->nV);
    K.setFromTriplets(t.begin(), t.end());
    g.unrequireTransportVectorsAlongHalfedge();
    g.unrequireEdgeCotanWeights();
    g.unrequireVertexIndices();
    return K;
}

TangentBasis IntrinsicSurface::tangentBasis() {
    auto& E = embedded();
    auto& G = inputGeometry();
    E.requireVertexNormals();
    E.requireVertexPositions();
    G.requireHalfedgeVectorsInVertex();
    TangentBasis B;
    B.e1.resize(3 * static_cast<size_t>(p_->nV)); B.e2 = B.e1; B.n = B.e1;
    for (gcs::Vertex v : inputMesh().vertices()) {
        // computeVertexTangentBasis's own formula, over the (mollified) input's gauge angles
        const gc::Vector3 N = E.vertexNormals[v];
        gc::Vector3 sum = gc::Vector3::zero();
        for (gcs::Halfedge he : v.outgoingHalfedges()) {
            gc::Vector3 eVec = E.vertexPositions[he.next().vertex()] - E.vertexPositions[he.vertex()];
            eVec = eVec.removeComponent(N);
            const double angle = G.halfedgeVectorsInVertex[he].arg();
            sum += eVec.rotateAround(N, -angle);
        }
        const gc::Vector3 e1 = gc::unit(sum), e2 = gc::cross(N, e1);
        const size_t i = 3 * v.getIndex();
        B.e1[i] = e1.x; B.e1[i + 1] = e1.y; B.e1[i + 2] = e1.z;
        B.e2[i] = e2.x; B.e2[i + 1] = e2.y; B.e2[i + 2] = e2.z;
        B.n[i] = N.x;  B.n[i + 1] = N.y;  B.n[i + 2] = N.z;
    }
    return B;
}

std::vector<double> IntrinsicSurface::barycentricMass() {
    auto& g = geometry();
    g.requireFaceAreas();
    std::vector<double> M(static_cast<size_t>(p_->nV), 0.0);
    for (gcs::Face f : g.mesh.faces())
        for (gcs::Vertex v : f.adjacentVertices()) M[v.getIndex()] += g.faceAreas[f] / 3.0;
    g.unrequireFaceAreas();
    return M;
}

Eigen::SparseMatrix<double> IntrinsicSurface::galerkinMassMatrix() {
    auto& g = geometry();
    g.requireVertexGalerkinMassMatrix();
    Eigen::SparseMatrix<double> M = g.vertexGalerkinMassMatrix;
    g.unrequireVertexGalerkinMassMatrix();
    return M;
}

std::vector<double> IntrinsicSurface::inputSignpostAngles() {
    auto& g = inputGeometry();
    g.requireCornerAngles();
    std::vector<double> a(p_->mesh->nHalfedges(), 0.0);
    for (gcs::Vertex v : p_->mesh->vertices()) {
        gcs::Halfedge he = v.halfedge();
        double sum = 0.0;
        for (int guard = 0; guard < 1 << 20; ++guard) {
            a[he.getIndex()] = sum;
            sum += g.cornerAngles[he.corner()];
            if (he.next().next().twin() == v.halfedge() || !he.next().next().twin().isInterior()) break;
            he = he.next().next().twin();
        }
    }
    g.unrequireCornerAngles();
    return a;
}

IntrinsicSurface::Complex IntrinsicSurface::complex() {
    auto& g = geometry();
    gcs::ManifoldSurfaceMesh& M = mesh();
    g.requireEdgeLengths();
    g.requireCornerAngles();
    Complex c;
    const size_t nF = M.nFaces(), nE = M.nEdges();
    c.faces.resize(3 * nF); c.faceEdges.resize(3 * nF); c.angle.resize(3 * nF);
    c.edges.resize(2 * nE); c.length.resize(nE);
    c.angleSum.assign(static_cast<size_t>(p_->nV), 0.0);
    c.reference.resize(static_cast<size_t>(p_->nV));
    const std::vector<double> inAngle = p_->tri ? std::vector<double>() : inputSignpostAngles();
    for (gcs::Edge e : M.edges()) {
        c.edges[2 * e.getIndex()] = static_cast<int32_t>(e.halfedge().vertex().getIndex());
        c.edges[2 * e.getIndex() + 1] = static_cast<int32_t>(e.halfedge().tipVertex().getIndex());
        c.length[e.getIndex()] = g.edgeLengths[e];
    }
    for (gcs::Face f : M.faces()) {
        gcs::Halfedge he = f.halfedge();
        for (int k = 0; k < 3; ++k, he = he.next()) {
            const size_t h = 3 * f.getIndex() + static_cast<size_t>(k);
            c.faces[h] = static_cast<int32_t>(he.vertex().getIndex());
            c.faceEdges[h] = (he.edge().halfedge() == he ? 1 : -1) * static_cast<int32_t>(he.edge().getIndex() + 1);
            c.angle[h] = p_->tri ? p_->tri->signpostAngle[he] : inAngle[he.getIndex()];
            c.angleSum[he.vertex().getIndex()] += g.cornerAngles[he.corner()];
        }
    }
    for (gcs::Vertex v : p_->mesh->vertices()) c.reference[v.getIndex()] = static_cast<int32_t>(v.halfedge().tipVertex().getIndex());
    g.unrequireEdgeLengths();
    g.unrequireCornerAngles();
    return c;
}

std::vector<double> IntrinsicSurface::inputMixedVoronoiMass() {
    return mixedVoronoi(*p_->mesh, p_->input());
}

}  // namespace nxr::manifold
