// nxr::geometry::IntrinsicSurface: see nxr/geometry/intrinsic_surface.h and docs/specs/2026-11-17-geometry-central-switch.md.
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/surface_arrays.h"

#include "nxr/core/errors.h"

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

namespace nxr::geometry {

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

// The intrinsic triangulation rebuilt from its arrays: its lengths, and the gauge (halfedge vectors, cone angles) exactly as the signpost triangulation held
// them when exported. Every other quantity is geometry-central's own, from the same mesh and lengths.
class Rebuilt : public MassServing<gcs::EdgeLengthGeometry> {
public:
    Rebuilt(const std::vector<double>* mass, gcs::ManifoldSurfaceMesh& mesh, const gcs::EdgeData<double>& len, const IntrinsicTriangulation& t)
        : MassServing<gcs::EdgeLengthGeometry>(mass, mesh, len), t_(t) {}
protected:
    void computeHalfedgeVectorsInVertex() override {
        this->halfedgeVectorsInVertex = gcs::HalfedgeData<gc::Vector2>(this->mesh);
        for (size_t h = 0; h < this->mesh.nHalfedges(); ++h)
            this->halfedgeVectorsInVertex[h] = gc::Vector2{t_.halfedgeVectorInVertex[2 * h], t_.halfedgeVectorInVertex[2 * h + 1]};
    }
    void computeVertexAngleSums() override {
        this->vertexAngleSums = gcs::VertexData<double>(this->mesh);
        for (size_t v = 0; v < this->mesh.nVertices(); ++v) this->vertexAngleSums[v] = t_.vertexAngleSum[v];
    }
private:
    const IntrinsicTriangulation& t_;
};

// geometry-central's mesh from its internal arrays (the constructor RichSurfaceMeshData uses; protected, reached through a subclass), and the arrays of a mesh
// (through pointers to the protected members, formed in the subclass).
struct RawMesh : gcs::ManifoldSurfaceMesh {
    RawMesh(const std::vector<size_t>& next, const std::vector<size_t>& vertex, const std::vector<size_t>& face, const std::vector<size_t>& vHe,
            const std::vector<size_t>& fHe, size_t loops)
        : gcs::ManifoldSurfaceMesh(next, vertex, face, vHe, fHe, loops) {}
    static const std::vector<size_t>& next(const gcs::SurfaceMesh& m) { return m.*(&RawMesh::heNextArr); }
    static const std::vector<size_t>& vertex(const gcs::SurfaceMesh& m) { return m.*(&RawMesh::heVertexArr); }
    static const std::vector<size_t>& face(const gcs::SurfaceMesh& m) { return m.*(&RawMesh::heFaceArr); }
    static const std::vector<size_t>& vHe(const gcs::SurfaceMesh& m) { return m.*(&RawMesh::vHalfedgeArr); }
    static const std::vector<size_t>& fHe(const gcs::SurfaceMesh& m) { return m.*(&RawMesh::fHalfedgeArr); }
    static size_t facesFill(const gcs::SurfaceMesh& m) { return m.*(&RawMesh::nFacesFillCount); }
    static size_t loopsFill(const gcs::SurfaceMesh& m) { return m.*(&RawMesh::nBoundaryLoopsFillCount); }
};

std::vector<int64_t> toI64(const std::vector<size_t>& a, size_t n) {
    std::vector<int64_t> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = a[i] == gc::INVALID_IND ? -1 : static_cast<int64_t>(a[i]);
    return r;
}
std::vector<size_t> toSize(const std::vector<int64_t>& a) {
    std::vector<size_t> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] < 0 ? gc::INVALID_IND : static_cast<size_t>(a[i]);
    return r;
}

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
    std::unique_ptr<Signpost> tri;                             // intrinsic mode only, when constructed
    std::unique_ptr<gcs::ManifoldSurfaceMesh> imesh;           // intrinsic mode only, when rebuilt from arrays: the triangulation's mesh
    std::unique_ptr<Rebuilt> rtri;                             // ... and its geometry
    const IntrinsicTriangulation* arrays = nullptr;            // the arrays rebuilt from (signposts)
    int nV = 0, nF = 0;

    gcs::IntrinsicGeometryInterface& input() { return inputGeom ? static_cast<gcs::IntrinsicGeometryInterface&>(*inputGeom) : *embedded; }
    gcs::IntrinsicGeometryInterface& geom() {
        if (tri) return *tri;
        if (rtri) return *rtri;
        return *embedded;
    }
    bool intrinsic() const { return tri || rtri; }
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

IntrinsicSurface::IntrinsicSurface(const SurfaceArrays& S) : p_(new Impl) {
    Impl& P = *p_;
    const IntrinsicTriangulation& T = S.triangulation;
    P.opt = T.options; P.rec = T.record;
    P.nV = S.nV(); P.nF = S.nF();
    std::vector<std::vector<size_t>> polys(static_cast<size_t>(P.nF));
    for (int f = 0; f < P.nF; ++f) {
        for (int k = 0; k < 3; ++k) {
            const int32_t v = S.faces(f, k);
            if (v < 0 || v >= P.nV) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface: a face index is outside [0, nV)");
            polys[static_cast<size_t>(f)].push_back(static_cast<size_t>(v));
        }
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> pos = S.positions;
    try {
        P.mesh.reset(new gcs::ManifoldSurfaceMesh(polys));
    } catch (const std::exception& e) {
        throw Error(ErrorCode::InvalidInput, std::string("IntrinsicSurface: the input is not an oriented manifold triangle mesh: ") + e.what());
    }
    P.mass = T.mass;
    const bool mass = T.options.mass == MassKind::InputMixedVoronoi && !P.mass.empty();
    if (!T.options.intrinsic) {
        P.embedded.reset(new Embedded(mass ? &P.mass : nullptr, *P.mesh, pos));
        return;
    }
    P.embedded.reset(new Embedded(nullptr, *P.mesh, pos));
    if (T.inputEdgeLength.size() != P.mesh->nEdges() || T.vertexHalfedge.size() != static_cast<size_t>(P.nV))
        throw Error(ErrorCode::InvalidInput, "IntrinsicSurface: the triangulation's arrays do not belong to this mesh");
    gcs::EdgeData<double> inLen(*P.mesh);
    for (size_t e = 0; e < P.mesh->nEdges(); ++e) inLen[e] = T.inputEdgeLength[e];
    P.inputGeom.reset(new gcs::EdgeLengthGeometry(*P.mesh, inLen));
    P.imesh.reset(new RawMesh(toSize(T.halfedgeNext), toSize(T.halfedgeVertex), toSize(T.halfedgeFace), toSize(T.vertexHalfedge), toSize(T.faceHalfedge),
                              static_cast<size_t>(T.boundaryLoops)));
    gcs::EdgeData<double> len(*P.imesh);
    for (size_t e = 0; e < P.imesh->nEdges(); ++e) len[e] = T.edgeLength[e];
    P.rtri.reset(new Rebuilt(mass ? &P.mass : nullptr, *P.imesh, len, T));
    P.arrays = &T;
}

IntrinsicTriangulation IntrinsicSurface::triangulation() {
    Impl& P = *p_;
    IntrinsicTriangulation T;
    T.options = P.opt; T.record = P.rec; T.mass = P.mass;
    if (!P.intrinsic()) return T;
    auto& in = P.input();
    in.requireEdgeLengths();
    T.inputEdgeLength.resize(P.mesh->nEdges());
    for (size_t e = 0; e < P.mesh->nEdges(); ++e) T.inputEdgeLength[e] = in.edgeLengths[e];
    gcs::ManifoldSurfaceMesh& M = mesh();
    if (!M.isCompressed()) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface::triangulation: the triangulation's mesh is not compressed");
    const size_t nH = M.nHalfedges(), nE = M.nEdges(), nV = M.nVertices();
    const size_t nFill = RawMesh::facesFill(M), nLoops = RawMesh::loopsFill(M);
    T.halfedgeNext = toI64(RawMesh::next(M), nH);
    T.halfedgeVertex = toI64(RawMesh::vertex(M), nH);
    T.halfedgeFace = toI64(RawMesh::face(M), nH);
    T.vertexHalfedge = toI64(RawMesh::vHe(M), nV);
    const auto& fHe = RawMesh::fHe(M);
    if (fHe.size() != nFill + nLoops) throw Error(ErrorCode::InvalidInput, "IntrinsicSurface::triangulation: the face buffer has spare capacity");
    T.faceHalfedge = toI64(fHe, fHe.size());
    T.boundaryLoops = static_cast<int64_t>(nLoops);
    auto& g = geometry();
    g.requireEdgeLengths(); g.requireVertexAngleSums(); g.requireHalfedgeVectorsInVertex();
    T.edgeLength.resize(nE); T.vertexAngleSum.resize(nV); T.halfedgeVectorInVertex.resize(2 * nH); T.signpost.resize(nH);
    for (size_t e = 0; e < nE; ++e) T.edgeLength[e] = g.edgeLengths[e];
    for (size_t v = 0; v < nV; ++v) T.vertexAngleSum[v] = g.vertexAngleSums[v];
    for (size_t h = 0; h < nH; ++h) {
        T.halfedgeVectorInVertex[2 * h] = g.halfedgeVectorsInVertex[h].x;
        T.halfedgeVectorInVertex[2 * h + 1] = g.halfedgeVectorsInVertex[h].y;
        T.signpost[h] = P.tri ? P.tri->signpostAngle[h] : P.arrays->signpost[h];
    }
    g.unrequireHalfedgeVectorsInVertex(); g.unrequireVertexAngleSums(); g.unrequireEdgeLengths();
    return T;
}

IntrinsicTriangulation intrinsicTriangulation(const double* positions, int nV, const int32_t* faces, int nF, const IntrinsicOptions& options) {
    IntrinsicSurface s(positions, nV, faces, nF, options);
    return s.triangulation();
}

IntrinsicOptions intrinsicOptions(bool intrinsicDelaunay, const DelaunayOptions& delaunay) {
    IntrinsicOptions o;
    o.intrinsic = intrinsicDelaunay;
    o.mollify = delaunay.mollify;
    o.flipEps = delaunay.flipEps;
    o.mass = MassKind::InputMixedVoronoi;
    return o;
}

SurfaceArrays surfaceArrays(const double* positions, int nV, const int32_t* faces, int nF, const IntrinsicOptions& options) {
    SurfaceArrays s;
    s.positions = Eigen::Map<const RowMat3>(positions, nV, 3);
    s.faces = Eigen::Map<const FaceMat>(faces, nF, 3);
    s.triangulation = intrinsicTriangulation(positions, nV, faces, nF, options);
    return s;
}

SurfaceArrays surfaceArrays(const RowMat3& positions, const FaceMat& faces, const IntrinsicOptions& options) {
    return surfaceArrays(positions.data(), static_cast<int>(positions.rows()), faces.data(), static_cast<int>(faces.rows()), options);
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
gcs::ManifoldSurfaceMesh& IntrinsicSurface::mesh() {
    if (p_->tri) return *p_->tri->intrinsicMesh;
    if (p_->imesh) return *p_->imesh;
    return *p_->mesh;
}

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

IntrinsicSurface::InputCorners IntrinsicSurface::inputCorners() {
    InputCorners c;
    const std::vector<double> sp = inputSignpostAngles();
    auto& g = inputGeometry();
    g.requireCornerAngles();
    const size_t nF = p_->mesh->nFaces(), nV = p_->mesh->nVertices();
    c.signpost.assign(3 * nF, 0.0); c.angle.assign(3 * nF, 0.0); c.angleSum.assign(nV, 0.0); c.boundary.assign(nV, 0);
    for (gcs::Face f : p_->mesh->faces()) {
        gcs::Halfedge he = f.halfedge();
        for (int k = 0; k < 3; ++k, he = he.next()) {
            const size_t h = 3 * f.getIndex() + static_cast<size_t>(k);
            c.signpost[h] = sp[he.getIndex()];
            c.angle[h] = g.cornerAngles[he.corner()];
            c.angleSum[he.vertex().getIndex()] += c.angle[h];
        }
    }
    for (gcs::Vertex v : p_->mesh->vertices()) c.boundary[v.getIndex()] = v.isBoundary() ? 1 : 0;
    g.unrequireCornerAngles();
    return c;
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
            if (he.next().next().twin() == v.halfedge()) break;
            if (!he.next().next().twin().isInterior()) {
                // a boundary vertex: the fan ends at the boundary edge whose outgoing halfedge is exterior; its direction is at the vertex's angle sum
                a[he.next().next().twin().getIndex()] = sum;
                break;
            }
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
    const std::vector<double> inAngle = p_->intrinsic() ? std::vector<double>() : inputSignpostAngles();
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
            c.angle[h] = p_->tri ? p_->tri->signpostAngle[he] : p_->rtri ? p_->arrays->signpost[he.getIndex()] : inAngle[he.getIndex()];
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

}  // namespace nxr::geometry
