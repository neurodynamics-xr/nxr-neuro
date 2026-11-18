// nxr::signal::connection — the Cortical Flow Studio's optical flow, ported line for line; see opticalflow_connection.h.
// Studio files: flow/opticalFlow.ts (componentPencil, frameTerms, embed, opticalFlow), differential/intrinsicGauge.ts, differential/faceGradient.ts,
// differential/connectionLaplacian.ts (tangentBasis' normals, rescaledAngle, coneTarget).
#include "nxr/signal/opticalflow_connection.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "nxr/core/errors.h"
#include "nxr/signal/forms.h"

namespace nxr::signal::connection {
using Eigen::VectorXd;
using Corners0 = nxr::geometry::IntrinsicSurface::InputCorners;
constexpr double kTwoPi = 6.283185307179586476925286766559;

namespace {
double coneTarget(const Corners0& m, int v) { return m.boundary[static_cast<size_t>(v)] ? m.angleSum[static_cast<size_t>(v)] : kTwoPi; }   // connectionLaplacian.ts
}

Corners cornersOf(const Corners0& m, const double* P, const int32_t* F) {
    const int nH = static_cast<int>(m.signpost.size()), n = static_cast<int>(m.angleSum.size()), nF = nH / 3;
    Corners c;
    c.psi0.assign(static_cast<size_t>(nH), 0); c.width.assign(static_cast<size_t>(nH), 0);
    c.edge.assign(3 * static_cast<size_t>(nH), 0); c.perp.assign(3 * static_cast<size_t>(nH), 0);
    c.scale.assign(static_cast<size_t>(n), 1); c.area.assign(static_cast<size_t>(nF), 0);
    for (int v = 0; v < n; ++v) c.scale[static_cast<size_t>(v)] = m.angleSum[static_cast<size_t>(v)] > 0 ? coneTarget(m, v) / m.angleSum[static_cast<size_t>(v)] : 1.0;
    for (int f = 0; f < nF; ++f) {
        const int a = F[3 * f], b = F[3 * f + 1], cc = F[3 * f + 2];
        const double ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
        const double wx = P[3 * cc] - P[3 * a], wy = P[3 * cc + 1] - P[3 * a + 1], wz = P[3 * cc + 2] - P[3 * a + 2];
        double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        const double l = std::hypot(nx, ny, nz);
        c.area[static_cast<size_t>(f)] = l / 2;
        const double li = l != 0 ? l : 1.0;
        nx /= li; ny /= li; nz /= li;
        for (int k = 0; k < 3; ++k) {
            const int h = 3 * f + k, v = F[h], w = F[3 * f + (k + 1) % 3];
            double ex = P[3 * w] - P[3 * v], ey = P[3 * w + 1] - P[3 * v + 1], ez = P[3 * w + 2] - P[3 * v + 2];
            double el = std::hypot(ex, ey, ez);
            if (el == 0) el = 1;
            ex /= el; ey /= el; ez /= el;
            const size_t H = static_cast<size_t>(h);
            c.edge[3 * H] = ex; c.edge[3 * H + 1] = ey; c.edge[3 * H + 2] = ez;
            c.perp[3 * H] = ny * ez - nz * ey; c.perp[3 * H + 1] = nz * ex - nx * ez; c.perp[3 * H + 2] = nx * ey - ny * ex;   // N x e
            c.psi0[H] = m.signpost[H] * c.scale[static_cast<size_t>(v)];       // the rescaled signpost (psi = theta x cone / angle sum)
            c.width[H] = m.angle[H] * c.scale[static_cast<size_t>(v)];
        }
    }
    c.ptr.assign(static_cast<size_t>(n) + 1, 0);
    for (int h = 0; h < nH; ++h) ++c.ptr[static_cast<size_t>(F[h]) + 1];
    for (int v = 0; v < n; ++v) c.ptr[static_cast<size_t>(v) + 1] += c.ptr[static_cast<size_t>(v)];
    std::vector<int32_t> fill(c.ptr.begin(), c.ptr.end() - 1);
    c.order.assign(static_cast<size_t>(nH), 0);
    for (int h = 0; h < nH; ++h) c.order[static_cast<size_t>(fill[static_cast<size_t>(F[h])]++)] = h;
    for (int v = 0; v < n; ++v)
        std::stable_sort(c.order.begin() + c.ptr[static_cast<size_t>(v)], c.order.begin() + c.ptr[static_cast<size_t>(v) + 1], [&](int32_t x, int32_t y) { return c.psi0[static_cast<size_t>(x)] < c.psi0[static_cast<size_t>(y)]; });
    c.faces.assign(F, F + 3 * nF);
    return c;
}

// The face gradients come from forms.cpp's `assembleFaceGradient` (the one float32 port of faceGradient.ts; built without FMA contraction): the flow reads its `grads`.
FaceGradient assembleFaceGradient(const float* position, const int32_t* index, int nV, int nF) {
    FaceGradient g;
    g.nF = nF;
    g.grads = nxr::signal::assembleFaceGradient(position, index, nV, nF).grads;
    return g;
}

VectorXd intrinsicGradient(const Corners& c, const FaceGradient& grad, const double* s) {
    const size_t n = c.scale.size();
    VectorXd out = VectorXd::Zero(2 * static_cast<Eigen::Index>(n));
    std::vector<double> wsum(n, 0.0);
    for (int f = 0; f < grad.nF; ++f) {
        const double A = c.area[static_cast<size_t>(f)];
        if (!(A > 0)) continue;
        double gx = 0, gy = 0, gz = 0;
        for (int k = 0; k < 3; ++k) {
            const double x = s[c.faces[3 * static_cast<size_t>(f) + static_cast<size_t>(k)]];
            const size_t o = 9 * static_cast<size_t>(f) + 3 * static_cast<size_t>(k);
            gx += x * grad.grads[o]; gy += x * grad.grads[o + 1]; gz += x * grad.grads[o + 2];
        }
        const double gl = std::hypot(gx, gy, gz);
        for (int k = 0; k < 3; ++k) {
            const size_t h = 3 * static_cast<size_t>(f) + static_cast<size_t>(k);
            const size_t v = static_cast<size_t>(c.faces[h]);
            wsum[v] += A;
            if (gl == 0) continue;
            const double th = std::atan2(gx * c.perp[3 * h] + gy * c.perp[3 * h + 1] + gz * c.perp[3 * h + 2], gx * c.edge[3 * h] + gy * c.edge[3 * h + 1] + gz * c.edge[3 * h + 2]);
            const double psi = c.psi0[h] + th * c.scale[v];
            out[2 * static_cast<Eigen::Index>(v)] += A * gl * std::cos(psi); out[2 * static_cast<Eigen::Index>(v) + 1] += A * gl * std::sin(psi);
        }
    }
    for (size_t v = 0; v < n; ++v) if (wsum[v] > 0) { out[2 * static_cast<Eigen::Index>(v)] /= wsum[v]; out[2 * static_cast<Eigen::Index>(v) + 1] /= wsum[v]; }
    return out;
}

void cornerAt(const Corners& c, int v, double psi, int& h, double& theta) {
    const int a = c.ptr[static_cast<size_t>(v)], b = c.ptr[static_cast<size_t>(v) + 1];
    if (a == b) { h = -1; theta = 0; return; }
    const int last = c.order[static_cast<size_t>(b) - 1];
    const double span = c.psi0[static_cast<size_t>(last)] + c.width[static_cast<size_t>(last)];
    double x = psi;
    if (span > kTwoPi - 1e-9) x -= kTwoPi * std::floor(x / kTwoPi);
    int k = a;
    while (k + 1 < b && c.psi0[static_cast<size_t>(c.order[static_cast<size_t>(k) + 1])] <= x) ++k;
    h = c.order[static_cast<size_t>(k)];
    theta = (x - c.psi0[static_cast<size_t>(h)]) / c.scale[static_cast<size_t>(v)];
}

void embedAt(const Corners& c, const double* nrm, int v, double a, double b, double* out) {
    const double zl = std::hypot(a, b);
    out[0] = out[1] = out[2] = 0;
    if (!zl) return;
    int h; double theta;
    cornerAt(c, v, std::atan2(b, a), h, theta);
    if (h < 0) return;
    const double co = std::cos(theta), si = std::sin(theta);
    const size_t H = static_cast<size_t>(h);
    double x = co * c.edge[3 * H] + si * c.perp[3 * H], y = co * c.edge[3 * H + 1] + si * c.perp[3 * H + 1], z = co * c.edge[3 * H + 2] + si * c.perp[3 * H + 2];
    const double nx = nrm[3 * v], ny = nrm[3 * v + 1], nz = nrm[3 * v + 2], d = x * nx + y * ny + z * nz;
    const double px = x - d * nx, py = y - d * ny, pz = z - d * nz, pl = std::hypot(px, py, pz);
    if (pl > 1e-6) { x = px / pl; y = py / pl; z = pz / pl; }
    out[0] = zl * x; out[1] = zl * y; out[2] = zl * z;
}

Eigen::SparseMatrix<double> realConnectionLaplacian(nxr::geometry::IntrinsicSurface& s) {
    const Eigen::SparseMatrix<std::complex<double>> K = s.connectionLaplacian(1);
    const int nV = s.nV();
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(4 * static_cast<size_t>(K.nonZeros()));
    for (int k = 0; k < K.outerSize(); ++k)
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(K, k); it; ++it) {
            const int i = static_cast<int>(it.row()), j = static_cast<int>(it.col());
            const double a = it.value().real(), b = it.value().imag();
            t.emplace_back(2 * i, 2 * j, a); t.emplace_back(2 * i, 2 * j + 1, -b); t.emplace_back(2 * i + 1, 2 * j, b); t.emplace_back(2 * i + 1, 2 * j + 1, a);
        }
    Eigen::SparseMatrix<double> C(2 * nV, 2 * nV);
    C.setFromTriplets(t.begin(), t.end());
    C.makeCompressed();
    return C;
}

Pencil componentPencil(const double* P, int n, const std::vector<int32_t>& F, const geometry::DelaunayOptions& o) {
    const int nF = static_cast<int>(F.size() / 3);
    nxr::geometry::IntrinsicOptions io;
    io.mollify = o.mollify; io.flipEps = o.flipEps; io.mass = nxr::geometry::MassKind::InputMixedVoronoi;
    nxr::geometry::IntrinsicSurface surf(P, n, F.data(), nF, io);   // geometry-central's signpost triangulation: the corners read its input, the operators its flipped mesh
    Pencil p;
    // the corners and normals of the INPUT faces, before the flips rewrite the halfedges
    std::vector<double> N(3 * static_cast<size_t>(n), 0.0);
    for (int f = 0; f < nF; ++f) {
        const int a = F[3 * f], b = F[3 * f + 1], c = F[3 * f + 2];
        const double ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
        const double vx = P[3 * c] - P[3 * a], vy = P[3 * c + 1] - P[3 * a + 1], vz = P[3 * c + 2] - P[3 * a + 2];
        const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        for (int v : {a, b, c}) { N[3 * static_cast<size_t>(v)] += nx; N[3 * static_cast<size_t>(v) + 1] += ny; N[3 * static_cast<size_t>(v) + 2] += nz; }
    }
    p.normals.assign(3 * static_cast<size_t>(n), 0.0);
    for (int v = 0; v < n; ++v) {
        double l = std::hypot(N[3 * static_cast<size_t>(v)], N[3 * static_cast<size_t>(v) + 1], N[3 * static_cast<size_t>(v) + 2]);
        if (l == 0) l = 1;
        for (int k = 0; k < 3; ++k) p.normals[3 * static_cast<size_t>(v) + static_cast<size_t>(k)] = static_cast<double>(static_cast<float>(N[3 * static_cast<size_t>(v) + static_cast<size_t>(k)] / l));   // Float32Array
    }
    p.corners = cornersOf(surf.inputCorners(), P, F.data());
    std::vector<float> Pf(3 * static_cast<size_t>(n));
    for (size_t k = 0; k < Pf.size(); ++k) Pf[k] = static_cast<float>(P[k]);                  // Float32Array.from(P)
    p.grad = assembleFaceGradient(Pf.data(), F.data(), n, nF);
    p.C = realConnectionLaplacian(surf);
    const std::vector<double> mass = surf.inputMixedVoronoiMass();                              // lumpedMass: the input's mixed Voronoi mass, recorded before the flips
    p.M = Eigen::Map<const VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size()));
    return p;
}

// ── the flow ──

struct ConnectionFlow::Impl {
    struct Comp { std::vector<int> verts; Pencil pen; double ridge = 0; std::vector<double> local; };
    Eigen::Index V = 0;
    double alpha = 1, scale = 1;
    std::vector<Comp> comps;
    std::vector<char> inFace;                                  // [V] 1 when the vertex belongs to a component with faces
    Eigen::SparseMatrix<double> system;                        // alpha C + the data blocks (interleaved: vertex v rows 2v, 2v+1)
    Eigen::SparseMatrix<double> smooth;                        // alpha C alone
    std::vector<double> smoothValues;
    std::vector<Eigen::Index> slot;                            // [4 V]
    Eigen::VectorXd M, z;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    bool analysed = false;
    optimisation::PcgPreconditioner precond;

    // the frame's data term: blocks into `system`'s slots and the right side; false when no gradient. dataScale: mean data weight per component
    void assemble(const VectorXd& previous, const VectorXd& next, double sc, VectorXd& rhs, double* dataScale) {
        if (previous.size() != V || next.size() != V) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "ConnectionFlow: a map has " + std::to_string(previous.size()) + " rows, the mesh " + std::to_string(V) + " vertices");
        const double s = std::isnan(sc) ? scale : sc, k = 1.0 / s;
        double* val = system.valuePtr();
        std::copy(smoothValues.begin(), smoothValues.end(), val);
        rhs = VectorXd::Zero(2 * V);
        double sum = 0, msum = 0;
        for (Comp& c : comps) {
            const size_t nl = c.verts.size();
            std::vector<double> next_(nl);
            for (size_t i = 0; i < nl; ++i) { c.local[i] = previous[c.verts[i]] * k; next_[i] = next[c.verts[i]] * k; }
            const VectorXd g = intrinsicGradient(c.pen.corners, c.pen.grad, c.local.data());
            for (size_t i = 0; i < nl; ++i) {
                const double g1 = g[2 * static_cast<Eigen::Index>(i)], g2 = g[2 * static_cast<Eigen::Index>(i) + 1], m = c.pen.M[static_cast<Eigen::Index>(i)], dI = next_[i] - c.local[i];
                const size_t gv = static_cast<size_t>(c.verts[i]);
                val[slot[4 * gv]] += m * g1 * g1 + c.ridge; val[slot[4 * gv + 1]] += m * g1 * g2;
                val[slot[4 * gv + 2]] += m * g1 * g2;       val[slot[4 * gv + 3]] += m * g2 * g2 + c.ridge;
                rhs[2 * static_cast<Eigen::Index>(gv)] = -m * g1 * dI; rhs[2 * static_cast<Eigen::Index>(gv) + 1] = -m * g2 * dI;
                sum += m * (g1 * g1 + g2 * g2); msum += m;
            }
        }
        if (dataScale) *dataScale = msum > 0 ? 0.5 * sum / msum : 0.0;
    }
    VectorXd embed(const VectorXd& zz) const {
        VectorXd out = VectorXd::Zero(3 * V);
        for (const Comp& c : comps)
            for (size_t i = 0; i < c.verts.size(); ++i) {
                const Eigen::Index gv = c.verts[i];
                double e[3];
                embedAt(c.pen.corners, c.pen.normals.data(), static_cast<int>(i), zz[2 * gv], zz[2 * gv + 1], e);
                out.segment<3>(3 * gv) << e[0], e[1], e[2];
            }
        return out;
    }
    void makePreconditioner(const optimisation::PcgOptions& o, double cdata) {
        Eigen::SparseMatrix<double> pre = system;
        double* val = pre.valuePtr();
        std::copy(smoothValues.begin(), smoothValues.end(), val);
        for (Eigen::Index v = 0; v < V; ++v) { val[slot[static_cast<size_t>(4 * v)]] += cdata * M[v]; val[slot[static_cast<size_t>(4 * v + 3)]] += cdata * M[v]; }
        precond.build(pre, o.preconditioner);
    }
};

ConnectionFlow::ConnectionFlow(const RowMat3& positions, const operators::FaceMat& faces, double alpha, double maxAbs, const geometry::DelaunayOptions& o) : p_(std::make_unique<Impl>()) {
    Impl& s = *p_;
    s.V = positions.rows(); s.alpha = alpha; s.scale = maxAbs;
    if (!(alpha > 0) || !std::isfinite(alpha)) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "opticalFlow: alpha = " + std::to_string(alpha) + ", the smoothness weight must be positive and finite");
    const Eigen::Index V = s.V;
    // connected components (union-find over the faces); each is solved by its own pencil (its own mesh, intrinsic Delaunay triangulation, C, M and corners)
    std::vector<int> parent(static_cast<size_t>(V));
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) { while (parent[static_cast<size_t>(x)] != x) { parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])]; x = parent[static_cast<size_t>(x)]; } return x; };
    for (Eigen::Index f = 0; f < faces.rows(); ++f) { const int a = find(faces(f, 0)); parent[static_cast<size_t>(find(faces(f, 1)))] = a; parent[static_cast<size_t>(find(faces(f, 2)))] = find(a); }
    std::vector<int> compOf(static_cast<size_t>(V), -1), localOf(static_cast<size_t>(V), -1);
    std::vector<char> used(static_cast<size_t>(V), 0);
    for (Eigen::Index f = 0; f < faces.rows(); ++f) for (int k = 0; k < 3; ++k) used[static_cast<size_t>(faces(f, k))] = 1;
    s.inFace = used;
    std::vector<int> rootToComp(static_cast<size_t>(V), -1);
    for (Eigen::Index v = 0; v < V; ++v) {
        if (!used[static_cast<size_t>(v)]) continue;
        const int r = find(static_cast<int>(v));
        if (rootToComp[static_cast<size_t>(r)] < 0) { rootToComp[static_cast<size_t>(r)] = static_cast<int>(s.comps.size()); s.comps.emplace_back(); }
        Impl::Comp& c = s.comps[static_cast<size_t>(rootToComp[static_cast<size_t>(r)])];
        compOf[static_cast<size_t>(v)] = rootToComp[static_cast<size_t>(r)]; localOf[static_cast<size_t>(v)] = static_cast<int>(c.verts.size());
        c.verts.push_back(static_cast<int>(v));
    }
    std::vector<std::vector<int32_t>> compFaces(s.comps.size());
    for (Eigen::Index f = 0; f < faces.rows(); ++f) { auto& cf = compFaces[static_cast<size_t>(compOf[static_cast<size_t>(faces(f, 0))])]; for (int k = 0; k < 3; ++k) cf.push_back(localOf[static_cast<size_t>(faces(f, k))]); }
    s.M = VectorXd::Zero(V);
    std::vector<Eigen::Triplet<double>> trip;
    for (size_t q = 0; q < s.comps.size(); ++q) {
        Impl::Comp& c = s.comps[q];
        const int n = static_cast<int>(c.verts.size());
        std::vector<double> P(3 * static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) P[3 * static_cast<size_t>(i) + static_cast<size_t>(k)] = positions(c.verts[static_cast<size_t>(i)], k);
        c.pen = componentPencil(P.data(), n, compFaces[q], o);
        c.local.assign(static_cast<size_t>(n), 0.0);
        double dC = 0;
        for (int k = 0; k < c.pen.C.outerSize(); ++k) for (Eigen::SparseMatrix<double>::InnerIterator it(c.pen.C, k); it; ++it) {
            if (it.row() == it.col()) dC += it.value();
            const Eigen::Index gr = 2 * c.verts[static_cast<size_t>(it.row() / 2)] + it.row() % 2, gc = 2 * c.verts[static_cast<size_t>(it.col() / 2)] + it.col() % 2;
            trip.emplace_back(gr, gc, alpha * it.value());
        }
        const double mean = dC / static_cast<double>(2 * n);
        c.ridge = 1e-9 * alpha * (mean != 0 ? mean : 1.0);            // 1e-9 · alpha · mean(C_ii) keeps a parallel field definite
        for (int i = 0; i < n; ++i) s.M[c.verts[static_cast<size_t>(i)]] = c.pen.M[i];
    }
    s.smooth.resize(2 * V, 2 * V);
    s.smooth.setFromTriplets(trip.begin(), trip.end());
    for (Eigen::Index v = 0; v < V; ++v) for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) trip.emplace_back(2 * v + a, 2 * v + b, 0.0);   // every vertex's 2 x 2 block is in the pattern
    s.system.resize(2 * V, 2 * V);
    s.system.setFromTriplets(trip.begin(), trip.end());
    s.smoothValues.assign(s.system.valuePtr(), s.system.valuePtr() + s.system.nonZeros());
    s.slot.resize(static_cast<size_t>(4 * V));
    for (Eigen::Index v = 0; v < V; ++v) for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) s.slot[static_cast<size_t>(4 * v + 2 * a + b)] = &s.system.coeffRef(2 * v + a, 2 * v + b) - s.system.valuePtr();
    // a vertex in no face has no smoothness and no data: the identity keeps the system definite (its z = 0)
    for (Eigen::Index v = 0; v < V; ++v) if (!used[static_cast<size_t>(v)]) { s.smoothValues[static_cast<size_t>(s.slot[static_cast<size_t>(4 * v)])] = 1; s.smoothValues[static_cast<size_t>(s.slot[static_cast<size_t>(4 * v + 3)])] = 1; }
    s.z = VectorXd::Zero(2 * V);
}
ConnectionFlow::~ConnectionFlow() = default;

const VectorXd& ConnectionFlow::gauge() const { return p_->z; }
const VectorXd& ConnectionFlow::mass() const { return p_->M; }
const Eigen::SparseMatrix<double>& ConnectionFlow::smoothness() const { return p_->smooth; }
VectorXd ConnectionFlow::embed(const VectorXd& z) const { return p_->embed(z); }
VectorXd ConnectionFlow::gaugeOf(const VectorXd& field) const {
    VectorXd out = VectorXd::Zero(2 * p_->V);
    for (const Impl::Comp& comp : p_->comps) {
        const Corners& c = comp.pen.corners;
        const size_t n = comp.verts.size();
        std::vector<double> wsum(n, 0.0), zx(n, 0.0), zy(n, 0.0);
        const size_t nF = c.area.size();
        for (size_t f = 0; f < nF; ++f) {
            const double A = c.area[f];
            if (!(A > 0)) continue;
            for (size_t k = 0; k < 3; ++k) {
                const size_t h = 3 * f + k, v = static_cast<size_t>(c.faces[h]);
                const Eigen::Vector3d u = field.segment<3>(3 * static_cast<Eigen::Index>(comp.verts[v]));
                wsum[v] += A;
                const double len = u.norm();
                if (len == 0) continue;
                const double th = std::atan2(u[0] * c.perp[3 * h] + u[1] * c.perp[3 * h + 1] + u[2] * c.perp[3 * h + 2], u[0] * c.edge[3 * h] + u[1] * c.edge[3 * h + 1] + u[2] * c.edge[3 * h + 2]);
                const double psi = c.psi0[h] + th * c.scale[v];
                zx[v] += A * len * std::cos(psi); zy[v] += A * len * std::sin(psi);
            }
        }
        for (size_t v = 0; v < n; ++v) if (wsum[v] > 0) { out[2 * comp.verts[v]] = zx[v] / wsum[v]; out[2 * comp.verts[v] + 1] = zy[v] / wsum[v]; }
    }
    return out;
}
VectorXd ConnectionFlow::gaugeGradient(const VectorXd& map, double sc) const {
    Impl& s = *p_;
    const double k = 1.0 / (std::isnan(sc) ? s.scale : sc);
    VectorXd out = VectorXd::Zero(2 * s.V);
    for (Impl::Comp& c : s.comps) {
        for (size_t i = 0; i < c.verts.size(); ++i) c.local[i] = map[c.verts[i]] * k;
        const VectorXd g = intrinsicGradient(c.pen.corners, c.pen.grad, c.local.data());
        for (size_t i = 0; i < c.verts.size(); ++i) { out[2 * c.verts[i]] = g[2 * static_cast<Eigen::Index>(i)]; out[2 * c.verts[i] + 1] = g[2 * static_cast<Eigen::Index>(i) + 1]; }
    }
    return out;
}

VectorXd ConnectionFlow::step(const VectorXd& previous, const VectorXd& next) {
    Impl& s = *p_;
    if (!(s.scale > 0)) return VectorXd::Zero(3 * s.V);                    // peak 0: no flow
    VectorXd rhs;
    s.assemble(previous, next, std::numeric_limits<double>::quiet_NaN(), rhs, nullptr);
    if (!(rhs.cwiseAbs().maxCoeff() > 0)) { s.z.setZero(); return VectorXd::Zero(3 * s.V); }   // no gradient in the frame: v = 0
    if (!s.analysed) { s.ldlt.analyzePattern(s.system); s.analysed = true; }
    s.ldlt.factorize(s.system);
    VectorXd z;
    if (s.ldlt.info() == Eigen::Success) z = s.ldlt.solve(rhs);
    if (s.ldlt.info() != Eigen::Success || !z.allFinite()) { Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(s.system); z = lu.solve(rhs); }
    s.z = z;
    return s.embed(z);
}

void ConnectionFlow::buildPreconditioner(const VectorXd& map, const optimisation::PcgOptions& o, double sc) {
    Impl& s = *p_;
    if (s.precond.built()) return;
    VectorXd rhs; double c = 0;
    s.assemble(map, map, sc, rhs, &c);
    s.makePreconditioner(o, std::max(c, 1e-12));
}

VectorXd ConnectionFlow::stepPcg(const VectorXd& previous, const VectorXd& next, VectorXd& zw, const optimisation::PcgOptions& o, optimisation::PcgInfo* info, double sc) {
    Impl& s = *p_;
    if (!((std::isnan(sc) ? s.scale : sc) > 0)) { zw = VectorXd::Zero(2 * s.V); return VectorXd::Zero(3 * s.V); }
    VectorXd rhs; double c = 0;
    s.assemble(previous, next, sc, rhs, &c);
    if (!s.precond.built()) s.makePreconditioner(o, std::max(c, 1e-12));
    const VectorXd x = optimisation::pcg(s.system, rhs, zw, s.precond, o, info);
    zw = x; s.z = x;
    return s.embed(x);
}

}  // namespace nxr::signal::connection
