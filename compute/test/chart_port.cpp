// THE PORT OF THE STUDIO'S CHARTS, kept ONLY for test_gc_port_comparison (the one-time comparison, G2 / appendix B) until the port is deleted (G7). Not part of the library:
// nxr/chart.h is the geometry-central wrapper since G4. Unchanged from nxr/chart.h and src/chart.cpp as of G3, in the namespace nxr::chart_port.
#include "chart_port.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "nxr/errors.h"
#include "nxr/opticalflow_connection.h"

namespace nxr::chart_port {

using Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
using flow::connection::Pencil;
using flow::connection::componentPencil;

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ── HeatMethod (toolbox/differential/heatMethod.ts) ──

struct HeatMethod::Impl {
    int n = 0;
    double t = 0;
    Eigen::SimplicialLDLT<SpMat> heat, heatD, poisson;
    bool hasD = false;
    std::vector<uint8_t> boundary;
    VectorXd M;
    struct Face { int v[3]; double P[3][2]; double A; };
    std::vector<Face> faces;
};

static SpMat shifted(const SpMat& A, double sigma, const VectorXd& B) {
    SpMat out = A;
    for (int i = 0; i < out.rows(); ++i) out.coeffRef(i, i) += sigma * B[i];
    out.makeCompressed();
    return out;
}

HeatMethod::HeatMethod(const delaunay::Mesh& m, double tCoef) : p_(std::make_unique<Impl>()) {
    Impl& s = *p_;
    s.n = m.nV;
    const SpMat L = delaunay::cotanLaplacian(m);
    const std::vector<double> mv = m.mass.size() == static_cast<size_t>(m.nV) ? m.mass : delaunay::mixedVoronoiMass(m);   // lumpedMass: the recorded mass, else the triangulation's own
    s.M = Eigen::Map<const VectorXd>(mv.data(), m.nV);
    double mean = 0;
    for (double l : m.length) mean += l;
    mean /= static_cast<double>(m.length.size());
    s.t = tCoef * mean * mean;
    const SpMat H = shifted(L, 1.0 / s.t, s.M);                  // (L + M/t) = (M + tL)/t
    s.heat.compute(H);
    if (!m.boundary.empty() && std::any_of(m.boundary.begin(), m.boundary.end(), [](uint8_t b) { return b != 0; })) {
        double big = 0;
        for (int i = 0; i < H.rows(); ++i) big = std::max(big, H.coeff(i, i));
        VectorXd b(m.nV);
        for (int i = 0; i < m.nV; ++i) b[i] = m.boundary[static_cast<size_t>(i)];
        s.heatD.compute(shifted(H, 1e10 * big, b));
        s.hasD = true;
        s.boundary = m.boundary;
    }
    const double area = s.M.sum();
    s.poisson.compute(shifted(L, 1e-8 / area * (4 * M_PI), s.M));
    for (int f = 0; f < m.nF; ++f) {
        const int h = 3 * f;
        Impl::Face fc;
        fc.v[0] = m.F[static_cast<size_t>(h)]; fc.v[1] = m.F[static_cast<size_t>(h) + 1]; fc.v[2] = m.F[static_cast<size_t>(h) + 2];
        const double lab = m.length[static_cast<size_t>(m.edgeOf[static_cast<size_t>(h)])], lbc = m.length[static_cast<size_t>(m.edgeOf[static_cast<size_t>(h) + 1])], lca = m.length[static_cast<size_t>(m.edgeOf[static_cast<size_t>(h) + 2])];
        const double x = (lab * lab + lca * lca - lbc * lbc) / (2 * lab), y = std::sqrt(std::max(0.0, lca * lca - x * x));
        fc.P[0][0] = 0; fc.P[0][1] = 0; fc.P[1][0] = lab; fc.P[1][1] = 0; fc.P[2][0] = x; fc.P[2][1] = y;
        fc.A = delaunay::triangleArea(lab, lbc, lca);
        s.faces.push_back(fc);
    }
}
HeatMethod::~HeatMethod() = default;
int HeatMethod::n() const { return p_->n; }
double HeatMethod::t() const { return p_->t; }

VectorXd HeatMethod::distance(const std::vector<int>& sources) const {
    const Impl& s = *p_;
    const int n = s.n;
    // 1 — heat: (L + M/t) u = M delta / t  (the Studio's right side: 1/t at each source)
    VectorXd rhs = VectorXd::Zero(n);
    for (int src : sources) rhs[src] = 1.0 / s.t;
    VectorXd u = s.heat.solve(rhs);
    bool onBoundary = false;
    if (s.hasD) for (int src : sources) if (s.boundary[static_cast<size_t>(src)]) onBoundary = true;
    if (s.hasD && !onBoundary) u = 0.5 * (u + s.heatD.solve(rhs));
    // 2 — the normalised direction per face, −∇u/|∇u|, 3 — its divergence, 4 — Poisson
    VectorXd div = VectorXd::Zero(n);
    auto cotAt = [](const double* A, const double* B, const double* C) {
        const double ux = B[0] - A[0], uy = B[1] - A[1], vx = C[0] - A[0], vy = C[1] - A[1];
        return (ux * vx + uy * vy) / std::abs(ux * vy - uy * vx);
    };
    for (const Impl::Face& f : s.faces) {
        if (!(f.A > 0)) continue;
        double gx = 0, gy = 0;
        for (int i = 0; i < 3; ++i) {
            const double* q = f.P[(i + 1) % 3];
            const double* r = f.P[(i + 2) % 3];
            const double ex = r[0] - q[0], ey = r[1] - q[1];
            gx += u[f.v[i]] * -ey; gy += u[f.v[i]] * ex;
        }
        const double g = std::hypot(gx, gy);
        if (!(g > 0)) continue;
        const double X[2] = {-gx / g, -gy / g};
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            const double* Pi = f.P[i]; const double* Pj = f.P[j]; const double* Pk = f.P[k];
            const double eij[2] = {Pj[0] - Pi[0], Pj[1] - Pi[1]}, eik[2] = {Pk[0] - Pi[0], Pk[1] - Pi[1]};
            div[f.v[i]] += 0.5 * (cotAt(Pk, Pi, Pj) * (eij[0] * X[0] + eij[1] * X[1]) + cotAt(Pj, Pk, Pi) * (eik[0] * X[0] + eik[1] * X[1]));
        }
    }
    const VectorXd phi = s.poisson.solve(-div);
    // 4 — the sources at 0 and the distances positive
    double at = 0;
    for (int src : sources) at += phi[src];
    at /= static_cast<double>(sources.size());
    const double mean = phi.dot(s.M);
    const double sign = mean / s.M.sum() >= at ? 1.0 : -1.0;
    return sign * (phi.array() - at).matrix();
}

// ── tangentBasis (connectionLaplacian.ts) ──

Basis tangentBasis(const delaunay::Mesh& m, const double* P, const int32_t* F, int nF) {
    if (!m.hasSignposts) throw core::Error(core::ErrorCode::InvalidInput, "tangentBasis: the mesh has no signposts");
    const int nV = m.nV;
    std::vector<double> N(3 * static_cast<size_t>(nV), 0.0);
    for (int f = 0; f < nF; ++f) {
        const int a = F[3 * f], b = F[3 * f + 1], c = F[3 * f + 2];
        const double ux = P[3 * b] - P[3 * a], uy = P[3 * b + 1] - P[3 * a + 1], uz = P[3 * b + 2] - P[3 * a + 2];
        const double vx = P[3 * c] - P[3 * a], vy = P[3 * c + 1] - P[3 * a + 1], vz = P[3 * c + 2] - P[3 * a + 2];
        const double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        for (int v : {a, b, c}) { N[3 * static_cast<size_t>(v)] += nx; N[3 * static_cast<size_t>(v) + 1] += ny; N[3 * static_cast<size_t>(v) + 2] += nz; }
    }
    Basis out;
    out.e1.assign(3 * static_cast<size_t>(nV), 0.f); out.e2 = out.e1; out.n = out.e1;
    for (int v = 0; v < nV; ++v) {
        const size_t o = 3 * static_cast<size_t>(v);
        double l = std::hypot(N[o], N[o + 1], N[o + 2]);
        if (l == 0) l = 1;
        const double nx = N[o] / l, ny = N[o + 1] / l, nz = N[o + 2] / l;
        const int r = m.ref[static_cast<size_t>(v)];
        double x = P[3 * r] - P[3 * v], y = P[3 * r + 1] - P[3 * v + 1], z = P[3 * r + 2] - P[3 * v + 2];
        const double d = x * nx + y * ny + z * nz;
        x -= d * nx; y -= d * ny; z -= d * nz;
        double k = std::hypot(x, y, z);
        if (k == 0) k = 1;
        x /= k; y /= k; z /= k;
        out.n[o] = static_cast<float>(nx); out.n[o + 1] = static_cast<float>(ny); out.n[o + 2] = static_cast<float>(nz);
        out.e1[o] = static_cast<float>(x); out.e1[o + 1] = static_cast<float>(y); out.e1[o + 2] = static_cast<float>(z);
        out.e2[o] = static_cast<float>(ny * z - nz * y); out.e2[o + 1] = static_cast<float>(nz * x - nx * z); out.e2[o + 2] = static_cast<float>(nx * y - ny * x);
    }
    return out;
}

// ── the kit ──

struct ChartKit::Impl {
    struct Comp {
        std::vector<int> verts;                 // global indices, ascending
        Pencil pen;
        Basis basis;
        std::unique_ptr<HeatMethod> heat;
        std::unique_ptr<Eigen::SimplicialLDLT<SpMat>> vheat;   // C + M/t, the vector heat flow's factor
        double t = 0;
    };
    int nV = 0;
    std::vector<double> P;
    std::vector<int32_t> F;
    std::vector<Comp> comps;
    std::vector<int> compOf, local;             // per vertex: its component (−1 in no face) and local index
    Comp& of(int v) {
        if (v < 0 || v >= nV || compOf[static_cast<size_t>(v)] < 0) throw core::Error(core::ErrorCode::InvalidInput, "ChartKit: vertex " + std::to_string(v) + " is on no component");
        return comps[static_cast<size_t>(compOf[static_cast<size_t>(v)])];
    }
    HeatMethod& heatOf(Comp& c) {
        if (!c.heat) c.heat = std::make_unique<HeatMethod>(c.pen.mesh);
        return *c.heat;
    }
    // #vectorHeat: C + M/t, t the mean intrinsic edge squared
    Eigen::SimplicialLDLT<SpMat>& vectorHeat(Comp& c) {
        if (!c.vheat) {
            double h = 0;
            for (double l : c.pen.mesh.length) h += l;
            h /= static_cast<double>(c.pen.mesh.length.size());
            c.t = h * h;
            VectorXd M2(2 * c.pen.M.size());
            for (Eigen::Index i = 0; i < c.pen.M.size(); ++i) { M2[2 * i] = c.pen.M[i]; M2[2 * i + 1] = c.pen.M[i]; }   // connectionMass
            c.vheat = std::make_unique<Eigen::SimplicialLDLT<SpMat>>();
            c.vheat->compute(shifted(c.pen.C, 1.0 / c.t, M2));
        }
        return *c.vheat;
    }
};

ChartKit::ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const delaunay::DelaunayOptions& o) : p_(std::make_unique<Impl>()) {
    Impl& s = *p_;
    s.nV = nV;
    s.P.assign(positions, positions + 3 * static_cast<size_t>(nV));
    s.F.assign(faces, faces + 3 * static_cast<size_t>(nF));
    // connected pieces of the faces
    std::vector<int> parent(static_cast<size_t>(nV));
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) { while (parent[static_cast<size_t>(x)] != x) { parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])]; x = parent[static_cast<size_t>(x)]; } return x; };
    std::vector<char> inFace(static_cast<size_t>(nV), 0);
    for (int f = 0; f < nF; ++f) {
        const int a = faces[3 * f], b = faces[3 * f + 1], c = faces[3 * f + 2];
        inFace[static_cast<size_t>(a)] = inFace[static_cast<size_t>(b)] = inFace[static_cast<size_t>(c)] = 1;
        parent[static_cast<size_t>(find(a))] = find(b); parent[static_cast<size_t>(find(b))] = find(c);
    }
    s.compOf.assign(static_cast<size_t>(nV), -1); s.local.assign(static_cast<size_t>(nV), 0);
    std::vector<int> rootComp(static_cast<size_t>(nV), -1);
    for (int v = 0; v < nV; ++v) {
        if (!inFace[static_cast<size_t>(v)]) continue;
        const int r = find(v);
        if (rootComp[static_cast<size_t>(r)] < 0) { rootComp[static_cast<size_t>(r)] = static_cast<int>(s.comps.size()); s.comps.emplace_back(); }
        Impl::Comp& c = s.comps[static_cast<size_t>(rootComp[static_cast<size_t>(r)])];
        s.compOf[static_cast<size_t>(v)] = rootComp[static_cast<size_t>(r)];
        s.local[static_cast<size_t>(v)] = static_cast<int>(c.verts.size());
        c.verts.push_back(v);
    }
    std::vector<std::vector<int32_t>> cf(s.comps.size());
    for (int f = 0; f < nF; ++f) {
        const int ci = s.compOf[static_cast<size_t>(faces[3 * f])];
        for (int k = 0; k < 3; ++k) cf[static_cast<size_t>(ci)].push_back(s.local[static_cast<size_t>(faces[3 * f + k])]);
    }
    for (size_t ci = 0; ci < s.comps.size(); ++ci) {
        Impl::Comp& c = s.comps[ci];
        std::vector<double> Pc(3 * c.verts.size());
        for (size_t k = 0; k < c.verts.size(); ++k) for (int d = 0; d < 3; ++d) Pc[3 * k + static_cast<size_t>(d)] = positions[3 * static_cast<size_t>(c.verts[k]) + static_cast<size_t>(d)];
        c.pen = componentPencil(Pc.data(), static_cast<int>(c.verts.size()), cf[ci], o);
        c.basis = tangentBasis(c.pen.mesh, Pc.data(), cf[ci].data(), static_cast<int>(cf[ci].size() / 3));
    }
}
ChartKit::~ChartKit() = default;
int ChartKit::n() const { return p_->nV; }

VectorXd ChartKit::heatDistance(int v) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const VectorXd d = s.heatOf(c).distance({s.local[static_cast<size_t>(v)]});
    VectorXd out = VectorXd::Constant(s.nV, std::numeric_limits<double>::infinity());
    for (size_t k = 0; k < c.verts.size(); ++k) out[c.verts[k]] = d[static_cast<Eigen::Index>(k)];
    return out;
}

VectorXd ChartKit::heatDistance(const std::vector<int>& sources) {
    Impl& s = *p_;
    VectorXd out = VectorXd::Constant(s.nV, kNaN);
    if (sources.empty()) return out;
    const int ci = s.compOf[static_cast<size_t>(sources[0])];
    Impl::Comp& c = s.of(sources[0]);
    std::vector<int> loc;
    for (int u : sources) if (s.compOf[static_cast<size_t>(u)] == ci) loc.push_back(s.local[static_cast<size_t>(u)]);
    const VectorXd d = s.heatOf(c).distance(loc);
    for (size_t k = 0; k < c.verts.size(); ++k) out[c.verts[k]] = d[static_cast<Eigen::Index>(k)];
    return out;
}

VectorXd ChartKit::transported(int v, double heading) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const int i = s.local[static_cast<size_t>(v)];
    VectorXd b = VectorXd::Zero(2 * static_cast<Eigen::Index>(c.verts.size()));
    b[2 * i] = std::cos(heading) * c.pen.M[i]; b[2 * i + 1] = std::sin(heading) * c.pen.M[i];
    const VectorXd x = s.vectorHeat(c).solve(b);
    VectorXd out = VectorXd::Constant(2 * s.nV, kNaN);
    for (size_t k = 0; k < c.verts.size(); ++k) { out[2 * c.verts[k]] = x[2 * static_cast<Eigen::Index>(k)]; out[2 * c.verts[k] + 1] = x[2 * static_cast<Eigen::Index>(k) + 1]; }
    return out;
}

Eigen::VectorXd ChartKit::gradient(const VectorXd& field) const {
    const Impl& s = *p_;
    const double* P = s.P.data();
    VectorXd out = VectorXd::Zero(3 * s.nV), w = VectorXd::Zero(s.nV);
    for (size_t f = 0; f < s.F.size(); f += 3) {
        const int idx[3] = {s.F[f], s.F[f + 1], s.F[f + 2]};
        const int a = idx[0], b = idx[1], c = idx[2];
        const double ab[3] = {P[3 * b] - P[3 * a], P[3 * b + 1] - P[3 * a + 1], P[3 * b + 2] - P[3 * a + 2]};
        const double ac[3] = {P[3 * c] - P[3 * a], P[3 * c + 1] - P[3 * a + 1], P[3 * c + 2] - P[3 * a + 2]};
        const double N[3] = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]};
        const double twoA = std::hypot(N[0], N[1], N[2]);
        if (!(twoA > 0)) continue;
        const double n[3] = {N[0] / twoA, N[1] / twoA, N[2] / twoA};
        double g[3] = {0, 0, 0};
        const int tri[3][3] = {{a, b, c}, {b, c, a}, {c, a, b}};
        for (const auto& t : tri) {
            const int i = t[0], j = t[1], k = t[2];
            const double e[3] = {P[3 * k] - P[3 * j], P[3 * k + 1] - P[3 * j + 1], P[3 * k + 2] - P[3 * j + 2]};   // the edge opposite i
            const double ne[3] = {n[1] * e[2] - n[2] * e[1], n[2] * e[0] - n[0] * e[2], n[0] * e[1] - n[1] * e[0]};
            for (int d = 0; d < 3; ++d) g[d] += field[i] * ne[d] / twoA;
        }
        const double A = twoA / 2;
        for (int v : idx) { for (int d = 0; d < 3; ++d) out[3 * v + d] += A * g[d]; w[v] += A; }
    }
    for (int v = 0; v < s.nV; ++v) if (w[v] > 0) for (int d = 0; d < 3; ++d) out[3 * v + d] = out[3 * v + d] / w[v];
    return out;
}

LogMap ChartKit::logMap(int v) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const int i = s.local[static_cast<size_t>(v)];
    VectorXd d = heatDistance(v);
    VectorXd dz = d;
    for (Eigen::Index k = 0; k < dz.size(); ++k) if (!std::isfinite(dz[k])) dz[k] = 0;   // the Studio zeroes the other components' Infinity
    const VectorXd g = gradient(dz);
    const VectorXd X = transported(v, 0.0);
    LogMap out{VectorXd::Constant(s.nV, kNaN), VectorXd::Constant(s.nV, kNaN)};
    const Basis& b = c.basis;
    for (size_t k = 0; k < c.verts.size(); ++k) {
        const int u = c.verts[k];
        const size_t o = 3 * k;
        const double gx = g[3 * u], gy = g[3 * u + 1], gz = g[3 * u + 2];
        const double gr = gx * b.e1[o] + gy * b.e1[o + 1] + gz * b.e1[o + 2], gi = gx * b.e2[o] + gy * b.e2[o + 1] + gz * b.e2[o + 2];
        out.r[u] = d[u];
        // the radial direction relative to the transported reference: arg(radial / reference)
        out.theta[u] = static_cast<int>(k) == i ? 0.0 : std::atan2(gi * X[2 * u] - gr * X[2 * u + 1], gr * X[2 * u] + gi * X[2 * u + 1]);
    }
    return out;
}

double ChartKit::gaugeHeading(int v, const double d[3]) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const size_t o = 3 * static_cast<size_t>(s.local[static_cast<size_t>(v)]);
    const Basis& b = c.basis;
    const double x = d[0] * b.e1[o] + d[1] * b.e1[o + 1] + d[2] * b.e1[o + 2], y = d[0] * b.e2[o] + d[1] * b.e2[o + 1] + d[2] * b.e2[o + 2];
    return std::hypot(x, y) > 1e-9 * std::hypot(d[0], d[1], d[2]) ? std::atan2(y, x) : kNaN;
}

void ChartKit::headingVector(int v, double psi, double out[3]) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const size_t o = 3 * static_cast<size_t>(s.local[static_cast<size_t>(v)]);
    for (size_t k = 0; k < 3; ++k) out[k] = static_cast<double>(static_cast<float>(std::cos(psi) * c.basis.e1[o + k] + std::sin(psi) * c.basis.e2[o + k]));   // Float32Array
}

std::vector<int> ChartKit::disc(int v, double radius, double maxRadius) {
    if (radius > maxRadius) throw core::Error(core::ErrorCode::InvalidInput, "ChartKit::disc: radius " + std::to_string(radius) + " is past the chart's validity radius " + std::to_string(maxRadius));
    const VectorXd d = heatDistance(v);
    std::vector<int> out;
    for (int i = 0; i < p_->nV; ++i) if (d[i] <= radius) out.push_back(i);
    return out;
}

Basis ChartKit::basis() const {
    const Impl& s = *p_;
    Basis out;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    out.e1.assign(3 * static_cast<size_t>(s.nV), nan); out.e2 = out.e1; out.n = out.e1;
    for (const Impl::Comp& c : s.comps)
        for (size_t k = 0; k < c.verts.size(); ++k) for (size_t d = 0; d < 3; ++d) {
            const size_t g = 3 * static_cast<size_t>(c.verts[k]) + d;
            out.e1[g] = c.basis.e1[3 * k + d]; out.e2[g] = c.basis.e2[3 * k + d]; out.n[g] = c.basis.n[3 * k + d];
        }
    return out;
}

}  // namespace nxr::chart_port
