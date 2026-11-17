#include "nxr/chart.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "nxr/errors.h"

#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vector_heat_method.h"

namespace nxr::chart {

using Eigen::VectorXd;
namespace gcs = geometrycentral::surface;
namespace gc = geometrycentral;

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct ChartKit::Impl {
    struct Comp {
        std::vector<int> verts;                                  // global indices, ascending
        std::unique_ptr<manifold::IntrinsicSurface> surface;
        manifold::TangentBasis basis;                            // the gauge in 3D, [3 n] local
        std::unique_ptr<gcs::HeatMethodDistanceSolver> heat;
        std::unique_ptr<gcs::VectorHeatMethodSolver> vheat;
    };
    int nV = 0;
    std::vector<double> P;
    std::vector<int32_t> F;
    std::vector<Comp> comps;
    std::vector<int> compOf, local;                              // per vertex: its component (-1 in no face) and local index
    Comp& of(int v) {
        if (v < 0 || v >= nV || compOf[static_cast<size_t>(v)] < 0) throw core::Error(core::ErrorCode::InvalidInput, "ChartKit: vertex " + std::to_string(v) + " is on no component");
        return comps[static_cast<size_t>(compOf[static_cast<size_t>(v)])];
    }
    gcs::HeatMethodDistanceSolver& heatOf(Comp& c) {
        if (!c.heat) c.heat = std::make_unique<gcs::HeatMethodDistanceSolver>(c.surface->geometry());
        return *c.heat;
    }
    gcs::VectorHeatMethodSolver& vectorHeatOf(Comp& c) {
        if (!c.vheat) c.vheat = std::make_unique<gcs::VectorHeatMethodSolver>(c.surface->geometry());
        return *c.vheat;
    }
};

ChartKit::ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const DelaunayOptions& d)
    : ChartKit(positions, nV, faces, nF, [&d] { manifold::IntrinsicOptions o; o.mollify = d.mollify; o.flipEps = d.flipEps; return o; }()) {}

ChartKit::ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const manifold::IntrinsicOptions& o) : p_(std::make_unique<Impl>()) {
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
        c.surface = std::make_unique<manifold::IntrinsicSurface>(Pc.data(), static_cast<int>(c.verts.size()), cf[ci].data(), static_cast<int>(cf[ci].size() / 3), o);
        c.basis = c.surface->tangentBasis();
    }
}
ChartKit::~ChartKit() = default;
int ChartKit::n() const { return p_->nV; }

manifold::IntrinsicSurface& ChartKit::surfaceOf(int v) { return *p_->of(v).surface; }
int ChartKit::localIndex(int v) const { return p_->local.at(static_cast<size_t>(v)); }

namespace {
// the distance of `sources` (local indices) over a component, as a local vector
VectorXd solveDistance(gcs::HeatMethodDistanceSolver& h, manifold::IntrinsicSurface& surf, const std::vector<int>& sources) {
    std::vector<gcs::Vertex> src;
    for (int u : sources) src.push_back(surf.mesh().vertex(static_cast<size_t>(u)));
    const gcs::VertexData<double> d = h.computeDistance(src);
    VectorXd out(surf.nV());
    for (int i = 0; i < surf.nV(); ++i) out[i] = d[static_cast<size_t>(i)];
    return out;
}
}  // namespace

VectorXd ChartKit::heatDistance(int v) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const VectorXd d = solveDistance(s.heatOf(c), *c.surface, {s.local[static_cast<size_t>(v)]});
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
    const VectorXd d = solveDistance(s.heatOf(c), *c.surface, loc);
    for (size_t k = 0; k < c.verts.size(); ++k) out[c.verts[k]] = d[static_cast<Eigen::Index>(k)];
    return out;
}

VectorXd ChartKit::transported(int v, double heading) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const gcs::VertexData<gc::Vector2> x = s.vectorHeatOf(c).transportTangentVector(c.surface->mesh().vertex(static_cast<size_t>(s.local[static_cast<size_t>(v)])), gc::Vector2{std::cos(heading), std::sin(heading)});
    VectorXd out = VectorXd::Constant(2 * s.nV, kNaN);
    for (size_t k = 0; k < c.verts.size(); ++k) { out[2 * c.verts[k]] = x[k].x; out[2 * c.verts[k] + 1] = x[k].y; }
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
    // theta is the angle of geometry-central's log map (VectorHeat strategy), in the source's gauge; r is the heat distance (the log map's own radius is a coarser estimate:
    // about 8% on a 642-vertex sphere against 1% for the heat distance, G4)
    const gcs::VertexData<gc::Vector2> z = s.vectorHeatOf(c).computeLogMap(c.surface->mesh().vertex(static_cast<size_t>(i)), gcs::LogMapStrategy::VectorHeat);
    const VectorXd d = heatDistance(v);
    LogMap out{VectorXd::Constant(s.nV, kNaN), VectorXd::Constant(s.nV, kNaN)};
    for (size_t k = 0; k < c.verts.size(); ++k) {
        const int u = c.verts[k];
        out.r[u] = d[u];
        out.theta[u] = static_cast<int>(k) == i ? 0.0 : std::atan2(z[k].y, z[k].x);
    }
    return out;
}

double ChartKit::gaugeHeading(int v, const double d[3]) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const size_t o = 3 * static_cast<size_t>(s.local[static_cast<size_t>(v)]);
    const manifold::TangentBasis& b = c.basis;
    const double x = d[0] * b.e1[o] + d[1] * b.e1[o + 1] + d[2] * b.e1[o + 2], y = d[0] * b.e2[o] + d[1] * b.e2[o + 1] + d[2] * b.e2[o + 2];
    return std::hypot(x, y) > 1e-9 * std::hypot(d[0], d[1], d[2]) ? std::atan2(y, x) : kNaN;
}

void ChartKit::headingVector(int v, double psi, double out[3]) {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const size_t o = 3 * static_cast<size_t>(s.local[static_cast<size_t>(v)]);
    for (size_t k = 0; k < 3; ++k) out[k] = std::cos(psi) * c.basis.e1[o + k] + std::sin(psi) * c.basis.e2[o + k];
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
    out.e1.assign(3 * static_cast<size_t>(s.nV), kNaN); out.e2 = out.e1; out.n = out.e1;
    for (const Impl::Comp& c : s.comps)
        for (size_t k = 0; k < c.verts.size(); ++k) for (size_t d = 0; d < 3; ++d) {
            const size_t g = 3 * static_cast<size_t>(c.verts[k]) + d;
            out.e1[g] = c.basis.e1[3 * k + d]; out.e2[g] = c.basis.e2[3 * k + d]; out.n[g] = c.basis.n[3 * k + d];
        }
    return out;
}

}  // namespace nxr::chart
