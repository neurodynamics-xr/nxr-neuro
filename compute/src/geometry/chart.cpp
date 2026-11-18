#include "nxr/geometry/charts.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

#include "nxr/core/errors.h"

#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vector_heat_method.h"

namespace nxr::geometry {

using Eigen::VectorXd;
namespace gcs = geometrycentral::surface;
namespace gc = geometrycentral;

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct ChartKit::Impl {
    struct Comp {
        std::vector<int> verts;                                  // global indices, ascending
        geometry::SurfaceArrays surface;                         // the component's arrays (positions, faces, intrinsic triangulation): data, no object
        geometry::TangentBasis basis;                            // the gauge in 3D, [3 n] local
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
};

ChartKit::ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const DelaunayOptions& d)
    : ChartKit(positions, nV, faces, nF, [&d] { geometry::IntrinsicOptions o; o.mollify = d.mollify; o.flipEps = d.flipEps; return o; }()) {}

ChartKit::ChartKit(const double* positions, int nV, const int32_t* faces, int nF, const geometry::IntrinsicOptions& o) : p_(std::make_unique<Impl>()) {
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
        c.surface = geometry::surfaceArrays(Pc.data(), static_cast<int>(c.verts.size()), cf[ci].data(), static_cast<int>(cf[ci].size() / 3), o);
        c.basis = geometry::IntrinsicSurface(c.surface).tangentBasis();
    }
}
ChartKit::~ChartKit() = default;
int ChartKit::n() const { return p_->nV; }

const geometry::SurfaceArrays& ChartKit::surfaceOf(int v) const { return const_cast<Impl&>(*p_).of(v).surface; }
int ChartKit::components() const { return static_cast<int>(p_->comps.size()); }
int ChartKit::componentOf(int v) const { return v >= 0 && v < p_->nV ? p_->compOf[static_cast<size_t>(v)] : -1; }
const std::vector<int>& ChartKit::componentVertices(int c) const { return p_->comps.at(static_cast<size_t>(c)).verts; }
const geometry::SurfaceArrays& ChartKit::componentSurface(int c) const { return p_->comps.at(static_cast<size_t>(c)).surface; }
int ChartKit::localIndex(int v) const { return p_->local.at(static_cast<size_t>(v)); }

// ── batch forms: one geometry-central solver per component per call, all anchors on it; nothing is held after the call ──
namespace {
struct ComponentSolvers {   // what a call builds for the components its anchors lie on; it lives for that call
    const ChartKit& kit;
    std::map<int, std::unique_ptr<geometry::IntrinsicSurface>> surf;
    std::map<int, std::unique_ptr<gcs::HeatMethodDistanceSolver>> heat;
    std::map<int, std::unique_ptr<gcs::VectorHeatMethodSolver>> vheat;
    explicit ComponentSolvers(const ChartKit& k) : kit(k) {}
    int componentOf(int v) const {
        if (kit.componentOf(v) < 0) throw core::Error(core::ErrorCode::InvalidInput, "ChartKit: vertex " + std::to_string(v) + " is on no component");
        return kit.componentOf(v);
    }
    geometry::IntrinsicSurface& surface(int c) {
        auto& p = surf[c];
        if (!p) p = std::make_unique<geometry::IntrinsicSurface>(kit.componentSurface(c));
        return *p;
    }
    gcs::HeatMethodDistanceSolver& heatSolver(int c) {
        auto& p = heat[c];
        if (!p) p = std::make_unique<gcs::HeatMethodDistanceSolver>(surface(c).geometry());
        return *p;
    }
    gcs::VectorHeatMethodSolver& vectorHeatSolver(int c) {
        auto& p = vheat[c];
        if (!p) p = std::make_unique<gcs::VectorHeatMethodSolver>(surface(c).geometry());
        return *p;
    }
};

// the distance of `sources` (local indices) over a component, as a local vector
VectorXd solveDistance(gcs::HeatMethodDistanceSolver& h, geometry::IntrinsicSurface& surf, const std::vector<int>& sources) {
    std::vector<gcs::Vertex> src;
    for (int u : sources) src.push_back(surf.mesh().vertex(static_cast<size_t>(u)));
    const gcs::VertexData<double> d = h.computeDistance(src);
    VectorXd out(surf.nV());
    for (int i = 0; i < surf.nV(); ++i) out[i] = d[static_cast<size_t>(i)];
    return out;
}
}  // namespace

void heatDistanceEach(const ChartKit& kit, const std::vector<int>& anchors, const std::function<void(std::size_t, const VectorXd&)>& visit) {
    ComponentSolvers cs(kit);
    for (std::size_t k = 0; k < anchors.size(); ++k) {
        const int v = anchors[k], c = cs.componentOf(v);
        const VectorXd d = solveDistance(cs.heatSolver(c), cs.surface(c), {kit.localIndex(v)});
        VectorXd out = VectorXd::Constant(kit.n(), std::numeric_limits<double>::infinity());
        const std::vector<int>& verts = kit.componentVertices(c);
        for (size_t i = 0; i < verts.size(); ++i) out[verts[i]] = d[static_cast<Eigen::Index>(i)];
        visit(k, out);
    }
}

VectorXd heatDistanceSet(const ChartKit& kit, const std::vector<int>& sources) {
    VectorXd out = VectorXd::Constant(kit.n(), kNaN);
    if (sources.empty()) return out;
    ComponentSolvers cs(kit);
    const int c = cs.componentOf(sources[0]);
    std::vector<int> loc;
    for (int u : sources) if (kit.componentOf(u) == c) loc.push_back(kit.localIndex(u));
    const VectorXd d = solveDistance(cs.heatSolver(c), cs.surface(c), loc);
    const std::vector<int>& verts = kit.componentVertices(c);
    for (size_t i = 0; i < verts.size(); ++i) out[verts[i]] = d[static_cast<Eigen::Index>(i)];
    return out;
}

void transportedEach(const ChartKit& kit, const std::vector<int>& anchors, const std::vector<double>& headings, const std::function<void(std::size_t, const VectorXd&)>& visit) {
    if (headings.size() != anchors.size() && !headings.empty()) throw core::Error(core::ErrorCode::InvalidInput, "transported: one heading per anchor (or none: all 0)");
    ComponentSolvers cs(kit);
    for (std::size_t k = 0; k < anchors.size(); ++k) {
        const int v = anchors[k], c = cs.componentOf(v);
        const double heading = headings.empty() ? 0.0 : headings[k];
        const gcs::VertexData<gc::Vector2> x = cs.vectorHeatSolver(c).transportTangentVector(cs.surface(c).mesh().vertex(static_cast<size_t>(kit.localIndex(v))), gc::Vector2{std::cos(heading), std::sin(heading)});
        VectorXd out = VectorXd::Constant(2 * kit.n(), kNaN);
        const std::vector<int>& verts = kit.componentVertices(c);
        for (size_t i = 0; i < verts.size(); ++i) { out[2 * verts[i]] = x[i].x; out[2 * verts[i] + 1] = x[i].y; }
        visit(k, out);
    }
}

void logMapEach(const ChartKit& kit, const std::vector<int>& anchors, const std::function<void(std::size_t, const LogMap&)>& visit) {
    ComponentSolvers cs(kit);
    for (std::size_t k = 0; k < anchors.size(); ++k) {
        const int v = anchors[k], c = cs.componentOf(v), i = kit.localIndex(v);
        // theta is the angle of geometry-central's log map (VectorHeat strategy), in the source's gauge; r is the heat distance (the log map's own radius is a coarser estimate:
        // about 8% on a 642-vertex sphere against 1% for the heat distance, G4)
        const gcs::VertexData<gc::Vector2> z = cs.vectorHeatSolver(c).computeLogMap(cs.surface(c).mesh().vertex(static_cast<size_t>(i)), gcs::LogMapStrategy::VectorHeat);
        const VectorXd d = solveDistance(cs.heatSolver(c), cs.surface(c), {i});
        LogMap out{VectorXd::Constant(kit.n(), kNaN), VectorXd::Constant(kit.n(), kNaN)};
        const std::vector<int>& verts = kit.componentVertices(c);
        for (size_t j = 0; j < verts.size(); ++j) {
            const int u = verts[j];
            out.r[u] = d[static_cast<Eigen::Index>(j)];
            out.theta[u] = static_cast<int>(j) == i ? 0.0 : std::atan2(z[j].y, z[j].x);
        }
        visit(k, out);
    }
}

std::vector<std::vector<int>> discs(const ChartKit& kit, const std::vector<int>& anchors, const std::vector<double>& radii, double maxRadius) {
    if (radii.size() != anchors.size()) throw core::Error(core::ErrorCode::InvalidInput, "discs: one radius per anchor");
    for (double r : radii)
        if (r > maxRadius) throw core::Error(core::ErrorCode::InvalidInput, "ChartKit::disc: radius " + std::to_string(r) + " is past the chart's validity radius " + std::to_string(maxRadius));
    std::vector<std::vector<int>> out(anchors.size());
    heatDistanceEach(kit, anchors, [&](std::size_t k, const VectorXd& d) {
        for (int i = 0; i < kit.n(); ++i) if (d[i] <= radii[k]) out[k].push_back(i);
    });
    return out;
}
std::vector<std::vector<int>> discs(const ChartKit& kit, const std::vector<int>& anchors, double radius, double maxRadius) {
    return discs(kit, anchors, std::vector<double>(anchors.size(), radius), maxRadius);
}

// single-anchor forms: the batch with one anchor (one solver, dropped on return)
VectorXd heatDistance(const ChartKit& kit, int v) { VectorXd r; heatDistanceEach(kit, {v}, [&](std::size_t, const VectorXd& d) { r = d; }); return r; }
VectorXd transported(const ChartKit& kit, int v, double heading) { VectorXd r; transportedEach(kit, {v}, {heading}, [&](std::size_t, const VectorXd& x) { r = x; }); return r; }
LogMap logMap(const ChartKit& kit, int v) { LogMap r; logMapEach(kit, {v}, [&](std::size_t, const LogMap& m) { r = m; }); return r; }
std::vector<int> disc(const ChartKit& kit, int v, double radius, double maxRadius) { return discs(kit, {v}, radius, maxRadius)[0]; }

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

double ChartKit::gaugeHeading(int v, const double d[3]) const {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const size_t o = 3 * static_cast<size_t>(s.local[static_cast<size_t>(v)]);
    const geometry::TangentBasis& b = c.basis;
    const double x = d[0] * b.e1[o] + d[1] * b.e1[o + 1] + d[2] * b.e1[o + 2], y = d[0] * b.e2[o] + d[1] * b.e2[o + 1] + d[2] * b.e2[o + 2];
    return std::hypot(x, y) > 1e-9 * std::hypot(d[0], d[1], d[2]) ? std::atan2(y, x) : kNaN;
}

void ChartKit::headingVector(int v, double psi, double out[3]) const {
    Impl& s = *p_;
    Impl::Comp& c = s.of(v);
    const size_t o = 3 * static_cast<size_t>(s.local[static_cast<size_t>(v)]);
    for (size_t k = 0; k < 3; ++k) out[k] = std::cos(psi) * c.basis.e1[o + k] + std::sin(psi) * c.basis.e2[o + k];
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

}  // namespace nxr::geometry
