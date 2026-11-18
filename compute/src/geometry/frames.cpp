// nxr::geometry — canonical (north / west) and anatomical (anterior / superior-or-medial) frames. See nxr/geometry/frames.h.
// Canonical: a line port of Rheome's +rheome/+geom/sphereframe.m (the same names, the same constants: eps floors, condition from the 2x2 Jacobian's singular values).
#include "nxr/geometry/frames.h"

#include "geometrycentral/surface/vertex_position_geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace nxr::geometry {

using Eigen::Vector3d;
using Eigen::VectorXd;
static const double kNaN = std::numeric_limits<double>::quiet_NaN();

namespace {
Vector3d row(const double* p, int i) { return Vector3d(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
// frames.py tangent_basis / sphereframe.m i_tangent: any orthonormal pair spanning the tangent plane (ref x, or y where |n_x| >= 0.9)
void anyTangent(const Vector3d& n, Vector3d& b1, Vector3d& b2) {
    const Vector3d ref = std::abs(n.x()) >= 0.9 ? Vector3d(0, 1, 0) : Vector3d(1, 0, 0);
    b1 = ref - ref.dot(n) * n;
    b1 /= b1.norm();
    b2 = n.cross(b1);
}
double wrap(double a) { return std::remainder(a, 2 * M_PI); }
}  // namespace

Canonical canonicalFrames(const double* V, int nV, const int32_t* F, int nF, const double* N, const double* Sph, const Vector3d& axisIn, double maxCondition) {
    const double eps = std::numeric_limits<double>::epsilon();
    const Vector3d ax = axisIn / axisIn.norm();
    Canonical o;
    o.axis = ax;
    o.e1.resize(nV, 3); o.e2.resize(nV, 3);
    o.colat.resize(nV); o.lon.resize(nV); o.condition.resize(nV);
    o.singular.assign(static_cast<size_t>(nV), 0);
    std::vector<Vector3d> u(static_cast<size_t>(nV)), nn(static_cast<size_t>(nV)), es(static_cast<size_t>(nV));
    std::vector<double> c(static_cast<size_t>(nV));
    // the longitude reference (i_basis): for +z it is (x, y), so lon = atan2(y, x) as on fsaverage
    Vector3d a1 = std::abs(ax.z()) > 0.9 ? Vector3d(1, 0, 0) : Vector3d(0, 0, 1);
    a1 -= a1.dot(ax) * ax; a1 /= a1.norm();
    const Vector3d a2 = ax.cross(a1);
    for (int i = 0; i < nV; ++i) {
        const Vector3d s = row(Sph, i);
        u[i] = s / s.norm();
        nn[i] = row(N, i) / row(N, i).norm();
        c[i] = u[i].dot(ax);
        const Vector3d mer = ax - c[i] * u[i];                       // the meridian towards +axis, tangent
        es[i] = mer / std::max(mer.norm(), eps);
        o.lon[i] = std::atan2(u[i].dot(a2), u[i].dot(a1));
        o.colat[i] = std::acos(std::max(-1.0, std::min(1.0, c[i])));
    }
    // per-vertex Jacobian sphere -> cortex, least squares over the directed 1-ring edges
    std::vector<Vector3d> sb1(static_cast<size_t>(nV)), sb2(static_cast<size_t>(nV)), db1(static_cast<size_t>(nV)), db2(static_cast<size_t>(nV));
    for (int i = 0; i < nV; ++i) { anyTangent(u[i], sb1[i], sb2[i]); anyTangent(nn[i], db1[i], db2[i]); }
    std::vector<std::pair<int, int>> E;
    E.reserve(static_cast<size_t>(3 * nF));
    for (int f = 0; f < nF; ++f) for (int k = 0; k < 3; ++k) {
        int a = F[3 * f + k], b = F[3 * f + (k + 1) % 3];
        E.emplace_back(std::min(a, b), std::max(a, b));
    }
    std::sort(E.begin(), E.end());
    E.erase(std::unique(E.begin(), E.end()), E.end());
    std::vector<double> S11(static_cast<size_t>(nV), 0), S12(static_cast<size_t>(nV), 0), S22(static_cast<size_t>(nV), 0), C11(static_cast<size_t>(nV), 0), C12(static_cast<size_t>(nV), 0),
        C21(static_cast<size_t>(nV), 0), C22(static_cast<size_t>(nV), 0);
    for (const auto& e : E) for (int dir = 0; dir < 2; ++dir) {
        const int A = dir ? e.second : e.first, B = dir ? e.first : e.second;
        const Vector3d ds = u[B] - u[A], dd = row(V, B) - row(V, A);
        const double s1 = ds.dot(sb1[A]), s2 = ds.dot(sb2[A]), c1 = dd.dot(db1[A]), c2 = dd.dot(db2[A]);
        S11[A] += s1 * s1; S12[A] += s1 * s2; S22[A] += s2 * s2;
        C11[A] += c1 * s1; C12[A] += c1 * s2; C21[A] += c2 * s1; C22[A] += c2 * s2;
    }
    for (int i = 0; i < nV; ++i) {
        const double dS = S11[i] * S22[i] - S12[i] * S12[i];       // inv(SS) = [S22 -S12; -S12 S11] / dS
        const double J11 = (C11[i] * S22[i] - C12[i] * S12[i]) / dS, J12 = (C12[i] * S11[i] - C11[i] * S12[i]) / dS;
        const double J21 = (C21[i] * S22[i] - C22[i] * S12[i]) / dS, J22 = (C22[i] * S11[i] - C21[i] * S12[i]) / dS;
        const double m1 = es[i].dot(sb1[i]), m2 = es[i].dot(sb2[i]);
        Vector3d e1 = (J11 * m1 + J12 * m2) * db1[i] + (J21 * m1 + J22 * m2) * db2[i];
        e1 /= std::max(e1.norm(), eps);
        const double tr = J11 * J11 + J12 * J12 + J21 * J21 + J22 * J22, dt = std::abs(J11 * J22 - J12 * J21);
        const double disc = std::sqrt(std::max(tr * tr - 4 * dt * dt, 0.0));
        o.condition[i] = std::sqrt((tr + disc) / std::max(tr - disc, std::numeric_limits<double>::min()));
        o.singular[i] = !(o.condition[i] <= maxCondition);
        o.e1.row(i) = e1.transpose();
        o.e2.row(i) = nn[i].cross(e1).transpose();
    }
    // the poles: the vertices closest to +axis / -axis, of EACH connected piece (a cortex's registered sphere carries both hemispheres; Rheome's is called per hemisphere)
    const std::vector<int> piece = pieces(F, nF, nV);
    int nPieces = 0;
    for (int k : piece) nPieces = std::max(nPieces, k + 1);
    std::vector<int> hi(static_cast<size_t>(nPieces), -1), lo(static_cast<size_t>(nPieces), -1);
    for (int i = 0; i < nV; ++i) { const int k = piece[i]; if (hi[k] < 0 || c[i] > c[hi[k]]) hi[k] = i; if (lo[k] < 0 || c[i] < c[lo[k]]) lo[k] = i; }
    for (int k = 0; k < nPieces; ++k) { o.singular[hi[k]] = 1; o.singular[lo[k]] = 1; o.poles.push_back(hi[k]); o.poles.push_back(lo[k]); }
    o.north = hi.empty() ? 0 : hi[0]; o.south = lo.empty() ? 0 : lo[0];
    for (int i = 0; i < nV; ++i) if (o.singular[i]) { o.e1.row(i).setConstant(kNaN); o.e2.row(i).setConstant(kNaN); }
    return o;
}

Mat3 vertexNormals(geometry::IntrinsicSurface& surface) {
    auto& E = surface.embedded();
    E.requireVertexNormals();
    const int nV = surface.nV();
    Mat3 n(nV, 3);
    for (int i = 0; i < nV; ++i) { const auto N = E.vertexNormals[static_cast<size_t>(i)]; n(i, 0) = N.x; n(i, 1) = N.y; n(i, 2) = N.z; }
    E.unrequireVertexNormals();
    return n;
}

Mat3 vertexNormals(const double* P, int nV, const int32_t* F, int nF) {
    // geometry-central's `vertexNormals` formula (the sum of the incident faces' unit normals, each weighted by the corner angle at the vertex, normalised), written on arrays so that a mesh
    // that is not (yet) a manifold, or has vertices in no face, still has normals; equal to `vertexNormals(surface)` (tested)
    Mat3 n = Mat3::Zero(nV, 3);
    for (int f = 0; f < nF; ++f) {
        const int v[3] = {F[3 * f], F[3 * f + 1], F[3 * f + 2]};
        const Vector3d p[3] = {row(P, v[0]), row(P, v[1]), row(P, v[2])};
        Vector3d fn = (p[1] - p[0]).cross(p[2] - p[0]);
        const double l = fn.norm();
        if (!(l > 0)) continue;
        fn /= l;
        for (int k = 0; k < 3; ++k) {
            const Vector3d a = (p[(k + 1) % 3] - p[k]).normalized(), b = (p[(k + 2) % 3] - p[k]).normalized();
            const double angle = std::atan2(a.cross(b).norm(), a.dot(b));
            n.row(v[k]) += angle * fn.transpose();
        }
    }
    for (int i = 0; i < nV; ++i) { const double l = n.row(i).norm(); if (l > 0) n.row(i) /= l; }
    return n;
}

VectorXd gaugeAngle(geometry::IntrinsicSurface& surface, const Mat3& d) {
    if (d.rows() != surface.nV()) throw std::invalid_argument("gaugeAngle: one direction per vertex of the surface");
    const geometry::TangentBasis b = surface.tangentBasis();
    VectorXd out(d.rows());
    for (int i = 0; i < d.rows(); ++i) {
        const size_t o = 3 * static_cast<size_t>(i);
        const Vector3d v = d.row(i).transpose();
        if (!v.allFinite()) { out[i] = kNaN; continue; }
        const double x = v[0] * b.e1[o] + v[1] * b.e1[o + 1] + v[2] * b.e1[o + 2], y = v[0] * b.e2[o] + v[1] * b.e2[o + 1] + v[2] * b.e2[o + 2];
        out[i] = std::hypot(x, y) > 1e-9 * v.norm() ? std::atan2(y, x) : kNaN;
    }
    return out;
}

VectorXd gaugeAngle(ChartKit& kit, const Mat3& d) {
    VectorXd out(d.rows());
    for (int i = 0; i < d.rows(); ++i) {
        const double v[3] = {d(i, 0), d(i, 1), d(i, 2)};
        out[i] = (std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2])) ? kit.gaugeHeading(i, v) : kNaN;
    }
    return out;
}

// ── anatomical ───────────────────────────────────────────────────────────────────────────────────────────

namespace {
// +1 left / -1 right per vertex: the option, or by connected piece the sign of the piece's mean position along `left` against the whole surface's mean
std::vector<int> sides(const double* P, int nV, const int32_t* F, int nF, const AnatomicalOptions& o) {
    std::vector<int> s(static_cast<size_t>(nV), o.side == Side::Right ? -1 : 1);
    if (o.side != Side::Auto) return s;
    std::vector<int> par(static_cast<size_t>(nV));
    std::iota(par.begin(), par.end(), 0);
    auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    for (int f = 0; f < nF; ++f) { par[find(F[3 * f])] = find(F[3 * f + 1]); par[find(F[3 * f + 1])] = find(F[3 * f + 2]); }
    const Vector3d L = o.axes.left.normalized();
    std::vector<double> sum(static_cast<size_t>(nV), 0.0), cnt(static_cast<size_t>(nV), 0.0);
    double all = 0;
    int pieces = 0;
    for (int i = 0; i < nV; ++i) { const double p = row(P, i).dot(L); all += p; const int r = find(i); if (cnt[r] == 0) ++pieces; sum[r] += p; cnt[r] += 1; }
    if (pieces < 2) throw std::invalid_argument("anatomicalFrame: a single-piece surface needs its hemisphere (Side::Left or Side::Right)");
    all /= nV;
    for (int i = 0; i < nV; ++i) { const int r = find(i); s[i] = sum[r] / cnt[r] - all >= 0 ? 1 : -1; }
    return s;
}
}  // namespace

Anatomical anatomicalProjection(const double* P, int nV, const int32_t* F, int nF, const Mat3& N, const AnatomicalOptions& o) {
    Anatomical a;
    a.angle = VectorXd::Constant(nV, kNaN); a.projection.resize(nV);
    a.anterior = Mat3::Constant(nV, 3, kNaN); a.secondAxis = Mat3::Constant(nV, 3, kNaN);
    a.second.assign(static_cast<size_t>(nV), 0); a.handed.assign(static_cast<size_t>(nV), 1); a.reliable.assign(static_cast<size_t>(nV), 0);
    a.definition = "scs";
    a.reference = "none";
    const std::vector<int> side = sides(P, nV, F, nF, o);
    const Vector3d A = o.axes.anterior.normalized(), S = o.axes.superior.normalized(), L = o.axes.left.normalized();
    for (int i = 0; i < nV; ++i) {
        const Vector3d n = N.row(i).transpose();
        Vector3d t = A - A.dot(n) * n;
        const double proj = t.norm();
        a.projection[i] = proj;
        if (!(proj > 1e-12)) continue;
        t /= proj;
        // the second axis: superior where it lies in the surface, medial where that does; its sign is anatomical
        const Vector3d q = n.cross(t);
        const Vector3d medial = -static_cast<double>(side[i]) * L;
        const double cs = S.dot(q), cm = medial.dot(q);
        if (std::abs(cs) >= std::abs(cm)) { a.second[i] = static_cast<int8_t>(Second::Superior); a.handed[i] = cs >= 0 ? 1 : -1; }
        else { a.second[i] = static_cast<int8_t>(Second::Medial); a.handed[i] = cm >= 0 ? 1 : -1; }
        a.anterior.row(i) = t.transpose();
        a.secondAxis.row(i) = (a.handed[i] * q).transpose();
        a.reliable[i] = proj >= o.minProjection;
    }
    return a;
}

Anatomical anatomicalFrame(const double* P, int nV, const int32_t* F, int nF, const Mat3& N, const Mat3& E1, const Mat3& E2, const AnatomicalOptions& o) {
    Anatomical a = anatomicalProjection(P, nV, F, nF, N, o);
    a.reference = "canonical frames";
    for (int i = 0; i < nV; ++i) {
        if (!(a.projection[i] > 1e-12)) continue;
        const Vector3d t = a.anterior.row(i).transpose(), e1 = E1.row(i).transpose(), e2 = E2.row(i).transpose();
        // the angle from north towards west
        a.angle[i] = (e1.allFinite() && e2.allFinite()) ? std::atan2(t.dot(e2), t.dot(e1)) : kNaN;
        a.reliable[i] = a.projection[i] >= o.minProjection && std::isfinite(a.angle[i]);
    }
    return a;
}

std::vector<int> pieces(const int32_t* F, int nF, int nV) {
    std::vector<int> par(static_cast<size_t>(nV));
    std::iota(par.begin(), par.end(), 0);
    auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    for (int f = 0; f < nF; ++f) { par[find(F[3 * f])] = find(F[3 * f + 1]); par[find(F[3 * f + 1])] = find(F[3 * f + 2]); }
    std::vector<int> id(static_cast<size_t>(nV), -1), out(static_cast<size_t>(nV));
    int k = 0;
    for (int i = 0; i < nV; ++i) { const int r = find(i); if (id[r] < 0) id[r] = k++; out[i] = id[r]; }
    return out;
}

namespace {
// the nearest points on the unit sphere: a grid over the unit cube; a query looks in growing rings of cells
struct SphereGrid {
    std::vector<Vector3d> p; std::vector<int> index; int G = 1; double h = 2;
    std::vector<std::vector<int>> bins;
    int cell(double x) const { return std::min(G - 1, std::max(0, static_cast<int>((x + 1.0) / h))); }
    void build(const std::vector<Vector3d>& pts, const std::vector<int>& idx) {
        p = pts; index = idx;
        G = std::max(1, static_cast<int>(std::cbrt(pts.size() / 2.0))); h = 2.0 / G;
        bins.assign(static_cast<size_t>(G) * G * G, {});
        for (size_t i = 0; i < pts.size(); ++i) bins[(static_cast<size_t>(cell(pts[i].x())) * G + cell(pts[i].y())) * G + cell(pts[i].z())].push_back(static_cast<int>(i));
    }
    int nearest(const Vector3d& q) const {
        const int cx = cell(q.x()), cy = cell(q.y()), cz = cell(q.z());
        int best = -1;
        double bd = std::numeric_limits<double>::infinity();
        for (int r = 0; r <= G; ++r) {
            for (int x = std::max(0, cx - r); x <= std::min(G - 1, cx + r); ++x) for (int y = std::max(0, cy - r); y <= std::min(G - 1, cy + r); ++y) for (int z = std::max(0, cz - r); z <= std::min(G - 1, cz + r); ++z) {
                if (std::max({std::abs(x - cx), std::abs(y - cy), std::abs(z - cz)}) != r) continue;
                for (int i : bins[(static_cast<size_t>(x) * G + y) * G + z]) { const double d = (p[i] - q).norm(); if (d < bd) { bd = d; best = i; } }
            }
            if (best >= 0 && bd <= r * h) break;
        }
        return best < 0 ? -1 : index[best];
    }
};
}  // namespace

Anatomical carryAnatomical(const Anatomical& T, const double* tSph, int nT, const std::vector<int>& tPiece, const double* sSph, int nS, const std::vector<int>& sPiece, const Mat3& subjectE1) {
    int nPieces = 1;
    for (int k : tPiece) nPieces = std::max(nPieces, k + 1);
    std::vector<SphereGrid> grids(static_cast<size_t>(nPieces));
    for (int k = 0; k < nPieces; ++k) {
        std::vector<Vector3d> pts; std::vector<int> idx;
        for (int i = 0; i < nT; ++i) if ((tPiece.empty() ? 0 : tPiece[i]) == k) { pts.push_back(row(tSph, i).normalized()); idx.push_back(i); }
        if (!pts.empty()) grids[k].build(pts, idx);
    }
    Anatomical o;
    o.angle.resize(nS); o.projection.resize(nS);
    o.second.assign(static_cast<size_t>(nS), 0); o.handed.assign(static_cast<size_t>(nS), 1); o.reliable.assign(static_cast<size_t>(nS), 0);
    o.definition = "template";
    o.reference = T.reference;
    for (int v = 0; v < nS; ++v) {
        const int k = sPiece.empty() ? 0 : sPiece[v];
        const int best = k < nPieces && !grids[k].p.empty() ? grids[k].nearest(row(sSph, v).normalized()) : -1;
        if (best < 0 || !subjectE1.row(v).allFinite()) { o.angle[v] = kNaN; o.projection[v] = best < 0 ? kNaN : T.projection[best]; continue; }
        o.angle[v] = T.angle[best]; o.projection[v] = T.projection[best];
        o.second[v] = T.second[best]; o.handed[v] = T.handed[best];
        o.reliable[v] = T.reliable[best] && std::isfinite(T.angle[best]);
    }
    return o;
}

Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> anatomicalComponents(const Anatomical& a, const Mat3& Vec) {
    if (a.anterior.rows() != Vec.rows()) throw std::invalid_argument("anatomicalComponents: the projection has no 3D axes for these vertices (frame-free SCS projection only; the carried template needs the frames)");
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> out(Vec.rows(), 2);
    for (int i = 0; i < Vec.rows(); ++i) {
        if (!a.reliable[i]) { out.row(i).setConstant(kNaN); continue; }
        out(i, 0) = Vec.row(i).dot(a.anterior.row(i));
        out(i, 1) = Vec.row(i).dot(a.secondAxis.row(i));
    }
    return out;
}

Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> anatomicalComponents(const Anatomical& a, const Mat3& E1, const Mat3& E2, const Mat3& Vec) {
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> out(Vec.rows(), 2);
    for (int i = 0; i < Vec.rows(); ++i) {
        if (!a.reliable[i]) { out.row(i).setConstant(kNaN); continue; }
        const double c1 = Vec.row(i).dot(E1.row(i)), c2 = Vec.row(i).dot(E2.row(i)), ca = std::cos(a.angle[i]), sa = std::sin(a.angle[i]);
        out(i, 0) = c1 * ca + c2 * sa;
        out(i, 1) = a.handed[i] * (-c1 * sa + c2 * ca);
    }
    return out;
}

VectorXd angleFromAnterior(const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>& c) {
    VectorXd out(c.rows());
    for (int i = 0; i < c.rows(); ++i) out[i] = std::atan2(c(i, 1), c(i, 0));
    return out;
}

VectorXd angleFromAnterior(const Anatomical& a, const VectorXd& heading) {
    VectorXd out(heading.size());
    for (int i = 0; i < heading.size(); ++i) out[i] = a.reliable[i] ? wrap(a.handed[i] * (heading[i] - a.angle[i])) : kNaN;
    return out;
}

}  // namespace nxr::geometry
