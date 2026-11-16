// THE ONE INTRINSIC DELAUNAY CONSTRUCTION: a line port of the Studio's intrinsicDelaunay.ts (see nxr/delaunay.h).
// Statements keep the reference's order and arithmetic so that the faces, lengths, mass, flip count and signposts agree to rounding.
#include "nxr/delaunay.h"

#include "nxr/errors.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>

namespace nxr::manifold::delaunay {

using core::Error;
using core::ErrorCode;

namespace {

// JavaScript's Math.min / Math.max: a NaN argument gives NaN (std::min / std::max would silently drop it).
inline double jmax(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? std::numeric_limits<double>::quiet_NaN() : (a > b ? a : b); }
inline double jmin(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? std::numeric_limits<double>::quiet_NaN() : (a < b ? a : b); }
inline double clamp1(double x) { return jmax(-1.0, jmin(1.0, x)); }

// JavaScript's Math.hypot (V8): scaled by the largest |x|, Kahan-summed — what the reference's edge lengths are made with. (With strict IEEE
// arithmetic: this file is compiled with -ffp-contract=off so that no fused multiply-add changes a last bit.)
inline double hypot3(double x, double y, double z) {
    const double a[3] = {std::fabs(x), std::fabs(y), std::fabs(z)};
    double mx = 0;
    for (double v : a) { if (std::isnan(v)) return std::numeric_limits<double>::quiet_NaN(); if (v > mx) mx = v; }
    if (std::isinf(mx)) return std::numeric_limits<double>::infinity();
    if (mx == 0) return 0;
    double sum = 0, compensation = 0;
    for (double v : a) {
        const double n = v / mx;
        const double summand = n * n - compensation;
        const double preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    return std::sqrt(sum) * mx;
}

inline double L(const Mesh& m, int h) { return m.length[static_cast<size_t>(m.edgeOf[static_cast<size_t>(h)])]; }

}  // namespace

Mesh makeMesh(const double* P, int nV, const int32_t* faces, int nF) {
    Mesh m;
    m.nV = nV; m.nF = nF;
    const size_t H = 3 * static_cast<size_t>(nF);
    m.F.assign(faces, faces + H);
    m.twin.assign(H, -1);
    m.edgeOf.assign(H, -1);
    std::unordered_map<int64_t, int> open;            // (tail, head) → halfedge, waiting for its twin
    auto key = [&](int64_t a, int64_t b) { return a * nV + b; };
    for (int h = 0; h < static_cast<int>(H); ++h) {
        const int a = m.F[h], b = m.F[nextHe(h)];
        if (a < 0 || a >= nV || b < 0 || b >= nV) throw Error(ErrorCode::InvalidInput, "delaunay::makeMesh: a face index is outside [0, nV)");
        auto it = open.find(key(b, a));
        if (it != open.end()) {
            const int t = it->second;
            open.erase(it);
            m.twin[h] = t; m.twin[t] = h; m.edgeOf[h] = m.edgeOf[t];
        } else {
            if (open.count(key(a, b)))
                throw Error(ErrorCode::InvalidInput, "delaunay::makeMesh: the edge " + std::to_string(a) + "->" + std::to_string(b) +
                            " appears twice in one orientation — not oriented or not manifold");
            open[key(a, b)] = h;
            const int e = static_cast<int>(m.length.size());
            m.edgeOf[h] = e; m.edgeHe.push_back(h);
            const double dx = P[3 * b] - P[3 * a], dy = P[3 * b + 1] - P[3 * a + 1], dz = P[3 * b + 2] - P[3 * a + 2];
            m.length.push_back(hypot3(dx, dy, dz));
        }
    }
    // what is still open is the BOUNDARY: its halfedges keep twin −1, their vertices are boundary vertices
    m.boundary.assign(static_cast<size_t>(nV), 0);
    for (const auto& kv : open) { m.boundary[m.F[kv.second]] = 1; m.boundary[m.F[nextHe(kv.second)]] = 1; }
    return m;
}

double triangleArea(double a, double b, double c) {
    const double s = (a + b + c) / 2;
    return std::sqrt(jmax(0.0, s * (s - a) * (s - b) * (s - c)));
}

double cornerAngle(const Mesh& m, int h) {
    const int h1 = nextHe(h), h2 = nextHe(h1);
    const double a = L(m, h), b = L(m, h2), opp = L(m, h1);
    return std::acos(clamp1((a * a + b * b - opp * opp) / (2 * a * b)));
}

double halfCot(const Mesh& m, int h) {
    const int h1 = nextHe(h), h2 = nextHe(h1);
    const double lij = L(m, h), ljk = L(m, h1), lki = L(m, h2);
    const double A = triangleArea(lij, ljk, lki);
    return (-lij * lij + ljk * ljk + lki * lki) / (4 * A) / 2;
}

double edgeWeight(const Mesh& m, int e) {
    const int h = m.edgeHe[static_cast<size_t>(e)], t = m.twin[static_cast<size_t>(h)];
    return halfCot(m, h) + (t >= 0 ? halfCot(m, t) : 0.0);
}

double mollify(Mesh& m, const DelaunayOptions& o) {
    const double relativeFactor = o.mollify;
    double mean = 0;
    for (double l : m.length) mean += l;
    const double delta = (mean / static_cast<double>(m.length.size())) * relativeFactor;
    double eps = 0;
    for (int h = 0; h < 3 * m.nF; ++h) {
        const double lA = L(m, h), lB = L(m, nextHe(h)), lC = L(m, nextHe(nextHe(h)));
        eps = jmax(eps, lC - lA - lB + delta);
    }
    if (eps > 0) for (double& l : m.length) l += eps;
    m.record.mollify = relativeFactor;
    m.record.epsilon = eps;
    return eps;
}

void addSignposts(Mesh& m) {
    const size_t H = 3 * static_cast<size_t>(m.nF);
    m.angle.assign(H, 0.0);
    m.angleSum.assign(static_cast<size_t>(m.nV), 0.0);
    m.ref.assign(static_cast<size_t>(m.nV), -1);
    std::vector<int> out(static_cast<size_t>(m.nV), -1);
    for (int h = 0; h < static_cast<int>(H); ++h) if (out[m.F[h]] == -1) out[m.F[h]] = h;
    // a BOUNDARY vertex's fan starts at its twinless outgoing halfedge and stops where the next has no twin
    for (int h = 0; h < static_cast<int>(H); ++h) if (m.twin[h] == -1) out[m.F[h]] = h;
    for (int v = 0; v < m.nV; ++v) {
        const int h0 = out[v];
        if (h0 < 0) continue;
        m.ref[v] = m.F[nextHe(h0)];
        int h = h0, guard = 0;
        double phi = 0;
        do {
            m.angle[h] = phi;
            phi += cornerAngle(m, h);
            h = m.twin[nextHe(nextHe(h))];                    // the next outgoing halfedge, counter-clockwise (−1: the boundary)
            if (++guard > (1 << 16)) throw Error(ErrorCode::GeometryDegenerate, "delaunay::addSignposts: the fan of vertex " + std::to_string(v) + " does not close");
        } while (h != h0 && h >= 0);
        m.angleSum[v] = phi;
    }
    m.hasSignposts = true;
}

namespace {

// THE LENGTH OF THE FLIPPED EDGE (Euclidean): the two triangles laid out in the plane across the edge a–b, the distance between their opposite
// vertices c and d — the angles at a add, and the law of cosines closes it.
double flippedLength(double lab, double lbc, double lca, double lad, double ldb) {
    auto ang = [](double x, double y, double opp) { return std::acos(clamp1((x * x + y * y - opp * opp) / (2 * x * y))); };
    const double theta = ang(lab, lca, lbc) + ang(lab, lad, ldb);       // at a: between ab and ac, and between ab and ad
    return std::sqrt(jmax(0.0, lca * lca + lad * lad - 2 * lca * lad * std::cos(theta)));
}

}  // namespace

bool flip(Mesh& m, int e) {
    const int h = m.edgeHe[static_cast<size_t>(e)], t = m.twin[static_cast<size_t>(h)];
    if (t < 0) return false;                                  // a BOUNDARY edge is never flipped
    const int h1 = nextHe(h), h2 = nextHe(h1), t1 = nextHe(t), t2 = nextHe(t1);
    if (h1 == t || t1 == h) return false;
    const int a = m.F[h], b = m.F[h1], c = m.F[h2], d = m.F[t2];
    const double len = flippedLength(m.length[static_cast<size_t>(e)], L(m, h1), L(m, h2), L(m, t1), L(m, t2));
    if (!std::isfinite(len) || len <= 0) return false;
    const int f1 = h - (h % 3), f2 = t - (t % 3);
    // the four outer halfedges before the rewrite: b→c (h1), c→a (h2), a→d (t1), d→b (t2); listed as the reference does: t2, h1, h2, t1
    const int oldH[4] = {t2, h1, h2, t1};
    int outerEdge[4], outerTwin[4];
    for (int i = 0; i < 4; ++i) { outerEdge[i] = m.edgeOf[oldH[i]]; outerTwin[i] = m.twin[oldH[i]]; }
    // THE FAR VERTICES MAY BE ONE (c = d): an outer twin inside the pair is REMAPPED to its new slot; one that is the flipped edge itself cannot be, and is refused.
    for (int i = 0; i < 4; ++i) if (outerTwin[i] == h || outerTwin[i] == t) return false;
    // the signposts of the four outer halfedges, read before they move
    double carried[4] = {0, 0, 0, 0};
    if (m.hasSignposts) for (int i = 0; i < 4; ++i) carried[i] = m.angle[oldH[i]];
    m.F[f1] = d; m.F[f1 + 1] = b; m.F[f1 + 2] = c;
    m.F[f2] = c; m.F[f2 + 1] = a; m.F[f2 + 2] = d;
    // f1: d→b (was t2), b→c (was h1), c→d (the new edge); f2: c→a (was h2), a→d (was t1), d→c (the new edge)
    const int slots[4] = {f1, f1 + 1, f2, f2 + 1};
    // where each old outer halfedge lands: d→b (t2) at f1, b→c (h1) at f1+1, c→a (h2) at f2, a→d (t1) at f2+1
    auto moved = [&](int x) -> int { for (int i = 0; i < 4; ++i) if (oldH[i] == x) return slots[i]; return -2; };
    for (int i = 0; i < 4; ++i) {
        const int s = slots[i], tw = outerTwin[i], mv = moved(tw);
        const int tw2 = (mv != -2) ? mv : tw;
        m.edgeOf[s] = outerEdge[i]; m.twin[s] = tw2; m.edgeHe[static_cast<size_t>(outerEdge[i])] = s;
        if (mv == -2 && tw >= 0) m.twin[tw] = s;              // a twin outside the pair points back; one inside is set in its own turn
    }
    m.edgeOf[f1 + 2] = e; m.edgeOf[f2 + 2] = e; m.twin[f1 + 2] = f2 + 2; m.twin[f2 + 2] = f1 + 2; m.edgeHe[static_cast<size_t>(e)] = f1 + 2;
    m.length[static_cast<size_t>(e)] = len;
    if (m.hasSignposts) {
        // the outer halfedges keep their angles in their new slots; the NEW edge's two halfedges are their counter-clockwise neighbour's angle
        // plus the corner between: c→d after c→a (face f2 = c, a, d), d→c after d→b (face f1 = d, b, c)
        m.angle[f1] = carried[0]; m.angle[f1 + 1] = carried[1]; m.angle[f2] = carried[2]; m.angle[f2 + 1] = carried[3];
        m.angle[f1 + 2] = std::fmod(m.angle[f2] + cornerAngle(m, f2), m.angleSum[c]);
        m.angle[f2 + 2] = std::fmod(m.angle[f1] + cornerAngle(m, f1), m.angleSum[d]);
    }
    return true;
}

int flipToDelaunay(Mesh& m, const DelaunayOptions& o) {
    const double eps = o.flipEps;
    const int nE = m.nE();
    std::deque<int> queue;
    for (int e = 0; e < nE; ++e) queue.push_back(e);
    std::vector<uint8_t> inQueue(static_cast<size_t>(nE), 1);
    if (m.mass.empty()) m.mass = mixedVoronoiMass(m);         // the mass is the surface's, taken before the flips
    int flips = 0;
    while (!queue.empty()) {
        const int e = queue.front(); queue.pop_front(); inQueue[static_cast<size_t>(e)] = 0;
        if (!(edgeWeight(m, e) < -eps)) continue;
        if (!flip(m, e)) continue;
        ++flips;
        const int h = m.edgeHe[static_cast<size_t>(e)], t = m.twin[static_cast<size_t>(h)];
        const int xs[4] = {nextHe(h), nextHe(nextHe(h)), nextHe(t), nextHe(nextHe(t))};
        for (int x : xs) { const int n = m.edgeOf[x]; if (!inQueue[static_cast<size_t>(n)]) { inQueue[static_cast<size_t>(n)] = 1; queue.push_back(n); } }
    }
    m.record.flipEps = eps;
    m.record.flips = flips;
    return flips;
}

std::vector<double> mixedVoronoiMass(const Mesh& m) {
    std::vector<double> M(static_cast<size_t>(m.nV), 0.0);
    for (int f = 0; f < m.nF; ++f) {
        const int h = 3 * f;
        const int v[3] = {m.F[h], m.F[h + 1], m.F[h + 2]};
        // l[k] is the edge from corner k to corner k+1; corner k's angle is opposite l[k+1]
        const double l[3] = {L(m, h), L(m, h + 1), L(m, h + 2)};
        const double A = triangleArea(l[0], l[1], l[2]);
        double cs[3];
        for (int k = 0; k < 3; ++k) { const double a = l[k], b = l[(k + 2) % 3], o = l[(k + 1) % 3]; cs[k] = (a * a + b * b - o * o) / (2 * a * b); }
        int obtuse = -1;
        for (int k = 0; k < 3; ++k) if (cs[k] < 0) { obtuse = k; break; }
        if (obtuse >= 0) { for (int k = 0; k < 3; ++k) M[static_cast<size_t>(v[k])] += (k == obtuse ? A / 2 : A / 4); continue; }
        double cot[3];
        for (int k = 0; k < 3; ++k) cot[k] = cs[k] / std::sqrt(jmax(1e-300, 1 - cs[k] * cs[k]));
        // corner k's Voronoi part: its two edges, each |e|² × the cot of the angle opposite it, over 8
        for (int k = 0; k < 3; ++k) {
            const int e2 = (k + 2) % 3;
            M[static_cast<size_t>(v[k])] += (l[k] * l[k] * cot[(k + 2) % 3] + l[e2] * l[e2] * cot[(e2 + 2) % 3]) / 8;
        }
    }
    return M;
}

std::vector<double> barycentricMass(const Mesh& m) {
    std::vector<double> M(static_cast<size_t>(m.nV), 0.0);
    for (int f = 0; f < m.nF; ++f) {
        const int h = 3 * f;
        const double A = triangleArea(L(m, h), L(m, h + 1), L(m, h + 2));
        for (int k = 0; k < 3; ++k) M[static_cast<size_t>(m.F[h + k])] += A / 3;
    }
    return M;
}

int nonDelaunayEdges(const Mesh& m, double eps) {
    int n = 0;
    for (int e = 0; e < m.nE(); ++e) if (edgeWeight(m, e) < -eps) ++n;
    return n;
}

Mesh prepare(const double* positions, int nV, const int32_t* faces, int nF, const DelaunayOptions& o, bool intrinsic) {
    Mesh m = makeMesh(positions, nV, faces, nF);
    if (intrinsic) {
        mollify(m, o);
        addSignposts(m);
        flipToDelaunay(m, o);
    } else {
        addSignposts(m);
        m.mass = mixedVoronoiMass(m);
        m.record.mollify = 0.0; m.record.epsilon = 0.0; m.record.flipEps = 0.0; m.record.flips = 0;
    }
    return m;
}

// ── operators ──

Eigen::SparseMatrix<double> cotanLaplacian(const Mesh& m) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(4 * static_cast<size_t>(m.nE()));
    for (int e = 0; e < m.nE(); ++e) {
        const int h = m.edgeHe[static_cast<size_t>(e)], i = m.F[h], j = m.F[nextHe(h)];
        if (i == j) continue;
        const double w = edgeWeight(m, e);
        t.emplace_back(i, j, -w); t.emplace_back(j, i, -w); t.emplace_back(i, i, w); t.emplace_back(j, j, w);
    }
    Eigen::SparseMatrix<double> Lm(m.nV, m.nV);
    Lm.setFromTriplets(t.begin(), t.end());
    Lm.makeCompressed();
    return Lm;
}

Eigen::VectorXd edgeCotanWeights(const Mesh& m) {
    Eigen::VectorXd w(m.nE());
    for (int e = 0; e < m.nE(); ++e) w(e) = edgeWeight(m, e);
    return w;
}

Eigen::VectorXd faceAreas(const Mesh& m) {
    Eigen::VectorXd a(m.nF);
    for (int f = 0; f < m.nF; ++f) a(f) = triangleArea(L(m, 3 * f), L(m, 3 * f + 1), L(m, 3 * f + 2));
    return a;
}

Eigen::SparseMatrix<double> galerkinMass(const Mesh& m) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(9 * static_cast<size_t>(m.nF));
    const Eigen::VectorXd A = faceAreas(m);
    for (int f = 0; f < m.nF; ++f)
        for (int r = 0; r < 3; ++r) {
            const int i = m.F[3 * f + r], j = m.F[3 * f + (r + 1) % 3], k = m.F[3 * f + (r + 2) % 3];
            t.emplace_back(i, i, A(f) / 6.0); t.emplace_back(i, j, A(f) / 12.0); t.emplace_back(i, k, A(f) / 12.0);
        }
    Eigen::SparseMatrix<double> M(m.nV, m.nV);
    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();
    return M;
}

static double coneTarget(const Mesh& m, int v) { return m.boundary[static_cast<size_t>(v)] ? m.angleSum[static_cast<size_t>(v)] : 2 * M_PI; }

double rescaledAngle(const Mesh& m, int h) {
    if (!m.hasSignposts) throw Error(ErrorCode::InvalidInput, "delaunay::rescaledAngle: the mesh has no signposts");
    const int v = m.F[static_cast<size_t>(h)];
    return (coneTarget(m, v) * m.angle[static_cast<size_t>(h)]) / m.angleSum[static_cast<size_t>(v)];
}

double reverseAngle(const Mesh& m, int h) {
    const int t = m.twin[static_cast<size_t>(h)];
    if (t >= 0) return rescaledAngle(m, t);
    const int n = nextHe(h), v = m.F[static_cast<size_t>(n)];
    return (coneTarget(m, v) * (m.angle[static_cast<size_t>(n)] + cornerAngle(m, n))) / m.angleSum[static_cast<size_t>(v)];
}

Eigen::SparseMatrix<std::complex<double>> connectionLaplacian(const Mesh& m, int nSym) {
    using C = std::complex<double>;
    std::vector<Eigen::Triplet<C>> t;
    t.reserve(4 * static_cast<size_t>(m.nE()));
    for (int e = 0; e < m.nE(); ++e) {
        const int h = m.edgeHe[static_cast<size_t>(e)], i = m.F[h], j = m.F[nextHe(h)];
        const double w = edgeWeight(m, e);
        // r_ji: a vector at j carried to i; r_ij = −r_ji, so the two off-diagonal entries are each other's conjugate
        const double r = nSym * (rescaledAngle(m, h) + M_PI - reverseAngle(m, h));
        t.emplace_back(i, i, C(w, 0)); t.emplace_back(j, j, C(w, 0));
        t.emplace_back(i, j, C(-w * std::cos(r), -w * std::sin(r)));
        t.emplace_back(j, i, C(-w * std::cos(r), w * std::sin(r)));
    }
    Eigen::SparseMatrix<C> K(m.nV, m.nV);
    K.setFromTriplets(t.begin(), t.end());
    K.makeCompressed();
    return K;
}

Dec dec(const Mesh& m) {
    Dec d;
    const int nE = m.nE();
    std::vector<Eigen::Triplet<double>> t0, t1;
    for (int e = 0; e < nE; ++e) {
        const int h = m.edgeHe[static_cast<size_t>(e)];
        t0.emplace_back(e, m.F[nextHe(h)], 1.0);               // the edge's orientation: tail → tip
        t0.emplace_back(e, m.F[h], -1.0);
    }
    for (int f = 0; f < m.nF; ++f)
        for (int k = 0; k < 3; ++k) {
            const int h = 3 * f + k, e = m.edgeOf[static_cast<size_t>(h)];
            t1.emplace_back(f, e, m.edgeHe[static_cast<size_t>(e)] == h ? 1.0 : -1.0);
        }
    d.d0.resize(nE, m.nV); d.d0.setFromTriplets(t0.begin(), t0.end());
    d.d1.resize(m.nF, nE); d.d1.setFromTriplets(t1.begin(), t1.end());
    d.d0.makeCompressed(); d.d1.makeCompressed();
    d.star1 = edgeCotanWeights(m);
    d.star2 = faceAreas(m).cwiseInverse();
    return d;
}

}  // namespace nxr::manifold::delaunay
