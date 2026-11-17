// Synthetic meshes of the geometry-central switch's tests (docs/specs/2026-11-17-geometry-central-switch.md): spheres, a UV sphere, tori, a planar disc.
// Random numbers come from mt19937's raw output only (the standard's distributions differ between libraries; the meshes must not).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

struct Mesh { std::vector<double> V; std::vector<int32_t> F; int nV() const { return (int)V.size() / 3; } int nF() const { return (int)F.size() / 3; } };

// ── meshes ── (random numbers come from mt19937's raw output only: the standard's distributions differ between libraries, the meshes must not)
inline double u01(std::mt19937& g) { return (g() + 0.5) / 4294967296.0; }
inline double gauss(std::mt19937& g) { return std::sqrt(-2 * std::log(u01(g))) * std::cos(2 * M_PI * u01(g)); }
inline Mesh icosphere(int lev) {
    const double t = (1 + std::sqrt(5.0)) / 2;
    Mesh m;
    const double p[12][3] = {{-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0},{0,-1,t},{0,1,t},{0,-1,-t},{0,1,-t},{t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1}};
    for (auto& q : p) { const double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]); for (int c = 0; c < 3; ++c) m.V.push_back(q[c] / n); }
    m.F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1};
    for (int l = 0; l < lev; ++l) {
        std::map<std::pair<int,int>, int> cache;
        auto mid = [&](int a, int b) {
            if (a > b) std::swap(a, b);
            auto it = cache.find({a, b}); if (it != cache.end()) return it->second;
            const int id = m.nV(); double x[3], n = 0;
            for (int c = 0; c < 3; ++c) { x[c] = (m.V[3*a+c] + m.V[3*b+c]) / 2; n += x[c]*x[c]; }
            for (int c = 0; c < 3; ++c) m.V.push_back(x[c] / std::sqrt(n));
            cache[{a, b}] = id; return id;
        };
        std::vector<int32_t> G;
        for (size_t f = 0; f < m.F.size(); f += 3) {
            const int a = m.F[f], b = m.F[f+1], c = m.F[f+2], ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            for (int v : {a, ab, ca,  b, bc, ab,  c, ca, bc,  ab, bc, ca}) G.push_back(v);
        }
        m.F = G;
    }
    return m;
}
inline double meanEdge(const Mesh& m) {
    double s = 0; int n = 0;
    for (int f = 0; f < m.nF(); ++f) for (int k = 0; k < 3; ++k) {
        const int a = m.F[3*f+k], b = m.F[3*f+(k+1)%3]; double d = 0;
        for (int c = 0; c < 3; ++c) d += std::pow(m.V[3*a+c] - m.V[3*b+c], 2);
        s += std::sqrt(d); ++n;
    }
    return s / n;
}
// tangential jitter of a sphere (many edges become non-Delaunay), the points stay on the sphere
inline Mesh jitterSphere(Mesh m, double frac, unsigned seed) {
    std::mt19937 g(seed);
    const double h = meanEdge(m);
    for (int v = 0; v < m.nV(); ++v) {
        double* x = &m.V[3*v]; double d[3] = {gauss(g), gauss(g), gauss(g)}; const double dn = d[0]*x[0] + d[1]*x[1] + d[2]*x[2];
        double n = 0; for (int c = 0; c < 3; ++c) { x[c] += frac * h * (d[c] - dn * x[c]); n += x[c]*x[c]; }
        for (int c = 0; c < 3; ++c) x[c] /= std::sqrt(n);
    }
    return m;
}
inline Mesh uvSphere(int nLat, int nLon) {   // with poles: fans of slivers
    Mesh m; m.V = {0,0,1};
    for (int i = 1; i < nLat; ++i) for (int j = 0; j < nLon; ++j) { const double th = M_PI * i / nLat, ph = 2 * M_PI * j / nLon; m.V.push_back(std::sin(th)*std::cos(ph)); m.V.push_back(std::sin(th)*std::sin(ph)); m.V.push_back(std::cos(th)); }
    m.V.insert(m.V.end(), {0,0,-1});
    const int S = m.nV() - 1;
    auto id = [&](int i, int j) { return 1 + (i - 1) * nLon + (j % nLon); };
    for (int j = 0; j < nLon; ++j) m.F.insert(m.F.end(), {0, id(1, j), id(1, j + 1)});
    for (int i = 1; i < nLat - 1; ++i) for (int j = 0; j < nLon; ++j) {
        m.F.insert(m.F.end(), {id(i, j), id(i + 1, j), id(i + 1, j + 1)});
        m.F.insert(m.F.end(), {id(i, j), id(i + 1, j + 1), id(i, j + 1)});
    }
    for (int j = 0; j < nLon; ++j) m.F.insert(m.F.end(), {S, id(nLat - 1, j + 1), id(nLat - 1, j)});
    return m;
}
// a torus of nu × nv quads (two triangles each); `skip` removes quads (for the double torus)
inline Mesh torus(int nu, int nv, double R, double r, const std::vector<std::pair<int,int>>& skip = {}, double dx = 0) {
    Mesh m;
    for (int i = 0; i < nu; ++i) for (int j = 0; j < nv; ++j) {
        const double u = 2 * M_PI * i / nu, w = 2 * M_PI * j / nv;
        m.V.insert(m.V.end(), {dx + (R + r * std::cos(w)) * std::cos(u), (R + r * std::cos(w)) * std::sin(u), r * std::sin(w)});
    }
    auto id = [&](int i, int j) { return (i % nu) * nv + (j % nv); };
    for (int i = 0; i < nu; ++i) for (int j = 0; j < nv; ++j) {
        if (std::find(skip.begin(), skip.end(), std::make_pair(i, j)) != skip.end()) continue;
        m.F.insert(m.F.end(), {id(i, j), id(i + 1, j), id(i + 1, j + 1)});
        m.F.insert(m.F.end(), {id(i, j), id(i + 1, j + 1), id(i, j + 1)});
    }
    return m;
}
// two tori joined by a strip over one removed quad each: genus 2
inline Mesh doubleTorus() {
    const int nu = 12, nv = 8;
    Mesh A = torus(nu, nv, 3, 1, {{0, 0}}, 0), B = torus(nu, nv, 3, 1, {{nu / 2, 0}}, 9);
    Mesh m = A; const int off = A.nV();
    m.V.insert(m.V.end(), B.V.begin(), B.V.end());
    for (int32_t f : B.F) m.F.push_back(f + off);
    auto id = [&](int i, int j, int o) { return o + (i % nu) * nv + (j % nv); };
    // hole of A: quad (0,0): the loop a0..a3 follows the boundary edges (no twin); found by the generic loop search below
    auto loop = [&](int o, int qi) {
        std::vector<int> L = {id(qi, 0, o), id(qi + 1, 0, o), id(qi + 1, 1, o), id(qi, 1, o)};   // the removed quad's corners, ccw in (u, w)
        return L;
    };
    std::vector<int> a = loop(0, 0), b = loop(off, nu / 2);
    // the remaining faces run a_{i+1} -> a_i around the hole, so the bridge needs a_i -> a_{i+1}; and b_j -> b_{j+1}, paired with j = -i
    for (int i = 0; i < 4; ++i) {
        const int j = ((-i) % 4 + 4) % 4;
        const int a0 = a[i], a1 = a[(i + 1) % 4], b0 = b[j], b1 = b[(j + 1) % 4];
        // the cycle a0 -> a1 -> b0 -> b1 as two triangles
        m.F.insert(m.F.end(), {a0, a1, b0});
        m.F.insert(m.F.end(), {a0, b0, b1});
    }
    return m;
}
// a jittered planar disc (rings of points; interior jittered)
inline Mesh disc(int rings, double jitter, unsigned seed) {
    Mesh m; m.V = {0,0,0};
    std::mt19937 g(seed);
    std::vector<int> prev = {0};
    for (int r = 1; r <= rings; ++r) {
        const int n = 6 * r;
        for (int k = 0; k < n; ++k) {
            double x = r * std::cos(2 * M_PI * k / n), y = r * std::sin(2 * M_PI * k / n);
            if (r < rings) { x += jitter * (2 * u01(g) - 1); y += jitter * (2 * u01(g) - 1); }
            m.V.insert(m.V.end(), {x, y, 0});
        }
    }
    auto ring = [&](int r, int k) { return r == 0 ? 0 : 1 + 3 * r * (r - 1) + ((k % (6 * r)) + 6 * r) % (6 * r); };
    for (int r = 0; r < rings; ++r) {
        const int n0 = std::max(1, 6 * r), n1 = 6 * (r + 1);
        // walk both rings by angle
        int i = 0, j = 0;
        if (r == 0) { for (j = 0; j < n1; ++j) m.F.insert(m.F.end(), {0, ring(1, j), ring(1, j + 1)}); continue; }
        while (i < n0 || j < n1) {
            const double ai = (i + 1.0) / n0, aj = (j + 1.0) / n1;
            if ((i < n0 && ai <= aj) || j >= n1) { m.F.insert(m.F.end(), {ring(r, i), ring(r + 1, j), ring(r, i + 1)}); ++i; }
            else { m.F.insert(m.F.end(), {ring(r, i), ring(r + 1, j), ring(r + 1, j + 1)}); ++j; }
        }
    }
    return m;
}


// UV spheres of few latitudes and many longitudes: needle and obtuse triangles on the unit sphere (the analytic spectrum still applies), the badly shaped
// meshes the intrinsic Delaunay triangulation is for.
inline Mesh needleSphere(int nLat, int nLon) { return uvSphere(nLat, nLon); }
