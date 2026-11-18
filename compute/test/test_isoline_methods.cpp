// The two isoline methods (nxr/signal/surface_calculus.h, IsolineMethod) on a cortex-sized synthetic mesh (an icosphere, 163842 vertices): the same
// segments (each segment's endpoints, as a set, to rounding), and the time of each (printed: the faster is the default).
#include "nxr/geometry/surface_arrays.h"
#include "nxr/signal/surface_calculus.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(c, msg) do { if (c) std::cout << "    ok   " << msg << "\n"; else { std::cout << "    FAIL " << msg << "\n"; ++failures; } } while (0)

#include "synthetic_meshes.h"

using Seg = std::array<double, 6>;
static std::vector<Seg> segments(const nxr::signal::IsolineResult& r) {
    std::vector<Seg> s(static_cast<size_t>(r.segmentCount));
    for (int i = 0; i < r.segmentCount; ++i) {
        // endpoints ordered by their coordinates rounded to 1e-9 (a crossing shared by two segments then sorts the same way in both results)
        auto q = [](double x) { return std::round(x * 1e9); };
        std::array<double, 3> a{r.positions(2 * i, 0), r.positions(2 * i, 1), r.positions(2 * i, 2)}, b{r.positions(2 * i + 1, 0), r.positions(2 * i + 1, 1), r.positions(2 * i + 1, 2)};
        if (std::array<double, 3>{q(b[0]), q(b[1]), q(b[2])} < std::array<double, 3>{q(a[0]), q(a[1]), q(a[2])}) std::swap(a, b);
        s[static_cast<size_t>(i)] = {a[0], a[1], a[2], b[0], b[1], b[2]};
    }
    std::sort(s.begin(), s.end(), [](const Seg& x, const Seg& y) {
        for (int k = 0; k < 6; ++k) {
            const double a = std::round(x[static_cast<size_t>(k)] * 1e9), b = std::round(y[static_cast<size_t>(k)] * 1e9);
            if (a != b) return a < b;
        }
        return false;
    });
    return s;
}

int main() {
    const Mesh m = icosphere(7);
    std::cout << "mesh: " << m.nV() << " vertices, " << m.nF() << " faces\n";
    const nxr::geometry::SurfaceArrays s = nxr::geometry::surfaceArrays(m.V.data(), m.nV(), m.F.data(), m.nF(), nxr::geometry::intrinsicOptions(false));
    std::mt19937 g(3);
    Eigen::VectorXd f(m.nV());
    for (int v = 0; v < m.nV(); ++v) f(v) = m.V[3 * v + 2] + 0.3 * std::sin(7 * m.V[3 * v]) + 1e-3 * gauss(g);
    const int levels = 20;
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    const auto a = nxr::signal::isoline(s, f, levels, 0.0, 0.0, nxr::signal::IsolineMethod::PerTriangle);
    auto t1 = clock::now();
    const auto b = nxr::signal::isoline(s, f, levels, 0.0, 0.0, nxr::signal::IsolineMethod::MarchingTriangles);
    auto t2 = clock::now();
    const double ta = std::chrono::duration<double>(t1 - t0).count(), tb = std::chrono::duration<double>(t2 - t1).count();
    std::printf("  per-triangle %.3f s, marching triangles %.3f s (each includes rebuilding the mesh from its arrays), %d segments\n", ta, tb, a.segmentCount);
    CHECK(a.segmentCount == b.segmentCount && a.segmentCount > 1000, "the same number of segments");
    const auto sa = segments(a), sb = segments(b);
    double worst = 0.0;
    for (size_t i = 0; i < std::min(sa.size(), sb.size()); ++i)
        for (int k = 0; k < 6; ++k) worst = std::max(worst, std::abs(sa[i][static_cast<size_t>(k)] - sb[i][static_cast<size_t>(k)]));
    std::printf("  worst endpoint difference %.3g\n", worst);
    CHECK(worst < 1e-12, "the same segments (endpoints to rounding)");
    std::cout << (failures ? "FAILED " : "PASSED ") << failures << "\n";
    return failures ? 1 : 0;
}
