// nxr::ladder — see nxr/ladder.h. A line port of the Cortical Flow Studio's engine/ladder.ts; compiled with -ffp-contract=off (JavaScript has no fused multiply-add).
#include "nxr/select/dyadic.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

#include "nxr/core/errors.h"

namespace nxr::select {

using Eigen::VectorXd;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ── tiles ────────────────────────────────────────────────────────────────────────────────

std::vector<int> tileCentres(const double* positions, const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const std::vector<char>& singular) {
    const size_t K = static_cast<size_t>(nodes);
    std::vector<double> w(K, 0.0), c(3 * K, 0.0);
    auto ok = [&](size_t v) { return node[v] >= 0 && (singular.empty() || !singular[v]); };
    for (size_t v = 0; v < node.size(); ++v) if (ok(v)) { const size_t t = static_cast<size_t>(node[v]); w[t] += area[v]; for (int d = 0; d < 3; ++d) c[3 * t + static_cast<size_t>(d)] += area[v] * positions[3 * v + static_cast<size_t>(d)]; }
    std::vector<int> best(K, -1);
    std::vector<double> bd(K, kInf);
    for (size_t v = 0; v < node.size(); ++v) {
        if (!ok(v)) continue;
        const size_t t = static_cast<size_t>(node[v]);
        double s2 = 0;
        for (int d = 0; d < 3; ++d) { const double x = positions[3 * v + static_cast<size_t>(d)] - c[3 * t + static_cast<size_t>(d)] / w[t]; s2 += x * x; }
        if (s2 < bd[t]) { bd[t] = s2; best[t] = static_cast<int>(v); }
    }
    return best;
}

std::vector<std::pair<int, int>> tileAdjacency(const int32_t* faces, int nF, const std::vector<int32_t>& node) {
    std::set<std::pair<int, int>> s;
    for (int f = 0; f < nF; ++f)
        for (int k = 0; k < 3; ++k) {
            const int a = node[static_cast<size_t>(faces[3 * f + k])], b = node[static_cast<size_t>(faces[3 * f + (k + 1) % 3])];
            if (a >= 0 && b >= 0 && a != b) s.insert({std::min(a, b), std::max(a, b)});
        }
    return {s.begin(), s.end()};
}

LeviCivita leviCivita(chart::ChartKit& kit, const geometry::Mat3& e1, const geometry::Mat3& e2, const VectorXd& gauge, const std::vector<int32_t>& node, int nodes, const std::vector<int>& centres) {
    (void)e1; (void)e2;
    const int V = static_cast<int>(node.size());
    LeviCivita lc;
    lc.centres = centres;
    lc.rho = VectorXd::Constant(V, kNaN);
    lc.omega = Eigen::MatrixXd::Constant(nodes, nodes, kNaN);
    std::vector<VectorXd> ang(static_cast<size_t>(nodes));
    for (int k = 0; k < nodes; ++k) {
        const int c = centres[static_cast<size_t>(k)];
        if (c < 0 || !std::isfinite(gauge[c])) continue;
        // e1 of the centre, carried everywhere: the unit vector headed at the centre's gauge angle of north
        const VectorXd T = kit.transported(c, gauge[c]);
        VectorXd a(V);
        for (int i = 0; i < V; ++i) a[i] = std::atan2(T[2 * i + 1], T[2 * i]) - gauge[i];      // its angle from the vertex's own e1, towards e2 (= n x e1)
        const double at0 = a[c];
        for (int i = 0; i < V; ++i) a[i] = std::remainder(a[i] - at0, 2 * M_PI);              // the identity at its own source (nxr: the heat solution is smoothed there)
        ang[static_cast<size_t>(k)] = a;
        for (int i = 0; i < V; ++i) if (node[static_cast<size_t>(i)] == k) lc.rho[i] = a[i];
    }
    for (int a = 0; a < nodes; ++a) {
        if (ang[static_cast<size_t>(a)].size() == 0) continue;
        for (int b = 0; b < nodes; ++b) {
            const int cb = centres[static_cast<size_t>(b)];
            if (cb >= 0) lc.omega(a, b) = ang[static_cast<size_t>(a)][cb];
        }
    }
    return lc;
}

Eigen::MatrixXd antisymmetric(const Eigen::MatrixXd& omega) {
    Eigen::MatrixXd out(omega.rows(), omega.cols());
    for (Eigen::Index a = 0; a < omega.rows(); ++a)
        for (Eigen::Index b = 0; b < omega.cols(); ++b) {
            const double ab = omega(a, b), ba = omega(b, a);
            out(a, b) = std::isfinite(ab) && std::isfinite(ba) ? std::remainder(ab - std::remainder(ab + ba, 2 * M_PI) / 2, 2 * M_PI) : kNaN;
        }
    return out;
}

Eigen::VectorXcd carriedToCentre(const geometry::Mat3& e1, const geometry::Mat3& e2, const Eigen::VectorXd& rho, const geometry::Mat3& v, std::vector<char>* finite) {
    const Eigen::Index n = rho.size();
    Eigen::VectorXcd out(n);
    if (finite) finite->assign(static_cast<size_t>(n), 0);
    for (Eigen::Index r = 0; r < n; ++r) {
        const double c1 = v.row(r).dot(e1.row(r)), c2 = v.row(r).dot(e2.row(r));
        const bool ok = std::isfinite(rho[r]) && std::isfinite(c1) && std::isfinite(c2);
        out[r] = ok ? std::complex<double>(c1, c2) * std::exp(std::complex<double>(0, -rho[r])) : std::complex<double>(kNaN, kNaN);
        if (finite) (*finite)[static_cast<size_t>(r)] = ok ? 1 : 0;
    }
    return out;
}

}  // namespace nxr::select
