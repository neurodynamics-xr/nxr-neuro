// nxr::ladder — see nxr/ladder.h. A line port of the Cortical Flow Studio's engine/ladder.ts; compiled with -ffp-contract=off (JavaScript has no fused multiply-add).
#include "nxr/ladder.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

#include "nxr/errors.h"

namespace nxr::ladder {

using Eigen::VectorXd;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ── tiles ────────────────────────────────────────────────────────────────────────────────

std::vector<double> tileAreas(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area) {
    std::vector<double> w(static_cast<size_t>(nodes), 0.0);
    for (size_t v = 0; v < node.size(); ++v) if (node[v] >= 0) w[static_cast<size_t>(node[v])] += area[v];
    return w;
}

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

// ── scalar summaries ─────────────────────────────────────────────────────────────────────

VectorXd ScalarSummary::mean() const {
    VectorXd m(w.size());
    for (Eigen::Index i = 0; i < w.size(); ++i) m[i] = w[i] > 0 ? s1[i] / w[i] : kNaN;
    return m;
}

ScalarSummary scalarSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const double* x) {
    ScalarSummary s;
    s.w = s.s1 = s.s2 = s.n = VectorXd::Zero(nodes);
    s.min = VectorXd::Constant(nodes, kInf);
    s.max = VectorXd::Constant(nodes, -kInf);
    for (size_t v = 0; v < node.size(); ++v) {
        if (node[v] < 0 || !std::isfinite(x[v])) continue;
        const Eigen::Index t = node[v];
        s.w[t] += area[v]; s.s1[t] += area[v] * x[v]; s.s2[t] += area[v] * x[v] * x[v]; s.n[t] += 1;
        s.min[t] = std::min(s.min[t], x[v]); s.max[t] = std::max(s.max[t], x[v]);
    }
    return s;
}

// ── vector summaries ─────────────────────────────────────────────────────────────────────

FrameSummary frameSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const Mat3& e1, const Mat3& e2, const Mat3& nrm, const Mat3& v) {
    FrameSummary s;
    s.w = s.c1 = s.c2 = s.cn = s.c11 = s.c22 = s.c12 = s.mag = VectorXd::Zero(nodes);
    for (size_t i = 0; i < node.size(); ++i) {
        if (node[i] < 0) continue;
        const Eigen::Index r = static_cast<Eigen::Index>(i);
        const double c1 = v.row(r).dot(e1.row(r)), c2 = v.row(r).dot(e2.row(r)), cn = v.row(r).dot(nrm.row(r));
        if (!std::isfinite(c1) || !std::isfinite(c2)) continue;                    // singular vertices out
        const Eigen::Index t = node[i];
        const double a = area[i], cnz = std::isfinite(cn) ? cn : 0.0;
        s.w[t] += a; s.c1[t] += a * c1; s.c2[t] += a * c2; s.cn[t] += a * cnz;
        s.c11[t] += a * c1 * c1; s.c22[t] += a * c2 * c2; s.c12[t] += a * c1 * c2; s.mag[t] += a * std::hypot(c1, c2);
    }
    return s;
}

LeviCivita leviCivita(chart::ChartKit& kit, const Mat3& e1, const Mat3& e2, const VectorXd& gauge, const std::vector<int32_t>& node, int nodes, const std::vector<int>& centres) {
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

Eigen::VectorXcd lcResultant(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const Mat3& e1, const Mat3& e2, const VectorXd& rho, const Mat3& v, VectorXd* weight, VectorXd* magnitude) {
    Eigen::VectorXcd out = Eigen::VectorXcd::Zero(nodes);
    VectorXd w = VectorXd::Zero(nodes), mg = VectorXd::Zero(nodes);
    for (size_t i = 0; i < node.size(); ++i) {
        if (node[i] < 0) continue;
        const Eigen::Index r = static_cast<Eigen::Index>(i);
        const double c1 = v.row(r).dot(e1.row(r)), c2 = v.row(r).dot(e2.row(r));
        if (!std::isfinite(rho[r]) || !std::isfinite(c1) || !std::isfinite(c2)) continue;
        const Eigen::Index t = node[i];
        out[t] += area[i] * std::complex<double>(c1, c2) * std::exp(std::complex<double>(0, -rho[r]));
        w[t] += area[i]; mg[t] += area[i] * std::hypot(c1, c2);
    }
    if (weight) *weight = w;
    if (magnitude) *magnitude = mg;
    return out;
}

VectorSummary vectorSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const Mat3& e1, const Mat3& e2, const Mat3& normal, const VectorXd& rho, const Mat3& v) {
    VectorSummary s;
    s.frame = frameSummary(node, nodes, area, e1, e2, normal, v);
    VectorXd mg;
    s.transported = lcResultant(node, nodes, area, e1, e2, rho, v, &s.weight, &mg);
    s.resultantLength.resize(nodes);
    for (int k = 0; k < nodes; ++k) s.resultantLength[k] = mg[k] > 0 ? std::abs(s.transported[k]) / mg[k] : kNaN;
    return s;
}


}  // namespace nxr::ladder
