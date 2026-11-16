// nxr::phase — the phase of a complex field, pointwise (phase.h): nxr's core_phase formulas.
#include "nxr/phase.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::phase {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

void check(const Eigen::MatrixXcd& Z, int c, Eigen::Index V) {
    if (c < 1) throw std::invalid_argument("phase: at least one component a vertex");
    if (Z.rows() != V * c) throw std::invalid_argument("phase: the field is not [vertices x components] rows");
}

// component a of Z as [V x m]
Eigen::MatrixXcd componentOf(const Eigen::MatrixXcd& Z, int c, int a) {
    const Eigen::Index V = Z.rows() / c;
    Eigen::MatrixXcd z(V, Z.cols());
    for (Eigen::Index v = 0; v < V; ++v) z.row(v) = Z.row(v * c + a);
    return z;
}

// the tangent gradient of one component: three [V x m] complex blocks, each vertex's vector projected on its tangent plane
std::array<Eigen::MatrixXcd, 3> tangentGradient(const Eigen::MatrixXcd& z, const Gradient& G, const Eigen::MatrixXd& N) {
    std::array<Eigen::MatrixXcd, 3> g;
    for (int d = 0; d < 3; ++d) g[static_cast<std::size_t>(d)] = G[static_cast<std::size_t>(d)].cast<std::complex<double>>() * z;
    for (Eigen::Index v = 0; v < z.rows(); ++v) {
        const Eigen::RowVectorXcd dot = g[0].row(v) * N(v, 0) + g[1].row(v) * N(v, 1) + g[2].row(v) * N(v, 2);
        for (int d = 0; d < 3; ++d) g[static_cast<std::size_t>(d)].row(v) -= dot * N(v, d);
    }
    return g;
}
}  // namespace

Gradient vertexGradient(const Eigen::MatrixXd& P, const Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>& F) {
    // per face: ∇φ_i = (N × e_i) / 2A, e_i the edge opposite vertex i, N the unit face normal (nxr gradient_and_kernel); then the area-weighted mean of the
    // incident faces' gradients at each vertex
    const Eigen::Index V = P.rows(), nF = F.rows();
    std::vector<Eigen::Triplet<double>> t[3];
    Eigen::VectorXd areaSum = Eigen::VectorXd::Zero(V);
    std::vector<double> area(static_cast<std::size_t>(nF));
    for (Eigen::Index f = 0; f < nF; ++f) {
        const Eigen::Vector3d p0 = P.row(F(f, 0)), p1 = P.row(F(f, 1)), p2 = P.row(F(f, 2));
        const Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
        const double a = n.norm() / 2;
        area[static_cast<std::size_t>(f)] = a;
        for (int s = 0; s < 3; ++s) areaSum(F(f, s)) += a;
    }
    for (Eigen::Index f = 0; f < nF; ++f) {
        const double a = area[static_cast<std::size_t>(f)];
        if (!(a > 0)) continue;
        const Eigen::Vector3d p[3] = {P.row(F(f, 0)), P.row(F(f, 1)), P.row(F(f, 2))};
        const Eigen::Vector3d N = (p[1] - p[0]).cross(p[2] - p[0]) / (2 * a);
        const int opp[3][2] = {{1, 2}, {2, 0}, {0, 1}};                                   // e_i = p_b − p_a, the edge opposite vertex i
        for (int i = 0; i < 3; ++i) {
            const Eigen::Vector3d gi = N.cross(p[opp[i][1]] - p[opp[i][0]]) / (2 * a);
            for (int s = 0; s < 3; ++s) {                                               // to every vertex of the face, weighted by the face's area
                const Eigen::Index v = F(f, s);
                for (int d = 0; d < 3; ++d) t[d].emplace_back(v, F(f, i), a * gi(d) / areaSum(v));
            }
        }
    }
    Gradient G;
    for (int d = 0; d < 3; ++d) { G[static_cast<std::size_t>(d)].resize(V, V); G[static_cast<std::size_t>(d)].setFromTriplets(t[d].begin(), t[d].end()); }
    return G;
}

Eigen::MatrixXd power(const Eigen::MatrixXcd& Z, int c) {
    if (c < 1 || Z.rows() % c != 0) throw std::invalid_argument("power: the field is not [vertices x components] rows");
    const Eigen::Index V = Z.rows() / c;
    Eigen::MatrixXd p = Eigen::MatrixXd::Zero(V, Z.cols());
    for (Eigen::Index v = 0; v < V; ++v)
        for (int a = 0; a < c; ++a) p.row(v) += Z.row(v * c + a).cwiseAbs2();
    return p;
}

Eigen::MatrixXd phaseCurrent(const Eigen::MatrixXcd& Z, int c, const Gradient& G, const Eigen::MatrixXd& N) {
    const Eigen::Index V = G[0].rows();
    check(Z, c, V);
    Eigen::MatrixXd j = Eigen::MatrixXd::Zero(3 * V, Z.cols());
    for (int a = 0; a < c; ++a) {
        const Eigen::MatrixXcd z = componentOf(Z, c, a);
        const auto g = tangentGradient(z, G, N);
        for (int d = 0; d < 3; ++d) {
            const Eigen::MatrixXd im = (z.conjugate().array() * g[static_cast<std::size_t>(d)].array()).imag();
            for (Eigen::Index v = 0; v < V; ++v) j.row(3 * v + d) += im.row(v);
        }
    }
    return j;
}

Eigen::MatrixXd gradientPower(const Eigen::MatrixXcd& Z, int c, const Gradient& G, const Eigen::MatrixXd& N) {
    const Eigen::Index V = G[0].rows();
    check(Z, c, V);
    Eigen::MatrixXd q = Eigen::MatrixXd::Zero(V, Z.cols());
    for (int a = 0; a < c; ++a) {
        const auto g = tangentGradient(componentOf(Z, c, a), G, N);
        for (int d = 0; d < 3; ++d) q += g[static_cast<std::size_t>(d)].cwiseAbs2();
    }
    return q;
}

Eigen::MatrixXd phaseGradient(const Eigen::MatrixXd& j, const Eigen::MatrixXd& p) {
    if (j.rows() != 3 * p.rows() || j.cols() != p.cols()) throw std::invalid_argument("phaseGradient: j is not [3V x m] for p [V x m]");
    Eigen::MatrixXd g(j.rows(), j.cols());
    for (Eigen::Index v = 0; v < p.rows(); ++v)
        for (Eigen::Index k = 0; k < p.cols(); ++k)
            for (int d = 0; d < 3; ++d) g(3 * v + d, k) = p(v, k) > 0 ? j(3 * v + d, k) / p(v, k) : kNaN;
    return g;
}

Velocity phaseVelocity(const Eigen::MatrixXd& g, double omega) {
    if (g.rows() % 3 != 0) throw std::invalid_argument("phaseVelocity: the gradient is not [3V x m]");
    const Eigen::Index V = g.rows() / 3;
    Velocity r{Eigen::MatrixXd(g.rows(), g.cols()), Eigen::MatrixXd(V, g.cols())};
    for (Eigen::Index v = 0; v < V; ++v)
        for (Eigen::Index k = 0; k < g.cols(); ++k) {
            const double g2 = g(3 * v, k) * g(3 * v, k) + g(3 * v + 1, k) * g(3 * v + 1, k) + g(3 * v + 2, k) * g(3 * v + 2, k);
            for (int d = 0; d < 3; ++d) r.velocity(3 * v + d, k) = g2 > 0 ? -omega * g(3 * v + d, k) / g2 : kNaN;
            r.speed(v, k) = g2 > 0 ? omega / std::sqrt(g2) : kNaN;
        }
    return r;
}

Eigen::MatrixXd travellingIndex(const Eigen::MatrixXd& j, const Eigen::MatrixXd& p, const Eigen::MatrixXd& q) {
    if (j.rows() != 3 * p.rows() || p.rows() != q.rows() || j.cols() != p.cols() || p.cols() != q.cols())
        throw std::invalid_argument("travellingIndex: j [3V x m], p and q [V x m]");
    Eigen::MatrixXd t(p.rows(), p.cols());
    for (Eigen::Index v = 0; v < p.rows(); ++v)
        for (Eigen::Index k = 0; k < p.cols(); ++k) {
            const double pq = p(v, k) * q(v, k);
            const double jn = std::sqrt(j(3 * v, k) * j(3 * v, k) + j(3 * v + 1, k) * j(3 * v + 1, k) + j(3 * v + 2, k) * j(3 * v + 2, k));
            t(v, k) = pq > 0 ? jn / std::sqrt(pq) : kNaN;
        }
    return t;
}

}  // namespace nxr::phase
