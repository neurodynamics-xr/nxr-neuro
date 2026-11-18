// nxr::measurement: the sums of squares behind a quadrature fraction (congruence norms), the quadrature energy of rows, and their grouped and pairwise forms.
#include "nxr/measurement/congruence_norms.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;

namespace nxr::measurement {

VectorXd quadratureEnergy(const MatrixXcd& Fr, const MatrixXcd& F) {
    if (Fr.cols() != F.cols()) throw std::invalid_argument("quadratureEnergy: Fr and F need the same number of columns");
    const MatrixXcd G = F.adjoint() * F;                         // sum_j conj(F_ja) F_jb
    const MatrixXcd G2 = F.adjoint() * F.conjugate();            // sum_j conj(F_ja) conj(F_jb)
    // sum_j |C_ij|^2 = f_i G f_i^H ;  sum_j C_ij^2 = f_i G2 f_i^T ;  (Im c)^2 = (|c|^2 - Re c^2) / 2
    const MatrixXcd A = Fr * G, B = Fr * G2;
    VectorXd e(Fr.rows());
    for (Eigen::Index i = 0; i < Fr.rows(); ++i) {
        const double n2 = (A.row(i).array() * Fr.row(i).conjugate().array()).sum().real();
        const double c2 = (B.row(i).array() * Fr.row(i).array()).sum().real();
        e[i] = std::max(0.0, (n2 - c2) / 2);
    }
    return e;
}

}  // namespace nxr::measurement

namespace nxr::measurement {

namespace {
// (‖T‖², ‖Im T‖²) of T = B C Bᵀ from the Gram G = BᵀB: tr(X G Xᵀ G) = Σ_ij (X G)_ij (G X)_ji... with X symmetric (Cr) or antisymmetric (Cq) after the symmetrisation, one product each
CongruenceNorms normsFromGram(const MatrixXd& G, const MatrixXcd& C) {
    const MatrixXcd H = 0.5 * (C + C.adjoint());                    // Hermitian: Re symmetric, Im antisymmetric
    const MatrixXd Cr = H.real(), Cq = H.imag();
    const MatrixXd P = Cr * G, Q = Cq * G;
    double re = 0, im = 0;
    for (Eigen::Index i = 0; i < P.rows(); ++i)
        for (Eigen::Index j = 0; j < P.cols(); ++j) { re += P(i, j) * P(j, i); im -= Q(i, j) * Q(j, i); }
    return {re + im, im};
}
}  // namespace

CongruenceNorms congruenceNorms(const MatrixXd& B, const MatrixXcd& C) {
    if (C.rows() != C.cols() || C.rows() != B.cols()) throw std::invalid_argument("congruenceNorms: C must be square with as many rows as B has columns");
    return normsFromGram(B.transpose() * B, C);
}

}  // namespace nxr::measurement

namespace nxr::measurement {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;

namespace {
struct GroupGrams {
    std::vector<std::vector<Eigen::Index>> rows;
    std::vector<MatrixXd> G;          // [groups] Bᵀ W B (m x m), empty for a group with no row
};
GroupGrams groupGrams(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<MatrixXcd>& Cs, const VectorXd& w, const char* who) {
    const std::string name = who;
    if (static_cast<Eigen::Index>(rowGroup.size()) != A.rows()) throw std::invalid_argument(name + ": one group per row of A");
    if (groups < 0) throw std::invalid_argument(name + ": a negative number of groups");
    if (w.size() != 0 && w.size() != A.rows()) throw std::invalid_argument(name + ": one weight per row of A");
    for (const MatrixXcd& C : Cs)
        if (C.size() != 0 && (C.rows() != C.cols() || C.rows() != A.cols())) throw std::invalid_argument(name + ": every C is square with as many rows as A has columns");
    GroupGrams g;
    g.rows.assign(static_cast<std::size_t>(groups), {});
    g.G.assign(static_cast<std::size_t>(groups), MatrixXd());
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        const int k = rowGroup[static_cast<std::size_t>(i)];
        if (k >= groups || k < -1) throw std::invalid_argument(name + ": a row's group is outside [0, groups)");
        if (k >= 0) g.rows[static_cast<std::size_t>(k)].push_back(i);
    }
    for (int k = 0; k < groups; ++k) {
        const auto& r = g.rows[static_cast<std::size_t>(k)];
        if (r.empty()) continue;
        MatrixXd Bg(static_cast<Eigen::Index>(r.size()), A.cols());
        for (std::size_t q = 0; q < r.size(); ++q) Bg.row(static_cast<Eigen::Index>(q)) = (w.size() ? std::sqrt(w[r[q]]) : 1.0) * A.row(r[q]);     // W^½ B: G = (W^½ B)ᵀ (W^½ B) = Bᵀ W B
        g.G[static_cast<std::size_t>(k)] = Bg.transpose() * Bg;                     // once per group
    }
    return g;
}
}  // namespace

GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const GroupGrams gg = groupGrams(A, rowGroup, groups, Cs, rowWeights, "groupedCongruenceNorms");
    GroupedCongruenceNorms out;
    out.total = MatrixXd::Constant(groups, static_cast<Eigen::Index>(Cs.size()), nan);
    out.quadrature = out.total;
    for (int g = 0; g < groups; ++g) {
        if (gg.rows[static_cast<std::size_t>(g)].empty()) continue;
        for (std::size_t t = 0; t < Cs.size(); ++t) {
            if (Cs[t].size() == 0) continue;
            const CongruenceNorms n = normsFromGram(gg.G[static_cast<std::size_t>(g)], Cs[t]);
            out.total(g, static_cast<Eigen::Index>(t)) = n.total;
            out.quadrature(g, static_cast<Eigen::Index>(t)) = n.quadrature;
        }
    }
    return out;
}

PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const GroupGrams gg = groupGrams(A, rowGroup, groups, Cs, rowWeights, "pairCongruenceNorms");
    for (const auto& p : pairs) if (p.first < 0 || p.second < 0 || p.first >= groups || p.second >= groups) throw std::invalid_argument("pairCongruenceNorms: a pair is outside [0, groups)");
    const Eigen::Index nC = static_cast<Eigen::Index>(Cs.size()), nP = static_cast<Eigen::Index>(pairs.size());
    PairCongruenceNorms out;
    out.tiles.total = MatrixXd::Constant(groups, nC, nan);
    out.tiles.quadrature = out.tiles.total;
    out.pairTotal = MatrixXd::Constant(nP, nC, nan);
    out.pairQuadrature = out.pairTotal;
    std::vector<char> used(static_cast<std::size_t>(groups), 0);
    for (int g = 0; g < groups; ++g) if (!gg.rows[static_cast<std::size_t>(g)].empty()) used[static_cast<std::size_t>(g)] = 1;
    std::vector<MatrixXd> Pr(static_cast<std::size_t>(groups)), Pq(static_cast<std::size_t>(groups));
    for (Eigen::Index t = 0; t < nC; ++t) {
        if (Cs[static_cast<std::size_t>(t)].size() == 0) continue;
        const MatrixXcd H = 0.5 * (Cs[static_cast<std::size_t>(t)] + Cs[static_cast<std::size_t>(t)].adjoint());
        const MatrixXd Cr = H.real(), Cq = H.imag();
        for (int g = 0; g < groups; ++g) {
            if (!used[static_cast<std::size_t>(g)]) continue;
            Pr[static_cast<std::size_t>(g)] = Cr * gg.G[static_cast<std::size_t>(g)];
            Pq[static_cast<std::size_t>(g)] = Cq * gg.G[static_cast<std::size_t>(g)];
            const double re = Pr[static_cast<std::size_t>(g)].cwiseProduct(Pr[static_cast<std::size_t>(g)].transpose()).sum();
            const double im = -Pq[static_cast<std::size_t>(g)].cwiseProduct(Pq[static_cast<std::size_t>(g)].transpose()).sum();
            out.tiles.total(g, t) = re + im;
            out.tiles.quadrature(g, t) = im;
        }
        for (Eigen::Index k = 0; k < nP; ++k) {
            const int a = pairs[static_cast<std::size_t>(k)].first, b = pairs[static_cast<std::size_t>(k)].second;
            if (!used[static_cast<std::size_t>(a)] || !used[static_cast<std::size_t>(b)]) continue;
            // tr(X G_b X G_a) = Σ_ij (X G_b)_ij (X G_a)_ji
            const double re = Pr[static_cast<std::size_t>(b)].cwiseProduct(Pr[static_cast<std::size_t>(a)].transpose()).sum();
            const double im = -Pq[static_cast<std::size_t>(b)].cwiseProduct(Pq[static_cast<std::size_t>(a)].transpose()).sum();
            out.pairTotal(k, t) = re + im;
            out.pairQuadrature(k, t) = im;
        }
    }
    return out;
}

namespace {
std::vector<int> groups_of(const selection::MemberOfElement& g) { return std::vector<int>(g.member_of.begin(), g.member_of.end()); }
}  // namespace

GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    return groupedCongruenceNorms(A, groups_of(groups), static_cast<int>(groups.members), Cs, rowWeights);
}

PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights) {
    return pairCongruenceNorms(A, groups_of(groups), static_cast<int>(groups.members), pairs, Cs, rowWeights);
}

}  // namespace nxr::measurement
