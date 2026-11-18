// nxr::hodge — see hodge_weights.h. A port of nxr.inverse.fused.
#include "nxr/operators/hodge_synthesis_weights.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace nxr::operators {

using Eigen::MatrixXd;
using Eigen::VectorXd;
constexpr double AREA_EPS = 1e-12;
constexpr int K_WALL = 3;

std::int64_t Basis::kept() const {
    std::int64_t n = 0;
    for (auto k : keep) n += k >= 0;
    return n;
}

FaceOperators::FaceOperators(const RowMat3& positions, const FaceMat& faces, bool roundToSingle) {
    nV = static_cast<int>(positions.rows());
    nF = static_cast<int>(faces.rows());
    RowMat3 P = roundToSingle ? RowMat3(positions.cast<float>().cast<double>()) : positions;       // the Studio's positions are float32
    std::vector<Eigen::Triplet<double>> tg[3], tc[3], ti;
    RowMat3 vn = RowMat3::Zero(nV, 3);
    std::vector<double> ws(static_cast<std::size_t>(nV), 0.0);
    std::vector<Eigen::Vector3d> Ns(nF);
    std::vector<double> areas(nF);
    for (int f = 0; f < nF; ++f) {
        const int v[3] = {faces(f, 0), faces(f, 1), faces(f, 2)};
        const Eigen::Vector3d a = P.row(v[0]), b = P.row(v[1]), c = P.row(v[2]);
        const Eigen::Vector3d n = (b - a).cross(c - a);
        const double twoA = n.norm();
        const bool live = twoA > AREA_EPS;
        const Eigen::Vector3d N = live ? Eigen::Vector3d(n / twoA) : Eigen::Vector3d::Zero();
        const double area = live ? twoA / 2 : 0.0;
        const double safe = live ? twoA : 1.0;
        Eigen::Vector3d g[3] = {N.cross(c - b) / safe, N.cross(a - c) / safe, N.cross(b - a) / safe};
        for (int s = 0; s < 3; ++s) {
            if (!live) g[s].setZero();
            const Eigen::Vector3d cg = N.cross(g[s]);
            for (int d = 0; d < 3; ++d) {
                tg[d].emplace_back(f, v[s], g[s][d]);
                tc[d].emplace_back(f, v[s], cg[d]);
            }
            ti.emplace_back(v[s], f, area);
            ws[static_cast<std::size_t>(v[s])] += area;
            vn.row(v[s]) += area * N.transpose();
        }
    }
    for (int d = 0; d < 3; ++d) {
        grad[d].resize(nF, nV); grad[d].setFromTriplets(tg[d].begin(), tg[d].end());
        curl[d].resize(nF, nV); curl[d].setFromTriplets(tc[d].begin(), tc[d].end());
    }
    toVertex.resize(nV, nF);
    toVertex.setFromTriplets(ti.begin(), ti.end());
    VectorXd inv(nV);
    for (int i = 0; i < nV; ++i) inv[i] = ws[static_cast<std::size_t>(i)] > 0 ? 1.0 / ws[static_cast<std::size_t>(i)] : 0.0;
    toVertex = inv.asDiagonal() * toVertex;
    normals.resize(nV, 3);
    for (int i = 0; i < nV; ++i) {
        const double ln = vn.row(i).norm();
        normals.row(i) = ln > 0 ? Eigen::RowVector3d(vn.row(i) / ln) : Eigen::RowVector3d::Zero();
    }
}

void FaceOperators::divCurl(const MatrixXd J[3], MatrixXd& div, MatrixXd& curlOut) const {
    const Eigen::Index m = J[0].cols();
    MatrixXd Jt[3] = {J[0], J[1], J[2]};
    for (int i = 0; i < nV; ++i) {                                  // the tangent part: J - (J . n) n
        const Eigen::RowVectorXd dot = J[0].row(i) * normals(i, 0) + J[1].row(i) * normals(i, 1) + J[2].row(i) * normals(i, 2);
        for (int d = 0; d < 3; ++d) Jt[d].row(i) -= dot * normals(i, d);
    }
    MatrixXd d = MatrixXd::Zero(nF, m), c = MatrixXd::Zero(nF, m);
    for (int k = 0; k < 3; ++k) { d += grad[k] * Jt[k]; c += curl[k] * Jt[k]; }
    div = toVertex * d;
    curlOut = toVertex * c;
}

void FaceOperators::vertexGradient(const MatrixXd& x, MatrixXd out[3]) const {
    for (int k = 0; k < 3; ++k) out[k] = toVertex * (grad[k] * x);
}

GluedCarry::GluedCarry(const Basis& basis, const RowMat3& inputPositions) {
    vInput_ = inputPositions.rows();
    kept_ = basis.kept();
    vGlued_ = static_cast<Eigen::Index>(basis.keep.size());
    const std::int64_t nb = vGlued_ - kept_;
    std::vector<char> isKept(static_cast<std::size_t>(vInput_), 0);
    keep_.assign(basis.keep.begin(), basis.keep.begin() + kept_);
    for (std::int64_t i = 0; i < kept_; ++i) isKept[static_cast<std::size_t>(basis.keep[static_cast<std::size_t>(i)])] = 1;
    if (nb == 0) return;
    std::vector<int> cut;
    for (Eigen::Index i = 0; i < vInput_; ++i) if (!isKept[static_cast<std::size_t>(i)]) cut.push_back(static_cast<int>(i));
    if (cut.size() < static_cast<std::size_t>(K_WALL)) throw std::invalid_argument("gluedKernel: fewer than 3 cut vertices to blend a bridge from");
    const RowMat3 Pin = inputPositions.cast<float>().cast<double>();
    wallVertex_.resize(static_cast<std::size_t>(K_WALL * nb));
    wallWeight_.resize(static_cast<std::size_t>(K_WALL * nb));
    for (std::int64_t b = 0; b < nb; ++b) {
        const Eigen::RowVector3d p = basis.positions.row(kept_ + b);
        std::vector<std::pair<double, int>> d;
        d.reserve(cut.size());
        for (int c : cut) d.emplace_back((Pin.row(c) - p).norm(), c);
        std::partial_sort(d.begin(), d.begin() + K_WALL, d.end());
        double w[K_WALL], sum = 0;
        for (int k = 0; k < K_WALL; ++k) { w[k] = 1.0 / std::max(d[static_cast<std::size_t>(k)].first, 1e-9); sum += w[k]; }
        for (int k = 0; k < K_WALL; ++k) {
            wallVertex_[static_cast<std::size_t>(K_WALL * b + k)] = d[static_cast<std::size_t>(k)].second;
            wallWeight_[static_cast<std::size_t>(K_WALL * b + k)] = w[k] / sum;
        }
    }
}

MatrixXd GluedCarry::apply(const MatrixXd& W) const {
    const Eigen::Index C = W.cols();
    if (W.rows() != 3 * vInput_) throw std::invalid_argument("gluedKernel: the kernel has " + std::to_string(W.rows()) + " rows, 3 x " + std::to_string(vInput_) + " input vertices expected");
    MatrixXd out(3 * vGlued_, C);
    for (std::int64_t i = 0; i < kept_; ++i) out.middleRows(3 * i, 3) = W.middleRows(3 * keep_[static_cast<std::size_t>(i)], 3);
    for (std::int64_t b = 0; b < vGlued_ - kept_; ++b) {
        MatrixXd acc = MatrixXd::Zero(3, C);
        for (int k = 0; k < K_WALL; ++k) acc += wallWeight_[static_cast<std::size_t>(K_WALL * b + k)] * W.middleRows(3 * wallVertex_[static_cast<std::size_t>(K_WALL * b + k)], 3);
        out.middleRows(3 * (kept_ + b), 3) = acc;
    }
    return out;
}

MatrixXd gluedKernel(const MatrixXd& W, const Basis& basis, const RowMat3& inputPositions) {
    if (W.rows() != 3 * inputPositions.rows()) throw std::invalid_argument("gluedKernel: the kernel has " + std::to_string(W.rows()) + " rows, 3 x " + std::to_string(inputPositions.rows()) + " input vertices expected");
    return GluedCarry(basis, inputPositions).apply(W);
}

static MatrixXd rows(const MatrixXd& M, const std::vector<int>& idx) {
    MatrixXd out(idx.size(), M.cols());
    for (std::size_t i = 0; i < idx.size(); ++i) out.row(static_cast<Eigen::Index>(i)) = M.row(idx[i]);
    return out;
}

static void split(const MatrixXd& W, Eigen::Index V, MatrixXd J[3]) {
    for (int d = 0; d < 3; ++d) {
        J[d].resize(V, W.cols());
        for (Eigen::Index v = 0; v < V; ++v) J[d].row(v) = W.row(3 * v + d);
    }
}

WeightsReader::WeightsReader(const Basis& basis, int block, const std::function<void(std::int64_t, std::int64_t)>& progress) : ops_(basis.positions, basis.faces) {
    V_ = ops_.nV; K_ = basis.P.rows();
    if (basis.kind.size() != static_cast<std::size_t>(K_)) throw std::invalid_argument("hodgeWeights: kind and P disagree");
    for (Eigen::Index k = 0; k < K_; ++k) (basis.kind[static_cast<std::size_t>(k)] == 2 ? co_ : ex_).push_back(static_cast<int>(k));
    // no copies of the basis' rows: P's rows scaled by the mass in place (the same products, element by element), and the Gram read from P itself
    const VectorXd& M = basis.mass;
    PcoM_ = rows(basis.P, co_);
    PcoM_.array().rowwise() *= M.transpose().array();
    PexM_ = rows(basis.P, ex_);
    PexM_.array().rowwise() *= M.transpose().array();
    if (!co_.empty()) {
        MatrixXd G = PcoM_ * basis.P(co_, Eigen::all).transpose();
        G = 0.5 * (G + G.transpose());
        gram_.compute(G);
    }
    order_ = ex_;
    order_.insert(order_.end(), co_.begin(), co_.end());
    const Eigen::Index n = order_.size();
    ne_ = ex_.size();
    MatrixXd A = MatrixXd::Zero(n, n);
    for (Eigen::Index k0 = 0; k0 < n; k0 += block) {                   // THE FLOW SYSTEM, a block of modes at a time
        const Eigen::Index w = std::min<Eigen::Index>(block, n - k0);
        MatrixXd x(V_, w), g[3], fl[3];
        for (Eigen::Index j = 0; j < w; ++j) x.col(j) = basis.P.row(order_[static_cast<std::size_t>(k0 + j)]).transpose();
        ops_.vertexGradient(x, g);
        for (int d = 0; d < 3; ++d) fl[d].resize(V_, w);
        for (Eigen::Index j = 0; j < w; ++j) {
            const bool exact = basis.kind[static_cast<std::size_t>(order_[static_cast<std::size_t>(k0 + j)])] != 2;
            for (Eigen::Index v = 0; v < V_; ++v) {
                const Eigen::Vector3d gv(g[0](v, j), g[1](v, j), g[2](v, j));
                const Eigen::Vector3d f = exact ? Eigen::Vector3d(-gv) : Eigen::Vector3d(-Eigen::Vector3d(ops_.normals.row(v)).cross(gv));
                for (int d = 0; d < 3; ++d) fl[d](v, j) = f[d];
            }
        }
        MatrixXd d2, c2;
        ops_.divCurl(fl, d2, c2);
        A.block(0, k0, ne_, w) = PexM_ * d2;
        A.block(ne_, k0, n - ne_, w) = PcoM_ * c2;
        if (progress) progress(k0 + w, n);
    }
    flowLu_.compute(A);
}

// the flow weights [K x C] (b, by mode), and with `normalReading` the coexact modes' normal-current readings e = Pco M (n . J)
MatrixXd WeightsReader::flowSolve(const MatrixXd& W, MatrixXd* normalReading) const {
    const Eigen::Index C = W.cols();
    if (W.rows() != 3 * V_) throw std::invalid_argument("hodgeWeights: the kernel has " + std::to_string(W.rows()) + " rows, 3 x " + std::to_string(V_) + " glued vertices expected");
    MatrixXd J[3];
    split(W, V_, J);
    MatrixXd div, curl;
    if (normalReading) {
        MatrixXd normal = MatrixXd::Zero(V_, C);                      // n . J
        for (int d = 0; d < 3; ++d) normal += ops_.normals.col(d).asDiagonal() * J[d];
        *normalReading = PcoM_ * normal;
    }
    ops_.divCurl(J, div, curl);
    const Eigen::Index n = order_.size();
    MatrixXd rhs(n, C);
    rhs << PexM_ * div, PcoM_ * curl;
    const MatrixXd b = flowLu_.solve(rhs);
    MatrixXd out = MatrixXd::Zero(K_, C);
    for (Eigen::Index i = 0; i < n; ++i) out.row(order_[static_cast<std::size_t>(i)]) = b.row(i);
    return out;
}

MatrixXd WeightsReader::flow(const MatrixXd& W) const { return flowSolve(W, nullptr); }

MatrixXd WeightsReader::weights(const MatrixXd& W) const {
    MatrixXd e;
    const MatrixXd b = flowSolve(W, &e);
    MatrixXd out = MatrixXd::Zero(2 * K_, W.cols());
    if (!co_.empty()) {
        const MatrixXd a = gram_.solve(e);
        for (std::size_t i = 0; i < co_.size(); ++i) out.row(co_[i]) = a.row(static_cast<Eigen::Index>(i));
    }
    out.bottomRows(K_) = b;
    return out;
}

MatrixXd hodgeWeights(const Basis& basis, const MatrixXd& W, int block, const std::function<void(std::int64_t, std::int64_t)>& progress) {
    return WeightsReader(basis, block, progress).weights(W);
}

Synthesis hodgeSynthesis(const Basis& basis, const MatrixXd& weights, const FaceOperators* opsIn) {
    std::unique_ptr<FaceOperators> own;
    if (!opsIn) { own = std::make_unique<FaceOperators>(basis.positions, basis.faces); opsIn = own.get(); }
    const FaceOperators& ops = *opsIn;
    const Eigen::Index K = basis.P.rows(), V = ops.nV, m = weights.cols();
    if (weights.rows() != 2 * K) throw std::invalid_argument("hodgeSynthesis: weights for " + std::to_string(weights.rows() / 2) + " modes, the basis has " + std::to_string(K));
    Synthesis s;
    s.normal = MatrixXd::Zero(V, m); s.A = MatrixXd::Zero(V, m); s.S = MatrixXd::Zero(V, m);
    for (Eigen::Index k = 0; k < K; ++k) {
        const Eigen::VectorXd p = basis.P.row(k).transpose();
        if (basis.kind[static_cast<std::size_t>(k)] == 2) { s.normal += p * weights.row(k); s.S += p * weights.row(K + k); }
        else s.A += p * weights.row(K + k);
    }
    MatrixXd gA[3], gB[3];
    ops.vertexGradient(s.A, gA);
    ops.vertexGradient(s.S, gB);
    s.J.resize(3 * V, m);
    for (Eigen::Index v = 0; v < V; ++v) {
        const Eigen::Vector3d n = ops.normals.row(v);
        Eigen::Matrix<double, 3, Eigen::Dynamic> J(3, m);
        for (Eigen::Index j = 0; j < m; ++j) {
            const Eigen::Vector3d ga(gA[0](v, j), gA[1](v, j), gA[2](v, j)), gb(gB[0](v, j), gB[1](v, j), gB[2](v, j));
            Eigen::Vector3d Jv = -ga - n.cross(gb);
            Jv += (s.normal(v, j) - Jv.dot(n)) * n;
            J.col(j) = Jv;
        }
        s.J.middleRows(3 * v, 3) = J;
    }
    return s;
}

MatrixXd modePower(const MatrixXd& B, const std::vector<MatrixXd>& S) {
    MatrixXd out(B.rows(), S.size());
    for (std::size_t b = 0; b < S.size(); ++b) out.col(static_cast<Eigen::Index>(b)) = ((B * S[b]).cwiseProduct(B)).rowwise().sum();
    return out;
}

}  // namespace nxr::hodge
