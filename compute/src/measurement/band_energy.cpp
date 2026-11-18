// The bodies of the filterbanks' energies (timefilterbank scalogram / energies, graphfilterbank scaleEnergy / memberEnergy / bandMoments / bandPower, jointfilterbank scalogram): moved, not edited.
#include "nxr/measurement/band_energy.h"
#include <stdexcept>
#include <string>

namespace nxr::measurement {
using Eigen::MatrixXd; using Eigen::VectorXd;

std::vector<MatrixXd> scalogram(const CoefficientsC& W) {
    std::vector<MatrixXd> P;
    for (const auto& w : W) P.push_back(w.cwiseAbs2());
    return P;
}

MatrixXd energies(const CoefficientsC& W) {
    MatrixXd E(static_cast<Eigen::Index>(W.size()), W.empty() ? 0 : W[0].cols());
    for (std::size_t m = 0; m < W.size(); ++m) E.row(static_cast<Eigen::Index>(m)) = W[m].cwiseAbs2().colwise().sum();
    return E;
}

VectorXd memberTotals(const CoefficientsC& W) {
    VectorXd e(static_cast<Eigen::Index>(W.size()));
    for (std::size_t m = 0; m < W.size(); ++m) e(static_cast<Eigen::Index>(m)) = W[m].cwiseAbs2().sum();
    return e;
}

MatrixXd scaleEnergy(const CoefficientsR& W) {
    MatrixXd E = MatrixXd::Zero(W.at(0).rows(), W[0].cols());
    for (const auto& w : W) E += w.array().square().matrix();
    return E;
}
MatrixXd memberEnergy(const CoefficientsR& W) {
    MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) E.row(static_cast<Eigen::Index>(m)) = W[m].array().square().colwise().sum();
    return E;
}

MatrixXd memberEnergy(const CoefficientsR& W, const VectorXd& weights) {
    MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) {
        if (W[m].rows() != weights.size()) throw std::invalid_argument("memberEnergy: " + std::to_string(weights.size()) + " weights for " + std::to_string(W[m].rows()) + " vertices");
        E.row(static_cast<Eigen::Index>(m)) = (W[m].array().square().colwise() * weights.array()).colwise().sum();
    }
    return E;
}

MatrixXd memberEnergy(const CoefficientsC& W) {
    MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) E.row(static_cast<Eigen::Index>(m)) = W[m].array().abs2().colwise().sum();
    return E;
}

MatrixXd memberEnergy(const CoefficientsC& W, const VectorXd& weights) {
    MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) {
        if (W[m].rows() != weights.size()) throw std::invalid_argument("memberEnergy: " + std::to_string(weights.size()) + " weights for " + std::to_string(W[m].rows()) + " vertices");
        E.row(static_cast<Eigen::Index>(m)) = (W[m].array().abs2().colwise() * weights.array()).colwise().sum();
    }
    return E;
}

BandMoments bandMoments(const MatrixXd& G, const Eigen::MatrixXcd& C, bool withMatrices) {
    if (C.rows() != C.cols() || C.rows() != G.rows())
        throw std::invalid_argument("bandMoments: C is " + std::to_string(C.rows()) + " x " + std::to_string(C.cols()) + " for " + std::to_string(G.rows()) + " modes");
    BandMoments r;
    r.power = G.array().square().matrix().transpose() * C.diagonal().real();
    if (withMatrices) {
        r.matrices.reserve(static_cast<std::size_t>(G.cols()));
        for (Eigen::Index m = 0; m < G.cols(); ++m) {
            const Eigen::VectorXcd g = G.col(m).cast<std::complex<double>>();
            r.matrices.push_back(g.asDiagonal() * C * g.asDiagonal());
        }
    }
    return r;
}

VectorXd bandPower(const MatrixXd& G, const VectorXd& modePower) {
    if (modePower.size() != G.rows()) throw std::invalid_argument("bandPower: " + std::to_string(modePower.size()) + " mode powers for " + std::to_string(G.rows()) + " modes");
    return G.array().square().matrix().transpose() * modePower;
}

}  // namespace nxr::measurement
