// nxr::pca — see include/nxr/pca.h.
#include "nxr/stats/pca.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace nxr::stats {

using Eigen::MatrixXd;
using Eigen::VectorXd;

PcaResult pca(const MatrixXd& X, const PcaOptions& options) {
    const Eigen::Index n = X.rows(), p = X.cols();
    if (n < 2) throw std::invalid_argument("pca: at least two observations (rows), got " + std::to_string(n));
    if (p < 1) throw std::invalid_argument("pca: at least one feature (column)");
    if (!X.allFinite()) throw std::invalid_argument("pca: the table holds a value that is not finite (NaN or infinite); leave out or fill the observation or feature first");
    PcaResult r;
    r.mean = X.colwise().mean().transpose();
    MatrixXd Xc = X.rowwise() - r.mean.transpose();
    r.scale = VectorXd::Ones(p);
    if (options.scale == Scale::UnitVariance) {
        for (Eigen::Index j = 0; j < p; ++j) {
            const double sd = std::sqrt(Xc.col(j).squaredNorm() / static_cast<double>(n - 1));
            if (!(sd > 0)) throw std::invalid_argument("pca: feature " + std::to_string(j) + " is constant, so it cannot be scaled to unit variance");
            r.scale[j] = sd;
        }
        Xc = Xc.array().rowwise() / r.scale.transpose().array();
    }
    const double total = Xc.squaredNorm() / static_cast<double>(n - 1);
    if (!(total > 0)) throw std::invalid_argument("pca: the table has no variance (every observation is the same)");
    Eigen::BDCSVD<MatrixXd> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const VectorXd S = svd.singularValues();
    const double tol = S[0] * static_cast<double>(std::max(n, p)) * std::numeric_limits<double>::epsilon();
    Eigen::Index k = 0;
    while (k < S.size() && S[k] > tol) ++k;
    r.rank = k;
    r.singularValues = S.head(k);
    r.components = svd.matrixV().leftCols(k).transpose();
    r.scores = svd.matrixU().leftCols(k) * r.singularValues.asDiagonal();
    for (Eigen::Index c = 0; c < k; ++c) {          // the sign: the entry of largest |loading| is positive
        Eigen::Index at = 0;
        r.components.row(c).cwiseAbs().maxCoeff(&at);
        if (r.components(c, at) < 0) { r.components.row(c) *= -1; r.scores.col(c) *= -1; }
    }
    r.explainedVariance = r.singularValues.array().square() / static_cast<double>(n - 1);
    r.explainedVarianceRatio = r.explainedVariance / total;
    return r;
}

MatrixXd reconstruct(const MatrixXd& scores, const MatrixXd& components, const VectorXd& mean, const VectorXd& scale, int k) {
    if (scores.cols() != components.rows()) throw std::invalid_argument("pca: the scores hold " + std::to_string(scores.cols()) + " components, the components " + std::to_string(components.rows()));
    if (components.cols() != mean.size() || mean.size() != scale.size()) throw std::invalid_argument("pca: the components, the mean and the scale must have the same number of features");
    if (k < 0 || k > components.rows()) throw std::invalid_argument("pca: k is 0 to " + std::to_string(components.rows()) + " (the components held), got " + std::to_string(k));
    MatrixXd out = k > 0 ? MatrixXd(scores.leftCols(k) * components.topRows(k)) : MatrixXd::Zero(scores.rows(), components.cols());
    out = out.array().rowwise() * scale.transpose().array();
    out.rowwise() += mean.transpose();
    return out;
}

}  // namespace nxr::stats
