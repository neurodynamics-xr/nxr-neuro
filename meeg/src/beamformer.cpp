// nxr::meeg — the LCMV beamformer with Brainstorm's 2016 normalisation (beamformer.h), written from Van Veen et al. 1997 and the documented behaviour of bst_inverse_linear_2016.
#include "nxr/meeg/beamformer.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "nxr/meeg/inverse.h"

namespace nxr::meeg {

namespace {

[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument("nxr::meeg: " + m); }

// the channel indices of each type, in order of first appearance (the blocks are disjoint, so their order changes nothing but the order of `ranks`)
std::vector<std::vector<Eigen::Index>> typeBlocks(const std::vector<std::string>& types) {
  std::vector<std::vector<Eigen::Index>> out;
  std::map<std::string, std::size_t> at;
  for (std::size_t i = 0; i < types.size(); ++i) {
    auto it = at.find(types[i]);
    if (it == at.end()) { it = at.emplace(types[i], out.size()).first; out.emplace_back(); }
    out[it->second].push_back(static_cast<Eigen::Index>(i));
  }
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void checkSquare(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const char* what) {
  if (C.rows() != C.cols()) bad(std::string(what) + " is not square");
  if (!C.allFinite()) bad(std::string(what) + " has a NaN or an infinity");
  if (static_cast<Eigen::Index>(types.size()) != C.rows()) bad("types has " + std::to_string(types.size()) + " entries for " + std::to_string(C.rows()) + " channels");
}

}  // namespace

TruncatedWhitening truncatedWhitener(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const TruncatedOptions& opt) {
  checkSquare(C, types, "the covariance");
  const Eigen::Index M = C.rows();
  TruncatedWhitening out;
  out.C = Eigen::MatrixXd::Zero(M, M);
  out.W = Eigen::MatrixXd::Zero(M, M);
  for (const auto& idx : typeBlocks(types)) {
    const Eigen::Index n = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd B(n, n);
    for (Eigen::Index i = 0; i < n; ++i) for (Eigen::Index j = 0; j < n; ++j) B(i, j) = C(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
    B = 0.5 * (B + B.transpose());
    // the block's singular values (a symmetric matrix's: the eigenvalues' magnitudes), in descending order, with their eigenvectors
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
    std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) { return std::abs(es.eigenvalues()(a)) > std::abs(es.eigenvalues()(b)); });
    const double largest = std::abs(es.eigenvalues()(order[0]));
    std::vector<double> s;
    Eigen::MatrixXd U(n, n);
    const double tol = static_cast<double>(n) * ulp32(std::sqrt(largest));   // n x eps(single(s_1)) on the singular values s = sqrt(eigenvalue)
    for (Eigen::Index k = 0; k < n; ++k) {
      const double sk = std::sqrt(std::abs(es.eigenvalues()(order[static_cast<std::size_t>(k)])));
      if (!(sk > tol)) continue;
      U.col(static_cast<Eigen::Index>(s.size())) = es.eigenvectors().col(order[static_cast<std::size_t>(k)]);
      s.push_back(sk);
    }
    const Eigen::Index r = static_cast<Eigen::Index>(s.size());
    if (r == 0) bad("a block of type " + types[static_cast<std::size_t>(idx.front())] + " has no eigenvalue above the rank tolerance");
    out.ranks.push_back(static_cast<int>(r));
    const Eigen::MatrixXd Ur = U.leftCols(r);
    Eigen::VectorXd sv = Eigen::Map<const Eigen::VectorXd>(s.data(), r);
    Eigen::MatrixXd Cb, Wb;
    switch (opt.regularisation) {
      case TruncatedRegularisation::Median: {
        const double m = median(s);
        sv = sv.cwiseMax(m);
        Cb = Ur * sv.array().square().matrix().asDiagonal() * Ur.transpose();
        Wb = Ur * sv.cwiseInverse().asDiagonal() * Ur.transpose();
        break;
      }
      case TruncatedRegularisation::Ridge: {
        const double rho = largest * opt.ridgeFraction;
        Cb = Ur * sv.array().square().matrix().asDiagonal() * Ur.transpose() + rho * Eigen::MatrixXd::Identity(n, n);
        Wb = Ur * (sv.array() + std::sqrt(rho)).inverse().matrix().asDiagonal() * Ur.transpose();
        break;
      }
      case TruncatedRegularisation::Diagonal: {
        const Eigen::MatrixXd T = Ur * sv.array().square().matrix().asDiagonal() * Ur.transpose();
        Cb = T.diagonal().asDiagonal();
        Wb = T.diagonal().cwiseSqrt().cwiseInverse().asDiagonal();
        break;
      }
      case TruncatedRegularisation::None:
        Cb = Ur * sv.array().square().matrix().asDiagonal() * Ur.transpose();
        Wb = Ur * sv.cwiseInverse().asDiagonal() * Ur.transpose();
        break;
    }
    for (Eigen::Index i = 0; i < n; ++i)
      for (Eigen::Index j = 0; j < n; ++j) {
        out.C(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]) = Cb(i, j);
        out.W(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]) = Wb(i, j);
      }
  }
  return out;
}

LcmvKernel lcmvKernel(const Eigen::MatrixXd& G, const Eigen::MatrixXd& orientations, const Eigen::MatrixXd& dataCovariance, const Eigen::MatrixXd& noiseCovariance,
                      const std::vector<std::string>& types, const LcmvOptions& opt) {
  checkSquare(dataCovariance, types, "the data covariance");
  checkSquare(noiseCovariance, types, "the noise covariance");
  const Eigen::Index M = G.rows();
  if (dataCovariance.rows() != M || noiseCovariance.rows() != M) bad("the covariances are not on the gain's " + std::to_string(M) + " channels");
  if (G.cols() % 3 != 0) bad("the gain is not channels x 3N");
  const Eigen::Index N = G.cols() / 3;
  if (orientations.rows() != N || orientations.cols() != 3) bad("the orientations are not " + std::to_string(N) + " x 3");
  if (!G.allFinite() || !orientations.allFinite()) bad("the gain or the orientations hold a NaN or an infinity");
  LcmvKernel out;
  TruncatedOptions rule = opt.covariance;
  if (opt.detectDiagonal && rule.regularisation != TruncatedRegularisation::Diagonal) {
    const double fd = noiseCovariance.diagonal().norm(), fa = noiseCovariance.norm();
    if (fa - fd < ulp32(fd)) {
      rule.regularisation = TruncatedRegularisation::Diagonal;
      out.notes.push_back("the noise covariance is diagonal: both covariances are whitened by their diagonal (the reference's rule)");
    }
  }
  const TruncatedWhitening d = truncatedWhitener(dataCovariance, types, rule), nz = truncatedWhitener(noiseCovariance, types, rule);
  out.dataWhitener = d.W; out.dataCovariance = d.C; out.dataRanks = d.ranks;
  out.noiseWhitener = nz.W; out.noiseCovariance = nz.C; out.noiseRanks = nz.ranks;
  out.K = Eigen::MatrixXd::Zero(N, M);
  out.gainNorm = Eigen::VectorXd::Zero(N);
  int blind = 0;
  for (Eigen::Index k = 0; k < N; ++k) {
    const Eigen::Vector3d o = orientations.row(k).transpose();
    const double on = o.norm();
    if (!(on > 0)) bad("source " + std::to_string(k) + " has a zero orientation");
    const Eigen::VectorXd l = d.W * (G.middleCols(3 * k, 3) * (o / on));
    const double ln = l.norm();
    out.gainNorm(k) = ln;
    if (!(ln > 0)) { ++blind; continue; }
    out.K.row(k) = (l / ln).transpose() * d.W;
  }
  if (blind) out.notes.push_back(std::to_string(blind) + " source(s) with no whitened gain: their rows are zero");
  return out;
}

}  // namespace nxr::meeg
