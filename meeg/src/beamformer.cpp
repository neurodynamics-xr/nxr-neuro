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
  out.applied = rule.regularisation;
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

// ── MNE-Python's conventions (beamformer.h, mne.beamformer.make_lcmv) ──────────────────────────────────────────────────────────────────────────────────────────────

std::string modalityOf(const std::string& type) { return type.rfind("MEG", 0) == 0 ? std::string("MEG") : type; }

std::vector<std::vector<Eigen::Index>> modalityBlocks(const std::vector<std::string>& types) {
  std::vector<std::string> m;
  m.reserve(types.size());
  for (const auto& t : types) m.push_back(modalityOf(t));
  return typeBlocks(m);
}

RankWhitening rankWhitener(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const std::vector<int>& ranks) {
  checkSquare(C, types, "the noise covariance");
  const auto blocks = modalityBlocks(types);
  if (ranks.size() != blocks.size()) bad(std::to_string(ranks.size()) + " ranks for " + std::to_string(blocks.size()) + " modalities");
  const Eigen::Index M = C.rows();
  RankWhitening out;
  out.W = Eigen::MatrixXd::Zero(M, M);
  out.ranks = ranks;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& idx = blocks[b];
    const Eigen::Index n = static_cast<Eigen::Index>(idx.size());
    const int r = ranks[b];
    out.modalities.push_back(modalityOf(types[static_cast<std::size_t>(idx.front())]));
    if (r < 1 || r > n) bad("the rank " + std::to_string(r) + " of the " + out.modalities.back() + " channels is outside [1, " + std::to_string(n) + "]");
    Eigen::MatrixXd B(n, n);
    for (Eigen::Index i = 0; i < n; ++i) for (Eigen::Index j = 0; j < n; ++j) B(i, j) = C(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (B + B.transpose()));   // ascending
    Eigen::VectorXd f = Eigen::VectorXd::Zero(n);
    for (Eigen::Index k = n - r; k < n; ++k) if (es.eigenvalues()(k) > 0) f(k) = 1.0 / std::sqrt(es.eigenvalues()(k));
    const Eigen::MatrixXd Wb = es.eigenvectors() * f.asDiagonal() * es.eigenvectors().transpose();
    for (Eigen::Index i = 0; i < n; ++i) for (Eigen::Index j = 0; j < n; ++j) out.W(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]) = Wb(i, j);
  }
  return out;
}

namespace {

// A^p of a symmetric positive semi-definite matrix, p in {-1, -1/2}: the eigenvalues at or below 1e-7 x the largest are dropped (MNE-Python's _sym_mat_pow)
Eigen::MatrixXd symPow(const Eigen::MatrixXd& A, double p) {
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
  const Eigen::VectorXd& s = es.eigenvalues();
  const double limit = s(s.size() - 1) * 1e-7;
  if ((s.array() < -limit).any()) bad("a matrix of the filter is not positive semi-definite");
  Eigen::VectorXd f(s.size());
  for (Eigen::Index i = 0; i < s.size(); ++i) f(i) = s(i) <= limit ? 0.0 : p == -1 ? 1.0 / s(i) : 1.0 / std::sqrt(s(i));
  return es.eigenvectors() * f.asDiagonal() * es.eigenvectors().transpose();
}
// the pseudo-inverse of a k x k denominator: 1 / x for one orientation (a non-finite one taken as 1, as the reference), symPow(-1) for three
Eigen::MatrixXd denominatorInverse(const Eigen::MatrixXd& D) {
  if (D.rows() == 1) { const double v = 1.0 / D(0, 0); return Eigen::MatrixXd::Constant(1, 1, std::isfinite(v) ? v : 1.0); }
  return symPow(D, -1);
}

}  // namespace

MneLcmvKernel mneLcmvKernel(const Eigen::MatrixXd& G, const Eigen::MatrixXd& normals, const Eigen::MatrixXd& dataCovariance, const Eigen::MatrixXd& whitener, int rank,
                            const MneLcmvOptions& opt) {
  const Eigen::Index M = G.rows();
  if (G.cols() % 3 != 0) bad("the gain is not channels x 3N");
  const Eigen::Index N = G.cols() / 3;
  if (normals.rows() != N || normals.cols() != 3) bad("the normals are not " + std::to_string(N) + " x 3");
  if (dataCovariance.rows() != M || dataCovariance.cols() != M || whitener.rows() != M || whitener.cols() != M) bad("the data covariance and the whitener are not on the gain's " + std::to_string(M) + " channels");
  if (!G.allFinite() || !normals.allFinite() || !dataCovariance.allFinite() || !whitener.allFinite()) bad("an input holds a NaN or an infinity");
  if (rank < 1 || rank > M) bad("the rank " + std::to_string(rank) + " is outside [1, " + std::to_string(M) + "]");
  if (!(opt.reg >= 0)) bad("reg is negative");
  MneLcmvKernel out;
  out.rank = rank;
  const Eigen::MatrixXd L = whitener * G;
  Eigen::MatrixXd C = whitener * dataCovariance * whitener.transpose();
  C = 0.5 * (C + C.transpose());
  // the regularised pseudo-inverse: loading mu on every singular value, then the `rank` largest of C + mu I inverted
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> e0(C);
  out.loading = opt.reg * e0.eigenvalues().cwiseAbs().mean();
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> e1(C + out.loading * Eigen::MatrixXd::Identity(M, M));
  std::vector<Eigen::Index> order(static_cast<std::size_t>(M));
  for (Eigen::Index i = 0; i < M; ++i) order[static_cast<std::size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) { return std::abs(e1.eigenvalues()(a)) > std::abs(e1.eigenvalues()(b)); });
  Eigen::MatrixXd Ci = Eigen::MatrixXd::Zero(M, M);
  for (int i = 0; i < rank; ++i) {
    const Eigen::Index k = order[static_cast<std::size_t>(i)];
    const double ev = e1.eigenvalues()(k);
    if (ev != 0) Ci += e1.eigenvectors().col(k) * e1.eigenvectors().col(k).transpose() / ev;
  }
  const Eigen::MatrixXd Ci2 = Ci * Ci;
  if (opt.weightNorm == LcmvWeightNorm::Nai) out.noise = std::max(e0.eigenvalues()(M - rank), out.loading);   // the rank-th largest eigenvalue of C (ascending order)
  const bool free = opt.orientation == LcmvOrientation::Free;
  const Eigen::Index rows = free ? 3 : 1;
  out.K = Eigen::MatrixXd::Zero(N * rows, M);
  if (opt.orientation == LcmvOrientation::MaxPower) out.orientations = Eigen::MatrixXd::Zero(N, 3);
  Eigen::MatrixXd F(N * rows, M);
  for (Eigen::Index k = 0; k < N; ++k) {
    Eigen::MatrixXd Lk = L.middleCols(3 * k, 3);
    const Eigen::Vector3d nk = normals.row(k).transpose();
    if (!(nk.norm() > 0)) bad("source " + std::to_string(k) + " has a zero normal");
    const Eigen::JacobiSVD<Eigen::MatrixXd> sv(Lk);
    if (sv.singularValues()(0) > 1e6 * sv.singularValues()(2))
      bad("source " + std::to_string(k) + ": its gain is singular (a spherical MEG model's radial direction); the reference's reduce_rank is not ported");
    if (opt.orientation == LcmvOrientation::Normal) {
      Lk = Lk * nk;
    } else if (opt.orientation == LcmvOrientation::MaxPower) {
      const Eigen::Matrix3d den = Lk.transpose() * Ci * Lk;
      const Eigen::Matrix3d pick = opt.weightNorm == LcmvWeightNorm::None ? Eigen::Matrix3d(symPow(den, -1)) : Eigen::Matrix3d(symPow(Lk.transpose() * Ci2 * Lk, -1) * den);
      const Eigen::EigenSolver<Eigen::Matrix3d> es(pick);
      Eigen::Index best = 0;
      for (Eigen::Index i = 1; i < 3; ++i) if (std::abs(es.eigenvalues()(i)) >= std::abs(es.eigenvalues()(best))) best = i;
      Eigen::Vector3d o = es.eigenvectors().col(best).real();
      o.normalize();
      if (o.dot(nk) < 0) o = -o;
      out.orientations.row(k) = o.transpose();
      Lk = Lk * o;
    }
    const Eigen::MatrixXd B = Lk.transpose() * Ci;                     // the numerator L_k^T C^-1
    Eigen::MatrixXd Fk;
    if (opt.weightNorm == LcmvWeightNorm::UnitNoiseGainInvariant) {
      Fk = B.rows() == 1 ? Eigen::MatrixXd(B / B.norm()) : Eigen::MatrixXd(symPow(B * B.transpose(), -0.5) * B);
    } else {
      Fk = denominatorInverse(B * Lk) * B;
      if (opt.weightNorm != LcmvWeightNorm::None)
        for (Eigen::Index i = 0; i < Fk.rows(); ++i) { const double nr = Fk.row(i).norm(); if (nr > 0) Fk.row(i) /= nr; }
      if (opt.weightNorm == LcmvWeightNorm::Nai) Fk /= std::sqrt(out.noise);
    }
    F.middleRows(k * rows, rows) = Fk;
  }
  out.K = F * whitener;
  return out;
}

}  // namespace nxr::meeg
