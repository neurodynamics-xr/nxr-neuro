// nxr::meeg — the unconstrained minimum-norm kernel, from the maths in docs/specs/2026-11-10-neuro-forward-inverse.md (section 2).
#include "nxr/meeg/inverse.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace nxr::meeg {

namespace {

[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument("nxr::meeg: " + m); }

double rankUnit(double largest, RankTolerance t) {
  if (largest <= 0) return 0;
  if (t == RankTolerance::SinglePrecision) return ulp32(largest);
  return std::ldexp(1.0, static_cast<int>(std::floor(std::log2(largest))) - 52);
}

// the channel indices of each type, in order of first appearance
std::vector<std::vector<int>> blocksOf(const std::vector<std::string>& types, bool cross) {
  std::vector<std::vector<int>> out;
  if (cross) {
    out.emplace_back();
    for (int i = 0; i < static_cast<int>(types.size()); ++i) out[0].push_back(i);
    return out;
  }
  std::map<std::string, std::size_t> at;
  for (int i = 0; i < static_cast<int>(types.size()); ++i) {
    auto it = at.find(types[static_cast<std::size_t>(i)]);
    if (it == at.end()) { it = at.emplace(types[static_cast<std::size_t>(i)], out.size()).first; out.emplace_back(); }
    out[it->second].push_back(i);
  }
  return out;
}

void checkTypes(const std::vector<std::string>& types, Eigen::Index m) {
  if (static_cast<Eigen::Index>(types.size()) != m) bad("types has " + std::to_string(types.size()) + " entries for " + std::to_string(m) + " channels");
}

}  // namespace

double ulp32(double x) {
  if (!(x > 0)) return 0;
  return std::ldexp(1.0, static_cast<int>(std::floor(std::log2(x))) - 23);
}

KernelOptions mneKernelOptions() {
  KernelOptions o;
  o.whitener.form = WhitenerForm::Pca;
  o.whitener.jointTypes = true;
  o.whitener.ridgeOverRank = true;
  o.depth.exponent = 0.8;
  o.depth.norm = DepthNorm::Spectral;
  o.depth.limitOn = DepthLimitOn::Norm;
  o.depth.channels = DepthChannels::OneType;
  o.snrNormalisation = SnrNormalisation::WhitenerRank;
  return o;
}

Whitener whitener(const Eigen::MatrixXd& C, const std::vector<std::string>& types, const WhitenerOptions& opt) {
  if (C.rows() != C.cols()) bad("the noise covariance is not square");
  if (!C.allFinite()) bad("the noise covariance has a NaN or an infinity");
  checkTypes(types, C.rows());
  const Eigen::Index M = C.rows();
  Whitener out;
  Regularisation reg = opt.regularisation;
  // a diagonal covariance: ||C||_F - ||diag C||_F below one ulp of ||diag C||_F
  if (opt.detectDiagonal && reg != Regularisation::Diagonal) {
    const double fd = C.diagonal().norm(), fa = C.norm();
    if (fa - fd < ulp32(fd)) {
      reg = Regularisation::Diagonal;
      out.notes.push_back("the noise covariance is diagonal: whitened by its diagonal");
    }
  }
  // JOINT TYPES (MNE-Python): the types of one modality (MEG MAG with MEG GRAD) are whitened together, their cross-covariance kept; the modalities (MEG, EEG, ...)
  // stay separate. The ridge is added per type, on the range of that type's block (rho_t U_r U_r^T: the null space of a projector stays null), then the
  // modality's matrix is whitened, in the scale of its own diagonal for conditioning.
  if (opt.jointTypes && reg != Regularisation::Diagonal) {
    std::map<std::string, std::vector<int>> groups;
    std::vector<std::string> order;
    for (int i = 0; i < static_cast<int>(types.size()); ++i) {
      const std::string& t = types[static_cast<std::size_t>(i)];
      const std::string g = t.rfind("MEG", 0) == 0 ? "MEG" : t;
      if (!groups.count(g)) order.push_back(g);
      groups[g].push_back(i);
    }
    std::vector<Eigen::MatrixXd> cores;
    int rows = 0;
    for (const std::string& gname : order) {
      const std::vector<int>& gi = groups[gname];
      const Eigen::Index m = static_cast<Eigen::Index>(gi.size());
      Eigen::MatrixXd Cr(m, m);
      for (Eigen::Index i = 0; i < m; ++i) for (Eigen::Index j = 0; j < m; ++j) Cr(i, j) = 0.5 * (C(gi[static_cast<std::size_t>(i)], gi[static_cast<std::size_t>(j)]) + C(gi[static_cast<std::size_t>(j)], gi[static_cast<std::size_t>(i)]));
      if (reg == Regularisation::Ridge) {
        std::vector<std::string> gt;
        for (int i : gi) gt.push_back(types[static_cast<std::size_t>(i)]);
        for (const auto& idx : blocksOf(gt, false)) {
          const Eigen::Index n = static_cast<Eigen::Index>(idx.size());
          Eigen::MatrixXd Ct(n, n);
          for (Eigen::Index i = 0; i < n; ++i) for (Eigen::Index j = 0; j < n; ++j) Ct(i, j) = Cr(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ct);
          const Eigen::VectorXd ev = es.eigenvalues().reverse();
          const Eigen::MatrixXd U = es.eigenvectors().rowwise().reverse();
          const double unit = rankUnit(ev(0), opt.rankTolerance);
          Eigen::Index r = 0;
          while (r < n && ev(r) > static_cast<double>(n) * unit && ev(r) > 0) ++r;
          const double rho = opt.ridgeFraction * Ct.trace() / static_cast<double>(opt.ridgeOverRank ? std::max<Eigen::Index>(r, 1) : n);
          const Eigen::MatrixXd Ur = U.leftCols(r);
          const Eigen::MatrixXd add = rho * Ur * Ur.transpose();
          for (Eigen::Index i = 0; i < n; ++i) for (Eigen::Index j = 0; j < n; ++j) Cr(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]) += add(i, j);
        }
      }
      const Eigen::MatrixXd& Cs = Cr;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Cs);
      const Eigen::VectorXd ev = es.eigenvalues().reverse();
      const Eigen::MatrixXd U = es.eigenvectors().rowwise().reverse();
      const double unit = rankUnit(ev(0), opt.rankTolerance);
      Eigen::Index r = 0;
      while (r < m && ev(r) > static_cast<double>(m) * unit && ev(r) > 0) ++r;
      if (r < m) out.notes.push_back(gname + " channels have rank " + std::to_string(r) + " of " + std::to_string(m));
      out.rank += static_cast<int>(r);
      Eigen::VectorXd f(r);
      for (Eigen::Index i = 0; i < r; ++i) f(i) = 1.0 / std::sqrt(ev(i));
      const Eigen::MatrixXd Ur = U.leftCols(r);
      Eigen::MatrixXd core = opt.form == WhitenerForm::Symmetric ? Eigen::MatrixXd(Ur * f.asDiagonal() * Ur.transpose()) : Eigen::MatrixXd(f.asDiagonal() * Ur.transpose());
      cores.push_back(core);
      rows += static_cast<int>(core.rows());
    }
    out.W = Eigen::MatrixXd::Zero(rows, M);
    int row0 = 0;
    for (std::size_t g = 0; g < order.size(); ++g) {
      const std::vector<int>& gi = groups[order[g]];
      for (Eigen::Index i = 0; i < cores[g].rows(); ++i)
        for (Eigen::Index j = 0; j < cores[g].cols(); ++j)
          out.W(opt.form == WhitenerForm::Symmetric ? gi[static_cast<std::size_t>(i)] : row0 + i, gi[static_cast<std::size_t>(j)]) = cores[g](i, j);
      if (opt.form != WhitenerForm::Symmetric) row0 += static_cast<int>(cores[g].rows());
    }
    return out;
  }
  const auto blocks = blocksOf(types, false);
  std::vector<Eigen::MatrixXd> parts;   // per block: W_t in the chosen form
  int rows = 0;
  for (const auto& idx : blocks) {
    const Eigen::Index n = static_cast<Eigen::Index>(idx.size());
    Eigen::MatrixXd Ct(n, n);
    for (Eigen::Index i = 0; i < n; ++i)
      for (Eigen::Index j = 0; j < n; ++j) Ct(i, j) = 0.5 * (C(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]) + C(idx[static_cast<std::size_t>(j)], idx[static_cast<std::size_t>(i)]));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ct);
    const Eigen::VectorXd ev = es.eigenvalues().reverse();                 // descending
    const Eigen::MatrixXd U = es.eigenvectors().rowwise().reverse();
    const double unit = rankUnit(ev(0), opt.rankTolerance);
    Eigen::Index r = 0;
    while (r < n && ev(r) > static_cast<double>(n) * unit && ev(r) > 0) ++r;
    if (r < n) out.notes.push_back("channel block of " + std::to_string(n) + " has rank " + std::to_string(r));
    out.rank += static_cast<int>(r);
    const Eigen::MatrixXd Ur = U.leftCols(r);
    Eigen::VectorXd f(r);                                                   // the diagonal of the form's middle factor
    if (reg == Regularisation::Ridge) {
      const double rho = opt.ridgeFraction * Ct.trace() / static_cast<double>(n);
      for (Eigen::Index i = 0; i < r; ++i) f(i) = 1.0 / std::sqrt(ev(i) + rho);
    } else if (reg == Regularisation::None) {
      for (Eigen::Index i = 0; i < r; ++i) f(i) = 1.0 / std::sqrt(ev(i));
    }
    Eigen::MatrixXd Wt;
    if (reg == Regularisation::Diagonal) {
      // C_r = U_r diag(ev_r) U_r^T; W = diag(1 / sqrt(diag C_r)), [n x n] whatever the form
      Eigen::VectorXd d = (Ur.array().square().matrix() * ev.head(r));
      Wt = Eigen::MatrixXd::Zero(n, n);
      for (Eigen::Index i = 0; i < n; ++i) Wt(i, i) = d(i) > 0 ? 1.0 / std::sqrt(d(i)) : 0.0;
      rows += static_cast<int>(n);
    } else if (opt.form == WhitenerForm::Symmetric) {
      Wt = Ur * f.asDiagonal() * Ur.transpose();
      rows += static_cast<int>(n);
    } else {
      Wt = f.asDiagonal() * Ur.transpose();
      rows += static_cast<int>(r);
    }
    parts.push_back(Wt);
  }
  // assemble blockdiag(W_t): a square block sits at its channels' rows and columns; a PCA block's rows are stacked in the blocks' order
  out.W = Eigen::MatrixXd::Zero(rows, M);
  int row0 = 0;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto& idx = blocks[b];
    const bool square = parts[b].rows() == parts[b].cols() && (opt.form == WhitenerForm::Symmetric || reg == Regularisation::Diagonal);
    for (Eigen::Index i = 0; i < parts[b].rows(); ++i)
      for (Eigen::Index j = 0; j < parts[b].cols(); ++j)
        out.W(square ? idx[static_cast<std::size_t>(i)] : row0 + i, idx[static_cast<std::size_t>(j)]) = parts[b](i, j);
    if (!square) row0 += static_cast<int>(parts[b].rows());
  }
  return out;
}

Eigen::VectorXd depthWeights(const Eigen::MatrixXd& G, const std::vector<std::string>& types, const DepthOptions& opt) {
  if (G.cols() % 3 != 0) bad("the gain has " + std::to_string(G.cols()) + " columns, not a multiple of 3");
  if (!G.allFinite()) bad("the gain has a NaN or an infinity");
  checkTypes(types, G.rows());
  const Eigen::Index N = G.cols() / 3;
  Eigen::VectorXd alpha = Eigen::VectorXd::Ones(N);
  if (!opt.enabled || N == 0) return alpha;
  // the rows the norm is computed on
  std::vector<int> rows;
  if (opt.channels == DepthChannels::OneType) {
    for (const char* want : {"MEG GRAD", "MEG MAG", "EEG"}) {
      for (int i = 0; i < static_cast<int>(types.size()); ++i) if (types[static_cast<std::size_t>(i)] == want) rows.push_back(i);
      if (!rows.empty()) break;
    }
  }
  if (rows.empty()) for (int i = 0; i < static_cast<int>(G.rows()); ++i) rows.push_back(i);
  Eigen::VectorXd nu(N);
  for (Eigen::Index k = 0; k < N; ++k) {
    Eigen::MatrixXd Gk(static_cast<Eigen::Index>(rows.size()), 3);
    for (std::size_t i = 0; i < rows.size(); ++i) Gk.row(static_cast<Eigen::Index>(i)) = G.block(rows[i], 3 * k, 1, 3);
    if (opt.norm == DepthNorm::Frobenius) nu(k) = Gk.squaredNorm();
    else {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Gk.transpose() * Gk, Eigen::EigenvaluesOnly);
      nu(k) = es.eigenvalues()(2);
    }
  }
  const double p = opt.exponent, L2 = opt.limit * opt.limit;
  if (opt.limitOn == DepthLimitOn::Weight) {
    Eigen::VectorXd a = nu.array().pow(p);
    const double amax = a.maxCoeff();
    a = a.cwiseMax(amax / L2);
    alpha = (amax / a.array()).sqrt();
  } else {
    Eigen::VectorXd u = nu.cwiseInverse();
    const double umin = u.minCoeff(), umax = u.maxCoeff();
    double uhat = umax;
    bool found = false;
    for (Eigen::Index k = 0; k < N; ++k) if (u(k) > L2 * umin && (!found || u(k) < uhat)) { uhat = u(k); found = true; }
    for (Eigen::Index k = 0; k < N; ++k) alpha(k) = std::pow(std::min(u(k) / uhat, 1.0), p / 2);
  }
  return alpha;
}

Kernel unconstrainedKernel(const Eigen::MatrixXd& G, const Eigen::MatrixXd& C, const std::vector<std::string>& types, const KernelOptions& opt) {
  if (C.rows() != G.rows()) bad("the gain has " + std::to_string(G.rows()) + " channels and the noise covariance " + std::to_string(C.rows()));
  if (!(opt.snr > 0)) bad("the SNR must be positive");
  Kernel out;
  const Eigen::VectorXd alpha = depthWeights(G, types, opt.depth);
  const Whitener w = whitener(C, types, opt.whitener);
  out.notes = w.notes;
  out.W = w.W;
  out.whitenerRank = w.rank;
  out.depthWeights = alpha;
  const Eigen::Index N = G.cols() / 3;
  Eigen::VectorXd a3(3 * N);
  for (Eigen::Index k = 0; k < N; ++k) a3.segment(3 * k, 3).setConstant(alpha(k));
  const Eigen::MatrixXd L = (w.W * G) * a3.asDiagonal();                    // [rows x 3N]
  const Eigen::MatrixXd LLt = L * L.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (LLt + LLt.transpose()));
  const Eigen::VectorXd s2 = es.eigenvalues().cwiseMax(0.0);
  const double denom = opt.snrNormalisation == SnrNormalisation::AllChannels ? static_cast<double>(G.rows()) : static_cast<double>(std::max(1, w.rank));
  const double mean = s2.sum() / denom;
  if (!(mean > 0)) bad("the whitened gain is zero: no kernel");
  out.lambda = opt.snr * opt.snr / mean;
  Eigen::VectorXd g = (out.lambda * s2.array() + 1.0).inverse();
  const Eigen::MatrixXd V = es.eigenvectors();
  // K = diag(alpha x 1_3) Lambda L^T V diag(g) V^T W
  out.K = a3.asDiagonal() * (out.lambda * (L.transpose() * (V * g.asDiagonal() * V.transpose()) * w.W));
  return out;
}

}  // namespace nxr::meeg
