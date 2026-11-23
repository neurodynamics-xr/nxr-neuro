// nxr::signal — the mode-space Fourier coefficients and their readings (see the header).
#include "nxr/signal/mode_spectrum.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "nxr/signal/fourier.h"

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }
}  // namespace

ModeAnalysis modeAnalysis(const Eigen::MatrixXd& P, const Eigen::VectorXd& mass, const std::vector<int>& kind, ModeKinds kinds, ModeNormalisation normalisation) {
  if (P.rows() != mass.size()) bad("modeAnalysis: " + std::to_string(P.rows()) + " vertices in the 0-form parts, " + std::to_string(mass.size()) + " masses");
  if (static_cast<Eigen::Index>(kind.size()) != P.cols()) bad("modeAnalysis: " + std::to_string(P.cols()) + " modes, " + std::to_string(kind.size()) + " kinds");
  ModeAnalysis out;
  for (Eigen::Index k = 0; k < P.cols(); ++k) {
    const int t = kind[static_cast<std::size_t>(k)];
    if (kinds == ModeKinds::all || (kinds == ModeKinds::exact && t == 0) || (kinds == ModeKinds::coexact && t == 2)) out.modes.push_back(k);
  }
  if (out.modes.empty()) bad("modeAnalysis: the selection keeps no mode");
  const Eigen::Index V = P.rows(), K = static_cast<Eigen::Index>(out.modes.size());
  out.synthesis.resize(V, K);
  for (Eigen::Index i = 0; i < K; ++i) {
    Eigen::VectorXd col = P.col(out.modes[static_cast<std::size_t>(i)]);
    if (normalisation == ModeNormalisation::unit_mass) {
      const double n2 = col.cwiseProduct(col).dot(mass);
      if (!(n2 > 0)) bad("modeAnalysis: mode " + std::to_string(out.modes[static_cast<std::size_t>(i)]) + " has no ⋆0 norm");
      col /= std::sqrt(n2);
    }
    out.synthesis.col(i) = col;
  }
  out.rows = (out.synthesis.array().colwise() * mass.array()).matrix().transpose();
  return out;
}

Eigen::MatrixXd modeAnalysisOfComponents(const Eigen::MatrixXd& rows, const Eigen::MatrixXd& W, int components) {
  if (components < 1) bad("modeAnalysisOfComponents: components >= 1");
  if (W.rows() != components * rows.cols()) bad("modeAnalysisOfComponents: the source operator has " + std::to_string(W.rows()) + " rows, not " + std::to_string(components) + " x " + std::to_string(rows.cols()) + " vertices");
  const Eigen::Index K = rows.rows(), V = rows.cols(), C = W.cols();
  Eigen::MatrixXd out(components * K, C);
  Eigen::MatrixXd Wc(V, C);
  for (int c = 0; c < components; ++c) {
    for (Eigen::Index v = 0; v < V; ++v) Wc.row(v) = W.row(components * v + c);
    out.middleRows(c * K, K).noalias() = rows * Wc;
  }
  return out;
}

Eigen::MatrixXcd spectralCoefficients(const Eigen::MatrixXd& op, const Eigen::MatrixXd& tile, std::int64_t firstBin, std::int64_t binCount) {
  if (op.cols() != tile.rows()) bad("spectralCoefficients: the operator reads " + std::to_string(op.cols()) + " channels, the tile has " + std::to_string(tile.rows()));
  const std::int64_t bins = static_cast<std::int64_t>(rfftBins(static_cast<std::size_t>(tile.cols())));
  if (firstBin < 0 || binCount < 1 || firstBin + binCount > bins) bad("spectralCoefficients: bins [" + std::to_string(firstBin) + ", " + std::to_string(firstBin + binCount) + ") outside the tile's " + std::to_string(bins) + " bins");
  const Eigen::MatrixXcd X = rfft(tile, 1);                                  // channels x bins
  return op.cast<std::complex<double>>() * X.middleCols(firstBin, binCount);
}

Eigen::MatrixXcd modeCrossSpectrum(const std::vector<Eigen::MatrixXcd>& tiles, const Eigen::VectorXd& w) {
  if (tiles.empty()) bad("modeCrossSpectrum: no tile");
  const Eigen::Index R = tiles.front().rows();
  Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(R, R);
  for (const auto& A : tiles) {
    if (A.rows() != R || A.cols() != w.size()) bad("modeCrossSpectrum: a tile is " + std::to_string(A.rows()) + " x " + std::to_string(A.cols()) + ", not " + std::to_string(R) + " x " + std::to_string(w.size()));
    const Eigen::MatrixXcd Aw = A * w.cast<std::complex<double>>().asDiagonal();
    T.noalias() += Aw * A.adjoint();
  }
  return T;
}

Eigen::MatrixXcd contractComponents(const Eigen::MatrixXcd& T, int components) {
  if (components < 1 || T.rows() != T.cols() || T.rows() % components != 0) bad("contractComponents: a square cross-spectrum of components x K rows");
  const Eigen::Index K = T.rows() / components;
  Eigen::MatrixXcd S = Eigen::MatrixXcd::Zero(K, K);
  for (int c = 0; c < components; ++c) S += T.block(c * K, c * K, K, K);
  return S;
}

FlowMap flowMap(const IntrinsicComplex& c, const Eigen::MatrixXd& Phi, const Eigen::MatrixXcd& S, const Eigen::VectorXd& h, std::int64_t blockVertices) {
  const Eigen::Index V = Phi.rows(), K = Phi.cols();
  if (V != c.nV) bad("flowMap: the synthesis has " + std::to_string(V) + " vertices, the complex " + std::to_string(c.nV));
  if (S.rows() != K || S.cols() != K || h.size() != K) bad("flowMap: the cross-spectrum and the weights are not over the synthesis' " + std::to_string(K) + " modes");
  const Eigen::MatrixXd Sh = h.asDiagonal() * S.real() * h.asDiagonal(), Ah = h.asDiagonal() * S.imag() * h.asDiagonal();
  FlowMap out;
  out.power.resize(V);
  out.phaseCurrent.setZero(c.nE);
  // the edges by their tail's block, so each block of Φ̂ Im S is made once
  const std::int64_t B = std::max<std::int64_t>(1, blockVertices);
  std::vector<std::vector<int>> byBlock(static_cast<std::size_t>((V + B - 1) / B));
  for (int e = 0; e < c.nE; ++e) byBlock[static_cast<std::size_t>(c.edges[2 * e] / B)].push_back(e);
  for (Eigen::Index v0 = 0; v0 < V; v0 += B) {
    const Eigen::Index n = std::min<Eigen::Index>(B, V - v0);
    const Eigen::MatrixXd Pb = Phi.middleRows(v0, n);
    out.power.segment(v0, n) = (Pb * Sh).cwiseProduct(Pb).rowwise().sum();
    const Eigen::MatrixXd U = Pb * Ah;                                        // [n x K]
    for (const int e : byBlock[static_cast<std::size_t>(v0 / B)]) out.phaseCurrent(e) = -U.row(c.edges[2 * e] - v0).dot(Phi.row(c.edges[2 * e + 1]));
  }
  return out;
}

CellMeasures cellMeasures(const Eigen::MatrixXcd& S, const Eigen::MatrixXd& H) {
  const Eigen::Index K = S.rows();
  if (S.cols() != K || H.cols() != K) bad("cellMeasures: the weights are not over the cross-spectrum's " + std::to_string(K) + " modes");
  const Eigen::VectorXd d = S.diagonal().real();
  Eigen::MatrixXd re2 = S.real().cwiseAbs2(), im2 = S.imag().cwiseAbs2(), noise = d * d.transpose() - re2;
  re2.diagonal().setZero(); im2.diagonal().setZero(); noise.diagonal().setZero();
  CellMeasures out;
  out.power = H * d;
  out.effectiveModes = H.rowwise().sum().cwiseAbs2().cwiseQuotient(H.cwiseAbs2().rowwise().sum());
  out.inPhaseEnergy = H * re2 * H.transpose();
  out.quadratureEnergy = H * im2 * H.transpose();
  out.quadratureNoise = H * noise * H.transpose();
  return out;
}

double effectiveDegreesOfFreedom(const Eigen::VectorXd& w, std::int64_t tiles, const Eigen::VectorXd& rho) {
  if (tiles < 1 || w.size() == 0) bad("effectiveDegreesOfFreedom: at least one tile and one bin");
  if (rho.size() > 0 && std::abs(rho(0) - 1.0) > 1e-12) bad("effectiveDegreesOfFreedom: the correlation at lag 0 is 1");
  double q = 0;
  for (Eigen::Index i = 0; i < w.size(); ++i)
    for (Eigen::Index j = 0; j < w.size(); ++j) {
      const Eigen::Index lag = std::abs(i - j);
      const double r = rho.size() == 0 ? (lag == 0 ? 1.0 : 0.0) : (lag < rho.size() ? rho(lag) : 0.0);
      q += w(i) * w(j) * r;
    }
  const double s = w.sum();
  return static_cast<double>(tiles) * s * s / q;
}

}  // namespace nxr::signal
