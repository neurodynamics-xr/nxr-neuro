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
  if (kinds == ModeKinds::exact_and_constant) out.modes.push_back(-1);                   // the constant first (mu 0): the field's coordinates are the constant and the exact modes
  for (Eigen::Index k = 0; k < P.cols(); ++k) {
    const int t = kind[static_cast<std::size_t>(k)];
    if (kinds == ModeKinds::all || ((kinds == ModeKinds::exact || kinds == ModeKinds::exact_and_constant) && t == 0) || (kinds == ModeKinds::coexact && t == 2)) out.modes.push_back(k);
  }
  if (out.modes.empty()) bad("modeAnalysis: the selection keeps no mode");
  const Eigen::Index V = P.rows(), K = static_cast<Eigen::Index>(out.modes.size());
  out.synthesis.resize(V, K);
  for (Eigen::Index i = 0; i < K; ++i) {
    if (out.modes[static_cast<std::size_t>(i)] < 0) {                                     // the constant at unit ⋆0 norm, whatever the normalisation (the package stores no constant)
      out.synthesis.col(i) = Eigen::VectorXd::Constant(V, 1.0 / std::sqrt(mass.sum()));
      continue;
    }
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







}  // namespace nxr::signal
