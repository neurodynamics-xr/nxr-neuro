#include "nxr/meeg/crossspectrum.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::meeg {

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
void checkBins(const std::vector<Eigen::MatrixXcd>& C, const char* who) {
  if (C.empty()) throw std::invalid_argument(std::string(who) + ": no bins");
  for (const auto& m : C)
    if (m.rows() != m.cols() || m.rows() != C[0].rows()) throw std::invalid_argument(std::string(who) + ": every bin is a square matrix of the same size");
}
}  // namespace

std::vector<Eigen::MatrixXcd> coherency(const std::vector<Eigen::MatrixXcd>& C) {
  checkBins(C, "coherency");
  std::vector<Eigen::MatrixXcd> out;
  for (const auto& m : C) {
    const Eigen::Index n = m.rows();
    Eigen::MatrixXcd o(n, n);
    for (Eigen::Index a = 0; a < n; ++a)
      for (Eigen::Index b = 0; b < n; ++b) {
        const double d = std::sqrt(m(a, a).real() * m(b, b).real());
        o(a, b) = d > 0 ? m(a, b) / d : std::complex<double>(kNaN, kNaN);
      }
    out.push_back(std::move(o));
  }
  return out;
}

PhaseSlope phaseSlope(const std::vector<Eigen::MatrixXcd>& C, const Eigen::VectorXd& f, const PhaseSlopeOptions& opt) {
  checkBins(C, "phaseSlope");
  const std::size_t F = C.size();
  if (static_cast<std::size_t>(f.size()) != F) throw std::invalid_argument("phaseSlope: one frequency per bin");
  const std::vector<Eigen::MatrixXcd> H = coherency(C);
  const Eigen::Index n = C[0].rows();
  PhaseSlope out;
  out.delayMs = Eigen::MatrixXd::Constant(n, n, kNaN);
  out.residual = Eigen::MatrixXd::Constant(n, n, kNaN);
  if (F < 2) return out;
  std::vector<double> phi(F), w(F);
  for (Eigen::Index a = 0; a < n; ++a)
    for (Eigen::Index b = 0; b < n; ++b) {
      bool ok = true;
      for (std::size_t k = 0; k < F; ++k) {
        const std::complex<double> h = H[k](a, b);
        if (!std::isfinite(h.real()) || !std::isfinite(h.imag())) { ok = false; break; }
        double p = std::arg(h);
        if (k > 0) {                                                 // adjacent unwrap: within +-pi of the previous bin
          p = phi[k - 1] + std::remainder(p - phi[k - 1], 2 * M_PI);
        }
        phi[k] = p;
        w[k] = opt.weights == PhaseSlopeOptions::Weights::Coherence ? std::norm(h) : 1.0;
      }
      if (!ok) continue;
      double sw = 0, sf = 0, sp = 0;
      for (std::size_t k = 0; k < F; ++k) { sw += w[k]; sf += w[k] * f[k]; sp += w[k] * phi[k]; }
      if (!(sw > 0)) continue;
      const double mf = sf / sw, mp = sp / sw;
      double sxx = 0, sxy = 0;
      for (std::size_t k = 0; k < F; ++k) { sxx += w[k] * (f[k] - mf) * (f[k] - mf); sxy += w[k] * (f[k] - mf) * (phi[k] - mp); }
      if (!(sxx > 0)) continue;
      const double slope = sxy / sxx;                                // rad per Hz
      double rss = 0;
      for (std::size_t k = 0; k < F; ++k) { const double r = phi[k] - mp - slope * (f[k] - mf); rss += w[k] * r * r; }
      out.delayMs(a, b) = slope / (2 * M_PI) * 1000.0;
      out.residual(a, b) = std::sqrt(rss / sw);
    }
  return out;
}

Eigen::MatrixXd phaseSlopeIndex(const std::vector<Eigen::MatrixXcd>& C) {
  checkBins(C, "phaseSlopeIndex");
  const std::vector<Eigen::MatrixXcd> H = coherency(C);
  const Eigen::Index n = C[0].rows();
  Eigen::MatrixXd psi = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t k = 0; k + 1 < H.size(); ++k)
    for (Eigen::Index a = 0; a < n; ++a)
      for (Eigen::Index b = 0; b < n; ++b) psi(a, b) += (std::conj(H[k](a, b)) * H[k + 1](a, b)).imag();
  return psi;
}

}  // namespace nxr::meeg
