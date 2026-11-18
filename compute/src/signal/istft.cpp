// istft: scipy.signal.istft's overlap-add.
#include "nxr/signal/istft.h"
#include "nxr/signal/fourier.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::signal {
namespace {
// segments [nperseg x nseg] (complex) -> overlap-add, normalised, boundary cut
Eigen::VectorXcd overlapAdd(const Eigen::MatrixXcd& segs, const Eigen::VectorXd& w, int noverlap, bool boundary, StftScaling scaling, double fs) {
  const long nper = w.size(), nseg = segs.cols();
  const long nov = noverlap < 0 ? nper / 2 : noverlap;
  if (nov >= nper) throw std::invalid_argument("istft: the overlap must be less than the segment length");
  const long step = nper - nov, len = nper + (nseg - 1) * step;
  const double scale = scaling == StftScaling::Spectrum ? w.sum() : std::sqrt(fs * w.squaredNorm());
  Eigen::VectorXcd x = Eigen::VectorXcd::Zero(len);
  Eigen::VectorXd norm = Eigen::VectorXd::Zero(len);
  for (long j = 0; j < nseg; ++j)
    for (long i = 0; i < nper; ++i) { x(j * step + i) += segs(i, j) * scale * w(i); norm(j * step + i) += w(i) * w(i); }
  long a = 0, b = len;
  if (boundary) { a = nper / 2; b = len - nper / 2; }
  if (b <= a) throw std::invalid_argument("istft: the segments are too few for the boundary cut");
  Eigen::VectorXcd y(b - a);
  for (long i = a; i < b; ++i) y(i - a) = x(i) / (norm(i) > 1e-10 ? norm(i) : 1.0);
  return y;
}
}  // namespace

Eigen::VectorXd istft(const Eigen::MatrixXcd& Zxx, const Eigen::VectorXd& window, int noverlap, int nfft, bool boundary, StftScaling scaling, double fs) {
  const long nper = window.size(), bins = Zxx.rows();
  const std::size_t nf = nfft > 0 ? static_cast<std::size_t>(nfft) : static_cast<std::size_t>(2 * (bins - 1));
  if (nper < 1 || nf < static_cast<std::size_t>(nper)) throw std::invalid_argument("istft: nfft must be at least the window's length");
  if (static_cast<std::size_t>(bins) != nf / 2 + 1) throw std::invalid_argument("istft: " + std::to_string(bins) + " one-sided bins do not fit nfft = " + std::to_string(nf));
  const Eigen::MatrixXd t = irfft(Zxx, nf, 0);   // [nfft x nseg]
  return overlapAdd(t.topRows(nper).cast<std::complex<double>>(), window, noverlap, boundary, scaling, fs).real();
}

Eigen::VectorXcd istftTwoSided(const Eigen::MatrixXcd& Zxx, const Eigen::VectorXd& window, int noverlap, bool boundary, StftScaling scaling, double fs) {
  if (window.size() < 1 || Zxx.rows() < window.size()) throw std::invalid_argument("istftTwoSided: the spectra must have at least as many bins as the window has samples");
  const Eigen::MatrixXcd t = ifft(Zxx, 0);
  return overlapAdd(t.topRows(window.size()), window, noverlap, boundary, scaling, fs);
}

}  // namespace nxr::signal
