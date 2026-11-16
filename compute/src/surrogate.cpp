// nxr::surrogate (surrogate.h).
#include "nxr/surrogate.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace nxr::surrogate {

namespace {
std::uint64_t splitmix(std::uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
double uniform01(std::uint64_t bits) { return (static_cast<double>(bits >> 11) + 0.5) * (1.0 / 9007199254740992.0); }   // (0, 1)
}  // namespace

double normalAt(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) {
  const std::uint64_t h = splitmix(splitmix(splitmix(seed) ^ (stream + 0x632be59bd9b4e019ULL)) ^ index);
  const double u1 = uniform01(h), u2 = uniform01(splitmix(h ^ 0xa0761d6478bd642fULL));
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

Eigen::MatrixXd whiteBlock(std::uint64_t seed, int rows, std::int64_t first, std::int64_t n) {
  if (rows < 1 || n < 0) throw std::invalid_argument("whiteBlock: rows >= 1 and n >= 0");
  Eigen::MatrixXd w(rows, n);
  for (int r = 0; r < rows; ++r)
    for (std::int64_t j = 0; j < n; ++j) w(r, j) = normalAt(seed, static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(first + j));   // a negative sample wraps: still one fixed value
  return w;
}

Eigen::VectorXd shapingTaps(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, double fs, int half) {
  if (freq.size() < 2 || freq.size() != psd.size()) throw std::invalid_argument("shapingTaps: freq and psd are of one length >= 2");
  if (!(fs > 0) || half < 1) throw std::invalid_argument("shapingTaps: fs > 0 and half >= 1");
  for (Eigen::Index i = 1; i < freq.size(); ++i) if (!(freq[i] > freq[i - 1])) throw std::invalid_argument("shapingTaps: freq increases");
  if (!(psd.maxCoeff() > 0)) throw std::invalid_argument("shapingTaps: the spectrum holds no power");
  // the amplitude on a fine grid up to Nyquist (linear in the power between the stored bins, zero outside them), and the taps as its cosine transform
  const double nyq = fs / 2, df = std::min(freq[1] - freq[0], nyq / 512.0) / 2;
  const std::int64_t nf = static_cast<std::int64_t>(std::floor(nyq / df)) + 1;
  Eigen::VectorXd a(nf);
  Eigen::Index j = 0;
  for (std::int64_t i = 0; i < nf; ++i) {
    const double f = static_cast<double>(i) * df;
    if (f < freq[0] || f > freq[freq.size() - 1]) { a[i] = 0; continue; }
    while (j + 2 < freq.size() && freq[j + 1] < f) ++j;
    const double t = (f - freq[j]) / (freq[j + 1] - freq[j]);
    a[i] = std::sqrt(std::max((1 - t) * psd[j] + t * psd[j + 1], 0.0));
  }
  Eigen::VectorXd h(2 * half + 1);
  for (int k = 0; k <= half; ++k) {
    double s = 0;
    for (std::int64_t i = 0; i < nf; ++i) s += a[i] * std::cos(6.283185307179586 * static_cast<double>(i) * df * k / fs) * (i == 0 || i == nf - 1 ? 0.5 : 1.0);
    const double w = 0.5 + 0.5 * std::cos(3.141592653589793 * k / (half + 1));
    h[half + k] = h[half - k] = s * w;
  }
  const double e = std::sqrt(h.squaredNorm());
  if (!(e > 0)) throw std::invalid_argument("shapingTaps: the taps carry no energy");
  return h / e;
}

Eigen::MatrixXd filterValid(const Eigen::MatrixXd& white, const Eigen::VectorXd& taps) {
  const Eigen::Index L = taps.size();
  if (L % 2 == 0 || white.cols() < L) throw std::invalid_argument("filterValid: odd taps and at least as many columns");
  const Eigen::Index n = white.cols() - L + 1;
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(white.rows(), n);
  for (Eigen::Index k = 0; k < L; ++k) y += taps[k] * white.middleCols(k, n);
  return y;
}

Eigen::MatrixXd sqrtPsd(const Eigen::MatrixXd& A) {
  if (A.rows() != A.cols()) throw std::invalid_argument("sqrtPsd: a square matrix");
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (A + A.transpose()));
  const Eigen::VectorXd e = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  return es.eigenvectors() * e.asDiagonal() * es.eigenvectors().transpose();
}

Eigen::MatrixXd lagfreeBlock(const Eigen::MatrixXd& Rh, const Eigen::VectorXd& taps, std::uint64_t seed, std::int64_t first, std::int64_t n, const Eigen::VectorXd& scale) {
  if (Rh.rows() != Rh.cols()) throw std::invalid_argument("lagfreeBlock: Rh is square");
  if (scale.size() != 0 && scale.size() != n) throw std::invalid_argument("lagfreeBlock: scale has one value a sample");
  if (taps.size() % 2 == 0) throw std::invalid_argument("lagfreeBlock: odd taps");
  const std::int64_t half = (taps.size() - 1) / 2;
  const Eigen::MatrixXd w = whiteBlock(seed, static_cast<int>(Rh.rows()), first - half, n + 2 * half);
  Eigen::MatrixXd x = Rh * filterValid(w, taps);
  if (scale.size() != 0) x = x * scale.asDiagonal();
  return x;
}

Eigen::VectorXd interpolateSeries(const std::vector<double>& v, std::int64_t stride, std::int64_t first, std::int64_t n) {
  if (v.empty() || stride < 1) throw std::invalid_argument("interpolateSeries: values and a stride >= 1");
  Eigen::VectorXd y(n);
  const double last = static_cast<double>(v.size() - 1);
  for (std::int64_t j = 0; j < n; ++j) {
    const double p = std::min(std::max(static_cast<double>(first + j) / static_cast<double>(stride), 0.0), last);
    const std::size_t i = static_cast<std::size_t>(std::floor(p)), i1 = std::min(i + 1, v.size() - 1);
    const double t = p - static_cast<double>(i);
    y[j] = (1 - t) * v[i] + t * v[i1];
  }
  return y;
}

double effectiveBandwidth(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, const Eigen::VectorXd& gain) {
  if (freq.size() < 2 || freq.size() != psd.size() || (gain.size() != 0 && gain.size() != psd.size())) throw std::invalid_argument("effectiveBandwidth: freq, psd (and gain) of one length >= 2");
  double s1 = 0, s2 = 0;
  for (Eigen::Index i = 0; i < psd.size(); ++i) { const double p = psd[i] * (gain.size() ? gain[i] * gain[i] : 1.0); s1 += p; s2 += p * p; }
  if (!(s2 > 0)) throw std::invalid_argument("effectiveBandwidth: the spectrum holds no power");
  const double df = (freq[freq.size() - 1] - freq[0]) / static_cast<double>(freq.size() - 1);
  return s1 * s1 * df / s2;
}

Eigen::MatrixXcd wishartDraw(const Eigen::MatrixXd& ReQ0, double dof, std::uint64_t seed, std::uint64_t draw) {
  if (ReQ0.rows() != ReQ0.cols()) throw std::invalid_argument("wishartDraw: a square matrix");
  const int k = std::max(static_cast<int>(std::lround(dof)), 2), C = static_cast<int>(ReQ0.rows());
  const Eigen::MatrixXd L = sqrtPsd(ReQ0);
  Eigen::MatrixXcd Z(C, k);
  const std::uint64_t s = seed ^ (0x517cc1b727220a95ULL * (draw + 1));
  for (int r = 0; r < C; ++r)
    for (int j = 0; j < k; ++j)
      Z(r, j) = std::complex<double>(normalAt(s, 2 * static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(j)), normalAt(s, 2 * static_cast<std::uint64_t>(r) + 1, static_cast<std::uint64_t>(j))) / std::sqrt(2.0);
  const Eigen::MatrixXcd X = L.cast<std::complex<double>>() * Z;
  return X * X.adjoint() / static_cast<double>(k);
}

double besagCliffordP(std::int64_t exceed, std::int64_t n, std::int64_t h) {
  if (n < 1 || exceed < 0 || exceed > n || h < 1) throw std::invalid_argument("besagCliffordP: n >= 1, 0 <= exceed <= n, h >= 1");
  return exceed >= h ? static_cast<double>(h) / static_cast<double>(n) : static_cast<double>(exceed + 1) / static_cast<double>(n + 1);
}

Eigen::VectorXd benjaminiHochberg(const Eigen::VectorXd& p) {
  std::vector<Eigen::Index> idx;
  for (Eigen::Index i = 0; i < p.size(); ++i) if (std::isfinite(p[i])) idx.push_back(i);
  std::sort(idx.begin(), idx.end(), [&](Eigen::Index a, Eigen::Index b) { return p[a] < p[b]; });
  Eigen::VectorXd q = Eigen::VectorXd::Constant(p.size(), std::numeric_limits<double>::quiet_NaN());
  const double m = static_cast<double>(idx.size());
  double run = 1.0;
  for (std::size_t r = idx.size(); r-- > 0;) { run = std::min(run, p[idx[r]] * m / static_cast<double>(r + 1)); q[idx[r]] = std::min(run, 1.0); }
  return q;
}

}  // namespace nxr::surrogate
