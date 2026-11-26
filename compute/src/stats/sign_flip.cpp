// nxr::stats — the sign-flip test of a direction's consistency and numpy's default_rng (see the header).
#include "nxr/stats/sign_flip.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nxr::stats {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }

// numpy's SeedSequence (numpy/random/bit_generator.pyx): the pool of 4 words mixed from the entropy, then generate_state
constexpr std::uint32_t kInitA = 0x43b0d7e5u, kMultA = 0x931e8875u, kInitB = 0x8b51f9ddu, kMultB = 0x58f38dedu, kMixMultL = 0xca01f9ddu, kMixMultR = 0x4973f715u;
constexpr int kShift = 16, kPool = 4;
std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hashConst) {
  value ^= hashConst;
  hashConst *= kMultA;
  value *= hashConst;
  value ^= value >> kShift;
  return value;
}
std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
  std::uint32_t r = kMixMultL * x - kMixMultR * y;
  r ^= r >> kShift;
  return r;
}
unsigned __int128 joined(std::uint64_t high, std::uint64_t low) { return (static_cast<unsigned __int128>(high) << 64) | low; }
const unsigned __int128 kMultiplier = joined(2549297995355413924ULL, 4865540595714422341ULL);

double quantileLinear(std::vector<double> v, double q) {          // numpy.percentile, method linear
  std::sort(v.begin(), v.end());
  const double at = q * static_cast<double>(v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(at));
  const std::size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (at - static_cast<double>(lo)) * (v[hi] - v[lo]);
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace

NumpyGenerator::NumpyGenerator(std::uint64_t seed) {
  std::vector<std::uint32_t> entropy{static_cast<std::uint32_t>(seed & 0xffffffffu)};
  if (seed >> 32) entropy.push_back(static_cast<std::uint32_t>(seed >> 32));
  std::uint32_t pool[kPool];
  std::uint32_t hashConst = kInitA;
  for (int i = 0; i < kPool; ++i) pool[i] = hashmix(i < static_cast<int>(entropy.size()) ? entropy[static_cast<std::size_t>(i)] : 0u, hashConst);
  for (int src = 0; src < kPool; ++src)
    for (int dst = 0; dst < kPool; ++dst)
      if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hashConst));
  for (std::size_t src = kPool; src < entropy.size(); ++src)
    for (int dst = 0; dst < kPool; ++dst) pool[dst] = mix(pool[dst], hashmix(entropy[src], hashConst));
  std::uint32_t words[8];
  std::uint32_t hashB = kInitB;
  for (int i = 0; i < 8; ++i) {
    std::uint32_t v = pool[i % kPool];
    v ^= hashB;
    hashB *= kMultB;
    v *= hashB;
    v ^= v >> kShift;
    words[i] = v;
  }
  std::uint64_t value[4];
  for (int i = 0; i < 4; ++i) value[i] = static_cast<std::uint64_t>(words[2 * i]) | (static_cast<std::uint64_t>(words[2 * i + 1]) << 32);   // the uint32 words viewed as uint64 (little-endian)
  const unsigned __int128 initialState = joined(value[0], value[1]), sequence = joined(value[2], value[3]);
  state_ = 0;
  increment_ = (sequence << 1) | 1u;                                    // pcg64_srandom_r
  state_ = state_ * kMultiplier + increment_;
  state_ += initialState;
  state_ = state_ * kMultiplier + increment_;
}

std::uint64_t NumpyGenerator::nextRaw() {
  state_ = state_ * kMultiplier + increment_;                           // step, then the XSL-RR output of the new state
  const std::uint64_t x = static_cast<std::uint64_t>(state_ >> 64) ^ static_cast<std::uint64_t>(state_);
  const unsigned rot = static_cast<unsigned>(state_ >> 122);
  return (x >> rot) | (x << ((64u - rot) & 63u));
}

std::uint32_t NumpyGenerator::nextUint32() {
  if (hasUint32_) { hasUint32_ = false; return uinteger_; }
  const std::uint64_t next = nextRaw();
  hasUint32_ = true;
  uinteger_ = static_cast<std::uint32_t>(next >> 32);
  return static_cast<std::uint32_t>(next & 0xffffffffu);
}

double NumpyGenerator::random() { return static_cast<double>(nextRaw() >> 11) * (1.0 / 9007199254740992.0); }

SignFlipConsistency signFlipConsistency(const Eigen::MatrixXd& values, int components, std::int64_t patterns, NumpyGenerator& generator, std::int64_t cellsPerFlip) {
  const Eigen::Index cells = values.rows();
  if (cells < 1) bad("signFlipConsistency: no cell");
  if (components < 1 || values.cols() % components != 0) bad("signFlipConsistency: the columns are not elements x components");
  if (patterns < 1 || cellsPerFlip < 1) bad("signFlipConsistency: at least one pattern and one cell per flip");
  const Eigen::Index elements = values.cols() / components;
  Eigen::MatrixXd unit = values;
  for (Eigen::Index s = 0; s < cells; ++s)
    for (Eigen::Index e = 0; e < elements; ++e) {
      const double length = values.row(s).segment(e * components, components).norm();
      unit.row(s).segment(e * components, components) /= std::max(length, 1e-30);
    }
  auto resultant = [&](const Eigen::RowVectorXd& sum) {
    Eigen::VectorXd r(elements);
    for (Eigen::Index e = 0; e < elements; ++e) r(e) = sum.segment(e * components, components).norm() / static_cast<double>(cells);
    return r;
  };
  SignFlipConsistency out;
  out.cells = cells; out.patterns = patterns;
  out.groups = (cells + cellsPerFlip - 1) / cellsPerFlip;
  out.consistency = resultant(unit.colwise().sum());
  // the groups' sums, so a pattern is one product over the groups
  Eigen::MatrixXd groupSums = Eigen::MatrixXd::Zero(out.groups, unit.cols());
  for (Eigen::Index s = 0; s < cells; ++s) groupSums.row(s / cellsPerFlip) += unit.row(s);
  Eigen::VectorXd exceed = Eigen::VectorXd::Zero(elements);
  std::vector<double> maxima, medians;
  maxima.reserve(static_cast<std::size_t>(patterns)); medians.reserve(static_cast<std::size_t>(patterns));
  Eigen::RowVectorXd signs(out.groups);
  for (std::int64_t k = 0; k < patterns; ++k) {
    for (Eigen::Index g = 0; g < out.groups; ++g) signs(g) = generator.random() < 0.5 ? -1.0 : 1.0;
    const Eigen::VectorXd r = resultant(signs * groupSums);
    for (Eigen::Index e = 0; e < elements; ++e) if (r(e) >= out.consistency(e)) exceed(e) += 1;
    maxima.push_back(r.maxCoeff());
    medians.push_back(median(std::vector<double>(r.data(), r.data() + r.size())));
  }
  const double denominator = static_cast<double>(patterns + 1);
  out.p = (exceed.array() + 1.0) / denominator;
  out.familywiseP.resize(elements);
  for (Eigen::Index e = 0; e < elements; ++e) {
    double n = 0;
    for (const double m : maxima) if (m >= out.consistency(e)) n += 1;
    out.familywiseP(e) = (n + 1.0) / denominator;
  }
  out.threshold = quantileLinear(maxima, 0.95);
  out.median = median(std::vector<double>(out.consistency.data(), out.consistency.data() + elements));
  out.medianNull95 = quantileLinear(medians, 0.95);
  double n = 0;
  for (const double m : medians) if (m >= out.median) n += 1;
  out.medianP = (n + 1.0) / denominator;
  if (cells > 1) {
    double sum = 0;
    for (Eigen::Index s = 0; s + 1 < cells; ++s) sum += unit.row(s).dot(unit.row(s + 1)) / static_cast<double>(elements);
    out.neighbourCorrelation = sum / static_cast<double>(cells - 1);
  }
  return out;
}

}  // namespace nxr::stats
