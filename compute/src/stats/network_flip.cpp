#include "nxr/stats/network_flip.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace nxr::stats {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }
// numpy random_bounded_uint64(state, 0, upper, 0, use_masked = false) for upper < 2^32: Lemire's method on 32-bit draws (buffered_bounded_lemire_uint32)
std::uint64_t boundedUint(NumpyGenerator& g, std::uint64_t upper) {
  if (upper == 0) return 0;
  if (upper > 0xffffffffULL) bad("numpyChoice: a bound above 2^32 - 1");
  if (upper == 0xffffffffULL) return g.nextUint32();
  const std::uint32_t excl = static_cast<std::uint32_t>(upper) + 1u;
  std::uint64_t m = static_cast<std::uint64_t>(g.nextUint32()) * excl;
  std::uint32_t leftover = static_cast<std::uint32_t>(m & 0xffffffffULL);
  if (leftover < excl) {
    const std::uint32_t threshold = static_cast<std::uint32_t>((0xffffffffULL - upper) % excl);
    while (leftover < threshold) { m = static_cast<std::uint64_t>(g.nextUint32()) * excl; leftover = static_cast<std::uint32_t>(m & 0xffffffffULL); }
  }
  return m >> 32;
}
}  // namespace

NetworkSignFlip networkSignFlip(const Eigen::MatrixXd& F, std::int64_t patterns, NumpyGenerator& generator) {
  const Eigen::Index S = F.rows(), E = F.cols();
  if (S < 1) bad("networkSignFlip: no segment");
  if (patterns < 1) bad("networkSignFlip: patterns >= 1");
  NetworkSignFlip out;
  out.segments = S; out.patterns = patterns;
  out.sum = F.colwise().sum().transpose();
  const Eigen::VectorXd scale = F.cwiseAbs().colwise().sum().transpose().cwiseMax(1e-30);
  const Eigen::VectorXd observedNormalised = out.sum.cwiseAbs().cwiseQuotient(scale);
  Eigen::VectorXd exceed = Eigen::VectorXd::Zero(E), familyExceed = Eigen::VectorXd::Zero(E);
  Eigen::RowVectorXd signs(S);
  for (std::int64_t q = 0; q < patterns; ++q) {
    for (Eigen::Index s = 0; s < S; ++s) signs(s) = generator.random() < 0.5 ? -1.0 : 1.0;
    const Eigen::RowVectorXd null = signs * F;
    double maximum = -1;
    for (Eigen::Index e = 0; e < E; ++e) {
      exceed(e) += std::abs(null(e)) >= std::abs(out.sum(e));
      maximum = std::max(maximum, std::abs(null(e) / scale(e)));
    }
    for (Eigen::Index e = 0; e < E; ++e) familyExceed(e) += maximum >= observedNormalised(e);
  }
  out.p = (exceed.array() + 1.0) / static_cast<double>(patterns + 1);
  out.familywiseP = (familyExceed.array() + 1.0) / static_cast<double>(patterns + 1);
  return out;
}

std::vector<std::int64_t> numpyChoice(std::int64_t population, std::int64_t size, NumpyGenerator& generator) {
  if (population < 0 || size < 0 || size > population) bad("numpyChoice: 0 <= size <= population");
  if (population > 10000) bad("numpyChoice: a population above 10 000 (numpy shuffles a tail there; not ported)");
  std::vector<std::int64_t> idx(static_cast<std::size_t>(size));
  std::unordered_set<std::int64_t> chosen;                                   // Floyd's algorithm: the hash set of numpy holds the same members
  for (std::int64_t j = population - size; j < population; ++j) {
    const std::int64_t value = static_cast<std::int64_t>(boundedUint(generator, static_cast<std::uint64_t>(j)));
    const std::int64_t pick = chosen.count(value) ? j : value;
    chosen.insert(pick);
    idx[static_cast<std::size_t>(j - population + size)] = pick;
  }
  for (std::int64_t i = size - 1; i >= 1; --i) {                            // _shuffle_int(bitgen, size, 1, idx)
    const std::int64_t j = static_cast<std::int64_t>(boundedUint(generator, static_cast<std::uint64_t>(i)));
    std::swap(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
  }
  return idx;
}

std::vector<std::int64_t> digitize(const Eigen::VectorXd& x, const std::vector<double>& edges) {
  for (std::size_t i = 1; i < edges.size(); ++i) if (!(edges[i] > edges[i - 1])) bad("digitize: the edges increase");
  std::vector<std::int64_t> out(static_cast<std::size_t>(x.size()));
  for (Eigen::Index i = 0; i < x.size(); ++i) out[static_cast<std::size_t>(i)] = std::upper_bound(edges.begin(), edges.end(), x(i)) - edges.begin();
  return out;
}

std::vector<std::uint8_t> distanceMatchedControl(const std::vector<std::uint8_t>& joined, const std::vector<std::int64_t>& bin, NumpyGenerator& generator) {
  if (joined.size() != bin.size()) bad("distanceMatchedControl: one bin per element");
  std::vector<std::uint8_t> control(joined.size(), 0);
  std::set<std::int64_t> occupied;
  for (std::size_t e = 0; e < joined.size(); ++e) if (joined[e]) occupied.insert(bin[e]);
  for (const std::int64_t b : occupied) {
    std::vector<std::int64_t> candidates;
    std::int64_t need = 0;
    for (std::size_t e = 0; e < joined.size(); ++e) {
      if (bin[e] != b) continue;
      if (joined[e]) ++need; else candidates.push_back(static_cast<std::int64_t>(e));
    }
    if (candidates.empty()) continue;
    const std::int64_t n = static_cast<std::int64_t>(candidates.size());
    for (const std::int64_t i : numpyChoice(n, std::min(n, need), generator)) control[static_cast<std::size_t>(candidates[static_cast<std::size_t>(i)])] = 1;
  }
  return control;
}

}  // namespace nxr::stats
