// nxr::partition: parity with the Studio's measureOverPartition (golden/partition_golden.txt from gen_partition_golden.mts: unit weights, area
// weights, an empty member), the roll-up law (two members merged by their row weights equal the measure of their union), and the departures from
// the reference, each counted: a negative code is left out, a non-finite value is counted and makes its member's moments NaN, a code past the
// member count is refused.
#include "nxr/partition.h"
#include "golden_io.h"
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace nxr::partition;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

static const std::vector<std::string> kMeasures = {"mean", "mean_square", "max", "min", "abs_max", "sum", "area", "n_elements"};

static void parity() {
  std::cout << "=== parity with the Studio's measureOverPartition (golden/partition_golden.txt) ===\n";
  Golden g(kDir + "/partition_golden.txt");
  EXPECT(g.good(), "partition golden opens");
  if (g.tok() != "cases") { ++fails; return; }
  const int cases = static_cast<int>(g.num());
  for (int c = 0; c < cases; ++c) {
    g.tok(); const int n = static_cast<int>(g.num()), members = static_cast<int>(g.num()); const bool weighted = g.num() > 0.5;
    const Eigen::MatrixXd codes = g.mat("codes"), values = g.mat("values"), weights = g.mat("weights"), want = g.mat("measures"), wantW = g.mat("row_weight");
    std::vector<std::int32_t> k(static_cast<size_t>(n)); std::vector<double> x(static_cast<size_t>(n)), w(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) { k[i] = static_cast<std::int32_t>(codes(0, i)); x[i] = values(0, i); w[i] = weights(0, i); }
    const Sums s = accumulate(k.data(), k.size(), members, x.data(), weighted ? w.data() : nullptr);
    double worst = 0; int mismatched = 0;
    for (int m = 0; m < members; ++m) {
      for (std::size_t j = 0; j < kMeasures.size(); ++j) {
        const double got = measure(s, m, kMeasures[j]), ref = want(m, static_cast<Eigen::Index>(j));
        if (std::isnan(ref) || std::isnan(got)) { if (std::isnan(ref) != std::isnan(got)) ++mismatched; continue; }
        worst = std::max(worst, std::abs(got - ref) / std::max(1e-300, std::abs(ref)));
      }
      if (!std::isnan(wantW(0, m))) worst = std::max(worst, std::abs(s.weight[static_cast<size_t>(m)] - wantW(0, m)) / std::max(1e-300, std::abs(wantW(0, m))));
    }
    EXPECT(mismatched == 0, "case " << c << ": " << mismatched << " measures written by one side only");
    EXPECT(worst < 1e-14, "case " << c << ": max rel err " << worst);
    std::cout << "  case " << c << " (" << n << " elements, " << members << " members, " << (weighted ? "area" : "unit") << " weights): max rel err " << worst << "\n";
  }
}

static void rollup() {
  std::cout << "=== the roll-up law: members merged by their row weights = the measure of their union ===\n";
  const std::vector<std::int32_t> fine = {0, 1, 2, 3, 0, 1, 2, 3, 3, 2}, coarse = {0, 0, 1, 1, 0, 0, 1, 1, 1, 1};   // coarse = fine >> 1
  const std::vector<double> x = {1, 4, -2, 5, 3, 0.5, 7, -1, 2, 6}, w = {0.3, 1.2, 0.7, 0.1, 0.9, 0.4, 0.5, 0.8, 0.2, 0.6};
  const Sums f = accumulate(fine.data(), fine.size(), 4, x.data(), w.data()), c = accumulate(coarse.data(), coarse.size(), 2, x.data(), w.data());
  for (int p = 0; p < 2; ++p) {
    double sw = 0, sm = 0, sq = 0;
    for (int ch = 2 * p; ch < 2 * p + 2; ++ch) { const double wc = f.weight[ch]; sw += wc; sm += measure(f, ch, "mean") * wc; sq += measure(f, ch, "mean_square") * wc; }
    EXPECT(std::abs(sm / sw - measure(c, p, "mean")) < 1e-14, "rolled-up mean of member " << p);
    EXPECT(std::abs(sq / sw - measure(c, p, "mean_square")) < 1e-14, "rolled-up mean_square of member " << p);
    EXPECT(std::abs(sw - c.weight[p]) < 1e-15, "rolled-up weight of member " << p);
  }
  // the raw moments x³ and x⁴ (skewness and kurtosis within a member, derived at read) roll up as the mean does
  for (int p = 0; p < 2; ++p) {
    double sw = 0, s3 = 0, s4 = 0, d3 = 0, d4 = 0, dw = 0;
    for (int ch = 2 * p; ch < 2 * p + 2; ++ch) { const double wc = f.weight[ch]; sw += wc; s3 += measure(f, ch, "mean_cube") * wc; s4 += measure(f, ch, "mean_fourth") * wc; }
    for (std::size_t i = 0; i < x.size(); ++i) if (coarse[i] == p) { d3 += w[i] * x[i] * x[i] * x[i]; d4 += w[i] * x[i] * x[i] * x[i] * x[i]; dw += w[i]; }
    EXPECT(std::abs(s3 / sw - d3 / dw) < 1e-12 * std::abs(d3 / dw) && std::abs(measure(c, p, "mean_cube") - d3 / dw) < 1e-12 * std::abs(d3 / dw), "rolled-up and direct mean_cube of member " << p);
    EXPECT(std::abs(s4 / sw - d4 / dw) < 1e-12 * d4 / dw && std::abs(measure(c, p, "mean_fourth") - d4 / dw) < 1e-12 * d4 / dw, "rolled-up and direct mean_fourth of member " << p);
  }
}

static void departures() {
  std::cout << "=== departures, counted: negative codes left out, non-finite values counted, a code past the count refused ===\n";
  const std::vector<std::int32_t> k = {-1, 0, 0, 1, -1};
  const std::vector<double> x = {100, 1, 3, std::numeric_limits<double>::quiet_NaN(), 100};
  const Sums s = accumulate(k.data(), k.size(), 2, x.data(), nullptr);
  EXPECT(s.unassigned == 2, "two elements with a negative code");
  EXPECT(measure(s, 0, "mean") == 2.0 && s.count[0] == 2, "member 0 is its two elements only");
  EXPECT(s.nonfinite[1] == 1 && std::isnan(measure(s, 1, "mean")), "member 1's NaN is counted and its mean is not written");
  bool refused = false;
  try { const std::vector<std::int32_t> bad = {0, 2}; accumulate(bad.data(), bad.size(), 2, nullptr, nullptr); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "a code past the member count is refused");
  const Sums geo = accumulate(k.data(), k.size(), 2, nullptr, nullptr);
  EXPECT(measure(geo, 0, "n_elements") == 2.0 && measure(geo, 0, "area") == 2.0, "without values: counts and weights only");
}

static void median_within_member() {
  std::cout << "=== the median within a member: numpy's, finite values only ===\n";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // member 0: odd (5 values), member 1: even (4 values), member 2: empty, member 3: only NaN; negative codes and NaNs left out
  const std::vector<std::int32_t> k = {0, 1, 0, -1, 1, 0, 1, 0, 1, 0, 3, 1, 0};
  const std::vector<double> x = {9, 4, 1, 1000, 2, 5, 8, nan, 6, 3, nan, nan, 7};
  const auto m = medians(k.data(), k.size(), 4, x.data());
  // member 0: 9 1 5 3 7 -> 1 3 5 7 9 -> 5;  member 1: 4 2 8 6 -> 2 4 6 8 -> 5
  EXPECT(m.size() == 4 && m[0] == 5.0, "odd count: the middle value, got " << m[0]);
  EXPECT(m[1] == 5.0, "even count: the mean of the two middle values, got " << m[1]);
  EXPECT(std::isnan(m[2]) && std::isnan(m[3]), "an empty member and a member of only NaN are NaN");
  bool refused = false;
  try { const std::vector<std::int32_t> bad = {0, 2}; const std::vector<double> v = {1, 2}; medians(bad.data(), 2, 2, v.data()); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "a code past the member count is refused");
}

static void mean_over_members() {
  std::cout << "=== the mean per member of every column at once: accumulate's mean a column at a time; an empty member NaN ===\n";
  const std::vector<std::int32_t> k = {0, 1, 0, -1, 1, 0};
  const std::vector<double> w = {1.0, 2.0, 3.0, 5.0, 0.5, 2.0};
  const std::vector<float> x = {1, 10, 2, 20, 3, 30, 100, 1000, 5, 50, 7, 70};     // 6 rows x 2 columns
  const auto m = meanOverMembers(x.data(), 6, 2, 2, k.data(), 3, w.data());
  double worst = 0;
  for (int c = 0; c < 2; ++c) {
    std::vector<double> col(6);
    for (int i = 0; i < 6; ++i) col[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i * 2 + c)];
    const Sums s = accumulate(k.data(), k.size(), 3, col.data(), w.data());
    for (int mem = 0; mem < 2; ++mem) worst = std::max(worst, std::abs(m(mem, c) - measure(s, mem, "mean")));
  }
  EXPECT(worst == 0.0, "equal to accumulate's mean, column by column, to the bit (worst " << worst << ")");
  EXPECT(std::isnan(m(2, 0)) && std::isnan(m(2, 1)), "a member of no element is NaN");
  const auto u = meanOverMembers(x.data(), 6, 2, 1, k.data(), 3, nullptr);
  EXPECT(u(0, 0) == (1.0 + 3.0 + 7.0) / 3.0, "no weights: the plain mean over the member's elements, the negative code left out");
  bool refused = false;
  try { const std::vector<std::int32_t> bad = {0, 3}; meanOverMembers(x.data(), 2, 2, 1, bad.data(), 3, nullptr); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "a code past the member count is refused");
}

static void sum_of_products() {
  using Eigen::MatrixXcd;
  std::vector<MatrixXcd> v;
  for (int i = 0; i < 5; ++i) v.push_back(MatrixXcd::Constant(2, 2, std::complex<double>(i + 1.0, -i)));
  const auto r = nxr::partition::sumOverMembers(v, {0, 0, -1, 2, 2}, 4);
  EXPECT(r.count[0] == 2 && r.count[1] == 0 && r.count[2] == 2 && r.count[3] == 0 && r.sum[1].size() == 0 && r.sum[3].size() == 0, "counts, and an empty member has an empty sum");
  EXPECT(r.sum[0](1, 0) == std::complex<double>(3.0, -1.0) && r.sum[2](0, 1) == std::complex<double>(9.0, -7.0), "a member's sum is the sum of its values; member -1 is in none");
  bool a = false, b = false;
  try { nxr::partition::sumOverMembers(v, {0, 0, 0, 0, 7}, 4); } catch (const std::invalid_argument&) { a = true; }
  try { std::vector<MatrixXcd> w = {MatrixXcd::Zero(2, 2), MatrixXcd::Zero(3, 3)}; nxr::partition::sumOverMembers(w, {0, 0}, 1); } catch (const std::invalid_argument&) { b = true; }
  EXPECT(a && b, "a member out of range and a shape mismatch are refused");
}

int main() {
  mean_over_members();
  sum_of_products();
  parity();
  median_within_member();
  rollup();
  departures();
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
