// nxr::stats sign_flip against numpy and nxr (golden sign_flip_golden.txt, gen_sign_flip_golden.py): NumpyGenerator's raw draws and random() equal numpy's
// default_rng for three seeds bit for bit; signFlipConsistency on synthetic cells with default_rng(0) equals nxr's flip_test (R1, p, family-wise p, the median, its
// null and p, the threshold); a refusal.
#include <cmath>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/stats/sign_flip.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_sign_flip\n";
  std::ifstream f(std::string(NXR_TEST_GOLDEN_DIR) + "/sign_flip_golden.txt");
  std::string line, key;
  std::getline(f, line);
  for (int s = 0; s < 3; ++s) {
    std::uint64_t seed; std::getline(f, line); std::istringstream(line) >> key >> seed;
    nxr::stats::NumpyGenerator raw(seed), draws(seed);
    std::getline(f, line); std::istringstream r(line); r >> key;
    bool same = true;
    for (int i = 0; i < 4; ++i) { std::uint64_t v; r >> v; same = same && raw.nextRaw() == v; }
    std::getline(f, line); std::istringstream d(line); d >> key;
    for (int i = 0; i < 6; ++i) { double v; d >> v; same = same && draws.random() == v; }
    EXPECT(same, "default_rng(" + std::to_string(seed) + "): raw draws and random() equal numpy's bit for bit");
  }
  int cells, elements, perms;
  std::getline(f, line); std::istringstream h(line); h >> key >> cells >> key >> elements >> key >> perms;
  auto row = [&](int n) { std::getline(f, line); std::istringstream in(line); in >> key; std::vector<double> v(static_cast<std::size_t>(n)); for (auto& x : v) in >> x; return v; };
  const std::vector<double> P = row(cells * elements * 3);
  Eigen::MatrixXd values(cells, elements * 3);
  for (int s = 0; s < cells; ++s) for (int j = 0; j < elements * 3; ++j) values(s, j) = P[static_cast<std::size_t>(s * elements * 3 + j)];
  const std::vector<double> R1 = row(elements), p = row(elements), pfwe = row(elements);
  auto scalar = [&]() { std::getline(f, line); std::istringstream in(line); double v; in >> key >> v; return v; };
  const double median = scalar(), null95 = scalar(), medianP = scalar(), threshold = scalar();
  nxr::stats::NumpyGenerator g(0);
  const nxr::stats::SignFlipConsistency t = nxr::stats::signFlipConsistency(values, 3, perms, g);
  double eR = 0, eP = 0;
  for (int e = 0; e < elements; ++e) { eR = std::max(eR, std::abs(t.consistency(e) - R1[static_cast<std::size_t>(e)])); eP = std::max({eP, std::abs(t.p(e) - p[static_cast<std::size_t>(e)]), std::abs(t.familywiseP(e) - pfwe[static_cast<std::size_t>(e)])}); }
  EXPECT(eR < 1e-12, "R1 per element equals nxr's flip_test");
  EXPECT(eP == 0.0, "p and family-wise p equal nxr's flip_test exactly (the same sign patterns)");
  EXPECT(std::abs(t.median - median) < 1e-12 && std::abs(t.medianNull95 - null95) < 1e-12 && t.medianP == medianP && std::abs(t.threshold - threshold) < 1e-12,
         "the median, its null's 95th percentile and p, the family-wise threshold");
  bool refused = false;
  try { nxr::stats::signFlipConsistency(values, 4, perms, g); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "refused: components that do not divide the columns");

  // THE GROUPING RULE on a synthetic series: each element an AR(1) series over the cells (coefficient 0.8), so that neighbouring cells are strongly
  // correlated and cells 8 apart are not: the fewest cells whose group sums decorrelate is found, more than one, and the rule stops at the minimum group count
  {
    const int n = 4000, width = 50;
    Eigen::MatrixXd series(n, width);
    std::mt19937 rng(7);
    std::normal_distribution<double> normal;
    Eigen::RowVectorXd state = Eigen::RowVectorXd::Zero(width);
    for (int s = 0; s < n; ++s) { for (int j = 0; j < width; ++j) state(j) = 0.8 * state(j) + normal(rng); series.row(s) = state; }
    const double one = nxr::stats::groupNeighbourCorrelation(series, 1, 1);
    const nxr::stats::DecorrelatedGrouping rule = nxr::stats::fewestDecorrelatedCells(series, 1, 0.1, 10);
    std::printf("  AR(0.8) cells: neighbour correlation %.3f; the rule: %lld cells a flip (correlation %.3f, reached %d)\n", one, static_cast<long long>(rule.cellsPerFlip), rule.correlation, rule.reached ? 1 : 0);
    EXPECT(one > 0.3 && rule.reached && rule.cellsPerFlip > 1 && rule.correlation < 0.1 && rule.correlations.size() == static_cast<std::size_t>(rule.cellsPerFlip),
           "the grouping rule: neighbouring cells correlated, the fewest cells whose group sums decorrelate found");
    EXPECT(rule.correlations.front() == one, "one cell a group is the neighbouring cells' correlation");
    Eigen::MatrixXd independent(n, width);
    for (int s = 0; s < n; ++s) for (int j = 0; j < width; ++j) independent(s, j) = normal(rng);
    EXPECT(nxr::stats::fewestDecorrelatedCells(independent, 1, 0.1, 10).cellsPerFlip == 1, "independent cells: one cell a flip");
    const nxr::stats::DecorrelatedGrouping never = nxr::stats::fewestDecorrelatedCells(series.topRows(40), 1, -0.5, 10);
    EXPECT(!never.reached && never.cellsPerFlip == 4, "not reached: the largest grouping that keeps the minimum group count, flagged");
  }
  std::cout << (failures ? "FAILED\n" : "all passed\n");
  return failures ? 1 : 0;
}
