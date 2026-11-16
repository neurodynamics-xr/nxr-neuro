// nxr::stats against its references (golden: test/golden/stats_golden.txt, gen_stats_golden.py: numpy and scipy on synthetic samples): mean, sd (ddof
// 1), median, the robust SD, percentiles (numpy linear), peak, kurtosis and skewness (MATLAB's: scipy fisher=False, bias=True), min, max; refusals.
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/stats.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_stats\n";
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/stats_golden.txt");
  EXPECT(in.good(), "the golden is there");
  std::string line;
  int samples = 0;
  double worst = 0;
  while (std::getline(in, line)) {
    std::istringstream h(line);
    std::string tag, name;
    std::size_t n = 0;
    h >> tag >> name >> n;
    if (tag != "sample") break;
    std::getline(in, line);
    std::istringstream vs(line);
    std::vector<double> x(n);
    for (auto& v : x) vs >> v;
    std::getline(in, line);
    std::istringstream ws(line);
    std::map<std::string, double> want;
    for (std::string kv; ws >> kv;) want[kv.substr(0, kv.find('='))] = std::stod(kv.substr(kv.find('=') + 1));
    using namespace nxr::stats;
    const std::map<std::string, double> got = {{"mean", mean(x)}, {"sd", sd(x)}, {"median", median(x)}, {"mad", mad(x)}, {"p10", percentile(x, 10)}, {"p50", percentile(x, 50)},
                                               {"p75", percentile(x, 75)}, {"p98", percentile(x, 98)}, {"peak", peak(x)}, {"kurtosis", kurtosis(x)}, {"skewness", skewness(x)},
                                               {"min", min(x)}, {"max", max(x)}};
    for (const auto& [k, w] : want) worst = std::max(worst, std::abs(got.at(k) - w) / std::max(1.0, std::abs(w)));
    ++samples;
  }
  std::printf("  %d samples, worst relative difference from numpy/scipy %.2e\n", samples, worst);
  EXPECT(samples == 3 && worst < 1e-12, "every summary equals its reference (numpy, scipy)");
  EXPECT(nxr::stats::summary("count", {2, 4, 9}) == 3 && nxr::stats::summary("sum", {2, 4, 9}) == 15 && nxr::stats::summary("count", {}) == 0 && nxr::stats::summary("sum", {}) == 0, "count and sum, 0 for none");
  {
    const auto t = nxr::stats::summarizeRows({1, 2, std::nan(""), std::nan(""), std::nan(""), std::nan("")}, 2, 3, nxr::stats::parseSummaries({"count", "sum", "median"}), "each", 1, nullptr);
    EXPECT(t[0] == 2 && t[1] == 3 && t[2] == 1.5 && t[3] == 0 && t[4] == 0 && std::isnan(t[5]), "per row: the count and sum of an empty row are 0, its median NaN");
  }
  EXPECT(nxr::stats::summary("percentile", {1, 2, 3, 4}, 75) == 3.25 && nxr::stats::summary("sd", {1, 2, 3, 4}, 0, 0) == std::sqrt(1.25), "summary by name, with p and ddof");
  {   // THE SUMMARIES ACROSS THE MEMBERS of a [rows x members] table: per row, pooled, of the mean over rows; NaN members left out and counted
    using namespace nxr::stats;
    const std::vector<double> t = {1, 2, 3, NAN, 10, 20, 30, 40};       // 2 rows x 4 members
    std::vector<std::int64_t> used;
    const auto each = summarizeRows(t, 2, 4, parseSummaries({"median", "max"}), "each", 1, &used);
    EXPECT(each == (std::vector<double>{2, 3, 25, 40}) && used == (std::vector<std::int64_t>{3, 4}), "per row: the NaN member left out and counted");
    const auto pool = summarizeRows(t, 2, 4, parseSummaries({"percentile:50"}), "pool", 1, &used);
    EXPECT(pool.size() == 1 && pool[0] == 10 && used[0] == 7, "pooled over rows");
    const auto mean = summarizeRows(t, 2, 4, parseSummaries({"mean"}), "mean", 1, &used);
    EXPECT(mean.size() == 1 && std::abs(mean[0] - (5.5 + 11 + 16.5) / 3) < 1e-12 && used[0] == 3, "of the mean over rows (a member with a NaN row left out)");
  }
  auto refused = [](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  EXPECT(refused([] { nxr::stats::median({}); }) && refused([] { nxr::stats::summary("mode", {1.0}); }) && refused([] { nxr::stats::sd({1.0}); }) &&
         refused([] { nxr::stats::percentile({1.0}, 101); }) && refused([] { nxr::stats::parseSummaries({"percentile"}); }) && refused([] { nxr::stats::parseSummaries({"mode"}); }),
         "refusals: an empty sample, an unknown summary, sd of one value, p outside [0, 100]");
  std::cout << (failures ? "FAIL" : "PASS") << " test_stats (" << failures << " failure(s))\n";
  return failures ? 1 : 0;
}
