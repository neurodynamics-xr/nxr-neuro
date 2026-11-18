// nxr::measurement: reduce, merge, roll_up, roll_values, summarize and estimate, on synthetic data only.
//   - parity with the Studio's measureOverPartition (golden/partition_golden.txt from gen_partition_golden.mts: unit weights, area weights, an empty member)
//   - the roll-up law: members merged equal the measure of their union (ingredients; stored values by the registry's merge rule: the in-memory side of io's SQL)
//   - the departures from the reference, each counted: a negative member number is left out, a non-finite value is counted, a member past the count is refused
//   - a median, a reduction of every row at once, a time rule and a product (cells chunk by chunk: bit-identical whatever the blocks), the outer form, the groups of a
//     coarser level, across subjects, summarize, SegmentMean and Histogram, the registry's derived measures
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "golden_io.h"
#include "nxr/measurement/estimate.h"
#include "nxr/measurement/reduce.h"
#include "nxr/measurement/roll_up.h"
#include "nxr/measurement/summarize.h"
#include "nxr/selection/partition.h"

static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace ms = nxr::measurement;
namespace sel = nxr::selection;

static const std::vector<std::string> kMeasures = {"mean", "mean_square", "max", "min", "abs_max", "sum", "area", "n_elements"};

// one row of values over the members of a member-number-per-element form
static ms::Ingredients over(const std::vector<std::int32_t>& codes, int members, const std::vector<double>& x, const std::vector<double>* w, bool hold = false) {
  sel::MemberOfElement mo;
  mo.member_of = codes;
  mo.members = members;
  ms::Ingredients g(1, members, 1, hold);
  ms::Request r;
  r.measures = {"mean", "median"};
  if (!hold) r.measures = {"mean"};
  r.weighting = w ? ms::Weighting::given : ms::Weighting::none;
  ms::reduce(g, x.empty() ? nullptr : x.data(), 1, static_cast<std::int64_t>(codes.size()), mo, w ? w->data() : nullptr, r);
  return g;
}
static double val(const ms::Ingredients& g, int m, const std::string& name, std::int64_t row = 0) { return ms::value(g, row, m, name); }

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
    const ms::Ingredients s = over(k, members, x, weighted ? &w : nullptr);
    double worst = 0; int mismatched = 0;
    for (int m = 0; m < members; ++m) {
      for (std::size_t j = 0; j < kMeasures.size(); ++j) {
        const double got = val(s, m, kMeasures[j]), ref = want(m, static_cast<Eigen::Index>(j));
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

static sel::Elements groups_of(const std::vector<std::vector<std::int64_t>>& g, std::int64_t size) {
  sel::Elements e;
  e.size = size;
  for (const auto& m : g) { e.elements.insert(e.elements.end(), m.begin(), m.end()); e.offsets.push_back(static_cast<std::int64_t>(e.elements.size())); }
  return e;
}

static void rollup_law() {
  std::cout << "=== the roll-up law: members merged by the ingredients' rules = the measure of their union ===\n";
  const std::vector<std::int32_t> fine = {0, 1, 2, 3, 0, 1, 2, 3, 3, 2}, coarse = {0, 0, 1, 1, 0, 0, 1, 1, 1, 1};   // coarse = fine >> 1
  const std::vector<double> x = {1, 4, -2, 5, 3, 0.5, 7, -1, 2, 6}, w = {0.3, 1.2, 0.7, 0.1, 0.9, 0.4, 0.5, 0.8, 0.2, 0.6};
  const ms::Ingredients f = over(fine, 4, x, &w), c = over(coarse, 2, x, &w);
  ms::Request req;
  req.measures = {"mean", "mean_square", "max", "min", "abs_max", "mean_cube", "mean_fourth", "sum"};
  const ms::Ingredients merged = ms::roll_up(f, groups_of({{0, 1}, {2, 3}}, 4), req);
  for (int p = 0; p < 2; ++p)
    for (const char* name : {"mean", "mean_square", "max", "min", "abs_max", "mean_cube", "mean_fourth", "sum", "area", "n_elements"}) {
      const double a = val(merged, p, name), b = val(c, p, name);
      EXPECT(std::abs(a - b) <= 1e-13 * std::max(1.0, std::abs(b)), "rolled-up " << name << " of member " << p << ": " << a << " vs " << b);
    }
  // stored VALUES by the registry's rule: a mean weighted by its weight measure, a max by max, a sum by sum (the in-memory side of io's SQL roll-up)
  Eigen::MatrixXd means(1, 4), areas(1, 4), maxes(1, 4), sums(1, 4);
  for (int m = 0; m < 4; ++m) { means(0, m) = val(f, m, "mean"); areas(0, m) = val(f, m, "area"); maxes(0, m) = val(f, m, "max"); sums(0, m) = val(f, m, "sum"); }
  const sel::Elements g = groups_of({{0, 1}, {2, 3}}, 4);
  const Eigen::MatrixXd rm = ms::roll_values(ms::Registry::standard(), "mean", means, &areas, g), rx = ms::roll_values(ms::Registry::standard(), "max", maxes, nullptr, g),
                        rs = ms::roll_values(ms::Registry::standard(), "sum", sums, nullptr, g);
  for (int p = 0; p < 2; ++p) {
    EXPECT(std::abs(rm(0, p) - val(c, p, "mean")) < 1e-13, "roll_values mean of member " << p);
    EXPECT(rx(0, p) == val(c, p, "max"), "roll_values max of member " << p);
    EXPECT(std::abs(rs(0, p) - val(c, p, "sum")) < 1e-13, "roll_values sum of member " << p);
  }
  bool refused = false;
  try { ms::roll_values(ms::Registry::standard(), "median", means, nullptr, g); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "a median is never rolled up (merge none)");
  ms::Request med; med.measures = {"median"};
  refused = false;
  try { ms::merge(f, g, med); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "merge refuses a median");
  // the moments' ratios are derived at read: the variance of the rolled-up member equals the direct one
  const auto stored = [&](const std::string& n) { return val(merged, 0, n); };
  EXPECT(std::abs(ms::Registry::standard().derive("variance", stored) - ms::Registry::standard().derive("variance", [&](const std::string& n) { return val(c, 0, n); })) < 1e-12, "derived variance from merged ingredients");
  // across subjects: the same codes added
  const ms::Ingredients two = ms::across({&f, &f}, req);
  EXPECT(two.weight[0] == 2 * f.weight[0] && two.max[1] == f.max[1] && two.count[2] == 2 * f.count[2], "across: weights and counts add, maxima merge");
}

static void departures() {
  std::cout << "=== departures, counted: negative numbers left out, non-finite values counted, a member past the count refused ===\n";
  const std::vector<std::int32_t> k = {-1, 0, 0, 1, -1};
  const std::vector<double> x = {100, 1, 3, std::numeric_limits<double>::quiet_NaN(), 100};
  const ms::Ingredients s = over(k, 2, x, nullptr);
  EXPECT(s.unassigned == 2, "two elements with a negative member number");
  EXPECT(val(s, 0, "mean") == 2.0 && s.count[0] == 2, "member 0 is its two elements only");
  EXPECT(s.nonfinite[1] == 1 && std::isnan(val(s, 1, "mean")), "member 1's NaN is counted and its mean is not written");
  bool refused = false;
  try { over({0, 2}, 2, {}, nullptr); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "a member past the count is refused");
  const ms::Ingredients geo = over(k, 2, {}, nullptr);
  EXPECT(val(geo, 0, "n_elements") == 2.0 && val(geo, 0, "area") == 2.0, "without values: counts and weights only");
  refused = false;
  try { sel::MemberOfElement mo; mo.member_of = {0, 0}; mo.members = 1; ms::Ingredients g(1, 1); ms::Request r; r.weighting = ms::Weighting::given; const double v[2] = {1, 2}; ms::reduce(g, v, 1, 2, mo, nullptr, r); }
  catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "Weighting::given without weights is refused");
}

static void median_within_member() {
  std::cout << "=== the median within a member: numpy's, finite values only ===\n";
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<std::int32_t> k = {0, 1, 0, -1, 1, 0, 1, 0, 1, 0, 3, 1, 0};
  const std::vector<double> x = {9, 4, 1, 1000, 2, 5, 8, nan, 6, 3, nan, nan, 7};
  const ms::Ingredients g = over(k, 4, x, nullptr, true);
  EXPECT(val(g, 0, "median") == 5.0, "odd count: the middle value, got " << val(g, 0, "median"));
  EXPECT(val(g, 1, "median") == 5.0, "even count: the mean of the two middle values, got " << val(g, 1, "median"));
  EXPECT(std::isnan(val(g, 2, "median")) && std::isnan(val(g, 3, "median")), "an empty member and a member of only NaN are NaN");
}

static void every_row_at_once() {
  std::cout << "=== a reduction of every row at once equals one row at a time, to the bit ===\n";
  const std::vector<std::int32_t> k = {0, 1, 0, -1, 1, 0};
  const std::vector<double> w = {1.0, 2.0, 3.0, 5.0, 0.5, 2.0};
  const std::vector<float> x = {1, 2, 3, 100, 5, 7, 10, 20, 30, 1000, 50, 70};     // 2 rows x 6 elements
  sel::MemberOfElement mo; mo.member_of = k; mo.members = 3;
  ms::Ingredients all(2, 3);
  ms::Request r; r.measures = {"mean"}; r.weighting = ms::Weighting::given;
  ms::reduce(all, x.data(), 2, 6, mo, w.data(), r);
  double worst = 0;
  for (int row = 0; row < 2; ++row) {
    std::vector<double> col(6);
    for (int i = 0; i < 6; ++i) col[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(row * 6 + i)];
    const ms::Ingredients one = over(k, 3, col, &w);
    for (int m = 0; m < 2; ++m) worst = std::max(worst, std::abs(val(all, m, "mean", row) - val(one, m, "mean")));
  }
  EXPECT(worst == 0.0, "equal to a row at a time (worst " << worst << ")");
  EXPECT(val(all, 2, "mean", 0) == 0.0 && all.count[all.at(0, 2)] == 0, "a member of no element: the reference's 0 for a mean, count 0");
}

static void overlapping_members() {
  std::cout << "=== overlapping members (listed elements): an element adds to every member holding it ===\n";
  sel::Elements e = groups_of({{0, 1, 2}, {2, 3}}, 4);
  const std::vector<double> x = {1, 2, 3, 4};
  ms::Ingredients g(1, 2);
  ms::Request r; r.measures = {"sum"};
  ms::reduce(g, x.data(), 1, 4, e, nullptr, r);
  EXPECT(val(g, 0, "sum") == 6.0 && val(g, 1, "sum") == 7.0, "the shared element 2 is in both members");
}

static void time_rule_and_product() {
  std::cout << "=== a time rule and a product, block by block: bit-identical whatever the blocks ===\n";
  const std::int64_t N = 103, C = 4;
  std::vector<float> x(static_cast<std::size_t>(C * N));
  for (std::int64_t c = 0; c < C; ++c) for (std::int64_t t = 0; t < N; ++t) x[static_cast<std::size_t>(c * N + t)] = static_cast<float>(std::sin(0.37 * static_cast<double>(t) + c) + 0.01 * static_cast<double>(c * t));
  sel::Rule rule; rule.origin = 0; rule.width = 10; rule.hop = 10; rule.count = 11; rule.size = N;
  // per row: a value per channel per tile
  ms::Request per_row; per_row.measures = {"mean"}; per_row.per_row = true;
  ms::Ingredients a(C, rule.count);
  ms::reduce(a, x.data(), C, N, 0, rule, per_row);
  const double direct = [&] { double s = 0; for (int t = 20; t < 30; ++t) s += static_cast<double>(x[static_cast<std::size_t>(2 * N + t)]); return s / 10.0; }();
  EXPECT(std::abs(val(a, 2, "mean", 2) - direct) < 1e-12, "tile 2 of channel 2 is the mean of its 10 samples");
  // two chunks cut at an arbitrary sample, pooled over the rows: the same doubles
  ms::Request pooled; pooled.measures = {"sum"};
  ms::Ingredients whole(1, rule.count), parts(1, rule.count);
  ms::reduce(whole, x.data(), C, N, 0, rule, pooled);
  const std::int64_t cut = 37;
  std::vector<float> b1(static_cast<std::size_t>(C * cut)), b2(static_cast<std::size_t>(C * (N - cut)));
  for (std::int64_t c = 0; c < C; ++c) {
    for (std::int64_t t = 0; t < cut; ++t) b1[static_cast<std::size_t>(c * cut + t)] = x[static_cast<std::size_t>(c * N + t)];
    for (std::int64_t t = cut; t < N; ++t) b2[static_cast<std::size_t>(c * (N - cut) + t - cut)] = x[static_cast<std::size_t>(c * N + t)];
  }
  ms::reduce(parts, b1.data(), C, cut, 0, rule, pooled);
  ms::reduce(parts, b2.data(), C, N - cut, cut, rule, pooled);
  EXPECT(whole.sum == parts.sum && whole.count == parts.count && whole.max == parts.max, "pooled tiles of two chunks equal the whole block's, to the last bit");
  // a product: time tiles (factor 0) x two groups of channels (factor 1), codes first factor fastest
  sel::ProductForm cells;
  cells.factors = {rule, sel::MemberOfElement{{0, 0, 1, 1}, 2}};
  cells.counts = {11, 2};
  cells.radices = {11, 2};
  ms::Request pr; pr.measures = {"sum"};
  ms::Ingredients cw(1, cells.members()), cp(1, cells.members());
  ms::reduce(cw, x.data(), C, N, 0, cells, nullptr, pr);
  ms::reduce(cp, b1.data(), C, cut, 0, cells, nullptr, pr);
  ms::reduce(cp, b2.data(), C, N - cut, cut, cells, nullptr, pr);
  EXPECT(cw.sum == cp.sum && cw.count == cp.count, "the product's cells of two chunks equal the whole block's, to the last bit");
  EXPECT(cw.count[static_cast<std::size_t>(3 + 11 * 1)] == 20, "cell (tile 3, channel group 1) holds 2 channels x 10 samples");
  // the weights of the space factor
  const double wt[4] = {1, 1, 2, 2};
  ms::Request wr; wr.measures = {"sum"}; wr.weighting = ms::Weighting::given;
  ms::Ingredients cwt(1, cells.members());
  ms::reduce(cwt, x.data(), C, N, 0, cells, wt, wr);
  EXPECT(std::abs(cwt.weight[static_cast<std::size_t>(3 + 11)] - 40.0) < 1e-12, "weights add: 2 channels x 10 samples x weight 2");
}

static void outer_form() {
  std::cout << "=== the outer form: z z^H summed over tiles; a burst's as the sum of its tiles' ===\n";
  using Eigen::MatrixXcd;
  MatrixXcd Z(2, 9);
  for (int t = 0; t < 9; ++t) { Z(0, t) = std::complex<double>(1 + t, 0.5 * t); Z(1, t) = std::complex<double>(-t, 2.0 + t); }
  sel::Rule rule; rule.origin = 0; rule.width = 3; rule.hop = 3; rule.count = 3; rule.size = 9;
  ms::MatrixIngredients whole(3), parts(3);
  ms::reduce(whole, Z, 0, rule);
  ms::reduce(parts, Z.leftCols(4), 0, rule);
  ms::reduce(parts, Z.rightCols(5), 4, rule);
  double worst = 0;
  for (int k = 0; k < 3; ++k) worst = std::max(worst, (whole.sum[static_cast<std::size_t>(k)] - parts.sum[static_cast<std::size_t>(k)]).cwiseAbs().maxCoeff());
  EXPECT(worst < 1e-12 && whole.count == parts.count, "a tile cut by a block edge is completed by the next block (worst " << worst << ")");
  const MatrixXcd direct = Z.middleCols(3, 3) * Z.middleCols(3, 3).adjoint();
  EXPECT((whole.sum[1] - direct).cwiseAbs().maxCoeff() < 1e-12 && whole.count[1] == 3, "a tile's sum is one complex product");
  const ms::MatrixIngredients burst = ms::merge(whole, groups_of({{0, 1}, {}}, 3));
  EXPECT((burst.sum[0] - whole.sum[0] - whole.sum[1]).cwiseAbs().maxCoeff() < 1e-12 && burst.count[0] == 6 && burst.sum[1].size() == 0, "a burst is the sum of its tiles; an empty group has an empty sum");
  EXPECT(std::abs(whole.mean(0)(0, 0) - whole.sum[0](0, 0) / 3.0) < 1e-12, "the mean is sum / count");
}

static void coarser_levels() {
  std::cout << "=== the groups of a coarser level, in closed form ===\n";
  // a dyadic surface grid of depth 3, one piece: 8 finest tiles; level 1 = 2 tiles
  std::vector<std::int32_t> codes(8);
  for (int i = 0; i < 8; ++i) codes[static_cast<std::size_t>(i)] = i;
  const sel::Selection grid = sel::dyadic_grid(codes, 3, 1);
  const sel::Elements g = ms::coarser_groups(grid, 3, 1);
  EXPECT(g.members() == 2 && g.offsets[1] == 4 && g.elements[4] == 4, "level 3 to level 1: tiles 0-3 and 4-7");
  EXPECT(ms::coarser_groups(grid, 3, ms::kWhole).members() == 1, "the whole is one group");
  // a dyadic time grid: tiles of 2^2 samples on 50 samples (13 tiles), to tiles of 2^4 (shift 2)
  const sel::Selection t2 = sel::time_grid(sel::Line{"s", 50, 0.0, 1.0}, 2);
  const sel::Elements gt = ms::coarser_groups(t2, 2, 4);
  EXPECT(gt.members() == 4 && gt.offsets[4] - gt.offsets[3] == 1, "13 tiles to 4 groups, the last holds the partial tile");
  // an irregular selection never rolls up
  bool refused = false;
  try { ms::coarser_groups(sel::set_of_spans({sel::SpanRow{0, 2, 0}, sel::SpanRow{1, 3, 1}}, "s"), 0, 1); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "an irregular selection's members never roll up (refused with the reason)");
  // one factor of a product: tiles (factor 0) x 3 elements
  const sel::Selection prod = sel::product({sel::time_grid(sel::Line{"s", 16, 0.0, 1.0}, 0), sel::elements_grid(3)});
  const sel::Elements gp = ms::coarser_groups(prod, 0, 0, 2);
  EXPECT(gp.members() == 4 * 3, "16 tiles to 4 (shift 2) x 3 elements: 12 coarse members");
}

static void summarize_and_estimate() {
  std::cout << "=== summarize, SegmentMean and Histogram ===\n";
  Eigen::MatrixXd v(2, 5);
  v << 1, 2, 3, 4, std::numeric_limits<double>::quiet_NaN(), 10, 20, 30, 40, 50;
  ms::Summaries s; s.names = {"mean", "count", "percentile:50"}; s.combine = ms::Combine::each;
  std::vector<std::int64_t> used;
  const Eigen::MatrixXd r = ms::summarize(v, s, &used);
  EXPECT(r.rows() == 2 && r(0, 0) == 2.5 && r(0, 1) == 4 && r(1, 0) == 30 && used[0] == 4 && used[1] == 5, "each row its own statistic; a NaN member skipped and counted");
  s.combine = ms::Combine::pool;
  EXPECT(ms::summarize(v, s).rows() == 1, "pool: one statistic");
  bool refused = false;
  try { ms::Summaries bad; bad.names = {"nonsense"}; ms::summarize(v, bad); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "an unknown summary is refused");

  ms::SegmentMean sm;
  Eigen::MatrixXd a(1, 2), b(1, 2);
  a << 1, 2; b << 3, 6;
  sm.add_real(0, a); sm.add_real(2, b); sm.skip(1);
  EXPECT(sm.used == 2 && sm.skipped == 1 && sm.mean_real()(0, 0) == 2.0 && sm.mean_real()(0, 1) == 4.0, "the mean over the segments added; a skipped one is counted");
  refused = false;
  try { sm.add_real(2, b); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "a segment added twice is refused");

  ms::Histogram h;
  h.edges = {1.0, 2.0, 4.0};
  const float vals[6] = {0.5f, 1.0f, 1.5f, 3.0f, 9.0f, std::numeric_limits<float>::infinity()};
  h.add(vals, 6);
  EXPECT(h.counts == std::vector<std::int64_t>({1, 2, 2}) && h.nonfinite == 1, "bins by upper edge, the last also holds what is above, a non-finite value is counted");
}

static void registry() {
  std::cout << "=== the registry: merge rules as data, derived measures ===\n";
  const ms::Registry& r = ms::Registry::standard();
  EXPECT(r.at("max").merge == ms::Merge::max && r.at("mean").merge == ms::Merge::mean && r.at("median").merge == ms::Merge::none && r.at("sum").merge == ms::Merge::sum, "merge rules");
  EXPECT(r.at("amplitude").form == "magnitude" && r.at("power").form == "power" && r.at("envelope_max").merge == ms::Merge::max, "forms and merges of the complex measures");
  EXPECT(r.derived("sd") && !r.derived("mean"), "derived measures are named by their ingredients");
  bool refused = false;
  try { r.at("no_such_measure"); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "an unknown measure is refused, naming the valid ones");
  const ms::Registry custom({ms::MeasureSpec{"my_max", "none", "max", ms::Merge::max, "", {}}});
  EXPECT(custom.at("my_max").merge == ms::Merge::max && custom.find("mean") == nullptr, "a registry made from io's rows");
  const double tr = r.derive("travelling_index", [](const std::string& n) { return n == "phase_current_1" ? 3.0 : (n == "phase_current_2" ? 4.0 : (n == "phase_power" ? 25.0 : 1.0)); });
  EXPECT(std::abs(tr - 5.0 / 5.0) < 1e-12, "travelling index: |sum j| / sqrt(power x gradient power), got " << tr);
}

int main() {
  every_row_at_once();
  overlapping_members();
  parity();
  median_within_member();
  rollup_law();
  departures();
  time_rule_and_product();
  outer_form();
  coarser_levels();
  summarize_and_estimate();
  registry();
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
