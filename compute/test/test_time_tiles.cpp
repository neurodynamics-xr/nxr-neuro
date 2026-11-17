// nxr::measurement::reduce over a time rule against the Studio's own reduceTiles (golden: test/golden/gen_tiles_golden.mts, backend/src/db/measurements.ts): the
// segmented reduce over time windows, pooled over the rows (min, max, Σx, Σx², count), on three cases (a partial last tile; one row, the per-row form of an
// array-backed measurement; wider windows). The Studio keeps float32 tiles (Σx added into float), `reduce` keeps double ingredients: the counts, the minima and the
// maxima are equal to the bit and the sums to float rounding. Then what the chunk loop relies on: a block read as two chunks, cut ANYWHERE (not only at a tile
// edge), gives the whole block's ingredients to the last bit (a cell's values are added in sample order whatever the blocks).
#include <algorithm>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/measurement/reduce.h"

static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace m = nxr::measurement;

static std::vector<float> floats(const std::string& line) {
  std::istringstream s(line);
  std::vector<float> v;
  for (std::string w; s >> w;) v.push_back(static_cast<float>(std::strtod(w.c_str(), nullptr)));
  return v;
}
static bool equal(double a, float b) { return std::isinf(a) ? (std::isinf(b) && (a > 0) == (b > 0)) : static_cast<float>(a) == b; }
static bool close(double a, float b) { return std::abs(a - b) <= 2e-6 * std::max(1.0f, std::abs(b)); }

static m::Ingredients run(const std::vector<float>& x, std::size_t rows, std::size_t samples, std::size_t width, std::size_t tiles, std::size_t cut) {
  nxr::selection::Rule rule;
  rule.origin = 0; rule.width = static_cast<std::int64_t>(width); rule.hop = static_cast<std::int64_t>(width); rule.count = static_cast<std::int64_t>(tiles); rule.size = static_cast<std::int64_t>(samples);
  m::Ingredients acc(1, static_cast<std::int64_t>(tiles));
  m::Request req;
  req.measures = {"sum", "max", "min", "count"};
  // the block in two chunks cut at sample `cut` (cut == samples: one chunk), each [rows x chunk samples] row-major
  const std::size_t parts[2] = {cut, samples - cut};
  std::size_t from = 0;
  for (std::size_t p = 0; p < 2; ++p) {
    if (parts[p] == 0) continue;
    std::vector<float> chunk(rows * parts[p]);
    for (std::size_t c = 0; c < rows; ++c) std::copy(x.begin() + static_cast<long>(c * samples + from), x.begin() + static_cast<long>(c * samples + from + parts[p]), chunk.begin() + static_cast<long>(c * parts[p]));
    m::reduce(acc, chunk.data(), static_cast<std::int64_t>(rows), static_cast<std::int64_t>(parts[p]), static_cast<std::int64_t>(from), rule, req);
    from += parts[p];
  }
  return acc;
}

int main() {
  std::ifstream in(kDir + "/tiles_golden.txt");
  EXPECT(in.good(), "no golden");
  std::string head;
  int cases = 0;
  while (std::getline(in, head)) {
    std::istringstream h(head);
    std::string tag;
    std::size_t rows, samples, width, tiles;
    h >> tag >> rows >> samples >> width >> tiles;
    std::string lx, lmin, lmax, lsum, lsq, lcount;
    std::getline(in, lx); std::getline(in, lmin); std::getline(in, lmax); std::getline(in, lsum); std::getline(in, lsq); std::getline(in, lcount);
    const auto x = floats(lx), wmin = floats(lmin), wmax = floats(lmax), wsum = floats(lsum), wsq = floats(lsq), wcount = floats(lcount);
    const m::Ingredients g = run(x, rows, samples, width, tiles, samples);
    bool ok = true;
    for (std::size_t t = 0; t < tiles; ++t) {
      const std::size_t k = g.at(0, static_cast<std::int64_t>(t));
      const double from = static_cast<double>(t * width), to = static_cast<double>(std::min((t + 1) * width, samples));
      if (to <= from) continue;                                                  // a tile past the data holds nothing (the Studio's negative count is not kept)
      ok = ok && equal(g.min[k], wmin[t]) && equal(g.max[k], wmax[t]) && static_cast<float>(g.count[k]) == wcount[t] && close(g.sum[k], wsum[t]) && close(g.sum_sq[k], wsq[t]);
    }
    std::printf("  %zu rows x %zu samples, windows of %zu (%zu tiles): %s\n", rows, samples, width, tiles, ok ? "equal to the Studio's (counts, min, max to the bit; sums to float rounding)" : "DIFFERENT");
    EXPECT(ok, "reduce differs from the Studio's (" << rows << " x " << samples << ")");
    // ADDITIVE across chunks, cut anywhere: bit-identical doubles
    for (const std::size_t cut : {samples / 3, (tiles / 2) * width}) {
      if (cut == 0 || cut >= samples) continue;
      const m::Ingredients two = run(x, rows, samples, width, tiles, cut);
      EXPECT(two.sum == g.sum && two.sum_sq == g.sum_sq && two.max == g.max && two.min == g.min && two.count == g.count, "two chunks cut at sample " << cut << " do not give the whole block's ingredients");
    }
    ++cases;
  }
  EXPECT(cases == 3, "the golden has 3 cases, read " << cases);
  // a tile wholly past the data holds nothing: count 0, the identities of max and min
  const std::vector<float> one(10, 1.0f);
  const m::Ingredients past = run(one, 1, 10, 8, 3, 10);
  EXPECT(past.count[1] == 2 && past.count[2] == 0 && std::isinf(past.max[2]) && past.max[2] < 0, "a tile past the data is empty: count 0, max -inf");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
