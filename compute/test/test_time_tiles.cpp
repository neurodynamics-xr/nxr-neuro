// nxr::partition::reduceTiles against the Studio's own reduceTiles (golden: test/golden/gen_tiles_golden.mts, backend/src/db/measurements.ts): the
// segmented reduce over time windows, pooled over the rows, to the bit (min, max, Σx, Σx², count), on three cases (a partial last tile; one row, the
// per-row form of an array-backed measurement; wider windows). Then what the chunk loop relies on: a block read as two chunks whose boundary is a tile edge
// gives exactly the whole block's tiles (the tiles are additive across chunks); a tile wholly past the data counts negative, as the Studio's.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/partition.h"

static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

static std::vector<float> floats(const std::string& line) {
  std::istringstream s(line);
  std::vector<float> v;
  for (std::string w; s >> w;) v.push_back(static_cast<float>(std::strtod(w.c_str(), nullptr)));
  return v;
}
static bool same(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) if (!(a[i] == b[i] || (std::isinf(a[i]) && std::isinf(b[i]) && (a[i] > 0) == (b[i] > 0)))) return false;
  return true;
}

int main() {
  using nxr::partition::reduceTiles;
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
    const auto x = floats(lx);
    const auto r = reduceTiles(x.data(), rows, samples, width, tiles);
    const bool ok = same(r.min, floats(lmin)) && same(r.max, floats(lmax)) && same(r.sum, floats(lsum)) && same(r.sumSq, floats(lsq)) && same(r.count, floats(lcount));
    std::printf("  %zu rows x %zu samples, windows of %zu (%zu tiles): %s\n", rows, samples, width, tiles, ok ? "equal to the Studio's, to the bit" : "DIFFERENT");
    EXPECT(ok, "reduceTiles differs from the Studio's (" << rows << " x " << samples << ")");
    // ADDITIVE across chunks: split each row at a tile edge, reduce the two chunks, and their tiles are the whole block's
    const std::size_t cut = (tiles / 2) * width;
    if (cut < samples) {
      std::vector<float> a(rows * cut), b(rows * (samples - cut));
      for (std::size_t c = 0; c < rows; ++c) {
        std::copy(x.begin() + static_cast<long>(c * samples), x.begin() + static_cast<long>(c * samples + cut), a.begin() + static_cast<long>(c * cut));
        std::copy(x.begin() + static_cast<long>(c * samples + cut), x.begin() + static_cast<long>((c + 1) * samples), b.begin() + static_cast<long>(c * (samples - cut)));
      }
      const auto ra = reduceTiles(a.data(), rows, cut, width, tiles / 2), rb = reduceTiles(b.data(), rows, samples - cut, width, tiles - tiles / 2);
      auto cat = [](std::vector<float> u, const std::vector<float>& v) { u.insert(u.end(), v.begin(), v.end()); return u; };
      EXPECT(same(cat(ra.sum, rb.sum), r.sum) && same(cat(ra.sumSq, rb.sumSq), r.sumSq) && same(cat(ra.min, rb.min), r.min) && same(cat(ra.max, rb.max), r.max) &&
                 same(cat(ra.count, rb.count), r.count),
             "two chunks split at a tile edge do not give the whole block's tiles");
    }
    ++cases;
  }
  EXPECT(cases == 3, "the golden has 3 cases, read " << cases);
  // a tile wholly past the data counts negative (the Studio's arithmetic, kept)
  const std::vector<float> one(10, 1.0f);
  const auto past = reduceTiles(one.data(), 1, 10, 8, 3);
  EXPECT(past.count[1] == 2.0f && past.count[2] == -6.0f, "a tile past the data: the Studio's count is to - from");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
