// nxr::io::rollup against nxr's `measure.rollup` (golden: test/golden/rollup_golden.txt, gen_rollup_golden.py): space by reshape at every level (−1 … D) and time on the tower at several levels and anchors,
// for sum, max and min; a derived density (Σ power / (node area × tile samples)) from rolled-up sums in both orders; mergeOf's rule; refusals.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/io/rollup.h"

using namespace nxr::io::rollup;

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; std::string t; while (s >> t) v.push_back(std::strtod(t.c_str(), nullptr)); return v; }
Merge op_of(const std::string& n) { return n == "sum" ? Merge::Sum : n == "max" ? Merge::Max : Merge::Min; }
bool same(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-12) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i])) { if (std::isnan(a[i]) != std::isnan(b[i])) return false; continue; }
    if (std::abs(a[i] - b[i]) > tol * (1 + std::abs(b[i]))) return false;
  }
  return true;
}
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_IO_DATA_DIR) + "/../golden/rollup_golden.txt");
  CHECK(in.good());
  std::string line;
  std::vector<double> x, power, area, samples;
  std::string kind, op;
  std::int64_t rows = 0, n = 0, code0 = 0;
  int depth = 0, cases = 0;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string tag; s >> tag;
    if (tag == "space") { kind = "space"; s >> op >> depth >> rows >> n; }
    else if (tag == "time") { kind = "time"; s >> op >> code0 >> rows >> n; }
    else if (tag == "x") x = nums(s);
    else if (tag == "level") {
      int lvl; std::int64_t cnt; s >> lvl >> cnt;
      std::getline(in, line);
      std::istringstream ys(line); std::string yt; ys >> yt;
      const std::vector<double> want = nums(ys);
      std::int64_t got_n = 0;
      const std::vector<double> got = kind == "space" ? space(x, rows, n, depth, lvl, op_of(op), &got_n) : time(x, rows, n, lvl, op_of(op), code0, &got_n);
      CHECK(got_n == cnt && same(got, want));
      ++cases;
    } else if (tag == "density") { s >> depth >> n >> rows >> code0; }              // leaves, frames, code0
    else if (tag == "power") power = nums(s);
    else if (tag == "area") area = nums(s);
    else if (tag == "samples") samples = nums(s);
    else if (tag == "dlevel") {
      int lvl, m; std::int64_t nn, nt; s >> lvl >> m >> nn >> nt;
      std::getline(in, line);
      std::istringstream ds(line); std::string dt; ds >> dt;
      const std::vector<double> want = nums(ds);
      // the leaves-by-frames power as [frames x leaves] for space, then time on [nodes x frames]
      const std::int64_t leaves = n, frames = rows;
      std::vector<double> t(power.size());
      for (std::int64_t l = 0; l < leaves; ++l) for (std::int64_t f = 0; f < frames; ++f) t[static_cast<std::size_t>(f * leaves + l)] = power[static_cast<std::size_t>(l * frames + f)];
      std::int64_t nodes = 0, tiles = 0;
      const std::vector<double> sp = space(t, frames, leaves, depth, lvl, Merge::Sum, &nodes);      // [frames x nodes]
      std::vector<double> u(sp.size());
      for (std::int64_t f = 0; f < frames; ++f) for (std::int64_t k = 0; k < nodes; ++k) u[static_cast<std::size_t>(k * frames + f)] = sp[static_cast<std::size_t>(f * nodes + k)];
      const std::vector<double> p = time(u, nodes, frames, m, Merge::Sum, code0, &tiles);
      const std::vector<double> a = space(area, 1, leaves, depth, lvl, Merge::Sum), sm = time(samples, 1, frames, m, Merge::Sum, code0);
      std::vector<double> d(p.size());
      for (std::int64_t k = 0; k < nodes; ++k) for (std::int64_t j = 0; j < tiles; ++j) { const double w = a[static_cast<std::size_t>(k)] * sm[static_cast<std::size_t>(j)]; d[static_cast<std::size_t>(k * tiles + j)] = w > 0 ? p[static_cast<std::size_t>(k * tiles + j)] / w : std::nan(""); }
      CHECK(nodes == nn && tiles == nt && same(d, want, 1e-11));
      // the other order (time, then space) gives the same sums
      const std::vector<double> tm = time(power, leaves, frames, m, Merge::Sum, code0);               // [leaves x tiles]
      std::vector<double> tt(tm.size());
      for (std::int64_t l = 0; l < leaves; ++l) for (std::int64_t j = 0; j < tiles; ++j) tt[static_cast<std::size_t>(j * leaves + l)] = tm[static_cast<std::size_t>(l * tiles + j)];
      const std::vector<double> st = space(tt, tiles, leaves, depth, lvl, Merge::Sum);                // [tiles x nodes]
      std::vector<double> p2(p.size());
      for (std::int64_t j = 0; j < tiles; ++j) for (std::int64_t k = 0; k < nodes; ++k) p2[static_cast<std::size_t>(k * tiles + j)] = st[static_cast<std::size_t>(j * nodes + k)];
      CHECK(same(p2, p, 1e-12));
      ++cases;
    }
  }
  std::printf("  %d roll-up cases against nxr's rollup\n", cases);
  CHECK(cases > 50);
  CHECK(merge_of("envelope_max") == Merge::Max && merge_of("max") == Merge::Max && merge_of("min") == Merge::Min && merge_of("power_sum") == Merge::Sum && merge_of("tensor_12_re") == Merge::Sum && merge_of("maximum") == Merge::Sum);
  auto refused = [](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  CHECK(refused([] { space({1, 2, 3, 4}, 1, 4, 3, 0, Merge::Sum); }));            // 4 leaves are not a whole number of pieces of 8
  CHECK(refused([] { space(std::vector<double>(16, 1.0), 1, 16, 3, 4, Merge::Sum); }));   // a level finer than the depth
  CHECK(refused([] { time({1, 2}, 1, 2, -1, Merge::Sum, 0); }));
  std::printf("test_rollup: ok\n");
  return 0;
}
