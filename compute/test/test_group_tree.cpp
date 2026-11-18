// nxr::select::groupTreeFor against nxr's `geom.trees.Template.tree_for` (golden: test/golden/group_tree_golden.txt, gen_group_tree_golden.py; synthetic random spheres and codes):
// every subject vertex's code at four depths IDENTICAL, the median and maximum match distance to rounding; refusals (a depth beyond the template's finest, a piece-count mismatch).
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/select/dyadic.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/group_tree_golden.txt");
  EXPECT(in.good(), "the golden is there");
  std::string line, tag;
  int finest = 0;
  std::vector<double> tsph, ssph;
  std::vector<std::int32_t> tcodes;
  std::vector<std::pair<int, int>> truns, sruns;
  int checked = 0, depth = 0;
  double med = 0, mx = 0;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    s >> tag;
    if (tag == "finest") s >> finest;
    else if (tag == "tpl_runs" || tag == "sub_runs") { int a, b, c, d; s >> a >> b >> c >> d; (tag == "tpl_runs" ? truns : sruns) = {{a, b}, {c, d}}; }
    else if (tag == "tpl_sphere") { double x; while (s >> x) tsph.push_back(x); }
    else if (tag == "sub_sphere") { double x; while (s >> x) ssph.push_back(x); }
    else if (tag == "tpl_codes") { int x; while (s >> x) tcodes.push_back(x); }
    else if (tag == "depth") { std::string a, b; s >> depth >> a >> med >> b >> mx; }
    else if (tag == "codes") {
      std::vector<std::int32_t> gold;
      int x;
      while (s >> x) gold.push_back(x);
      const int nV = static_cast<int>(ssph.size() / 3);
      double m = 0, M = 0;
      const nxr::select::Tree t = nxr::select::groupTreeFor(tsph.data(), tcodes.data(), truns, finest, ssph.data(), nV, sruns, depth, &m, &M);
      EXPECT(t.codes == gold, "codes at depth " << depth << " identical to nxr's tree_for");
      EXPECT(std::abs(m - med) < 1e-13 && std::abs(M - mx) < 1e-13, "match distances at depth " << depth << ": " << m << " " << M);
      EXPECT(t.depth == depth && t.finest == finest && t.runs == sruns, "tree fields");
      ++checked;
    }
  }
  EXPECT(checked == 4, "four depths checked");
  bool deep = false, pieces = false;
  const int nV = static_cast<int>(ssph.size() / 3);
  try { nxr::select::groupTreeFor(tsph.data(), tcodes.data(), truns, finest, ssph.data(), nV, sruns, finest + 1); } catch (const std::invalid_argument&) { deep = true; }
  try { nxr::select::groupTreeFor(tsph.data(), tcodes.data(), truns, finest, ssph.data(), nV, {{0, nV}}, 2); } catch (const std::invalid_argument&) { pieces = true; }
  EXPECT(deep, "a depth beyond the template's finest is refused");
  EXPECT(pieces, "a piece-count mismatch is refused");
  std::cout << (fails ? "test_group_tree FAILED\n" : "test_group_tree OK\n");
  return fails ? 1 : 0;
}
