// nxr::stats network nulls (nxr/stats/network_flip.h) against numpy (golden/network_flip_golden.txt, golden/gen_network_flip_golden.py): the reference's sign flip and its
// family-wise p, numpy's choice without replacement and the distance-matched control, drawn from ONE generator in the reference's order (bit for bit).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "check.h"
#include "nxr/stats/network_flip.h"

using namespace nxr::stats;

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/network_flip_golden.txt");
  CHECK(in.good());
  std::vector<std::vector<double>> rows;
  std::vector<std::string> keys;
  for (std::string line; std::getline(in, line);) {
    std::istringstream s(line);
    std::string key; s >> key; keys.push_back(key);
    std::vector<double> v; double x; while (s >> x) v.push_back(x);
    rows.push_back(v);
  }
  const int S = 12, E = 40, patterns = 200;
  std::vector<std::uint8_t> joined(E);
  Eigen::VectorXd dist(E);
  for (int e = 0; e < E; ++e) { joined[static_cast<std::size_t>(e)] = (e * 7 % 5) < 2; dist(e) = e * 37 % 300; }
  const std::vector<std::int64_t> bins = digitize(dist, {0, 30, 45, 60, 80, 100, 130, 170, 250});
  for (int e = 0; e < E; ++e) CHECK(bins[static_cast<std::size_t>(e)] == static_cast<std::int64_t>(rows[0][static_cast<std::size_t>(e)]));
  NumpyGenerator g(0);
  for (int k = 0; k < 3; ++k) {
    Eigen::MatrixXd F(S, E);
    for (int s = 0; s < S; ++s) for (int e = 0; e < E; ++e) F(s, e) = std::sin(1.7 * s + 0.3 * e * e + k) + 0.2 * std::cos(e + k) + 0.15;
    const NetworkSignFlip f = networkSignFlip(F, patterns, g);
    const std::vector<std::uint8_t> control = distanceMatchedControl(joined, bins, g);
    const auto& p = rows[static_cast<std::size_t>(1 + 3 * k)];
    const auto& pf = rows[static_cast<std::size_t>(2 + 3 * k)];
    const auto& c = rows[static_cast<std::size_t>(3 + 3 * k)];
    int bad = 0;
    for (int e = 0; e < E; ++e) {
      bad += std::abs(f.p(e) - p[static_cast<std::size_t>(1 + e)]) > 1e-12;
      bad += std::abs(f.familywiseP(e) - pf[static_cast<std::size_t>(1 + e)]) > 1e-12;
      bad += control[static_cast<std::size_t>(e)] != static_cast<std::uint8_t>(c[static_cast<std::size_t>(1 + e)]);
    }
    std::printf("  frequency %d: %d differences from numpy\n", k, bad);
    CHECK(bad == 0);
  }
  NumpyGenerator h(5);
  h.random(); h.nextUint32();
  const auto c1 = numpyChoice(500, 37, h), c2 = numpyChoice(9, 9, h), c3 = numpyChoice(3, 0, h);
  CHECK(c1.size() == rows[10].size() && c2.size() == rows[11].size() && c3.empty());
  for (std::size_t i = 0; i < c1.size(); ++i) CHECK(c1[i] == static_cast<std::int64_t>(rows[10][i]));
  for (std::size_t i = 0; i < c2.size(); ++i) CHECK(c2[i] == static_cast<std::int64_t>(rows[11][i]));
  CHECK(std::abs(h.random() - rows[13][0]) < 1e-15);
  return nxrtest::finish("test_network_flip");
}
