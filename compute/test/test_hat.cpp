// nxr::filterbank::mexicanHat against nxr (golden: test/golden/gen_hat_golden.py, the gain lines of
// nxr/detect/vortex_cores.py hat_operator on a synthetic sphere spectrum, levels 1 … 5): nxr's λs²e^(−λs²/2) (s² = 2/centre) equals the hat's raw gain; nxr scales it by the sample
// maximum, which on a dense spectrum is the analytic one (2/e), so the gain here is the raw one divided by 2/e. The centre nxr takes, 4π·2^(l−0.5)/A, is the geometric centre of
// `weyl(A, …)`'s level-l band. Then: unit peak at the centre, 0 at 0, the reach (gain under the floor), refusals.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/filterbank.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/hat_golden.txt");
  if (!in) { std::cout << "no golden\n"; return 1; }
  double area = 0;
  std::vector<double> lam;
  std::map<int, std::vector<double>> raw, gain;
  std::map<int, std::pair<double, double>> peak;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag;
    s >> tag;
    if (tag == "area") s >> area;
    else if (tag == "lam") { int n; s >> n; lam.resize(static_cast<std::size_t>(n)); for (auto& v : lam) s >> v; }
    else if (tag == "peak") { int l; double c, m; s >> l >> c >> m; peak[l] = {c, m}; }
    else if (tag == "raw" || tag == "gain") { int l, n; s >> l >> n; auto& v = (tag == "raw" ? raw : gain)[l]; v.resize(static_cast<std::size_t>(n)); for (auto& x : v) s >> x; }
  }
  const auto bank = nxr::filterbank::weyl(area, 6);
  for (int l = 1; l <= 5; ++l) {
    const auto& m = bank.members[static_cast<std::size_t>(l)];
    const double centre = std::sqrt(m.lo * m.hi);
    EXPECT(std::fabs(centre - peak[l].first) < 1e-12 * centre, "level " << l << ": the geometric centre of the Weyl band is not nxr's lam_peak");
    double worst = 0, worst_n = 0;
    for (std::size_t i = 0; i < lam.size(); ++i) {
      worst = std::max(worst, std::fabs(nxr::filterbank::mexicanHat(lam[i], centre) - raw[l][i] / (2.0 / M_E)));
      worst_n = std::max(worst_n, std::fabs(nxr::filterbank::mexicanHat(lam[i], centre) * (2.0 / M_E) / peak[l].second - gain[l][i]));   // nxr's own scaling by its sample maximum
    }
    std::printf("  level %d (centre %.4g): max difference from nxr's gain %.1e (raw/analytic max), %.1e (its sample-max scaling)\n", l, centre, worst, worst_n);
    EXPECT(worst < 1e-12 && worst_n < 1e-12, "level " << l << ": the hat differs from nxr's");
    // the reach: the gain there is the floor, and it is under the floor beyond
    const double reach = nxr::filterbank::mexicanHatReach(centre, 1e-3);
    EXPECT(reach > centre && std::fabs(nxr::filterbank::mexicanHat(reach, centre) - 1e-3) < 1e-12 && nxr::filterbank::mexicanHat(1.5 * reach, centre) < 1e-3, "level " << l << ": the reach");
  }
  EXPECT(std::fabs(nxr::filterbank::mexicanHat(7.0, 7.0) - 1.0) < 1e-15, "unit peak at the centre");
  EXPECT(nxr::filterbank::mexicanHat(0.0, 7.0) == 0.0, "zero at 0");
  bool refused = false;
  try { nxr::filterbank::mexicanHat(1.0, 0.0); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "a centre of 0 not refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
