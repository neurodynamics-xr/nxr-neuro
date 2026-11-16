// nxr::risefall against Rheome's rheome.detect.risefall (golden: test/golden/risefall_golden.txt, gen_risefall_golden.m, MATLAB on synthetic double-sigmoid episodes with known parameters and
// noise): the fitted tOn, tOff, tauR, tauF, b, h, r² and the plateau, for the linear and the quadrature model, to the optimizer's tolerance; the planted parameters recovered; a window under 11
// samples and a series of no amplitude not ok; the posterior model refused by name.
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "nxr/risefall.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_risefall\n";
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/risefall_golden.txt");
  EXPECT(in.good(), "the golden is there");
  std::map<std::string, std::vector<double>> g;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream h(line);
    std::string k; std::size_t n;
    h >> k >> n;
    std::vector<double> v(n);
    for (auto& x : v) h >> x;
    g[k] = v;
  }
  for (const char* c : {"lin1", "lin2", "quad1", "quad2"}) {
    const std::string n = c;
    const auto& a = g[n + "_a"];
    nxr::risefall::Options o;
    o.model = g[n + "_model"][0] == 1 ? nxr::risefall::Model::Quadrature : nxr::risefall::Model::Linear;
    const auto F = nxr::risefall::fit(Eigen::Map<const Eigen::VectorXd>(a.data(), static_cast<Eigen::Index>(a.size())), g[n + "_fs"][0], g[n + "_span"][0], g[n + "_span"][1], o);
    const auto& m = g[n + "_fit"];       // tOn tOff tauR tauF b h r2 ok atBound
    const auto& truth = g[n + "_truth"]; // b h on off tr tf
    double worst_s = std::max({std::abs(F.tOn - m[0]), std::abs(F.tOff - m[1]), std::abs(F.tauR - m[2]), std::abs(F.tauF - m[3])});
    double worst_a = std::max({std::abs(F.b - m[4]), std::abs(F.h - m[5]), std::abs(F.r2 - m[6])});
    std::printf("  %s: tOn %.5f (Rheome %.5f), tOff %.5f (%.5f), tauR %.4f (%.4f), tauF %.4f (%.4f), r2 %.5f (%.5f); worst |difference| %.1e s, %.1e\n", c, F.tOn, m[0], F.tOff, m[1], F.tauR, m[2], F.tauF, m[3], F.r2, m[6], worst_s, worst_a);
    EXPECT(F.ok && m[7] == 1 && worst_s < 2e-3 && worst_a < 2e-3, std::string(c) + ": the fit equals Rheome's (timings within 2 ms, amplitudes and r² within 2e-3)");
    EXPECT(std::abs(F.tOn - truth[2]) < 0.05 && std::abs(F.tOff - truth[3]) < 0.1, std::string(c) + ": the planted onset and offset recovered (0.05 s, 0.1 s)");
    const auto& pl = g[n + "_plateau"];
    EXPECT(F.hasPlateau == (pl[0] == 2) && (!F.hasPlateau || (std::abs(F.plateau[0] - pl[1]) < 2e-3 && std::abs(F.plateau[1] - pl[2]) < 2e-3)), std::string(c) + ": the plateau equals Rheome's");
  }
  {
    const auto& a = g["short_a"];
    const auto F = nxr::risefall::fit(Eigen::Map<const Eigen::VectorXd>(a.data(), static_cast<Eigen::Index>(a.size())), 50, 2, 2.1, nxr::risefall::Options{0.0, 0.1, 1.0, nxr::risefall::Model::Linear});
    EXPECT(!F.ok && g["short_fit"][0] == 0, "a window under 11 samples is not ok, as Rheome's");
    const Eigen::VectorXd z = Eigen::VectorXd::Zero(600);
    EXPECT(!nxr::risefall::fit(z, 50, 3, 6).ok && g["zero_fit"][0] == 0, "a series of no amplitude is not ok, as Rheome's");
  }
  bool refused = false;
  try { nxr::risefall::parseModel("posterior"); } catch (const std::invalid_argument& e) { refused = std::string(e.what()).find("not built") != std::string::npos; }
  EXPECT(refused, "the posterior model is refused by name as not built");
  std::cout << (failures ? "FAILED\n" : "ok\n");
  return failures ? 1 : 0;
}
