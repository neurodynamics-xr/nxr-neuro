// nxr::risefall::stagesOf and passesGate (global_bursts' `stages` and `fit_gate`;) against nxr's `vortex.burst_stages` rule (golden: test/golden/gen_burst_stages_golden.py): on three synthetic
// amplitude series (a clean episode, one whose fall is at its bound, one in noise) the Model "quadrature" fit (margin 10 cycles, smoothing 1 cycle, tau <= 10 cycles) gives the same gate decision as nxr
// and the same onset (tOn ± 2 tauR), within (the plateau) and offset (tOff ± 2 tauF) windows; the fit's own numbers agree with nxr's to the optimiser's tolerance (the paths differ: Levenberg-Marquardt here, scipy's
// trust-region there). Then the stages' arithmetic on given numbers (a plateau that closes, a bad number) and the gate's three refusals.
#include <cmath>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/select/risefall.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace {
double num(const std::string& s) { return s == "nan" ? NAN : std::stod(s); }
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/burst_stages_golden.txt");
  if (!in) { std::cout << "no golden\n"; return 1; }
  std::string ln;
  double fs = 0, cyc = 0;
  std::getline(in, ln);
  { std::istringstream s(ln); std::string k; s >> k >> fs >> k >> cyc; }
  std::string name;
  double t0 = 0, t1 = 0;
  Eigen::VectorXd a;
  int cases = 0;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag;
    s >> tag;
    if (tag == "case") { s >> name >> t0 >> t1; }
    else if (tag == "series") { std::vector<double> v; double x; while (s >> x) v.push_back(x); a = Eigen::Map<Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size())); }
    else if (tag == "fit") {
      std::string k, v;
      std::map<std::string, std::string> m;
      while (s >> k >> v) m[k] = v;
      std::string l1, l2, l3;
      std::getline(in, l1); std::getline(in, l2); std::getline(in, l3);
      auto two = [&](const std::string& l) { std::istringstream q(l); std::string t, x, y; q >> t >> x >> y; return std::make_pair(num(x), num(y)); };
      const auto on = two(l1), wi = two(l2), of = two(l3);
      nxr::risefall::Options o;
      o.model = nxr::risefall::Model::Quadrature; o.margin = 10 * cyc; o.smooth = cyc; o.tauMax = 10 * cyc;
      const auto F = nxr::risefall::fit(a, fs, t0, t1, o);
      const bool gate = nxr::risefall::passesGate(F, 0.7);
      std::printf("  %s: ok %d at_bound %d r2 %.4f (nxr %.4f), tOn %.4f (%.4f), tOff %.4f (%.4f), gate %d (nxr %s)\n", name.c_str(), F.ok, F.atBound, F.r2, num(m["r2"]), F.tOn, num(m["tOn"]), F.tOff, num(m["tOff"]), gate, m["gate"].c_str());
      EXPECT(F.ok == (m["ok"] == "1") && F.atBound == (m["at_bound"] == "1") && gate == (m["gate"] == "1"), name << ": the gate differs from nxr's");
      const bool well = name == "clean";      // a well posed fit agrees to 5 ms; the two that are not (a fall at its bound, a fit in noise) only to where the optimisers stop: the gate's decision is the test
      EXPECT(std::fabs(F.tOn - num(m["tOn"])) < (well ? 5e-3 : 5e-2) && std::fabs(F.tOff - num(m["tOff"])) < (well ? 5e-3 : 0.1), name << ": tOn / tOff differ from nxr's");
      EXPECT(std::fabs(F.r2 - num(m["r2"])) < (well ? 5e-3 : 5e-2), name << ": r2");
      // the stages of nxr's own numbers equal nxr's windows exactly (the arithmetic), and of ours agree to the fit's tolerance
      const auto S = nxr::risefall::stagesOf(num(m["tOn"]), num(m["tOff"]), num(m["tauR"]), num(m["tauF"]));
      EXPECT(S.valid && std::fabs(S.onset[0] - on.first) < 1e-12 && std::fabs(S.onset[1] - on.second) < 1e-12 && std::fabs(S.offset[0] - of.first) < 1e-12 && std::fabs(S.offset[1] - of.second) < 1e-12, name << ": onset / offset");
      EXPECT(S.hasWithin == std::isfinite(wi.first) && (!S.hasWithin || (std::fabs(S.within[0] - wi.first) < 1e-12 && std::fabs(S.within[1] - wi.second) < 1e-12)), name << ": within");
      const auto S2 = nxr::risefall::stagesOf(F.tOn, F.tOff, F.tauR, F.tauF);
      EXPECT(S2.valid && std::fabs(S2.onset[0] - on.first) < 2e-2 && std::fabs(S2.offset[1] - of.second) < 5e-2 && std::fabs(S2.onset[0] - F.buildup[0]) < 1e-12 && std::fabs(S2.offset[1] - F.decay[1]) < 1e-12, name << ": ours");
      ++cases;
    }
  }
  EXPECT(cases == 3, "three cases");
  // a plateau that closes: the fall's decay begins before the rise's ends: no within
  const auto c = nxr::risefall::stagesOf(10.0, 10.5, 0.2, 0.2);
  EXPECT(c.valid && !c.hasWithin && std::isnan(c.within[0]), "no plateau when onset and offset meet");
  EXPECT(!nxr::risefall::stagesOf(NAN, 1, 0.1, 0.1).valid && !nxr::risefall::stagesOf(1, 2, 0.0, 0.1).valid, "a bad number gives no stages");
  nxr::risefall::Fit f; f.ok = true; f.r2 = 0.9; f.atBound = false;
  EXPECT(nxr::risefall::passesGate(f, 0.7) && !nxr::risefall::passesGate(f, 0.95), "r2 gate");
  f.atBound = true;
  EXPECT(!nxr::risefall::passesGate(f, 0.7), "a tau at its bound is refused");
  f.atBound = false; f.ok = false;
  EXPECT(!nxr::risefall::passesGate(f, 0.0), "a fit that is not ok is refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
