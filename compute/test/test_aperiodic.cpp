// nxr::signal aperiodic against nxr's specparam port (golden aperiodic_golden.txt, gen_aperiodic_golden.py): the fit without and with the knee on a synthetic
// spectrum over 1-45 Hz (offset, exponent, knee, r², points used); evaluateAperiodic reproduces the model.
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/signal/aperiodic.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_aperiodic\n";
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/aperiodic_golden.txt");
  std::string line, key;
  std::getline(in, line);
  auto row = [&]() { std::getline(in, line); std::istringstream s(line); s >> key; std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; };
  const std::vector<double> fv = row(), pv = row();
  const Eigen::VectorXd f = Eigen::Map<const Eigen::VectorXd>(fv.data(), static_cast<Eigen::Index>(fv.size()));
  const Eigen::VectorXd P = Eigen::Map<const Eigen::VectorXd>(pv.data(), static_cast<Eigen::Index>(pv.size()));
  for (int k = 0; k < 2; ++k) {
    const std::vector<double> g = row();
    const nxr::signal::AperiodicFit a = nxr::signal::fitAperiodic(P, f, g[0] > 0.5, 3, 2.0, 1.0, 45.0);
    EXPECT(std::abs(a.offset - g[1]) < 1e-9 && std::abs(a.exponent - g[2]) < 1e-9 && std::abs(a.knee - g[3]) < 1e-9 && std::abs(a.r2 - g[4]) < 1e-9 && a.pointsUsed == static_cast<Eigen::Index>(g[5]),
           std::string(k ? "with" : "without") + " the knee: offset, exponent, knee, r², points used equal nxr's");
    const Eigen::VectorXd m = nxr::signal::evaluateAperiodic(a, f.head(3));
    EXPECT(std::abs(std::log10(m(1)) - (a.knee > 0 ? a.offset - std::log10(a.knee + std::pow(f(1), a.exponent)) : a.offset - a.exponent * std::log10(f(1)))) < 1e-12, "evaluateAperiodic: the model");
  }
  std::cout << (failures ? "FAILED\n" : "all passed\n");
  return failures ? 1 : 0;
}
