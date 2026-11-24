// nxr::meeg::channelPositions: Brainstorm's rule on synthetic channels (an axial gradiometer of eight points, a planar gradiometer of two, a magnetometer of one,
// an electrode without coils, coils stored out of channel order) and the refusals.
#include "nxr/meeg/channels.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

static bool refuses(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

int main() {
  // channel 0: 8 points, the first four at z = 1 (x = 0, 1, 2, 3), the reference four at z = 5; channel 1: 2 points; channel 2: 1 point; channel 3: none
  std::vector<double> coils;
  for (int j = 0; j < 4; ++j) { coils.push_back(j); coils.push_back(0); coils.push_back(1); }
  for (int j = 0; j < 4; ++j) { coils.push_back(j); coils.push_back(0); coils.push_back(5); }
  coils.insert(coils.end(), {10, 0, 0, 12, 0, 0});        // coils 8, 9
  coils.insert(coils.end(), {7, 7, 7});                   // coil 10
  const std::vector<double> points = {0, 0, 1, 10, 0, 0, 7, 7, 7, -1, -2, -3};
  const std::vector<std::int32_t> indptr = {0, 8, 10, 11, 11};
  std::vector<std::int32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const auto P = nxr::meeg::channelPositions(points, coils, indptr, indices, 4);
  EXPECT(std::abs(P[0] - 1.5) < 1e-12 && P[1] == 0 && std::abs(P[2] - 1) < 1e-12, "an axial gradiometer: the mean of its first four points (the pickup coil)");
  EXPECT(std::abs(P[3] - 11) < 1e-12, "a planar gradiometer: the mean of its two points");
  EXPECT(P[6] == 7 && P[7] == 7 && P[8] == 7, "a magnetometer: its one point");
  EXPECT(P[9] == -1 && P[10] == -2 && P[11] == -3, "a channel without coils keeps its stored point");
  const auto all8 = nxr::meeg::channelPositions(points, coils, indptr, indices, 8);
  EXPECT(std::abs(all8[2] - 3) < 1e-12, "coilPoints 8: the mean of all eight points");
  // coils stored in another order: channel 0's are listed through the indices
  std::vector<std::int32_t> shuffled = {7, 6, 5, 4, 0, 1, 2, 3, 8, 9, 10};
  const auto S = nxr::meeg::channelPositions(points, coils, indptr, shuffled, 4);
  EXPECT(std::abs(S[2] - 5) < 1e-12, "the first four coils are the CSR row's first four indices");
  EXPECT(nxr::meeg::channelPositions(points, {}, {}, {}) == points, "no coils: every channel keeps its point");
  EXPECT(refuses([&] { nxr::meeg::channelPositions(points, coils, {0, 8, 10}, indices); }), "an indptr of the wrong length is refused");
  EXPECT(refuses([&] { nxr::meeg::channelPositions(points, coils, indptr, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99}); }), "a coil index outside the coils is refused");
  EXPECT(refuses([&] { nxr::meeg::channelPositions(points, coils, indptr, indices, 0); }), "coilPoints 0 is refused");
  std::cout << (fails ? "FAILED " : "OK ") << fails << "\n";
  return fails ? 1 : 0;
}
