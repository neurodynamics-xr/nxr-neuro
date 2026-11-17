// The distance gate: a fiber with an end farther than the gate is left out; `none` (keepAll) leaves none out; a gate between two distances separates exactly those fibers;
// the distances come out of nearestVertices either way (stored, so a changed gate needs no new ends).
#include <cmath>
#include <iostream>

#include "connectome_helpers.h"
#include "nxr/connectome.h"
#include "nxr/nearest.h"

using namespace nxr::connectome;

int main() {
  std::vector<double> vertexPositions; std::vector<int32_t> triangles;
  icosphere(2, vertexPositions, triangles, 0.08);
  // fiber f has its first end `d(f)` above vertex f along the outward normal, its last end 0.3 mm above vertex 100 + f: the nearest vertex is that vertex and the distance is d
  const std::vector<double> offsets_mm = {0.5, 2, 4.9, 5.1, 7, 12};   // first ends
  const int nF = static_cast<int>(offsets_mm.size());
  std::vector<double> ends(static_cast<std::size_t>(6 * nF));
  auto above = [&](int v, double mm, double* out) {
    double n = 0;
    for (int k = 0; k < 3; ++k) n += vertexPositions[static_cast<std::size_t>(3 * v + k)] * vertexPositions[static_cast<std::size_t>(3 * v + k)];
    n = std::sqrt(n);
    for (int k = 0; k < 3; ++k) out[k] = vertexPositions[static_cast<std::size_t>(3 * v + k)] * (1 + mm * 1e-3 / n);
  };
  for (int f = 0; f < nF; ++f) { above(f, offsets_mm[static_cast<std::size_t>(f)], &ends[static_cast<std::size_t>(3 * f)]); above(100 + f, 0.3, &ends[static_cast<std::size_t>(3 * (nF + f))]); }
  const auto near = nxr::nearest::nearestVertices<double>(vertexPositions, ends);
  bool right = true;
  for (int f = 0; f < nF; ++f) right = right && near.vertex[static_cast<std::size_t>(f)] == f && near.vertex[static_cast<std::size_t>(nF + f)] == 100 + f && std::abs(near.distance[static_cast<std::size_t>(f)] - offsets_mm[static_cast<std::size_t>(f)] * 1e-3) < 1e-12;
  EXPECT(right, "each end is above its vertex at the distance made: the distances are stored whatever the gate");

  auto kept = [&](double mm) { return withinGate<double>(near.distance, mm * 1e-3); };
  EXPECT((kept(5) == std::vector<std::uint8_t>{1, 1, 1, 0, 0, 0}), "the default gate, 5 mm: leaves out exactly the fibers with an end farther than 5 mm");
  EXPECT((kept(5.0) == kept(5.05)) && (kept(4.5) == std::vector<std::uint8_t>{1, 1, 0, 0, 0, 0}), "a gate between two distances separates exactly those fibers");
  EXPECT((kept(6) == std::vector<std::uint8_t>{1, 1, 1, 1, 0, 0}) && kept(100) == keepAll(nF), "a wider gate keeps more; a gate wider than every distance keeps all");
  EXPECT(keepAll(nF) == std::vector<std::uint8_t>(6, 1), "`none` (keepAll) leaves none out");
  std::int64_t left_out = 0;
  for (auto k : kept(5)) left_out += !k;
  EXPECT(left_out == 3, "the number left out, of nF, for the report");
  // either end can leave a fiber out
  std::vector<double> d = {1e-3, 1e-3, 9e-3, 1e-3};   // fibers 0 (ends 1,9) , 1 (ends 1, 1)
  EXPECT((withinGate<double>(d, 5e-3) == std::vector<std::uint8_t>{0, 1}), "a fiber is kept only when both ends are within the gate");
  // the counts change with the gate and equal the counts of the kept fibers alone
  const auto all = endConnectome<double>(near.vertex, {}, 162), gated = endConnectome<double>(near.vertex, kept(5), 162);
  EXPECT(gated.nonZeros() == 6 && all.nonZeros() == 12, "the gate leaves out the fibers' counts too");
  bool nan_out = true;
  { std::vector<double> dn = {std::nan(""), 0.0}; nan_out = withinGate<double>(dn, 1.0)[0] == 0; }
  EXPECT(nan_out, "a NaN distance is not within the gate");
  bool refusedNeg = false;
  try { withinGate<double>(d, -1); } catch (const std::invalid_argument&) { refusedNeg = true; }
  EXPECT(refusedNeg, "a negative gate is refused");
  std::cout << (g_failures ? "FAILED\n" : "all gate tests passed\n");
  return g_failures ? 1 : 0;
}
