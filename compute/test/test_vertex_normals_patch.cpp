// nxr::geometry::vertexNormalsPatch against Brainstorm's tess_normals of the release of 2018-10-12 (golden/gen_vertex_normals_golden.m: an uneven convex mesh); and that it differs from the
// corner-angle normals (nxr::geometry::vertexNormals) in direction, not only in sign. Error = max |a - b| (unit vectors).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/geometry/frames.h"

int main() {
  std::ifstream f(std::string(NXR_TEST_GOLDEN_DIR) + "/vertex_normals_golden.txt");
  if (!f) { std::printf("FAIL test_vertex_normals_patch: no golden\n"); return 1; }
  std::map<std::string, std::vector<double>> g;
  std::map<std::string, int> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream h(line);
    std::string name; int r, c;
    h >> name >> r >> c;
    std::getline(f, line);
    std::istringstream v(line);
    std::vector<double> x(static_cast<std::size_t>(r * c));
    for (auto& e : x) v >> e;
    g[name] = x; rows[name] = r;
  }
  const int nV = rows["V"], nF = rows["F"];
  std::vector<std::int32_t> F(g["F"].begin(), g["F"].end());
  const nxr::geometry::Mat3 N = nxr::geometry::vertexNormalsPatch(g["V"].data(), nV, F.data(), nF);
  const nxr::geometry::Mat3 C = nxr::geometry::vertexNormals(g["V"].data(), nV, F.data(), nF);
  double err = 0, apart = 0;
  for (int i = 0; i < nV; ++i)
    for (int k = 0; k < 3; ++k) {
      err = std::max(err, std::abs(N(i, k) - g["N"][static_cast<std::size_t>(3 * i + k)]));
      apart = std::max(apart, std::min(std::abs(N(i, k) - C(i, k)), std::abs(N(i, k) + C(i, k))));
    }
  std::printf("  Brainstorm 2018 tess_normals: %.2e; corner-angle normals differ by up to %.2e (sign aside)\n", err, apart);
  int failed = 0;
  if (!(err < 1e-6)) { std::printf("  CHECK failed: err < 1e-6 (MATLAB's patch computes in single precision)\n"); ++failed; }
  if (!(apart > 1e-3)) { std::printf("  CHECK failed: the weighting changes the direction\n"); ++failed; }
  std::printf(failed ? "FAIL test_vertex_normals_patch\n" : "PASS test_vertex_normals_patch\n");
  return failed ? 1 : 0;
}
