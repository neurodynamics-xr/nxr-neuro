// isolines of a scalar field on a 642-vertex sphere (icosphere level 3), 20 levels, the default method. Synthetic data, fixed field.
#include "bench.h"
#include "nxr/signal/surface_calculus.h"
#include "synthetic_meshes.h"
#include <cmath>

int main() {
  const Mesh m = icosphere(3);   // 642 vertices (compute/test/synthetic_meshes.h)
  const nxr::geometry::SurfaceArrays s = nxr::geometry::surfaceArrays(m.V.data(), m.nV(), m.F.data(), m.nF(), nxr::geometry::intrinsicOptions(false));
  Eigen::VectorXd f(m.nV());
  for (int v = 0; v < m.nV(); ++v) f(v) = m.V[3 * v + 2] + 0.3 * std::sin(7 * m.V[3 * v]);
  nxr_bench::run("isolines_sphere", "vertices=642 levels=20 method=per_triangle", 20, [&] {
    const auto r = nxr::signal::isoline(s, f, 20, 0.0, 0.0, nxr::signal::IsolineMethod::PerTriangle);
    nxr_bench::keep(r.segmentCount);
  });
  return 0;
}
