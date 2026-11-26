// nxr::geometry::pointset: the triangulation of a point set on a cap (synthetic points only).
//   planarDelaunay: every triangle counter-clockwise, its circumcircle empty of the other points, 2n - 2 - h triangles (h the hull's points); refusals.
//   pointSetMesh on a spherical cap and on an ellipsoidal helmet (jittered spirals): every point a vertex, one boundary loop, Euler characteristic 1,
//   every edge on one or two triangles and each interior edge once each way (consistently oriented), no degenerate triangle, normals outward;
//   the options: as_triangulated keeps the planar orientation, a pole along -z on the cap turned upside down gives the same triangles; refusals.
#include "nxr/geometry/point_set_mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace ps = nxr::geometry::pointset;

namespace {

struct Lcg {   // a fixed pseudo-random sequence in [0, 1)
  std::uint64_t s;
  double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; }
};

// n points on a cap of the ellipsoid (a, b, c) up to the polar angle `top` from +z: a Fibonacci spiral, jittered so that no four are cocircular
std::vector<double> cap_points(int n, double a, double b, double c, double top, std::uint64_t seed) {
  Lcg r{seed};
  std::vector<double> p;
  const double golden = M_PI * (3.0 - std::sqrt(5.0)), z0 = std::cos(top);
  for (int i = 0; i < n; ++i) {
    const double z = 1.0 - (1.0 - z0) * (i + 0.5) / n + 0.002 * (r.next() - 0.5);
    const double rho = std::sqrt(std::max(0.0, 1.0 - z * z)), phi = golden * i + 0.05 * (r.next() - 0.5);
    p.push_back(a * rho * std::cos(phi)); p.push_back(b * rho * std::sin(phi)); p.push_back(c * z);
  }
  return p;
}

bool refuses(const std::function<void()>& f) {
  try { f(); } catch (const std::invalid_argument&) { return true; }
  return false;
}

void check_mesh(const std::string& name, const std::vector<double>& V, const ps::PointSetMesh& m, bool outward) {
  const std::int64_t n = static_cast<std::int64_t>(V.size() / 3), F = static_cast<std::int64_t>(m.faces.size() / 3);
  std::cout << "  " << name << ": " << n << " points, " << F << " triangles (Delaunay " << m.delaunay_triangles << ", cut " << m.cut_triangles << ", pinches " << m.pinches_closed
            << ", fans dropped " << m.fans_dropped << ", hole triangles " << m.hole_triangles << "), loops " << m.boundary_loops << ", chi " << m.euler_characteristic << "\n";
  EXPECT(m.points_without_face.empty(), name + ": every point is a vertex");
  EXPECT(m.boundary_loops == 1, name + ": one boundary loop (a disc)");
  EXPECT(m.euler_characteristic == 1, name + ": Euler characteristic 1");
  std::map<std::pair<int, int>, int> directed;
  double c[3] = {0, 0, 0};
  for (std::int64_t i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) c[k] += V[static_cast<std::size_t>(3 * i + k)] / static_cast<double>(n);
  double min_area = 1e300, mean_area = 0;
  int inward = 0;
  for (std::int64_t f = 0; f < F; ++f) {
    const int t[3] = {m.faces[static_cast<std::size_t>(3 * f)], m.faces[static_cast<std::size_t>(3 * f + 1)], m.faces[static_cast<std::size_t>(3 * f + 2)]};
    for (int k = 0; k < 3; ++k) ++directed[{t[k], t[(k + 1) % 3]}];
    double a[3], b[3], nr[3], mid[3];
    for (int k = 0; k < 3; ++k) {
      a[k] = V[static_cast<std::size_t>(3 * t[1] + k)] - V[static_cast<std::size_t>(3 * t[0] + k)];
      b[k] = V[static_cast<std::size_t>(3 * t[2] + k)] - V[static_cast<std::size_t>(3 * t[0] + k)];
      mid[k] = (V[static_cast<std::size_t>(3 * t[0] + k)] + V[static_cast<std::size_t>(3 * t[1] + k)] + V[static_cast<std::size_t>(3 * t[2] + k)]) / 3.0 - c[k];
    }
    nr[0] = a[1] * b[2] - a[2] * b[1]; nr[1] = a[2] * b[0] - a[0] * b[2]; nr[2] = a[0] * b[1] - a[1] * b[0];
    const double area = 0.5 * std::sqrt(nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]);
    min_area = std::min(min_area, area); mean_area += area / static_cast<double>(F);
    inward += nr[0] * mid[0] + nr[1] * mid[1] + nr[2] * mid[2] <= 0;
  }
  int twice_same_way = 0, unmatched = 0;
  for (const auto& [e, k] : directed) { twice_same_way += k > 1; unmatched += !directed.count({e.second, e.first}); }
  EXPECT(twice_same_way == 0, name + ": no edge twice in one direction (consistent orientation, a manifold)");
  EXPECT(unmatched > 0, name + ": the rim's edges have one triangle");
  EXPECT(min_area > 1e-3 * mean_area, name + ": no degenerate triangle (smallest area " + std::to_string(min_area / mean_area) + " of the mean)");
  if (outward) EXPECT(inward == 0, name + ": every normal points away from the centroid");
}

}  // namespace

int main() {
  std::cout << "=== planarDelaunay ===\n";
  {
    Lcg r{7};
    std::vector<double> xy;
    const int n = 200;
    for (int i = 0; i < n; ++i) { xy.push_back(r.next()); xy.push_back(r.next()); }
    const auto T = ps::planarDelaunay(xy);
    const std::size_t nt = T.size() / 3;
    int ccw = 0, empty = 0;
    for (std::size_t t = 0; t < nt; ++t) {
      const int a = T[3 * t], b = T[3 * t + 1], c = T[3 * t + 2];
      const double ax = xy[2 * a], ay = xy[2 * a + 1], bx = xy[2 * b], by = xy[2 * b + 1], cx = xy[2 * c], cy = xy[2 * c + 1];
      ccw += (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0;
      // the circumcircle
      const double d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
      const double ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
      const double uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
      const double r2 = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
      bool ok = true;
      for (int q = 0; q < n && ok; ++q) {
        if (q == a || q == b || q == c) continue;
        ok = (xy[2 * q] - ux) * (xy[2 * q] - ux) + (xy[2 * q + 1] - uy) * (xy[2 * q + 1] - uy) > r2 * (1 - 1e-9);
      }
      empty += ok;
    }
    // the hull's point count (Andrew's monotone chain), for 2n - 2 - h
    std::vector<std::pair<double, double>> P;
    for (int i = 0; i < n; ++i) P.emplace_back(xy[2 * i], xy[2 * i + 1]);
    std::sort(P.begin(), P.end());
    auto cross = [](const auto& o, const auto& a, const auto& b) { return (a.first - o.first) * (b.second - o.second) - (a.second - o.second) * (b.first - o.first); };
    std::vector<std::pair<double, double>> H(2 * P.size());
    std::size_t k = 0;
    for (std::size_t i = 0; i < P.size(); ++i) { while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0) --k; H[k++] = P[i]; }
    for (std::size_t i = P.size() - 1, t = k + 1; i > 0; --i) { while (k >= t && cross(H[k - 2], H[k - 1], P[i - 1]) <= 0) --k; H[k++] = P[i - 1]; }
    const std::size_t h = k - 1;
    std::cout << "  " << n << " random points: " << nt << " triangles (2n - 2 - h = " << 2 * n - 2 - h << "), counter-clockwise " << ccw << ", empty circumcircles " << empty << "\n";
    EXPECT(nt == 2 * n - 2 - h, "2n - 2 - h triangles");
    EXPECT(ccw == static_cast<int>(nt), "every triangle counter-clockwise");
    EXPECT(empty == static_cast<int>(nt), "every circumcircle empty (Delaunay)");
    EXPECT(refuses([] { ps::planarDelaunay({0, 0, 1, 1, 2, 2, 3, 3}); }), "points on one line are refused");
    EXPECT(refuses([] { ps::planarDelaunay({0, 0, 1, 0, 0, 1, 1, 0}); }), "coincident points are refused");
    // a collinear start (the first points in order on one line) is fanned from the first point off it
    const auto C = ps::planarDelaunay({0, 0, 1, 0, 2, 0, 3, 0, 1.5, 1, 1.4, -1.2});
    EXPECT(C.size() / 3 == 6, "a collinear start: 2n - 2 - h = 6 triangles (4 points on the hull)");
  }

  std::cout << "=== pointSetMesh ===\n";
  {
    const auto cap = cap_points(160, 0.1, 0.1, 0.1, 1.75, 11);                 // a spherical cap to 100 degrees (a helmet's span)
    check_mesh("spherical cap", cap, ps::pointSetMesh(cap), true);
    const auto helmet = cap_points(270, 0.09, 0.11, 0.12, 1.6, 23);           // an ellipsoidal helmet, as many points as a whole-head MEG
    const ps::PointSetMesh hm = ps::pointSetMesh(helmet);
    check_mesh("ellipsoidal helmet", helmet, hm, true);
    EXPECT(hm.median_edge > 0.005 && hm.median_edge < 0.05, "the median edge is in the points' unit");

    // as_triangulated: the same triangles (as vertex sets), the planar orientation (counter-clockwise seen from the pole: also outward on a cap about +z)
    ps::PointSetMeshOptions o;
    o.orientation = ps::Orientation::as_triangulated;
    const ps::PointSetMesh hp = ps::pointSetMesh(helmet, o);
    auto as_sets = [](const std::vector<std::int32_t>& f) {
      std::set<std::vector<std::int32_t>> s;
      for (std::size_t t = 0; t < f.size() / 3; ++t) { std::vector<std::int32_t> v{f[3 * t], f[3 * t + 1], f[3 * t + 2]}; std::sort(v.begin(), v.end()); s.insert(v); }
      return s;
    };
    EXPECT(as_sets(hp.faces) == as_sets(hm.faces), "the orientation option changes no triangle");
    // the pole: the cap turned upside down (z -> -z, y -> -y: a rotation) with the pole along -z gives the same triangles
    std::vector<double> flipped = helmet;
    for (std::size_t i = 0; i < flipped.size(); i += 3) { flipped[i + 1] = -flipped[i + 1]; flipped[i + 2] = -flipped[i + 2]; }
    ps::PointSetMeshOptions down;
    down.pole = {{0.0, 0.0, -2.0}};
    const ps::PointSetMesh fm = ps::pointSetMesh(flipped, down);
    check_mesh("helmet upside down, pole -z", flipped, fm, true);
    EXPECT(as_sets(fm.faces) == as_sets(hm.faces), "a rotated cap with its pole rotated gives the same triangles");
    // the cut: a larger cut keeps more of the rim's chords
    ps::PointSetMeshOptions wide;
    wide.cut = 4.0;
    EXPECT(ps::pointSetMesh(helmet, wide).cut_triangles <= hm.cut_triangles, "a larger cut removes no more triangles");

    EXPECT(refuses([] { ps::pointSetMesh({0, 0, 1, 1, 0, 1}); }), "fewer than 3 points are refused");
    EXPECT(refuses([&] { auto v = helmet; v[4] = std::nan(""); ps::pointSetMesh(v); }), "a non-finite coordinate is refused");
    EXPECT(refuses([&] { ps::PointSetMeshOptions z; z.pole = {{0, 0, 0}}; ps::pointSetMesh(helmet, z); }), "a zero pole is refused");
    EXPECT(refuses([&] { ps::PointSetMeshOptions z; z.cut = 0; ps::pointSetMesh(helmet, z); }), "a cut of 0 is refused");
  }
  std::cout << (fails ? "FAILED " : "OK ") << fails << "\n";
  return fails ? 1 : 0;
}
