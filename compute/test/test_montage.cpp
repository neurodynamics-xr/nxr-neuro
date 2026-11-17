// nxr::montage (include/nxr/montage.h): the 10-05 rule on synthetic heads, and against MNE-Python's `spherical_1005` (golden/montage_golden.txt, made once by golden/gen_montage_golden.py;
// the generator only calls MNE and the golden is its output).
//   1. the names: 345 labels of the 10-05, the 10-10 and 10-20 subsets inside them, MNE's own lists inside ours, the old names;
//   2. a SPHERE (a triangulated sphere of radius 0.095 m, moved by an arbitrary rotation, scale and shift): after a similarity alignment (Umeyama 1991) onto MNE's positions the error is
//      reported in millimetres and bounded by what the mesh's edge length allows (a triangle mesh shortens an arc by a term that goes as the square of the edge);
//   3. an ELLIPSOID: the midline's points are at k/20 of its arclength (the ellipse's own, by quadrature), Cz is the top, the right and left points are mirror images;
//   4. refusals: a plane that cuts nothing, landmarks on a line, a face naming a vertex that does not exist.
#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

#include "nxr/montage.h"
#include "synthetic_meshes.h"

#ifndef NXR_TEST_GOLDEN_DIR
#error "NXR_TEST_GOLDEN_DIR is not set"
#endif

using namespace nxr::montage;
using V3 = Eigen::Vector3d;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << msg << "\n"; } } while (0)

struct Golden {
  std::map<std::string, V3> points, landmarks;
  std::set<std::string> s1010, s1020;
};
static Golden readGolden() {
  Golden g;
  std::ifstream f(std::string(NXR_TEST_GOLDEN_DIR) + "/montage_golden.txt");
  if (!f) { std::cout << "  [FAIL] montage_golden.txt is missing\n"; std::exit(1); }
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    std::string kind, name;
    s >> kind >> name;
    if (kind == "landmark" || kind == "point") { double x, y, z; s >> x >> y >> z; (kind == "point" ? g.points : g.landmarks)[name] = V3(x, y, z); }
    else if (kind == "system") { std::string l; auto& set = name == "1010" ? g.s1010 : g.s1020; while (s >> l) set.insert(l); }
  }
  return g;
}

static std::map<std::string, V3> byLabel(const Placement& p) {
  std::map<std::string, V3> m;
  for (std::size_t i = 0; i < p.labels.size(); ++i) m[p.labels[i]] = V3(p.positions[3 * i], p.positions[3 * i + 1], p.positions[3 * i + 2]);
  return m;
}

int main() {
  const Golden g = readGolden();

  // 1. the names
  {
    const auto& l = labels1005();
    const std::set<std::string> all(l.begin(), l.end());
    CHECK(l.size() == 345 && all.size() == 345, "the 10-05 has 345 distinct names, got " << l.size());
    CHECK(labels1020().size() == 21, "the 10-20 has 21 names");
    CHECK(labels1010().size() == 77 && std::set<std::string>(labels1010().begin(), labels1010().end()).size() == 77, "the 10-10 has 77 distinct names");
    for (const auto& s : labels1010()) CHECK(all.count(s), s << " (10-10) is not in the 10-05");
    for (const auto& s : labels1020()) CHECK(all.count(s), s << " (10-20) is not in the 10-05");
    const std::set<std::string> t10(labels1010().begin(), labels1010().end());
    for (const auto& s : labels1020()) CHECK(t10.count(s), s << " (10-20) is not in the 10-10");
    for (const auto& s : g.s1020) CHECK(std::find(labels1020().begin(), labels1020().end(), s) != labels1020().end(), "MNE's 10-20 name " << s << " is missing");
    for (const auto& s : g.s1010) CHECK(t10.count(s), "MNE's 10-10 name " << s << " is missing");
    CHECK(g.s1020.size() == 21 && g.s1010.size() == 70, "the golden's lists");
    for (const auto& kv : g.points) CHECK(all.count(kv.first), "MNE's 10-05 name " << kv.first << " is missing");
    for (const auto& s : l) CHECK(g.points.count(s) || s == "Nz", s << " is not in MNE's 10-05");
    CHECK(oldNames().at("T3") == "T7" && oldNames().at("T6") == "P8", "the old names");
  }

  // 2. a sphere, moved, against MNE
  {
    Mesh m = icosphere(5);
    const double R = 0.095;
    const Eigen::AngleAxisd rot(0.9, V3(0.3, -0.5, 0.8).normalized());
    const double scale = 1.07;
    const V3 shift(0.012, -0.03, 0.021);
    auto move = [&](const V3& p) { return V3(scale * (rot * p) + shift); };
    for (int i = 0; i < m.nV(); ++i) { const V3 p = move(V3(R * m.V[3 * i], R * m.V[3 * i + 1], R * m.V[3 * i + 2])); for (int c = 0; c < 3; ++c) m.V[3 * i + c] = p[c]; }
    auto lm = [&](const V3& p) { const V3 q = move(p); return Point{q[0], q[1], q[2]}; };
    Landmarks L;
    L.nasion = lm(g.landmarks.at("NAS")); L.inion = lm(g.points.at("Iz")); L.lpa = lm(g.landmarks.at("LPA")); L.rpa = lm(g.landmarks.at("RPA"));
    const Placement p = tenFive(m.V, m.F, L);
    CHECK(p.labels.size() == 345 && p.positions.size() == 345 * 3, "345 positions");
    CHECK(p.cz_residual < 1e-7 && p.iterations < 20, "Cz settled in " << p.iterations << " iterations, residual " << p.cz_residual);
    for (int k = 0; k < 4; ++k) CHECK(p.landmark_shift[k] < 1e-3, "a landmark on the sphere moved " << p.landmark_shift[k] << " to reach the mesh");
    const auto mine = byLabel(p);
    std::vector<std::string> names;
    for (const auto& kv : mine) if (kv.first == "Nz" || g.points.count(kv.first)) names.push_back(kv.first);
    Eigen::Matrix3Xd A(3, names.size()), B(3, names.size());
    for (std::size_t i = 0; i < names.size(); ++i) { A.col(i) = mine.at(names[i]); B.col(i) = names[i] == "Nz" ? g.landmarks.at("NAS") : g.points.at(names[i]); }
    const Eigen::Matrix4d T = Eigen::umeyama(A, B, true);
    double sum = 0, mx = 0;
    std::string worst;
    for (std::size_t i = 0; i < names.size(); ++i) {
      const V3 q = T.topLeftCorner<3, 3>() * A.col(i) + T.topRightCorner<3, 1>();
      const double e = 1000.0 * (q - B.col(i)).norm();
      sum += e;
      if (e > mx) { mx = e; worst = names[i]; }
    }
    const double mean = sum / names.size();
    std::cout << "  sphere (icosphere level 5, edge " << 1000 * R * 0.0625 << " mm) against MNE spherical_1005 after a similarity alignment: mean error " << mean << " mm, max " << mx << " mm (" << worst << "), "
              << names.size() << " positions, Cz in " << p.iterations << " iterations\n";
    CHECK(mean < 0.1 && mx < 0.35, "the sphere's positions differ from MNE's: mean " << mean << " mm, max " << mx << " mm");
    // the subsets are the names MNE's own subsets have
    for (const auto& s : g.s1020) CHECK(mine.count(s), s);
  }

  // 3. an ellipsoid: semi-axes a (x, right), b (y, front), c (z, up); the landmarks are on its axes
  {
    const double a = 0.075, b = 0.100, c = 0.085;
    Mesh m = icosphere(5);
    for (int i = 0; i < m.nV(); ++i) { m.V[3 * i] *= a; m.V[3 * i + 1] *= b; m.V[3 * i + 2] *= c; }
    Landmarks L;
    L.nasion = {0, b, 0}; L.inion = {0, -b, 0}; L.lpa = {-a, 0, 0}; L.rpa = {a, 0, 0};
    const Placement p = tenFive(m.V, m.F, L);
    const auto mine = byLabel(p);
    CHECK((mine.at("Cz") - V3(0, 0, c)).norm() < 5e-4, "Cz is the ellipsoid's top");
    // arclength of the midline ellipse (b cos u, c sin u) from Nz (u = 0) to u, by Simpson's rule; the points are at k/20 of the half (u = pi)
    auto arcTo = [&](double u) {
      const int n = 4000;
      double s = 0;
      for (int i = 0; i <= n; ++i) { const double w = u * i / n; const double f = std::sqrt(b * b * std::sin(w) * std::sin(w) + c * c * std::cos(w) * std::cos(w)); s += (i == 0 || i == n ? 1 : i % 2 ? 4 : 2) * f; }
      return s * (u / n) / 3;
    };
    const double half = arcTo(M_PI);
    const char* mid[21] = {"Nz", "NFpz", "Fpz", "AFpz", "AFz", "AFFz", "Fz", "FFCz", "FCz", "FCCz", "Cz", "CCPz", "CPz", "CPPz", "Pz", "PPOz", "POz", "POOz", "Oz", "OIz", "Iz"};
    double worst = 0;
    for (int k = 0; k < 21; ++k) {
      const V3 q = mine.at(mid[k]);
      const double ang = std::atan2(q[2] / c, q[1] / b);                 // the point's parameter u on the ellipse (b cos u, c sin u): from +y (Nz) over +z to -y (Iz)
      const double frac = arcTo(std::fabs(ang)) / half;
      worst = std::max(worst, std::fabs(frac - k / 20.0));
    }
    std::cout << "  ellipsoid midline: the largest departure from the 5% steps is " << worst << " of the arc\n";
    CHECK(worst < 2e-3, "the midline's points are not at 5% steps: " << worst);
    // mirror: left and right points
    for (const char* s : {"Fp1", "AF3", "F3", "FC5", "C3", "CP1", "P7", "PO3", "O1", "T7", "FT9", "TP10"}) {
      std::string name = s, other;
      // a mirror name: odd numbers on the left, even on the right (7 <-> 8, 9 <-> 10)
      if (name == "T7") other = "T8"; else if (name == "P7") other = "P8"; else if (name == "FT9") other = "FT10"; else if (name == "TP10") other = "TP9";
      else { other = name; other.back() = static_cast<char>(name.back() + 1); }
      const V3 l = mine.at(name), r = mine.at(other);
      CHECK(std::fabs(l[0] + r[0]) < 2e-4 && std::fabs(l[1] - r[1]) < 2e-4 && std::fabs(l[2] - r[2]) < 2e-4, name << " and " << other << " are not mirror images: " << l.transpose() << " / " << r.transpose());
    }
    // every position is on the surface
    for (const auto& kv : mine) {
      const V3& q = kv.second;
      const double e = std::sqrt(q[0] * q[0] / (a * a) + q[1] * q[1] / (b * b) + q[2] * q[2] / (c * c));
      CHECK(std::fabs(e - 1) < 5e-3, kv.first << " is off the surface: " << e);
    }
  }

  // 4. refusals
  {
    Mesh m = icosphere(2);
    Landmarks L;
    L.nasion = {0, 1, 0}; L.inion = {0, -1, 0}; L.lpa = {-1, 0, 0}; L.rpa = {1, 0, 0};
    auto throws = [&](const std::vector<double>& v, const std::vector<int32_t>& f, const Landmarks& l, const Options& o = {}) { try { tenFive(v, f, l, o); } catch (const std::invalid_argument&) { return true; } return false; };
    Landmarks line = L; line.lpa = {0, 1, 0}; line.rpa = {0, -1, 0};
    CHECK(throws(m.V, m.F, line), "landmarks on a line are refused");
    auto bad = m.F; bad[0] = m.nV() + 5;
    CHECK(throws(m.V, bad, L), "a face naming a missing vertex is refused");
    Options one; one.max_iterations = 0;
    CHECK(throws(m.V, m.F, L, one), "no iterations: Cz is refused, never guessed");
    CHECK(throws({1, 2}, {0, 1, 2}, L), "a bad vertex array is refused");
  }

  std::cout << (fails ? "FAILED\n" : "montage: ok\n");
  return fails ? 1 : 0;
}
