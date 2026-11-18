// nxr::signal against nxr's phase reader (golden: test/golden/gen_phase_golden.py, from nxr/detect/core_phase.py's `gradient_and_kernel` GV and `read_out`'s
// formulas): the phase gradient and the velocity of a random complex field on a bumped icosphere (642 vertices, mm), three frames. Then what the maths
// promises, on a sphere (radius 50 mm):
//   TRAVELLING  z = e^{-i m λ} (m = 2, λ the longitude): ∇φ along e_λ with |∇φ| = m/ρ, the speed ω ρ/m, the travelling index 1;
//   STANDING    z = cos(m λ): the travelling index 0;
//   A VECTOR    J = e^{-i m λ} n (the current along the normal, which turns over the sphere): the same phase gradient as the scalar (the turning orientation
//               does not leak into the phase: d·∇d = 0);
//   POINTWISE   every value depends on its own frame only (a frame computed alone equals it computed with others);
// the phase current lies on the tangent plane; refusals.
#include "nxr/signal/phase_gradient.h"

#include <algorithm>
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

static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

using namespace nxr::signal;
using Faces = Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

static std::map<std::string, std::vector<double>> load(const std::string& path, int& nV, int& nF) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("no golden at " + path);
  std::map<std::string, std::vector<double>> r;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag;
    s >> tag;
    if (tag == "mesh") { s >> nV >> nF; continue; }
    std::size_t n = 0;
    s >> n;
    std::vector<double> v(n);
    for (auto& x : v) { std::string w; s >> w; x = std::strtod(w.c_str(), nullptr); }
    r[tag] = std::move(v);
  }
  return r;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? NAN : v[v.size() / 2]; }

int main() {
  int nV = 0, nF = 0;
  const auto g = load(kDir + "/phase_golden.txt", nV, nF);
  Eigen::MatrixXd P(nV, 3), N(nV, 3);
  Faces F(nF, 3);
  for (int v = 0; v < nV; ++v) for (int d = 0; d < 3; ++d) { P(v, d) = g.at("positions")[3 * v + d]; N(v, d) = g.at("normals")[3 * v + d]; }
  for (int f = 0; f < nF; ++f) for (int s = 0; s < 3; ++s) F(f, s) = static_cast<int32_t>(g.at("faces")[3 * f + s]);
  const Gradient G = vertexGradient(P, F);
  const int m = 3;
  Eigen::MatrixXcd Z(nV, m);
  for (int v = 0; v < nV; ++v) for (int k = 0; k < m; ++k) Z(v, k) = {g.at("re")[v * m + k], g.at("im")[v * m + k]};

  // AGAINST nxr: the phase gradient and the velocity (golden layout [V x m x 3])
  const Eigen::MatrixXd j = phaseCurrent(Z, 1, G, N), p = power(Z, 1), grad = phaseGradient(j, p);
  const Velocity u = phaseVelocity(grad, 2 * M_PI * 10.0);
  double eg = 0, eu = 0, sg = 0, su = 0;
  for (int v = 0; v < nV; ++v) for (int k = 0; k < m; ++k) for (int d = 0; d < 3; ++d) {
    const std::size_t i = static_cast<std::size_t>((v * m + k) * 3 + d);
    eg = std::max(eg, std::fabs(grad(3 * v + d, k) - g.at("grad_phi")[i])); sg = std::max(sg, std::fabs(g.at("grad_phi")[i]));
    eu = std::max(eu, std::fabs(u.velocity(3 * v + d, k) - g.at("velocity")[i])); su = std::max(su, std::fabs(g.at("velocity")[i]));
  }
  std::printf("  against nxr's core_phase: phase gradient %.1e, velocity %.1e (relative to the largest)\n", eg / sg, eu / su);
  EXPECT(eg / sg < 1e-10 && eu / su < 1e-10, "the phase gradient or velocity differs from nxr's");
  // the phase current lies on the tangent plane
  double off = 0, js = 0;
  for (int v = 0; v < nV; ++v) for (int k = 0; k < m; ++k) {
    off = std::max(off, std::fabs(j(3 * v, k) * N(v, 0) + j(3 * v + 1, k) * N(v, 1) + j(3 * v + 2, k) * N(v, 2)));
    js = std::max(js, std::sqrt(j(3 * v, k) * j(3 * v, k) + j(3 * v + 1, k) * j(3 * v + 1, k) + j(3 * v + 2, k) * j(3 * v + 2, k)));
  }
  EXPECT(off < 1e-12 * js, "the phase current has a normal part");
  // POINTWISE: frame 1 alone equals frame 1 among the three
  const Eigen::MatrixXcd one = Z.col(1);
  const Eigen::MatrixXd j1 = phaseCurrent(one, 1, G, N);
  EXPECT((j1.col(0) - j.col(1)).cwiseAbs().maxCoeff() == 0.0, "a frame's phase current depends on the other frames");

  // THE SPHERE (radius 50 mm, unbumped): waves round the z axis
  Eigen::MatrixXd S = P.rowwise().normalized() * 50.0, Ns = P.rowwise().normalized();
  const Gradient Gs = vertexGradient(S, F);
  const int mm = 2;
  const double omega = 2 * M_PI * 10.0;
  Eigen::MatrixXcd trav(nV, 1), stand(nV, 1), vec(3 * nV, 1);
  for (int v = 0; v < nV; ++v) {
    const double lam = std::atan2(S(v, 1), S(v, 0));
    trav(v, 0) = std::exp(std::complex<double>(0, -mm * lam));
    stand(v, 0) = std::cos(mm * lam);
    for (int d = 0; d < 3; ++d) vec(3 * v + d, 0) = trav(v, 0) * Ns(v, d);
  }
  const Eigen::MatrixXd jt = phaseCurrent(trav, 1, Gs, Ns), pt = power(trav, 1), qt = gradientPower(trav, 1, Gs, Ns);
  const Eigen::MatrixXd gt = phaseGradient(jt, pt), it = travellingIndex(jt, pt, qt);
  const Velocity ut = phaseVelocity(gt, omega);
  const Eigen::MatrixXd js_ = phaseCurrent(stand, 1, Gs, Ns), ps = power(stand, 1), qs = gradientPower(stand, 1, Gs, Ns);
  const Eigen::MatrixXd is = travellingIndex(js_, ps, qs);
  const Eigen::MatrixXd jv = phaseCurrent(vec, 3, Gs, Ns), pv = power(vec, 3), gv = phaseGradient(jv, pv);
  std::vector<double> espeed, cosang, itr, ist, evec;
  for (int v = 0; v < nV; ++v) {
    const double rho = std::hypot(S(v, 0), S(v, 1));
    if (rho < 25.0) continue;                                     // away from the poles (the phase's singularities)
    const Eigen::Vector3d el(-S(v, 1) / rho, S(v, 0) / rho, 0), uv(ut.velocity(3 * v, 0), ut.velocity(3 * v + 1, 0), ut.velocity(3 * v + 2, 0));
    espeed.push_back(std::fabs(ut.speed(v, 0) - omega * rho / mm) / (omega * rho / mm));
    cosang.push_back(uv.normalized().dot(el));
    itr.push_back(it(v, 0));
    if (std::isfinite(is(v, 0))) ist.push_back(is(v, 0));
    const Eigen::Vector3d a(gt(3 * v, 0), gt(3 * v + 1, 0), gt(3 * v + 2, 0)), b(gv(3 * v, 0), gv(3 * v + 1, 0), gv(3 * v + 2, 0));
    evec.push_back((a - b).norm() / a.norm());
  }
  std::printf("  sphere, %zu vertices off the poles: travelling wave speed error median %.3f, direction cos min %.4f, travelling index median %.3f;"
              " standing %.3f; a vector along the turning normal vs the scalar: median %.1e\n",
              espeed.size(), median(espeed), *std::min_element(cosang.begin(), cosang.end()), median(itr), median(ist), median(evec));
  EXPECT(median(espeed) < 0.03 && *std::min_element(cosang.begin(), cosang.end()) > 0.98, "the travelling wave's velocity is not ω ρ / m along e_λ");
  EXPECT(median(itr) > 0.95 && median(ist) < 0.1, "the travelling index does not tell a travelling wave from a standing one");
  EXPECT(median(evec) < 0.02, "a vector whose orientation turns does not give the scalar's phase gradient");

  // REFUSALS
  auto refused = [](auto&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  EXPECT(refused([&] { phaseCurrent(Z, 2, G, N); }), "rows not vertices x components: not refused");
  EXPECT(refused([&] { phaseGradient(j, p.topRows(10)); }), "mismatched shapes: not refused");

  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
