// nxr::signal::formOfFreeColumns (apply_kernel's `form`: the kernel's free currents → tangent part → curl_tan / div_tan / n·x, one linear operator;) against nxr's
// `Cortex.readings` lines (golden: test/golden/gen_curl_golden.py, nxr FaceOperators.div_curl(tangent(q)), on a jittered icosphere with three planted currents, a rigid rotation among them):
// the three forms equal nxr's (float32 inside, so 1e-4 of the largest value); a rigid rotation about z has curl_tan ≈ 2 ω cos(colatitude)/R of the sphere (sense: counterclockwise about +z is +);
// linearity (the form of a sum is the sum of the forms: the operator folds into the kernel); refusals.
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/signal/forms.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
double worst(const Eigen::MatrixXd& a, const std::vector<double>& ref) {   // ref: [nv × cols] row-major
  double w = 0, m = 0;
  for (Eigen::Index v = 0; v < a.rows(); ++v) for (Eigen::Index c = 0; c < a.cols(); ++c) m = std::max(m, std::fabs(ref[static_cast<std::size_t>(v * a.cols() + c)]));
  for (Eigen::Index v = 0; v < a.rows(); ++v) for (Eigen::Index c = 0; c < a.cols(); ++c) w = std::max(w, std::fabs(a(v, c) - ref[static_cast<std::size_t>(v * a.cols() + c)]));
  return w / m;
}
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/curl_golden.txt");
  if (!in) { std::cout << "no golden\n"; return 1; }
  int nv = 0, nf = 0, cols = 0;
  std::vector<double> pos, q, curl, div, nrm;
  std::vector<int> faces;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag;
    s >> tag;
    if (tag == "nv") { std::string a, b; s >> nv >> a >> nf >> b >> cols; }
    else if (tag == "positions") pos = nums(s);
    else if (tag == "faces") { for (double f : nums(s)) faces.push_back(static_cast<int>(f)); }
    else if (tag == "q") q = nums(s);
    else if (tag == "curl_tan") curl = nums(s);
    else if (tag == "div_tan") div = nums(s);
    else if (tag == "normal") nrm = nums(s);
  }
  std::vector<float> P(pos.begin(), pos.end());
  const auto op = nxr::signal::assembleFaceGradient(P.data(), faces.data(), nv, nf);
  Eigen::MatrixXd K(3 * nv, cols);
  for (int r = 0; r < 3 * nv; ++r) for (int c = 0; c < cols; ++c) K(r, c) = q[static_cast<std::size_t>(r * cols + c)];
  using nxr::signal::Form;
  const Eigen::MatrixXd C = nxr::signal::formOfFreeColumns(Form::curl_tan, K, op), D = nxr::signal::formOfFreeColumns(Form::div_tan, K, op), N = nxr::signal::formOfFreeColumns(Form::nx, K, op);
  const double ec = worst(C, curl), ed = worst(D, div), en = worst(N, nrm);
  std::printf("  vs nxr FaceOperators: curl_tan %.1e, div_tan %.1e, normal %.1e (of the largest value)\n", ec, ed, en);
  EXPECT(ec < 1e-4 && ed < 1e-4 && en < 1e-4, "a form differs from nxr's");
  // the planted rigid rotation about +z: ω = 1/R (1/s), a tangent flow with curl 2 ω cos(colatitude) along the normal; column 0 also holds a normal part (no curl in the tangent flow's curl)
  const double R = 0.08;
  double num = 0, den = 0;
  for (int v = 0; v < nv; ++v) {
    const double r = std::sqrt(pos[3 * v] * pos[3 * v] + pos[3 * v + 1] * pos[3 * v + 1] + pos[3 * v + 2] * pos[3 * v + 2]);
    const double expect = 2.0 / R * pos[3 * v + 2] / r;
    num += C(v, 0) * expect; den += expect * expect;
  }
  std::printf("  rigid rotation, curl_tan projected on 2 cos(colatitude)/R: %.3f\n", num / den);
  EXPECT(num / den > 0.8 && num / den < 1.2, "the rigid rotation's curl is not 2 cos(colatitude)/R of the right sign");
  // linear: form(K0 + 2 K1) = form(K0) + 2 form(K1)
  Eigen::MatrixXd Ks = K.col(0) + 2.0 * K.col(1);
  const Eigen::MatrixXd Cs = nxr::signal::formOfFreeColumns(Form::curl_tan, Ks, op);
  EXPECT((Cs.col(0) - (C.col(0) + 2.0 * C.col(1))).cwiseAbs().maxCoeff() < 1e-4 * C.cwiseAbs().maxCoeff(), "the form is not linear");
  // refusals: a non-linear form, a form of another type, a K of other rows
  auto refused = [&](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  EXPECT(refused([&] { nxr::signal::formOfFreeColumns(Form::absq, K, op); }), "absq not refused");
  EXPECT(refused([&] { nxr::signal::formOfFreeColumns(Form::xt, K, op); }), "xt (a vector form) not refused");
  EXPECT(refused([&] { nxr::signal::formOfFreeColumns(Form::curl_tan, K.topRows(3 * nv - 3), op); }), "wrong rows not refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
