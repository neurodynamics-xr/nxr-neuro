// nxr::meeg forward models: the simplex, the sphere fit, Sarvas's field and the multi-shell EEG sphere, against what the maths requires (no reference program):
// a radial dipole makes no field; the radial field of a dipole in a sphere is the Biot-Savart field of the primary current alone; invariance under a rotation and a
// translation; the EEG series against the closed form of one shell, and the Berg-Scherg equivalent against the series.
#include <cmath>
#include <cstdio>
#include <random>

#include "check.h"
#include "nxr/meeg/forward.h"

using namespace nxr::meeg;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {
double relmax(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }
template <class F>
bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
}  // namespace

int main() {
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  const Vector3d c(0.003, -0.002, 0.045);

  // ---- the simplex: the Rosenbrock valley, then a budget too small
  {
    auto rosen = [](const VectorXd& x) { return 100 * std::pow(x(1) - x(0) * x(0), 2) + std::pow(1 - x(0), 2); };
    VectorXd x0(2);
    x0 << -1.2, 1;
    const nxr::optimisation::MinimiseResult r = nxr::optimisation::nelderMead(rosen, x0);
    CHECK(r.converged);
    CHECK((r.x - Vector3d(1, 1, 0).head<2>()).norm() < 1e-5);
    nxr::optimisation::NelderMeadOptions o;
    o.maxEvaluations = 10;
    CHECK(!nelderMead(rosen, x0, o).converged);
    CHECK(throws([&] { nxr::optimisation::nelderMead(rosen, VectorXd()); }));
  }
  // ---- the sphere fit
  {
    MatrixXd P(200, 3);
    for (int i = 0; i < 200; ++i) { Vector3d u(nd(rng), nd(rng), nd(rng)); P.row(i) = (Vector3d(0.01, -0.02, 0.03) + 0.08 * u.normalized()).transpose(); }
    for (SphereCost cost : {SphereCost::Geometric, SphereCost::Algebraic}) {
      SphereFitOptions o;
      o.cost = cost;
      const Sphere s = fitSphere(P, VectorXd(), o);
      CHECK((s.centre - Vector3d(0.01, -0.02, 0.03)).norm() < 1e-10);
      CHECK(std::abs(s.radius - 0.08) < 1e-10);
    }
    // a cap only (the points of a hemisphere), weights that single out the cap's centre; noisy points: the geometric centre is the least-squares one (zero gradient)
    MatrixXd Q = P;
    for (int i = 0; i < 200; ++i) Q.row(i) *= 1 + 0.02 * nd(rng);
    const Sphere g = fitSphere(Q);
    const auto cost = [&](const Vector3d& cc) { const VectorXd d = (Q.rowwise() - cc.transpose()).rowwise().norm(); return (d.array() - d.mean()).square().sum(); };
    for (int k = 0; k < 3; ++k) { Vector3d e = Vector3d::Zero(); e(k) = 1e-5; CHECK(cost(g.centre) <= cost(g.centre + e) && cost(g.centre) <= cost(g.centre - e)); }
    CHECK(throws([&] { fitSphere(P.topRows(3)); }));
    CHECK(throws([&] { fitSphere(P, VectorXd::Zero(200)); }));
    VectorXd bad = VectorXd::Ones(200);
    bad(4) = -1;
    CHECK(throws([&] { fitSphere(P, bad); }));
    // sensor spheres: the sphere under a sensor of a surface that IS a sphere is that sphere, whatever the weights
    MatrixXd sp(1, 3), sw(1, 1);
    sp << 0.01, -0.02, 0.2;
    const std::vector<Sphere> ss = sensorSpheres(P, sp, VectorXd::Ones(1), {0}, 1);
    CHECK(ss.size() == 1 && (ss[0].centre - Vector3d(0.01, -0.02, 0.03)).norm() < 1e-9);
  }
  // ---- Sarvas
  MatrixXd src(6, 3);
  for (int i = 0; i < 6; ++i) src.row(i) = (c + 0.05 * Vector3d(nd(rng), nd(rng), nd(rng)).normalized() * (0.3 + 0.7 * std::abs(nd(rng)) / 2)).transpose();
  const Vector3d sensor = c + Vector3d(0.05, 0.07, 0.08);
  {
    // a radial dipole (Q along r0 from the centre) makes no field at all
    for (int k = 0; k < 6; ++k) {
      const Vector3d r0 = src.row(k).transpose() - c;
      const MatrixXd g = sarvasPointGain(src.row(k), sensor, Vector3d(0.3, -0.5, 0.8).normalized(), c);
      CHECK(std::abs(g.row(0).dot(r0)) < 1e-12 * g.row(0).norm() * r0.norm() + 1e-30);
    }
    // the radial component of B is the primary current's alone (Biot-Savart): B_r = mu0/(4 pi) (Q x (r - r0)) . r^ / |r - r0|^3
    const Vector3d rhat = (sensor - c).normalized();
    const MatrixXd g = sarvasPointGain(src, sensor, rhat, c);
    MatrixXd bs(6, 3);
    for (int k = 0; k < 6; ++k) {
      const Vector3d d = sensor - src.row(k).transpose();
      // Q x d . rhat = Q . (d x rhat)
      bs.row(k) = (1e-7 * d.cross(rhat) / std::pow(d.norm(), 3)).transpose();
    }
    CHECK(relmax(g, bs) < 1e-12);
    // translation and rotation of the whole configuration
    const Vector3d t(0.1, -0.2, 0.3);
    MatrixXd shifted = src.rowwise() + t.transpose();
    const Vector3d on(0.3, -0.5, 0.8);
    CHECK(relmax(sarvasPointGain(shifted, sensor + t, on, c + t), sarvasPointGain(src, sensor, on, c)) < 1e-10);
    const Eigen::Matrix3d Rm = Eigen::AngleAxisd(0.7, Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    MatrixXd rot = (Rm * (src.rowwise() - c.transpose()).transpose()).transpose();
    rot.rowwise() += c.transpose();
    // the gain row is the coefficient of Q: rotating Q too, g' = R g
    const MatrixXd g0 = sarvasPointGain(src, sensor, on, c), g1 = sarvasPointGain(rot, c + Rm * (sensor - c), Rm * on, c);
    CHECK(relmax(g1, (Rm * g0.transpose()).transpose()) < 1e-10);
    CHECK(throws([&] { sarvasPointGain(sensor.transpose(), sensor, on, c); }));
  }
  {
    // a channel is the weighted sum of its points; a gradiometer of two coaxial coils is a difference
    Coils k;
    k.position.resize(3, 3);
    k.position << 0.05, 0.07, 0.12, 0.05, 0.07, 0.17, 0.0, 0.1, 0.1;
    k.position.rowwise() += c.transpose();
    k.orientation = MatrixXd::Zero(3, 3);
    k.orientation.col(2).setOnes();
    k.weight.resize(3);
    k.weight << 20, -20, 1;
    k.channel = {0, 0, 1};
    k.channels = 2;
    const MatrixXd G = sarvasGain(src, k, c);
    const Vector3d z(0, 0, 1);
    const MatrixXd a = sarvasPointGain(src, k.position.row(0).transpose(), z, c), b = sarvasPointGain(src, k.position.row(1).transpose(), z, c);
    for (int s = 0; s < 6; ++s) CHECK((G.block<1, 3>(0, 3 * s) - 20 * (a.row(s) - b.row(s))).norm() < 1e-12 * G.cwiseAbs().maxCoeff());
    // one centre per channel = the single-sphere gain when they agree
    CHECK(relmax(sarvasGain(src, k, std::vector<Vector3d>{c, c}), G) < 1e-15);
    CHECK(throws([&] { sarvasGain(src, k, std::vector<Vector3d>{c}); }));
    Coils bad = k;
    bad.orientation(1, 0) = std::nan("");
    CHECK(throws([&] { sarvasGain(src, bad, c); }));
    // the CTF correction: G - C R
    MatrixXd Cc = MatrixXd::Random(2, 4), Rg = MatrixXd::Random(4, 18);
    CHECK(relmax(referenceCompensatedGain(G, Cc, Rg), G - Cc * Rg) < 1e-15);
    CHECK(throws([&] { referenceCompensatedGain(G, Cc, MatrixXd::Random(3, 18)); }));
  }
  // ---- EEG
  {
    const VectorXd r1 = (VectorXd(1) << 1.0).finished(), s1 = (VectorXd(1) << 0.33).finished();
    CHECK(multiShellCoefficients(r1, s1, 5).isApproxToConstant(1.0, 1e-15));
    // equal conductivities everywhere: the shells are not there
    const VectorXd r3 = (VectorXd(3) << 0.88, 0.93, 1.0).finished(), same = VectorXd::Constant(3, 0.33);
    CHECK(multiShellCoefficients(r3, same, 20).isApproxToConstant(1.0, 1e-12));
    // a skull that conducts less lowers the series
    const VectorXd f = multiShellCoefficients(r3, (VectorXd(3) << 0.33, 0.0042, 0.33).finished(), 200);
    CHECK(f(0) < 1 && f(0) > 0 && f(199) > 0 && f(199) < f(0));
    // the Berg-Scherg fit: sum of magnitudes is f_1, small residual, descending eccentricities in (-1, 1)
    const BergScherg b = fitBergScherg(r3, (VectorXd(3) << 0.33, 0.0042, 0.33).finished());
    CHECK(std::abs(b.magnitude.sum() - f(0)) < 1e-12);
    CHECK(b.residual < 1e-2);
    CHECK(b.eccentricity(0) >= b.eccentricity(1) && b.eccentricity(1) >= b.eccentricity(2) && b.eccentricity.cwiseAbs().maxCoeff() < 1);
    // the series gain of ONE shell is the closed form (one equivalent dipole at the source, magnitude 1)
    EegSphereOptions one;
    one.radius = r1;
    one.conductivity = s1;
    one.approximation = EegApproximation::Series;
    one.seriesTerms = 400;
    EegSphereOptions cf = one;
    cf.approximation = EegApproximation::BergScherg;
    cf.useGivenBerg = true;
    cf.given.eccentricity = r1;
    cf.given.magnitude = r1;
    Coils el;
    const int ne = 6;
    el.position.resize(ne, 3);
    for (int i = 0; i < ne; ++i) el.position.row(i) = (c + 0.09 * Vector3d(nd(rng), nd(rng), nd(rng)).normalized()).transpose();
    el.weight = VectorXd::Ones(ne);
    for (int i = 0; i < ne; ++i) el.channel.push_back(i);
    el.channels = ne;
    CHECK(relmax(eegSphereGain(src, el, c, one), eegSphereGain(src, el, c, cf)) < 1e-9);
    // the three-shell: Berg-Scherg against the series of the same shells; more terms, the same
    EegSphereOptions three, ser;
    three.useRecordedBerg = false;                   // our own fit against the series
    ser.approximation = EegApproximation::Series;
    const MatrixXd Gb = eegSphereGain(src, el, c, three), Gs = eegSphereGain(src, el, c, ser);
    ser.seriesTerms = 400;
    CHECK(relmax(eegSphereGain(src, el, c, ser), Gs) < 1e-8);
    std::printf("  three-shell Berg-Scherg vs series: %.2e\n", relmax(Gb, Gs));
    CHECK(relmax(Gb, Gs) < 5e-3);
    // a source at the centre: the limit, continuous with the formula just beside it
    MatrixXd two(2, 3);
    two.row(0) = c.transpose();
    two.row(1) = (c + Vector3d(1e-5, 0, 0)).transpose();
    const MatrixXd Gc = eegSphereGain(two, el, c, three);
    CHECK(relmax(Gc.leftCols(3), Gc.rightCols(3)) < 1e-3);
    // electrodes in one channel add: a channel of two electrodes with weights +1, -1 is a difference
    Coils d = el;
    d.channel = {0, 0, 1, 2, 3, 4};
    d.weight(1) = -1;
    d.channels = 5;
    const MatrixXd Gd = eegSphereGain(src, d, c, three);
    CHECK((Gd.row(0) - (Gb.row(0) - Gb.row(1))).cwiseAbs().maxCoeff() < 1e-12 * Gb.cwiseAbs().maxCoeff());
    // errors
    EegSphereOptions badl = three;
    badl.conductivity = VectorXd::Constant(2, 0.3);
    CHECK(throws([&] { eegSphereGain(src, el, c, badl); }));
    CHECK(throws([&] { multiShellCoefficients((VectorXd(2) << 1.0, 0.5).finished(), VectorXd::Ones(2), 5); }));
  }
  return nxrtest::finish("test_forward");
}
