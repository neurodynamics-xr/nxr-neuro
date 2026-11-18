// nxr::monogenic: the spatial analytic signal (Felsberg and Sommer 2001), on the unit sphere (icosphere, 642 vertices), by what the maths promises:
//   a planted band pattern f = P_5(cos θ) (a zonal spherical harmonic, degree 5, λ = 30) with out-of-band degrees 2 and 9 added, band-passed by a gain at λ = 30;
//   the continuum Riesz part is ∇P_5 / √30, so A² = P_5² + sin²θ P_5'² / 30;
//   NO ZERO-CROSSING DIPS: where the band's field passes through zero, A stays at the pattern's amplitude (|f| does not);
//   THE PHASE advances across lobes: < π/2 on a positive lobe, > π/2 on a negative one, ≈ π/2 at the zero crossings;
//   THE ORIENTATION is the pattern's gradient direction (along the meridian, with the sign of P_5');
//   R by the eigenmodes equals R by a dense reference (the symmetric matrix function Δ^{-1/2}), the Chebyshev series follows it within its tolerance;
//   a mode's sign flip leaves R and A unchanged;  A is the same for a complex field read part by part (A² adds Re and Im);  refusals.
#include "nxr/signal/riesz.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>

#include "mesh_helpers.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/phase_gradient.h"

using namespace nxr::signal;

static double P5(double x) { return (63 * std::pow(x, 5) - 70 * std::pow(x, 3) + 15 * x) / 8; }
static double dP5(double x) { return (315 * std::pow(x, 4) - 210 * x * x + 15) / 8; }
static double P2(double x) { return (3 * x * x - 1) / 2; }
static double P9(double x) { return (12155 * std::pow(x, 9) - 25740 * std::pow(x, 7) + 18018 * std::pow(x, 5) - 4620 * std::pow(x, 3) + 315 * x) / 128; }

int main() {
  std::vector<double> V;
  std::vector<int32_t> Fv;
  icosphere(3, V, Fv);
  const int n = static_cast<int>(V.size() / 3), nF = static_cast<int>(Fv.size() / 3);
  Eigen::MatrixXd P(n, 3);
  Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> F(nF, 3);
  for (int i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) P(i, d) = V[3 * i + d];
  for (int f = 0; f < nF; ++f) for (int s = 0; s < 3; ++s) F(f, s) = Fv[3 * f + s];
  const nxr::signal::Gradient G = nxr::signal::vertexGradient(P, F);

  nxr::geometry::IntrinsicSurface isurf(V.data(), n, Fv.data(), nF);
  const Eigen::SparseMatrix<double> Lap = isurf.cotanLaplacian();
  const std::vector<double> mass = isurf.inputMixedVoronoiMass();
  Eigen::MatrixXd Ld(Lap), Md = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd m(n);
  for (int i = 0; i < n; ++i) { Md(i, i) = mass[i]; m(i) = mass[i]; }
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Ld, Md);
  const Eigen::MatrixXd Phi = es.eigenvectors();
  const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(0.0);

  const auto band = [](double l) { return std::exp(-std::pow((l - 30.0) / 5.0, 2)); };
  const auto gR = rieszGain(band, 1e-3);
  EXPECT(gR(0.0) == 0.0 && gR(1e-9) == 0.0, "the constant mode has no Riesz part");
  EXPECT(std::abs(gR(30.0) - band(30.0) / std::sqrt(30.0)) < 1e-15, "rieszGain = g / sqrt(lambda)");

  Eigen::MatrixXd f0(n, 1);
  for (int i = 0; i < n; ++i) { const double z = P(i, 2); f0(i, 0) = P5(z) + 0.8 * P2(z) + 0.5 * P9(z) / 6.0; }
  // the band's field and its Riesz part by the eigenmodes: S = Φ diag(g_R) Φᵀ ⋆0 f0, R = G S;  f = Φ diag(g) Φᵀ ⋆0 f0
  auto bandOf = [&](const Eigen::MatrixXd& Ph, const std::function<double(double)>& g) {
    Eigen::VectorXd gv(lam.size());
    for (int k = 0; k < lam.size(); ++k) gv(k) = g(lam(k));
    return Eigen::MatrixXd(Ph * (gv.asDiagonal() * (Ph.transpose() * m.asDiagonal() * f0)));
  };
  const Eigen::MatrixXd f = bandOf(Phi, band), S = bandOf(Phi, gR), R = rieszTransform(G, S, 1);
  EXPECT(R.rows() == 3 * n, "R is [3V x m]");

  // the dense reference: Δ^{-1/2} as the symmetric matrix function of M^{-1/2} L M^{-1/2}, applied to M^{1/2}-scaled f
  {
    const Eigen::VectorXd mh = m.cwiseSqrt(), mih = mh.cwiseInverse();
    const Eigen::MatrixXd Ls = mih.asDiagonal() * Ld * mih.asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ss(Ls);
    Eigen::VectorXd gv(n);
    for (int k = 0; k < n; ++k) gv(k) = gR(std::max(ss.eigenvalues()(k), 0.0));
    const Eigen::MatrixXd Sd = mih.asDiagonal() * (ss.eigenvectors() * (gv.asDiagonal() * (ss.eigenvectors().transpose() * (mh.asDiagonal() * f0))));
    EXPECT((Sd - S).cwiseAbs().maxCoeff() < 1e-9 * std::max(1.0, S.cwiseAbs().maxCoeff()), "R by the eigenmodes equals R by the dense matrix function (S: " << (Sd - S).cwiseAbs().maxCoeff() << ")");
  }
  // the Chebyshev series (the fallback) follows it
  {
    const nxr::signal::graphfilterbank::ChebyshevOperator op(Lap, Eigen::SparseMatrix<double>(Md.sparseView()), 0, 320);
    const Eigen::MatrixXd Sc = op.apply(gR, f0);
    EXPECT((Sc - S).cwiseAbs().maxCoeff() < 2e-3 * S.cwiseAbs().maxCoeff(), "R by the Chebyshev series follows the eigenmodes (" << (Sc - S).cwiseAbs().maxCoeff() / S.cwiseAbs().maxCoeff() << " of the peak)");
  }
  // a mode's sign flip changes nothing
  {
    Eigen::MatrixXd Pf = Phi;
    for (int k = 0; k < Pf.cols(); k += 3) Pf.col(k) *= -1;
    const Eigen::MatrixXd R2 = rieszTransform(G, bandOf(Pf, gR), 1), f2 = bandOf(Pf, band);
    const Eigen::MatrixXd A1 = amplitude(f.cast<std::complex<double>>(), 1, R.cast<std::complex<double>>(), 3), A2 = amplitude(f2.cast<std::complex<double>>(), 1, R2.cast<std::complex<double>>(), 3);
    EXPECT((A1 - A2).cwiseAbs().maxCoeff() < 1e-10, "A is unchanged by a flip of a mode's sign");
  }

  const Eigen::MatrixXd A = amplitude(f.cast<std::complex<double>>(), 1, R.cast<std::complex<double>>(), 3), ph = phase(f, R), ori = orientation(R);
  // continuum values of the band's pattern: f ≈ g(30) P_5, R f ≈ g(30) ∇P_5 / √30
  // the band's gain at the discrete degree-5 eigenvalue (the mesh's, not 30): the pattern's scale, fitted by the mass inner product with P_5
  double num = 0, den = 0;
  for (int i = 0; i < n; ++i) { num += m(i) * f(i, 0) * P5(P(i, 2)); den += m(i) * P5(P(i, 2)) * P5(P(i, 2)); }
  const double g30 = num / den;
  double worstA = 0, peakA = 0, minAnearZero = 1e9, minAbsNearZero = 1e9, medA = 0, worstOri = 0;
  std::vector<double> inband;
  int zeros = 0, phaseWrong = 0, oriChecked = 0;
  for (int i = 0; i < n; ++i) {
    const double z = P(i, 2), s2 = 1 - z * z;
    if (std::abs(z) > 0.85) continue;                                  // away from the poles, where the zonal pattern's own amplitude falls
    const double Ac = g30 * std::sqrt(P5(z) * P5(z) + s2 * dP5(z) * dP5(z) / 30.0);
    worstA = std::max(worstA, std::abs(A(i, 0) - Ac));
    peakA = std::max(peakA, Ac);
    inband.push_back(A(i, 0));
    if (std::abs(f(i, 0)) < 0.03 * g30) { ++zeros; minAnearZero = std::min(minAnearZero, A(i, 0)); minAbsNearZero = std::min(minAbsNearZero, std::abs(f(i, 0))); }
    if (std::abs(f(i, 0)) > 0.25 * g30 && ((f(i, 0) > 0) != (ph(i, 0) < M_PI / 2))) ++phaseWrong;
    // orientation: along the meridian e_θ-direction ±: o · t = ±1 for t the unit tangent of ∇z; sign = sign(P_5')
    if (std::abs(dP5(z)) > 0.5 && s2 > 0.05) {
      Eigen::Vector3d t(-z * P(i, 0), -z * P(i, 1), 1 - z * z);       // ∇_S z = e_z − z n
      t.normalize();
      const Eigen::Vector3d o = ori.block<3, 1>(3 * i, 0);
      worstOri = std::max(worstOri, std::abs(std::abs(o.dot(t)) - 1.0));
      if ((o.dot(t) > 0) != (dP5(z) > 0)) worstOri = 1;
      ++oriChecked;
    }
  }
  std::sort(inband.begin(), inband.end());
  medA = inband[inband.size() / 2];
  EXPECT(worstA < 0.08 * peakA, "A follows sqrt(P5^2 + sin^2 P5'^2 / 30) of the continuum (worst " << worstA / peakA << " of the peak)");
  EXPECT(zeros >= 4, "the band's field passes through zero (" << zeros << " vertices within 3% of the amplitude)");
  EXPECT(minAnearZero > 0.3 * medA, "NO DIP: A at the field's zero crossings (min " << minAnearZero << ") stays above 0.3 of its median " << medA << ", where |f| reads " << minAbsNearZero);
  EXPECT(phaseWrong == 0, "the phase is < pi/2 on positive lobes and > pi/2 on negative ones (" << phaseWrong << " wrong)");
  EXPECT(oriChecked > 50 && worstOri < 0.08, "the orientation is the pattern's gradient direction, with the sign of P5' (" << oriChecked << " checked, worst " << worstOri << ")");
  {
    double nearHalf = 0;
    for (int i = 0; i < n; ++i) if (std::abs(P(i, 2)) < 0.85 && std::abs(f(i, 0)) < 0.03 * g30) nearHalf = std::max(nearHalf, std::abs(ph(i, 0) - M_PI / 2));
    EXPECT(nearHalf < 0.25, "the phase is about pi/2 at the zero crossings (worst deviation " << nearHalf << ")");
  }

  // a complex field: A² = A²(Re) + |Rim|² ...: amplitude of z = f e^{iθ}+... equals sqrt(Σ|z|² + Σ|R z|²)
  {
    Eigen::MatrixXcd Z(n, 2), RZ(3 * n, 2);
    Z.col(0) = f.cast<std::complex<double>>(); Z.col(1) = std::complex<double>(0, 1) * f.array();
    RZ.col(0) = R.cast<std::complex<double>>(); RZ.col(1) = std::complex<double>(0, 1) * R.array();
    const Eigen::MatrixXd Ac = amplitude(Z, 1, RZ, 3);
    EXPECT((Ac.col(0) - A.col(0)).cwiseAbs().maxCoeff() < 1e-12 && (Ac.col(1) - A.col(0)).cwiseAbs().maxCoeff() < 1e-12, "a phase factor on z leaves A unchanged (R is real and linear)");
    // two components add their squares
    Eigen::MatrixXcd Z2(2 * n, 1), R2(6 * n, 1);
    for (int i = 0; i < n; ++i) { Z2(2 * i, 0) = f(i, 0); Z2(2 * i + 1, 0) = f(i, 0); for (int d = 0; d < 3; ++d) { R2(6 * i + d, 0) = R(3 * i + d, 0); R2(6 * i + 3 + d, 0) = R(3 * i + d, 0); } }
    EXPECT((amplitude(Z2, 2, R2, 6).col(0) - std::sqrt(2.0) * A.col(0)).cwiseAbs().maxCoeff() < 1e-12, "components add their squares");
  }
  // pointwise: a frame computed alone equals it computed with others
  {
    Eigen::MatrixXd two(n, 2);
    two.col(0) = f.col(0); two.col(1) = 3 * f.col(0);
    const Eigen::MatrixXd R2 = rieszTransform(G, Eigen::MatrixXd(Eigen::MatrixXd(two.cols() == 2 ? S.replicate(1, 2) : S)), 1);
    EXPECT((R2.col(0) - R.col(0)).cwiseAbs().maxCoeff() < 1e-14, "R of a frame does not depend on the other frames");
  }
  // refusals
  {
    bool threw = false;
    try { rieszTransform(G, Eigen::MatrixXd(Eigen::MatrixXd::Zero(n + 1, 1)), 1); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "a field of the wrong size is refused");
    threw = false;
    try { amplitude(Eigen::MatrixXcd::Zero(n, 1), 1, Eigen::MatrixXcd::Zero(n, 1), 3); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "a Riesz part of the wrong size is refused");
  }
  std::cout << (g_failures ? "FAILED" : "all passed") << "\n";
  return g_failures ? 1 : 0;
}
