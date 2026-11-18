// The eigenspectrum (filterbank::forward / inverse, Rheome's graphtransform.eigen): round trip, Parseval, projection, complex fields.
#include "nxr/signal/graphfilterbank.h"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdio>
#include <random>

using namespace nxr::signal::graphfilterbank;

static int failures = 0;
static void check(bool ok, const char* what, double value) {
    if (!ok) { std::printf("FAIL %s (%.3e)\n", what, value); ++failures; }
}

int main() {
    const int n = 40, nT = 5;
    std::mt19937 rng(7);
    std::normal_distribution<double> N01;
    std::uniform_real_distribution<double> U(0.5, 2.0);

    // a pencil (L, B): L symmetric positive semi-definite, B diagonal positive (a lumped mass)
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) A(i, j) = N01(rng);
    const Eigen::MatrixXd L = A.transpose() * A;
    Eigen::VectorXd b(n);
    for (int i = 0; i < n; ++i) b(i) = U(rng);
    const Eigen::MatrixXd Bd = b.asDiagonal();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(L, Bd);

    EigenTransform t;
    t.phi = es.eigenvectors();          // B-orthonormal
    t.lambda = es.eigenvalues();
    t.mass = Bd.sparseView();

    Eigen::MatrixXd F(n, nT);
    for (int i = 0; i < n; ++i) for (int j = 0; j < nT; ++j) F(i, j) = N01(rng);

    // full basis: exact round trip and Parseval
    const Eigen::MatrixXd C = forward(t, F);
    check(C.rows() == n && C.cols() == nT, "forward shape", 0);
    const double rt = (inverse(t, C) - F).norm() / F.norm();
    check(rt < 1e-12, "inverse(forward(F)) = F", rt);
    const double e1 = C.squaredNorm(), e0 = (F.transpose() * Bd * F).trace();
    check(std::abs(e1 - e0) / e0 < 1e-12, "Parseval ||C||^2 = F'BF", std::abs(e1 - e0) / e0);

    // truncated basis: inverse(forward) is the B-orthogonal projection (idempotent)
    EigenTransform k = t;
    k.phi = t.phi.leftCols(12);
    k.lambda = t.lambda.head(12);
    const Eigen::MatrixXd P1 = inverse(k, forward(k, F));
    const double idem = (inverse(k, forward(k, P1)) - P1).norm() / P1.norm();
    check(idem < 1e-12, "projection is idempotent", idem);

    // complex fields (one FFT bin per column): the parts transform separately
    Eigen::MatrixXcd Z(n, nT);
    Z.real() = F;
    for (int i = 0; i < n; ++i) for (int j = 0; j < nT; ++j) Z(i, j).imag(N01(rng));
    const Eigen::MatrixXcd Cz = forward(t, Z);
    const double cr = (Cz.real() - forward(t, Eigen::MatrixXd(Z.real()))).norm() + (Cz.imag() - forward(t, Eigen::MatrixXd(Z.imag()))).norm();
    check(cr < 1e-12, "complex forward = forward of the parts", cr);
    const double crt = (inverse(t, Cz) - Z).norm() / Z.norm();
    check(crt < 1e-12, "complex round trip", crt);

    // no mass: plain orthonormal basis
    EigenTransform u;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eu(L);
    u.phi = eu.eigenvectors();
    u.lambda = eu.eigenvalues();
    const double urt = (inverse(u, forward(u, F)) - F).norm() / F.norm();
    check(urt < 1e-12, "identity-mass round trip", urt);

    // shape errors throw
    bool threw = false;
    try { forward(t, Eigen::MatrixXd(n + 1, 1)); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "forward rejects a field of the wrong length", 0);
    threw = false;
    try { inverse(k, Eigen::MatrixXd(n, 1)); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "inverse rejects coefficients of the wrong length", 0);

    if (failures) return 1;
    std::printf("test_eigenspectrum: ok\n");
    return 0;
}
