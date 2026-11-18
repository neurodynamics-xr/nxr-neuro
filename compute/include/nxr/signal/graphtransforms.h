#pragma once
// nxr::signal::graphtransforms: the transforms of a field on a self-adjoint operator, and the response a bank designs and a transform applies.
//
// PURE MATHS: arrays and operators in, arrays out; no file I/O, no mesh. The EIGEN route (exact, through the operator's modes: gft and igft, Rheome's graphtransform.eigen) and the CHEBYSHEV route (a
// polynomial of the pencil (L, M), no eigenvectors: chebyshev, Rheome's graphtransform.chebyshev). The banks that design the responses (graphfilterbank.h) include this header.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <functional>
#include <memory>

namespace nxr::signal::graphtransforms {

using Gain = std::function<double(double)>;   // g(λ), real, for one scalar λ

// ── the transforms ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** EXACT: the operator's modes. Φ is mass-orthonormal (Φᵀ B Φ = I; Parseval depends on it). Empty `mass` = identity. */
struct EigenTransform {
    Eigen::MatrixXd phi;                 // [n × K]
    Eigen::VectorXd lambda;              // [K]
    Eigen::SparseMatrix<double> mass;    // B, [n × n] (0 × 0: identity)
};
/** The eigenspectrum, the spatial equivalent of the FFT (Rheome's graphtransform.eigen: gft and igft).
 *  gft: field → one coefficient per mode, C = Φᵀ B F  [K × nT]; igft: C → field, F = Φ C  [n × nT].
 *  Φ mass-orthonormal ⇒ Parseval: ‖C‖² = Fᵀ B F, and igft(gft(F)) is F's projection onto the K modes (F itself when K = n).
 *  Trailing time/trial columns pass through. Complex fields (e.g. one FFT bin per column) are transformed part by part. */
Eigen::MatrixXd gft(const EigenTransform& t, const Eigen::MatrixXd& F);
Eigen::MatrixXcd gft(const EigenTransform& t, const Eigen::MatrixXcd& F);
Eigen::MatrixXd igft(const EigenTransform& t, const Eigen::MatrixXd& C);
Eigen::MatrixXcd igft(const EigenTransform& t, const Eigen::MatrixXcd& C);

/** POLYNOMIAL: g(L) as a Chebyshev series in M⁻¹L on [0, λmax] by Clenshaw recursion; no eigenvectors. Factors the mass once. */
class ChebyshevOperator {
public:
    /** L: the stiffness, [n × n]; `mass`: M (0 × 0 = identity; else SPD, factored by Cholesky). lmax ≤ 0: estimated (Lanczos × 1.01).
     *  λmax MUST bound the operator's whole spectrum — a smaller value makes the series diverge; one below 0.99 of a cheap
     *  estimate throws. `order` is the series degree (Rheome default 40). */
    ChebyshevOperator(const Eigen::SparseMatrix<double>& L, const Eigen::SparseMatrix<double>& mass = {}, double lmax = 0, int order = 40);
    ~ChebyshevOperator();
    ChebyshevOperator(ChebyshevOperator&&) noexcept;
    ChebyshevOperator& operator=(ChebyshevOperator&&) noexcept;
    int rows() const;
    double lmax() const;
    int order() const;
    friend Eigen::MatrixXd chebyshev(const ChebyshevOperator& op, const Gain& g, const Eigen::MatrixXd& X);
private:
    struct Impl; std::unique_ptr<Impl> p_;
};

/** g(M⁻¹L) X (Rheome's graphtransform.chebyshev). The complex overload applies the real operator to the real and imaginary parts apart. */
Eigen::MatrixXd chebyshev(const ChebyshevOperator& op, const Gain& g, const Eigen::MatrixXd& X);
Eigen::MatrixXcd chebyshev(const ChebyshevOperator& op, const Gain& g, const Eigen::MatrixXcd& X);

/** Largest eigenvalue of the pencil (L, M) by Lanczos (Spectra), as MATLAB's eigs(L, M, 1, 'largestabs'); tolerance 1e-4. 0 on failure. */
double estimateLmax(const Eigen::SparseMatrix<double>& L, const Eigen::SparseMatrix<double>& mass = {});

/** The Chebyshev coefficients of g on [0, lmax] at order+1 Chebyshev nodes (Rheome's graphtransform.chebyshev; c₀ halved). */
Eigen::VectorXd chebyshevCoefficients(const Gain& g, double lmax, int order);
/** The series Σ c_k T_k(2λ/lmax − 1) at λ (Clenshaw). */
double chebyshevEval(const Eigen::VectorXd& c, double lmax, double lambda);

/** How well the series of degree `order` stands for the gain g on [0, λmax]: the largest |g(λ) − Σ c_k T_k| over `samples` points. A narrow band at a fine level of a
 *  large mesh needs a high order (the gain is nearly a spike on [0, λmax]): the caller checks this before trusting a polynomial band. */
double chebyshevError(const Gain& g, double lmax, int order, int samples = 4096);

}  // namespace nxr::signal::graphtransforms
