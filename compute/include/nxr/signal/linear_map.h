#pragma once
// nxr::signal::linearMap — a linear map of the channels with a COMPLEX operator: Z = W X, W [R x C] complex, X [C x T] real, Z [R x T] complex (Z = Re W . X + i Im W . X).
//
// The generic linear-operator step (a stored operator applied to a block: the members stage of a chain). One row w = p + i q is the pair of maps z = p.X + i q.X: with
// the rows (fx, fy) of a subject's rotation plane it is the complex series whose power at +f is the pair turning from x towards y (nxr `rotationplane`; the two-sided
// spectrum is nxr::signal::spectral::ComplexWelch). A real operator is the case Im W = 0.
//
// PURE MATHS: matrices in, a matrix out; POINTWISE (every sample of Z depends on that sample of X alone: no reduction over time or space). It never knows where W or X came from.
#include <Eigen/Dense>
#include <complex>
#include <stdexcept>
#include <string>

namespace nxr::signal {

namespace detail {
inline Eigen::MatrixXcd linearMapComplex(const Eigen::MatrixXcd& weights, const Eigen::MatrixXd& X) {
    if (weights.cols() != X.rows())
        throw std::invalid_argument("linearMap: the operator acts on " + std::to_string(weights.cols()) + " channels, the data has " + std::to_string(X.rows()));
    const Eigen::Index R = weights.rows();
    Eigen::MatrixXd stacked(2 * R, weights.cols());      // real parts over imaginary parts: one real product, as the pair of rows always was
    stacked.topRows(R) = weights.real();
    stacked.bottomRows(R) = weights.imag();
    const Eigen::MatrixXd Y = stacked * X;
    Eigen::MatrixXcd Z(R, X.cols());
    for (Eigen::Index r = 0; r < R; ++r)
        for (Eigen::Index t = 0; t < X.cols(); ++t) Z(r, t) = std::complex<double>(Y(r, t), Y(R + r, t));
    return Z;
}
inline Eigen::MatrixXd linearMapReal(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& X) {
    if (weights.cols() != X.rows())
        throw std::invalid_argument("linearMap: the operator acts on " + std::to_string(weights.cols()) + " channels, the data has " + std::to_string(X.rows()));
    return weights * X;
}
}  // namespace detail

/** ONE function for both operators, chosen by the operator's scalar type (so any Eigen expression resolves, with every compiler): a COMPLEX W [R x C] gives Z = W X complex (Re W . X +
 *  i Im W . X); a REAL W gives Y = W X real (a region map: one row a region's signal). X [C x T] real. */
template <class Weights, class Data>
auto linearMap(const Eigen::MatrixBase<Weights>& weights, const Eigen::MatrixBase<Data>& X) {
    if constexpr (Eigen::NumTraits<typename Weights::Scalar>::IsComplex) return detail::linearMapComplex(weights.template cast<std::complex<double>>(), X.template cast<double>());
    else return detail::linearMapReal(weights.template cast<double>(), X.template cast<double>());
}

}  // namespace nxr::signal
