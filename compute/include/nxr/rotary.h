#pragma once
// nxr::rotary — a pair of linear maps of the channels read as ONE complex series: z = a_x . X + i a_y . X (the subject's rotation plane: rows fx, fy over the kernel's channels;
// nxr `nxr kernel rotationplane`, nxr.pipelines.plane / lagged_flow: z = fx @ X + 1j * (fy @ X)).
//
// PURE MATHS: matrices in, a matrix out; a POINTWISE node (every sample of z depends on that sample of X alone, no reduction over time or space). It never knows where the
// rows or the data came from. A pair turning from x towards y has its power at +f in the two-sided spectrum of z (nxr::signal::spectral::ComplexWelch): +f is forward when the
// operator already carries the sense (nxr negates fy when -f wins).
#include <Eigen/Dense>
#include <complex>
#include <stdexcept>

namespace nxr::rotary {

/** rows [2 x C] (x then y), X [C x T] -> z [1 x T] complex. */
inline Eigen::MatrixXcd pairSeries(const Eigen::MatrixXd& rows, const Eigen::MatrixXd& X) {
    if (rows.rows() != 2) throw std::invalid_argument("rotary: the pair is two rows (x, y), got " + std::to_string(rows.rows()));
    if (rows.cols() != X.rows()) throw std::invalid_argument("rotary: the rows act on " + std::to_string(rows.cols()) + " channels, the data has " + std::to_string(X.rows()));
    const Eigen::MatrixXd xy = rows * X;
    Eigen::MatrixXcd z(1, X.cols());
    for (Eigen::Index t = 0; t < X.cols(); ++t) z(0, t) = std::complex<double>(xy(0, t), xy(1, t));
    return z;
}

}  // namespace nxr::rotary
