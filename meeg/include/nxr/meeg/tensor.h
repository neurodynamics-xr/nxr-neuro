// nxr::meeg — the CROSS-SPECTRAL TENSOR of a free-orientation source current, and what is read from it at a vertex. The vector analogue of the band
// power map: where the band power of a source is the trace, the tensor keeps how the current oscillates — along which axis, how much along the
// normal, whether it swings or turns.
//
//   sourceTensor(K, S, Sq)   per source v, the complex 3x3 Hermitian T_v = K_v (S + i Sq) K_v^T, K_v the source's three kernel rows (x y z, ambient,
//                            the kernel's coordinates), S + i Sq a band's sensor cross-spectral density (nxr::signal::spectral::CrossWelch with quadrature:
//                            S its real part, Sq its imaginary part Im(X X^H)). Re T is symmetric, Im T antisymmetric; tr T = the source's band power
//                            summed over its orientations (sourcePower, Orientation::Norm) — exactly. Never a source time series: channels^2 per band.
//   stokes(T, e1, e2)        the TANGENTIAL Stokes parameters in the frame (e1, e2) (the canonical frames: e1 north, e2 = n x e1): with
//                            P_ab = e_a^T T e_b,  S0 = P11 + P22 (tangential power), S1 = P11 - P22, S2 = 2 Re P12 (linear polarisation: its axis
//                            1/2 arg(S1 + i S2) from e1, its strength |S1 + i S2|), S3 = 2 Im P12 (circular polarisation: > 0 a current turning from
//                            e1 towards e2, counter-clockwise about n). Under a turn of the frame by psi, S1 + i S2 turns by 2 psi; S0, S3, |S1 + i S2|
//                            do not. Linear in T, so they add over time and over vertices.
//   normalPower(T, n)        n^T Re(T) n, the power along the normal (unit n)
//   crossTensor(K, S, Sq, i, j)   the 3x3 block BETWEEN two sources, T_ij = K_i (S + i Sq) K_j^T: the cross-spectral density of source i's x y z with source j's.
//                            T_ji = T_ij^H and T_ii is sourceTensor's. Every block of the 3N x 3N source cross-spectrum is K C K^T read on demand from the band's
//                            sensor cross-spectrum C (M x M, stored by `crossspectrum compute`) and the kernel; the full matrix (rank <= M) is never formed.
#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>

namespace nxr::meeg {

/** [N] tensors, each 9 real and 9 imaginary parts, row-major 3x3 (x y z). */
struct SourceTensors {
  Eigen::Index sources = 0;
  std::vector<double> re, im;                 // [N x 9] each
  const double* reOf(Eigen::Index v) const { return re.data() + 9 * v; }
  const double* imOf(Eigen::Index v) const { return im.data() + 9 * v; }
};
/** K [3N x M] (three rows a source: x y z), S and Sq [M x M] (Sq antisymmetric). Throws std::invalid_argument on a shape mismatch. */
SourceTensors sourceTensor(const Eigen::MatrixXd& K, const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq);

struct Stokes { double s0 = 0, s1 = 0, s2 = 0, s3 = 0; };
/** The tangential Stokes parameters of one tensor (re, im: 9 each, row-major) in the frame (e1, e2), both unit and orthogonal. */
Stokes stokes(const double* re, const double* im, const std::array<double, 3>& e1, const std::array<double, 3>& e2);
/** n^T Re(T) n for a unit normal n. */
double normalPower(const double* re, const std::array<double, 3>& n);
/** tr Re(T): the source's power summed over its orientations. */
double tracePower(const double* re);
/** The block T_ij (re, im: 9 each, row-major) between sources i and j of K [3N x M]. Throws std::invalid_argument on a shape mismatch or a source outside K. */
void crossTensor(const Eigen::MatrixXd& K, const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq, Eigen::Index i, Eigen::Index j, double* re, double* im);

}  // namespace nxr::meeg
