#pragma once
// nxr::signal — the POLARISATION of a source current's cross-spectral tensor at a vertex (from meeg/tensor.h): the tangential Stokes parameters in the canonical frame (e1, e2), the power along
// the normal and the trace of a 3x3 Hermitian tensor T given by its real and imaginary parts (9 each, row-major x y z).
//   stokes(T, e1, e2)   with P_ab = e_a^T T e_b: S0 = P11 + P22 (tangential power), S1 = P11 - P22, S2 = 2 Re P12 (linear polarisation: axis 1/2 arg(S1 + i S2) from e1, strength |S1 + i S2|),
//                       S3 = 2 Im P12 (circular: > 0 a current turning from e1 towards e2, counter-clockwise about n). Under a turn of the frame by psi, S1 + i S2 turns by 2 psi; S0, S3,
//                       |S1 + i S2| do not. Linear in T, so they add over time and over vertices.
//   normalPower(T, n)   n^T Re(T) n, the power along the unit normal;  tracePower(T) = tr Re(T): the power summed over the three orientations.
#include <array>

namespace nxr::signal {

struct Stokes { double s0 = 0, s1 = 0, s2 = 0, s3 = 0; };
/** The tangential Stokes parameters of one tensor (re, im: 9 each, row-major) in the frame (e1, e2), both unit and orthogonal. */
Stokes stokes(const double* re, const double* im, const std::array<double, 3>& e1, const std::array<double, 3>& e2);
/** n^T Re(T) n for a unit normal n. */
double normalPower(const double* re, const std::array<double, 3>& n);
/** tr Re(T): the source's power summed over its orientations. */
double tracePower(const double* re);

}  // namespace nxr::signal
