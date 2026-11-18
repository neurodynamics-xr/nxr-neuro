#pragma once
// nxr::signal — the WINDOWED GRAPH FOURIER TRANSFORM (Shuman, Ricaud & Vandergheynst 2016, "Vertex-frequency analysis on graphs", Appl. Comput. Harmon. Anal. 40:260-291):
// the graph analogue of the short-time Fourier transform: a signal is localised by a window centred on each vertex i, and analysed at each graph frequency k.
//
// PURE MATHS on an eigenbasis handed in (graphtransforms::EigenTransform: orthonormal eigenvectors phi_l of the graph Laplacian, eigenvalues lambda_l; the mass must be the identity, as
// in the paper: a mass-weighted pencil is refused). With N vertices and a kernel g^(lambda) (the window's spectrum, a function of the eigenvalue):
//   generalised translation   (T_i g)(n) = sqrt(N) sum_l g^(lambda_l) phi_l(i) phi_l(n)           the window centred on vertex i
//   modulation                (M_k g)(n) = sqrt(N) phi_k(n) g(n)
//   atom                      g_{i,k}(n) = (M_k T_i g)(n) = N phi_k(n) sum_l g^(lambda_l) phi_l(i) phi_l(n)
//   transform                 S f(i, k)  = < f, g_{i,k} > = sum_n f(n) g_{i,k}(n)                    (real eigenvectors; the window is real)
// `wgft` returns S f as [vertices x frequencies] (row i = the window at vertex i, column k = the graph frequency k, in the order of the eigenvectors given).
// With all N eigenvectors the transform is a frame and is inverted by `iwgft`:
//   f(m) = sum_{i,k} S f(i,k) g_{i,k}(m) / ( N sum_i (T_i g)(m)^2 )                                   (the denominator: the window's energy over the vertices at m; > 0 unless the kernel has no mass)
// LIMIT (stated, tested): the identity mass only. A transform with a mass matrix (a mass-weighted pencil, the cortex's lumped mass) is refused with the reason, never computed as if the mass were the identity;
// the mass-weighted windowed transform is not designed here.
// What it cannot show: a window g^ that is zero on part of the spectrum cannot see those frequencies; with K < N eigenvectors the inverse is not exact.
#include "nxr/signal/graphtransforms.h"
#include <Eigen/Dense>

namespace nxr::signal::graphtransforms {

/** S f(i, k), [N x K], of a real signal f [N] (one value per vertex) with the window of spectrum `kernel` (> 0 somewhere). Throws std::invalid_argument on a non-identity mass or a size mismatch. */
Eigen::MatrixXd wgft(const EigenTransform& t, const Gain& kernel, const Eigen::VectorXd& f);
/** The synthesis of S [N x K] back to a signal [N]; exact for the full eigenbasis (K = N) and a kernel with mass at every vertex. */
Eigen::VectorXd iwgft(const EigenTransform& t, const Gain& kernel, const Eigen::MatrixXd& S);

}  // namespace nxr::signal::graphtransforms
