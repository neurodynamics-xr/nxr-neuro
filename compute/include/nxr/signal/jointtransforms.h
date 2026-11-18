#pragma once
// nxr::signal — the JOINT FOURIER TRANSFORM of a time-vertex signal: `jft` / `ijft` (GSPBox gsp_jft / gsp_ijft; Grassi, Loukas, Perraudin & Ricaud 2018, "Time-vertex signal processing").
//
// PURE MATHS: the graph Fourier transform over the vertices followed by the discrete Fourier transform over time. F is [vertices x T] (time along the columns, as everywhere in nxr::signal);
// the transform is [modes x T]: row k = the graph frequency lambda_k (the eigenvector order given), column j = the temporal frequency j fs / T, j = 0 .. T-1 in fft order (two-sided; a real
// signal gives a Hermitian-symmetric spectrum along j). Conventions: the graph transform is EigenTransform's (C = Phi^T B F, Parseval for a mass-orthonormal Phi); the time transform is
// the UNITARY DFT (numpy norm="ortho": both directions scaled by 1/sqrt(T), GSPBox's), so ||jft(F)||^2 = F^T B F.
#include "nxr/signal/graphfilterbank.h"
#include <Eigen/Dense>

namespace nxr::signal::jointtransforms {

/** [vertices x T] -> [modes x T] complex. A complex signal is transformed part by part (EigenTransform's rule). */
Eigen::MatrixXcd jft(const graphfilterbank::EigenTransform& t, const Eigen::MatrixXd& F);
Eigen::MatrixXcd jft(const graphfilterbank::EigenTransform& t, const Eigen::MatrixXcd& F);
/** [modes x T] -> [vertices x T] complex (the inverse of jft on its range; the projection onto the K modes when K < N). */
Eigen::MatrixXcd ijft(const graphfilterbank::EigenTransform& t, const Eigen::MatrixXcd& Fhat);
/** The real part of ijft: the signal whose spectrum is Hermitian-symmetric along time (the transform of a real signal). */
Eigen::MatrixXd ijftReal(const graphfilterbank::EigenTransform& t, const Eigen::MatrixXcd& Fhat);

}  // namespace nxr::signal::jointtransforms
