#pragma once
// nxr::speedspectrum — the SPEED SPECTRUM of cortical activity: how much energy moves at each phase speed, at each spatial scale.
//
// A wave travelling at phase speed c puts its energy where frequency and wavenumber meet on the line  2 pi f = c k,  k = mu = sqrt(lambda)
// (mu the Hodge-Dirac modes' wavenumber, rad per unit length). So the mode-by-frequency power  P[k, f] = diag(B S(f) B^T)  (B the fused
// weights; S the sensor cross-spectrum per frequency bin), tiled by the JOINT filterbank (a graph member over lambda = mu^2 x a time member
// over f), says how much energy travels at each speed. PURE MATHS: arrays in, arrays out; no optical flow.
//
// What it returns, for EXACT (divergence, kind 0) and COEXACT (rotation, kind 2) modes SEPARATELY:
//   tileEnergy[G x T]   sum over modes k of the kind and bins f of  g_g(lambda_k)^2 h_t(f)^2 P[k, f]   (the joint member's gain at (lambda_k, f); the
//                       bank is separable, so the member's gain is the product of its two factors). A tight bank (sum of squares = G^2 on the interior)
//                       conserves energy: the tiles sum to G^2 times the power inside the bank's support.
//   tileSpeed[G x T]    the tile's centre speed  2 pi f_c / mu_c  (f_c the time member's centre, mu_c the graph member's centre wavenumber).
//   histogram[G x nb]   per spatial-scale member g: the energy g_g(lambda_k)^2 P[k, f] binned on  c = 2 pi f / mu_k  over ALL (k, f) (log-spaced
//                       bins; `below` / `above` hold what lies outside; modes with mu <= 0 have no speed and are not binned).
//   quadrature[G x T]   the travelling-vs-standing indicator: ||Im C||_F / ||C||_F of the tile's mode cross-spectrum
//                       C = sum_f h_t(f)^2 D_g (Re S(f) + i Im S(f)) D_g, D_g = diag(g_g(lambda_k)) on the kind's modes (nxr::quadrature's im_fraction).
//                       A standing oscillation puts energy at (lambda, f) too, but its modes are in phase: Im C = 0. Only a wave that moves has
//                       modes in quadrature. (Two modes in exact quadrature give 1/sqrt(2).)
// Frequencies are the one-sided bins of the cross-spectrum (f > 0); the time bank is evaluated there (a two-sided bank's negative members are empty).
#include "nxr/filterbank.h"
#include "nxr/jointbank.h"
#include "nxr/timebank.h"
#include <Eigen/Dense>
#include <vector>

namespace nxr::signal::speedspectrum {

/** The mode cross-spectra at the kept frequency bins: S[b] the real part, Sq[b] the imaginary part (antisymmetric), both [K x K]. */
struct ModeSpectra {
    Eigen::VectorXd frequencies;               // Hz, [B], ascending, > 0
    std::vector<Eigen::MatrixXd> S, Sq;        // [B] of [K x K]; Sq may be empty (then the quadrature fraction is not computed: NaN)
};

struct Options {
    int speedBins = 40;                        // log-spaced
    double speedMin = 0, speedMax = 0;         // length/s; 0: from the spectrum (the smallest and largest 2 pi f / mu over the modes and bins)
};

/** One kind of mode (exact or coexact). */
struct Part {
    int kind = 0;                              // 0 exact (divergence) · 2 coexact (rotation)
    int modes = 0;
    double power = 0;                          // sum over the kind's modes and the bins of P[k, f]
    Eigen::MatrixXd tileEnergy, tileSpeed, quadrature;   // [G x T]
    Eigen::MatrixXd histogram;                 // [G x speedBins]
    Eigen::MatrixXd outside;                   // [G x 2]: energy below the first and above the last speed bin
};

struct Result {
    Eigen::VectorXd speedEdges;                // [speedBins + 1], length/s
    Eigen::VectorXd speedCentres;              // [speedBins] geometric centres
    Eigen::VectorXd graphCentres;              // [G] centre wavenumber of each graph member, rad per unit length
    Eigen::VectorXd timeCentres;               // [T] centre frequency of each time member, Hz
    std::vector<Part> parts;                   // exact first, then coexact (a kind with no mode is left out)
};

/** `mu` [K] and `kind` [K] (0 exact, 2 coexact) of the modes the spectra are on. The bank is the separable product of `graph` and `time`. */
Result compute(const filterbank::Bank& graph, const timebank::TimeBank& time, const Eigen::VectorXd& mu, const std::vector<int>& kind, const ModeSpectra& spectra, const Options& options = {});

}  // namespace nxr::signal::speedspectrum
