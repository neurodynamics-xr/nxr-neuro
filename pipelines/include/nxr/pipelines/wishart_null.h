// nxr-pipelines — THE WISHART NULL OF THE SECOND-MOMENT MEASURES (`wishart compute`; spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10, decision 3): a STATISTIC NODE. A burst's second moment Q_b = sum of z z^H over its time tiles (the stored
// `crossspectrum compute --estimator analytic` windows, summed over the burst's tiles first: products add, exact) is a sample covariance; a quantity LINEAR in Q_b (the quadrature, the lagged share, a phase current) has a null with no lag at all: Q_null ~ complex Wishart(Re Q_b / nu, nu), the
// burst's own zero-lag covariance with `nu` degrees of freedom (nxr `vortex.wishart`), `nu` = the burst's duration x the band's effective bandwidth (nxr `dof_fraction` x the rate; `nxr::surrogate::effectiveBandwidth` of the stored PSD through the band's analytic gain), 1000 draws, seeded. The
// draws are cheap because the null is a distribution of Q, not of a recording. It complements the sign-flip test: sign flips ask whether a direction is SHARED across bursts, the Wishart null asks whether LAGS EXIST AT ALL in each burst.
//
// WHAT IS MEASURED here is the SENSOR-LEVEL LAGGED SHARE of each burst, ||Im Q_b||_F^2 / ||Q_b||_F^2 (the share of the burst's second moment that is quadrature, between the kernel's sensors): per burst the draws at least as large (p = (e + 1) / (draws + 1)), Benjamini-Hochberg over the
// bursts, and for the run the mean share against the null's mean over draws. The same draws serve any functional linear in Q (the cortical quadrature and phase current are built the same way from Q: B Q B^T and the gradient form): they need the operator of the level and tile
// applied to each draw, which is NOT built here (the cost of the bound operator times 1000 draws per burst; open).
// STORED (statistics of the recording): `wishart.<windows>.<bursts>` [bursts x (lagged_share, null_mean, p, q, dof)] (measure `null_p`, method `wishart_null`) and `wishart.<windows>.<bursts>.run` [(mean share, null mean, null sd, p)].
// WHAT IT CANNOT SAY: the null is a zero-lag Gaussian process with the burst's covariance and `dof` independent samples: a burst beyond it has lags beyond sampling noise, not a wave; `dof` is the duration x bandwidth of a flat-ish band (the PSD's effective bandwidth), so a spectrum with structure inside the band changes it.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct WishartNullOptions {
  std::string recording;                          // REQUIRED: the recording the windows are of
  std::string windows;                            // REQUIRED: the stored analytic cross-spectrum windows (`<stem>_<band>_windows`)
  std::string bursts;                             // REQUIRED: a stored spans Selection on the recording's time line (tile-aligned: `global_bursts`)
  std::string psd;                                // REQUIRED: the recording's stored Welch PSD (its spectrum gives the band's effective bandwidth)
  std::vector<double> band;                       // REQUIRED: [lo, hi] Hz, the band the windows are of
  std::string shape = "morlet";                   // the band's analytic gain: morlet | meyer (half_width 1/3)
  int draws = 1000;
  std::uint64_t seed = 1;
  bool replace = false;
};
struct WishartNullResult { nxr::io::Json info; bool hit = false; };
WishartNullResult wishart_null_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const WishartNullOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
