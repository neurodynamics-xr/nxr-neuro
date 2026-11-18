#pragma once
// nxr::select — WHERE AN AMPLITUDE EPISODE SWITCHES ON AND OFF: the double-sigmoid fit of Rheome's `+rheome/+detect/risefall.m` (the Rheome code; the maths written from it).
// PURE MATHS: an amplitude series (any: the mean over a region, the max over tiles; the function never knows which), its rate, the candidate episode [t0, t1] s, options → the fit.
//
//   a(t) = b + h S((t − tOn)/tauR) S(−(t − tOff)/tauF)                          Model::Linear          S(x) = 1/(1 + e^(−x))
//   a(t) = sqrt(b² + (h S((t − tOn)/tauR) S(−(t − tOff)/tauF))²)                Model::Quadrature      an episode on a background adds in POWER (incoherent sources)
//
// fitted by bounded nonlinear least squares to the series, smoothed by a symmetric moving mean (`smooth` s) and scaled to unit maximum, over [t0 − margin, t1 + margin]. tOn and tOff are the
// HALF-rise and HALF-fall times; the phases: build-up [tOn − 2 tauR, tOn + 2 tauR], plateau [tOn + 2 tauR, tOff − 2 tauF] (none when they cross), decay [tOff − 2 tauF, tOff + 2 tauF]. r² is of the
// fit on the unit scale (scale-free). The fit is Rheome's `lsqcurvefit` (trust-region-reflective); here a bounded Levenberg–Marquardt on the same bounds and start: the minimum is the same where the fit is
// well posed, the path is not (tested against Rheome's outputs on synthetic episodes, test/golden/gen_risefall_golden.m).
// NOT BUILT: Rheome's Model "posterior" (a grid posterior of tOn and tOff): refused by name (std::invalid_argument).
#include <Eigen/Dense>
#include <limits>
#include <string>

namespace nxr::select {

enum class Model { Linear, Quadrature };
Model parseModel(const std::string& name);       // "linear" | "quadrature"; "posterior" is refused as not built; others refused naming the valid ones

struct Options {
  double margin = 1.0;                           // s either side of [t0, t1] included in the fit
  double smooth = 0.1;                           // s: the symmetric moving mean applied first (movmean over round(smooth·fs) samples, MATLAB's window rule)
  double tauMax = 1.0;                           // s: the upper bound of tauR and tauF (a fit within 1% of it is flagged `atBound`)
  Model model = Model::Linear;
};
struct Fit {
  double tOn = NAN, tOff = NAN, tauR = NAN, tauF = NAN;   // s, on the series' clock (sample i is at (i)/fs, i from 0)
  double b = NAN, h = NAN;                                // the amplitude's units
  double r2 = NAN;
  bool ok = false, atBound = false;
  double buildup[2] = {NAN, NAN}, decay[2] = {NAN, NAN}, plateau[2] = {NAN, NAN};
  bool hasPlateau = false;
  int iterations = 0;
};
/** `a` [n] the amplitude, `fs` samples a second, [t0, t1] the candidate episode (s). A window under 11 samples, a series of no amplitude, or a fit that ends with tOff <= tOn: ok = false (every number NaN). */
Fit risefall(const Eigen::Ref<const Eigen::VectorXd>& a, double fs, double t0, double t1, const Options& opt = {});

/** THE STAGES of an episode from its fit's numbers (s;; nxr `burst_stages`, rheome.flow.labelalpha's episodes): ONSET [tOn − 2 tauR, tOn + 2 tauR], WITHIN the plateau [tOn + 2 tauR, tOff − 2 tauF] (none
 *  when they meet), OFFSET [tOff − 2 tauF, tOff + 2 tauF]. Pure: four numbers in, three intervals out (NaN in → no stage). */
struct Stages {
  double onset[2] = {NAN, NAN}, within[2] = {NAN, NAN}, offset[2] = {NAN, NAN};
  bool hasWithin = false;
  bool valid = false;                            // the four numbers were finite and tauR, tauF above 0
};
Stages stagesOf(double tOn, double tOff, double tauR, double tauF);
/** THE FIT GATE (nxr `burst_stages`, labelalpha's): keep an episode when its fit is ok, its r² is at least `minR2` and its rise and fall times are not at their bound (`atBound`: within 1% of tauMax) */
bool passesGate(const Fit& f, double minR2);

}  // namespace nxr::select
