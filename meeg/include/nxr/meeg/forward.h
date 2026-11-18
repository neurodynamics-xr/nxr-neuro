// nxr::meeg — the forward model (gain matrix) of sphere head models, written from the published maths (docs/specs/2026-11-10-neuro-forward-inverse.md, section 8):
//   MEG: Sarvas, Phys. Med. Biol. 32:11-22, 1987 (a current dipole in a conducting sphere); overlapping spheres: Huang, Mosher & Leahy, Phys. Med. Biol. 44:423-440, 1999.
//   EEG: the multi-shell sphere as a series (Brody, Terry & Ideker 1973; Zhang, Phys. Med. Biol. 40:335-349, 1995) and its Berg & Scherg (1994) equivalent of J dipoles in one shell.
// MNE-Python (BSD) is a second reference. Brainstorm (GPLv3) is a source of reference OUTPUTS only: nothing here is translated from it.
// Pure: arrays in, arrays out. Depends on Eigen only; no file is read.
//
// The gain G [M x 3N] has the channels as rows and x, y, z of each source (a unit dipole, A.m) as columns 3k..3k+2: T/(A.m) for MEG, V/(A.m) for EEG, in SI units.
// A channel is the weighted sum of its sensor points (a magnetometer's integration points, a gradiometer's two coils with weights of opposite sign, an electrode's one point):
//   G_channel = sum_k weight_k . G_point(k).
#pragma once
#include <Eigen/Dense>
#include <vector>

#include "nxr/optimisation/fminsearch.h"
#include "nxr/meeg/sphere.h"

namespace nxr::meeg {

// The sensor points of M channels, flattened: point p belongs to channel `channel[p]`.
struct Coils {
  Eigen::MatrixXd position;        // [P x 3] metres
  Eigen::MatrixXd orientation;     // [P x 3] unit vectors the point measures along (MEG; ignored for EEG)
  Eigen::VectorXd weight;          // [P] the point's weight in its channel
  std::vector<int> channel;        // [P] 0..channels-1
  int channels = 0;
};

// ---- MEG ---------------------------------------------------------------------------------------------------------------------------------------------------
struct SarvasOptions {
  double permeability = 1.2566370614359173e-6;   // mu_0, T.m/A (4 pi x 1e-7)
};
// B at `position` along `orientation` of unit dipoles at `sources` [N x 3] in a sphere about `centre`: [N x 3] (x, y, z moments), T/(A.m).
//   B(r) = mu_0 / (4 pi F^2) [ F Q x r0 - (Q x r0 . r) grad F ],  a = r - r0,  F = a (r a + r^2 - r0.r),
//   grad F = (a^2/r + a.r/a + 2a + 2r) r - (a + 2r + a.r/a) r0       (r, r0 from the centre); the gain row is the coefficient of Q.
// std::invalid_argument when a source coincides with the point.
Eigen::MatrixXd sarvasPointGain(const Eigen::MatrixXd& sources, const Eigen::Vector3d& position, const Eigen::Vector3d& orientation, const Eigen::Vector3d& centre,
                                const SarvasOptions& opt = {});
// The gain of every channel with ONE sphere centre (single sphere), or with a sphere centre per channel (overlapping spheres: `centres` has `coils.channels` entries).
// std::invalid_argument for sizes that disagree, a NaN, or a point with no orientation.
Eigen::MatrixXd sarvasGain(const Eigen::MatrixXd& sources, const Coils& coils, const Eigen::Vector3d& centre, const SarvasOptions& opt = {});
Eigen::MatrixXd sarvasGain(const Eigen::MatrixXd& sources, const Coils& coils, const std::vector<Eigen::Vector3d>& centres, const SarvasOptions& opt = {});
// CTF reference compensation, the one-line correction: G_MEG - C . G_REF (C [M x R]: the compensation coefficients, G_REF [R x 3N]). Nothing calls it yet.
Eigen::MatrixXd referenceCompensatedGain(const Eigen::MatrixXd& megGain, const Eigen::MatrixXd& coefficients, const Eigen::MatrixXd& referenceGain);

// ---- EEG ---------------------------------------------------------------------------------------------------------------------------------------------------
// The coefficients f_n, n = 1..terms, of the multi-shell sphere's series relative to one homogeneous sphere (all 1 for one shell): the Zhang (1995) transfer-matrix recursion over the
// interfaces. radius and conductivity run from the innermost to the outermost shell (radius only through its ratios).
Eigen::VectorXd multiShellCoefficients(const Eigen::VectorXd& radius, const Eigen::VectorXd& conductivity, int terms);

// How the Berg-Scherg parameters are fitted: f_n ~ sum_j lambda_j mu_j^(n-1), n = 1..terms, with sum_j lambda_j = f_1 (so the first term is exact), weighted per n by
// w_n = sqrt((2n+1)(3n+1)/n) q^(n-1) with q = r_min / r_max (the weighting MNE-Python uses), n >= 2; the mu_j in (-1, 1) by simplex, the lambda_j by linear least squares.
struct BergFitOptions {
  int dipoles = 3;                 // J
  int terms = 200;                 // the series terms fitted
  int starts = 12;                 // simplex starts (deterministic, spread over (-1, 1)); the best minimum is kept: the fit has local minima
  optimisation::NelderMeadOptions minimiser = []{ optimisation::NelderMeadOptions o; o.zeroStep = 0.05; return o; }();
};
struct BergScherg {
  Eigen::VectorXd eccentricity;    // mu_j, largest first
  Eigen::VectorXd magnitude;       // lambda_j: sum = f_1, relative to the outer shell's conductivity (the gain divides by it)
  double residual = 0;             // sqrt(sum_n w_n^2 r_n^2 / sum_n w_n^2 f_n^2) of the fit
};
BergScherg fitBergScherg(const Eigen::VectorXd& radius, const Eigen::VectorXd& conductivity, const BergFitOptions& opt = {});

enum class EegApproximation {
  BergScherg,   // J equivalent dipoles in one shell (Brainstorm's, MNE-Python's)
  Series        // the multi-shell series itself, `seriesTerms` terms (exact up to the truncation; slower)
};
struct EegSphereOptions {
  Eigen::VectorXd radius = (Eigen::VectorXd(3) << 0.88, 0.93, 1.0).finished();            // relative radii, innermost to outermost (brain, skull, scalp): Brainstorm's defaults
  Eigen::VectorXd conductivity = (Eigen::VectorXd(3) << 0.33, 0.0042, 0.33).finished();  // S/m, innermost to outermost: Brainstorm's defaults
  EegApproximation approximation = EegApproximation::BergScherg;
  BergFitOptions berg;
  bool useRecordedBerg = true;     // the layers of Brainstorm's defaults or of MNE-Python's use the reference's own recorded Berg-Scherg parameters (below); any other layers are fitted. false: always fit
  bool useGivenBerg = false;       // true: use `given` instead (any parameters; wins over the two above)
  BergScherg given;
  int seriesTerms = 200;           // for EegApproximation::Series
  double centreTolerance = 1e-6;   // an equivalent dipole nearer the centre than this fraction of the electrode's distance takes the limit at the centre
};
// The recorded parameters (`recordedBergScherg`): the fit depends only on the relative radii and the conductivities, so the references' values for their default layers are constants of this code,
// obtained by RUNNING bst_berg (Brainstorm: radii 0.88, 0.93, 1.0; 0.33, 0.0042, 0.33 S/m) and MNE-Python's make_sphere_model (see mneEegSphereOptions) and copying their output (not from any source).
// Returns false when the layers are neither of those. Our own fit (fitBergScherg) is the more accurate approximation of the series (spec section 8).
bool recordedBergScherg(const Eigen::VectorXd& radius, const Eigen::VectorXd& conductivity, BergScherg& out);

// MNE-Python's defaults (make_sphere_model): relative radii 0.90, 0.92, 0.97, 1.00 (brain, CSF, skull, scalp), conductivities 0.33, 1.0, 0.004, 0.33 S/m.
EegSphereOptions mneEegSphereOptions();

// The gain of every electrode (the electrode points are taken as given, on the outer shell) for unit dipoles at `sources` [N x 3], with the spheres about `centre`: [M x 3N], V/(A.m).
//   V = (1/sigma_L) sum_j lambda_j (m1 mu_j r0 + m2 e) / (4 pi |mu_j r0|^2) . Q      (Zhang 1995 eq. 4 for a dipole at mu_j r0 in a homogeneous sphere)
// The sources are not tested against the shells (the formula is evaluated as it stands, as the reference does). std::invalid_argument for bad sizes or layers.
Eigen::MatrixXd eegSphereGain(const Eigen::MatrixXd& sources, const Coils& electrodes, const Eigen::Vector3d& centre, const EegSphereOptions& opt = {}, BergScherg* usedParameters = nullptr);

}  // namespace nxr::meeg
