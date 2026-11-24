#pragma once
// nxr::signal — THE GAINS OF A KERNEL SET: the one evaluator of the selection system's kernel members (an anchor + a profile + a chart) on the points of a line
// (a frequency line in Hz, an eigenvalue line in mu). A kernel set is a selection (`select.kernels` makes and stores it); the filterbanks are how its kernels are
// designed: a family that is a bank's shape is evaluated by that bank's own function, never by a copy of its formula.
//
// FAMILIES (one per set; `Profile::params` holds its numbers, every one stated):
//   gaussian        g = gain exp(-u^2 / (2 sd^2)), sd = FWHM / (2 sqrt(2 ln 2)), u = c(x) - c(centre) in the chart (linear or log2); the FWHM is `fwhm`, or
//                   `fwhm_per_centre` times the centre (constant Q on a linear chart). Exactly one of the two.
//   box             g = gain where |u| <= FWHM / 2, else 0 (the same width parameters).
//   cos2_partition  an amplitude partition in log2 (nxr::signal::cosineSquaredPartition over the set's centres, in order); no parameter; the gains sum to one.
//   morlet_band     timefilterbank::morletBand over the member's span [lo, hi] (linear chart).
//   meyer_band      timefilterbank::meyerBand over the member's span [lo, hi], `half_width` octaves (log2 chart).
//   mexican_hat     graphfilterbank::mexicanHat(x, centre) (linear chart): unit peak at the centre.
// A member's ANCHOR is a span on axis 0: [centre, centre] (start = stop) for gaussian, box, cos2_partition and mexican_hat; [lo, hi] for the band families.
//
// THE EDGE RULE is stated by the caller: `NonPositiveCoordinates::zero`, a coordinate <= 0 has gain 0 (a frequency's mean, an eigenvalue line's constant); the
// cos2 partition takes such a coordinate as -infinity in log2, so its first member is flat there. std::invalid_argument names every refusal.
#include <Eigen/Dense>
#include <string>

#include "nxr/selection/selection.h"

namespace nxr::signal {

enum class NonPositiveCoordinates { zero };

struct KernelGains {
  Eigen::MatrixXd gains;          // [members x points]
  Eigen::VectorXd centres;        // per member, on the line's coordinate (a band's: the midpoint of [lo, hi])
  Eigen::VectorXd fullWidths;     // per member, the FWHM (gaussian) or full width (box) in the chart; NaN for the other families
  std::string family;             // the set's family
  std::string chart;              // the set's chart
};

/** The gains of every member of `kernelSet` (a set whose members are kernels) at the points `line`, under the edge rule `rule`. */
KernelGains kernelGains(const nxr::selection::Selection& kernelSet, const Eigen::VectorXd& line, NonPositiveCoordinates rule);

}  // namespace nxr::signal
