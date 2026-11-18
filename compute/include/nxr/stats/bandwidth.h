#pragma once
// nxr::stats — the bandwidths of a power spectrum or a filter's squared gain: ONE general function each, on a uniform grid (spacing `df`), the old copies (a filter bank's equivalent and noise bandwidth, the lag-free
// surrogates' effective bandwidth) its cases. PURE MATHS on arrays.
//
//   enbw                 the EQUIVALENT NOISE BANDWIDTH of a power response P(f): the width of a rectangle of height `reference` that passes as much white noise, Hz: (integral of P df) / reference.
//                        With reference = max P (the default) it is the textbook ENBW of a filter; a bank member's analytic noise bandwidth is enbw((g/2)^2, df, reference 1).
//   statisticalBandwidth the STATISTICAL (effective) bandwidth: (integral P df)^2 / integral P^2 df, the number of independent samples a second of a signal with this power spectrum holds; a flat spectrum of width B
//                        gives B. A bank member's equivalent bandwidth is statisticalBandwidth(g^2, df).
// `Rule`: Trapezoid weights the two end points by one half (the filter banks' integral), Sum weights every point alike (the lag-free surrogates' sum). They differ by the end points, which the formulas keep apart.
#include <Eigen/Dense>

namespace nxr::stats {

enum class Rule { Trapezoid, Sum };

/** (integral P df) / reference; reference <= 0: max P. Throws std::invalid_argument on an empty or all-zero response. */
double enbw(const Eigen::VectorXd& power, double df, double reference = 0, Rule rule = Rule::Trapezoid);
/** (integral P df)^2 / integral P^2 df. Throws std::invalid_argument when P holds no power. */
double statisticalBandwidth(const Eigen::VectorXd& power, double df, Rule rule = Rule::Trapezoid);

}  // namespace nxr::stats
