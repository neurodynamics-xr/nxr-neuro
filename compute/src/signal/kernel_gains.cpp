// nxr::signal::kernelGains: the one evaluator of a kernel set (include/nxr/signal/kernel_gains.h).
#include "nxr/signal/kernel_gains.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/spectral_flow.h"
#include "nxr/signal/timefilterbank.h"

namespace nxr::signal {
namespace {

[[noreturn]] void refuse(const std::string& why) { throw std::invalid_argument("kernelGains: " + why); }

bool anchoredAtCentre(const std::string& family) { return family == "gaussian" || family == "box" || family == "cos2_partition" || family == "mexican_hat"; }
bool isBand(const std::string& family) { return family == "morlet_band" || family == "meyer_band"; }

const char* chartOf(const std::string& family) {
  if (family == "cos2_partition" || family == "meyer_band") return "log2";
  if (family == "morlet_band" || family == "mexican_hat") return "linear";
  return nullptr;   // gaussian, box: linear or log2, the caller's
}

double parameter(const nxr::selection::Profile& profile, const char* name) {
  const auto it = profile.params.find(name);
  if (it == profile.params.end()) refuse("a " + profile.family + " profile states the parameter " + name);
  if (!std::isfinite(it->second) || !(it->second > 0)) refuse(std::string("the parameter ") + name + " is a positive number");
  return it->second;
}

// the FWHM of a gaussian or box member: `fwhm`, or `fwhm_per_centre` times the centre; exactly one is stated
double fullWidthOf(const nxr::selection::Profile& profile, double centre) {
  const bool absolute = profile.params.count("fwhm") > 0, proportional = profile.params.count("fwhm_per_centre") > 0;
  if (absolute == proportional) refuse("a " + profile.family + " profile states exactly one of fwhm and fwhm_per_centre");
  if (absolute) return parameter(profile, "fwhm");
  double fwhm = parameter(profile, "fwhm_per_centre");
  fwhm *= centre;
  return fwhm;
}

}  // namespace

KernelGains kernelGains(const nxr::selection::Selection& kernelSet, const Eigen::VectorXd& line, NonPositiveCoordinates rule) {
  if (rule != NonPositiveCoordinates::zero) refuse("the edge rule is not stated");
  if (kernelSet.type != nxr::selection::Type::set) refuse("a kernel set is a set of kernel members");
  const auto& members = kernelSet.members;
  const Eigen::Index M = static_cast<Eigen::Index>(members.size()), N = line.size();
  if (M == 0) refuse("a kernel set with no member");

  KernelGains out;
  for (const auto& m : members) {
    if (!m.profile) refuse("a member without a profile is not a kernel");
    if (m.type != nxr::selection::Type::span || m.extents.size() != 1 || m.extents[0].axis != 0) refuse("a kernel's anchor is a span on axis 0");
  }
  out.family = members.front().profile->family;
  out.chart = members.front().profile->chart;
  for (const auto& m : members)
    if (m.profile->family != out.family || m.profile->chart != out.chart) refuse("a kernel set's members have one family and one chart");
  const std::string& family = out.family;
  if (!anchoredAtCentre(family) && !isBand(family)) refuse("the family '" + family + "' is not one kernelGains holds (gaussian, box, cos2_partition, morlet_band, meyer_band, mexican_hat)");
  if (const char* chart = chartOf(family)) {
    if (out.chart != chart) refuse("a " + family + " profile lies in the " + chart + " chart");
  } else if (out.chart != "linear" && out.chart != "log2") {
    refuse("the chart of a " + family + " profile is linear or log2");
  }

  out.gains = Eigen::MatrixXd::Zero(M, N);
  out.centres.resize(M);
  out.fullWidths = Eigen::VectorXd::Constant(M, std::numeric_limits<double>::quiet_NaN());
  for (Eigen::Index k = 0; k < M; ++k) {
    const auto& extent = members[static_cast<std::size_t>(k)].extents[0];
    if (anchoredAtCentre(family) && extent.start != extent.stop) refuse("a " + family + " member is anchored at its centre (a span with start = stop)");
    if (isBand(family) && !(extent.stop > extent.start)) refuse("a " + family + " member is a band [lo, hi] with lo < hi");
    out.centres(k) = anchoredAtCentre(family) ? extent.start : 0.5 * (extent.start + extent.stop);
    if (!(out.centres(k) > 0)) refuse("a kernel's centre is above 0");
  }

  if (family == "cos2_partition") {
    std::vector<double> log2Centres;
    for (Eigen::Index k = 0; k < M; ++k) log2Centres.push_back(std::log2(out.centres(k)));
    Eigen::VectorXd log2Coordinate(N);
    for (Eigen::Index i = 0; i < N; ++i) log2Coordinate(i) = line(i) > 0 ? std::log2(line(i)) : -std::numeric_limits<double>::infinity();
    out.gains = cosineSquaredPartition(log2Coordinate, log2Centres);
    for (Eigen::Index k = 0; k < M; ++k) {
      const double gain = members[static_cast<std::size_t>(k)].profile->gain;
      if (gain != 1.0) out.gains.row(k) *= gain;
    }
    return out;
  }

  for (Eigen::Index k = 0; k < M; ++k) {
    const auto& member = members[static_cast<std::size_t>(k)];
    const nxr::selection::Profile& profile = *member.profile;
    const double centre = out.centres(k), gain = profile.gain, lo = member.extents[0].start, hi = member.extents[0].stop;
    if (!std::isfinite(gain)) refuse("a kernel's gain is a finite number");
    double fwhm = 0, halfWidth = 0;
    if (family == "gaussian" || family == "box") { fwhm = fullWidthOf(profile, centre); out.fullWidths(k) = fwhm; }
    if (family == "meyer_band") halfWidth = parameter(profile, "half_width");
    const bool logChart = out.chart == "log2";
    for (Eigen::Index i = 0; i < N; ++i) {
      const double x = line(i);
      if (!(x > 0)) continue;   // the edge rule: a coordinate <= 0 has gain 0
      if (family == "gaussian" || family == "box") {
        const double u = logChart ? std::log2(x) - std::log2(centre) : x - centre;
        if (family == "gaussian") {
          const double sd = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
          out.gains(k, i) = gain * std::exp(-u * u / (2.0 * sd * sd));
        } else {
          out.gains(k, i) = std::abs(u) <= fwhm / 2.0 ? gain : 0.0;
        }
      } else if (family == "morlet_band") {
        out.gains(k, i) = gain * timefilterbank::morletBand(x, lo, hi);
      } else if (family == "meyer_band") {
        out.gains(k, i) = gain * timefilterbank::meyerBand(x, lo, hi, halfWidth);
      } else {   // mexican_hat
        out.gains(k, i) = gain * graphfilterbank::mexicanHat(x, centre);
      }
    }
  }
  return out;
}

}  // namespace nxr::signal
