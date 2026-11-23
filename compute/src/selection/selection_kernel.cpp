// nxr::selection: kernel members resolved onto a line (include/nxr/selection/kernel.h).
#include <cmath>
#include <stdexcept>

#include "nxr/selection/kernel.h"
#include "nxr/selection/resolved.h"

namespace nxr::selection {
namespace {

[[noreturn]] void refuse(const std::string& m) { throw std::invalid_argument("resolve_weights: " + m); }

bool known_chart(const std::string& c) { return c == "linear" || c == "log2" || c == "log10"; }

// the chart's value of a coordinate; NaN when the chart has none (f <= 0 in a log chart)
double chart_value(const std::string& chart, double f) {
  if (chart == "linear") return f;
  if (!(f > 0)) return std::nan("");
  return chart == "log2" ? std::log2(f) : std::log10(f);
}

double anchor_centre(const Selection& m, const std::vector<double>& line) {
  if (m.type == Type::span) {
    if (m.extents.size() != 1 || m.extents[0].axis != 0) refuse("a kernel's anchor span has one extent on axis 0");
    return 0.5 * (m.extents[0].start + m.extents[0].stop);
  }
  if (m.type == Type::point) {
    if (!m.center || *m.center < 0 || *m.center >= static_cast<std::int64_t>(line.size())) refuse("a kernel's anchor point is an element of the line");
    return line[static_cast<std::size_t>(*m.center)];
  }
  refuse(std::string("a kernel anchored on a ") + type_name(m.type) + " is not yet made");
}

void append_member(const Selection& m, const std::vector<double>& line, const WeightOptions& o, WeightedElements& out) {
  const std::int64_t n = static_cast<std::int64_t>(line.size());
  auto edge = [&](std::int64_t i) { return i == 0 && line[0] == 0.0 ? true : (o.nyquist_bin && i == *o.nyquist_bin); };
  const bool box = !m.profile || m.profile->family == "box";
  if (m.profile) {
    if (const auto why = profile_refusal(*m.profile)) refuse(*why);
  }
  if (box) {
    // the bins a span selects today: resolve_intervals' outward rounding on the line's uniform spacing
    if (m.type != Type::span) refuse(std::string("a box is a span, not a ") + type_name(m.type));
    if (n < 2) refuse("a box needs a line of at least two points");
    const double spacing = line[1] - line[0];
    for (std::int64_t i = 2; i < n; ++i)
      if (std::abs((line[i] - line[i - 1]) - spacing) > 1e-9 * std::abs(spacing)) refuse("a box needs a uniformly spaced line");
    Selection plain = m;
    plain.profile.reset();
    Line ln;
    ln.unit = m.unit;
    ln.count = n;
    ln.origin = line[0];
    ln.spacing = spacing;
    const Intervals iv = resolve_intervals(plain, ln);
    const double g = m.profile ? m.profile->gain : 1.0;
    for (std::int64_t k = iv.offsets[0]; k < iv.offsets[1]; ++k)
      for (std::int64_t i = iv.start[k]; i < iv.stop[k]; ++i) {
        if (edge(i) && *o.edge_bins == EdgeBins::drop) continue;
        out.elements.push_back(i);
        out.weights.push_back(g);
      }
    return;
  }
  const Profile& p = *m.profile;   // gaussian
  const double fc = chart_value(p.chart, anchor_centre(m, line));
  if (std::isnan(fc)) refuse("the kernel's centre has no value in the chart " + p.chart);
  const double s = p.params.at("fwhm") / (2.0 * std::sqrt(2.0 * std::log(2.0)));
  for (std::int64_t i = 0; i < n; ++i) {
    const double c = chart_value(p.chart, line[static_cast<std::size_t>(i)]);
    if (std::isnan(c)) continue;                                  // DC in a log chart: weight 0
    if (edge(i) && *o.edge_bins == EdgeBins::drop) continue;
    const double g = p.gain * std::exp(-(c - fc) * (c - fc) / (2.0 * s * s));
    if (g > 0) { out.elements.push_back(i); out.weights.push_back(g); }
  }
}

}  // namespace

std::optional<std::string> profile_refusal(const Profile& p) {
  if (p.family.empty()) return std::string("a kernel's profile states its family");
  if (p.chart.empty()) return std::string("a kernel's profile states its chart (linear | log2 | log10)");
  if (!known_chart(p.chart)) return "unknown chart '" + p.chart + "' (linear | log2 | log10)";
  if (!std::isfinite(p.gain)) return std::string("a kernel's profile states a finite gain");
  if (p.family == "box") return std::nullopt;
  if (p.family == "gaussian") {
    const auto f = p.params.find("fwhm");
    if (f == p.params.end()) return std::string("a gaussian profile states the parameter fwhm");
    if (!(f->second > 0) || !std::isfinite(f->second)) return std::string("a gaussian profile's fwhm is positive");
    return std::nullopt;
  }
  return "the profile family '" + p.family + "' is not yet made (box, gaussian)";
}

WeightedElements resolve_weights(const Selection& selection, const std::vector<double>& line, const WeightOptions& options) {
  if (!options.edge_bins) refuse("the parameter edge_bins (single | drop) is not stated");
  WeightedElements out;
  out.offsets.push_back(0);
  if (selection.type == Type::set) {
    if (selection.members.empty() && selection.tiling) refuse("a set of listed members, not a tiling");
    for (const Selection& m : selection.members) {
      if (m.type == Type::set) refuse("a member that is a set");
      append_member(m, line, options, out);
      out.offsets.push_back(static_cast<std::int64_t>(out.elements.size()));
    }
  } else {
    append_member(selection, line, options, out);
    out.offsets.push_back(static_cast<std::int64_t>(out.elements.size()));
  }
  return out;
}

WeightedElements quadratic(WeightedElements w) {
  for (double& g : w.weights) g *= g;
  return w;
}

}  // namespace nxr::selection
