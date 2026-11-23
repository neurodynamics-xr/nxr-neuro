// nxr::selection (compute) — KERNEL MEMBERS RESOLVED: a member with a `Profile` (selection.h) is an anchor + a profile + a chart; resolved onto a line it gives the
// elements it weights and each one's weight, per member (CSR). Pure: the line is an array of coordinates (e.g. bin frequencies k fs / n), no file, no stored id.
//
// WEIGHT CONVENTION: a kernel stores the AMPLITUDE g (peak `gain`); `resolve_weights` returns g. A LINEAR field takes g; a QUADRATIC field (a power, a spectrum
// of a product) takes g^2: `quadratic` squares the weights. The caller states which.
//
// Families: `box` (the bins a span selects today, weight `gain`; resolved_h's outward rounding) and `gaussian` (g = gain exp(-(c(f) - c(fc))^2 / (2 s^2)), s = fwhm / (2 sqrt(2 ln 2)),
// c the chart; fc the anchor's centre, the midpoint of its extent). Charts: `linear` (Hz), `log2` (octaves), `log10`. In a log chart the coordinate f <= 0 has no
// chart value: such an element (DC) gets weight 0 and is not listed. Other families are refused as not yet made, by name.
//
// EDGE BINS (DC, and the Nyquist bin when the caller names it): a real transform's interior bins each stand for two (+f and -f), the edge bins for one. The rule is a
// STATED parameter of the resolution, no default: `single` keeps an edge bin once, at its weight; `drop` leaves it out.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/selection/selection.h"

namespace nxr::selection {

enum class EdgeBins { single, drop };

/** the elements each member weights, CSR: member m holds elements[offsets[m] .. offsets[m+1]) with weights[...] (the amplitude g, elements ascending) */
struct WeightedElements {
  std::vector<std::int64_t> offsets;   // members + 1
  std::vector<std::int64_t> elements;
  std::vector<double> weights;
  std::int64_t members() const { return offsets.empty() ? 0 : static_cast<std::int64_t>(offsets.size()) - 1; }
};

struct WeightOptions {
  std::optional<EdgeBins> edge_bins;                  // REQUIRED (std::invalid_argument naming `edge_bins` when absent)
  std::optional<std::int64_t> nyquist_bin;            // the element that is the Nyquist bin, when the line has one (an edge bin beside DC)
};

/** the weights of a member (a point, span or distance with a profile, or a plain span: a box) or of a set of listed members, one member each, on the line of
 *  coordinates `line` (ascending, uniformly spaced for a box). A member without a profile is a box (its span's bins, weight 1). std::invalid_argument names
 *  the missing parameter, the unknown family/chart or the member that is not resolvable. */
WeightedElements resolve_weights(const Selection& selection, const std::vector<double>& line, const WeightOptions& options);

/** the weights squared (a quadratic field's: g^2) */
WeightedElements quadratic(WeightedElements w);

/** why a profile is refused (a missing family, chart or parameter, a non-positive fwhm, a family not yet made), or nothing */
std::optional<std::string> profile_refusal(const Profile& profile);

}  // namespace nxr::selection
