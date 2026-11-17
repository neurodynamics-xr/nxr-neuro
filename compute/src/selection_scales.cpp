// nxr::selection (compute) — named spatial scales on the eigenvalue line (scales.h).
#include "nxr/selection/scales.h"

#include <cmath>
#include <stdexcept>

namespace nxr::selection {

namespace {
constexpr double kTwoPi = 2 * M_PI, kFourPi = 4 * M_PI;

Spans named(std::vector<Span> list, const std::vector<Scale>& scales, const std::string& unit) {
  Lookup lookup;
  for (std::size_t i = 0; i < scales.size(); ++i) lookup.push_back(Entry{static_cast<std::int64_t>(i), scales[i].name, std::nullopt});
  return spans("eigenvalue", "1/" + unit + "^2", std::move(list), std::move(lookup));
}
}  // namespace

Spans scales_by_wavelength(const std::vector<Scale>& scales, const std::string& unit) {
  if (unit.empty()) throw std::invalid_argument("scales_by_wavelength: the wavelengths' unit is needed");
  std::vector<Span> list;
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const Scale& s = scales[i];
    if (!(s.from > 0) || !(s.to > s.from)) throw std::invalid_argument("scales_by_wavelength: '" + s.name + "' needs 0 < from < to");
    // a longer wavelength is a smaller eigenvalue: [from, to) in wavelength is ((2 pi / to)^2, (2 pi / from)^2] in eigenvalue
    const double lo = std::isinf(s.to) ? 0.0 : std::pow(kTwoPi / s.to, 2), hi = std::pow(kTwoPi / s.from, 2);
    list.push_back(Span{lo, hi, static_cast<std::int64_t>(i)});
  }
  return named(std::move(list), scales, unit);
}

Spans scales_by_parts(const std::vector<Scale>& scales, double piece_area, const std::string& unit) {
  if (unit.empty()) throw std::invalid_argument("scales_by_parts: the area's unit is needed");
  if (!(piece_area > 0) || !std::isfinite(piece_area)) throw std::invalid_argument("scales_by_parts: the piece's area is > 0");
  std::vector<Span> list;
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const Scale& s = scales[i];
    if (!(s.from >= 0) || !(s.to > s.from) || !std::isfinite(s.to)) throw std::invalid_argument("scales_by_parts: '" + s.name + "' needs 0 <= from < to");
    list.push_back(Span{kFourPi * s.from / piece_area, kFourPi * s.to / piece_area, static_cast<std::int64_t>(i)});
  }
  return named(std::move(list), scales, unit);
}

}  // namespace nxr::selection
