// nxr::selection (compute) — the band sets (bands.h).
#include "nxr/select/bands.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace nxr::select {

namespace {
constexpr double kTwoPi = 2 * M_PI, kFourPi = 4 * M_PI;

std::string number(double x) { char b[32]; std::snprintf(b, sizeof b, "%.4g", x); return b; }

bool near(double a, double b) { return std::abs(a - b) <= 1e-9 * std::max(1.0, std::max(std::abs(a), std::abs(b))); }

// the bands as spans, the lookup naming them; the family from the listing: adjacent bands with one ratio hi / lo are a GRID
Selection make(const std::vector<BandSpec>& bands, const std::string& unit, bool must_tile) {
  if (bands.empty()) throw std::invalid_argument("bands: there are none");
  std::vector<SpanRow> rows;
  Lookup lookup;
  for (std::size_t i = 0; i < bands.size(); ++i) {
    const BandSpec& b = bands[i];
    if (!(b.lo >= 0) || !(b.hi > b.lo) || std::isnan(b.hi)) throw std::invalid_argument("bands: '" + b.name + "' needs 0 <= lo < hi (got " + number(b.lo) + " to " + number(b.hi) + " " + unit + ")");
    rows.push_back(SpanRow{b.lo, b.hi, static_cast<std::int64_t>(i)});
    lookup.push_back(Entry{static_cast<std::int64_t>(i), b.name.empty() ? number(b.lo) + "-" + number(b.hi) + " " + unit : b.name, std::nullopt});
  }
  bool adjacent = true, one_ratio = bands.front().lo > 0;
  for (std::size_t i = 0; i + 1 < bands.size(); ++i) {
    // in ascending order or highest first: adjacent when one band's edge is the next one's other edge
    const bool asc = near(bands[i].hi, bands[i + 1].lo), desc = near(bands[i].lo, bands[i + 1].hi);
    if (!(asc || desc)) adjacent = false;
  }
  if (must_tile && !adjacent) {
    std::vector<std::pair<double, double>> b;
    for (const BandSpec& x : bands) b.emplace_back(x.lo, x.hi);
    std::sort(b.begin(), b.end());
    for (std::size_t i = 0; i + 1 < b.size(); ++i)
      if (!near(b[i].second, b[i + 1].first)) throw std::invalid_argument("bands: a gap or an overlap at " + number(b[i].second) + " " + unit);
  }
  if (adjacent && one_ratio) {
    const double r = bands.front().hi / bands.front().lo;
    for (const BandSpec& b : bands) if (!(b.lo > 0) || !near(b.hi / b.lo, r)) one_ratio = false;
  } else {
    one_ratio = false;
  }
  Selection s = set_of_spans(std::move(rows), unit, std::move(lookup));
  s.family = (adjacent && (one_ratio || must_tile)) ? Family::grid : Family::irregular;
  return s;
}
}  // namespace

Selection bands_by_frequency(const std::vector<BandSpec>& bands, const std::string& unit) { return make(bands, unit, false); }

const std::vector<BandSpec>& band_preset(const std::string& name) {
  static const std::vector<BandSpec> brainstorm = {{"delta", 2, 4}, {"theta", 5, 7}, {"alpha", 8, 12}, {"beta", 15, 29}, {"gamma1", 30, 59}, {"gamma2", 60, 90}};
  if (name == "brainstorm") return brainstorm;
  throw std::invalid_argument("the band preset is 'brainstorm', not '" + name + "'");
}

Selection bands_preset(const std::string& name) { return bands_by_frequency(band_preset(name), "Hz"); }

Selection bands_by_eigenvalue(const std::vector<BandSpec>& bands, const std::string& unit) {
  if (unit.empty()) throw std::invalid_argument("bands_by_eigenvalue: the eigenvalues' unit is needed (1/mm^2, 1/m^2)");
  return make(bands, unit, false);
}

Selection bands_by_wavelength(const std::vector<Scale>& scales, const std::string& unit) {
  if (unit.empty()) throw std::invalid_argument("bands_by_wavelength: the wavelengths' unit is needed");
  std::vector<BandSpec> list;
  for (const Scale& s : scales) {
    if (!(s.from > 0) || !(s.to > s.from)) throw std::invalid_argument("bands_by_wavelength: '" + s.name + "' needs 0 < from < to");
    // a longer wavelength is a smaller eigenvalue: [from, to) in wavelength is ((2 pi / to)^2, (2 pi / from)^2] in eigenvalue
    list.push_back(BandSpec{s.name, std::isinf(s.to) ? 0.0 : std::pow(kTwoPi / s.to, 2), std::pow(kTwoPi / s.from, 2)});
  }
  return make(list, "1/" + unit + "^2", false);
}

Selection bands_by_parts(const std::vector<Scale>& scales, double piece_area, const std::string& unit) {
  if (unit.empty()) throw std::invalid_argument("bands_by_parts: the area's unit is needed");
  if (!(piece_area > 0) || !std::isfinite(piece_area)) throw std::invalid_argument("bands_by_parts: the piece's area is > 0");
  std::vector<BandSpec> list;
  for (const Scale& s : scales) {
    if (!(s.from >= 0) || !(s.to > s.from) || !std::isfinite(s.to)) throw std::invalid_argument("bands_by_parts: '" + s.name + "' needs 0 <= from < to");
    list.push_back(BandSpec{s.name, kFourPi * s.from / piece_area, kFourPi * s.to / piece_area});
  }
  return make(list, "1/" + unit + "^2", false);
}

Selection bands_by_levels(const Selection& levels) {
  std::vector<BandSpec> list;
  std::string unit;
  for (const Selection& l : levels.members)
    if (l.band) { list.push_back(BandSpec{{}, l.band->first, l.band->second}); unit = l.band_unit; }
  if (list.empty()) throw std::invalid_argument("bands_by_levels: the selection has no levels with bands");
  return make(list, unit, true);
}

std::vector<BandSpec> octave_bands(double f_lo, int n) {
  if (!(f_lo > 0) || n < 1) throw std::invalid_argument("octave_bands: f_lo > 0 and n >= 1");
  std::vector<BandSpec> out;
  for (int b = 0; b < n; ++b) out.push_back(BandSpec{{}, f_lo * std::ldexp(1.0, b), f_lo * std::ldexp(1.0, b + 1)});
  return out;
}

std::vector<BandSpec> bands_of_design(const ConstantQDesign& design) {
  std::vector<BandSpec> out;
  for (const auto& b : design.bands) out.push_back(BandSpec{{}, b.first, b.second});
  std::sort(out.begin(), out.end(), [](const BandSpec& a, const BandSpec& b) { return a.lo < b.lo; });
  return out;
}

std::pair<std::int64_t, std::int64_t> bins_of_band(double lo_hz, double hi_hz, std::int64_t tile_samples, double sfreq) {
  if (tile_samples < 1 || !(sfreq > 0) || !(hi_hz > lo_hz) || lo_hz < 0) throw std::invalid_argument("bins_of_band: a band 0 <= lo < hi (Hz), a tile of at least 1 sample and sfreq > 0");
  const double df = sfreq / static_cast<double>(tile_samples);
  const std::int64_t nbins = tile_samples / 2 + 1;
  const std::int64_t first = static_cast<std::int64_t>(std::ceil(lo_hz / df - 1e-9));
  const std::int64_t last = std::min(static_cast<std::int64_t>(std::ceil(hi_hz / df - 1e-9)), nbins);
  if (last <= first || first >= nbins) throw std::invalid_argument("bins_of_band: the band " + number(lo_hz) + "-" + number(hi_hz) + " Hz holds no bin of a " + std::to_string(tile_samples) + "-sample tile at " + number(sfreq) + " Hz (spacing " + number(df) + " Hz)");
  return {first, last};
}

}  // namespace nxr::select
