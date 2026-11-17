// nxr::selection (compute) — coupled levels and grid addresses (grid.h).
#include "nxr/selection/grid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>


namespace nxr::selection {

namespace {
constexpr double kFourPi = 4 * M_PI;
constexpr double kTolerance = 1e-12;                                     // a ratio within this of a power of two is that power
}  // namespace

int natural_level(double support) {
  if (!std::isfinite(support) || support < 0) throw std::invalid_argument("natural_level: a support >= 0 samples");
  if (support <= 1) return 0;
  return static_cast<int>(std::ceil(std::log2(support) - kTolerance));
}

int time_level_of_band(double lo_hz, double hi_hz, double sfreq, double cycles, std::optional<int> max_level) {
  if (!(lo_hz > 0) || !(hi_hz >= lo_hz) || !(sfreq > 0) || !(cycles > 0)) throw std::invalid_argument("time_level_of_band: a band 0 < lo <= hi (Hz), sfreq > 0 and cycles > 0");
  const double centre = std::sqrt(lo_hz * hi_hz);
  const int k = natural_level(std::ceil(cycles * sfreq / centre));       // the Studio's `need`: samples holding `cycles` cycles of the centre
  return max_level ? std::min(k, std::max(0, *max_level)) : k;
}

int weyl_level(double lambda, double piece_area) {
  if (!(piece_area > 0) || !std::isfinite(lambda) || lambda < 0) throw std::invalid_argument("weyl_level: an eigenvalue >= 0 and a piece area > 0");
  const double modes = piece_area * lambda / kFourPi;                  // Weyl: lambda(m) = 4 pi m / A
  if (modes <= 1) return 0;
  return static_cast<int>(std::ceil(std::log2(modes) - kTolerance));
}

int space_level_of_band(double lambda_lo, double lambda_hi, double piece_area, int depth) {
  if (depth < 0 || !(lambda_lo >= 0) || !(lambda_hi >= lambda_lo)) throw std::invalid_argument("space_level_of_band: 0 <= lo <= hi and depth >= 0");
  const double centre = lambda_lo > 0 ? std::sqrt(lambda_lo * lambda_hi) : lambda_hi / 2;
  return std::min(weyl_level(centre, piece_area), depth);
}

LevelOf level_of_duration(double seconds, double sfreq, std::optional<int> max_level) {
  if (!(seconds > 0) || !std::isfinite(seconds) || !(sfreq > 0)) throw std::invalid_argument("level_of_duration: a duration > 0 s and a sampling rate > 0 Hz");
  int k = natural_level(seconds * sfreq);
  if (max_level) k = std::min(k, std::max(0, *max_level));
  return LevelOf{k, std::ldexp(1.0, k) / sfreq, "s"};
}

namespace {
// the coarsest level l (0 .. depth) with values[l] <= limit, where the values fall as the levels deepen; the refusal names the finest value
LevelOf coarsest_at_most(const std::vector<double>& values, double limit, const std::string& what, const std::string& unit) {
  if (values.empty() || !(limit > 0) || !std::isfinite(limit)) throw std::invalid_argument("level_of_" + what + ": a " + what + " > 0 and the measured " + what + " of levels 0 .. depth");
  for (std::size_t l = 0; l < values.size(); ++l)
    if (values[l] <= limit * (1 + 1e-9)) return LevelOf{static_cast<int>(l), values[l], unit};
  throw std::invalid_argument("level_of_" + what + ": no level is fine enough: a " + what + " of " + std::to_string(limit) + " " + unit + " is below the finest level's " + std::to_string(values.back()) + " " + unit);
}
}  // namespace

LevelOf level_of_diameter(const std::vector<double>& tile_diameters, double diameter, const std::string& unit) { return coarsest_at_most(tile_diameters, diameter, "diameter", unit); }

LevelOf level_of_area(const std::vector<double>& tile_areas, double area, const std::string& unit) { return coarsest_at_most(tile_areas, area, "area", unit + "^2"); }

LevelOf level_of_eigenvalue(double lambda, double piece_area, int depth, const std::string& unit) {
  if (depth < 0) throw std::invalid_argument("level_of_eigenvalue: depth >= 0");
  const int l = std::min(weyl_level(lambda, piece_area), depth);
  const double top = kFourPi * std::ldexp(1.0, l) / piece_area;                     // the level's band [4 pi 2^(l-1) / A, 4 pi 2^l / A): its shortest wavelength is at the top
  return LevelOf{l, 2 * M_PI / std::sqrt(top), unit};
}

LevelOf level_of_wavelength(double wavelength, double piece_area, int depth, const std::string& unit) {
  if (!(wavelength > 0)) throw std::invalid_argument("level_of_wavelength: a wavelength > 0");
  const double lambda = std::isinf(wavelength) ? 0.0 : std::pow(2 * M_PI / wavelength, 2);
  return level_of_eigenvalue(lambda, piece_area, depth, unit);
}

LevelOf level_of_parts(double parts, int depth) {
  if (!(parts >= 1) || !std::isfinite(parts) || depth < 0) throw std::invalid_argument("level_of_parts: parts >= 1 and depth >= 0");
  const double l = std::log2(parts);
  const double r = std::round(l);
  if (std::abs(l - r) > 1e-9) throw std::invalid_argument("level_of_parts: " + std::to_string(parts) + " parts is not a power of two (the dyadic grid halves each tile): the nearest are " +
                                                         std::to_string(std::ldexp(1.0, static_cast<int>(std::floor(l)))) + " and " + std::to_string(std::ldexp(1.0, static_cast<int>(std::ceil(l)))));
  if (r > depth) throw std::invalid_argument("level_of_parts: " + std::to_string(parts) + " parts needs level " + std::to_string(static_cast<int>(r)) + ", the grid's finest is " + std::to_string(depth));
  return LevelOf{static_cast<int>(r), std::ldexp(1.0, static_cast<int>(r)), "parts"};
}

Support support_of(double lo_hz, double hi_hz, double tc, double sfreq, int log2n, std::optional<int> level, bool clamp) {
  if (log2n < 0 || log2n > 62 || !std::isfinite(tc)) throw std::invalid_argument("support_of: a line of 2^log2n samples (0..62) and a finite centre time");
  const int fit = time_level_of_band(lo_hz, hi_hz, sfreq, kJointCycles, log2n);
  // the Studio counts levels by tiles of the line; here k is the tile's power of two: "never finer than the fit" is k >= fit, "down to FINEST" is k >= kFinestTileLog2
  const int want = level.value_or(fit);
  const int lowest = clamp ? fit : std::min(fit, kFinestTileLog2);
  Support s;
  s.level = std::clamp(want, lowest, log2n);
  s.samples = std::int64_t(1) << s.level;
  const std::int64_t tiles = std::int64_t(1) << (log2n - s.level);
  s.tile = std::clamp<std::int64_t>(static_cast<std::int64_t>(std::floor(tc * sfreq / static_cast<double>(s.samples))), 0, tiles - 1);
  s.t0 = static_cast<double>(s.tile * s.samples) / sfreq;
  s.t1 = static_cast<double>((s.tile + 1) * s.samples) / sfreq;
  return s;
}

Address address_on_line(double center_sample, double extent_samples, std::int64_t samples, int finest, int coarsest) {
  if (!std::isfinite(center_sample) || samples < 1 || finest < 0 || coarsest < finest || coarsest > 62)
    throw std::invalid_argument("address_on_line: a finite centre, a line of samples >= 1 and levels 0 <= finest <= coarsest <= 62");
  Address a;
  a.level = std::clamp(natural_level(extent_samples), finest, coarsest);
  const std::int64_t c = std::clamp<std::int64_t>(std::llround(center_sample), 0, samples - 1);   // the sample nearest the centre
  a.member = c >> a.level;
  return a;
}

CorticalLevel cortical_level_of_extent(double sigma, const std::vector<double>& d, double tile_sigma) {
  if (!(sigma > 0) || !std::isfinite(sigma) || d.empty() || !(tile_sigma > 0))
    throw std::invalid_argument("cortical_level_of_extent: a feature width > 0, the measured tile diameters of levels 0 .. depth and tile_sigma > 0");
  CorticalLevel out;
  const double limit = tile_sigma * sigma;
  for (std::size_t l = 1; l < d.size(); ++l) {                           // coarsest first: a coarser tile pools more samples (jointcell)
    if (d[l] <= limit + 1e-9) {
      out.level = static_cast<int>(l);
      out.why = "level " + std::to_string(l) + ": tiles of diameter " + std::to_string(d[l]) + ", the coarsest <= " + std::to_string(tile_sigma) + " sigma = " + std::to_string(limit);
      return out;
    }
  }
  out.why = "no level is fine enough: a feature of width " + std::to_string(sigma) + " needs tiles <= " + std::to_string(limit) + " (tile_sigma " + std::to_string(tile_sigma) +
            "); the finest here are " + std::to_string(d.back());
  return out;
}

Address dyadic_address(const Selection& g, std::int64_t center_vertex, double radius, double tile_sigma) {
  if (g.type != Type::set || !g.grid || g.grid->kind != Grid::Kind::dyadic) throw std::invalid_argument("dyadic_address: a dyadic grid (partition.h `dyadic`)");
  if (center_vertex < 0 || center_vertex >= static_cast<std::int64_t>(g.member_of.size()))
    throw std::invalid_argument("dyadic_address: centre vertex " + std::to_string(center_vertex) + " outside the grid");
  const std::int32_t code = g.member_of[static_cast<std::size_t>(center_vertex)];
  if (code < 0) throw std::invalid_argument("dyadic_address: centre vertex " + std::to_string(center_vertex) + " is in no tile");
  Address a;
  if (radius == 0) {                                                    // a point-like member (a distance holding its centre): the finest tile
    a.level = g.grid->depth;
  } else {
    std::vector<double> diameters;                                      // the measured tile diameter per level, from the grid's level dictionary
    for (const Selection& level : g.members) {
      if (!level.diameter) throw std::invalid_argument("dyadic_address: the grid's levels carry no measured tile diameter (make it with partition.h `dyadic`)");
      diameters.push_back(*level.diameter);
    }
    if (static_cast<int>(diameters.size()) != g.grid->depth + 1) throw std::invalid_argument("dyadic_address: the grid has no level dictionary (levels 0 .. depth) to read tile diameters from");
    const CorticalLevel c = cortical_level_of_extent(radius, diameters, tile_sigma);
    if (!c.level) throw std::invalid_argument("dyadic_address: " + c.why);
    a.level = *c.level;
  }
  a.member = static_cast<std::int64_t>(code) >> (g.grid->depth - a.level);   // the closed form
  return a;
}

}  // namespace nxr::selection
