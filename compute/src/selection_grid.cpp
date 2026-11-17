// nxr::selection (compute) — coupled levels and grid addresses (grid.h).
#include "nxr/selection/grid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "nxr/selection/constant_q.h"

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

int time_level_of_band(double lo_hz, double sfreq, std::optional<double> cycles) {
  if (!(lo_hz > 0) || !(sfreq > 0)) throw std::invalid_argument("time_level_of_band: a band's low edge > 0 Hz on a line of sfreq > 0");
  return natural_level(static_cast<double>(constantq::windowFor(sfreq, lo_hz, cycles)));
}

int weyl_level(double lambda, double piece_area) {
  if (!(piece_area > 0) || !std::isfinite(lambda) || lambda < 0) throw std::invalid_argument("weyl_level: an eigenvalue >= 0 and a piece area > 0");
  const double modes = piece_area * lambda / kFourPi;                  // Weyl: lambda(m) = 4 pi m / A
  if (modes <= 1) return 0;
  return static_cast<int>(std::ceil(std::log2(modes) - kTolerance));
}

int space_level_of_band(double lambda_lo, double piece_area, int depth) {
  if (depth < 0) throw std::invalid_argument("space_level_of_band: depth >= 0");
  return std::min(weyl_level(lambda_lo, piece_area), depth);
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

int dyadic_level_of_extent(double radius, double piece_area, int depth) {
  if (!(piece_area > 0) || !std::isfinite(radius) || radius < 0 || depth < 0) throw std::invalid_argument("dyadic_level_of_extent: a radius >= 0, a piece area > 0, depth >= 0");
  if (radius == 0) return depth;
  const double ratio = piece_area / (M_PI * radius * radius);           // tiles a piece whose area still holds the disc
  if (ratio < 1) return 0;
  return std::min(depth, static_cast<int>(std::floor(std::log2(ratio) + kTolerance)));
}

Address dyadic_address(const Selection& g, std::int64_t center_vertex, double radius, double piece_area) {
  if (g.type != Type::set || !g.grid || g.grid->kind != Grid::Kind::dyadic) throw std::invalid_argument("dyadic_address: a dyadic grid (partition.h `dyadic`)");
  if (center_vertex < 0 || center_vertex >= static_cast<std::int64_t>(g.member_of.size()))
    throw std::invalid_argument("dyadic_address: centre vertex " + std::to_string(center_vertex) + " outside the grid");
  const std::int32_t code = g.member_of[static_cast<std::size_t>(center_vertex)];
  if (code < 0) throw std::invalid_argument("dyadic_address: centre vertex " + std::to_string(center_vertex) + " is in no tile");
  Address a;
  a.level = dyadic_level_of_extent(radius, piece_area, g.grid->depth);
  a.member = static_cast<std::int64_t>(code) >> (g.grid->depth - a.level);   // the closed form
  return a;
}

}  // namespace nxr::selection
