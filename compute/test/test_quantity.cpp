// nxr::quantity, the levels it resolves, the band sets and the time rule, on synthetic data only:
//   - the parser: values, ranges, open ranges, chart words, units with superscripts, the index chart, refusals with the reason
//   - the charts' transitions (identity, affine, log, power: a wavelength from an eigenvalue) and unit conversion
//   - the levels of section 3.3 of the consolidation spec: a duration, a band, a tile diameter, an area, a wavelength, parts, each with its snapped quantity
//   - the band sets (frequency, eigenvalue, wavelength, parts, levels), their families
//   - the time rule: tiles, band, frames (per tile, per cycle, an origin), segments, windows, chunks
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "nxr/quantity.h"
#include "nxr/selection/bands.h"
#include "nxr/selection/grid.h"
#include "nxr/selection/partition.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)
template <class F> static bool refused(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

namespace q = nxr::quantity;
namespace sel = nxr::selection;

static void parsing() {
  std::cout << "=== the parser ===\n";
  q::Quantity a = q::parse("8-12 Hz");
  EXPECT(a.lo == 8 && a.hi == 12 && a.unit == "Hz" && a.chart == "linear", "a frequency range");
  a = q::parse("2.5 s");
  EXPECT(a.is_value() && a.lo == 2.5 && a.chart == "clock", "a duration");
  a = q::parse("tile 20 mm");
  EXPECT(a.lo == 20 && a.unit == "mm" && a.chart == "diameter", "a tile diameter");
  a = q::parse("wavelength 40 mm");
  EXPECT(a.chart == "wavelength" && a.lo == 40 && a.unit == "mm", "a wavelength");
  a = q::parse("area 100-400 mm\xc2\xb2");
  EXPECT(a.chart == "area" && a.lo == 100 && a.hi == 400 && a.unit == "mm^2", "an area range with a superscript");
  a = q::parse("parts 16/hemisphere");
  EXPECT(a.chart == "parts" && a.lo == 16 && a.unit == "hemisphere", "parts per hemisphere");
  a = q::parse(">= 5 mm");
  EXPECT(a.lo == 5 && std::isinf(a.hi) && a.chart == "diameter", "an open range");
  a = q::parse("index:5");
  EXPECT(a.chart == "index" && q::as_index(a) == 5, "the one way to state an index");
  a = q::parse("log10 1-2 Hz");
  EXPECT(a.chart == "log10" && a.unit == "Hz", "a chart word with a unit");
  EXPECT(q::format(q::parse("8-12 Hz")) == "8-12 Hz" && q::format(q::parse("wavelength 40 mm")) == "wavelength 40 mm", "format reads back");
  EXPECT(refused([] { q::parse("5"); }) && refused([] { q::parse("12-8 Hz"); }) && refused([] { q::parse(""); }) && refused([] { q::parse("fast 5 Hz"); }) && refused([] { q::parse("5 furlongs"); }),
         "a bare number, a reversed range, empty text, an unknown word and an unknown unit are refused");
  EXPECT(refused([] { q::as_index(q::parse("5 mm")); }), "an index is stated through the index chart only");
}

static void charts() {
  std::cout << "=== charts: transitions and units ===\n";
  q::Chart linear{"linear", "Hz", "identity", {}}, log10{"log10", "Hz", "log", {10}}, mm{"diameter", "mm", "affine", {1000.0, 0.0}}, wl{"wavelength", "mm", "power", {2 * M_PI * 1000.0, -0.5}};
  EXPECT(std::abs(q::to_reference(q::parse("10 Hz"), 10, linear) - 10) < 1e-12 && std::abs(q::to_reference(q::parse("2 kHz"), 2, linear) - 2000) < 1e-9, "identity, and kHz converted to the chart's Hz");
  EXPECT(std::abs(q::to_reference(q::parse("log10 1 Hz"), 1, log10) - 10) < 1e-12, "log10 chart: 1 is 10 Hz");
  EXPECT(std::abs(q::to_reference(q::parse("tile 20 mm"), 20, mm) - 0.02) < 1e-12, "affine chart: a chart in mm of a reference in metres");
  // a wavelength of 40 mm: lambda = (2 pi / 0.04 m)^2 in 1/m^2 (the chart's power map c = 2 pi 1000 lambda^(-1/2), c in mm)
  const double lambda = q::to_reference(q::parse("wavelength 40 mm"), 40, wl);
  EXPECT(std::abs(lambda - std::pow(2 * M_PI * 1000.0 / 40.0, 2)) < 1e-6 * lambda, "a wavelength is an eigenvalue by the power map");
  const q::Quantity r = q::to_reference(q::parse("wavelength 20-40 mm"), wl);
  EXPECT(r.lo < r.hi && std::abs(r.lo - lambda) < 1e-6 * lambda, "a decreasing map swaps the ends");
  EXPECT(refused([&] { q::to_reference(q::parse("10 s"), 10, linear); }) && refused([&] { q::to_reference(q::parse("10 mm"), 10, linear); }), "another chart and another dimension are refused");
  EXPECT(refused([&] { q::Chart op{"linear", "Hz", "operator", {}}; q::to_reference(q::parse("linear 1 Hz"), 1, op); }), "a chart defined by an operator cannot be computed here");
}

static void levels() {
  std::cout << "=== levels resolved from quantities ===\n";
  const double fs = 1200;
  const sel::LevelOf d = sel::level_of_duration(2.0, fs);
  EXPECT(d.level == 12 && std::abs(d.snapped - 4096.0 / fs) < 1e-12 && d.unit == "s", "2 s at 1200 Hz: 2400 samples, tiles of 2^12, snapped to 3.413 s");
  const sel::LevelOf d2 = sel::level_of_duration(100, fs, 10);
  EXPECT(d2.level == 10, "held at most the line's level");
  // diameters falling with the level (the surface ladder's measured mean tile diameter)
  const std::vector<double> diam = {200, 120, 70, 40, 22, 12};
  const sel::LevelOf c = sel::level_of_diameter(diam, 50, "mm");
  EXPECT(c.level == 3 && c.snapped == 40, "the coarsest level whose tile is at most 50 mm");
  EXPECT(refused([&] { sel::level_of_diameter(diam, 5, "mm"); }), "no level is fine enough: refused");
  const std::vector<double> areas = {40000, 14400, 4900, 1600, 484, 144};
  EXPECT(sel::level_of_area(areas, 2000, "mm").level == 3, "the coarsest level whose tile area is at most 2000 mm^2");
  // Weyl: level l pairs with [4 pi 2^(l-1) / A, 4 pi 2^l / A); a piece of area 100000 mm^2
  const double A = 100000.0;
  const sel::LevelOf w = sel::level_of_wavelength(50, A, 8, "mm");
  const double lambda = std::pow(2 * M_PI / 50.0, 2);
  EXPECT(w.level == sel::weyl_level(lambda, A) && w.unit == "mm" && w.snapped <= 50.0 + 1e-9, "a wavelength is the Weyl level holding it; snapped to the level's shortest wavelength");
  EXPECT(sel::level_of_eigenvalue(lambda, A, 8, "mm").level == w.level, "an eigenvalue gives the same level");
  EXPECT(sel::level_of_parts(16, 6).level == 4 && sel::level_of_parts(16, 6).snapped == 16 && refused([] { sel::level_of_parts(12, 6); }) && refused([] { sel::level_of_parts(128, 6); }), "parts: exact powers of two within the depth");
}

static void bands() {
  std::cout << "=== band sets ===\n";
  const sel::Selection oct = sel::bands_by_frequency(sel::octave_bands());
  EXPECT(sel::count(oct) == 6 && oct.family == sel::Family::grid, "the octave bank: six bands, adjacent with one ratio: a grid");
  const sel::Selection named = sel::bands_by_frequency({{"alpha", 8, 12}, {"beta", 13, 30}});
  EXPECT(named.family == sel::Family::irregular && sel::count(named) == 2 && named.lookup.size() == 2 && named.lookup[0].name == "alpha", "named bands with a gap: irregular, named through the lookup");
  EXPECT(refused([] { sel::bands_by_frequency({}); }) && refused([] { sel::bands_by_frequency({{"x", 5, 5}}); }), "no bands and an empty band are refused");
  const sel::Selection wl = sel::bands_by_wavelength({{"region", 20, 80}, {"hemisphere", 80, std::numeric_limits<double>::infinity()}}, "mm");
  EXPECT(sel::count(wl) == 2 && wl.unit == "1/mm^2", "scales by wavelength become eigenvalue spans");
  const sel::Selection parts = sel::bands_by_parts({{"lobe", 2, 8}}, 100000.0, "mm");
  EXPECT(sel::count(parts) == 1, "scales by parts");
  const sel::BandSpec b{"", 0, 1};
  EXPECT(refused([&] { sel::bins_of_band(0.1, 0.2, 100, 100); }) && sel::bins_of_band(8, 12, 1024, 1024).first == 8 && sel::bins_of_band(8, 12, 1024, 1024).second == 12, "the bins of a band on a transform's line; a band holding none is refused");
  (void)b;
}

static void time_rules() {
  std::cout << "=== the time rule ===\n";
  const double fs = 1000;
  const sel::Line line{"s", 100000, 0.0, 1.0 / fs};
  sel::TimeRule t;
  t.kind = sel::TimeRule::Kind::tiles; t.duration_s = 0.5;
  sel::TimeGrid g = sel::time_rule_resolved(line, t);
  EXPECT(g.level == 9 && g.width_samples == 512 && g.hop_samples == 512 && sel::count(g.grid) == 196 && g.grid.level == 9, "tiles of at least 0.5 s: 2^9 samples, 196 tiles, one level of the dyadic time grid");
  t.kind = sel::TimeRule::Kind::band; t.band_hz = std::make_pair(8.0, 12.0);
  g = sel::time_rule_resolved(line, t);
  EXPECT(g.level == sel::time_level_of_band(8, 12, fs), "a band's tile holds 4 cycles of its centre");
  t.kind = sel::TimeRule::Kind::chunks;
  EXPECT(sel::time_rule_resolved(line, t).level == 15, "the chunk grid is level 15");
  sel::TimeRule s;
  s.kind = sel::TimeRule::Kind::segments; s.duration_s = 4.0; s.overlap = 0.5; s.segment_rule = "round";
  g = sel::time_rule_resolved(line, s);
  EXPECT(g.width_samples == 4000 && g.hop_samples == 2000 && sel::count(g.grid) == 49, "Welch segments of 4 s, 50 % overlap: 49 whole segments");
  s.segment_rule = "tile"; s.duration_s = 4.0;
  g = sel::time_rule_resolved(line, s);
  EXPECT(g.width_samples == 4096 && g.hop_samples == 2048, "the tile rule snaps the segment to 2^12 samples");
  sel::TimeRule w;
  w.kind = sel::TimeRule::Kind::windows; w.duration_s = 1.0; w.step_s = 0.25;
  g = sel::time_rule_resolved(line, w);
  EXPECT(g.width_samples == 1024 && g.hop_samples == 256 && sel::count(g.grid) == (100000 - 1024) / 256 + 1, "windows and steps snapped to 2^k samples");
  // an origin: a point, or an event's span (its start); the grid runs from there
  sel::TimeRule o = t;
  o.kind = sel::TimeRule::Kind::tiles; o.duration_s = 0.5; o.origin = sel::point(10000);
  g = sel::time_rule_resolved(line, o);
  EXPECT(g.origin_sample == 10000 && sel::count(g.grid) == (90000 + 511) / 512, "an origin at a point starts the first tile there");
  o.origin = sel::span({sel::Extent{0, 20.0, 30.0}});
  EXPECT(sel::time_rule_resolved(line, o).origin_sample == 20000, "an origin given as an event: its start");
  sel::TimeRule bad = o; bad.origin = sel::point(-3);
  EXPECT(refused([&] { sel::time_rule(line, bad); }) && refused([&] { sel::TimeRule none; none.kind = sel::TimeRule::Kind::tiles; sel::time_rule(line, none); }), "an origin outside the line, and tiles with no duration, are refused");
}

int main() {
  parsing();
  charts();
  levels();
  bands();
  time_rules();
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
