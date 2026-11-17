#pragma once
// nxr::quantity (compute) — A QUANTITY: a number (or a range) with a unit, in a named chart of the space it lies in (spec docs/specs/2026-11-17-consolidation.md,
// section 3). Every selection parameter of a definition, preset or command is a quantity ("8-12 Hz", "2.5 s", "tile 20 mm", "wavelength 40 mm", "parts 16/hemisphere",
// "area 100-400 mm^2", "index:5"); a bare level index is never a parameter, an index is written through the named `index` chart only.
//
// Ported from the Studio's schema (`space` and `coordinate_system`): a SPACE has several charts, one the REFERENCE, the others defined by a TRANSITION from the reference
// (identity, affine, log, power, operator), each with a unit. The charts are read from io (the space's coordinate_system rows) and handed here as data (`Chart`).
// Pure functions of text and numbers: no file, no row. std::invalid_argument names what is refused and what was expected.
#include <string>
#include <vector>

namespace nxr::quantity {

/** a stated quantity: a value has lo == hi; an open range has an infinite end ("at least 5 mm": hi = +infinity). `chart` is one of linear, log10, clock, eigenvalue,
 *  wavelength, area, diameter, parts, index (a chart word of the text, else the one the unit implies: Hz linear, s clock, mm diameter, mm^2 area, 1/m^2 eigenvalue);
 *  `unit` is canonical (superscript two as ^2). */
struct Quantity {
  double lo = 0, hi = 0;
  std::string unit, chart;
  bool is_value() const { return lo == hi; }
};

/** one chart of a space, as the schema stores it: `transition` is the map FROM THE REFERENCE chart to this one, coordinate = f(reference):
 *    identity   c = r
 *    affine     c = params[0] r + params[1]               (scale, offset; millimetres from metres: 1000, 0)
 *    log        c = log_b(r), b = params[0] (10 when none)
 *    power      c = params[0] r^params[1]                  (a wavelength from an eigenvalue: 2 pi, -1/2)
 *    operator   defined by an operator: not computable here (refused) */
struct Chart {
  std::string name, unit;
  std::string transition = "identity";
  std::vector<double> params;
};

/** "8-12 Hz", "2.5 s", "20 mm", "tile 20 mm", "wavelength 40 mm", "parts 16", "parts 16/hemisphere", "area 100-400 mm^2", ">= 5 mm", "index:5".
 *  std::invalid_argument for text it cannot read, a range with lo > hi, a unit with no chart and no chart word. */
Quantity parse(const std::string& text);
/** the text of a quantity as `parse` reads it (for provenance): "8-12 Hz", "wavelength 40 mm" */
std::string format(const Quantity&);

/** `v` (in the quantity's unit and chart) as a coordinate of the REFERENCE chart of the space, by the chart's transition (inverted) after the unit is made the chart's
 *  (m, mm, cm, um; s, ms; Hz, kHz). The chart is the quantity's own: std::invalid_argument when its name differs, when the unit is not the chart's (or convertible to it:
 *  the message names the chart's unit), or for an `operator` transition. */
double to_reference(const Quantity& q, double v, const Chart& chart);
/** both ends in the reference chart, ordered lo <= hi (a decreasing map swaps them) */
Quantity to_reference(const Quantity& q, const Chart& chart);

/** the canonical spelling of a unit ("mm²" and "mm2" to "mm^2", "µm" to "um", "1/m²" to "1/m^2") */
std::string canonical_unit(const std::string& unit);
/** the factor that turns a value in `from` into `to` (the same dimension: length, time, frequency, area, eigenvalue); std::invalid_argument for units of different
 *  dimensions or an unknown unit */
double unit_factor(const std::string& from, const std::string& to);

/** the index written through the `index` chart ("index:5"): its integer; std::invalid_argument for a quantity of another chart */
long long as_index(const Quantity&);

}  // namespace nxr::quantity
