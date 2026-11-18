// nxr::selection (compute) — THE BAND SETS (spec docs/specs/2026-11-17-consolidation.md, section 1.3): `select.bands`' one maths. A band set is a set of SPANS on a spectral line, one
// member a band, labelled through a lookup (its name). Five rules, one header:
//
//   frequency    a list of named bands in Hz on the Frequency line (a standard bank, a constant-Q design's bands, an octave rule)
//   eigenvalue   a list of named spans in the Eigenvalue line's reference chart (1/unit^2)
//   wavelength   named scales given by wavelength: [shortest, longest) in the length unit, held as eigenvalue spans, lambda = (2 pi / w)^2
//   parts        named scales given by the number of parts of a connected piece of area A: lambda = 4 pi m / A (Weyl's law)
//   levels       the levels of a dyadic grid, each level's Weyl band (the bands tile the line: a GRID)
//
// A single band SELECTS (it is no frequency axis); several bands are a band SET, one selection whose members include the bands. The ranges and their names are the caller's, never
// defaulted. The family is recorded: a set whose bands are adjacent with one ratio hi / lo (an octave rule, a constant-Q ladder) or that tile the line (levels) is a GRID, every other
// listing is IRREGULAR (gaps and overlaps allowed: named bands). Pure functions of their arguments; std::invalid_argument names what is refused.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "nxr/select/partition.h"
#include "nxr/select/scales.h"
#include "nxr/selection/selection.h"

namespace nxr::select {

using namespace nxr::selection;   // the makers build the class side's Selection (selection/selection.h); it is declared there

/** one named band [lo, hi) in the line's unit (an empty name: "lo-hi unit") */
struct BandSpec {
  std::string name;
  double lo = 0, hi = 0;
};

/** rule `frequency`: the bands in `unit` (Hz). std::invalid_argument for a band that is not 0 <= lo < hi or an empty list. */
Selection bands_by_frequency(const std::vector<BandSpec>& bands, const std::string& unit = "Hz");
/** rule `eigenvalue`: the bands in `unit` (1/unit^2 of the surface's length unit) */
Selection bands_by_eigenvalue(const std::vector<BandSpec>& bands, const std::string& unit);
/** rule `wavelength`: the scales given by WAVELENGTH, in `unit` (`to` may be +infinity: the largest scale, from eigenvalue 0); codes 0.. in the order given. std::invalid_argument for
 *  from <= 0, to <= from, or an empty unit. */
Selection bands_by_wavelength(const std::vector<Scale>& scales, const std::string& unit);
/** rule `parts`: the scales given by the NUMBER OF PARTS of a connected piece of area `piece_area` (in `unit`^2); codes 0.. in the order given. std::invalid_argument for from < 0,
 *  to <= from, a non-positive area or an empty unit. */
Selection bands_by_parts(const std::vector<Scale>& scales, double piece_area, const std::string& unit);
/** rule `levels`: the Weyl band of each level of a dyadic grid (or of any set with levels), one member a level, in level order; the bands must tile the line without gap or overlap */
Selection bands_by_levels(const Selection& levels);

/** the octave bank: [f_lo 2^b, f_lo 2^(b+1)) Hz, b = 0 .. n - 1 (nxr's `octave_bands`: 1-2 ... 32-64 Hz for f_lo 1, n 6). std::invalid_argument for f_lo <= 0 or n < 1. */
std::vector<BandSpec> octave_bands(double f_lo = 1.0, int n = 6);
/** a constant-Q design's bands (highest first) as a list for the rule `frequency` */
std::vector<BandSpec> bands_of_design(const ConstantQDesign& design);

/** the bins of a band on the line a transform makes: the bins [first, last) of the one-sided spectrum of `tile_samples` samples at `sfreq` Hz (spacing df = sfreq / tile_samples):
 *  [ceil(lo / df), ceil(hi / df)), the top held inside the spectrum's tile_samples / 2 + 1 bins. std::invalid_argument for a band that holds no bin. (A transform's bins are the made
 *  line of its output domain, a factor of the field's type.) */
std::pair<std::int64_t, std::int64_t> bins_of_band(double lo_hz, double hi_hz, std::int64_t tile_samples, double sfreq);

}  // namespace nxr::select
