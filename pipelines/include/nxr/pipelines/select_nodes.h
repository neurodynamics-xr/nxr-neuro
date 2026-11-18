// nxr-pipelines — THE FIVE SELECTION MAKERS AS NODES (docs/specs/2026-11-17-consolidation.md, section 1): compute creates, io stores.
//
//   select.dyadic_surface   setup   a surface -> its dyadic grid (rule subject | group | transfer); the finest tile is a QUANTITY (a diameter, an area, a wavelength, parts)
//   select.time_grid        setup   a time line -> one rule family of members [origin + i hop, origin + i hop + width): tiles | band | frames | constant_q | segments | windows | chunks;
//                                   frames per tile or per cycle and the frame ORIGIN (the line's start, a clock time, or an EVENT) are options of the node
//   select.bands            array   a band set (spans on the Frequency or Eigenvalue line, with names): frequency | eigenvalue | wavelength | parts | levels
//   select.product          array   the product of grids (Studio D100; D129 joint layout), levels per axis or COUPLED to a band (the Studio's fitLevelOf, Weyl's law)
//   select.events.*         detection: spans (rules detect | keep_where | stages), regions, cores, tracks: irregular selections that record the grid they were found over
//
// Each maker reads no data: its inputs are manifolds and stored selections, its parameters are quantities (compute/include/nxr/quantity.h), and what it made is handed to the next
// node through its typed edge. The level and the SNAPPED quantity (what the level actually is) are in the output's description. A retired name is refused with the name of
// what replaces it (registry.cpp).
#pragma once
#include <vector>

#include "nxr/selection/selection.h"

#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

/** takes the old makers' steps out of `r` and appends the makers; registers the native makers (select.bands, select.product) in nxr::graph::registry() */
void add_select_nodes(std::vector<StepSpec>& r);
/** registers the native makers; idempotent, called with the other native nodes before the registry of steps is first read */
void register_select_nodes();

/** THE LEVEL a stated quantity resolves to on a dyadic grid (spec 3.3), the one place: a tile diameter, an area, a wavelength (the Weyl level), parts per hemisphere, or `index:N` (diagnostics).
 *  `grid`: the grid with its levels' measured `diameter` and `band` (a grid made in memory, or rebuilt from a stored ladder's level records); `unit` the surface's length unit ("m", "cm", "mm").
 *  `snapped` gets what the level actually is. A bare number is refused. */
int level_of_quantity(const nxr::selection::Selection& grid, const Json& quantity, const std::string& unit, const std::string& who, Json& snapped);

/** A surface grid's LEVEL DICTIONARY as JSON ([{level, diameter?, lambda_min?, lambda_max?}], levels 0 .. depth) and back as the Selection `level_of_quantity` reads. A node that holds a field of the ladder's
 *  scales records the dictionary in its output's facts, so a later node resolves ITS quantity without the ladder (at_level, the scale space's readers). */
Json ladder_dictionary(const nxr::selection::Selection& grid);
nxr::selection::Selection grid_of_dictionary(const Json& dictionary);
/** a stored tile ladder's level dictionary as the grid the resolver reads (the ladder's level records: measured diameter, Weyl band), and the surface's length unit ("m" when the ladder records none) */
nxr::selection::Selection ladder_grid(const Bound& bound, const Value& tiles, std::string* unit = nullptr);
/** THE LEVEL a quantity states on a stored tile ladder (`tiles`: the ladder selection value), the one resolver for every node that names a scale (tile_crossings, patches, cellquadrature, run_repeat, ...) */
int level_of_ladder(const Bound& bound, const Value& tiles, const Json& quantity, const std::string& who, Json* snapped = nullptr);
/** the same for a node that has the SURFACE and no ladder input (the kernel's target): the surface's own tile ladder (`<surface> tiles`) is found and read */
int level_of_surface(const Bound& bound, const SurfaceInput& surface, const Json& quantity, const std::string& who, Json* snapped = nullptr);
/** the level a quantity (a scale) states among the levels a field already holds: resolved on the dictionary, then required to be one of `held` (refused naming them with their snapped quantities) */
int level_among(const Json& dictionary, const std::string& unit, const Json& quantity, const std::vector<int>& held, const std::string& who, Json& snapped);

}  // namespace nxr::pipelines::graph
