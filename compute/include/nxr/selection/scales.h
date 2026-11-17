// nxr::selection (compute) — NAMED SPATIAL SCALES ON THE EIGENVALUE LINE: the surface's counterpart of named frequency bands. A scale (hemisphere, lobe, region,
// gyrus, …) is a labelled span of Laplacian eigenvalues, made from what the caller states about it, in either of two ways:
//
//   by wavelength   a scale whose patterns have wavelengths w in [shortest, longest): lambda = (2 pi / w)^2
//   by parts        a scale that divides a connected piece of area A into m in [fewest, most) parts: lambda = 4 pi m / A (Weyl's law, as the dyadic levels);
//                   the counts may be read from an atlas (a hemisphere 1; DK's regions on a hemisphere, …)
//
// The ranges and their names are the caller's, never defaulted. The result is a set of spans on the eigenvalue line, in 1/unit^2, one member a scale, labelled by its name.
#pragma once
#include <string>
#include <vector>

#include "nxr/selection/selection.h"

namespace nxr::selection {

/** one named scale: [from, to) in the method's quantity (a wavelength in the length unit; a number of parts) */
struct Scale {
  std::string name;
  double from = 0, to = 0;
};

/** the scales given by WAVELENGTH, in `unit` (to may be +infinity: the largest scale, from eigenvalue 0); codes 0.. in the order given.
 *  std::invalid_argument for from <= 0, to <= from, or an empty unit. */
Selection scales_by_wavelength(const std::vector<Scale>& scales, const std::string& unit);

/** the scales given by the NUMBER OF PARTS of a connected piece of area `piece_area` (in `unit`^2); codes 0.. in the order given.
 *  std::invalid_argument for from < 0, to <= from, a non-positive area or an empty unit. */
Selection scales_by_parts(const std::vector<Scale>& scales, double piece_area, const std::string& unit);

}  // namespace nxr::selection
