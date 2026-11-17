// nxr::selection (compute) — NAMED SPATIAL SCALES (the data): the surface's counterpart of named frequency bands. A scale (hemisphere, lobe, region,
// gyrus, …) is a labelled span of Laplacian eigenvalues, made from what the caller states about it, in either of two ways:
//
//   by wavelength   a scale whose patterns have wavelengths w in [shortest, longest): lambda = (2 pi / w)^2
//   by parts        a scale that divides a connected piece of area A into m in [fewest, most) parts: lambda = 4 pi m / A (Weyl's law, as the dyadic levels);
//                   the counts may be read from an atlas (a hemisphere 1; DK's regions on a hemisphere, …)
//
// The ranges and their names are the caller's, never defaulted. The band set made from them is a set of spans on the eigenvalue line, in 1/unit^2, one member a scale, labelled by its name (bands.h).
#pragma once
#include <string>
#include <vector>


namespace nxr::selection {

/** one named scale: [from, to) in the method's quantity (a wavelength in the length unit; a number of parts) */
struct Scale {
  std::string name;
  double from = 0, to = 0;
};

// The scales become a band set by `bands_by_wavelength` and `bands_by_parts` (selection/bands.h), the one place the band sets are made.

}  // namespace nxr::selection
