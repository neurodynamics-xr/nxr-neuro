// nxr::selection (compute) — LABELLED REGIONS: sets of a manifold's elements (a surface's vertices, a montage's channels), each with a code read through a
// lookup. Unlike a partition they may leave elements out and overlap: a disc, an atlas region, one tile, a spatial burst's region, a region drawn by hand.
// The counterpart of labelled spans on a line. Made in memory here; io stores them as indices selections with their dictionary.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/selection/lookup.h"
#include "nxr/selection/partition.h"
#include "nxr/selection/spans.h"

namespace nxr::chart { class ChartKit; }

namespace nxr::selection {

/** one region: its elements (sorted, each once) and its code */
struct Region {
  std::int64_t code = 0;
  std::vector<std::int32_t> elements;
};

struct Regions {
  std::string on;                  // the manifold: surface · channels · …
  std::int64_t elements = 0;       // how many elements the manifold has
  std::vector<Region> regions;
  Lookup lookup;                   // the codes' names (every code a region carries has an entry, unless the lookup is empty)
};

/** labelled regions of the manifold `on` of `elements` elements: each region's elements sorted and made unique; std::invalid_argument for an element outside
 *  0..elements-1, an empty region, or a code the lookup lacks */
Regions regions(std::string on, std::int64_t elements, std::vector<Region> list, Lookup lookup);

/** the indices (into `r.regions`) of the regions holding `element`: none, one, or several where they overlap */
std::vector<std::size_t> regions_at(const Regions& r, std::int32_t element);

// ── makers of one region or span ────────────────────────────────────────────────────────
/** a DISC: the vertices within `radius` (the surface's unit) of `centre` by heat-method geodesic distance (geometry-central, through the chart kit) */
Region disc(chart::ChartKit& kit, std::int32_t centre, double radius, std::int64_t code);

/** ONE MEMBER of a partition by codes as a region (an atlas region, a tile of a level): the elements whose member at `level` is `member` */
Region member_region(const Partition& p, int level, std::int64_t member, std::int64_t code);

/** ONE MEMBER of a partition of a line as a span in the line's unit (a time tile, a band of a spectrum's partition) */
Span member_span(const Partition& p, int level, std::int64_t member, std::int64_t code);

}  // namespace nxr::selection
