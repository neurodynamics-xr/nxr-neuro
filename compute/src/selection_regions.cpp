// nxr::selection (compute) — labelled regions and the makers of one region or span (regions.h).
#include "nxr/selection/regions.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "nxr/chart.h"

namespace nxr::selection {

Regions regions(std::string on, std::int64_t elements, std::vector<Region> list, Lookup lookup) {
  check(lookup);
  if (elements < 0) throw std::invalid_argument("regions: a manifold of " + std::to_string(elements) + " elements");
  for (Region& r : list) {
    std::sort(r.elements.begin(), r.elements.end());
    r.elements.erase(std::unique(r.elements.begin(), r.elements.end()), r.elements.end());
    if (r.elements.empty()) throw std::invalid_argument("regions: region " + std::to_string(r.code) + " holds no element");
    if (r.elements.front() < 0 || r.elements.back() >= elements)
      throw std::invalid_argument("regions: region " + std::to_string(r.code) + " has an element outside 0.." + std::to_string(elements - 1));
    if (!lookup.empty()) entry(lookup, r.code);                       // throws for a code the lookup lacks
  }
  return Regions{std::move(on), elements, std::move(list), std::move(lookup)};
}

std::vector<std::size_t> regions_at(const Regions& r, std::int32_t element) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < r.regions.size(); ++i)
    if (std::binary_search(r.regions[i].elements.begin(), r.regions[i].elements.end(), element)) out.push_back(i);
  return out;
}

Region disc(chart::ChartKit& kit, std::int32_t centre, double radius, std::int64_t code) {
  if (centre < 0 || centre >= kit.n()) throw std::invalid_argument("disc: centre " + std::to_string(centre) + " outside 0.." + std::to_string(kit.n() - 1));
  if (!(radius >= 0) || !std::isfinite(radius)) throw std::invalid_argument("disc: the radius is >= 0");
  Region r;
  r.code = code;
  for (int v : kit.disc(centre, radius)) r.elements.push_back(static_cast<std::int32_t>(v));
  std::sort(r.elements.begin(), r.elements.end());
  return r;
}

Region member_region(const Partition& p, int level, std::int64_t m, std::int64_t code) {
  if (p.codes.empty()) throw std::invalid_argument("member_region: the " + p.method + " partition has no codes (a line's member is a span: member_span)");
  Region r;
  r.code = code;
  for (std::int64_t v = 0; v < p.elements; ++v)
    if (member(p, level, v) == m) r.elements.push_back(static_cast<std::int32_t>(v));
  if (r.elements.empty()) throw std::invalid_argument("member_region: member " + std::to_string(m) + " of level " + std::to_string(level) + " holds no element");
  return r;
}

Span member_span(const Partition& p, int level, std::int64_t m, std::int64_t code) {
  for (const Level& l : p.levels)
    if (l.level == level && !l.edges.empty()) {
      if (m < 0 || m + 1 >= static_cast<std::int64_t>(l.edges.size())) throw std::invalid_argument("member_span: member " + std::to_string(m) + " outside the level");
      return Span{l.edges[static_cast<std::size_t>(m)], l.edges[static_cast<std::size_t>(m + 1)], code};
    }
  if (!p.codes.empty() || !(p.spacing > 0)) throw std::invalid_argument("member_span: the " + p.method + " partition is not a partition of a line");
  if (level < 0 || level > 62) throw std::invalid_argument("member_span: level " + std::to_string(level) + " outside 0..62");
  const std::int64_t first = m << level, last = std::min(p.elements, (m + 1) << level);
  if (m < 0 || first >= p.elements) throw std::invalid_argument("member_span: member " + std::to_string(m) + " outside the line");
  return Span{p.origin + static_cast<double>(first) * p.spacing, p.origin + static_cast<double>(last) * p.spacing, code};
}

}  // namespace nxr::selection
