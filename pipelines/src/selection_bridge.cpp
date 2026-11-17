// nxr-pipelines — the bridge between compute's selection and io's description (selection_bridge.h): a field-by-field copy, nothing decided.
#include "nxr/pipelines/selection_bridge.h"

namespace nxr::pipelines {

namespace cs = nxr::selection;
namespace is = nxr::io::selection;

namespace {
is::Type to_io(cs::Type t) {
  switch (t) {
    case cs::Type::point: return is::Type::point;
    case cs::Type::span: return is::Type::span;
    case cs::Type::distance: return is::Type::distance;
    case cs::Type::indices: return is::Type::indices;
    case cs::Type::set: return is::Type::set;
  }
  return is::Type::set;
}
cs::Type to_compute(is::Type t) {
  switch (t) {
    case is::Type::point: return cs::Type::point;
    case is::Type::span: return cs::Type::span;
    case is::Type::distance: return cs::Type::distance;
    case is::Type::indices: return cs::Type::indices;
    case is::Type::set: return cs::Type::set;
  }
  return cs::Type::set;
}
// a grid's kind, one name each side: elements, dyadic, and a line's time grid (the closed form io stores)
is::Grid::Kind grid_kind_to_io(cs::Grid::Kind k) {
  switch (k) {
    case cs::Grid::Kind::elements: return is::Grid::Kind::elements;
    case cs::Grid::Kind::dyadic: return is::Grid::Kind::dyadic;
    case cs::Grid::Kind::time: return is::Grid::Kind::time;
  }
  return is::Grid::Kind::elements;
}
cs::Grid::Kind grid_kind_to_compute(is::Grid::Kind k) {
  switch (k) {
    case is::Grid::Kind::elements: return cs::Grid::Kind::elements;
    case is::Grid::Kind::dyadic: return cs::Grid::Kind::dyadic;
    case is::Grid::Kind::time: return cs::Grid::Kind::time;
  }
  return cs::Grid::Kind::elements;
}
}  // namespace

is::Description describe(const cs::Selection& m) {
  is::Description d;
  d.type = to_io(m.type);
  if (m.type == cs::Type::set) d.family = m.family == cs::Family::grid ? is::Family::grid : is::Family::irregular;
  d.code = m.code;
  d.unit = m.unit;
  d.center = m.center;
  d.radius = m.radius;
  for (const auto& e : m.extents) d.extents.push_back(is::Extent{e.axis, e.start, e.stop});
  d.indices = m.indices;
  for (const auto& x : m.members) d.members.push_back(describe(x));
  d.member_of = m.member_of;
  d.member_codes = m.member_codes;
  if (m.tiling) d.tiling = is::Tiling{m.tiling->origin, m.tiling->width, m.tiling->hop, m.tiling->count};
  for (const auto& x : m.factors) d.factors.push_back(describe(x));
  d.radices = m.radices;
  if (m.grid) d.grid = is::Grid{grid_kind_to_io(m.grid->kind), m.grid->size, m.grid->depth, m.grid->slots, m.grid->finest, m.grid->match_median, m.grid->match_max, m.grid->origin, m.grid->rate};
  d.level = m.level;
  d.diameter = m.diameter;
  d.band = m.band;
  d.band_unit = m.band_unit;
  for (const auto& e : m.lookup) d.lookup.push_back(is::Entry{e.code, e.name, e.color});
  return d;
}

cs::Selection made(const is::Description& d) {
  cs::Selection m;
  m.type = to_compute(d.type);
  m.family = d.family == is::Family::grid ? cs::Family::grid : cs::Family::irregular;   // a load always says (unmarked only from a caller that did not)
  m.code = d.code;
  m.unit = d.unit;
  m.center = d.center;
  m.radius = d.radius;
  for (const auto& e : d.extents) m.extents.push_back(cs::Extent{e.axis, e.start, e.stop});
  m.indices = d.indices;
  for (const auto& x : d.members) m.members.push_back(made(x));
  m.member_of = d.member_of;
  m.member_codes = d.member_codes;
  if (d.tiling) m.tiling = cs::Tiling{d.tiling->origin, d.tiling->width, d.tiling->hop, d.tiling->count};
  for (const auto& x : d.factors) m.factors.push_back(made(x));
  m.radices = d.radices;
  if (d.grid) m.grid = cs::Grid{grid_kind_to_compute(d.grid->kind), d.grid->size, d.grid->depth, d.grid->slots, d.grid->finest, d.grid->match_median, d.grid->match_max, d.grid->origin, d.grid->rate};
  m.level = d.level;
  m.diameter = d.diameter;
  m.band = d.band;
  m.band_unit = d.band_unit;
  for (const auto& e : d.lookup) m.lookup.push_back(cs::Entry{e.code, e.name, e.color});
  return m;
}

}  // namespace nxr::pipelines
