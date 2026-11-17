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
}  // namespace

is::Description describe(const cs::Selection& m) {
  is::Description d;
  d.type = to_io(m.type);
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
  if (m.grid) d.grid = is::Grid{m.grid->kind == cs::Grid::Kind::dyadic ? is::Grid::Kind::dyadic : is::Grid::Kind::elements, m.grid->size, m.grid->depth, m.grid->slots};
  d.level = m.level;
  d.band = m.band;
  d.band_unit = m.band_unit;
  for (const auto& e : m.lookup) d.lookup.push_back(is::Entry{e.code, e.name, e.color});
  return d;
}

cs::Selection made(const is::Description& d) {
  cs::Selection m;
  m.type = to_compute(d.type);
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
  if (d.grid) m.grid = cs::Grid{d.grid->kind == is::Grid::Kind::dyadic ? cs::Grid::Kind::dyadic : cs::Grid::Kind::elements, d.grid->size, d.grid->depth, d.grid->slots};
  m.level = d.level;
  m.band = d.band;
  m.band_unit = d.band_unit;
  for (const auto& e : d.lookup) m.lookup.push_back(cs::Entry{e.code, e.name, e.color});
  return m;
}

}  // namespace nxr::pipelines
