// nxr-io — the selection (selection.h): the Studio's entities/selection.ts.
#include "nxr/io/selection.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "nxr/io/array_reader.h"
#include "nxr/io/database.h"
#include "nxr/io/errors.h"
#include "nxr/io/records/records.h"
#include "nxr/io/types.h"

namespace nxr::io::selection {

namespace {
[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }
}  // namespace

Selection Selection::read(const Database& db, const Store& store, std::string_view id) {
  const auto r = records::Selections(db).read(id);
  if (!r) throw Error(code::no_such_row, "selection " + std::string(id) + ": no such row");
  return Selection(db, store, *r);
}

std::vector<records::SelectionRow> Selection::members() const { return records::Selections(*db_).members(row_.id); }
std::vector<records::SelectionExtentRow> Selection::extents() const { return records::Selections(*db_).extents(row_.id); }
std::vector<records::SelectionMeasurementRow> Selection::measurements(bool own) const { return records::Selections(*db_).measurements(row_.id, own); }

std::optional<records::DictionaryEntryRow> Selection::entry() const {
  if (!row_.code) return std::nullopt;
  std::optional<std::string> dict;
  if (row_.member_of_id)
    if (const auto set = records::Selections(*db_).read(*row_.member_of_id); set && set->dictionary_id) dict = set->dictionary_id;
  if (!dict) dict = row_.dictionary_id;
  if (!dict) return std::nullopt;
  const auto r = db_->one("SELECT * FROM dictionary_entry WHERE dictionary_id = ? AND code = ?", {V(*dict), V(*row_.code)});
  if (!r) return std::nullopt;
  return records::DictionaryEntryRow::from_row(*r);
}

std::optional<double> Selection::measurement(const std::string& measure, const std::optional<std::string>& of_field_id) const {
  for (const auto& m : measurements(true))
    if (m.measure == measure && m.of_field_id == of_field_id && m.value) return *m.value;
  return std::nullopt;
}

std::optional<Tiling> Selection::tiling() const {
  const Json& p = row_.params_json;
  if (!p.is_object() || !p.contains("tiling") || !p["tiling"].is_object()) return std::nullopt;
  const Json& t = p["tiling"];
  return Tiling{t.value("origin", 0.0), t.value("width", 0.0), t.value("hop", 0.0), t.value("count", std::int64_t(0))};
}

double Selection::end() const {
  const Json& p = row_.params_json;
  if (p.is_object() && p.contains("end") && p["end"].is_number()) return p["end"].get<double>();
  if (row_.manifold_id)
    for (const auto& d : records::Manifolds(*db_).factors(*row_.manifold_id))
      if (d.spacing) return d.origin.value_or(0.0) + static_cast<double>(d.n_vertices) * *d.spacing;
  return INFINITY;
}

std::int64_t Selection::count() const {
  if (const auto t = tiling()) return t->count;
  if (row_.type == "spans") return static_cast<std::int64_t>(records::Selections(*db_).spans(row_.id).size());
  return static_cast<std::int64_t>(members().size());
}

Span Selection::at(std::int64_t k) const {
  // both forms of a tessellation answer this: the closed form by arithmetic (an O(1) multiply, D80), a set of rows by lookup
  if (row_.type != "spans") refuse("selection " + row_.name + ": not a tessellation (" + row_.type + "): only a spans set has members by ordinal");
  if (const auto t = tiling()) {
    if (k < 0 || k >= t->count) refuse("selection " + row_.name + ": no member " + std::to_string(k) + " (it has " + std::to_string(t->count) + ")");
    const double start = t->origin + static_cast<double>(k) * t->hop;
    return Span{k, start, std::min(start + t->width, end()), k};
  }
  const auto r = db_->one("SELECT * FROM selection_span WHERE selection_id = ? AND ordinal = ?", {V(row_.id), V(k)});
  if (!r) refuse("selection " + row_.name + ": no member " + std::to_string(k) + " (it has " + std::to_string(count()) + ")");
  const auto s = records::SelectionSpanRow::from_row(*r);
  return Span{s.ordinal, s.start, s.stop, s.code};
}

std::optional<std::int64_t> Selection::index_at(double value) const {
  if (const auto t = tiling()) {
    if (!(t->hop > 0) || value < t->origin || value >= end()) return std::nullopt;
    // the member whose [start, stop) holds the value: the last start at or before it, when it reaches past it
    const std::int64_t k = std::min<std::int64_t>(t->count - 1, static_cast<std::int64_t>(std::floor((value - t->origin) / t->hop)));
    if (k < 0) return std::nullopt;
    const double start = t->origin + static_cast<double>(k) * t->hop;
    return value < std::min(start + t->width, end()) ? std::optional<std::int64_t>(k) : std::nullopt;
  }
  for (const auto& p : records::Selections(*db_).spans(row_.id)) if (value >= p.start && value < p.stop) return p.ordinal;
  return std::nullopt;
}

Resolution Selection::resolve(const std::optional<std::string>& on_id) const {
  const std::string on = on_id.value_or(row_.manifold_id.value_or(std::string()));
  if (on.empty()) refuse("selection " + row_.name + ": a position on a space resolves on a line (vertex_on), not by itself");
  const auto rec = records::Manifolds(*db_).read(on);
  if (!rec) throw Error(code::no_such_row, "selection " + row_.name + ": manifold " + on + " has no row");
  const bool product = rec->row.type == "product";
  const std::vector<records::GeometryDimensionRow> dims = rec->dimensions;   // a product's factor lines, a lattice's axes, a line's one axis
  Resolution r;
  if (row_.type == "span") {
    // A BOX (D97): one extent per constrained axis; on a PRODUCT an axis is a FACTOR, by ordinal. An axis with a closed form rounds its extent onto samples;
    // one without (points: the sensors) takes its extent as ELEMENT INDICES.
    for (const auto& d : dims) r.slice.emplace_back(0, d.n_vertices);
    if (r.slice.empty()) r.slice.emplace_back(0, rec->row.n_vertices);
    for (const auto& e : extents()) {
      if (e.axis < 0 || e.axis >= static_cast<std::int64_t>(r.slice.size())) refuse("selection " + row_.name + ": '" + rec->row.name + "' has no axis " + std::to_string(e.axis));
      const std::int64_t n = r.slice[static_cast<std::size_t>(e.axis)].second;
      std::int64_t lo, hi;
      const records::GeometryDimensionRow* d = static_cast<std::size_t>(e.axis) < dims.size() ? &dims[static_cast<std::size_t>(e.axis)] : nullptr;
      if (d && d->spacing) {
        // a CALLER-STATED extent (rule 7, as compute's resolve): the sample nearest its centre is always held, the extent rounds OUTWARD (start down, stop up), never empty
        constexpr double kTolerance = 1e-9;                                // a corner within this many samples of a sample is on it
        const double pa = (e.start - d->origin.value_or(0.0)) / *d->spacing, pb = (e.stop - d->origin.value_or(0.0)) / *d->spacing;
        const std::int64_t c = std::llround((pa + pb) / 2);
        lo = std::min(static_cast<std::int64_t>(std::floor(pa + kTolerance)), c);
        hi = std::max(static_cast<std::int64_t>(std::ceil(pb - kTolerance)), c + 1);
        lo = std::max<std::int64_t>(0, lo);
        hi = std::min<std::int64_t>(d->n_vertices, hi);
      } else {
        if (e.start != std::floor(e.start) || e.stop != std::floor(e.stop))
          refuse("selection " + row_.name + ": axis " + std::to_string(e.axis) + " of '" + rec->row.name + "' has no closed form, so its extent is element indices (whole numbers)");
        lo = std::max<std::int64_t>(0, static_cast<std::int64_t>(e.start));
        hi = std::min<std::int64_t>(n, static_cast<std::int64_t>(e.stop));
      }
      r.slice[static_cast<std::size_t>(e.axis)] = {lo, std::max(lo, hi)};
    }
    (void)product;
    return r;
  }
  if (row_.type == "point") {
    if (!row_.center) refuse("selection " + row_.name + ": a point on a space resolves on a line (vertex_on)");
    r.slice.emplace_back(*row_.center, *row_.center + 1);
    for (std::size_t a = 1; a < dims.size(); ++a) r.slice.emplace_back(0, dims[a].n_vertices);
    return r;
  }
  if (row_.type == "indices" || row_.type == "set") {
    if (!row_.array_path) refuse("selection " + row_.name + ": a " + row_.type + " with no array: resolve its members");
    r.kind = Resolution::Kind::indices;
    r.indices = elements();
    return r;
  }
  if (row_.type == "kernel") {
    r.kind = Resolution::Kind::weights;
    r.weight_fn = row_.weight_fn;
    r.params = row_.params_json;
    return r;
  }
  if (row_.type == "spans") refuse("selection " + row_.name + ": a spans set resolves member by member (at)");
  refuse("selection " + row_.name + ": " + row_.type + " cannot be resolved yet");
}

std::optional<std::int64_t> Selection::vertex_on(const std::string& line) const {
  if (!row_.position) return std::nullopt;
  const auto rec = records::Manifolds(*db_).read(line);
  if (!rec || rec->dimensions.empty() || !rec->dimensions[0].spacing) refuse("selection " + row_.name + ": " + line + " is not a closed-form line");
  if (rec->geometry && rec->geometry->coordinate_system_id && row_.coordinate_system_id && *rec->geometry->coordinate_system_id != *row_.coordinate_system_id)
    refuse("selection " + row_.name + ": the line is written in another coordinate system (its transition is not applied)");
  const auto& d = rec->dimensions[0];
  const double k = std::round((*row_.position - d.origin.value_or(0.0)) / *d.spacing);
  if (k < 0 || k >= static_cast<double>(d.n_vertices)) return std::nullopt;
  return static_cast<std::int64_t>(k);
}

std::vector<std::int32_t> Selection::elements() const {
  if (!row_.array_path) refuse("selection " + row_.name + ": no array is stored for its elements");
  ArrayReader ar = ArrayReader::open(*store_, *row_.array_path);
  const auto raw = ar.read();
  const DType t = ar.dtype();
  const std::size_t n = raw.size() / dtype_size(t);
  std::vector<std::int32_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (t == DType::Int32) { std::int32_t v; std::memcpy(&v, raw.data() + 4 * i, 4); out[i] = v; }
    else if (t == DType::Int64) { std::int64_t v; std::memcpy(&v, raw.data() + 8 * i, 8); out[i] = static_cast<std::int32_t>(v); }
    else if (t == DType::Float32) { float v; std::memcpy(&v, raw.data() + 4 * i, 4); out[i] = static_cast<std::int32_t>(v); }
    else if (t == DType::Float64) { double v; std::memcpy(&v, raw.data() + 8 * i, 8); out[i] = static_cast<std::int32_t>(v); }
    else refuse("selection " + row_.name + ": its array's type is not an integer or float");
  }
  return out;
}

std::map<std::int32_t, std::int64_t> Selection::partition() const {
  std::map<std::int32_t, std::int64_t> out;
  for (const auto c : elements()) ++out[c];
  return out;
}

}  // namespace nxr::io::selection
