// The runner's side of a node's needs (include/nxr/pipelines/bound.h): the reads a node used to make itself. Each method does exactly the read the node did.
#include "nxr/pipelines/bound.h"

#include "nxr/io/array_reader.h"
#include <cstring>

#include "nxr/io/errors.h"

namespace nxr::pipelines::graph {

namespace io = nxr::io;

namespace {
// a stored node's array (a kernel, a selection's indices) as T: read by name, never a recording (the chunk service's)
template <class T>
std::vector<T> read_array(const io::Store& store, const std::string& key, std::vector<std::int64_t>* shape = nullptr) {
  io::ArrayReader a = io::ArrayReader::open(store, key);
  if (shape) *shape = a.shape();
  const auto raw = a.read();
  const auto t = a.dtype();
  const std::size_t n = raw.size() / io::dtype_size(t);
  std::vector<T> out(n);
  auto conv = [&](auto tag) {
    using S = decltype(tag);
    for (std::size_t i = 0; i < n; ++i) { S v; std::memcpy(&v, raw.data() + i * sizeof(S), sizeof(S)); out[i] = static_cast<T>(v); }
  };
  switch (t) {
    case io::DType::Float32: conv(float{}); break;
    case io::DType::Float64: conv(double{}); break;
    case io::DType::Int32: conv(std::int32_t{}); break;
    case io::DType::Int64: conv(std::int64_t{}); break;
    default: throw io::Error(io::code::invalid_argument, "an array of type " + std::string(io::dtype_to_string(t)) + " is not read here");
  }
  return out;
}

}  // namespace

Bound::Bound(io::SubjectWriter& sub, io::DatasetWriter* ds, std::function<void(const io::Json&)> say) : subject_(task::Subject::of(sub)), sub_(&sub), ds_(ds), say_(std::move(say)) {}

io::SubjectWriter& Bound::writer() const {
  if (!sub_) throw io::Error(io::code::invalid_argument, "a stored constant is read in a run, not in a plan: the binding holds no subject");
  return *sub_;
}

namespace {
// a found node as the row the node code reads (the columns by name; a JSON column as its text)
io::Row row_of(const task::Node& n) {
  io::Row r;
  for (const auto& [k, v] : n.row.items()) {
    if (v.is_null()) r[k] = std::monostate{};
    else if (v.is_boolean()) r[k] = static_cast<std::int64_t>(v.get<bool>() ? 1 : 0);
    else if (v.is_number_integer()) r[k] = v.get<std::int64_t>();
    else if (v.is_number()) r[k] = v.get<double>();
    else if (v.is_string()) r[k] = v.get<std::string>();
    else r[k] = v.dump();
  }
  return r;
}
std::optional<io::Row> row_or_none(const std::optional<task::Node>& n) { return n ? std::optional<io::Row>(row_of(*n)) : std::nullopt; }
}  // namespace

std::optional<io::Row> Bound::manifold(const std::string& id) const { return row_or_none(subject_.by_id(task::Cls::manifold, id)); }
std::optional<io::Row> Bound::selection(const std::string& id) const { return row_or_none(subject_.by_id(task::Cls::selection, id)); }
std::optional<io::Row> Bound::field(const std::string& id) const { return row_or_none(subject_.by_id(task::Cls::field, id)); }
std::string Bound::manifold_type(const std::string& id) const {
  const auto m = subject_.by_id(task::Cls::manifold, id);
  return m && m->type ? *m->type : std::string();
}
std::optional<std::string> Bound::glued_surface_id(const std::string& manifold_id) const {
  const auto tm = subject_.by_id(task::Cls::manifold, manifold_id);
  if (!tm) return std::nullopt;
  task::Want w{.cls = task::Cls::manifold};
  w.name = tm->name + "_glued";
  w.type = "surface";
  const auto gm = subject_.find(w);
  if (!gm) return std::nullopt;
  return gm->id;
}
std::vector<std::pair<double, double>> Bound::spans(const std::string& selection_id) const {
  std::vector<std::pair<double, double>> out;
  const auto sel = subject_.by_id(task::Cls::selection, selection_id);
  if (sel) for (const auto& s : subject_.spans(*sel)) out.emplace_back(s.start, s.stop);
  return out;
}
std::vector<io::Row> Bound::level_members(const std::string& ladder_id) const {
  std::vector<io::Row> out;
  const auto ladder = subject_.by_id(task::Cls::selection, ladder_id);
  if (ladder) for (const auto& m : subject_.members(*ladder)) out.push_back(row_of(m));
  return out;
}
std::optional<io::Row> Bound::manifold_named(const std::string& name) const {
  task::Want w{.cls = task::Cls::manifold};
  w.name = name;
  return row_or_none(subject_.find(w));
}
SurfaceInput Bound::surface(const std::optional<std::string>& name, bool as_stored) const { return read_surface(writer(), name, as_stored); }
SurfaceLevel Bound::surface_level(const SurfaceInput& s, int level) const { return read_surface_level(writer(), s, level); }
StoredFrames Bound::stored_frames(const std::string& surface_name, std::int64_t n_vertices) const { return read_stored_frames(writer(), surface_name, n_vertices); }
internal::GluedDomain Bound::glued_domain(const SurfaceInput& s) const { return internal::glued_domain(writer(), s); }
StoredCrossSpectrum Bound::cross_spectrum(const std::string& name) const { return crossspectrum_read(writer(), name); }
io::StoredStatistic Bound::statistic(const std::string& name) const { return io::read_statistic(writer(), name); }
std::vector<std::int32_t> Bound::array_int32(const std::string& path) const { return read_array<std::int32_t>(writer().store(), path); }
std::vector<float> Bound::array_float(const std::string& path, std::vector<std::int64_t>* shape) const { return read_array<float>(writer().store(), path, shape); }
std::vector<double> Bound::array_double(const std::string& path, std::vector<std::int64_t>* shape) const { return read_array<double>(writer().store(), path, shape); }
SpatialBandPlan Bound::plan_spatial_band(const SurfaceInput& s, const SurfaceLevel& l, const SpatialBandOptions& o) const { return nxr::pipelines::plan_spatial_band(writer(), s, l, o); }
SpatialBand Bound::make_spatial_band(const SurfaceInput& s, const SurfaceLevel& l, const SpatialBandOptions& o) const {
  if (!ds_) throw io::Error(io::code::invalid_argument, "a spatial band is prepared with the dataset (the runner may have made its eigenmodes first)");
  // eigenmodes first, and never hidden inside this run: a package that is not stored is made by the runner as its own run (its own provenance and cache key), and this run starts again
  if (o.transform == "auto" || o.transform == "eigen") {
    SpatialBandPlan p = nxr::pipelines::plan_spatial_band(writer(), s, l, o);
    if (p.transform == "eigen" && p.source == "computed") throw MissingEigenmodes(s.name, std::move(p));
  }
  return nxr::pipelines::make_spatial_band(*ds_, writer(), s, l, o, say_);
}

}  // namespace nxr::pipelines::graph
