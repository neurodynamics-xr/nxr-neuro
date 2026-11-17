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

Bound::Bound(io::SubjectWriter& sub, io::DatasetWriter* ds, std::function<void(const io::Json&)> say) : db_(&sub.db()), sub_(&sub), ds_(ds), say_(std::move(say)) {}

io::SubjectWriter& Bound::subject() const {
  if (!sub_) throw io::Error(io::code::invalid_argument, "a stored constant is read in a run, not in a plan: the binding holds no subject");
  return *sub_;
}

std::optional<io::Row> Bound::manifold(const std::string& id) const { return db_->read("manifold", id); }
std::optional<io::Row> Bound::selection(const std::string& id) const { return db_->read("selection", id); }
std::optional<io::Row> Bound::field(const std::string& id) const { return db_->read("field", id); }
std::string Bound::manifold_type(const std::string& id) const {
  const auto m = db_->one("SELECT type FROM manifold WHERE id = ?", {io::V(id)});
  return m ? io::text(*m, "type") : std::string();
}
std::optional<std::string> Bound::glued_surface_id(const std::string& manifold_id) const {
  const auto tm = db_->one("SELECT name, subject_id FROM manifold WHERE id = ?", {io::V(manifold_id)});
  if (!tm) return std::nullopt;
  const auto gm = db_->one("SELECT id FROM manifold WHERE subject_id = ? AND name = ? AND type = 'surface'", {io::V(io::text(*tm, "subject_id")), io::V(io::text(*tm, "name") + "_glued")});
  if (!gm) return std::nullopt;
  return io::text(*gm, "id");
}
std::vector<io::Row> Bound::spans(const std::string& selection_id) const {
  return db_->all("SELECT ordinal, start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {io::V(selection_id)});
}
std::vector<io::Row> Bound::level_members(const std::string& ladder_id) const {
  return db_->all("SELECT id, name, member_ordinal, params_json FROM selection WHERE member_of_id = ? AND status = 'complete' ORDER BY member_ordinal", {io::V(ladder_id)});
}
std::optional<io::Row> Bound::manifold_named(const std::string& name) const {
  return subject().db().one("SELECT id, n_vertices FROM manifold WHERE subject_id = ? AND name = ?", {io::V(subject().id()), io::V(name)});
}
SurfaceInput Bound::surface(const std::optional<std::string>& name, bool as_stored) const { return read_surface(subject(), name, as_stored); }
SurfaceLevel Bound::surface_level(const SurfaceInput& s, int level) const { return read_surface_level(subject(), s, level); }
StoredFrames Bound::stored_frames(const std::string& surface_name, std::int64_t n_vertices) const { return read_stored_frames(subject(), surface_name, n_vertices); }
internal::GluedDomain Bound::glued_domain(const SurfaceInput& s) const { return internal::glued_domain(subject(), s); }
StoredCrossSpectrum Bound::cross_spectrum(const std::string& name) const { return crossspectrum_read(subject(), name); }
io::StoredStatistic Bound::statistic(const std::string& name) const { return io::read_statistic(subject(), name); }
std::vector<std::int32_t> Bound::array_int32(const std::string& path) const { return read_array<std::int32_t>(subject().store(), path); }
std::vector<float> Bound::array_float(const std::string& path, std::vector<std::int64_t>* shape) const { return read_array<float>(subject().store(), path, shape); }
SpatialBandPlan Bound::plan_spatial_band(const SurfaceInput& s, const SurfaceLevel& l, const SpatialBandOptions& o) const { return nxr::pipelines::plan_spatial_band(subject(), s, l, o); }
SpatialBand Bound::make_spatial_band(const SurfaceInput& s, const SurfaceLevel& l, const SpatialBandOptions& o) const {
  if (!ds_) throw io::Error(io::code::invalid_argument, "a spatial band is prepared with the dataset (it may store the eigenmodes it computes)");
  return nxr::pipelines::make_spatial_band(*ds_, subject(), s, l, o, say_);
}

}  // namespace nxr::pipelines::graph
