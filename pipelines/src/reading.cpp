// nxr-pipelines — the one door to the chunk service (reading.h).
#include "nxr/pipelines/reading.h"

namespace nxr::pipelines {

namespace {
nxr::io::Span span_with(std::int64_t width, std::int64_t n_samples, double sfreq, double t0, const std::optional<std::string>& chunks, const std::optional<double>& start,
                        const std::optional<double>& stop, int level) {
  return nxr::io::resolve_span({chunks, start, stop, level}, n_samples, sfreq, t0, width);
}
nxr::io::ChunkSource source_with(nxr::io::SubjectWriter& sub, const std::string& path, std::int64_t width, double sfreq, double t0, std::vector<std::int32_t> rows,
                                 const std::optional<std::vector<std::pair<double, double>>>& bad) {
  return nxr::io::ChunkSource(sub.store(), path, width, sfreq, t0, std::move(rows), bad);
}
}  // namespace

nxr::io::Span span_of(const nxr::io::Row& field, std::int64_t n_samples, double sfreq, double t0, const std::optional<std::string>& chunks,
                      const std::optional<double>& start, const std::optional<double>& stop, int level) {
  return span_with(nxr::io::chunk_width(field), n_samples, sfreq, t0, chunks, start, stop, level);
}

nxr::io::Span span_of(const task::Node& field, std::int64_t n_samples, double sfreq, double t0, const std::optional<std::string>& chunks,
                      const std::optional<double>& start, const std::optional<double>& stop, int level) {
  return span_with(nxr::io::chunk_width(field.fact("chunk_shape_json")), n_samples, sfreq, t0, chunks, start, stop, level);
}

nxr::io::ChunkSource source_of(nxr::io::SubjectWriter& sub, const nxr::io::Row& field, double sfreq, double t0, std::vector<std::int32_t> rows,
                               const std::optional<std::vector<std::pair<double, double>>>& bad) {
  return source_with(sub, nxr::io::text(field, "path"), nxr::io::chunk_width(field), sfreq, t0, std::move(rows), bad);
}

nxr::io::ChunkSource source_of(nxr::io::SubjectWriter& sub, const task::Node& field, double sfreq, double t0, std::vector<std::int32_t> rows,
                               const std::optional<std::vector<std::pair<double, double>>>& bad) {
  return source_with(sub, field.path.value_or(std::string()), nxr::io::chunk_width(field.fact("chunk_shape_json")), sfreq, t0, std::move(rows), bad);
}

}  // namespace nxr::pipelines
