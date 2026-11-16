// nxr-pipelines — the one door to the chunk service (reading.h).
#include "nxr/pipelines/reading.h"

namespace nxr::pipelines {

nxr::io::Span span_of(const nxr::io::Row& field, std::int64_t n_samples, double sfreq, double t0, const std::optional<std::string>& chunks,
                      const std::optional<double>& start, const std::optional<double>& stop, int level) {
  return nxr::io::resolve_span({chunks, start, stop, level}, n_samples, sfreq, t0, nxr::io::chunk_width(field));
}

nxr::io::ChunkSource source_of(nxr::io::SubjectWriter& sub, const nxr::io::Row& field, double sfreq, double t0, std::vector<std::int32_t> rows,
                               const std::optional<std::vector<std::pair<double, double>>>& bad) {
  return nxr::io::ChunkSource(sub.store(), nxr::io::text(field, "path"), nxr::io::chunk_width(field), sfreq, t0, std::move(rows), bad);
}

}  // namespace nxr::pipelines
