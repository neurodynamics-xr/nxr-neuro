// nxr-pipelines — THE ONE DOOR to the chunk service for every pipeline (the paramount architecture: loading is io's chunk service; a pipeline only declares
// what it reads). A time-resolved Field (a recording, a source series: [rows x samples] on a product whose second factor is its time line) is read by
// naming its row, its time line's rate and origin, the rows wanted, the bad spans, and the span of the run (one chunk by default). The layering guard
// (cmake/check_layering.cmake) allows the chunk service to be built only here.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io.h"
#include "nxr/pipelines/task.h"

namespace nxr::pipelines {

/** the span a run reads: chunks "k" | "a-b" | "all", or [start, stop) seconds; neither is an error (the caller states the span). `level`: the tile level the run measures on (its window or
 *  segment, 2^level samples): the span's ends snap to the nearest tile boundary; 0 for a pointwise run */
nxr::io::Span span_of(const nxr::io::Row& field, std::int64_t n_samples, double sfreq, double t0, const std::optional<std::string>& chunks,
                      const std::optional<double>& start, const std::optional<double>& stop, int level = 0);
nxr::io::Span span_of(const task::Node& field, std::int64_t n_samples, double sfreq, double t0, const std::optional<std::string>& chunks,
                      const std::optional<double>& start, const std::optional<double>& stop, int level = 0);
/** the span of the samples [start, stop) of a source's field, as a read of a stored field by its samples asks (not a run's range, which is chunks or seconds): an Error (invalid_argument)
 *  unless 0 <= start <= stop <= its samples */
nxr::io::Span sample_span(const nxr::io::ChunkSource& src, std::int64_t start, std::int64_t stop);
/** the Field's chunks: the rows asked for (empty: all), the bad spans' mask when given; the recording's active projectors applied to every chunk unless `projectors` is
 *  none (the chunk service's rule; what was applied is the source's `projection_record()`, for the run's provenance and cache key) */
nxr::io::ChunkSource source_of(nxr::io::SubjectWriter& sub, const nxr::io::Row& field, double sfreq, double t0, std::vector<std::int32_t> rows = {},
                               const std::optional<std::vector<std::pair<double, double>>>& bad = std::nullopt, nxr::io::Projectors projectors = nxr::io::Projectors::active);
nxr::io::ChunkSource source_of(nxr::io::SubjectWriter& sub, const task::Node& field, double sfreq, double t0, std::vector<std::int32_t> rows = {},
                               const std::optional<std::vector<std::pair<double, double>>>& bad = std::nullopt, nxr::io::Projectors projectors = nxr::io::Projectors::active);

/** the Field's SHAPE only (spec docs/specs/2026-11-17-surrogates.md, section 6): its channels, rate, span and chunk grid, and the bad spans' mask when given; no sample is ever read (a surrogate's draw,
 *  declared `read: shape`). The chunks it hands carry no data. */
nxr::io::ChunkSource shape_source_of(nxr::io::SubjectWriter& sub, const nxr::io::Row& field, double sfreq, double t0,
                                     const std::optional<std::vector<std::pair<double, double>>>& bad = std::nullopt);

}  // namespace nxr::pipelines
