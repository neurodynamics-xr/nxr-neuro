// nxr-io — EXPORT TSV (docs/specs/2026-11-17-interoperability.md, 5.2): stored measurements and statistics as BIDS-style tables with a JSON sidecar, for R, Python, SPSS and the
// BIDS tools.
//
//   measurements  one row per member of a labelling (a partition), `member` (its dictionary name) and `code`, then one column per measure named by the caller, of one Field.
//                 The values are what is stored, rolled up as io's measurement query rolls them; a measure that is DERIVED at read (variance, sd, a travelling index) is
//                 written as a derived column and named as derived in the sidecar. Scalar measurement rows only (an array-backed pyramid is read through `query measurements`).
//   statistic     long form, one row per value: a column per axis (a frequency axis as its coordinate in the axis's unit, a bin axis as `<axis>_low` and `<axis>_high`, a
//                 selection axis as the member's name, any other axis as the index), `component` when the statistic has components, and `value`.
// `<out>.tsv` and the BIDS sidecar `<out>.json`: every column's `Description`, `Units` (when the store records one) and `Levels` (the member names), the partition and its
// dictionary (`Partition`), the statistic's record (`Statistic`: measure, method, scaling, span, n_members), and the provenance record under `GeneratedBy` (nxr/exporters/common.h:
// the producer, command, options, inputs with the digests of their arrays, terms, output hash). Values are written as the shortest decimal that reads back to the same double; NaN is `n/a`.
#pragma once
#include <filesystem>
#include <string>
#include <vector>

#include "nxr/exporters/common.h"

namespace nxr::exporters {

struct TsvExport {
  std::string partition, field;            // measurements: the labelling and the Field (by name), both required
  std::vector<std::string> measures;       //   the measures to write, by name (at least one)
  std::string statistic;                   // or a statistic, by name (not with the measurement options)
  std::string out;                         // the path stem: <out>.tsv and <out>.json
  Context ctx;
};

// Writes the table and its sidecar; returns {files: [...], sidecar, rows, columns}.
Json export_tsv(const std::filesystem::path& package, const TsvExport& options);

}  // namespace nxr::exporters
