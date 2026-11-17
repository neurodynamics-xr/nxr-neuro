// nxr-io — EXPORT BIDS-DERIVATIVES (docs/specs/2026-11-17-interoperability.md, 5.4; question 9: `desc-<what>` and documented suffixes now): the exports of 5.1 to 5.3, named with BIDS
// entities under a derivatives folder, each with a BIDS sidecar, and a `dataset_description.json` that lists nxr-neuro's suffixes.
//
//   <root>/dataset_description.json                 DatasetType derivative, GeneratedBy, SourceDatasets, License, and `NxrNeuro.Suffixes` (below)
//   <root>/sub-<S>/[ses-<E>/]<sub-S>[_ses-E][_task-T][_run-R]_space-<SP>[_hemi-<L|R>]_desc-<name>_<suffix>.<ext>   and  ....json
//
//   suffix  map     a scalar map on a surface            .func.gii (.shape.gii is not used)   one file per hemisphere when the cortex stores them
//           series  a scalar series on surface x time    .func.gii (the time range stated by the caller)
//           dseg    a labelling (parcellation)           .label.gii
//           stc     a source map or series               <stem>_stc-lh.stc and -rh.stc (MNE), one sidecar <stem>_stc.json
//           surf    the surface itself (--with-surface)  .surf.gii beside each map or series
//           stat    a statistic, long form               .tsv + .json
//           measures  a partition's measurements         .tsv + .json
// `<name>` is the stored node's name reduced to letters and digits; two names that reduce to the same label are refused. The sidecar of each file is the exporter's provenance record
// (GeneratedBy) with `Sources` as BIDS URIs (`bids::<path>`) when the input was imported from a BIDS dataset. A re-run into the same folder merges `dataset_description.json`.
// participants.tsv is never written here. Nothing is defaulted: the nodes (a regular expression over their names), the space and the time range of a series are the caller's.
#pragma once
#include <filesystem>
#include <optional>
#include <string>

#include "nxr/exporters/common.h"

namespace nxr::exporters {

struct BidsDerivativesExport {
  std::string root;                        // the derivatives folder (made if absent)
  std::string results;                     // a regular expression matched against the WHOLE name of each field, labelling and statistic to export
  std::optional<std::string> space;        // the `space-` entity; unset: the name of the template the subject is registered to (refused when it is neither)
  std::optional<std::string> session, task, run;   // the `ses-`, `task-` and `run-` entities, as the caller names them (none by default)
  std::optional<SeriesRange> range;        // the time range of every series matched (REQUIRED when one is)
  bool with_surface = false;               // a .surf.gii beside each map and series
  bool stc = false;                        // also each map and series as MNE source estimates (the cortex must store both hemispheres)
  std::string measurements;                // "PARTITION:FIELD:measure,measure": also that table (empty: none)
  Context ctx;
};

// Writes the files; returns {root, files: [...], sidecars: [...], description, skipped: [{name, reason}]}.
Json export_bids_derivatives(const std::filesystem::path& package, const BidsDerivativesExport& options);

}  // namespace nxr::exporters
