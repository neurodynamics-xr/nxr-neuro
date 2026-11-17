// nxr-io — EXPORT GIFTI (docs/specs/2026-11-17-interoperability.md, 5.1): a cortical field, or a labelling, on a surface, written as GIFTI files for Workbench, freeview and nibabel.
//
//   a Field on a surface (a map)                  -> <stem>.func.gii (or .shape.gii): one data array [n_vertices]
//   a Field on surface x time line (a series)     -> <stem>.func.gii: one data array per frame of the stated time range, `nxr-neuro:time` (s) on each
//   a labelling (a partition or set Selection)    -> <stem>.label.gii: an int32 array and the label table (the dictionary's names and colours)
//   the surface itself, on request                -> <stem>.surf.gii (the pointset and the triangles)
//   hemispheres, on request                       -> one file of each kind per hemisphere, <stem>_hemi-L / _hemi-R, `AnatomicalStructurePrimary` CortexLeft / CortexRight
// and `<first file>.json`, the provenance sidecar (nxr/exporters/common.h). Each file's metadata carries `nxr-neuro:version`, `nxr-neuro:command`, `nxr-neuro:producer` (the field's
// own producer record, JSON), `nxr-neuro:origin` (the terms carried from the inputs' origin records, JSON) and, per data array, `Name`, `nxr-neuro:field` and `nxr-neuro:unit`.
//
// An export writes what is stored, in the units stored, or refuses: no maths. Two limits are GIFTI's and are recorded in the sidecar, never silent: GIFTI holds float32, int32 and
// uint8 only (float64 values are written as float32; `details.converted`), and a surface's coordinates are written in the unit they are stored in unless `surface_unit` asks for
// metres or millimetres (a scale by a power of ten). A labelling's code -1 (no label) is written as key 0 when the dictionary does not use 0, else refused.
//
// A series is READ chunk by chunk through io's chunk service over the range the caller states (`range`: required for a series, refused for a map; never defaulted); a GIFTI file is one
// XML document, so the frames of the range are held (as float32) to be written: the range bounds the memory.
#pragma once
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include "nxr/exporters/common.h"

namespace nxr::exporters {

struct GiftiExport {
  std::string field;                       // a Field on a surface (or on a surface x time line), by name; or
  std::string labelling;                   // a partition / set Selection on a surface, by name (exactly one of the two)
  std::string out;                         // the path stem the files are written under (the folder must exist)
  std::string kind = "func";               // a Field: "func" (.func.gii) or "shape" (.shape.gii, static data such as curvature; a map only)
  bool surface = false;                    // also write the surface (.surf.gii)
  bool hemispheres = false;                // one file of each kind per hemisphere (the surface must have its hemispheres as components)
  std::optional<std::string> surface_unit; // "m" or "mm": the surface's coordinates scaled to it; unset: as stored
  std::optional<SeriesRange> range;        // a series: the time range read (REQUIRED); a map: must be unset
  // names the files of one hemisphere ("L" or "R") when set: the path stem to write under (the BIDS-derivatives exporter puts `hemi-` among the entities); default `<out>_hemi-<L|R>`
  std::function<std::string(const std::string& hemisphere)> hemisphere_stem;
  Context ctx;
};

// Writes the files; returns {files: [paths], sidecar, frames, ...}. Errors are nxr::io::Error: `no_such_row` (nothing by that name), `invalid_argument` (not on a surface, a
// range missing or given for a map, a value type GIFTI cannot hold, ...).
Json export_gifti(const std::filesystem::path& package, const GiftiExport& options);

}  // namespace nxr::exporters
