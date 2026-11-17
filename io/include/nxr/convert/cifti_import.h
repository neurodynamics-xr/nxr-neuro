// nxr-io — IMPORT OF A CIFTI-2 DENSE MAP (docs/audits/2026-11-17-format-mapping-and-coordinates.md, gap 11) into a subject PACKAGE, with the pure reader (`nxr/formats/cifti.h`).
//
//   dscalar / dlabel   the cortical brain models (surface vertices of CORTEX_LEFT / CORTEX_RIGHT) are placed on the template surfaces the caller NAMES (`left`, `right`:
//                      manifolds of the package whose vertex count is the file's): an `indices` Selection of the vertices the file covers (the medial wall is left out),
//                      and per map a Field on the surface (NaN outside the Selection); a dlabel becomes a codes Selection per map, ONLY when `parcellation` is named.
//   voxel models       are not imported yet (the importer reports the structures it left out); dtseries and parcellated files are refused by name.
#pragma once
#include <string>

#include "nxr/bst/session.h"
#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct CiftiImport {
  std::string name;               // the base name of what is written (default: the file's name without .dscalar.nii / .nii)
  std::string left, right;        // the package's surfaces for CORTEX_LEFT and CORTEX_RIGHT
  std::string parcellation;       // the parcellation a dlabel is
  std::string field_kind = "overlay";
  std::string unit;
  std::string source, source_file;
  std::string licence = "unknown", attribution = "unknown";
  bool redistributable = false;
  bool anonymize = false;
  nxr::bst::Emit progress;
};

nxr::io::Json import_cifti(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const CiftiImport& opt);

}  // namespace nxr::convert
