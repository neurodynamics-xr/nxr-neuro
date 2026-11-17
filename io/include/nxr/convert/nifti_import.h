// nxr-io — IMPORT OF A NIfTI VOLUME (docs/audits/2026-11-17-format-mapping-and-coordinates.md, gap 9) into a subject PACKAGE, made with the pure reader
// (`nxr/formats/nifti.h`).
//
//   volume       a lattice manifold in its own voxel chart `voxel:<name>`: one dimension row per axis (i, j, k) carries the voxel size as SPACING in mm (the Studio's
//                convention); a Field of the intensities on it (C order, scaled as the file says, precision as the file's)
//   world frame  a transition OPERATOR `<name>_<frame>`: voxel index -> the file's world system, 3 x 4, MILLIMETRES, the voxel size folded in
//                (`world_mm = M @ [i, j, k, 1]`). The system is the one the file's own code names: sform_code (else qform_code) 1 scanner RAS (`scanner-ras`, m), 2 aligned to
//                another image (`nifti-aligned`), 3 Talairach, 4 MNI (`mni`, mm), 5 another template (`template-other`). A file with no code states no world frame: the
//                voxel chart stands alone, with no transition, and nothing is guessed. NIfTI's axes are RAS by definition: nothing is converted.
//   labels       a codes Selection on the volume, ONLY when the caller names the parcellation (`parcellation`); the entries are named from `label_names` (a file of
//                `code name` lines), or by their code
// Volumes of 4 dimensions or more (a series of volumes) are refused; `anonymize` drops the file's free-text description and auxiliary file name.
#pragma once
#include <string>

#include "nxr/bst/session.h"
#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct NiftiImport {
  std::string name;                   // the base name of what is written (default: the file's name without .nii / .nii.gz)
  std::string kind;                   // volume | labels; empty: labels when `parcellation` is named, else volume
  std::string parcellation;           // the name of the parcellation a label volume is (the importer never guesses it)
  std::string label_names;            // a file of `code name` lines for the entries of the parcellation
  std::string field_kind = "intensity";
  std::string unit;                   // the unit of the values, when the caller knows it
  std::string source, source_file;
  std::string licence = "unknown", attribution = "unknown";
  bool redistributable = false;
  bool anonymize = false;
  nxr::bst::Emit progress;
};

nxr::io::Json import_nifti(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const NiftiImport& opt);

}  // namespace nxr::convert
