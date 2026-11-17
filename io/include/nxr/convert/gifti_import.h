// nxr-io — IMPORT OF A GIFTI FILE (docs/audits/2026-11-17-format-mapping-and-coordinates.md, gap 10) into a subject PACKAGE, with the pure reader (`nxr/formats/gifti.h`).
//
//   .surf.gii    a surface manifold (the POINTSET and TRIANGLE arrays). Its positions are in the system the file's CoordinateSystemTransformMatrix names as the DATA space
//                (NIFTI_XFORM_SCANNER_ANAT -> scanner-ras, ALIGNED_ANAT, TALAIRACH, MNI_152, TEMPLATE_OTHER), in that system's unit (the file's millimetres are converted
//                to the unit of the system: metres for scanner RAS). A file whose data space is UNKNOWN but which carries FreeSurfer's volume geometry (`VolGeomC_R/A/S`)
//                is FreeSurfer's surface RAS (`freesurfer-tkr`, metres), placed in scanner RAS by c_ras. Any other UNKNOWN file stands in a system of its own, `gifti:<name>`
//                (metres, the format's millimetres converted), with no transition: not comparable to anything. When the matrix names ANOTHER system than the data space,
//                a transition operator `<name>_<from>_<to>` carries it (in the target's unit). The matrix is always recorded.
//   .func.gii / .shape.gii   one Field per data array on the surface named by `surface` (the vertex counts must agree)
//   .label.gii   a codes Selection per label array on that surface, ONLY when the caller names the parcellation; entries named from the file's label table
// The kind follows the arrays (`kind` to say). `anonymize` drops the file's free-text metadata (Description, UserName, Date, ...).
#pragma once
#include <string>

#include "nxr/bst/session.h"
#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct GiftiImport {
  std::string name;               // the base name of what is written (default: the file's name without .surf.gii / .gii)
  std::string kind;               // surface | overlay | labels; empty: from the arrays
  std::string surface;            // the surface (a manifold of the package) an overlay or a label file belongs on
  std::string parcellation;       // the parcellation a label file is: named by the caller, never guessed
  std::string field_kind = "overlay";
  std::string unit;
  std::string source, source_file;
  std::string licence = "unknown", attribution = "unknown";
  bool redistributable = false;
  bool anonymize = false;
  nxr::bst::Emit progress;
};

nxr::io::Json import_gifti(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const GiftiImport& opt);

}  // namespace nxr::convert
