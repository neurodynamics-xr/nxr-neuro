// nxr-io — IMPORT OF A FREESURFER SUBJECT (docs/audits/2026-11-17-format-mapping-and-coordinates.md, gap 10) into a subject PACKAGE, with the readers of `nxr/io/surface.h`.
//
//   surfaces     `surf/lh.<s>` and `surf/rh.<s>` read as ONE surface (the hemispheres are its components), in `freesurfer-tkr`: FreeSurfer's surface RAS, METRES (the file's
//                millimetres / 1000), the system the template importers use
//   scanner RAS  a transition OPERATOR `<name>_freesurfer-tkr_scanner-ras`, 3 x 4, metres: the surfaces' tkr coordinates + c_ras (the volume's centre in scanner RAS, which
//                the volume `mri/orig.mgz` holds, or else the surface file's own volume-geometry tail). With no c_ras the system is not placed and nothing is guessed
//   annotations  `label/lh.<a>.annot` + `rh.<a>.annot`, each a codes Selection on the surface (a set, one code per vertex), the entries named from the file's colour table
//                with the hemisphere. Only the annotations the caller NAMES are imported (a parcellation is imported only when named)
//   overlays     `surf/lh.<o>` + `rh.<o>` (thickness, curv, sulc, area, ... : FreeSurfer's curvature format), each a Field on the surface
//   `sphere.reg`  when both exist, the registration sphere of the first surface (a sibling geometry in `freesurfer-sphere`)
// RAS only: FreeSurfer's axes are RAS in its surface frame and nothing is converted. `anonymize` drops the "created by" line and the volume's file name the surfaces record.
#pragma once
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/session.h"
#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct FreeSurferImport {
  std::string name = "cortex";                // the prefix of what is written: <name>_<surface>
  std::vector<std::string> surfaces = {"white"};
  std::vector<std::string> annotations;       // `aparc`, ...: named by the caller
  std::vector<std::string> overlays;          // `thickness`, ...
  std::string orig;                           // the volume that holds c_ras; empty: <folder>/mri/orig.mgz when there is one, else the surface's own tail
  std::string field_kind = "overlay";
  std::string source, source_file;
  std::string licence = "unknown", attribution = "unknown";
  bool redistributable = false;
  bool anonymize = false;
  nxr::bst::Emit progress;
};

// `folder` is a FreeSurfer subject folder (surf/, label/, mri/)
nxr::io::Json import_freesurfer_subject(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& folder, const FreeSurferImport& opt);

// c_ras of a FreeSurfer volume (`.mgh` / `.mgz`) or of a surface file's volume-geometry tail (nibabel's `volume_info['cras']`), in millimetres; none when the file has none
std::optional<std::array<double, 3>> freesurfer_c_ras(const std::string& file);
// a FreeSurfer curvature / overlay file (the new format: magic 0xFFFFFF, vertices, faces, values per vertex, big-endian float32 values; or the old three-byte-count format)
std::vector<float> read_freesurfer_curv(const std::string& file);

}  // namespace nxr::convert
