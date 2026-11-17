// nxr-io — IMPORT OF MNE's FIF OBJECT FILES (docs/audits/2026-11-17-format-mapping-and-coordinates.md, gap 8): a source space (-src.fif), a noise or data covariance
// (-cov.fif), a forward solution (-fwd.fif), an inverse operator (-inv.fif), an event list (-eve.fif), an annotation file (-annot.fif) and a head-to-MRI transform (-trans.fif)
// become entities of a subject PACKAGE, made with the pure reader (`nxr/formats/fif_objects.h`).
//
//   source space   a surface source space is the FULL surface (the hemispheres in one surface manifold) in the file's frame (MRI: `freesurfer-tkr`; head: `head`, metres) and an
//                  `indices` Selection `<name>_source_space` of its used vertices (MNE's `vertno`): a source space is never a manifold of its own (D30, D158); the decimated
//                  mesh (`use_tris`) is counted in the provenance. A volume source space is a lattice manifold in its voxel chart (spacing in mm, the voxel -> frame transition in mm)
//                  with an `indices` Selection of the used voxels; a discrete one a points manifold.
//   covariance     a STATISTIC (measure `covariance`) on the channels' Selection, detached (computed elsewhere); `nfree` is its `n_members` and `dof`, the method, bads, projectors and
//                  score are its options
//   forward        an Operator `forward`, source -> channels, [M x 3V] (free orientation, vector3) or [M x V] (fixed), with the source-space Selection and the channels' Selection; the
//                  components' frame is in the provenance
//   inverse        the PARTS of the operator, never a kernel: `<n>_eigen_leads` (rank -> source), `<n>_eigen_fields` (channels -> rank), the singular values a Field on the rank axis, the
//                  noise covariance a statistic, the source covariance, depth and orientation priors Fields on the sources (diagonal ones)
//   events         instants of an existing recording, as the recording's event Selection (`<rec>_events`)
//   annotations    spans, with their channels, of an existing recording
//   transform      a transition operator `head` -> `freesurfer-tkr`
// Channels (forward, inverse, covariance) are those of an EXISTING sensor manifold of the package, matched by name (a channel the manifold lacks is refused, naming it). The head-to-MRI
// transform a forward or inverse file carries is also written as a transition. `anonymize` drops the file names, working folder and command line the files record.
#pragma once
#include <string>

#include "nxr/bst/session.h"
#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct FifObjectImport {
  std::string kind;                   // src | cov | fwd | inv | eve | annot | trans; empty: from the file
  std::string name;                   // the base name of what is written (default: the file's name without its suffix)
  std::string acquisition = "rest";
  std::string channels;               // the sensor manifold of the package the file's channels are in (default: the only one)
  std::string recording;              // the recording an event or annotation file belongs to
  std::string surface;                // an existing surface manifold of the package to put a surface source space's Selection on (its vertex count must match)
  std::string source;                 // the file's path as it is recorded (relative), and the file to hash for the origin record
  std::string source_file;
  std::string licence = "unknown", attribution = "unknown";
  bool redistributable = false;
  bool anonymize = false;
  nxr::bst::Emit progress;
};

// The kind of object a FIF file holds (src, cov, fwd, inv, eve, annot, trans); refused when it holds none of them.
std::string fif_object_kind(const std::string& path);
// Import the file into the subject `sub`; returns {kind, name, items..., origin, origin_items}.
nxr::io::Json import_fif_object(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const std::string& path, const FifObjectImport& opt);

}  // namespace nxr::convert
