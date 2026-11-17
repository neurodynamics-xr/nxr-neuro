// nxr-io — the world frames an imported image or surface can name (docs/audits/2026-11-17-format-mapping-and-coordinates.md, Q8): RAS-family systems only, each on the
// subject's head space with its unit. A system is made when first named; the transition that places it is the importer's own operator, written in the same compose.
#pragma once
#include <optional>
#include <string>

#include "nxr/io/writer.h"

namespace nxr::convert {

struct World {
  std::string key, unit, description;
  double mm_per_unit() const { return unit == "m" ? 1000.0 : 1.0; }   // the length of one unit of the system, in millimetres
};

// the system a NIfTI transform code (1 scanner, 2 aligned, 3 Talairach, 4 MNI, 5 another template) names; none for 0
std::optional<World> world_of_nifti_code(int code);
// the same by GIFTI's name (`NIFTI_XFORM_SCANNER_ANAT`, ...); none for NIFTI_XFORM_UNKNOWN
std::optional<World> world_of_gifti_space(const std::string& xform);
// FreeSurfer's scanner RAS is NIfTI's code 1
World scanner_ras();
// make the system on the head space when it is new
void ensure_world(nxr::io::SubjectWriter& sub, const World& w);

}  // namespace nxr::convert
