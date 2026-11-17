#include "nxr/convert/world_frames.h"

namespace nxr::convert {

std::optional<World> world_of_nifti_code(int code) {
  switch (code) {
    case 1: return World{"scanner-ras", "m", "the scanner's RAS frame (NIfTI code 1): x right, y anterior, z superior, from the scanner's own coordinates"};
    case 2: return World{"nifti-aligned", "mm", "the frame of another image this one is aligned to (NIfTI code 2)"};
    case 3: return World{"talairach", "mm", "Talairach space (NIfTI code 3)"};
    case 4: return World{"mni", "mm", "MNI template space"};
    case 5: return World{"template-other", "mm", "another template's space (NIfTI code 5)"};
    default: return std::nullopt;
  }
}

std::optional<World> world_of_gifti_space(const std::string& x) {
  if (x == "NIFTI_XFORM_SCANNER_ANAT") return world_of_nifti_code(1);
  if (x == "NIFTI_XFORM_ALIGNED_ANAT") return world_of_nifti_code(2);
  if (x == "NIFTI_XFORM_TALAIRACH") return world_of_nifti_code(3);
  if (x == "NIFTI_XFORM_MNI_152") return world_of_nifti_code(4);
  if (x == "NIFTI_XFORM_TEMPLATE_OTHER") return world_of_nifti_code(5);
  return std::nullopt;
}

World scanner_ras() { return *world_of_nifti_code(1); }

void ensure_world(nxr::io::SubjectWriter& sub, const World& w) { sub.frame({"head", w.key, w.unit, w.description, ""}); }

}  // namespace nxr::convert
