// nxr-pipelines — THE INSTRUMENT'S BLUR (`pointspread compute`; spec docs/specs/2026-11-14-spatial-events.md, "Statistics and nulls": "the instrument's blur is a separate measurement, never built into a tool"). A unit current is planted at a sample of
// vertices through the STORED forward model (the head model's gain, `headmodel compute`: sensors <- sources) and read back through the STORED inverse kernel (`inverse compute`): the resolution matrix's column of that source. The column is read by the SAME chain the spatial events are:
// the free current through each level's hat on the glued cortex, its power summed over the components, the largest L2-normalised response among the levels at the patch's centre (nxr::scalespace, Lindeberg's scale selection), the patch cut where the power at the reference level is at
// least `fraction` of its peak (a point source has no noise floor: half the maximum, not the events' false-alarm threshold; the connected patch by nxr::cores::corePatches). Per sampled vertex and per Cartesian orientation of the source (no orientation is assumed: x, y and z each, as the inverse
// gives it), stored as ONE statistic of a recording (the band's chain is the recording's), `pointspread.<kernel>` [vertices x orientations x measures]: the peak's distance from the source (cm), the patch's equal-area diameter (cm), its area (cm2), the characteristic level and
// its size (cm: the level's centre wavelength) and the peak power. A reader compares the events' sizes with it: a patch no larger than a point source's is the instrument's blur, not the source's extent.
// WHAT IT CANNOT SAY: the blur of THIS head model and kernel (their noise model and regularisation), for a source ALONE (many sources blur together differently); sizes are octave-spaced levels; a source on the bridge or at a vertex the sample misses is not measured.
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct PointSpreadOptions {
  std::string recording;                          // REQUIRED: the recording whose band the chain is (the statistic is OF it)
  std::optional<std::string> forward;             // the head model Operator (kind forward) by name; default: the subject's only one
  std::optional<std::string> kernel;              // the inverse kernel by name; default: the subject's only one of the forward's sources
  std::vector<int> levels = {2, 3, 4, 5, 6, 7, 8};   // the scale space (as `spatial_events`)
  int reference_level = 6;                        // the level whose power the patch is cut from
  std::string spatial_shape = "mexhat", transform = "auto";
  int modes = 0, order = 0;
  double tolerance = 0.01;
  double fraction = 0.5;                          // the patch: connected power at least this fraction of the peak at the reference level
  int vertices = 20;                              // the sample: this many native vertices spread evenly over the surface (lowest index first) ...
  std::vector<int> list;                          // ... or these native vertices (wins when given)
  std::optional<std::string> name;                // default `pointspread.<kernel>`
  bool replace = false;
};
struct PointSpreadResult { nxr::io::Json info; bool hit = false; };
PointSpreadResult pointspread_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const PointSpreadOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
