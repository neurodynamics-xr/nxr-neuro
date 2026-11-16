// nxr-pipelines — the cortical frames of a subject's surface, stored as the dataset stores them (Step 2 of the cortical coordinate system; maths in nxr/frames.h).
//
//   canonical frames   `<surface>_canonical_frames`  Field [V,3,3] float32, kind "canonical frames", value_type matrix3; rows e1 = north, e2 = n x e1 = west, n outward; NaN at singular vertices
//                      `<surface>_frame_singularities`  Selection (indices) of the singular vertices (every hemisphere's two poles, and the ill-conditioned push-forwards)
//                      `<surface>_canonical_gauge_angle`  Field [V] float32, rad: north's angle in the signpost basis (ChartKit::gaugeHeading), to turn a heading in the signpost gauge into one from north
//                      READ when stored (the Studio's, nxr's, ours), COMPUTED from `<surface>_sphere` (the registered sphere) when absent: the same as nxr `hemisphere_frames(domain="sphere")`.
//   anatomical frames  `<surface>_anatomical_angle`     Field [V] float32, rad, from north towards west to ANTERIOR (NaN where the reference frame is singular)
//                      `<surface>_anatomical_second`    Field [V] integer: 0 the second axis points SUPERIOR, 1 MEDIAL
//                      `<surface>_anatomical_handed`    Field [V] integer: +1 / -1, the second axis is handed * (n x anterior)
//                      `<surface>_anatomical_reliable`  Selection (indices) of the vertices where the anterior projection is long enough (not the poles)
//                      the record (producer_json of the angle): the definition used ("template:<subject>" carried through the registered spheres, or "scs" / "ras": the surface's own axes projected on its
//                      tangent planes), the reference ("canonical frames", or "signpost gauge" when there is no registration), the threshold, the axes.
// Everything lives in the subject's SCS (x anterior, y left, z up), never RAS; a template's own convention (`convention`) enters once, when ITS frame is made.
#pragma once
#include <Eigen/Dense>
#include <optional>
#include <string>

#include "nxr/frames.h"
#include "nxr/io.h"

namespace nxr::pipelines {

struct FramesComputeOptions {
  std::optional<std::string> surface;        // the subject's surface (default: the primary one)
  std::optional<std::string> templ;          // the TEMPLATE subject of the same dataset whose anatomical frame is carried through the registered spheres; unset: the surface's own SCS axes
  std::optional<std::string> template_surface;   // the template's surface (default: its primary one)
  std::string convention = "scs";            // the coordinates the surface's own axes are given in, when they are used: scs (anterior +x, superior +z, left +y) or ras (anterior +y, superior +z, left -x)
  std::string side = "auto";                 // a single-piece surface's hemisphere: auto (each piece by position), left or right
  double min_projection = 0.34;              // reliable where the anterior axis projects on the tangent plane with at least this length (sin 20 degrees)
  double max_condition = 50.0;               // canonical frames: a vertex whose push-forward Jacobian is worse is singular (frames.py)
  bool replace = false;                      // an existing anatomical frame is moved aside
};
struct FramesComputeResult {
  nxr::io::Json info;
  bool canonical_computed = false;           // false: the stored ones were read
};
FramesComputeResult frames_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const FramesComputeOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

// What a surface has stored (read by `opticalflow compute`): the canonical frames and the anatomical frame, or nothing.
struct StoredFrames {
  bool canonical = false, anatomical = false;
  nxr::frames::Mat3 e1, e2, n;               // [V x 3] (when canonical; NaN at a singular vertex); n the stored third row
  nxr::frames::Anatomical anatomy;           // (when anatomical)
  std::string anatomical_id, anatomical_modified;   // the angle field's identity (a cache key part)
};
StoredFrames read_stored_frames(nxr::io::SubjectWriter& sub, const std::string& surface_name, std::int64_t n_vertices);

}  // namespace nxr::pipelines
