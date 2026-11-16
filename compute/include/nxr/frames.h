#pragma once
// nxr::frames — the CANONICAL frames (north / west) and the ANATOMICAL frames (anterior / superior-or-medial) of a cortex: the one tangent frame per vertex that means the same in
// every subject. Step 2 of the cortical coordinate system (docs/audits/2026-11-11-cortical-coordinates.md; the charts of nxr/chart.h are step 1).
//
// CANONICAL FRAMES. A port of Rheome's `+rheome/+geom/sphereframe.m` (MIT) = nxr's `nxr.geom.frames.hemisphere_frames(domain="sphere")` in closed form = the frames the Cortical Flow Studio
// reads as the Field `<surface>_canonical_frames` ([V,3,3] float32, rows e1, e2, n; the Selection `<surface>_frame_singularities` lists the singular vertices):
//   e1 = NORTH, the meridian of the subject's REGISTERED SPHERE (FreeSurfer sphere.reg, the Field/surface `<surface>_sphere`) towards +axis, pushed forward to the cortex by a 2x2 least-squares
//        Jacobian per vertex (fitted on its 1-ring edges): the same anatomical direction in every subject.
//   e2 = n x e1, WEST (the Studio's, the spec's and nxr's frames.py word). NOTE: nxr `coords.py` calls the same vector "EAST" (its header); it is the same vector, n x north.
//   n  = the outward normal. Right-handed (e1, e2, n). NaN e1/e2 at SINGULAR vertices (the two poles and any vertex whose Jacobian has condition above maxCondition).
// An angle from north is measured from e1 TOWARDS e2 (counter-clockwise seen from outside, as gaugeHeading's angle in the signpost basis).
//
// ANATOMICAL FRAMES (). Anterior and the second axis as ONE rotation angle per vertex from north, a label and a mask:
//   * DEFAULT: the anatomical axes are defined ONCE on the TEMPLATE (its anterior direction projected on its tangent planes) and CARRIED to each subject through the sphere registration (the nearest
//     template vertex on the registered spheres), as north / west are: the angle FROM NORTH is what is carried, because north means the same everywhere.
//   * Stored in the subject's SCS (x towards the nasion = anterior, y towards the left ear, z up), never RAS; the template's own convention only enters in `Axes`, once.
//   * OPTION (no registration): SCS-native axes, +x (anterior) and +z (superior) projected on the tangent plane. The record says which definition was used (`Anatomical::definition`).
//   * SECOND AXIS has an ANATOMICAL sign, not a cross product: it points SUPERIOR where superior lies in the surface (lateral and medial cortex) and MEDIAL (towards the midline: -left on the left
//     hemisphere, +left on the right) where medial lies in the surface (dorsal and ventral cortex). `second` says which (kSuperior | kMedial); `handed` is the sign of n x anterior it takes
//     (+1 / -1), so the second axis is  handed * (n x anterior)  and the pair (anterior, second) is a FIXED ROTATION of (north, west) by `angle` (and a flip where handed = -1).
//   * RELIABILITY: where the anterior axis has a short projection on the tangent plane (the frontal and occipital poles, where the surface faces forward or back), the direction is not defined;
//     `reliable` is false below `minProjection` (default 0.34, the sine of 20 degrees: the surface normal within 20 degrees of the anterior axis).
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/chart.h"

namespace nxr::frames {

using Mat3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;   // [V x 3]

/** `sphereframe.m`'s output. e1, e2 are NaN at singular vertices. */
struct Canonical {
    Mat3 e1, e2;                       // north, west (unit, tangent)
    Eigen::VectorXd colat, lon;        // on the sphere from +axis, radians
    Eigen::VectorXd condition;         // of each push-forward Jacobian
    std::vector<char> singular;        // the two poles and the ill-conditioned push-forwards
    int north = 0, south = 0;          // the vertices closest to +axis / -axis (of the first connected piece: Rheome's, on one hemisphere)
    std::vector<int> poles;            // north, south of every connected piece (a cortex: both hemispheres); all are singular
    Eigen::Vector3d axis{0, 0, 1};
};

/** `rheome.geom.sphereframe(V, F, N, Sph, Axis=, MaxCondition=)`: `positions` [3 nV], `faces` [3 nF], `normals` [3 nV] outward, `sphere` [3 nV] the registered-sphere point of each vertex (any radius). */
Canonical canonicalFrames(const double* positions, int nV, const int32_t* faces, int nF, const double* normals, const double* sphere,
                          const Eigen::Vector3d& axis = Eigen::Vector3d(0, 0, 1), double maxCondition = 50.0);

/** The area-weighted outward vertex normals [V x 3] (unit), the Studio's `tangentBasis` normals. */
Mat3 vertexNormals(const double* positions, int nV, const int32_t* faces, int nF);

/** The gauge angle of a direction per vertex: its angle in each vertex's signpost basis (`ChartKit::gaugeHeading`); NaN where the direction is NaN, zero, or normal to the surface. */
Eigen::VectorXd gaugeAngle(chart::ChartKit& kit, const Mat3& direction);

/** A tangent vector field's angle from NORTH at each vertex, in the signpost gauge: gaugeHeading(v) - gaugeHeading(e1), wrapped to (-pi, pi]. */
Eigen::VectorXd angleFromNorth(chart::ChartKit& kit, const Mat3& vectors, const Mat3& e1);

// ── anatomical frames ────────────────────────────────────────────────────────────────────────────────────

enum class Second : int8_t { Superior = 0, Medial = 1 };
enum class Side { Left, Right, Auto };

/** The anatomical axes as unit vectors in the COORDINATES OF THE SURFACE they are given for. SCS: anterior +x, superior +z, left +y. A template in RAS: anterior +y, superior +z, left -x. */
struct Axes {
    Eigen::Vector3d anterior{1, 0, 0}, superior{0, 0, 1}, left{0, 1, 0};
};
inline Axes scsAxes() { return Axes{}; }

struct AnatomicalOptions {
    Axes axes;                         // the axes of the surface's own coordinates
    double minProjection = 0.34;       // reliable where |anterior projected on the tangent plane| >= this (sin 20 degrees)
    Side side = Side::Auto;            // the hemisphere of a single-piece surface; Auto: each connected piece by the sign of its mean position along `left` (needs two pieces)
};

/** One rotation angle per vertex from north, the label, the handedness and the mask. */
struct Anatomical {
    Eigen::VectorXd angle;             // radians, from north (e1) towards west (e2) to ANTERIOR; NaN where the reference frame is singular
    std::vector<int8_t> second;        // Second::Superior | Second::Medial
    std::vector<int8_t> handed;        // +1 / -1: the second axis is handed * (n x anterior)
    std::vector<char> reliable;        // the anterior projection is long enough (and the reference frame defined)
    Eigen::VectorXd projection;        // |anterior projected on the tangent plane|
    std::string definition;            // "scs" | "template" (the record)
    std::string reference;             // "canonical frames" | "signpost gauge" (what the angle is measured from)
};

/** The anatomical frame from the surface's OWN axes (SCS-native, or the template's own once). `e1`, `e2` are the reference frame (canonical north / west, or the signpost basis). `normals` outward. */
Anatomical anatomicalFrame(const double* positions, int nV, const int32_t* faces, int nF, const Mat3& normals, const Mat3& e1, const Mat3& e2, const AnatomicalOptions& o = {});

/** CARRY a template's anatomical frame to a subject through the registered spheres (the nearest template vertex of each subject vertex): the angle from north, the label, the handedness and the
 *  mask are copied; the nearest vertex is looked up within the piece of the same number; a subject vertex whose canonical frame is singular (NaN e1) is unreliable. */
Anatomical carryAnatomical(const Anatomical& templ, const double* templateSphere, int nTemplate, const std::vector<int>& templatePiece,
                           const double* subjectSphere, int nSubject, const std::vector<int>& subjectPiece, const Mat3& subjectE1);

/** The connected piece of every vertex, numbered in order of its first vertex (a cortex: its hemispheres, lh first as FreeSurfer lists them). A registered sphere places the hemispheres on the
 *  same sphere, so `carryAnatomical` matches pieces with the same number (empty vectors: one piece). */
std::vector<int> pieces(const int32_t* faces, int nF, int nV);

/** The (anterior, second) components of tangent vectors [V x 3], read with the frames (e1, e2, n): column 0 anterior, column 1 the second axis (superior or medial by `second`). NaN where unreliable. */
Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> anatomicalComponents(const Anatomical& a, const Mat3& e1, const Mat3& e2, const Mat3& vectors);

/** The angle of a vector from ANTERIOR towards the second axis, atan2(second, anterior), radians. */
Eigen::VectorXd angleFromAnterior(const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>& components);

/** The same from a heading measured from north (e.g. angleFromNorth): (heading - angle) wrapped, towards the second axis with the handedness. Gauge-free. */
Eigen::VectorXd angleFromAnterior(const Anatomical& a, const Eigen::VectorXd& headingFromNorth);

}  // namespace nxr::frames
