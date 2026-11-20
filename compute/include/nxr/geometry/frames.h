#pragma once
// nxr::geometry — the CANONICAL frames (north / west) and the ANATOMICAL frames (anterior / superior-or-medial) of a cortex: the one tangent frame per vertex that means the same in
// every subject. Step 2 of the cortical coordinate system (docs/audits/2026-11-11-cortical-coordinates.md; the charts of nxr/geometry/charts.h are step 1).
//
// CANONICAL FRAMES. A port of Rheome's `+rheome/+geom/sphereframe.m` (MIT) = nxr's `nxr.geom.frames.hemisphere_frames(domain="sphere")` in closed form = the frames the Cortical Flow Studio
// reads as the Field `<surface>_canonical_frames` ([V,3,3] float32, rows e1, e2, n; the Selection `<surface>_frame_singularities` lists the singular vertices):
//   e1 = NORTH, the meridian of the subject's REGISTERED SPHERE (FreeSurfer sphere.reg, the Field/surface `<surface>_sphere`) towards +axis, pushed forward to the cortex by a 2x2 least-squares
//        Jacobian per vertex (fitted on its 1-ring edges): the same anatomical direction in every subject.
//   e2 = n x e1, WEST (the Studio's, the spec's and nxr's frames.py word). NOTE: nxr `coords.py` calls the same vector "EAST" (its header); it is the same vector, n x north.
//   n  = the outward normal. Right-handed (e1, e2, n). NaN e1/e2 at SINGULAR vertices (the two poles and any vertex whose Jacobian has condition above maxCondition).
// An angle from north is measured from e1 TOWARDS e2 (counter-clockwise seen from outside, as gaugeHeading's angle in the gauge).
// STORED, north / west is ONE ANGLE ARRAY RELATIVE TO THE GAUGE: `gaugeAngle(surface, e1)`, the angle of north in each vertex's gauge (geometry-central's vertex gauge of the input mesh, one gauge
// for every tangent quantity; docs/specs/2026-11-17-geometry-central-switch.md section 6). A heading h in the gauge is read from north as h - gaugeAngle (wrapped); the normal the frames project
// on is the surface's own `vertexNormals` (corner-angle weighted, geometry-central's), the one tangent plane of the gauge's basis too.
//
// ANATOMICAL PROJECTION (a REPORTING projection of a tangent vector, no longer a frame; was "anatomical frames"). Anterior and the second axis as ONE rotation angle per vertex from north, a label and a mask:
//   * DEFAULT: the anatomical axes are defined ONCE on the TEMPLATE (its anterior direction projected on its tangent planes) and CARRIED to each subject through the sphere registration (the nearest
//     template vertex on the registered spheres), as north / west are: the angle FROM NORTH is what is carried, because north means the same everywhere.
//   * Stored in the subject's SCS (x towards the nasion = anterior, y towards the left ear, z up), never RAS; the template's own convention only enters in `Axes`, once.
//   * OPTION (no registration): SCS-native axes, +x (anterior) and +z (superior) projected on the tangent plane and DOTTED WITH THE 3D VECTOR (`anatomicalProjection`, `anatomicalComponents(a, vectors)`):
//     it needs no frame and no gauge. The record says which definition was used (`Anatomical::definition`).
//   * SECOND AXIS has an ANATOMICAL sign, not a cross product: it points SUPERIOR where superior lies in the surface (lateral and medial cortex) and MEDIAL (towards the midline: -left on the left
//     hemisphere, +left on the right) where medial lies in the surface (dorsal and ventral cortex). `second` says which (kSuperior | kMedial); `handed` is the sign of n x anterior it takes
//     (+1 / -1), so the second axis is  handed * (n x anterior)  and the pair (anterior, second) is a FIXED ROTATION of (north, west) by `angle` (and a flip where handed = -1).
//   * RELIABILITY: where the anterior axis has a short projection on the tangent plane (the frontal and occipital poles, where the surface faces forward or back), the direction is not defined;
//     `reliable` is false below `minProjection` (default 0.34, the sine of 20 degrees: the surface normal within 20 degrees of the anterior axis).
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/geometry/charts.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"

namespace nxr::geometry {

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

/** The surface's outward unit vertex normals [V x 3]: geometry-central's `vertexNormals` of the input geometry (corner-angle weighted), the one tangent plane of the gauge. */
Mat3 vertexNormals(geometry::IntrinsicSurface& surface);
/** The same formula from arrays (a convenience of the pipelines, which take the normals of their IntrinsicSurface once they hold one; no manifold check, a vertex in no face has a zero normal). */
Mat3 vertexNormals(const double* positions, int nV, const int32_t* faces, int nF);
/** Brainstorm's vertex normals (its surfaces' `VertNormals`, tess_normals of the release of 2018-10-12: MATLAB's patch `VertexNormals`), from their documented behaviour: minus the sum of the
 *  incident faces' UNIT normals (no area or angle weight; MATLAB's patch points them against the right-hand rule), normalised [V x 3]; a vertex in no face has a zero normal. They differ from
 *  `vertexNormals` (corner-angle weighted) in direction as well as sign. Held to a golden made by that release (compute/test/golden/gen_vertex_normals_golden.m). */
Mat3 vertexNormalsPatch(const double* positions, int nV, const int32_t* faces, int nF);

/** The gauge angle of a direction per vertex: its angle in each vertex's gauge (atan2 against `IntrinsicSurface::tangentBasis`, i.e. `vertexTangentBasis`); NaN where the direction is NaN, zero, or
 *  normal to the surface. `direction` has one row per vertex of `surface`. The canonical frame's stored form is `gaugeAngle(surface, e1)`. */
Eigen::VectorXd gaugeAngle(geometry::IntrinsicSurface& surface, const Mat3& direction);
/** The same on a chart kit (one connected piece or more; `ChartKit::gaugeHeading` per vertex). */
Eigen::VectorXd gaugeAngle(const ChartKit& kit, const Mat3& direction);
// angle from north of a tangent vector field: frame.rotate(field, canonical_gauge_angle, 1), then arg: a pipeline step (spec 5.3), no longer here

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

/** The anatomical projection: the label, the handedness and the mask, and either the 3D axes (SCS definition, frame-free) or one rotation angle per vertex from north (template definition, carried). */
struct Anatomical {
    Eigen::VectorXd angle;             // radians, from north (e1) towards west (e2) to ANTERIOR; NaN where the reference frame is singular (NaN for the frame-free `anatomicalProjection`)
    Mat3 anterior, secondAxis;         // frame-free SCS definition (`anatomicalProjection`): the unit 3D axes at each vertex, secondAxis = handed * (n x anterior); empty otherwise
    std::vector<int8_t> second;        // Second::Superior | Second::Medial
    std::vector<int8_t> handed;        // +1 / -1: the second axis is handed * (n x anterior)
    std::vector<char> reliable;        // the anterior projection is long enough (and the reference frame defined)
    Eigen::VectorXd projection;        // |anterior projected on the tangent plane|
    std::string definition;            // "scs" | "template" (the record)
    std::string reference;             // "canonical frames" | "signpost gauge" (what `angle` is measured from, recorded) | "none" (frame-free). The "signpost gauge" reference is retired with the pipelines' own use (G6a)
};
using AnatomicalProjection = Anatomical;

/** The FRAME-FREE anatomical projection of the SCS-native axes (`o.axes`): at each vertex the anterior axis projected on the tangent plane of `normals` (outward, unit) and the second axis, as 3D
 *  unit vectors; `angle` is NaN and `reference` "none". Read with `anatomicalComponents(a, vectors)`. Needs no frame and no gauge. */
Anatomical anatomicalProjection(const double* positions, int nV, const int32_t* faces, int nF, const Mat3& normals, const AnatomicalOptions& o = {});

/** The anatomical frame from the surface's OWN axes (SCS-native, or the template's own once). `e1`, `e2` are the reference frame (canonical north / west, or the signpost basis). `normals` outward. */
Anatomical anatomicalFrame(const double* positions, int nV, const int32_t* faces, int nF, const Mat3& normals, const Mat3& e1, const Mat3& e2, const AnatomicalOptions& o = {});

/** CARRY a template's anatomical frame to a subject through the registered spheres (the nearest template vertex of each subject vertex): the angle from north, the label, the handedness and the
 *  mask are copied; the nearest vertex is looked up within the piece of the same number; a subject vertex whose canonical frame is singular (NaN e1) is unreliable. */
Anatomical carryAnatomical(const Anatomical& templ, const double* templateSphere, int nTemplate, const std::vector<int>& templatePiece,
                           const double* subjectSphere, int nSubject, const std::vector<int>& subjectPiece, const Mat3& subjectE1);

/** The connected piece of every vertex, numbered in order of its first vertex (a cortex: its hemispheres, lh first as FreeSurfer lists them). A registered sphere places the hemispheres on the
 *  same sphere, so `carryAnatomical` matches pieces with the same number (empty vectors: one piece). */
std::vector<int> pieces(const int32_t* faces, int nF, int nV);

/** The (anterior, second) components of tangent vectors [V x 3] by the frame-free projection (`anatomicalProjection`): column 0 = vector . anterior, column 1 = vector . secondAxis. NaN where unreliable. */
Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> anatomicalComponents(const Anatomical& a, const Mat3& vectors);

/** The (anterior, second) components of tangent vectors [V x 3], read with the frames (e1, e2, n): column 0 anterior, column 1 the second axis (superior or medial by `second`). NaN where unreliable. */
Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> anatomicalComponents(const Anatomical& a, const Mat3& e1, const Mat3& e2, const Mat3& vectors);

/** The angle of a vector from ANTERIOR towards the second axis, atan2(second, anterior), radians. */
Eigen::VectorXd angleFromAnterior(const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>& components);

/** The same from a heading measured from north (e.g. angleFromNorth): (heading - angle) wrapped, towards the second axis with the handedness. Gauge-free. */
Eigen::VectorXd angleFromAnterior(const Anatomical& a, const Eigen::VectorXd& headingFromNorth);

}  // namespace nxr::geometry

namespace nxr::geometry {
// ── Normal Estimators ────────────────────────────────────────

enum class NormalType {
    AngleWeighted,    // default (matches gpjs "Tip Angle Weighted")
    AreaWeighted,     // weighted by face area
    EqualWeighted,    // equal weighting (normalized sum)
    SphereInscribed,  // (u × v) / (|u|² |v|²) per corner
    MeanCurvature,    // cotangent-weighted (Laplace-Beltrami)
    GaussCurvature    // dihedral angle / edge length weighted
};

/** Compute per-vertex normals using a given estimator.
 *  Returns [nV * 3] flat array (row-major). */
Eigen::MatrixXd normals(const SurfaceArrays& surface, NormalType type);

// ── Face Frames ──────────────────────────────────────────────

/** Per-face orthonormal tangent frame.
 *
 *  Each face f has an in-plane orthonormal basis (e1, e2) and a
 *  normal n, all 3D vectors in world coordinates. Used by:
 *    - GPU-side particle advection (frames define each face's
 *      local 2D coordinate system)
 *    - Tangent vector field visualization (face vectors expressed
 *      as 2D coords in (e1, e2) reconstruct in JS / shaders)
 *    - Texture mapping where a per-face frame is needed
 *
 *  Convention matches direction_field.cpp::faceOrthonormalBasis:
 *  e1 is the unit edge vector along f.halfedge(), e2 = n × e1.
 *  This is the same frame the trivial-connection direction field
 *  uses, so tangent vectors transported between faces are
 *  interpretable in the same coordinate system. */
struct FaceFrames {
    Eigen::MatrixXd e1;       // [nF, 3] — first tangent vector
    Eigen::MatrixXd e2;       // [nF, 3] — second tangent (n × e1)
    Eigen::MatrixXd normals;  // [nF, 3] — face normals
};

FaceFrames frames(const SurfaceArrays& surface);

/* Per-vertex orthonormal tangent frame = geometry-central's gauge of the (mollified) input mesh (IntrinsicSurface::tangentBasis; equal to its
 * `vertexTangentBasis` when nothing was mollified): the 3D realization of the per-vertex tangent space in which the vertex connection-Laplacian's complex
 * eigenvector coordinates, the vector heat method's and the transport live:
 *   n  = the corner-angle-weighted vertex normal
 *   e1 = the average of the vertex's outgoing edges, each projected on the tangent plane and rotated back by its own rescaled angle (angle 0 = the input's
 *        v.halfedge())
 *   e2 = n x e1
 * A connection-mode coordinate z=(a,b) at vertex i decodes to a*e1_i + b*e2_i. */
struct VertexFrames {
    Eigen::MatrixXd e1;       // [nV, 3] — first tangent
    Eigen::MatrixXd e2;       // [nV, 3] — second tangent (n × e1)
    Eigen::MatrixXd normals;  // [nV, 3] — vertex normals (area-weighted)
};

VertexFrames vertexFrames(const SurfaceArrays& surface);

// ── Complex frame ("grid") ───────────────────────────────────
//
// The per-element tangent frame packed as a complex 3-vector
// c = e1 + i·e2 ∈ ℂ³ (real part = e1, imag part = e2). The unit
// normal is recovered as real(c) × imag(c) and is not stored
// separately. This is the representation Brainstorm's Geometry.*.grid
// fields use; see the bundle design spec §5.1.
Eigen::MatrixXcd vertexGrid(const SurfaceArrays& surface);   // [nV, 3] complex, from vertexFrames (geometry-central's gauge: what the Levi-Civita connection Laplacian's coordinates decode against)
Eigen::MatrixXcd faceGrid(const SurfaceArrays& surface);     // [nF, 3] complex
} // namespace nxr::geometry

