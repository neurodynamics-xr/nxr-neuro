#pragma once
// nxr::forms — a field's TYPE and its FORMS: the step between a source estimate and any measure (the `form` step of a pipeline; the paramount architecture:
// a pure node, arrays in, arrays out). A line port of the Cortical Flow Studio's `fieldForms.ts` (`FORMS`, `formInfo`, `widthOf`, `formValues`) and of the
// parts of `faceGradient.ts` the forms read (`assembleFaceGradient`, `flowFaceScalarsCpu`, `faceToVertexCpu`, `flowFieldCpu`, `normalPartCpu`,
// `flowFieldPartCpu`, `meanCurvatureCpu`, `divCurv2HCpu`), the reference. JavaScript's
// arithmetic is kept: values held in double, stored as float32 where the Studio stores a Float32Array; Math.hypot as V8 computes it; compiled without fused
// multiply-add. test_forms holds it to the Studio's own functions (golden: test/golden/gen_forms_golden.mts).
//
//   TYPE   what the value at a vertex IS:   Real ℝ a number (1 a vertex) · Complex ℂ a tangent vector, EMBEDDED (3 a vertex: x y z)
//                                           · Quaternion ℍ [w x y z] (4 a vertex: a number w and an ambient vector x; a free-orientation current is ℍ, w = 0)
//   FORM   a quantity drawn from it, of its own type (the Studio's ids and names):
//            ℝ  f
//            ℂ  abs (|v|) · div · curl · div_tan · div_3d · div_curv · div_2H · curl_tan · curl_3d · curl_curv (ℝ) · v · Jv (ℂ: Jv = n × v)
//            ℍ  absq (|q|) · nx (n·x) · w · the div / curl forms of x (ℝ) · xt (x − (n·x) n, ℂ) · x (ℍ: the vector in space)
//          div / curl are the AMBIENT face-gradient forms of the full 3-D vector (Brainstorm's bst_divergence / bst_curl); _tan the tangent flow's own,
//          _3d the full vector (= div / curl), _curv the operator on the normal part J_n n alone, div_2H its continuum estimate 2H J_n.
//   STACKED   values may hold several frames one after another (length = frames x widthOf(type, n)); a form is computed frame by frame, stacked the same way.
//
// COMPLEX VALUES (the analytic signal; an extension of the Studio's real forms, for the phase): `formComplex` takes a field's real and imaginary parts.
//   A LINEAR form (every form but abs and absq) is applied to each part: form(re) + i form(im), so the phase is kept and the result may feed a phase gradient.
//   abs / absq give the ENVELOPE, sqrt of the sum of the squares of every real and imaginary component (|v| over C^3, |q| over C^4): a real number with no
//   phase. `isLinear` says which is which; a pipeline's plan refuses a non-linear form as the input of a phase.
#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nxr::forms {

enum class FieldType { Real, Complex, Quaternion };
enum class Form { f, abs, div, curl, div_tan, div_3d, div_curv, div_2H, curl_tan, curl_3d, curl_curv, v, Jv, absq, nx, w, xt, x };

struct FormInfo {
    Form id;
    const char* key;    // the Studio's FormId ("nx", "div_tan", …)
    const char* sym;    // its symbol ("n·x")
    const char* name;   // its name ("out of the surface")
    FieldType out;      // the type of the form's values
};

/** a type's forms, in the Studio's order (`FORMS[t]`) */
const std::vector<FormInfo>& formsOf(FieldType t);
/** the form of a type, or nothing when the type has no such form (`formInfo`) */
std::optional<FormInfo> formInfo(FieldType t, Form id);
/** a form by its Studio id ("nx"); nothing for an unknown id */
std::optional<Form> formOf(const std::string& key);
/** the numbers one frame of a field of this type holds: n, 3n or 4n (`widthOf`) */
std::size_t widthOf(FieldType t, std::size_t n);
/** a divergence / rotation form (`isDivCurl`) */
bool isDivCurl(Form id);
/** a LINEAR form: every form but abs and absq (it keeps a complex field's phase) */
bool isLinear(Form id);

/** `assembleFaceGradient`: the face-gradient operator of a mesh, as the Studio builds it (float32 positions in, its arrays as Float32Array / Int32Array). */
struct FaceGradientOperator {
    int nF = 0, nV = 0;
    std::vector<float> grads;                  // [9 nF] the three hat gradients a face: g0.xyz g1.xyz g2.xyz
    std::vector<float> normals;                // [3 nF] unit face normals (outward by CCW winding)
    std::vector<float> areas;                  // [nF] face areas, 0 for a degenerate face
    std::vector<int32_t> faces;                // [3 nF]
    std::vector<int32_t> vertexFaceOffsets;    // [nV + 1] vertex -> face CSR
    std::vector<int32_t> vertexFaceIndices;    // [3 nF live]
    std::vector<float> vertexAreaSum;          // [nV] the sum of the incident faces' areas
    std::vector<float> vertexNormals;          // [3 nV] area-weighted unit vertex normals from the face normals
    double signedVolume = 0;                   // the divergence theorem's volume: > 0 iff the winding is outward (a check, never a correction)
    int degenerateFaces = 0;
    std::vector<float> position;               // [3 nV] the positions (the mean curvature reads them)
};
FaceGradientOperator assembleFaceGradient(const float* position, const int32_t* faces, int nV, int nF);

/** `meanCurvatureCpu`: signed mean curvature H [nV] (1/length, +1/R on a sphere with outward normals) */
std::vector<float> meanCurvature(const FaceGradientOperator& op);

/** `formValues`: the form `id` of a field of type `t` (`values` [frames x widthOf(t, nV)], float32), stacked frame by frame: nV a frame for a scalar form,
 *  3 nV for a vector one, 4 nV for ℍ's `x`. Throws std::invalid_argument when the type has no such form or the length is not whole frames. */
std::vector<float> formValues(FieldType t, Form id, const float* values, std::size_t length, const FaceGradientOperator& op);

/** a complex field's form: a linear form of re + i im is form(re) + i form(im); abs / absq give the envelope (`im` empty). */
struct ComplexForm {
    std::vector<float> re, im;
};
ComplexForm formComplex(FieldType t, Form id, const float* re, const float* im, std::size_t length, const FaceGradientOperator& op);

/** a LINEAR form of a free current's COLUMNS.
 *  K [3 nV × C]: column c is one free current (x y z a vertex, w = 0, the Studio's ℍ); the result [nV × C] holds its form `id` (a scalar form of ℍ: curl_tan, div_tan, nx, …, per column, as
 *  `formValues`). Pure: a matrix in, a matrix out; float32 inside, as the Studio. Throws std::invalid_argument for a form that is not a linear scalar form of ℍ, or a K of other rows. */
Eigen::MatrixXd formOfFreeColumns(Form id, const Eigen::MatrixXd& K, const FaceGradientOperator& op);

}  // namespace nxr::forms
