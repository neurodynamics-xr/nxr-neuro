#pragma once
// nxr::scalespace — SCALE SELECTION at a point: the hat level whose scale-normalised response is the largest (Lindeberg 1998, "Feature detection with automatic scale selection"; spec
// docs/specs/2026-11-14-spatial-events.md, decision 4). PURE MATHS on matrices: the power of a field's hat response at several levels, the squared L2 norm of each level's atom, numbers in, the chosen level out. The
// function never knows where the matrices came from (a recording's band power through a kernel, a planted blob), which vertices or frames they hold, or what a level is in centimetres.
//
//   the response   at an element e (a vertex) and a level l, the hat operator's response is r_l(e) = <atom_l(e), f> with atom_l(e) the operator's column at e. Its SQUARE is the power P_l(e) = |r_l(e)|² (the caller sums
//                  a free current's components, as `power` does). Normalised to the UNIT-ENERGY atom, the correlation of f with it is r_l(e) / ‖atom_l(e)‖, so the normalised power is P_l(e) / n_l(e) with
//                  n_l(e) = ‖atom_l(e)‖² the squared L2 norm the caller gives. A blob of the atom's own size gives its largest normalised power at that level, and white noise the same at every level.
//   the choice     the level l* of the largest normalised power at each element and frame (the lowest index on a tie); its value is returned beside it.
// Layout: rows are element-major, level-minor (row = e · levels + l), as `apply_kernel`'s multi-level output and `power` of it are. Pointwise in the frames (columns) and the elements: nothing is reduced across time or space;
// the maximum is over the level axis, the scale space.
#include <Eigen/Dense>

namespace nxr::scalespace {

struct LevelChoice {
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> index;   // [elements × frames] the chosen level's position in the list (0 … levels − 1)
  Eigen::MatrixXd response;                                   // [elements × frames] the largest normalised power P_l / n_l
};

/** `power` [elements · levels × frames] (≥ 0, finite), `norm2` [elements · levels] (> 0, finite: the squared L2 norm of each element's atom at each level), `levels` ≥ 1.
 *  std::invalid_argument for sizes that disagree, a norm that is not finite and above 0, or a power that is negative or not finite. */
LevelChoice selectLevel(const Eigen::Ref<const Eigen::MatrixXd>& power, const Eigen::Ref<const Eigen::VectorXd>& norm2, int levels);

}  // namespace nxr::scalespace
