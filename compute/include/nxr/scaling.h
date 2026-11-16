#pragma once
// nxr::scaling — a DIAGONAL SCALING of a matrix's rows: general algebra, no domain. D⁻¹ X, each row i of X divided by d_i. The spatial events' `relative` step uses it to put a band's power per vertex in
// units of that vertex's own baseline power (spec docs/specs/2026-11-14-spatial-events.md, piece 2), but the function knows nothing of power, vertices or baselines: a matrix and a vector of divisors in, a matrix out.
// Pointwise: it never reduces across rows or columns, so a matrix cut into blocks of columns gives the same result block by block.
#include <Eigen/Dense>

namespace nxr::scaling {

/** Y(i, j) = X(i, j) / d(i). std::invalid_argument when d does not have X's rows, or an entry of d is not finite and above 0 (the message names the first). */
Eigen::MatrixXd divideRows(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::VectorXd>& d);

}  // namespace nxr::scaling
