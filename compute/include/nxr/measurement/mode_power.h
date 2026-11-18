#pragma once
// Each mode's band power (toolbox refactor P1: moved from hodge_weights.h, declaration unchanged).
#include <Eigen/Dense>
#include <vector>

namespace nxr::measurement {

/** Each mode's band power, [2K x bands]: diag(B S_b B^T) (the real part of S_b enters). */
Eigen::MatrixXd modePower(const Eigen::MatrixXd& B, const std::vector<Eigen::MatrixXd>& S);

}  // namespace nxr::measurement
