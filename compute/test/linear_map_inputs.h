// The synthetic inputs of test_psd_linear_map: a deterministic kernel (free: 3 rows a source, and fixed), no random generator.
#pragma once
#include <Eigen/Dense>
#include <cmath>
namespace lmi {
using Eigen::MatrixXd;
constexpr int kM = 7, kN = 5;
inline MatrixXd kernel(int rowsPerSource) { MatrixXd K(rowsPerSource * kN, kM); for (int r = 0; r < K.rows(); ++r) for (int c = 0; c < kM; ++c) K(r, c) = std::sin(0.7 * r + 1.3 * c + 0.2 * r * c); return K; }
}  // namespace lmi
