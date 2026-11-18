// Shared by golden/gen_psd_linear_map_golden.cpp (the OLD meeg::sourcePower) and test_psd_linear_map.cpp (the NEW psd of a linear map): a deterministic kernel (free: 3 rows a source, and fixed) and symmetric
// cross-spectra, no random generator.
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>
namespace lmi {
using Eigen::MatrixXd;
constexpr int kM = 7, kN = 5;
inline MatrixXd kernel(int rowsPerSource) { MatrixXd K(rowsPerSource * kN, kM); for (int r = 0; r < K.rows(); ++r) for (int c = 0; c < kM; ++c) K(r, c) = std::sin(0.7 * r + 1.3 * c + 0.2 * r * c); return K; }
inline std::vector<MatrixXd> spectra() {
    std::vector<MatrixXd> S;
    for (int g = 0; g < 3; ++g) { MatrixXd A(kM, kM); for (int i = 0; i < kM; ++i) for (int j = 0; j < kM; ++j) A(i, j) = std::cos(0.9 * i - 0.4 * j + g); S.push_back(A * A.transpose()); }
    return S;
}
}  // namespace lmi
