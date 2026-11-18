// Shared by the frames tests: a synthetic two-hemisphere ellipsoid "brain" with a registered sphere (the pre-image), in SCS (x anterior, y left, z up) or RAS.
#pragma once
#include <Eigen/Dense>
#include <vector>

#include "mesh_helpers.h"
#include "nxr/geometry/frames.h"

using Eigen::Vector3d;
using nxr::geometry::Mat3;
using nxr::geometry::vertexNormals;

struct Brain {
    std::vector<double> P, Sph;
    std::vector<int32_t> F;
    Mat3 N;
    int n = 0, nF = 0, nHalf = 0;
};
// two hemispheres (left first), each an ellipsoid with a bump, their registered sphere the pre-image; SCS: x anterior, y left, z up. `ras`: the same brain in RAS (x right, y anterior).
inline Brain brain(int level, bool ras, double bump = 0.1, const Vector3d& D = Vector3d(2.0, 1.0, 1.5)) {
    std::vector<double> U; std::vector<int32_t> F0;
    icosphere(level, U, F0);
    const int m = static_cast<int>(U.size() / 3);
    Brain b;
    b.nHalf = m; b.n = 2 * m; b.nF = 2 * static_cast<int>(F0.size() / 3);
    for (int h = 0; h < 2; ++h) {
        const double cy = h == 0 ? 1.3 : -1.3;
        for (int i = 0; i < m; ++i) {
            const Vector3d u(U[3 * i], U[3 * i + 1], U[3 * i + 2]);
            const double r = 1 + bump * std::sin(3 * u.x()) * std::cos(2 * u.z());
            Vector3d x = Vector3d(D.x() * u.x() * r, cy + D.y() * u.y() * r, D.z() * u.z() * r);
            if (ras) x = Vector3d(-x.y(), x.x(), x.z());
            for (int k = 0; k < 3; ++k) { b.P.push_back(x[k]); b.Sph.push_back(u[k]); }
        }
        for (int32_t v : F0) b.F.push_back(v + h * m);
    }
    b.N = vertexNormals(b.P.data(), b.n, b.F.data(), b.nF);
    return b;
}
