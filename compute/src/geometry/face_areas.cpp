// nxr::geometry: face areas.
#include "nxr/geometry/mesh.h"

#include <Eigen/Geometry>   // cross() is declared by Eigen/Core and defined here; gcc/ld reports it undefined without this include

namespace nxr::geometry {

Eigen::VectorXd faceAreas(const RowMat3& positions, const FaceMat& faces) {
    Eigen::VectorXd a(faces.rows());
    for (Eigen::Index f = 0; f < faces.rows(); ++f) {
        const Eigen::Vector3d p0 = positions.row(faces(f, 0)), p1 = positions.row(faces(f, 1)), p2 = positions.row(faces(f, 2));
        a[f] = 0.5 * (p1 - p0).cross(p2 - p0).norm();
    }
    return a;
}


}  // namespace nxr::geometry
