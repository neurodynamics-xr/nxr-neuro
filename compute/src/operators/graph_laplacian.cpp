#include "nxr/geometry/mesh.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/operators/laplacians.h"

namespace nxr::operators {

// L = d0ᵀ d0. d0 is the signed vertex→edge incidence (exterior derivative on
// 0-forms). d0ᵀd0 gives degree on the diagonal and −1 between adjacent vertices.
Eigen::SparseMatrix<double> graphLaplacian(const geometry::SurfaceArrays& surface) {
    const Eigen::SparseMatrix<double>& D0 = d0(surface);   // passthrough accessor
    Eigen::SparseMatrix<double> L = D0.transpose() * D0;
    L.makeCompressed();
    return L;
}

} // namespace nxr::operators
