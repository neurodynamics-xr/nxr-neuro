#pragma once
// nxr::operators — the POINTWISE weak forms of a tangent flow's divergence and curl, per vertex, as exact scalars (nxr's `measure/integrals.py`, the reference).
//
//   weakDivergence   D_i = -sum_{f of i} A_f grad(lambda_i)_f . v_f           the flux of v out of vertex i's cell: mass_i x (div v)_i
//   weakCurl         C_i = -sum_{f of i} A_f grad(lambda_i)_f . (v_f x n_f)    the circulation of v round vertex i's cell: mass_i x ((curl v) . n)_i
//
// The field is ONE TANGENT VECTOR A FACE, in its plane: a vertex field (3-D, [3 nV x T], rows x1 y1 z1 x2 ...) is the mean of each face's corners projected on the face's plane
// (`tangentFaceField`; its normal part is dropped). A tile's flux (circulation) is NOT computed here: it is the SUM of these over the tile's vertices, a measurement over a partition
// (`nxr::measurement::reduce`, measure sum, over the tile's member-of-element), and the tile's area is the sum of the barycentric mass (`massMatrix(Barycentric)`). Additive up a tile tree and
// linear in the field; exact on any mesh, folded or not. Units: the field's unit x the positions' length unit. Pure functions of arrays (no file, no chunk, no source).
#include <Eigen/Dense>
#include <cstdint>

namespace nxr::operators {

/** one tangent vector a face, in its plane: [3 nF x T] (rows 3f..3f+2 the face's vector) from a vertex field [3 nV x T] (std::invalid_argument for a field of the wrong size) */
Eigen::MatrixXd tangentFaceField(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField);
/** [nV x T] the flux out of each vertex's cell, per column of the field (std::invalid_argument: a missing array, a face off the vertices, a field of the wrong size or with a NaN or Inf) */
Eigen::MatrixXd weakDivergence(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField);
/** [nV x T] the circulation round each vertex's cell (anticlockwise seen from outside, along the face normals), same refusals */
Eigen::MatrixXd weakCurl(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField);

}  // namespace nxr::operators
