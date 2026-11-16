// nxr::integrals — a flow's net source and net rotation in each tile, as exact scalars: a port of nxr's `measure/integrals.py` (the reference).
//
//   flux         Phi = oint_{dtile} v.nu ds  = int_tile div v dA      the net source of v in the tile: its outflow
//   circulation  Gam = oint_{dtile} v.tau ds = int_tile (curl v).n dA  the net rotation of v in the tile
//
// The field is ONE TANGENT VECTOR A FACE, in its plane: a vertex field (3-D, [3 nV x T], rows x1 y1 z1 x2 ...) is the mean of each face's corners projected on the
// face's plane (its normal part is dropped: the integrals are the TANGENT flow's). Both are taken in the weak (Galerkin) form the cotan Laplacian is built on: with chi
// the tile's indicator (1 on its vertices, linear on the faces), Phi = -sum_f A_f grad(chi)_f . v_f and Gam = -sum_f A_f grad(chi)_f . (v_f x n_f). A face wholly inside the
// tile adds nothing; a face on its edge adds exactly the outflow (circulation) through its share of the tile's boundary: EXACT on any mesh, folded or not.
// ADDITIVE: a parent tile's integral is the sum of its children's; LINEAR in the field (so in time). Units: the field's unit x the positions' length unit; the area the length unit squared.
// A tile is a code per vertex (a level of the surface ladder, an ROI labelling's codes; < 0: off every tile). Pure functions of arrays (no file, no chunk, no source).
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::integrals {

struct TileIntegral {
  Eigen::MatrixXd value;               // [codes x T]: the integral of each tile (0 for a code no vertex holds)
  std::vector<double> area;            // [codes]: the tile's area (its vertices' lumped areas)
};

// the lumped vertex area a_i = sum over the faces of i of A_f / 3 (the barycentric cell)
std::vector<double> vertexAreas(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF);

// one tangent vector a face, in its plane: [3 nF x T] (rows 3f..3f+2 the face's vector) from a vertex field [3 nV x T]
Eigen::MatrixXd faceField(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField);

// Phi: the outflow of the field out of each tile (codes in [0, nCodes)), per column of the field
TileIntegral flux(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField, const std::int32_t* codes, int nCodes);
// Gam: the circulation of the field round each tile (anticlockwise seen from outside, along the face normals)
TileIntegral circulation(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField, const std::int32_t* codes, int nCodes);

}  // namespace nxr::integrals
