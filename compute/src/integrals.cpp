#include "nxr/integrals.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::integrals {

namespace {
using V3 = Eigen::Vector3d;
V3 pt(const double* p, std::int32_t i) { return V3(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }

void check(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& field, const std::int32_t* codes) {
  if (!positions || !faces || !codes) throw std::invalid_argument("integrals: positions, faces and codes are required");
  if (field.rows() != 3 * nV) throw std::invalid_argument("integrals: a vertex field of " + std::to_string(field.rows()) + " rows for " + std::to_string(nV) + " vertices (3 a vertex: x1 y1 z1 x2 ...)");
  for (std::int64_t f = 0; f < 3 * nF; ++f) if (faces[f] < 0 || faces[f] >= nV) throw std::invalid_argument("integrals: a face indexes outside the vertices");
  if (!field.allFinite()) throw std::invalid_argument("integrals: the field has NaN or Inf (an integral leaves nothing out)");
}

// the weak form: value[tile] += -A_f grad(lambda_k) . w_f for each corner k of each face in a tile; w the face field (or its quarter turn)
TileIntegral integral(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField, const std::int32_t* codes, int nCodes, bool turn) {
  check(positions, nV, faces, nF, vertexField, codes);
  const Eigen::MatrixXd vf = faceField(positions, nV, faces, nF, vertexField);
  const Eigen::Index T = vertexField.cols();
  TileIntegral out;
  out.value = Eigen::MatrixXd::Zero(nCodes, T);
  out.area.assign(static_cast<std::size_t>(nCodes), 0.0);
  for (std::int64_t f = 0; f < nF; ++f) {
    const std::int32_t c[3] = {faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]};
    const V3 p[3] = {pt(positions, c[0]), pt(positions, c[1]), pt(positions, c[2])};
    const V3 cr = (p[1] - p[0]).cross(p[2] - p[0]);
    const double A = 0.5 * cr.norm();
    if (!(A > 0)) continue;
    const V3 n = cr / (2 * A);
    for (int k = 0; k < 3; ++k) {
      const std::int32_t t = codes[c[k]];
      if (t < 0 || t >= nCodes) continue;
      out.area[static_cast<std::size_t>(t)] += A / 3;
      const V3 g = n.cross(p[(k + 2) % 3] - p[(k + 1) % 3]) / (2 * A);       // grad lambda_k: the quarter turn of the opposite edge, over twice the area
      for (Eigen::Index j = 0; j < T; ++j) {
        V3 w(vf(3 * f, j), vf(3 * f + 1, j), vf(3 * f + 2, j));
        if (turn) w = w.cross(n);
        out.value(t, j) -= A * g.dot(w);
      }
    }
  }
  return out;
}
}  // namespace

std::vector<double> vertexAreas(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF) {
  std::vector<double> a(static_cast<std::size_t>(nV), 0.0);
  for (std::int64_t f = 0; f < nF; ++f) {
    const double A = 0.5 * (pt(positions, faces[3 * f + 1]) - pt(positions, faces[3 * f])).cross(pt(positions, faces[3 * f + 2]) - pt(positions, faces[3 * f])).norm();
    for (int k = 0; k < 3; ++k) a[static_cast<std::size_t>(faces[3 * f + k])] += A / 3;
  }
  return a;
}

Eigen::MatrixXd faceField(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField) {
  if (vertexField.rows() != 3 * nV) throw std::invalid_argument("integrals: a vertex field of " + std::to_string(vertexField.rows()) + " rows for " + std::to_string(nV) + " vertices (3 a vertex)");
  const Eigen::Index T = vertexField.cols();
  Eigen::MatrixXd out(3 * nF, T);
  for (std::int64_t f = 0; f < nF; ++f) {
    const V3 cr = (pt(positions, faces[3 * f + 1]) - pt(positions, faces[3 * f])).cross(pt(positions, faces[3 * f + 2]) - pt(positions, faces[3 * f]));
    const double len = cr.norm();
    const V3 n = len > 0 ? V3(cr / len) : V3(0, 0, 0);
    for (Eigen::Index j = 0; j < T; ++j) {
      V3 m = V3::Zero();
      for (int k = 0; k < 3; ++k) m += V3(vertexField(3 * faces[3 * f + k], j), vertexField(3 * faces[3 * f + k] + 1, j), vertexField(3 * faces[3 * f + k] + 2, j));
      m /= 3;
      m -= m.dot(n) * n;                                                    // in the face's plane
      out(3 * f, j) = m.x(); out(3 * f + 1, j) = m.y(); out(3 * f + 2, j) = m.z();
    }
  }
  return out;
}

TileIntegral flux(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField, const std::int32_t* codes, int nCodes) {
  return integral(positions, nV, faces, nF, vertexField, codes, nCodes, false);
}
TileIntegral circulation(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField, const std::int32_t* codes, int nCodes) {
  return integral(positions, nV, faces, nF, vertexField, codes, nCodes, true);
}

}  // namespace nxr::integrals
