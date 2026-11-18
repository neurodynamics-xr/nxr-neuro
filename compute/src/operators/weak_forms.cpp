// nxr::operators — the weak forms of divergence and curl (weak_forms.h), the vertex-wise parts of nxr's measure.integrals.
#include "nxr/operators/weak_forms.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::operators {

namespace {
using V3 = Eigen::Vector3d;
V3 pt(const double* p, std::int32_t i) { return V3(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }

void check(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& field) {
  if (!positions || !faces) throw std::invalid_argument("weak forms: positions and faces are required");
  if (field.rows() != 3 * nV) throw std::invalid_argument("weak forms: a vertex field of " + std::to_string(field.rows()) + " rows for " + std::to_string(nV) + " vertices (3 a vertex: x1 y1 z1 x2 ...)");
  for (std::int64_t f = 0; f < 3 * nF; ++f) if (faces[f] < 0 || faces[f] >= nV) throw std::invalid_argument("weak forms: a face indexes outside the vertices");
  if (!field.allFinite()) throw std::invalid_argument("weak forms: the field has NaN or Inf (an integral leaves nothing out)");
}

Eigen::MatrixXd form(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField, bool turn) {
  check(positions, nV, faces, nF, vertexField);
  const Eigen::MatrixXd vf = tangentFaceField(positions, nV, faces, nF, vertexField);
  const Eigen::Index T = vertexField.cols();
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(nV, T);
  for (std::int64_t f = 0; f < nF; ++f) {
    const std::int32_t c[3] = {faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]};
    const V3 p[3] = {pt(positions, c[0]), pt(positions, c[1]), pt(positions, c[2])};
    const V3 cr = (p[1] - p[0]).cross(p[2] - p[0]);
    const double A = 0.5 * cr.norm();
    if (!(A > 0)) continue;
    const V3 n = cr / (2 * A);
    for (int k = 0; k < 3; ++k) {
      const V3 g = n.cross(p[(k + 2) % 3] - p[(k + 1) % 3]) / (2 * A);       // grad lambda_k: the quarter turn of the opposite edge, over twice the area
      for (Eigen::Index j = 0; j < T; ++j) {
        V3 w(vf(3 * f, j), vf(3 * f + 1, j), vf(3 * f + 2, j));
        if (turn) w = w.cross(n);
        out(c[k], j) -= A * g.dot(w);
      }
    }
  }
  return out;
}
}  // namespace

Eigen::MatrixXd tangentFaceField(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField) {
  if (vertexField.rows() != 3 * nV) throw std::invalid_argument("weak forms: a vertex field of " + std::to_string(vertexField.rows()) + " rows for " + std::to_string(nV) + " vertices (3 a vertex)");
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

Eigen::MatrixXd weakDivergence(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField) { return form(positions, nV, faces, nF, vertexField, false); }
Eigen::MatrixXd weakCurl(const double* positions, std::int64_t nV, const std::int32_t* faces, std::int64_t nF, const Eigen::MatrixXd& vertexField) { return form(positions, nV, faces, nF, vertexField, true); }

}  // namespace nxr::operators
