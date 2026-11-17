#include "nxr/formats/geometry.h"

#include <cmath>
#include <utility>

#include "nxr/formats/errors.h"

namespace nxr::formats::geom {

Point apply(const Matrix4& m, const Point& p) {
  Point o;
  for (int r = 0; r < 3; ++r) o[static_cast<std::size_t>(r)] = m[static_cast<std::size_t>(r * 4)] * p[0] + m[static_cast<std::size_t>(r * 4 + 1)] * p[1] + m[static_cast<std::size_t>(r * 4 + 2)] * p[2] + m[static_cast<std::size_t>(r * 4 + 3)];
  return o;
}

Matrix4 multiply(const Matrix4& a, const Matrix4& b) {
  Matrix4 o{};
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += a[static_cast<std::size_t>(r * 4 + k)] * b[static_cast<std::size_t>(k * 4 + c)];
      o[static_cast<std::size_t>(r * 4 + c)] = s;
    }
  return o;
}

Matrix4 invert(const Matrix4& m) {
  double a[4][8];
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 8; ++c) a[r][c] = c < 4 ? m[static_cast<std::size_t>(r * 4 + c)] : (c - 4 == r ? 1.0 : 0.0);
  for (int col = 0; col < 4; ++col) {
    int piv = col;
    for (int r = col + 1; r < 4; ++r) if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
    if (std::fabs(a[piv][col]) < 1e-300) throw format_error("a coordinate transform is singular");
    if (piv != col) for (int c = 0; c < 8; ++c) std::swap(a[piv][c], a[col][c]);
    const double d = a[col][col];
    for (int c = 0; c < 8; ++c) a[col][c] /= d;
    for (int r = 0; r < 4; ++r) {
      if (r == col) continue;
      const double f = a[r][col];
      for (int c = 0; c < 8; ++c) a[r][c] -= f * a[col][c];
    }
  }
  Matrix4 o{};
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) o[static_cast<std::size_t>(r * 4 + c)] = a[r][c + 4];
  return o;
}

Matrix4 neuromag_head_from_landmarks(const Point& nas, const Point& lpa, const Point& rpa) {
  auto sub = [](const Point& a, const Point& b) { return Point{a[0] - b[0], a[1] - b[1], a[2] - b[2]}; };
  auto dot = [](const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  auto unit = [&](Point a) { const double n = std::sqrt(dot(a, a)); if (n == 0) throw format_error("the landmarks coincide"); for (double& x : a) x /= n; return a; };
  auto cross = [](const Point& a, const Point& b) { return Point{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; };
  const auto right = unit(sub(rpa, lpa));
  const double t = dot(sub(nas, lpa), right);
  const Point origin{lpa[0] + t * right[0], lpa[1] + t * right[1], lpa[2] + t * right[2]};
  const auto anterior = unit(sub(nas, origin));
  const auto superior = unit(cross(right, anterior));
  Matrix4 m{};
  const Point* rows[3] = {&right, &anterior, &superior};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) m[static_cast<std::size_t>(r * 4 + c)] = (*rows[r])[static_cast<std::size_t>(c)];
    m[static_cast<std::size_t>(r * 4 + 3)] = -dot(*rows[r], origin);
  }
  m[15] = 1.0;
  return m;
}

Matrix4 fit_rigid(const std::vector<Point>& from, const std::vector<Point>& to) {
  if (from.size() != to.size() || from.size() < 3) throw format_error("a rigid fit needs at least three matched points");
  const std::size_t n = from.size();
  Point cf{}, ct{};
  for (std::size_t i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) { cf[static_cast<std::size_t>(k)] += from[i][static_cast<std::size_t>(k)] / static_cast<double>(n); ct[static_cast<std::size_t>(k)] += to[i][static_cast<std::size_t>(k)] / static_cast<double>(n); }
  double S[3][3] = {};
  for (std::size_t i = 0; i < n; ++i)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) S[a][b] += (from[i][static_cast<std::size_t>(a)] - cf[static_cast<std::size_t>(a)]) * (to[i][static_cast<std::size_t>(b)] - ct[static_cast<std::size_t>(b)]);
  // Horn's symmetric 4 x 4 matrix; its eigenvector of the largest eigenvalue is the rotation's unit quaternion
  double N[4][4] = {
      {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
      {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
      {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
      {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
  double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int sweep = 0; sweep < 100; ++sweep) {   // cyclic Jacobi
    double off = 0;
    for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) off += N[p][q] * N[p][q];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) {
        if (std::fabs(N[p][q]) < 1e-300) continue;
        const double theta = (N[q][q] - N[p][p]) / (2.0 * N[p][q]);
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int k = 0; k < 4; ++k) { const double a = N[k][p], b = N[k][q]; N[k][p] = c * a - s * b; N[k][q] = s * a + c * b; }
        for (int k = 0; k < 4; ++k) { const double a = N[p][k], b = N[q][k]; N[p][k] = c * a - s * b; N[q][k] = s * a + c * b; }
        for (int k = 0; k < 4; ++k) { const double a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
      }
  }
  int best = 0;
  for (int k = 1; k < 4; ++k) if (N[k][k] > N[best][best]) best = k;
  double q0 = V[0][best], q1 = V[1][best], q2 = V[2][best], q3 = V[3][best];
  const double qn = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  q0 /= qn; q1 /= qn; q2 /= qn; q3 /= qn;
  double R[3][3] = {
      {q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)},
      {2 * (q2 * q1 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q0 * q1)},
      {2 * (q3 * q1 - q0 * q2), 2 * (q3 * q2 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3}};
  Matrix4 m{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) m[static_cast<std::size_t>(r * 4 + c)] = R[r][c];
    m[static_cast<std::size_t>(r * 4 + 3)] = ct[static_cast<std::size_t>(r)] - (R[r][0] * cf[0] + R[r][1] * cf[1] + R[r][2] * cf[2]);
  }
  m[15] = 1.0;
  return m;
}


Point rotate(const Matrix4& m, const Point& v) {
  Point o;
  for (int r = 0; r < 3; ++r) o[static_cast<std::size_t>(r)] = m[static_cast<std::size_t>(r * 4)] * v[0] + m[static_cast<std::size_t>(r * 4 + 1)] * v[1] + m[static_cast<std::size_t>(r * 4 + 2)] * v[2];
  return o;
}

Matrix4 scs_from_landmarks(const Point& nas, const Point& lpa, const Point& rpa) {
  auto sub = [](const Point& a, const Point& b) { return Point{a[0] - b[0], a[1] - b[1], a[2] - b[2]}; };
  auto dot = [](const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  auto unit = [&](Point a) { const double n = std::sqrt(dot(a, a)); if (n == 0) throw format_error("the landmarks coincide"); for (double& x : a) x /= n; return a; };
  auto cross = [](const Point& a, const Point& b) { return Point{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; };
  const Point origin{(lpa[0] + rpa[0]) / 2, (lpa[1] + rpa[1]) / 2, (lpa[2] + rpa[2]) / 2};
  const Point x = unit(sub(nas, origin));
  Point y = sub(lpa, rpa);
  const double d = dot(y, x);
  for (int k = 0; k < 3; ++k) y[static_cast<std::size_t>(k)] -= d * x[static_cast<std::size_t>(k)];
  y = unit(y);
  const Point z = unit(cross(x, y));
  Matrix4 m{};
  const Point* rows[3] = {&x, &y, &z};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) m[static_cast<std::size_t>(r * 4 + c)] = (*rows[r])[static_cast<std::size_t>(c)];
    m[static_cast<std::size_t>(r * 4 + 3)] = -dot(*rows[r], origin);
  }
  m[15] = 1.0;
  return m;
}

}  // namespace nxr::formats::geom
