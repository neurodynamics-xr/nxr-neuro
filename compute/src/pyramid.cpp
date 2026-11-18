// nxr::pyramid (pyramid.h).
#include "nxr/pyramid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "nxr/signal/fourier.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/core/parallel.h"

namespace nxr::pyramid {

FrameKernel frameKernel(const Eigen::MatrixXd& K3, const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2, const Eigen::MatrixXd& n, const std::vector<char>& singular, int threads) {
  const Eigen::Index V = e1.rows();
  if (e1.cols() != 3 || e2.rows() != V || e2.cols() != 3 || n.rows() != V || n.cols() != 3 || static_cast<Eigen::Index>(singular.size()) != V) throw std::invalid_argument("frameKernel: e1, e2, n are [V x 3] and singular [V]");
  if (K3.rows() != 3 * V) throw std::invalid_argument("frameKernel: kernel has " + std::to_string(K3.rows()) + " rows, expected 3 x " + std::to_string(V));
  const Eigen::Index M = K3.cols();
  FrameKernel out;
  out.K1.resize(V, M); out.K2.resize(V, M); out.Kn.resize(V, M);
  auto num = [](double x) { return std::isnan(x) ? 0.0 : x; };           // numpy nan_to_num (the frames hold NaN, never inf)
  core::forRanges(V, std::max(1, threads), [&](std::int64_t v0, std::int64_t v1) {         // each vertex writes its own rows: any split is bit-identical
  for (Eigen::Index v = v0; v < v1; ++v) {
    Eigen::Vector3d a, b, c;
    for (int k = 0; k < 3; ++k) { a[k] = num(e1(v, k)); b[k] = num(e2(v, k)); c[k] = num(n(v, k)); }
    if (singular[static_cast<std::size_t>(v)]) {
      Eigen::Vector3d nf;                                                // nxr: nan_to_num(normal) + [0, 0, 1] where the component is NaN
      for (int k = 0; k < 3; ++k) nf[k] = num(n(v, k)) + (std::isnan(n(v, k)) && k == 2 ? 1.0 : 0.0);
      const Eigen::Vector3d ref = std::abs(nf[0]) < 0.9 ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 1, 0);   // tangent_basis
      Eigen::Vector3d b1 = ref - ref.dot(nf) * nf;
      b1 /= b1.norm();
      a = b1; b = nf.cross(b1); c = nf;                                  // (b1, b2, nf) is orthonormal: the whole |J|^2, normal share included, goes to the total
    }
    out.K1.row(v) = a[0] * K3.row(3 * v) + a[1] * K3.row(3 * v + 1) + a[2] * K3.row(3 * v + 2);
    out.K2.row(v) = b[0] * K3.row(3 * v) + b[1] * K3.row(3 * v + 1) + b[2] * K3.row(3 * v + 2);
    out.Kn.row(v) = c[0] * K3.row(3 * v) + c[1] * K3.row(3 * v + 1) + c[2] * K3.row(3 * v + 2);
  }
  });
  return out;
}

}  // namespace nxr::pyramid
