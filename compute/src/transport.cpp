// nxr::transport — carrying a vector along the cortex (transport.h).
#include "nxr/transport.h"

#include <cmath>
#include <stdexcept>

namespace nxr::transport {

Transport transportOf(const std::string& name) {
  if (name == "levi_civita") return Transport::LeviCivita;
  if (name == "frames") return Transport::Frames;
  throw std::invalid_argument("transport '" + name + "': levi_civita or frames");
}

std::vector<Eigen::Matrix3d> leviCivita(const Eigen::MatrixXd& normals, const Edges& edges) {
  if (normals.cols() != 3) throw std::invalid_argument("leviCivita: normals are [n x 3]");
  std::vector<Eigen::Matrix3d> R(static_cast<std::size_t>(edges.rows()));
  for (Eigen::Index e = 0; e < edges.rows(); ++e) {
    const Eigen::Index u = edges(e, 0), v = edges(e, 1);
    if (u < 0 || v < 0 || u >= normals.rows() || v >= normals.rows()) throw std::invalid_argument("leviCivita: edge " + std::to_string(e) + " outside the vertices");
    const Eigen::Vector3d a = normals.row(v).normalized(), b = normals.row(u).normalized();
    const Eigen::Vector3d k = a.cross(b);
    const double s = k.norm(), c = a.dot(b);
    Eigen::Matrix3d Rot;
    if (s < 1e-12) {
      if (c > 0) Rot.setIdentity();
      else {                                                              // antiparallel: a half-turn about an axis perpendicular to a
        Eigen::Vector3d p = std::abs(a.x()) < 0.9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
        p = (p - p.dot(a) * a).normalized();
        Rot = 2 * p * p.transpose() - Eigen::Matrix3d::Identity();
      }
    } else {                                                              // Rodrigues: about k/s by the angle with cos c, sin s
      const Eigen::Vector3d ax = k / s;
      Eigen::Matrix3d K;
      K << 0, -ax.z(), ax.y(), ax.z(), 0, -ax.x(), -ax.y(), ax.x(), 0;
      Rot = Eigen::Matrix3d::Identity() + s * K + (1 - c) * K * K;
    }
    R[static_cast<std::size_t>(e)] = Rot;
  }
  return R;
}

std::vector<Eigen::Matrix3d> frames(const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2, const Eigen::MatrixXd& normals, const Edges& edges) {
  if (e1.cols() != 3 || e2.cols() != 3 || normals.cols() != 3 || e1.rows() != normals.rows() || e2.rows() != normals.rows())
    throw std::invalid_argument("frames: e1, e2, normals are [n x 3] each");
  auto F = [&](Eigen::Index v) { Eigen::Matrix3d f; f.col(0) = e1.row(v).transpose(); f.col(1) = e2.row(v).transpose(); f.col(2) = normals.row(v).transpose(); return f; };
  std::vector<Eigen::Matrix3d> R(static_cast<std::size_t>(edges.rows()));
  for (Eigen::Index e = 0; e < edges.rows(); ++e) {
    const Eigen::Index u = edges(e, 0), v = edges(e, 1);
    if (u < 0 || v < 0 || u >= normals.rows() || v >= normals.rows()) throw std::invalid_argument("frames: edge " + std::to_string(e) + " outside the vertices");
    R[static_cast<std::size_t>(e)] = F(u) * F(v).transpose();
  }
  return R;
}

Compared compare(const std::vector<Eigen::MatrixXcd>& T, const std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& Tuu, const Eigen::VectorXd& Tvv) {
  const std::size_t E = T.size();
  if (R.size() != E || static_cast<std::size_t>(Tuu.size()) != E || static_cast<std::size_t>(Tvv.size()) != E)
    throw std::invalid_argument("compare: " + std::to_string(E) + " blocks, " + std::to_string(R.size()) + " rotations, " + std::to_string(Tuu.size()) + " / " +
                                std::to_string(Tvv.size()) + " powers");
  Compared c;
  c.S.resize(E);
  c.lag.resize(static_cast<Eigen::Index>(E));
  c.strength.resize(static_cast<Eigen::Index>(E));
  for (std::size_t e = 0; e < E; ++e) {
    if (T[e].rows() != 3 || T[e].cols() != 3) throw std::invalid_argument("compare: block " + std::to_string(e) + " is not 3 x 3");
    const std::complex<double> s = (T[e] * R[e].transpose().cast<std::complex<double>>()).trace();
    c.S[e] = s;
    c.lag(static_cast<Eigen::Index>(e)) = std::arg(s);
    const double d = std::sqrt(std::max(Tuu(static_cast<Eigen::Index>(e)), 0.0) * std::max(Tvv(static_cast<Eigen::Index>(e)), 0.0));
    c.strength(static_cast<Eigen::Index>(e)) = d > 0 ? std::abs(s) / d : 0.0;
  }
  return c;
}

}  // namespace nxr::transport
