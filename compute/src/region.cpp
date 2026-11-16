// nxr::region — collapsing a region to one signal (region.h).
#include "nxr/region.h"

#include <cmath>
#include <stdexcept>

namespace nxr::region {

Collapse collapseOf(const std::string& name) {
  if (name == "pca_flip") return Collapse::PcaFlip;
  throw std::invalid_argument("region collapse '" + name + "': pca_flip");
}

Weights collapse(Collapse how, const std::vector<int32_t>& codes, int regions, const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M, const Eigen::MatrixXd& normals) {
  (void)how;   // one option today: pca_flip
  const Eigen::Index n = A.rows();
  if (static_cast<Eigen::Index>(codes.size()) != n || normals.rows() != n || normals.cols() != 3 || M.rows() != A.cols() || M.cols() != A.cols())
    throw std::invalid_argument("region::collapse: codes [n], A [n x m], M [m x m], normals [n x 3] (n = " + std::to_string(n) + ")");
  std::vector<std::vector<Eigen::Index>> members(static_cast<std::size_t>(regions));
  for (Eigen::Index v = 0; v < n; ++v) {
    const int32_t c = codes[static_cast<std::size_t>(v)];
    if (c < 0) continue;
    if (c >= regions) throw std::invalid_argument("region::collapse: code " + std::to_string(c) + " ≥ " + std::to_string(regions) + " regions");
    members[static_cast<std::size_t>(c)].push_back(v);
  }
  Weights out;
  out.share = Eigen::VectorXd::Zero(regions);
  out.size = Eigen::VectorXi::Zero(regions);
  std::vector<Eigen::Triplet<double>> t;
  for (int r = 0; r < regions; ++r) {
    const auto& mem = members[static_cast<std::size_t>(r)];
    const Eigen::Index k = static_cast<Eigen::Index>(mem.size());
    out.size(r) = static_cast<int>(k);
    if (k == 0) continue;
    Eigen::MatrixXd Ar(k, A.cols()), Nr(k, 3);
    for (Eigen::Index i = 0; i < k; ++i) { Ar.row(i) = A.row(mem[static_cast<std::size_t>(i)]); Nr.row(i) = normals.row(mem[static_cast<std::size_t>(i)]); }
    const Eigen::MatrixXcd Arc = Ar.cast<std::complex<double>>();
    const Eigen::MatrixXd B = (Arc * M * Arc.transpose()).real();          // the region's real covariance block
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (B + B.transpose()));
    Eigen::VectorXd w = es.eigenvectors().col(k - 1);                       // the leading component (ascending order)
    const double tr = es.eigenvalues().sum();
    out.share(r) = tr > 0 ? es.eigenvalues()(k - 1) / tr : 0.0;
    // the sign: agreeing with the region's dominant orientation (MNE's flip)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Nr, Eigen::ComputeThinV);
    Eigen::Vector3d nbar = svd.matrixV().col(0);
    double majority = 0;                                                    // n̄'s own sign: the majority of the normals along it (MNE's label_sign_flip)
    for (Eigen::Index i = 0; i < k; ++i) majority += Nr.row(i).dot(nbar) >= 0 ? 1.0 : -1.0;
    if (majority < 0) nbar = -nbar;
    double agree = 0;
    for (Eigen::Index i = 0; i < k; ++i) agree += (Nr.row(i).dot(nbar) >= 0 ? 1.0 : -1.0) * w(i);
    if (agree < 0) w = -w;
    for (Eigen::Index i = 0; i < k; ++i) t.emplace_back(r, mem[static_cast<std::size_t>(i)], w(i));
  }
  out.W.resize(regions, n);
  out.W.setFromTriplets(t.begin(), t.end());
  return out;
}

}  // namespace nxr::region
