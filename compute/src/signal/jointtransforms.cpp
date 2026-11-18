// jft / ijft: the graph Fourier transform (EigenTransform) over the vertices, the unitary DFT over time.
#include "nxr/signal/jointtransforms.h"
#include "nxr/signal/fourier.h"

namespace nxr::signal::jointtransforms {

Eigen::MatrixXcd jft(const graphtransforms::EigenTransform& t, const Eigen::MatrixXcd& F) {
  const Eigen::MatrixXcd modes = graphtransforms::gft(t, F);   // [modes x T]
  return fft(Eigen::MatrixXcd(modes.transpose()), 0, Norm::Ortho).transpose();   // time along the columns: transform the rows
}
Eigen::MatrixXcd jft(const graphtransforms::EigenTransform& t, const Eigen::MatrixXd& F) { return jft(t, Eigen::MatrixXcd(F.cast<std::complex<double>>())); }

Eigen::MatrixXcd ijft(const graphtransforms::EigenTransform& t, const Eigen::MatrixXcd& Fhat) {
  const Eigen::MatrixXcd spatial = ifft(Eigen::MatrixXcd(Fhat.transpose()), 0, Norm::Ortho).transpose();   // time along the columns again
  return graphtransforms::igft(t, spatial);
}
Eigen::MatrixXd ijftReal(const graphtransforms::EigenTransform& t, const Eigen::MatrixXcd& Fhat) { return ijft(t, Fhat).real(); }

}  // namespace nxr::signal::jointtransforms
