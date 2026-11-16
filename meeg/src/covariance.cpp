// nxr::meeg — the chunk-wise sample covariance (docs/specs/2026-11-10-neuro-forward-inverse.md, section 7).
#include "nxr/meeg/covariance.h"

#include <stdexcept>
#include <string>

namespace nxr::meeg {

CovarianceAccumulator::CovarianceAccumulator(int channels, const CovarianceOptions& opt)
    : channels_(channels), opt_(opt), sum_(Eigen::MatrixXd::Zero(channels, channels)) {
  if (channels < 1) throw std::invalid_argument("nxr::meeg: a covariance needs at least one channel");
  open_.S = Eigen::MatrixXd::Zero(channels, channels);
  open_.s = Eigen::VectorXd::Zero(channels);
}

Eigen::MatrixXd CovarianceAccumulator::closed(const Block& b) const {
  if (b.n == 0) return Eigen::MatrixXd::Zero(channels_, channels_);
  if (!opt_.removeMean) return b.S + b.pivot * b.s.transpose() + b.s * b.pivot.transpose() + static_cast<double>(b.n) * b.pivot * b.pivot.transpose();
  return b.S - (b.s * b.s.transpose()) / static_cast<double>(b.n);        // the sum about the block's mean, from the sums about the pivot
}

void CovarianceAccumulator::beginBlock() {
  if (open_.n > 0) {
    sum_ += closed(open_);
    samples_ += open_.n;
    ++closedBlocks_;
  }
  open_ = Block();
  open_.S = Eigen::MatrixXd::Zero(channels_, channels_);
  open_.s = Eigen::VectorXd::Zero(channels_);
}

void CovarianceAccumulator::add(const Eigen::MatrixXd& chunk) {
  if (chunk.rows() != channels_) throw std::invalid_argument("nxr::meeg: a chunk has " + std::to_string(chunk.rows()) + " channels, expected " + std::to_string(channels_));
  if (!chunk.allFinite()) throw std::invalid_argument("nxr::meeg: a chunk has a NaN or an infinity");
  if (chunk.cols() == 0) return;
  if (open_.n == 0) open_.pivot = chunk.col(0);
  const Eigen::MatrixXd X = chunk.colwise() - open_.pivot;
  open_.S.noalias() += X * X.transpose();
  open_.s += X.rowwise().sum();
  open_.n += chunk.cols();
}

int CovarianceAccumulator::blocks() const { return closedBlocks_ + (open_.n > 0 ? 1 : 0); }

Eigen::MatrixXd CovarianceAccumulator::covariance() const {
  const std::int64_t n = samples();
  const std::int64_t b = blocks();
  double div = 0;
  switch (opt_.divisor) {
    case CovarianceDivisor::SamplesMinusBlocks: div = static_cast<double>(n - (opt_.removeMean ? b : 0)); break;
    case CovarianceDivisor::Samples: div = static_cast<double>(n); break;
    case CovarianceDivisor::SamplesMinusOne: div = static_cast<double>(n - 1); break;
  }
  if (!(div > 0)) throw std::runtime_error("nxr::meeg: too few samples for a covariance (" + std::to_string(n) + " samples in " + std::to_string(b) + " blocks)");
  Eigen::MatrixXd C = sum_ + closed(open_);
  return 0.5 * (C + C.transpose()) / div;
}

}  // namespace nxr::meeg
