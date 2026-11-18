// nxr::stats — the sample covariance of channels, accumulated chunk by chunk (docs/specs/2026-11-10-neuro-forward-inverse.md, section 7).
// Pure: arrays in, arrays out. Depends on Eigen only; the caller reads the recording and hands over its chunks. An ESTIMATE in the sense of `measurement/estimate.h`: a mean over a family of
// segments (the blocks), folded chunk by chunk; `cov` is the one-shot form (MATLAB / NumPy `cov`: one block of all the samples).
//
//   C = [ sum over blocks b ( sum_t (x_t - m_b)(x_t - m_b)^T ) ] / divisor,     m_b = the mean of block b
//
// A BLOCK is a stretch of samples whose own mean (DC offset) is removed, as the reference removes it per data file. The sums are kept per block as shifted
// sums, so a chunk is seen once and nothing of the recording is held.
#pragma once
#include <Eigen/Dense>
#include <cstdint>

namespace nxr::stats {

// What the accumulated sum of squares is divided by.
enum class CovarianceDivisor {
  SamplesMinusBlocks,   // N - B: the unbiased estimate when each of B blocks has its own mean removed (one block: N - 1, MNE-Python's)
  Samples,              // N
  SamplesMinusOne       // N - 1 whatever the number of blocks
};
struct CovarianceOptions {
  bool removeMean = true;                                          // remove each block's mean (the reference's 'remove DC offset' per file); false: raw second moment
  CovarianceDivisor divisor = CovarianceDivisor::SamplesMinusBlocks;
};

class CovarianceEstimate {
 public:
  explicit CovarianceEstimate(int channels, const CovarianceOptions& opt = {});
  // Start a new block (the first one starts itself). Closes the one open.
  void beginBlock();
  // Add a chunk [channels x samples] to the open block. std::invalid_argument for the wrong number of rows or a NaN.
  void add(const Eigen::MatrixXd& chunk);
  // The covariance [channels x channels] of everything added (the open block is counted, not closed). std::runtime_error when the divisor is not positive.
  Eigen::MatrixXd covariance() const;
  std::int64_t samples() const { return samples_ + open_.n; }      // N: the samples recorded in the estimate
  int blocks() const;                                              // B: the blocks with at least one sample
  const CovarianceOptions& options() const { return opt_; }

 private:
  struct Block {
    Eigen::MatrixXd S;      // sum of (x - pivot)(x - pivot)^T
    Eigen::VectorXd s;      // sum of (x - pivot)
    Eigen::VectorXd pivot;  // the block's first sample
    std::int64_t n = 0;
  };
  Eigen::MatrixXd closed(const Block& b) const;
  int channels_;
  CovarianceOptions opt_;
  Eigen::MatrixXd sum_;                // the closed blocks' sum of squares
  std::int64_t samples_ = 0;
  int closedBlocks_ = 0;
  Block open_;
};

/** The covariance [channels x channels] of x [channels x samples] as ONE block (MATLAB / NumPy `cov`, observations in the columns): the estimate fed the whole matrix. */
Eigen::MatrixXd cov(const Eigen::MatrixXd& x, const CovarianceOptions& opt = {});

}  // namespace nxr::stats
