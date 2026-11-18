// nxr::stats covariance: the chunk-wise accumulator against the direct formula, for any chunking, with blocks, each divisor, and a raw second moment.
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <random>

#include "check.h"
#include "nxr/stats/covariance.h"

using namespace nxr::stats;
using Eigen::MatrixXd;

namespace {
MatrixXd direct(const std::vector<MatrixXd>& blocks, bool demean, double divisor) {
  const int c = static_cast<int>(blocks[0].rows());
  MatrixXd S = MatrixXd::Zero(c, c);
  for (const auto& b : blocks) {
    MatrixXd X = demean ? MatrixXd(b.colwise() - b.rowwise().mean()) : b;
    S += X * X.transpose();
  }
  return S / divisor;
}
}  // namespace

int main() {
  // parity with MNE-Python's raw covariance (one block, N - 1) over a window, fed in chunks of 37 samples
  {
    std::ifstream f(std::string(NXR_TEST_GOLDEN_DIR) + "/cov_golden.txt");
    CHECK(static_cast<bool>(f));
    std::map<std::string, MatrixXd> gd;
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream h(line);
      std::string nm; int r, c; h >> nm >> r >> c;
      MatrixXd m(r, c);
      std::getline(f, line);
      std::istringstream v(line);
      for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) v >> m(i, j);
      gd[nm] = m;
    }
    const MatrixXd X = gd.at("X"), ref = gd.at("C_mne");
    const int s0 = static_cast<int>(gd.at("window")(0, 0)), s1 = static_cast<int>(gd.at("window")(0, 1));
    CovarianceEstimate acc(static_cast<int>(X.rows()));
    for (int s = s0; s < s1; s += 37) acc.add(X.middleCols(s, std::min(37, s1 - s)));
    CHECK_EQ(acc.samples(), static_cast<std::int64_t>(gd.at("nsamples_mne")(0, 0)));
    const double e = (acc.covariance() - ref).cwiseAbs().maxCoeff() / ref.cwiseAbs().maxCoeff();
    std::printf("  MNE-Python raw covariance: max rel diff %.2e\n", e);
    CHECK(e < 1e-9);
  }
  std::mt19937 g(5);
  std::normal_distribution<double> nd;
  auto draw = [&](int r, int n, double dc) { MatrixXd m(r, n); for (int i = 0; i < r; ++i) for (int j = 0; j < n; ++j) m(i, j) = (i + 1) * nd(g) + dc + i; return m; };
  const MatrixXd a = draw(5, 700, 1e3), b = draw(5, 301, -50);   // large offsets: the shifted sums must hold
  // one block, any chunking
  for (int step : {1, 7, 64, 700}) {
    CovarianceEstimate acc(5);
    for (int s = 0; s < a.cols(); s += step) acc.add(a.middleCols(s, std::min<int>(step, static_cast<int>(a.cols()) - s)));
    CHECK_EQ(acc.samples(), 700);
    CHECK_EQ(acc.blocks(), 1);
    CHECK((acc.covariance() - direct({a}, true, 699)).cwiseAbs().maxCoeff() < 1e-8);
  }
  // two blocks: each block's own mean removed; N - B
  {
    CovarianceEstimate acc(5);
    acc.add(a.leftCols(300));
    acc.add(a.rightCols(400));
    acc.beginBlock();
    acc.add(b);
    CHECK_EQ(acc.blocks(), 2);
    CHECK_EQ(acc.samples(), 1001);
    CHECK((acc.covariance() - direct({a, b}, true, 999)).cwiseAbs().maxCoeff() < 1e-8);
  }
  for (auto [dv, want] : {std::pair{CovarianceDivisor::Samples, 1001.0}, std::pair{CovarianceDivisor::SamplesMinusOne, 1000.0}}) {
    CovarianceOptions o;
    o.divisor = dv;
    CovarianceEstimate acc(5, o);
    acc.add(a);
    acc.beginBlock();
    acc.add(b);
    CHECK((acc.covariance() - direct({a, b}, true, want)).cwiseAbs().maxCoeff() < 1e-8);
  }
  // no mean removal: the raw second moment
  {
    CovarianceOptions o;
    o.removeMean = false;
    o.divisor = CovarianceDivisor::Samples;
    CovarianceEstimate acc(5, o);
    acc.add(a.leftCols(10));
    acc.add(a.rightCols(690));
    CHECK((acc.covariance() - direct({a}, false, 700)).cwiseAbs().maxCoeff() < 1e-6 * 1e6);
  }
  // errors
  {
    CovarianceEstimate acc(5);
    bool threw = false;
    try { acc.add(MatrixXd::Zero(4, 3)); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { acc.covariance(); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
  }
  return nxrtest::finish("test_covariance");
}
