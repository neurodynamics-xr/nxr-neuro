// Parity of the LCMV beamformer (nxr::meeg::lcmvKernel, fixed orientation, the pseudo neural activity index) and of the region's first principal component (nxr::meeg::scoutSignFlip,
// regionComponent, regionComponentSign) with Brainstorm's release of 2018-10-12 (bst_inverse_linear_2016 and bst_scout_value 'pca' with the sign flip), on the synthetic case of
// golden/gen_lcmv_golden.m: 200 sources on a sphere, 26 MEG channels of two types, a planted source, closed-form recordings. Error = max|a-b| / max|b|.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/meeg/beamformer.h"
#include "nxr/meeg/scouts.h"

using namespace nxr::meeg;
using Eigen::MatrixXd;

namespace {
std::map<std::string, MatrixXd> readGolden(const std::string& path) {
  std::ifstream f(path);
  if (!f) { std::printf("  cannot open %s\n", path.c_str()); std::exit(1); }
  std::map<std::string, MatrixXd> out;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream h(line);
    std::string name;
    int r, c;
    h >> name >> r >> c;
    MatrixXd m(r, c);
    std::getline(f, line);
    std::istringstream v(line);
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) v >> m(i, j);
    out[name] = m;
  }
  return out;
}
double rel(const MatrixXd& a, const MatrixXd& b) { return a.rows() != b.rows() || a.cols() != b.cols() ? 1e9 : (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }

// the generator's recording, in closed form: the planted source along its orientation plus 34 sinusoids per channel (the same expression, term by term)
MatrixXd recording(const MatrixXd& G, const MatrixXd& orient, const MatrixXd& gscale, int M, int T, double fs, int p) {
  const double pi = 3.14159265358979323846;
  const Eigen::VectorXd lp = G.middleCols(3 * p, 3) * orient.row(p).transpose();
  MatrixXd X(M, T);
  for (int c = 0; c < M; ++c)
    for (int j = 0; j < T; ++j) {
      const double s = 2e-8 * std::sin(2 * pi * 11 * j / fs) * (1 + 0.5 * std::sin(2 * pi * 0.7 * j / fs));
      double b = 0;
      for (int k = 1; k <= 34; ++k) b = b + std::cos(0.37 * c * k + 0.31 * k) * std::sin(2 * pi * (2.5 + 2.9 * k) * j / fs + 0.29 * c * k + 1.3 * k);
      X(c, j) = lp(c) * s + 1e-13 * gscale(0, c) * b;
    }
  return X;
}
}  // namespace

int main() {
  const auto g = readGolden(std::string(NXR_TEST_GOLDEN_DIR) + "/lcmv_golden.txt");
  const MatrixXd& prm = g.at("params");
  const int N = static_cast<int>(prm(0, 0)), M = static_cast<int>(prm(0, 1)), T = static_cast<int>(prm(0, 2)), p = static_cast<int>(prm(0, 4));
  const int span0 = static_cast<int>(prm(0, 5)), spanN = static_cast<int>(prm(0, 6));
  const double fs = prm(0, 3);
  const MatrixXd &G = g.at("G"), &orient = g.at("orient"), &Cd = g.at("Cd"), &Cn = g.at("Cn");
  std::vector<std::string> types;
  for (int c = 0; c < M; ++c) types.push_back(g.at("types")(0, c) == 0 ? "MEG MAG" : "MEG GRAD");

  // ── the kernel, the data whitener and the noise whitener, per rule ──
  struct Variant { const char* name; TruncatedRegularisation rule; bool diagonalNoise; };
  const Variant vs[] = {{"median", TruncatedRegularisation::Median, false}, {"reg", TruncatedRegularisation::Ridge, false},
                        {"none", TruncatedRegularisation::None, false}, {"diagnoise", TruncatedRegularisation::Median, true}};
  double worstK = 0, worstW = 0;
  for (const Variant& v : vs) {
    LcmvOptions o;
    o.covariance.regularisation = v.rule;
    const MatrixXd noise = v.diagonalNoise ? MatrixXd(Cn.diagonal().asDiagonal()) : Cn;
    const LcmvKernel k = lcmvKernel(G, orient, Cd, noise, types, o);
    const std::string key = v.name;
    const double ek = rel(k.K, g.at(key + ".K")), ew = rel(k.noiseWhitener, g.at(key + ".W")), ed = rel(k.dataWhitener, g.at(key + ".DW"));
    std::printf("  Brainstorm 2018 LCMV %-10s K %.2e  noise whitener %.2e  data whitener %.2e%s\n", v.name, ek, ew, ed, k.notes.empty() ? "" : ("  (" + k.notes[0] + ")").c_str());
    worstK = std::max(worstK, ek);
    worstW = std::max({worstW, ew, ed});
    CHECK(k.K.rows() == N && k.K.cols() == M);
    if (v.diagonalNoise) CHECK(!k.notes.empty());
  }
  std::printf("  worst: kernel %.2e, whiteners %.2e\n", worstK, worstW);
  // the whiteners agree to the golden's 12 digits; the kernel divides by each source's whitened gain, which carries the data covariance's conditioning (a strong planted source
  // over a weak background) into the rounding of the eigensolvers: a few 1e-9 of its largest entry. A different rule or orientation would differ at order one.
  CHECK(worstK < 5e-8);
  CHECK(worstW < 1e-10);

  // ── the region's first principal component, over the span, from the golden's median kernel and the closed-form recording ──
  const MatrixXd X = recording(G, orient, g.at("gscale"), M, T, fs, p);
  const MatrixXd Xs = X.middleCols(span0, spanN);
  MatrixXd Xc = Xs;
  Xc.colwise() -= Xs.rowwise().mean();
  const MatrixXd C = Xc * Xc.transpose() / static_cast<double>(spanN - 1);   // the channels' covariance over the span, its mean removed
  const MatrixXd& K = g.at("median.K");
  const MatrixXd& Y = g.at("scout_pca");
  const MatrixXd& region = g.at("region");
  double worstY = 0;
  for (int r = 0; r < Y.rows(); ++r) {
    std::vector<int> iv;
    for (int i = 0; i < N; ++i) if (static_cast<int>(region(0, i)) == r) iv.push_back(i);
    MatrixXd Kr(static_cast<Eigen::Index>(iv.size()), M), Or(static_cast<Eigen::Index>(iv.size()), 3);
    for (std::size_t i = 0; i < iv.size(); ++i) { Kr.row(static_cast<Eigen::Index>(i)) = K.row(iv[i]); Or.row(static_cast<Eigen::Index>(i)) = orient.row(iv[i]); }
    const Eigen::VectorXd flip = scoutSignFlip(Or);
    const RegionComponent rc = regionComponent(Kr, C, flip);
    const Eigen::RowVectorXd z = rc.component * Xs, pk = rc.peak * Xs;
    const double s = regionComponentSign(pk.maxCoeff(), pk.minCoeff(), pk.mean(), z.maxCoeff(), z.minCoeff(), z.mean());
    const Eigen::RowVectorXd y = (s * (z.array() - z.mean()) + pk.mean()).matrix();
    const double e = rel(y, Y.row(r));
    std::printf("  region %d: %zu sources, %d flipped, peak %lld, explained %.4f, sign %+.0f: error %.2e\n", r, iv.size(), static_cast<int>((flip.array() < 0).count()),
                static_cast<long long>(rc.peakSource), rc.explained, s, e);
    worstY = std::max(worstY, e);
  }
  std::printf("  worst region signal (sign included): %.2e\n", worstY);
  CHECK(worstY < 1e-8);                                        // a wrong sign or offset would be an error of order one
  return nxrtest::finish("test_beamformer_parity");
}
