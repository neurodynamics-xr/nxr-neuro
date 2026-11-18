// nxr::cores::corePatches against nxr's `_core_patches` (golden: test/golden/cores_golden.txt, gen_cores_golden.py; a synthetic icosphere cortex with planted blobs of both senses): the same cores, in the same order
// (frame, sense, lowest vertex), with the same centres (the |value|·area-weighted centroid) and areas; the centre vertex and the peak (nxr-neuro's own attributes) as the generator derives them.
// The threshold is read off nxr::histogram (the project's 4096 log bins) folded over the frames: it does not depend on how the frames are cut into blocks, and is within one bin of numpy's exact percentile.
// The cores do not depend on the block of frames either: any cut into blocks, each given its first frame, gives the same list. Hand-made graphs: a patch is the connected above-threshold vertices of ONE sense (two
// touching patches of opposite senses are two cores; two of the same sense joined by a vertex above threshold are one); refusals.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/select/cores.h"
#include "measurement_helpers.h"
#include "nxr/histogram.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
template <class F> bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
bool same(const nxr::cores::Cores& a, const nxr::cores::Cores& b) {
  return a.frame == b.frame && a.sense == b.sense && a.centre_vertex == b.centre_vertex && a.area == b.area && a.peak == b.peak && a.centre.rows() == b.centre.rows() && (a.centre.rows() == 0 || a.centre == b.centre);
}
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/cores_golden.txt");
  if (!in) { std::cout << "no golden\n"; return 1; }
  int nv = 0, nf = 0, frames = 0, ncores = 0;
  double thr = 0;
  std::vector<double> pos, area, map, centre, carea, peak;
  std::vector<int> faces, frame, sense, cvert;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag;
    s >> tag;
    if (tag == "nv") { std::string a, b, c; s >> nv >> a >> nf >> b >> frames >> c >> ncores; }
    else if (tag == "positions") pos = nums(s);
    else if (tag == "faces") { for (double f : nums(s)) faces.push_back(static_cast<int>(f)); }
    else if (tag == "area") area = nums(s);
    else if (tag == "map") map = nums(s);
    else if (tag == "threshold") s >> thr;
    else if (tag == "frame") { for (double f : nums(s)) frame.push_back(static_cast<int>(f)); }
    else if (tag == "sense") { for (double f : nums(s)) sense.push_back(static_cast<int>(f)); }
    else if (tag == "centre") centre = nums(s);
    else if (tag == "core_area") carea = nums(s);
    else if (tag == "centre_vertex") { for (double f : nums(s)) cvert.push_back(static_cast<int>(f)); }
    else if (tag == "peak") peak = nums(s);
  }
  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
  for (int f = 0; f < nf; ++f) for (int k = 0; k < 3; ++k) pairs.emplace_back(faces[static_cast<std::size_t>(3 * f + k)], faces[static_cast<std::size_t>(3 * f + (k + 1) % 3)]);
  const Eigen::MatrixXd X = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(map.data(), nv, frames);
  const Eigen::MatrixXd P = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(pos.data(), nv, 3);
  const Eigen::VectorXd A = Eigen::Map<const Eigen::VectorXd>(area.data(), nv);
  const nxr::cores::Cores c = nxr::cores::corePatches(X, pairs, A, P, thr);
  bool ok = c.size() == ncores;
  double dc = 0, da = 0, dp = 0;
  for (std::int64_t k = 0; k < c.size() && ok; ++k) {
    ok = c.frame[static_cast<std::size_t>(k)] == frame[static_cast<std::size_t>(k)] && c.sense[static_cast<std::size_t>(k)] == sense[static_cast<std::size_t>(k)] && c.centre_vertex[static_cast<std::size_t>(k)] == cvert[static_cast<std::size_t>(k)];
    for (int d = 0; d < 3; ++d) dc = std::max(dc, std::abs(c.centre(k, d) - centre[static_cast<std::size_t>(3 * k + d)]));
    da = std::max(da, std::abs(c.area[static_cast<std::size_t>(k)] - carea[static_cast<std::size_t>(k)]) / carea[static_cast<std::size_t>(k)]);
    dp = std::max(dp, std::abs(c.peak[static_cast<std::size_t>(k)] - peak[static_cast<std::size_t>(k)]));
  }
  std::printf("  vs nxr _core_patches: %lld cores (nxr %d); centre max %.1e mm, area %.1e relative, peak %.1e\n", static_cast<long long>(c.size()), ncores, dc, da, dp);
  EXPECT(ok && dc < 1e-9 && da < 1e-12 && dp < 1e-12, "the cores differ from nxr's");
  EXPECT(ncores >= 10, "the golden holds cores of both senses");
  // the cores do not depend on how the frames are cut into blocks
  for (const int w : {1, 3, 5, frames}) {
    nxr::cores::Cores all;
    for (int f0 = 0; f0 < frames; f0 += w) nxr::cores::append(all, nxr::cores::corePatches(X.middleCols(f0, std::min(w, frames - f0)), pairs, A, P, thr, f0));
    EXPECT(same(all, c), "blocks of " + std::to_string(w) + " frames give another list");
  }
  // the threshold: nxr::histogram's 4096 log bins over the frames, edges fixed once from the first frames (the reference's first chunk): any cut into blocks gives the same counts and threshold, within one bin of numpy's
  {
    const int first = 3;
    std::vector<float> a(static_cast<std::size_t>(nv) * frames);
    for (int v = 0; v < nv; ++v) for (int f = 0; f < frames; ++f) a[static_cast<std::size_t>(v) * frames + f] = static_cast<float>(std::abs(X(v, f)));
    float mx = 0;
    for (int v = 0; v < nv; ++v) for (int f = 0; f < first; ++f) mx = std::max(mx, a[static_cast<std::size_t>(v) * frames + f]);
    const nxr::histogram::LogBins bins = nxr::histogram::logBinsFor(mx);
    double ref = 0;
    for (const int w : {1, 3, 8}) {
      std::vector<std::int64_t> total(static_cast<std::size_t>(bins.bins), 0);
      for (int f0 = 0; f0 < frames; f0 += w) {
        const int n = std::min(w, frames - f0);
        std::vector<float> blk(static_cast<std::size_t>(nv) * n);
        for (int v = 0; v < nv; ++v) for (int f = 0; f < n; ++f) blk[static_cast<std::size_t>(v) * n + f] = a[static_cast<std::size_t>(v) * frames + f0 + f];
        const nxr::histogram::Counts h = histogramTiles(blk.data(), static_cast<std::size_t>(nv), static_cast<std::size_t>(n), static_cast<std::size_t>(n), 1, bins);
        for (int b = 0; b < bins.bins; ++b) total[static_cast<std::size_t>(b)] += h.counts[static_cast<std::size_t>(b)];
      }
      const double t = nxr::histogram::percentile(total, bins, 98);
      if (w == 1) ref = t;
      EXPECT(t == ref, "the threshold depends on the block of frames (" + std::to_string(w) + ")");
      EXPECT(t >= thr * (1 - 1e-6) && t <= thr * 1.0036, "the histogram threshold is within one bin of the exact percentile");
    }
    std::printf("  threshold: histogram %.5g, exact %.5g (one bin = 0.34 %%)\n", ref, thr);
  }
  // hand-made graphs: a path of 7 vertices 0-1-2-3-4-5-6, threshold 1
  {
    std::vector<std::pair<std::int32_t, std::int32_t>> path;
    for (int i = 0; i < 6; ++i) path.emplace_back(i, i + 1);
    Eigen::MatrixXd Q(7, 1);
    Q << 2, 3, -2, -4, 0.5, 1.5, 1;                                  // +{0,1}, −{2,3}, none at 4, +{5,6}: opposite senses touch at 1|2 and stay two cores
    Eigen::VectorXd w = Eigen::VectorXd::Ones(7);
    Eigen::MatrixXd pos7(7, 3);
    for (int i = 0; i < 7; ++i) pos7.row(i) << i, 0, 0;
    const nxr::cores::Cores h = nxr::cores::corePatches(Q, path, w, pos7, 1.0);
    EXPECT(h.size() == 3 && h.sense[0] == 1 && h.sense[1] == 1 && h.sense[2] == -1 && h.frame[0] == 0, "a patch is one sense (+1 patches first, by lowest vertex, then −1): two touching opposite patches are two cores");
    EXPECT(std::abs(h.centre(0, 0) - (0 * 2 + 1 * 3) / 5.0) < 1e-12 && h.centre_vertex[0] == 1 && std::abs(h.area[0] - 2) < 1e-12 && std::abs(h.peak[0] - 3) < 1e-12, "the centre is the |value|-weighted centroid");
    EXPECT(h.centre_vertex[1] == 5 && std::abs(h.peak[1] - 1.5) < 1e-12 && h.centre_vertex[2] == 3 && std::abs(h.peak[2] - 4) < 1e-12 && std::abs(h.area[2] - 2) < 1e-12, "centre vertex and peak");
    Q(4, 0) = 2;                                                      // 4 above and of the sense of {5,6}: they join; 3 is of the other sense
    const nxr::cores::Cores j = nxr::cores::corePatches(Q, path, w, pos7, 1.0);
    EXPECT(j.size() == 3 && j.area[1] == 3, "a vertex above threshold joins the same-sense patches it touches");
    EXPECT(nxr::cores::corePatches(Q, path, w, pos7, 10.0).size() == 0, "nothing above the threshold, no cores");
    EXPECT(throws([&] { nxr::cores::corePatches(Q, path, w, pos7, 0.0); }) && throws([&] { nxr::cores::corePatches(Q, path, Eigen::VectorXd::Ones(6), pos7, 1.0); }) &&
           throws([&] { std::vector<std::pair<std::int32_t, std::int32_t>> bad{{0, 9}}; nxr::cores::corePatches(Q, bad, w, pos7, 1.0); }) &&
           throws([&] { Eigen::MatrixXd N = Q; N(0, 0) = std::nan(""); nxr::cores::corePatches(N, path, w, pos7, 1.0); }), "refusals");
  }
  std::printf("%s test_cores\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
