// nxr::pyramid against nxr's band power (golden: test/golden/pyramid_golden.txt, gen_pyramid_golden.py; synthetic two-hemisphere mesh, kernel and sensor data):
//   - the subject tree's leaf codes and areas (nxr::ladder against nxr geom.trees), the power-partition gains, one band's analytic signal, and the band power per (band, leaf, frame)
//     (power, env, envmax, the good samples a frame) with the power-partition bank on the whole record in one block (exact parity up to nxr's float32);
//   - the result does not depend on the block size or on how the record is cut into frames of blocks (sums add, maxima merge);
//   - the power partition's band powers add to the broadband power; the Morlet shape chunked with a halo matches the same filter over the whole record.
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/ladder.h"
#include "nxr/pyramid.h"
#include "nxr/tower.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)
using namespace nxr;

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
double relmax(const std::vector<double>& a, const std::vector<double>& b) {   // max |a - b| / max |b| over finite entries
  double e = 0, sc = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) continue; e = std::max(e, std::abs(a[i] - b[i])); sc = std::max(sc, std::abs(b[i])); }
  return e / sc;
}
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/pyramid_golden.txt");
  EXPECT(in.good(), "the golden is there");
  std::string line;
  int V = 0, nF = 0, M = 0, n = 0, depth = 0, finest = 0, nb = 0, nLeaves = 0, nFrames = 0;
  double sfreq = 0, startS = 0;
  int level = 0;
  std::int64_t code0 = 0;
  std::vector<double> P, area, K, X, freqs;
  std::vector<std::int32_t> F, codes, sampleCodes;
  std::vector<std::uint8_t> good;
  std::vector<std::vector<double>> gains;
  pyramid::Bands bands;
  std::vector<double> power, env, envmax, w, samples, are, aim;
  int analyticBand = -1;
  std::vector<int> singular;
  std::vector<double> E1, E2, EN, K3;
  std::map<std::string, std::vector<double>> T;
  std::vector<std::pair<int, std::pair<std::vector<double>, std::vector<double>>>> analytic;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string tag; s >> tag;
    if (tag == "mesh") { int m; s >> V >> nF >> m; }
    else if (tag == "P") P = nums(s);
    else if (tag == "F") { for (double v : nums(s)) F.push_back(static_cast<std::int32_t>(v)); }
    else if (tag == "tree") s >> depth >> finest;
    else if (tag == "codes") { for (double v : nums(s)) codes.push_back(static_cast<std::int32_t>(v)); }
    else if (tag == "area") area = nums(s);
    else if (tag == "kernel") s >> V >> M;
    else if (tag == "K") K = nums(s);
    else if (tag == "data") s >> M >> n >> sfreq;
    else if (tag == "X") X = nums(s);
    else if (tag == "good") { for (double v : nums(s)) good.push_back(static_cast<std::uint8_t>(v)); }
    else if (tag == "placement") s >> startS >> level >> code0 >> nFrames;
    else if (tag == "sample_codes") { for (double v : nums(s)) sampleCodes.push_back(static_cast<std::int32_t>(v)); }
    else if (tag == "bands") s >> nb;
    else if (tag == "band") { double lo, hi; s >> lo >> hi; bands.emplace_back(lo, hi); }
    else if (tag == "freqs") freqs = nums(s);
    else if (tag == "gain") gains.push_back(nums(s));
    else if (tag == "analytic") { s >> analyticBand; analytic.push_back({analyticBand, {}}); }
    else if (tag == "re") analytic.back().second.first = nums(s);
    else if (tag == "im") analytic.back().second.second = nums(s);
    else if (tag == "result") s >> nb >> nLeaves >> nFrames;
    else if (tag == "power") power = nums(s);
    else if (tag == "env") env = nums(s);
    else if (tag == "envmax") envmax = nums(s);
    else if (tag == "w") w = nums(s);
    else if (tag == "samples") samples = nums(s);
    else if (tag == "singular") { for (double v : nums(s)) singular.push_back(static_cast<int>(v)); }
    else if (tag == "E1") E1 = nums(s);
    else if (tag == "E2") E2 = nums(s);
    else if (tag == "EN") EN = nums(s);
    else if (tag == "K3") K3 = nums(s);
    else if (tag.rfind("T_", 0) == 0 && tag != "T_samples") T[tag.substr(2)] = nums(s);
  }
  EXPECT(static_cast<int>(bands.size()) == nb && nb == 6, "six octave bands");
  const pyramid::Bands oct = pyramid::octaveBands();
  for (int b = 0; b < nb; ++b) EXPECT(oct[b] == bands[b], "octaveBands = nxr's DEFAULT_BANDS");

  // the tree: nxr-neuro's ladder on the same mesh gives nxr's leaf codes and leaf areas
  ladder::Mesh mesh; mesh.n = V; mesh.positions = P; mesh.faces = F;
  const ladder::Tree tree = ladder::subjectTree(P.data(), V, F.data(), nF, depth);
  EXPECT(tree.finest == finest && tree.depth == depth, "the tree reaches nxr's finest level");
  EXPECT(tree.codes == codes, "the leaf codes are nxr's");
  const std::vector<double> a = ladder::vertexAreas(mesh);
  EXPECT(relmax(a, area) < 1e-12, "the vertex areas are nxr's");
  const std::vector<double> wl = ladder::tileAreas(tree.codes, tree.nLeaves(), a);
  EXPECT(static_cast<int>(wl.size()) == nLeaves && relmax(wl, w) < 1e-12, "the leaf areas w are nxr's");

  // the power-partition gains
  Eigen::VectorXd fr = Eigen::Map<const Eigen::VectorXd>(freqs.data(), static_cast<Eigen::Index>(freqs.size()));
  const Eigen::MatrixXd g = pyramid::bandGains(fr, bands, pyramid::Shape::PowerPartition);
  for (int b = 0; b < nb; ++b) { std::vector<double> gb(g.cols()); for (int k = 0; k < g.cols(); ++k) gb[k] = g(b, k); EXPECT(relmax(gb, gains[b]) < 1e-12, "gain of band " << b << " is nxr's band_gains"); }
  { // squares sum to one inside [1, 64] away from the outer tapers
    Eigen::VectorXd f2(3); f2 << 3.0, 9.0, 20.0;
    const Eigen::MatrixXd g2 = pyramid::bandGains(f2, bands, pyramid::Shape::PowerPartition);
    for (int k = 0; k < 3; ++k) EXPECT(std::abs(g2.col(k).squaredNorm() - 1) < 1e-12, "power partition: the squared gains sum to 1");
  }
  // whole record, one block
  Eigen::MatrixXd Kmat(V, M); for (int v = 0; v < V; ++v) for (int c = 0; c < M; ++c) Kmat(v, c) = K[static_cast<std::size_t>(v) * M + c];
  Eigen::MatrixXd block(n, M); for (int c = 0; c < M; ++c) for (int t = 0; t < n; ++t) block(t, c) = X[static_cast<std::size_t>(c) * n + t];
  const pyramid::Spectrum spec = pyramid::spectrumOf(block, sfreq);
  const Eigen::MatrixXd gn = pyramid::bandGains(spec.freq, bands, pyramid::Shape::PowerPartition);
  for (const auto& an : analytic) {
    const Eigen::MatrixXcd A = pyramid::analyticBand(spec, gn.row(an.first).transpose());
    std::vector<double> re, im, gre = an.second.first, gim = an.second.second;
    for (int c = 0; c < 2; ++c) for (int t = 0; t < n; ++t) { re.push_back(A(t, c).real()); im.push_back(A(t, c).imag()); }
    EXPECT(relmax(re, gre) < 1e-5 && relmax(im, gim) < 1e-5, "analytic signal of band " << an.first << " (nxr float32): " << relmax(re, gre) << " " << relmax(im, gim));
  }

  // the band power per (band, leaf, frame)
  const tower::Placement pl{sfreq, n, startS};
  (void)pl;
  const tower::FrameTiling ft = tower::makeFrameTiling(sfreq, n, level, startS);
  EXPECT(ft.code0 == code0 && ft.count == nFrames, "the frame tiling is nxr's (code0, count)");
  const std::vector<std::int32_t> fidx = pyramid::frameOfSamples(ft, 0, n, 0);
  for (int i = 0; i < n; ++i) if (fidx[i] != sampleCodes[i] - code0) { EXPECT(false, "frame of sample " << i); break; }
  std::vector<std::int64_t> counts(nFrames, 0);
  pyramid::addSampleCounts(ft, 0, n, good.data(), 0, counts);
  { std::vector<double> c(counts.begin(), counts.end()); EXPECT(relmax(c, samples) < 1e-12, "the good samples a frame are nxr's"); }
  auto run = [&](pyramid::Shape shape, std::int64_t sub, std::vector<partition::AnalyticSums>& outs, std::int64_t cut) {
    outs.assign(nb, partition::AnalyticSums(tree.nLeaves(), nFrames));
    const Eigen::MatrixXd gs = pyramid::bandGains(spec.freq, bands, shape);
    for (int b = 0; b < nb; ++b) {
      const Eigen::MatrixXcd A = pyramid::analyticBand(spec, gs.row(b).transpose());
      for (std::int64_t s0 = 0; s0 < n; s0 += cut) {                  // the record handed over in pieces: frames cut by an edge are completed by the next piece
        const std::int64_t w2 = std::min<std::int64_t>(cut, n - s0);
        pyramid::bandSums(outs[b], Kmat, A.middleRows(s0, w2), good.data() + s0, tree.codes.data(), a.data(), fidx.data() + s0, sub);
      }
    }
  };
  std::vector<partition::AnalyticSums> r1, r2;
  run(pyramid::Shape::PowerPartition, 512, r1, n);
  std::vector<double> gp, ge, gm, gotp, gote, gotm;
  for (int b = 0; b < nb; ++b) for (int l = 0; l < nLeaves; ++l) for (int f = 0; f < nFrames; ++f) {
    const std::size_t at = static_cast<std::size_t>(f) * nLeaves + l;
    gotp.push_back(r1[b].power[at]); gote.push_back(r1[b].env[at]); gotm.push_back(r1[b].envmax[at]);
  }
  const double ep = relmax(gotp, power), ee = relmax(gote, env), em = relmax(gotm, envmax);
  std::cout << "  power_partition vs nxr (float32): power " << ep << ", env " << ee << ", envmax " << em << "\n";
  EXPECT(ep < 2e-5 && ee < 2e-5 && em < 2e-5, "power, env, envmax per (band, leaf, frame) are nxr's");
  for (std::size_t i = 0; i < gotm.size(); ++i) if (std::isinf(envmax[i]) != std::isinf(gotm[i])) { EXPECT(false, "an empty leaf keeps envmax = -inf as nxr's"); break; }
  // block size independence
  run(pyramid::Shape::PowerPartition, 37, r2, 211);
  double d = 0;
  for (int b = 0; b < nb; ++b) for (std::size_t i = 0; i < r1[b].power.size(); ++i) {
    d = std::max({d, std::abs(r1[b].power[i] - r2[b].power[i]) / (1 + std::abs(r1[b].power[i])), std::abs(r1[b].env[i] - r2[b].env[i]) / (1 + std::abs(r1[b].env[i]))});
    if (std::isfinite(r1[b].envmax[i])) d = std::max(d, std::abs(r1[b].envmax[i] - r2[b].envmax[i]));
  }
  EXPECT(d < 1e-9, "the sums do not depend on the block size or the cut: " << d);

  // the bank partitions power: sum over bands of the band power ~ broadband (Parseval) inside the covered range; and the 0 frequency / above 64 Hz are outside it
  {
    Eigen::MatrixXd tone(n, 1);
    for (int t = 0; t < n; ++t) tone(t, 0) = std::sin(2 * M_PI * 9.3 * t / sfreq);     // a tone inside the bank
    const pyramid::Spectrum st = pyramid::spectrumOf(tone, sfreq);
    const Eigen::MatrixXd gs = pyramid::bandGains(st.freq, bands, pyramid::Shape::PowerPartition);
    double total = 0, broad = 0;
    for (int b = 0; b < nb; ++b) total += pyramid::analyticBand(st, gs.row(b).transpose()).cwiseAbs2().sum();
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(st.freq.size());
    broad = pyramid::analyticBand(st, ones).cwiseAbs2().sum();
    EXPECT(std::abs(total / broad - 1) < 1e-3, "band powers add to broadband: " << total / broad);
  }
  // Morlet: a halo'd block (the chunk service's way) equals the whole-record filter over the samples kept
  {
    const int N = 6000;
    Eigen::MatrixXd x(N, 2);
    for (int t = 0; t < N; ++t) { x(t, 0) = std::sin(2 * M_PI * 10.7 * t / 250.0) + 0.3 * std::sin(2 * M_PI * 3.1 * t / 250.0 + 1) + 0.2 * std::cos(t * 0.37); x(t, 1) = std::cos(2 * M_PI * 6.1 * t / 250.0); }
    const pyramid::Bands bd{{8.0, 16.0}};
    const pyramid::Spectrum whole = pyramid::spectrumOf(x, 250.0);
    const Eigen::MatrixXcd Aw = pyramid::analyticBand(whole, pyramid::bandGains(whole.freq, bd, pyramid::Shape::Morlet).row(0).transpose());
    const int own0 = 2000, own1 = 3500, halo = 8 * 250;                             // 8 s of context
    const Eigen::MatrixXd part = x.middleRows(own0 - halo, own1 - own0 + 2 * halo);
    const pyramid::Spectrum sp = pyramid::spectrumOf(part, 250.0);
    const Eigen::MatrixXcd Ap = pyramid::analyticBand(sp, pyramid::bandGains(sp.freq, bd, pyramid::Shape::Morlet).row(0).transpose());
    double e = 0, sc = 0;
    for (int t = 0; t < own1 - own0; ++t) for (int c = 0; c < 2; ++c) { e = std::max(e, std::abs(Ap(halo + t, c) - Aw(own0 + t, c))); sc = std::max(sc, std::abs(Aw(own0 + t, c))); }
    std::cout << "  morlet 8-16 Hz, 8 s halo vs whole record: max rel error " << e / sc << "\n";
    EXPECT(e / sc < 1e-3, "Morlet in a halo'd block matches the whole-record filter: " << e / sc);
  }
  // THE FREE-ORIENTATION TENSOR against nxr's band_tensor (the power-partition bank, the whole record in one block; synthetic frames with singular vertices, a free kernel [3V x M])
  {
    const Eigen::Index NV = V;
    Eigen::MatrixXd e1m(NV, 3), e2m(NV, 3), enm(NV, 3), K3m(3 * NV, M);
    for (int v = 0; v < NV; ++v) for (int k = 0; k < 3; ++k) { e1m(v, k) = E1[3 * v + k]; e2m(v, k) = E2[3 * v + k]; enm(v, k) = EN[3 * v + k]; }
    for (int r = 0; r < 3 * NV; ++r) for (int c = 0; c < M; ++c) K3m(r, c) = K3[static_cast<std::size_t>(r) * M + c];
    std::vector<char> sing(NV, 0);
    for (int i : singular) { sing[i] = 1; for (int k = 0; k < 3; ++k) e1m(i, k) = e2m(i, k) = enm(i, k) = std::numeric_limits<double>::quiet_NaN(); }
    const pyramid::FrameKernel fk = pyramid::frameKernel(K3m, e1m, e2m, enm, sing);
    std::vector<double> wt(NV), wtot = a;
    for (int v = 0; v < NV; ++v) wt[v] = sing[v] ? 0.0 : a[v];                 // a vertex with no frame is out of the nine sums, in the total
    const Eigen::MatrixXd gs = pyramid::bandGains(spec.freq, bands, pyramid::Shape::PowerPartition);
    auto runT = [&](std::int64_t sub, std::int64_t cut, std::vector<partition::TensorSums>& o) {
      o.assign(nb, partition::TensorSums(tree.nLeaves(), nFrames));
      for (int b = 0; b < nb; ++b) {
        const Eigen::MatrixXcd A = pyramid::analyticBand(spec, gs.row(b).transpose());
        for (std::int64_t s0 = 0; s0 < n; s0 += cut) pyramid::tensorBandSums(o[b], fk, A.middleRows(s0, std::min<std::int64_t>(cut, n - s0)), good.data() + s0, tree.codes.data(), wt.data(), wtot.data(), fidx.data() + s0, sub);
      }
    };
    std::vector<partition::TensorSums> t1, t2;
    runT(128, n, t1);
    double worst = 0;
    for (int k = 0; k < 10; ++k) {
      const std::string key = partition::TensorSums::keys()[k];
      std::vector<double> got, want = T.at(key);
      EXPECT(!want.empty(), "the golden has tensor " << key);
      for (int b = 0; b < nb; ++b) for (int l = 0; l < nLeaves; ++l) for (int f = 0; f < nFrames; ++f) got.push_back((*t1[b].arrays()[k])[static_cast<std::size_t>(f) * nLeaves + l]);
      const double e = relmax(got, want);
      worst = std::max(worst, e);
      EXPECT(e < 2e-5, "tensor " << key << " is nxr's band_tensor: " << e);
    }
    std::cout << "  free-orientation tensor vs nxr (float32): worst relative error " << worst << "\n";
    runT(29, 173, t2);
    double dd = 0;
    for (int b = 0; b < nb; ++b) for (int k = 0; k < 10; ++k) { const auto& x = *t1[b].arrays()[k]; const auto& y = *t2[b].arrays()[k]; for (std::size_t i = 0; i < x.size(); ++i) dd = std::max(dd, std::abs(x[i] - y[i]) / (1 + std::abs(x[i]))); }
    EXPECT(dd < 1e-9, "the tensor sums do not depend on the block size or the cut: " << dd);
    // the trace: t11 + t22 + tnn is the total where no vertex lacks a frame; a singular vertex is only in the total, with its whole |J|^2 (nxr's, corrected 2026-11-14)
    double tr = 0, tot = 0;
    for (std::size_t i = 0; i < t1[2].t11.size(); ++i) { tr += t1[2].t11[i] + t1[2].t22[i] + t1[2].tnn[i]; tot += t1[2].total[i]; }
    EXPECT(tot > tr, "a vertex with no frame is in the total only");
  }
  // refusals
  { bool threw = false; try { pyramid::shapeOf("meyer"); } catch (const std::invalid_argument&) { threw = true; } EXPECT(threw, "an unknown shape is refused"); }
  { bool threw = false; try { pyramid::bandGains(fr, {{1, 2}, {3, 4}}, pyramid::Shape::PowerPartition); } catch (const std::invalid_argument&) { threw = true; } EXPECT(threw, "non-adjacent bands are refused for a power partition"); }
  if (fails) { std::cout << fails << " failed\n"; return 1; }
  std::cout << "test_pyramid: ok\n";
  return 0;
}
