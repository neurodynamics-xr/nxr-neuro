// nxr::pyramid against nxr's band power (golden: test/golden/pyramid_golden.txt, gen_pyramid_golden.py; synthetic two-hemisphere mesh, kernel and sensor data):
//   - the subject tree's leaf codes and areas (nxr::select against nxr geom.trees), the power-partition gains, one band's analytic signal, and the band power per (band, leaf, frame)
//     (power, env, envmax, the good samples a frame): `nxr::measurement::reduce` over the PRODUCT of the frame of each sample and the leaf of each vertex (the joint partition), with the
//     power-partition bank on the whole record in one block (parity up to nxr's float32); the frame of a sample is the golden's (the frame grid is a selection, its tiles one of the inputs);
//   - the result does not depend on the block size or on how the record is cut into blocks (sums add, maxima merge);
//   - the power partition's band powers add to the broadband power; the Morlet shape chunked with a halo matches the same filter over the whole record.
#include "nxr/signal/wavelet.h"
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <algorithm>
#include <complex>
#include "nxr/select/dyadic.h"
#include "nxr/measurement/reduce.h"
#include "nxr/pyramid.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/select/bands.h"
#include "nxr/selection/resolved.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)
using namespace nxr;
namespace tb = nxr::signal::timefilterbank;

namespace {
using Bands = std::vector<std::pair<double, double>>;
// A block [samples x channels] and the frequencies of its spectrum (numpy's fftfreq).
struct Spec { Eigen::MatrixXd block; Eigen::VectorXd freq; double sfreq = 0; };
Spec spectrumOf(const Eigen::MatrixXd& block, double sfreq) { return {block, nxr::signal::fftfreq(static_cast<std::size_t>(block.rows()), 1.0 / sfreq), sfreq}; }
// The analytic signal of band `b` of `bands` (shape `family`): the wavelet filterbank, analytic, gain 2, DC and Nyquist dropped (the band's gain on positive frequencies, doubled; what the pyramid's analyticBand was).
Eigen::MatrixXcd bandAnalytic(const Spec& s, const Bands& bands, tb::Family family, std::size_t b) {
  tb::Design d(family, 2.0, tb::Sides::Analytic);
  d.bands = bands;
  d.edgeBins = tb::EdgeBins::Drop;
  return nxr::signal::wt(tb::design(d, static_cast<std::size_t>(s.block.rows()), s.sfreq), s.block)[b];
}
// The broadband analytic signal: positive frequencies doubled, the rest 0 (the plain FFT Hilbert).
Eigen::MatrixXcd analyticBroadband(const Spec& s) {
  Eigen::MatrixXcd Y = nxr::signal::fft(s.block.cast<std::complex<double>>());
  for (Eigen::Index k = 0; k < Y.rows(); ++k) Y.row(k) *= s.freq(k) > 0 ? 2.0 : 0.0;
  return nxr::signal::ifft(Y);
}
}  // namespace

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
double relmax(const std::vector<double>& a, const std::vector<double>& b) {   // max |a - b| / max |b| over finite entries
  double e = 0, sc = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { if (!std::isfinite(b[i])) continue; e = std::max(e, std::abs(a[i] - b[i])); sc = std::max(sc, std::abs(b[i])); }
  return e / sc;
}

namespace ms = nxr::measurement;
namespace sel = nxr::selection;

// the joint cells of the pyramid: the product of the frame of each sample (factor 0, a member number per sample of the line) and the leaf of each vertex (factor 1), code frame + frames x leaf
sel::ProductForm joint_cells(const std::vector<std::int32_t>& frame_of_sample, int frames, const std::vector<std::int32_t>& leaf_of_vertex, int leaves) {
  sel::MemberOfElement f, l;
  f.member_of = frame_of_sample; f.members = frames;
  l.member_of = leaf_of_vertex; l.members = leaves;
  sel::ProductForm c;
  c.factors = {f, l};
  c.counts = {frames, leaves};
  c.radices = {frames, leaves};
  return c;
}

// one block of J = K A (K [rows x channels], A [samples x channels]); a bad sample carries J = 0 (nxr): values [vertices x samples] row-major of the given pointwise form
Eigen::MatrixXcd project(const Eigen::MatrixXd& K, const Eigen::MatrixXcd& A, const std::uint8_t* good) {
  Eigen::MatrixXcd Ab = A;
  if (good) for (Eigen::Index s = 0; s < Ab.rows(); ++s) if (!good[s]) Ab.row(s).setZero();
  return K.cast<std::complex<double>>() * Ab.transpose();      // [vertices x samples]
}
std::vector<double> rowmajor(const Eigen::MatrixXd& m) {
  std::vector<double> v(static_cast<std::size_t>(m.size()));
  for (Eigen::Index r = 0; r < m.rows(); ++r) for (Eigen::Index c = 0; c < m.cols(); ++c) v[static_cast<std::size_t>(r * m.cols() + c)] = m(r, c);
  return v;
}

// the band's cortical-current sums over the joint cells, the record handed over in blocks of `sub` samples starting at `first`: the power (sum a |J|^2), the envelope (sum a |J|, and its max)
struct BandCells {
  ms::Ingredients power, env;
  BandCells(std::int64_t cells) : power(1, cells), env(1, cells) {}
};
void add_block(BandCells& out, const sel::ProductForm& cells, const Eigen::MatrixXd& K, const Eigen::MatrixXcd& A, const std::uint8_t* good, const double* area, std::int64_t first, std::int64_t sub) {
  ms::Request r; r.measures = {"sum", "max"}; r.weighting = ms::Weighting::given;
  const std::int64_t n = A.rows();
  for (std::int64_t s0 = 0; s0 < n; s0 += sub) {
    const std::int64_t w = std::min<std::int64_t>(sub, n - s0);
    const Eigen::MatrixXcd J = project(K, A.middleRows(s0, w), good ? good + s0 : nullptr);
    const std::vector<double> p = rowmajor(J.cwiseAbs2()), m = rowmajor(J.cwiseAbs());
    ms::reduce(out.power, p.data(), J.rows(), w, first + s0, cells, area, r);
    ms::reduce(out.env, m.data(), J.rows(), w, first + s0, cells, area, r);
  }
}

// the free-orientation tensor of a band's current in the canonical frames (nxr measure.vectors.band_tensor): the Hermitian form's ten pointwise sums, one `reduce` each, over the joint cells
const char* const kTensorKeys[10] = {"t11", "t22", "tnn", "t12r", "t12i", "t1nr", "t1ni", "t2nr", "t2ni", "total"};
struct TensorCells {
  std::vector<ms::Ingredients> sums;
  explicit TensorCells(std::int64_t cells) : sums(10, ms::Ingredients(1, cells)) {}
};
void add_tensor_block(TensorCells& out, const sel::ProductForm& cells, const pyramid::FrameKernel& fk, const Eigen::MatrixXcd& A, const std::uint8_t* good, const double* wt, const double* wtot,
                      std::int64_t first, std::int64_t sub) {
  ms::Request r; r.measures = {"sum"}; r.weighting = ms::Weighting::given;
  const std::int64_t n = A.rows();
  for (std::int64_t s0 = 0; s0 < n; s0 += sub) {
    const std::int64_t w = std::min<std::int64_t>(sub, n - s0);
    const Eigen::MatrixXcd J1 = project(fk.K1, A.middleRows(s0, w), good ? good + s0 : nullptr), J2 = project(fk.K2, A.middleRows(s0, w), good ? good + s0 : nullptr),
                           Jn = project(fk.Kn, A.middleRows(s0, w), good ? good + s0 : nullptr);
    const Eigen::MatrixXd p11 = J1.cwiseAbs2(), p22 = J2.cwiseAbs2(), pnn = Jn.cwiseAbs2();
    const Eigen::ArrayXXcd c12 = J1.array() * J2.conjugate().array(), c1n = J1.array() * Jn.conjugate().array(), c2n = J2.array() * Jn.conjugate().array();
    const Eigen::MatrixXd forms[10] = {p11, p22, pnn, c12.real().matrix(), c12.imag().matrix(), c1n.real().matrix(), c1n.imag().matrix(), c2n.real().matrix(), c2n.imag().matrix(), p11 + p22 + pnn};
    for (int k = 0; k < 10; ++k) {
      const std::vector<double> v = rowmajor(forms[k]);
      ms::reduce(out.sums[static_cast<std::size_t>(k)], v.data(), J1.rows(), w, first + s0, cells, k == 9 ? wtot : wt, r);      // the nine are weighted by the area of a vertex with a frame, the total by every vertex's
    }
  }
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
  Bands bands;
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
  const std::vector<nxr::select::BandSpec> oct = nxr::select::octave_bands();
  for (int b = 0; b < nb; ++b) EXPECT(oct[static_cast<std::size_t>(b)].lo == bands[static_cast<std::size_t>(b)].first && oct[static_cast<std::size_t>(b)].hi == bands[static_cast<std::size_t>(b)].second, "octave_bands = nxr's DEFAULT_BANDS");

  // the tree: nxr-neuro's ladder on the same mesh gives nxr's leaf codes and leaf areas
  select::Mesh mesh; mesh.n = V; mesh.positions = P; mesh.faces = F;
  const select::Tree tree = select::subjectTree(P.data(), V, F.data(), nF, depth);
  EXPECT(tree.finest == finest && tree.depth == depth, "the tree reaches nxr's finest level");
  EXPECT(tree.codes == codes, "the leaf codes are nxr's");
  const std::vector<double> a = select::vertexAreas(mesh);
  EXPECT(relmax(a, area) < 1e-12, "the vertex areas are nxr's");
  std::vector<double> wl;                                                       // the leaf areas: the member's own measure `area`, `reduce` with the vertex areas as weights
  {
    sel::MemberOfElement leaves; leaves.member_of = tree.codes; leaves.members = tree.nLeaves();
    ms::Ingredients ar(1, tree.nLeaves());
    ms::Request rq; rq.measures = {"area"}; rq.weighting = ms::Weighting::given;
    ms::reduce(ar, static_cast<const double*>(nullptr), 1, V, leaves, a.data(), rq);
    for (int l = 0; l < tree.nLeaves(); ++l) wl.push_back(ms::value(ar, 0, l, "area"));
  }
  EXPECT(static_cast<int>(wl.size()) == nLeaves && relmax(wl, w) < 1e-12, "the leaf areas w are nxr's");

  // the power-partition gains
  Eigen::VectorXd fr = Eigen::Map<const Eigen::VectorXd>(freqs.data(), static_cast<Eigen::Index>(freqs.size()));
  const Eigen::MatrixXd g = tb::bandGains(fr, bands, tb::Family::PowerPartition);
  for (int b = 0; b < nb; ++b) { std::vector<double> gb(g.cols()); for (int k = 0; k < g.cols(); ++k) gb[k] = g(b, k); EXPECT(relmax(gb, gains[b]) < 1e-12, "gain of band " << b << " is nxr's band_gains"); }
  { // squares sum to one inside [1, 64] away from the outer tapers
    Eigen::VectorXd f2(3); f2 << 3.0, 9.0, 20.0;
    const Eigen::MatrixXd g2 = tb::bandGains(f2, bands, tb::Family::PowerPartition);
    for (int k = 0; k < 3; ++k) EXPECT(std::abs(g2.col(k).squaredNorm() - 1) < 1e-12, "power partition: the squared gains sum to 1");
  }
  // whole record, one block
  Eigen::MatrixXd Kmat(V, M); for (int v = 0; v < V; ++v) for (int c = 0; c < M; ++c) Kmat(v, c) = K[static_cast<std::size_t>(v) * M + c];
  Eigen::MatrixXd block(n, M); for (int c = 0; c < M; ++c) for (int t = 0; t < n; ++t) block(t, c) = X[static_cast<std::size_t>(c) * n + t];
  const Spec spec = spectrumOf(block, sfreq);
  const Eigen::MatrixXd gn = tb::bandGains(spec.freq, bands, tb::Family::PowerPartition);
  for (const auto& an : analytic) {
    const Eigen::MatrixXcd A = bandAnalytic(spec, bands, tb::Family::PowerPartition, static_cast<std::size_t>(an.first));
    std::vector<double> re, im, gre = an.second.first, gim = an.second.second;
    for (int c = 0; c < 2; ++c) for (int t = 0; t < n; ++t) { re.push_back(A(t, c).real()); im.push_back(A(t, c).imag()); }
    EXPECT(relmax(re, gre) < 1e-5 && relmax(im, gim) < 1e-5, "analytic signal of band " << an.first << " (nxr float32): " << relmax(re, gre) << " " << relmax(im, gim));
  }

  // the band power per (band, leaf, frame): the frame of each sample is the golden's (nxr's tower code minus the first frame's)
  std::vector<std::int32_t> fidx(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) fidx[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(sampleCodes[static_cast<std::size_t>(i)] - code0);
  std::vector<std::int64_t> counts(static_cast<std::size_t>(nFrames), 0);
  for (int i = 0; i < n; ++i) if (good[static_cast<std::size_t>(i)]) ++counts[static_cast<std::size_t>(fidx[static_cast<std::size_t>(i)])];
  { std::vector<double> c(counts.begin(), counts.end()); EXPECT(relmax(c, samples) < 1e-12, "the good samples a frame are nxr's"); }
  const sel::ProductForm cells = joint_cells(fidx, nFrames, tree.codes, tree.nLeaves());
  auto run = [&](tb::Family shape, std::int64_t sub, std::vector<BandCells>& outs, std::int64_t cut) {
    outs.assign(static_cast<std::size_t>(nb), BandCells(cells.members()));
    for (int b = 0; b < nb; ++b) {
      const Eigen::MatrixXcd A = bandAnalytic(spec, bands, shape, static_cast<std::size_t>(b));
      for (std::int64_t s0 = 0; s0 < n; s0 += cut) {                  // the record handed over in pieces: frames cut by an edge are completed by the next piece
        const std::int64_t w2 = std::min<std::int64_t>(cut, n - s0);
        add_block(outs[static_cast<std::size_t>(b)], cells, Kmat, A.middleRows(s0, w2), good.data() + s0, a.data(), s0, sub);
      }
    }
  };
  std::vector<BandCells> r1, r2;
  run(tb::Family::PowerPartition, 512, r1, n);
  std::vector<double> gotp, gote, gotm;
  for (int b = 0; b < nb; ++b) for (int l = 0; l < nLeaves; ++l) for (int f = 0; f < nFrames; ++f) {
    const std::size_t at = static_cast<std::size_t>(f) + static_cast<std::size_t>(nFrames) * static_cast<std::size_t>(l);      // the product's code, the first factor (frames) fastest
    gotp.push_back(r1[static_cast<std::size_t>(b)].power.sum[at]); gote.push_back(r1[static_cast<std::size_t>(b)].env.sum[at]); gotm.push_back(r1[static_cast<std::size_t>(b)].env.max[at]);
  }
  const double ep = relmax(gotp, power), ee = relmax(gote, env), em = relmax(gotm, envmax);
  std::cout << "  power_partition vs nxr (float32): power " << ep << ", env " << ee << ", envmax " << em << "\n";
  EXPECT(ep < 2e-5 && ee < 2e-5 && em < 2e-5, "power, env, envmax per (band, leaf, frame) are nxr's");
  for (std::size_t i = 0; i < gotm.size(); ++i) if (std::isinf(envmax[i]) != std::isinf(gotm[i])) { EXPECT(false, "an empty leaf keeps envmax = -inf as nxr's"); break; }
  // block size independence
  run(tb::Family::PowerPartition, 37, r2, 211);
  double d = 0;
  for (int b = 0; b < nb; ++b) for (std::size_t i = 0; i < r1[static_cast<std::size_t>(b)].power.sum.size(); ++i) {
    const auto& x1 = r1[static_cast<std::size_t>(b)]; const auto& x2 = r2[static_cast<std::size_t>(b)];
    d = std::max({d, std::abs(x1.power.sum[i] - x2.power.sum[i]) / (1 + std::abs(x1.power.sum[i])), std::abs(x1.env.sum[i] - x2.env.sum[i]) / (1 + std::abs(x1.env.sum[i]))});
    if (std::isfinite(x1.env.max[i])) d = std::max(d, std::abs(x1.env.max[i] - x2.env.max[i]));
  }
  EXPECT(d < 1e-9, "the sums do not depend on the block size or the cut: " << d);

  // the bank partitions power: sum over bands of the band power ~ broadband (Parseval) inside the covered range; and the 0 frequency / above 64 Hz are outside it
  {
    Eigen::MatrixXd tone(n, 1);
    for (int t = 0; t < n; ++t) tone(t, 0) = std::sin(2 * M_PI * 9.3 * t / sfreq);     // a tone inside the bank
    const Spec st = spectrumOf(tone, sfreq);
    double total = 0, broad = 0;
    for (int b = 0; b < nb; ++b) total += bandAnalytic(st, bands, tb::Family::PowerPartition, static_cast<std::size_t>(b)).cwiseAbs2().sum();
    broad = analyticBroadband(st).cwiseAbs2().sum();
    EXPECT(std::abs(total / broad - 1) < 1e-3, "band powers add to broadband: " << total / broad);
  }
  // Morlet: a halo'd block (the chunk service's way) equals the whole-record filter over the samples kept
  {
    const int N = 6000;
    Eigen::MatrixXd x(N, 2);
    for (int t = 0; t < N; ++t) { x(t, 0) = std::sin(2 * M_PI * 10.7 * t / 250.0) + 0.3 * std::sin(2 * M_PI * 3.1 * t / 250.0 + 1) + 0.2 * std::cos(t * 0.37); x(t, 1) = std::cos(2 * M_PI * 6.1 * t / 250.0); }
    const Bands bd{{8.0, 16.0}};
    const Spec whole = spectrumOf(x, 250.0);
    const Eigen::MatrixXcd Aw = bandAnalytic(whole, bd, tb::Family::Morlet, 0);
    const int own0 = 2000, own1 = 3500, halo = 8 * 250;                             // 8 s of context
    const Eigen::MatrixXd part = x.middleRows(own0 - halo, own1 - own0 + 2 * halo);
    const Spec sp = spectrumOf(part, 250.0);
    const Eigen::MatrixXcd Ap = bandAnalytic(sp, bd, tb::Family::Morlet, 0);
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
    auto runT = [&](std::int64_t sub, std::int64_t cut, std::vector<TensorCells>& o) {
      o.assign(static_cast<std::size_t>(nb), TensorCells(cells.members()));
      for (int b = 0; b < nb; ++b) {
        const Eigen::MatrixXcd A = bandAnalytic(spec, bands, tb::Family::PowerPartition, static_cast<std::size_t>(b));
        for (std::int64_t s0 = 0; s0 < n; s0 += cut) add_tensor_block(o[static_cast<std::size_t>(b)], cells, fk, A.middleRows(s0, std::min<std::int64_t>(cut, n - s0)), good.data() + s0, wt.data(), wtot.data(), s0, sub);
      }
    };
    std::vector<TensorCells> t1, t2;
    runT(128, n, t1);
    double worst = 0;
    for (int k = 0; k < 10; ++k) {
      const std::string key = kTensorKeys[k];
      std::vector<double> got, want = T.at(key);
      EXPECT(!want.empty(), "the golden has tensor " << key);
      for (int b = 0; b < nb; ++b) for (int l = 0; l < nLeaves; ++l) for (int f = 0; f < nFrames; ++f) got.push_back(t1[static_cast<std::size_t>(b)].sums[static_cast<std::size_t>(k)].sum[static_cast<std::size_t>(f) + static_cast<std::size_t>(nFrames) * static_cast<std::size_t>(l)]);
      const double e = relmax(got, want);
      worst = std::max(worst, e);
      EXPECT(e < 2e-5, "tensor " << key << " is nxr's band_tensor: " << e);
    }
    std::cout << "  free-orientation tensor vs nxr (float32): worst relative error " << worst << "\n";
    runT(29, 173, t2);
    double dd = 0;
    for (int b = 0; b < nb; ++b) for (int k = 0; k < 10; ++k) { const auto& x = t1[static_cast<std::size_t>(b)].sums[static_cast<std::size_t>(k)].sum; const auto& y = t2[static_cast<std::size_t>(b)].sums[static_cast<std::size_t>(k)].sum; for (std::size_t i = 0; i < x.size(); ++i) dd = std::max(dd, std::abs(x[i] - y[i]) / (1 + std::abs(x[i]))); }
    EXPECT(dd < 1e-9, "the tensor sums do not depend on the block size or the cut: " << dd);
    // the trace: t11 + t22 + tnn is the total where no vertex lacks a frame; a singular vertex is only in the total, with its whole |J|^2 (nxr's, corrected 2026-11-14)
    double tr = 0, tot = 0;
    for (std::size_t i = 0; i < t1[2].sums[0].sum.size(); ++i) { tr += t1[2].sums[0].sum[i] + t1[2].sums[1].sum[i] + t1[2].sums[2].sum[i]; tot += t1[2].sums[9].sum[i]; }
    EXPECT(tot > tr, "a vertex with no frame is in the total only");
  }
  // refusals
  { bool threw = false; try { tb::bandFamilyOf("meyer"); } catch (const std::invalid_argument&) { threw = true; } EXPECT(threw, "an unknown shape is refused"); }
  { bool threw = false; try { tb::bandGains(fr, {{1, 2}, {3, 4}}, tb::Family::PowerPartition); } catch (const std::invalid_argument&) { threw = true; } EXPECT(threw, "non-adjacent bands are refused for a power partition"); }
  if (fails) { std::cout << fails << " failed\n"; return 1; }
  std::cout << "test_pyramid: ok\n";
  return 0;
}
