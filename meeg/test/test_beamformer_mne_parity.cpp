// Parity of the LCMV beamformer under MNE-Python's conventions (nxr::meeg::rankWhitener, nxr::meeg::mneLcmvKernel) with mne.beamformer.make_lcmv itself, on the synthetic case of
// golden/gen_lcmv_mne_golden.py: 30 EEG electrodes on a four-shell sphere, 200 sources with random orientations, the average-reference projector, a data and a noise covariance.
// Every orientation (free, normal, max-power) x weight normalisation (none, unit-noise-gain, unit-noise-gain-invariant, nai) with the noise covariance, and four without it (MNE-Python's
// ad hoc identity covariance: the whitener is the projector). The kernel compared is the operator on the channels as recorded (weights x whitener). Error = max|a-b| / max|b|.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/meeg/beamformer.h"

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
bool refused(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
}  // namespace

int main() {
  const auto g = readGolden(std::string(NXR_TEST_GOLDEN_DIR) + "/lcmv_mne_golden.txt");
  const MatrixXd &G = g.at("G"), &nn = g.at("nn"), &Cd = g.at("Cd"), &Cn = g.at("Cn"), &P = g.at("P");
  const int M = static_cast<int>(G.rows()), rank = static_cast<int>(g.at("rank")(0, 0));
  const std::vector<std::string> types(static_cast<std::size_t>(M), "EEG");
  const MatrixXd& sampled = g.at("sampled");

  // ── the whiteners: of the projected noise covariance, and of the projected identity (no noise covariance) ──
  const RankWhitening wn = rankWhitener(P * Cn * P.transpose(), types, {rank}), wa = rankWhitener(P * P.transpose(), types, {rank});
  const double ewn = rel(wn.W, g.at("noise.W")), ewa = rel(wa.W, g.at("adhoc.W"));
  std::printf("  whitener (rank %d): with the noise covariance %.2e, without %.2e\n", rank, ewn, ewa);
  CHECK(ewn < 1e-10 && ewa < 1e-10);

  // ── the kernels ──
  const std::pair<const char*, LcmvOrientation> oris[] = {{"free", LcmvOrientation::Free}, {"normal", LcmvOrientation::Normal}, {"max-power", LcmvOrientation::MaxPower}};
  const std::pair<const char*, LcmvWeightNorm> norms[] = {{"none", LcmvWeightNorm::None}, {"unit-noise-gain", LcmvWeightNorm::UnitNoiseGain},
                                                          {"unit-noise-gain-invariant", LcmvWeightNorm::UnitNoiseGainInvariant}, {"nai", LcmvWeightNorm::Nai}};
  int cases = 0;
  double worst = 0, worstOri = 0;
  for (const char* noise : {"noise", "adhoc"})
    for (const auto& [oname, o] : oris)
      for (const auto& [wname, w] : norms) {
        const std::string key = std::string(noise) + "." + oname + "." + wname;
        if (!g.count(key + ".K")) continue;
        MneLcmvOptions opt;
        opt.orientation = o; opt.weightNorm = w;
        const MneLcmvKernel k = mneLcmvKernel(G, nn, Cd, std::string(noise) == "noise" ? wn.W : wa.W, rank, opt);
        const int rows = o == LcmvOrientation::Free ? 3 : 1;
        MatrixXd Ks(sampled.cols() * rows, M), Os(sampled.cols(), 3);
        for (Eigen::Index i = 0; i < sampled.cols(); ++i) {
          const Eigen::Index s = static_cast<Eigen::Index>(sampled(0, i));
          Ks.middleRows(i * rows, rows) = k.K.middleRows(s * rows, rows);
          if (o == LcmvOrientation::MaxPower) Os.row(i) = k.orientations.row(s);
        }
        const double e = rel(Ks, g.at(key + ".K"));
        const double eo = o == LcmvOrientation::MaxPower ? rel(Os, g.at(key + ".ori")) : 0.0;
        if (o == LcmvOrientation::MaxPower) std::printf("  %-42s K %.2e  orientation %.2e\n", key.c_str(), e, eo);
        else std::printf("  %-42s K %.2e\n", key.c_str(), e);
        worst = std::max(worst, e);
        worstOri = std::max(worstOri, eo);
        ++cases;
      }
  std::printf("  %d cases: worst kernel %.2e, worst max-power orientation %.2e\n", cases, worst, worstOri);
  CHECK(cases == 16);
  CHECK(worst < 1e-9);
  CHECK(worstOri < 1e-9);

  // ── refusals ──
  CHECK(refused([&] { rankWhitener(Cn, types, {M + 1}); }));                                   // a rank above the channels
  CHECK(refused([&] { rankWhitener(Cn, types, {rank, rank}); }));                             // one rank a modality
  CHECK(refused([&] { mneLcmvKernel(G, nn, Cd, wn.W, 0); }));                                 // a rank of zero
  CHECK(refused([&] { mneLcmvKernel(G, nn.topRows(3), Cd, wn.W, rank); }));                   // normals of other sources
  {   // a gain with no radial sensitivity (a spherical MEG model): MNE-Python asks for reduce_rank, not ported here
    MatrixXd Gs = G;
    Gs.col(2).setZero();
    CHECK(refused([&] { mneLcmvKernel(Gs, nn, Cd, wn.W, rank); }));
  }
  return nxrtest::finish("test_beamformer_mne_parity");
}
