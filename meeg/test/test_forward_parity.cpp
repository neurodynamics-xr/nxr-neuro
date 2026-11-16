// Parity of nxr::meeg's sphere forward models with Brainstorm's (bst_meg_sph, bst_os, bst_eeg_sph, bst_berg) and MNE-Python's (_sphere_field, _eeg_spherepot_coil)
// on the synthetic case of golden/gen_forward_golden.py. Error = max|a-b| / max|b|.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

#include "check.h"
#include "nxr/meeg/forward.h"

using namespace nxr::meeg;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

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
}  // namespace

int main() {
  const auto g = readGolden(std::string(NXR_TEST_GOLDEN_DIR) + "/forward_golden.txt");
  Coils meg;
  meg.position = g.at("coil_pos");
  meg.orientation = g.at("coil_ori");
  meg.weight = g.at("coil_w").col(0);
  for (int i = 0; i < g.at("coil_ch").rows(); ++i) meg.channel.push_back(static_cast<int>(g.at("coil_ch")(i, 0)));
  meg.channels = meg.channel.back() + 1;
  const MatrixXd src = g.at("src"), srcOut = g.at("src_out");
  const Vector3d c0 = g.at("c0").row(0).transpose(), ec = g.at("ec").row(0).transpose();

  // ---- MEG, single sphere
  const double eBstS = rel(sarvasGain(src, meg, c0), g.at("meg_single.bst")), eMneS = rel(sarvasGain(src, meg, c0), g.at("meg_single.mne"));
  const double eBstO = rel(sarvasGain(srcOut, meg, c0), g.at("meg_single_out.bst")), eMneO = rel(sarvasGain(srcOut, meg, c0), g.at("meg_single_out.mne"));
  std::printf("  MEG single sphere: Brainstorm %.2e (sources outside %.2e), MNE-Python %.2e (outside %.2e)\n", eBstS, eBstO, eMneS, eMneO);
  CHECK(eBstS < 1e-10 && eBstO < 1e-10 && eMneS < 1e-10 && eMneO < 1e-10);

  // ---- MEG, overlapping spheres: the gain from the reference's own spheres (exact), then from ours
  std::vector<Vector3d> bstCentres;
  for (int i = 0; i < g.at("os_centre.bst").rows(); ++i) bstCentres.push_back(g.at("os_centre.bst").row(i).transpose());
  const double eOsGiven = rel(sarvasGain(src, meg, bstCentres), g.at("meg_os.bst"));
  const MatrixXd inner = g.at("inner");
  const std::vector<Sphere> ours = sensorSpheres(inner, meg.position, meg.weight, meg.channel, meg.channels);
  std::vector<Vector3d> ourCentres;
  double dc = 0, dr = 0;
  for (int i = 0; i < meg.channels; ++i) {
    ourCentres.push_back(ours[static_cast<std::size_t>(i)].centre);
    dc = std::max(dc, (ours[static_cast<std::size_t>(i)].centre - bstCentres[static_cast<std::size_t>(i)]).norm());
    dr = std::max(dr, std::abs(ours[static_cast<std::size_t>(i)].radius - g.at("os_radius.bst")(i, 0)));
  }
  const MatrixXd Gos = sarvasGain(src, meg, ourCentres);
  const double eOs = rel(Gos, g.at("meg_os.bst")), eSingleVsOs = rel(sarvasGain(src, meg, c0), g.at("meg_os.bst"));
  const double eOsFro = (Gos - g.at("meg_os.bst")).norm() / g.at("meg_os.bst").norm();
  std::printf("  MEG overlapping spheres: Sarvas from Brainstorm's spheres %.2e; our spheres: centres within %.2f mm, radii within %.2f mm, gain max %.2e (Frobenius %.2e); single sphere vs Brainstorm's OS: %.2e\n",
              eOsGiven, dc * 1e3, dr * 1e3, eOs, eOsFro, eSingleVsOs);
  CHECK(eOsGiven < 1e-10);
  CHECK(eOs < 0.1);                    // the sphere fit is not the reference's (spec section 8): about 4% on this case
  CHECK(eOs < eSingleVsOs);            // but closer to its overlapping spheres than one sphere is

  // ---- EEG, three shells
  Coils el;
  el.position = g.at("eeg");
  el.weight = VectorXd::Ones(el.position.rows());
  for (int i = 0; i < el.position.rows(); ++i) el.channel.push_back(i);
  el.channels = static_cast<int>(el.position.rows());
  auto given = [&](const MatrixXd& mu, const MatrixXd& lam) {
    BergScherg b;
    b.eccentricity = mu.col(0);
    b.magnitude = lam.col(0);
    return b;
  };
  EegSphereOptions bst;                                           // Brainstorm's layers are the defaults
  const BergScherg bb = given(g.at("berg_mu.bst"), g.at("berg_lam.bst"));
  EegSphereOptions bstGiven = bst;
  bstGiven.useGivenBerg = true;
  bstGiven.given = bb;
  const double eF = (multiShellCoefficients(bst.radius, bst.conductivity, 200) - g.at("berg_f.bst").col(0)).cwiseAbs().maxCoeff();
  const double eBstE = rel(eegSphereGain(src, el, ec, bstGiven), g.at("eeg_bst.bst")), eBstEo = rel(eegSphereGain(srcOut, el, ec, bstGiven), g.at("eeg_bst_out.bst"));
  EegSphereOptions oneShell;
  oneShell.radius = VectorXd::Constant(1, 1.0);
  oneShell.conductivity = VectorXd::Constant(1, 0.33);
  oneShell.approximation = EegApproximation::Series;
  oneShell.seriesTerms = 400;
  const double eOne = rel(eegSphereGain(src, el, ec, oneShell), g.at("eeg_bst_1.bst"));
  std::printf("  EEG: f_n vs the reference %.2e; from Brainstorm's Berg parameters: gain %.2e (sources outside %.2e); one shell (series) %.2e\n", eF, eBstE, eBstEo, eOne);
  CHECK(eF < 1e-10 && eBstE < 1e-10 && eBstEo < 1e-10);
  CHECK(eOne < 1e-5);                  // Brainstorm's single shell differs at this level from the closed form (its own truncation)
  EegSphereOptions mne = mneEegSphereOptions();
  EegSphereOptions mneGiven = mne;
  mneGiven.useGivenBerg = true;
  mneGiven.given = given(g.at("eeg_mnelayers.mne_mu"), g.at("eeg_mnelayers.mne_lam"));
  const double eMneE = rel(eegSphereGain(src, el, ec, mneGiven), g.at("eeg_mnelayers.mne")),
               eMneF = (multiShellCoefficients(mne.radius, mne.conductivity, 200) - g.at("eeg_mnelayers.f").col(0)).cwiseAbs().maxCoeff();
  std::printf("  EEG with MNE-Python's layers: f_n %.2e; from its Berg parameters, gain %.2e\n", eMneF, eMneE);
  CHECK(eMneE < 1e-10 && eMneF < 1e-10);
  // the DEFAULT options (recorded parameters): the references' gains with no parameters passed in
  {
    const double eD = rel(eegSphereGain(src, el, ec), g.at("eeg_bst.bst")), eDo = rel(eegSphereGain(srcOut, el, ec), g.at("eeg_bst_out.bst")), eDm = rel(eegSphereGain(src, el, ec, mneEegSphereOptions()), g.at("eeg_mnelayers.mne"));
    std::printf("  EEG default options (recorded parameters): Brainstorm %.2e (outside %.2e), MNE-Python layers %.2e\n", eD, eDo, eDm);
    CHECK(eD < 1e-10 && eDo < 1e-10 && eDm < 1e-10);
    BergScherg rec;
    CHECK(recordedBergScherg(bst.radius, bst.conductivity, rec) && recordedBergScherg(mne.radius, mne.conductivity, rec));
    CHECK(!recordedBergScherg((VectorXd(3) << 0.8, 0.9, 1.0).finished(), bst.conductivity, rec));
    // the sphere our fit gives, the same sphere handed to the reference's formula: the gain is the reference's (here the sphere is the case's own centre)
  }
  bst.useRecordedBerg = false;
  mne.useRecordedBerg = false;
  // our own Berg-Scherg fit: against the references' fits, and against the series both approximate
  BergScherg fitted;
  const MatrixXd Gfit = eegSphereGain(src, el, ec, bst, &fitted);
  EegSphereOptions ser = bst;
  ser.approximation = EegApproximation::Series;
  const MatrixXd Gser = eegSphereGain(src, el, ec, ser);
  const double eVsBst = rel(Gfit, g.at("eeg_bst.bst")), eVsSer = rel(Gfit, Gser), eBstSer = rel(g.at("eeg_bst.bst"), Gser), eMneSer = rel(g.at("eeg_mnelayers.mne"), eegSphereGain(src, el, ec, [&] { auto o = mne; o.approximation = EegApproximation::Series; return o; }()));
  const double eFitMne = rel(eegSphereGain(src, el, ec, mne), g.at("eeg_mnelayers.mne"));
  std::printf("  EEG own fit (residual %.1e): vs Brainstorm %.2e; vs the series %.2e (Brainstorm's fit vs the series %.2e, MNE-Python's %.2e); MNE layers: ours vs MNE-Python %.2e\n",
              fitted.residual, eVsBst, eVsSer, eBstSer, eMneSer, eFitMne);
  {
    // the documented fminsearch behaviour, one start (a start of the published kind): how near does it come to the reference's parameters?
    EegSphereOptions fm = bst;
    fm.berg.starts = 1;
    fm.berg.minimiser = fminsearchOptions(3);
    BergScherg one;
    const MatrixXd Gone = eegSphereGain(src, el, ec, fm, &one);
    std::printf("  EEG single start, fminsearch rules: vs Brainstorm %.2e, vs the series %.2e (residual %.1e)\n", rel(Gone, g.at("eeg_bst.bst")), rel(Gone, Gser), one.residual);
  }
  CHECK(eVsSer < 2e-3);
  CHECK(eVsSer < eBstSer);             // closer to the exact series than the reference's fit
  CHECK(eVsBst < 2e-2 && eFitMne < 2e-2);
  return nxrtest::finish("test_forward_parity");
}
