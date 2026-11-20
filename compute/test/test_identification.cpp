// nxr::stats' identification statistics (identification.h) against the published method's own notebooks (golden: test/golden/fingerprint_golden.txt, made by gen_fingerprint_golden.py, which
// RUNS the authors' release megFingerprinting v1.0.0 on a synthetic cohort of 12 members x 612 features; synthetic data only), and their properties:
//   the feature vectors (a region x bin spectrum flattened row-major; a connectome's upper triangle WITH its diagonal per band, bands concatenated) against the release's cell 4;
//   the correlation, both accuracies, D_self (the notebook's `all` and the paper's `others`), ICC(1,1), differential power and group consistency against the release;
//   the PCA reconstruction curve and the best k against the release run with scikit-learn's FULL solver (the exact SVD nxr computes) to rounding, and against the release as published (its
//   automatic, randomised solver) within the tolerance that solver leaves;
//   a planted cohort identified perfectly and a null cohort at chance; the refusals.
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "golden_io.h"
#include "nxr/stats/identification.h"
#include "nxr/stats/pca.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace nxr::stats;

static bool refused(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
static double maxAbs(const VectorXd& a, const MatrixXd& b) { return (a - b.row(0).transpose()).cwiseAbs().maxCoeff(); }
// the golden's order-sensitive checksum of a vector: its length, its sum and the index-weighted mean
static VectorXd checksum(const VectorXd& v) {
  VectorXd c(3);
  double w = 0;
  for (Eigen::Index k = 0; k < v.size(); ++k) w += static_cast<double>(k + 1) * v[k];
  c << static_cast<double>(v.size()), v.sum(), w / static_cast<double>(v.size());
  return c;
}

int main() {
  std::cout << "test_identification\n";
  Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/fingerprint_golden.txt");
  EXPECT(g.good(), "the golden is there");

  // ── step 20: the feature vectors, against the release's cell 4 (member 2 of a closed-form spectrum and connectome)
  {
    MatrixXd psd(1, 68 * 301);
    for (int r = 0; r < 68; ++r) for (int b = 0; b < 301; ++b) psd(0, r * 301 + b) = 1.0 + 0.01 * 2 + 0.5 * std::sin(0.37 * r + 0.11 * b) + 0.001 * r * b;
    const VectorXd v = featureTable(psd, {68, 301}, FeatureLayout::Flatten).row(0).transpose();
    const VectorXd c = checksum(v), ref = g.mat("psd_feature").row(0).transpose();
    EXPECT(c[0] == ref[0] && std::abs(c[1] - ref[1]) < 1e-9 * std::abs(ref[1]) && std::abs(c[2] - ref[2]) < 1e-9 * std::abs(ref[2]), "a region x bin spectrum flattened row-major, as the release's");
    MatrixXd aec(1, 68 * 68 * 6);
    for (int band = 0; band < 6; ++band)
      for (int r = 0; r < 68; ++r)
        for (int q = 0; q < 68; ++q) aec(0, (r * 68 + q) * 6 + band) = r == q ? 1.0 : 0.5 * std::cos(0.13 * std::min(r, q) + 0.07 * std::max(r, q) + 0.5 * band);
    const VectorXd u = featureTable(aec, {68, 68, 6}, FeatureLayout::UpperTriangle).row(0).transpose();
    const VectorXd cu = checksum(u), refu = g.mat("aec_feature").row(0).transpose();
    EXPECT(u.size() == 2346 * 6 && cu[0] == refu[0] && std::abs(cu[1] - refu[1]) < 1e-9 * std::abs(refu[1]) && std::abs(cu[2] - refu[2]) < 1e-9 * std::abs(refu[2]),
           "a connectome's upper triangle with its diagonal per band, bands concatenated, as the release's (2346 x 6)");
  }

  // ── steps 21-23, 25 against the release
  const MatrixXd T = g.mat("target"), D = g.mat("database");
  const MatrixXd R = g.mat("correlation"), acc = g.mat("accuracy");
  const Identification id = identification(T, D);
  const double r_err = (id.correlation - R).cwiseAbs().maxCoeff();
  std::cout << "  correlation " << sci(r_err) << "\n";
  EXPECT(r_err < 1e-12, "the target x database correlation is the release's (Pearson's r of the z-scored vectors)");
  EXPECT((identificationCorrelation(T, D) - R).cwiseAbs().maxCoeff() < 1e-12, "identificationCorrelation alone gives the same");
  EXPECT(id.accuracy.accuracyByRow == acc(0, 0) && id.accuracy.accuracyByColumn == acc(0, 1), "both accuracies are the release's (" + std::to_string(acc(0, 0)) + ", " + std::to_string(acc(0, 1)) + ")");
  const double s_all = maxAbs(id.selfIdentifiability, g.mat("self_all")), s_oth = maxAbs(selfIdentifiability(id.correlation, SelfReference::Others), g.mat("self_others"));
  std::cout << "  D_self all " << sci(s_all) << ", others " << sci(s_oth) << "\n";
  EXPECT(s_all < 1e-10, "D_self (all, the notebook's) is the release's");
  EXPECT(s_oth < 1e-10, "D_self (others, the paper's formula) is the formula's");
  const double i_err = maxAbs(id.icc, g.mat("icc")), p_err = maxAbs(id.differentialPower, g.mat("differential_power")), c_err = maxAbs(id.groupConsistency, g.mat("group_consistency"));
  std::cout << "  ICC " << sci(i_err) << ", differential power " << sci(p_err) << ", group consistency " << sci(c_err) << "\n";
  EXPECT(i_err < 1e-10 && p_err < 1e-10 && c_err < 1e-12, "the edgewise ICC(1,1), differential power and group consistency are the release's");

  // ── step 24: the curve, from nxr::stats::pca of the stacked table
  MatrixXd X(T.rows() * 2, T.cols());
  X.topRows(T.rows()) = T; X.bottomRows(T.rows()) = D;
  const PcaResult p = pca(X);
  std::vector<std::int64_t> steps;
  const MatrixXd gs = g.mat("steps");
  for (int j = 0; j < gs.cols(); ++j) steps.push_back(static_cast<std::int64_t>(gs(0, j)));
  EXPECT(steps == identificationSteps(X.rows()), "the notebook's steps (2, 5, 10, 20 for 24 rows)");
  const IdentificationCurve c = identificationCurve(p.scores, p.components, p.mean, p.scale, steps);
  for (const std::string tag : {"full", "release"}) {
    const MatrixXd curve = g.mat("curve_" + tag), best = g.mat("best_" + tag), best_acc = g.mat("best_accuracy_" + tag);
    const double cs = maxAbs(c.meanSelfIdentifiability, curve.row(2)), ct = maxAbs(c.accuracyByRow, curve.row(0)), cd = maxAbs(c.accuracyByColumn, curve.row(1));
    const double bs = maxAbs(c.atBest.selfIdentifiability, g.mat("best_self_" + tag)), bi = maxAbs(c.atBest.icc, g.mat("best_icc_" + tag));
    const double bp = maxAbs(c.atBest.differentialPower, g.mat("best_differential_power_" + tag)), bg = maxAbs(c.atBest.groupConsistency, g.mat("best_group_consistency_" + tag));
    std::cout << "  curve vs " << tag << ": mean D_self " << sci(cs) << ", accuracy " << sci(std::max(ct, cd)) << "; at the best k: D_self " << sci(bs) << ", ICC " << sci(bi) << ", DP " << sci(bp) << ", GC " << sci(bg) << "\n";
    const bool same_best = c.best == static_cast<std::int64_t>(best(0, 0)) && c.atBest.accuracy.accuracyByRow == best_acc(0, 0) && c.atBest.accuracy.accuracyByColumn == best_acc(0, 1);
    if (tag == "full") {
      EXPECT(cs < 1e-9 && ct == 0 && cd == 0 && same_best, "the curve and the best k are the release's with the full solver (the exact SVD)");
      EXPECT(bs < 1e-9 && bi < 1e-9 && bp < 1e-9 && bg < 1e-9, "the edgewise statistics at the best k are the release's with the full solver");
    } else {
      // the release's automatic solver is randomised for k < 0.8 min(2n, p) at this size (unseeded in the release; seeded in the generator): its reconstruction is not exact. The tolerance
      // is what it leaves here (printed by the generator), not a property of nxr.
      // measured: 1.4e-6 on the mean D_self at k = 2, 5, 10 (randomised); the best k here is 20, above 80 % of min(2n, p) = 19.2, where scikit-learn uses its full solver: exact
      EXPECT(cs < 1e-5 && ct == 0 && cd == 0 && same_best, "the curve and the best k equal the release's randomised solver within its tolerance (1e-5)");
      EXPECT(bs < 1e-5 && bi < 1e-5 && bg < 1e-5 && bp < 1e-5, "the edgewise statistics at the best k equal the release's within the randomised solver's tolerance (1e-5)");
    }
  }

  // ── a planted cohort and a null one
  {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> z;
    const int n = 20, f = 300;
    MatrixXd own(n, f), a(n, f), b(n, f), nb(n, f);
    for (int i = 0; i < n; ++i) for (int j = 0; j < f; ++j) own(i, j) = z(rng);
    for (int i = 0; i < n; ++i) for (int j = 0; j < f; ++j) { a(i, j) = own(i, j) + 0.3 * z(rng); b(i, j) = own(i, j) + 0.3 * z(rng); nb(i, j) = z(rng); }
    const Identification planted = identification(a, b);
    EXPECT(planted.accuracy.accuracyByRow == 1 && planted.accuracy.accuracyByColumn == 1 && planted.selfIdentifiability.minCoeff() > 2, "a planted cohort: identified both ways, every D_self high");
    EXPECT(planted.icc.mean() > 0.8, "a planted cohort: the features are reliable (mean ICC " + std::to_string(planted.icc.mean()) + ")");
    double hits = 0;
    const int draws = 200;
    for (int d = 0; d < draws; ++d) {
      MatrixXd x(n, 40), y(n, 40);
      for (int i = 0; i < n; ++i) for (int j = 0; j < 40; ++j) { x(i, j) = z(rng); y(i, j) = z(rng); }
      hits += identificationAccuracy(identificationCorrelation(x, y)).accuracyByRow;
    }
    const double mean_acc = hits / draws, chance = 1.0 / n, se = std::sqrt(chance * (1 - chance) / (n * draws));
    EXPECT(std::abs(mean_acc - chance) < 4 * se, "a null cohort is identified at chance (" + std::to_string(mean_acc) + " against 1/n = " + std::to_string(chance) + ")");
    const Identification null = identification(a, nb);
    EXPECT(std::abs(null.icc.mean()) < 0.1, "a null cohort: the features are not reliable (mean ICC " + std::to_string(null.icc.mean()) + ")");
  }

  // ── refusals
  EXPECT(refused([&] { identification(T, D.leftCols(10)); }), "tables of different widths are refused");
  EXPECT(refused([&] { featureTable(T, {5, 5}, FeatureLayout::Flatten); }), "a shape that is not the row's size is refused");
  EXPECT(refused([&] { featureTable(MatrixXd::Ones(1, 12), {3, 4}, FeatureLayout::UpperTriangle); }), "the upper triangle of a matrix that is not square is refused");
  EXPECT(refused([&] { zscoreRows(MatrixXd::Ones(2, 5)); }), "a constant feature vector is refused");
  EXPECT(refused([&] { identificationCurve(p.scores, p.components, p.mean, p.scale, {25}); }), "a step above min(2n, features) is refused");
  EXPECT(refused([&] { selfIdentifiability(MatrixXd::Identity(2, 2), SelfReference::Others); }), "D_self over the others needs three members");

  std::cout << (failures ? "FAILED: " : "all passed: ") << failures << "\n";
  return failures ? 1 : 0;
}
