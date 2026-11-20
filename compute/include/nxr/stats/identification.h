#pragma once
// nxr::stats — IDENTIFICATION OF INDIVIDUALS FROM TWO MEASUREMENTS EACH ("brain fingerprinting"): how much more alike two measurements of one person are than measurements of
// different people, and which features carry that difference.
//
// PURE MATHS (matrices in, matrices out; no file I/O). Called with any two tables of the same shape, row i of each the same individual: where the rows came from (the stored spectra
// of a cohort's members, connectomes, synthetic arrays) is the caller's business. The reference is the published method:
//   da Silva Castanheira J, Orozco Perez HD, Misic B, Baillet S (2021) Brief segments of neurophysiological activity enable individual differentiation. Nat Commun 12:5713,
//   doi:10.1038/s41467-021-25895-8; its authors' release megFingerprinting v1.0.0 (06fdba8), the notebooks 4_PSD_fingerprinting_* and 4_AEC_fingerprinting_*;
//   the identification statistics after Finn ES et al. (2015) Nat Neurosci 18:1664 and Amico E, Goni J (2018) Sci Rep 8:8254.
// Written from the paper and the notebooks' documented behaviour; the authors' code (GPL-3) is only run to make the goldens (stats/test/golden/gen_fingerprint_*.py).
//
//   featureTable          each row a member's stored values (C order over `shape`) -> the member's feature vector: `flatten` keeps them as they are (a region x bin spectrum, row-major),
//                         `upper-triangle` keeps, per trailing slice (a band), the upper triangle of the leading square [r x r] WITH its diagonal, row-major, the slices concatenated
//                         (the connectome feature: 68 x 69 / 2 = 2346 per band). The release orders the regions by name (a pandas pivot); here they keep the stored order.
//   zscoreRows            each row z-scored across its columns, population SD (ddof 0; scipy.stats.zscore)
//   identificationCorrelation   r_ij = z(t_i) . z(d_j) / p: Pearson's r between target row i and database row j. (The notebooks divide by the table's width minus its label column, p.)
//   identificationAccuracy      target from database: the share of rows whose largest r is on the diagonal (argmax per row, the first on a tie, numpy's); database from target: per column
//   selfIdentifiability   D_self,i: `all` (the default; the notebooks', which made the published numbers): the diagonal of each row of r z-scored over ALL its n entries, the participant
//                         included (ddof 0); `others` (the paper's formula): (r_ii - mean_{j != i} r_ij) / sd_{j != i} r_ij (ddof 0, as the notebooks' SD)
//   intraclassCorrelation ICC(1,1), one-way random effects, k = 2 measurements, per column: (MS_b - MS_w) / (MS_b + (k - 1) MS_w), MS_b over n - 1, MS_w over n (k - 1)
//   differentialPower     Finn 2015, per column: for each member i, p_i = (#{j != i: a_i b_j > a_i b_i} + #{j != i: b_i a_j > a_i b_i}) / (2 (n - 1)), strict >, floored at 1 / (2 (n - 1));
//                         the column's power is the sum over i of -ln p_i
//   groupConsistency      per column, the mean over the members of a_i b_i
// The edgewise statistics (the last three) are taken, as the notebooks take them, on the z-scored features (`identification` z-scores first): they measure the reliability of a feature's
// place within each person's profile, not of its absolute value.
//
// THE PCA RECONSTRUCTION CURVE (Amico & Goni 2018; the notebooks): target and database stacked [2n x p], centred, decomposed (nxr::stats::pca, an EXACT thin SVD), reconstructed at k
// components with the mean (nxr::stats::reconstruct), split back and identified; k in the steps (default: 2, 5, then 10, 20, ... below 2n); the best k is the first of the largest mean
// reconstructed D_self. The release's scikit-learn PCA(k) uses its automatic solver, randomised (and unseeded) for k below 80 % of min(2n, p) when the table is larger than 500 x 500:
// its curve equals this one only to that solver's tolerance.
//
// What it cannot show: identification says the two measurements of a person are more alike than those of different people, not why (anatomy, the head's position, the kernel, recurring
// artefacts and the day's noise are individual too); accuracy falls as the cohort grows and is not comparable between cohorts; the curve and its best k are fitted on the data they score.
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::stats {

enum class FeatureLayout { Flatten, UpperTriangle };
enum class SelfReference { All, Others };

/** [n x F] from [n x prod(shape)]: see the header. UpperTriangle needs shape [r, r] or [r, r, slices]. std::invalid_argument for a row of the wrong size or a shape that is not square. */
Eigen::MatrixXd featureTable(const Eigen::MatrixXd& members, const std::vector<std::int64_t>& shape, FeatureLayout layout);

/** each row minus its mean, over its population SD (ddof 0). A row with no variance, or a value that is not finite, is refused (std::invalid_argument naming the row). */
Eigen::MatrixXd zscoreRows(const Eigen::MatrixXd& X);

/** Pearson's r between every target row and every database row [n_target x n_database]; both tables have p >= 2 columns. */
Eigen::MatrixXd identificationCorrelation(const Eigen::MatrixXd& target, const Eigen::MatrixXd& database);

struct IdentificationAccuracy {
    double accuracyByRow = 0;      // TARGET FROM DATABASE: the share of rows i (targets) whose largest r_ij is at j = i
    double accuracyByColumn = 0;   // DATABASE FROM TARGET: the share of columns j (database entries) whose largest r_ij is at i = j
    double chance = 0;               // 1 / n
};
/** the two accuracies of a square correlation matrix (row i and column i the same member) */
IdentificationAccuracy identificationAccuracy(const Eigen::MatrixXd& correlation);

/** D_self per member of a square correlation matrix (n >= 2; `others` needs n >= 3). A row with no spread is refused. */
Eigen::VectorXd selfIdentifiability(const Eigen::MatrixXd& correlation, SelfReference reference = SelfReference::All);

/** per column of two tables [n x p] (row i of each the same member; n >= 2): ICC(1,1) with k = 2 */
Eigen::VectorXd intraclassCorrelation(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);
/** per column: Finn's differential power (n >= 2) */
Eigen::VectorXd differentialPower(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);
/** per column: the mean over the members of a_i b_i */
Eigen::VectorXd groupConsistency(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

/** everything one comparison gives: the rows z-scored, then the correlation, the accuracies, the self-correlation (r_ii) and D_self, and the edgewise statistics on the z-scored features */
struct Identification {
    Eigen::MatrixXd correlation;
    IdentificationAccuracy accuracy;
    Eigen::VectorXd selfCorrelation, selfIdentifiability;
    Eigen::VectorXd icc, differentialPower, groupConsistency;
};
/** target and database [n x p], row i of each the same member (n >= 2, p >= 2) */
Identification identification(const Eigen::MatrixXd& target, const Eigen::MatrixXd& database, SelfReference reference = SelfReference::All);

/** the notebooks' steps for a stacked table of `rows` (= 2n) rows: 2, 5, then 10, 20, ... below `rows` (those below `rows` only) */
std::vector<std::int64_t> identificationSteps(std::int64_t rows);

struct IdentificationCurve {
    std::vector<std::int64_t> steps;              // the k asked
    std::vector<std::int64_t> used;               // the components each used: k, or the table's rank when k exceeds it (the components beyond the rank are zero)
    Eigen::VectorXd accuracyByRow, accuracyByColumn, meanSelfIdentifiability;   // per step
    std::int64_t best = 0;                        // the index into steps of the largest mean D_self (the first, on a tie: numpy's argmax)
    Identification atBest;                        // the identification of the reconstruction at the best k
};
/** the curve from a PCA of the stacked table [target; database] (2n rows, the n target rows first): its scores [2n x r], components [r x p], mean and divisor [p] (nxr::stats::pca);
 *  every k reconstructed with nxr::stats::reconstruct. A step below 1 or above min(2n, p) (scikit-learn's bound) is refused. */
IdentificationCurve identificationCurve(const Eigen::MatrixXd& scores, const Eigen::MatrixXd& components, const Eigen::VectorXd& mean, const Eigen::VectorXd& scale,
                                        const std::vector<std::int64_t>& steps, SelfReference reference = SelfReference::All);

}  // namespace nxr::stats
