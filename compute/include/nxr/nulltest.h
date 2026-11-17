#pragma once
// nxr::nulltest — COMPARING A MEASUREMENT WITH ITS SURROGATE DRAWS (spec docs/specs/2026-11-17-surrogates.md, sections 2 and 7). PURE MATHS on arrays: the data's values per member of a selection
// and each draw's values per member of the SAME selection in, counts, p and q out. A function never knows what the members are (tiles, vertices, patch-size entries, bursts) nor where a draw came from.
// References: Besag and Clifford (1991), the sequential Monte Carlo p; Benjamini and Hochberg (1995); Rheome +rheome/+cores bh, tilepq.
//
// THE DRAW LINE: draws i = 0, 1, … lie on a line; a batch is a member of a grid on it. `tallyDraw` folds ONE draw into a tally (a measurement of the batch, draw by draw), and the tally of the run
// is the roll-up of its batches (a sum). The stopping rule (`undecidedMembers`) and the p (`sequentialPs`) read the tally. The loop over draws is the caller's (orchestration).
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::nulltest {

/** the sequential p of Besag and Clifford (1991): `exceedances` null values at least the data's after `draws` draws, h the exceedances at which sampling stops: h / n when it stopped
 *  (exceedances >= h), else (exceedances + 1) / (n + 1). Refused (std::invalid_argument) for n < 1, exceedances outside 0 .. n, h < 1. */
double sequentialP(std::int64_t exceedances, std::int64_t draws, std::int64_t h);
/** per member: sequentialP(exceedances, draws, h) where the data has a statistic (finite) and draws > 0, NaN elsewhere */
Eigen::VectorXd sequentialPs(const Eigen::VectorXd& data, const Eigen::VectorXd& exceedances, const Eigen::VectorXd& draws, std::int64_t h);
/** per member: the Monte Carlo p of a FIXED number of draws, (exceedances + 1) / (draws + 1) (Rheome tilepq), where the data has a statistic; NaN elsewhere */
Eigen::VectorXd fixedDrawPs(const Eigen::VectorXd& data, const Eigen::VectorXd& exceedances, std::int64_t draws);
/** Benjamini-Hochberg q-values of p (NaN stays NaN and does not count among the tests; Rheome bh) */
Eigen::VectorXd benjaminiHochbergQ(const Eigen::VectorXd& p);

/** which way a draw is as extreme as the data: at least (upper) or at most (lower) */
enum class Side { upper, lower };
/** per member: 1 where the draw is at least (upper) or at most (lower) the data's, 0 where it is not, NaN where either is not finite (no comparison) */
Eigen::VectorXd exceedanceIndicator(const Eigen::VectorXd& data, const Eigen::VectorXd& draw, Side side);

/** THE TALLY of the draws, per member: the upper and lower exceedances, the draws that had a value (finite), and the sum of those values */
struct Tally {
  Eigen::VectorXd upper, lower, finite, sum;
  std::int64_t draws = 0;                       // the draws folded in (every one, finite or not)
};
/** a tally of `members` with no draw */
Tally emptyTally(Eigen::Index members);
/** the tally with ONE more draw folded in: its exceedances against `data` (both sides), its finite values counted and summed. Refused (std::invalid_argument) when the sizes differ. */
Tally tallyDraw(const Tally& tally, const Eigen::VectorXd& data, const Eigen::VectorXd& draw);
/** the roll-up of two tallies of the same members (two batches of the draw line): their sum */
Tally rollUp(const Tally& a, const Tally& b);
/** the members with a statistic (finite data) and fewer than h exceedances on `side`: the stopping rule's question (none left: every member is decided) */
std::int64_t undecidedMembers(const Eigen::VectorXd& data, const Tally& tally, std::int64_t h, Side side = Side::upper);
/** the draws' mean per member (sum / finite; NaN where no draw had a value) */
Eigen::VectorXd nullMean(const Tally& tally);

/** THE MEAN OVER MEMBERS against the draws' (a `summary` statistic over the members of an irregular selection, e.g. the bursts of a run): `data` [members], `draws` [draws x members]. The mean is over
 *  the members where the data has a statistic (the same members for every draw; a draw's non-finite value there is left out of its mean). */
struct MeanComparison { double mean = 0, nullMean = 0, nullSd = 0, p = 0; std::int64_t members = 0, draws = 0; };
/** the data's mean, the draws' means' mean and SD (n - 1), and p = (1 + #(draw mean >= data mean)) / (draws + 1). Refused (std::invalid_argument) with no member or fewer than 2 draws. */
MeanComparison compareMeanOverMembers(const Eigen::VectorXd& data, const Eigen::MatrixXd& draws);

/** per member: the data's count over the null's mean count (`nullCountSum` / `draws`); NaN where the null's mean is 0 */
Eigen::VectorXd countRatio(const Eigen::VectorXd& dataCounts, const Eigen::VectorXd& nullCountSum, std::int64_t draws);

// ── the vortex chain's crossings ──
/** per tile: the unbiased rho^2 = (|S|^2 - n) / (n (n - 1)) of the tile's pooled sum S [tiles x 3] of n crossings (never R: places with different counts are compared); NaN with fewer than
 *  max(minCount, 2) crossings */
Eigen::VectorXd unbiasedConsistency(const Eigen::MatrixXd& sum, const Eigen::VectorXd& n, int minCount);
/** per tile: the mean resultant sum / n [tiles x 3] (a zero row where n is 0) */
Eigen::MatrixXd meanResultants(const Eigen::MatrixXd& sum, const Eigen::VectorXd& n);

// ── the spatial events ──
/** per vertex, from [vertices x 2] (frames in a patch, frames; row-major): the fraction of frames in a patch. `occupiedOnly` (the data's): NaN where the vertex is never in a patch or has no frame
 *  (no test); otherwise (a draw's) 0 where it has no frame. */
Eigen::VectorXd occupancyFraction(const std::vector<double>& inPatchAndFrames, bool occupiedOnly);
/** the size distribution of the patches: each count's share of the patches (NaN with no patch), then the patches per frame (NaN with no frame) */
Eigen::VectorXd patchShares(const Eigen::VectorXd& counts, double patches, double frames);

}  // namespace nxr::nulltest
