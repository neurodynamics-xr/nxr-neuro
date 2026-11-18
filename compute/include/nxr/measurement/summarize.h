// nxr::measurement (compute) — `summarize`: members -> statistic (spec docs/specs/2026-11-17-consolidation.md, section 2.4). A measurement over a selection's members reduces to a
// STATISTIC across those members: mean, sd, median, mad, percentile, min, max, skewness, kurtosis, peak, count (nxr::stats is its arithmetic, each summary naming its reference).
// Over a grid's members it is a statistic of a partition; across an irregular set's members (bursts, events, scouts) it is a `summary` statistic. Never of a field: a reduction over
// time or space is `reduce`.
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::measurement {

/** how the table's rows combine: each row its own statistic; pool every value of every row into one sample; mean the rows per member (a member with a NaN row is left out) */
enum class Combine { each, pool, mean };

struct Summaries {
  std::vector<std::string> names;          // mean, sd, median, mad, percentile:P, min, max, skewness, kurtosis, peak, count, sum
  Combine combine = Combine::each;
  int ddof = 1;                            // the standard deviation's degrees of freedom (MATLAB's N - 1)
};

/** values [rows x members]; the result [rows' x summaries] (rows' = rows for `each`, else 1). A member that is NaN is skipped and counted (`used`: the members each row' used); a row' with
 *  fewer values than a summary needs gives NaN for it. std::invalid_argument for an unknown summary (naming the valid ones) or no summary. */
Eigen::MatrixXd summarize(const Eigen::MatrixXd& values, const Summaries& summaries, std::vector<std::int64_t>* used = nullptr);

}  // namespace nxr::measurement
