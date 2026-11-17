// nxr::measurement — `summarize` (measurement/summarize.h): the across-member summaries, nxr::stats' arithmetic.
#include "nxr/measurement/summarize.h"

#include <stdexcept>

#include "nxr/stats.h"

namespace nxr::measurement {

Eigen::MatrixXd summarize(const Eigen::MatrixXd& values, const Summaries& summaries, std::vector<std::int64_t>* used) {
  if (values.rows() < 1 || values.cols() < 1) throw std::invalid_argument("summarize: a table of at least one row and one member");
  const std::vector<stats::Summary> parsed = stats::parseSummaries(summaries.names);
  const char* combine = summaries.combine == Combine::each ? "each" : (summaries.combine == Combine::pool ? "pool" : "mean");
  std::vector<double> flat(static_cast<std::size_t>(values.size()));
  for (Eigen::Index r = 0; r < values.rows(); ++r)
    for (Eigen::Index m = 0; m < values.cols(); ++m) flat[static_cast<std::size_t>(r * values.cols() + m)] = values(r, m);       // row-major, as summarizeRows reads
  const std::vector<double> out = stats::summarizeRows(flat, values.rows(), values.cols(), parsed, combine, summaries.ddof, used);
  const Eigen::Index rows = summaries.combine == Combine::each ? values.rows() : 1, cols = static_cast<Eigen::Index>(parsed.size());
  Eigen::MatrixXd result(rows, cols);
  for (Eigen::Index r = 0; r < rows; ++r)
    for (Eigen::Index c = 0; c < cols; ++c) result(r, c) = out[static_cast<std::size_t>(r * cols + c)];
  return result;
}

}  // namespace nxr::measurement
