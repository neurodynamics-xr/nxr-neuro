#pragma once
// nxr::stats — SUMMARIES OF A SAMPLE (spec docs/specs/2026-11-13-statistic-summaries.md): what a partition's members reduce to when a measurement is summarized
// (field -> selection/measurement -> statistic). Pure: a vector of values in, numbers out; non-finite values are left out by the caller (counted there).
// Each summary names its reference:
//   mean                 Σx / n                                                     MATLAB Mean
//   sd(ddof)             √(Σ(x − mean)² / (n − ddof)); ddof 1 by default            MATLAB StandardDeviation (N − 1)
//   median               the middle value; the mean of the two middle ones          numpy.median
//   mad                  1.4826 · median |x − median| (the robust SD)               Rheome detect.spindle, nxr spindle_envelope
//   percentile(p)        linear interpolation between the order statistics,         numpy.percentile (method "linear"), the Studio's percentileLinear
//                        at rank p/100 · (n − 1)
//   peak                 max |x|                                                    MATLAB PeakValue
//   kurtosis, skewness   the standardized 4th and 3rd moments, 1/n (not excess)     MATLAB Kurtosis, Skewness (scipy.stats with fisher=False, bias=True)
//   min, max
//   count                the number of values (the finite values of the members)    the burst count; rate = count / good time, derived at read
//   sum                  Σx                                                         the burst duration's sum; occupancy = Σ duration / good time, derived at read
#include <cstdint>
#include <string>
#include <vector>


namespace nxr::stats {

double mean(const std::vector<double>& x);
double sd(const std::vector<double>& x, int ddof = 1);
double median(std::vector<double> x);
double mad(const std::vector<double>& x);
double percentile(std::vector<double> x, double p);
double peak(const std::vector<double>& x);
double kurtosis(const std::vector<double>& x);
double skewness(const std::vector<double>& x);
double min(const std::vector<double>& x);
double max(const std::vector<double>& x);
double count(const std::vector<double>& x);   // the number of values (0 for none; never NaN)
double sum(const std::vector<double>& x);     // Σx (0 for none)

/** a summary by name ("mean", "sd", "median", "mad", "percentile" with p, "peak", "kurtosis", "skewness", "min", "max", "count", "sum"); std::invalid_argument for an
 *  unknown name or an empty sample */
double summary(const std::string& name, const std::vector<double>& x, double p = 50, int ddof = 1);
/** the names `summary` takes */
const std::vector<std::string>& summaries();

/** one requested summary: its name and its parameter (the percentile's p) */
struct Summary { std::string name; double p = 50; };
/** "median" · "mad" · "percentile:75" … as Summaries (std::invalid_argument naming the valid ones) */
std::vector<Summary> parseSummaries(const std::vector<std::string>& names);

/** THE SUMMARIES ACROSS THE MEMBERS of a [rows x members] table (row-major; NaN = a member with no value, left out and counted): per row (`each`), over every value of
 *  every row (`pool`), or of the mean over the rows per member (`mean`: a member with any NaN row left out). Out [rows' x summaries] (rows' = rows for
 *  each, else 1) and the members used per row' (`used`). A row' with fewer values than a summary needs gives NaN for it. */
std::vector<double> summarizeRows(const std::vector<double>& values, std::int64_t rows, std::int64_t members, const std::vector<Summary>& summaries,
                                  const std::string& combine, int ddof, std::vector<std::int64_t>* used);

}  // namespace nxr::stats
