#pragma once
// nxr::signal — THE BISPECTRUM OF CHUNK SPECTRA WEIGHTED BY FREQUENCY ATOMS (spec docs/specs/2026-11-24-bispectrum.md). PURE MATHS: arrays in, arrays out; nothing here knows a file,
// a chunk or a subject.
//
// For one chunk's spectrum x [rows x bins] (bin i of the array is bin firstBin + i of a transform) and three gains per triple, one per axis (g1 on the first frequency, g2 on the
// second, g3 on their sum), the triple product summed over every pair of bins whose sum is a kept bin:
//
//     bispectrum  B = Σ_{i, j} g1(i) x(i) · g2(j) x(j) · conj(g3(i + j + firstBin) x(i + j + firstBin))
//
// (the direct, FFT-based estimate of Hagihira et al. 2001 and Sigl and Chamoun 1994, here weighted by atoms; with every gain 1 at one bin it is the classic estimate at that
// pair of bins). The weighting w = g1 g2 g3 (non-negative gains) gives the two normalisations their bounds:
//   threenorm   (Shahbazi Avarvand et al. 2018; Bartz et al. 2019; Zandvoort and Nolte 2021): the cube sums  Σ w |x(i)|³,  Σ w |x(j)|³,  Σ w |x(i + j)|³ ;
//               bicoherence = |B| / cbrt(product of the three) ≤ 1 (Hölder over chunks and pairs of bins)
//   kim_powers  (Kim and Powers 1979; Hagihira et al. 2001): the product power  Σ w |x(i) x(j)|²  and the sum power  Σ w |x(i + j)|² ;
//               bicoherence² = |B|² / (product power · sum power) ≤ 1 (Cauchy-Schwarz)
// All are SUMS: the sums of several chunks are the sums of each (the caller adds them), and every ratio is derived at read from them.
//
// THE CHEAPEST ORDER. B is never formed per pair of bins (rows x bins² per triple). Triples that share their (second, sum) gains share one correlation along the frequency line,
// W(i) = Σ_j g2 x(j) conj(g3 x(i + j)), read with each first gain by one dot product; triples that share their (first, second) gains share one convolution
// V(s) = Σ_{i + j = s} g1 x(i) g2 x(j), read with each sum gain. `atomTriples` picks the grouping with fewer distinct pairs (phase-amplitude coupling: the (high, high) pairs;
// harmonics: either). Each correlation or convolution is one transform of the gains' supports (nxr's FFT), rows at a time. The cube sums and the sum power need only the gains'
// own correlations and convolutions (no data), made once per triple.
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::signal {

/** one reading of the bispectrum: the gain rows (indices into the gains) on the first frequency, the second, and their sum */
struct AtomTriple {
  std::int64_t first = 0, second = 0, sum = 0;
};

/** the triples, their gains and their data-free parts, made once for every chunk */
struct AtomTriples {
  Eigen::MatrixXd gains;                                       // [gain rows x bins], non-negative
  std::int64_t firstBin = 0;                                   // bin 0 of the arrays is bin firstBin of the transform
  std::vector<AtomTriple> triples;
  bool groupedBySecondAndSum = true;                           // true: (second, sum) correlated once per pair; false: (first, second) convolved once per pair
  std::vector<std::vector<std::size_t>> groups;                // the triples of each distinct pair
  Eigen::MatrixXd secondSumCorrelation;                        // [bins x triples]  Σ_j g2(j) g3(i + j + firstBin), on the first axis
  Eigen::MatrixXd firstSumCorrelation;                         // [bins x triples]  Σ_i g1(i) g3(i + j + firstBin), on the second axis
  Eigen::MatrixXd firstSecondConvolution;                      // [bins x triples]  Σ_{i + j + firstBin = s} g1(i) g2(j), on the sum axis
  Eigen::VectorXd weight;                                      // [triples]  Σ_{i, j} g1 g2 g3: the triple's support (0: no pair of bins carries it)
};
/** `gains` [gain rows x bins] (non-negative), `firstBin` >= 0, `triples` indexing the rows. Throws std::invalid_argument on a negative gain or an index outside the rows. */
AtomTriples atomTriples(const Eigen::MatrixXd& gains, std::int64_t firstBin, const std::vector<AtomTriple>& triples);

/** the normalisation whose ingredients are summed beside the bispectrum */
enum class BicoherenceNormalisation { none, threenorm, kim_powers };
/** "none" | "threenorm" | "kim_powers"; throws std::invalid_argument naming the choices */
BicoherenceNormalisation bicoherenceNormalisationOf(const std::string& name);

struct TripleSums {
  Eigen::MatrixXcd bispectrum;                                 // [rows x triples]
  Eigen::MatrixXd firstCube, secondCube, sumCube;              // [rows x triples] threenorm (empty otherwise)
  Eigen::MatrixXd productPower, sumPower;                      // [rows x triples] kim_powers (empty otherwise)
  /** adds another chunk's sums (the same rows, triples and normalisation); an empty left side takes the right's */
  TripleSums& operator+=(const TripleSums& other);
};
/** one chunk's sums: `spectrum` [rows x bins] (the bins of `triples.gains`). Throws std::invalid_argument on bins that disagree. */
TripleSums atomBispectrum(const AtomTriples& triples, const Eigen::MatrixXcd& spectrum, BicoherenceNormalisation normalisation);

}  // namespace nxr::signal
