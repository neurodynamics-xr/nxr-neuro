// nxr::measurement (compute) — `reduce`: a field (or a measurement) over the members of ANY selection (spec docs/specs/2026-11-17-consolidation.md, section 2.2). The ONE
// implementation of a reduction across time or space: every other copy (partition::accumulate, reduceTiles, crossSpectrum, analyticSums, tensorSums, pyramid::bandSums, the
// ladder summaries, intrinsic tile flux, band amplitude, histogram tiles) is retired into it. The reference is the Studio's `measureOverPartition` and `reduceTiles`.
//
// WHAT IT TAKES: values (float or double: PRECISION FOLLOWS THE INPUT; the ingredients are accumulated in double) and the selection's RESOLVED FORM (selection/resolved.h): a member
// number per element, listed elements (overlapping members), a rule on a time line, or a product (time x space). WHAT IT GIVES: the additive INGREDIENTS (measures.h), per row and
// member, folded block by block: a member read chunk by chunk is the chunks' ingredients added, so the result does not depend on how the record was cut. A reduction across time or
// space is never a node's own: it is this, over a partition's members.
//
// LAYOUT. A reduction over SPACE (member-of-element, elements) takes values [rows x elements], row-major, one row a frame or a sample: time is left unreduced (rows). A reduction over
// TIME (a rule, a product) takes values [rows x samples], row-major, one row a channel or a vertex: with `per_row` the rows stay (a value per row per member: the band amplitude of
// every channel), else they are POOLED into one value per member (reduceTiles: pooled over the rows). A product reduces its two factors at once: factor 0 lies on the block's samples
// (a time rule), factor 1 on its rows' elements (a member number or listed elements: the channels, the leaves of a surface grid); the cells are the product's codes, first factor
// fastest, one row of ingredients.
//
// WEIGHTS are the caller's, stated: Weighting::none (1 per element: Brainstorm's scout mean), or given (a barycentric vertex area, nxr's atlas; a mixed-Voronoi area, the Studio's
// `lumpedMass`). `values` null: only the weights and counts (a member's own measures: area, n_elements). A non-finite value is counted (`nonfinite`); it makes the member's sums
// non-finite, which `value` reads as NaN (the reference leaves it unwritten). An element with a negative member number is in no member and counted in `unassigned`.
//
// MASKS. Bad samples are an EXCLUDED selection's: the caller removes them from the blocks it hands (a sample masked carries no weight), never a factor.
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/measurement/measures.h"
#include "nxr/selection/resolved.h"

namespace nxr::measurement {

enum class Weighting { none, given };

struct Request {
  std::vector<std::string> measures;       // the measures wanted (no default: the caller's or the preset's); each must be in the registry; a median keeps the values
  Weighting weighting = Weighting::none;
  int component = 0;                       // a vector field is reduced one component per call: this call's component of `Ingredients::sum`
  bool per_row = false;                    // a reduction over time: keep a value per row (else pool the rows)
};

// ── over SPACE: members of elements ────────────────────────────────────────────────────────
/** values [rows x elements]; acc must be Ingredients(rows, members(m), components). A member number >= members is an error (std::invalid_argument). */
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t elements, const selection::MemberOfElement& members, const double* weights, const Request&);
void reduce(Ingredients& acc, const double* values, std::int64_t rows, std::int64_t elements, const selection::MemberOfElement& members, const double* weights, const Request&);
/** overlapping members (a drawn region, a disc, picked channels): each member's elements, an element in several members adds to each */
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t elements, const selection::Elements& members, const double* weights, const Request&);
void reduce(Ingredients& acc, const double* values, std::int64_t rows, std::int64_t elements, const selection::Elements& members, const double* weights, const Request&);

// ── over TIME: a rule on the line (a tiling, Welch's segments, chunks) ─────────────────────
/** values [rows x samples] holding samples first_sample .. first_sample + samples - 1 of the line; acc is Ingredients(per_row ? rows : 1, tiles.count, ...). A sample in no member adds
 *  nothing; where members overlap (hop < width) a sample adds to every member holding it. */
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const selection::Rule& tiles, const Request&);
void reduce(Ingredients& acc, const double* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const selection::Rule& tiles, const Request&);

// ── over a PRODUCT: time x space cells ─────────────────────────────────────────────────────
/** values [rows x samples] where the rows are the elements of the product's factor 1; acc is Ingredients(1, cells.members(), ...). factor 0: a Rule (a time rule) or a MemberOfElement over
 *  the line's samples; factor 1: a MemberOfElement or Elements over the rows. `weights` [rows] (per element of factor 1; null with Weighting::none). */
void reduce(Ingredients& acc, const float* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const selection::ProductForm& cells, const double* weights, const Request&);
void reduce(Ingredients& acc, const double* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const selection::ProductForm& cells, const double* weights, const Request&);

// ── the OUTER form: a matrix per member (z z^H summed) ─────────────────────────────────────
/** the sums of matrices over members, additive as the ingredients are: Σ_t z(t) z(t)^H of a band's analytic coefficients over each tile (the cross-spectrum of the band over the
 *  window), or a burst's cross-spectrum as the sum of its tiles' (`merge`). The mean (the cross-spectral density) is sum / count at read, and the coherency Σ_uv / √(Σ_uu Σ_vv) (ratios
 *  are derived). The averaging span is the CALLER's (the rule), never the block it is handed: a result does not depend on how the record was chunked. */
struct MatrixIngredients {
  std::vector<Eigen::MatrixXcd> sum;       // [members], each [n x n]; 0 x 0 until a value is added
  std::vector<std::int64_t> count;         // [members] values (samples) summed
  MatrixIngredients() = default;
  explicit MatrixIngredients(std::int64_t members) : sum(static_cast<std::size_t>(members)), count(static_cast<std::size_t>(members), 0) {}
  std::int64_t members() const { return static_cast<std::int64_t>(sum.size()); }
  /** sum / count of a member (std::invalid_argument for a member with no value) */
  Eigen::MatrixXcd mean(std::int64_t member) const;
};
/** Z [channels x samples] complex: samples first_sample .. of the line; per member of `tiles` that the block touches, Σ z z^H over the block's part of it (one complex product Z_t Z_t^H),
 *  added to acc.sum. acc must hold tiles.count members. */
void reduce(MatrixIngredients& acc, const Eigen::MatrixXcd& Z, std::int64_t first_sample, const selection::Rule& tiles);
/** add one value (a matrix) to a member */
void add(MatrixIngredients& acc, std::int64_t member, const Eigen::MatrixXcd& value);

}  // namespace nxr::measurement
