// nxr::partition — a field measured over the MEMBERS of a partition (a labelling: one code per element; Desikan-Killiany, a level of a tile ladder, a
// channel grouping). A port of the Cortical Flow Studio's `measureOverPartition` (backend/src/db/measurements.ts), the reference: one pass over the
// elements accumulates, per member, the weight Σw, the weighted sums Σw·x and Σw·x², the count, max, min and max |x|; the measures are read from those.
//
//   mean         Σw·x / Σw            (0 for a member of zero weight, as the reference)
//   mean_square  Σw·x² / Σw
//   sum          Σw·x
//   max · min · abs_max               over the member's elements (unweighted)
//   area         Σw                   (the member's area when w is a vertex area)
//   n_elements   the member's element count
//   mean_abs     Σw·|x| / Σw,  mean_sqrt_abs  Σw·√|x| / Σw   (nxr-neuro's own, 2026-11-13: the ingredients of MATLAB's shape, impulse and clearance factors)
//   mean_cube    Σw·x³ / Σw,   mean_fourth    Σw·x⁴ / Σw     (nxr-neuro's own, 2026-11-13: the raw moments from which skewness and kurtosis WITHIN a
//                                                           member are derived at read, measures_nxr.json; they roll up as the mean does)
//
// Each stored row carries `weight` = Σw, so a mean rolls up exactly (Σ mean·weight / Σ weight = Σw·x / Σw) and the variance is derived at read from
// the rolled-up mean and mean_square (the measure registry, measures.json). The WEIGHTS are the caller's (an option, never assumed): none (1 per element:
// Brainstorm's scout mean), a barycentric vertex area (nxr's atlas, Rheome), a mixed-Voronoi area (the Studio's `lumpedMass`).
// Departures from the reference, each counted, never silent: an element with a NEGATIVE code belongs to no member and is left out (the Studio's
// `tileMeans` rule; the reference refuses it); a non-finite value is counted per member (`nonfinite`) — it makes that member's moments non-finite, which
// the reference then leaves unwritten.
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::partition {

struct Sums {
  int members = 0;
  std::vector<double> weight, sum, sum_square, max, min, abs_max;   // [members]
  std::vector<double> sum_abs, sum_sqrt_abs;                        // [members] Σw·|x|, Σw·√|x|: nxr-neuro's own (2026-11-13), for MATLAB's shape, impulse and clearance factors
  std::vector<double> sum_cube, sum_fourth;                         // [members] Σw·x³, Σw·x⁴: nxr-neuro's own (2026-11-13), for skewness and kurtosis within a member
  std::vector<std::int64_t> count, nonfinite;                      // [members]
  std::int64_t unassigned = 0;                                     // elements with a negative code
};

// Accumulate `values` over the members of `codes` (n elements). `weights` null: 1 per element. A code ≥ `members` is an error (std::invalid_argument).
// `values` null: only the weights and counts (a geometric measurement: area, n_elements).
Sums accumulate(const std::int32_t* codes, std::size_t n, int members, const double* values, const double* weights);

// The MEAN within each member for EVERY column of a block at once (time left unreduced): `data` row-major, `rows` elements, `stride` floats between rows, `cols` columns; `codes` one per element (a
// negative code in no member), `weights` null: 1 per element. Per column the `mean` of `accumulate` (Σw·x / Σw, summed in element order, so the same doubles); a member of no weight is NaN
// here (a tile of no vertex has no mean; `accumulate`'s 0 is the reference's convention for a stored row). A code ≥ `members` is an error (std::invalid_argument).
Eigen::MatrixXd meanOverMembers(const float* data, std::size_t rows, std::size_t stride, std::size_t cols, const std::int32_t* codes, int members, const double* weights);

// The MEDIAN within a member: per member, the median of its FINITE values (the mean of the two middle ones for an even
// count), NaN for an empty member. Exact and unweighted: a median cannot be accumulated from sums, so it takes all of a member's values at once, and it
// does not roll up (reduction "none"). A negative code is left out; a code >= members is an error (std::invalid_argument), as `accumulate`.
std::vector<double> medians(const std::int32_t* codes, std::size_t n, int members, const double* values);

// The measures this producer writes, in the reference's order of defaults: with a field, mean · max · min · mean_square · abs_max; without, area · n_elements.
const std::vector<std::string>& field_measures();
const std::vector<std::string>& geometric_measures();
bool known_measure(const std::string& name);

// A measure of member `m`, as the reference computes it; NaN when it is not finite (an empty member for max/min; a non-finite value). The caller writes
// only finite values, as the reference does.
double measure(const Sums& s, int m, const std::string& name);

// ── TIME PARTITIONS (the same Studio file, measurements.ts `reduceTiles`, the reference) ──
// A SEGMENTED REDUCE over `tiles` contiguous windows of `width` samples, across every row (channel, vertex) of a [rows x samples] block (row-major): per
// tile the min, max, Σx, Σx² and the count of samples, pooled over the rows. The Studio's "ONE arithmetic: main's batch producer runs it over a whole
// recording, and the app's CPU node runs it a chunk at a time" — it is additive (sums and counts add, min and max take the min and max), so the tiles of
// a block read chunk by chunk are the chunks' tiles combined. For values PER ROW per window (an array-backed measurement, D96: the envelope's min and max
// per channel), reduce each row as a block of one. Float32 as the Studio's arrays: Σx and Σx² summed in double over a row's run, added into float.
// The tiles start at the block's first sample; the last tile may be partial (its count says so).
struct TileReduction {
  std::vector<float> min, max, sum, sumSq, count;                  // [tiles]
};
TileReduction reduceTiles(const float* values, std::size_t rows, std::size_t samples, std::size_t width, std::size_t tiles);

// ── THE CROSS-SPECTRUM over time windows: a measurement over a time partition of the pointwise outer product z zᴴ ──
// Z [channels x samples] complex: a band's (or a filterbank member's) analytic coefficients of every channel. Per tile of `width` samples (`tiles` of them,
// from the block's first sample; the last may be partial), Σ_t z(t) z(t)ᴴ ([channels x channels], Hermitian) and the samples summed. The outer product is
// never formed sample by sample: a tile's sum is one complex product Z_t Z_tᴴ. ADDITIVE, as reduceTiles: a window read chunk by chunk is the sum of its
// parts (a window straddling two chunks: each chunk's part, added). The mean (the cross-spectral density of the band over the window) is Σ / count, and
// the coherency Σ_uv / √(Σ_uu Σ_vv), at read (ratios are derived). The averaging span is the CALLER's (the tiling), never the block it is handed: a result
// does not depend on how the record was chunked. In sensor space C is the compact statistic of every source pair: K_u C K_vᵀ (the kernel is linear).
struct CrossSpectrumTiles {
  std::vector<Eigen::MatrixXcd> sum;      // [tiles], each [channels x channels]
  std::vector<std::int64_t> count;        // [tiles] samples summed (0 for a tile past the data)
};
CrossSpectrumTiles crossSpectrum(const Eigen::MatrixXcd& Z, std::size_t width, std::size_t tiles);

// ── THE REDUCTION OF PRODUCTS over a partition's members: the sum of matrices (z zᴴ sums add) ──
// `value[i]` belongs to member `member[i]` of `members` (−1: none); the member's sum is the plain sum of its values (all the same shape) and its count the number of values summed; a member with no value gets an
// EMPTY (0 x 0) matrix. EXACT for products (a burst's cross-spectrum is the sum of its time tiles'); never use it on phases or on ratios. Throws std::invalid_argument on mismatched shapes or a member out of range.
struct MatrixSums {
  std::vector<Eigen::MatrixXcd> sum;      // [members]
  std::vector<std::int64_t> count;        // [members] values summed
};
MatrixSums sumOverMembers(const std::vector<Eigen::MatrixXcd>& value, const std::vector<int>& member, int members);

// ── THE ANALYTIC SUMS over a joint partition: members of a spatial partition × the tiles of a time partition (the measurement pyramid, nxr `measure.scalars.band_power`) ──
// A complex field z (a band's cortical current J = K A, [rows x samples]: real and imaginary parts) measured over (member of `codes`, time tile of `tile`): per cell Σ a·|z|² (`power`),
// Σ a·|z| (`env`, the envelope) and max |z| (`envmax`), a the row's weight (the vertex's lumped area); every sample of a tile adds, so the cells are MERGEABLE (sums add, maxima take the
// maximum): a tile cut by a block edge is completed by the next block, and the result does not depend on the block size. `tile[s]` indexes the cell's time axis (< 0: the sample is in no tile);
// a row with a negative code is in no member. Cells are laid out [tile x member] (row-major: a tile's members contiguous). `envmax` starts at -inf (an empty cell; the identity of max);
// a sample masked by the caller carries z = 0 (it adds 0 and holds a max of 0), so it needs no special case here.
struct AnalyticSums {
  int members = 0;
  std::int64_t tiles = 0;
  std::vector<double> power, env, envmax;   // [tiles x members]
  AnalyticSums() = default;
  AnalyticSums(int members_, std::int64_t tiles_);
};
// Adds the block to `out`. `re`, `im`: [rows x samples]; `codes`, `weights`: [rows]; `tile`: [samples], each < out.tiles. A code >= out.members is an error (std::invalid_argument).
void analyticSums(AnalyticSums& out, const Eigen::MatrixXd& re, const Eigen::MatrixXd& im, const std::int32_t* codes, const double* weights, const std::int32_t* tile);

// ── THE ORIENTATION TENSOR SUMS over the same joint partition (nxr `measure.vectors.band_tensor`; the free-orientation current of a band) ──
// A band's complex current in a vertex's frame (1, 2, n) = (e1 north, e2 west, n outward): J1, J2, Jn, each given by its real and imaginary parts [rows x samples]. Per cell the Hermitian tensor's nine
// real numbers, summed with the row's weight: t11 = Σ a|J1|², t22, tnn, t12r = Σ a Re(J1 J2*), t12i = Σ a Im(J1 J2*) = Σ a (J1i J2r − J1r J2i), t1nr/t1ni (1 with n), t2nr/t2ni (2 with n); and `total` = Σ b (|J1|² + |J2|² + |Jn|²),
// the trace with ITS OWN weight b (a vertex with no frame is out of the nine sums, weight a = 0, yet in the total with its area b). All are sums, so the cells are MERGEABLE (a tile cut by a block edge is completed by the
// next block; the result does not depend on the block size). Layout [tile x member], as AnalyticSums. `tile[s] < 0`: the sample is in no tile; `codes[v] < 0`: the row is in no member.
struct TensorSums {
  int members = 0;
  std::int64_t tiles = 0;
  std::vector<double> t11, t22, tnn, t12r, t12i, t1nr, t1ni, t2nr, t2ni, total;   // [tiles x members]
  TensorSums() = default;
  TensorSums(int members_, std::int64_t tiles_);
  /** the ten arrays in nxr's order TENSOR_KEYS: t11 t22 tnn t12r t12i t1nr t1ni t2nr t2ni total */
  std::vector<std::vector<double>*> arrays();
  static const char* const* keys();
};
// Adds the block. `J[0..5]` = J1r, J1i, J2r, J2i, Jnr, Jni, each [rows x samples]; `weights` (the nine sums) and `totalWeights` (the total): [rows]. Throws std::invalid_argument on a shape mismatch or a code beyond the members.
void tensorSums(TensorSums& out, const Eigen::MatrixXd* J, const std::int32_t* codes, const double* weights, const double* totalWeights, const std::int32_t* tile);

}  // namespace nxr::partition
