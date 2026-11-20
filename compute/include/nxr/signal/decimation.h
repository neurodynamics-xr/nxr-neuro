// nxr::signal — DECIMATION: a signal at a lower rate (rate / q), anti-aliased and with no delay, as scipy.signal.decimate(x, q, ftype='fir', zero_phase=True)
// (its default design: a window-method low-pass of order 20 q, Hamming, its edge at the new Nyquist frequency, 1 / q in units of the old one; applied centred, so the phase
// is zero; then every q-th sample kept, the first sample of the signal first).
//
// THE HALO: the low-pass has order / 2 taps on each side of its centre, so a block read with order / 2 samples of context on each side (`decimationHalo`) gives, on its own
// columns, exactly what the whole signal gives there; the samples beyond the block count as zero (scipy's padding at the signal's ends). Which of the block's own columns
// are kept is the caller's `offset` (the first one whose index in the whole signal is a multiple of q), so decimated blocks laid end to end are the whole signal decimated,
// whatever the block size.
//
// What it cannot resolve: nothing above the new Nyquist frequency survives; the band just below it is attenuated by the filter's transition (the window method's), and
// within order / 2 samples of the signal's ends the output is the zero-padded signal's.
//
// PURE MATHS: a matrix in, a matrix out ([channels x samples], one row a channel). Precision follows the input: float32 in, float32 out.
#pragma once
#include <Eigen/Dense>

namespace nxr::signal {

/** the order of the low-pass a decimation by `q` uses: `order` when given (even, >= 2), else 20 q (scipy's) */
int decimationOrder(int q, int order = 0);
/** the samples of context a block needs on each side so its decimated columns equal the whole signal's: order / 2 */
Eigen::Index decimationHalo(int q, int order = 0);

/** x [channels x Wr] decimated by the integer factor `q` (>= 1): the block's own columns are [left, left + width) (width < 0: to the block's end), the samples kept are the own columns
 *  left + offset, left + offset + q, ... (0 <= offset < q), each the zero-phase low-pass's output there (samples beyond the block are zero). Returns [channels x kept].
 *  q = 1 returns the own columns unchanged. Throws std::invalid_argument on a bad factor, order, offset or column range. */
Eigen::MatrixXd decimate(const Eigen::MatrixXd& x, int q, Eigen::Index left = 0, Eigen::Index width = -1, Eigen::Index offset = 0, int order = 0);
Eigen::MatrixXf decimate(const Eigen::MatrixXf& x, int q, Eigen::Index left = 0, Eigen::Index width = -1, Eigen::Index offset = 0, int order = 0);

}  // namespace nxr::signal
