#pragma once
// nxr::signal — the Fourier transform of nxr_compute (pocketfft, the FFT inside NumPy and SciPy, BSD-3; vendored in deps/pocketfft).
//
// Conventions are NumPy's (numpy.fft), so a result here equals numpy's to rounding:
//   forward   X[k] = sum_j x[j] exp(-2 pi i j k / n)        inverse   x[j] = sum_k X[k] exp(+2 pi i j k / n)
//   Norm::Backward (default)  forward unscaled, inverse scaled by 1/n          (numpy norm="backward")
//   Norm::Ortho               both scaled by 1/sqrt(n)                         (norm="ortho")
//   Norm::Forward             forward scaled by 1/n, inverse unscaled          (norm="forward")
// A real transform of n samples has n/2+1 non-negative-frequency bins (numpy rfft); c2r needs n, the length of the real axis,
// because n/2+1 bins fit both 2m and 2m+1 samples.
//
// ARRAYS are dense and ROW-MAJOR (C order) with a shape; the transform runs along one `axis` of it. An Eigen column-major
// [n x m] matrix (one signal per column) is the row-major array of shape {m, n} and its columns are transformed along axis 1.
// The matrix helpers below do this for you. Pure: no file I/O, no threads (pocketfft is built single-threaded, so it
// is safe inside a binding's worker and under WebAssembly).
#include <Eigen/Dense>
#include <complex>
#include <cstddef>
#include <vector>

namespace nxr::signal {

enum class Norm { Backward, Ortho, Forward };
using Shape = std::vector<std::size_t>;
using cd = std::complex<double>;

/** Complex-to-complex transform of a row-major array along `axis`. `in` and `out` may be the same pointer. */
void c2c(const cd* in, cd* out, const Shape& shape, std::size_t axis, bool forward, Norm norm = Norm::Backward);

/** Real-to-complex (numpy rfft): `in` has `shape`, `out` has the same shape with shape[axis] replaced by shape[axis]/2 + 1. */
void r2c(const double* in, cd* out, const Shape& shape, std::size_t axis, Norm norm = Norm::Backward);

/** Complex-to-real (numpy irfft): `in` has shape[axis]/2 + 1 bins along `axis`; `out` has `shape` (the real length is shape[axis]).
 *  The imaginary parts of the zero-frequency (and, for even n, Nyquist) bins are ignored, as numpy does. Inverse direction. */
void c2r(const cd* in, double* out, const Shape& shape, std::size_t axis, Norm norm = Norm::Backward);

/** Number of non-negative-frequency bins of a real transform of n samples: n/2 + 1. */
inline std::size_t rfftBins(std::size_t n) { return n / 2 + 1; }

/** numpy.fft.fftfreq(n, d): 0, 1, ..., ceil(n/2)-1, -floor(n/2), ..., -1, all divided by n*d. */
Eigen::VectorXd fftfreq(std::size_t n, double d = 1.0);

/** numpy.fft.fftshift of a vector: the zero-frequency element moved to the centre ([0 .. n/2-1, -n/2 .. -1] becomes [-n/2 .. -1, 0 .. n/2-1]); a permutation, exact. */
Eigen::VectorXd fftshift(const Eigen::VectorXd& x);

/** numpy.fft.rfftfreq(n, d): 0, 1, ..., n/2, divided by n*d. */
Eigen::VectorXd rfftfreq(std::size_t n, double d = 1.0);

/** The smallest length >= n whose prime factors are all small, so the transform is fast (scipy.fft.next_fast_len):
 *  real = true allows {2,3,5}, false allows {2,3,5,7,11}. */
std::size_t next_fast_len(std::size_t n, bool real = true);

// ── Eigen forms: ONE function per transform with an `axis` argument ──────────────────────────────────────────────────
// axis 0 transforms down the columns of a column-major matrix (one signal per column, time down the rows); axis 1 along its rows (one signal per row).
// A vector is a single column. The row-major shape and axis of the raw c2c / r2c / c2r above are the general form.

/** rfft (numpy.fft.rfft) of a real matrix along `axis`: the transformed axis has n/2+1 bins. */
Eigen::MatrixXcd rfft(const Eigen::MatrixXd& x, int axis = 0, Norm norm = Norm::Backward);
/** irfft (numpy.fft.irfft) along `axis` back to n real samples (n is required: n/2+1 bins fit 2m and 2m+1). */
Eigen::MatrixXd irfft(const Eigen::MatrixXcd& X, std::size_t n, int axis = 0, Norm norm = Norm::Backward);
/** fft / ifft (numpy.fft.fft / ifft) of a complex matrix along `axis`. */
Eigen::MatrixXcd fft(const Eigen::MatrixXcd& x, int axis = 0, Norm norm = Norm::Backward);
Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, int axis = 0, Norm norm = Norm::Backward);

/** A vector is a single column: the same functions, the norm as the second argument. */
inline Eigen::MatrixXcd rfft(const Eigen::MatrixXd& x, Norm norm) { return rfft(x, 0, norm); }
inline Eigen::MatrixXd irfft(const Eigen::MatrixXcd& X, std::size_t n, Norm norm) { return irfft(X, n, 0, norm); }
inline Eigen::MatrixXcd fft(const Eigen::MatrixXcd& x, Norm norm) { return fft(x, 0, norm); }
inline Eigen::MatrixXcd ifft(const Eigen::MatrixXcd& X, Norm norm) { return ifft(X, 0, norm); }

}  // namespace nxr::signal
