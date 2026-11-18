#pragma once
// nxr::signal — the INVERSE short-time Fourier transform: `istft`, as SciPy's scipy.signal.istft (MATLAB's istft is the same overlap-add of windowed inverse FFTs).
//
// PURE MATHS: a matrix of segment spectra in, the signal out. Zxx is [bins x segments] (column j = the spectrum of segment j, as scipy.signal.stft's output): one-sided
// (n/2 + 1 bins, a real signal; `nfft` says whether the length was even or odd) or two-sided (nfft bins, a complex signal). The caller's window is the analysis window (nperseg samples).
//   - each column is inverse-transformed (irfft or ifft of length nfft) and cut to nperseg samples;
//   - scaling `Spectrum` (scipy's default: the stft divided the spectrum by the window's sum) or `Psd` (divided by sqrt(fs sum w^2)), undone here;
//   - the segments are multiplied by the window again and overlap-added at hop nperseg - noverlap; the sum is divided by the overlap-added w^2 wherever that exceeds 1e-10
//     (the nonzero-overlap-add condition: a signal is recovered exactly where it holds);
//   - `boundary` true: the first and last nperseg/2 samples, which stft's zero extension added, are cut.
// Not here: the forward transform in this layout (the segment-spectrum accumulator is nxr::signal::stft in spectral.h, a Welch step).
#include <Eigen/Dense>

namespace nxr::signal {

enum class StftScaling { Spectrum, Psd };

/** Real signal from one-sided segment spectra [(nfft/2 + 1) x segments]. `window`: nperseg samples; `noverlap` < 0: nperseg / 2; `nfft` 0: 2 (bins - 1) (scipy's default; pass
 *  the true nfft for an odd one); `fs` only matters for Psd scaling. Throws std::invalid_argument on an inconsistent size. */
Eigen::VectorXd istft(const Eigen::MatrixXcd& Zxx, const Eigen::VectorXd& window, int noverlap = -1, int nfft = 0, bool boundary = true, StftScaling scaling = StftScaling::Spectrum, double fs = 1.0);
/** Complex signal from two-sided segment spectra [nfft x segments] (scipy's input_onesided=False). */
Eigen::VectorXcd istftTwoSided(const Eigen::MatrixXcd& Zxx, const Eigen::VectorXd& window, int noverlap = -1, bool boundary = true, StftScaling scaling = StftScaling::Spectrum, double fs = 1.0);

}  // namespace nxr::signal
