#pragma once
// nxr::signal — IIR filter DESIGN: `butter`, the Butterworth filter, as MATLAB's butter and SciPy's scipy.signal.butter (output = 'ba').
//
// PURE MATHS: numbers in, polynomial coefficients out; no signal is filtered here. Conventions are MATLAB's and SciPy's:
//   - the order N and the edge(s) Wn in units of the NYQUIST frequency (Wn = f / (fs / 2), 0 < Wn < 1); one edge for a low- or high-pass, two for a band-pass or band-stop
//     (a band-pass or band-stop of order N has 2N poles);
//   - the analogue prototype's poles, the frequency transformation (lp2lp, lp2hp, lp2bp, lp2bs) on the frequency pre-warped by tan(pi Wn / 2), the bilinear transform;
//   - the result is the transfer function  H(z) = (b[0] + b[1] z^-1 + ... ) / (a[0] + a[1] z^-1 + ... ), a[0] = 1, the gain 1 at the passband's reference
//     (DC for a low-pass, Nyquist for a high-pass, the centre of the passband for a band-pass, DC for a band-stop).
// A transfer function in b, a loses precision as the order grows (the polynomial's coefficients, not the poles, are stored): use it up to order ~10; a higher order wants the
// second-order sections (`butterSos`) or the zeros, poles and gain (`butterZpk`), which are scipy.signal.butter's output = 'sos' and 'zpk'.
#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace nxr::signal {

/** The type of a classic frequency-selective filter: MATLAB's 'low' | 'high' | 'bandpass' | 'stop'. */
enum class Band { Lowpass, Highpass, Bandpass, Bandstop };

struct TransferFunction {
  Eigen::VectorXd b;   ///< numerator, descending powers of z (z^0 first as a delay line: b[0] + b[1] z^-1 + ...)
  Eigen::VectorXd a;   ///< denominator, a[0] = 1
};

/** The zeros, the poles and the gain of a digital filter: H(z) = k prod (1 - z_i z^-1) / prod (1 - p_i z^-1). */
struct ZeroPoleGain {
  std::vector<std::complex<double>> z, p;
  double k = 1.0;
};

/** Butterworth filter of order `order` >= 1 with the edge(s) `Wn` in units of Nyquist: one value (Lowpass, Highpass) or two ascending (Bandpass, Bandstop).
 *  MATLAB butter(order, Wn, type), scipy.signal.butter(order, Wn, btype, output='ba'). Throws std::invalid_argument on a bad order, edge count or edge. */
TransferFunction butter(int order, const std::vector<double>& Wn, Band type);
/** The same filter as zeros, poles and gain, in scipy.signal.butter(order, Wn, btype, output='zpk')'s order (the roots of each prototype pole, "+" before "-"; the zeros at -1, +1 or +-j on the unit circle last).
 *  Same arguments and errors as `butter`. */
ZeroPoleGain butterZpk(int order, const std::vector<double>& Wn, Band type);
/** Second-order sections [n x 6], a row b0 b1 b2 a0 a1 a2 (a0 = 1), the gain in the first row, the sections ordered as scipy.signal.zpk2sos(z, p, k) with pairing 'nearest' (the digital default): the pole closest
 *  to the unit circle is paired last. scipy.signal.butter(order, Wn, btype, output='sos'). Same arguments and errors as `butter`. */
Eigen::MatrixXd butterSos(int order, const std::vector<double>& Wn, Band type);
/** scipy.signal.zpk2sos(z, p, k, pairing='nearest', analog=False): the sections of a digital filter given by its zeros, poles (in conjugate pairs) and gain. std::invalid_argument for a complex root without its conjugate. */
Eigen::MatrixXd zpkToSos(const std::vector<std::complex<double>>& zeros, const std::vector<std::complex<double>>& poles, double gain);

}  // namespace nxr::signal
