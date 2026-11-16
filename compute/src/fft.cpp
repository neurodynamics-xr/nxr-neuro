// nxr::fft over pocketfft (deps/pocketfft). Single-threaded: no pthreads, so it is safe in the wasm and N-API builds.
#define POCKETFFT_NO_MULTITHREADING
#include "nxr/fft.h"
#include "../deps/pocketfft/pocketfft_hdronly.h"
#include <cmath>
#include <stdexcept>

namespace nxr::fft {
namespace {

pocketfft::stride_t rowMajorStrides(const Shape& shape, std::size_t elemBytes) {
    pocketfft::stride_t s(shape.size());
    std::ptrdiff_t acc = static_cast<std::ptrdiff_t>(elemBytes);
    for (std::size_t i = shape.size(); i-- > 0;) { s[i] = acc; acc *= static_cast<std::ptrdiff_t>(shape[i]); }
    return s;
}

void checkAxis(const Shape& shape, std::size_t axis, const char* who) {
    if (shape.empty() || axis >= shape.size()) throw std::invalid_argument(std::string(who) + ": axis out of range");
}

double scale(Norm norm, std::size_t n, bool forward) {
    if (n == 0) return 1.0;
    switch (norm) {
        case Norm::Backward: return forward ? 1.0 : 1.0 / static_cast<double>(n);
        case Norm::Ortho:    return 1.0 / std::sqrt(static_cast<double>(n));
        case Norm::Forward:  return forward ? 1.0 / static_cast<double>(n) : 1.0;
    }
    return 1.0;
}

std::size_t total(const Shape& s) { std::size_t p = 1; for (auto v : s) p *= v; return p; }

}  // namespace

void c2c(const cd* in, cd* out, const Shape& shape, std::size_t axis, bool forward, Norm norm) {
    checkAxis(shape, axis, "fft::c2c");
    if (total(shape) == 0) return;
    const auto st = rowMajorStrides(shape, sizeof(cd));
    pocketfft::c2c(shape, st, st, pocketfft::shape_t{axis}, forward, in, out, scale(norm, shape[axis], forward));
}

void r2c(const double* in, cd* out, const Shape& shape, std::size_t axis, Norm norm) {
    checkAxis(shape, axis, "fft::r2c");
    if (total(shape) == 0) return;
    Shape so = shape; so[axis] = shape[axis] / 2 + 1;
    pocketfft::r2c(shape, rowMajorStrides(shape, sizeof(double)), rowMajorStrides(so, sizeof(cd)), axis, true, in, out,
                   scale(norm, shape[axis], true));
}

void c2r(const cd* in, double* out, const Shape& shape, std::size_t axis, Norm norm) {
    checkAxis(shape, axis, "fft::c2r");
    if (total(shape) == 0) return;
    Shape si = shape; si[axis] = shape[axis] / 2 + 1;
    pocketfft::c2r(shape, rowMajorStrides(si, sizeof(cd)), rowMajorStrides(shape, sizeof(double)), axis, false, in, out,
                   scale(norm, shape[axis], false));
}

Eigen::VectorXd fftfreq(std::size_t n, double d) {
    Eigen::VectorXd f(static_cast<Eigen::Index>(n));
    const double v = 1.0 / (static_cast<double>(n) * d);
    const std::size_t half = (n - 1) / 2 + 1;   // numpy: N = (n-1)//2 + 1 non-negative bins
    for (std::size_t i = 0; i < n; ++i)
        f(static_cast<Eigen::Index>(i)) = (i < half ? static_cast<double>(i) : static_cast<double>(i) - static_cast<double>(n)) * v;
    return f;
}

Eigen::VectorXd rfftfreq(std::size_t n, double d) {
    const std::size_t m = n / 2 + 1;
    Eigen::VectorXd f(static_cast<Eigen::Index>(m));
    const double v = 1.0 / (static_cast<double>(n) * d);
    for (std::size_t i = 0; i < m; ++i) f(static_cast<Eigen::Index>(i)) = static_cast<double>(i) * v;
    return f;
}

std::size_t nextFastLen(std::size_t n, bool real) {
    if (n <= 6) return n == 0 ? 1 : n;
    const std::size_t primes[] = {2, 3, 5, 7, 11};
    const std::size_t np = real ? 3 : 5;
    for (std::size_t m = n;; ++m) {
        std::size_t r = m;
        for (std::size_t i = 0; i < np; ++i) while (r % primes[i] == 0) r /= primes[i];
        if (r == 1) return m;
    }
}

// Column-major [n x m] = row-major {m, n}; the transform axis is the last.
Eigen::MatrixXcd rfftColumns(const Eigen::MatrixXd& x, Norm norm) {
    const std::size_t n = static_cast<std::size_t>(x.rows()), m = static_cast<std::size_t>(x.cols());
    Eigen::MatrixXcd X(static_cast<Eigen::Index>(rfftBins(n)), x.cols());
    if (n && m) r2c(x.data(), X.data(), {m, n}, 1, norm);
    return X;
}

Eigen::MatrixXd irfftColumns(const Eigen::MatrixXcd& X, std::size_t n, Norm norm) {
    if (static_cast<std::size_t>(X.rows()) != rfftBins(n)) throw std::invalid_argument("fft::irfftColumns: rows must be n/2 + 1");
    Eigen::MatrixXd x(static_cast<Eigen::Index>(n), X.cols());
    if (n && X.cols()) c2r(X.data(), x.data(), {static_cast<std::size_t>(X.cols()), n}, 1, norm);
    return x;
}

Eigen::MatrixXcd fftColumns(const Eigen::MatrixXcd& x, bool forward, Norm norm) {
    Eigen::MatrixXcd y(x.rows(), x.cols());
    if (x.size()) c2c(x.data(), y.data(), {static_cast<std::size_t>(x.cols()), static_cast<std::size_t>(x.rows())}, 1, forward, norm);
    return y;
}

Eigen::VectorXcd rfft(const Eigen::VectorXd& x, Norm norm) { return rfftColumns(x, norm).col(0); }
Eigen::VectorXd irfft(const Eigen::VectorXcd& X, std::size_t n, Norm norm) { return irfftColumns(X, n, norm).col(0); }
Eigen::VectorXcd fft(const Eigen::VectorXcd& x, Norm norm) { return fftColumns(x, true, norm).col(0); }
Eigen::VectorXcd ifft(const Eigen::VectorXcd& X, Norm norm) { return fftColumns(X, false, norm).col(0); }

}  // namespace nxr::fft
