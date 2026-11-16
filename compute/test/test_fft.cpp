// nxr::fft against numpy.fft (goldens from golden/gen_fft_golden.py: random inputs, numpy's outputs) and its own round trips.
#include "nxr/fft.h"
#include "mesh_helpers.h"
#include <fstream>
#include <sstream>
#include <string>
using namespace nxr::fft;

static std::vector<double> readValues(std::istream& is, const std::string& tag) {
    std::string t; is >> t;
    if (t != tag) { std::cout << "  [FAIL] golden: expected '" << tag << "', got '" << t << "'\n"; ++g_failures; return {}; }
    std::string line; std::getline(is, line);
    std::istringstream ls(line); std::vector<double> v; double x;
    while (ls >> x) v.push_back(x);
    return v;
}
static std::string sci(double v) { std::ostringstream s; s.precision(2); s << std::scientific << v; return s.str(); }
static double relErr(const double* a, const double* b, size_t n) {   // max |a-b| / max|b|
    double d = 0, m = 0;
    for (size_t i = 0; i < n; ++i) { d = std::max(d, std::abs(a[i] - b[i])); m = std::max(m, std::abs(b[i])); }
    return d / std::max(m, 1e-300);
}

int main() {
    std::cout << "\n=== nxr::fft vs numpy.fft ===\n";
    std::ifstream f(std::string(NXR_TEST_GOLDEN_DIR) + "/fft_golden.txt");
    EXPECT(f.good(), "golden file opens");
    std::string tok; int cases = 0; double worst = 0;
    while (f >> tok) {
        if (tok == "case") {
            std::string kind, normS; size_t axis, nd; f >> kind >> normS >> axis >> nd;
            Shape shape(nd); size_t total = 1; for (auto& s : shape) { f >> s; total *= s; }
            Norm norm = normS == "ortho" ? Norm::Ortho : normS == "forward" ? Norm::Forward : Norm::Backward;
            auto in = readValues(f, "in"); auto want = readValues(f, "out");
            std::vector<double> got;
            Shape sh2 = shape; sh2[axis] = shape[axis] / 2 + 1; size_t bins = total / shape[axis] * sh2[axis];
            if (kind == "c2c_fwd" || kind == "c2c_inv") {
                std::vector<cd> a(total), y(total);
                for (size_t i = 0; i < total; ++i) a[i] = {in[2*i], in[2*i+1]};
                c2c(a.data(), y.data(), shape, axis, kind == "c2c_fwd", norm);
                for (auto z : y) { got.push_back(z.real()); got.push_back(z.imag()); }
            } else if (kind == "r2c") {
                std::vector<cd> y(bins);
                r2c(in.data(), y.data(), shape, axis, norm);
                for (auto z : y) { got.push_back(z.real()); got.push_back(z.imag()); }
            } else {   // c2r
                std::vector<cd> a(bins); std::vector<double> y(total);
                for (size_t i = 0; i < bins; ++i) a[i] = {in[2*i], in[2*i+1]};
                c2r(a.data(), y.data(), shape, axis, norm);
                got = y;
            }
            bool sized = got.size() == want.size();
            double e = sized ? relErr(got.data(), want.data(), got.size()) : 1.0;
            worst = std::max(worst, e); ++cases;
            if (!sized || e > 1e-12) { std::cout << "  [FAIL] " << kind << " " << normS << " axis " << axis << " n " << shape[axis] << " rel err " << e << "\n"; ++g_failures; }
        } else if (tok == "freq") {
            size_t n; double d; f >> n >> d;
            auto ff = readValues(f, "fftfreq"); auto rf = readValues(f, "rfftfreq");
            auto a = fftfreq(n, d), b = rfftfreq(n, d);
            bool ok = (size_t)a.size() == ff.size() && (size_t)b.size() == rf.size();
            for (size_t i = 0; ok && i < ff.size(); ++i) ok = std::abs(a(i) - ff[i]) <= 1e-15 * std::max(1.0, std::abs(ff[i]));
            for (size_t i = 0; ok && i < rf.size(); ++i) ok = std::abs(b(i) - rf[i]) <= 1e-15 * std::max(1.0, std::abs(rf[i]));
            EXPECT(ok, "fftfreq / rfftfreq(" + std::to_string(n) + ", " + std::to_string(d) + ") == numpy");
        }
    }
    EXPECT(cases > 60 && g_failures == 0, std::to_string(cases) + " numpy cases (c2c, r2c, c2r, three norms, 1-D and 2/3-D axes), worst relative error " + sci(worst) + " <= 1e-12");

    std::cout << "=== round trips and helpers ===\n";
    {   // ifft(fft(x)) == x; irfft(rfft(x), n) == x for odd and even n; Parseval under ortho
        for (size_t n : {1u, 6u, 7u, 128u, 255u}) {
            Eigen::VectorXd x = Eigen::VectorXd::Random(n);
            double e = (irfft(rfft(x), n) - x).cwiseAbs().maxCoeff();
            EXPECT(e < 1e-13, "irfft(rfft(x), " + std::to_string(n) + ") == x, err " + std::to_string(e));
            Eigen::VectorXcd z = Eigen::VectorXcd::Random(n);
            double e2 = (ifft(fft(z)) - z).cwiseAbs().maxCoeff();
            EXPECT(e2 < 1e-13, "ifft(fft(z)) == z, n " + std::to_string(n) + ", err " + std::to_string(e2));
        }
        Eigen::VectorXd x = Eigen::VectorXd::Random(100);
        double pe = (rfft(x, Norm::Ortho).cwiseAbs2().sum() * 2 - std::norm(rfft(x, Norm::Ortho)(0)) - std::norm(rfft(x, Norm::Ortho)(50))) - x.squaredNorm();
        EXPECT(std::abs(pe) < 1e-12, "ortho: Parseval (one-sided sum doubled) err " + std::to_string(pe));
    }
    {   // column helper: one signal per column of a column-major matrix
        Eigen::MatrixXd X = Eigen::MatrixXd::Random(50, 4);
        Eigen::MatrixXcd C = rfftColumns(X);
        EXPECT(C.rows() == 26 && C.cols() == 4, "rfftColumns shape [n/2+1 x m]");
        double e = 0; for (int j = 0; j < 4; ++j) e = std::max(e, (C.col(j) - rfft(X.col(j))).cwiseAbs().maxCoeff());
        EXPECT(e < 1e-13, "rfftColumns == rfft of each column");
        EXPECT((irfftColumns(C, 50) - X).cwiseAbs().maxCoeff() < 1e-13, "irfftColumns(rfftColumns(X), n) == X");
        EXPECT((fftColumns(fftColumns(X.cast<std::complex<double>>()), false) - X.cast<std::complex<double>>()).cwiseAbs().maxCoeff() < 1e-13, "fftColumns round trip");
    }
    EXPECT(nextFastLen(13, true) == 15 && nextFastLen(17, true) == 18 && nextFastLen(97, true) == 100 && nextFastLen(7, false) == 7 && nextFastLen(1031, false) == 1050,
           "nextFastLen (scipy: real {2,3,5}, complex {2,3,5,7,11})");
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
