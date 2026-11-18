// nxr::stats: circular statistics (mean direction, mean resultant length, Rayleigh test).
#include "nxr/stats/circular.h"

#include <cmath>
#include <complex>

namespace nxr::stats {
using Eigen::VectorXd;
using cd = std::complex<double>;

double meanResultantLength(const VectorXd& a) {
    if (a.size() == 0) return 0;
    cd m(0, 0);
    for (Eigen::Index i = 0; i < a.size(); ++i) m += std::polar(1.0, a[i]);
    return std::abs(m / static_cast<double>(a.size()));
}

double meanDirection(const VectorXd& a) {
    cd m(0, 0);
    for (Eigen::Index i = 0; i < a.size(); ++i) m += std::polar(1.0, a[i]);
    return std::arg(m);
}

Rayleigh rayleighTest(const VectorXd& a) {
    Rayleigh r;
    r.n = a.size();
    r.R = meanResultantLength(a);
    r.p = std::exp(-static_cast<double>(r.n) * r.R * r.R);
    r.meanDirection = meanDirection(a);
    return r;
}

}  // namespace nxr::stats
