// nxr::stats: circular statistics (mean direction, mean resultant length, Rayleigh test).
#include "nxr/stats/circular.h"

#include <cmath>
#include <complex>

namespace nxr::stats {
using Eigen::VectorXd;
using cd = std::complex<double>;

double circ_r(const VectorXd& a) {
    if (a.size() == 0) return 0;
    cd m(0, 0);
    for (Eigen::Index i = 0; i < a.size(); ++i) m += std::polar(1.0, a[i]);
    return std::abs(m / static_cast<double>(a.size()));
}

double circ_mean(const VectorXd& a) {
    cd m(0, 0);
    for (Eigen::Index i = 0; i < a.size(); ++i) m += std::polar(1.0, a[i]);
    return std::arg(m);
}

Rayleigh circ_rtest(const VectorXd& a) {
    Rayleigh r;
    r.n = a.size();
    r.R = circ_r(a);
    r.p = std::exp(-static_cast<double>(r.n) * r.R * r.R);
    r.meanDirection = circ_mean(a);
    return r;
}

}  // namespace nxr::stats
