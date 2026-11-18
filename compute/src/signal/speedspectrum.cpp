#include "nxr/signal/speedspectrum.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::signal {

using Eigen::MatrixXd;
using Eigen::VectorXd;

Result compute(const filterbank::Bank& graph, const timebank::TimeBank& time, const VectorXd& mu, const std::vector<int>& kind, const ModeSpectra& sp, const Options& opt) {
    constexpr double kTwoPi = 6.283185307179586;
    const Eigen::Index K = mu.size(), B = sp.frequencies.size();
    if (static_cast<Eigen::Index>(kind.size()) != K) throw std::invalid_argument("speedspectrum: kind and mu differ in length");
    if (static_cast<Eigen::Index>(sp.S.size()) != B) throw std::invalid_argument("speedspectrum: one S per frequency");
    const bool haveQ = !sp.Sq.empty();
    if (haveQ && static_cast<Eigen::Index>(sp.Sq.size()) != B) throw std::invalid_argument("speedspectrum: one Sq per frequency");
    for (Eigen::Index b = 0; b < B; ++b) {
        if (!(sp.frequencies(b) > 0)) throw std::invalid_argument("speedspectrum: the frequencies are the one-sided bins, f > 0");
        if (sp.S[static_cast<std::size_t>(b)].rows() != K || sp.S[static_cast<std::size_t>(b)].cols() != K) throw std::invalid_argument("speedspectrum: S must be [K x K]");
    }
    if (opt.speedBins < 1) throw std::invalid_argument("speedspectrum: speedBins must be at least 1");
    const int G = graph.size(), T = time.size();
    if (G < 1 || T < 1) throw std::invalid_argument("speedspectrum: an empty bank");
    // the joint bank (jointbank::make) is separable here, so a member's gain at (lambda, f) is the product of its two factors
    VectorXd lambda = mu.array().square();
    const MatrixXd gg = filterbank::gains(graph, lambda).transpose();   // [G x K] (the library's is [K x G])
    const MatrixXd ht = timebank::gains(time, sp.frequencies);       // [B x T]
    // the speed range
    double cmin = opt.speedMin, cmax = opt.speedMax;
    if (!(cmin > 0) || !(cmax > cmin)) {
        double lo = std::numeric_limits<double>::infinity(), hi = 0;
        for (Eigen::Index k = 0; k < K; ++k) if (mu(k) > 0) for (Eigen::Index b = 0; b < B; ++b) {
            const double c = kTwoPi * sp.frequencies(b) / mu(k);
            lo = std::min(lo, c); hi = std::max(hi, c);
        }
        if (!(hi > 0)) throw std::invalid_argument("speedspectrum: no mode with mu > 0 and no frequency");
        cmin = opt.speedMin > 0 ? opt.speedMin : lo;
        cmax = opt.speedMax > cmin ? opt.speedMax : std::max(hi, cmin * 1.0000001) * 1.0000001;
    }
    Result r;
    const int nb = opt.speedBins;
    r.speedEdges.resize(nb + 1);
    r.speedCentres.resize(nb);
    const double l0 = std::log(cmin), dl = (std::log(cmax) - l0) / nb;
    for (int i = 0; i <= nb; ++i) r.speedEdges(i) = std::exp(l0 + dl * i);
    for (int i = 0; i < nb; ++i) r.speedCentres(i) = std::exp(l0 + dl * (i + 0.5));
    r.graphCentres = filterbank::centerWavenumbers(graph);
    r.timeCentres = timebank::centerFrequencies(time);
    for (int ki : {0, 2}) {
        std::vector<Eigen::Index> idx;
        for (Eigen::Index k = 0; k < K; ++k) if (kind[static_cast<std::size_t>(k)] == ki) idx.push_back(k);
        if (idx.empty()) continue;
        Part p;
        p.kind = ki;
        p.modes = static_cast<int>(idx.size());
        p.tileEnergy = MatrixXd::Zero(G, T);
        p.tileSpeed.resize(G, T);
        p.quadrature = MatrixXd::Constant(G, T, std::numeric_limits<double>::quiet_NaN());
        p.histogram = MatrixXd::Zero(G, nb);
        p.outside = MatrixXd::Zero(G, 2);
        for (int g = 0; g < G; ++g) for (int t = 0; t < T; ++t) p.tileSpeed(g, t) = r.graphCentres(g) > 0 ? kTwoPi * r.timeCentres(t) / r.graphCentres(g) : std::numeric_limits<double>::quiet_NaN();
        // A_t = sum_f h_t(f)^2 S(f) (and Im), restricted to the kind's modes
        const Eigen::Index n = static_cast<Eigen::Index>(idx.size());
        std::vector<MatrixXd> Are(T, MatrixXd::Zero(n, n)), Aim(T, MatrixXd::Zero(n, n));
        for (Eigen::Index b = 0; b < B; ++b) {
            VectorXd pf(n);
            for (Eigen::Index a = 0; a < n; ++a) pf(a) = sp.S[static_cast<std::size_t>(b)](idx[static_cast<std::size_t>(a)], idx[static_cast<std::size_t>(a)]);
            p.power += pf.sum();
            for (Eigen::Index a = 0; a < n; ++a) {
                const Eigen::Index k = idx[static_cast<std::size_t>(a)];
                if (!(mu(k) > 0)) continue;
                const double c = kTwoPi * sp.frequencies(b) / mu(k);
                const int bin = static_cast<int>(std::floor((std::log(c) - l0) / dl));
                for (int g = 0; g < G; ++g) {
                    const double e = gg(g, k) * gg(g, k) * pf(a);
                    if (c < cmin || bin < 0) p.outside(g, 0) += e;
                    else if (c >= r.speedEdges(nb) || bin >= nb) p.outside(g, 1) += e;
                    else p.histogram(g, bin) += e;
                }
            }
            for (int t = 0; t < T; ++t) {
                const double h2 = ht(b, t) * ht(b, t);
                if (h2 == 0) continue;
                for (int g = 0; g < G; ++g) for (Eigen::Index a = 0; a < n; ++a) p.tileEnergy(g, t) += h2 * gg(g, idx[static_cast<std::size_t>(a)]) * gg(g, idx[static_cast<std::size_t>(a)]) * pf(a);
                for (Eigen::Index a = 0; a < n; ++a) for (Eigen::Index c2 = 0; c2 < n; ++c2) {
                    Are[static_cast<std::size_t>(t)](a, c2) += h2 * sp.S[static_cast<std::size_t>(b)](idx[static_cast<std::size_t>(a)], idx[static_cast<std::size_t>(c2)]);
                    if (haveQ) Aim[static_cast<std::size_t>(t)](a, c2) += h2 * sp.Sq[static_cast<std::size_t>(b)](idx[static_cast<std::size_t>(a)], idx[static_cast<std::size_t>(c2)]);
                }
            }
        }
        if (haveQ) for (int t = 0; t < T; ++t) {
            const MatrixXd im2 = Aim[static_cast<std::size_t>(t)].array().square(), all2 = Are[static_cast<std::size_t>(t)].array().square() + im2.array();
            for (int g = 0; g < G; ++g) {
                VectorXd w(n);
                for (Eigen::Index a = 0; a < n; ++a) w(a) = gg(g, idx[static_cast<std::size_t>(a)]) * gg(g, idx[static_cast<std::size_t>(a)]);
                const double num = w.dot(im2 * w), den = w.dot(all2 * w);
                p.quadrature(g, t) = den > 0 ? std::sqrt(std::max(0.0, num) / den) : std::numeric_limits<double>::quiet_NaN();
            }
        }
        r.parts.push_back(std::move(p));
    }
    return r;
}

}  // namespace nxr::signal
