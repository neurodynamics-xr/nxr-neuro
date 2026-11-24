// nxr::signal::timefilterbank — see include/nxr/timebank.h and docs/specs/2026-11-10-filterbanks.md.
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/graphtransforms.h"
#include "nxr/signal/filters_fir.h"
#include "nxr/signal/fourier.h"
#include "nxr/stats/bandwidth.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nxr::signal::timefilterbank {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace fb = nxr::signal::graphfilterbank;
namespace gt = nxr::signal::graphtransforms;
static constexpr double kPi = 3.14159265358979323846;

namespace {
double angle(double x, double e, double h) { return (kPi / 2) * fb::meyerNu((std::log2(x / e) + h) / (2 * h)); }
}

double meyerBand(double x, double lo, double hi, double h) {
    if (!(x > 0)) return lo > 0 ? 0.0 : 1.0;
    const double rise = lo > 0 ? std::sin(angle(x, lo, h)) : 1.0;
    const double fall = std::isfinite(hi) ? std::cos(angle(x, hi, h)) : 1.0;
    return rise * fall;
}

double morletBand(double x, double lo, double hi) {
    if (!(hi > lo)) throw std::invalid_argument("timefilterbank::morletBand: needs lo < hi");
    const double f0 = (lo + hi) / 2, sigma = (hi - lo) / (2 * std::sqrt(2 * std::log(2.0)));
    const double z = (std::fabs(x) - f0) / sigma;
    return std::exp(-0.5 * z * z);
}

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Family bandFamilyOf(const std::string& name) {
    if (name == "morlet") return Family::Morlet;
    if (name == "power_partition") return Family::PowerPartition;
    throw std::invalid_argument("the band shape is morlet or power_partition, not '" + name + "'");
}
Family familyOf(const std::string& name) {
    if (name == "itersine") return Family::Itersine;
    if (name == "meyer") return Family::Meyer;
    if (name == "morlet") return Family::Morlet;
    if (name == "morse") return Family::Morse;
    if (name == "amor") return Family::Amor;
    if (name == "bump") return Family::Bump;
    if (name == "power_partition") return Family::PowerPartition;
    if (name == "gabor") return Family::Gabor;
    throw std::invalid_argument("time bank family '" + name + "' is not one of itersine, meyer, morlet, morse, amor, bump, power_partition, gabor");
}
Sides sidesOf(const std::string& name) {
    if (name == "analytic") return Sides::Analytic;
    if (name == "two_sided") return Sides::TwoSided;
    throw std::invalid_argument("time bank sides '" + name + "' is not one of analytic, two_sided");
}
const char* bandFamilyName(Family family) { return family == Family::PowerPartition ? "power_partition" : "morlet"; }

MatrixXd bandGains(const VectorXd& freqs, const std::vector<std::pair<double, double>>& bands, Family family) {
  if (family != Family::Morlet && family != Family::PowerPartition) throw std::invalid_argument("bandGains: the shapes are Morlet and PowerPartition");
  const Eigen::Index nb = static_cast<Eigen::Index>(bands.size()), nf = freqs.size();
  MatrixXd g = MatrixXd::Zero(nb, nf);
  if (bands.empty()) return g;
  for (const auto& b : bands) if (!(b.second > b.first) || !(b.first > 0)) throw std::invalid_argument("a band needs 0 < lo < hi");
  if (family == Family::Morlet) {
    for (Eigen::Index b = 0; b < nb; ++b)
      for (Eigen::Index k = 0; k < nf; ++k) g(b, k) = freqs(k) == 0 ? 0.0 : morletBand(std::abs(freqs(k)), bands[b].first, bands[b].second);
    return g;
  }
  // nxr band_gains: edges in log2 f, a cos/sin crossover half an octave wide centred on each shared edge, the outer edges taper to zero the same way
  std::vector<double> edges{bands[0].first};
  for (const auto& b : bands) edges.push_back(b.second);
  for (Eigen::Index b = 0; b + 1 < nb; ++b)
    if (std::abs(bands[b].second - bands[b + 1].first) > 1e-12 * bands[b].second) throw std::invalid_argument("power_partition: the bands must be adjacent (each starts where the last ends)");
  const double half = 0.25;
  for (Eigen::Index k = 0; k < nf; ++k) {
    const double f = std::abs(freqs(k));
    if (!(f > 0)) continue;
    const double lf = std::log2(std::max(f, 1e-12));
    auto rise = [&](double e) { const double t = std::clamp((lf - (e - half)) / (2 * half), 0.0, 1.0); return std::sin(0.5 * M_PI * t); };
    for (Eigen::Index b = 0; b < nb; ++b) {
      const double up = rise(std::log2(edges[b])), r = rise(std::log2(edges[b + 1]));
      g(b, k) = up * std::sqrt(std::clamp(1 - r * r, 0.0, 1.0));
    }
  }
  return g;
}

static std::vector<double> logCentres(double fs, double f0, double f1, int voices) {
    f1 = std::min(f1, fs / 2.5);
    f0 = std::min(std::max(f0, 1e-3), f1);
    const int F = std::max(1, static_cast<int>(std::floor(voices * std::log2(f1 / f0) + 1e-9)) + 1);
    std::vector<double> c;
    for (int k = 0; k < F; ++k) c.push_back(f0 * std::pow(2.0, static_cast<double>(k) / voices));
    return c;
}

static std::vector<double> cwtCentres(double fmin, double fmax, int voices) {
    if (!(fmin > 0 && fmax >= fmin && voices >= 1)) throw std::invalid_argument("cwtCentres: 0 < fmin <= fmax and voices >= 1");
    std::vector<double> c;
    for (int k = 0;; ++k) {
        const double f = fmax * std::pow(2.0, -static_cast<double>(k) / voices);
        if (f < fmin * (1 - 1e-12)) break;
        c.push_back(f);
    }
    std::reverse(c.begin(), c.end());
    return c;
}

std::vector<double> centerFrequencies(CentreGrid grid, double lo, double hi, int voices, double fs) {
  return grid == CentreGrid::Studio ? logCentres(fs, lo, hi, voices) : cwtCentres(lo, hi, voices);
}

Freqz freqz(const TimeFilterbank& bank) { return {gains(bank), bins(bank)}; }

GaborGrid gaborGrid(double sigmaMin, double sigmaMax, int perOctave, double fMin, double fMax, double minFSigma) {
    if (!(sigmaMin > 0) || !(sigmaMax >= sigmaMin)) throw std::invalid_argument("gaborGrid: 0 < sigmaMin <= sigmaMax (s)");
    if (perOctave < 1) throw std::invalid_argument("gaborGrid: at least one width an octave");
    if (!(fMin >= 0) || !(fMax >= fMin)) throw std::invalid_argument("gaborGrid: 0 <= fMin <= fMax (Hz)");
    if (!(minFSigma >= 0)) throw std::invalid_argument("gaborGrid: minFSigma (f·σ) is at least 0");
    GaborGrid g;
    const int K = static_cast<int>(std::floor(perOctave * std::log2(sigmaMax / sigmaMin) + 1e-9));
    for (int k = 0; k <= K; ++k) g.widths.push_back(sigmaMin * std::pow(2.0, static_cast<double>(k) / perOctave));
    for (int k = 0; k <= K; ++k) {
        const double s = g.widths[static_cast<std::size_t>(k)], sf = 1.0 / (2 * kPi * s), f0 = std::max(fMin - sf, minFSigma / s);
        if (f0 > fMax) continue;
        // numpy arange(f0, fMax + sf + 1e-9, sf / 2): the count ceil((stop - start) / step)
        const double stop = fMax + sf + 1e-9, step = sf / 2;
        const std::int64_t n = static_cast<std::int64_t>(std::ceil((stop - f0) / step));
        for (std::int64_t i = 0; i < n; ++i) { g.atoms.push_back({s, f0 + static_cast<double>(i) * step}); g.widthOf.push_back(k); }
    }
    return g;
}

Gain1 bandGain(const BandShape& shape) {
    const double lo = shape.lo, hi = shape.hi;
    if (!(hi > lo)) throw std::invalid_argument("bandGain: a band is [lo, hi] with lo < hi");
    if (shape.family == "morlet") return [lo, hi](double f) { return morletBand(f, lo, hi); };
    if (shape.family == "meyer") {
        if (!shape.halfWidth) throw std::invalid_argument("bandGain: a meyer band states its half-width (octaves)");
        const double halfWidth = *shape.halfWidth;
        return [lo, hi, halfWidth](double f) { return meyerBand(f, lo, hi, halfWidth); };
    }
    if (shape.family == "morlet_ladder") {
        if (!shape.ladderMinimum || !shape.ladderMaximum || !shape.voices) throw std::invalid_argument("bandGain: a morlet_ladder band states its ladder's minimum, maximum and voices");
        return morletLadderGain(lo, hi, *shape.ladderMinimum, *shape.ladderMaximum, *shape.voices);
    }
    throw std::invalid_argument("bandGain: the band shape '" + shape.family + "' is not morlet, meyer or morlet_ladder");
}

std::function<double(double)> morletLadderGain(double lo, double hi, double ladderMin, double ladderMax, int voices) {
    if (!(ladderMin > 0) || !(ladderMax > ladderMin) || voices < 1) throw std::invalid_argument("morletLadderGain: 0 < ladderMin < ladderMax and voices >= 1");
    const int K = static_cast<int>(std::lround(voices * std::log2(ladderMax / ladderMin)));
    Design d(Family::Morlet, 1.0, Sides::Analytic);
    for (int m = 0; m <= K; ++m) d.centres.push_back(ladderMin * std::pow(2.0, static_cast<double>(m) / voices));
    d.cycles = 1 / (std::pow(2.0, 1.0 / (2 * voices)) - std::pow(2.0, -1.0 / (2 * voices)));
    const TimeFilterbank bank = timeBank(d, 16, 2 * ladderMax * 4);       // the members' gains are continuous functions of f: the grid plays no part
    auto sum = [bank](double f, double flo, double fhi) {
        double s = 0;
        for (const Member& m : bank.members) if (m.fc >= flo && m.fc <= fhi) { const double v = m.gain(f); s += v * v; }
        return s;
    };
    double norm = 0;
    const int n = 1 << 16;
    for (int i = 0; i <= n; ++i) { const double f = 2 * ladderMin + (ladderMax / 2 - 2 * ladderMin) * i / n; norm = std::max(norm, sum(f, 0, std::numeric_limits<double>::infinity())); }
    return [sum, lo, hi, norm](double f) { const double a = std::abs(f); return a > 0 ? sum(a, lo, hi) / norm : 0.0; };
}

std::size_t gridLength(std::size_t n, Padding padding) {
    if (n == 0) throw std::invalid_argument("timebank: an empty record");
    return padding == Padding::None ? n : next_fast_len(2 * n, true);
}

MatrixXd extendSymmetric(const MatrixXd& x) {
    MatrixXd e(2 * x.rows(), x.cols());
    e.topRows(x.rows()) = x;
    e.bottomRows(x.rows()) = x.colwise().reverse();
    return e;
}

std::vector<double> windowEdges(std::size_t nfft, double fs, const std::optional<std::pair<double, double>>& limits) {
    std::vector<double> edges;
    for (int k = 0;; ++k) {
        const double e = fs * std::pow(2.0, k) / static_cast<double>(nfft);
        if (e > fs / 4 * (1 + 1e-9)) break;
        if (limits && (e < limits->first * (1 - 1e-9) || e > limits->second * (1 + 1e-9))) continue;
        edges.push_back(e);
    }
    if (edges.empty()) throw std::invalid_argument("timebank: no octave edge of the window within the range");
    return edges;
}

namespace {

double itersine(double u) { return std::abs(u) <= 1 ? std::sin(0.5 * kPi * std::pow(std::cos(0.5 * kPi * u), 2)) : 0.0; }

// the three continuous-wavelet mothers in frequency, peak 2 (cwtfilterbank's bandpass normalisation); x = ω_peak · f / fc
double morse(double x, double beta, double gamma) {
    if (!(x > 0)) return 0.0;
    const double loga = std::log(2.0) + (beta / gamma) * (1.0 + std::log(gamma / beta));
    return std::exp(loga + beta * std::log(x) - std::pow(x, gamma));
}
double amor(double x) { return x > 0 ? 2.0 * std::exp(-0.5 * (x - 6) * (x - 6)) : 0.0; }
double bump(double x) {
    const double u = (x - 5.0) / 0.6;
    return std::abs(u) < 1 ? 2.0 * std::exp(1.0 - 1.0 / (1.0 - u * u)) : 0.0;
}

}  // namespace

TimeFilterbank timeBank(const Design& d, std::size_t nfft, double fs) {
    if (nfft < 2 || !(fs > 0)) throw std::invalid_argument("timebank::design: nfft >= 2 and fs > 0");
    if (!(d.gain > 0) || !std::isfinite(d.gain)) throw std::invalid_argument("timebank::design: the gain is the peak in-band response, a positive number");
    const bool analytic = d.sides == Sides::Analytic;
    const double G = d.gain;                 // the peak in-band response of every member, every family
    const double sc = G;                     // tight families: unit-sum gains times G (so Σ g² = G²)
    TimeFilterbank b;
    b.family = d.family; b.gain = d.gain; b.sides = d.sides; b.edgeBins = d.edgeBins; b.nfft = nfft; b.fs = fs;
    b.normalisation = d.family == Family::Gabor ? Normalisation::UnitEnergy : Normalisation::Peak;

    // a member given on |f|: analytic members live on f > 0; a two-sided bank splits them into one at +f and one at -f
    auto add = [&](Role role, double fc, double lo, double hi, std::function<double(double)> onAbs) {
        auto make = [&](int s) {
            Member m; m.role = role; m.sign = s; m.fc = fc; m.lo = lo; m.hi = hi;
            m.gain = [onAbs, s](double f) { return (s > 0 ? f > 0 : f < 0) ? onAbs(std::abs(f)) : 0.0; };
            b.members.push_back(std::move(m));
        };
        make(+1);
        if (!analytic) make(-1);
    };
    // the low-pass / scaling member. Analytic: one member on f >= 0. Two-sided: a member at +f and one at -f (a rotation too slow for the first band keeps its
    // sense), and the DC bin alone in a member of its own (it has no sense), so every bin has S = 1.
    auto addLow = [&](Role role, double lo, double hi, std::function<double(double)> onAbs) {
        if (analytic) {
            Member m; m.role = role; m.sign = 0; m.lo = lo; m.hi = hi;
            m.gain = [onAbs](double f) { return f >= 0 ? onAbs(f) : 0.0; };
            b.members.push_back(std::move(m));
            return;
        }
        Member dc; dc.role = Role::Dc; dc.sign = 0; dc.fc = 0.0; dc.lo = 0.0; dc.hi = 0.0;
        dc.gain = [onAbs](double f) { return f == 0.0 ? onAbs(0.0) : 0.0; };
        b.members.push_back(std::move(dc));
        add(role, kNaN, lo, hi, onAbs);
    };

    switch (d.family) {
    case Family::Itersine: {
        const int V = d.voices; const double a = d.anchor;
        if (V < 1 || !(a > 0)) throw std::invalid_argument("timebank::design: voices >= 1 and anchor > 0");
        const double T = static_cast<double>(nfft) / fs;
        const double fLoRep = d.minBins / ((std::pow(2.0, 1.0 / V) - std::pow(2.0, -1.0 / V)) * T);
        const double fLo = d.limits ? std::max(d.limits->first, fLoRep) : fLoRep;
        const double fHi = d.limits ? std::min(d.limits->second, fs / 2) : fs / 2;
        const int kLo = static_cast<int>(std::ceil(V * std::log2(fLo / a)));
        const int kHi = static_cast<int>(std::floor(V * std::log2(fHi / a))) - 1;
        if (kHi < kLo) throw std::invalid_argument("timebank::design: no band member fits between the limits at this many voices");
        auto xi = [V, a](double af) { return V * std::log2(std::max(af, std::numeric_limits<double>::min()) / a); };
        const double K0 = kLo, KE = kHi;
        addLow(Role::Lowpass, 0.0, a * std::pow(2.0, K0 / V),
                  [=](double af) { const double x = xi(af); return x < K0 ? sc * std::sqrt(std::max(1.0 - std::pow(itersine(x - K0), 2), 0.0)) : 0.0; });
        for (int k = kLo; k <= kHi; ++k)
            add(Role::Band, a * std::pow(2.0, static_cast<double>(k) / V), a * std::pow(2.0, (k - 1.0) / V), a * std::pow(2.0, (k + 1.0) / V),
                [=](double af) { return sc * itersine(xi(af) - k); });
        add(Role::Highpass, kNaN, a * std::pow(2.0, KE / V), fs / 2,
            [=](double af) { const double x = xi(af); return x > KE ? sc * std::sqrt(std::max(1.0 - std::pow(itersine(x - KE), 2), 0.0)) : 0.0; });
        b.total = G * G;
        break;
    }
    case Family::Meyer: {
        if (!d.bands.empty()) {   // the Signal chain's single bands (the Studio's meyerBand), not the frame: one member a band, not tight
            for (const auto& [lo, hi] : d.bands) {
                if (!(lo >= 0 && hi > lo)) throw std::invalid_argument("timebank::design: a Meyer band is [lo, hi] Hz, 0 <= lo < hi");
                const double hw = d.halfWidth;
                add(Role::Band, std::sqrt(std::max(lo, 1e-12) * hi), lo, hi, [sc, lo, hi, hw](double af) { return sc * meyerBand(af, lo, hi, hw); });
            }
            b.total = kNaN;
            break;
        }
        const std::vector<double> edges = d.edges.empty() ? windowEdges(nfft, fs, d.limits) : d.edges;
        const fb::GraphFilterbank dy = fb::dyadic(edges, d.halfWidth);
        for (const auto& m : dy.members) {
            const gt::Gain g = m.gain;
            auto onAbs = [g, sc](double af) { return sc * g(af); };
            const double lo = m.lo, hi = m.hi;
            if (m.role == fb::Role::Scaling) addLow(Role::Scaling, 0.0, hi, onAbs);
            else if (m.role == fb::Role::Wavelet) add(Role::Wavelet, std::sqrt(lo * hi), lo, hi, onAbs);
            else add(Role::Top, kNaN, lo, fs / 2, onAbs);
        }
        b.total = G * G;
        break;
    }
    case Family::PowerPartition: {
        if (d.bands.empty()) throw std::invalid_argument("timebank::design: PowerPartition needs bands");
        const std::vector<std::pair<double, double>> bands = d.bands;
        for (std::size_t k = 0; k < bands.size(); ++k) {
            const Eigen::VectorXd probe = Eigen::VectorXd::Ones(1);
            (void)bandGains(probe, std::vector<std::pair<double, double>>{bands[k]}, Family::PowerPartition);   // each band needs 0 < lo < hi
        }
        if (bands.size() > 1) (void)bandGains(Eigen::VectorXd::Ones(1), bands, Family::PowerPartition);          // adjacent, ascending
        for (std::size_t k = 0; k < bands.size(); ++k) {
            add(Role::Band, std::sqrt(bands[k].first * bands[k].second), bands[k].first, bands[k].second, [bands, k, sc](double af) {
                if (!(af > 0)) return 0.0;
                return sc * bandGains(Eigen::VectorXd::Constant(1, af), bands, Family::PowerPartition)(static_cast<Eigen::Index>(k), 0);
            });
        }
        b.total = kNaN;
        break;
    }
    case Family::Gabor: {
        // the unit-energy atom ψ = (πσ²)^(−1/4) e^(−t²/2σ²) e^(i2πft): Ψ(ν) = (πσ²)^(−1/4) √(2π) σ e^(−2π²σ²(ν−f)²), on ν > 0 (analytic), times G
        if (!analytic) throw std::invalid_argument("timebank::design: Gabor atoms are analytic (one-sided) only");
        if (d.atoms.empty()) throw std::invalid_argument("timebank::design: Gabor needs atoms (see gaborGrid)");
        for (const auto& [s, f] : d.atoms) {
            if (!(s > 0) || !(f > 0)) throw std::invalid_argument("timebank::design: a Gabor atom needs σ > 0 and f > 0");
            const double c = G * std::pow(kPi * s * s, -0.25) * std::sqrt(2 * kPi) * s, sf = 1.0 / (2 * kPi * s);
            Member m; m.role = Role::Band; m.sign = +1; m.fc = f; m.lo = std::max(0.0, f - sf); m.hi = f + sf; m.width = s;
            m.gain = [c, s, f](double nu) { return nu > 0 ? c * std::exp(-2 * kPi * kPi * s * s * (nu - f) * (nu - f)) : 0.0; };
            b.members.push_back(std::move(m));
        }
        b.total = kNaN;
        break;
    }
    case Family::Morlet: case Family::Morse: case Family::Amor: case Family::Bump: {
        if (d.family == Family::Morlet && !d.bands.empty()) {   // the fixed-band Morlet: one member a band
            for (const auto& [lo, hi] : d.bands) {
                if (!(lo > 0 && hi > lo)) throw std::invalid_argument("timebank::design: a band needs 0 < lo < hi");
                add(Role::Band, 0.5 * (lo + hi), lo, hi, [sc, lo, hi](double af) { return sc * morletBand(af, lo, hi); });
            }
            b.total = kNaN;
            break;
        }
        if (d.centres.empty()) throw std::invalid_argument("timebank::design: Morlet / Morse / Amor / Bump need centres (see centerFrequencies(CentreGrid, ...))");
        for (double fc : d.centres) {
            if (!(fc > 0)) throw std::invalid_argument("timebank::design: centres must be positive");
            std::function<double(double)> onAbs;
            if (d.family == Family::Morlet) {
                const double cyc = d.cycles, w = fc / cyc;
                if (!analytic) {   // nxr's joint.morlet: a Gaussian about the SIGNED centre ±fc over the whole two-sided axis (its tail across f = 0 is e^(−cycles²/2), kept)
                    for (int sgn : {+1, -1}) {
                        Member m; m.role = Role::Band; m.sign = sgn; m.fc = fc;
                        m.gain = [G, fc, w, sgn](double f) { return G * std::exp(-0.5 * std::pow((f - sgn * fc) / w, 2)); };
                        b.members.push_back(std::move(m));
                    }
                    continue;
                }
                onAbs = [G, fc, w](double af) { return G * std::exp(-0.5 * std::pow((af - fc) / w, 2)); };
            } else if (d.family == Family::Morse) {
                const double gam = d.gamma, beta = d.timeBandwidth / d.gamma, wp = std::pow(beta / gam, 1.0 / gam);
                onAbs = [G, fc, gam, beta, wp](double af) { return 0.5 * G * morse(wp * af / fc, beta, gam); };
            } else if (d.family == Family::Amor) {
                onAbs = [G, fc](double af) { return 0.5 * G * amor(6.0 * af / fc); };
            } else {
                onAbs = [G, fc](double af) { return 0.5 * G * bump(5.0 * af / fc); };
            }
            add(Role::Band, fc, kNaN, kNaN, onAbs);
        }
        break;
    }
    }

    // support bins (analytic grid)
    if (analytic) {
        const VectorXd f = rfftfreq(nfft, 1.0 / fs);
        const std::size_t nb = static_cast<std::size_t>(f.size());
        b.support.assign(b.members.size(), {0, nb - 1});
        for (std::size_t m = 0; m < b.members.size(); ++m) {
            const Member& mem = b.members[m];
            std::size_t i1 = nb, i2 = 0;
            if (d.family == Family::Itersine) {   // Rheome's bins: the open interval (lo, hi); the low-pass from DC to the first centre, the high-pass from the last centre to Nyquist
                const std::size_t M = b.members.size();
                for (std::size_t k = 0; k < nb; ++k) {
                    if (mem.role == Role::Lowpass) { i1 = 0; if (f(k) < b.members[1].fc) i2 = k; }
                    else if (mem.role == Role::Highpass) { i2 = nb - 1; if (i1 == nb && f(k) > b.members[M - 2].fc) i1 = k; }
                    else if (f(k) > mem.lo && f(k) < mem.hi) { if (i1 == nb) i1 = k; i2 = k; }
                }
            } else {
                for (std::size_t k = 0; k < nb; ++k) if (mem.gain(f(k)) != 0.0) { if (i1 == nb) i1 = k; i2 = k; }
            }
            if (i1 == nb) { i1 = 0; i2 = 0; }
            b.support[m] = {i1, i2};
        }
    } else {
        b.support.assign(b.members.size(), {0, nfft - 1});
    }
    return b;
}

VectorXd bins(const TimeFilterbank& b) {
    return b.sides == Sides::Analytic ? rfftfreq(b.nfft, 1.0 / b.fs) : fftfreq(b.nfft, 1.0 / b.fs);
}

MatrixXd gains(const TimeFilterbank& b, const VectorXd& f) {
    MatrixXd G(f.size(), b.size());
    for (int m = 0; m < b.size(); ++m)
        for (Eigen::Index k = 0; k < f.size(); ++k) G(k, m) = b.members[static_cast<std::size_t>(m)].gain(f(k));
    return G;
}

// the bins of an analytic grid that are real and unmirrored: DC, and Nyquist for an even length
static bool isEdgeBin(const TimeFilterbank& b, Eigen::Index k) {
    return b.sides == Sides::Analytic && (k == 0 || (b.nfft % 2 == 0 && static_cast<std::size_t>(k) == b.nfft / 2));
}

MatrixXd gains(const TimeFilterbank& b) {
    MatrixXd G = gains(b, bins(b));
    for (Eigen::Index k = 0; k < G.rows(); ++k)
        if (isEdgeBin(b, k)) G.row(k) *= (b.edgeBins == EdgeBins::Drop ? 0.0 : 1.0 / std::sqrt(2.0));
    return G;
}

VectorXd frameOperator(const TimeFilterbank& b) { return gains(b).array().square().rowwise().sum(); }

MatrixXd dualGains(const TimeFilterbank& b) {
    const MatrixXd G = gains(b);
    const VectorXd S = G.array().square().rowwise().sum();
    const double tol = 1e-12 * S.maxCoeff();
    MatrixXd D = G;
    for (Eigen::Index k = 0; k < G.rows(); ++k) {
        if (S(k) <= tol) D.row(k).setZero(); else D.row(k) /= S(k);
    }
    return D;
}

fb::Bounds framebounds(const TimeFilterbank& b) {
    const VectorXd S = frameOperator(b);
    double A = std::numeric_limits<double>::infinity(), B = 0, unc = 0; Eigen::Index count = 0;
    for (Eigen::Index k = 0; k < S.size(); ++k) {
        if (isEdgeBin(b, k)) continue;
        A = std::min(A, S(k)); B = std::max(B, S(k)); if (S(k) <= 1e-12) ++unc; ++count;
    }
    fb::Bounds r;
    r.A = A; r.B = B; r.tightness = B / std::max(A, 2.220446049250313e-16); r.uncovered = count ? unc / static_cast<double>(count) : 0.0;
    return r;
}

bool isframetight(const TimeFilterbank& b, double tol) { const fb::Bounds r = framebounds(b); return std::abs(r.B / std::max(r.A, 2.220446049250313e-16) - 1) <= tol; }

VectorXd centerFrequencies(const TimeFilterbank& b) {
    VectorXd fc(b.size());
    const VectorXd f = bins(b);
    const MatrixXd G = gains(b);
    for (int m = 0; m < b.size(); ++m) {
        const Member& mem = b.members[static_cast<std::size_t>(m)];
        if (!std::isnan(mem.fc)) { fc(m) = mem.fc; continue; }
        double num = 0, den = 0;
        const auto [i1, i2] = b.support[static_cast<std::size_t>(m)];
        for (std::size_t k = (b.sides == Sides::Analytic ? i1 : 0); k <= (b.sides == Sides::Analytic ? i2 : b.nfft - 1); ++k) {
            const double g2 = G(static_cast<Eigen::Index>(k), m) * G(static_cast<Eigen::Index>(k), m);
            num += std::abs(f(static_cast<Eigen::Index>(k))) * g2; den += g2;
        }
        fc(m) = den > 0 ? num / den : kNaN;
    }
    return fc;
}

std::pair<VectorXd, std::vector<std::size_t>> rates(const TimeFilterbank& b, double oversample) {
    if (b.sides != Sides::Analytic) throw std::invalid_argument("timebank::rates: own-rate sub-bands are defined for the analytic bank");
    VectorXd r(b.size()); std::vector<std::size_t> Nm;
    for (int m = 0; m < b.size(); ++m) {
        const auto [i1, i2] = b.support[static_cast<std::size_t>(m)];
        const std::size_t nb = i2 - i1 + 1, n = next_fast_len(static_cast<std::size_t>(std::ceil(oversample * static_cast<double>(nb))), true);
        Nm.push_back(n); r(m) = static_cast<double>(n) * b.fs / static_cast<double>(b.nfft);
    }
    return {r, Nm};
}


std::size_t haloOfTaps(const TimeFilterbank& bank, int member) {
  if (member < 0 || member >= bank.size()) throw std::invalid_argument("haloOf: no member " + std::to_string(member));
  const Gain1 g = bank.members[static_cast<std::size_t>(member)].gain;
  const auto onAbs = [g](double f) { return g(std::abs(f)); };
  return std::max(nxr::signal::gainTaps(onAbs, bank.fs, false).size(), nxr::signal::gainTaps(onAbs, bank.fs, true).size()) - 1;
}

std::size_t haloOf(const TimeFilterbank& bank, int member, double tolerance) {
  if (member < 0 || member >= bank.size()) throw std::invalid_argument("haloOf: no member " + std::to_string(member));
  if (!(tolerance > 0) || !(tolerance < 1)) throw std::invalid_argument("haloOf: the tolerance is in (0, 1)");
  using i64 = std::int64_t;
  constexpr i64 kMaxReach = i64(1) << 16;
  const std::size_t m = static_cast<std::size_t>(member);
  // the impulse response of the member's masked gain on a fine grid (zero outside the member's support on the bank's grid), the grid doubled while the tail has not decayed
  const double df = bank.fs / static_cast<double>(bank.nfft);
  const double lo = static_cast<double>(bank.support[m].first) * df - 1e-9 * df, hi = static_cast<double>(bank.support[m].second) * df + 1e-9 * df;
  std::size_t g = nxr::signal::next_fast_len(std::max<std::size_t>(4 * bank.nfft, 4096), true);
  i64 reach = 0;
  for (;;) {
    VectorXd f = VectorXd::LinSpaced(static_cast<Eigen::Index>(g / 2 + 1), 0, static_cast<double>(g / 2)) * (bank.fs / static_cast<double>(g));
    MatrixXd G = gains(bank, f);
    Eigen::VectorXcd Y = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(g));
    for (Eigen::Index i = 0; i < G.rows(); ++i) if (f(i) >= lo && f(i) <= hi) Y(i) = G(i, member);
    const Eigen::VectorXcd h = nxr::signal::ifft(Eigen::MatrixXcd(Y)).col(0);
    VectorXd a = VectorXd::Zero(static_cast<Eigen::Index>(g / 2 + 1));   // sum of |h| at |t| = i, both sides
    double total = 0;
    for (std::size_t i = 0; i < g; ++i) { const std::size_t t = std::min(i, g - i); a(static_cast<Eigen::Index>(t)) += std::abs(h(static_cast<Eigen::Index>(i))); total += std::abs(h(static_cast<Eigen::Index>(i))); }
    double tail = 0; reach = static_cast<i64>(g / 2);
    for (i64 t = static_cast<i64>(g / 2); t >= 0; --t) {      // grow the tail from the far end while it stays under the tolerance
      if (tail + a(t) > tolerance * total) { reach = t; break; }
      tail += a(t); reach = t;
    }
    if (reach < static_cast<i64>(g / 2) - 1 || static_cast<i64>(g) >= 2 * kMaxReach) break;
    g = nxr::signal::next_fast_len(2 * g, true);
  }
  return static_cast<std::size_t>(std::max<i64>(1, std::min<i64>(reach, kMaxReach)));
}

double equivalentBandwidth(const TimeFilterbank& bank, int member, int grid) {
  if (member < 0 || member >= bank.size()) throw std::invalid_argument("equivalentBandwidth: no member " + std::to_string(member));
  if (grid < 16) throw std::invalid_argument("equivalentBandwidth: grid >= 16");
  const Gain1& g = bank.members[static_cast<std::size_t>(member)].gain;
  const double df = bank.fs / 2 / (grid - 1);
  VectorXd p(grid);
  for (int i = 0; i < grid; ++i) { const double v = g(i * df); p(i) = v * v; }
  return nxr::stats::statisticalBandwidth(p, df, nxr::stats::Rule::Trapezoid);
}

double noiseBandwidth(const TimeFilterbank& bank, int member, int grid) {
  if (member < 0 || member >= bank.size()) throw std::invalid_argument("noiseBandwidth: no member " + std::to_string(member));
  if (grid < 16) throw std::invalid_argument("noiseBandwidth: grid >= 16");
  const Gain1& g = bank.members[static_cast<std::size_t>(member)].gain;
  const double df = bank.fs / 2 / (grid - 1);
  VectorXd p(grid);
  for (int i = 0; i < grid; ++i) { const double v = g(i * df) / 2; p(i) = v * v; }
  return nxr::stats::enbw(p, df, 1.0, nxr::stats::Rule::Trapezoid);
}

double independentEstimates(const TimeFilterbank& bank, int member, std::int64_t samples) {
  if (samples <= 0) throw std::invalid_argument("independentEstimates: a window of at least one sample");
  return equivalentBandwidth(bank, member) * static_cast<double>(samples) / bank.fs;
}


}  // namespace nxr::signal::timefilterbank
