// nxr::signal — the bispectrum of chunk spectra weighted by frequency atoms (bispectrum.h; spec docs/specs/2026-11-24-bispectrum.md).
#include "nxr/signal/bispectrum.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <stdexcept>
#include <utility>

#include "nxr/signal/fourier.h"

namespace nxr::signal {

namespace {

using cd = std::complex<double>;

// the range [first, end) of a gain row's non-zero entries (empty when every entry is zero): the bins a sum can reach, exactly
struct Support {
  Eigen::Index first = 0, end = 0;
  Eigen::Index size() const { return end - first; }
};
Support supportOf(const Eigen::RowVectorXd& gain) {
  Support s;
  Eigen::Index i = 0;
  while (i < gain.size() && gain(i) == 0) ++i;
  if (i == gain.size()) return s;
  Eigen::Index j = gain.size();
  while (gain(j - 1) == 0) --j;
  s.first = i; s.end = j;
  return s;
}

// the linear convolution of every column of `a` [La x R] with the column of `b` [Lb x R], or with its one column [Lb x 1]: [La + Lb - 1 x R], one transform each (nxr's FFT)
Eigen::MatrixXcd convolveColumns(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
  const Eigen::Index n = a.rows() + b.rows() - 1, R = a.cols();
  const Eigen::Index L = static_cast<Eigen::Index>(next_fast_len(static_cast<std::size_t>(n), false));
  Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(L, R), B = Eigen::MatrixXcd::Zero(L, b.cols());
  A.topRows(a.rows()) = a;
  B.topRows(b.rows()) = b;
  Eigen::MatrixXcd FA = fft(A, 0);
  const Eigen::MatrixXcd FB = fft(B, 0);
  if (FB.cols() == 1) FA = FA.array().colwise() * FB.col(0).array();
  else FA = FA.cwiseProduct(FB);
  return ifft(FA, 0).topRows(n);
}

// W(i) = Σ_j u2(j) u3(i + j + firstBin) for i in [0, bins), u2 given on its support s2 ([s2.size() x R]), u3 on s3 ([s3.size() x R or 1]): [bins x R]
Eigen::MatrixXcd correlationOnFirst(const Eigen::MatrixXcd& u2, const Support& s2, const Eigen::MatrixXcd& u3, const Support& s3, Eigen::Index bins, std::int64_t firstBin) {
  Eigen::MatrixXcd W = Eigen::MatrixXcd::Zero(bins, u2.cols());
  if (s2.size() == 0 || s3.size() == 0) return W;
  const Eigen::MatrixXcd c = convolveColumns(u2.colwise().reverse(), u3);   // c(p) = Σ_q u2(end2 - 1 - q) u3(first3 + p - q)
  for (Eigen::Index i = 0; i < bins; ++i) {
    const Eigen::Index p = i - s3.first + s2.end - 1 + static_cast<Eigen::Index>(firstBin);
    if (p >= 0 && p < c.rows()) W.row(i) = c.row(p);
  }
  return W;
}
// V(s) = Σ_{i + j + firstBin = s} u1(i) u2(j) for s in [0, bins): [bins x R]
Eigen::MatrixXcd convolutionOnSum(const Eigen::MatrixXcd& u1, const Support& s1, const Eigen::MatrixXcd& u2, const Support& s2, Eigen::Index bins, std::int64_t firstBin) {
  Eigen::MatrixXcd V = Eigen::MatrixXcd::Zero(bins, std::max(u1.cols(), u2.cols()));
  if (s1.size() == 0 || s2.size() == 0) return V;
  const Eigen::MatrixXcd c = u1.cols() >= u2.cols() ? convolveColumns(u1, u2) : convolveColumns(u2, u1);
  for (Eigen::Index p = 0; p < c.rows(); ++p) {
    const Eigen::Index s = s1.first + s2.first + static_cast<Eigen::Index>(firstBin) + p;
    if (s >= bins) break;
    V.row(s) = c.row(p);
  }
  return V;
}
Eigen::VectorXcd column(const Eigen::RowVectorXd& gain, const Support& s) { return gain.segment(s.first, s.size()).transpose().cast<cd>(); }

}  // namespace

BicoherenceNormalisation bicoherenceNormalisationOf(const std::string& name) {
  if (name == "none") return BicoherenceNormalisation::none;
  if (name == "threenorm") return BicoherenceNormalisation::threenorm;
  if (name == "kim_powers") return BicoherenceNormalisation::kim_powers;
  throw std::invalid_argument("bicoherence normalisation is none, threenorm or kim_powers, not " + name);
}

AtomTriples atomTriples(const Eigen::MatrixXd& gains, std::int64_t firstBin, const std::vector<AtomTriple>& triples) {
  if (firstBin < 0) throw std::invalid_argument("atomTriples: firstBin >= 0");
  if ((gains.array() < 0).any()) throw std::invalid_argument("atomTriples: the gains are non-negative (the normalisations' bounds need it)");
  for (const auto& t : triples)
    for (const std::int64_t k : {t.first, t.second, t.sum})
      if (k < 0 || k >= gains.rows()) throw std::invalid_argument("atomTriples: a triple's gain row is outside the gains");
  AtomTriples out;
  out.gains = gains; out.firstBin = firstBin; out.triples = triples;
  const Eigen::Index bins = gains.cols(), T = static_cast<Eigen::Index>(triples.size());
  // the grouping with fewer distinct pairs: each distinct pair is one transform of every row per chunk
  std::map<std::pair<std::int64_t, std::int64_t>, std::vector<std::size_t>> bySecondSum, byFirstSecond;
  for (std::size_t t = 0; t < triples.size(); ++t) { bySecondSum[{triples[t].second, triples[t].sum}].push_back(t); byFirstSecond[{triples[t].first, triples[t].second}].push_back(t); }
  out.groupedBySecondAndSum = bySecondSum.size() <= byFirstSecond.size();
  for (auto& [key, members] : out.groupedBySecondAndSum ? bySecondSum : byFirstSecond) { (void)key; out.groups.push_back(members); }
  // the data-free parts, once per triple
  out.secondSumCorrelation = Eigen::MatrixXd::Zero(bins, T);
  out.firstSumCorrelation = Eigen::MatrixXd::Zero(bins, T);
  out.firstSecondConvolution = Eigen::MatrixXd::Zero(bins, T);
  out.weight = Eigen::VectorXd::Zero(T);
  for (Eigen::Index t = 0; t < T; ++t) {
    const AtomTriple& tr = triples[static_cast<std::size_t>(t)];
    const Eigen::RowVectorXd g1 = gains.row(tr.first), g2 = gains.row(tr.second), g3 = gains.row(tr.sum);
    const Support s1 = supportOf(g1), s2 = supportOf(g2), s3 = supportOf(g3);
    out.secondSumCorrelation.col(t) = correlationOnFirst(column(g2, s2), s2, column(g3, s3), s3, bins, firstBin).col(0).real();
    out.firstSumCorrelation.col(t) = correlationOnFirst(column(g1, s1), s1, column(g3, s3), s3, bins, firstBin).col(0).real();
    out.firstSecondConvolution.col(t) = convolutionOnSum(column(g1, s1), s1, column(g2, s2), s2, bins, firstBin).col(0).real();
    out.weight(t) = g1.dot(out.secondSumCorrelation.col(t));
  }
  return out;
}

TripleSums& TripleSums::operator+=(const TripleSums& other) {
  if (bispectrum.size() == 0) { *this = other; return *this; }
  if (other.bispectrum.rows() != bispectrum.rows() || other.bispectrum.cols() != bispectrum.cols()) throw std::invalid_argument("TripleSums: sums of different rows or triples");
  bispectrum += other.bispectrum;
  auto add = [](Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { if (a.size() == 0) a = b; else if (b.size() != 0) a += b; };
  add(firstCube, other.firstCube); add(secondCube, other.secondCube); add(sumCube, other.sumCube);
  add(productPower, other.productPower); add(sumPower, other.sumPower);
  return *this;
}

std::vector<TripleSums> bandTripleBispectrum(const AtomTriples& at, const std::vector<Eigen::MatrixXcd>& bands, BicoherenceNormalisation normalisation) {
  const Eigen::Index bins = at.gains.cols(), T = static_cast<Eigen::Index>(at.triples.size()), Q = static_cast<Eigen::Index>(bands.size());
  if (Q == 0) throw std::invalid_argument("bandTripleBispectrum: no band spectrum");
  const Eigen::Index R = bands.front().rows();
  for (const auto& b : bands)
    if (b.cols() != bins || b.rows() != R) throw std::invalid_argument("bandTripleBispectrum: each band's spectrum is [rows x " + std::to_string(bins) + " bins], the same rows");
  std::vector<Eigen::MatrixXcd> X;                                                     // [bins x rows] each: one transform per row runs down a column
  std::vector<Eigen::MatrixXd> power;
  for (const auto& b : bands) { X.push_back(b.transpose()); power.push_back(X.back().cwiseAbs2()); }
  const bool kimPowers = normalisation == BicoherenceNormalisation::kim_powers;
  auto at3 = [Q](Eigen::Index p, Eigen::Index q, Eigen::Index r) { return static_cast<std::size_t>((p * Q + q) * Q + r); };
  std::vector<TripleSums> out(static_cast<std::size_t>(Q * Q * Q));
  for (auto& s : out) { s.bispectrum = Eigen::MatrixXcd::Zero(R, T); if (kimPowers) s.productPower = Eigen::MatrixXd::Zero(R, T); }
  auto gainOf = [&](std::int64_t row) { return Eigen::RowVectorXd(at.gains.row(row)); };
  for (const auto& members : at.groups) {
    const AtomTriple& key = at.triples[members.front()];
    if (at.groupedBySecondAndSum) {
      const Eigen::RowVectorXd g2 = gainOf(key.second), g3 = gainOf(key.sum);
      const Support s2 = supportOf(g2), s3 = supportOf(g3);
      if (s2.size() == 0 || s3.size() == 0) continue;
      for (Eigen::Index q = 0; q < Q; ++q) {
        const Eigen::MatrixXcd u2 = X[q].middleRows(s2.first, s2.size()).array().colwise() * column(g2, s2).array();
        if (kimPowers) {                                                                // the product power pairs the first and second bands only
          const Eigen::MatrixXcd W2 = correlationOnFirst(power[q].middleRows(s2.first, s2.size()).cast<cd>().array().colwise() * column(g2, s2).array(), s2, column(g3, s3), s3, bins, at.firstBin);
          for (const std::size_t t : members) {
            const Eigen::VectorXd g1 = at.gains.row(at.triples[t].first).transpose();
            for (Eigen::Index p = 0; p < Q; ++p) {
              const Eigen::VectorXd v = (power[p].array().colwise() * g1.array() * W2.real().array()).colwise().sum().transpose();
              for (Eigen::Index r = 0; r < Q; ++r) out[at3(p, q, r)].productPower.col(static_cast<Eigen::Index>(t)) = v;
            }
          }
        }
        for (Eigen::Index r = 0; r < Q; ++r) {
          const Eigen::MatrixXcd u3 = (X[r].middleRows(s3.first, s3.size()).array().colwise() * column(g3, s3).array()).conjugate();
          const Eigen::MatrixXcd W = correlationOnFirst(u2, s2, u3, s3, bins, at.firstBin);   // [bins x rows], once per (second band, sum band)
          for (const std::size_t t : members) {
            const Eigen::VectorXcd g1 = at.gains.row(at.triples[t].first).transpose().cast<cd>();
            for (Eigen::Index p = 0; p < Q; ++p)
              out[at3(p, q, r)].bispectrum.col(static_cast<Eigen::Index>(t)) = (X[p].array().colwise() * g1.array() * W.array()).colwise().sum().transpose();
          }
        }
      }
    } else {
      const Eigen::RowVectorXd g1 = gainOf(key.first), g2 = gainOf(key.second);
      const Support s1 = supportOf(g1), s2 = supportOf(g2);
      if (s1.size() == 0 || s2.size() == 0) continue;
      for (Eigen::Index p = 0; p < Q; ++p)
        for (Eigen::Index q = 0; q < Q; ++q) {
          const Eigen::MatrixXcd u1 = X[p].middleRows(s1.first, s1.size()).array().colwise() * column(g1, s1).array();
          const Eigen::MatrixXcd u2 = X[q].middleRows(s2.first, s2.size()).array().colwise() * column(g2, s2).array();
          const Eigen::MatrixXcd V = convolutionOnSum(u1, s1, u2, s2, bins, at.firstBin);  // [bins x rows], once per (first band, second band)
          Eigen::MatrixXcd V2;
          if (kimPowers) V2 = convolutionOnSum(power[p].middleRows(s1.first, s1.size()).cast<cd>().array().colwise() * column(g1, s1).array(), s1,
                                               power[q].middleRows(s2.first, s2.size()).cast<cd>().array().colwise() * column(g2, s2).array(), s2, bins, at.firstBin);
          for (const std::size_t t : members) {
            const Eigen::VectorXd g3 = at.gains.row(at.triples[t].sum).transpose();
            Eigen::VectorXd product;
            if (kimPowers) product = (V2.real().array().colwise() * g3.array()).colwise().sum().transpose();
            for (Eigen::Index r = 0; r < Q; ++r) {
              out[at3(p, q, r)].bispectrum.col(static_cast<Eigen::Index>(t)) = (V.array() * (X[r].array().colwise() * g3.cast<cd>().array()).conjugate()).colwise().sum().transpose();
              if (kimPowers) out[at3(p, q, r)].productPower.col(static_cast<Eigen::Index>(t)) = product;
            }
          }
        }
    }
  }
  // the ingredients that need only the gains' own correlations and convolutions: one product [rows x bins] x [bins x triples] per band and axis
  auto weighted = [&](const Eigen::MatrixXd& values, auto gainRow, const Eigen::MatrixXd& partner) {
    Eigen::MatrixXd M(bins, T);
    for (Eigen::Index t = 0; t < T; ++t) M.col(t) = at.gains.row(gainRow(at.triples[static_cast<std::size_t>(t)])).transpose().cwiseProduct(partner.col(t));
    return Eigen::MatrixXd(values.transpose() * M);
  };
  auto first = [](const AtomTriple& t) { return t.first; };
  auto second = [](const AtomTriple& t) { return t.second; };
  auto sum = [](const AtomTriple& t) { return t.sum; };
  for (Eigen::Index b = 0; b < Q; ++b) {
    if (normalisation == BicoherenceNormalisation::threenorm) {
      const Eigen::MatrixXd cube = power[b].cwiseSqrt().cwiseProduct(power[b]);
      const Eigen::MatrixXd c1 = weighted(cube, first, at.secondSumCorrelation), c2 = weighted(cube, second, at.firstSumCorrelation), c3 = weighted(cube, sum, at.firstSecondConvolution);
      for (Eigen::Index i = 0; i < Q; ++i)
        for (Eigen::Index j = 0; j < Q; ++j) { out[at3(b, i, j)].firstCube = c1; out[at3(i, b, j)].secondCube = c2; out[at3(i, j, b)].sumCube = c3; }
    }
    if (kimPowers) {
      const Eigen::MatrixXd sp = weighted(power[b], sum, at.firstSecondConvolution);
      for (Eigen::Index i = 0; i < Q; ++i) for (Eigen::Index j = 0; j < Q; ++j) out[at3(i, j, b)].sumPower = sp;
    }
  }
  return out;
}

TripleSums atomBispectrum(const AtomTriples& at, const Eigen::MatrixXcd& spectrum, BicoherenceNormalisation normalisation) {
  if (spectrum.cols() != at.gains.cols()) throw std::invalid_argument("atomBispectrum: the spectrum has " + std::to_string(spectrum.cols()) + " bins, the gains " + std::to_string(at.gains.cols()));
  return bandTripleBispectrum(at, {spectrum}, normalisation).front();
}

Eigen::MatrixXcd randomisePhases(const Eigen::MatrixXcd& spectrum, const Eigen::VectorXd& phases) {
  if (phases.size() != spectrum.cols()) throw std::invalid_argument("randomisePhases: one phase per bin");
  Eigen::VectorXcd turn(phases.size());
  for (Eigen::Index k = 0; k < phases.size(); ++k) turn(k) = std::polar(1.0, phases(k));
  return spectrum * turn.asDiagonal();
}
Eigen::MatrixXd bicoherenceOf(const TripleSums& s, BicoherenceNormalisation normalisation) {
  const Eigen::MatrixXd magnitude = s.bispectrum.cwiseAbs();
  Eigen::MatrixXd b = magnitude;
  if (normalisation == BicoherenceNormalisation::none) return b;
  for (Eigen::Index r = 0; r < b.rows(); ++r)
    for (Eigen::Index t = 0; t < b.cols(); ++t) {
      const double n = normalisation == BicoherenceNormalisation::threenorm ? std::cbrt(s.firstCube(r, t) * s.secondCube(r, t) * s.sumCube(r, t)) : s.productPower(r, t) * s.sumPower(r, t);
      b(r, t) = n > 0 ? (normalisation == BicoherenceNormalisation::threenorm ? magnitude(r, t) / n : magnitude(r, t) * magnitude(r, t) / n) : 0.0;
    }
  return b;
}

ExceedancePValues exceedancePValues(const Eigen::VectorXd& observed, const Eigen::MatrixXd& null) {
  if (null.rows() == 0 || null.cols() != observed.size()) throw std::invalid_argument("exceedancePValues: [draws x cells] with at least one draw");
  const double D = static_cast<double>(null.rows());
  Eigen::VectorXd largest = null.rowwise().maxCoeff();
  ExceedancePValues out;
  out.p = Eigen::VectorXd::Zero(observed.size()); out.familyP = out.p;
  for (Eigen::Index c = 0; c < observed.size(); ++c) {
    out.p(c) = ((null.col(c).array() >= observed(c)).count() + 1.0) / (D + 1.0);
    out.familyP(c) = ((largest.array() >= observed(c)).count() + 1.0) / (D + 1.0);
  }
  std::vector<double> sorted(largest.data(), largest.data() + largest.size());
  std::sort(sorted.begin(), sorted.end());
  out.familyThreshold = sorted[static_cast<std::size_t>(std::min<double>(D - 1, std::ceil(0.95 * D) - 1))];
  return out;
}

}  // namespace nxr::signal
