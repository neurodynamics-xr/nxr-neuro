// nxr-pipelines — THE BISPECTRUM OF THE CHUNK SPECTRA, AND THE FLOW ACROSS FREQUENCIES, as graph nodes (spec docs/specs/2026-11-24-bispectrum.md; definitions `sensor_bispectrum`,
// `sensor_flow_across_frequencies`).
//
// signal.bispectrum: the stored chunk spectra (signal.chunk_spectra: each chunk's spectrum in a surface's field coordinates, the cross-spectrum in its factored form) + the frequency atoms
//   (a kernel set) [+ the Hodge-Dirac package and eigenvalue bands, for the reading on the surface] --> per triple of atoms (one gain per axis: the first frequency, the second, their sum)
//   and per coordinate (or per vertex of the surface, per band) the bispectrum summed over the chunks and the atoms' pairs of bins, with its normalisation's ingredients (sums; every
//   bicoherence, biphase and asymmetry is derived at read). Readings: phase_amplitude_coupling (low, high, high), harmonics (atom, atom, the atom at twice its frequency).
// signal.flow_across_frequencies: the flow readout's stored per-atom results (signal.spectral_flow: edges, modes, vertices) + the package + the atoms --> per pair of atoms the cosines of
//   their flows (edges, travel, rotation), per vertex the direction consistency across atoms and the slope of the apparent speed against frequency.
// Every node is a function of arrays: no file, no subject, no recording (the chunk spectra and the flow are arrays handed over by the runner).
#include "nxr/pipelines/bispectrum_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/pipelines/form_steps.h"
#include "chunk_spectra_readers.h"
#include "nxr/signal/bispectrum.h"
#include "nxr/signal/flow_across_frequencies.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/spectral_flow.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;
using cd = std::complex<double>;

namespace {

const ng::Array& handed(const ng::Arrays& c, const char* port, const char* who) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(who) + ": " + port + " was not handed over");
  return it->second;
}
bool given(const ng::Arrays& c, const char* port) { const auto it = c.find(port); return it != c.end() && !it->second.meta.value("placeholder", false); }
std::string text_of(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_string()) ng::refuse(std::string(who) + ": " + key + " has no default: name it");
  return p[key].get<std::string>();
}
// the description an array carries: its own (made in this graph) or the stored statistic's (shape and producer at the top)
Json shape_of(const ng::Array& a) { return a.meta.contains("shape") ? a.meta["shape"] : a.meta.value("describe", Json::object()).value("shape", Json()); }
Json producer_of(const ng::Array& a) { return a.meta.contains("producer") ? a.meta["producer"] : a.meta.value("describe", Json::object()).value("producer", Json::object()); }
Json indexed_axis(const char* name, std::int64_t size, const Json& values = Json()) {
  Json v = values;
  if (v.is_null()) { v = Json::array(); for (std::int64_t i = 0; i < size; ++i) v.push_back(i); }
  return Json{{"name", name}, {"size", size}, {"values", v}};
}

// ── the chunk spectra as signal.chunk_spectra stores them: [chunks x bins x coordinates x (re, im)] with the transform's line in the producer ──
// (the same reading as sensor_flow_steps.cpp's; one shared reader once the sensor-flow nodes settle: issue filed)
struct ChunkSpectra {
  std::int64_t chunks = 0, bins = 0, rows = 0, first = 0, transform = 0;
  double sfreq = 0;
  std::vector<std::int64_t> modes;
  std::vector<double> mu;
  bool unit_mass = true;
  const ng::Array* array = nullptr;
  double frequency(std::int64_t k) const { return static_cast<double>(first + k) * sfreq / static_cast<double>(transform); }
  Eigen::MatrixXcd chunk(std::int64_t t) const {                     // [coordinates x bins]
    Eigen::MatrixXcd X(rows, bins);
    for (std::int64_t k = 0; k < bins; ++k)
      for (std::int64_t r = 0; r < rows; ++r) {
        const std::size_t i = static_cast<std::size_t>(((t * bins + k) * rows + r) * 2);
        X(r, k) = cd(array->at(i), array->at(i + 1));
      }
    return X;
  }
};
ChunkSpectra chunk_spectra_of(const ng::Array& A, const char* who) {
  const Json shape = shape_of(A), pr = producer_of(A);
  if (!shape.is_array() || shape.size() != 4 || shape[3] != 2) ng::refuse(std::string(who) + ": spectra are not [chunks x bins x coordinates x 2] (signal.chunk_spectra)");
  for (const char* k : {"sfreq", "transform_samples", "first_bin", "modes"}) if (!pr.contains(k)) ng::refuse(std::string(who) + ": the spectra do not record " + k);
  ChunkSpectra s;
  s.chunks = shape[0]; s.bins = shape[1]; s.rows = shape[2];
  s.sfreq = pr["sfreq"]; s.transform = pr["transform_samples"]; s.first = pr["first_bin"];
  s.modes = pr["modes"].get<std::vector<std::int64_t>>();
  s.mu = pr.value("mu", std::vector<double>());
  s.unit_mass = pr.value("normalisation", Json("unit_mass")) != Json("as_stored");
  if (static_cast<std::int64_t>(s.modes.size()) != s.rows) ng::refuse(std::string(who) + ": the spectra's coordinates are not their modes");
  if (static_cast<std::int64_t>(A.size()) != s.chunks * s.bins * s.rows * 2) ng::refuse(std::string(who) + ": the spectra hold " + std::to_string(A.size()) + " values, not their shape's");
  s.array = &A;
  return s;
}
// the coordinates' synthesis Psi [S x K] (the constant 1/sqrt(area); each mode at unit star0 norm under unit_mass) and their eigenvalues
std::pair<Eigen::MatrixXd, Eigen::VectorXd> synthesis_of(const ChunkSpectra& s, const nxr::signal::Modes& m, const Eigen::VectorXd& mass, const char* who) {
  const Eigen::Index S = m.P.rows(), K = static_cast<Eigen::Index>(s.modes.size());
  if (mass.size() != S) ng::refuse(std::string(who) + ": the package's vertex mass does not fit its vertices");
  Eigen::MatrixXd Psi(S, K);
  Eigen::VectorXd mu(K);
  for (Eigen::Index j = 0; j < K; ++j) {
    const std::int64_t k = s.modes[static_cast<std::size_t>(j)];
    if (k < 0) { Psi.col(j).setConstant(1.0 / std::sqrt(mass.sum())); mu(j) = 0.0; continue; }
    if (k >= m.P.cols()) ng::refuse(std::string(who) + ": a coordinate's mode is outside the package's modes");
    Psi.col(j) = m.P.col(k);
    if (s.unit_mass) Psi.col(j) /= std::sqrt(Psi.col(j).cwiseAbs2().dot(mass));
    mu(j) = m.mu(k);
  }
  return {Psi, mu};
}
// the atoms' gains at the points x, normalised as the parameter says (the same choice as signal.spectral_flow's, so the atoms are the flow's)
Eigen::MatrixXd atom_gains(const ng::Array& atoms, const Eigen::VectorXd& x, const Json& p, const char* who) {
  Eigen::MatrixXd g = spectra_readers::kernel_gains_of(atoms, x, who, "frequency").gains;
  const std::string normalisation = text_of(p, "frequency_normalisation", who);
  if (normalisation == "sum_of_squared_gains") {
    if (!p.contains("support_floor") || !p["support_floor"].is_number()) ng::refuse(std::string(who) + ": support_floor has no default: name it");
    g = nxr::signal::normaliseBySumOfSquaredGains(g, p["support_floor"].get<double>());
  } else if (normalisation != "none") ng::refuse(std::string(who) + ": frequency_normalisation is sum_of_squared_gains or none");
  return g;
}
std::vector<double> centres_of(const ng::Array& atoms, const char* who) {
  // each atom's centre as the one evaluator reports it (a band family's: the midpoint of [lo, hi], never its low edge)
  const Eigen::VectorXd centres = spectra_readers::kernel_gains_of(atoms, Eigen::VectorXd(0), who, "frequency").centres;
  return std::vector<double>(centres.data(), centres.data() + centres.size());
}

const char* const kWarnings[] = {
    "the field on the helmet only (or on the surface the coordinates belong to): no source, no field spread, no cortical claim",
    "a sum over the chunks and the atoms' pairs of bins: coupling that changes its phase over the run cancels (a whole-run statistic; single events are the burst pipelines')",
    "each sample counted twice (an overlap of half a chunk on each side); the chunks' transforms are rectangular cuts (the atoms are the taper)",
    "the recording's two ends are reflected into the first and last transforms' overlaps: time runs backwards there, which reverses a rise-decay asymmetry in those samples",
    "phase-amplitude coupling is read only where the high atom is wider than the low frequency (Zandvoort and Nolte 2021): a narrow high atom gives a small value that is not an absence "
    "of coupling (each triple's weight is stored: 0 where no pair of bins carries it)",
    "a harmonic whose centre (twice the atom's) lies above the span is not read (left out of the triples); bicoherence at (f, f, 2f) is a harmonic or a coupling of f with itself: it does not tell them apart",
    "bicoherence is biased upward by few chunks: compare with a null (surrogates or noise of the same length), never with 0",
    "per coordinate: each coordinate with itself only (no coupling between coordinates is read); on the surface: each vertex's own field (no coupling between vertices)"};

// ── signal.bispectrum ──────────────────────────────────────────────────────────────────────────────────────────────
const char* const kBispectrum = "signal.bispectrum";
struct BispectrumPrepared {
  ng::Array spectra;
  ng::Array atoms;                    // the atoms: a stored kernel set (select.kernels) on the frequency line
  std::size_t atom_count = 0;
  Eigen::MatrixXd bands;              // [bands x K] (empty: the whole field)
  std::shared_ptr<const nxr::signal::IntrinsicComplex> complex;
  std::shared_ptr<const nxr::signal::Modes> modes;
  Json p;
};

// the parameters, checked before any data (by plan and again by prepare, which a `compute` call reaches without a plan)
void check_bispectrum_parameters(const Json& p) {
  if (!p.contains("readings") || !p["readings"].is_array() || p["readings"].empty()) ng::refuse(std::string(kBispectrum) + ": readings has no default: name a list of phase_amplitude_coupling, harmonics");
  for (const auto& r : p["readings"]) if (r != "phase_amplitude_coupling" && r != "harmonics") ng::refuse(std::string(kBispectrum) + ": a reading is phase_amplitude_coupling or harmonics");
  try { nxr::signal::bicoherenceNormalisationOf(text_of(p, "normalisation", kBispectrum)); } catch (const std::invalid_argument& e) { ng::refuse(std::string(kBispectrum) + ": " + e.what()); }
  const std::string space = text_of(p, "space", kBispectrum);
  if (space != "coordinates" && space != "surface" && space != "band_triples") ng::refuse(std::string(kBispectrum) + ": space is coordinates, surface or band_triples");
  const std::string fn = text_of(p, "frequency_normalisation", kBispectrum);
  if (fn == "sum_of_squared_gains" && !(p.contains("support_floor") && p["support_floor"].is_number())) ng::refuse(std::string(kBispectrum) + ": support_floor has no default: name it");
}

ng::NodeSpec bispectrum() {
  ng::NodeSpec n;
  n.name = kBispectrum;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in a surface's field coordinates (signal.chunk_spectra), stored or made in the same graph", false},
              {"frequency", "the frequency atoms: a kernel set on the spectra's frequency line (select_kernels), one gain per axis taken from it", false},
              {"package", "the Hodge-Dirac package of the surface the coordinates are on (needed with space surface)", true},
              {"scale", "the eigenvalue bands, a kernel set on the eigenvalue line (space surface only; absent: the whole field)", true}};
  n.outputs = {{"bispectrum", "per element (coordinate, or vertex), per triple of atoms (x band on the surface), the bispectrum's real and imaginary parts and its normalisation's sums", false}};
  n.parameters = {
      {"readings", Json(), "the triples read, a list of: phase_amplitude_coupling (low atom, high atom, high atom: every pair with the high centre above the low) | harmonics (atom, atom, "
                           "the atom at twice its frequency: the harmonic and the waveform's asymmetry) (no default)", "-", {}, false},
      {"normalisation", Json(), "the sums kept for the bicoherence: threenorm (Shahbazi Avarvand et al. 2018: the cube sums) | kim_powers (Kim and Powers 1979, Hagihira et al. 2001: the product "
                                "and sum powers) | none (no default)", "-", {"threenorm", "kim_powers", "none"}, false},
      {"space", Json(), "where the triple products are read: coordinates (each field coordinate with itself) | surface (each vertex of the surface: the coordinates synthesised, per band "
                        "when bands are given; the synthesis composed with the band's gains first) | band_triples (each vertex, every TRIPLE of eigenvalue bands: the first axis read on band "
                        "p's field, the second on band q's, the sum on band r's: coupling across spatial scales) (no default)", "-", {"coordinates", "surface", "band_triples"}, false},
      {"frequency_normalisation", Json(), "the atoms as signal.spectral_flow weights them: sum_of_squared_gains | none (no default)", "-", {"sum_of_squared_gains", "none"}, false},
      {"support_floor", Json(), "with sum_of_squared_gains: as signal.spectral_flow's (no default)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run, chunk by chunk of the stored spectra: the triple products of the atoms' bins summed over every pair of bins and chunk (a sum: the statistic's definition), "
                             "one correlation or convolution along the frequency line per distinct pair of axes");
  n.matches = "the direct (FFT-based) bispectrum (Hagihira et al. 2001; Sigl and Chamoun 1994) weighted by atoms; threenorm bicoherence (Shahbazi Avarvand et al. 2018), Kim and Powers (1979); "
              "bispectral phase-amplitude coupling (Zandvoort and Nolte 2021); biphase and waveform shape (Bartz et al. 2019); compute nxr::signal::atomBispectrum";
  n.description = "the bispectrum of the chunk spectra weighted by frequency atoms (one gain per axis), per field coordinate or per vertex of the surface: phase-amplitude coupling and harmonics, "
                  "with the normalisation's sums; bicoherence, biphase and asymmetry derived at read";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "frequency"}) if (!in.count(k)) ng::refuse(std::string(kBispectrum) + ": " + k + " is not given");
    check_bispectrum_parameters(p);
    const std::string space = p["space"].get<std::string>();
    if (space != "coordinates" && !in.count("package")) ng::refuse(std::string(kBispectrum) + ": space " + space + " needs the package (the coordinates' synthesis)");
    if (space == "band_triples" && !in.count("scale")) ng::refuse(std::string(kBispectrum) + ": space band_triples needs the eigenvalue bands (scale)");
    if (space == "coordinates" && in.count("scale")) ng::refuse(std::string(kBispectrum) + ": bands apply on the surface only (per coordinate, a band is a selection of the coordinate rows at read)");
    ng::Planned out;
    ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "statistic"; out.out["bispectrum"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    check_bispectrum_parameters(p);
    BispectrumPrepared pr;
    pr.spectra = handed(c, "spectra", kBispectrum);
    pr.atoms = handed(c, "frequency", kBispectrum);
    pr.atom_count = spectra_readers::kernelSetOf(pr.atoms, kBispectrum, "frequency").members.size();
    pr.p = p;
    if (p.at("space") != "coordinates") {
      const BoundPackage b = bind_hodge_package(handed(c, "package", kBispectrum), kBispectrum);
      pr.complex = b.complex; pr.modes = b.modes;
      if (given(c, "scale")) {
        const ChunkSpectra s = chunk_spectra_of(pr.spectra, kBispectrum);
        const auto [Psi, mu] = synthesis_of(s, *pr.modes, pr.complex->star0, kBispectrum);
        (void)Psi;
        pr.bands = spectra_readers::kernel_gains_of(c.at("scale"), mu, kBispectrum, "scale").gains;
        for (Eigen::Index j = 0; j < mu.size(); ++j) if (!(mu(j) > 0) && pr.bands.col(j).sum() == 0) pr.bands(0, j) = 1.0;   // the constant lies in the coarsest band
      }
    }
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const BispectrumPrepared& pr = std::any_cast<const BispectrumPrepared&>(prepared);
    const ChunkSpectra s = chunk_spectra_of(pr.spectra, kBispectrum);
    Eigen::VectorXd f(s.bins);
    for (std::int64_t k = 0; k < s.bins; ++k) f(k) = s.frequency(k);
    const Eigen::Index A = static_cast<Eigen::Index>(pr.atom_count);
    // the gain rows: the atoms [0, A), then each atom at twice its frequency [A, 2A) (its gain at f / 2), the same normalisation
    Eigen::MatrixXd gains(2 * A, s.bins);
    gains.topRows(A) = atom_gains(pr.atoms, f, pr.p, kBispectrum);
    gains.bottomRows(A) = atom_gains(pr.atoms, f / 2.0, pr.p, kBispectrum);
    const std::vector<double> centres = centres_of(pr.atoms, kBispectrum);
    std::vector<nxr::signal::AtomTriple> triples;
    Json listed = Json::array();
    for (const auto& reading : pr.p.at("readings")) {
      if (reading == "phase_amplitude_coupling")
        for (Eigen::Index lo = 0; lo < A; ++lo)
          for (Eigen::Index hi = 0; hi < A; ++hi)
            if (centres[static_cast<std::size_t>(hi)] > centres[static_cast<std::size_t>(lo)]) {
              triples.push_back({lo, hi, hi});
              listed.push_back(Json{{"reading", "phase_amplitude_coupling"}, {"first", lo}, {"second", hi}, {"sum", hi}, {"low_hz", centres[static_cast<std::size_t>(lo)]}, {"high_hz", centres[static_cast<std::size_t>(hi)]}});
            }
      if (reading == "harmonics")
        for (Eigen::Index a = 0; a < A; ++a) {
          if (!(2 * centres[static_cast<std::size_t>(a)] < s.frequency(s.bins - 1))) continue;   // the harmonic's centre above the span: not read (only its gain's tail would be)
          triples.push_back({a, a, A + a});
          listed.push_back(Json{{"reading", "harmonics"}, {"first", a}, {"second", a}, {"sum", A + a}, {"fundamental_hz", centres[static_cast<std::size_t>(a)]}, {"harmonic_hz", 2 * centres[static_cast<std::size_t>(a)]}});
        }
    }
    const nxr::signal::AtomTriples plan = nxr::signal::atomTriples(gains, s.first, triples);
    for (std::size_t t = 0; t < triples.size(); ++t) listed[t]["weight"] = plan.weight(static_cast<Eigen::Index>(t));
    const auto normalisation = nxr::signal::bicoherenceNormalisationOf(pr.p.at("normalisation").get<std::string>());
    const std::string space = pr.p.at("space").get<std::string>();
    const bool surface = space != "coordinates", bandTriples = space == "band_triples";
    // the operators from the coordinates to the rows read: none (each coordinate), or the synthesis composed with each band's gains (one [S x K] each, applied to the coefficients)
    std::vector<Eigen::MatrixXd> operators;
    if (surface) {
      const auto [Psi, mu] = synthesis_of(s, *pr.modes, pr.complex->star0, kBispectrum);
      (void)mu;
      if (pr.bands.rows() == 0) operators.push_back(Psi);
      for (Eigen::Index q = 0; q < pr.bands.rows(); ++q) operators.push_back(Psi * pr.bands.row(q).transpose().asDiagonal());
    }
    const Eigen::Index Q = static_cast<Eigen::Index>(operators.size());
    const Eigen::Index groups = !surface ? 1 : bandTriples ? Q * Q * Q : Q;              // the members beside the triples: each band (surface), each band triple (band_triples)
    // one chunk at a time: its coefficients, or each band's field (the composed operator applied to them), summed into the triples
    std::vector<nxr::signal::TripleSums> sums(static_cast<std::size_t>(groups));
    for (std::int64_t t = 0; t < s.chunks; ++t) {
      const Eigen::MatrixXcd X = s.chunk(t);
      std::vector<Eigen::MatrixXcd> chunk;
      if (!surface) chunk.push_back(X);
      else for (const auto& op : operators) chunk.push_back(op.cast<cd>() * X);
      if (bandTriples) {
        const std::vector<nxr::signal::TripleSums> each = nxr::signal::bandTripleBispectrum(plan, chunk, normalisation);
        for (std::size_t g = 0; g < each.size(); ++g) sums[g] += each[g];
      } else {
        for (std::size_t q = 0; q < chunk.size(); ++q) sums[q] += nxr::signal::atomBispectrum(plan, chunk[q], normalisation);
      }
    }
    const Eigen::Index R = sums[0].bispectrum.rows(), T = static_cast<Eigen::Index>(triples.size()), cells = T * groups;
    // the stored array: element x (triple, member: the triple fastest) x measure
    std::vector<std::string> measures = {"bispectrum_real", "bispectrum_imaginary"};
    if (normalisation == nxr::signal::BicoherenceNormalisation::threenorm) for (const char* m : {"first_cube", "second_cube", "sum_cube"}) measures.push_back(m);
    if (normalisation == nxr::signal::BicoherenceNormalisation::kim_powers) for (const char* m : {"product_power", "sum_power"}) measures.push_back(m);
    const Eigen::Index Mn = static_cast<Eigen::Index>(measures.size());
    ng::Array out;
    out.rows = R * cells; out.cols = Mn;
    out.data.assign(static_cast<std::size_t>(R * cells * Mn), 0.0);
    for (Eigen::Index q = 0; q < groups; ++q) {
      const nxr::signal::TripleSums& g = sums[static_cast<std::size_t>(q)];
      for (Eigen::Index r = 0; r < R; ++r)
        for (Eigen::Index t = 0; t < T; ++t) {
          const Eigen::Index cell = r * cells + q * T + t;
          double* row = &out.data[static_cast<std::size_t>(cell * Mn)];
          row[0] = g.bispectrum(r, t).real(); row[1] = g.bispectrum(r, t).imag();
          if (normalisation == nxr::signal::BicoherenceNormalisation::threenorm) { row[2] = g.firstCube(r, t); row[3] = g.secondCube(r, t); row[4] = g.sumCube(r, t); }
          if (normalisation == nxr::signal::BicoherenceNormalisation::kim_powers) { row[2] = g.productPower(r, t); row[3] = g.sumPower(r, t); }
        }
    }
    Json measureList = Json::array(), measureNames = Json::array();
    for (const auto& m : measures) { measureList.push_back(Json{{"measure", m}}); measureNames.push_back(m); }
    Json elements = Json::array();
    if (!surface) for (std::size_t j = 0; j < s.modes.size(); ++j) elements.push_back(Json{{"mode", s.modes[j]}, {"mu", j < s.mu.size() ? Json(s.mu[j]) : Json()}});
    Json warnings = Json::array();
    for (const char* w : kWarnings) warnings.push_back(w);
    const Json derived = Json{
        {"bicoherence_threenorm", "|B| / cbrt(first_cube x second_cube x sum_cube), in [0, 1]"},
        {"bicoherence_kim_powers", "|B|^2 / (product_power x sum_power), in [0, 1]"},
        {"biphase", "arg B (radians)"},
        {"peak_trough_asymmetry", "Re(B) / normaliser (harmonics: > 0 sharp peaks, < 0 sharp troughs)"},
        {"rise_decay_asymmetry", "Im(B) / normaliser (harmonics: > 0 slow rise and fast decay, < 0 fast rise and slow decay)"},
        {"phase_amplitude_coupling", "B of (low, high, high) = N^2 x the sum over the transform's samples of z_low |z_high|^2 (z: the atoms' band signals): its modulus the coupling, its "
                                     "argument the low phase at which the high amplitude peaks"}};
    const Json producer = Json{{"warnings", warnings},
                               {"references", Json::array({"Hagihira et al. 2001 (Anesth Analg 93:966)", "Sigl and Chamoun 1994 (J Clin Monit 10:392)", "Kim and Powers 1979 (IEEE Trans Plasma Sci 7:120)",
                                                           "Shahbazi Avarvand et al. 2018 (NeuroImage 174:352)", "Bartz et al. 2019 (NeuroImage 188:833)", "Zandvoort and Nolte 2021 (J Neurosci Methods 350:109032)"})},
                               {"normalisation", pr.p.at("normalisation")}, {"space", pr.p.at("space")}, {"frequency_normalisation", pr.p.value("frequency_normalisation", Json())},
                               {"grouping", plan.groupedBySecondAndSum ? "(second, sum) correlated once per pair, read with each first atom" : "(first, second) convolved once per pair, read with each sum atom"},
                               {"distinct_pairs", plan.groups.size()}, {"bands", Q}, {"chunks", s.chunks}, {"bins", s.bins}, {"sfreq", s.sfreq}, {"transform_samples", s.transform},
                               {"sign", "numpy's forward transform (exp(-2 pi i j k / n)); B = x(f1) x(f2) conj(x(f1 + f2)) is invariant to the chunk's time origin"}};
    out.meta["describe"] = Json{{"method", "bispectrum"}, {"measure", "bispectrum"}, {"shape", Json::array({R, cells, Mn})},
                                {"axes", Json::array({indexed_axis("element", R, surface ? Json() : elements), indexed_axis("member", cells), indexed_axis("measure", Mn, measureNames)})},
                                {"measures", measureList}, {"values_shape", Json::array({R, cells})}, {"unit", "(spectra unit)^3"},
                                {"attributes", Json{{"rows", surface ? "the vertices of the surface" : "the field coordinates (each with itself)"},
                                                    {"members", bandTriples ? "triple x band triple (p, q, r: index (p Q + q) Q + r), the triple fastest" : groups > 1 ? "triple x band, the triple fastest" : "the triples"}, {"triples", listed}, {"bands", surface ? pr.bands.rows() : 0},
                                                    {"sum_over", "the chunks and every pair of bins (f1, f2) with f1 + f2 a kept bin, weighted by the triple's three gains"},
                                                    {"derived_at_read", derived}}},
                                {"producer", producer},
                                {"description", "the bispectrum of the chunk spectra weighted by atoms: per element and triple, B and its normalisation's sums"}};
    return ng::Arrays{{"bispectrum", std::move(out)}};
  };
  n.record = [](const std::any& prepared) {
    const BispectrumPrepared& pr = std::any_cast<const BispectrumPrepared&>(prepared);
    return Json{{"readings", pr.p.at("readings")}, {"normalisation", pr.p.at("normalisation")}, {"space", pr.p.at("space")}, {"atoms", pr.atom_count}, {"bands", pr.bands.rows()}};
  };
  return n;
}

// ── signal.flow_across_frequencies ─────────────────────────────────────────────────────────────────────────────────
const char* const kAcross = "signal.flow_across_frequencies";
struct AcrossPrepared {
  BoundPackage package;
  ng::Array edges, modes, vertices;
  std::vector<ng::Array> stored;      // power, current_x, current_y, current_z when given as stored measurements
  std::vector<double> centres;
};
// a flow readout's per-atom values [elements x atoms] of one measure (signal.spectral_flow's arrays: element x member x measure)
Eigen::MatrixXd measure_of(const ng::Array& a, std::int64_t measure, const char* who, const char* port) {
  if (a.meta.contains("measure") && measure == 0) {                       // a STORED measurement, read by its measure: its values [rows x members]
    Eigen::MatrixXd m(a.rows, a.cols);
    for (std::int64_t r = 0; r < a.rows; ++r) for (std::int64_t c = 0; c < a.cols; ++c) m(r, c) = a.at(static_cast<std::size_t>(r * a.cols + c));
    return m;
  }
  const Json shape = shape_of(a);
  if (!shape.is_array() || shape.size() != 3) ng::refuse(std::string(who) + ": " + port + " is not a flow readout's [elements x members x measures] (signal.spectral_flow)");
  const std::int64_t rows = shape[0], members = shape[1], measures = shape[2];
  if (measure >= measures || static_cast<std::int64_t>(a.size()) != rows * members * measures) ng::refuse(std::string(who) + ": " + port + "'s values do not fit its shape");
  Eigen::MatrixXd m(rows, members);
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t c = 0; c < members; ++c) m(r, c) = a.at(static_cast<std::size_t>((r * members + c) * measures + measure));
  return m;
}

ng::NodeSpec flow_across_frequencies() {
  ng::NodeSpec n;
  n.name = kAcross;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"edges", "the flow's phase current per edge and atom (signal.spectral_flow's whole_edges)", false},
              {"modes", "the flow's Hodge-Dirac coefficient per mode and atom (signal.spectral_flow's whole_modes)", false},
              {"vertices", "the flow's vertex measures per atom as one readout (signal.spectral_flow's whole_vertices in a run: power, potential, stream, vertex phase current x y z); or the four ports below", true},
              {"power", "the stored measurement power per vertex and atom (with current_x, current_y, current_z, in place of vertices)", true},
              {"current_x", "the stored measurement vertex_phase_current_x", true},
              {"current_y", "the stored measurement vertex_phase_current_y", true},
              {"current_z", "the stored measurement vertex_phase_current_z", true},
              {"package", "the Hodge-Dirac package the flow is on (its edges' Hodge weights and its modes' kinds)", false},
              {"frequency", "the atoms the flow was read with (the kernel set: their centres are the frequencies compared)", false}};
  n.outputs = {{"atom_pairs", "per pair of atoms: the cosine of their edge phase currents, of their travel coefficients and of their rotation coefficients", false},
               {"dispersion", "per vertex: the direction consistency across atoms, the slope and intercept of log apparent speed against log frequency, the atoms used", false}};
  n.parameters = {};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once: a statistic over the members of the atoms' level, from the flow's stored per-atom sums (no recording)");
  n.matches = "the speed across frequencies readout of the sensor flow (spec 2026-11-24-bispectrum); compute nxr::signal::flowAcrossFrequencies";
  n.description = "whether the flow moves at the same speed and in the same direction across frequencies: cosines of the atoms' flows, direction consistency and the slope of speed against frequency";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"edges", "modes", "package", "frequency"}) if (!in.count(k)) ng::refuse(std::string(kAcross) + ": " + k + " is not given");
    const bool four = in.count("power") && in.count("current_x") && in.count("current_y") && in.count("current_z");
    if (in.count("vertices") == four) ng::refuse(std::string(kAcross) + ": give the vertex measures either as one readout (vertices) or as the four stored measurements (power, current_x, current_y, current_z)");
    ng::Planned out;
    for (const char* o : {"atom_pairs", "dispersion"}) { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "statistic"; out.out[o] = t; }
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json&, ng::Shared&) {
    AcrossPrepared pr;
    pr.package = bind_hodge_package(handed(c, "package", kAcross), kAcross);
    pr.edges = handed(c, "edges", kAcross); pr.modes = handed(c, "modes", kAcross);
    if (c.count("vertices")) pr.vertices = handed(c, "vertices", kAcross);
    else for (const char* k : {"power", "current_x", "current_y", "current_z"}) pr.stored.push_back(handed(c, k, kAcross));
    pr.centres = centres_of(handed(c, "frequency", kAcross), kAcross);
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const AcrossPrepared& pr = std::any_cast<const AcrossPrepared&>(prepared);
    const Eigen::MatrixXd edges = measure_of(pr.edges, 0, kAcross, "edges"), modes = measure_of(pr.modes, 0, kAcross, "modes");
    const bool stored = !pr.stored.empty();
    const Eigen::MatrixXd power = stored ? measure_of(pr.stored[0], 0, kAcross, "power") : measure_of(pr.vertices, 0, kAcross, "vertices");
    const Eigen::Index S = power.rows(), A = power.cols();
    Eigen::MatrixXd current(3 * S, A);
    for (int c = 0; c < 3; ++c) {
      const Eigen::MatrixXd m = stored ? measure_of(pr.stored[static_cast<std::size_t>(1 + c)], 0, kAcross, "current") : measure_of(pr.vertices, 3 + c, kAcross, "vertices");
      if (m.rows() != S || m.cols() != A) ng::refuse(std::string(kAcross) + ": the vertex measures do not share their vertices and atoms");
      for (Eigen::Index v = 0; v < S; ++v) current.row(3 * v + c) = m.row(v);
    }
    if (static_cast<Eigen::Index>(pr.centres.size()) != A) ng::refuse(std::string(kAcross) + ": the flow has " + std::to_string(A) + " members, the atoms " + std::to_string(pr.centres.size()));
    const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
    const nxr::signal::Modes& md = *pr.package.modes;
    if (edges.rows() != cx.nE || modes.rows() != md.mu.size()) ng::refuse(std::string(kAcross) + ": the flow is not on this package's edges and modes");
    std::vector<bool> exact(static_cast<std::size_t>(md.mu.size()));
    for (std::size_t k = 0; k < exact.size(); ++k) exact[k] = md.kind[k] == 0;
    const Eigen::VectorXd frequencies = Eigen::Map<const Eigen::VectorXd>(pr.centres.data(), A);
    const nxr::signal::FlowAcrossFrequencies r = nxr::signal::flowAcrossFrequencies(edges, cx.star1, modes, exact, power, current, frequencies);
    ng::Array pairs;
    pairs.rows = A * A; pairs.cols = 3;
    for (Eigen::Index a = 0; a < A; ++a)
      for (Eigen::Index b = 0; b < A; ++b) for (const double v : {r.edgeCosine(a, b), r.travelCosine(a, b), r.rotationCosine(a, b)}) pairs.data.push_back(v);
    ng::Array vertices;
    vertices.rows = S; vertices.cols = 4;
    for (Eigen::Index v = 0; v < S; ++v) for (const double x : {r.directionConsistency(v), r.speedSlope(v), r.speedIntercept(v), r.atomsUsed(v)}) vertices.data.push_back(x);
    Json atoms = Json::array();
    for (const double c : pr.centres) atoms.push_back(c);
    const Json producer = Json{{"warnings", Json::array({"the apparent speed is an upper bound (2 pi f power / |vertex phase current|); its slope against frequency is 0 for the same speed, 1 for the same wavelength",
                                                         "the field on the helmet only (or the surface the flow is on); no source claim",
                                                         "a whole-run flow keeps the net quadrature: an event's flow is compared across atoms by the same node from the atoms' time cells within the event (graph in the spec)",
                                                         "no null here: a single run's cosines and slopes are descriptions; a test needs events or runs as replicates"})},
                               {"frequencies_hz", atoms}};
    pairs.meta["describe"] = Json{{"method", "flow_across_frequencies"}, {"measure", "flow_agreement"}, {"shape", Json::array({A, A, 3})},
                                  {"axes", Json::array({indexed_axis("atom", A, atoms), indexed_axis("atom", A, atoms), indexed_axis("measure", 3, Json::array({"edge_cosine", "travel_cosine", "rotation_cosine"}))})},
                                  {"measures", Json::array({Json{{"measure", "edge_cosine"}}, Json{{"measure", "travel_cosine"}}, Json{{"measure", "rotation_cosine"}}})},
                                  {"producer", producer}, {"description", "per pair of atoms, the cosines of their flows: edges (Hodge inner product), travel and rotation coefficients"}};
    vertices.meta["describe"] = Json{{"method", "flow_across_frequencies"}, {"measure", "flow_dispersion"}, {"shape", Json::array({S, 4})},
                                     {"axes", Json::array({indexed_axis("vertex", S), indexed_axis("measure", 4, Json::array({"direction_consistency", "speed_slope", "speed_intercept", "atoms_used"}))})},
                                     {"measures", Json::array({Json{{"measure", "direction_consistency"}}, Json{{"measure", "speed_slope"}}, Json{{"measure", "speed_intercept"}}, Json{{"measure", "atoms_used"}}})},
                                     {"producer", producer},
                                     {"description", "per vertex, across atoms: the resultant length of the unit vertex phase currents, and log apparent speed = intercept + slope x log frequency"}};
    return ng::Arrays{{"atom_pairs", std::move(pairs)}, {"dispersion", std::move(vertices)}};
  };
  n.record = [](const std::any& prepared) { return Json{{"atoms", std::any_cast<const AcrossPrepared&>(prepared).centres.size()}}; };
  return n;
}

}  // namespace

void register_bispectrum_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {bispectrum(), flow_across_frequencies()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
