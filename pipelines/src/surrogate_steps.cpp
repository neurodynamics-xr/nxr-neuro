// nxr-pipelines — the registered steps of the LAG-FREE SURROGATES (spec docs/specs/2026-11-17-surrogates.md; compute nxr::synthesis):
//
//   power_series              accumulate   the data's band power in TIME TILES: Σ |z|² over the channels the surrogate covers and over each tile's good samples, and the count (a reduction over a time
//                                          partition, folded chunk by chunk: the dyadic tiles of 2^level samples on the recording's line); a STATISTIC [tiles × (sum, count)], the mean derived at read. It is
//                                          the envelope the surrogate is scaled to (Rheome envelopenull's one-cycle envelope: a tile of 2^level samples, the nearest power of two to a cycle).
//   surrogate.lagfree_mixing  setup        the ZERO-LAG MIXING of the band: √(Re C) of its stored cross-spectrum at unit mean channel variance (nxr::synthesis::zeroLagMixing), an operator [M × M] over the
//                                          cross-spectrum's channels, with its provenance; composed into the first linear map of the chain (apply_kernel's `pre_operator`).
//   surrogate.lagfree         chunk        ONE DRAW, UNMIXED: per channel of the stored cross-spectrum, Gaussian noise with the recording's own mean spectrum (the stored Welch PSD: never flat), scaled in time
//                                          to the data's tile power over the draw's expected power through the band's analytic gain (nxr::synthesis::shapedNoiseBlock). A function of the SAMPLE INDEX (a
//                                          counter-based generator): the draw does not depend on the chunk size; another `seed` is another draw. It reads NO sample of x (the recording declared `read: shape`):
//                                          only its rows, rate and samples. Its output AWAITS the mixing (`awaits_operator`): the time steps after it are linear and the same on every channel, so the mixing
//                                          commutes with them and is applied once, composed into the spatial operator (spec section 4).
//
// WHAT IT CANNOT SAY: the surrogate has the data's covariance at zero lag, its spectrum and its envelope, and nothing else. Its spatial structure is the band's (the cross-spectrum's real part) at every
// frequency; the draw's envelope follows the data's up to the noise's own sampling variation (a pointwise scale: no reduction inside the node, the cardinal rule). Channels the cross-spectrum does not
// cover are zero in the draw (the kernel uses none of them).
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>

#include "nxr/filterbank.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/registry.h"
#include "nxr/synthesis/lagfree.h"

namespace nxr::pipelines::graph {

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
void recording_input(const Value& x, const std::string& who, bool complex) {
  need(x.cls == "field" && x.time && x.chunked, who + ": x must be a field on … × time, inside the chunk loop");
  need(x.facts.value("axis", std::string("samples")) == "samples", who + ": x is on the frame tiling; this step reads the recording's own samples");
  need(x.complex == complex, who + (complex ? ": x must be the band's analytic signal (`analytic`)" : ": x must be the real recording (before the analytic signal)"));
  need(x.value_type == "scalar", who + ": x is " + x.value_type + "; one number a channel");
}
void statistic_input(const Value& t, const std::string& who, const std::string& port, const std::string& method, const std::vector<std::string>& measures) {
  need(t.cls == "statistic" && t.row, who + ": " + port + " must be a stored statistic (read by name)");
  const std::string m = nxr::io::text(*t.row, "method"), me = nxr::io::text(*t.row, "measure");
  need(m == method && std::find(measures.begin(), measures.end(), me) != measures.end(), who + ": " + port + " " + nxr::io::text(*t.row, "name") + " is a statistic of method " + m + ", measure " + me + "; it must be method " + method + " (" + measures[0] + ")");
}
std::function<double(double)> gain_of(const Json& p) {
  const double lo = p["band"][0].get<double>(), hi = p["band"][1].get<double>(), hw = p["half_width"].get<double>();
  if (p["shape"] == "morlet") return [lo, hi](double f) { return nxr::filterbank::morletBand(f, lo, hi); };
  return [lo, hi, hw](double f) { return nxr::filterbank::meyerBand(f, lo, hi, hw); };
}
void band_params(const Json& p, const std::string& who, double rate) {
  const Json& b = p["band"];
  need(b.is_array() && b.size() == 2 && b[0].is_number() && b[1].is_number() && b[0].get<double>() >= 0 && b[0].get<double>() < b[1].get<double>() && b[1].get<double>() < rate / 2, who + ": band " + b.dump() + ": [lo, hi] Hz below Nyquist");
  need(p["shape"] == "morlet" || p["shape"] == "meyer", who + ": shape " + p["shape"].dump() + ": morlet or meyer");
}
// what the mixing is OF: the cross-spectrum's name (a value awaiting the mixing names the same, and apply_kernel checks they meet)
Json operator_of(const Value& cross_spectrum) { return Json{{"cross_spectrum", nxr::io::text(*cross_spectrum.row, "name")}, {"operator", "zero-lag mixing"}}; }
// ── power_series ──
struct SeriesState { std::vector<std::int32_t> channels; int level = 7; double rate = 0; Json record; };
struct SeriesAcc { std::map<std::int64_t, std::pair<double, double>> tiles; };
// ── surrogate.lagfree_mixing ──
struct MixingState { std::string name, id; };
// ── surrogate.lagfree ──
struct SurState {
  std::vector<std::int32_t> channels;       // the M recording rows the draw fills (the cross-spectrum's), stream m at row channels[m]
  std::int64_t rows = 0;                    // the recording's rows (the block the draw makes)
  Eigen::VectorXd taps;
  std::vector<double> power;                // the data's mean |z|^2 per tile (gaps filled)
  std::int64_t stride = 1;
  double expected = 1;                      // the unscaled, mixed draw's mean |z|^2 through the band's analytic gain
  std::uint64_t seed = 0;
  Json record;
};
}  // namespace

void add_surrogate_steps(std::vector<StepSpec>& r) {
  {
    StepSpec s;
    s.name = "power_series"; s.kind = "accumulate";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "adds each good sample's power into its time tile's sum and count: keyed sums, whatever the order of the blocks (up to rounding)", {});
    s.inputs = {"x", "cross_spectrum", "exclude"}; s.optional = {"exclude"}; s.outputs = {"series"};
    s.parameters = Json{{"level", 7}, {"bad_segments", "exclude"}};
    s.matches = "Rheome envelopenull's one-cycle envelope (the sensors' band power, per tile of 2^level samples: the nearest power of two to a cycle); the dyadic time tiles of the recording's line";
    s.description = "the band power of the analytic recording in TIME TILES of 2^level samples (the recording's line, from sample 0): per tile the sum of |z|² over the channels of the stored cross-spectrum `cross_spectrum` "
                    "(the channels the surrogate covers) and over the tile's good samples, and the number of those values; the mean (the one-cycle power the surrogate is scaled to) is DERIVED at read. A reduction over a time partition "
                    "folded chunk by chunk (the result does not depend on the chunk size); a STATISTIC [tiles × (sum, count)], measure power_sum, method power_tiles";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value& x = ins.at("x");
      recording_input(x, "power_series", true);
      need(p["level"].is_number_integer() && p["level"].get<int>() >= 1 && p["level"].get<int>() <= 15, "power_series: level is the tile's size 2^level samples, 1 to 15");
      need(p["bad_segments"] == "exclude" || p["bad_segments"] == "ignore", "power_series: bad_segments " + p["bad_segments"].dump() + ": exclude or ignore");
      if (ins.count("exclude")) need(ins.at("exclude").cls == "selection" && ins.at("exclude").manifolds[0] == *x.time, "power_series: exclude must be a selection on x's time line");
      statistic_input(ins.at("cross_spectrum"), "power_series", "cross_spectrum", "welch", {"cross_spectral_density"});
      Value y = x;
      y.row.reset(); y.cls = "statistic"; y.chunked = false; y.complex = false; y.unit.reset(); y.manifolds.clear(); y.time.reset();
      y.facts = Json{{"measure", "power_sum"}, {"generic", true}};
      return Values{{"series", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
      SeriesState st;
      const auto cs = c.bound.cross_spectrum(nxr::io::text(*ins.at("cross_spectrum").row, "name"));
      st.channels = cs.channels; st.level = p["level"].get<int>(); st.rate = ins.at("x").rate;
      st.record = Json{{"level", st.level}, {"tile_samples", std::int64_t(1) << st.level}, {"channels", st.channels.size()}, {"cross_spectrum", nxr::io::text(*ins.at("cross_spectrum").row, "name")}, {"band", ins.at("x").facts.value("band", Json())}};
      return std::any(std::make_shared<SeriesState>(std::move(st)));
    };
    s.record = [](const std::any& st) { return std::any_cast<const std::shared_ptr<SeriesState>&>(st)->record; };
    s.start = [](const std::any&) { return std::any(SeriesAcc{}); };
    s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& at) {
      const SeriesState& st = *std::any_cast<const std::shared_ptr<SeriesState>&>(st_);
      SeriesAcc& acc = std::any_cast<SeriesAcc&>(acc_);
      const Block& b = ins.at("x");
      for (std::int64_t j = 0; j < at.n; ++j) {
        if (at.good && !at.good->empty() && !(*at.good)[static_cast<std::size_t>(j)]) continue;
        double sum = 0;
        for (const std::int32_t ch : st.channels) {
          const std::size_t i = static_cast<std::size_t>(ch * b.width + b.left + j);
          const double re = b.re[i], im = b.complex() ? b.im[i] : 0.0;
          sum += re * re + im * im;
        }
        auto& t = acc.tiles[(at.first + j) >> st.level];
        t.first += sum; t.second += static_cast<double>(st.channels.size());
      }
    };
    s.finish = [](const std::any& st_, std::any& acc_) {
      const SeriesState& st = *std::any_cast<const std::shared_ptr<SeriesState>&>(st_);
      const SeriesAcc& acc = std::any_cast<SeriesAcc&>(acc_);
      if (acc.tiles.empty()) refuse("power_series: no good sample in the span");
      const std::int64_t n = acc.tiles.rbegin()->first + 1, tile = std::int64_t(1) << st.level;
      Array y{n, 2, std::vector<double>(static_cast<std::size_t>(n * 2), 0.0)};
      std::int64_t good = 0;
      for (const auto& [k, v] : acc.tiles) { y.data[static_cast<std::size_t>(2 * k)] = v.first; y.data[static_cast<std::size_t>(2 * k + 1)] = v.second; good += v.second > 0; }
      Json edges = Json::array();
      for (std::int64_t k = 0; k < n; ++k) edges.push_back(static_cast<double>((k + 1) * tile));
      y.meta = Json{{"generic_statistic", Json{{"measure", "power_sum"}, {"method", "power_tiles"}, {"shape", Json::array({n, std::int64_t(2)})},
                                               {"axes", Json::array({Json{{"name", "time_tile"}, {"edges", edges}, {"unit", "samples"}, {"size", n}}, Json{{"name", "component"}, {"size", 2}}})},
                                               {"components", Json::array({Json{{"name", "sum"}}, Json{{"name", "count"}}})}, {"n_members", good},
                                               {"description", "the band power of the recording in time tiles of " + std::to_string(tile) + " samples: the sum of |z|^2 over the surrogate's channels and the tile's good samples, and the number of values (mean = sum / count at read)"},
                                               {"options", Json{{"level", st.level}, {"tile_samples", tile}, {"rate", st.rate}, {"channels", st.channels}, {"record", st.record}, {"upper_edges", "the time axis's `edges` are the tiles' upper edges in samples, from sample 0 of the recording's line"}}}}}};
      return Arrays{{"series", std::move(y)}};
    };
    r.push_back(std::move(s));
  }

  {
    StepSpec s;
    s.name = "surrogate.lagfree_mixing"; s.kind = "setup";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "once, from a stored cross-spectrum (no time axis), in double", {});
    s.inputs = {"cross_spectrum"}; s.outputs = {"mixing"};
    s.parameters = Json::object();
    s.matches = "Rheome +rheome/+cores lagfreenull (Rh = the symmetric square root of the zero-lag covariance), pairsqrt; compute nxr::synthesis::zeroLagMixing";
    s.description = "the ZERO-LAG MIXING of the band: the symmetric square root of the real part of the stored cross-spectrum at unit mean channel variance, an operator [M × M] over the cross-spectrum's channels "
                    "(meta.channels, in order) with its provenance (meta.provenance: the cross-spectrum's id and name). A surrogate's unmixed draw awaits it; it is composed ONCE into the first linear map of the chain "
                    "(apply_kernel's `pre_operator`), never applied per sample";
    s.plan = [](const Values& ins, const Json&, const PlanContext&) {
      statistic_input(ins.at("cross_spectrum"), "surrogate.lagfree_mixing", "cross_spectrum", "welch", {"cross_spectral_density"});
      Value y;
      y.cls = "operator";
      y.facts = Json{{"operator_of", operator_of(ins.at("cross_spectrum"))}, {"linear", true}, {"made_by", "surrogate.lagfree_mixing"}};
      return Values{{"mixing", y}};
    };
    s.prepare = [](const Values& ins, const Json&, ExecContext&) {
      const nxr::io::Row& row = *ins.at("cross_spectrum").row;
      return std::any(MixingState{nxr::io::text(row, "name"), nxr::io::text(row, "id")});
    };
    s.record = [](const std::any& st) { const MixingState& m = std::any_cast<const MixingState&>(st); return Json{{"cross_spectrum", m.name}, {"cross_spectrum_id", m.id}}; };
    s.run = [](const std::any& st_, const Arrays&, ExecContext& c) {
      const MixingState& st = std::any_cast<const MixingState&>(st_);
      const auto cs = c.bound.cross_spectrum(st.name);
      Eigen::MatrixXd B;
      try { B = nxr::synthesis::zeroLagMixing(cs.S); } catch (const std::invalid_argument& e) { refuse(std::string("surrogate.lagfree_mixing: ") + st.name + ": " + e.what()); }
      const std::int64_t M = B.rows();
      Array a;
      a.rows = M; a.cols = M;
      a.data.resize(static_cast<std::size_t>(M * M));
      for (std::int64_t i = 0; i < M; ++i) for (std::int64_t j = 0; j < M; ++j) a.data[static_cast<std::size_t>(i * M + j)] = B(i, j);    // row-major
      a.meta["channels"] = cs.channels;
      a.meta["provenance"] = Json{{"cross_spectrum", st.name}, {"cross_spectrum_id", st.id}, {"operator", "zero-lag mixing: sqrt(Re C / (trace Re C / M))"}, {"reference", "Rheome lagfreenull, pairsqrt"}};
      return Arrays{{"mixing", std::move(a)}};
    };
    r.push_back(std::move(s));
  }

  {
    StepSpec s;
    s.name = "surrogate.lagfree"; s.kind = "chunk";
    s.reads_shape_only = true;
    s.traits = node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "the noise is a function of the sample index (a counter-based generator) and reads no sample of the input; each output sample is that noise filtered by the shaping taps, made from the sample index on each side, so it asks no halo of the input; computed in double, stored as float32", "none from the input: the taps' reach (`taps_half` samples each side) is made from the sample index");
    s.inputs = {"x", "psd", "cross_spectrum", "power_series"}; s.outputs = {"y"};
    s.parameters = Json{{"band", nullptr}, {"shape", "morlet"}, {"half_width", 1.0 / 3.0}, {"taps_half", 0}, {"seed", 0}};
    s.matches = "Rheome +rheome/+cores lagfreenull and envelopenull (Gaussian noise with the data's mean spectrum, rescaled to the data's one-cycle envelope; the mixing is applied later, composed into the spatial operator); compute nxr::synthesis (shapedNoiseBlock, spectrumShapingTaps, channelMeanSpectrum, analyticGainPower, envelopeScale, interpolateTileSeries); the scale is pointwise (expected, not realised)";
    s.description = "ONE LAG-FREE DRAW, UNMIXED, in place of the recording: per channel of the stored cross-spectrum one stream of Gaussian noise with the recording's own mean spectrum (the stored Welch PSD over those channels: never flat), "
                    "scaled in time to the data's tile power (`power_series`) over the draw's expected power through the band's analytic gain, so the mixed draw's analytic signal has the data's band envelope. A function of the SAMPLE "
                    "INDEX (a counter-based generator of `seed`): the draw does not depend on the chunk size; another `seed` is another draw. x is the recording's SHAPE (rows, rate, samples; `read: shape`): no sample is read. "
                    "Channels the cross-spectrum does not cover are zero. The output AWAITS the zero-lag mixing (`surrogate.lagfree_mixing`), applied once in the first linear map (apply_kernel's `pre_operator`): it commutes "
                    "with every linear time step before it (spec docs/specs/2026-11-17-surrogates.md, section 4). `taps_half` 0: from the PSD's resolution (half = fs / (2 df), at most 4096)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value& x = ins.at("x");
      recording_input(x, "surrogate.lagfree", false);
      band_params(p, "surrogate.lagfree", x.rate);
      need(p["seed"].is_number_integer() && p["seed"].get<std::int64_t>() >= 0, "surrogate.lagfree: seed is a whole number >= 0 (a draw)");
      need(p["taps_half"].is_number_integer() && p["taps_half"].get<int>() >= 0, "surrogate.lagfree: taps_half is a whole number >= 0 (0: from the PSD's resolution)");
      statistic_input(ins.at("psd"), "surrogate.lagfree", "psd", "welch", {"power_density", "power"});
      statistic_input(ins.at("cross_spectrum"), "surrogate.lagfree", "cross_spectrum", "welch", {"cross_spectral_density"});
      statistic_input(ins.at("power_series"), "surrogate.lagfree", "power_series", "power_tiles", {"power_sum"});
      Value y = x;
      y.row.reset();
      y.facts.erase("read");                                                  // the draw has values: the steps after it read them
      y.facts["made_by"] = "surrogate.lagfree";
      y.facts["awaits_operator"] = Json{{"of", operator_of(ins.at("cross_spectrum"))}};
      y.facts["surrogate"] = Json{{"generator", "surrogate.lagfree"}, {"seed", p["seed"]}, {"psd", nxr::io::text(*ins.at("psd").row, "name")}, {"cross_spectrum", nxr::io::text(*ins.at("cross_spectrum").row, "name")},
                                  {"power_series", nxr::io::text(*ins.at("power_series").row, "name")}};
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
      SurState st;
      const Value& x = ins.at("x");
      const auto man = c.bound.manifold(x.manifolds[0]);
      need(man.has_value(), "surrogate.lagfree: x's channels are not among the subject's manifolds");
      st.rows = nxr::io::integer(*man, "n_vertices");
      const auto cs = c.bound.cross_spectrum(nxr::io::text(*ins.at("cross_spectrum").row, "name"));
      st.channels = cs.channels;
      const std::int64_t M = static_cast<std::int64_t>(cs.channels.size());
      for (const std::int32_t ch : st.channels) need(ch >= 0 && ch < st.rows, "surrogate.lagfree: the cross-spectrum's channel " + std::to_string(ch) + " is not one of x's " + std::to_string(st.rows) + " rows");
      // the spectrum: the PSD's mean over the cross-spectrum's channels, on its frequency line (k df)
      const nxr::io::StoredStatistic P = c.bound.statistic(nxr::io::text(*ins.at("psd").row, "name"));
      need(P.shape.size() == 2, "surrogate.lagfree: the PSD " + nxr::io::text(*ins.at("psd").row, "name") + " is not [channels × frequencies]");
      const std::int64_t nf = P.shape[1];
      const double df = P.producer.value("frequency_resolution_hz", 0.0), fs = P.producer.value("sfreq", x.rate);
      need(df > 0 && std::abs(fs - x.rate) < 1e-6 * x.rate, "surrogate.lagfree: the PSD's resolution or sampling rate is not recorded, or it is not the recording's");
      Eigen::VectorXd freq(nf), gain(nf), psd;
      for (std::int64_t k = 0; k < nf; ++k) freq[k] = static_cast<double>(k) * df;
      const auto g = gain_of(p);
      for (std::int64_t k = 0; k < nf; ++k) gain[k] = g(freq[k]);
      const int half = p["taps_half"].get<int>() > 0 ? p["taps_half"].get<int>() : static_cast<int>(std::min(4096.0, std::max(64.0, std::round(fs / (2 * df)))));
      try {
        psd = nxr::synthesis::channelMeanSpectrum(P.values, nf, st.channels);
        st.taps = nxr::synthesis::spectrumShapingTaps(freq, psd, fs, half);
        st.expected = nxr::synthesis::analyticGainPower(freq, psd, gain);             // the unscaled draw's mean |z|^2 through the analytic gain (unit-variance streams, unit mean channel variance after the mixing)
      } catch (const std::invalid_argument& e) { refuse(std::string("surrogate.lagfree: ") + e.what() + " (band " + p["band"].dump() + " Hz)"); }
      // the data's tile power: the mean derived at read (sum / count); a tile with no good sample takes the one before it
      const nxr::io::StoredStatistic S = c.bound.statistic(nxr::io::text(*ins.at("power_series").row, "name"));
      need(S.shape.size() == 2 && S.shape[1] == 2, "surrogate.lagfree: power_series is not [tiles × (sum, count)]");
      const std::int64_t n = S.shape[0];
      st.stride = std::int64_t(1) << S.options.value("level", 0);
      need(st.stride > 1, "surrogate.lagfree: the power series records no tile level");
      st.power.assign(static_cast<std::size_t>(n), std::nan(""));
      for (std::int64_t k = 0; k < n; ++k) if (S.values[static_cast<std::size_t>(2 * k + 1)] > 0) st.power[static_cast<std::size_t>(k)] = S.values[static_cast<std::size_t>(2 * k)] / S.values[static_cast<std::size_t>(2 * k + 1)];
      std::int64_t first_ok = -1;
      for (std::int64_t k = 0; k < n; ++k) if (std::isfinite(st.power[static_cast<std::size_t>(k)])) { first_ok = k; break; }
      need(first_ok >= 0, "surrogate.lagfree: the power series holds no good tile");
      for (std::int64_t k = 0; k < n; ++k) { if (!std::isfinite(st.power[static_cast<std::size_t>(k)])) st.power[static_cast<std::size_t>(k)] = k > 0 ? st.power[static_cast<std::size_t>(k - 1)] : st.power[static_cast<std::size_t>(first_ok)]; }
      st.seed = static_cast<std::uint64_t>(p["seed"].get<std::int64_t>());
      st.record = Json{{"generator", "surrogate.lagfree"}, {"seed", p["seed"]}, {"channels", M}, {"taps", st.taps.size()}, {"taps_half", half}, {"frequency_resolution_hz", df}, {"expected_power", st.expected}, {"tile_samples", st.stride},
                       {"psd", nxr::io::text(*ins.at("psd").row, "name")}, {"cross_spectrum", nxr::io::text(*ins.at("cross_spectrum").row, "name")}, {"power_series", nxr::io::text(*ins.at("power_series").row, "name")},
                       {"reads", "the recording's shape only (no sample)"}, {"order", "white -> taps -> scale; the mixing composed into the first linear map (spec docs/specs/2026-11-17-surrogates.md, section 4)"},
                       {"reference", "Rheome lagfreenull, envelopenull"},
                       {"warnings", Json::array({"the surrogate has the data's zero-lag covariance, spectrum and envelope and nothing else: a result beyond it is beyond what mixing and amplitude modulation produce, not a proof of a wave",
                                                 "the draw's envelope follows the data's up to the noise's own sampling variation (a pointwise scale), where Rheome's envelopenull rescales each draw to its realised envelope",
                                                 "the draw's spatial structure is the band's (the cross-spectrum's real part), the same at every frequency"})}};
      return std::any(std::make_shared<SurState>(std::move(st)));
    };
    s.record = [](const std::any& st) { return std::any_cast<const std::shared_ptr<SurState>&>(st)->record; };
    s.halo = [](const std::any&) { return std::int64_t(0); };
    s.halo_at_plan = [](const Values&, const Json&) { return std::int64_t(0); };   // the draw is made on the halo the later steps ask, from the sample index: it needs none of the recording's
    s.chunk_at = [](const std::any& st_, const Blocks& ins, std::int64_t keep, const Place& at) {
      const SurState& st = *std::any_cast<const std::shared_ptr<SurState>&>(st_);
      const Block& b = ins.at("x");                                           // the SHAPE: its width and halo (no rows, no samples)
      const std::int64_t kl = std::min(keep, b.left), kr = std::min(keep, b.right), w = b.own() + kl + kr, first = at.first - kl;
      const Eigen::VectorXd pw = nxr::synthesis::interpolateTileSeries(st.power, st.stride, first - st.stride / 2, w);   // the tiles' value k sits at the tile's centre
      const Eigen::MatrixXd Y = nxr::synthesis::shapedNoiseBlock(st.taps, st.seed, static_cast<int>(st.channels.size()), first, w, nxr::synthesis::envelopeScale(pw, st.expected));
      Block y{st.rows, w, kl, kr, std::vector<float>(static_cast<std::size_t>(st.rows * w), 0.0f), {}};
      for (std::size_t m = 0; m < st.channels.size(); ++m)
        for (std::int64_t j = 0; j < w; ++j) y.re[static_cast<std::size_t>(st.channels[m] * w + j)] = static_cast<float>(Y(static_cast<Eigen::Index>(m), j));
      return Blocks{{"y", std::move(y)}};
    };
    r.push_back(std::move(s));
  }
}

}  // namespace nxr::pipelines::graph
