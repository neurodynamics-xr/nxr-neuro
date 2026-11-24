// nxr-pipelines — the registered steps of the LAG-FREE SURROGATES (spec docs/specs/2026-11-17-surrogates.md; compute nxr::synthesis):
//
//   (power_series is gone: the data's band power per tile is a stored MEASUREMENT, `measure` preset surrogate_lagfree_envelope, read by name by surrogate.lagfree; so is the PSD's channel mean, preset psd_channel_mean: rule 13/32)
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

#include "nxr/signal/timefilterbank.h"
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
  nxr::signal::timefilterbank::BandShape shape;
  shape.family = p["shape"].get<std::string>();
  shape.lo = p["band"][0].get<double>();
  shape.hi = p["band"][1].get<double>();
  if (p.contains("half_width") && p["half_width"].is_number()) shape.halfWidth = p["half_width"].get<double>();
  return nxr::signal::timefilterbank::bandGain(shape);
}
void band_params(const Json& p, const std::string& who, double rate) {
  const Json& b = p["band"];
  need(b.is_array() && b.size() == 2 && b[0].is_number() && b[1].is_number() && b[0].get<double>() >= 0 && b[0].get<double>() < b[1].get<double>() && b[1].get<double>() < rate / 2, who + ": band " + b.dump() + ": [lo, hi] Hz below Nyquist");
  need(p["shape"] == "morlet" || p["shape"] == "meyer", who + ": shape " + p["shape"].dump() + ": morlet or meyer");
}
// what the mixing is OF: the cross-spectrum's name (a value awaiting the mixing names the same, and apply_kernel checks they meet)
Json operator_of(const Value& cross_spectrum) { return Json{{"cross_spectrum", nxr::io::text(*cross_spectrum.row, "name")}, {"operator", "zero-lag mixing"}}; }
// a STORED MEASUREMENT read by name (rule 13: a reduction a chain needs as a constant is a measurement stored by its own definition first): its selection (`facts.over.selection_id`) and, when it is
// of a field, that field (`facts.of_field_id`). The values are read through the binding (io's measurement_values); nothing is recomputed here.
void measurement_input(const Value& m, const std::string& who, const std::string& port) {
  need(m.cls == "measurement" && m.facts.contains("over") && m.facts["over"].contains("selection_id"), who + ": " + port + " must be a stored measurement (read by name; `nxr-neuro run measure --set ...` first)");
}
nxr::io::MeasurementValues read_measurement(const Bound& b, const Value& m, const std::string& measure) {
  if (m.facts.contains("of_statistic_id") && m.facts["of_statistic_id"].is_string())            // a measurement OF A STORED STATISTIC (the PSD's channel mean)
    return b.measurement_values_of_statistic(m.facts["over"]["selection_id"].get<std::string>(), m.facts["of_statistic_id"].get<std::string>(), measure);
  return b.measurement_values(m.facts["over"]["selection_id"].get<std::string>(), m.facts.contains("of_field_id") && m.facts["of_field_id"].is_string() ? std::optional<std::string>(m.facts["of_field_id"].get<std::string>()) : std::nullopt, std::nullopt, measure);
}
// the tile (samples) of the time factor of a product selection: the time grid the measurement was made over
std::int64_t time_tile_of(const Bound& b, const Value& m, const std::string& who) {
  const auto sel = b.selection(m.facts["over"]["selection_id"].get<std::string>());
  need(sel.has_value(), who + ": the measurement's selection is not stored");
  const Json pj = Json::parse(nxr::io::opt_text(*sel, "params_json").value_or("{}"), nullptr, false);
  if (pj.is_object() && pj.contains("product") && pj["product"].is_array()) {                    // the stored product: its factors are Selections, in order; the time factor's grid is the tile
    for (const auto& e : pj["product"]) {
      if (!e.is_object() || !e.contains("selection") || !e["selection"].is_string()) continue;
      const auto f = b.selection(e["selection"].get<std::string>());
      if (!f) continue;
      const Json fj = Json::parse(nxr::io::opt_text(*f, "params_json").value_or("{}"), nullptr, false);
      if (fj.is_object() && fj.contains("grid") && fj["grid"].is_object() && fj["grid"].value("kind", std::string()) == "time" && fj["grid"].value("width", std::int64_t(0)) > 0) return fj["grid"]["width"].get<std::int64_t>();
    }
  } else if (pj.is_object() && pj.contains("product"))
    for (const auto& f : pj["product"]["factors"]) if (f.value("kind", std::string()) == "time" && f.value("width", std::int64_t(0)) > 0) return f["width"].get<std::int64_t>();
  if (pj.is_object() && pj.contains("grid") && pj["grid"].value("kind", std::string()) == "time") return pj["grid"].value("width", std::int64_t(0));
  refuse(who + ": the measurement's selection " + nxr::io::text(*sel, "name") + " is not over a time grid (a product with a time factor)");
}
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
    s.name = "synthesis.lagfree_mixing"; s.kind = "setup";
    s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "once, from a stored cross-spectrum (no time axis), in double", {});
    s.inputs = {"cross_spectrum"}; s.outputs = {"mixing"};
    s.parameters = Json::object();
    s.matches = "Rheome +rheome/+cores lagfreenull (Rh = the symmetric square root of the zero-lag covariance), pairsqrt; compute nxr::synthesis::zeroLagMixing";
    s.description = "the ZERO-LAG MIXING of the band: the symmetric square root of the real part of the stored cross-spectrum at unit mean channel variance, an operator [M × M] over the cross-spectrum's channels "
                    "(meta.channels, in order) with its provenance (meta.provenance: the cross-spectrum's id and name). A surrogate's unmixed draw awaits it; it is composed ONCE into the first linear map of the chain "
                    "(apply_kernel's `pre_operator`), never applied per sample";
    s.plan = [](const Values& ins, const Json&, const PlanContext&) {
      statistic_input(ins.at("cross_spectrum"), "synthesis.lagfree_mixing", "cross_spectrum", "welch", {"cross_spectral_density"});
      Value y;
      y.cls = "operator";
      y.facts = Json{{"operator_of", operator_of(ins.at("cross_spectrum"))}, {"linear", true}, {"made_by", "synthesis.lagfree_mixing"}};
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
      try { B = nxr::synthesis::zeroLagMixing(cs.S); } catch (const std::invalid_argument& e) { refuse(std::string("synthesis.lagfree_mixing: ") + st.name + ": " + e.what()); }
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
    s.name = "synthesis.lagfree"; s.kind = "chunk";
    s.reads_shape_only = true;
    s.traits = node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::follows_input, "the noise is a function of the sample index (a counter-based generator) and reads no sample of the input; each output sample is that noise filtered by the shaping taps, made from the sample index on each side, so it asks no halo of the input; computed in double, stored as float32", "none from the input: the taps' reach (`taps_half` samples each side) is made from the sample index");
    s.inputs = {"x", "psd", "psd_channel_mean", "cross_spectrum", "power_series"}; s.outputs = {"y"};
    s.parameters = Json{{"band", nullptr}, {"shape", "morlet"}, {"half_width", 1.0 / 3.0}, {"taps_half", 0}, {"seed", 0}};
    s.matches = "Rheome +rheome/+cores lagfreenull and envelopenull (Gaussian noise with the data's mean spectrum, rescaled to the data's one-cycle envelope; the mixing is applied later, composed into the spatial operator); compute nxr::synthesis (shapedNoiseBlock, spectrumShapingTaps, analyticGainPower, envelopeScale, interpolateTileSeries); the scale is pointwise (expected, not realised)";
    s.description = "ONE LAG-FREE DRAW, UNMIXED, in place of the recording: per channel of the stored cross-spectrum one stream of Gaussian noise with the recording's own mean spectrum (the stored Welch PSD over those channels: never flat), "
                    "scaled in time to the data's tile power (`power_series`) over the draw's expected power through the band's analytic gain, so the mixed draw's analytic signal has the data's band envelope. A function of the SAMPLE "
                    "INDEX (a counter-based generator of `seed`): the draw does not depend on the chunk size; another `seed` is another draw. x is the recording's SHAPE (rows, rate, samples; `read: shape`): no sample is read. "
                    "Channels the cross-spectrum does not cover are zero. The output AWAITS the zero-lag mixing (`synthesis.lagfree_mixing`), applied once in the first linear map (apply_kernel's `pre_operator`): it commutes "
                    "with every linear time step before it (spec docs/specs/2026-11-17-surrogates.md, section 4). `taps_half` 0: from the PSD's resolution (half = fs / (2 df), at most 4096)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value& x = ins.at("x");
      recording_input(x, "synthesis.lagfree", false);
      band_params(p, "synthesis.lagfree", x.rate);
      need(p["seed"].is_number_integer() && p["seed"].get<std::int64_t>() >= 0, "synthesis.lagfree: seed is a whole number >= 0 (a draw)");
      need(p["taps_half"].is_number_integer() && p["taps_half"].get<int>() >= 0, "synthesis.lagfree: taps_half is a whole number >= 0 (0: from the PSD's resolution)");
      statistic_input(ins.at("psd"), "synthesis.lagfree", "psd", "welch", {"power_density", "power"});
      statistic_input(ins.at("cross_spectrum"), "synthesis.lagfree", "cross_spectrum", "welch", {"cross_spectral_density"});
      measurement_input(ins.at("psd_channel_mean"), "synthesis.lagfree", "psd_channel_mean");
      measurement_input(ins.at("power_series"), "synthesis.lagfree", "power_series");
      Value y = x;
      y.row.reset();
      y.facts.erase("read");                                                  // the draw has values: the steps after it read them
      y.facts["made_by"] = "synthesis.lagfree";
      y.facts["awaits_operator"] = Json{{"of", operator_of(ins.at("cross_spectrum"))}};
      y.facts["surrogate"] = Json{{"generator", "synthesis.lagfree"}, {"seed", p["seed"]}, {"psd", nxr::io::text(*ins.at("psd").row, "name")}, {"cross_spectrum", nxr::io::text(*ins.at("cross_spectrum").row, "name")},
                                  {"power_series", ins.at("power_series").facts["over"].value("selection", std::string())}, {"psd_channel_mean", ins.at("psd_channel_mean").facts["over"].value("selection", std::string())}};
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
      SurState st;
      const Value& x = ins.at("x");
      const auto man = c.bound.manifold(x.manifolds[0]);
      need(man.has_value(), "synthesis.lagfree: x's channels are not among the subject's manifolds");
      st.rows = nxr::io::integer(*man, "n_vertices");
      const auto cs = c.bound.cross_spectrum(nxr::io::text(*ins.at("cross_spectrum").row, "name"));
      st.channels = cs.channels;
      const std::int64_t M = static_cast<std::int64_t>(cs.channels.size());
      for (const std::int32_t ch : st.channels) need(ch >= 0 && ch < st.rows, "synthesis.lagfree: the cross-spectrum's channel " + std::to_string(ch) + " is not one of x's " + std::to_string(st.rows) + " rows");
      // the spectrum: the PSD's mean over the cross-spectrum's channels, on its frequency line (k df)
      const nxr::io::StoredStatistic P = c.bound.statistic(nxr::io::text(*ins.at("psd").row, "name"));
      need(P.shape.size() == 2, "synthesis.lagfree: the PSD " + nxr::io::text(*ins.at("psd").row, "name") + " is not [channels × frequencies]");
      const std::int64_t nf = P.shape[1];
      const double df = P.producer.value("frequency_resolution_hz", 0.0), fs = P.producer.value("sfreq", x.rate);
      need(df > 0 && std::abs(fs - x.rate) < 1e-6 * x.rate, "synthesis.lagfree: the PSD's resolution or sampling rate is not recorded, or it is not the recording's");
      Eigen::VectorXd freq(nf), gain(nf), psd;
      for (std::int64_t k = 0; k < nf; ++k) freq[k] = static_cast<double>(k) * df;
      const auto g = gain_of(p);
      for (std::int64_t k = 0; k < nf; ++k) gain[k] = g(freq[k]);
      const int half = p["taps_half"].get<int>() > 0 ? p["taps_half"].get<int>() : static_cast<int>(std::min(4096.0, std::max(64.0, std::round(fs / (2 * df)))));
      try {
        // the PSD's mean over the cross-spectrum's channels is a STORED MEASUREMENT read by name, one value per frequency
        const nxr::io::MeasurementValues pm = read_measurement(c.bound, ins.at("psd_channel_mean"), "mean");
        if (pm.members != 1 || pm.rows != nf) throw std::invalid_argument("psd_channel_mean is [" + std::to_string(pm.rows) + " frequencies x " + std::to_string(pm.members) + " members]; the PSD has " + std::to_string(nf) + " frequencies and the mean one member");
        psd = Eigen::Map<const Eigen::VectorXd>(pm.values.data(), nf);
        st.taps = nxr::synthesis::spectrumShapingTaps(freq, psd, fs, half);
        st.expected = nxr::synthesis::analyticGainPower(freq, psd, gain);             // the unscaled draw's mean |z|^2 through the analytic gain (unit-variance streams, unit mean channel variance after the mixing)
      } catch (const std::invalid_argument& e) { refuse(std::string("synthesis.lagfree: ") + e.what() + " (band " + p["band"].dump() + " Hz)"); }
      // the data's tile power: the mean derived at read (sum / count); a tile with no good sample takes the one before it
      const nxr::io::MeasurementValues S = read_measurement(c.bound, ins.at("power_series"), "power");            // mean |z|^2 per tile over the cross-spectrum's channels
      need(S.rows == 1, "synthesis.lagfree: power_series is [" + std::to_string(S.rows) + " rows x " + std::to_string(S.members) + " tiles]; one row of tiles (the channel set one member)");
      const std::int64_t n = S.members;
      st.stride = time_tile_of(c.bound, ins.at("power_series"), "synthesis.lagfree");
      need(st.stride > 1, "synthesis.lagfree: the power series' time grid records no tile");
      st.power.assign(static_cast<std::size_t>(n), std::nan(""));
      for (std::int64_t k = 0; k < n; ++k) if (std::isfinite(S.values[static_cast<std::size_t>(k)])) st.power[static_cast<std::size_t>(k)] = S.values[static_cast<std::size_t>(k)];   // a tile with no good sample is NaN: it takes the one before it
      std::int64_t first_ok = -1;
      for (std::int64_t k = 0; k < n; ++k) if (std::isfinite(st.power[static_cast<std::size_t>(k)])) { first_ok = k; break; }
      need(first_ok >= 0, "synthesis.lagfree: the power series holds no good tile");
      for (std::int64_t k = 0; k < n; ++k) { if (!std::isfinite(st.power[static_cast<std::size_t>(k)])) st.power[static_cast<std::size_t>(k)] = k > 0 ? st.power[static_cast<std::size_t>(k - 1)] : st.power[static_cast<std::size_t>(first_ok)]; }
      st.seed = static_cast<std::uint64_t>(p["seed"].get<std::int64_t>());
      st.record = Json{{"generator", "synthesis.lagfree"}, {"seed", p["seed"]}, {"channels", M}, {"taps", st.taps.size()}, {"taps_half", half}, {"frequency_resolution_hz", df}, {"expected_power", st.expected}, {"tile_samples", st.stride},
                       {"psd", nxr::io::text(*ins.at("psd").row, "name")}, {"cross_spectrum", nxr::io::text(*ins.at("cross_spectrum").row, "name")}, {"power_series", ins.at("power_series").facts["over"].value("selection", std::string())}, {"psd_channel_mean", ins.at("psd_channel_mean").facts["over"].value("selection", std::string())},
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
