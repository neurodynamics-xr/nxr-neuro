// nxr-pipelines — the registered steps of the LAG-FREE NULL (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10; compute nxr::surrogate):
//
//   power_series       accumulate   the data's band power in TIME TILES: Σ |z|² over the channels the null covers and over each tile's good samples, and the count (a reduction over a time partition, folded chunk by chunk: the
//                                   dyadic tiles of 2^level samples on the recording's line); a STATISTIC [tiles × (sum, count)], the mean derived at read. It is the envelope the surrogate is scaled to (nxr `envelope_matched_null`'s
//                                   one-cycle RMS: a tile of 2^level samples, the nearest power of two to a cycle).
//   lagfree_surrogate  chunk        replaces the recording's samples by a LAG-FREE draw: Gaussian noise with the recording's own spectrum (the stored Welch PSD: never flat), mixed by √(Re C) of the band's stored cross-spectrum (the
//                                   zero-lag structure kept, no lag), scaled to the data's tile power (the stored `power_series`) over the surrogate's expected power through the band's analytic gain. A function of the SAMPLE INDEX
//                                   (a counter-based generator): the draw does not depend on the chunk size; a draw is the same graph with another `seed`. The input x gives the recording's rows, rate and samples; its values are not read.
//
// WHAT IT CANNOT SAY: the null has the data's covariance at zero lag, its spectrum and its envelope, and nothing else. Its spatial structure is the band's (the cross-spectrum's real part) at every frequency; the draw's
// envelope follows the data's up to the noise's own sampling variation (a pointwise scale: no reduction inside the node, the cardinal rule). Channels the cross-spectrum does not cover are zero in the draw (the kernel uses none of them).
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>

#include "nxr/filterbank.h"
#include "nxr/io/errors.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/registry.h"
#include "nxr/surrogate.h"

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
// ── power_series ──
struct SeriesState { std::vector<std::int32_t> channels; int level = 7; double rate = 0; Json record; };
struct SeriesAcc { std::map<std::int64_t, std::pair<double, double>> tiles; };
// ── lagfree_surrogate ──
struct SurState {
  Eigen::MatrixXd Rh;                       // [M x M] sqrt(Re C) over unit mean variance
  std::vector<std::int32_t> channels;       // the M recording rows the draw fills
  Eigen::VectorXd taps;
  std::vector<double> power;                // the data's mean |z|^2 per tile (gaps filled)
  std::int64_t stride = 1;
  double expected = 1;                      // the unscaled draw's mean |z|^2 through the band's analytic gain
  std::uint64_t seed = 0;
  Json record;
};
}  // namespace

void add_surrogate_steps(std::vector<StepSpec>& r) {
  {
    StepSpec s;
    s.name = "power_series"; s.kind = "accumulate";
    s.inputs = {"x", "cross_spectrum", "exclude"}; s.optional = {"exclude"}; s.outputs = {"series"};
    s.parameters = Json{{"level", 7}, {"bad_segments", "exclude"}};
    s.matches = "nxr vortex.envelope_matched_null's one-cycle RMS envelope (the sensors' band power, per tile of 2^level samples: the nearest power of two to a cycle); the dyadic time tiles of the recording's line";
    s.description = "the band power of the analytic recording in TIME TILES of 2^level samples (the recording's line, from sample 0): per tile the sum of |z|² over the channels of the stored cross-spectrum `cross_spectrum` "
                    "(the channels the null covers) and over the tile's good samples, and the number of those values; the mean (the one-cycle power the surrogate is scaled to) is DERIVED at read. A reduction over a time partition "
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
      const auto cs = crossspectrum_read(c.sub, nxr::io::text(*ins.at("cross_spectrum").row, "name"));
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
                                               {"description", "the band power of the recording in time tiles of " + std::to_string(tile) + " samples: the sum of |z|^2 over the null's channels and the tile's good samples, and the number of values (mean = sum / count at read)"},
                                               {"options", Json{{"level", st.level}, {"tile_samples", tile}, {"rate", st.rate}, {"channels", st.channels}, {"record", st.record}, {"upper_edges", "the time axis's `edges` are the tiles' upper edges in samples, from sample 0 of the recording's line"}}}}}};
      return Arrays{{"series", std::move(y)}};
    };
    r.push_back(std::move(s));
  }

  {
    StepSpec s;
    s.name = "lagfree_surrogate"; s.kind = "chunk";
    s.inputs = {"x", "psd", "cross_spectrum", "power_series"}; s.outputs = {"y"};
    s.parameters = Json{{"band", nullptr}, {"shape", "morlet"}, {"half_width", 1.0 / 3.0}, {"taps_half", 0}, {"seed", 0}};
    s.matches = "nxr vortex.lagfree_real and envelope_matched_null (Gaussian noise with the data's mean spectrum, mixed by sqrt(Re cov), rescaled to the data's one-cycle envelope); compute nxr::surrogate (lagfreeBlock, shapingTaps, sqrtPsd, interpolateSeries); the scale is pointwise (expected, not realised)";
    s.description = "a LAG-FREE DRAW in place of the recording (the project's null): per channel of the stored cross-spectrum, Gaussian noise with the recording's own mean spectrum (the stored Welch PSD over those channels: never flat), "
                    "mixed by sqrt(Re C) of the band's cross-spectrum (the zero-lag structure kept, every lag removed) and scaled in time to the data's tile power (`power_series`) over the draw's expected power through the band's analytic gain, so the "
                    "analytic signal of the draw has the data's band envelope. A function of the SAMPLE INDEX (a counter-based generator of `seed`): the draw does not depend on the chunk size; another `seed` is another draw. x gives the recording's "
                    "rows, rate and samples only; channels the cross-spectrum does not cover are zero. `taps_half` 0: from the PSD's resolution (half = fs / (2 df), at most 4096)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      const Value& x = ins.at("x");
      recording_input(x, "lagfree_surrogate", false);
      band_params(p, "lagfree_surrogate", x.rate);
      need(p["seed"].is_number_integer() && p["seed"].get<std::int64_t>() >= 0, "lagfree_surrogate: seed is a whole number >= 0 (a draw)");
      need(p["taps_half"].is_number_integer() && p["taps_half"].get<int>() >= 0, "lagfree_surrogate: taps_half is a whole number >= 0 (0: from the PSD's resolution)");
      statistic_input(ins.at("psd"), "lagfree_surrogate", "psd", "welch", {"power_density", "power"});
      statistic_input(ins.at("cross_spectrum"), "lagfree_surrogate", "cross_spectrum", "welch", {"cross_spectral_density"});
      statistic_input(ins.at("power_series"), "lagfree_surrogate", "power_series", "power_tiles", {"power_sum"});
      Value y = x;
      y.row.reset();
      y.facts["made_by"] = "lagfree_surrogate";
      y.facts["null"] = Json{{"seed", p["seed"]}, {"psd", nxr::io::text(*ins.at("psd").row, "name")}, {"cross_spectrum", nxr::io::text(*ins.at("cross_spectrum").row, "name")}, {"power_series", nxr::io::text(*ins.at("power_series").row, "name")}};
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
      SurState st;
      const Value& x = ins.at("x");
      const auto cs = crossspectrum_read(c.sub, nxr::io::text(*ins.at("cross_spectrum").row, "name"));
      st.channels = cs.channels;
      const std::int64_t M = static_cast<std::int64_t>(cs.channels.size());
      const double tr = cs.S.trace() / static_cast<double>(M);
      need(tr > 0, "lagfree_surrogate: the cross-spectrum holds no power");
      st.Rh = nxr::surrogate::sqrtPsd(cs.S / tr);
      // the spectrum: the PSD's mean over the cross-spectrum's channels, on its frequency line (k df)
      const nxr::io::StoredStatistic P = nxr::io::read_statistic(c.sub, nxr::io::text(*ins.at("psd").row, "name"));
      need(P.shape.size() == 2, "lagfree_surrogate: the PSD " + nxr::io::text(*ins.at("psd").row, "name") + " is not [channels × frequencies]");
      const std::int64_t Cp = P.shape[0], nf = P.shape[1];
      const double df = P.producer.value("frequency_resolution_hz", 0.0), fs = P.producer.value("sfreq", x.rate);
      need(df > 0 && std::abs(fs - x.rate) < 1e-6 * x.rate, "lagfree_surrogate: the PSD's resolution or sampling rate is not recorded, or it is not the recording's");
      Eigen::VectorXd freq(nf), psd = Eigen::VectorXd::Zero(nf);
      for (std::int64_t k = 0; k < nf; ++k) freq[k] = static_cast<double>(k) * df;
      for (const std::int32_t ch : st.channels) {
        need(ch >= 0 && ch < Cp, "lagfree_surrogate: the PSD has no channel " + std::to_string(ch));
        for (std::int64_t k = 0; k < nf; ++k) psd[k] += P.values[static_cast<std::size_t>(ch * nf + k)] / static_cast<double>(M);
      }
      const int half = p["taps_half"].get<int>() > 0 ? p["taps_half"].get<int>() : static_cast<int>(std::min(4096.0, std::max(64.0, std::round(fs / (2 * df)))));
      try { st.taps = nxr::surrogate::shapingTaps(freq, psd, fs, half); } catch (const std::invalid_argument& e) { refuse(std::string("lagfree_surrogate: ") + e.what()); }
      // the unscaled draw's mean |z|^2 through the analytic gain: 2 sum PSD g^2 / sum PSD (the analytic signal's power is twice the band-passed real signal's)
      const auto g = gain_of(p);
      double num = 0, den = 0;
      for (std::int64_t k = 0; k < nf; ++k) { const double w = (k == 0 || k == nf - 1) ? 0.5 : 1.0, gk = g(freq[k]); num += w * psd[k] * gk * gk; den += w * psd[k]; }
      need(num > 0 && den > 0, "lagfree_surrogate: the PSD holds no power in the band " + p["band"].dump() + " Hz");
      st.expected = 2.0 * num / den;
      // the data's tile power
      const nxr::io::StoredStatistic S = nxr::io::read_statistic(c.sub, nxr::io::text(*ins.at("power_series").row, "name"));
      need(S.shape.size() == 2 && S.shape[1] == 2, "lagfree_surrogate: power_series is not [tiles × (sum, count)]");
      const std::int64_t n = S.shape[0];
      st.stride = std::int64_t(1) << S.options.value("level", 0);
      need(st.stride > 1, "lagfree_surrogate: the power series records no tile level");
      st.power.assign(static_cast<std::size_t>(n), std::nan(""));
      for (std::int64_t k = 0; k < n; ++k) if (S.values[static_cast<std::size_t>(2 * k + 1)] > 0) st.power[static_cast<std::size_t>(k)] = S.values[static_cast<std::size_t>(2 * k)] / S.values[static_cast<std::size_t>(2 * k + 1)];
      std::int64_t first_ok = -1;
      for (std::int64_t k = 0; k < n; ++k) if (std::isfinite(st.power[static_cast<std::size_t>(k)])) { first_ok = k; break; }
      need(first_ok >= 0, "lagfree_surrogate: the power series holds no good tile");
      for (std::int64_t k = 0; k < n; ++k) { if (!std::isfinite(st.power[static_cast<std::size_t>(k)])) st.power[static_cast<std::size_t>(k)] = k > 0 ? st.power[static_cast<std::size_t>(k - 1)] : st.power[static_cast<std::size_t>(first_ok)]; }
      st.seed = static_cast<std::uint64_t>(p["seed"].get<std::int64_t>());
      st.record = Json{{"seed", p["seed"]}, {"channels", M}, {"taps", st.taps.size()}, {"taps_half", half}, {"frequency_resolution_hz", df}, {"expected_power", st.expected}, {"tile_samples", st.stride}, {"psd", nxr::io::text(*ins.at("psd").row, "name")},
                       {"cross_spectrum", nxr::io::text(*ins.at("cross_spectrum").row, "name")}, {"power_series", nxr::io::text(*ins.at("power_series").row, "name")}, {"reference", "nxr vortex.lagfree_real, envelope_matched_null"},
                       {"warnings", Json::array({"the null has the data's zero-lag covariance, spectrum and envelope and nothing else: a result beyond it is beyond what mixing and amplitude modulation produce, not a proof of a wave",
                                                 "the draw's envelope follows the data's up to the noise's own sampling variation (a pointwise scale), where nxr rescales each draw to the data's realised envelope",
                                                 "the draw's spatial structure is the band's (the cross-spectrum's real part), the same at every frequency"})}};
      return std::any(std::make_shared<SurState>(std::move(st)));
    };
    s.record = [](const std::any& st) { return std::any_cast<const std::shared_ptr<SurState>&>(st)->record; };
    s.halo = [](const std::any&) { return std::int64_t(0); };            // the draw is made on the halo the later steps ask, from the sample index: it needs none of the recording's
    s.chunk_at = [](const std::any& st_, const Blocks& ins, std::int64_t keep, const Place& at) {
      const SurState& st = *std::any_cast<const std::shared_ptr<SurState>&>(st_);
      const Block& b = ins.at("x");
      const std::int64_t kl = std::min(keep, b.left), kr = std::min(keep, b.right), w = b.own() + kl + kr, first = at.first - kl;
      const Eigen::VectorXd pw = nxr::surrogate::interpolateSeries(st.power, st.stride, first - st.stride / 2, w);   // the tiles' value k sits at the tile's centre
      Eigen::VectorXd scale(w);
      for (std::int64_t j = 0; j < w; ++j) scale[j] = std::sqrt(std::max(pw[j], 0.0) / st.expected);
      const Eigen::MatrixXd Y = nxr::surrogate::lagfreeBlock(st.Rh, st.taps, st.seed, first, w, scale);
      Block y{b.rows, w, kl, kr, std::vector<float>(static_cast<std::size_t>(b.rows * w), 0.0f), {}};
      for (std::size_t m = 0; m < st.channels.size(); ++m)
        for (std::int64_t j = 0; j < w; ++j) y.re[static_cast<std::size_t>(st.channels[m] * w + j)] = static_cast<float>(Y(static_cast<Eigen::Index>(m), j));
      return Blocks{{"y", std::move(y)}};
    };
    r.push_back(std::move(s));
  }
}

}  // namespace nxr::pipelines::graph
