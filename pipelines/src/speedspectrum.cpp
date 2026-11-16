#include "nxr/pipelines/speedspectrum.h"

#include <cmath>

#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/meeg_internal.h"

#ifndef NXR_COMPUTE_VERSION
#define NXR_COMPUTE_VERSION "unknown"
#endif

namespace nxr::pipelines {

using namespace nxr::io;
namespace tb = nxr::timebank;
namespace fb = nxr::filterbank;
namespace in = nxr::pipelines::internal;

namespace {

[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }

const char* tb_family(tb::Family f) {
  return f == tb::Family::Itersine ? "itersine" : f == tb::Family::Meyer ? "meyer" : f == tb::Family::Morlet ? "morlet" : f == tb::Family::Morse ? "morse" : f == tb::Family::Amor ? "amor" : "bump";
}
const char* fb_family(fb::Family f) {
  return f == fb::Family::MexHat ? "mexhat" : f == fb::Family::Heat ? "heat" : f == fb::Family::Itersine ? "itersine" : "logitersine";
}

tb::Design time_design(const SpeedSpectrumOptions& o) {
  tb::Design d(*o.family, o.gain, *o.sides);
  d.voices = o.voices; d.anchor = o.anchor; d.limits = o.limits; d.halfWidth = o.half_width; d.cycles = o.cycles; d.gamma = o.gamma; d.timeBandwidth = o.time_bandwidth;
  d.centres = o.centres;
  return d;
}

// A line of member or bin NUMBERS (spacing 1) of `n` points. nxr.subject@6.0 has folders only for time, frequency and eigenvalue lines, so the graph members sit on an eigenvalue-role line
// and the time members and speed bins on frequency-role lines; the line is an index, the members' real centres are in the fields' producer_json (open: the format ruling for a member axis).
std::string index_line(SubjectWriter& sub, const std::string& name, const char* role, const char* dim, std::int64_t n, const std::optional<std::string>& acquisition_id) {
  if (const auto l = sub.db().one("SELECT m.id AS id FROM manifold m WHERE m.subject_id = ? AND m.name = ? AND m.type = 'line'", {V(sub.id()), V(name)})) return text(*l, "id");
  ManifoldSpec ms;
  ms.name = name;
  ms.path = folder_for("line", "", std::string(role)) + "/" + name;
  ms.type = "line";
  ms.n_vertices = n;
  ms.topology = {"lattice", 1, 1, std::nullopt, {}};
  const auto lc = lattice_counts({n});
  if (acquisition_id) ms.cols.set("acquisition_id", *acquisition_id); else ms.cols.null("acquisition_id");
  if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
  if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
  if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
  GeometrySpec g;
  g.form = "closed_form";
  const bool eigen = std::string(role) == "eigen";
  g.cs_key = eigen ? "eigenvalue:speedspectrum" : "hz";
  DimensionSpec d;
  d.name = dim;
  d.n_vertices = n;
  d.spacing = 1.0;
  d.origin = 0.0;
  if (!eigen) d.unit = "Hz";
  g.dimensions.push_back(d);
  ms.geometry = g;
  return sub.manifold(std::move(ms));
}

Json vec_json(const Eigen::VectorXd& v) {
  Json a = Json::array();
  for (Eigen::Index i = 0; i < v.size(); ++i) a.push_back(std::isfinite(v(i)) ? Json(v(i)) : Json());
  return a;
}

}  // namespace

SpeedSpectrumResult speedspectrum_compute(DatasetWriter& ds, SubjectWriter& sub, const SpeedSpectrumOptions& opt, const std::function<void(const Json&)>& progress) {
  if (!opt.graph_family) invalid("speedspectrum: the graph filterbank family is required (mexhat, heat, itersine or logitersine)");
  if (!opt.family) invalid("speedspectrum: the time filterbank family is required (itersine, meyer, morlet, morse, amor or bump)");
  if (!(opt.gain > 0)) invalid("speedspectrum: the time filterbank gain is required and positive");
  if (!opt.sides) invalid("speedspectrum: the time filterbank sides are required (analytic or two-sided)");
  if (opt.speed.speedBins < 1) invalid("speedspectrum: speed bins must be at least 1");
  in::HodgeSpectraRequest req;
  req.recording = opt.recording; req.kernel = opt.kernel; req.package = opt.package; req.modes = opt.modes; req.chunks = opt.chunks; req.start = opt.start; req.stop = opt.stop;
  req.segment_seconds = opt.segment_seconds; req.segment_samples = opt.segment_samples; req.segment_level = opt.segment_level; req.welch = opt.welch;
  req.option_set = opt.option_set; req.exclude_bad_segments = opt.exclude_bad_segments; req.fmin = opt.fmin; req.fmax = opt.fmax; req.who = "speedspectrum";
  const in::HodgeSpectraPlan plan = in::plan_hodge_spectra(sub, req);
  const tb::Design design = time_design(opt);
  try { (void)tb::design(design, plan.welch.segmentSamples, plan.sfreq); } catch (const std::invalid_argument& e) { invalid(std::string("speedspectrum: the time bank: ") + e.what()); }
  const std::string stem = opt.name ? *opt.name : plan.recording + "_" + plan.kernel + "_speedspectrum";
  Json banks = Json{{"graph", Json{{"family", fb_family(*opt.graph_family)}, {"voices", opt.graph_voices}, {"filters", opt.graph_filters}}},
                    {"time", Json{{"family", tb_family(*opt.family)}, {"gain", opt.gain}, {"sides", *opt.sides == tb::Sides::Analytic ? "analytic" : "two-sided"}, {"centres", opt.centres}, {"cycles", opt.cycles},
                                  {"voices", opt.voices}, {"anchor", opt.anchor}, {"limits", opt.limits ? Json::array({opt.limits->first, opt.limits->second}) : Json()},
                                  {"half_width", opt.half_width}, {"gamma", opt.gamma}, {"time_bandwidth", opt.time_bandwidth}, {"nfft", plan.welch.segmentSamples}}},
                    {"speed", Json{{"bins", opt.speed.speedBins}, {"min", opt.speed.speedMin}, {"max", opt.speed.speedMax}}}};
  CacheKey key;
  key.pipeline = "speedspectrum";
  key.params = plan.params;
  key.params["name"] = stem;
  key.params["banks"] = banks;
  key.params["fmin_hz"] = opt.fmin;
  key.params["fmax_hz"] = opt.fmax;
  key.input = plan.input;
  key.compute = NXR_COMPUTE_VERSION;

  auto say = [&](const Json& j) { if (progress) progress(j); };
  SpeedSpectrumResult out;
  std::int64_t segments = 0, skipped = 0;
  double excluded_fraction = 0;
  Json axes = Json::object();
  const char* kind_names[2] = {"exact", "coexact"};
  const char* parts[4] = {"energy", "tile_speed", "quadrature", "histogram"};
  std::vector<std::pair<const char*, std::string>> have;
  std::string moved_aside;
  bool replaced = false;
  std::int64_t modes_exact = 0, modes_coexact = 0;
  auto load = [&](const Row& f) {
    const Json pj = Json::parse(opt_text(f, "producer_json").value_or("{}"), nullptr, false);
    if (pj.is_object()) { segments = pj.value("segments", std::int64_t(0)); skipped = pj.value("segments_skipped", std::int64_t(0)); excluded_fraction = pj.value("excluded_fraction", 0.0); }
    return 0;
  };
  auto compute = [&]() {
    const in::HodgeSpectraRun run = in::run_hodge_spectra(sub, req, progress);
    segments = run.segments; skipped = run.skipped; excluded_fraction = run.excluded_fraction;
    const Eigen::VectorXd lambda = run.mu.array().square();
    fb::DesignOptions dop;
    dop.voices = opt.graph_voices;
    dop.numFilters = opt.graph_filters;
    const fb::Bank graph = fb::design(*opt.graph_family, fb::Range::fromSpectrum(lambda), dop);
    const tb::TimeBank time = tb::design(design, plan.welch.segmentSamples, plan.sfreq);
    say(Json{{"stage", "bank"}, {"graph_members", graph.size()}, {"time_members", time.size()}});
    out.result = nxr::speedspectrum::compute(graph, time, run.mu, run.kind, run.spectra, opt.speed);
    for (const auto& p : out.result.parts) (p.kind == 0 ? modes_exact : modes_coexact) = p.modes;
    return 0;
  };
  auto store = [&](const int&) {
    const auto& R = out.result;
    const std::int64_t G = static_cast<std::int64_t>(R.graphCentres.size()), T = static_cast<std::int64_t>(R.timeCentres.size()), nb = static_cast<std::int64_t>(R.speedCentres.size());
    const std::string gl = index_line(sub, stem + "_graph_member", "eigen", "graph_member", G, plan.acquisition_id);
    const std::string tl = index_line(sub, stem + "_time_member", "spectral", "time_member", T, plan.acquisition_id);
    const std::string sl = index_line(sub, stem + "_speed_bin", "spectral", "speed_bin", nb, plan.acquisition_id);
    const std::string gt = sub.product({gl, tl}, plan.acquisition_id), gs = sub.product({gl, sl}, plan.acquisition_id);
    // STATISTICS (2026-11-13: computed over the recording from the segment partition): `<stem>_<kind>_<part>` on the axes graph member x time member (the
    // histogram: x speed bin), the index lines kept as the axes' manifolds
    have.push_back({"selection", in::segments_path_of(sub, plan.recording, stem)});
    for (const auto& p : R.parts) for (const char* part : parts) have.push_back({"statistic", in::statistic_path_of(sub, plan.recording, stem + "_" + kind_names[p.kind == 0 ? 0 : 1] + "_" + part)});
    moved_aside = in::clear_stored_nodes(ds, sub, have, opt.replace, stem, replaced);
    const std::string seg_id = in::welch_segments_of(sub, plan.recording, stem, plan.s0, plan.s1, plan.welch, segments, skipped);
    const Json pr = plan.projectors;
    const auto& wo = plan.welch;
    for (const auto& p : R.parts) {
      const char* kn = kind_names[p.kind == 0 ? 0 : 1];
      for (const char* part : parts) {
        const bool hist = std::string(part) == "histogram";
        const Eigen::MatrixXd& M = std::string(part) == "energy" ? p.tileEnergy : std::string(part) == "tile_speed" ? p.tileSpeed : std::string(part) == "quadrature" ? p.quadrature : p.histogram;
        std::vector<float> data(static_cast<std::size_t>(M.rows() * M.cols()));
        for (Eigen::Index g = 0; g < M.rows(); ++g) for (Eigen::Index c = 0; c < M.cols(); ++c) data[static_cast<std::size_t>(g * M.cols() + c)] = static_cast<float>(M(g, c));
        StatisticSpec f;
        f.name = stem + "_" + kn + "_" + part;
        f.path = in::statistic_path_of(sub, plan.recording, f.name);
        const std::string ps = part;
        f.measure = ps == "tile_speed" ? "speed" : ps == "quadrature" ? "quadrature_fraction" : "power";
        f.method = "welch";
        f.of_field_id = plan.recording_id;
        f.partition_selection_id = seg_id;
        f.n_members = segments;
        f.members_skipped = skipped;
        f.axes = Json::array({Json{{"name", "graph_member"}, {"manifold_id", gl}, {"size", M.rows()}},
                              Json{{"name", hist ? "speed_bin" : "time_member"}, {"manifold_id", hist ? sl : tl}, {"size", M.cols()}}});
        f.span = Json{{"start_sample", plan.s0}, {"stop_sample", plan.s1}, {"start_s", plan.t0 + static_cast<double>(plan.s0) / plan.sfreq}, {"stop_s", plan.t0 + static_cast<double>(plan.s1) / plan.sfreq},
                      {"chunks", plan.span}, {"chunks_read", plan.chunks_read}};
        f.data = view(data, {M.rows(), M.cols()});
        if (ps == "tile_speed") f.unit = "length/s";
        f.description = std::string("the speed spectrum (") + part + ") of the " + kn + " Hodge-Dirac modes of the sources of " + plan.recording + " by " + plan.kernel +
                        ": joint (eigenvalue x frequency) filterbank tiles; a wave of phase speed c lies on 2 pi f = c mu";
        f.options = in::welch_options_of(wo, opt.option_set, opt.exclude_bad_segments);
        Json pj = stamp(key);
        pj["stage"] = "speedspectrum";
        pj["by"] = "nxr-neuro speedspectrum compute";
        pj["producer"] = in::producer_string("speedspectrum compute");
        pj["part"] = part;
        pj["mode_kind"] = kn;
        pj["modes_of_kind"] = p.modes;
        pj["power_of_kind"] = p.power;
        pj["recording"] = plan.recording; pj["recording_id"] = plan.recording_id;
        pj["kernel"] = plan.kernel; pj["kernel_id"] = plan.kernel_id;
        pj["package"] = plan.package; pj["package_id"] = plan.package_id;
        pj["modes"] = plan.modes;
        pj["fused"] = "nxr-compute hodgeWeights: B [2K x C] = [a; b]; the flow weights b feed CrossWelch (one-bin groups, Re and Im)";
        pj["banks"] = banks;
        pj["graph_centre_wavenumbers"] = vec_json(R.graphCentres);
        pj["time_centre_frequencies_hz"] = vec_json(R.timeCentres);
        pj["speed_edges"] = vec_json(R.speedEdges);
        pj["speed_centres"] = vec_json(R.speedCentres);
        Json below = Json::array(), above = Json::array();
        for (Eigen::Index g = 0; g < p.outside.rows(); ++g) { below.push_back(p.outside(g, 0)); above.push_back(p.outside(g, 1)); }
        pj["energy_below_speed_range"] = below;
        pj["energy_above_speed_range"] = above;
        pj["frequency_range_hz"] = Json::array({opt.fmin, opt.fmax});
        pj["frequency_bins"] = Json::array({plan.first_bin, plan.last_bin});
        pj["frequency_resolution_hz"] = plan.df;
        pj["option_set"] = opt.option_set;
        pj["segment_samples"] = static_cast<std::int64_t>(wo.segmentSamples);
        pj["window_s"] = static_cast<double>(wo.segmentSamples) / plan.sfreq;
        pj["overlap"] = wo.overlap;
        pj["window"] = in::window_label(wo.window);
        pj["window_form"] = wo.periodic ? "periodic" : "symmetric";
        pj["detrend"] = in::detrend_label(wo.detrend);
        pj["scaling"] = in::scaling_label(wo.scaling);
        pj["sfreq"] = plan.sfreq;
        pj["analysis_window"] = Json{{"start_sample", plan.s0}, {"stop_sample", plan.s1}, {"chunks", plan.span}, {"chunks_read", plan.chunks_read}};
        pj["segments"] = segments;
        pj["segments_skipped"] = skipped;
        pj["bad_segments"] = plan.bad_segments;
        pj["excluded_fraction"] = excluded_fraction;
        pj["chunk_width"] = plan.chunk_width;
        pj["projectors"] = pr["projectors"];
        pj["projectors_applied_in_samples"] = pr["applied_in_samples"];
        pj["projectors_not_applied"] = pr["not_applied"];
        pj["ctf_compensation"] = pr["ctf_compensation"];
        f.producer = pj;
        sub.statistic(std::move(f));
      }
    }
  };
  const Outcome<int> o = cached<int>(sub, "statistic", std::nullopt, key, load, compute, store);
  out.hit = o.hit;
  out.seconds = o.seconds;
  Json fields = Json::array();
  if (!o.hit) for (const auto& p : out.result.parts) for (const char* part : parts) fields.push_back(stem + "_" + kind_names[p.kind == 0 ? 0 : 1] + "_" + part);
  out.info = Json{{"name", stem}, {"statistics", fields}, {"recording", plan.recording}, {"kernel", plan.kernel}, {"package", plan.package}, {"modes", plan.modes}, {"banks", banks},
                  {"frequency_range_hz", Json::array({opt.fmin, opt.fmax})}, {"segment_samples", static_cast<std::int64_t>(plan.welch.segmentSamples)}, {"segments", segments},
                  {"segments_skipped", skipped}, {"excluded_fraction", excluded_fraction}, {"replaced", replaced}, {"moved_aside", moved_aside.empty() ? Json() : Json(moved_aside)}, {"cache", o.hit ? "hit" : "miss"}};
  return out;
}

}  // namespace nxr::pipelines
