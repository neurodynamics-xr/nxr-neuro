// nxr-pipelines — THE VERBS AS REGISTERED STEPS: what the hand-wired verbs computed, as steps of the registry that read through the run's bound services (surfaces, their
// ladders, spatial bands, glued domains; nothing else of the store) and hand back described arrays the definition stores. One step a verb, its maths the pure functions of
// compute/ and meeg/; the YAML definitions in pipelines/definitions/ are the verbs.
//
//   pointspread   setup  forward + kernel (+ the recording it is of)  ->  spread: the instrument's blur at sampled vertices (a statistic)
//   wishart_null  setup  recording + bursts + psd (+ the analytic windows by name)  ->  per_burst, run: the lagged share against a Wishart null
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "nxr/cores.h"
#include "nxr/filterbank.h"
#include "nxr/surrogate.h"
#include <limits>
#include <map>
#include "nxr/io/finder.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/scalespace.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;
using nxr::io::text;

namespace {

[[noreturn]] void refuse(const std::string& m) { throw std::invalid_argument(m); }

Value statistic_value() { Value v; v.cls = "statistic"; return v; }

// ── pointspread ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE INSTRUMENT'S BLUR (the hand-wired `pointspread compute` it replaces): a unit current at each sampled vertex (x, y and z separately) through the head model's gain and the
// kernel (the resolution matrix's column), read by the spatial events' chain: its power at each level of the surface's ladder (the spatial bands, on the glued domain), the peak of
// the reference level, the patch around it cut at a fraction of the peak, and the characteristic level of its centre. Per source and orientation: the peak's distance from the
// source, the patch's equal-area diameter and area, the characteristic level and its size, the peak power.
struct SpreadState { Value forward, kernel; Json p; };
StepSpec pointspread_step() {
  StepSpec s;
  s.name = "pointspread";
  s.kind = "setup";
  s.inputs = {"forward", "kernel"};
  s.outputs = {"spread"};
  s.parameters = Json{{"levels", nullptr}, {"reference_level", nullptr}, {"fraction", 0.5}, {"vertices", 64}, {"list", Json::array()}, {"transform", "auto"}, {"modes", nullptr},
                      {"order", nullptr}, {"tolerance", nullptr}, {"spatial_shape", "mexhat"}};
  s.matches = "the spatial events' chain on a planted point source (nxr.detect spatial events); meeg resolution matrix K G";
  s.description = "the instrument's blur: a unit current at sampled vertices through the head model and the kernel, read by the spatial events' chain";
  s.traits = NodeTraits{};
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double, once a run");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    if (!ins.count("forward") || !ins.count("kernel")) refuse("pointspread: forward and kernel are its inputs");
    if (!p["levels"].is_array() || p["levels"].empty()) refuse("pointspread: levels is a list of the surface ladder's levels");
    if (p["reference_level"].is_null()) refuse("pointspread: reference_level is one of the levels");
    const double f = p.value("fraction", 0.5);
    if (!(f > 0 && f < 1)) refuse("pointspread: fraction is in (0, 1)");
    return Values{{"spread", statistic_value()}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) { return std::any(SpreadState{ins.at("forward"), ins.at("kernel"), p}); };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const SpreadState& st = std::any_cast<const SpreadState&>(state);
    const Json& p = st.p;
    const Bound& b = ctx.bound;
    const nxr::io::Row& fwd = *st.forward.row;
    const nxr::io::Row& ker = *st.kernel.row;
    const std::string kname = text(ker, "name");
    if (nxr::io::opt_text(ker, "to_manifold_id") != nxr::io::opt_text(fwd, "from_manifold_id")) refuse("pointspread: the kernel " + kname + " reads other sources than the head model " + text(fwd, "name"));
    const auto csel = b.selection(nxr::io::opt_text(ker, "from_selection_id").value_or(""));
    if (!csel || nxr::io::is_null(*csel, "array_path")) refuse(kname + ": its channels (from_selection) are not stored");
    const std::vector<std::int32_t> chans = b.array_int32(text(*csel, "array_path"));
    const std::int64_t M = static_cast<std::int64_t>(chans.size());
    std::vector<std::int64_t> gshape, kshape;
    const std::vector<double> gf = b.array_double(text(fwd, "path"), &gshape), kf = b.array_double(text(ker, "path"), &kshape);
    if (gshape.size() != 2 || gshape[1] % 3 != 0) refuse(text(fwd, "name") + ": the gain is not [channels x 3N]");
    const std::int64_t Mall = gshape[0], N = gshape[1] / 3;
    if (kshape.size() != 2 || kshape[0] != 3 * N || kshape[1] != M) refuse(kname + ": its array is not [3N x the channels]");
    Eigen::MatrixXd G(M, 3 * N), K(3 * N, M);
    for (std::int64_t i = 0; i < M; ++i) {
      const std::int32_t c = chans[static_cast<std::size_t>(i)];
      if (c < 0 || c >= Mall) refuse(kname + ": a channel index is outside the gain's channels");
      for (std::int64_t j = 0; j < 3 * N; ++j) G(i, j) = gf[static_cast<std::size_t>(c * 3 * N + j)];
    }
    for (std::int64_t r = 0; r < 3 * N; ++r) for (std::int64_t c = 0; c < M; ++c) K(r, c) = kf[static_cast<std::size_t>(r * M + c)];
    const auto man = b.manifold(nxr::io::opt_text(fwd, "from_manifold_id").value_or(""));
    if (!man) refuse(text(fwd, "name") + ": its source surface is not stored");
    const SurfaceInput S = b.surface(text(*man, "name"), true);
    if (S.n_vertices() != N) refuse("pointspread: the head model's sources are not the vertices of " + S.name);
    const internal::GluedDomain dom = b.glued_domain(S);
    if (!dom.found) refuse("pointspread: " + dom.why);
    const std::int64_t Vg = dom.surface.n_vertices();
    std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
    for (std::int64_t f = 0; f < dom.surface.n_faces(); ++f)
      for (int k = 0; k < 3; ++k) {
        std::int32_t a = dom.surface.faces[static_cast<std::size_t>(3 * f + k)], c = dom.surface.faces[static_cast<std::size_t>(3 * f + (k + 1) % 3)];
        if (a > c) std::swap(a, c);
        pairs.emplace_back(a, c);
      }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(Vg); mesh.positions = dom.surface.vertices; mesh.faces = dom.surface.faces;
    const std::vector<double> va = nxr::ladder::vertexAreas(mesh);
    const Eigen::VectorXd area = Eigen::Map<const Eigen::VectorXd>(va.data(), static_cast<Eigen::Index>(va.size()));
    const Eigen::MatrixXd pos = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(dom.surface.vertices.data(), Vg, 3);
    std::vector<int> sample = p.value("list", std::vector<int>());
    if (sample.empty()) { const int n = std::max(1, std::min<int>(p.value("vertices", 64), static_cast<int>(N))); for (int i = 0; i < n; ++i) sample.push_back(static_cast<int>(static_cast<std::int64_t>(i) * N / n)); }
    for (const int v : sample) if (v < 0 || v >= N) refuse("pointspread: vertex " + std::to_string(v) + " is not one of the " + std::to_string(N) + " vertices");
    const std::vector<int> levels = p["levels"].get<std::vector<int>>();
    const int reference = p["reference_level"].get<int>();
    const auto refit = std::find(levels.begin(), levels.end(), reference);
    if (refit == levels.end()) refuse("pointspread: the reference level must be one of the levels");
    const std::int64_t NL = static_cast<std::int64_t>(levels.size()), ref = refit - levels.begin();
    std::vector<SpatialBand> bands;
    std::vector<double> size_cm;
    Eigen::VectorXd norm2(Vg * NL);
    for (std::int64_t li = 0; li < NL; ++li) {
      const SurfaceLevel L = b.surface_level(S, levels[static_cast<std::size_t>(li)]);
      SpatialBandOptions bo;
      bo.transform = p.value("transform", std::string("auto"));
      if (!p["modes"].is_null()) bo.modes = p["modes"].get<std::int64_t>();
      if (!p["order"].is_null()) bo.order = p["order"].get<int>();
      if (!p["tolerance"].is_null()) bo.tolerance = p["tolerance"].get<double>();
      bo.spatial_shape = p.value("spatial_shape", std::string("mexhat"));
      bo.glued = true;
      bands.push_back(b.make_spatial_band(S, L, bo));
      const auto& m = L.bank.members[static_cast<std::size_t>(L.l)];
      const double lc = m.lo > 0 ? std::sqrt(m.lo * m.hi) : m.hi / std::sqrt(2.0);
      size_cm.push_back(100.0 * 2.0 * M_PI / std::sqrt(lc));
      std::vector<double> n2(static_cast<std::size_t>(Vg), 0.0);
      for (std::int64_t k0 = 0; k0 < N; k0 += 256) {
        const std::int64_t w = std::min<std::int64_t>(256, N - k0);
        Eigen::MatrixXd I = Eigen::MatrixXd::Zero(N, w);
        for (std::int64_t i = 0; i < w; ++i) I(k0 + i, i) = 1.0;
        const Eigen::MatrixXd A = bands.back().apply_glued(I);
        for (std::int64_t v = 0; v < Vg; ++v) n2[static_cast<std::size_t>(v)] += A.row(v).squaredNorm();
      }
      for (std::int64_t v = 0; v < Vg; ++v) {
        if (!(n2[static_cast<std::size_t>(v)] > 0)) refuse("pointspread: the atom of glued vertex " + std::to_string(v) + " has norm 0 at level " + std::to_string(levels[static_cast<std::size_t>(li)]));
        norm2(v * NL + li) = n2[static_cast<std::size_t>(v)];
      }
    }
    const double fraction = p.value("fraction", 0.5);
    const int NS = static_cast<int>(sample.size());
    Array out;
    out.rows = NS; out.cols = 18;
    out.data.assign(static_cast<std::size_t>(NS) * 18, std::nan(""));
    for (int si = 0; si < NS; ++si) {
      for (int o = 0; o < 3; ++o) {
        const Eigen::VectorXd shat = K * G.col(3 * static_cast<std::int64_t>(sample[static_cast<std::size_t>(si)]) + o);   // the resolution matrix's column of the unit current
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Vg * NL, 1);
        for (std::int64_t li = 0; li < NL; ++li)
          for (int c = 0; c < 3; ++c) {
            Eigen::MatrixXd comp(N, 1);
            for (std::int64_t v = 0; v < N; ++v) comp(v, 0) = shat[3 * v + c];
            const Eigen::MatrixXd h = bands[static_cast<std::size_t>(li)].apply_glued(comp);
            for (std::int64_t v = 0; v < Vg; ++v) P(v * NL + li, 0) += h(v, 0) * h(v, 0);
          }
        Eigen::MatrixXd Pr(Vg, 1);
        std::int64_t peak = 0;
        for (std::int64_t v = 0; v < Vg; ++v) { Pr(v, 0) = P(v * NL + ref, 0); if (Pr(v, 0) > Pr(peak, 0)) peak = v; }
        const double mx = Pr(peak, 0);
        if (!(mx > 0)) continue;
        const nxr::cores::Cores C = nxr::cores::corePatches(Pr, pairs, area, pos, fraction * mx, 0);
        std::int64_t pi = -1;
        for (std::int64_t i = 0; i < C.size(); ++i) if (pi < 0 || C.peak[static_cast<std::size_t>(i)] > C.peak[static_cast<std::size_t>(pi)]) pi = i;
        if (pi < 0) continue;
        const std::int32_t cv = C.centre_vertex[static_cast<std::size_t>(pi)];
        Eigen::MatrixXd Pc(NL, 1);
        Eigen::VectorXd n2c(NL);
        for (std::int64_t li = 0; li < NL; ++li) { Pc(li, 0) = P(static_cast<std::int64_t>(cv) * NL + li, 0); n2c[li] = norm2(static_cast<std::int64_t>(cv) * NL + li); }
        const nxr::scalespace::LevelChoice ch = nxr::scalespace::selectLevel(Pc, n2c, 1);
        std::int64_t src = -1;
        for (std::size_t k = 0; k < dom.support.size(); ++k) if (dom.support[k] == sample[static_cast<std::size_t>(si)]) { src = static_cast<std::int64_t>(k); break; }
        double* r = &out.data[static_cast<std::size_t>((si * 3 + o) * 6)];
        r[0] = src >= 0 ? 100.0 * (pos.row(peak) - pos.row(src)).norm() : std::nan("");
        r[1] = 100.0 * 2.0 * std::sqrt(C.area[static_cast<std::size_t>(pi)] / M_PI);
        r[2] = 1e4 * C.area[static_cast<std::size_t>(pi)];
        r[3] = levels[static_cast<std::size_t>(ch.index(0, 0))];
        r[4] = size_cm[static_cast<std::size_t>(ch.index(0, 0))];
        r[5] = mx;
      }
      if (ctx.say) ctx.say(Json{{"stage", "pointspread"}, {"vertex", sample[static_cast<std::size_t>(si)]}, {"done", si + 1}, {"of", NS}});
    }
    Json vals = Json::array(), ors = Json::array({Json{{"name", "x"}}, Json{{"name", "y"}}, Json{{"name", "z"}}});
    for (const int v : sample) vals.push_back(Json{{"name", "vertex" + std::to_string(v)}, {"vertex", v}});
    out.meta["describe"] = Json{
        {"measure", "point_spread"}, {"method", "point_source"}, {"precision", "float64"}, {"n_members", NS}, {"shape", Json::array({NS, 3, 6})},
        {"axes", Json::array({Json{{"name", "source_vertex"}, {"values", vals}, {"size", NS}}, Json{{"name", "orientation"}, {"values", ors}, {"size", 3}},
                              Json{{"name", "measure"}, {"size", 6}, {"values", Json::array({Json{{"name", "peak_error_cm"}}, Json{{"name", "diameter_cm"}}, Json{{"name", "area_cm2"}},
                                                                                           Json{{"name", "characteristic_level"}}, Json{{"name", "size_cm"}}, Json{{"name", "peak_power"}}})}}})},
        {"options", Json{{"forward", text(fwd, "name")}, {"kernel", kname}, {"levels", levels}, {"reference_level", reference}, {"fraction", fraction}, {"sample", sample}, {"surface", S.name}}},
        {"producer", Json{{"warnings", Json::array({"the blur of THIS head model and kernel (their noise model and regularisation) for a source ALONE: many sources blur together differently",
                                                    "sizes are octave-spaced levels; the patch is cut at a fraction of the peak (a point source has no noise floor), not at the events' false-alarm threshold",
                                                    "a source is planted along each Cartesian axis, no orientation is assumed; a vertex the sample misses is not measured"})}}},
        {"description", "the instrument's blur: a unit current at each sampled vertex (x, y and z separately) through " + text(fwd, "name") + " and " + kname + ", read by the spatial events' chain"}};
    return Arrays{{"spread", std::move(out)}};
  };
  return s;
}


// ── wishart_null ────────────────────────────────────────────────────────────────────────────────────────────────
// THE WISHART NULL OF THE BURSTS' LAGGED SHARE (the hand-wired `wishart compute` it replaces): per burst, the analytic windows wholly inside it summed (Q, the burst's second moment
// between the sensors), its lagged share |Im Q|^2 / |Q|^2, and the same share under a zero-lag Gaussian null with the burst's own covariance and dof = its duration x the band's
// effective bandwidth (the PSD's, through the band's analytic gain): the p, the BH q; and the run's mean share against the null's.
double lagged_share(const Eigen::MatrixXcd& Q) { const double t = Q.squaredNorm(); return t > 0 ? Q.imag().squaredNorm() / t : std::numeric_limits<double>::quiet_NaN(); }
struct WishartState { Value recording, bursts, psd; Json p; };
StepSpec wishart_step() {
  StepSpec s;
  s.name = "wishart_null";
  s.kind = "setup";
  s.inputs = {"recording", "bursts", "psd"};
  s.outputs = {"per_burst", "run"};
  s.parameters = Json{{"windows", nullptr}, {"band", nullptr}, {"shape", "morlet"}, {"draws", 1000}, {"seed", 0}};
  s.matches = "nxr vortex.wishart, dof_fraction; compute nxr::surrogate::wishartDraw, effectiveBandwidth, benjaminiHochberg";
  s.description = "per burst, the lagged share of its sensors' second moment against a zero-lag Wishart null of its own covariance and degrees of freedom";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double from stored windows");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    for (const char* k : {"recording", "bursts", "psd"}) if (!ins.count(k)) refuse(std::string("wishart_null: ") + k + " is not given");
    if (!p["windows"].is_string()) refuse("wishart_null: windows names the analytic cross-spectrum series of the band (<name>_<band>_windows)");
    if (!p["band"].is_array() || p["band"].size() != 2 || !(p["band"][0].get<double>() >= 0 && p["band"][0].get<double>() < p["band"][1].get<double>())) refuse("wishart_null: band is [lo, hi] Hz, lo < hi");
    if (p.value("draws", 1000) < 20) refuse("wishart_null: draws is at least 20");
    const std::string sh = p.value("shape", std::string("morlet"));
    if (sh != "morlet" && sh != "meyer") refuse("wishart_null: shape is morlet or meyer");
    return Values{{"per_burst", statistic_value()}, {"run", statistic_value()}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) { return std::any(WishartState{ins.at("recording"), ins.at("bursts"), ins.at("psd"), p}); };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const WishartState& st = std::any_cast<const WishartState&>(state);
    const Json& p = st.p;
    const Bound& b = ctx.bound;
    const std::string windows = p["windows"].get<std::string>();
    const CrossSpectrumWindows win = b.cross_spectrum_windows(windows);
    if (win.level < 0 || win.sum.empty()) refuse("wishart_null: " + windows + " is not an analytic cross-spectrum series on time tiles");
    std::vector<std::pair<double, double>> spans = b.spans(text(*st.bursts.row, "id"));
    if (spans.empty()) refuse("wishart_null: the bursts hold no burst");
    if (!st.recording.time || nxr::io::opt_text(*st.bursts.row, "manifold_id") != st.recording.time) refuse("wishart_null: the bursts are on another time line than the recording's");
    const double rate = st.recording.rate, dt = 1.0 / rate, t0 = st.recording.origin;
    const double width = static_cast<double>(std::int64_t(1) << win.level) * dt, eps = 0.5 * dt;
    const nxr::io::StoredStatistic P = b.statistic(text(*st.psd.row, "name"));
    if (P.shape.size() != 2) refuse("wishart_null: the PSD is not [channels x frequencies]");
    const double df = P.producer.value("frequency_resolution_hz", 0.0);
    if (!(df > 0)) refuse("wishart_null: the PSD records no frequency resolution");
    const std::int64_t nf = P.shape[1];
    const std::vector<double> band = p["band"].get<std::vector<double>>();
    const std::string shape = p.value("shape", std::string("morlet"));
    Eigen::VectorXd freq(nf), psd = Eigen::VectorXd::Zero(nf), gain(nf);
    for (std::int64_t k = 0; k < nf; ++k) freq[k] = static_cast<double>(k) * df;
    for (const std::int32_t ch : win.channels) {
      if (ch < 0 || ch >= P.shape[0]) refuse("wishart_null: the PSD has no channel " + std::to_string(ch));
      for (std::int64_t k = 0; k < nf; ++k) psd[k] += P.values[static_cast<std::size_t>(ch * nf + k)] / static_cast<double>(win.channels.size());
    }
    for (std::int64_t k = 0; k < nf; ++k) gain[k] = shape == "morlet" ? nxr::filterbank::morletBand(freq[k], band[0], band[1]) : nxr::filterbank::meyerBand(freq[k], band[0], band[1], 1.0 / 3.0);
    double Beff = 0;
    try { Beff = nxr::surrogate::effectiveBandwidth(freq, psd, gain); } catch (const std::invalid_argument& e) { refuse(std::string("wishart_null: ") + e.what()); }
    std::map<std::int64_t, std::size_t> at;
    for (std::size_t w = 0; w < win.index.size(); ++w) at[win.index[w]] = w;
    const std::int64_t B = static_cast<std::int64_t>(spans.size());
    const Eigen::Index C = static_cast<Eigen::Index>(win.channels.size());
    std::vector<Eigen::MatrixXcd> Q(static_cast<std::size_t>(B));
    std::vector<double> dof(static_cast<std::size_t>(B), std::numeric_limits<double>::quiet_NaN());
    std::int64_t used = 0;
    for (std::int64_t k = 0; k < B; ++k) {
      const double a = spans[static_cast<std::size_t>(k)].first, z = spans[static_cast<std::size_t>(k)].second;
      const std::int64_t k0 = std::max<std::int64_t>(static_cast<std::int64_t>(std::ceil((a - t0 - eps) / width)), 0), k1 = static_cast<std::int64_t>(std::floor((z - t0 + eps) / width)) - 1;
      Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(C, C);
      bool whole = k1 >= k0;
      for (std::int64_t w = k0; whole && w <= k1; ++w) { const auto it = at.find(w); if (it == at.end()) whole = false; else sum += win.sum[it->second]; }
      if (!whole) continue;
      Q[static_cast<std::size_t>(k)] = sum;
      dof[static_cast<std::size_t>(k)] = static_cast<double>(k1 - k0 + 1) * width * Beff;
      ++used;
    }
    if (used == 0) refuse("wishart_null: no burst is whole in the stored windows (tile-aligned bursts inside the windows' span)");
    const int draws = p.value("draws", 1000);
    const std::uint64_t seed = p.value("seed", std::uint64_t(0));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd share(B), nullmean = Eigen::VectorXd::Zero(B), exceed = Eigen::VectorXd::Zero(B), pv(B);
    std::vector<double> run_null(static_cast<std::size_t>(draws), 0.0);
    for (std::int64_t k = 0; k < B; ++k) {
      share[k] = Q[static_cast<std::size_t>(k)].size() ? lagged_share(Q[static_cast<std::size_t>(k)]) : nan;
      if (!std::isfinite(share[k])) { pv[k] = nan; continue; }
      const Eigen::MatrixXd R = Q[static_cast<std::size_t>(k)].real();
      for (int j = 0; j < draws; ++j) {
        const double sj = lagged_share(nxr::surrogate::wishartDraw(R, dof[static_cast<std::size_t>(k)], seed + static_cast<std::uint64_t>(k) * 7919u, static_cast<std::uint64_t>(j)));
        nullmean[k] += sj / draws; run_null[static_cast<std::size_t>(j)] += sj;
        if (sj >= share[k]) exceed[k] += 1;
      }
      pv[k] = static_cast<double>(exceed[k] + 1) / static_cast<double>(draws + 1);
    }
    const Eigen::VectorXd q = nxr::surrogate::benjaminiHochberg(pv);
    double mean_share = 0, nmean = 0, nsd = 0;
    for (std::int64_t k = 0; k < B; ++k) if (std::isfinite(share[k])) mean_share += share[k] / static_cast<double>(used);
    for (int j = 0; j < draws; ++j) { run_null[static_cast<std::size_t>(j)] /= static_cast<double>(used); nmean += run_null[static_cast<std::size_t>(j)] / draws; }
    for (int j = 0; j < draws; ++j) nsd += std::pow(run_null[static_cast<std::size_t>(j)] - nmean, 2) / (draws - 1);
    nsd = std::sqrt(nsd);
    const double prun = static_cast<double>(1 + std::count_if(run_null.begin(), run_null.end(), [&](double x) { return x >= mean_share; })) / static_cast<double>(draws + 1);
    const Json warnings = Json::array({"the null is a zero-lag Gaussian process with the burst's own covariance and `dof` independent samples: a burst beyond it has lags beyond sampling noise, not a wave",
                                       "dof = the burst's duration x the band's effective bandwidth (the PSD's, through the analytic gain): a spectrum with structure inside the band changes it",
                                       "the lagged share is a SENSOR-level quantity (the share of the second moment in quadrature between the sensors)"});
    const Json opts = Json{{"draws", draws}, {"seed", seed}, {"band", band}, {"shape", shape}, {"effective_bandwidth_hz", Beff}, {"windows", windows}, {"window_level", win.level}};
    Json names = Json::array();
    for (std::int64_t k = 0; k < B; ++k) names.push_back(Json{{"name", "burst" + std::to_string(k)}, {"start", spans[static_cast<std::size_t>(k)].first}, {"stop", spans[static_cast<std::size_t>(k)].second}});
    Array pb;
    pb.rows = B; pb.cols = 5;
    pb.data.resize(static_cast<std::size_t>(B * 5));
    for (std::int64_t k = 0; k < B; ++k) {
      double* r = &pb.data[static_cast<std::size_t>(5 * k)];
      r[0] = share[k]; r[1] = std::isfinite(share[k]) ? nullmean[k] : nan; r[2] = pv[k]; r[3] = q[k]; r[4] = dof[static_cast<std::size_t>(k)];
    }
    pb.meta["describe"] = Json{{"measure", "null_p"}, {"method", "wishart_null"}, {"precision", "float64"}, {"n_members", used}, {"options", opts}, {"producer", Json{{"warnings", warnings}}},
                               {"axes", Json::array({Json{{"name", "burst"}, {"values", names}, {"size", B}},
                                                     Json{{"name", "test"}, {"size", 5}, {"values", Json::array({Json{{"name", "lagged_share"}}, Json{{"name", "null_mean"}}, Json{{"name", "p"}}, Json{{"name", "q"}}, Json{{"name", "dof"}}})}}})},
                               {"description", "per burst: the share of the second moment in quadrature between the sensors, its mean under the Wishart null (" + std::to_string(draws) + " draws), the p, the BH q and the dof"}};
    Array pr;
    pr.rows = 4; pr.cols = 1;
    pr.data = {mean_share, nmean, nsd, prun};
    pr.meta["describe"] = Json{{"measure", "null_p"}, {"method", "wishart_null"}, {"precision", "float64"}, {"n_members", used}, {"options", opts}, {"producer", Json{{"warnings", warnings}}},
                               {"axes", Json::array({Json{{"name", "test"}, {"size", 4}, {"values", Json::array({Json{{"name", "mean_share"}}, Json{{"name", "null_mean"}}, Json{{"name", "null_sd"}}, Json{{"name", "p"}}})}}})},
                               {"description", "the run's mean lagged share over the bursts against the Wishart null's mean over the same bursts (" + std::to_string(draws) + " draws)"}};
    return Arrays{{"per_burst", std::move(pb)}, {"run", std::move(pr)}};
  };
  return s;
}

}  // namespace

void append_verb_steps(std::vector<StepSpec>& r) {
  r.push_back(pointspread_step());
  r.push_back(wishart_step());
}

}  // namespace nxr::pipelines::graph
