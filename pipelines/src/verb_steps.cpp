// nxr-pipelines — THE VERBS AS REGISTERED STEPS: what the hand-wired verbs computed, as steps of the registry that read through the run's bound services (surfaces, their
// ladders, spatial bands, glued domains; nothing else of the store) and hand back described arrays the definition stores. One step a verb, its maths the pure functions of
// compute/ and meeg/; the YAML definitions in pipelines/definitions/ are the verbs.
//
//   pointspread  setup  forward + kernel (+ the recording it is of)  ->  spread: the instrument's blur at sampled vertices (a statistic)
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "nxr/cores.h"
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

}  // namespace

void append_verb_steps(std::vector<StepSpec>& r) {
  r.push_back(pointspread_step());
}

}  // namespace nxr::pipelines::graph
