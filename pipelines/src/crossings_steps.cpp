// nxr-pipelines — the registered step of the CROSSINGS of tracks through tiles (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 7):
//
//   tile_crossings   after   a crossing is ONE TRACK'S STEPS inside ONE TILE of the surface ladder at `level` (one sense): the steps' unit directions carried to the tile's centre (the vector heat method's transport: the
//                            Levi-Civita form, nxr `Tiles.crossings`, compute nxr::crossings), added, and made a unit vector. The crossings of a tile are added and counted, per sense and per STAGE (the burst frames, and the onset /
//                            within / offset stages when given). Stored as ONE MEASUREMENT over the level's tiles, `tile_crossings`: [rows = stage × sense × members = the tiles + a `bridge` member × 4 = the vector (x y z)
//                            then the count]; rows 2i and 2i+1 are stage i's senses +1 and −1. A crossing inside the callosal bridge (no tile, no centre to carry to) is in the `bridge` member (the last), its unit
//                            vectors added in 3-D.
//
// The steps are the tracks' (the cores of a track in consecutive frames: each core after the first is a step from the previous core's centre), read from the cores and the labelling `track` that `tracks` made; a
// track not stored (shorter than `min_frames`) makes no crossing. Reads no recording data. DERIVED AT READ (never stored): `crossing_consistency` R = |S| / n and the unbiased `crossing_rho2` = (|S|² − n) / (n (n − 1)).
// What it cannot resolve: a crossing is the direction a core moved through a tile, between frames of the band's signal; a tile's R is large only where cores keep going the same way, and by itself it is no evidence of
// anything (the lag-free null is, piece 10: the unbiased rho2 and the data-to-null crossing ratio compare places with different counts).
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>

#include "nxr/chart.h"
#include "nxr/crossings.h"
#include "nxr/frames.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/spatial_band.h"

namespace nxr::pipelines::graph {

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
struct CrossState {
  nxr::crossings::Surface surface;          // over the GLUED vertices: positions, normals, e1, e2, rho of each (the bridge has none), tile
  nxr::frames::Mat3 e1, e2;                 // the native surface's frames, for the embedding at the centres
  std::vector<int> centres;                 // [tiles] the native centre vertex of each tile (-1 empty)
  int tiles = 0;
  std::vector<std::string> stages;
  std::vector<std::vector<std::pair<double, double>>> spans;   // per stage: its [start, stop) in s (empty for `burst`: every step)
  double fps = 0, origin = 0;
  Json record;
  bool as_statistic = false;
  int level = 0;
};
const char* const kStageNames[] = {"burst", "onset", "within", "offset"};
const char* const kWarnings[] = {
    "a crossing is the direction a core moved through a tile between frames of the band's signal: a tile's consistency R is large only where cores keep going the same way, and it is no evidence of travelling by itself (the lag-free null is, piece 10)",
    "compare places with different counts by the unbiased rho2, never R, and report the crossing counts (the data-to-null ratio); the unit directions are carried to the tile centre by the vector heat method, which on a folded cortex depends on the path (a fraction of a degree on a sphere)",
    "a crossing inside the callosal bridge has no tile centre: its unit vectors are added in 3-D in the `bridge` member"};
}  // namespace

void add_crossings_steps(std::vector<StepSpec>& r) {
  StepSpec s;
  s.name = "tile_crossings"; s.kind = "after";
  s.traits = node_traits(TimeBehaviour::windowed, Precision::float64, OutputPrecision::float64, "each step of a track is read from the previous core of the same track", "one core back along the same track (a step runs from the previous core's centre)", true);
  s.inputs = {"events", "frame", "sense", "centre", "track", "tiles", "onset", "within", "offset"};
  s.optional = {"onset", "within", "offset"};
  s.outputs = {"crossings"};
  s.parameters = Json{{"level", nullptr}, {"stages", Json::array({"burst"})}, {"as", "measurement"}};
  s.matches = "nxr vortex_cores.Tiles.crossings (a track's steps in one tile, carried to the tile centre by the vector heat method, added, made a unit vector; per tile and sense); compute nxr::crossings, nxr::ladder::leviCivita";
  s.description = "the CROSSINGS of the run's tracks through the tiles of the surface ladder at `level`: each core after the first of a stored track is a STEP from the previous core's centre; a crossing is one track's steps in "
                  "one tile and one sense, their unit directions (projected on the tangent plane at the nearest vertex) carried to the tile's centre along the surface (the Levi-Civita form, the vector heat method), added and made a "
                  "unit vector. Per tile, sense and STAGE (`burst`: every step; `onset`, `within`, `offset`: the steps whose frame starts inside that stage's spans, the inputs of those names): the crossings' unit vectors added "
                  "(a 3-D tangent vector at the tile's centre) and their number. ONE MEASUREMENT over the level's tiles, `tile_crossings`, [rows × members × 4]: rows 2i and 2i+1 are stage i's senses +1 and −1; members are the "
                  "level's tiles then a `bridge` member (the last: crossings whose nearest vertex is on the callosal bridge, their unit vectors added in 3-D); components the vector (crossing_sum_x/y/z) then the count "
                  "(crossings). R = |S|/n and the unbiased rho2 = (|S|² − n)/(n (n − 1)) are DERIVED at read, never stored. Reads the cores' list, not the chunks";
  s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
    const Value& ev = ins.at("events");
    need(ev.cls == "spans" && ev.facts.value("events_kind", std::string()) == "cores", "tile_crossings: events must be the spans of the `cores` step");
    for (const char* port : {"frame", "sense", "centre"}) {
      const Value& m = ins.at(port);
      need(m.cls == "measurement" && m.facts["over"].value("events_kind", std::string()) == "cores" && m.facts.value("measure", std::string()) == port, std::string("tile_crossings: ") + port + " must be the cores' own `" + port + "` measurement");
    }
    const Value& tr = ins.at("track");
    need(tr.cls == "measurement" && tr.facts.value("measure", std::string()) == "track" && tr.facts["over"].value("events_kind", std::string()) == "cores", "tile_crossings: track must be the labelling core → track (`tracks`' `track`)");
    const Value& t = ins.at("tiles");
    need(t.cls == "selection" && t.row && nxr::io::text(*t.row, "type") == "ladder", "tile_crossings: tiles must be a surface's tile ladder (`selection: ladder`)");
    need(p["level"].is_number_integer() && p["level"].get<int>() >= 0, "tile_crossings: level is the ladder level whose tiles the crossings are counted in (a whole number ≥ 0)");
    need(p["stages"].is_array() && !p["stages"].empty(), "tile_crossings: stages is a list: burst, and any of onset, within, offset");
    for (const auto& st : p["stages"]) {
      need(st.is_string() && std::find_if(std::begin(kStageNames), std::end(kStageNames), [&](const char* k) { return st == k; }) != std::end(kStageNames), "tile_crossings: stage " + st.dump() + ": burst, onset, within or offset");
      if (st != "burst") {
        need(ins.count(st.get<std::string>()) != 0, "tile_crossings: stage " + st.get<std::string>() + " needs its spans (the input `" + st.get<std::string>() + "`: the stage's Selection, e.g. global_bursts.burst_" + st.get<std::string>() + ".<recording>)");
        const Value& sp = ins.at(st.get<std::string>());
        need(sp.cls == "selection" && sp.row && nxr::io::text(*sp.row, "type") == "spans" && sp.manifolds[0] == *ev.time, "tile_crossings: the input " + st.get<std::string>() + " must be a spans Selection on the recording's time line");
      }
    }
    std::optional<nxr::io::Row> lv;
    for (const auto& m : c.bound.level_members(nxr::io::text(*t.row, "id")))
      if (nxr::io::integer(m, "member_ordinal") == static_cast<std::int64_t>(p["level"].get<int>())) { lv = m; break; }
    need(lv.has_value(), "tile_crossings: the ladder " + nxr::io::text(*t.row, "name") + " has no level " + std::to_string(p["level"].get<int>()));
    need(p["as"] == "measurement" || p["as"] == "statistic", "tile_crossings: as is measurement (the run's own crossings, over the level's tiles) or statistic (a null draw's: a node of its own, never the data's rows)");
    if (p["as"] == "statistic") {                      // a lag-free null's draw: the same array as a statistic beside the recording, so the data's measurement is never touched
      Value z = ev;
      z.row.reset(); z.cls = "statistic"; z.chunked = false; z.unit.reset(); z.manifolds.clear(); z.time.reset();
      z.facts = Json{{"measure", "crossing_sum"}, {"generic", true}, {"over", Json{{"selection_id", nxr::io::text(*lv, "id")}, {"selection", nxr::io::text(*lv, "name")}, {"level", p["level"]}}}, {"stages", p["stages"]}};
      return Values{{"crossings", z}};
    }
    Value y = ev;
    y.row.reset();
    y.cls = "measurement";
    y.chunked = false;
    y.unit.reset();
    y.facts = Json{{"measure", "tile_crossings"}, {"over", Json{{"selection_id", nxr::io::text(*lv, "id")}, {"selection", nxr::io::text(*lv, "name")}, {"level", p["level"]}}}, {"stages", p["stages"]}};
    return Values{{"crossings", y}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    CrossState st;
    const Value &ev = ins.at("events"), &t = ins.at("tiles");
    st.fps = ev.rate; st.origin = ev.origin;     // the frame tiling's rate and origin are the events' (tiles_of); re-read at run from the cores' own meta
    const int level = p["level"].get<int>();
    const auto man = c.bound.manifold(t.manifolds[0]);
    const SurfaceInput S = c.bound.surface(nxr::io::text(*man, "name"), true);
    const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
    const SurfaceLevel SL = c.bound.surface_level(S, level);
    st.tiles = static_cast<int>(SL.nodes);
    st.as_statistic = p["as"] == "statistic"; st.level = level;
    // the frames: the canonical ones when stored, else the signpost basis (said); the transport of each tile centre's frame to every vertex of its tile
    nxr::chart::ChartKit kit(S.vertices.data(), nV, S.faces.data(), nF);
    const StoredFrames sf = c.bound.stored_frames(S.name, nV);
    std::string frame = "canonical";                                     // the frame (G6: "frame", never "gauge")
    if (sf.canonical) { st.e1 = sf.e1; st.e2 = sf.e2; }
    else {
      const nxr::chart::Basis b = kit.basis();
      st.e1.resize(nV, 3); st.e2.resize(nV, 3);
      for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) { st.e1(i, k) = b.e1[static_cast<std::size_t>(3 * i + k)]; st.e2(i, k) = b.e2[static_cast<std::size_t>(3 * i + k)]; }
      frame = "gauge";
      if (c.say) c.say(Json{{"stage", "tile_crossings"}, {"message", "no canonical frames stored for " + S.name + " (`frames compute`): the signpost basis carries the crossings"}});
    }
    const Eigen::VectorXd gaugeAngle = nxr::frames::gaugeAngle(kit, st.e1);
    nxr::ladder::Mesh mesh; mesh.n = nV; mesh.positions = S.vertices; mesh.faces = S.faces;
    const std::vector<double> area = nxr::ladder::vertexAreas(mesh);
    std::vector<char> singular(static_cast<std::size_t>(nV), 0);
    for (int i = 0; i < nV; ++i) singular[static_cast<std::size_t>(i)] = !(std::isfinite(st.e1(i, 0)) && std::isfinite(st.e1(i, 1)) && std::isfinite(st.e1(i, 2)));
    st.centres = nxr::ladder::tileCentres(S.vertices.data(), SL.tile, st.tiles, area, singular);
    const nxr::ladder::LeviCivita lc = nxr::ladder::leviCivita(kit, st.e1, st.e2, gaugeAngle, SL.tile, st.tiles, st.centres);
    const nxr::frames::Mat3 nrm = nxr::frames::vertexNormals(S.vertices.data(), nV, S.faces.data(), nF);
    // the GLUED vertices (the cores live there): the kept ones in `support` order, then the bridge's
    const internal::GluedDomain dom = c.bound.glued_domain(S);
    if (!dom.found) refuse("tile_crossings: " + dom.why);
    const std::int64_t Vg = dom.surface.n_vertices(), kept = static_cast<std::int64_t>(dom.support.size());
    nxr::crossings::Surface& G = st.surface;
    G.positions = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(dom.surface.vertices.data(), Vg, 3);
    const nxr::frames::Mat3 gn = nxr::frames::vertexNormals(dom.surface.vertices.data(), static_cast<int>(Vg), dom.surface.faces.data(), static_cast<int>(dom.surface.n_faces()));
    G.normals = gn; G.e1 = Eigen::MatrixXd::Constant(Vg, 3, std::nan("")); G.e2 = G.e1;
    G.rho = Eigen::VectorXd::Constant(Vg, std::nan(""));
    G.node.assign(static_cast<std::size_t>(Vg), nxr::crossings::kBridge);
    for (std::int64_t i = 0; i < kept; ++i) {
      const std::int32_t v = dom.support[static_cast<std::size_t>(i)];
      G.normals.row(i) = nrm.row(v); G.e1.row(i) = st.e1.row(v); G.e2.row(i) = st.e2.row(v);
      G.rho[i] = lc.rho[v];
      G.node[static_cast<std::size_t>(i)] = SL.tile[static_cast<std::size_t>(v)];
    }
    // the stages' spans
    for (const auto& name : p["stages"]) {
      st.stages.push_back(name.get<std::string>());
      std::vector<std::pair<double, double>> sp;
      if (name != "burst")
        for (const auto& r : c.bound.spans(nxr::io::text(*ins.at(name.get<std::string>()).row, "id")))
          sp.emplace_back(r.first, r.second);
      st.spans.push_back(std::move(sp));
    }
    st.record = Json{{"level", level}, {"tiles", st.tiles}, {"surface", S.name}, {"glued", Json{{"vertices", Vg}, {"kept", kept}, {"bridge", Vg - kept}}}, {"frame", frame}, {"gauge", nxr::pipelines::gauge_block(kit, nV, S.id)}, {"stages", p["stages"]},
                     {"rows", "stage × sense: row 2i the stage i's sense +1, row 2i+1 sense -1"}, {"members", "the level's tiles, then `bridge`"}, {"components", "crossing_sum_x, crossing_sum_y, crossing_sum_z, crossings"},
                     {"reference", "nxr vortex_cores.Tiles.crossings"}, {"warnings", Json::array({kWarnings[0], kWarnings[1], kWarnings[2]})}};
    return std::any(std::make_shared<CrossState>(std::move(st)));
  };
  s.record = [](const std::any& st) { return std::any_cast<const std::shared_ptr<CrossState>&>(st)->record; };
  s.run = [](const std::any& st_, const Arrays& ins, ExecContext&) {
    const CrossState& st = *std::any_cast<const std::shared_ptr<CrossState>&>(st_);
    const Array &fr = ins.at("frame"), &sn = ins.at("sense"), &ce = ins.at("centre"), &tk = ins.at("track"), &ev = ins.at("events");
    const std::int64_t E = fr.cols;
    if (sn.cols != E || tk.cols != E || ce.cols != E || ce.comps != 3) throw std::invalid_argument("tile_crossings: the cores' attributes are of different lengths");
    const double fps = ev.meta["tiles_of"]["fps"].get<double>(), origin = ev.meta["tiles_of"]["origin"].get<double>();
    // the steps of the stored tracks: each core after the first of its track, from the previous core's centre
    nxr::crossings::Steps steps;
    std::vector<std::int64_t> sframe;
    std::map<std::int64_t, std::int64_t> last;
    std::vector<std::array<double, 3>> from, disp;
    for (std::int64_t i = 0; i < E; ++i) {
      const std::int64_t t = static_cast<std::int64_t>(std::llround(tk.data[static_cast<std::size_t>(i)]));
      if (t < 0) continue;
      const auto it = last.find(t);
      if (it != last.end()) {
        const std::int64_t q = it->second;
        std::array<double, 3> a, d;
        for (int k = 0; k < 3; ++k) { a[static_cast<std::size_t>(k)] = ce.data[static_cast<std::size_t>(3 * q + k)]; d[static_cast<std::size_t>(k)] = ce.data[static_cast<std::size_t>(3 * i + k)] - a[static_cast<std::size_t>(k)]; }
        from.push_back(a); disp.push_back(d);
        steps.sense.push_back(sn.data[static_cast<std::size_t>(i)] > 0 ? 1 : -1);
        steps.track.push_back(static_cast<std::int32_t>(t));
        sframe.push_back(static_cast<std::int64_t>(std::llround(fr.data[static_cast<std::size_t>(i)])));
      }
      last[t] = i;
    }
    const std::int64_t n = static_cast<std::int64_t>(from.size());
    steps.from.resize(n, 3); steps.disp.resize(n, 3);
    for (std::int64_t i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) { steps.from(i, k) = from[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]; steps.disp(i, k) = disp[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]; }
    const std::int64_t R = 2 * static_cast<std::int64_t>(st.stages.size()), T = st.tiles + 1, C = 4;
    Array out{R, T, std::vector<double>(static_cast<std::size_t>(R * T * C), 0.0)};
    out.comps = C;
    Json used = Json::array();
    for (std::size_t g = 0; g < st.stages.size(); ++g) {
      std::vector<std::uint8_t> use;
      if (!st.spans[g].empty() || st.stages[g] != "burst") {
        use.assign(static_cast<std::size_t>(n), 0);
        for (std::int64_t i = 0; i < n; ++i) {
          const double tj = origin + static_cast<double>(sframe[static_cast<std::size_t>(i)]) / fps;
          for (const auto& [a, b] : st.spans[g]) if (tj >= a && tj < b) { use[static_cast<std::size_t>(i)] = 1; break; }
        }
      }
      const nxr::crossings::Result res = nxr::crossings::crossings(steps, use, st.surface, st.tiles);
      for (int sx = 0; sx < 2; ++sx) {
        const std::int64_t row = 2 * static_cast<std::int64_t>(g) + sx;
        const Eigen::MatrixXd v = nxr::crossings::embedAtCentres(res.sum.col(sx), st.centres, st.e1, st.e2);
        for (int t = 0; t < st.tiles; ++t) {
          double* o = out.data.data() + ((row * T + t) * C);
          for (int k = 0; k < 3; ++k) o[k] = v(t, k);
          o[3] = res.n(t, sx);
        }
        double* b = out.data.data() + ((row * T + st.tiles) * C);
        for (int k = 0; k < 3; ++k) b[k] = res.bridge(sx, k);
        b[3] = res.bridge_n[sx];
      }
      used.push_back(Json{{"stage", st.stages[g]}, {"steps", res.steps_in}, {"crossed", res.steps_used}, {"no_direction", res.no_direction}, {"off_tiles", res.off_tiles}});
    }
    out.meta = Json{{"crossings", st.record}, {"steps_total", n}, {"stages_used", used}, {"rows", R}, {"members", T}, {"bridge_member", st.tiles}};
    if (st.as_statistic) {
      Json names = Json::array();
      for (int t = 0; t < st.tiles; ++t) names.push_back(Json{{"name", std::to_string(t)}});
      names.push_back(Json{{"name", "bridge"}});
      Json rows = Json::array();
      for (const auto& sg : st.stages) for (const char* sn : {"+1", "-1"}) rows.push_back(Json{{"name", sg + " " + sn}});
      out.meta["generic_statistic"] = Json{{"measure", "crossing_sum"}, {"method", "lagfree_null"}, {"shape", Json::array({R, T, C})},
                                           {"axes", Json::array({Json{{"name", "stage_sense"}, {"values", rows}, {"size", R}}, Json{{"name", "tile"}, {"values", names}, {"size", T}}, Json{{"name", "component"}, {"size", C}}})},
                                           {"components", Json::array({Json{{"name", "crossing_sum_x"}}, Json{{"name", "crossing_sum_y"}}, Json{{"name", "crossing_sum_z"}}, Json{{"name", "crossings"}}})},
                                           {"n_members", n},
                                           {"description", "the crossings of the tracks of a lag-free null draw through the tiles of level " + std::to_string(st.level) + ": per stage and sense, per tile then the bridge, the vector sum (x y z) and the count"},
                                           {"options", Json{{"level", st.level}, {"tiles", st.tiles}, {"stages_used", used}, {"crossings", st.record}}}};
    }
    return Arrays{{"crossings", std::move(out)}};
  };
  r.push_back(std::move(s));
}

}  // namespace nxr::pipelines::graph
