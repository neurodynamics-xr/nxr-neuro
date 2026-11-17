// nxr-pipelines — the registered steps of the VORTEX CORES (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 5; docs/specs/2026-11-14-spatial-events.md, decisions 1 and 5):
//
//   core_histogram   accumulate   the distribution of |curl| over the burst frames (nxr `_core_threshold`: 4096 log bins, the range fixed from the run's first 3000 good frames), a STATISTIC (counts); the
//                                 percentile (98 by default) is derived at read
//   cores            accumulate   the connected above-threshold patches of the curl per frame and sense on the (glued) mesh (nxr `_core_patches`), the threshold read BY NAME from the stored histogram;
//                                 stored as POINT EVENTS (point_events.h): a spans Selection, one member per core on its frame, and the attributes as measurements over it
//
// Both read a field on the FRAME TILING (the `frames` step, with `within` the bursts): the good frames are the burst frames. The maths is `nxr::histogram` and `nxr::cores::corePatches` (compute/): pure functions of
// matrices; this file prepares their inputs (the mesh's edges, areas and positions, the stored threshold), folds the frames block by block and hands the result to the runner, which stores it.
// The step never reduces across time or space on its own beyond what a statistic over a partition is: the histogram's counts are additive over frames (a measurement over the frame partition).
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "nxr/cores.h"
#include "nxr/histogram.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/point_events.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

Json relative_chain(const Value& x);   // spatial_events_steps.cpp: what a relative power's threshold distribution and its patches share

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }

// the field the cores are cut from: a scalar on a surface, on the frame tiling (made by `frames`), linear in the recording (a complex one is read by its real part: Re(M z) = M Re(z) for a real linear map M)
void curl_input(const Value& x, const PlanContext& c, const std::string& who) {
  need(x.cls == "field" && x.time && x.chunked, who + ": x must be a field on … × time, inside the chunk loop");
  need(x.facts.value("axis", std::string("samples")) == "frames", who + ": x must be on the frame tiling (`frames`, with `within` the bursts): the cores are cut from the burst frames, one frame at a time");
  need(!x.facts.value("nonlinear", false), who + ": x was made by a non-linear step (" + x.facts.value("made_by", std::string("?")) + "): the cores are patches of the signed curl, not of an envelope");
  need(x.value_type == "scalar" && !x.manifolds.empty(), who + ": x is " + x.value_type + "; one number a vertex (the curl: `apply_kernel` with form curl_tan)");
  const auto m = c.bound.manifold(x.manifolds[0]);
  need(m && nxr::io::text(*m, "type") == "surface", who + ": x's first factor is not a surface (the cores are patches on the mesh)");
}
void bad_input(const Value& x, const Values& ins, const Json& p, const std::string& who) {
  need(p["bad_segments"] == "exclude" || p["bad_segments"] == "ignore", who + ": bad_segments " + p["bad_segments"].dump() + ": exclude or ignore");
  if (ins.count("exclude")) {
    const Value& ex = ins.at("exclude");
    need(ex.cls == "selection" && ex.manifolds[0] == *x.time, who + ": exclude must be a selection on x's time line (" + *x.time + "), not on " + ex.manifolds[0]);
  }
}
// what of x's chain the threshold and the cores must share: the band, the spatial level and shape, the form and the frame tiling
Json chain_of(const Value& x) {
  Json c = Json::object();
  c["band"] = x.facts.contains("band") ? x.facts["band"] : Json();
  c["form"] = x.facts.contains("form") ? x.facts["form"] : Json();
  c["spatial_band"] = x.facts.contains("spatial_band") ? Json{{"level", x.facts["spatial_band"]["level"]}, {"spatial_shape", x.facts["spatial_band"]["spatial_shape"]}} : Json();
  c["tiling"] = x.facts.contains("frames") ? x.facts["frames"]["tiling"] : Json();
  c["within"] = x.facts.contains("frames") ? x.facts["frames"]["within"] : Json();
  c["glued"] = x.facts.value("glued", false);
  return c;
}

// ── THE HISTOGRAM ──
struct HistState { int bins = 4096; std::int64_t range = 3000, block = 256; Json chain; double fps = 0, origin = 0; std::string reference = "nxr vortex._core_threshold"; };
struct HistAcc {
  std::int64_t frames = 0, buffered = 0, first_frame = -1;
  std::vector<float> buffer;                       // |value| of the first `range` good frames, a frame after another [V each]; freed when the bins are fixed
  bool fixed = false;
  nxr::histogram::LogBins bins;
  std::vector<std::int64_t> counts;
  std::int64_t negative = 0, nonfinite = 0;
};
void count_into(HistAcc& a, const float* values, std::size_t rows, std::size_t samples) {   // values [rows × samples] row-major
  const nxr::histogram::Counts c = nxr::histogram::histogramTiles(values, rows, samples, samples, 1, a.bins);
  for (int k = 0; k < c.bins; ++k) a.counts[static_cast<std::size_t>(k)] += c.counts[static_cast<std::size_t>(k)];
  a.negative += c.negative[0];
  a.nonfinite += c.nonfinite[0];
}
void fix_bins(HistAcc& a, std::int64_t V, int bins) {
  double mx = 0;
  for (const float v : a.buffer) if (std::isfinite(v)) mx = std::max(mx, static_cast<double>(v));
  try { a.bins = nxr::histogram::logBinsFor(mx, bins); } catch (const std::invalid_argument& e) { refuse(std::string("core_histogram: ") + e.what()); }
  a.counts.assign(static_cast<std::size_t>(bins), 0);
  a.fixed = true;
  if (a.buffered > 0) count_into(a, a.buffer.data(), static_cast<std::size_t>(a.buffered), static_cast<std::size_t>(V));   // the buffer's frames are rows of V values: the pool is over both
  a.buffer.clear();
  a.buffer.shrink_to_fit();
}

// ── THE CORES ──
struct CoreState {
  double threshold = 0, percentile = 98;
  std::int64_t block = 256, V = 0;
  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
  Eigen::VectorXd area;
  Eigen::MatrixXd positions;
  double fps = 0, origin = 0;
  Json record;
};
struct CoreAcc {
  std::int64_t frames = 0;
  std::vector<std::int64_t> frame;
  std::vector<std::int8_t> sense;
  std::vector<std::int32_t> centre_vertex;
  std::vector<double> centre, area, peak;
};
}  // namespace

void add_cores_steps(std::vector<StepSpec>& r) {
  // ───────────────────────── core_histogram ─────────────────────────
  {
    StepSpec s;
    s.name = "core_histogram"; s.kind = "accumulate";
    s.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "buffers the first range_frames good frames until the histogram's range is fixed, so what a block adds depends on the blocks before it", {});
    s.inputs = {"x", "exclude"}; s.optional = {"exclude"}; s.outputs = {"y"};
    s.parameters = Json{{"bins", 4096}, {"range_frames", 3000}, {"block_frames", 256}, {"bad_segments", "exclude"}};
    s.matches = "nxr vortex._core_threshold (4096 log bins, hi = 4 × the first chunk's maximum, lo = hi·1e-6; the first 3000 frames); compute nxr::histogram (logBinsFor, histogramTiles)";
    s.description = "the DISTRIBUTION of |x| over the good frames of a field on the frame tiling (the curl of the current at the burst frames), pooled over its vertices: counts in `bins` log bins (4096), a STATISTIC "
                    "(measure count, method histogram). The bins' range is fixed once from the run's first `range_frames` (3000, nxr's) good frames, counted from the run's first good frame: hi = 4 × their largest "
                    "|x|, lo = hi·1e-6, so the result does not depend on the chunk size; those frames are held until the range is known (MEMORY: range_frames × vertices float32). Counts are additive over frames "
                    "(a measurement over the frame partition); a percentile (the cores' threshold, 98 by default) is DERIVED at read from the stored counts and edges, exact to one bin (0.34 %). A complex x is "
                    "read by its real part (the real band's curl). Bad segments' frames are left out with `exclude`";
    s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
      const Value& x = ins.at("x");
      curl_input(x, c, "core_histogram");
      bad_input(x, ins, p, "core_histogram");
      need(p["bins"].is_number_integer() && p["bins"].get<int>() >= 2, "core_histogram: bins is a whole number, at least 2 (nxr: 4096)");
      need(p["range_frames"].is_number_integer() && p["range_frames"].get<std::int64_t>() >= 1, "core_histogram: range_frames is at least 1 (nxr: 3000)");
      need(p["block_frames"].is_number_integer() && p["block_frames"].get<std::int64_t>() >= 1, "core_histogram: block_frames is at least 1");
      Value y = x;
      y.row.reset();
      y.cls = "statistic";
      y.manifolds = {*x.time};
      y.chunked = false;
      y.complex = false;
      y.unit.reset();
      y.facts = Json{{"measure", "count"}, {"method", "histogram"}, {"chain", chain_of(x)}, {"frames", x.facts["frames"]}};
      return Values{{"y", y}};
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
      const Value& x = ins.at("x");
      HistState st;
      st.bins = p["bins"].get<int>(); st.range = p["range_frames"].get<std::int64_t>(); st.block = p["block_frames"].get<std::int64_t>();
      st.chain = chain_of(x); st.fps = x.rate; st.origin = x.origin;
      return std::any(std::make_shared<HistState>(std::move(st)));
    };
    s.start = [](const std::any&) { return std::any(HistAcc{}); };
    s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& place) {
      const HistState& st = *std::any_cast<const std::shared_ptr<HistState>&>(st_);
      HistAcc& a = std::any_cast<HistAcc&>(acc_);
      const Block& b = ins.at("x");
      const std::size_t V = static_cast<std::size_t>(b.rows);
      std::vector<std::int64_t> cols;
      for (std::int64_t c = 0; c < place.n; ++c) if (!place.good || place.good->empty() || (*place.good)[static_cast<std::size_t>(c)]) cols.push_back(c);
      if (!cols.empty() && a.first_frame < 0) a.first_frame = place.first + cols.front();
      std::size_t s0 = 0;
      while (s0 < cols.size()) {
        if (!a.fixed) {                                                  // the first `range` good frames are held until the range is known
          const std::size_t take = std::min<std::size_t>(cols.size() - s0, static_cast<std::size_t>(st.range - a.buffered));
          for (std::size_t j = 0; j < take; ++j) {
            const std::size_t col = static_cast<std::size_t>(b.left + cols[s0 + j]);
            for (std::size_t v = 0; v < V; ++v) a.buffer.push_back(std::abs(b.re[v * static_cast<std::size_t>(b.width) + col]));
          }
          a.buffered += static_cast<std::int64_t>(take); a.frames += static_cast<std::int64_t>(take); s0 += take;
          if (a.buffered >= st.range) fix_bins(a, static_cast<std::int64_t>(V), st.bins);
          continue;
        }
        const std::size_t k = std::min<std::size_t>(cols.size() - s0, static_cast<std::size_t>(st.block));
        std::vector<float> A(V * k);
        for (std::size_t j = 0; j < k; ++j) {
          const std::size_t col = static_cast<std::size_t>(b.left + cols[s0 + j]);
          for (std::size_t v = 0; v < V; ++v) A[v * k + j] = std::abs(b.re[v * static_cast<std::size_t>(b.width) + col]);
        }
        count_into(a, A.data(), V, k);
        a.frames += static_cast<std::int64_t>(k); s0 += k;
      }
    };
    s.finish = [](const std::any& st_, std::any& acc_) {
      const HistState& st = *std::any_cast<const std::shared_ptr<HistState>&>(st_);
      HistAcc& a = std::any_cast<HistAcc&>(acc_);
      if (a.frames == 0) refuse("core_histogram: no good frame in the run (the bursts hold none in the span, or every frame is a bad segment's): there is no distribution to read a threshold from");
      if (!a.fixed) {                                                    // fewer frames than the range: the range is theirs (the reference's first chunk is then the whole run)
        std::int64_t V = a.buffered > 0 ? static_cast<std::int64_t>(a.buffer.size()) / a.buffered : 0;
        fix_bins(a, V, st.bins);
      }
      Array y{1, st.bins, {}};
      y.data.assign(a.counts.begin(), a.counts.end());
      y.meta = Json{{"histogram", Json{{"lo", a.bins.lo}, {"hi", a.bins.hi}, {"bins", st.bins}, {"frames", a.frames}, {"range_frames", st.range}, {"first_frame", a.first_frame},
                                        {"negative", a.negative}, {"nonfinite", a.nonfinite}, {"chain", st.chain}, {"fps", st.fps}, {"origin", st.origin},
                                        {"edges", "upper edge of bin k: lo * (hi / lo)^(k / (bins - 1)) (np.geomspace(lo, hi, bins)[k]); bin 0 is [0, lo), values above hi are clipped into the last bin"},
                                        {"reference", st.reference}}}};
      return Arrays{{"y", std::move(y)}};
    };
    // ── relative_histogram: the SAME accumulator (the log-bin counts, the range fixed from the first frames), over the relative power of a spatial-events chain OUTSIDE the bursts ──
    StepSpec t = s;
    t.name = "relative_histogram";
    t.inputs = {"x", "exclude", "outside"}; t.optional = {"exclude"};
    t.matches = "the time burst detector's threshold from the envelope's own distribution; compute nxr::histogram (logBinsFor, histogramTiles), as core_histogram";
    t.description = "the DISTRIBUTION of the relative power (`relative`, then `at_level`) of a band over the good frames of the frame tiling OUTSIDE the named bursts (`outside`) and the bad segments (`exclude`), pooled over the "
                    "glued vertices: counts in `bins` log bins (4096), a STATISTIC (measure count, method histogram), the false-alarm distribution of the spatial events' threshold (a percentile of it, 99 by default, is "
                    "DERIVED at read by `patches`). The bins' range is fixed once from the run's first `range_frames` (3000) good frames, counted from the run's first good frame: hi = 4 × their largest value, lo = hi·1e-6, "
                    "so the result does not depend on the chunk size; those frames are held until the range is known (MEMORY: range_frames × vertices float32). Counts are additive over frames (a measurement over the "
                    "frame partition). The chain (band and its shape, spatial band, levels, reference level, tiling, baseline) is in its options: `patches` refuses a distribution of another chain. The frames are NOT "
                    "inside the bursts (`frames` without `within`): the bursts are left out here, as `baseline` leaves them out";
    t.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
      const Value& x = ins.at("x");
      need(x.cls == "field" && x.time && x.chunked && x.facts.value("axis", std::string("samples")) == "frames", "relative_histogram: x must be a field on the frame tiling (`frames`, without `within`: the bursts are left out by `outside`)");
      need(!x.complex && x.value_type == "scalar" && x.facts.value("made_by", std::string()) == "relative" && !x.facts.contains("levels"), "relative_histogram: x must be the relative power of ONE level (`relative`, then `at_level`); x was made by " + x.facts.value("made_by", std::string("no non-linear step")));
      need(x.facts["frames"].value("within", Json()).is_null(), "relative_histogram: the frames are inside " + x.facts["frames"]["within"].dump() + " (`frames` within): the distribution is of the frames OUTSIDE the bursts, so the frame tiling must not be restricted to them");
      const auto m = c.bound.manifold(x.manifolds[0]);
      need(m && nxr::io::text(*m, "type") == "surface", "relative_histogram: x's first factor is not a surface");
      need(p["bad_segments"] == "exclude", "relative_histogram: `outside` leaves a Selection out of the run: bad_segments must be exclude (the run's exclusion)");
      need(p["bins"].is_number_integer() && p["bins"].get<int>() >= 2, "relative_histogram: bins is a whole number, at least 2 (nxr: 4096)");
      need(p["range_frames"].is_number_integer() && p["range_frames"].get<std::int64_t>() >= 1, "relative_histogram: range_frames is at least 1 (nxr: 3000)");
      need(p["block_frames"].is_number_integer() && p["block_frames"].get<std::int64_t>() >= 1, "relative_histogram: block_frames is at least 1");
      for (const char* port : {"exclude", "outside"}) {
        if (!ins.count(port)) continue;
        const Value& ex = ins.at(port);
        need(ex.cls == "selection" && ex.row && nxr::io::text(*ex.row, "type") == "spans" && ex.manifolds[0] == *x.time, std::string("relative_histogram: ") + port + " must be a spans Selection on x's time line (" + *x.time + ")");
      }
      Json chain = relative_chain(x);
      chain["outside"] = nxr::io::text(*ins.at("outside").row, "name");
      Value y = x;
      y.row.reset();
      y.cls = "statistic";
      y.manifolds = {*x.time};
      y.chunked = false;
      y.complex = false;
      y.unit.reset();
      y.facts = Json{{"measure", "count"}, {"method", "histogram"}, {"chain", chain}, {"frames", x.facts["frames"]}};
      return Values{{"y", y}};
    };
    t.prepare = [](const Values& ins, const Json& p, ExecContext&) {
      const Value& x = ins.at("x");
      HistState st;
      st.bins = p["bins"].get<int>(); st.range = p["range_frames"].get<std::int64_t>(); st.block = p["block_frames"].get<std::int64_t>();
      st.chain = relative_chain(x); st.chain["outside"] = nxr::io::text(*ins.at("outside").row, "name"); st.fps = x.rate; st.origin = x.origin;
      st.reference = "the time burst detector's threshold from the envelope's own distribution; nxr::histogram";
      return std::any(std::make_shared<HistState>(std::move(st)));
    };
    r.push_back(std::move(s));
    r.push_back(std::move(t));
  }

  // ───────────────────────── cores ─────────────────────────
  {
    StepSpec s;
    s.name = "cores"; s.kind = "accumulate";
    s.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "finds the cores of each frame on its own, but appends them to one list in frame order, so the blocks must meet in time order", {});
    s.inputs = {"x", "threshold", "exclude"}; s.optional = {"exclude"};
    s.outputs = {"spans", "frame", "sense", "centre_vertex", "centre", "area", "peak"};
    s.parameters = Json{{"percentile", 98.0}, {"block_frames", 256}, {"bad_segments", "exclude"}};
    s.matches = "nxr vortex._core_patches (connected patches of one sense per frame; the |value|·area-weighted centroid); compute nxr::cores::corePatches; the threshold nxr::histogram::percentile of the stored histogram";
    s.description = "the VORTEX CORES of a field on the frame tiling (the hatted curl at the burst frames): per frame and per sense (+1 first) the connected vertices with sense·x ≥ the threshold, joined by the mesh's edges "
                    "(no merging of patches that do not touch). The threshold is the `percentile` (98, nxr's) of the stored histogram `threshold` (`core_histogram`, another run's statistic read BY NAME), derived "
                    "here from its counts and edges: a core is cut from a threshold it is given. The cores are POINT EVENTS (point_events.h): a spans Selection, one member per core on its frame (members may share "
                    "a frame; in the order frame, sense +1 first, lowest vertex), and per core as measurements over it: frame, sense, centre_vertex (the patch's vertex nearest its centre), centre (x y z, the "
                    "|value|·area-weighted centroid, the surface's unit), area (Σ the patch's vertex areas, m²) and peak (the largest sense·x). Frames are independent, so the result does not depend on the chunk "
                    "size. A complex x is read by its real part. Mesh: the vertices' edges from the faces of x's surface, the barycentric vertex areas, the stored positions";
    s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
      const Value &x = ins.at("x"), &t = ins.at("threshold");
      curl_input(x, c, "cores");
      bad_input(x, ins, p, "cores");
      need(p["percentile"].is_number() && p["percentile"].get<double>() > 0 && p["percentile"].get<double>() <= 100, "cores: percentile is in (0, 100] (nxr: 98)");
      need(p["block_frames"].is_number_integer() && p["block_frames"].get<std::int64_t>() >= 1, "cores: block_frames is at least 1");
      need(t.cls == "statistic" && t.row && nxr::io::text(*t.row, "method") == "histogram" && nxr::io::text(*t.row, "measure") == "count",
           "cores: threshold must be the stored histogram (`core_histogram`: a statistic of method histogram, measure count), read by name");
      const Json op = Json::parse(nxr::io::opt_text(*t.row, "options_json").value_or("{}"), nullptr, false);
      need(op.is_object() && op.contains("lo") && op.contains("hi") && op.contains("bins") && op.contains("chain"), "cores: " + nxr::io::text(*t.row, "name") + " records no log-bin range (lo, hi, bins) or chain: it is not a `core_histogram`'s");
      // the threshold's chain and the cores' chain are the same: the band, the spatial level and shape, the form, the frame tiling (a percentile of another chain's distribution is no threshold of this one)
      const Json mine = chain_of(x), theirs = op["chain"];
      for (const char* k : {"band", "form", "spatial_band", "tiling", "glued"})
        need(mine[k] == theirs.value(k, Json()), std::string("cores: the threshold ") + nxr::io::text(*t.row, "name") + " was made from another " + k + " (" + theirs.value(k, Json()).dump() + ") than the field's (" + mine[k].dump() + "): run `core_histogram` on the same chain");
      Values out;
      const Json record = Json{{"threshold_statistic", nxr::io::text(*t.row, "name")}, {"percentile", p["percentile"]}, {"chain", mine}};
      out["spans"] = point_events_value(x, Json{{"cores", record}}, "cores");
      out["frame"] = point_attribute_value(x, "frame", std::nullopt, "cores");
      out["sense"] = point_attribute_value(x, "sense", std::nullopt, "cores");
      out["centre_vertex"] = point_attribute_value(x, "centre_vertex", std::nullopt, "cores");
      out["centre"] = point_attribute_value(x, "centre", "m", "cores");
      out["area"] = point_attribute_value(x, "area", "m\xc2\xb2", "cores");
      out["peak"] = point_attribute_value(x, "peak", std::nullopt, "cores");
      return out;
    };
    s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
      const Value &x = ins.at("x"), &t = ins.at("threshold");
      CoreState st;
      st.percentile = p["percentile"].get<double>(); st.block = p["block_frames"].get<std::int64_t>(); st.fps = x.rate; st.origin = x.origin;
      // THE THRESHOLD, derived from the stored distribution (read by name)
      const nxr::io::StoredStatistic hs = c.bound.statistic(nxr::io::text(*t.row, "name"));
      const std::int64_t bins = hs.options["bins"].get<std::int64_t>();
      if (static_cast<std::int64_t>(hs.values.size()) != bins) refuse("cores: " + nxr::io::text(*t.row, "name") + ": " + std::to_string(hs.values.size()) + " counts for " + std::to_string(bins) + " bins");
      std::vector<std::int64_t> counts(hs.values.size());
      for (std::size_t i = 0; i < counts.size(); ++i) counts[i] = static_cast<std::int64_t>(std::llround(hs.values[i]));
      nxr::histogram::LogBins lb;
      lb.lo = hs.options["lo"].get<double>(); lb.hi = hs.options["hi"].get<double>(); lb.bins = static_cast<int>(bins);
      try { st.threshold = nxr::histogram::percentile(counts, lb, st.percentile); } catch (const std::invalid_argument& e) { refuse(std::string("cores: the threshold: ") + e.what()); }
      // THE MESH: x's surface (the glued one for the curl of the one cortex): edges, vertex areas, positions
      const auto man = c.bound.manifold(x.manifolds[0]);
      const nxr::pipelines::SurfaceInput S = c.bound.surface(nxr::io::text(*man, "name"), true);
      st.V = S.n_vertices();
      nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(S.n_vertices()); mesh.positions = S.vertices; mesh.faces = S.faces;
      const std::vector<double> va = nxr::ladder::vertexAreas(mesh);
      st.area = Eigen::Map<const Eigen::VectorXd>(va.data(), static_cast<Eigen::Index>(va.size()));
      st.positions = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(S.vertices.data(), S.n_vertices(), 3);
      std::vector<std::pair<std::int32_t, std::int32_t>> pr;
      for (std::int64_t f = 0; f < S.n_faces(); ++f)
        for (int k = 0; k < 3; ++k) {
          std::int32_t a = S.faces[static_cast<std::size_t>(3 * f + k)], b = S.faces[static_cast<std::size_t>(3 * f + (k + 1) % 3)];
          if (a > b) std::swap(a, b);
          pr.emplace_back(a, b);
        }
      std::sort(pr.begin(), pr.end());
      pr.erase(std::unique(pr.begin(), pr.end()), pr.end());
      st.pairs = std::move(pr);
      st.record = Json{{"threshold", st.threshold}, {"percentile", st.percentile}, {"threshold_statistic", Json{{"name", nxr::io::text(*t.row, "name")}, {"id", nxr::io::text(*t.row, "id")}}},
                       {"histogram", Json{{"lo", lb.lo}, {"hi", lb.hi}, {"bins", bins}, {"frames", hs.options.value("frames", Json())}}}, {"surface", Json{{"name", nxr::io::text(*man, "name")}, {"vertices", st.V}, {"edges", static_cast<std::int64_t>(st.pairs.size())}}},
                       {"positions_unit", "the surface's own (m)"}, {"reference", "nxr vortex._core_patches"},
                       {"warnings", Json::array({"a core is a patch of the curl above a percentile of the run's own burst frames: it describes where the field rotates, not that rotation is there (the null is the evidence)",
                                                 "the sense read from the curl map; the current's own sense is chance"})}};
      return std::any(std::make_shared<CoreState>(std::move(st)));
    };
    s.record = [](const std::any& st_) { return std::any_cast<const std::shared_ptr<CoreState>&>(st_)->record; };
    s.start = [](const std::any&) { return std::any(CoreAcc{}); };
    s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& place) {
      const CoreState& st = *std::any_cast<const std::shared_ptr<CoreState>&>(st_);
      CoreAcc& a = std::any_cast<CoreAcc&>(acc_);
      const Block& b = ins.at("x");
      if (b.rows != st.V) refuse("cores: x has " + std::to_string(b.rows) + " rows, the surface " + std::to_string(st.V) + " vertices");
      std::vector<std::int64_t> cols;
      for (std::int64_t c = 0; c < place.n; ++c) if (!place.good || place.good->empty() || (*place.good)[static_cast<std::size_t>(c)]) cols.push_back(c);
      for (std::size_t s0 = 0; s0 < cols.size(); s0 += static_cast<std::size_t>(st.block)) {
        const std::size_t k = std::min<std::size_t>(cols.size() - s0, static_cast<std::size_t>(st.block));
        Eigen::MatrixXd M(st.V, static_cast<Eigen::Index>(k));
        for (std::size_t j = 0; j < k; ++j) {
          const std::size_t col = static_cast<std::size_t>(b.left + cols[s0 + j]);
          for (std::int64_t v = 0; v < st.V; ++v) M(v, static_cast<Eigen::Index>(j)) = b.re[static_cast<std::size_t>(v) * static_cast<std::size_t>(b.width) + col];
        }
        nxr::cores::Cores C;
        try { C = nxr::cores::corePatches(M, st.pairs, st.area, st.positions, st.threshold, 0); } catch (const std::invalid_argument& e) { refuse(std::string("cores: ") + e.what()); }
        for (std::int64_t i = 0; i < C.size(); ++i) {
          a.frame.push_back(place.first + cols[s0 + static_cast<std::size_t>(C.frame[static_cast<std::size_t>(i)])]);
          a.sense.push_back(C.sense[static_cast<std::size_t>(i)]);
          a.centre_vertex.push_back(C.centre_vertex[static_cast<std::size_t>(i)]);
          for (int d = 0; d < 3; ++d) a.centre.push_back(C.centre(i, d));
          a.area.push_back(C.area[static_cast<std::size_t>(i)]);
          a.peak.push_back(C.peak[static_cast<std::size_t>(i)]);
        }
      }
      a.frames += static_cast<std::int64_t>(cols.size());
    };
    s.finish = [](const std::any& st_, std::any& acc_) {
      const CoreState& st = *std::any_cast<const std::shared_ptr<CoreState>&>(st_);
      const CoreAcc& a = std::any_cast<const CoreAcc&>(acc_);
      PointEvents e;
      e.fps = st.fps; e.origin = st.origin; e.frame = a.frame;
      e.record = st.record;
      e.record["frames_scanned"] = a.frames;
      e.record["cores"] = static_cast<std::int64_t>(a.frame.size());
      auto col = [&](const char* m, std::int64_t comps, std::vector<double> v) { e.attributes.push_back(PointAttribute{m, comps, std::move(v)}); };
      std::vector<double> frame(a.frame.begin(), a.frame.end()), sense(a.sense.begin(), a.sense.end()), cv(a.centre_vertex.begin(), a.centre_vertex.end());
      col("frame", 1, frame); col("sense", 1, sense); col("centre_vertex", 1, cv); col("centre", 3, a.centre); col("area", 1, a.area); col("peak", 1, a.peak);
      Arrays all = point_event_arrays(e);
      Arrays out;
      out["spans"] = std::move(all.at("spans"));
      for (const char* m : {"frame", "sense", "centre_vertex", "centre", "area", "peak"}) out[m] = std::move(all.at(m));
      return out;
    };
    r.push_back(std::move(s));
  }
}

}  // namespace nxr::pipelines::graph
