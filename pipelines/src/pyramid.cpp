// nxr-pipelines — what the measurement pyramid's verbs still share (pyramid.h): the leaves, the frame cells and their sums (`phasegradient`, `opticalflow`, `pyramid reduce`). The pyramid itself is `run pyramid`.
#include "nxr/pipelines/pyramid.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/selection_measurement.h"
#include "nxr/io/selection_store.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>

#include "nxr/bst/frequency.h"
#include "nxr/io/finder.h"
#include "nxr/io/hash.h"
#include "nxr/io/layout.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/ladder.h"
#include "nxr/parallel.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/frame_tiling.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/group.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/joint.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/reading.h"
#include "nxr/pipelines/tiles.h"
#include "nxr/pyramid.h"
#include "nxr/tower.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace lad = nxr::ladder;
namespace py = nxr::pyramid;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }
std::string g(double v) { std::ostringstream s; s << v; return s.str(); }

template <class T> std::vector<T> read_vec(const nxr::io::Store& store, const std::string& key, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(store, key);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

// the stored bank of `atlas bands` (nxr D158), or made: a spans Selection on the dataset's Frequency manifold, code = the band's index
std::string bands_selection_impl(SubjectWriter& sub, const py::Bands& bands, bool check_only = false, DatasetWriter* dset = nullptr) {   // check_only: refuse what would be refused, write nothing (the run checks BEFORE it computes)
  if (const auto have = nxr::io::selection::find_by_name(sub.db(), sub.id(), "atlas bands", std::string("spans"))) {
    const auto rows = nxr::io::selection::spans_of(sub.db(), have->id);
    bool same = rows.size() == bands.size();
    for (std::size_t k = 0; same && k < rows.size(); ++k)
      same = std::abs(rows[k].start - bands[k].first) < 1e-9 && std::abs(rows[k].stop - bands[k].second) < 1e-9 && rows[k].code && *rows[k].code == static_cast<std::int64_t>(k);
    if (!same) invalid(sub.name() + ": the stored `atlas bands` are another bank than the one asked for; a band's code is its index in the bank, so the bank of a subject is one (name the stored bands, or rebuild with --replace after removing them)");
    return have->id;
  }
  auto f = nxr::bst::find_frequency(sub.db());
  const bool is_template = text(sub.row(), "kind") == "template";
  // the Frequency manifold (with its standard bands) lives in the dataset's default subject; made there when absent, as `atlas default-subject` (nxr) makes it: additive, nothing existing changes
  const std::string no_default = sub.name() + ": the bands are spans on the Frequency manifold of the dataset's default subject, which has none and the dataset has no default subject to make it in (create the default subject first)";
  if (check_only && (f || is_template)) return "";
  if (check_only) { if (!dset || !dset->template_()) throw Error(code::no_such_row, no_default); return ""; }
  if (!f && is_template) { nxr::bst::ensure_frequency(sub); f = nxr::bst::find_frequency(sub.db()); }
  if (!f && dset) {
    if (const auto def = dset->template_()) {
      SubjectWriter dsub = SubjectWriter::open(*dset, text(*def, "name"));
      dsub.compose([&] { nxr::bst::ensure_frequency(dsub); });
      f = nxr::bst::find_frequency(sub.db());
    }
  }
  if (!f) throw Error(code::no_such_row, no_default);
  std::string id;
  sub.compose([&] {
    // a dictionary of the bank already here (a group package carries the members' copies as anchor rows) is the bank's when it is this bank
    std::string existing_dictionary;
    if (const auto have = nxr::io::selection::find_lookup(sub.db(), "dataset", "atlas bands")) {
      auto entries = nxr::io::selection::lookup_entries(sub.db(), *have);
      std::sort(entries.begin(), entries.end(), [](const auto& x, const auto& y) { return x.code < y.code; });
      bool same = entries.size() == bands.size();
      for (std::size_t k = 0; same && k < entries.size(); ++k) {
        const Json& a = entries[k].attributes_json;
        same = entries[k].code == static_cast<std::int64_t>(k) && a.is_object() && std::abs(a.value("low_hz", -1.0) - bands[k].first) < 1e-9 && std::abs(a.value("high_hz", -1.0) - bands[k].second) < 1e-9;
      }
      if (same) existing_dictionary = *have;   // (another bank: left to the dictionary's own rules, as before)
    }
    DictionarySpec ds;
    ds.name = "atlas bands"; ds.scope = "dataset";
    ds.description = "the measurement pyramid's band bank: code = the band's index";
    ds.entries = std::vector<DictionaryEntry>();
    for (std::size_t k = 0; k < bands.size(); ++k) {
      DictionaryEntry e;
      e.code = static_cast<std::int64_t>(k); e.ordinal = static_cast<std::int64_t>(k);
      e.name = g(bands[k].first) + "\xe2\x80\x93" + g(bands[k].second) + " Hz";
      e.cols.set("measurement", std::sqrt(bands[k].first * bands[k].second)).set("measurement_unit", "Hz").json("attributes_json", Json{{"low_hz", bands[k].first}, {"high_hz", bands[k].second}});
      ds.entries->push_back(e);
    }
    const std::string dictionary_id = existing_dictionary.empty() ? sub.dictionary(ds).id : existing_dictionary;
    nxr::io::selection::Description bank;
    bank.type = nxr::io::selection::Type::set;
    bank.unit = "Hz";
    for (std::size_t k = 0; k < bands.size(); ++k) {
      nxr::io::selection::Description band;
      band.type = nxr::io::selection::Type::span;
      band.code = static_cast<std::int64_t>(k);
      band.extents.push_back(nxr::io::selection::Extent{0, bands[k].first, bands[k].second});
      bank.members.push_back(std::move(band));
    }
    nxr::io::selection::Placement at;
    at.name = "atlas bands"; at.manifold_id = text(f->manifold, "id"); at.cell = "1"; at.dictionary_id = dictionary_id;
    at.description = "the pyramid's frequency bands: a span [lo, hi) of the Frequency manifold each, code = the band's index";
    id = nxr::io::selection::save(sub, bank, at).id;
  });
  return id;
}
}  // namespace

std::string pyramid_bands_selection(SubjectWriter& sub, const std::vector<std::pair<double, double>>& bands) { return bands_selection_impl(sub, bands); }

PyramidLeaves pyramid_leaves(DatasetWriter& ds, SubjectWriter& sub, const SurfaceInput& S, std::optional<int> depth, const std::string& who, const std::function<void(const Json&)>& say, const std::string& tree) {
  const std::string surface_name = S.name;
  if (tree != "subject" && tree != "group") invalid(who + ": --tree is subject or group, not " + tree);
  if (tree == "group" && text(sub.row(), "kind") != "template") return group_leaves(ds, sub, S, depth, who, say);
  const std::int64_t N = S.n_vertices();
  auto ladder_of = [&]() { return nxr::io::selection::ladder_row(sub.db(), sub.id(), surface_name + " tiles", S.id); };
  if (!ladder_of()) { TilesComputeOptions to; to.surface = surface_name; if (say) say(Json{{"stage", "tiles"}, {"surface", surface_name}, {"made", true}}); tiles_compute(ds, sub, to); }
  const auto ladder = ladder_of();
  const Json lp = Json::parse(text(*ladder, "params_json"), nullptr, false);
  if (!lp.is_object() || !lp.contains("depth") || !lp.contains("finest")) invalid(surface_name + " tiles: its record has no depth (a ladder made by another producer)");
  PyramidLeaves out;
  const int ladder_depth = lp["depth"].get<int>();
  out.finest = lp["finest"].get<int>();
  out.depth = depth ? *depth : ladder_depth;
  if (out.depth < 0 || out.depth > ladder_depth) invalid(who + ": --depth " + std::to_string(out.depth) + " is outside the surface's ladder (levels 0 \xe2\x80\xa6 " + std::to_string(ladder_depth) + ")");
  out.name = surface_name + " tiles L" + std::to_string(out.depth);
  out.part = nxr::io::selection::member_row(sub.db(), text(*ladder, "id"), out.name);
  if (!out.part) invalid(sub.name() + ": the ladder of " + surface_name + " has no level " + std::to_string(out.depth));
  out.record = Json::parse(text(*out.part, "params_json"), nullptr, false);
  if (!out.record.is_object() || !out.record.contains("node") || !out.record.contains("areas")) invalid(out.name + ": its record lacks the node of its arrays (a ladder made by another producer)");
  out.tree = lp.value("tree", std::string("subject"));
  out.node = out.record["node"].get<std::string>();
  out.leaves = integer(*out.part, "n_members");
  out.codes = read_vec<std::int32_t>(sub.store(), text(*out.part, "array_path"));
  out.leaf_area = read_vec<double>(sub.store(), out.record["areas"].get<std::string>());
  if (static_cast<std::int64_t>(out.codes.size()) != N || static_cast<std::int64_t>(out.leaf_area.size()) != out.leaves) invalid(out.name + ": its codes or areas do not fit the surface");
  return out;
}

void pyramid_leaves_record(SubjectWriter& sub, PyramidLeaves& leaves, const Json& gauge) {
  leaves.record["atlas"] = Json{{"tree", leaves.tree}, {"level", leaves.depth}, {"finest", leaves.finest}, {"frame", gauge}, {"areas", leaves.record["areas"]}, {"node", leaves.node}};
  nxr::io::selection::set_params(sub.db(), text(*leaves.part, "id"), leaves.record);
}

FrameCells frame_cells(DatasetWriter& ds, SubjectWriter& sub, const std::string& recording, int frame_level, std::optional<double> start_s, const std::string& who) {
  if (frame_level > 0) invalid(who + ": the frame level is 0 (the day) or finer (negative)");
  const auto recrow = find_field(sub.db(), sub.id(), FieldQuery{recording, "recording"});
  if (!recrow) missing(sub.name() + ": no recording '" + recording + "'");
  const finder::Value rv = finder::value_of(sub.db(), "field", *recrow);
  if (!rv.time || !(rv.rate > 0)) invalid(recording + ": it has no time line with a rate");
  FrameCells fc;
  fc.time_id = *rv.time;
  const std::int64_t n_samples = integer(*sub.db().read("manifold", fc.time_id), "n_vertices");
  try { fc.ft = nxr::tower::makeFrameTiling(rv.rate, n_samples, frame_level, start_s); } catch (const std::invalid_argument& e) { invalid(who + ": " + e.what()); }
  fc.n_frames = fc.ft.count;
  fc.bits = joint_bits(fc.n_frames);
  FrameTilingOptions fo;
  fo.recording = recording; fo.frame_level = frame_level; fo.start_s = start_s;
  frame_tiling_compute(ds, sub, fo);
  const auto frow = nxr::io::selection::named_row(sub.db(), sub.id(), frame_tiling_name(recording, frame_level), std::nullopt, fc.time_id);
  if (!frow) missing(who + ": the frame tiling was not stored");
  fc.frames_id = text(*frow, "id");
  const Json fparams = Json::parse(opt_text(*frow, "params_json").value_or("{}"), nullptr, false);
  if (!fparams.is_object() || !fparams.contains("atlas") || fparams["atlas"].value("frame_code0", std::int64_t(-1)) != fc.ft.code0 || integer(*frow, "n_members") != fc.n_frames)
    invalid(frame_tiling_name(recording, frame_level) + ": a tiling already stored is anchored differently (--start-s); replace it with `run frametiling --replace`");
  return fc;
}

std::string frame_sums_write(SubjectWriter& sub, const PyramidLeaves& PL, const FrameCells& fc, const std::string& joint_name, const std::string& description, std::int64_t frame0, std::int64_t frames_span,
                             const std::vector<FrameSumsSeries>& series, const std::string& computed_by, const std::optional<std::string>& acquisition_id) {
  const std::int64_t L = PL.leaves, NF = fc.n_frames;
  const std::string surface_id = text(*PL.part, "manifold_id");
  const std::string product = sub.product({surface_id, fc.time_id}, acquisition_id);
  nxr::io::selection::Placement js;
  js.name = joint_name; js.manifold_id = product; js.cell = "0,1"; js.n_members = L * NF; js.description = description;
  js.params = Json{{"joint", Json{{"space", text(*PL.part, "id")}, {"time", fc.frames_id}, {"bits", fc.bits}, {"tiles", L}, {"frames", NF}, {"layout", Json::array({"band", "tile", "frame"})}}},
                   {"atlas", Json{{"tree", PL.tree}, {"frame", PL.record["atlas"].value("frame", Json(nullptr))}, {"node", PL.node}}}};
  if (acquisition_id) js.acquisition_id = *acquisition_id;
  for (const FrameSumsSeries& s : series) {
    if (static_cast<std::int64_t>(s.counts.size()) != frames_span) invalid("frame sums: the counts do not fit the span's frames");
    // the per-frame count of the series' steps, on the frame tiling, as the build stores its samples (a measurement row `count` of the Field)
    std::vector<double> counts(static_cast<std::size_t>(NF), 0.0);
    for (std::int64_t f = 0; f < frames_span; ++f) counts[static_cast<std::size_t>(frame0 + f)] = s.counts[static_cast<std::size_t>(f)];
    const std::string cpath = PL.node + "/" + s.group + "/count";
    sub.write_array(cpath, view(counts, {NF}));
    Cols cnt;
    cnt.set("measure", "count").set("of_field_id", s.field_id).set("array_path", cpath).set("computed_by", computed_by);
    nxr::io::selection::save_measurement(sub, fc.frames_id, cnt);
    for (const FrameSumsRow& r : s.rows) {
      if (static_cast<std::int64_t>(r.v.size()) != L * frames_span) invalid("frame sums: " + r.measure + " does not fit the leaves x the span's frames");
      std::vector<double> out(static_cast<std::size_t>(L * NF), 0.0);
      for (std::int64_t l = 0; l < L; ++l)
        for (std::int64_t f = 0; f < frames_span; ++f) out[static_cast<std::size_t>(l * NF + frame0 + f)] = r.v[static_cast<std::size_t>(l * frames_span + f)];
      const std::string path = PL.node + "/" + s.group + "/" + r.rel;
      sub.write_array(path, view(out, {1, L, NF}));
      Cols c;
      c.set("measure", r.measure).set("of_field_id", s.field_id).set("array_path", path).set("computed_by", computed_by);
      if (!r.unit.empty()) c.set("unit", r.unit);
      js.measurements.push_back(c);
    }
  }
  return nxr::io::selection::save(sub, nxr::io::selection::Description{}, js).id;
}

SpatialBandsResult pyramid_spatial_bands(DatasetWriter& ds, SubjectWriter& sub, const SpatialBandsOptions& opt) {
  const SurfaceInput S = read_surface(sub, opt.surface, true);
  const std::string surface_name = S.name;
  auto ladder_of = [&]() { return nxr::io::selection::ladder_row(sub.db(), sub.id(), surface_name + " tiles", S.id); };
  if (!ladder_of()) { TilesComputeOptions to; to.surface = surface_name; tiles_compute(ds, sub, to); }
  const auto ladder = ladder_of();
  if (!ladder) missing(sub.name() + ": the ladder of " + surface_name + " was not stored");
  std::vector<SpatialBandEdge> edges;
  Json levels = Json::array();
  for (const Row& m : nxr::io::selection::member_rows(sub.db(), text(*ladder, "id"))) {
    const Json r = Json::parse(opt_text(m, "params_json").value_or("{}"), nullptr, false);
    if (!r.is_object() || !r.contains("level") || !r.contains("lambda_min") || !r.contains("lambda_max")) invalid(surface_name + " tiles: a level's record has no lambda range (a ladder made by another producer)");
    SpatialBandEdge e;
    e.level = r["level"].get<std::int64_t>(); e.lo = r["lambda_min"].get<double>(); e.hi = r["lambda_max"].get<double>();
    edges.push_back(e);
    levels.push_back(Json{{"level", e.level}, {"lambda_min", e.lo}, {"lambda_max", e.hi}, {"wavelength_m", r.value("wavelength_m", std::nan(""))}});
  }
  // the Eigenvalue line lives in the dataset's default subject; made there when absent (additive, as the Frequency manifold is made), so a verb that needs a spatial band never asks for a separate step
  if (!find_eigenvalue(sub.db()) && text(sub.row(), "kind") != "template") {
    if (const auto def = ds.template_()) {
      SubjectWriter dsub = SubjectWriter::open(ds, text(*def, "name"));
      dsub.compose([&] { ensure_eigenvalue(dsub); });
    }
  }
  const std::string id = spatial_bands_selection(sub, surface_name, edges);
  SpatialBandsResult res;
  res.selection_id = id;
  res.edges = edges;
  {
    const Json gp = Json::parse(text(*ladder, "params_json"), nullptr, false);
    if (!gp.is_object() || !gp.contains("area_m2") || !gp.contains("pieces")) invalid(surface_name + " tiles: its record lacks the surface's area (a ladder made by another producer)");
    res.piece_area = gp["area_m2"].get<double>() / std::max(1, gp["pieces"].get<int>());
  }
  res.info = Json{{"surface", surface_name}, {"selection", spatial_bands_name(surface_name)}, {"selection_id", id}, {"levels", edges.size()}, {"bands", levels}, {"line", kEigenvalueName}, {"unit", "1/m^2"}};
  return res;
}

}  // namespace nxr::pipelines
