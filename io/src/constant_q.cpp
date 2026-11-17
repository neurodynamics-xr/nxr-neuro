// nxr::io::constantq — the constant-Q tiling of a time Line (constant_q.h): the Studio's backend/src/db/constant-q.ts, line for line (JavaScript's
// Math.round rounds halves up; its Number formatting is the io's js_json).
#include "nxr/io/constant_q.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

#include "nxr/io/errors.h"
#include "nxr/selection/partition.h"
#include "nxr/io/writer.h"

namespace nxr::io::constantq {

namespace {
double jsRound(double x) { return std::floor(x + 0.5); }                 // Math.round: halves toward +∞
double pow2ceil(double x) { return std::pow(2.0, std::ceil(std::log2(x))); }
std::string fixed(double x, int d) { char b[64]; std::snprintf(b, sizeof b, "%.*f", d, x); return b; }
std::string fmtHz(double x) { return x >= 10 ? fixed(x, 1) : x >= 1 ? fixed(x, 2) : fixed(x, 3); }
std::string i2s(std::int64_t x) { return std::to_string(x); }
const char* methodName(Method m) { return m == Method::Hertz ? "hertz" : "cycles"; }
}  // namespace

std::pair<double, double> anchorsFor(Method method, double sfreq, double target) {
  const double nyquist = sfreq / 2;
  if (method == Method::Cycles) return {target, nyquist};
  return {std::pow(2.0, jsRound(std::log2(target * sfreq))) / sfreq, std::min(std::pow(2.0, std::floor(std::log2(nyquist))), kCapHz)};
}

std::int64_t windowFor(double sfreq, double lo, std::optional<double> cycles) {
  if (!(sfreq > 0) || !(lo > 0)) throw Error(code::invalid_argument, "windowFor: " + std::to_string(sfreq) + " Hz at " + std::to_string(lo) + " Hz");
  const double c = cycles ? *cycles : anchorsFor(Method::Hertz, sfreq).first;
  if (!(c > 0)) throw Error(code::invalid_argument, "windowFor: cycles must be > 0");
  return static_cast<std::int64_t>(std::llround(pow2ceil(c * sfreq / lo)));
}

int levelOf(double seconds, double sfreq) {
  if (!(seconds > 0) || !(sfreq > 0)) throw Error(code::invalid_argument, "levelOf: " + std::to_string(seconds) + " s at " + std::to_string(sfreq) + " Hz");
  return std::max(0, static_cast<int>(std::lround(std::log2(seconds * sfreq))));
}

int levelOfWidth(std::int64_t samples) {
  if (samples < 1 || (samples & (samples - 1)) != 0) return -1;
  int k = 0;
  while ((std::int64_t(1) << k) < samples) ++k;
  return k;
}

std::int64_t strideFor(double sfreq, double hz, int min) {
  if (!(sfreq > 0) || !(hz > 0)) throw Error(code::invalid_argument, "strideFor: " + std::to_string(sfreq) + " Hz at " + std::to_string(hz) + " Hz");
  return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::pow(2.0, std::floor(std::log2(sfreq / (min * hz))))));
}

std::vector<Level> levelsOf(const std::vector<Band>& bands, std::int64_t samples, double sfreq) {
  std::map<std::int64_t, std::vector<const Band*>> byWindow;
  for (const Band& b : bands) byWindow[b.window_samples].push_back(&b);
  const std::int64_t finest = byWindow.empty() ? 1 : byWindow.begin()->first;
  std::vector<Level> out;
  int level = 0;
  for (const auto& [window, share] : byWindow) {
    Level l;
    l.level = level++;
    l.window_samples = window;
    l.window_s = share[0]->window_s;
    l.hop_samples = window;
    l.count = (samples + window - 1) / window;
    l.finest_tiles = window / finest;
    l.resolves_lo = share[0]->lo_hz;
    l.resolves_hi = share[0]->hi_hz;
    for (const Band* b : share) { l.resolves_lo = std::min(l.resolves_lo, b->lo_hz); l.resolves_hi = std::max(l.resolves_hi, b->hi_hz); }
    l.bands = static_cast<int>(share.size());
    l.stride = strideFor(sfreq, l.resolves_lo);
    l.rate_hz = sfreq / static_cast<double>(l.stride);
    l.points = window / l.stride;
    l.spc_lo = l.rate_hz / l.resolves_lo;
    l.spc_hi = l.rate_hz / l.resolves_hi;
    l.nyquist_bound = l.stride == 1 && l.rate_hz / l.resolves_lo < kMultirateMin;
    out.push_back(l);
  }
  return out;
}

Tiling constantQTiling(double sfreq, std::int64_t samples, double origin, const Params& p) {
  if (!(sfreq > 0) || samples < 1) throw Error(code::invalid_argument, "constantQ: a line of " + i2s(samples) + " samples at " + std::to_string(sfreq) + " Hz");
  Tiling t;
  t.sfreq = sfreq;
  t.samples = samples;
  t.origin = origin;
  t.voices = p.voices.value_or(6);
  t.method = p.method.value_or(Method::Hertz);
  const auto [aCycles, aFMax] = anchorsFor(t.method, sfreq);
  t.cycles = p.cycles.value_or(aCycles);
  if (t.voices < 1) throw Error(code::invalid_argument, "constantQ: voices must be a whole number >= 1, got " + std::to_string(t.voices));
  if (!(t.cycles > 0)) throw Error(code::invalid_argument, "constantQ: cycles must be > 0");
  const double nyquist = sfreq / 2;
  const double fMax = std::min(p.fMax.value_or(aFMax), nyquist);
  if (!(fMax > 0)) throw Error(code::invalid_argument, "constantQ: fMax must be > 0");
  const double fMin = std::max(p.fMin.value_or(t.method == Method::Hertz ? kFloorHz : 0.0), 0.0);
  // THE BANDS: the selection module's time partition (one design of constant-Q, nxr/selection/partition.h); this adds what the stored rows record of each
  const double w = 1.0 / t.voices, q = 1.0 / (std::pow(2.0, w / 2) - std::pow(2.0, -w / 2)), dt = 1.0 / sfreq;
  nxr::selection::ConstantQ cq;
  cq.method = methodName(t.method); cq.voices = t.voices; cq.cycles = t.cycles; cq.f_max = fMax; cq.f_min = fMin;
  nxr::selection::Partition part;
  try { part = nxr::selection::partition(nxr::selection::Line{samples, sfreq}, cq); } catch (const std::invalid_argument& e) { throw Error(code::invalid_argument, std::string("constantQ: ") + e.what()); }
  for (std::size_t k = 0; k < part.bands.size(); ++k) {
    const auto& pb = part.bands[k];
    Band b;
    b.ordinal = static_cast<int>(k); b.octave = b.ordinal / t.voices; b.voice = b.ordinal % t.voices;
    b.lo_hz = pb.lo; b.hi_hz = pb.hi; b.centre_hz = std::sqrt(pb.lo * pb.hi); b.q = q;
    b.window_samples = std::int64_t(1) << pb.level; b.window_s = static_cast<double>(b.window_samples) * dt; b.hop_samples = b.window_samples;
    b.count = (samples + b.window_samples - 1) / b.window_samples;
    t.bands.push_back(b);
  }
  t.f_max = fMax;
  t.f_min = t.bands.empty() ? fMax : t.bands.back().lo_hz;
  t.levels = levelsOf(t.bands, samples, sfreq);
  return t;
}

std::pair<double, double> tileOf(const Closed& t, std::int64_t m, double end) {
  const double start = t.origin + static_cast<double>(m) * t.hop;
  return {start, std::min(start + t.width, end)};
}

std::optional<std::int64_t> tileIndexAt(const Closed& t, double value, double end) {
  if (value < t.origin || value >= end) return std::nullopt;
  const std::int64_t m = static_cast<std::int64_t>(std::floor((value - t.origin) / t.hop));
  return m < t.count ? std::optional<std::int64_t>(m) : std::nullopt;
}

std::int64_t tilesWithin(std::int64_t coarser, std::int64_t finer) {
  if (!(finer > 0) || coarser % finer != 0)
    throw Error(code::invalid_argument, "tilesWithin: " + i2s(coarser) + " is not a whole multiple of " + i2s(finer));
  return coarser / finer;
}

Json levelInfo(const Tiling& t, const Level& l) {
  const std::int64_t finest = t.levels.empty() ? 1 : t.levels[0].window_samples;
  return Json{{"level", l.level}, {"window_samples", l.window_samples}, {"window_s", l.window_s}, {"hop_samples", l.hop_samples}, {"count", l.count},
              {"finest_tiles", l.finest_tiles}, {"resolves_hz", Json{{"lo", l.resolves_lo}, {"hi", l.resolves_hi}}}, {"bands", l.bands}, {"stride", l.stride},
              {"rate_hz", l.rate_hz}, {"points", l.points}, {"samples_per_cycle", Json{{"lo", l.spc_lo}, {"hi", l.spc_hi}}}, {"nyquist_bound", l.nyquist_bound},
              {"duration_s", l.window_s}, {"finest_window_samples", finest}};
}

Json setParams(const Tiling& t) {
  const double end = t.origin + static_cast<double>(t.samples) / t.sfreq;
  return Json{{"sfreq", t.sfreq}, {"samples", t.samples}, {"origin", t.origin}, {"method", methodName(t.method)}, {"voices", t.voices}, {"cycles", t.cycles},
              {"f_max", t.f_max}, {"f_min", t.f_min}, {"levels", t.levels.size()}, {"bands", t.bands.size()}, {"end", end}};
}

Json memberParams(const Tiling& t, const Level& l) {
  const double end = t.origin + static_cast<double>(t.samples) / t.sfreq;
  return Json{{"tiling", Json{{"origin", t.origin}, {"width", static_cast<double>(l.window_samples) / t.sfreq}, {"hop", static_cast<double>(l.hop_samples) / t.sfreq},
                              {"count", l.count}}},
              {"end", end}, {"level", l.level}, {"window_samples", l.window_samples}, {"hop_samples", l.hop_samples}, {"finest_tiles", l.finest_tiles},
              {"resolves_hz", Json{{"lo", l.resolves_lo}, {"hi", l.resolves_hi}}}};
}

Written write(SubjectWriter& sub, const std::string& manifold_id, const Params& p, const std::optional<std::string>& by) {
  // the time Line: a closed-form `line` (timeLineOf)
  const auto m = sub.db().read("manifold", manifold_id);
  if (!m) throw Error(code::no_such_row, "constantQ: manifold " + manifold_id + " has no row");
  if (text(*m, "type") != "line" || is_null(*m, "geometry_id")) throw Error(code::invalid_argument, "constantQ: '" + text(*m, "name") + "' is not a closed-form line");
  const auto d = sub.db().one("SELECT n_vertices, spacing, origin FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal LIMIT 1", {V(text(*m, "geometry_id"))});
  if (!d || is_null(*d, "spacing")) throw Error(code::invalid_argument, "constantQ: '" + text(*m, "name") + "' has no closed-form dimension");
  const double sfreq = 1.0 / std::stod(text(*d, "spacing"));
  const std::int64_t samples = integer(*d, "n_vertices");
  const double origin = is_null(*d, "origin") ? 0.0 : std::stod(text(*d, "origin"));
  const std::string name = text(*m, "name") + "_constantQ";
  Written w;
  w.name = name;
  w.tiling = constantQTiling(sfreq, samples, origin, p);
  // the tiling already on the line (constantQOf: a ladder on it whose params name `voices`)
  for (const Row& s : sub.db().all("SELECT * FROM selection WHERE subject_id = ? AND type = 'ladder' AND manifold_id = ? AND status = 'complete'", {V(sub.id()), V(manifold_id)})) {
    if (opt_text(s, "params_json").value_or("").find("\"voices\"") == std::string::npos) continue;
    w.existed = true;
    w.set_id = text(s, "id");
    for (const Row& mm : sub.db().all("SELECT id FROM selection WHERE member_of_id = ? ORDER BY member_ordinal", {V(w.set_id)})) w.member_ids.push_back(text(mm, "id"));
    return w;
  }
  const Tiling& t = w.tiling;
  const auto acquisition = opt_text(*m, "acquisition_id");
  const auto prov = [&](Cols& c) { if (by) c.set("created_by", *by).set("modified_by", *by); };
  // the LEVEL dictionary (constantQDictionaryRows)
  DictionarySpec dict;
  dict.name = name;
  dict.description = "the levels of the constant-Q tiling '" + name + "': code = the level, 0 the finest; an entry carries the level's window in samples and seconds, its tile count, "
                     "how many finest tiles one tile holds, and what that window resolves in Hz \xe2\x80\x94 derived from the window and the sample rate, not a coordinate on any manifold";
  std::vector<DictionaryEntry> entries;
  for (const Level& l : t.levels) {
    DictionaryEntry e;
    e.code = l.level;
    e.name = i2s(l.window_samples) + " samples";
    e.ordinal = l.level;
    e.cols.set("measurement", l.window_s).set("measurement_unit", "s").set("attributes_json", jdump(levelInfo(t, l)));
    entries.push_back(std::move(e));
  }
  dict.entries = std::move(entries);
  const std::string dict_id = sub.dictionary(dict).id;
  // a TILE dictionary per level, EMPTY (levelTileDictionaryRows)
  std::vector<std::string> tileDicts;
  for (const Level& l : t.levels) {
    DictionarySpec td;
    td.name = name + " L" + std::to_string(l.level) + " tiles";
    td.entries = std::vector<DictionaryEntry>{};
    td.description = "the " + i2s(l.count) + " tiles of level " + std::to_string(l.level) +
                     " \xe2\x80\x94 a tile's code is its index in time order, the PRIMARY KEY a measurement, a label or a note hangs on (D83); what a tile IS (" +
                     i2s(l.window_samples) + " samples, " + fixed(l.window_s, 3) + " s) is the level's entry in the ladder dictionary";
    tileDicts.push_back(sub.dictionary(td).id);
  }
  // THE LADDER and its members (constantQRows)
  const std::size_t L = t.levels.size();
  SelectionSpec set;
  set.name = name;
  set.type = "ladder";
  set.manifold_id = manifold_id;
  set.no_elements = true;
  if (acquisition) set.cols.set("acquisition_id", *acquisition);
  set.cols.set("label", "constant-Q, " + std::to_string(L) + " levels")
      .set("description", "the constant-Q tiling of the time line: " + std::to_string(L) + " time resolutions, windows " + i2s(L ? t.levels.front().window_samples : 0) + "\xe2\x80\x93" +
                              i2s(L ? t.levels.back().window_samples : 0) + " samples, designed at " + std::to_string(t.voices) + " voices/octave and " +
                              js_json(Json(t.cycles)) + " cycles per window over " + fmtHz(t.f_min) + "\xe2\x80\x93" + fmtHz(t.f_max) +
                              " Hz \xe2\x80\x94 a tiling of TIME, on the time manifold and no other; the hertz describe what the windows resolve and reference nothing (D84)")
      .set("cell", "1").set("dictionary_id", dict_id).set("n_elements", t.samples).set("n_members", static_cast<std::int64_t>(L)).set("unit", "s");
  set.cols.json("params_json", setParams(t));
  prov(set.cols);
  w.set_id = sub.selection(std::move(set));
  for (std::size_t i = 0; i < L; ++i) {
    const Level& l = t.levels[i];
    SelectionSpec mem;
    mem.name = name + " L" + std::to_string(l.level);
    mem.type = "spans";
    mem.manifold_id = manifold_id;
    if (acquisition) mem.cols.set("acquisition_id", *acquisition);
    mem.cols.set("label", i2s(l.window_samples) + " samples \xc2\xb7 " + fixed(l.window_s, 3) + " s")
        .set("cell", "1").set("member_of_id", w.set_id).set("member_ordinal", static_cast<std::int64_t>(l.level)).set("code", static_cast<std::int64_t>(l.level))
        .set("dictionary_id", tileDicts[i]).set("n_elements", t.samples).set("n_members", l.count).set("unit", "s");
    mem.cols.json("params_json", memberParams(t, l));
    prov(mem.cols);
    w.member_ids.push_back(sub.selection(std::move(mem)));
  }
  return w;
}

}  // namespace nxr::io::constantq
