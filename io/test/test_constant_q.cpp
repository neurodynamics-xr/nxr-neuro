// nxr::io::constantq against the Studio's constant-Q tiling (golden: test/golden/gen_constant_q_golden.mts, backend/src/db/constant-q.ts): on five lines
// (1200 Hz like the fixture's MEG, 2400 Hz, 250 Hz from origin 1.5 s, the `cycles` method, 3 voices with fMin 2 Hz) the bands and levels (every integer
// exactly, every number to rounding), the level dictionary's entries (name, measurement, attributes), the ladder's label, description and params, its
// members' names, labels, tile counts and closed forms, and the tile dictionaries' descriptions — the rows the Studio writes. Then the writer on a synthetic
// dataset: the ladder on a time line (a set of `spans` members whose membership is the closed form), its dictionaries, idempotent (a second write finds it);
// the tile arithmetic (tileOf, tileIndexAt, tilesWithin); refusals.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/constant_q.h"

using namespace nxr::io;
namespace cq = nxr::io::constantq;
namespace fs = std::filesystem;

// two JSON values equal: integers, strings and booleans exactly, numbers to rounding (std::pow and V8's Math.pow may differ in the last place)
static bool close(const Json& a, const Json& b, std::string& where, const std::string& at = "") {
  if (a.is_number() && b.is_number()) {
    const double x = a.get<double>(), y = b.get<double>();
    if (x == y || std::fabs(x - y) <= 1e-12 * std::max(std::fabs(x), std::fabs(y))) return true;
    where = at + ": " + a.dump() + " vs " + b.dump();
    return false;
  }
  if (a.is_object() && b.is_object()) {
    if (a.size() != b.size()) { where = at + ": " + std::to_string(a.size()) + " keys vs " + std::to_string(b.size()); return false; }
    auto ia = a.begin();
    auto ib = b.begin();
    for (; ia != a.end(); ++ia, ++ib) {
      if (ia.key() != ib.key()) { where = at + ": key " + ia.key() + " vs " + ib.key(); return false; }
      if (!close(ia.value(), ib.value(), where, at + "." + ia.key())) return false;
    }
    return true;
  }
  if (a.is_array() && b.is_array()) {
    if (a.size() != b.size()) { where = at + ": length"; return false; }
    for (std::size_t i = 0; i < a.size(); ++i) if (!close(a[i], b[i], where, at + "[" + std::to_string(i) + "]")) return false;
    return true;
  }
  if (a == b) return true;
  where = at + ": " + a.dump() + " vs " + b.dump();
  return false;
}

int main() {
  std::ifstream in(std::string(NXR_IO_DATA_DIR) + "/../golden/constant_q_golden.json");
  CHECK(in.good());
  const Json golden = Json::parse(in);
  for (const Json& c : golden) {
    const std::string name = c["name"];
    cq::Params p;
    if (c["params"].contains("method")) p.method = c["params"]["method"] == "cycles" ? cq::Method::Cycles : cq::Method::Hertz;
    if (c["params"].contains("voices")) p.voices = c["params"]["voices"].get<int>();
    if (c["params"].contains("fMin")) p.fMin = c["params"]["fMin"].get<double>();
    const cq::Tiling t = cq::constantQTiling(c["line"]["sfreq"].get<double>(), c["line"]["samples"].get<std::int64_t>(), c["line"]["origin"].get<double>(), p);
    std::string where;
    // the tiling's own numbers and its levels, as the Studio's levelInfos (the dictionary's attributes)
    const Json& gt = c["tiling"];
    bool ok = t.bands.size() == gt["bands"].size() && t.levels.size() == gt["levels"].size();
    ok = ok && close(Json(t.cycles), gt["cycles"], where, "cycles") && close(Json(t.f_max), gt["f_max"], where, "f_max") && close(Json(t.f_min), gt["f_min"], where, "f_min");
    for (std::size_t i = 0; ok && i < t.bands.size(); ++i) {
      const Json& b = gt["bands"][i];
      ok = t.bands[i].window_samples == b["window_samples"].get<std::int64_t>() && t.bands[i].count == b["count"].get<std::int64_t>() &&
           close(Json(t.bands[i].lo_hz), b["lo_hz"], where, "band lo") && close(Json(t.bands[i].centre_hz), b["centre_hz"], where, "band centre") &&
           close(Json(t.bands[i].q), b["q"], where, "band q");
    }
    for (std::size_t i = 0; ok && i < t.levels.size(); ++i) {
      const Json want = Json::parse(c["entries"][i]["attributes_json"].get<std::string>());
      ok = close(cq::levelInfo(t, t.levels[i]), want, where, "level " + std::to_string(i)) && c["entries"][i]["name"] == std::to_string(t.levels[i].window_samples) + " samples" &&
           close(Json(t.levels[i].window_s), c["entries"][i]["measurement"], where, "measurement");
    }
    // the ladder's row and its members'
    ok = ok && close(cq::setParams(t), Json::parse(c["set"]["params_json"].get<std::string>()), where, "set params");
    for (std::size_t i = 0; ok && i < t.levels.size(); ++i)
      ok = close(cq::memberParams(t, t.levels[i]), Json::parse(c["members"][i]["params_json"].get<std::string>()), where, "member params " + std::to_string(i)) &&
           c["members"][i]["n_members"].get<std::int64_t>() == t.levels[i].count;
    std::printf("  %s: %zu bands, %zu levels (windows %lld..%lld samples): %s\n", name.c_str(), t.bands.size(), t.levels.size(),
                static_cast<long long>(t.levels.empty() ? 0 : t.levels.front().window_samples), static_cast<long long>(t.levels.empty() ? 0 : t.levels.back().window_samples),
                ok ? "equal to the Studio's" : ("DIFFERENT at " + where).c_str());
    CHECK(ok);
  }

  // THE WRITER on a synthetic dataset: a 1200 Hz line of 360000 samples (the golden's first case)
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_constant_q_" + std::to_string(::getpid()));
  fs::remove_all(root);
  DatasetWriter ds = create_dataset(root, "demo");
  std::string tid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      ManifoldSpec t;
      t.name = "rest_time"; t.path = "time/rest_time"; t.type = "line"; t.n_vertices = 360000;
      t.topology = {"path", 1, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = 360000; d.spacing = 1.0 / 1200.0; d.origin = 0.0; d.unit = "s";
      g.dimensions.push_back(d); t.geometry = g;
      t.cols.set("n_edges", 359999);
      tid = sub.manifold(t);
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  cq::Written w;
  sub.compose([&] { w = cq::write(sub, tid, {}, std::string("test")); });
  CHECK(!w.existed && w.member_ids.size() == golden[0]["members"].size());
  const auto set = sub.db().read("selection", w.set_id);
  CHECK(set && text(*set, "type") == "ladder" && text(*set, "name") == "rest_time_constantQ" && text(*set, "label") == golden[0]["set"]["label"].get<std::string>());
  CHECK(text(*set, "description") == golden[0]["set"]["description"].get<std::string>());
  const auto mem = sub.db().read("selection", w.member_ids[2]);
  CHECK(mem && text(*mem, "type") == "spans" && integer(*mem, "code") == 2 && text(*mem, "label") == golden[0]["members"][2]["label"].get<std::string>() &&
        text(*mem, "name") == "rest_time_constantQ L2" && is_null(*mem, "array_path"));
  const auto entries = sub.db().all("SELECT * FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", {V(text(*set, "dictionary_id"))});
  CHECK_EQ(static_cast<int>(entries.size()), static_cast<int>(golden[0]["entries"].size()));
  const auto tileDict = sub.db().one("SELECT * FROM dictionary WHERE id = ?", {V(text(*mem, "dictionary_id"))});
  CHECK(tileDict && text(*tileDict, "description") == golden[0]["tileDescriptions"][2].get<std::string>());
  cq::Written again;
  sub.compose([&] { again = cq::write(sub, tid); });
  CHECK(again.existed && again.set_id == w.set_id && again.member_ids == w.member_ids);

  // THE TILE ARITHMETIC (tileOf, tileIndexAt, tilesWithin)
  const cq::Closed c{0.0, 0.5, 0.5, 4};
  CHECK(cq::tileOf(c, 3, 1.8).second == 1.8 && cq::tileOf(c, 1, 1.8).first == 0.5);
  CHECK(cq::tileIndexAt(c, 0.7, 1.8) == 1 && !cq::tileIndexAt(c, 1.9, 1.8) && !cq::tileIndexAt(c, -0.1, 1.8));
  CHECK(cq::tilesWithin(4096, 512) == 8);
  bool refused = false;
  try { cq::tilesWithin(1000, 512); } catch (const Error&) { refused = true; }
  CHECK(refused);
  refused = false;
  try { cq::constantQTiling(1200, 0, 0); } catch (const Error&) { refused = true; }
  CHECK(refused);
  // THE TILE HELPERS (2026-11-13): a band's window as the ladder sizes it; seconds to the nearest level
  CHECK(cq::windowFor(1200, 8) == 1024 && cq::windowFor(1200, 15) == 1024 && cq::windowFor(1200, 4) == 2048 && cq::windowFor(2400, 8) == 2048);
  for (const cq::Band& b : cq::constantQTiling(1200, 1 << 20, 0).bands) CHECK(cq::windowFor(1200, b.lo_hz) == b.window_samples);   // the ladder's own
  CHECK(cq::levelOf(4, 1200) == 12 && cq::levelOf(4, 2400) == 13 && cq::levelOf(4, 1000) == 12 && cq::levelOf(0.0001, 1000) == 0);
  CHECK(cq::levelOfWidth(4096) == 12 && cq::levelOfWidth(1) == 0 && cq::levelOfWidth(4800) == -1);
  CHECK(cq::strideFor(1200, 8) == 4 && cq::strideFor(1200, 15) == 2);
  fs::remove_all(root);
  return nxrtest::finish("test_constant_q");
}
