// nxr::io::constantq (the tile grid: tileOf, tileIndexAt, tilesWithin, levelOf, levelOfWidth, windowFor, strideFor) and the selection module's ladder: io STORES a
// designed ladder (the design itself is compute's, tested there against the Studio's golden test/golden/gen_constant_q_golden.mts). Here a ladder of three levels
// (1024, 2048, 4096 samples at 1200 Hz) is made by hand as a description, saved on a time line (a `ladder` Selection whose members are `spans` holding the closed form,
// a level dictionary, an empty tile dictionary per level), found again by its parameters, loaded back as the same description, and updated and deleted.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/selection_store.h"

using namespace nxr::io;
namespace cq = nxr::io::constantq;
namespace fs = std::filesystem;

int main() {
  // THE LADDER, stored: a 1200 Hz line of 360000 samples
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
  selection::Description ladder;
  ladder.type = selection::Type::set;
  ladder.unit = "s";
  for (const std::int64_t window : {1024, 2048, 4096}) {
    selection::Description level;
    level.type = selection::Type::set;
    level.unit = "s";
    level.level = cq::levelOfWidth(window);
    level.tiling = selection::Tiling{0.0, static_cast<double>(window) / 1200.0, static_cast<double>(window) / 1200.0, (360000 + window - 1) / window};
    level.band = std::make_pair(1200.0 / window * 8, 1200.0 / window * 12);
    level.band_unit = "Hz";
    ladder.members.push_back(level);
  }
  selection::Placement at;
  at.name = "rest_time_ladder";
  at.manifold_id = tid;
  at.created_by = "test";
  at.params = Json{{"method", "hertz"}, {"voices", 6}};
  selection::Saved saved;
  sub.compose([&] { saved = selection::save(sub, ladder, at); });
  CHECK(saved.member_ids.size() == 3);
  const auto set = sub.db().read("selection", saved.id);
  CHECK(set && text(*set, "type") == "ladder" && text(*set, "name") == "rest_time_ladder" && integer(*set, "n_members") == 3);
  const auto mem = sub.db().read("selection", saved.member_ids[2]);
  CHECK(mem && text(*mem, "type") == "spans" && integer(*mem, "code") == 2 && text(*mem, "name") == "rest_time_ladder L2" && is_null(*mem, "array_path") &&
        integer(*mem, "n_members") == (360000 + 4095) / 4096);
  CHECK_EQ(static_cast<int>(sub.db().all("SELECT * FROM dictionary_entry WHERE dictionary_id = ?", {V(text(*set, "dictionary_id"))}).size()), 3);
  const auto tile_dict = sub.db().one("SELECT * FROM dictionary WHERE id = ?", {V(text(*mem, "dictionary_id"))});
  CHECK(tile_dict && !sub.db().one("SELECT 1 FROM dictionary_entry WHERE dictionary_id = ?", {V(text(*tile_dict, "id"))}));
  // found again by its parameters (a second ladder is not made)
  const auto found = selection::find_with_param(sub.db(), sub.id(), tid, "ladder", "voices");
  CHECK(found && found->id == saved.id && found->member_ids == saved.member_ids);
  // loaded back as the description it was made from
  const auto loaded = selection::load(sub.db(), sub.store(), saved.id);
  CHECK(loaded.description.type == selection::Type::set && loaded.description.members.size() == 3);
  for (std::size_t k = 0; k < 3; ++k) {
    const auto& m = loaded.description.members[k];
    CHECK(m.tiling && m.tiling->count == ladder.members[k].tiling->count && std::fabs(m.tiling->width - ladder.members[k].tiling->width) < 1e-12);
    CHECK(m.level == ladder.members[k].level && m.band && std::fabs(m.band->first - ladder.members[k].band->first) < 1e-9);
  }
  CHECK(loaded.description.lookup.size() == 3);
  // a ladder is updated member by member; its levels are not replaced in one call
  bool refused = false;
  try { sub.compose([&] { selection::update(sub, saved.id, ladder); }); } catch (const Error&) { refused = true; }
  CHECK(refused);
  // a level's tiling updated in place
  selection::Description shorter = loaded.description.members[0];
  shorter.tiling->count = 10;
  sub.compose([&] { selection::update(sub, saved.member_ids[0], shorter); });
  CHECK(integer(*sub.db().read("selection", saved.member_ids[0]), "n_members") == 10);
  sub.compose([&] { selection::remove(sub, saved.id); });
  CHECK(!sub.db().read("selection", saved.id) && !sub.db().read("selection", saved.member_ids[1]));

  // THE TILE ARITHMETIC (tileOf, tileIndexAt, tilesWithin)
  const cq::Closed c{0.0, 0.5, 0.5, 4};
  CHECK(cq::tileOf(c, 3, 1.8).second == 1.8 && cq::tileOf(c, 1, 1.8).first == 0.5);
  CHECK(cq::tileIndexAt(c, 0.7, 1.8) == 1 && !cq::tileIndexAt(c, 1.9, 1.8) && !cq::tileIndexAt(c, -0.1, 1.8));
  CHECK(cq::tilesWithin(4096, 512) == 8);
  refused = false;
  try { cq::tilesWithin(1000, 512); } catch (const Error&) { refused = true; }
  CHECK(refused);
  // THE TILE HELPERS (2026-11-13): a band's window as the ladder sizes it; seconds to the nearest level
  CHECK(cq::windowFor(1200, 8) == 1024 && cq::windowFor(1200, 15) == 1024 && cq::windowFor(1200, 4) == 2048 && cq::windowFor(2400, 8) == 2048);
  CHECK(cq::levelOf(4, 1200) == 12 && cq::levelOf(4, 2400) == 13 && cq::levelOf(4, 1000) == 12 && cq::levelOf(0.0001, 1000) == 0);
  CHECK(cq::levelOfWidth(4096) == 12 && cq::levelOfWidth(1) == 0 && cq::levelOfWidth(4800) == -1);
  CHECK(cq::strideFor(1200, 8) == 4 && cq::strideFor(1200, 15) == 2);
  fs::remove_all(root);
  return nxrtest::finish("test_constant_q");
}
