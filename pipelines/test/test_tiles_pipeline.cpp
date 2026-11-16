// The tile ladder pipeline on a synthetic two-hemisphere brain: the ladder is stored as Selections (a `ladder` Selection with a level dictionary, one `set` member per level with the tile code of
// every vertex), a re-run keeps it and --replace rewrites it, the stored codes are the library's subject tree, and tile summaries of a scalar and a vector map are written in both vector forms
// (frame components in the canonical frames; the Levi-Civita resultant), labelled and recorded on the level Selection.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/io/measures.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/tiles.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

template <class T> static std::vector<T> read_array(SubjectWriter& sub, const std::string& path) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_tiles_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(3, false);
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S");
    sub.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph; add_surface(sub, so, b.P, b.F); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S");
  FramesComputeOptions fo;
  frames_compute(ds, sub, fo);

  TilesComputeOptions to;
  const TilesComputeResult r = tiles_compute(ds, sub, to);
  const int depth = r.info["depth"].get<int>();
  std::printf("  ladder: %d levels over %d pieces, %d tiles at depth %d\n", r.info["levels"].get<int>(), r.info["pieces"].get<int>(), r.info["tiles_at_depth"].get<int>(), depth);
  CHECK(!r.existed && depth >= 3 && r.info["pieces"].get<int>() == 2);
  const auto lad = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'ladder'", {V(sub.id()), V(std::string("cortex tiles"))});
  CHECK(lad.has_value() && !is_null(*lad, "dictionary_id") && integer(*lad, "n_members") == depth + 1);
  if (!lad) return 1;
  const auto members = sub.db().all("SELECT * FROM selection WHERE member_of_id = ? ORDER BY member_ordinal", {V(text(*lad, "id"))});
  CHECK_EQ(static_cast<int>(members.size()), depth + 1);
  const auto cnt = sub.db().one("SELECT COUNT(*) AS c FROM dictionary_entry WHERE dictionary_id = ?", {V(text(*lad, "dictionary_id"))});
  CHECK(cnt.has_value() && integer(*cnt, "c") == depth + 1);
  // the tiles are Selections: no manifold, no folder of their own besides the surface's atlas node
  CHECK(!sub.db().one("SELECT id FROM manifold WHERE subject_id = ? AND name LIKE '%tiles%'", {V(sub.id())}).has_value());

  const SurfaceInput S = read_surface(sub, std::nullopt, true);      // float64 as stored: the ladder's input (the Studio's stored ladder, nxr)
  const nxr::ladder::Tree tree = nxr::ladder::subjectTree(S.vertices.data(), static_cast<int>(S.n_vertices()), S.faces.data(), static_cast<int>(S.n_faces()));
  const std::vector<std::int32_t> codes = read_array<std::int32_t>(sub, text(members.back(), "array_path"));
  CHECK(codes == tree.codes);
  const std::vector<double> w = read_array<double>(sub, Json::parse(text(members[2], "params_json"))["areas"].get<std::string>());
  double sum = 0;
  for (double x : w) sum += x;
  double area = 0;
  for (double a : tree.pieceArea) area += a;
  CHECK(w.size() == 8 && std::abs(sum - area) < 1e-9 * area);
  const auto adj = read_array<std::int32_t>(sub, Json::parse(text(members[2], "params_json"))["node"].get<std::string>() + "/lc/L2/adjacency");
  CHECK(!adj.empty() && adj.size() % 2 == 0);

  // a re-run keeps; --replace rewrites
  CHECK(tiles_compute(ds, sub, to).existed);
  to.replace = true;
  CHECK(!tiles_compute(ds, sub, to).existed);
  const auto lad2 = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'ladder'", {V(sub.id()), V(std::string("cortex tiles"))});
  CHECK(lad2.has_value() && text(*lad2, "id") != text(*lad, "id"));
  if (!lad2) return 1;
  const auto members2 = sub.db().all("SELECT * FROM selection WHERE member_of_id = ? ORDER BY member_ordinal", {V(text(*lad2, "id"))});
  CHECK_EQ(static_cast<int>(members2.size()), depth + 1);

  // summaries: a scalar (x coordinate) and a tangent vector (the canonical north e1, then a raw ambient constant)
  const StoredFrames sf = read_stored_frames(sub, "cortex", S.n_vertices());
  CHECK(sf.canonical);
  TileMaps maps;
  maps.stem = "demo";
  Eigen::VectorXd x(S.n_vertices());
  for (int i = 0; i < S.n_vertices(); ++i) x[i] = S.vertices[3 * static_cast<size_t>(i)];
  maps.scalars.push_back({"x", x});
  {   // the Field the scalar map is of: its summaries are measurement rows of it
    FieldSpec fs;
    fs.name = "x_map"; fs.kind = "map"; fs.manifold_id = S.id; fs.value_type = "scalar";
    std::string fid;
    sub.compose([&] { fid = sub.field(std::move(fs)); });
    TileMapSource src; src.field_id = fid; src.name = "x_map";
    maps.sources["x"] = src;
  }
  nxr::frames::Mat3 v = sf.e1;
  for (int i = 0; i < S.n_vertices(); ++i) if (!std::isfinite(v(i, 0))) v.row(i).setZero();
  maps.vectors.push_back({"north", v});
  TileSummaryOptions so;
  so.levels = {1, 2, 3};
  const Json out = tile_summaries_write(ds, sub, maps, so);
  CHECK(out["levels"].size() == 3 && out["gauge"].get<std::string>() == "canonical frames");
  const Json p2 = Json::parse(text(*sub.db().one("SELECT * FROM selection WHERE id = ?", {V(text(members2[2], "id"))}), "params_json"));
  CHECK(p2["summaries"].size() == 2 && p2["summaries"][1]["forms"].size() == 2);
  CHECK(p2["summaries"][1]["forms"][0]["reduction"].get<std::string>() == "frame" && p2["summaries"][1]["forms"][1]["reduction"].get<std::string>() == "levi-civita");
  const std::string dir = p2["node"].get<std::string>();
  const std::vector<double> len = read_array<double>(sub, dir + "/summary/demo/north/levi_civita/length");
  const std::vector<double> c1 = read_array<double>(sub, dir + "/summary/demo/north/frame/c1"), wv = read_array<double>(sub, dir + "/summary/demo/north/frame/w");
  CHECK(len.size() == 8 && c1.size() == 8);
  {   // THE SCALAR SUMMARY IS MEASUREMENT ROWS (nxr's maps/<field>/{w,s1,s2,n,min,max}); the mean is derived at read and equals the direct area-weighted mean of the vertices
    const std::string fid = maps.sources["x"].field_id;
    CHECK(!sub.db().one("SELECT measure FROM selection_measurement WHERE selection_id = ? AND measure = 'mean'", {V(text(members2[2], "id"))}).has_value());
    int rows = 0;
    for (const char* m : {"weight", "sum", "sum_square", "count", "min", "max"})
      if (sub.db().one("SELECT measure FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = ? AND array_path = ?", {V(text(members2[2], "id")), V(fid), V(std::string(m)), V(dir + "/maps/x_map/" + std::string(m == std::string("weight") ? "w" : m == std::string("sum") ? "s1" : m == std::string("sum_square") ? "s2" : m == std::string("count") ? "n" : m))}).has_value()) ++rows;
    CHECK_EQ(rows, 6);
    nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(S.n_vertices()); mesh.positions = S.vertices; mesh.faces = S.faces;
    const std::vector<double> va = nxr::ladder::vertexAreas(mesh);
    const std::vector<std::int32_t> c2 = read_array<std::int32_t>(sub, text(members2[2], "array_path"));
    std::vector<double> num(8, 0.0), den(8, 0.0);
    for (int i = 0; i < S.n_vertices(); ++i) if (c2[static_cast<size_t>(i)] >= 0) { num[static_cast<size_t>(c2[static_cast<size_t>(i)])] += va[static_cast<size_t>(i)] * x[i]; den[static_cast<size_t>(c2[static_cast<size_t>(i)])] += va[static_cast<size_t>(i)]; }
    double worst = 0;
    int got = 0;
    for (const Json& r : query_measurements(sub.db(), sub.store(), Json{{"selectionId", text(members2[2], "id")}, {"measure", "mean"}, {"ofFieldId", fid}, {"spaceLevel", 2}})) {
      const size_t c = r["code"].get<size_t>();
      worst = std::max(worst, std::abs(r["value"].get<double>() - num[c] / den[c]) / (std::abs(num[c] / den[c]) + 1e-12));
      ++got;
    }
    // and rolled up to level 1 (two tiles of the level 2 each) it is the weighted mean of its children
    int got1 = 0;
    for (const Json& r : query_measurements(sub.db(), sub.store(), Json{{"selectionId", text(members2[2], "id")}, {"measure", "mean"}, {"ofFieldId", fid}, {"spaceLevel", 1}})) {
      const size_t c = r["code"].get<size_t>();
      const double n1 = num[2 * c] + num[2 * c + 1], d1 = den[2 * c] + den[2 * c + 1];
      worst = std::max(worst, std::abs(r["value"].get<double>() - n1 / d1) / (std::abs(n1 / d1) + 1e-12));
      ++got1;
    }
    std::printf("  scalar summary as rows: %d tiles at level 2, %d at level 1; mean at read vs the direct area-weighted mean: worst relative difference %.2e\n", got, got1, worst);
    CHECK(got == 8 && got1 == 4 && worst < 1e-12);
  }
  double lmin = 1;
  for (double l : len) if (std::isfinite(l)) lmin = std::min(lmin, l);
  std::printf("  north (e1) on 8 tiles: frame c1 / w = %.3f %.3f ..., Levi-Civita length min %.3f\n", c1[0] / wv[0], c1[1] / wv[1], lmin);
  CHECK(std::abs(c1[0] / wv[0] - 1) < 1e-6);                 // e1 in its own frame is (1, 0) everywhere
  CHECK(lmin > 0.3 && lmin <= 1 + 1e-9);                     // transported, a non-parallel frame field still has a resultant in (0, 1]
  bool refused = false;
  try { tile_summaries_write(ds, sub, maps, so); } catch (const std::exception&) { refused = true; }
  CHECK(refused);
  so.replace = true;
  tile_summaries_write(ds, sub, maps, so);
  const Json p2b = Json::parse(text(*sub.db().one("SELECT * FROM selection WHERE id = ?", {V(text(members2[2], "id"))}), "params_json"));
  CHECK(p2b["summaries"].size() == 2);                       // replaced, not appended
  // the GROUP TREE is the default subject's only: a regular subject is refused; its definitions are nxr's defaults
  bool group_refused = false;
  try { group_tree_compute(ds, sub, {}); } catch (const Error& e) { group_refused = e.code() == std::string(code::invalid_argument); }
  CHECK(group_refused);
  const Json defs = group_tree_definitions({});
  CHECK(defs["tree"] == "group" && defs["depth"] == 8 && defs["frame_level"] == -18 && std::abs(defs["frame_s"].get<double>() - 86400.0 / 262144.0) < 1e-15);
  CHECK(defs["bands_hz"].size() == 6 && defs["bands_hz"][5][1] == 64.0 && defs["frames"].is_null());
  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_tiles_pipeline");
}
