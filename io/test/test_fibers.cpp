// Fiber bundles in a package (nxr/io/fibers.h) and the .trk import (nxr/convert/fibers.h), on synthetic data: a bundle stored in float32 and in float64 and read back
// equal, its ends and distances (stored order), the origin carried to a derived result, the refusals, precision following the input, and a TrackVis file imported with a caller's transform.
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/convert/fibers.h"
#include "nxr/convert/templates.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/fibers.h"
#include "nxr/io/subject_package.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}
std::string what_of(const std::function<void()>& f) {
  try { f(); } catch (const std::exception& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// an octahedron of radius 0.1 m as the surface "pial"
void add_surface(SubjectWriter& sub) {
  const std::vector<double> v = {0.1, 0, 0, -0.1, 0, 0, 0, 0.1, 0, 0, -0.1, 0, 0, 0, 0.1, 0, 0, -0.1};
  const std::vector<std::int32_t> f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  ManifoldSpec m;
  m.name = "pial";
  m.path = "source/pial";
  m.type = "surface";
  m.n_vertices = 6;
  m.topology = {"simplicial", 2, 8, std::string("ccw-outward"), {{2, "source/pial/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = "source/pial/vertices";
  m.geometry = g;
  m.populate = [&](Store&, const std::string&) {
    sub.write_array("source/pial/vertices", view(v, {6, 3}));
    sub.write_array("source/pial/faces", view(f, {8, 3}));
  };
  sub.manifold(m);
}

void write_trk(const fs::path& p, const std::vector<std::vector<std::array<float, 3>>>& streamlines) {
  std::vector<char> h(1000, 0);
  std::memcpy(h.data(), "TRACK", 5);
  const std::int16_t dim[3] = {10, 12, 14};
  const float vs[3] = {1.0f, 1.0f, 1.0f};
  const float aff[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  std::memcpy(h.data() + 6, dim, 6);
  std::memcpy(h.data() + 12, vs, 12);
  std::memcpy(h.data() + 440, aff, 64);
  std::memcpy(h.data() + 948, "RAS", 3);
  const std::int32_t version = 2, size = 1000;
  std::memcpy(h.data() + 992, &version, 4);
  std::memcpy(h.data() + 996, &size, 4);
  std::ofstream o(p, std::ios::binary);
  o.write(h.data(), 1000);
  for (const auto& s : streamlines) {
    const std::int32_t n = static_cast<std::int32_t>(s.size());
    o.write(reinterpret_cast<const char*>(&n), 4);
    for (const auto& q : s) o.write(reinterpret_cast<const char*>(q.data()), 12);
  }
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_fibers_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  CreatePackageOptions opt;
  opt.home_dataset = "demo";
  const Json origin = Json{{"schema", "nxr-origin/1"}, {"source", Json{{"name", "a synthetic bundle"}}}, {"citation", Json::array({"A. Author (2026) A paper."})},
                           {"licence", Json{{"id", "CC-BY-SA-4.0"}, {"share_alike", true}}}};
  const fs::path pkg = root / "S1.nxr.zarr";
  const std::int64_t nF = 5, nP = 4;
  fibers::Bundle<float> bf;
  bf.name = "tracts";
  bf.n_fibers = nF; bf.points_per_fiber = nP;
  for (std::int64_t i = 0; i < nF * nP; ++i) { bf.points.push_back(0.01f * static_cast<float>(i)); bf.points.push_back(-0.001f * static_cast<float>(i)); bf.points.push_back(0.1f); }
  bf.source = "synthetic"; bf.origin = origin;
  fibers::Ends<float> ef;
  ef.vertex = {0, 1, 2, 3, 4, 5, 0, 0, 5, 4};
  ef.distance = {0.001f, 0.002f, 0.003f, 0.004f, 0.005f, 0.006f, 0.007f, 0.008f, 0.009f, 0.010f};
  {
    DatasetWriter w = create_package(pkg, "S1", opt);
    SubjectWriter sub = SubjectWriter::open(w, "S1");
    add_surface(sub);
    const auto stored = fibers::store_bundle<float>(sub, bf);
    CHECK_EQ(stored.precision, std::string("float32"));
    CHECK_EQ(stored.path, std::string("fibre/tracts"));
    CHECK_EQ(code_of([&] { fibers::store_bundle<float>(sub, bf); }), std::string(code::invalid_argument));   // the name is taken
    fibers::Bundle<float> bad = bf; bad.name = "bad"; bad.points.pop_back();
    CHECK(has(what_of([&] { fibers::store_bundle<float>(sub, bad); }), "points are ["));
    bad = bf; bad.name = "bad"; bad.points_per_fiber = 1;
    CHECK(has(what_of([&] { fibers::store_bundle<float>(sub, bad); }), "at least 2 points"));
    fibers::Bundle<double> bd;                                                                                // a bundle in float64, beside it
    bd.name = "wide"; bd.n_fibers = 1; bd.points_per_fiber = 2; bd.points = {0, 0, 0.1, 0.01, 0, 0.1};
    fibers::store_bundle<double>(sub, bd);
    fibers::store_ends<float>(sub, "tracts", "pial", ef, Json{{"gate", "none"}});
    CHECK_EQ(code_of([&] { fibers::store_ends<float>(sub, "tracts", "pial", ef); }), std::string(code::invalid_argument));   // the pair is stored once
    fibers::Ends<double> out_of_range;
    out_of_range.vertex = {6, 0}; out_of_range.distance = {0.0, 0.0};
    CHECK(has(what_of([&] { fibers::store_ends<double>(sub, "wide", "pial", out_of_range); }), "outside"));
    CHECK(has(what_of([&] { fibers::store_ends<float>(sub, "tracts", "nosuch", ef); }), "no surface"));
    sub.complete();
  }
  {
    SubjectPackage p = SubjectPackage::open(pkg);
    const task::Subject s = task::Subject::of(p);
    const auto b = fibers::read_bundle<float>(s, p.store(), "tracts");
    CHECK_EQ(b.n_fibers, nF);
    CHECK_EQ(b.points_per_fiber, nP);
    CHECK(b.points == bf.points);
    CHECK_EQ(fibers::bundle_precision(s, p.store(), "tracts"), std::string("float32"));
    CHECK_EQ(fibers::bundle_precision(s, p.store(), "wide"), std::string("float64"));
    CHECK(has(what_of([&] { fibers::read_bundle<double>(s, p.store(), "tracts"); }), "precision follows the input"));
    CHECK_EQ(fibers::read_bundle<double>(s, p.store(), "wide").points.size(), std::size_t(6));
    CHECK_EQ(code_of([&] { fibers::read_bundle<float>(s, p.store(), "none"); }), std::string(code::no_such_row));
    const auto e = fibers::read_ends<float>(s, p.store(), "tracts", "pial");
    CHECK(e.vertex == ef.vertex);                               // fiber f's first end at 2f, its last at 2f + 1
    CHECK(e.distance == ef.distance);                           // in metres, the fibers' precision
    CHECK(has(what_of([&] { fibers::read_ends<double>(s, p.store(), "tracts", "pial"); }), "float32"));
    // the origin is in the bundle's record and travels to a derived result with its licence and attribution
    task::Want w; w.cls = task::Cls::manifold; w.name = "tracts";
    const task::Node n = s.require(w, "tracts");
    const Json derived = fibers::derived_origin(n.producer, "tracts");
    CHECK_EQ(derived["licence"]["id"].get<std::string>(), std::string("CC-BY-SA-4.0"));
    CHECK_EQ(derived["citation"][0].get<std::string>(), std::string("A. Author (2026) A paper."));
    CHECK_EQ(derived["derived_from"].get<std::string>(), std::string("tracts"));
    CHECK(has(derived["derived_notice"].get<std::string>(), "ShareAlike"));
    task::Want ow; ow.cls = task::Cls::operator_; ow.name = "tracts_ends_pial";
    const task::Node op = s.require(ow, "the ends");
    CHECK_EQ(op.producer["origin"]["licence"]["id"].get<std::string>(), std::string("CC-BY-SA-4.0"));   // the ends carry the licence too
    CHECK_EQ(op.producer["gate"].get<std::string>(), std::string("none"));
    CHECK(fibers::derived_origin(Json::object(), "x").is_null());
    CHECK(p.verify().ok);
  }

  // ── the .trk import ──
  {
    write_trk(root / "t.trk", {{{0, 0, 0}, {10, 0, 0}, {20, 0, 0}}, {{0, 5, 0}, {0, 5, 10}}, {{1, 1, 1}}});   // the last has one point: dropped
    const fs::path pkg2 = root / "S2.nxr.zarr";
    DatasetWriter w = create_package(pkg2, "S2", opt);
    SubjectWriter sub = SubjectWriter::open(w, "S2");
    nxr::convert::TrkImport rq;
    rq.trk = root / "t.trk";
    rq.name = "tracts";
    rq.points = 5;
    CHECK(has(what_of([&] { nxr::convert::import_trk(sub, rq); }), "transform"));          // the transform is the caller's: no default
    rq.to_scs_m = std::array<double, 12>{0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001, 0, 0, 0.05};   // mm to m, a shift of 5 cm along z
    rq.origin = origin;
    const Json r = nxr::convert::import_trk(sub, rq);
    CHECK_EQ(r["n_fibers"].get<std::int64_t>(), std::int64_t(2));
    CHECK_EQ(r["n_dropped"].get<std::int64_t>(), std::int64_t(1));
    sub.complete();
    SubjectPackage p = SubjectPackage::open(pkg2);
    const auto b = fibers::read_bundle<float>(task::Subject::of(p), p.store(), "tracts");
    CHECK_EQ(b.points_per_fiber, std::int64_t(5));
    auto near = [](float a, double c) { return std::abs(a - c) < 1e-5; };
    // fiber 0 runs from x = 0 to 20 mm at equal arc length: 5 points at 0, 5, 10, 15, 20 mm (the .trk's own world is RAS: the identity here), then mm -> m and + 5 cm in z
    // the reader's own world coordinates (the half-voxel shift of TrackVis' corner convention included) give the expectation: the fiber's first point, then mm -> m and + 5 cm in z
    const nxr::convert::Tractogram tg = nxr::convert::read_trk(root / "t.trk");
    const double x0 = tg.points[0], y0 = tg.points[1], z0 = tg.points[2], y1 = tg.points[3 * 3 + 1], z1 = tg.points[3 * 3 + 2];
    CHECK(near(b.points[0], x0 * 1e-3) && near(b.points[1], y0 * 1e-3) && near(b.points[2], z0 * 1e-3 + 0.05));
    CHECK(near(b.points[3] - b.points[0], 0.005) && near(b.points[12] - b.points[0], 0.020));             // fiber 0: 5 points at equal arc length over 20 mm
    CHECK(near(b.points[3 * 5 + 1], y1 * 1e-3) && near(b.points[3 * 9 + 2], (z1 + 10) * 1e-3 + 0.05));    // fiber 1: its last point is 10 mm above its first in z
    CHECK_EQ(b.origin["licence"]["id"].get<std::string>(), std::string("CC-BY-SA-4.0"));
    CHECK_EQ(b.source, std::string("t.trk"));
  }
  fs::remove_all(root);
  return nxrtest::finish("fibers");
}
