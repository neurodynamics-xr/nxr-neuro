// export gifti (nxr/exporters/gifti.h) on a synthetic package: two tetrahedra as the left and right hemispheres of a "cortex", a labelling with a dictionary, a map and a
// series on the surface. Each file is read back with the pure GIFTI reader; provenance, terms and refusals are checked. With NXR_EXPORT_GOLDEN_DIR set the files are
// kept there with expected.json, for io/test/parity/test_export_gifti.py (nibabel reads them).
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <filesystem>

#include "check.h"
#include "nxr/exporters/gifti.h"
#include "nxr/formats/gifti.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
using namespace nxr::io::task;
namespace fs = std::filesystem;
namespace ex = nxr::exporters;
namespace gifti = nxr::formats::gifti;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}

constexpr int T = 40;                      // samples of the series: 2 Hz from t = 1 s
double series_value(int v, int t) { return v + 100.0 * t; }

const gifti::DataArray* named(const gifti::Image& img, const std::string& intent, std::size_t nth = 0) {
  std::size_t k = 0;
  for (const auto& a : img.arrays) if (a.intent == intent && k++ == nth) return &a;
  return nullptr;
}
std::string meta_of(const gifti::DataArray& a, const std::string& k) { const std::string* m = a.meta(k); return m ? *m : std::string("<none>"); }

ex::Context ctx_of(const std::string& what) {
  ex::Context c;
  c.version = "9.9.9";
  c.command = "nxr-neuro export gifti " + what;
  c.options = Json{{"what", what}};
  return c;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_export_gifti_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path pkg = root / "S.nxr.zarr";
  const char* golden = std::getenv("NXR_EXPORT_GOLDEN_DIR");

  // two tetrahedra: vertices 0-3 the left hemisphere, 4-7 the right
  std::vector<double> xyz;
  for (int h = 0; h < 2; ++h) {
    const double dx = h ? 0.1 : 0.0;
    for (const auto& p : std::vector<std::array<double, 3>>{{0, 0, 0}, {0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}}) { xyz.push_back(p[0] + dx); xyz.push_back(p[1]); xyz.push_back(p[2]); }
  }
  std::vector<std::int32_t> faces;
  for (int h = 0; h < 2; ++h)
    for (const auto& f : std::vector<std::array<int, 3>>{{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}) for (int c = 0; c < 3; ++c) faces.push_back(4 * h + f[c]);
  std::vector<double> map(8);
  for (int v = 0; v < 8; ++v) map[v] = 0.1 * (v + 1) + 1e-9;           // float64: the export converts to float32

  std::string cortex_id, map_id, labelling_id;
  {
    CreatePackageOptions po;
    po.home_dataset = "demo";
    DatasetWriter w = create_package(pkg, "S", po);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    SurfaceOptions so;
    so.name = "cortex";
    so.hemispheres = {{"left", 0, 4}, {"right", 4, 4}};
    AtlasMember atlas;
    atlas.family = "demo-atlas";
    atlas.codes = {0, 1, 1, 1, 2, 2, 2, 0};
    atlas.levels = {"unassigned", "alpha", "beta"};
    atlas.colors = {{0, 0, 0}, {255, 0, 128}, {0, 0, 255}};
    so.atlases = {atlas};
    const SurfaceInfo si = add_surface(sub, so, xyz, faces);
    cortex_id = si.id;
    FieldSpec fm;
    fm.name = "map"; fm.path = "source/map"; fm.kind = "map"; fm.manifold_id = si.id; fm.data = view(map, {8});
    fm.cols.set("unit", "V");
    map_id = sub.field(fm);
    set_origin(w.db(), "field", map_id, OriginRecord{"fixture:map", "CC-BY-SA-4.0", "A. Author", true});
    set_origin(w.db(), "manifold", si.id, OriginRecord{"fixture:surface", "CC0-1.0", "B. Builder", true});
    sub.complete();
  }
  {   // the series on surface x time line, and a labelling with a code of -1
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    Subject t = Subject::of(sub);
    Unit u = t.unit(By{"test", Json::object()});
    LineOut ln;
    ln.name = "t"; ln.n = T; ln.spacing = 0.5; ln.origin = 1.0; ln.unit = "s"; ln.dimension = "time"; ln.folder = "time"; ln.cs_key = "clock";
    const std::string line_id = u.line(ln);
    const std::string prod = u.product(ProductOut{{cortex_id, line_id}, std::nullopt});
    FieldOut fo;
    fo.name = "series"; fo.output = "series"; fo.kind = "map"; fo.manifold_id = prod; fo.unit = "A/m"; fo.data.shape = {8, T};
    for (int v = 0; v < 8; ++v) for (int k = 0; k < T; ++k) fo.data.f32.push_back(static_cast<float>(series_value(v, k)));
    fo.producer = Json{{"method", "synthetic"}};
    u.field(fo);
    SelectionOut part;
    part.name = "uses_zero"; part.output = "uses_zero"; part.type = "partition"; part.manifold_id = cortex_id; part.member_of = {-1, 0, 0, 0, 1, 1, 1, 1}; part.member_codes = {0, 1}; part.entries = {"a", "b"};
    u.selection(part);
    u.commit();
    // a labelling whose dictionary leaves code 0 free: -1 (no label) can be key 0
    DictionarySpec d;
    d.name = "halves-dict";
    d.entries = std::vector<DictionaryEntry>{DictionaryEntry{3, "three", 0, std::array<int, 3>{1, 2, 3}, {}}, DictionaryEntry{4, "four", 1, std::nullopt, {}}};
    const DictionaryResult dr = sub.dictionary(d);
    std::vector<std::int32_t> codes = {-1, 3, 3, 3, 4, 4, 4, 4};
    SelectionSpec ss;
    ss.name = "halves"; ss.path = "source/halves"; ss.type = "set"; ss.manifold_id = cortex_id; ss.data = view(codes, {8});
    ss.cols.set("dictionary_id", dr.id).set("n_elements", static_cast<std::int64_t>(8));
    sub.selection(std::move(ss));
    sub.complete();
  }
  {   // a field that is not a scalar, on the same surface
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    std::vector<double> vec(24, 1.0);
    FieldSpec fv;
    fv.name = "current"; fv.path = "source/current"; fv.kind = "map"; fv.manifold_id = cortex_id; fv.value_type = "vector3"; fv.data = view(vec, {8, 3});
    sub.field(fv);
    sub.complete();
  }
  CHECK(SubjectPackage::open(pkg).verify().ok);

  // ---- a map: .shape.gii with the surface and the hemispheres
  {
    ex::GiftiExport o;
    o.field = "map"; o.out = (root / "map").string(); o.kind = "shape"; o.surface = true; o.hemispheres = true; o.ctx = ctx_of("map");
    const Json r = ex::export_gifti(pkg, o);
    CHECK_EQ(r["files"].size(), static_cast<std::size_t>(4));
    for (const auto& f : r["files"]) CHECK(fs::exists(f.get<std::string>()));
    const gifti::Image left = gifti::read_file((root / "map_hemi-L.shape.gii").string());
    const gifti::Image right = gifti::read_file((root / "map_hemi-R.shape.gii").string());
    CHECK_EQ(left.arrays.size(), static_cast<std::size_t>(1));
    const gifti::DataArray& a = left.arrays[0];
    CHECK_EQ(a.intent, std::string("NIFTI_INTENT_SHAPE"));
    CHECK_EQ(a.dims, std::vector<std::int64_t>({4}));
    for (int v = 0; v < 4; ++v) CHECK_EQ(a.f32[v], static_cast<float>(map[v]));
    for (int v = 0; v < 4; ++v) CHECK_EQ(right.arrays[0].f32[v], static_cast<float>(map[4 + v]));
    CHECK_EQ(meta_of(a, "nxr-neuro:unit"), std::string("V"));
    CHECK_EQ(meta_of(a, "Name"), std::string("map"));
    CHECK_EQ(*left.meta("AnatomicalStructurePrimary"), std::string("CortexLeft"));
    CHECK_EQ(*right.meta("AnatomicalStructurePrimary"), std::string("CortexRight"));
    CHECK_EQ(*left.meta("nxr-neuro:version"), std::string("9.9.9"));
    CHECK(left.meta("nxr-neuro:command")->find("export gifti") != std::string::npos);
    const Json origin = Json::parse(*left.meta("nxr-neuro:origin"));
    CHECK_EQ(origin["licence"].get<std::string>(), std::string("CC-BY-SA-4.0"));   // ShareAlike stays ShareAlike
    CHECK(origin["share_alike"].get<bool>());
    CHECK(origin["redistributable"].get<bool>());
    CHECK_EQ(*left.meta("nxr-neuro:attribution"), std::string("A. Author; B. Builder"));
    // the surface of the right hemisphere: its own vertices and triangles, indices from 0
    const gifti::Image surf = gifti::read_file((root / "map_hemi-R.surf.gii").string());
    const gifti::DataArray* p = named(surf, "NIFTI_INTENT_POINTSET");
    const gifti::DataArray* t = named(surf, "NIFTI_INTENT_TRIANGLE");
    CHECK(p && t);
    CHECK_EQ(p->dims, std::vector<std::int64_t>({4, 3}));
    CHECK_EQ(t->dims, std::vector<std::int64_t>({4, 3}));
    for (int i = 0; i < 12; ++i) CHECK_EQ(p->f32[i], static_cast<float>(xyz[12 + i]));
    for (const auto idx : t->i32) CHECK(idx >= 0 && idx < 4);
    // the sidecar: producer, options, inputs with the digests of their arrays, terms, the files with their hashes, no time range
    std::ifstream sf(r["sidecar"].get<std::string>());
    const Json side = Json::parse(sf);
    CHECK_EQ(side["producer"]["version"].get<std::string>(), std::string("9.9.9"));
    CHECK_EQ(side["schema"].get<std::string>(), std::string("nxr-export/1"));
    CHECK(side["span"].is_null());
    CHECK(side["inputs"][0]["arrays"]["keys"].get<int>() > 0);
    CHECK(!side["inputs"][0]["arrays"]["sha256"].is_null());
    CHECK_EQ(side["terms"]["licence"].get<std::string>(), std::string("CC-BY-SA-4.0"));
    CHECK_EQ(side["outputs"].size(), static_cast<std::size_t>(4));
    for (const auto& f : side["outputs"]) CHECK_EQ(f["sha256"].get<std::string>(), ex::file_sha256((root / f["file"].get<std::string>()).string()));
    CHECK(side["details"]["converted"][0].get<std::string>().find("float64") != std::string::npos);
    // the same command again writes the same files
    const std::string h1 = ex::file_sha256((root / "map_hemi-L.shape.gii").string());
    ex::export_gifti(pkg, o);
    CHECK_EQ(ex::file_sha256((root / "map_hemi-L.shape.gii").string()), h1);
    if (golden) {
      fs::create_directories(golden);
      for (const char* n : {"map_hemi-L.shape.gii", "map_hemi-R.shape.gii", "map_hemi-R.surf.gii"}) fs::copy_file(root / n, fs::path(golden) / n, fs::copy_options::overwrite_existing);
    }
  }

  // ---- a map on the whole surface, as .func.gii, with a unit conversion
  {
    ex::GiftiExport o;
    o.field = "map"; o.out = (root / "whole").string(); o.surface = true; o.ctx = ctx_of("whole");
    const Json r = ex::export_gifti(pkg, o);
    const gifti::Image f = gifti::read_file((root / "whole.func.gii").string());
    CHECK_EQ(f.arrays[0].intent, std::string("NIFTI_INTENT_NONE"));
    CHECK_EQ(f.arrays[0].dims, std::vector<std::int64_t>({8}));
    CHECK(f.meta("AnatomicalStructurePrimary") == nullptr);
    const gifti::Image s = gifti::read_file((root / "whole.surf.gii").string());
    CHECK_EQ(named(s, "NIFTI_INTENT_TRIANGLE")->dims, std::vector<std::int64_t>({8, 3}));
    const Geometry g = [&] { const SubjectPackage p = SubjectPackage::open(pkg); const Subject t = Subject::of(p); return t.geometry(cortex_id); }();
    o.out = (root / "mm").string();
    o.surface_unit = "mm";
    if (g.coordinate_unit && (*g.coordinate_unit == "m" || *g.coordinate_unit == "mm")) {
      ex::export_gifti(pkg, o);
      const gifti::Image smm = gifti::read_file((root / "mm.surf.gii").string());
      const double scale = *g.coordinate_unit == "m" ? 1000.0 : 1.0;
      CHECK_EQ(named(smm, "NIFTI_INTENT_POINTSET")->f32[3], static_cast<float>(xyz[3] * scale));
    } else {
      CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));
    }
    (void)r;
  }

  // ---- a series: the range is required, and read back frame by frame
  {
    ex::GiftiExport o;
    o.field = "series"; o.out = (root / "series").string(); o.ctx = ctx_of("series");
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));            // no range, no default
    o.range = ex::SeriesRange{false, 1.5, 3.0};                                                           // samples 1, 2, 3
    o.hemispheres = true;
    const Json r = ex::export_gifti(pkg, o);
    CHECK_EQ(r["frames"].get<int>(), 3);
    const gifti::Image right = gifti::read_file((root / "series_hemi-R.func.gii").string());
    CHECK_EQ(right.arrays.size(), static_cast<std::size_t>(3));
    for (int j = 0; j < 3; ++j) {
      const gifti::DataArray& a = right.arrays[j];
      CHECK_EQ(a.intent, std::string("NIFTI_INTENT_TIME_SERIES"));
      for (int v = 0; v < 4; ++v) CHECK_EQ(a.f32[v], static_cast<float>(series_value(4 + v, 1 + j)));
      CHECK(std::abs(std::stod(meta_of(a, "nxr-neuro:time")) - (1.0 + 0.5 * (1 + j))) < 1e-9);
      CHECK_EQ(meta_of(a, "nxr-neuro:frame"), std::to_string(1 + j));
    }
    CHECK(std::abs(std::stod(*right.meta("TimeStep")) - 0.5) < 1e-12);
    std::ifstream sf(r["sidecar"].get<std::string>());
    const Json side = Json::parse(sf);
    CHECK(!side["span"].is_null());                                                                       // the range read is stated
    CHECK(side["span"].is_object());
    // the whole recording
    o.hemispheres = false; o.range = ex::SeriesRange{true, 0, 0}; o.out = (root / "series_all").string();
    const Json ra = ex::export_gifti(pkg, o);
    CHECK_EQ(ra["frames"].get<int>(), T);
    const gifti::Image all = gifti::read_file((root / "series_all.func.gii").string());
    CHECK_EQ(all.arrays.size(), static_cast<std::size_t>(T));
    CHECK_EQ(all.arrays[T - 1].f32[7], static_cast<float>(series_value(7, T - 1)));
    CHECK(all.meta("AnatomicalStructurePrimary") == nullptr);
    if (golden) {
      fs::copy_file(root / "series_hemi-R.func.gii", fs::path(golden) / "series_hemi-R.func.gii", fs::copy_options::overwrite_existing);
      fs::copy_file(root / "series_all.func.gii", fs::path(golden) / "series_all.func.gii", fs::copy_options::overwrite_existing);
    }
    // a time range on a map, and one that holds no sample
    ex::GiftiExport m;
    m.field = "map"; m.out = (root / "m2").string(); m.range = ex::SeriesRange{true, 0, 0}; m.ctx = ctx_of("m2");
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, m); }), std::string(code::invalid_argument));
    o.range = ex::SeriesRange{false, 3.0, 2.0};
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));
  }

  // ---- labellings: with a dictionary (names, colours); with codes of -1 and no dictionary
  {
    ex::GiftiExport o;
    o.labelling = "demo-atlas"; o.out = (root / "atlas").string(); o.ctx = ctx_of("atlas");
    ex::export_gifti(pkg, o);
    const gifti::Image img = gifti::read_file((root / "atlas.label.gii").string());
    CHECK_EQ(img.arrays.size(), static_cast<std::size_t>(1));
    CHECK_EQ(img.arrays[0].intent, std::string("NIFTI_INTENT_LABEL"));
    CHECK_EQ(img.arrays[0].i32, std::vector<std::int32_t>({0, 1, 1, 1, 2, 2, 2, 0}));
    CHECK_EQ(img.labels.size(), static_cast<std::size_t>(3));
    bool alpha = false;
    for (const auto& l : img.labels) if (l.name == "alpha") { alpha = true; CHECK(l.has_colour); CHECK_EQ(l.rgba[0], 1.0f); CHECK_EQ(l.rgba[2], 128.0f / 255.0f); }
    CHECK(alpha);
    o.hemispheres = true;
    ex::export_gifti(pkg, o);
    CHECK_EQ(gifti::read_file((root / "atlas_hemi-R.label.gii").string()).arrays[0].i32, std::vector<std::int32_t>({2, 2, 2, 0}));
    if (golden) fs::copy_file(root / "atlas.label.gii", fs::path(golden) / "atlas.label.gii", fs::copy_options::overwrite_existing);

    ex::GiftiExport h;
    h.labelling = "halves"; h.out = (root / "halves").string(); h.ctx = ctx_of("halves");
    const Json r = ex::export_gifti(pkg, h);
    const gifti::Image hi = gifti::read_file((root / "halves.label.gii").string());
    CHECK_EQ(hi.arrays[0].i32, std::vector<std::int32_t>({0, 3, 3, 3, 4, 4, 4, 4}));   // -1 is key 0, "???"
    CHECK(r["details"]["unlabelled_as_key_0"].get<bool>());
    bool unknown = false;
    for (const auto& l : hi.labels) if (l.key == 0 && l.name == "???") unknown = true;
    CHECK(unknown);
    ex::GiftiExport z;
    z.labelling = "uses_zero"; z.out = (root / "zero").string(); z.ctx = ctx_of("zero");
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, z); }), std::string(code::invalid_argument));   // -1 with code 0 in use: no free key
  }

  // ---- terms: no origin gives "unknown"; two different ShareAlike licences are refused
  {
    ex::GiftiExport o;
    o.field = "series"; o.out = (root / "terms").string(); o.range = ex::SeriesRange{true, 0, 0}; o.ctx = ctx_of("terms");
    const Json r = ex::export_gifti(pkg, o);
    CHECK_EQ(r["terms"]["licence"].get<std::string>(), std::string("CC0-1.0 + unknown"));  // the surface states one (not ShareAlike), the series none: both are listed
    CHECK(!r["terms"]["redistributable"].get<bool>());                                      // the series has no record: not redistributable
    {
      DatasetWriter w = open_package_writer(pkg);
      SubjectWriter sub = SubjectWriter::open(w, "S");
      set_origin(w.db(), "manifold", cortex_id, OriginRecord{"fixture:surface", "ODbL-1.0", "B. Builder", true});
    }
    o.field = "map"; o.out = (root / "terms2").string(); o.range.reset();
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));   // CC-BY-SA with ODbL
    CHECK(ex::is_share_alike("CC-BY-SA-4.0") && ex::is_share_alike("ODbL-1.0") && !ex::is_share_alike("CC-BY-4.0") && !ex::is_share_alike("CC0-1.0"));
  }

  // ---- refusals
  {
    ex::GiftiExport o;
    o.out = (root / "x").string();
    o.ctx = ctx_of("x");
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));          // neither a field nor a labelling
    o.field = "nope";
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::no_such_row));
    o.field = "current";
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));          // a vector field
    o.field = "map"; o.kind = "other";
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));
    o.kind = "func"; o.hemispheres = true; o.surface_unit = "km";
    CHECK_EQ(code_of([&] { ex::export_gifti(pkg, o); }), std::string(code::invalid_argument));
  }
  fs::remove_all(root);
  return nxrtest::finish("export_gifti");
}
