// export fif (nxr/exporters/fif.h): source estimates as -lh.stc and -rh.stc on a synthetic package (two tetrahedra as the hemispheres of a cortex): a map, a series over a stated
// range (one chunk and several), refusals. Each file is read back with the pure .stc reader. With NXR_EXPORT_GOLDEN_DIR set the files are kept there for
// io/test/parity/test_export_fif.py (MNE-Python reads them).
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/exporters/fif.h"
#include "nxr/formats/fif_objects.h"
#include "nxr/formats/byte_source.h"
#include "nxr/formats/stc.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
using namespace nxr::io::task;
namespace fs = std::filesystem;
namespace ex = nxr::exporters;
namespace stc = nxr::formats::stc;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}
constexpr int T = 40;                       // 2 Hz from t = 1 s
double series_value(int v, int t) { return v + 100.0 * t; }
ex::Context ctx_of() {
  ex::Context c;
  c.version = "9.9.9";
  c.command = "nxr-neuro export fif test";
  return c;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_export_fif_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path pkg = root / "S.nxr.zarr";
  const char* golden = std::getenv("NXR_EXPORT_GOLDEN_DIR");

  std::vector<double> xyz;
  for (int h = 0; h < 2; ++h)
    for (const auto& p : std::vector<std::array<double, 3>>{{0, 0, 0}, {0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}}) { xyz.push_back(p[0] + (h ? 0.1 : 0.0)); xyz.push_back(p[1]); xyz.push_back(p[2]); }
  std::vector<std::int32_t> faces;
  for (int h = 0; h < 2; ++h)
    for (const auto& f : std::vector<std::array<int, 3>>{{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}) for (int c = 0; c < 3; ++c) faces.push_back(4 * h + f[c]);
  std::vector<float> map(8);
  for (int v = 0; v < 8; ++v) map[v] = 0.5f * (v + 1);

  std::string cortex_id, product_id;
  {
    CreatePackageOptions po;
    po.home_dataset = "demo";
    DatasetWriter w = create_package(pkg, "S", po);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    SurfaceOptions so;
    so.name = "cortex";
    so.hemispheres = {{"left", 0, 4}, {"right", 4, 4}};
    const SurfaceInfo si = add_surface(sub, so, xyz, faces);
    cortex_id = si.id;
    FieldSpec fm;
    fm.name = "map"; fm.path = "source/map"; fm.kind = "map"; fm.manifold_id = si.id; fm.data = view(map, {8});
    fm.cols.set("unit", "A/m");
    const std::string fid = sub.field(fm);
    set_origin(w.db(), "field", fid, OriginRecord{"fixture:map", "CC-BY-SA-4.0", "A. Author", true});
    std::vector<double> vec(24, 1.0);
    FieldSpec fv;
    fv.name = "current"; fv.path = "source/current"; fv.kind = "map"; fv.manifold_id = si.id; fv.value_type = "vector3"; fv.data = view(vec, {8, 3});
    sub.field(fv);
    sub.complete();
  }
  {
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    Subject t = Subject::of(sub);
    Unit u = t.unit(By{"test", Json::object()});
    LineOut ln;
    ln.name = "t"; ln.n = T; ln.spacing = 0.5; ln.origin = 1.0; ln.unit = "s"; ln.dimension = "time"; ln.folder = "time"; ln.cs_key = "clock";
    const std::string line_id = u.line(ln);
    product_id = u.product(ProductOut{{cortex_id, line_id}, std::nullopt});
    FieldOut fo;
    fo.name = "series"; fo.output = "series"; fo.kind = "map"; fo.manifold_id = product_id; fo.unit = "A/m"; fo.data.shape = {8, T};
    for (int v = 0; v < 8; ++v) for (int k = 0; k < T; ++k) fo.data.f32.push_back(static_cast<float>(series_value(v, k)));
    u.field(fo);
    // a covariance statistic on 8 "channels" (the cortex's vertices stand in): their names labelling and the channel Selection the statistic's axes name
    SelectionOut nm;
    nm.name = "cortex_names"; nm.output = "cortex_names"; nm.type = "partition"; nm.manifold_id = cortex_id;
    SelectionOut chs = nm;
    chs.name = "chans"; chs.output = "chans";
    for (int v = 0; v < 8; ++v) { nm.codes.push_back(v); chs.codes.push_back(v); nm.entries.push_back("EEG" + std::to_string(v + 1)); chs.entries.push_back("c" + std::to_string(v)); }
    u.selection(nm);
    const std::string chans_id = u.selection(chs);
    StatisticOut cv;
    cv.name = "cov"; cv.output = "cov"; cv.measure = "covariance"; cv.method = "sample"; cv.unit = "V\xc2\xb2"; cv.dof = 99.0; cv.detached = true;
    cv.axes = Json::array({Json{{"name", "channel"}, {"selection_id", chans_id}, {"size", 8}}, Json{{"name", "channel"}, {"selection_id", chans_id}, {"size", 8}}});
    cv.data.shape = {8, 8};
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) cv.data.f32.push_back(i == j ? 2.0f + i : 0.25f);
    u.statistic(cv);
    u.commit();
  }
  {   // the same series stored in chunks of 16 samples: three chunks, the last partial
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    std::vector<float> data;
    for (int v = 0; v < 8; ++v) for (int k = 0; k < T; ++k) data.push_back(static_cast<float>(series_value(v, k)));
    FieldSpec fs2;
    fs2.name = "series_chunked"; fs2.path = "source/series_chunked"; fs2.kind = "map"; fs2.manifold_id = product_id; fs2.data = view(data, {8, T});
    fs2.array.chunks = Shape{8, 16};
    fs2.array.compress = false;
    sub.field(fs2);
    sub.complete();
  }
  CHECK(SubjectPackage::open(pkg).verify().ok);

  // ---- a map: one time point on each hemisphere
  {
    ex::StcExport o;
    o.field = "map"; o.out = (root / "map").string(); o.ctx = ctx_of();
    const Json r = ex::export_stc(pkg, o);
    CHECK_EQ(r["files"].size(), static_cast<std::size_t>(2));
    const stc::Estimate l = stc::read_file((root / "map-lh.stc").string()), rr = stc::read_file((root / "map-rh.stc").string());
    CHECK_EQ(l.n_times, 1u);
    CHECK_EQ(l.tmin, 0.0);
    CHECK_EQ(l.tstep, 1.0);
    CHECK_EQ(l.vertices, std::vector<std::uint32_t>({0, 1, 2, 3}));
    for (int v = 0; v < 4; ++v) { CHECK_EQ(l.at(v, 0), map[v]); CHECK_EQ(rr.at(v, 0), map[4 + v]); }
    std::ifstream sf(r["sidecar"].get<std::string>());
    const Json side = Json::parse(sf);
    CHECK_EQ(side["details"]["unit"].get<std::string>(), std::string("A/m"));
    CHECK_EQ(side["terms"]["licence"].get<std::string>(), std::string("CC-BY-SA-4.0"));
    CHECK(side["terms"]["share_alike"].get<bool>());
    CHECK_EQ(side["outputs"].size(), static_cast<std::size_t>(2));
    CHECK_EQ(side["outputs"][0]["sha256"].get<std::string>(), ex::file_sha256((root / "map-lh.stc").string()));
    CHECK(side["span"].is_null());
    if (golden) {
      fs::create_directories(golden);
      for (const char* n : {"map-lh.stc", "map-rh.stc"}) fs::copy_file(root / n, fs::path(golden) / n, fs::copy_options::overwrite_existing);
    }
  }

  // ---- series: the range is required; one chunk, and across chunk boundaries
  for (const char* name : {"series", "series_chunked"}) {
    ex::StcExport o;
    o.field = name; o.out = (root / name).string(); o.ctx = ctx_of();
    CHECK_EQ(code_of([&] { ex::export_stc(pkg, o); }), std::string(code::invalid_argument));            // no range, no default
    o.range = ex::SeriesRange{true, 0, 0};
    const Json r = ex::export_stc(pkg, o);
    CHECK_EQ(r["frames"].get<int>(), T);
    const stc::Estimate l = stc::read_file((root / (std::string(name) + "-lh.stc")).string()), rr = stc::read_file((root / (std::string(name) + "-rh.stc")).string());
    CHECK_EQ(l.n_times, static_cast<std::uint32_t>(T));
    CHECK_EQ(l.tmin, 1.0);
    CHECK_EQ(l.tstep, 0.5);
    for (int t = 0; t < T; ++t) for (int v = 0; v < 4; ++v) {
      CHECK_EQ(l.at(v, t), static_cast<float>(series_value(v, t)));
      CHECK_EQ(rr.at(v, t), static_cast<float>(series_value(4 + v, t)));
    }
    // samples 14 to 19 (seconds 8 to 11): inside chunk 0 and chunk 1 of the chunked field
    ex::StcExport p = o;
    p.out = (root / (std::string(name) + "_part")).string();
    p.range = ex::SeriesRange{false, 8.0, 11.0};
    const Json rp = ex::export_stc(pkg, p);
    CHECK_EQ(rp["frames"].get<int>(), 6);
    const stc::Estimate pl = stc::read_file(p.out + "-lh.stc");
    CHECK_EQ(pl.n_times, 6u);
    CHECK_EQ(pl.tmin, 8.0);
    for (int t = 0; t < 6; ++t) for (int v = 0; v < 4; ++v) CHECK_EQ(pl.at(v, t), static_cast<float>(series_value(v, 14 + t)));
    std::ifstream sf(rp["sidecar"].get<std::string>());
    CHECK(Json::parse(sf)["span"].is_object());
    if (golden && std::string(name) == "series_chunked") {
      for (const char* n : {"series_chunked-lh.stc", "series_chunked-rh.stc"}) fs::copy_file(root / n, fs::path(golden) / n, fs::copy_options::overwrite_existing);
    }
  }

  // ---- a covariance statistic: -cov.fif, read back with the pure reader (MNE reads it in the parity test)
  {
    ex::CovarianceExport o;
    o.statistic = "cov"; o.out = (root / "cov").string(); o.ctx = ctx_of();
    const Json r = ex::export_covariance(pkg, o);
    CHECK_EQ(r["channels"].get<int>(), 8);
    const nxr::formats::fif::File f(std::make_shared<nxr::formats::FileSource>((root / "cov-cov.fif").string()));
    const nxr::formats::fif::Covariance c = nxr::formats::fif::read_covariance(f);
    CHECK_EQ(c.dim, 8);
    CHECK_EQ(c.nfree, 99);
    CHECK_EQ(c.names[0], std::string("EEG1"));
    CHECK_EQ(c.names[7], std::string("EEG8"));
    CHECK_EQ(c.data[3 * 8 + 3], 5.0);
    CHECK_EQ(c.data[1 * 8 + 6], 0.25);
    CHECK_EQ(c.data[6 * 8 + 1], 0.25);
    std::ifstream sf(r["sidecar"].get<std::string>());
    CHECK_EQ(Json::parse(sf)["details"]["unit"].get<std::string>(), std::string("V\xc2\xb2"));
    if (golden) fs::copy_file(root / "cov-cov.fif", fs::path(golden) / "cov-cov.fif", fs::copy_options::overwrite_existing);
    o.statistic = "series";
    CHECK_EQ(code_of([&] { ex::export_covariance(pkg, o); }), std::string(code::no_such_row));          // not a statistic
    o.statistic = "";
    CHECK_EQ(code_of([&] { ex::export_covariance(pkg, o); }), std::string(code::invalid_argument));
  }

  // ---- refusals
  {
    ex::StcExport o;
    o.out = (root / "x").string();
    o.ctx = ctx_of();
    CHECK_EQ(code_of([&] { ex::export_stc(pkg, o); }), std::string(code::invalid_argument));            // no field
    o.field = "nope";
    CHECK_EQ(code_of([&] { ex::export_stc(pkg, o); }), std::string(code::no_such_row));
    o.field = "current";
    CHECK_EQ(code_of([&] { ex::export_stc(pkg, o); }), std::string(code::invalid_argument));            // not scalar
    o.field = "map"; o.range = ex::SeriesRange{true, 0, 0};
    CHECK_EQ(code_of([&] { ex::export_stc(pkg, o); }), std::string(code::invalid_argument));            // a map has no range
    o.field = "series"; o.range = ex::SeriesRange{false, 5.0, 4.0};
    CHECK_EQ(code_of([&] { ex::export_stc(pkg, o); }), std::string(code::invalid_argument));            // the range ends before it starts
  }
  fs::remove_all(root);
  return nxrtest::finish("export_fif");
}
