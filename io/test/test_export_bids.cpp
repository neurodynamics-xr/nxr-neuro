// export bids-derivatives (nxr/exporters/bids.h) on a synthetic package: names with BIDS entities, sidecars with Sources as BIDS URIs, dataset_description.json merged across runs,
// the licence rule, refusals. Files are read back with the pure readers.
#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/exporters/bids.h"
#include "nxr/formats/gifti.h"
#include "nxr/formats/stc.h"
#include "nxr/formats/tsv.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
using namespace nxr::io::task;
namespace fs = std::filesystem;
namespace ex = nxr::exporters;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}
Json read_json(const fs::path& p) { std::ifstream f(p); return Json::parse(f); }
ex::Context ctx_of() {
  ex::Context c;
  c.version = "9.9.9";
  c.command = "nxr-neuro export bids-derivatives test";
  return c;
}
constexpr int T = 6;
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_export_bids_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path pkg = root / "sub01.nxr.zarr";

  std::vector<double> xyz;
  for (int h = 0; h < 2; ++h)
    for (const auto& p : std::vector<std::array<double, 3>>{{0, 0, 0}, {0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}}) { xyz.push_back(p[0] + (h ? 0.1 : 0.0)); xyz.push_back(p[1]); xyz.push_back(p[2]); }
  std::vector<std::int32_t> faces;
  for (int h = 0; h < 2; ++h)
    for (const auto& f : std::vector<std::array<int, 3>>{{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {0, 3, 2}}) for (int c = 0; c < 3; ++c) faces.push_back(4 * h + f[c]);
  std::vector<float> map(8), other(8, 2.0f);
  for (int v = 0; v < 8; ++v) map[v] = static_cast<float>(v);

  std::string cortex_id, map_id;
  {
    CreatePackageOptions po;
    po.home_dataset = "demo";
    DatasetWriter w = create_package(pkg, "sub-01", po);
    SubjectWriter sub = SubjectWriter::open(w, "sub-01");
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
    FieldSpec fo;
    fo.name = "ma_p"; fo.path = "source/ma_p"; fo.kind = "map"; fo.manifold_id = si.id; fo.data = view(other, {8});
    sub.field(fo);
    set_origin(w.db(), "field", map_id, OriginRecord{"bids:sub-01/meg/sub-01_task-rest_meg.fif#sha256=abc", "CC-BY-SA-4.0", "A. Author", true});
    set_origin(w.db(), "manifold", si.id, OriginRecord{"fixture:surface", "CC0-1.0", "B. Builder", true});
    sub.complete();
  }
  {
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "sub-01");
    Subject t = Subject::of(sub);
    Unit u = t.unit(By{"test", Json::object()});
    LineOut ln;
    ln.name = "t"; ln.n = T; ln.spacing = 0.5; ln.origin = 1.0; ln.unit = "s"; ln.dimension = "time"; ln.folder = "time"; ln.cs_key = "clock";
    const std::string line_id = u.line(ln);
    const std::string prod = u.product(ProductOut{{cortex_id, line_id}, std::nullopt});
    FieldOut fo;
    fo.name = "series"; fo.output = "series"; fo.kind = "map"; fo.manifold_id = prod; fo.data.shape = {8, T};
    for (int v = 0; v < 8; ++v) for (int k = 0; k < T; ++k) fo.data.f32.push_back(static_cast<float>(v + 10 * k));
    u.field(fo);
    LineOut fl;
    fl.name = "freq"; fl.n = 3; fl.spacing = 1.0; fl.origin = 1.0;
    const std::string freq_id = u.line(fl);
    StatisticOut st;
    st.name = "psd"; st.output = "psd"; st.measure = "power_density"; st.method = "welch"; st.of_field_id = map_id;
    st.axes = Json::array({Json{{"name", "frequency"}, {"manifold_id", freq_id}, {"size", 3}, {"unit", "Hz"}}});
    st.data.shape = {3};
    st.data.f32 = {1.0f, 2.0f, 3.0f};
    u.statistic(st);
    u.commit();
  }

  const fs::path out = root / "derivatives" / "nxr-neuro";
  ex::BidsDerivativesExport o;
  o.root = out.string(); o.results = "map|demo-atlas|psd"; o.space = "fsaverage"; o.session = "01"; o.task = "rest"; o.run = "1"; o.with_surface = true; o.stc = true; o.ctx = ctx_of();
  const Json r = ex::export_bids_derivatives(pkg, o);
  const fs::path sd = out / "sub-sub01" / "ses-01";
  const std::string base = "sub-sub01_ses-01_task-rest_run-1_space-fsaverage";
  for (const char* n : {"_hemi-L_desc-map_map.func.gii", "_hemi-R_desc-map_map.func.gii", "_hemi-L_desc-map_map.surf.gii", "_hemi-L_desc-map_map.json", "_hemi-R_desc-map_map.json",
                        "_hemi-L_desc-demoatlas_dseg.label.gii", "_hemi-R_desc-demoatlas_dseg.json", "_desc-map_stc-lh.stc", "_desc-map_stc-rh.stc", "_desc-map_stc.json",
                        "_desc-psd_stat.tsv", "_desc-psd_stat.json"}) CHECK(fs::exists(sd / (base + n)));
  CHECK(!fs::exists(sd / (base + "_hemi-L_desc-map_map.func.gii.json")));                              // the exporter's own sidecar is replaced by the BIDS one
  // the map and the atlas read back
  const auto left = nxr::formats::gifti::read_file((sd / (base + "_hemi-L_desc-map_map.func.gii")).string());
  CHECK_EQ(left.arrays[0].f32, std::vector<float>({0, 1, 2, 3}));
  const auto stcr = nxr::formats::stc::read_file((sd / (base + "_desc-map_stc-rh.stc")).string());
  CHECK_EQ(stcr.at(0, 0), 4.0f);
  CHECK_EQ(nxr::formats::tsv::read_file((sd / (base + "_desc-psd_stat.tsv")).string()).rows.size(), static_cast<std::size_t>(3));
  // the sidecar: the provenance, and the BIDS URI of the input imported from a BIDS dataset
  const Json side = read_json(sd / (base + "_hemi-L_desc-map_map.json"));
  CHECK_EQ(side["Sources"][0].get<std::string>(), std::string("bids::sub-01/meg/sub-01_task-rest_meg.fif"));
  CHECK_EQ(side["GeneratedBy"][0]["producer"]["version"].get<std::string>(), std::string("9.9.9"));
  CHECK(!read_json(sd / (base + "_hemi-L_desc-demoatlas_dseg.json")).contains("Sources"));
  // dataset_description.json: the derivative, the generator, the sources, ShareAlike carried, the suffixes listed
  const Json desc = read_json(out / "dataset_description.json");
  CHECK_EQ(desc["DatasetType"].get<std::string>(), std::string("derivative"));
  CHECK_EQ(desc["GeneratedBy"][0]["Name"].get<std::string>(), std::string("nxr-neuro"));
  CHECK_EQ(desc["License"].get<std::string>(), std::string("CC-BY-SA-4.0"));
  CHECK_EQ(desc["SourceDatasets"][0]["URL"].get<std::string>(), std::string("bids::sub-01/meg/sub-01_task-rest_meg.fif"));
  for (const char* k : {"map", "series", "dseg", "stc", "surf", "stat", "measures"}) CHECK(desc["NxrNeuro"]["Suffixes"].contains(k));
  CHECK(!fs::exists(out / "participants.tsv"));
  CHECK_EQ(r["skipped"].size(), static_cast<std::size_t>(0));

  // a second run into the same folder: the description is merged, not duplicated
  ex::BidsDerivativesExport o2 = o;
  o2.results = "series"; o2.stc = false; o2.with_surface = false; o2.range = ex::SeriesRange{true, 0, 0};
  ex::export_bids_derivatives(pkg, o2);
  CHECK(fs::exists(sd / (base + "_hemi-L_desc-series_series.func.gii")));
  CHECK_EQ(nxr::formats::gifti::read_file((sd / (base + "_hemi-R_desc-series_series.func.gii")).string()).arrays.size(), static_cast<std::size_t>(T));
  CHECK_EQ(read_json(out / "dataset_description.json")["GeneratedBy"].size(), static_cast<std::size_t>(1));

  // refusals
  {
    ex::BidsDerivativesExport b = o;
    b.root = (root / "b").string(); b.results = "series"; b.range.reset();
    CHECK_EQ(code_of([&] { ex::export_bids_derivatives(pkg, b); }), std::string(code::invalid_argument));        // a series needs its range
    b.results = "map|ma_p"; b.range = o.range;
    CHECK_EQ(code_of([&] { ex::export_bids_derivatives(pkg, b); }), std::string(code::invalid_argument));        // two names, one desc label
    b.results = "nothing-like-this";
    CHECK_EQ(code_of([&] { ex::export_bids_derivatives(pkg, b); }), std::string(code::invalid_argument));
    b.results = "map"; b.space.reset();
    CHECK_EQ(code_of([&] { ex::export_bids_derivatives(pkg, b); }), std::string(code::invalid_argument));        // no space named, no template
    b.space = "x"; b.results = "(";
    CHECK_EQ(code_of([&] { ex::export_bids_derivatives(pkg, b); }), std::string(code::invalid_argument));        // not a regular expression
  }
  fs::remove_all(root);
  return nxrtest::finish("export_bids");
}
