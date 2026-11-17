// CIFTI-2 against goldens made by nibabel (test/golden/gen_cifti.py): dtseries, dscalar, dlabel, pscalar, ptseries and a float64 file: the XML
// mappings (series, named maps and label tables, brain models with their vertices and voxels, parcels, the volume), the matrix by block; the refusals.
#include <cmath>
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/cifti.h"
#include "nxr/formats/errors.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/cifti/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

static bool near(double a, double b, double rel = 1e-9, double abs_ = 1e-12) { return std::fabs(a - b) <= rel * std::fabs(b) + abs_; }

static void check_volume(const std::optional<cifti::VolumeSpace>& v, const json& shape, const json& affine) {
  if (shape.is_null()) { CHECK(!v); return; }
  CHECK(v.has_value());
  if (!v) return;
  for (int i = 0; i < 3; ++i) CHECK_EQ(v->dims[static_cast<std::size_t>(i)], shape[i].get<std::int64_t>());
  CHECK_EQ(v->meter_exponent, -3);
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) CHECK(near(v->ijk_to_xyz[static_cast<std::size_t>(r * 4 + c)], affine[r][c].get<double>(), 1e-9, 1e-12));
}

static void check_axis(const cifti::Mapping& m, const json& a) {
  const std::string kind = a["kind"].get<std::string>();
  if (kind == "series") {
    CHECK_EQ(m.type, std::string("CIFTI_INDEX_TYPE_SERIES"));
    CHECK(m.series.has_value());
    if (!m.series) return;
    CHECK_EQ(m.series->points, a["size"].get<std::uint64_t>());
    CHECK(near(m.series->start, a["start"].get<double>()));
    CHECK(near(m.series->step, a["step"].get<double>()));
    CHECK_EQ(m.series->unit, a["unit"].get<std::string>());
    CHECK_EQ(m.length(), a["size"].get<std::uint64_t>());
  } else if (kind == "scalars" || kind == "labels") {
    CHECK_EQ(m.type, std::string(kind == "scalars" ? "CIFTI_INDEX_TYPE_SCALARS" : "CIFTI_INDEX_TYPE_LABELS"));
    CHECK_EQ(m.named_maps.size(), a["names"].size());
    for (std::size_t i = 0; i < m.named_maps.size() && i < a["names"].size(); ++i) {
      const cifti::NamedMap& n = m.named_maps[i];
      CHECK_EQ(n.name, a["names"][i].get<std::string>());
      CHECK_EQ(n.metadata.size(), a["meta"][i].size());
      for (const auto& kv : n.metadata) CHECK(a["meta"][i].contains(kv.first) && a["meta"][i][kv.first].get<std::string>() == kv.second);
      if (kind == "labels") {
        CHECK_EQ(n.labels.size(), a["labels"][i].size());
        for (const auto& e : n.labels) {
          const std::string key = std::to_string(e.key);
          CHECK(a["labels"][i].contains(key));
          if (!a["labels"][i].contains(key)) continue;
          CHECK_EQ(e.name, a["labels"][i][key][0].get<std::string>());
          CHECK(near(e.red, a["labels"][i][key][1][0].get<double>(), 1e-6));
          CHECK(near(e.green, a["labels"][i][key][1][1].get<double>(), 1e-6));
          CHECK(near(e.blue, a["labels"][i][key][1][2].get<double>(), 1e-6));
          CHECK(near(e.alpha, a["labels"][i][key][1][3].get<double>(), 1e-6));
        }
      } else {
        CHECK(n.labels.empty());
      }
    }
  } else if (kind == "brain_models") {
    CHECK_EQ(m.type, std::string("CIFTI_INDEX_TYPE_BRAIN_MODELS"));
    CHECK_EQ(m.brain_models.size(), a["structures"].size());
    for (std::size_t i = 0; i < m.brain_models.size() && i < a["structures"].size(); ++i) {
      const cifti::BrainModel& b = m.brain_models[i];
      const json& w = a["structures"][i];
      CHECK_EQ(b.structure, w["name"].get<std::string>());
      CHECK_EQ(b.index_offset, w["start"].get<std::uint64_t>());
      CHECK_EQ(b.index_offset + b.index_count, w["stop"].get<std::uint64_t>());
      const bool surf = w["is_surface"].get<bool>();
      CHECK_EQ(b.model_type, std::string(surf ? "CIFTI_MODEL_TYPE_SURFACE" : "CIFTI_MODEL_TYPE_VOXELS"));
      if (surf) {
        CHECK_EQ(b.vertex_indices.size(), w["vertex"].size());
        for (std::size_t k = 0; k < b.vertex_indices.size() && k < w["vertex"].size(); ++k) CHECK_EQ(b.vertex_indices[k], w["vertex"][k].get<std::uint32_t>());
        CHECK(b.surface_vertex_count.has_value() && *b.surface_vertex_count == w["nvertices"].get<std::uint64_t>());
      } else {
        CHECK_EQ(b.voxel_ijk.size(), w["voxel"].size());
        for (std::size_t k = 0; k < b.voxel_ijk.size() && k < w["voxel"].size(); ++k)
          for (int c = 0; c < 3; ++c) CHECK_EQ(b.voxel_ijk[k][static_cast<std::size_t>(c)], w["voxel"][k][c].get<std::int32_t>());
      }
    }
    check_volume(m.volume, a["volume_shape"], a["affine"]);
  } else if (kind == "parcels") {
    CHECK_EQ(m.type, std::string("CIFTI_INDEX_TYPE_PARCELS"));
    CHECK_EQ(m.parcels.size(), a["names"].size());
    for (std::size_t i = 0; i < m.parcels.size() && i < a["names"].size(); ++i) {
      const cifti::Parcel& p = m.parcels[i];
      CHECK_EQ(p.name, a["names"][i].get<std::string>());
      CHECK_EQ(p.vertices.size(), a["vertices"][i].size());
      for (const auto& pv : p.vertices) {
        CHECK(a["vertices"][i].contains(pv.structure));
        if (!a["vertices"][i].contains(pv.structure)) continue;
        CHECK_EQ(pv.vertices.size(), a["vertices"][i][pv.structure].size());
        for (std::size_t k = 0; k < pv.vertices.size() && k < a["vertices"][i][pv.structure].size(); ++k) CHECK_EQ(pv.vertices[k], a["vertices"][i][pv.structure][k].get<std::uint32_t>());
      }
      CHECK_EQ(p.voxel_ijk.size(), a["voxels"][i].size());
      for (std::size_t k = 0; k < p.voxel_ijk.size() && k < a["voxels"][i].size(); ++k)
        for (int c = 0; c < 3; ++c) CHECK_EQ(p.voxel_ijk[k][static_cast<std::size_t>(c)], a["voxels"][i][k][c].get<std::int32_t>());
    }
    CHECK_EQ(m.surfaces.size(), a["nvertices"].size());
    for (const auto& s : m.surfaces) CHECK(a["nvertices"].contains(s.structure) && a["nvertices"][s.structure].get<std::uint64_t>() == s.vertex_count);
    check_volume(m.volume, a["volume_shape"], a["affine"]);
  }
}

int main() {
  std::ifstream gf(kGold + "cifti_golden.json");
  const json G = json::parse(gf);
  for (auto it = G["cases"].begin(); it != G["cases"].end(); ++it) {
    const json& g = it.value();
    std::printf("  case %s\n", it.key().c_str());
    auto r = cifti::open_file(kData + g["file"].get<std::string>());
    CHECK_EQ(r->rows(), g["shape"][0].get<std::uint64_t>());
    CHECK_EQ(r->cols(), g["shape"][1].get<std::uint64_t>());
    CHECK_EQ(r->nifti_header().version, 2);
    CHECK_EQ(r->nifti_header().intent_code, g["intent_code"].get<int>());
    CHECK_EQ(r->version(), std::string("2.0"));
    CHECK_EQ(r->mappings().size(), std::size_t(2));
    check_axis(r->map(0), g["axes"][0]);
    check_axis(r->map(1), g["axes"][1]);
    CHECK(r->xml().rfind("<CIFTI", 0) == 0);
    // the matrix, whole and by blocks of rows and columns
    const bool f64 = g["float64"].get<bool>();
    CHECK(r->precision() == (f64 ? Precision::float64 : Precision::float32));
    std::ifstream f(kGold + "cifti_" + it.key() + ".f32", std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::uint64_t rows = r->rows(), cols = r->cols();
    std::vector<double> want(rows * cols);
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (f64) { double v; std::memcpy(&v, &raw[i * 8], 8); want[i] = v; }
      else { float v; std::memcpy(&v, &raw[i * 4], 4); want[i] = v; }
    }
    std::vector<double> whole(rows * cols), part;
    r->read(0, rows, 0, cols, whole.data());
    bool ok = true;
    for (std::size_t i = 0; i < whole.size(); ++i) ok = ok && whole[i] == want[i];
    CHECK(ok);
    // a block in the middle: rows [1, rows), columns [2, cols - 1) (the last two columns for a narrow matrix)
    const std::uint64_t r0 = 1, n = rows - 1, c0 = cols > 3 ? 2 : 1, c1 = cols > 3 ? cols - 1 : cols;
    part.assign(n * (c1 - c0), -1.0);
    r->read(r0, n, c0, c1, part.data());
    ok = true;
    for (std::uint64_t i = 0; i < n; ++i) for (std::uint64_t k = c0; k < c1; ++k) ok = ok && part[i * (c1 - c0) + (k - c0)] == want[(r0 + i) * cols + k];
    CHECK(ok);
    if (!f64) {
      std::vector<float> fl(rows * cols);
      r->read(0, rows, 0, cols, fl.data());
      ok = true;
      for (std::size_t i = 0; i < fl.size(); ++i) ok = ok && static_cast<double>(fl[i]) == want[i];
      CHECK(ok);
    } else {
      std::vector<float> fl(4);
      CHECK(rg::throws_code([&] { r->read(0, 1, 0, 4, fl.data()); }, code::refused));
    }
    CHECK(rg::throws_code([&] { std::vector<double> x(4); r->read(rows, 1, 0, 1, x.data()); }, code::out_of_range));
    CHECK(rg::throws_code([&] { std::vector<double> x(4); r->read(0, 1, 0, cols + 1, x.data()); }, code::out_of_range));
  }
  auto refusal = [&](const char* file, const char* code_, const char* needle) {
    try {
      cifti::open_file(kData + file);
      std::printf("  %s was not refused\n", file);
      CHECK(false);
    } catch (const nxr::io::Error& e) {
      CHECK(e.code() == code_);
      const bool found = std::string(e.what()).find(needle) != std::string::npos;
      if (!found) std::printf("  %s: message '%s' lacks '%s'\n", file, e.what(), needle);
      CHECK(found);
    }
  };
  refusal("truncated.nii", code::format, "truncated");
  refusal("nifti1.nii", code::refused, "NIfTI-1");
  refusal("nifti2_plain.nii", code::format, "no extension of code 32");
  refusal("cifti1.nii", code::refused, "version '1.0'");
  refusal("int16.nii", code::refused, "datatype 4");
  refusal("missing.nii", code::io, "");
  refusal("whatever.nii.gz", code::refused, "gzipped");
  return nxrtest::finish("formats_cifti");
}
