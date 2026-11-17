// NIfTI-1/2 against goldens made with nibabel (test/golden/gen_nifti.py), the writer's round trips, and the refusals.
#include <cmath>
#include <fstream>
#include <cstdio>
#include <functional>

#include "check.h"
#include "nlohmann/json.hpp"
#include "nxr/formats/nifti.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";

static bool throws_code(const std::function<void()>& f, const char* code) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code() == code; }
  return false;
}

static bool near(double a, double b, double rel = 2e-6) { return std::fabs(a - b) <= rel * std::fabs(b) + 1e-9; }

int main() {
  std::ifstream gf(std::string(NXR_FORMATS_TEST_DIR) + "/golden/nifti_golden.json");
  const json g = json::parse(gf);
  for (const auto& [name, c] : g["cases"].items()) {
    const nifti::Volume v = nifti::read_file(kData + c["file"].get<std::string>());
    const nifti::Header& h = v.header;
    CHECK_EQ(h.version, c["version"].get<int>());
    CHECK_EQ(static_cast<int>(h.datatype), c["datatype"].get<int>());
    CHECK_EQ(h.shape.size(), c["shape"].size());
    for (std::size_t i = 0; i < h.shape.size() && i < c["shape"].size(); ++i) CHECK_EQ(h.shape[i], c["shape"][i].get<std::int64_t>());
    for (std::size_t i = 0; i < h.pixdim.size() && i < c["zooms"].size(); ++i) CHECK(near(h.pixdim[i], c["zooms"][i].get<double>()));
    CHECK_EQ(h.qform_code, c["qform_code"].get<int>());
    CHECK_EQ(h.sform_code, c["sform_code"].get<int>());
    CHECK_EQ(h.xyzt_units, c["xyzt_units"].get<int>());
    CHECK_EQ(h.descrip, c["descrip"].get<std::string>());
    if (c["scl_slope"].is_number()) { CHECK(near(h.scl_slope, c["scl_slope"].get<double>())); CHECK(near(h.scl_inter, c["scl_inter"].get<double>())); }
    CHECK_EQ(h.little_endian, !c.value("big_endian", false));
    const auto aff = h.affine();
    for (int i = 0; i < 12; ++i) {
      const double want = c["best_affine"][i].get<double>();
      CHECK(std::fabs(aff[i] - want) < 1e-5 * (1.0 + std::fabs(want)));
    }
    if (c["qform_affine"].is_array() && h.sform_code == 0) CHECK(h.affine_source() == nifti::Header::AffineSource::qform);
    const auto& d = c["data"];
    CHECK_EQ(v.size(), d.size());
    bool all = v.size() == d.size();
    for (std::size_t i = 0; all && i < d.size(); ++i) all = near(v.at(i), d[i].get<double>());
    if (!all) std::printf("  data differ in case %s\n", name.c_str());
    CHECK(all);
    // precision follows the file
    const int dt = c["datatype"].get<int>();
    const bool wide = dt == 64 || dt == 8;
    CHECK(v.precision == (wide ? Precision::float64 : Precision::float32));
  }

  // the unit and the affine in metres
  {
    const nifti::Volume v = nifti::read_file(kData + "nifti_f32_sform.nii");
    CHECK_EQ(v.header.spatial_unit(), std::string("mm"));
    CHECK_EQ(v.header.time_unit(), std::string("s"));
    CHECK(near(v.header.unit_to_metres(), 1e-3));
  }

  // writer: read, write, read again: header and values equal; and the bytes of a float32 file are stable through a second write
  for (const char* f : {"nifti_f32_sform.nii", "nifti2_f64.nii", "nifti_i16_qform_scaled.nii", "nifti_u8_4d.nii.gz", "nifti_i32.nii"}) {
    const nifti::Volume a = nifti::read_file(kData + f);
    std::vector<double> vals(a.size());
    for (std::size_t i = 0; i < vals.size(); ++i) vals[i] = a.at(i);
    MemorySink out;
    nifti::write_volume(out, a.header, vals);
    MemorySource back(out.data());
    const nifti::Volume b = nifti::read_volume(back);
    CHECK_EQ(b.header.version, a.header.version);
    CHECK_EQ(b.header.datatype, a.header.datatype);
    CHECK(b.header.shape == a.header.shape);
    CHECK(b.header.affine() == a.header.affine());
    CHECK_EQ(b.header.descrip, a.header.descrip);
    CHECK_EQ(b.size(), a.size());
    bool same = b.size() == a.size();
    for (std::size_t i = 0; same && i < a.size(); ++i) same = near(b.at(i), a.at(i), 1e-6);
    CHECK(same);
    MemorySink out2;
    std::vector<double> v2(b.size());
    for (std::size_t i = 0; i < v2.size(); ++i) v2[i] = b.at(i);
    nifti::write_volume(out2, b.header, v2);
    CHECK(out.data() == out2.data());   // stable bytes
  }
  // a header made here, with an extension, round trips, as NIfTI-1 and NIfTI-2, float32 input
  for (int ver : {1, 2}) {
    nifti::Header h = nifti::make_header(ver, {3, 2}, {1.0, 2.0}, {1, 0, 0, 5, 0, 2, 0, 6, 0, 0, 1, 7});
    h.descrip = "made here";
    nifti::Extension e;
    e.code = 6;
    e.bytes = {'h', 'i'};
    h.extensions.push_back(e);
    const std::vector<float> vals = {1, 2, 3, 4, 5, 6.5f};
    MemorySink out;
    nifti::write_volume(out, h, vals);
    MemorySource src(out.data());
    const nifti::Volume v = nifti::read_volume(src);
    CHECK(v.f32 == vals);
    CHECK_EQ(v.header.extensions.size(), 1u);
    if (!v.header.extensions.empty()) { CHECK_EQ(v.header.extensions[0].code, 6); CHECK_EQ(v.header.extensions[0].bytes[0], 'h'); }
    CHECK_EQ(v.header.affine()[3], 5.0);
    { FileSink keep(std::string(NXR_FORMATS_WORK_DIR) + "/written_v" + std::to_string(ver) + ".nii"); keep.write(out.data()); keep.close(); }   // for a by-hand read with nibabel
  }
  // integers out of the type's range are refused by the writer
  {
    nifti::Header h = nifti::make_header(1, {2}, {1.0}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0});
    h.datatype = 4; h.bitpix = 16;
    MemorySink out;
    CHECK(throws_code([&] { nifti::write_volume(out, h, std::vector<double>{1.0, 40000.0}); }, code::out_of_range));
  }
  // refusals
  CHECK(throws_code([&] { nifti::read_file(kData + "nifti_complex64.nii"); }, code::refused));
  CHECK(throws_code([&] { nifti::read_file(kData + "nifti_truncated.nii"); }, code::format));
  CHECK(throws_code([&] { nifti::read_file(kData + "not_nifti.nii"); }, code::format));
  CHECK(throws_code([&] { nifti::read_file(kData + "missing.nii"); }, code::io));
  CHECK(throws_code([&] { MemorySource m(std::string("tiny")); nifti::read_header(m); }, code::format));
  return nxrtest::finish("formats_nifti");
}
