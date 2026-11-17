// GIFTI against goldens made with nibabel (test/golden/gen_gifti.py), the writer's round trips, column-major arrays, and the refusals.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>

#include "check.h"
#include "nlohmann/json.hpp"
#include "nxr/formats/errors.h"
#include "nxr/formats/gifti.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";

static bool throws_code(const std::function<void()>& f, const char* code) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code() == code; }
  return false;
}

static gifti::Image from_text(const std::string& s) {
  MemorySource m(s);
  return gifti::read(m);
}

static bool same_image(const gifti::Image& a, const gifti::Image& b) {
  if (a.arrays.size() != b.arrays.size() || a.metadata != b.metadata || a.labels.size() != b.labels.size()) return false;
  for (std::size_t i = 0; i < a.arrays.size(); ++i) {
    const auto &x = a.arrays[i], &y = b.arrays[i];
    if (x.intent != y.intent || x.datatype != y.datatype || x.dims != y.dims || x.f32 != y.f32 || x.i32 != y.i32 || x.u8 != y.u8 ||
        x.metadata != y.metadata || x.transforms.size() != y.transforms.size())
      return false;
    for (std::size_t t = 0; t < x.transforms.size(); ++t)
      if (x.transforms[t].matrix != y.transforms[t].matrix || x.transforms[t].dataspace != y.transforms[t].dataspace) return false;
  }
  for (std::size_t i = 0; i < a.labels.size(); ++i)
    if (a.labels[i].key != b.labels[i].key || a.labels[i].name != b.labels[i].name || a.labels[i].rgba != b.labels[i].rgba) return false;
  return true;
}

int main() {
  std::ifstream gf(std::string(NXR_FORMATS_TEST_DIR) + "/golden/gifti_golden.json");
  const json g = json::parse(gf);
  for (const auto& [name, c] : g["cases"].items()) {
    const gifti::Image img = gifti::read_file(kData + c["file"].get<std::string>());
    CHECK_EQ(img.arrays.size(), c["arrays"].size());
    for (const auto& [k, v] : c["metadata"].items()) {
      const std::string* m = img.meta(k);
      CHECK(m != nullptr && *m == v.get<std::string>());
    }
    for (std::size_t i = 0; i < img.arrays.size() && i < c["arrays"].size(); ++i) {
      const gifti::DataArray& a = img.arrays[i];
      const json& w = c["arrays"][i];
      CHECK_EQ(a.intent, w["intent"].get<std::string>());
      CHECK_EQ(a.datatype, w["datatype"].get<std::string>());
      CHECK_EQ(a.big_endian_in_file, w["endian"].get<std::string>() == "BigEndian");
      CHECK(static_cast<int>(a.encoding) == (w["encoding"] == "ASCII" ? 0 : w["encoding"] == "Base64Binary" ? 1 : 2));
      CHECK_EQ(a.dims.size(), w["dims"].size());
      for (std::size_t d = 0; d < a.dims.size() && d < w["dims"].size(); ++d) CHECK_EQ(a.dims[d], w["dims"][d].get<std::int64_t>());
      CHECK_EQ(a.size(), w["data"].size());
      bool all = a.size() == w["data"].size();
      for (std::size_t k = 0; all && k < a.size(); ++k) all = std::fabs(a.at(k) - w["data"][k].get<double>()) <= 1e-6 * std::fabs(w["data"][k].get<double>()) + 1e-9;
      if (!all) std::printf("  data differ in %s array %zu\n", name.c_str(), i);
      CHECK(all);
      for (const auto& [k, v] : w["metadata"].items()) { const std::string* m = a.meta(k); CHECK(m != nullptr && *m == v.get<std::string>()); }
      if (w["coordsys"].is_object()) {
        CHECK_EQ(a.transforms.size(), 1u);
        if (!a.transforms.empty()) {
          CHECK_EQ(a.transforms[0].dataspace, w["coordsys"]["dataspace"].get<std::string>());
          CHECK_EQ(a.transforms[0].xformspace, w["coordsys"]["xformspace"].get<std::string>());
          for (int k = 0; k < 16; ++k) CHECK(std::fabs(a.transforms[0].matrix[k] - w["coordsys"]["xform"][k].get<double>()) < 1e-6);
        }
      }
    }
    CHECK_EQ(img.labels.size(), c["labels"].size());
    for (const gifti::Label& l : img.labels) {
      const json& w = c["labels"][std::to_string(l.key)];
      CHECK(w.is_object());
      if (w.is_object()) { CHECK_EQ(l.name, w["name"].get<std::string>()); CHECK(std::fabs(l.rgba[1] - w["rgba"][1].get<double>()) < 1e-6); }
    }

    // writer: every encoding reads back to the same image, and a second write is byte-identical
    for (gifti::Encoding e : {gifti::Encoding::ascii, gifti::Encoding::base64, gifti::Encoding::gzip_base64}) {
      gifti::Image copy = img;
      for (auto& a : copy.arrays) a.encoding = e;
      MemorySink out;
      gifti::write(out, copy);
      const gifti::Image back = from_text(out.str());
      CHECK(same_image(copy, back));
      MemorySink out2;
      gifti::write(out2, back);
      CHECK(out.data() == out2.data());
      if (name == "surf_b64" && e == gifti::Encoding::gzip_base64) { FileSink keep(std::string(NXR_FORMATS_WORK_DIR) + "/written_surf.gii"); keep.write(out.data()); keep.close(); }   // for a by-hand read with nibabel
    }
  }

  // a column-major 2 x 3 array: stored values 1..6 with the first index fastest are the matrix [[1,3,5],[2,4,6]] in row-major order 1 3 5 2 4 6
  {
    const std::string xml =
        "<?xml version=\"1.0\"?><GIFTI Version=\"1.0\" NumberOfDataArrays=\"1\"><DataArray Intent=\"NIFTI_INTENT_NONE\" DataType=\"NIFTI_TYPE_INT32\" "
        "ArrayIndexingOrder=\"ColumnMajorOrder\" Dimensionality=\"2\" Encoding=\"ASCII\" Endian=\"LittleEndian\" Dim0=\"2\" Dim1=\"3\">"
        "<Data>1 2 3 4 5 6</Data></DataArray></GIFTI>";
    const gifti::Image img = from_text(xml);
    CHECK((img.arrays[0].i32 == std::vector<std::int32_t>{1, 3, 5, 2, 4, 6}));
    CHECK(img.arrays[0].column_major_in_file);
  }
  // uint8 arrays
  {
    const std::string xml =
        "<GIFTI NumberOfDataArrays=\"1\"><DataArray Intent=\"NIFTI_INTENT_NONE\" DataType=\"NIFTI_TYPE_UINT8\" Dimensionality=\"1\" Encoding=\"ASCII\" Dim0=\"3\">"
        "<Data>0 7 255</Data></DataArray></GIFTI>";
    const gifti::Image img = from_text(xml);
    CHECK((img.arrays[0].u8 == std::vector<std::uint8_t>{0, 7, 255}));
    MemorySink out;
    gifti::write(out, img);
    CHECK(same_image(img, from_text(out.str())));
  }
  // builders and the size check
  {
    gifti::Image img;
    img.arrays.push_back(gifti::make_float32("NIFTI_INTENT_POINTSET", {2, 3}, {0, 1, 2, 3, 4, 5}));
    MemorySink ok;
    gifti::write(ok, img);
    CHECK((from_text(ok.str()).arrays[0].f32 == std::vector<float>{0, 1, 2, 3, 4, 5}));
    img.arrays[0].dims = {2, 4};
    MemorySink bad;
    CHECK(throws_code([&] { gifti::write(bad, img); }, code::out_of_range));
  }
  // refusals and malformed input
  const std::string head = "<GIFTI NumberOfDataArrays=\"1\"><DataArray Intent=\"NIFTI_INTENT_NONE\" ";
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT32\" Dimensionality=\"1\" Encoding=\"ExternalFileBinary\" ExternalFileName=\"x.bin\" Dim0=\"2\"><Data/></DataArray></GIFTI>"); }, code::refused));
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT64\" Dimensionality=\"1\" Encoding=\"ASCII\" Dim0=\"2\"><Data>1 2</Data></DataArray></GIFTI>"); }, code::refused));
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT32\" Dimensionality=\"1\" Encoding=\"ASCII\" Dim0=\"3\"><Data>1 2</Data></DataArray></GIFTI>"); }, code::format));
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT32\" Dimensionality=\"1\" Encoding=\"ASCII\" Dim0=\"1\"><Data>1 2</Data></DataArray></GIFTI>"); }, code::format));
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT32\" Dimensionality=\"2\" Encoding=\"ASCII\" Dim0=\"2\"><Data>1 2</Data></DataArray></GIFTI>"); }, code::format));
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT32\" Dimensionality=\"1\" Encoding=\"Base64Binary\" Dim0=\"1\"><Data>!!!!</Data></DataArray></GIFTI>"); }, code::format));
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT32\" Dimensionality=\"1\" Encoding=\"Base64Binary\" Dim0=\"2\"><Data>AAAAAA==</Data></DataArray></GIFTI>"); }, code::format));
  CHECK(throws_code([&] { from_text(head + "DataType=\"NIFTI_TYPE_FLOAT32\" Dimensionality=\"1\" Encoding=\"GZipBase64Binary\" Dim0=\"1\"><Data>AAAAAA==</Data></DataArray></GIFTI>"); }, code::format));
  CHECK(throws_code([&] { from_text("<GIFTI NumberOfDataArrays=\"2\"></GIFTI>"); }, code::format));
  CHECK(throws_code([&] { from_text("<notgifti/>"); }, code::format));
  CHECK(throws_code([&] { from_text("<GIFTI><unclosed></GIFTI>"); }, code::format));
  return nxrtest::finish("formats_gifti");
}
