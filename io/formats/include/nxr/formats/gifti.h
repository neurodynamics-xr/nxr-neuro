// GIFTI (surfaces, shape and functional maps, label maps): read and write, as plain objects. XML through pugixml.
//
// Arrays come out in row-major order whatever the file's `ArrayIndexingOrder`, as the values of their datatype (float32, int32, uint8: the
// three GIFTI defines), decoded from ASCII, Base64Binary or GZipBase64Binary in either byte order. ExternalFileBinary arrays are refused
// (the reader sees one file). The writer writes ASCII, Base64Binary or GZipBase64Binary, little-endian, row-major.
// Names (intent, datatype, spaces) are the file's own strings, e.g. "NIFTI_INTENT_POINTSET", "NIFTI_TYPE_FLOAT32".
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "nxr/formats/byte_source.h"

namespace nxr::formats::gifti {

enum class Encoding { ascii, base64, gzip_base64 };

struct Transform {
  std::string dataspace = "NIFTI_XFORM_UNKNOWN";
  std::string xformspace = "NIFTI_XFORM_UNKNOWN";
  std::array<double, 16> matrix{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};   // 4 x 4, row-major
};

struct DataArray {
  std::string intent;                      // "NIFTI_INTENT_POINTSET", "_TRIANGLE", "_SHAPE", "_LABEL", "_NONE", ...
  std::string datatype = "NIFTI_TYPE_FLOAT32";   // "NIFTI_TYPE_FLOAT32", "NIFTI_TYPE_INT32" or "NIFTI_TYPE_UINT8"
  std::vector<std::int64_t> dims;          // Dim0, Dim1, ...
  Encoding encoding = Encoding::base64;    // as read; what the writer uses
  bool column_major_in_file = false;       // the file's ArrayIndexingOrder (the values below are row-major either way)
  bool big_endian_in_file = false;
  std::vector<std::pair<std::string, std::string>> metadata;
  std::vector<Transform> transforms;
  // The values, one of these three according to `datatype`, in row-major order.
  std::vector<float> f32;
  std::vector<std::int32_t> i32;
  std::vector<std::uint8_t> u8;

  std::uint64_t count() const;             // product of dims
  std::size_t size() const { return f32.size() + i32.size() + u8.size(); }
  double at(std::size_t i) const;          // value i as a double
  const std::string* meta(const std::string& name) const;   // a metadata value, or nullptr
};

struct Label {
  std::int32_t key = 0;
  std::string name;
  std::array<float, 4> rgba{0, 0, 0, 0};   // red, green, blue, alpha, 0 to 1 (absent in the file: 0 0 0 0, has_colour false)
  bool has_colour = false;
};

struct Image {
  std::string version = "1.0";
  std::vector<std::pair<std::string, std::string>> metadata;
  std::vector<Label> labels;
  std::vector<DataArray> arrays;
  const std::string* meta(const std::string& name) const;
};

Image read(const ByteSource& src);
Image read_file(const std::string& path);

void write(ByteSink& sink, const Image& image);
void write_file(const std::string& path, const Image& image);

// Builders for the common arrays (the writer checks dims against the values).
DataArray make_float32(const std::string& intent, std::vector<std::int64_t> dims, std::vector<float> values);
DataArray make_int32(const std::string& intent, std::vector<std::int64_t> dims, std::vector<std::int32_t> values);

}  // namespace nxr::formats::gifti
