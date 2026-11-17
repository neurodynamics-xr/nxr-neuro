// NIfTI-1 and NIfTI-2 (single file `.nii`, and the `.hdr` + `.img` pair): the header, the volume, and a writer. Plain objects only.
//
// The header is parsed whole (every field the format has). Volumes are small and read whole. Data come out as the file stores them (x fastest),
// scaled by scl_slope and scl_inter when the file sets them. Precision follows the file: 8- and 16-bit integers and float32 read as 32-bit
// floats, 32-bit and 64-bit integers and float64 as 64-bit floats. Complex, RGB and 128-bit types are refused.
// Coordinates are in the file's own spatial unit (`spatial_unit`, millimetres by convention); `unit_to_metres` converts.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/recording.h"   // Precision

namespace nxr::formats::nifti {

struct Extension {
  std::int32_t code = 0;
  std::vector<unsigned char> bytes;
};

struct Header {
  int version = 1;                       // 1 or 2
  bool little_endian = true;             // the file's byte order
  bool single_file = true;               // "n+1"/"n+2": data follow the header in the same file; false for "ni1" (a .hdr + .img pair)
  std::int16_t datatype = 16;            // NIfTI datatype code (16 = float32)
  std::int16_t bitpix = 32;
  std::vector<std::int64_t> shape;       // dim[1..dim[0]]
  std::vector<double> pixdim;            // pixdim[1..dim[0]]
  double qfac = 1.0;                     // pixdim[0]: the sign of the third axis of the qform
  std::uint64_t vox_offset = 352;
  double scl_slope = 1.0, scl_inter = 0.0;
  double cal_max = 0.0, cal_min = 0.0;
  double slice_duration = 0.0, toffset = 0.0;
  std::int64_t slice_start = 0, slice_end = 0;
  int slice_code = 0;
  int dim_info = 0;
  int xyzt_units = 0;
  int intent_code = 0;
  double intent_p1 = 0.0, intent_p2 = 0.0, intent_p3 = 0.0;
  std::string intent_name, descrip, aux_file;
  int qform_code = 0, sform_code = 0;
  double quatern_b = 0.0, quatern_c = 0.0, quatern_d = 0.0;
  std::array<double, 3> qoffset{0.0, 0.0, 0.0};
  std::array<double, 12> srow{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};   // rows x, y, z of the sform, 3 x 4 row-major
  std::vector<Extension> extensions;

  // Derived, not stored: the voxel -> world map of the best form (sform when its code is set, else qform when its code is set, else the
  // scaling-only map of the pixdims), 3 x 4 row-major, in `spatial_unit`.
  std::array<double, 12> affine() const;
  enum class AffineSource { scaling, qform, sform };
  AffineSource affine_source() const;
  std::string spatial_unit() const;      // "m", "mm", "um", or "" when unstated
  std::string time_unit() const;         // "s", "ms", "us", "hz", "ppm", "rads" or ""
  double unit_to_metres() const;         // metres per spatial unit; 1e-3 when the file states none (the format's convention is millimetres)
  std::uint64_t n_voxels() const;        // product of shape
};

struct Volume {
  Header header;
  Precision precision = Precision::float32;
  std::vector<float> f32;                // the values when precision is float32 (empty otherwise)
  std::vector<double> f64;               // the values when precision is float64
  std::size_t size() const { return precision == Precision::float32 ? f32.size() : f64.size(); }
  double at(std::size_t i) const { return precision == Precision::float32 ? static_cast<double>(f32[i]) : f64[i]; }
};

// The header only (348 or 540 bytes at the start of `src`, plus extensions). Throws code::format on bad bytes.
Header read_header(const ByteSource& src);

// A single-file volume, or a pair (`header_src` the .hdr, `data_src` the .img). Gzipped files are inflated by the caller (see read_file).
Volume read_volume(const ByteSource& src);
Volume read_volume(const ByteSource& header_src, const ByteSource& data_src);

// A file by path: `.nii`, `.nii.gz`, or a `.hdr`/`.img` pair named by either. A `.gz` is inflated whole (volumes are small).
Volume read_file(const std::string& path);

// The writer. `values` are the real-world values (after scaling); stored as header.datatype with scl_slope and scl_inter honoured
// (integers are rounded to nearest, and refused when out of the type's range). Writes little-endian, a single file (the version is the header's).
// `shape` and `datatype` come from the header; the sizes must agree with `values`. Extensions are written after the header.
void write_volume(ByteSink& sink, const Header& header, const std::vector<float>& values);
void write_volume(ByteSink& sink, const Header& header, const std::vector<double>& values);
void write_file(const std::string& path, const Header& header, const std::vector<double>& values);   // `.nii.gz` is gzipped

// A header for a float32 volume of the given shape, voxel size and sform (3 x 4 row-major), written as a NIfTI-1 or 2 file.
Header make_header(int version, std::vector<std::int64_t> shape, std::vector<double> voxel_size, const std::array<double, 12>& sform);

}  // namespace nxr::formats::nifti
