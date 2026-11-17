// CIFTI-2 (dense and parcellated: dtseries, dscalar, dlabel, pscalar, ptseries, and any other two-dimensional map): a NIfTI-2 file whose extension
// of code 32 is XML describing what each index of the data matrix means. The XML is parsed whole (pugixml); the matrix is read by rows and columns.
//
//   auto r = cifti::open_file("sub-01.dtseries.nii");   // the header and the XML, never the data
//   r->map(1).brain_models;                              // what the columns are: surface vertices and voxels, by structure
//   r->map(0).series;                                    // what the rows are: a time line
//   r->read(r0, n, c0, c1, out);                         // rows (dimension 0) [r0, r0 + n), columns (dimension 1) [c0, c1), row-major
//
// Matrix orientation is nibabel's: the matrix is `rows() x cols()` where rows index matrix dimension 0 and columns dimension 1, as the XML's
// `AppliesToMatrixDimension` says: for a dtseries dimension 0 is the series (time) and dimension 1 the brain models (grayordinates); for a dscalar,
// dimension 0 the named maps and dimension 1 the brain models; for a pscalar dimension 1 is the parcels; for a ptseries dimension 0 is the series and
// dimension 1 the parcels. In the file dimension 0 varies fastest (NIfTI `dim[5]` is the length of dimension 0, `dim[6]` of dimension 1): all values of
// the first index of dimension 1 come first. `read` returns any rectangle in row-major order; it reads one contiguous run per column.
// What is parsed: the whole NIfTI-2 header, the Matrix metadata, and per mapping every attribute and child of CIFTI-2's `MatrixIndicesMap`: series
// (points, start, step, unit, exponent), named maps (name, metadata, label table with keys and colours), brain models (type, structure, index offset
// and count, surface vertex count, vertex indices, voxel indices), parcels (name, vertices per structure, voxels), surfaces, and the volume (its
// dimensions and voxel-to-world matrix with the meter exponent). Indices are zero-based as the file has them.
// Values: float32 and float64 files (read as 32-bit floats, and 64-bit through the double read); other data types, CIFTI-1 files, and gzipped
// files are refused with the reason. Precision follows the file.
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/nifti.h"
#include "nxr/formats/recording.h"   // Precision

namespace nxr::formats::cifti {

struct VolumeSpace {
  std::array<std::int64_t, 3> dims{};              // VolumeDimensions
  std::array<double, 16> ijk_to_xyz{};             // TransformationMatrixVoxelIndicesIJKtoXYZ, 4 x 4 row-major, in units of 10^meter_exponent metres
  int meter_exponent = -3;                         // millimetres by default
};

struct BrainModel {
  std::string model_type;                          // "CIFTI_MODEL_TYPE_SURFACE" or "CIFTI_MODEL_TYPE_VOXELS"
  std::string structure;                           // "CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_THALAMUS_LEFT", ...
  std::uint64_t index_offset = 0, index_count = 0; // the model's span along the dimension
  std::optional<std::uint64_t> surface_vertex_count;
  std::vector<std::uint32_t> vertex_indices;       // surface models: the surface vertex of each index of the model
  std::vector<std::array<std::int32_t, 3>> voxel_ijk;   // voxel models: the voxel of each index of the model
};

struct SurfaceInfo { std::string structure; std::uint64_t vertex_count = 0; };

struct ParcelVertices { std::string structure; std::vector<std::uint32_t> vertices; };

struct Parcel {
  std::string name;
  std::vector<ParcelVertices> vertices;            // per surface structure
  std::vector<std::array<std::int32_t, 3>> voxel_ijk;
};

struct LabelEntry { std::int32_t key = 0; std::string name; float red = 0, green = 0, blue = 0, alpha = 0; };

struct NamedMap {
  std::string name;
  std::vector<std::pair<std::string, std::string>> metadata;
  std::vector<LabelEntry> labels;                  // a label table (dlabel, plabel); empty for scalars
};

struct Series {
  std::uint64_t points = 0;
  double start = 0.0, step = 0.0;
  int exponent = 0;                                // the unit's power of ten (seconds: 0)
  std::string unit;                                // "SECOND", "HERTZ", "METER", "RADIAN"
};

struct Mapping {
  std::vector<int> applies_to;                     // the matrix dimensions it describes
  std::string type;                                // "CIFTI_INDEX_TYPE_BRAIN_MODELS", "_PARCELS", "_SCALARS", "_LABELS", "_SERIES"
  std::vector<BrainModel> brain_models;
  std::vector<SurfaceInfo> surfaces;               // parcellated maps: the surfaces the parcels' vertices are on
  std::vector<Parcel> parcels;
  std::vector<NamedMap> named_maps;
  std::optional<Series> series;
  std::optional<VolumeSpace> volume;
  std::uint64_t length() const;                    // the number of indices the mapping covers
};

class Reader {
 public:
  const nifti::Header& nifti_header() const { return nifti_; }
  const std::string& xml() const { return xml_; }
  const std::string& version() const { return version_; }                // "2" or "2.0", as the file has it
  const std::vector<std::pair<std::string, std::string>>& metadata() const { return metadata_; }   // the Matrix's own
  const std::vector<Mapping>& mappings() const { return mappings_; }
  const Mapping& map(int dimension) const;                                // the mapping of matrix dimension 0 or 1; throws when none covers it
  std::uint64_t rows() const { return rows_; }
  std::uint64_t cols() const { return cols_; }
  Precision precision() const { return precision_; }
  // Rows [r0, r0 + n), columns [c0, c1), row-major (out[(r - r0) * (c1 - c0) + (c - c0)]).
  void read(std::uint64_t r0, std::uint64_t n, std::uint64_t c0, std::uint64_t c1, float* out) const;
  void read(std::uint64_t r0, std::uint64_t n, std::uint64_t c0, std::uint64_t c1, double* out) const;

 private:
  friend std::unique_ptr<Reader> open(std::shared_ptr<const ByteSource>);
  template <class T> void read_values(std::uint64_t r0, std::uint64_t n, std::uint64_t c0, std::uint64_t c1, T* out) const;
  nifti::Header nifti_;
  std::string xml_, version_;
  std::vector<std::pair<std::string, std::string>> metadata_;
  std::vector<Mapping> mappings_;
  std::uint64_t rows_ = 0, cols_ = 0;
  Precision precision_ = Precision::float32;
  std::shared_ptr<const ByteSource> src_;
};

std::unique_ptr<Reader> open(std::shared_ptr<const ByteSource> file);
std::unique_ptr<Reader> open_file(const std::string& path);   // a `.nii` file; a gzipped one is refused

// The XML alone, parsed: for a caller that already holds the extension.
void parse_xml(const std::string& xml, std::string* version, std::vector<std::pair<std::string, std::string>>* metadata, std::vector<Mapping>* mappings);

}  // namespace nxr::formats::cifti
