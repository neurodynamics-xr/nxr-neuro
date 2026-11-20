// nxr-io — NumPy's .npy files, read (format versions 1–3, C order, little-endian or one-byte types) and written (version 1.0, C order, little-endian): the files an
// external solver writes, and the command line's array format (`read <class> --out`, `add <class> --values`).
//
//   NpyFile f = NpyFile::open("fsaverage_hodge_exact_modes.npy");      // the header only
//   std::vector<double> rows = f.rows<double>(0, 64);                   // rows [0, 64) of a 2-D array, converted to double
//   std::vector<std::int64_t> keep = f.all<std::int64_t>();             // the whole array
//
// A row range is read from its offset in the file, so a GB array is never held whole. The values are converted from the stored type
// (f4, f8, i1–i8, u1–u8, b1) to the type asked for; Fortran order and big-endian multi-byte types are refused.
#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nxr::io {

class NpyFile {
 public:
  static NpyFile open(const std::filesystem::path& path);

  const std::filesystem::path& path() const { return path_; }
  const std::vector<std::int64_t>& shape() const { return shape_; }
  const std::string& descr() const { return descr_; }   // numpy's dtype string, e.g. "<f8"
  std::int64_t size() const;                            // the number of elements
  std::int64_t row_size() const;                        // the elements of one row (the product of every axis but the first; 1 for a 1-D array)

  // rows [r0, r1) of axis 0, converted to T (C order: row after row)
  template <class T> std::vector<T> rows(std::int64_t r0, std::int64_t r1) const;
  template <class T> std::vector<T> all() const { return rows<T>(0, shape_.empty() ? 1 : shape_[0]); }

 private:
  std::filesystem::path path_;
  std::vector<std::int64_t> shape_;
  std::string descr_;
  std::size_t itemsize_ = 0;
  std::uint64_t data_offset_ = 0;
};

/** the store's name of a .npy dtype ("<f4" -> "float32", "|u1" -> "uint8", "|b1" -> "bool"); an Error for one io does not read */
std::string npy_dtype_name(const std::string& descr);
/** a .npy file (version 1.0, C order, little-endian) of `shape` holding `bytes`: the values of `dtype` as the store names it (float32, float64, int8 to int64,
 *  uint8 to uint64, bool), little-endian; an Error when the bytes are not the shape's values of that dtype */
void write_npy(const std::filesystem::path& path, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<std::uint8_t>& bytes);

}  // namespace nxr::io
