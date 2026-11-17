// FIF tag writing: the file id, blocks and tags, as MNE-Python writes them (mne/_fiff/write.py, BSD-3-Clause, the MNE-Python contributors; see
// THIRD_PARTY_NOTICES.md). The writers of the FIF objects (fif_objects.h) and, later, the exporters are built on it.
//
//   MemorySink sink;  fif::TagWriter w(sink);
//   w.start_file(id);  w.start_block(fif::block::kMneEvents);  w.write_int(fif::tag::kMneEventList, {...});  w.end_block(...);  w.end_file();
//
// A file is a sequence of tags (kind, type, size, next = 0; big-endian) and nested blocks, as MNE writes it: the file id, a directory pointer of
// -1 and a free list of -1 first, and a last NOP tag with next = -1 at the end. No directory is written (readers walk the tags, as MNE's do).
// Values are the file's types: floats are 32-bit (`write_float`, `write_float_matrix`) or 64-bit (`write_double`, `write_double_matrix`).
#pragma once
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/fif.h"

namespace nxr::formats::fif {

// A sparse matrix as the file stores it (single-precision values, 32-bit indices): compressed by rows (`row_compressed`) or by columns.
struct SparseMatrix {
  bool row_compressed = true;
  std::int32_t nrow = 0, ncol = 0;
  std::vector<double> data;            // nnz values (the file's 32-bit floats)
  std::vector<std::int32_t> indices;   // nnz column indices (rows when column-compressed)
  std::vector<std::int32_t> indptr;    // nrow + 1 (ncol + 1 when column-compressed)
};

class TagWriter {
 public:
  explicit TagWriter(ByteSink& sink) : sink_(sink) {}

  void start_file(const Id& file_id);   // file id, directory pointer -1, free list -1
  void end_file();                      // the closing NOP tag
  void start_block(std::int32_t kind);
  void end_block(std::int32_t kind);

  void write_int(std::int32_t kind, const std::vector<std::int32_t>& v);
  void write_int(std::int32_t kind, std::int32_t v) { write_int(kind, std::vector<std::int32_t>{v}); }
  void write_float(std::int32_t kind, const std::vector<double>& v);   // 32-bit
  void write_float(std::int32_t kind, double v) { write_float(kind, std::vector<double>{v}); }
  void write_double(std::int32_t kind, const std::vector<double>& v);
  void write_double(std::int32_t kind, double v) { write_double(kind, std::vector<double>{v}); }
  void write_string(std::int32_t kind, const std::string& utf8);       // ISO 8859-1; characters outside it as MNE's "&#nnnnnn;"
  void write_name_list(std::int32_t kind, const std::vector<std::string>& names);   // ':'-joined, a colon in a name as "{COLON}"
  void write_id(std::int32_t kind, const Id& id);
  void write_ch_info(const ChannelInfo& ch);
  void write_coord_trans(const CoordTrans& t);   // the transform and its inverse, as the file stores both
  void write_dig_point(const DigPoint& d);

  // dense matrices, row-major `rows x cols` (`dims` for 3-D: the file stores them last dimension first)
  void write_float_array(std::int32_t kind, const std::vector<double>& data);   // a one-dimensional matrix (32-bit)
  void write_float_matrix(std::int32_t kind, const std::vector<double>& data, std::int32_t rows, std::int32_t cols);
  void write_double_matrix(std::int32_t kind, const std::vector<double>& data, std::int32_t rows, std::int32_t cols);
  void write_int_matrix(std::int32_t kind, const std::vector<std::int32_t>& data, std::int32_t rows, std::int32_t cols);
  void write_complex_matrix(std::int32_t kind, const std::vector<std::complex<double>>& data, std::int32_t rows, std::int32_t cols, bool double_precision);
  void write_sparse(std::int32_t kind, const SparseMatrix& m);

  // a tag as another reader gave it (`Value` as `File::value` decoded it, for the plain types and matrices); used to keep what a writer does not model
  void write_generic(const GenericTag& t);
  void write_raw(std::int32_t kind, std::uint32_t type, const std::vector<unsigned char>& data);

  ByteSink& sink() { return sink_; }

 private:
  void header(std::int32_t kind, std::uint32_t type, std::uint64_t size);
  void matrix(std::int32_t kind, std::uint32_t elem_type, const std::vector<std::int32_t>& dims_rowmajor, const std::vector<unsigned char>& payload);
  ByteSink& sink_;
};

// The file id MNE writes for a file made now would carry the machine and time: callers pass what they want recorded (nothing machine-specific here).
Id make_id(std::int32_t secs, std::int32_t usecs);

}  // namespace nxr::formats::fif
