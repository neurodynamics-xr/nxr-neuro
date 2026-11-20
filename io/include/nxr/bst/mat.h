// nxr-io — the `bst` reading layer: a Brainstorm `.mat` (MATLAB v4 to v7.3) opened with matio, its variables listed, one variable
// read as a typed Value. The counterpart of nxr.convert.matio (pymatreader), without its squeezing: a struct is a struct, a struct array
// has its elements, a cell its cells, a char its text, a number its class — exactly as stored.
//
//   auto mat = bst::MatFile::open("tess_cortex_pial_low.mat");        // v4 to v7.3 (HDF5); a v7.3 file reads as its v7 twin
//   for (auto& v : mat.list()) ...                                    // name · class · dims
//   bst::Value v = mat.read("Vertices");                              // one variable
//   auto xyz = v.row_major();                                         // a 2-D numeric array as nRows x nCols doubles, row after row
//   bst::Value s = *bst::read_variable(path, "Atlas");                // or one variable by file and name
//   for (std::size_t e = 0; e < s.numel(); ++e) s.field("Name", e)->string();
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/errors.h"

namespace nxr::bst {

using Error = nxr::io::Error;

enum class Class { Empty, Cell, Struct, Object, Char, Sparse, Double, Single, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Function, Opaque };
const char* class_name(Class c);   // MATLAB's name: "double", "char", "struct", ...

class Value {
 public:
  Class cls = Class::Empty;
  std::vector<std::size_t> dims;           // MATLAB's, column-major order of the data
  bool is_complex = false;
  bool is_logical = false;                 // a logical array (stored as uint8)
  std::vector<unsigned char> re, im;       // numeric: the stored values in the class's native type (little-endian host order); sparse: its nonzeros
  std::vector<std::string> text;           // char: the rows, UTF-8
  std::vector<std::string> fields;         // struct: the field names
  std::vector<Value> elems;                // cell: the cells; struct: element after element, `fields.size()` values each
  std::vector<std::uint64_t> ir, jc;       // sparse: row indices and column pointers

  std::size_t numel() const;
  bool empty() const { return numel() == 0; }
  bool is_numeric() const;                 // a dense numeric (or logical) array
  bool is_struct() const { return cls == Class::Struct; }
  bool is_cell() const { return cls == Class::Cell; }
  bool is_char() const { return cls == Class::Char; }

  // numeric (and logical, char as its code points): the values widened to double / int64, in MATLAB's (column-major) order
  std::vector<double> doubles() const;
  std::vector<double> imag() const;        // the imaginary part (empty unless complex)
  std::vector<std::int64_t> int64s() const;
  // a 2-D numeric array as nRows x nCols doubles, row after row (a vector is 1 x n)
  std::vector<double> row_major() const;
  std::size_t rows() const { return dims.empty() ? 0 : dims[0]; }
  std::size_t cols() const;                // the product of the other dimensions
  double scalar() const;                   // the first element, as double (throws on an empty or non-numeric value)

  // char: the text of a row vector (or an empty char); more than one row is an error
  std::string string() const;

  // struct: a field of element `elem`; nullptr when the struct has no such field (or no such element)
  const Value* field(const std::string& name, std::size_t elem = 0) const;
  bool has_field(const std::string& name) const;
  // cell: the i-th cell (column-major)
  const Value& cell(std::size_t i) const { return elems.at(i); }
};

struct VariableInfo {
  std::string name;
  Class cls = Class::Empty;
  std::vector<std::size_t> dims;
  bool is_complex = false, is_logical = false;
};

class MatFile {
 public:
  // Open a `.mat` for reading (v4, v5, v6, v7, and v7.3 = HDF5, which reads as its v7 twin). NotFound when it is not a file; ValueError when matio
  // cannot read it as a MAT-file. `read(name)` of a v7.3 variable that holds a MATLAB object stored opaquely (string, datetime, table,
  // categorical, a user class) throws Unsupported naming the file and the variable; `read_all` returns such a value as Class::Opaque.
  static MatFile open(const std::filesystem::path& file);
  MatFile(MatFile&&) noexcept;
  MatFile& operator=(MatFile&&) noexcept;
  ~MatFile();

  const std::filesystem::path& path() const { return path_; }
  std::string header() const;               // the 116-byte descriptive text
  int version() const;                      // 4 · 5 (v6, v7 are 5 as well: the file format of the header) · 73 (v7.3, HDF5)
  std::vector<VariableInfo> list();         // the variables' names, classes and sizes, without their data
  // False when a v4/v5 file ends inside a variable (truncated): matio then stops quietly, while nxr's reader (pymatreader) reads every variable and fails.
  bool complete();
  std::optional<Value> read(const std::string& name);   // one variable (nullopt when the file has none of that name)
  std::vector<std::pair<std::string, Value>> read_all();

  // A 2-D real numeric variable [rows x cols] read WITHOUT its data (nullopt: absent, not 2-D or not real numeric) ...
  struct MatrixInfo { Class cls = Class::Empty; std::size_t rows = 0, cols = 0; };
  std::optional<MatrixInfo> matrix_info(const std::string& name);
  // ... and its columns [c0, c0 + n) read alone, as float32 [rows x n] row after row (a time slab of channels x time): the file is not loaded whole
  void read_columns_f32(const std::string& name, std::size_t c0, std::size_t n, std::vector<float>& out);

 private:
  MatFile() = default;
  std::filesystem::path path_;
  void* mat_ = nullptr;   // mat_t*
};

// One variable of a file (nullopt when absent); the variables named, as pymatreader's `variable_names`.
std::optional<Value> read_variable(const std::filesystem::path& file, const std::string& name);
std::map<std::string, Value> read_variables(const std::filesystem::path& file, const std::vector<std::string>& names);
std::map<std::string, Value> read_all_variables(const std::filesystem::path& file);

// Whether a file begins as a MATLAB v7.3 (HDF5) file: the 128-byte header says "MATLAB 7.3", or the HDF5 signature is at its start or after a user block.
bool is_v73(const std::filesystem::path& file);

// ---- A MATLAB v7.3 FILE THIS BUILD CANNOT READ ----
// v7.3 is HDF5. The native builds read it (matio over HDF5); the WebAssembly build has no HDF5 (and is 32-bit), so there `MatFile::open` refuses a v7.3 file with
// UnreadableV73 (an Error `unsupported`, naming the file), and the Brainstorm importers (`import subject | condition | template | group`) SKIP that one file, import the rest,
// and name it with kUnreadableV73 in the import's result and job row (`skipped`). Any other unreadable file is refused as before.
constexpr const char* kUnreadableV73 = "MATLAB v7.3 (HDF5): not readable in this build; import it with a native build";
struct UnreadableV73 : nxr::io::Error {
  explicit UnreadableV73(std::filesystem::path f);
  std::filesystem::path file;
};
// Whether this build reads v7.3 files: true natively, false in WebAssembly. `set_v73_readable(false)` makes a native build behave as the WebAssembly one (its tests do; the
// default comes back with `set_v73_readable(true)`).
bool v73_readable();
void set_v73_readable(bool readable);

// The files an import skipped, gathered while one is alive (the innermost scope of the thread); `note_skipped` adds one to it (nothing when none is open).
struct SkippedFile { std::string file, reason; };
class SkippedFiles {
 public:
  SkippedFiles();
  ~SkippedFiles();
  SkippedFiles(const SkippedFiles&) = delete;
  SkippedFiles& operator=(const SkippedFiles&) = delete;
  const std::vector<SkippedFile>& files() const { return files_; }

 private:
  friend void note_skipped(const std::string& file, const std::string& reason);
  std::vector<SkippedFile> files_;
  SkippedFiles* outer_ = nullptr;
};
void note_skipped(const std::string& file, const std::string& reason);

}  // namespace nxr::bst
