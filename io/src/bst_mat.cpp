// nxr::bst — Brainstorm `.mat` files over matio (zlib; HDF5 for v7.3): see nxr/bst/mat.h.
#include "nxr/bst/mat.h"

#include <cstring>
#include <fstream>
#include <mutex>

extern "C" {
#include "matio.h"
}

namespace nxr::bst {

namespace {
thread_local std::string g_last_message;

void log_message(int, const char* message) { g_last_message = message ? message : ""; }

void init_logging() {
  static std::once_flag once;
  std::call_once(once, [] {
    Mat_LogInitFunc("nxr-io", log_message);
    Mat_SetVerbose(0, 1);   // nothing printed: the messages are kept for the error that follows
  });
}

std::size_t class_size(Class c) {
  switch (c) {
    case Class::Double: return 8;
    case Class::Single: return 4;
    case Class::Int8: case Class::UInt8: return 1;
    case Class::Int16: case Class::UInt16: return 2;
    case Class::Int32: case Class::UInt32: return 4;
    case Class::Int64: case Class::UInt64: return 8;
    default: return 0;
  }
}

Class class_of(int c) {
  switch (c) {
    case MAT_C_CELL: return Class::Cell;
    case MAT_C_STRUCT: return Class::Struct;
    case MAT_C_OBJECT: return Class::Object;
    case MAT_C_CHAR: return Class::Char;
    case MAT_C_SPARSE: return Class::Sparse;
    case MAT_C_DOUBLE: return Class::Double;
    case MAT_C_SINGLE: return Class::Single;
    case MAT_C_INT8: return Class::Int8;
    case MAT_C_UINT8: return Class::UInt8;
    case MAT_C_INT16: return Class::Int16;
    case MAT_C_UINT16: return Class::UInt16;
    case MAT_C_INT32: return Class::Int32;
    case MAT_C_UINT32: return Class::UInt32;
    case MAT_C_INT64: return Class::Int64;
    case MAT_C_UINT64: return Class::UInt64;
    case MAT_C_FUNCTION: return Class::Function;
    case MAT_C_OPAQUE: return Class::Opaque;
    // v7.3: matio gives class 0 to a MATLAB_class it does not know (string, datetime, table, categorical, user classes: MATLAB stores
    // them as an opaque uint32 reference array + the class name in a hidden subsystem); a v5 file never has class 0
    case MAT_C_EMPTY: return Class::Opaque;
    default: return Class::Empty;
  }
}

template <class T>
double widen(const unsigned char* p) { T v; std::memcpy(&v, p, sizeof v); return static_cast<double>(v); }
template <class T>
std::int64_t widen_i(const unsigned char* p) { T v; std::memcpy(&v, p, sizeof v); return static_cast<std::int64_t>(v); }

double element(Class c, const unsigned char* p) {
  switch (c) {
    case Class::Double: return widen<double>(p);
    case Class::Single: return widen<float>(p);
    case Class::Int8: return widen<std::int8_t>(p);
    case Class::UInt8: return widen<std::uint8_t>(p);
    case Class::Int16: return widen<std::int16_t>(p);
    case Class::UInt16: return widen<std::uint16_t>(p);
    case Class::Int32: return widen<std::int32_t>(p);
    case Class::UInt32: return widen<std::uint32_t>(p);
    case Class::Int64: return widen<std::int64_t>(p);
    case Class::UInt64: return widen<std::uint64_t>(p);
    default: return 0;
  }
}

std::int64_t element_i(Class c, const unsigned char* p) {
  switch (c) {
    case Class::Double: return static_cast<std::int64_t>(widen<double>(p));
    case Class::Single: return static_cast<std::int64_t>(widen<float>(p));
    case Class::Int8: return widen_i<std::int8_t>(p);
    case Class::UInt8: return widen_i<std::uint8_t>(p);
    case Class::Int16: return widen_i<std::int16_t>(p);
    case Class::UInt16: return widen_i<std::uint16_t>(p);
    case Class::Int32: return widen_i<std::int32_t>(p);
    case Class::UInt32: return widen_i<std::uint32_t>(p);
    case Class::Int64: return widen_i<std::int64_t>(p);
    case Class::UInt64: return widen_i<std::uint64_t>(p);
    default: return 0;
  }
}

void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) out += static_cast<char>(cp);
  else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
  else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
  else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
}

std::size_t product(const std::vector<std::size_t>& d) {
  if (d.empty()) return 0;
  std::size_t n = 1;
  for (auto x : d) n *= x;
  return n;
}

Value convert(const matvar_t* mv, const std::string& file);

// the name of the first opaque value inside v (a MATLAB object matio cannot decode), or empty
bool has_opaque(const Value& v) {
  if (v.cls == Class::Opaque) return true;
  for (const auto& e : v.elems) if (has_opaque(e)) return true;
  return false;
}

// char data: code units in the stored type; the rows of a char matrix are strided by the row count
void read_chars(Value& v, const matvar_t* mv) {
  const std::size_t n = product(v.dims);
  const std::size_t rows = v.dims.empty() ? 0 : v.dims[0];
  const std::size_t cols = rows ? n / rows : 0;
  if (!mv->data || n == 0) { v.text.assign(rows ? rows : 0, std::string()); if (rows == 0) v.text.clear(); return; }
  const std::size_t unit = static_cast<std::size_t>(mv->data_size);
  const unsigned char* p = static_cast<const unsigned char*>(mv->data);
  if (unit == 1) {
    // bytes: a row vector's text is UTF-8 as stored; a matrix is read character by character
    if (mv->nbytes != n || rows == 1) { v.text.push_back(std::string(reinterpret_cast<const char*>(p), mv->nbytes)); return; }
    for (std::size_t r = 0; r < rows; ++r) {
      std::string row;
      for (std::size_t c = 0; c < cols; ++c) append_utf8(row, p[r + c * rows]);
      v.text.push_back(std::move(row));
    }
    return;
  }
  auto unit16 = [&](std::size_t k) { std::uint16_t u; std::memcpy(&u, p + 2 * k, 2); return static_cast<std::uint32_t>(u); };
  if (unit != 2) throw Error(nxr::io::code::unsupported, "char data of " + std::to_string(unit) + " bytes per character");
  for (std::size_t r = 0; r < rows; ++r) {
    std::string row;
    for (std::size_t c = 0; c < cols; ++c) {
      std::uint32_t cp = unit16(r + c * rows);
      if (cp >= 0xD800 && cp < 0xDC00 && c + 1 < cols) {
        const std::uint32_t lo = unit16(r + (c + 1) * rows);
        if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); ++c; }
      }
      append_utf8(row, cp);
    }
    v.text.push_back(std::move(row));
  }
}

Value convert(const matvar_t* mv, const std::string& file) {
  Value v;
  v.cls = class_of(mv->class_type);
  v.dims.assign(mv->dims, mv->dims + mv->rank);
  v.is_complex = mv->isComplex != 0;
  v.is_logical = mv->isLogical != 0;
  const std::size_t n = product(v.dims);
  switch (v.cls) {
    case Class::Cell:
      if (n && mv->data) {
        for (std::size_t i = 0; i < n; ++i) {
          const matvar_t* c = Mat_VarGetCell(mv, static_cast<int>(i));
          v.elems.push_back(c ? convert(c, file) : Value{});
        }
      }
      break;
    case Class::Struct: {
      const unsigned nf = Mat_VarGetNumberOfFields(mv);
      char* const* names = Mat_VarGetStructFieldnames(mv);
      for (unsigned f = 0; f < nf; ++f) v.fields.push_back(names[f]);
      if (nf && n && mv->data) {
        for (std::size_t e = 0; e < n; ++e)
          for (unsigned f = 0; f < nf; ++f) {
            const matvar_t* c = Mat_VarGetStructFieldByIndex(mv, f, e);
            v.elems.push_back(c ? convert(c, file) : Value{});
          }
      }
      break;
    }
    case Class::Char:
      read_chars(v, mv);
      break;
    case Class::Sparse: {
      const mat_sparse_t* sp = static_cast<const mat_sparse_t*>(mv->data);
      if (sp) {
        v.ir.assign(sp->ir, sp->ir + sp->nir);
        v.jc.assign(sp->jc, sp->jc + sp->njc);
        const std::size_t nz = sp->ndata;
        if (mv->isComplex) {
          const mat_complex_split_t* z = static_cast<const mat_complex_split_t*>(sp->data);
          v.re.assign(static_cast<const unsigned char*>(z->Re), static_cast<const unsigned char*>(z->Re) + nz * 8);
          v.im.assign(static_cast<const unsigned char*>(z->Im), static_cast<const unsigned char*>(z->Im) + nz * 8);
        } else {
          v.re.assign(static_cast<const unsigned char*>(sp->data), static_cast<const unsigned char*>(sp->data) + nz * 8);
        }
      }
      break;
    }
    case Class::Double: case Class::Single: case Class::Int8: case Class::UInt8: case Class::Int16: case Class::UInt16:
    case Class::Int32: case Class::UInt32: case Class::Int64: case Class::UInt64: {
      const std::size_t sz = class_size(v.cls);
      if (n && mv->data) {
        if (static_cast<std::size_t>(mv->data_size) != sz)
          throw Error(nxr::io::code::unsupported, file + ": a " + std::string(class_name(v.cls)) + " array stored as " + std::to_string(mv->data_size) + "-byte values");
        if (mv->isComplex) {
          const mat_complex_split_t* z = static_cast<const mat_complex_split_t*>(mv->data);
          v.re.assign(static_cast<const unsigned char*>(z->Re), static_cast<const unsigned char*>(z->Re) + n * sz);
          v.im.assign(static_cast<const unsigned char*>(z->Im), static_cast<const unsigned char*>(z->Im) + n * sz);
        } else {
          v.re.assign(static_cast<const unsigned char*>(mv->data), static_cast<const unsigned char*>(mv->data) + n * sz);
        }
      }
      break;
    }
    default:
      break;   // objects, functions, opaque: named and sized, nothing more
  }
  return v;
}

}  // namespace

const char* class_name(Class c) {
  switch (c) {
    case Class::Empty: return "empty";
    case Class::Cell: return "cell";
    case Class::Struct: return "struct";
    case Class::Object: return "object";
    case Class::Char: return "char";
    case Class::Sparse: return "sparse";
    case Class::Double: return "double";
    case Class::Single: return "single";
    case Class::Int8: return "int8";
    case Class::UInt8: return "uint8";
    case Class::Int16: return "int16";
    case Class::UInt16: return "uint16";
    case Class::Int32: return "int32";
    case Class::UInt32: return "uint32";
    case Class::Int64: return "int64";
    case Class::UInt64: return "uint64";
    case Class::Function: return "function_handle";
    case Class::Opaque: return "opaque";
  }
  return "?";
}

// ── Value ──
std::size_t Value::numel() const { return product(dims); }

bool Value::is_numeric() const {
  switch (cls) {
    case Class::Double: case Class::Single: case Class::Int8: case Class::UInt8: case Class::Int16: case Class::UInt16:
    case Class::Int32: case Class::UInt32: case Class::Int64: case Class::UInt64: return true;
    default: return false;
  }
}

std::vector<double> Value::doubles() const {
  std::vector<double> out;
  if (is_numeric()) {
    const std::size_t sz = class_size(cls), n = re.size() / sz;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(element(cls, re.data() + i * sz));
  } else if (cls == Class::Sparse) {
    out.assign(numel(), 0.0);
    const std::size_t nr = rows();
    for (std::size_t c = 0; c + 1 < jc.size(); ++c)
      for (std::uint64_t k = jc[c]; k < jc[c + 1]; ++k) { double x; std::memcpy(&x, re.data() + 8 * k, 8); out[ir[k] + c * nr] = x; }
  } else {
    throw Error(nxr::io::code::invalid_argument, std::string("a ") + class_name(cls) + " is not numeric");
  }
  return out;
}

std::vector<double> Value::imag() const {
  std::vector<double> out;
  if (!is_complex || !is_numeric()) return out;
  const std::size_t sz = class_size(cls), n = im.size() / sz;
  for (std::size_t i = 0; i < n; ++i) out.push_back(element(cls, im.data() + i * sz));
  return out;
}

std::vector<std::int64_t> Value::int64s() const {
  std::vector<std::int64_t> out;
  if (!is_numeric()) throw Error(nxr::io::code::invalid_argument, std::string("a ") + class_name(cls) + " is not numeric");
  const std::size_t sz = class_size(cls), n = re.size() / sz;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) out.push_back(element_i(cls, re.data() + i * sz));
  return out;
}

std::size_t Value::cols() const {
  if (dims.empty()) return 0;
  const std::size_t n = numel();
  return dims[0] ? n / dims[0] : 0;
}

std::vector<double> Value::row_major() const {
  if (dims.size() > 2) throw Error(nxr::io::code::invalid_argument, "row_major: an array of more than two dimensions");
  const std::vector<double> a = doubles();
  const std::size_t r = rows(), c = cols();
  std::vector<double> out(a.size());
  for (std::size_t i = 0; i < r; ++i)
    for (std::size_t j = 0; j < c; ++j) out[i * c + j] = a[i + j * r];
  return out;
}

double Value::scalar() const {
  if (!is_numeric() || re.empty()) throw Error(nxr::io::code::invalid_argument, "scalar: an empty or non-numeric value");
  return element(cls, re.data());
}

std::string Value::string() const {
  if (cls != Class::Char) throw Error(nxr::io::code::invalid_argument, std::string("string: a ") + class_name(cls) + " is not char");
  if (text.empty()) return "";
  if (text.size() > 1) throw Error(nxr::io::code::invalid_argument, "string: a char array of " + std::to_string(text.size()) + " rows");
  return text[0];
}

const Value* Value::field(const std::string& name, std::size_t elem) const {
  if (cls != Class::Struct || elem >= numel()) return nullptr;
  for (std::size_t f = 0; f < fields.size(); ++f)
    if (fields[f] == name) return &elems[elem * fields.size() + f];
  return nullptr;
}

bool Value::has_field(const std::string& name) const {
  for (const auto& f : fields) if (f == name) return true;
  return false;
}

// ── MatFile ──
bool is_v73(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return false;
  char h[128] = {0};
  in.read(h, sizeof h);
  const std::size_t got = static_cast<std::size_t>(in.gcount());
  static const unsigned char sig[8] = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
  if (got >= 8 && std::memcmp(h, sig, 8) == 0) return true;
  if (got >= 128) {
    if (std::strncmp(h, "MATLAB 7.3", 10) == 0) return true;
    // the version word: 0x0200 in the byte order the endian marker names
    const bool little = h[126] == 'I' && h[127] == 'M';
    const unsigned ver = little ? (static_cast<unsigned char>(h[124]) | static_cast<unsigned char>(h[125]) << 8)
                                : (static_cast<unsigned char>(h[125]) | static_cast<unsigned char>(h[124]) << 8);
    if (std::strncmp(h, "MATLAB", 6) == 0 && ver == 0x0200) return true;
  }
  return false;
}

MatFile MatFile::open(const std::filesystem::path& file) {
  init_logging();
  std::error_code ec;
  if (!std::filesystem::is_regular_file(file, ec)) throw Error(nxr::io::code::not_found, "not a file: " + file.string());
  g_last_message.clear();
#ifdef __EMSCRIPTEN__
  // the WebAssembly build's matio has no HDF5: a v7.3 file is refused by name, not with matio's generic failure
  if (is_v73(file))
    throw Error(nxr::io::code::unsupported, file.string() + ": a MATLAB v7.3 (HDF5) file, which nxr-neuro's WebAssembly build does not read (it has no HDF5); "
                                            "re-save it as v7 (MATLAB: save -v7), or convert it with the native nxr-neuro");
#endif
  mat_t* m = Mat_Open(file.string().c_str(), MAT_ACC_RDONLY);
  if (!m) throw Error(nxr::io::code::invalid_argument, file.string() + ": not a MATLAB MAT-file matio can read" + (g_last_message.empty() ? "" : " (" + g_last_message + ")"));
  MatFile f;
  f.path_ = file;
  f.mat_ = m;
  return f;
}

MatFile::MatFile(MatFile&& o) noexcept : path_(std::move(o.path_)), mat_(o.mat_) { o.mat_ = nullptr; }
MatFile& MatFile::operator=(MatFile&& o) noexcept {
  if (this != &o) { if (mat_) Mat_Close(static_cast<mat_t*>(mat_)); path_ = std::move(o.path_); mat_ = o.mat_; o.mat_ = nullptr; }
  return *this;
}
MatFile::~MatFile() { if (mat_) Mat_Close(static_cast<mat_t*>(mat_)); }

std::string MatFile::header() const {
  const char* h = Mat_GetHeader(static_cast<mat_t*>(mat_));
  return h ? h : "";
}
int MatFile::version() const {
  switch (Mat_GetVersion(static_cast<mat_t*>(mat_))) {
    case MAT_FT_MAT4: return 4;
    case MAT_FT_MAT5: return 5;
    case MAT_FT_MAT73: return 73;   // 7.3 (HDF5)
    default: return 0;
  }
}

std::vector<VariableInfo> MatFile::list() {
  std::vector<VariableInfo> out;
  mat_t* m = static_cast<mat_t*>(mat_);
  Mat_Rewind(m);
  while (matvar_t* mv = Mat_VarReadNextInfo(m)) {
    VariableInfo i;
    i.name = mv->name ? mv->name : "";
    i.cls = class_of(mv->class_type);
    i.dims.assign(mv->dims, mv->dims + mv->rank);
    i.is_complex = mv->isComplex != 0;
    i.is_logical = mv->isLogical != 0;
    out.push_back(std::move(i));
    Mat_VarFree(mv);
  }
  Mat_Rewind(m);
  return out;
}

bool MatFile::complete() {
  mat_t* m = static_cast<mat_t*>(mat_);
  if (Mat_GetVersion(m) == MAT_FT_MAT73) return true;   // HDF5 has no stream to end inside a variable; a damaged one fails to open
  Mat_Rewind(m);
  g_last_message.clear();
  while (matvar_t* mv = Mat_VarReadNextInfo(m)) Mat_VarFree(mv);
  const bool ok = g_last_message.empty();
  Mat_Rewind(m);
  return ok;
}

std::optional<Value> MatFile::read(const std::string& name) {
  mat_t* m = static_cast<mat_t*>(mat_);
  Mat_Rewind(m);
  g_last_message.clear();
  matvar_t* mv = Mat_VarRead(m, name.c_str());
  if (!mv) return std::nullopt;
  std::optional<Value> out;
  try {
    out = convert(mv, path_.string());
  } catch (...) {
    Mat_VarFree(mv);
    throw;
  }
  Mat_VarFree(mv);
  // a v7.3 file stores MATLAB objects (string, datetime, table, ...) opaquely: refuse them by name rather than hand back a stub
  if (Mat_GetVersion(m) == MAT_FT_MAT73 && has_opaque(*out))
    throw Error(nxr::io::code::unsupported, path_.string() + ": variable '" + name + "' holds a MATLAB object (string, datetime, table, categorical or a class "
                "of the user's) that a v7.3 file stores opaquely; nxr-io does not read those. Re-save it with the object converted to plain data (char, cellstr, struct)");
  return out;
}

std::vector<std::pair<std::string, Value>> MatFile::read_all() {
  std::vector<std::pair<std::string, Value>> out;
  mat_t* m = static_cast<mat_t*>(mat_);
  Mat_Rewind(m);
  while (matvar_t* mv = Mat_VarReadNext(m)) {
    try {
      out.emplace_back(mv->name ? mv->name : "", convert(mv, path_.string()));
    } catch (...) {
      Mat_VarFree(mv);
      throw;
    }
    Mat_VarFree(mv);
  }
  Mat_Rewind(m);
  return out;
}

std::optional<MatFile::MatrixInfo> MatFile::matrix_info(const std::string& name) {
  mat_t* m = static_cast<mat_t*>(mat_);
  Mat_Rewind(m);
  matvar_t* mv = Mat_VarReadInfo(m, name.c_str());
  if (!mv) return std::nullopt;
  std::optional<MatrixInfo> out;
  const Class c = class_of(mv->class_type);
  if (mv->rank == 2 && !mv->isComplex && class_size(c) != 0) out = MatrixInfo{c, mv->dims[0], mv->dims[1]};
  Mat_VarFree(mv);
  return out;
}

void MatFile::read_columns_f32(const std::string& name, std::size_t c0, std::size_t n, std::vector<float>& out) {
  mat_t* m = static_cast<mat_t*>(mat_);
  Mat_Rewind(m);
  matvar_t* mv = Mat_VarReadInfo(m, name.c_str());
  if (!mv) throw Error(nxr::io::code::no_such_row, name);
  struct Free { matvar_t* v; ~Free() { Mat_VarFree(v); } } guard{mv};
  const Class c = class_of(mv->class_type);
  if (mv->rank != 2 || mv->isComplex || class_size(c) == 0) throw Error(nxr::io::code::invalid_argument, name + ": not a real 2-D numeric matrix");
  const std::size_t rows = mv->dims[0];
  if (c0 + n > mv->dims[1]) throw Error(nxr::io::code::invalid_argument, name + ": columns outside the matrix");
  int start[2] = {0, static_cast<int>(c0)}, stride[2] = {1, 1}, edge[2] = {static_cast<int>(rows), static_cast<int>(n)};
  std::vector<unsigned char> raw(rows * n * class_size(c));
  if (Mat_VarReadData(m, mv, raw.data(), start, stride, edge) != 0) throw Error(nxr::io::code::invalid_argument, path_.string() + ": cannot read " + name + " (" + g_last_message + ")");
  out.resize(rows * n);
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < rows; ++i) {
      const unsigned char* p = raw.data() + (j * rows + i) * class_size(c);
      out[i * n + j] = static_cast<float>(element(c, p));   // double -> float32 rounds as numpy's astype does
    }
}

std::optional<Value> read_variable(const std::filesystem::path& file, const std::string& name) {
  MatFile f = MatFile::open(file);
  return f.read(name);
}

std::map<std::string, Value> read_variables(const std::filesystem::path& file, const std::vector<std::string>& names) {
  MatFile f = MatFile::open(file);
  std::map<std::string, Value> out;
  for (const auto& n : names)
    if (auto v = f.read(n)) out.emplace(n, std::move(*v));
  return out;
}

std::map<std::string, Value> read_all_variables(const std::filesystem::path& file) {
  MatFile f = MatFile::open(file);
  std::map<std::string, Value> out;
  for (auto& kv : f.read_all()) out.emplace(std::move(kv));
  return out;
}

}  // namespace nxr::bst
