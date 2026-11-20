// nxr-io — NumPy .npy files, read and written (see npy.h).
#include "nxr/io/npy.h"

#include <cstring>
#include <fstream>
#include <regex>

#include "nxr/io/errors.h"

namespace nxr::io {

namespace fs = std::filesystem;

namespace {

[[noreturn]] void bad(const fs::path& p, const std::string& m) { throw Error(code::invalid_argument, p.string() + ": " + m); }

// the value of `key` in the header's Python dict literal (the text after `'key':`, up to its comma at depth 0)
std::string dict_value(const std::string& h, const std::string& key, const fs::path& p) {
  const auto k = h.find("'" + key + "'");
  if (k == std::string::npos) bad(p, "the .npy header has no '" + key + "'");
  auto i = h.find(':', k) + 1;
  while (i < h.size() && h[i] == ' ') ++i;
  int depth = 0;
  std::size_t j = i;
  for (; j < h.size(); ++j) {
    const char c = h[j];
    if (c == '(' || c == '[') ++depth;
    else if (c == ')' || c == ']') --depth;
    else if ((c == ',' || c == '}') && depth == 0) break;
  }
  return h.substr(i, j - i);
}

template <class T, class S>
void convert(const unsigned char* src, std::size_t n, T* out) {
  for (std::size_t i = 0; i < n; ++i) {
    S v;
    std::memcpy(&v, src + i * sizeof(S), sizeof(S));
    out[i] = static_cast<T>(v);
  }
}

}  // namespace

NpyFile NpyFile::open(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error(code::not_found, path.string() + ": no such file");
  char magic[8];
  if (!f.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0) bad(path, "not a .npy file");
  const int major = static_cast<unsigned char>(magic[6]);
  std::uint32_t hlen = 0;
  std::uint64_t pre = 0;
  if (major == 1) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    hlen = static_cast<std::uint32_t>(b[0] | (b[1] << 8));
    pre = 10;
  } else if (major == 2 || major == 3) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    hlen = static_cast<std::uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24));
    pre = 12;
  } else {
    bad(path, ".npy format version " + std::to_string(major) + " is not supported");
  }
  std::string h(hlen, '\0');
  if (!f.read(h.data(), hlen)) bad(path, "a truncated .npy header");

  NpyFile n;
  n.path_ = path;
  n.data_offset_ = pre + hlen;
  std::string descr = dict_value(h, "descr", path);
  if (descr.size() < 2 || (descr.front() != '\'' && descr.front() != '"')) bad(path, "a structured dtype is not supported (" + descr + ")");
  n.descr_ = descr.substr(1, descr.size() - 2);
  if (dict_value(h, "fortran_order", path) != "False") bad(path, "a Fortran-order array is not supported");
  const std::string shape = dict_value(h, "shape", path);
  static const std::regex num("[0-9]+");
  for (auto it = std::sregex_iterator(shape.begin(), shape.end(), num); it != std::sregex_iterator(); ++it) n.shape_.push_back(std::stoll(it->str()));
  if (n.descr_.size() < 3) bad(path, "dtype '" + n.descr_ + "' is not supported");
  const char order = n.descr_[0], kind = n.descr_[1];
  n.itemsize_ = static_cast<std::size_t>(std::stoul(n.descr_.substr(2)));
  if (n.itemsize_ > 1 && order != '<' && order != '=') bad(path, "a big-endian array is not supported (" + n.descr_ + ")");
  if (!((kind == 'f' && (n.itemsize_ == 4 || n.itemsize_ == 8)) || ((kind == 'i' || kind == 'u') && (n.itemsize_ == 1 || n.itemsize_ == 2 || n.itemsize_ == 4 || n.itemsize_ == 8)) ||
        (kind == 'b' && n.itemsize_ == 1)))
    bad(path, "dtype '" + n.descr_ + "' is not supported");
  std::error_code ec;
  const auto bytes = fs::file_size(path, ec);
  if (ec || bytes < n.data_offset_ + static_cast<std::uint64_t>(n.size()) * n.itemsize_) bad(path, "the file is shorter than its header says");
  return n;
}

std::int64_t NpyFile::size() const {
  std::int64_t s = 1;
  for (auto d : shape_) s *= d;
  return s;
}

std::int64_t NpyFile::row_size() const {
  std::int64_t s = 1;
  for (std::size_t i = 1; i < shape_.size(); ++i) s *= shape_[i];
  return s;
}

template <class T>
std::vector<T> NpyFile::rows(std::int64_t r0, std::int64_t r1) const {
  const std::int64_t nrows = shape_.empty() ? 1 : shape_[0];
  if (r0 < 0 || r1 < r0 || r1 > nrows) bad(path_, "rows [" + std::to_string(r0) + ", " + std::to_string(r1) + ") of " + std::to_string(nrows));
  const std::size_t n = static_cast<std::size_t>((r1 - r0) * row_size());
  std::vector<unsigned char> raw(n * itemsize_);
  std::ifstream f(path_, std::ios::binary);
  f.seekg(static_cast<std::streamoff>(data_offset_ + static_cast<std::uint64_t>(r0 * row_size()) * itemsize_));
  if (!f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()))) bad(path_, "a short read");
  std::vector<T> out(n);
  const char kind = descr_[1];
  const unsigned char* s = raw.data();
  if (kind == 'f') {
    if (itemsize_ == 4) convert<T, float>(s, n, out.data()); else convert<T, double>(s, n, out.data());
  } else if (kind == 'i') {
    switch (itemsize_) {
      case 1: convert<T, std::int8_t>(s, n, out.data()); break;
      case 2: convert<T, std::int16_t>(s, n, out.data()); break;
      case 4: convert<T, std::int32_t>(s, n, out.data()); break;
      default: convert<T, std::int64_t>(s, n, out.data());
    }
  } else {   // u, b
    switch (itemsize_) {
      case 1: convert<T, std::uint8_t>(s, n, out.data()); break;
      case 2: convert<T, std::uint16_t>(s, n, out.data()); break;
      case 4: convert<T, std::uint32_t>(s, n, out.data()); break;
      default: convert<T, std::uint64_t>(s, n, out.data());
    }
  }
  return out;
}

template std::vector<double> NpyFile::rows<double>(std::int64_t, std::int64_t) const;
template std::vector<float> NpyFile::rows<float>(std::int64_t, std::int64_t) const;
template std::vector<std::int64_t> NpyFile::rows<std::int64_t>(std::int64_t, std::int64_t) const;
template std::vector<std::int32_t> NpyFile::rows<std::int32_t>(std::int64_t, std::int64_t) const;
template std::vector<std::uint8_t> NpyFile::rows<std::uint8_t>(std::int64_t, std::int64_t) const;

namespace {
// numpy's descr of a store dtype, and back
const std::vector<std::pair<std::string, std::string>>& dtype_descrs() {
  static const std::vector<std::pair<std::string, std::string>> t = {{"float32", "<f4"}, {"float64", "<f8"}, {"int8", "|i1"}, {"int16", "<i2"}, {"int32", "<i4"}, {"int64", "<i8"},
                                                                     {"uint8", "|u1"}, {"uint16", "<u2"}, {"uint32", "<u4"}, {"uint64", "<u8"}, {"bool", "|b1"}};
  return t;
}
}  // namespace

std::string npy_dtype_name(const std::string& descr) {
  std::string d = descr;
  if (d.size() == 3 && d[0] == '=') d[0] = d[2] == '1' ? '|' : '<';
  if (d.size() == 3 && d[2] == '1' && d[0] == '<') d[0] = '|';
  for (const auto& [name, de] : dtype_descrs()) if (de == d) return name;
  throw Error(code::invalid_argument, "the .npy dtype '" + descr + "' is not one io reads (float32, float64, int8 to int64, uint8 to uint64, bool; little-endian)");
}

void write_npy(const fs::path& path, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<std::uint8_t>& bytes) {
  std::string descr;
  for (const auto& [name, de] : dtype_descrs()) if (name == dtype) descr = de;
  if (descr.empty()) throw Error(code::invalid_argument, path.string() + ": no .npy dtype for '" + dtype + "'");
  std::int64_t n = 1;
  for (const auto d : shape) n *= d;
  const std::size_t item = static_cast<std::size_t>(std::stoul(descr.substr(2)));
  if (bytes.size() != static_cast<std::size_t>(n) * item)
    throw Error(code::invalid_argument, path.string() + ": " + std::to_string(bytes.size()) + " bytes are not " + std::to_string(n) + " values of " + dtype);
  std::string sh;
  for (const auto d : shape) sh += std::to_string(d) + ", ";
  if (!sh.empty()) sh.erase(sh.size() - (shape.size() > 1 ? 2 : 1));   // (n,) for one axis, (a, b) for more, () for none
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" + sh + "), }";
  while ((10 + header.size() + 1) % 64) header += ' ';
  header += '\n';
  std::ofstream o(path, std::ios::binary | std::ios::trunc);
  if (!o) throw Error(code::not_writable, path.string() + ": cannot be written");
  o.write("\x93NUMPY\x01\x00", 8);
  const unsigned char len[2] = {static_cast<unsigned char>(header.size() & 0xff), static_cast<unsigned char>(header.size() >> 8)};
  o.write(reinterpret_cast<const char*>(len), 2);
  o << header;
  o.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!o) throw Error(code::not_writable, path.string() + ": a short write");
}

}  // namespace nxr::io
