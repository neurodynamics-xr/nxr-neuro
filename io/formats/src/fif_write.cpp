// FIF tag writing. Ported from MNE-Python's mne/_fiff/write.py (BSD-3-Clause, Copyright the MNE-Python contributors; see THIRD_PARTY_NOTICES.md).
#include "nxr/formats/fif_write.h"

#include <cstdio>
#include <cstring>

#include "fif_internal.h"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"

namespace nxr::formats::fif {

namespace {

void put_i32(std::vector<unsigned char>& b, std::int32_t v) {
  unsigned char t[4];
  store_u32(t, static_cast<std::uint32_t>(v), false);
  b.insert(b.end(), t, t + 4);
}
void put_f32(std::vector<unsigned char>& b, double v) {
  unsigned char t[4];
  store_f32(t, static_cast<float>(v), false);
  b.insert(b.end(), t, t + 4);
}
void put_f64(std::vector<unsigned char>& b, double v) {
  unsigned char t[8];
  store_f64(t, v, false);
  b.insert(b.end(), t, t + 8);
}

// UTF-8 to ISO 8859-1; what Latin-1 cannot hold becomes a six-digit decimal character reference, which MNE reads back (html.unescape) and so do we
std::string utf8_to_latin1(const std::string& s) {
  std::string out;
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    unsigned cp = c;
    std::size_t n = 1;
    if (c >= 0xF0 && i + 3 < s.size()) { cp = ((c & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu); n = 4; }
    else if (c >= 0xE0 && i + 2 < s.size()) { cp = ((c & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu); n = 3; }
    else if (c >= 0xC0 && i + 1 < s.size()) { cp = ((c & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu); n = 2; }
    if (cp < 256) out += static_cast<char>(cp);
    else { char buf[16]; std::snprintf(buf, sizeof buf, "&#%06u;", cp); out += buf; }
    i += n;
  }
  return out;
}

}  // namespace

Id make_id(std::int32_t secs, std::int32_t usecs) {
  Id id;
  id.version = (1 << 16) | 2;   // FIFFC_MAJOR_VERSION 1, MINOR 2
  id.machid = {0, 0};
  id.secs = secs;
  id.usecs = usecs;
  return id;
}

void TagWriter::header(std::int32_t kind, std::uint32_t type, std::uint64_t size) {
  if (size > 0x7FFFFFFFu) throw refused("FIF: a tag of " + std::to_string(size) + " bytes does not fit the format (kind " + std::to_string(kind) + ")");
  unsigned char h[16];
  store_u32(h, static_cast<std::uint32_t>(kind), false);
  store_u32(h + 4, type, false);
  store_u32(h + 8, static_cast<std::uint32_t>(size), false);
  store_u32(h + 12, 0, false);   // FIFFV_NEXT_SEQ
  sink_.write(h, 16);
}

void TagWriter::write_raw(std::int32_t kind, std::uint32_t type, const std::vector<unsigned char>& data) {
  header(kind, type, data.size());
  if (!data.empty()) sink_.write(data.data(), data.size());
}

void TagWriter::start_file(const Id& id) {
  write_id(tag::kFileId, id);
  write_int(tag::kDirPointer, -1);
  write_int(tag::kFreeList, -1);
}

void TagWriter::end_file() {
  unsigned char h[16];
  store_u32(h, static_cast<std::uint32_t>(tag::kNop), false);
  store_u32(h + 4, dtype::kVoid, false);
  store_u32(h + 8, 0, false);
  store_u32(h + 12, 0xFFFFFFFFu, false);   // FIFFV_NEXT_NONE
  sink_.write(h, 16);
}

void TagWriter::start_block(std::int32_t kind) { write_int(tag::kBlockStart, kind); }
void TagWriter::end_block(std::int32_t kind) { write_int(tag::kBlockEnd, kind); }

void TagWriter::write_int(std::int32_t kind, const std::vector<std::int32_t>& v) {
  std::vector<unsigned char> b;
  for (std::int32_t x : v) put_i32(b, x);
  write_raw(kind, dtype::kInt, b);
}
void TagWriter::write_float(std::int32_t kind, const std::vector<double>& v) {
  std::vector<unsigned char> b;
  for (double x : v) put_f32(b, x);
  write_raw(kind, dtype::kFloat, b);
}
void TagWriter::write_double(std::int32_t kind, const std::vector<double>& v) {
  std::vector<unsigned char> b;
  for (double x : v) put_f64(b, x);
  write_raw(kind, dtype::kDouble, b);
}
void TagWriter::write_string(std::int32_t kind, const std::string& utf8) {
  const std::string l = utf8_to_latin1(utf8);
  if (l.empty()) return;   // MNE writes nothing for an empty string
  write_raw(kind, dtype::kString, std::vector<unsigned char>(l.begin(), l.end()));
}
void TagWriter::write_name_list(std::int32_t kind, const std::vector<std::string>& names) {
  std::string s;
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].find("{COLON}") != std::string::npos) throw refused("FIF: the substring \"{COLON}\" in a name is not supported");
    if (i) s += ':';
    for (char c : names[i]) { if (c == ':') s += "{COLON}"; else s += c; }
  }
  write_string(kind, s);
}
void TagWriter::write_id(std::int32_t kind, const Id& id) {
  std::vector<unsigned char> b;
  put_i32(b, id.version);
  put_i32(b, id.machid[0]);
  put_i32(b, id.machid[1]);
  put_i32(b, id.secs);
  put_i32(b, id.usecs);
  write_raw(kind, dtype::kIdStruct, b);
}
void TagWriter::write_ch_info(const ChannelInfo& ch) {
  std::vector<unsigned char> b;
  put_i32(b, ch.scanno);
  put_i32(b, ch.logno);
  put_i32(b, ch.kind);
  put_f32(b, ch.range);
  put_f32(b, ch.cal);
  put_i32(b, ch.coil_type);
  for (double x : ch.loc) put_f32(b, x);
  put_i32(b, ch.unit);
  put_i32(b, ch.unit_mul);
  std::string name = utf8_to_latin1(ch.name).substr(0, 15);
  name.resize(16, '\0');
  b.insert(b.end(), name.begin(), name.end());
  write_raw(tag::kChInfo, dtype::kChInfoStruct, b);
}
void TagWriter::write_coord_trans(const CoordTrans& t) {
  const CoordTrans inv = detail::invert(t);
  std::vector<unsigned char> b;
  put_i32(b, t.from);
  put_i32(b, t.to);
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) put_f32(b, t.trans[r * 4 + c]);
  for (int r = 0; r < 3; ++r) put_f32(b, t.trans[r * 4 + 3]);
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) put_f32(b, inv.trans[r * 4 + c]);
  for (int r = 0; r < 3; ++r) put_f32(b, inv.trans[r * 4 + 3]);
  write_raw(tag::kCoordTrans, dtype::kCoordTransStruct, b);
}
void TagWriter::write_dig_point(const DigPoint& d) {
  std::vector<unsigned char> b;
  put_i32(b, d.kind);
  put_i32(b, d.ident);
  for (double x : d.r) put_f32(b, x);
  write_raw(tag::kDigPoint, dtype::kDigPointStruct, b);
}

void TagWriter::matrix(std::int32_t kind, std::uint32_t elem_type, const std::vector<std::int32_t>& dims, const std::vector<unsigned char>& payload) {
  std::vector<unsigned char> b = payload;
  for (std::size_t i = dims.size(); i-- > 0;) put_i32(b, dims[i]);   // last dimension first
  put_i32(b, static_cast<std::int32_t>(dims.size()));
  write_raw(kind, dtype::kMatrix | elem_type, b);
}
void TagWriter::write_float_array(std::int32_t kind, const std::vector<double>& d) {
  std::vector<unsigned char> p;
  for (double x : d) put_f32(p, x);
  matrix(kind, dtype::kFloat, {static_cast<std::int32_t>(d.size())}, p);
}
void TagWriter::write_float_matrix(std::int32_t kind, const std::vector<double>& d, std::int32_t rows, std::int32_t cols) {
  if (d.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) throw format_error("FIF: a matrix of " + std::to_string(rows) + " x " + std::to_string(cols) + " given " + std::to_string(d.size()) + " values");
  std::vector<unsigned char> p;
  for (double x : d) put_f32(p, x);
  matrix(kind, dtype::kFloat, {rows, cols}, p);
}
void TagWriter::write_double_matrix(std::int32_t kind, const std::vector<double>& d, std::int32_t rows, std::int32_t cols) {
  if (d.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) throw format_error("FIF: a matrix of " + std::to_string(rows) + " x " + std::to_string(cols) + " given " + std::to_string(d.size()) + " values");
  std::vector<unsigned char> p;
  for (double x : d) put_f64(p, x);
  matrix(kind, dtype::kDouble, {rows, cols}, p);
}
void TagWriter::write_int_matrix(std::int32_t kind, const std::vector<std::int32_t>& d, std::int32_t rows, std::int32_t cols) {
  if (d.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) throw format_error("FIF: a matrix of " + std::to_string(rows) + " x " + std::to_string(cols) + " given " + std::to_string(d.size()) + " values");
  std::vector<unsigned char> p;
  for (std::int32_t x : d) put_i32(p, x);
  matrix(kind, dtype::kInt, {rows, cols}, p);
}
void TagWriter::write_complex_matrix(std::int32_t kind, const std::vector<std::complex<double>>& d, std::int32_t rows, std::int32_t cols, bool dp) {
  if (d.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) throw format_error("FIF: a complex matrix of " + std::to_string(rows) + " x " + std::to_string(cols) + " given " + std::to_string(d.size()) + " values");
  std::vector<unsigned char> p;
  for (const auto& z : d) { if (dp) { put_f64(p, z.real()); put_f64(p, z.imag()); } else { put_f32(p, z.real()); put_f32(p, z.imag()); } }
  matrix(kind, dp ? dtype::kComplexDouble : dtype::kComplexFloat, {rows, cols}, p);
}
void TagWriter::write_sparse(std::int32_t kind, const SparseMatrix& m) {
  const std::size_t nnz = m.data.size();
  const std::size_t ptr = static_cast<std::size_t>(m.row_compressed ? m.nrow : m.ncol) + 1;
  if (m.indices.size() != nnz || m.indptr.size() != ptr) throw format_error("FIF: a sparse matrix whose index arrays do not match its values");
  std::vector<unsigned char> b;
  for (double x : m.data) put_f32(b, x);
  for (std::int32_t x : m.indices) put_i32(b, x);
  for (std::int32_t x : m.indptr) put_i32(b, x);
  put_i32(b, static_cast<std::int32_t>(nnz));
  put_i32(b, m.nrow);
  put_i32(b, m.ncol);
  put_i32(b, 2);
  write_raw(kind, dtype::kMatrix | (m.row_compressed ? dtype::kSparseRcs : dtype::kSparseCcs) | dtype::kFloat, b);
}

void TagWriter::write_generic(const GenericTag& t) {
  if (!t.value.bytes.empty() && t.value.ints.empty() && t.value.reals.empty() && t.value.complexes.empty() && t.value.text.empty()) {
    write_raw(t.kind, t.type, t.value.bytes);   // a structured or undecoded tag, as stored
    return;
  }
  const std::uint32_t coding = t.type & 0xFFFF0000u, base = t.type & 0xFFFFu;
  if (coding == dtype::kMatrix) {
    const std::int32_t r = t.value.dims.size() > 0 ? t.value.dims[0] : 0, c = t.value.dims.size() > 1 ? t.value.dims[1] : 1;
    if (t.value.dims.size() > 2) throw refused("FIF: a generic 3-D matrix tag (kind " + std::to_string(t.kind) + ") is not written");
    if (base == dtype::kInt || base == dtype::kJulian) { std::vector<std::int32_t> v; for (auto x : t.value.ints) v.push_back(static_cast<std::int32_t>(x)); write_int_matrix(t.kind, v, r, c); }
    else if (base == dtype::kFloat) write_float_matrix(t.kind, t.value.reals, r, c);
    else if (base == dtype::kDouble) write_double_matrix(t.kind, t.value.reals, r, c);
    else write_complex_matrix(t.kind, t.value.complexes, r, c, base == dtype::kComplexDouble);
    return;
  }
  switch (t.type) {
    case dtype::kInt: case dtype::kJulian: { std::vector<unsigned char> b; for (auto x : t.value.ints) put_i32(b, static_cast<std::int32_t>(x)); write_raw(t.kind, t.type, b); break; }
    case dtype::kFloat: write_float(t.kind, t.value.reals); break;
    case dtype::kDouble: write_double(t.kind, t.value.reals); break;
    case dtype::kString: write_string(t.kind, t.value.text); break;
    case dtype::kVoid: write_raw(t.kind, t.type, {}); break;
    case dtype::kShort: case dtype::kUShort: case dtype::kDauPack16: {
      std::vector<unsigned char> b;
      for (auto x : t.value.ints) { unsigned char q[2]; store_u16(q, static_cast<std::uint16_t>(x), false); b.insert(b.end(), q, q + 2); }
      write_raw(t.kind, t.type, b);
      break;
    }
    case dtype::kByte: { std::vector<unsigned char> b; for (auto x : t.value.ints) b.push_back(static_cast<unsigned char>(x)); write_raw(t.kind, t.type, b); break; }
    case dtype::kUInt: { std::vector<unsigned char> b; for (auto x : t.value.ints) put_i32(b, static_cast<std::int32_t>(static_cast<std::uint32_t>(x))); write_raw(t.kind, t.type, b); break; }
    case dtype::kComplexFloat: case dtype::kComplexDouble: {
      std::vector<unsigned char> b;
      for (const auto& z : t.value.complexes) { if (t.type == dtype::kComplexDouble) { put_f64(b, z.real()); put_f64(b, z.imag()); } else { put_f32(b, z.real()); put_f32(b, z.imag()); } }
      write_raw(t.kind, t.type, b);
      break;
    }
    default: throw refused("FIF: a generic tag of data type " + std::to_string(t.type) + " (kind " + std::to_string(t.kind) + ") is not written");
  }
}

}  // namespace nxr::formats::fif
