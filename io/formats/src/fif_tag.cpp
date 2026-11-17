// FIF tag layer: the file id, the tag directory, the block tree, and tag values. Ported from MNE-Python's mne/_fiff/{open,tree,tag}.py
// (BSD-3-Clause, Copyright the MNE-Python contributors; see THIRD_PARTY_NOTICES.md).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "fif_internal.h"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"
#include "nxr/formats/fif.h"

namespace nxr::formats::fif {

namespace detail {

Id decode_id(const std::vector<unsigned char>& b) {
  if (b.size() < 20) throw format_error("FIF: an id structure has " + std::to_string(b.size()) + " bytes, expected 20");
  Id id;
  id.version = load_i32(b.data(), false);
  id.machid = {load_i32(b.data() + 4, false), load_i32(b.data() + 8, false)};
  id.secs = load_i32(b.data() + 12, false);
  id.usecs = load_i32(b.data() + 16, false);
  return id;
}

ChannelInfo decode_ch_info(const std::vector<unsigned char>& b) {
  if (b.size() < 96) throw format_error("FIF: a channel info structure has " + std::to_string(b.size()) + " bytes, expected 96");
  ChannelInfo c;
  const unsigned char* p = b.data();
  c.scanno = load_i32(p, false);
  c.logno = load_i32(p + 4, false);
  c.kind = load_i32(p + 8, false);
  c.range = load_f32(p + 12, false);
  c.cal = load_f32(p + 16, false);
  c.coil_type = load_i32(p + 20, false);
  for (int i = 0; i < 12; ++i) c.loc[i] = load_f32(p + 24 + 4 * i, false);
  c.unit = load_i32(p + 72, false);
  c.unit_mul = load_i32(p + 76, false);
  std::size_t n = 0;
  while (n < 16 && p[80 + n] != 0) ++n;
  c.name = std::string(reinterpret_cast<const char*>(p + 80), n);
  return c;
}

DigPoint decode_dig_point(const unsigned char* p) {
  DigPoint d;
  d.kind = load_i32(p, false);
  d.ident = load_i32(p + 4, false);
  for (int i = 0; i < 3; ++i) d.r[i] = load_f32(p + 8 + 4 * i, false);
  return d;
}

std::vector<DigPoint> decode_dig_string(const std::vector<unsigned char>& b) {
  if (b.size() < 12) throw format_error("FIF: a digitisation string structure is shorter than its header");
  const std::int32_t kind = load_i32(b.data(), false), ident = load_i32(b.data() + 4, false), n = load_i32(b.data() + 8, false);
  if (n < 0 || b.size() < 12 + static_cast<std::size_t>(n) * 12) throw format_error("FIF: a digitisation string structure is truncated");
  std::vector<DigPoint> out(static_cast<std::size_t>(n));
  for (std::int32_t i = 0; i < n; ++i) {
    out[i].kind = kind;
    out[i].ident = ident;
    for (int k = 0; k < 3; ++k) out[i].r[k] = load_f32(b.data() + 12 + 12 * i + 4 * k, false);
  }
  return out;
}

CoordTrans decode_coord_trans(const std::vector<unsigned char>& b) {
  if (b.size() < 56) throw format_error("FIF: a coordinate transform structure is truncated");
  CoordTrans t;
  const unsigned char* p = b.data();
  t.from = load_i32(p, false);
  t.to = load_i32(p + 4, false);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) t.trans[r * 4 + c] = load_f32(p + 8 + 4 * (r * 3 + c), false);
    t.trans[r * 4 + 3] = load_f32(p + 44 + 4 * r, false);
  }
  t.trans[12] = t.trans[13] = t.trans[14] = 0.0;
  t.trans[15] = 1.0;
  return t;   // the file's stored inverse (the next 48 bytes) is not read, as MNE does not
}

CoordTrans invert(const CoordTrans& t) {
  // Gauss-Jordan on [A | I] for a 4 x 4 matrix
  double a[4][8];
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 8; ++c) a[r][c] = c < 4 ? t.trans[r * 4 + c] : (c - 4 == r ? 1.0 : 0.0);
  for (int col = 0; col < 4; ++col) {
    int piv = col;
    for (int r = col + 1; r < 4; ++r)
      if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
    if (std::fabs(a[piv][col]) < 1e-300) throw format_error("FIF: a coordinate transform is singular");
    if (piv != col)
      for (int c = 0; c < 8; ++c) std::swap(a[piv][c], a[col][c]);
    const double d = a[col][col];
    for (int c = 0; c < 8; ++c) a[col][c] /= d;
    for (int r = 0; r < 4; ++r) {
      if (r == col) continue;
      const double f = a[r][col];
      for (int c = 0; c < 8; ++c) a[r][c] -= f * a[col][c];
    }
  }
  CoordTrans o;
  o.from = t.to;
  o.to = t.from;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) o.trans[r * 4 + c] = a[r][c + 4];
  return o;
}

static void append_utf8(std::string& s, unsigned cp) {
  if (cp < 0x80) s += static_cast<char>(cp);
  else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
  else if (cp < 0x10000) { s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
  else { s += static_cast<char>(0xF0 | (cp >> 18)); s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
}

// ISO 8859-1 (the FIF standard) to UTF-8, then MNE's `&#hhhhhh;` escapes (how it stores characters outside Latin-1) back to characters.
std::string latin1_to_utf8(const unsigned char* p, std::size_t n) {
  while (n > 0 && p[n - 1] == 0) --n;
  std::string s;
  s.reserve(n);
  for (std::size_t i = 0; i < n; ++i) append_utf8(s, p[i]);
  std::string out;
  for (std::size_t i = 0; i < s.size();) {
    if (s[i] == '&' && i + 9 <= s.size() && s[i + 1] == '#' && s[i + 8] == ';') {
      bool hex = true;
      for (int k = 2; k < 8; ++k) hex = hex && std::isxdigit(static_cast<unsigned char>(s[i + k]));
      if (hex) {
        // html.unescape reads "&#" followed by digits as a decimal reference; "&#x" would be hexadecimal, which these six characters cannot start with
        bool dec = true;
        for (int k = 2; k < 8; ++k) dec = dec && std::isdigit(static_cast<unsigned char>(s[i + k]));
        if (dec) {
          append_utf8(out, static_cast<unsigned>(std::stoul(s.substr(i + 2, 6))));
          i += 9;
          continue;
        }
      }
    }
    out += s[i++];
  }
  return out;
}

std::vector<std::string> split_name_list(const std::string& s) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  std::size_t start = 0;
  for (;;) {
    const std::size_t at = s.find(':', start);
    std::string name = s.substr(start, at == std::string::npos ? std::string::npos : at - start);
    for (std::size_t k; (k = name.find("{COLON}")) != std::string::npos;) name.replace(k, 7, ":");
    out.push_back(std::move(name));
    if (at == std::string::npos) break;
    start = at + 1;
  }
  return out;
}

}  // namespace detail

namespace {

struct Header16 {
  TagHeader h;
  bool ok = false;
};

Header16 read_header(const ByteSource& src, std::uint64_t pos) {
  Header16 r;
  if (pos > src.size() || src.size() - pos < 16) return r;
  unsigned char b[16];
  src.read(pos, 16, b);
  r.h.kind = load_i32(b, false);
  r.h.type = load_u32(b + 4, false);
  r.h.size = load_i32(b + 8, false);
  r.h.next = load_i32(b + 12, false);
  r.h.pos = pos;
  r.ok = true;
  return r;
}

std::size_t element_size(std::uint32_t t) {
  switch (t) {
    case dtype::kByte: return 1;
    case dtype::kShort: case dtype::kUShort: case dtype::kDauPack16: return 2;
    case dtype::kInt: case dtype::kUInt: case dtype::kFloat: case dtype::kJulian: return 4;
    case dtype::kDouble: case dtype::kComplexFloat: return 8;
    case dtype::kComplexDouble: return 16;
    default: return 0;
  }
}

}  // namespace

File::File(std::shared_ptr<const ByteSource> source) : source_(std::move(source)) {
  const ByteSource& src = *source_;
  const Header16 first = read_header(src, 0);
  if (!first.ok || first.h.kind != tag::kFileId || first.h.type != dtype::kIdStruct || first.h.size != 20)
    throw format_error("FIF: the file does not start with a file id tag (not a FIF file)");
  file_id_ = detail::decode_id(bytes(first.h));
  const Header16 second = read_header(src, 16 + 20);
  if (!second.ok || second.h.kind != tag::kDirPointer || second.h.size < 4) throw format_error("FIF: the file has no directory pointer after the file id");
  std::uint8_t b4[4];
  src.read(second.h.data_pos(), 4, b4);
  const std::int64_t dirpos = load_i32(b4, false);

  bool slow = true;
  if (dirpos > 0 && static_cast<std::uint64_t>(dirpos) < src.size()) {
    const Header16 d = read_header(src, static_cast<std::uint64_t>(dirpos));
    if (d.ok && d.h.type == dtype::kDirEntryStruct && d.h.size > 0 && d.h.size % 16 == 0 && d.h.data_pos() + d.h.size <= src.size()) {
      const std::vector<unsigned char> raw = bytes(d.h);
      std::vector<TagHeader> dir;
      bool good = true;
      for (std::size_t i = 0; good && i + 16 <= raw.size(); i += 16) {
        TagHeader e;
        e.kind = load_i32(&raw[i], false);
        e.type = load_u32(&raw[i + 4], false);
        e.size = load_i32(&raw[i + 8], false);
        e.pos = static_cast<std::uint64_t>(load_u32(&raw[i + 12], false));   // the directory keeps each tag's position where a tag keeps `next`
        e.next = -1;
        good = e.pos < src.size();
        dir.push_back(e);
      }
      if (good) {
        directory_ = std::move(dir);
        slow = false;
      }
    }
  }
  if (slow) {   // no usable directory: walk the tags
    std::uint64_t pos = 0;
    for (;;) {
      const Header16 t = read_header(src, pos);
      if (!t.ok) break;   // a file ending with a partial tag header
      directory_.push_back(t.h);
      if (t.h.next == 0) {
        if (t.h.size < 0) throw format_error("FIF: a tag at " + std::to_string(pos) + " has a negative size");
        pos = pos + 16 + static_cast<std::uint64_t>(t.h.size);
      } else if (t.h.next > 0) {
        if (static_cast<std::uint64_t>(t.h.next) <= pos) throw format_error("FIF: the tag at " + std::to_string(pos) + " points backwards (to " + std::to_string(t.h.next) + ")");
        pos = static_cast<std::uint64_t>(t.h.next);
      } else {
        break;
      }
    }
  }

  // the block tree (mne/_fiff/tree.py make_dir_tree): a block ends at the first block-end tag
  root_.kind = 0;
  root_.id = file_id_;
  std::vector<Block*> stack{&root_};
  for (const TagHeader& e : directory_) {
    Block* top = stack.back();
    if (e.kind == tag::kBlockStart) {
      if (e.size < 4) throw format_error("FIF: a block-start tag at " + std::to_string(e.pos) + " has no block kind");
      std::uint8_t k4[4];
      src.read(e.data_pos(), 4, k4);
      top->children.emplace_back();
      Block* child = &top->children.back();
      child->kind = load_i32(k4, false);
      stack.push_back(child);
    } else if (e.kind == tag::kBlockEnd) {
      if (stack.size() > 1) stack.pop_back();
    } else {
      top->tags.push_back(e);
      if (top != &root_ && (e.kind == tag::kBlockId || e.kind == tag::kParentBlockId) && e.size == 20) {
        (e.kind == tag::kBlockId ? top->id : top->parent_id) = detail::decode_id(bytes(e));
      }
    }
  }
}

std::vector<unsigned char> File::bytes(const TagHeader& t) const {
  if (t.size < 0) throw format_error("FIF: a tag at " + std::to_string(t.pos) + " has a negative size");
  if (t.data_pos() + static_cast<std::uint64_t>(t.size) > source_->size())
    throw format_error("FIF: the tag at " + std::to_string(t.pos) + " runs past the end of the file (truncated file)");
  std::vector<unsigned char> b(static_cast<std::size_t>(t.size));
  if (!b.empty()) source_->read(t.data_pos(), b.size(), b.data());
  return b;
}

Value File::value(const TagHeader& t) const {
  Value v;
  if (t.size <= 0) return v;
  const std::vector<unsigned char> b = bytes(t);
  const unsigned char* p = b.data();
  const std::uint32_t coding = t.type & 0xFFFF0000u;
  if (coding != 0) {
    if (coding != dtype::kMatrix) throw refused("FIF: a sparse matrix tag (kind " + std::to_string(t.kind) + ") is not read");
    const std::uint32_t mtype = t.type & 0x0000FFFFu;
    const std::size_t es = element_size(mtype);
    if (es == 0 || mtype == dtype::kShort || mtype == dtype::kByte || mtype == dtype::kUShort || mtype == dtype::kUInt || mtype == dtype::kDauPack16)
      throw refused("FIF: a matrix of element type " + std::to_string(mtype) + " (tag kind " + std::to_string(t.kind) + ") is not read");
    if (b.size() < 4) throw format_error("FIF: a matrix tag is too short");
    const std::int32_t ndim = load_i32(p + b.size() - 4, false);
    if (ndim < 0 || ndim > 3 || b.size() < 4u * (static_cast<std::size_t>(ndim) + 1)) throw refused("FIF: only matrices of up to 3 dimensions are read (tag kind " + std::to_string(t.kind) + ")");
    v.dims.resize(static_cast<std::size_t>(ndim));
    std::uint64_t count = 1;
    for (std::int32_t i = 0; i < ndim; ++i) {   // stored last-dimension first
      const std::int32_t d = load_i32(p + b.size() - 4 * (static_cast<std::size_t>(ndim) + 1) + 4 * i, false);
      v.dims[static_cast<std::size_t>(ndim - 1 - i)] = d;
      if (d < 0) throw format_error("FIF: a matrix has a negative dimension");
      count *= static_cast<std::uint64_t>(d);
    }
    if (count * es + 4u * (static_cast<std::size_t>(ndim) + 1) > b.size()) throw format_error("FIF: a matrix tag is shorter than its dimensions say");
    if (mtype == dtype::kInt || mtype == dtype::kJulian) {
      for (std::uint64_t i = 0; i < count; ++i) v.ints.push_back(load_i32(p + 4 * i, false));
    } else if (mtype == dtype::kFloat) {
      for (std::uint64_t i = 0; i < count; ++i) v.reals.push_back(load_f32(p + 4 * i, false));
    } else if (mtype == dtype::kDouble) {
      for (std::uint64_t i = 0; i < count; ++i) v.reals.push_back(load_f64(p + 8 * i, false));
    } else if (mtype == dtype::kComplexFloat) {
      for (std::uint64_t i = 0; i < count; ++i) v.complexes.emplace_back(load_f32(p + 8 * i, false), load_f32(p + 8 * i + 4, false));
    } else {
      for (std::uint64_t i = 0; i < count; ++i) v.complexes.emplace_back(load_f64(p + 16 * i, false), load_f64(p + 16 * i + 8, false));
    }
    return v;
  }
  const std::size_t es = element_size(t.type);
  if (es != 0) {
    if (b.size() % es != 0) throw format_error("FIF: a tag of element size " + std::to_string(es) + " holds " + std::to_string(b.size()) + " bytes (kind " + std::to_string(t.kind) + ")");
    const std::size_t n = b.size() / es;
    for (std::size_t i = 0; i < n; ++i) {
      const unsigned char* q = p + i * es;
      switch (t.type) {
        case dtype::kByte: v.ints.push_back(q[0]); break;
        case dtype::kShort: case dtype::kDauPack16: v.ints.push_back(load_i16(q, false)); break;
        case dtype::kUShort: v.ints.push_back(load_u16(q, false)); break;
        case dtype::kInt: case dtype::kJulian: v.ints.push_back(load_i32(q, false)); break;
        case dtype::kUInt: v.ints.push_back(load_u32(q, false)); break;
        case dtype::kFloat: v.reals.push_back(load_f32(q, false)); break;
        case dtype::kDouble: v.reals.push_back(load_f64(q, false)); break;
        case dtype::kComplexFloat: v.complexes.emplace_back(load_f32(q, false), load_f32(q + 4, false)); break;
        default: v.complexes.emplace_back(load_f64(q, false), load_f64(q + 8, false)); break;
      }
    }
    return v;
  }
  switch (t.type) {
    case dtype::kString: v.text = detail::latin1_to_utf8(p, b.size()); return v;
    case dtype::kVoid: return v;
    case dtype::kOldPack: {
      if (b.size() < 8) throw format_error("FIF: an old-pack tag is too short");
      const double offset = load_f32(p, false), scale = load_f32(p + 4, false);
      for (std::size_t i = 8; i + 2 <= b.size(); i += 2) v.reals.push_back(load_i16(p + i, false) * scale + offset);
      return v;
    }
    case dtype::kChInfoStruct: case dtype::kIdStruct: case dtype::kDirEntryStruct: case dtype::kDigPointStruct: case dtype::kCoordTransStruct:
    case dtype::kDigStringStruct:
      v.bytes = b;
      return v;
    default: throw refused("FIF: a tag of unimplemented data type " + std::to_string(t.type) + " (kind " + std::to_string(t.kind) + ")");
  }
}

std::vector<const Block*> find_blocks(const Block& root, std::int32_t kind) {
  std::vector<const Block*> out;
  // pre-order, children in file order
  struct Walk {
    static void go(const Block& b, std::int32_t k, std::vector<const Block*>& o) {
      if (b.kind == k) o.push_back(&b);
      for (const Block& c : b.children) go(c, k, o);
    }
  };
  Walk::go(root, kind, out);
  return out;
}

const TagHeader* find_tag(const Block& b, std::int32_t kind) {
  for (const TagHeader& t : b.tags)
    if (t.kind == kind) return &t;
  return nullptr;
}

}  // namespace nxr::formats::fif
