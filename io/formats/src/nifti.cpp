#include "nxr/formats/nifti.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "nxr/formats/deflate.h"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"

namespace nxr::formats::nifti {

namespace {

std::string trim_nul(const unsigned char* p, std::size_t n) {
  std::size_t k = 0;
  while (k < n && p[k] != 0) ++k;
  return std::string(reinterpret_cast<const char*>(p), k);
}

struct TypeInfo { int bytes; bool is_float; bool is_signed; bool wide; };   // wide: needs float64 to hold exactly

bool type_info(int code, TypeInfo& t) {
  switch (code) {
    case 2:    t = {1, false, false, false}; return true;   // uint8
    case 256:  t = {1, false, true, false};  return true;   // int8
    case 4:    t = {2, false, true, false};  return true;   // int16
    case 512:  t = {2, false, false, false}; return true;   // uint16
    case 8:    t = {4, false, true, true};   return true;   // int32
    case 768:  t = {4, false, false, true};  return true;   // uint32
    case 1024: t = {8, false, true, true};   return true;   // int64
    case 1280: t = {8, false, false, true};  return true;   // uint64
    case 16:   t = {4, true, true, false};   return true;   // float32
    case 64:   t = {8, true, true, true};    return true;   // float64
    default: return false;
  }
}

std::string datatype_name(int code) {
  switch (code) {
    case 32: return "complex64"; case 1792: return "complex128"; case 2048: return "complex256";
    case 128: return "RGB24"; case 2304: return "RGBA32"; case 1536: return "float128"; case 0: return "unknown (0)";
    default: return "code " + std::to_string(code);
  }
}

// the stored value at `p` as a double
double stored_value(const unsigned char* p, int code, bool little) {
  switch (code) {
    case 2:    return p[0];
    case 256:  return static_cast<std::int8_t>(p[0]);
    case 4:    return load_i16(p, little);
    case 512:  return load_u16(p, little);
    case 8:    return load_i32(p, little);
    case 768:  return load_u32(p, little);
    case 1024: return static_cast<double>(load_i64(p, little));
    case 1280: return static_cast<double>(load_u64(p, little));
    case 16:   return load_f32(p, little);
    default:   return load_f64(p, little);
  }
}

constexpr double kUnitToMetres[] = {1e-3, 1.0, 1e-3, 1e-6};   // unstated (mm convention), m, mm, um

}  // namespace

std::uint64_t Header::n_voxels() const {
  std::uint64_t n = 1;
  for (std::int64_t d : shape) {
    if (d < 0) throw format_error("NIfTI: a negative dimension");
    if (d != 0 && n > UINT64_MAX / static_cast<std::uint64_t>(d)) throw format_error("NIfTI: the dimensions overflow");
    n *= static_cast<std::uint64_t>(d);
  }
  return n;
}

std::string Header::spatial_unit() const {
  switch (xyzt_units & 7) { case 1: return "m"; case 2: return "mm"; case 3: return "um"; default: return ""; }
}
std::string Header::time_unit() const {
  switch (xyzt_units & 0x38) { case 8: return "s"; case 16: return "ms"; case 24: return "us"; case 32: return "hz"; case 40: return "ppm"; case 48: return "rads"; default: return ""; }
}
double Header::unit_to_metres() const {
  const int u = xyzt_units & 7;
  return (u >= 1 && u <= 3) ? kUnitToMetres[u] : kUnitToMetres[0];
}

Header::AffineSource Header::affine_source() const {
  if (sform_code > 0) return AffineSource::sform;
  if (qform_code > 0) return AffineSource::qform;
  return AffineSource::scaling;
}

std::array<double, 12> Header::affine() const {
  std::array<double, 12> m{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  switch (affine_source()) {
    case AffineSource::sform: return srow;
    case AffineSource::qform: {
      const double b = quatern_b, c = quatern_c, d = quatern_d;
      double a = 1.0 - (b * b + c * c + d * d);
      double s = 1.0;
      double bb = b, cc = c, dd = d;
      if (a < 1e-7) { const double n = std::sqrt(b * b + c * c + d * d); if (n > 0) { bb /= n; cc /= n; dd /= n; } a = 0.0; }
      else a = std::sqrt(a);
      (void)s;
      const double R[3][3] = {{a * a + bb * bb - cc * cc - dd * dd, 2 * (bb * cc - a * dd), 2 * (bb * dd + a * cc)},
                              {2 * (bb * cc + a * dd), a * a + cc * cc - bb * bb - dd * dd, 2 * (cc * dd - a * bb)},
                              {2 * (bb * dd - a * cc), 2 * (cc * dd + a * bb), a * a + dd * dd - bb * bb - cc * cc}};
      const double dx = pixdim.size() > 0 ? pixdim[0] : 1.0, dy = pixdim.size() > 1 ? pixdim[1] : 1.0, dz = (pixdim.size() > 2 ? pixdim[2] : 1.0) * (qfac < 0 ? -1.0 : 1.0);
      for (int r = 0; r < 3; ++r) {
        m[r * 4 + 0] = R[r][0] * dx;
        m[r * 4 + 1] = R[r][1] * dy;
        m[r * 4 + 2] = R[r][2] * dz;
        m[r * 4 + 3] = qoffset[r];
      }
      return m;
    }
    default:
      for (int i = 0; i < 3; ++i) m[i * 4 + i] = (pixdim.size() > static_cast<std::size_t>(i)) ? pixdim[i] : 1.0;
      return m;
  }
}

Header read_header(const ByteSource& src) {
  if (src.size() < 348) throw format_error("NIfTI: fewer than 348 bytes, not a NIfTI header");
  unsigned char s4[4];
  src.read(0, 4, s4);
  bool little = true;
  int version = 0;
  for (int pass = 0; pass < 2 && !version; ++pass) {
    const bool le = pass == 0;
    const std::int32_t sz = load_i32(s4, le);
    if (sz == 348) { version = 1; little = le; }
    else if (sz == 540) { version = 2; little = le; }
  }
  if (!version) throw format_error("NIfTI: the first field is neither 348 (NIfTI-1) nor 540 (NIfTI-2): not a NIfTI header (NIfTI-2 and Analyze 7.5 aside)");
  const std::size_t hsize = version == 1 ? 348 : 540;
  if (src.size() < hsize) throw format_error("NIfTI: truncated header");
  const std::vector<unsigned char> h = src.bytes(0, hsize);
  const unsigned char* p = h.data();

  Header o;
  o.version = version;
  o.little_endian = little;
  auto i16 = [&](std::size_t off) { return load_i16(p + off, little); };
  auto i32 = [&](std::size_t off) { return load_i32(p + off, little); };
  auto i64 = [&](std::size_t off) { return load_i64(p + off, little); };
  auto f32 = [&](std::size_t off) { return static_cast<double>(load_f32(p + off, little)); };
  auto f64 = [&](std::size_t off) { return load_f64(p + off, little); };

  if (version == 1) {
    const std::string magic = trim_nul(p + 344, 4);
    if (magic == "n+1") o.single_file = true;
    else if (magic == "ni1") o.single_file = false;
    else throw format_error("NIfTI-1: the magic is \"" + magic + "\", not n+1 or ni1 (an Analyze 7.5 file is not NIfTI)");
    o.dim_info = p[39];
    const int nd = i16(40);
    if (nd < 1 || nd > 7) throw format_error("NIfTI-1: dim[0] = " + std::to_string(nd) + " is not 1 to 7 (a byte-swapped or damaged header)");
    for (int i = 1; i <= nd; ++i) o.shape.push_back(i16(40 + 2 * i));
    o.intent_p1 = f32(56); o.intent_p2 = f32(60); o.intent_p3 = f32(64);
    o.intent_code = i16(68);
    o.datatype = i16(70);
    o.bitpix = i16(72);
    o.slice_start = i16(74);
    o.qfac = f32(76);
    for (int i = 1; i <= nd; ++i) o.pixdim.push_back(f32(76 + 4 * i));
    o.vox_offset = static_cast<std::uint64_t>(std::max(0.0, f32(108)));
    o.scl_slope = f32(112); o.scl_inter = f32(116);
    o.slice_end = i16(120);
    o.slice_code = p[122];
    o.xyzt_units = p[123];
    o.cal_max = f32(124); o.cal_min = f32(128);
    o.slice_duration = f32(132); o.toffset = f32(136);
    o.descrip = trim_nul(p + 148, 80);
    o.aux_file = trim_nul(p + 228, 24);
    o.qform_code = i16(252); o.sform_code = i16(254);
    o.quatern_b = f32(256); o.quatern_c = f32(260); o.quatern_d = f32(264);
    o.qoffset = {f32(268), f32(272), f32(276)};
    for (int i = 0; i < 12; ++i) o.srow[i] = f32(280 + 4 * i);
    o.intent_name = trim_nul(p + 328, 16);
  } else {
    static const unsigned char magic2[8] = {'n', '+', '2', 0, '\r', '\n', 0x1a, '\n'};
    static const unsigned char magic2i[8] = {'n', 'i', '2', 0, '\r', '\n', 0x1a, '\n'};
    if (std::memcmp(p + 4, magic2, 8) == 0) o.single_file = true;
    else if (std::memcmp(p + 4, magic2i, 8) == 0) o.single_file = false;
    else throw format_error("NIfTI-2: the magic (with its newline check bytes) is damaged, as after a text-mode transfer");
    o.datatype = i16(12);
    o.bitpix = i16(14);
    const std::int64_t nd = i64(16);
    if (nd < 1 || nd > 7) throw format_error("NIfTI-2: dim[0] = " + std::to_string(nd) + " is not 1 to 7");
    for (int i = 1; i <= nd; ++i) o.shape.push_back(i64(16 + 8 * i));
    o.intent_p1 = f64(80); o.intent_p2 = f64(88); o.intent_p3 = f64(96);
    o.qfac = f64(104);
    for (int i = 1; i <= nd; ++i) o.pixdim.push_back(f64(104 + 8 * i));
    const std::int64_t vo = i64(168);
    if (vo < 0) throw format_error("NIfTI-2: a negative vox_offset");
    o.vox_offset = static_cast<std::uint64_t>(vo);
    o.scl_slope = f64(176); o.scl_inter = f64(184);
    o.cal_max = f64(192); o.cal_min = f64(200);
    o.slice_duration = f64(208); o.toffset = f64(216);
    o.slice_start = i64(224); o.slice_end = i64(232);
    o.descrip = trim_nul(p + 240, 80);
    o.aux_file = trim_nul(p + 320, 24);
    o.qform_code = i32(344); o.sform_code = i32(348);
    o.quatern_b = f64(352); o.quatern_c = f64(360); o.quatern_d = f64(368);
    o.qoffset = {f64(376), f64(384), f64(392)};
    for (int i = 0; i < 12; ++i) o.srow[i] = f64(400 + 8 * i);
    o.slice_code = i32(496);
    o.xyzt_units = i32(500);
    o.intent_code = i32(504);
    o.intent_name = trim_nul(p + 508, 16);
    o.dim_info = p[524];
  }
  for (std::int64_t d : o.shape) if (d < 0) throw format_error("NIfTI: a negative dimension");
  if (!(o.scl_slope == o.scl_slope) || !(o.scl_inter == o.scl_inter)) { o.scl_slope = 1.0; o.scl_inter = 0.0; }   // NaN: no scaling

  // extensions: four extender bytes, then (esize, ecode, data) records, up to vox_offset (single files only)
  if (o.single_file && src.size() >= hsize + 4 && o.vox_offset > hsize + 4) {
    unsigned char ext[4];
    src.read(hsize, 4, ext);
    if (ext[0] != 0) {
      std::uint64_t pos = hsize + 4;
      while (pos + 8 <= o.vox_offset && pos + 8 <= src.size()) {
        unsigned char eh[8];
        src.read(pos, 8, eh);
        const std::int32_t esize = load_i32(eh, little), ecode = load_i32(eh + 4, little);
        if (esize < 16 || esize % 16 != 0 || pos + static_cast<std::uint64_t>(esize) > o.vox_offset) break;   // not a valid chain: stop, keep what parsed
        Extension e;
        e.code = ecode;
        e.bytes = src.bytes(pos + 8, static_cast<std::size_t>(esize) - 8);
        o.extensions.push_back(std::move(e));
        pos += static_cast<std::uint64_t>(esize);
      }
    }
  }
  return o;
}

namespace {

Volume decode(const Header& h, const ByteSource& data, std::uint64_t offset) {
  TypeInfo t;
  if (!type_info(h.datatype, t))
    throw refused("NIfTI: the datatype " + datatype_name(h.datatype) + " is not read (supported: integers of 8 to 64 bits, float32, float64); complex, RGB and 128-bit types are refused");
  if (h.bitpix != t.bytes * 8) throw format_error("NIfTI: bitpix " + std::to_string(h.bitpix) + " does not match datatype " + std::to_string(h.datatype));
  const std::uint64_t n = h.n_voxels();
  if (offset > data.size() || n > (data.size() - offset) / static_cast<std::uint64_t>(t.bytes))
    throw format_error("NIfTI: the data are truncated: " + std::to_string(n) + " voxels of " + std::to_string(t.bytes) + " bytes do not fit after offset " + std::to_string(offset) + " in " + std::to_string(data.size()) + " bytes");
  const std::vector<unsigned char> raw = data.bytes(offset, static_cast<std::size_t>(n) * t.bytes);
  const bool scaled = h.scl_slope != 0.0 && (h.scl_slope != 1.0 || h.scl_inter != 0.0);
  const double slope = scaled ? h.scl_slope : 1.0, inter = scaled ? h.scl_inter : 0.0;
  Volume v;
  v.header = h;
  const bool wide = t.wide;
  v.precision = wide ? Precision::float64 : Precision::float32;
  if (wide) v.f64.resize(n); else v.f32.resize(n);
  for (std::uint64_t i = 0; i < n; ++i) {
    double x = stored_value(raw.data() + i * t.bytes, h.datatype, h.little_endian);
    if (scaled) x = x * slope + inter;
    if (wide) v.f64[i] = x; else v.f32[i] = static_cast<float>(x);
  }
  return v;
}

}  // namespace

Volume read_volume(const ByteSource& src) {
  Header h = read_header(src);
  if (!h.single_file) throw refused("NIfTI: this header says its data are in a separate .img file (magic ni1/ni2): give both the .hdr and the .img");
  return decode(h, src, h.vox_offset);
}

Volume read_volume(const ByteSource& header_src, const ByteSource& data_src) {
  Header h = read_header(header_src);
  return decode(h, data_src, h.single_file ? h.vox_offset : 0);
}

namespace {
bool ends_with(const std::string& s, const std::string& suf) { return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0; }
}  // namespace

Volume read_file(const std::string& path) {
  auto load = [](const std::string& p) -> std::vector<unsigned char> {
    FileSource f(p);
    std::vector<unsigned char> b = f.bytes(0, static_cast<std::size_t>(f.size()));
    if (b.size() >= 2 && b[0] == 0x1f && b[1] == 0x8b) b = inflate_bytes(b.data(), b.size(), DeflateWrapper::gzip);
    return b;
  };
  std::string base = path;
  if (ends_with(base, ".gz")) base.resize(base.size() - 3);
  if (ends_with(base, ".hdr") || ends_with(base, ".img")) {
    const std::string stem = base.substr(0, base.size() - 4);
    const bool gz = ends_with(path, ".gz");
    MemorySource hdr(load(stem + ".hdr" + (gz ? ".gz" : ""))), img(load(stem + ".img" + (gz ? ".gz" : "")));
    return read_volume(hdr, img);
  }
  MemorySource m(load(path));
  return read_volume(m);
}

Header make_header(int version, std::vector<std::int64_t> shape, std::vector<double> voxel_size, const std::array<double, 12>& sform) {
  Header h;
  h.version = version;
  h.datatype = 16;
  h.bitpix = 32;
  h.shape = std::move(shape);
  h.pixdim = std::move(voxel_size);
  h.pixdim.resize(h.shape.size(), 1.0);
  h.xyzt_units = 2;   // mm
  h.sform_code = 1;
  h.srow = sform;
  h.qform_code = 0;
  h.vox_offset = version == 1 ? 352 : 544;
  return h;
}

namespace {

template <class T>
void write_impl(ByteSink& sink, const Header& h, const std::vector<T>& values) {
  if (h.version != 1 && h.version != 2) throw range_error("NIfTI: version must be 1 or 2");
  TypeInfo t;
  if (!type_info(h.datatype, t)) throw refused("NIfTI writer: datatype " + datatype_name(h.datatype) + " is not written");
  if (h.shape.empty() || h.shape.size() > 7) throw range_error("NIfTI writer: 1 to 7 dimensions");
  if (values.size() != h.n_voxels()) throw range_error("NIfTI writer: " + std::to_string(values.size()) + " values for shape of " + std::to_string(h.n_voxels()) + " voxels");
  const std::size_t hsize = h.version == 1 ? 348 : 540;
  std::size_t ext_bytes = 0;
  for (const Extension& e : h.extensions) {
    std::size_t sz = 8 + e.bytes.size();
    sz = (sz + 15) / 16 * 16;
    ext_bytes += sz;
  }
  const std::uint64_t vox_offset = hsize + 4 + ext_bytes;
  std::vector<unsigned char> b(hsize, 0);
  unsigned char* p = b.data();
  const bool le = true;
  const int nd = static_cast<int>(h.shape.size());
  auto w16 = [&](std::size_t o, std::int64_t v) { store_u16(p + o, static_cast<std::uint16_t>(v), le); };
  auto w32 = [&](std::size_t o, std::int64_t v) { store_u32(p + o, static_cast<std::uint32_t>(v), le); };
  auto w64 = [&](std::size_t o, std::int64_t v) { store_u64(p + o, static_cast<std::uint64_t>(v), le); };
  auto wf = [&](std::size_t o, double v) { store_f32(p + o, static_cast<float>(v), le); };
  auto wd = [&](std::size_t o, double v) { store_f64(p + o, v, le); };
  auto str = [&](std::size_t o, const std::string& s, std::size_t n) { std::memcpy(p + o, s.data(), std::min(s.size(), n - 1)); };
  auto pd = [&](int i) { return i < static_cast<int>(h.pixdim.size()) ? h.pixdim[i] : 1.0; };
  if (h.version == 1) {
    w32(0, 348);
    p[39] = static_cast<unsigned char>(h.dim_info);
    w16(40, nd);
    for (int i = 0; i < 7; ++i) w16(42 + 2 * i, i < nd ? h.shape[i] : 1);
    wf(56, h.intent_p1); wf(60, h.intent_p2); wf(64, h.intent_p3);
    w16(68, h.intent_code); w16(70, h.datatype); w16(72, h.bitpix); w16(74, h.slice_start);
    wf(76, h.qfac < 0 ? -1.0 : 1.0);
    for (int i = 0; i < 7; ++i) wf(80 + 4 * i, i < nd ? pd(i) : 1.0);
    wf(108, static_cast<double>(vox_offset));
    wf(112, h.scl_slope); wf(116, h.scl_inter);
    w16(120, h.slice_end); p[122] = static_cast<unsigned char>(h.slice_code); p[123] = static_cast<unsigned char>(h.xyzt_units);
    wf(124, h.cal_max); wf(128, h.cal_min); wf(132, h.slice_duration); wf(136, h.toffset);
    str(148, h.descrip, 80); str(228, h.aux_file, 24);
    w16(252, h.qform_code); w16(254, h.sform_code);
    wf(256, h.quatern_b); wf(260, h.quatern_c); wf(264, h.quatern_d);
    wf(268, h.qoffset[0]); wf(272, h.qoffset[1]); wf(276, h.qoffset[2]);
    for (int i = 0; i < 12; ++i) wf(280 + 4 * i, h.srow[i]);
    str(328, h.intent_name, 16);
    std::memcpy(p + 344, "n+1", 4);
  } else {
    w32(0, 540);
    static const unsigned char magic2[8] = {'n', '+', '2', 0, '\r', '\n', 0x1a, '\n'};
    std::memcpy(p + 4, magic2, 8);
    w16(12, h.datatype); w16(14, h.bitpix);
    w64(16, nd);
    for (int i = 0; i < 7; ++i) w64(24 + 8 * i, i < nd ? h.shape[i] : 1);
    wd(80, h.intent_p1); wd(88, h.intent_p2); wd(96, h.intent_p3);
    wd(104, h.qfac < 0 ? -1.0 : 1.0);
    for (int i = 0; i < 7; ++i) wd(112 + 8 * i, i < nd ? pd(i) : 1.0);
    w64(168, static_cast<std::int64_t>(vox_offset));
    wd(176, h.scl_slope); wd(184, h.scl_inter); wd(192, h.cal_max); wd(200, h.cal_min);
    wd(208, h.slice_duration); wd(216, h.toffset);
    w64(224, h.slice_start); w64(232, h.slice_end);
    str(240, h.descrip, 80); str(320, h.aux_file, 24);
    w32(344, h.qform_code); w32(348, h.sform_code);
    wd(352, h.quatern_b); wd(360, h.quatern_c); wd(368, h.quatern_d);
    wd(376, h.qoffset[0]); wd(384, h.qoffset[1]); wd(392, h.qoffset[2]);
    for (int i = 0; i < 12; ++i) wd(400 + 8 * i, h.srow[i]);
    w32(496, h.slice_code); w32(500, h.xyzt_units); w32(504, h.intent_code);
    str(508, h.intent_name, 16);
    p[524] = static_cast<unsigned char>(h.dim_info);
  }
  sink.write(b);
  unsigned char ext[4] = {static_cast<unsigned char>(h.extensions.empty() ? 0 : 1), 0, 0, 0};
  sink.write(ext, 4);
  for (const Extension& e : h.extensions) {
    std::size_t sz = (8 + e.bytes.size() + 15) / 16 * 16;
    std::vector<unsigned char> rec(sz, 0);
    store_u32(rec.data(), static_cast<std::uint32_t>(sz), le);
    store_u32(rec.data() + 4, static_cast<std::uint32_t>(e.code), le);
    std::copy(e.bytes.begin(), e.bytes.end(), rec.begin() + 8);
    sink.write(rec);
  }

  const bool scaled = h.scl_slope != 0.0 && (h.scl_slope != 1.0 || h.scl_inter != 0.0);
  const double slope = scaled ? h.scl_slope : 1.0, inter = scaled ? h.scl_inter : 0.0;
  std::vector<unsigned char> blk;
  const std::size_t per = 1 << 16;
  for (std::size_t i0 = 0; i0 < values.size(); i0 += per) {
    const std::size_t m = std::min(per, values.size() - i0);
    blk.assign(m * t.bytes, 0);
    for (std::size_t k = 0; k < m; ++k) {
      double x = static_cast<double>(values[i0 + k]);
      unsigned char* q = blk.data() + k * t.bytes;
      if (t.is_float) {
        const double s = scaled ? (x - inter) / slope : x;
        if (t.bytes == 4) store_f32(q, static_cast<float>(s), le); else store_f64(q, s, le);
        continue;
      }
      double s = scaled ? std::nearbyint((x - inter) / slope) : std::nearbyint(x);
      double lo, hi;
      const int bits = t.bytes * 8;
      if (t.is_signed) { lo = -std::ldexp(1.0, bits - 1); hi = std::ldexp(1.0, bits - 1) - 1; } else { lo = 0; hi = std::ldexp(1.0, bits) - 1; }
      if (!(s >= lo && s <= hi)) throw range_error("NIfTI writer: value " + std::to_string(x) + " does not fit datatype " + std::to_string(h.datatype));
      if (t.is_signed) {
        const std::int64_t v = static_cast<std::int64_t>(s);
        switch (t.bytes) { case 1: q[0] = static_cast<unsigned char>(v); break; case 2: store_u16(q, static_cast<std::uint16_t>(v), le); break;
                           case 4: store_u32(q, static_cast<std::uint32_t>(v), le); break; default: store_u64(q, static_cast<std::uint64_t>(v), le); }
      } else {
        const std::uint64_t v = static_cast<std::uint64_t>(s);
        switch (t.bytes) { case 1: q[0] = static_cast<unsigned char>(v); break; case 2: store_u16(q, static_cast<std::uint16_t>(v), le); break;
                           case 4: store_u32(q, static_cast<std::uint32_t>(v), le); break; default: store_u64(q, v, le); }
      }
    }
    sink.write(blk);
  }
}

}  // namespace

void write_volume(ByteSink& sink, const Header& header, const std::vector<float>& values) { write_impl(sink, header, values); }
void write_volume(ByteSink& sink, const Header& header, const std::vector<double>& values) { write_impl(sink, header, values); }

void write_file(const std::string& path, const Header& header, const std::vector<double>& values) {
  MemorySink m;
  write_volume(m, header, values);
  FileSink f(path);
  if (ends_with(path, ".gz")) f.write(deflate_bytes(m.data().data(), m.data().size(), DeflateWrapper::gzip));
  else f.write(m.data());
  f.close();
}

}  // namespace nxr::formats::nifti
