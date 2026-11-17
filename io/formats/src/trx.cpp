// TRX: see nxr/formats/trx.h.
#include "nxr/formats/trx.h"

#include <cmath>
#include <cstring>

#include "nxr/formats/errors.h"
#include "nxr/formats/zip.h"

namespace nxr::formats::trx {

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else {   // a subnormal half: normalise
      int e = -1;
      std::uint32_t m = man;
      do { ++e; m <<= 1; } while (!(m & 0x400u));
      bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

std::uint16_t float_to_half(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  std::uint32_t man = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<std::uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));   // inf, nan
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);                                              // overflow
  if (exp <= 0) {                                                                                                // subnormal or zero
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    man |= 0x800000u;
    const int shift = 14 - exp;
    std::uint32_t h = man >> shift;
    const std::uint32_t rem = man & ((1u << shift) - 1u), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = (static_cast<std::uint32_t>(exp) << 10) | (man >> 13);
  const std::uint32_t rem = man & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;   // a carry into the exponent is the right rounding
  return static_cast<std::uint16_t>(sign | h);
}

namespace {
int dtype_size(const std::string& d) {
  if (d == "float16") return 2;
  if (d == "float32" || d == "uint32" || d == "int32") return 4;
  if (d == "float64" || d == "uint64" || d == "int64") return 8;
  return 0;
}

// `positions.3.float32` -> "float32" (the second dot's tail); `offsets.uint64` -> "uint64"
bool entry_dtype(const std::string& name, const std::string& stem, std::string& dtype) {
  if (name.compare(0, stem.size(), stem) != 0) return false;
  const auto dot = name.rfind('.');
  if (dot == std::string::npos || dot < stem.size() - 1) return false;
  dtype = name.substr(dot + 1);
  return true;
}

// the bytes of an array entry streamed through `fn(first_element, count, bytes)`, whole elements at a time
template <class F> void stream_elements(const zip::Reader& z, const zip::Entry& e, int size, F fn) {
  if (e.size % static_cast<std::uint64_t>(size)) throw format_error("trx: '" + e.name + "' is not a whole number of elements");
  std::vector<unsigned char> carry;
  std::uint64_t first = 0;
  z.stream(e, [&](const unsigned char* d, std::size_t n) {
    std::size_t at = 0;
    if (!carry.empty()) {   // complete the element split across two blocks
      while (at < n && carry.size() < static_cast<std::size_t>(size)) carry.push_back(d[at++]);
      if (carry.size() == static_cast<std::size_t>(size)) { fn(first, 1, carry.data()); ++first; carry.clear(); }
    }
    const std::size_t whole = (n - at) / static_cast<std::size_t>(size);
    if (whole) { fn(first, whole, d + at); first += whole; at += whole * static_cast<std::size_t>(size); }
    for (; at < n; ++at) carry.push_back(d[at]);
  });
}

template <class T> T load(const unsigned char* p) { T v; std::memcpy(&v, p, sizeof v); return v; }   // the host is little-endian (the format is; every supported platform is)

void check_header(const nlohmann::json& j, Header& h) {
  if (!j.is_object()) throw format_error("trx: header.json is not an object");
  for (const char* k : {"NB_STREAMLINES", "NB_VERTICES"})
    if (!j.contains(k) || !j[k].is_number_integer() || j[k].get<std::int64_t>() < 0) throw format_error(std::string("trx: header.json has no valid ") + k);
  h.n_streamlines = j["NB_STREAMLINES"].get<std::int64_t>();
  h.n_vertices = j["NB_VERTICES"].get<std::int64_t>();
  if (!j.contains("DIMENSIONS") || !j["DIMENSIONS"].is_array() || j["DIMENSIONS"].size() != 3) throw format_error("trx: header.json has no DIMENSIONS of three numbers");
  for (int i = 0; i < 3; ++i) h.dimensions[static_cast<std::size_t>(i)] = j["DIMENSIONS"][static_cast<std::size_t>(i)].get<int>();
  if (!j.contains("VOXEL_TO_RASMM") || !j["VOXEL_TO_RASMM"].is_array() || j["VOXEL_TO_RASMM"].size() != 4) throw format_error("trx: header.json has no 4x4 VOXEL_TO_RASMM");
  for (std::size_t r = 0; r < 4; ++r) {
    const auto& row = j["VOXEL_TO_RASMM"][r];
    if (!row.is_array() || row.size() != 4) throw format_error("trx: header.json has no 4x4 VOXEL_TO_RASMM");
    for (std::size_t c = 0; c < 4; ++c) h.voxel_to_rasmm[4 * r + c] = row[c].get<double>();
  }
  h.raw = j;
}
}  // namespace

Streamlines read(std::shared_ptr<const ByteSource> source) {
  const zip::Reader z = zip::Reader::open(std::move(source));
  Streamlines out;
  const zip::Entry* head = z.find("header.json");
  if (!head) throw format_error("trx: no header.json (not a TRX file)");
  nlohmann::json j;
  try {
    const auto b = z.read(*head, std::uint64_t(1) << 24);
    j = nlohmann::json::parse(b.begin(), b.end());
  } catch (const nlohmann::json::exception& e) {
    throw format_error(std::string("trx: header.json is not valid JSON: ") + e.what());
  }
  check_header(j, out.header);
  const zip::Entry *pos = nullptr, *off = nullptr;
  for (const auto& e : z.entries()) {
    if (e.is_directory() || e.name == "header.json") continue;
    std::string d;
    if (entry_dtype(e.name, "positions.3.", d) && e.name.find('/') == std::string::npos) { pos = &e; out.positions_dtype = d; }
    else if (entry_dtype(e.name, "offsets.", d) && e.name.find('/') == std::string::npos) { off = &e; out.offsets_dtype = d; }
    else out.other_entries.push_back(e.name);
  }
  if (!pos) throw format_error("trx: no positions.3.<dtype> array");
  if (!off) throw format_error("trx: no offsets.<dtype> array");
  const int psize = dtype_size(out.positions_dtype), osize = dtype_size(out.offsets_dtype);
  if (out.positions_dtype != "float16" && out.positions_dtype != "float32" && out.positions_dtype != "float64") throw refused("trx: positions of type '" + out.positions_dtype + "' (float16, float32 or float64)");
  if (out.offsets_dtype != "uint32" && out.offsets_dtype != "uint64" && out.offsets_dtype != "int32" && out.offsets_dtype != "int64") throw refused("trx: offsets of type '" + out.offsets_dtype + "' (uint32, uint64, int32 or int64)");
  const Header& h = out.header;
  if (pos->size != static_cast<std::uint64_t>(h.n_vertices) * 3u * static_cast<std::uint64_t>(psize))
    throw format_error("trx: the positions hold " + std::to_string(pos->size / static_cast<std::uint64_t>(psize)) + " values, NB_VERTICES = " + std::to_string(h.n_vertices) + " needs " + std::to_string(3 * h.n_vertices));
  const std::uint64_t n_off = off->size / static_cast<std::uint64_t>(osize);
  if (off->size % static_cast<std::uint64_t>(osize) || (n_off != static_cast<std::uint64_t>(h.n_streamlines) && n_off != static_cast<std::uint64_t>(h.n_streamlines) + 1))
    throw format_error("trx: " + std::to_string(n_off) + " offsets for NB_STREAMLINES = " + std::to_string(h.n_streamlines) + " (N or N + 1 are valid)");

  out.offsets.reserve(static_cast<std::size_t>(h.n_streamlines) + 1);
  stream_elements(z, *off, osize, [&](std::uint64_t, std::size_t n, const unsigned char* p) {
    for (std::size_t i = 0; i < n; ++i) {
      std::int64_t v;
      if (out.offsets_dtype == "uint32") v = load<std::uint32_t>(p + 4 * i);
      else if (out.offsets_dtype == "int32") v = load<std::int32_t>(p + 4 * i);
      else if (out.offsets_dtype == "int64") v = load<std::int64_t>(p + 8 * i);
      else {
        const std::uint64_t u = load<std::uint64_t>(p + 8 * i);
        if (u > static_cast<std::uint64_t>(INT64_MAX)) throw format_error("trx: an offset beyond 2^63");
        v = static_cast<std::int64_t>(u);
      }
      out.offsets.push_back(v);
    }
  });
  if (n_off == static_cast<std::uint64_t>(h.n_streamlines)) out.offsets.push_back(h.n_vertices);
  for (std::size_t s = 0; s + 1 < out.offsets.size(); ++s)
    if (out.offsets[s] < 0 || out.offsets[s + 1] < out.offsets[s]) throw format_error("trx: the offsets decrease at streamline " + std::to_string(s));
  if (out.offsets.back() != h.n_vertices) throw format_error("trx: the offsets end at " + std::to_string(out.offsets.back()) + ", NB_VERTICES = " + std::to_string(h.n_vertices));
  if (!out.offsets.empty() && out.offsets.front() != 0) throw format_error("trx: the first offset is " + std::to_string(out.offsets.front()) + ", not 0");

  out.points.resize(static_cast<std::size_t>(h.n_vertices) * 3);
  stream_elements(z, *pos, psize, [&](std::uint64_t first, std::size_t n, const unsigned char* p) {
    float* dst = out.points.data() + first;
    for (std::size_t i = 0; i < n; ++i) {
      if (out.positions_dtype == "float16") dst[i] = half_to_float(load<std::uint16_t>(p + 2 * i));
      else if (out.positions_dtype == "float32") dst[i] = load<float>(p + 4 * i);
      else dst[i] = static_cast<float>(load<double>(p + 8 * i));
    }
  });
  for (float v : out.points) if (!std::isfinite(v)) throw format_error("trx: a position is not a finite number");
  return out;
}

Streamlines read_file(const std::string& path) { return read(std::make_shared<FileSource>(path)); }

void write(ByteSink& sink, const Streamlines& s, const WriteOptions& o) {
  if (o.positions_dtype != "float16" && o.positions_dtype != "float32" && o.positions_dtype != "float64") throw refused("trx: positions of type '" + o.positions_dtype + "'");
  if (o.offsets_dtype != "uint32" && o.offsets_dtype != "uint64") throw refused("trx: offsets of type '" + o.offsets_dtype + "'");
  if (s.offsets.empty() || s.offsets.front() != 0 || s.offsets.back() * 3 != static_cast<std::int64_t>(s.points.size())) throw format_error("trx: offsets and points disagree (offsets are N + 1 values, from 0 to the vertex count)");
  nlohmann::json j = s.header.raw.is_object() ? s.header.raw : nlohmann::json::object();
  j["NB_STREAMLINES"] = static_cast<std::int64_t>(s.offsets.size()) - 1;
  j["NB_VERTICES"] = s.offsets.back();
  if (!j.contains("DIMENSIONS")) j["DIMENSIONS"] = {s.header.dimensions[0], s.header.dimensions[1], s.header.dimensions[2]};
  if (!j.contains("VOXEL_TO_RASMM")) {
    j["VOXEL_TO_RASMM"] = nlohmann::json::array();
    for (int r = 0; r < 4; ++r) j["VOXEL_TO_RASMM"].push_back({s.header.voxel_to_rasmm[static_cast<std::size_t>(4 * r)], s.header.voxel_to_rasmm[static_cast<std::size_t>(4 * r + 1)],
                                                              s.header.voxel_to_rasmm[static_cast<std::size_t>(4 * r + 2)], s.header.voxel_to_rasmm[static_cast<std::size_t>(4 * r + 3)]});
  }
  std::vector<zip::NewEntry> entries;
  const std::string hj = j.dump();
  entries.push_back({"header.json", std::vector<unsigned char>(hj.begin(), hj.end())});
  zip::NewEntry pe{"positions.3." + o.positions_dtype, {}};
  for (float v : s.points) {
    unsigned char b[8];
    std::size_t n;
    if (o.positions_dtype == "float16") { const std::uint16_t h = float_to_half(v); std::memcpy(b, &h, 2); n = 2; }
    else if (o.positions_dtype == "float32") { std::memcpy(b, &v, 4); n = 4; }
    else { const double d = v; std::memcpy(b, &d, 8); n = 8; }
    pe.data.insert(pe.data.end(), b, b + n);
  }
  entries.push_back(std::move(pe));
  zip::NewEntry oe{"offsets." + o.offsets_dtype, {}};
  const std::size_t n_off = o.offsets_with_end ? s.offsets.size() : s.offsets.size() - 1;
  for (std::size_t i = 0; i < n_off; ++i) {
    unsigned char b[8];
    std::size_t n;
    if (o.offsets_dtype == "uint32") { const std::uint32_t v = static_cast<std::uint32_t>(s.offsets[i]); std::memcpy(b, &v, 4); n = 4; }
    else { const std::uint64_t v = static_cast<std::uint64_t>(s.offsets[i]); std::memcpy(b, &v, 8); n = 8; }
    oe.data.insert(oe.data.end(), b, b + n);
  }
  entries.push_back(std::move(oe));
  zip::write(sink, entries);
}

}  // namespace nxr::formats::trx
