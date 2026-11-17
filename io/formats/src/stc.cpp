// FreeSurfer / MNE `.stc` source estimates: see nxr/formats/stc.h.
#include "nxr/formats/stc.h"

#include <cmath>
#include <cstring>

#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"

namespace nxr::formats::stc {

namespace {
void put_f32(ByteSink& s, float v) { unsigned char b[4]; store_f32(b, v, false); s.write(b, 4); }
void put_u32(ByteSink& s, std::uint32_t v) { unsigned char b[4]; store_u32(b, v, false); s.write(b, 4); }
}  // namespace

Estimate read(const ByteSource& src) {
  if (src.size() < 16) throw format_error("an .stc file holds at least its 16 header bytes; this one has " + std::to_string(src.size()));
  Estimate e;
  unsigned char b[4];
  std::uint64_t off = 0;
  auto u32 = [&]() { src.read(off, 4, b); off += 4; return load_u32(b, false); };
  auto f32 = [&]() { src.read(off, 4, b); off += 4; return load_f32(b, false); };
  e.tmin = static_cast<double>(f32()) / 1000.0;
  e.tstep = static_cast<double>(f32()) / 1000.0;
  const std::uint32_t nv = u32();
  if (src.size() < off + 4ull * nv + 4) throw format_error("the .stc file is shorter than its " + std::to_string(nv) + " vertex numbers");
  e.vertices.resize(nv);
  for (auto& v : e.vertices) v = u32();
  e.n_times = u32();
  const std::uint64_t count = static_cast<std::uint64_t>(nv) * e.n_times;
  if (src.size() != off + 4 * count) throw format_error("the .stc file holds " + std::to_string((src.size() - off) / 4) + " values, its header says " + std::to_string(count));
  e.data.resize(count);
  std::vector<unsigned char> raw(count * 4);
  if (!raw.empty()) src.read(off, raw.size(), raw.data());
  for (std::uint64_t i = 0; i < count; ++i) e.data[i] = load_f32(raw.data() + 4 * i, false);
  return e;
}

Estimate read_file(const std::string& path) { FileSource s(path); return read(s); }

void write(ByteSink& sink, const Estimate& e) {
  if (e.data.size() != static_cast<std::size_t>(e.n_times) * e.vertices.size()) throw format_error("the .stc data hold " + std::to_string(e.data.size()) + " values for " + std::to_string(e.n_times) + " time points of " + std::to_string(e.vertices.size()) + " vertices");
  Writer w(sink, e.tmin, e.tstep, e.vertices, e.n_times);
  for (std::uint32_t t = 0; t < e.n_times; ++t) w.frame(e.data.data() + static_cast<std::size_t>(t) * e.vertices.size());
  w.finish();
}

void write_file(const std::string& path, const Estimate& e) {
  FileSink s(path);
  write(s, e);
  s.close();
}

Writer::Writer(ByteSink& sink, double tmin_s, double tstep_s, const std::vector<std::uint32_t>& vertices, std::uint32_t n_times) : sink_(sink), n_vertices_(vertices.size()), n_times_(n_times) {
  if (vertices.empty()) throw range_error("an .stc file has at least one vertex");
  for (std::size_t i = 1; i < vertices.size(); ++i) if (vertices[i] <= vertices[i - 1]) throw range_error("the vertex numbers of an .stc file are strictly increasing; vertex " + std::to_string(i) + " is not");
  if (!(tstep_s > 0) || !std::isfinite(tmin_s)) throw range_error("an .stc file needs a positive time step and a finite start time");
  put_f32(sink_, static_cast<float>(1000.0 * tmin_s));
  put_f32(sink_, static_cast<float>(1000.0 * tstep_s));
  put_u32(sink_, static_cast<std::uint32_t>(vertices.size()));
  for (const auto v : vertices) put_u32(sink_, v);
  put_u32(sink_, n_times);
}

void Writer::frame(const float* values) {
  if (written_ >= n_times_) throw range_error("more frames than the header's " + std::to_string(n_times_));
  std::vector<unsigned char> b(4 * n_vertices_);
  for (std::size_t i = 0; i < n_vertices_; ++i) store_f32(b.data() + 4 * i, values[i], false);
  sink_.write(b.data(), b.size());
  ++written_;
}

void Writer::finish() {
  if (written_ != n_times_) throw range_error("the .stc header says " + std::to_string(n_times_) + " time points but " + std::to_string(written_) + " were written");
}

}  // namespace nxr::formats::stc
