// FreeSurfer / MNE source estimate files, `-lh.stc` and `-rh.stc` (one hemisphere each): read and write, as plain objects.
//
// Layout (MNE-Python's `_write_stc`, big-endian throughout): tmin in milliseconds (float32), tstep in milliseconds (float32), the number of vertices (uint32), the vertex
// numbers (uint32 each), the number of time points (uint32), then the data in TIME-MAJOR order as float32: all vertices of the first time point, then of the second ...
// The file has no room for metadata (units, provenance): a producer keeps them in a sidecar. Times are exposed in seconds, as MNE reads them.
//
// `Writer` takes a file's header first and then one time point at a time, so a long series is written frame by frame, never held (the number of time points is stated up
// front, as the format's header needs it).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nxr/formats/byte_source.h"

namespace nxr::formats::stc {

struct Estimate {
  double tmin = 0;                          // seconds: the first time point
  double tstep = 0;                         // seconds between time points
  std::vector<std::uint32_t> vertices;      // vertex numbers (0-based, in the source space's surface)
  std::uint32_t n_times = 0;
  std::vector<float> data;                  // time-major [n_times x n_vertices]
  std::size_t n_vertices() const { return vertices.size(); }
  float at(std::size_t vertex, std::size_t time) const { return data[time * vertices.size() + vertex]; }
};

Estimate read(const ByteSource& src);
Estimate read_file(const std::string& path);
void write(ByteSink& sink, const Estimate& e);
void write_file(const std::string& path, const Estimate& e);

// A file written one time point at a time. The header goes out on construction; `frame` takes one value per vertex in the order of `vertices`; `finish` checks that exactly
// `n_times` frames were given (a short file is an error, never left behind silently).
class Writer {
 public:
  Writer(ByteSink& sink, double tmin_s, double tstep_s, const std::vector<std::uint32_t>& vertices, std::uint32_t n_times);
  void frame(const float* values);
  void finish();
  std::uint32_t written() const { return written_; }

 private:
  ByteSink& sink_;
  std::size_t n_vertices_;
  std::uint32_t n_times_, written_ = 0;
};

}  // namespace nxr::formats::stc
