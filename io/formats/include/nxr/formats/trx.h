// TRX (https://github.com/tee-ar-ex/trx-spec): a tractography file, a ZIP archive (stored, so its arrays can be memory-mapped) of
//
//   header.json                     {"DIMENSIONS": [x, y, z], "NB_STREAMLINES": N, "NB_VERTICES": M, "VOXEL_TO_RASMM": 4x4 row-major}   of the anatomy it was tracked on
//   positions.3.<dtype>             [M x 3] float16 | float32 | float64: every streamline's points, RAS+ mm (world coordinates), streamline after streamline
//   offsets.<dtype>                 uint32 | uint64 (or int32 | int64): the index of each streamline's first point; N values (the last streamline ends at M) or N + 1 (the last is M)
//   dpv/ dps/ groups/ dpg/          data per vertex, per streamline, groups of streamlines and data per group: NOT read (named in `other_entries`)
//
// Pure: a ByteSource in, the streamlines out (float32, the same layout as the TrackVis reader's `Tractogram`); the arrays are streamed out of the archive, never held twice. Refused:
// a missing header, positions or offsets, counts that disagree, offsets that decrease or run past the vertices, a dtype that is not listed. Little-endian, as the spec says.
// There is no reference reader here (nibabel does not read TRX): the tests read files this writer makes and files written byte by byte.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "nxr/formats/byte_source.h"

namespace nxr::formats::trx {

struct Header {
  std::array<int, 3> dimensions{};
  std::int64_t n_streamlines = 0, n_vertices = 0;
  std::array<double, 16> voxel_to_rasmm{};   // row-major 4x4
  nlohmann::json raw;                        // the whole file
};

struct Streamlines {
  Header header;
  std::vector<float> points;                 // [n_vertices x 3], RAS+ mm
  std::vector<std::int64_t> offsets;         // n_streamlines + 1: streamline s is points offsets[s] .. offsets[s+1]
  std::string positions_dtype, offsets_dtype;
  std::vector<std::string> other_entries;    // the archive's other files (dpv/, dps/, groups/, dpg/), not read
};

Streamlines read(std::shared_ptr<const ByteSource> source);
Streamlines read_file(const std::string& path);

struct WriteOptions {
  std::string positions_dtype = "float32";   // float16 | float32 | float64
  std::string offsets_dtype = "uint64";      // uint32 | uint64
  bool offsets_with_end = true;              // N + 1 offsets, the last the vertex count
};
// A TRX archive of the streamlines (the header's counts are set from them; its anatomy fields are the caller's). Positions are narrowed to the dtype asked for.
void write(ByteSink& sink, const Streamlines& s, const WriteOptions& options = {});

// IEEE half precision <-> single, exact for every half (exposed for the tests)
float half_to_float(std::uint16_t h);
std::uint16_t float_to_half(float f);   // round to nearest even; overflow to infinity

}  // namespace nxr::formats::trx
