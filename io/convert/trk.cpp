// TrackVis (.trk) reader and the streamline resampler of the HCP-1065 importer (io/convert).
//
// The reader follows the format's published header (TrackVis version 2: 1000 bytes) and reads the points into RAS+ mm the way nibabel's `nibabel.streamlines` does
// (written here from that behaviour, not ported): voxel-mm -> voxel (divide by the voxel size), the half-voxel corner shift, the change of orientation between the
// header's voxel order and the one the voxel-to-RAS matrix implies, then the matrix.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include "nxr/convert/templates.h"
#include "nxr/io/errors.h"

namespace nxr::convert {

namespace code = nxr::io::code;
namespace fs = std::filesystem;

namespace {

[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

template <class T>
T rd(const std::vector<char>& h, std::size_t off) {
  T v;
  std::memcpy(&v, h.data() + off, sizeof(T));
  return v;
}

struct Orient { int axis[3]; int flip[3]; };   // for voxel axis i: the RAS axis and the sign

// 'L' 'R' 'P' 'A' 'I' 'S' as nibabel's axcodes2ornt (labels (L,R), (P,A), (I,S))
Orient from_codes(const std::string& c) {
  Orient o{};
  for (int i = 0; i < 3; ++i) {
    switch (c[static_cast<std::size_t>(i)]) {
      case 'R': o.axis[i] = 0; o.flip[i] = 1; break;
      case 'L': o.axis[i] = 0; o.flip[i] = -1; break;
      case 'A': o.axis[i] = 1; o.flip[i] = 1; break;
      case 'P': o.axis[i] = 1; o.flip[i] = -1; break;
      case 'S': o.axis[i] = 2; o.flip[i] = 1; break;
      case 'I': o.axis[i] = 2; o.flip[i] = -1; break;
      default: refuse(std::string("the .trk's voxel order '") + c + "' is not three of L R A P S I");
    }
  }
  return o;
}
// the axis codes an affine implies (nibabel's aff2axcodes for an axis-aligned affine: each voxel axis goes along the RAS axis of its largest component)
Orient from_affine(const double a[4][4]) {
  Orient o{};
  bool used[3] = {false, false, false};
  for (int j = 0; j < 3; ++j) {
    int best = 0;
    for (int i = 1; i < 3; ++i) if (std::abs(a[i][j]) > std::abs(a[best][j])) best = i;
    const double main = std::abs(a[best][j]);
    for (int i = 0; i < 3; ++i) if (i != best && std::abs(a[i][j]) > 1e-3 * main) refuse("the .trk's voxel-to-RAS matrix is not axis-aligned: only axis-aligned voxel orders are read");
    if (used[best]) refuse("the .trk's voxel-to-RAS matrix sends two voxel axes along one RAS axis");
    used[best] = true;
    o.axis[j] = best;
    o.flip[j] = a[best][j] < 0 ? -1 : 1;
  }
  return o;
}

}  // namespace

Tractogram read_trk(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw nxr::io::Error(code::not_found, path.string() + ": cannot be opened");
  std::vector<char> h(1000);
  f.read(h.data(), 1000);
  if (f.gcount() != 1000) refuse(path.filename().string() + ": shorter than a TrackVis header (1000 bytes)");
  if (std::strncmp(h.data(), "TRACK", 5) != 0) refuse(path.filename().string() + ": not a TrackVis file (no TRACK id)");
  if (rd<std::int32_t>(h, 996) != 1000) refuse(path.filename().string() + ": the header size is not 1000 (a TrackVis file is read little-endian)");
  const std::int32_t version = rd<std::int32_t>(h, 992);
  if (version != 2) refuse(path.filename().string() + ": TrackVis version " + std::to_string(version) + ": only version 2 (with a voxel-to-RAS matrix) is read");
  Tractogram t;
  for (int i = 0; i < 3; ++i) {
    t.dimensions[static_cast<std::size_t>(i)] = rd<std::int16_t>(h, 6 + 2 * static_cast<std::size_t>(i));
    t.voxel_size[static_cast<std::size_t>(i)] = rd<float>(h, 12 + 4 * static_cast<std::size_t>(i));
  }
  const int n_scalars = rd<std::int16_t>(h, 36), n_properties = rd<std::int16_t>(h, 238);
  t.voxel_order = std::string(h.data() + 948, 3);
  double a[4][4];
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) a[i][j] = rd<float>(h, 440 + 4 * static_cast<std::size_t>(4 * i + j));
  if (a[3][3] == 0) refuse(path.filename().string() + ": the voxel-to-RAS matrix is all zero (an old TrackVis file): it is not read");
  for (int i = 0; i < 3; ++i) if (!(t.voxel_size[static_cast<std::size_t>(i)] > 0)) refuse(path.filename().string() + ": a voxel size is not positive");

  // the world affine (nibabel's get_affine_trackvis_to_rasmm): voxmm -> vox -> corner shift -> change of orientation -> matrix
  double aff[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  auto mul = [](const double x[4][4], const double y[4][4], double out[4][4]) {
    double r[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { r[i][j] = 0; for (int k = 0; k < 4; ++k) r[i][j] += x[i][k] * y[k][j]; }
    std::memcpy(out, r, sizeof(r));
  };
  double scale[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  for (int i = 0; i < 3; ++i) scale[i][i] = 1.0 / t.voxel_size[static_cast<std::size_t>(i)];
  double offset[4][4] = {{1, 0, 0, -0.5}, {0, 1, 0, -0.5}, {0, 0, 1, -0.5}, {0, 0, 0, 1}};
  mul(scale, aff, aff);
  mul(offset, aff, aff);
  {
    // ornt_transform(header's order, the matrix's) and inv_ornt_aff over the dimensions
    const Orient start = from_codes(t.voxel_order), end = from_affine(a);
    int out_axis[3], out_flip[3];
    for (int i = 0; i < 3; ++i) {
      int found = -1;
      for (int k = 0; k < 3; ++k) if (end.axis[k] == start.axis[i]) found = k;
      out_axis[i] = found;
      out_flip[i] = start.flip[i] * end.flip[found];
    }
    double reorder[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};   // np.eye(4)[[ornt axes..., 3], :]
    for (int i = 0; i < 3; ++i) reorder[i][out_axis[i]] = 1;
    double flip[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    for (int i = 0; i < 3; ++i) {
      const double center = -(static_cast<double>(t.dimensions[static_cast<std::size_t>(i)]) - 1) / 2.0;
      flip[i][i] = out_flip[i];
      flip[i][3] = out_flip[i] * center - center;
    }
    double M[4][4];
    mul(flip, reorder, M);
    mul(M, aff, aff);
  }
  mul(a, aff, aff);

  // the records: n (int32), n x (3 + n_scalars) floats, n_properties floats
  const std::size_t per_point = static_cast<std::size_t>(3 + n_scalars);
  t.offsets.push_back(0);
  std::vector<float> rec;
  for (;;) {
    std::int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (f.gcount() == 0) break;
    if (f.gcount() != 4 || n < 0) refuse(path.filename().string() + ": a truncated or corrupt record");
    rec.resize(static_cast<std::size_t>(n) * per_point + static_cast<std::size_t>(n_properties));
    f.read(reinterpret_cast<char*>(rec.data()), static_cast<std::streamsize>(rec.size() * sizeof(float)));
    if (static_cast<std::size_t>(f.gcount()) != rec.size() * sizeof(float)) refuse(path.filename().string() + ": a truncated record (streamline " + std::to_string(t.n_streamlines()) + ")");
    for (std::int32_t p = 0; p < n; ++p) {
      const float* q = rec.data() + static_cast<std::size_t>(p) * per_point;
      for (int i = 0; i < 3; ++i)
        t.points.push_back(static_cast<float>(aff[i][0] * q[0] + aff[i][1] * q[1] + aff[i][2] * q[2] + aff[i][3]));
    }
    t.offsets.push_back(static_cast<std::int64_t>(t.points.size() / 3));
  }
  return t;
}

std::vector<float> resample_streamlines(const Tractogram& t, int n_points, std::vector<std::int64_t>* kept) {
  if (n_points < 2) refuse("a streamline is resampled to at least 2 points");
  std::vector<float> out;
  if (kept) kept->clear();
  const std::size_t np = static_cast<std::size_t>(n_points);
  for (std::int64_t s = 0; s < t.n_streamlines(); ++s) {
    const std::int64_t a = t.offsets[static_cast<std::size_t>(s)], b = t.offsets[static_cast<std::size_t>(s) + 1];
    const std::int64_t n = b - a;
    if (n < 2) continue;
    auto P = [&](std::int64_t i, int k) { return static_cast<double>(t.points[static_cast<std::size_t>(3 * (a + i) + k)]); };
    std::vector<double> cum(static_cast<std::size_t>(n), 0.0);
    for (std::int64_t i = 1; i < n; ++i) {
      double d = 0;
      for (int k = 0; k < 3; ++k) d += (P(i, k) - P(i - 1, k)) * (P(i, k) - P(i - 1, k));
      cum[static_cast<std::size_t>(i)] = cum[static_cast<std::size_t>(i) - 1] + std::sqrt(d);
    }
    const double total = cum.back();
    if (!(total > 0)) continue;   // every point the same: no direction
    if (kept) kept->push_back(s);
    std::int64_t seg = 0;
    for (std::size_t j = 0; j < np; ++j) {
      const double target = total * static_cast<double>(j) / static_cast<double>(np - 1);
      while (seg + 2 < n && cum[static_cast<std::size_t>(seg) + 1] < target) ++seg;
      const double c0 = cum[static_cast<std::size_t>(seg)], c1 = cum[static_cast<std::size_t>(seg) + 1];
      const double u = c1 > c0 ? std::min(1.0, std::max(0.0, (target - c0) / (c1 - c0))) : 0.0;
      for (int k = 0; k < 3; ++k) out.push_back(static_cast<float>(P(seg, k) + u * (P(seg + 1, k) - P(seg, k))));
    }
  }
  return out;
}

}  // namespace nxr::convert
