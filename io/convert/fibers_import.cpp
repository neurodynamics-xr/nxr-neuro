// Import of a TrackVis (.trk) fiber bundle into a package (nxr/convert/fibers.h).
#include "nxr/convert/fibers.h"

#include <cmath>

#include "nxr/convert/templates.h"
#include "nxr/formats/trx.h"
#include "nxr/io/errors.h"

namespace nxr::convert {

namespace code = nxr::io::code;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
}  // namespace

namespace {
// resample, transform and store a tractogram (the same for every file format); `format` names the file's own coordinates in the provenance
nxr::io::Json store_tractogram(nxr::io::SubjectWriter& sub, const Tractogram& t, const std::string& what, const std::string& file_name, const std::string& name,
                               const std::array<double, 12>& a, int points, const nxr::io::Json& origin, const std::string& comment, const std::string& read_step) {
  std::vector<std::int64_t> kept;
  const std::vector<float> res = resample_streamlines(t, points, &kept);
  const std::int64_t nf = static_cast<std::int64_t>(kept.size());
  if (nf == 0) refuse(what + ": " + file_name + " holds no usable streamline");
  nxr::io::fibers::Bundle<float> b;
  b.name = name;
  b.n_fibers = nf;
  b.points_per_fiber = points;
  b.points.resize(res.size());
  for (std::size_t i = 0; i < res.size() / 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      b.points[3 * i + j] = static_cast<float>(static_cast<double>(res[3 * i]) * a[j] + static_cast<double>(res[3 * i + 1]) * a[3 + j] + static_cast<double>(res[3 * i + 2]) * a[6 + j] + a[9 + j]);
  b.source = file_name;
  b.comment = comment;
  b.origin = origin;
  b.producer = nxr::io::Json{{"processing", nxr::io::Json::array({read_step, "RAS mm to the package's SCS metres by the caller's affine",
                                                                    "each streamline resampled to " + std::to_string(points) + " points at equal arc length (its two ends kept exactly)"})},
                             {"input_streamlines", t.n_streamlines()}, {"input_points", static_cast<std::int64_t>(t.points.size() / 3)}};
  const auto s = nxr::io::fibers::store_bundle<float>(sub, b);
  return nxr::io::Json{{"name", s.name}, {"id", s.id}, {"n_fibers", nf}, {"n_dropped", t.n_streamlines() - nf}, {"points_per_fiber", points}, {"n_input_points", static_cast<std::int64_t>(t.points.size() / 3)}};
}
}  // namespace

nxr::io::Json import_trk(nxr::io::SubjectWriter& sub, const TrkImport& rq) {
  if (rq.name.empty()) refuse("trk import: name the bundle");
  if (!rq.to_scs_m) refuse("trk import: give the transform from the .trk's RAS mm to the package's SCS metres: it depends on the anatomy the tractography was made against and has no default");
  if (rq.points < 2) refuse("trk import: points per fiber is at least 2");
  const Tractogram t = read_trk(rq.trk);
  return store_tractogram(sub, t, "trk import", rq.trk.filename().string(), rq.name, *rq.to_scs_m, rq.points, rq.origin, rq.comment, "streamlines read from the TrackVis .trk (voxel-mm to RAS mm as nibabel reads them)");
}

nxr::io::Json import_trx(nxr::io::SubjectWriter& sub, const TrxImport& rq) {
  if (rq.name.empty()) refuse("trx import: name the bundle");
  if (!rq.to_scs_m) refuse("trx import: give the transform from the TRX's RAS mm to the package's SCS metres: it depends on the anatomy the tractography was made against and has no default");
  if (rq.points < 2) refuse("trx import: points per fiber is at least 2");
  const nxr::formats::trx::Streamlines f = nxr::formats::trx::read_file(rq.trx.string());
  Tractogram t;
  t.points = f.points;
  t.offsets = f.offsets;
  t.dimensions = f.header.dimensions;
  for (int c = 0; c < 3; ++c) {
    double n = 0;
    for (int r = 0; r < 3; ++r) n += f.header.voxel_to_rasmm[static_cast<std::size_t>(4 * r + c)] * f.header.voxel_to_rasmm[static_cast<std::size_t>(4 * r + c)];
    t.voxel_size[static_cast<std::size_t>(c)] = static_cast<float>(std::sqrt(n));
  }
  nxr::io::Json out = store_tractogram(sub, t, "trx import", rq.trx.filename().string(), rq.name, *rq.to_scs_m, rq.points, rq.origin, rq.comment,
                                       "streamlines read from the TRX archive (positions in RAS mm as the file has them: " + f.positions_dtype + ")");
  out["not_imported"] = f.other_entries;
  return out;
}

}  // namespace nxr::convert
