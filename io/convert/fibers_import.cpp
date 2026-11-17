// Import of a TrackVis (.trk) fiber bundle into a package (nxr/convert/fibers.h).
#include "nxr/convert/fibers.h"

#include "nxr/convert/templates.h"
#include "nxr/io/errors.h"

namespace nxr::convert {

namespace code = nxr::io::code;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
}  // namespace

nxr::io::Json import_trk(nxr::io::SubjectWriter& sub, const TrkImport& rq) {
  if (rq.name.empty()) refuse("trk import: name the bundle");
  if (!rq.to_scs_m) refuse("trk import: give the transform from the .trk's RAS mm to the package's SCS metres: it depends on the anatomy the tractography was made against and has no default");
  if (rq.points < 2) refuse("trk import: points per fiber is at least 2");
  const Tractogram t = read_trk(rq.trk);
  std::vector<std::int64_t> kept;
  const std::vector<float> res = resample_streamlines(t, rq.points, &kept);
  const std::int64_t nf = static_cast<std::int64_t>(kept.size());
  if (nf == 0) refuse("trk import: " + rq.trk.filename().string() + " holds no usable streamline");
  const auto& a = *rq.to_scs_m;
  nxr::io::fibers::Bundle<float> b;
  b.name = rq.name;
  b.n_fibers = nf;
  b.points_per_fiber = rq.points;
  b.points.resize(res.size());
  for (std::size_t i = 0; i < res.size() / 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      b.points[3 * i + j] = static_cast<float>(static_cast<double>(res[3 * i]) * a[j] + static_cast<double>(res[3 * i + 1]) * a[3 + j] + static_cast<double>(res[3 * i + 2]) * a[6 + j] + a[9 + j]);
  b.source = rq.trk.filename().string();
  b.comment = rq.comment;
  b.origin = rq.origin;
  b.producer = nxr::io::Json{{"processing", nxr::io::Json::array({"streamlines read from the TrackVis .trk (voxel-mm to RAS mm as nibabel reads them)", "RAS mm to the package's SCS metres by the caller's affine",
                                                                    "each streamline resampled to " + std::to_string(rq.points) + " points at equal arc length (its two ends kept exactly)"})},
                             {"input_streamlines", t.n_streamlines()}, {"input_points", static_cast<std::int64_t>(t.points.size() / 3)}};
  const auto s = nxr::io::fibers::store_bundle<float>(sub, b);
  return nxr::io::Json{{"name", s.name}, {"id", s.id}, {"n_fibers", nf}, {"n_dropped", t.n_streamlines() - nf}, {"points_per_fiber", rq.points}, {"n_input_points", static_cast<std::int64_t>(t.points.size() / 3)}};
}

}  // namespace nxr::convert
