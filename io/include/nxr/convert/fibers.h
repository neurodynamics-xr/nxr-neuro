// nxr-io — IMPORT a fiber bundle from a DSI Studio / TrackVis `.trk` into a subject or template package (io/convert; the reader is `read_trk`, nxr/convert/templates.h).
//
// The .trk's points are RAS+ mm in the scanner's world as nibabel reads them. A package's fibers are in its SCS, in metres, so the caller gives the transform: a choice of
// data (it depends on the anatomy the tractography was made against), never defaulted. Every streamline is resampled to a fixed number of points at equal arc length
// (the ends kept exactly) because a bundle is fiber-major with equal points per fiber. Storage is `nxr::io::fibers::store_bundle` (float32, as the file).
//
// TRX (`import_trx`) is the same, from a TRX archive (nxr/formats/trx.h): positions in RAS+ mm as the file has them, then the caller's transform and the resampling. The file's data per
// vertex, per streamline and its groups are not imported (the result lists them).
#pragma once
#include <array>
#include <filesystem>
#include <string>

#include "nxr/io/fibers.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct TrkImport {
  std::filesystem::path trk;                 // the .trk file
  std::string name;                          // the bundle's name in the package (required)
  std::optional<std::array<double, 12>> to_scs_m;   // the 4x3 affine (row-major, last row the translation) from the .trk's RAS mm to the package's SCS metres (required)
  int points = 64;                           // points per fiber after resampling (>= 2)
  nxr::io::Json origin;                      // the origin document (source, citation, licence, terms); null: none
  std::string comment;
};

struct TrxImport {
  std::filesystem::path trx;                 // the .trx file
  std::string name;                          // the bundle's name in the package (required)
  std::optional<std::array<double, 12>> to_scs_m;   // the 4x3 affine (row-major, last row the translation) from the TRX's RAS mm to the package's SCS metres (required)
  int points = 64;                           // points per fiber after resampling (>= 2)
  nxr::io::Json origin;                      // the origin document (source, citation, licence, terms); null: none
  std::string comment;
};

/** Reads, transforms, resamples and stores the bundle. Returns {name, id, n_fibers, n_dropped, points_per_fiber, n_input_points}. Refused (Error invalid_argument): no name, no
 *  transform, points < 2, a file with no usable streamline. */
nxr::io::Json import_trk(nxr::io::SubjectWriter& sub, const TrkImport& request);
/** The same for a TRX archive. Returns what `import_trk` returns, and `not_imported`: the archive's data per vertex, per streamline and groups, by name. */
nxr::io::Json import_trx(nxr::io::SubjectWriter& sub, const TrxImport& request);

}  // namespace nxr::convert
