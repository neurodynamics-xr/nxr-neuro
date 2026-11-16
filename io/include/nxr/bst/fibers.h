// nxr-io — TRACTOGRAPHY: `nxr.convert.fibers`. A Brainstorm fibres surface (`tess_fibers_*.mat`) as a bundle of curves (a `curve` manifold in `fibre/`, path topology,
// nF components of nP points, fibre-major), the map of their ends onto the cortex (the operator `<fibers>_ends`, csr, one 1 per end), the connectome built from
// those ends (a sparse CSC field on the product cortex x cortex, `source-source/<fibers>_connectome`, D159) and Brainstorm's own region-level connectomes
// (`Scouts(k).Assignment`, per atlas) as MEASUREMENTS `streamlines` on that atlas's Selection (D158).
//
//   if (bst::is_fibers_file(f)) { Json info = bst::export_fibers(sub, f, "S01", "source/cortex_pial_low", anat / "tess_cortex_pial_low.mat", emit); }
#pragma once
#include <filesystem>
#include <string>

#include "nxr/bst/subject.h"

namespace nxr::bst {

// Whether a file name begins `tess_fibers` (a fibres file has `Points` and no `Vertices`/`Faces`).
bool is_fibers_file(const std::filesystem::path& path);

// The fibres, their ends on the cortex at `cortex_path` (`source/<surface>`, already a manifold of the subject, converted from `cortex_file`) and one connectome per atlas
// Brainstorm assigned. Returns {name, id, n_fibers, n_points, connectome, connectomes}. ValueError when `Points` is not [nFibers, nPoints, 3] or the cortex is not in the subject.
Json export_fibers(nxr::io::SubjectWriter& sub, const std::filesystem::path& fibers_file, const std::string& subject, const std::string& cortex_path,
                   const std::filesystem::path& cortex_file, const Emit& progress = nullptr);

}  // namespace nxr::bst
