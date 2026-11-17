// nxr::tower — the frame grid (frame_grid.h).
#include "nxr/frame_grid.h"

#include <stdexcept>

namespace nxr::tower {

nxr::selection::Selection frameGrid(const FrameTiling& t) {
  if (!(t.sfreq > 0) || t.count < 1) throw std::invalid_argument("frameGrid: the tiling has no frame");
  return nxr::selection::tiled(nxr::selection::Tiling{t.origin_s(), t.frame_s(), t.frame_s(), t.count}, "s");
}

nxr::selection::Intervals frameIntervals(const FrameTiling& t) {
  nxr::selection::Line line;
  line.unit = "s";
  line.count = t.samples;
  line.origin = 0.0;
  line.spacing = 1.0 / t.sfreq;
  return nxr::selection::resolve_intervals(frameGrid(t), line);
}

}  // namespace nxr::tower
