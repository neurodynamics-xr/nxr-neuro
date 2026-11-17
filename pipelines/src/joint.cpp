// nxr-pipelines — the joint partition of space × time (joint.h).
#include "nxr/pipelines/joint.h"

#include "nxr/pipelines/task.h"

#include <cmath>

#include "nxr/io/finder.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/spatial_band.h"

namespace nxr::pipelines {

using namespace nxr::io;


int joint_bits(std::int64_t frames) {
  int b = 1;
  while ((std::int64_t(1) << b) < std::max<std::int64_t>(2, frames)) ++b;
  return b;
}


}  // namespace nxr::pipelines
