# nxr_compute: the selection toolbox (compute/include/nxr/selection/, compute/src/selection/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/selection/selection.cpp
  src/selection/selection_resolved.cpp
  src/selection/selection_kernel.cpp
)

