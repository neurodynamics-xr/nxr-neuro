# Sources of nxr_compute owned by refactor batch B4 (core + optimisation + stats + synthesis + meeg (compute part)), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

include(${CMAKE_CURRENT_LIST_DIR}/nxr_core.cmake)
target_link_libraries(nxr_compute PUBLIC nxr_core)

target_sources(nxr_compute PRIVATE
  src/meeg/montage.cpp
  src/meeg/connectome.cpp
  src/stats/lag.cpp
  src/select/distance_pairs.cpp
  src/stats/summary.cpp
  src/stats/histogram.cpp
  src/synthesis/lagfree.cpp
  src/stats/nulltest.cpp
  src/stats/pca.cpp
)
