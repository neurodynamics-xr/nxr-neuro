# Sources of nxr_compute owned by refactor batch B4 (core + optimisation + stats + synthesis + meeg (compute part)), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/congruence.cpp
  src/montage.cpp
  src/connectome.cpp
  src/lag.cpp
  src/stats.cpp
  src/histogram.cpp
  src/synthesis_lagfree.cpp
  src/nulltest.cpp
  src/parallel.cpp
  src/pca.cpp
)
