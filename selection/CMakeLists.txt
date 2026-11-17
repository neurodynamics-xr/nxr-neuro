# nxr_selection — SELECTIONS AND PARTITIONS OF MANIFOLDS (docs/specs/2026-11-17-selection-module.md): pure arithmetic of the parts of a manifold (grids, tilings,
# their members, their nesting), below io and compute, which both call on it. It depends on nothing but the standard library.
add_library(nxr_selection
  src/time_partition.cpp
)
target_include_directories(nxr_selection PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(nxr_selection PUBLIC cxx_std_17)
set_target_properties(nxr_selection PROPERTIES POSITION_INDEPENDENT_CODE ON)
