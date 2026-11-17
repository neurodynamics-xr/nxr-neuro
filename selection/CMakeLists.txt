# nxr_selection — SELECTIONS AND PARTITIONS OF MANIFOLDS (docs/specs/2026-11-17-selection-module.md): a partition of a manifold by a method, below io and
# compute, which both call on it. It depends on the standard library and Eigen (header-only) alone.
add_library(nxr_selection
  src/partition.cpp
  src/dyadic.cpp
)
# dyadic.cpp is a port of the Studio's ladder.ts (no FMA in JavaScript): contraction off, so areas, distances and the equal-mass cuts tie as they do there
set_source_files_properties(src/dyadic.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_include_directories(nxr_selection PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(nxr_selection PRIVATE Eigen3::Eigen)
target_compile_features(nxr_selection PUBLIC cxx_std_17)
set_target_properties(nxr_selection PROPERTIES POSITION_INDEPENDENT_CODE ON)
