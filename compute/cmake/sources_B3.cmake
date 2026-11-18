# Sources of nxr_compute owned by refactor batch B3 (selection + select + measurement + quantity), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/region.cpp
  src/intervals.cpp
  src/regions.cpp
  src/extent.cpp
  src/scale.cpp
  src/risefall.cpp
  src/cores.cpp
  src/tracks.cpp
  src/crossings.cpp
  src/scaling.cpp
  src/scalespace.cpp
  src/integrals.cpp
  src/ladder.cpp
  src/selection.cpp
  src/selection_partition.cpp
  src/selection_dyadic.cpp
  src/selection_bands.cpp
  src/selection_time_rule.cpp
  src/quantity.cpp
  src/selection_constant_q.cpp
  src/selection_resolved.cpp
  src/selection_grid.cpp
  src/measurement_measures.cpp
  src/measurement_reduce.cpp
  src/measurement_roll_up.cpp
  src/measurement_summarize.cpp
  src/measurement_estimate.cpp
)

# ladder.cpp is a port of the Studio's ladder.ts (no FMA in JavaScript): contraction off, so areas, distances and the equal-mass cuts tie as they do there
set_source_files_properties(src/ladder.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(src/selection_dyadic.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # the Studio's ladder.ts, moved here
# measurement_measures.cpp / measurement_reduce.cpp are the Studio's measureOverPartition (no FMA in JavaScript): contraction off, so the sums equal the Studio's to the last bit
set_source_files_properties(src/measurement_measures.cpp src/measurement_reduce.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # the same arithmetic, moved into the measurement module
