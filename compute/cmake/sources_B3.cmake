# Sources of nxr_compute owned by refactor batch B3 (selection + select + measurement + quantity), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/region.cpp
  src/select/intervals.cpp
  src/select/regions.cpp
  src/select/extent.cpp
  src/select/scale.cpp
  src/select/risefall.cpp
  src/select/cores.cpp
  src/select/tracks.cpp
  src/select/crossings.cpp
  src/select/scalespace.cpp
  src/integrals.cpp
  src/select/ladder.cpp
  src/selection/selection.cpp
  src/select/selection_partition.cpp
  src/select/selection_dyadic.cpp
  src/select/selection_bands.cpp
  src/select/selection_time_rule.cpp
  src/quantity/quantity.cpp
  src/select/selection_constant_q.cpp
  src/selection/selection_resolved.cpp
  src/select/selection_grid.cpp
  src/measurement/measurement_measures.cpp
  src/measurement/measurement_reduce.cpp
  src/measurement/measurement_roll_up.cpp
  src/measurement/measurement_summarize.cpp
  src/measurement/measurement_estimate.cpp
)

# ladder.cpp is a port of the Studio's ladder.ts (no FMA in JavaScript): contraction off, so areas, distances and the equal-mass cuts tie as they do there
set_source_files_properties(src/select/ladder.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(src/select/selection_dyadic.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # the Studio's ladder.ts, moved here
# measurement_measures.cpp / measurement_reduce.cpp are the Studio's measureOverPartition (no FMA in JavaScript): contraction off, so the sums equal the Studio's to the last bit
set_source_files_properties(src/measurement/measurement_measures.cpp src/measurement/measurement_reduce.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # the same arithmetic, moved into the measurement module
