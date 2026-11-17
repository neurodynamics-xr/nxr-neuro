# nxr_pipelines — the thin layer where nxr_io and nxr_compute meet (read with nxr_io, compute with nxr_compute, write with nxr_io, cached).
# Included by ../CMakeLists.txt when both libraries are built. Neither library includes the other; they link into this one directly.

# nxr_compute's identity, recorded in every cached result: compute's version (bindings/package.json is the project's published version of it; none
# here, so the toolkit's) + this checkout's commit (read only; "-dirty" if compute's sources differ from it).
set(NXR_COMPUTE_VERSION "${PROJECT_VERSION}")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_SOURCE_DIR} rev-parse --short=10 HEAD OUTPUT_VARIABLE _rev OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _rc)
  if(_rc EQUAL 0)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_SOURCE_DIR} status --porcelain --untracked-files=no -- compute/src compute/include compute/CMakeLists.txt
                    OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(NXR_COMPUTE_VERSION "${NXR_COMPUTE_VERSION}+${_rev}")
    if(_dirty)
      set(NXR_COMPUTE_VERSION "${NXR_COMPUTE_VERSION}-dirty")
    endif()
  endif()
endif()
message(STATUS "nxr_pipelines: nxr_compute ${NXR_COMPUTE_VERSION}")

# nxr-neuro's own pipeline definitions (definitions/*.yaml), compiled in as {name, text} (definition.cpp `load` finds them by name)
file(GLOB NXR_DEFINITION_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/definitions/*.yaml)
set(_defs "")
foreach(f ${NXR_DEFINITION_FILES})
  get_filename_component(_n ${f} NAME_WE)
  file(READ ${f} _t)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${f})   # an edited definition re-embeds on the next build
  string(APPEND _defs "{\"${_n}\", R\"NXRDEF(${_t})NXRDEF\"},\n")
endforeach()
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gen/definitions.inc "${_defs}")

add_library(nxr_pipelines
  src/cache.cpp
  src/definition.cpp
  src/dispatch.cpp
  src/registry.cpp
  src/extent_steps.cpp
  src/cores_steps.cpp
  src/tracks_steps.cpp
  src/crossings_steps.cpp
  src/surrogate_steps.cpp
  src/run_repeat.cpp
  src/null_compute.cpp
  src/spatial_null.cpp
  src/pointspread.cpp
  src/wishart_null.cpp
  src/spatial_events_steps.cpp
  src/point_events.cpp
  src/runner.cpp
  src/input.cpp
  src/eigen.cpp
  src/frames.cpp
  src/registration.cpp
  src/dataset.cpp
  src/listing.cpp
  src/measure.cpp
  src/meeg.cpp
  src/spectral_record.cpp
  src/reading.cpp
  src/measure_time.cpp
  src/statistics.cpp
  src/patches.cpp
  src/spatial_band.cpp
  src/joint.cpp
  src/frame_tiling.cpp
  src/pyramid.cpp
  src/group.cpp
  src/cellquadrature.cpp
  src/scaleselection.cpp
  src/speedspectrum.cpp
  src/tiles.cpp
)
target_include_directories(nxr_pipelines PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(nxr_pipelines PUBLIC nxr_io nxr_compute nxr_meeg)
target_compile_definitions(nxr_pipelines PUBLIC NXR_COMPUTE_VERSION="${NXR_COMPUTE_VERSION}" NXR_NEURO_VERSION="${PROJECT_VERSION}")
if(MSVC)
  target_compile_options(nxr_pipelines PRIVATE /W4)
else()
  target_compile_options(nxr_pipelines PRIVATE -Wall -Wextra)
endif()

# the cache's own test (a key, a lookup, a hit and a miss on a synthetic dataset)
if(NXR_NEURO_BUILD_TESTS)
  add_executable(test_pipelines test/test_pipelines.cpp)
  target_link_libraries(test_pipelines PRIVATE nxr_pipelines)
  target_include_directories(test_pipelines PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  add_test(NAME test_pipelines COMMAND test_pipelines)
  add_executable(test_dataset_with_template test/test_dataset_with_template.cpp)
  target_link_libraries(test_dataset_with_template PRIVATE nxr_pipelines)
  target_include_directories(test_dataset_with_template PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  add_test(NAME test_dataset_with_template COMMAND test_dataset_with_template)
  add_executable(test_registration test/test_registration.cpp)
  target_link_libraries(test_registration PRIVATE nxr_pipelines)
  target_include_directories(test_registration PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  add_test(NAME test_registration COMMAND test_registration)
  add_executable(test_meeg_pipelines test/test_meeg_pipelines.cpp)
  target_link_libraries(test_meeg_pipelines PRIVATE nxr_pipelines)
  target_include_directories(test_meeg_pipelines PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  if(NXR_NEURO_BUILD_CLI)   # the commands too: the test runs the built nxr-neuro
    add_dependencies(test_meeg_pipelines nxr-neuro)
    target_compile_definitions(test_meeg_pipelines PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_meeg_pipelines COMMAND test_meeg_pipelines)
  add_executable(test_psd_pipeline test/test_psd_pipeline.cpp)
  target_link_libraries(test_psd_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_psd_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test)
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_psd_pipeline nxr-neuro)
    target_compile_definitions(test_psd_pipeline PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_psd_pipeline COMMAND test_psd_pipeline)
  add_executable(test_source_psd test/test_source_psd.cpp)
  target_link_libraries(test_source_psd PRIVATE nxr_pipelines)
  target_include_directories(test_source_psd PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  target_compile_definitions(test_source_psd PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/meeg/test/golden")
  add_test(NAME test_source_psd COMMAND test_source_psd)
  add_executable(test_package_pipelines test/test_package_pipelines.cpp)
  target_link_libraries(test_package_pipelines PRIVATE nxr_pipelines)
  target_include_directories(test_package_pipelines PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_package_pipelines COMMAND test_package_pipelines)
  add_executable(test_quadrature_pipeline test/test_quadrature_pipeline.cpp)
  target_link_libraries(test_quadrature_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_quadrature_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_quadrature_pipeline COMMAND test_quadrature_pipeline)
  add_executable(test_hodge_power test/test_hodge_power.cpp)
  target_link_libraries(test_hodge_power PRIVATE nxr_pipelines)
  target_include_directories(test_hodge_power PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_hodge_power COMMAND test_hodge_power)

  add_executable(test_speedspectrum_pipeline test/test_speedspectrum_pipeline.cpp)
  target_link_libraries(test_speedspectrum_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_speedspectrum_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_speedspectrum_pipeline COMMAND test_speedspectrum_pipeline)

  add_executable(test_opticalflow_pipeline test/test_opticalflow_pipeline.cpp)
  target_link_libraries(test_opticalflow_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_opticalflow_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_opticalflow_pipeline COMMAND test_opticalflow_pipeline)
  add_executable(test_frames_pipeline test/test_frames_pipeline.cpp)
  target_link_libraries(test_frames_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_frames_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_frames_pipeline COMMAND test_frames_pipeline)
  add_executable(test_tiles_pipeline test/test_tiles_pipeline.cpp)
  target_link_libraries(test_tiles_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_tiles_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_tiles_pipeline COMMAND test_tiles_pipeline)
  add_executable(test_source_tensor test/test_source_tensor.cpp)
  target_link_libraries(test_source_tensor PRIVATE nxr_pipelines)
  target_include_directories(test_source_tensor PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_source_tensor COMMAND test_source_tensor)
  add_executable(test_crossspectrum test/test_crossspectrum.cpp)
  target_link_libraries(test_crossspectrum PRIVATE nxr_pipelines)
  target_include_directories(test_crossspectrum PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_crossspectrum COMMAND test_crossspectrum)
  add_executable(test_phasegradient test/test_phasegradient.cpp)
  target_link_libraries(test_phasegradient PRIVATE nxr_pipelines)
  target_include_directories(test_phasegradient PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_phasegradient COMMAND test_phasegradient)
  add_executable(test_phasegradient_band test/test_phasegradient_band.cpp)
  target_link_libraries(test_phasegradient_band PRIVATE nxr_pipelines)
  target_include_directories(test_phasegradient_band PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_phasegradient_band COMMAND test_phasegradient_band)
  set_tests_properties(test_phasegradient_band PROPERTIES LABELS slow)   # minutes long: CI runs it on main and nightly, not on every pull request
  add_executable(test_cyclepattern test/test_cyclepattern.cpp)
  target_link_libraries(test_cyclepattern PRIVATE nxr_pipelines)
  target_include_directories(test_cyclepattern PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_cyclepattern COMMAND test_cyclepattern)
  set_tests_properties(test_cyclepattern PROPERTIES LABELS slow)   # minutes long: CI runs it on main and nightly, not on every pull request
  add_executable(test_measure_pipeline test/test_measure_pipeline.cpp)
  target_link_libraries(test_measure_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_measure_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_measure_pipeline COMMAND test_measure_pipeline)

  add_executable(test_patches_pipeline test/test_patches_pipeline.cpp)
  target_link_libraries(test_patches_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_patches_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_patches_pipeline COMMAND test_patches_pipeline)

  add_executable(test_cortical_cells test/test_cortical_cells.cpp)
  target_link_libraries(test_cortical_cells PRIVATE nxr_pipelines)
  target_include_directories(test_cortical_cells PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_cortical_cells COMMAND test_cortical_cells)

  add_executable(test_cortical_cell_pairs test/test_cortical_cell_pairs.cpp)
  target_link_libraries(test_cortical_cell_pairs PRIVATE nxr_pipelines)
  target_include_directories(test_cortical_cell_pairs PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_cortical_cell_pairs COMMAND test_cortical_cell_pairs)

  add_executable(test_cortical_cell_pairs_long test/test_cortical_cell_pairs_long.cpp)
  target_link_libraries(test_cortical_cell_pairs_long PRIVATE nxr_pipelines)
  target_include_directories(test_cortical_cell_pairs_long PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_cortical_cell_pairs_long COMMAND test_cortical_cell_pairs_long)

  add_executable(test_cellquadrature test/test_cellquadrature.cpp)
  target_link_libraries(test_cellquadrature PRIVATE nxr_pipelines)
  target_include_directories(test_cellquadrature PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_cellquadrature COMMAND test_cellquadrature)
  add_executable(test_scale_selection test/test_scale_selection.cpp)
  target_link_libraries(test_scale_selection PRIVATE nxr_pipelines)
  target_include_directories(test_scale_selection PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_scale_selection COMMAND test_scale_selection)

  add_executable(test_spatial_band test/test_spatial_band.cpp)
  target_link_libraries(test_spatial_band PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_band PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_band COMMAND test_spatial_band)
  add_executable(test_spatial_hat test/test_spatial_hat.cpp)
  target_link_libraries(test_spatial_hat PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_hat PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_hat COMMAND test_spatial_hat)

  add_executable(test_glued_form test/test_glued_form.cpp)
  target_link_libraries(test_glued_form PRIVATE nxr_pipelines)
  target_include_directories(test_glued_form PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  target_compile_definitions(test_glued_form PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/compute/test/golden")
  add_test(NAME test_glued_form COMMAND test_glued_form)

  add_executable(test_apply_kernel_form test/test_apply_kernel_form.cpp)
  target_link_libraries(test_apply_kernel_form PRIVATE nxr_pipelines)
  target_include_directories(test_apply_kernel_form PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_apply_kernel_form COMMAND test_apply_kernel_form)

  add_executable(test_apply_kernel_levels test/test_apply_kernel_levels.cpp)
  target_link_libraries(test_apply_kernel_levels PRIVATE nxr_pipelines)
  target_include_directories(test_apply_kernel_levels PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_apply_kernel_levels COMMAND test_apply_kernel_levels)

  add_executable(test_pyramid_pipeline test/test_pyramid_pipeline.cpp)
  target_link_libraries(test_pyramid_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_pyramid_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_pyramid_pipeline COMMAND test_pyramid_pipeline)
  add_executable(test_pyramid_reduce test/test_pyramid_reduce.cpp)
  target_link_libraries(test_pyramid_reduce PRIVATE nxr_pipelines)
  target_include_directories(test_pyramid_reduce PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_pyramid_reduce COMMAND test_pyramid_reduce)
  add_executable(test_spatial_bands test/test_spatial_bands.cpp)
  target_link_libraries(test_spatial_bands PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_bands PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_bands COMMAND test_spatial_bands)
  add_executable(test_frame_tiling_pipeline test/test_frame_tiling_pipeline.cpp)
  target_link_libraries(test_frame_tiling_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_frame_tiling_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_frame_tiling_pipeline COMMAND test_frame_tiling_pipeline)

  add_executable(test_vortex_cores_pipeline test/test_vortex_cores_pipeline.cpp)
  target_link_libraries(test_vortex_cores_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_vortex_cores_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_vortex_cores_pipeline COMMAND test_vortex_cores_pipeline)

  add_executable(test_spatial_events_pipeline test/test_spatial_events_pipeline.cpp)
  target_link_libraries(test_spatial_events_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_events_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_events_pipeline COMMAND test_spatial_events_pipeline)

  add_executable(test_scale_step test/test_scale_step.cpp)
  target_link_libraries(test_scale_step PRIVATE nxr_pipelines)
  target_include_directories(test_scale_step PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_scale_step COMMAND test_scale_step)

  add_executable(test_wishart_null test/test_wishart_null.cpp)
  target_link_libraries(test_wishart_null PRIVATE nxr_pipelines)
  target_include_directories(test_wishart_null PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_wishart_null COMMAND test_wishart_null)
  add_executable(test_pointspread test/test_pointspread.cpp)
  target_link_libraries(test_pointspread PRIVATE nxr_pipelines)
  target_include_directories(test_pointspread PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_pointspread COMMAND test_pointspread)
  add_executable(test_spatial_events_planted test/test_spatial_events_planted.cpp)
  target_link_libraries(test_spatial_events_planted PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_events_planted PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_events_planted COMMAND test_spatial_events_planted)
  set_tests_properties(test_spatial_events_planted PROPERTIES LABELS slow)   # minutes long: CI runs it on main and nightly, not on every pull request
  add_executable(test_spatial_threshold test/test_spatial_threshold.cpp)
  target_link_libraries(test_spatial_threshold PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_threshold PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_threshold COMMAND test_spatial_threshold)
  set_tests_properties(test_spatial_threshold PROPERTIES LABELS slow)   # minutes long: CI runs it on main and nightly, not on every pull request

  add_executable(test_cell_phase_current test/test_cell_phase_current.cpp)
  target_link_libraries(test_cell_phase_current PRIVATE nxr_pipelines)
  target_include_directories(test_cell_phase_current PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_cell_phase_current COMMAND test_cell_phase_current)
  add_executable(test_global_bursts test/test_global_bursts.cpp)
  target_link_libraries(test_global_bursts PRIVATE nxr_pipelines)
  target_include_directories(test_global_bursts PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_global_bursts COMMAND test_global_bursts)

  add_executable(test_spatial_analytic test/test_spatial_analytic.cpp)
  target_link_libraries(test_spatial_analytic PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_analytic PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_analytic COMMAND test_spatial_analytic)
  add_executable(test_burst_extent test/test_burst_extent.cpp)
  target_link_libraries(test_burst_extent PRIVATE nxr_pipelines)
  target_include_directories(test_burst_extent PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_burst_extent COMMAND test_burst_extent)
  add_executable(test_measure_time test/test_measure_time.cpp)
  target_link_libraries(test_measure_time PRIVATE nxr_pipelines)
  target_include_directories(test_measure_time PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_measure_time nxr-neuro)
    target_compile_definitions(test_measure_time PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_measure_time COMMAND test_measure_time)
  add_executable(test_runner_plan test/test_runner_plan.cpp)
  target_link_libraries(test_runner_plan PRIVATE nxr_pipelines)
  target_include_directories(test_runner_plan PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  target_compile_definitions(test_runner_plan PRIVATE NXR_SHIPPED_DEFINITIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/definitions" NXR_TEST_DEFINITIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/definitions")
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_runner_plan nxr-neuro)
    target_compile_definitions(test_runner_plan PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_runner_plan COMMAND test_runner_plan)
  add_executable(test_node_traits test/test_node_traits.cpp)
  target_link_libraries(test_node_traits PRIVATE nxr_pipelines)
  target_include_directories(test_node_traits PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  add_test(NAME test_node_traits COMMAND test_node_traits)
  
  add_executable(test_runner_execute test/test_runner_execute.cpp)
  target_link_libraries(test_runner_execute PRIVATE nxr_pipelines)
  target_include_directories(test_runner_execute PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  target_compile_definitions(test_runner_execute PRIVATE NXR_TEST_DEFINITIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/definitions")
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_runner_execute nxr-neuro)
    target_compile_definitions(test_runner_execute PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_runner_execute COMMAND test_runner_execute)
  add_executable(test_band_bursts test/test_band_bursts.cpp)
  target_link_libraries(test_band_bursts PRIVATE nxr_pipelines)
  target_include_directories(test_band_bursts PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_band_bursts COMMAND test_band_bursts)
  add_executable(test_filter_pipeline test/test_filter_pipeline.cpp)
  target_link_libraries(test_filter_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_filter_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test)
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_filter_pipeline nxr-neuro)
    target_compile_definitions(test_filter_pipeline PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_filter_pipeline COMMAND test_filter_pipeline)
  add_executable(test_headmodel test/test_headmodel.cpp)
  target_link_libraries(test_headmodel PRIVATE nxr_pipelines)
  target_include_directories(test_headmodel PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test)
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_headmodel nxr-neuro)
    target_compile_definitions(test_headmodel PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_headmodel COMMAND test_headmodel)
endif()
