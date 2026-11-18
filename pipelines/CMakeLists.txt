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
  src/bound.cpp
  src/step_nodes.cpp
  src/welch_steps.cpp
  src/meeg_steps.cpp
  src/quadrature_steps.cpp
  src/partition_steps.cpp
  src/pca_steps.cpp
  src/flow_steps.cpp
  src/verb_steps.cpp
  src/steps_measure.cpp
  src/scale_inputs.cpp
  src/steps_geometry.cpp
  src/steps_patches.cpp
  src/registry.cpp
  src/extent_steps.cpp
  src/cores_steps.cpp
  src/tracks_steps.cpp
  src/crossings_steps.cpp
  src/connectome_steps.cpp
  src/montage_steps.cpp
  src/select_nodes.cpp
  src/measurement_nodes.cpp
  src/tiles_steps.cpp
  src/surrogate_steps.cpp
  src/run_repeat.cpp
  src/surrogate_compare.cpp
  src/spatial_events_steps.cpp
  src/point_events.cpp
  src/runner.cpp
  src/declared_outputs.cpp
  src/input.cpp
  src/eigen.cpp
  src/eigen_steps.cpp
  src/frames.cpp
  src/registration.cpp
  src/montage_create.cpp
  src/dataset.cpp
  src/listing.cpp
  src/measure.cpp
  src/meeg.cpp
  src/spectral_record.cpp
  src/reading.cpp
  src/spatial_band.cpp
  src/group.cpp
  src/face.cpp
  src/selection_bridge.cpp
)
target_include_directories(nxr_pipelines PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(nxr_pipelines PUBLIC nxr_io nxr_compute nxr_meeg nxr_graph)
target_compile_definitions(nxr_pipelines PUBLIC NXR_COMPUTE_VERSION="${NXR_COMPUTE_VERSION}" NXR_NEURO_VERSION="${PROJECT_VERSION}")
if(MSVC)
  target_compile_options(nxr_pipelines PRIVATE /W4)
else()
  target_compile_options(nxr_pipelines PRIVATE -Wall -Wextra)
endif()

# the cache's own test (a key, a lookup, a hit and a miss on a synthetic dataset)
if(NXR_NEURO_BUILD_TESTS)
  option(NXR_BUILD_UNPORTED_TESTS "build the pipeline tests still written for retired verbs (they do not compile until ported)" OFF)
  add_executable(test_definitions_load test/test_definitions_load.cpp)   # every shipped definition loads on its own (all failures printed)
  target_link_libraries(test_definitions_load PRIVATE nxr_pipelines)
  target_compile_definitions(test_definitions_load PRIVATE NXR_TEST_DEFINITIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/definitions")
  add_test(NAME test_definitions_load COMMAND test_definitions_load)
  add_executable(test_pipelines test/test_pipelines.cpp)
  target_link_libraries(test_pipelines PRIVATE nxr_pipelines)
  target_include_directories(test_pipelines PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  add_test(NAME test_pipelines COMMAND test_pipelines)
  add_executable(test_dataset_with_template test/test_dataset_with_template.cpp)
  target_link_libraries(test_dataset_with_template PRIVATE nxr_pipelines)
  target_include_directories(test_dataset_with_template PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  add_test(NAME test_dataset_with_template COMMAND test_dataset_with_template)
  add_executable(test_montage_create test/test_montage_create.cpp)
  target_link_libraries(test_montage_create PRIVATE nxr_pipelines)
  target_include_directories(test_montage_create PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_montage_create COMMAND test_montage_create)
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
  add_executable(test_face test/test_face.cpp)   # what the Python and MATLAB faces call (face.h): run, list, read, on a synthetic dataset; the same result as the command line's
  target_link_libraries(test_face PRIVATE nxr_pipelines)
  target_include_directories(test_face PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test)
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_face nxr-neuro)
    target_compile_definitions(test_face PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_face COMMAND test_face)
  add_executable(test_psd_golden test/test_psd_golden.cpp)
  target_link_libraries(test_psd_golden PRIVATE nxr_pipelines)
  target_include_directories(test_psd_golden PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  target_compile_definitions(test_psd_golden PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/compute/test/golden")
  add_test(NAME test_psd_golden COMMAND test_psd_golden)
  add_executable(test_source_psd test/test_source_psd.cpp)
  target_link_libraries(test_source_psd PRIVATE nxr_pipelines)
  target_include_directories(test_source_psd PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  target_compile_definitions(test_source_psd PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/meeg/test/golden")
  add_test(NAME test_source_psd COMMAND test_source_psd)
  add_executable(test_package_pipelines test/test_package_pipelines.cpp)
  target_link_libraries(test_package_pipelines PRIVATE nxr_pipelines)
  target_include_directories(test_package_pipelines PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_package_pipelines COMMAND test_package_pipelines)
  add_executable(test_import_in_place test/test_import_in_place.cpp)
  target_link_libraries(test_import_in_place PRIVATE nxr_pipelines)
  target_include_directories(test_import_in_place PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test)
  target_compile_definitions(test_import_in_place PRIVATE NXR_FORMATS_DATA_DIR="${CMAKE_SOURCE_DIR}/io/formats/test/data")
  add_test(NAME test_import_in_place COMMAND test_import_in_place)
  add_executable(test_quadrature_pipeline test/test_quadrature_pipeline.cpp)
  target_link_libraries(test_quadrature_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_quadrature_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_quadrature_pipeline COMMAND test_quadrature_pipeline)
  add_executable(test_hodge_power test/test_hodge_power.cpp)
  target_link_libraries(test_hodge_power PRIVATE nxr_pipelines)
  target_include_directories(test_hodge_power PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_hodge_power COMMAND test_hodge_power)

  if(NXR_BUILD_UNPORTED_TESTS)   # written for the hand-wired verb the round replaced by a definition; to be ported (docs: the build-2 report)
    add_executable(test_speedspectrum_pipeline test/test_speedspectrum_pipeline.cpp)
    target_link_libraries(test_speedspectrum_pipeline PRIVATE nxr_pipelines)
    target_include_directories(test_speedspectrum_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/meeg/test ${CMAKE_SOURCE_DIR}/compute/test)
    add_test(NAME test_speedspectrum_pipeline COMMAND test_speedspectrum_pipeline)
  endif()

  add_executable(test_frames_pipeline test/test_frames_pipeline.cpp)
  target_link_libraries(test_frames_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_frames_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_frames_pipeline COMMAND test_frames_pipeline)
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
  add_executable(test_measure_pipeline test/test_measure_pipeline.cpp)
  target_link_libraries(test_measure_pipeline PRIVATE nxr_pipelines)
  target_include_directories(test_measure_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_measure_pipeline COMMAND test_measure_pipeline)

  if(NXR_BUILD_UNPORTED_TESTS)   # written for the hand-wired verb the round replaced by a definition; to be ported (docs: the build-2 report)
    add_executable(test_patches_pipeline test/test_patches_pipeline.cpp)
    target_link_libraries(test_patches_pipeline PRIVATE nxr_pipelines)
    target_include_directories(test_patches_pipeline PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
    add_test(NAME test_patches_pipeline COMMAND test_patches_pipeline)
  endif()




  if(NXR_BUILD_UNPORTED_TESTS)   # written for the hand-wired verb the round replaced by a definition; to be ported (docs: the build-2 report)
    add_executable(test_cellquadrature test/test_cellquadrature.cpp)
    target_link_libraries(test_cellquadrature PRIVATE nxr_pipelines)
    target_include_directories(test_cellquadrature PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
    add_test(NAME test_cellquadrature COMMAND test_cellquadrature)
  endif()
  if(NXR_BUILD_UNPORTED_TESTS)   # written for the hand-wired verb the round replaced by a definition; to be ported (docs: the build-2 report)
    add_executable(test_scale_selection test/test_scale_selection.cpp)
    target_link_libraries(test_scale_selection PRIVATE nxr_pipelines)
    target_include_directories(test_scale_selection PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
    add_test(NAME test_scale_selection COMMAND test_scale_selection)
  endif()

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

  add_executable(test_select_makers test/test_select_makers.cpp)
  target_link_libraries(test_select_makers PRIVATE nxr_pipelines)
  target_include_directories(test_select_makers PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_select_makers COMMAND test_select_makers)
  add_executable(test_rollup_parity test/test_rollup_parity.cpp)
  target_link_libraries(test_rollup_parity PRIVATE nxr_pipelines)
  target_include_directories(test_rollup_parity PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_rollup_parity COMMAND test_rollup_parity)
  add_executable(test_spatial_bands test/test_spatial_bands.cpp)
  target_link_libraries(test_spatial_bands PRIVATE nxr_pipelines)
  target_include_directories(test_spatial_bands PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_spatial_bands COMMAND test_spatial_bands)

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

  if(NXR_BUILD_UNPORTED_TESTS)   # written for the hand-wired verb the round replaced by a definition; to be ported (docs: the build-2 report)
    add_executable(test_wishart_null test/test_wishart_null.cpp)
    target_link_libraries(test_wishart_null PRIVATE nxr_pipelines)
    target_include_directories(test_wishart_null PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
    add_test(NAME test_wishart_null COMMAND test_wishart_null)
  endif()
  if(NXR_BUILD_UNPORTED_TESTS)   # written for the hand-wired verb the round replaced by a definition; to be ported (docs: the build-2 report)
    add_executable(test_pointspread test/test_pointspread.cpp)
    target_link_libraries(test_pointspread PRIVATE nxr_pipelines)
    target_include_directories(test_pointspread PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
    add_test(NAME test_pointspread COMMAND test_pointspread)
  endif()
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
  add_executable(test_runner_plan test/test_runner_plan.cpp)
  target_link_libraries(test_runner_plan PRIVATE nxr_pipelines)
  target_include_directories(test_runner_plan PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  target_compile_definitions(test_runner_plan PRIVATE NXR_SHIPPED_DEFINITIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/definitions" NXR_TEST_DEFINITIONS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/definitions")
  if(NXR_NEURO_BUILD_CLI)
    add_dependencies(test_runner_plan nxr-neuro)
    target_compile_definitions(test_runner_plan PRIVATE NXR_TEST_CLI="$<TARGET_FILE:nxr-neuro>")
  endif()
  add_test(NAME test_runner_plan COMMAND test_runner_plan)
  # one name per thing across the C++ function, the registered node and the bindings (pipelines/test/names.tsv)
  add_executable(test_names_across_layers test/test_names_across_layers.cpp)
  target_link_libraries(test_names_across_layers PRIVATE nxr_pipelines)
  target_include_directories(test_names_across_layers PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  target_compile_definitions(test_names_across_layers PRIVATE NXR_REPO_ROOT="${CMAKE_SOURCE_DIR}")
  add_test(NAME test_names_across_layers COMMAND test_names_across_layers)
  add_test(NAME test_names_across_layers_catches COMMAND test_names_across_layers ${CMAKE_CURRENT_SOURCE_DIR}/test/names_bad.tsv)
  set_tests_properties(test_names_across_layers_catches PROPERTIES WILL_FAIL TRUE)
  add_executable(test_node_traits test/test_node_traits.cpp)
  target_link_libraries(test_node_traits PRIVATE nxr_pipelines)
  target_include_directories(test_node_traits PRIVATE ${CMAKE_SOURCE_DIR}/io/test)
  add_test(NAME test_node_traits COMMAND test_node_traits)
  
  add_executable(test_montage_nodes test/test_montage_nodes.cpp)
  target_link_libraries(test_montage_nodes PRIVATE nxr_pipelines)
  target_include_directories(test_montage_nodes PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_montage_nodes COMMAND test_montage_nodes)
  add_executable(test_connectome_nodes test/test_connectome_nodes.cpp)
  target_link_libraries(test_connectome_nodes PRIVATE nxr_pipelines)
  target_include_directories(test_connectome_nodes PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_connectome_nodes COMMAND test_connectome_nodes)
  add_executable(test_connectome_definition test/test_connectome_definition.cpp)
  target_link_libraries(test_connectome_definition PRIVATE nxr_pipelines)
  target_compile_definitions(test_connectome_definition PRIVATE NXR_PENDING_DEFINITIONS="${CMAKE_CURRENT_SOURCE_DIR}/definitions/pending" NXR_DEFINITIONS="${CMAKE_CURRENT_SOURCE_DIR}/definitions")
  add_test(NAME test_connectome_definition COMMAND test_connectome_definition)

  add_executable(test_runner_outputs test/test_runner_outputs.cpp)
  target_link_libraries(test_runner_outputs PRIVATE nxr_pipelines)
  target_include_directories(test_runner_outputs PRIVATE ${CMAKE_SOURCE_DIR}/io/test ${CMAKE_SOURCE_DIR}/compute/test)
  add_test(NAME test_runner_outputs COMMAND test_runner_outputs)
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
  # the type system on the shipped definitions (spec docs/specs/2026-11-17-type-system.md, T2): synthetic input types, no data
  add_executable(test_definitions_typed test/test_definitions_typed.cpp)
  target_link_libraries(test_definitions_typed PRIVATE nxr_pipelines)
  add_test(NAME test_definitions_typed COMMAND test_definitions_typed)
  # every pipeline test may use io's test-only probe (nxr/io/testing/probe.h) for assertions about what was written
  get_directory_property(_pipeline_targets BUILDSYSTEM_TARGETS)
  foreach(_t ${_pipeline_targets})
    if(_t MATCHES "^test_")
      target_link_libraries(${_t} PRIVATE nxr_io_testing)
    endif()
  endforeach()
endif()

# CI runs a pull request only the tests of the libraries it touches (.github/workflows/ci.yml): every test added here carries this label.
set_property(DIRECTORY APPEND PROPERTY LABELS pipelines)
