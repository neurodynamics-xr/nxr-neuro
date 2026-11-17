# nxr_compute — included by ../CMakeLists.txt (the top-level project, options and Eigen are there).

# ── Workspace defaults ───────────────────────────────────────
# C++17 floor. geometry-central + Eigen 3.4 under Emscripten Clang
# trip a template-deduction bug at C++20 (visible in
# boundary_first_flattening.cpp). Until Eigen is upgraded or the
# trip site patched, the entire workspace stays at C++17.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Dependencies ─────────────────────────────────────────────
# Per CONTRIBUTING.md, the dependency policy is locked: Eigen + Spectra
# + geometry-central + pocketfft (header-only, vendored in deps/pocketfft;
# BSD-3), nothing else. All are header-only or vendored, so no system
# packages are required.

add_subdirectory(deps/geometry-central)

include(FetchContent)
FetchContent_Declare(
  spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra.git
  GIT_TAG        v1.2.0        # 2025-06-29: the complex Hermitian solver HermEigsSolver (v1.1.0) for the magnetic Laplacian
  GIT_SHALLOW    TRUE
)
FetchContent_Populate(spectra)

# ── nxr_compute static library ───────────────────────────────
# The pure compute kernel. Depends only on Eigen (transitively
# via geometry-central), Spectra (header-only), and
# geometry-central (vendored submodule). All consumer targets
# (the N-API addon, MATLAB MEX, WASM, CLI) link against this
# library and add nothing more.
#
# CMake target: nxr_compute
# Headers:      include/nxr/{compute,errors,cancellation,progress}.h
# Namespace:    nxr::compute


add_library(nxr_compute STATIC
  src/mesh_operators.cpp
  src/normalize_delaunay.cpp
  src/eigensolver.cpp
  src/congruence.cpp
  src/nearest.cpp
  src/montage.cpp
  src/connectome.cpp
  src/magnetic.cpp
  src/lag.cpp
  src/region.cpp
  src/transport.cpp
  src/normalize.cpp
  src/cholesky_cache.cpp
  src/poisson_solver.cpp
  src/geodesic.cpp
  src/chart.cpp
  src/frames.cpp
  src/hodge.cpp
  src/field_generators.cpp
  src/curvatures.cpp
  src/normals.cpp
  src/vector_field.cpp
  src/isolines.cpp
  src/direction_field.cpp
  src/face_frames.cpp
  src/vertex_frames.cpp
  src/geodesic_path.cpp
  src/parametrization.cpp
  src/vector_heat.cpp
  src/signed_heat.cpp
  src/smooth_field.cpp
  src/stripe_patterns.cpp
  src/connection_laplacian.cpp
  src/covariant_laplacian.cpp
  src/query_measure.cpp
  src/geometry_grid.cpp
  src/intrinsic_surface.cpp
  src/gauge.cpp
  src/topology.cpp
  src/geometry_bundle.cpp
  src/graph_laplacian.cpp
  src/facets.cpp
  src/covariant_differential.cpp
  src/extrinsic_weitzenbock.cpp
  src/dirac_operator.cpp
  src/face_gradient.cpp
  src/hodge_weights.cpp
  src/intrinsic_forms.cpp
  src/intervals.cpp
  src/regions.cpp
  src/extent.cpp
  src/scale.cpp
  src/risefall.cpp
  src/stats.cpp
  src/histogram.cpp
  src/cores.cpp
  src/tracks.cpp
  src/crossings.cpp
  src/sphere_registration.cpp
  src/surrogate.cpp
  src/scaling.cpp
  src/scalespace.cpp
  src/patterns.cpp
  src/spectral_scaling.cpp
  src/eigenproblem.cpp
  src/hodge_dirac.cpp
  src/spin_dirac.cpp
  src/operator_registry.cpp
  src/field_registry.cpp
  src/fft.cpp
  src/filterbank.cpp
  src/frame_tiling.cpp
  src/tower.cpp
  src/pyramid.cpp
  src/parallel.cpp
  src/integrals.cpp
  src/timebank.cpp
  src/jointbank.cpp
  src/spacetime_kernels.cpp
  src/welch.cpp
  src/band_amplitude.cpp
  src/pca.cpp
  src/quadrature.cpp
  src/filter.cpp
  src/wt.cpp
  src/analytic.cpp
  src/opticalflow.cpp
  src/opticalflow_connection.cpp
  src/speedspectrum.cpp
  src/opticalflow_methods.cpp
  src/ladder.cpp
  src/selection.cpp
  src/selection_partition.cpp
  src/selection_dyadic.cpp
  src/selection_scales.cpp
  src/selection_constant_q.cpp
  src/selection_resolved.cpp
  src/selection_grid.cpp
  src/partition.cpp
  src/glue.cpp
  src/forms.cpp
  src/taps.cpp
  src/phase.cpp
  src/monogenic.cpp
)

# Treat an unhandled enumerator in a no-default switch as an error for the
# operator-registry TU (so a new OperatorId can't ship without metadata).
# GCC/Clang: -Werror=switch. MSVC: C4062 is the no-default analogue; promote
# it to an error with /we4062. The flags are mutually exclusive — cl.exe parses
# "/Wswitch" as "/W<arg>" and fails with D8021.
if(MSVC)
    set_source_files_properties(src/operator_registry.cpp PROPERTIES
        COMPILE_OPTIONS "/we4062")
else()
    set_source_files_properties(src/operator_registry.cpp PROPERTIES
        COMPILE_OPTIONS "-Wswitch;-Werror=switch")
endif()

target_include_directories(nxr_compute PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${spectra_SOURCE_DIR}/include
)

target_compile_features(nxr_compute PUBLIC cxx_std_17)

# ladder.cpp is a port of the Studio's ladder.ts (no FMA in JavaScript): contraction off, so areas, distances and the equal-mass cuts tie as they do there
set_source_files_properties(src/ladder.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(src/selection_dyadic.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # the Studio's ladder.ts, moved here
# partition.cpp is a port of the Studio's measureOverPartition (no FMA in JavaScript): contraction off, so its sums equal the Studio's to the last bit
set_source_files_properties(src/partition.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(src/taps.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")    # nxr's taps.py (numpy: no FMA)
set_source_files_properties(src/forms.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # a port of the Studio's fieldForms.ts / faceGradient.ts (no FMA in JavaScript)
# glue.cpp is a port of the Studio's glue.ts (no FMA in JavaScript): contraction off, so the bridge's positions and the stitch's ties are the Studio's
set_source_files_properties(src/glue.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_link_libraries(nxr_compute PUBLIC
  geometry-central
)
# The pinned geometry-central commit (7 hex), read from the submodule at configure time: nxr::manifold::geometryCentralVersion() and the gauge identity
# (docs/specs/2026-11-17-geometry-central-switch.md section 7). "unknown" outside a git checkout (a source tarball).
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/deps/geometry-central rev-parse --short=7 HEAD
                OUTPUT_VARIABLE NXR_GC_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE NXR_GC_RC)
if(NOT NXR_GC_RC EQUAL 0 OR NXR_GC_COMMIT STREQUAL "")
  set(NXR_GC_COMMIT "unknown")
endif()
set_property(SOURCE src/intrinsic_surface.cpp APPEND PROPERTY COMPILE_DEFINITIONS NXR_GEOMETRY_CENTRAL_COMMIT="${NXR_GC_COMMIT}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../.gitmodules)
# std::thread (parallel.cpp): the platform's own threads, not a library dependency
if(NOT EMSCRIPTEN)
  find_package(Threads REQUIRED)
  target_link_libraries(nxr_compute PUBLIC Threads::Threads)
endif()

# AVX2 intentionally NOT enabled — /arch:AVX2 on nxr_compute alone
# causes an Eigen alignment ABI mismatch with geometry-central.
# Re-enable only by either propagating the flag through
# geometry-central or defining EIGEN_MAX_ALIGN_BYTES=16 globally.

# ── Standalone tests (no Node.js) ────────────────────────────
# Skipped under Emscripten — these are native-only smoke tests;
# the WASM build has its own JS-side smoke (scripts/_smoke-wasm.mjs).

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)

    add_executable(test_eigen test/test_eigen.cpp)
    target_link_libraries(test_eigen PRIVATE nxr_compute)
    add_test(NAME test_eigen COMMAND test_eigen)
    add_executable(test_eigen_consistency test/test_eigen_consistency.cpp)
    target_link_libraries(test_eigen_consistency PRIVATE nxr_compute)
    add_test(NAME test_eigen_consistency COMMAND test_eigen_consistency)

    add_executable(test_cholesky_cache test/test_cholesky_cache.cpp)
    target_link_libraries(test_cholesky_cache PRIVATE nxr_compute)
    add_test(NAME test_cholesky_cache COMMAND test_cholesky_cache)

    add_executable(test_field_generators test/test_field_generators.cpp)
    target_link_libraries(test_field_generators PRIVATE nxr_compute)
    add_test(NAME test_field_generators COMMAND test_field_generators)

    add_executable(test_visualization_primitives test/test_visualization_primitives.cpp)
    target_link_libraries(test_visualization_primitives PRIVATE nxr_compute)
    add_test(NAME test_visualization_primitives COMMAND test_visualization_primitives)

    add_executable(test_cancellation test/test_cancellation.cpp)
    target_link_libraries(test_cancellation PRIVATE nxr_compute)
    add_test(NAME test_cancellation COMMAND test_cancellation)

    add_executable(test_progress test/test_progress.cpp)
    target_link_libraries(test_progress PRIVATE nxr_compute)
    add_test(NAME test_progress COMMAND test_progress)

    add_executable(test_graph_agnostic test/test_graph_agnostic.cpp)
    target_link_libraries(test_graph_agnostic PRIVATE nxr_compute)
    add_test(NAME test_graph_agnostic COMMAND test_graph_agnostic)

    add_executable(test_mass_variants test/test_mass_variants.cpp)
    target_link_libraries(test_mass_variants PRIVATE nxr_compute)
    add_test(NAME test_mass_variants COMMAND test_mass_variants)

    add_executable(test_operator_registry test/test_operator_registry.cpp)
    target_link_libraries(test_operator_registry PRIVATE nxr_compute)
    add_test(NAME test_operator_registry COMMAND test_operator_registry)

    add_executable(test_field_registry test/test_field_registry.cpp)
    target_link_libraries(test_field_registry PRIVATE nxr_compute)
    add_test(NAME test_field_registry COMMAND test_field_registry)

    add_executable(test_geometry_central_extras test/test_geometry_central_extras.cpp)
    target_link_libraries(test_geometry_central_extras PRIVATE nxr_compute)
    add_test(NAME test_geometry_central_extras COMMAND test_geometry_central_extras)

    add_executable(test_connection_laplacian test/test_connection_laplacian.cpp)
    target_link_libraries(test_connection_laplacian PRIVATE nxr_compute)
    add_test(NAME test_connection_laplacian COMMAND test_connection_laplacian)

    add_executable(test_geometry_bundle test/test_geometry_bundle.cpp)
    target_link_libraries(test_geometry_bundle PRIVATE nxr_compute)
    add_test(NAME test_geometry_bundle COMMAND test_geometry_bundle)

    add_executable(test_passthrough_accessors test/test_passthrough_accessors.cpp)
    target_link_libraries(test_passthrough_accessors PRIVATE nxr_compute)
    add_test(NAME test_passthrough_accessors COMMAND test_passthrough_accessors)

    add_executable(test_query_measure test/test_query_measure.cpp)
    target_link_libraries(test_query_measure PRIVATE nxr_compute)
    add_test(NAME test_query_measure COMMAND test_query_measure)

    add_executable(test_normalize test/test_normalize.cpp)
    target_link_libraries(test_normalize PRIVATE nxr_compute)
    add_test(NAME test_normalize COMMAND test_normalize)

    # the analytic tests of the geometry-central switch (docs/specs/2026-11-17-geometry-central-switch.md, section 4) and the version pin
    add_executable(test_intrinsic_surface test/test_intrinsic_surface.cpp)
    target_link_libraries(test_intrinsic_surface PRIVATE nxr_compute)
    target_compile_definitions(test_intrinsic_surface PRIVATE NXR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
    add_test(NAME test_intrinsic_surface COMMAND test_intrinsic_surface)

    add_executable(test_intrinsic_delaunay test/test_intrinsic_delaunay.cpp)
    target_link_libraries(test_intrinsic_delaunay PRIVATE nxr_compute)
    add_test(NAME test_intrinsic_delaunay COMMAND test_intrinsic_delaunay)

    add_executable(test_facets test/test_facets.cpp)
    target_link_libraries(test_facets PRIVATE nxr_compute)
    add_test(NAME test_facets COMMAND test_facets)

    add_executable(test_gauge_workflow test/test_gauge_workflow.cpp)
    target_link_libraries(test_gauge_workflow PRIVATE nxr_compute)
    add_test(NAME test_gauge_workflow COMMAND test_gauge_workflow)

    add_executable(test_operators_facet test/test_operators_facet.cpp)
    target_link_libraries(test_operators_facet PRIVATE nxr_compute)
    add_test(NAME test_operators_facet COMMAND test_operators_facet)

    add_executable(test_covariant_differential test/test_covariant_differential.cpp)
    target_link_libraries(test_covariant_differential PRIVATE nxr_compute)
    add_test(NAME test_covariant_differential COMMAND test_covariant_differential)

    add_executable(test_dirac_operator test/test_dirac_operator.cpp)
    target_link_libraries(test_dirac_operator PRIVATE nxr_compute)
    add_test(NAME test_dirac_operator COMMAND test_dirac_operator)

    add_executable(test_dirac_face_operator test/test_dirac_face_operator.cpp)
    target_link_libraries(test_dirac_face_operator PRIVATE nxr_compute)
    add_test(NAME test_dirac_face_operator COMMAND test_dirac_face_operator)

    add_executable(test_operator_eigensolve test/test_operator_eigensolve.cpp)
    target_link_libraries(test_operator_eigensolve PRIVATE nxr_compute)
    add_test(NAME test_operator_eigensolve COMMAND test_operator_eigensolve)

    add_executable(test_hodge_dirac test/test_hodge_dirac.cpp)
    target_link_libraries(test_hodge_dirac PRIVATE nxr_compute)
    add_test(NAME test_hodge_dirac COMMAND test_hodge_dirac)

    add_executable(test_spin_dirac test/test_spin_dirac.cpp)
    target_link_libraries(test_spin_dirac PRIVATE nxr_compute)
    add_test(NAME test_spin_dirac COMMAND test_spin_dirac)

    add_executable(test_graph_eigen test/test_graph_eigen.cpp)
    target_link_libraries(test_graph_eigen PRIVATE nxr_compute)
    add_test(NAME test_graph_eigen COMMAND test_graph_eigen)

    add_executable(test_cf_parity test/test_cf_parity.cpp)
    target_link_libraries(test_cf_parity PRIVATE nxr_compute)
    add_test(NAME test_cf_parity COMMAND test_cf_parity)

    add_executable(test_face_gradient test/test_face_gradient.cpp)
    target_link_libraries(test_face_gradient PRIVATE nxr_compute)
    add_test(NAME test_face_gradient COMMAND test_face_gradient)

    add_executable(test_intrinsic_forms test/test_intrinsic_forms.cpp)
    target_link_libraries(test_intrinsic_forms PRIVATE nxr_compute)
    add_test(NAME test_intrinsic_forms COMMAND test_intrinsic_forms)
    add_executable(test_intervals test/test_intervals.cpp)
    target_link_libraries(test_intervals PRIVATE nxr_compute)
    add_test(NAME test_intervals COMMAND test_intervals)
    add_executable(test_regions test/test_regions.cpp)
    target_link_libraries(test_regions PRIVATE nxr_compute)
    add_test(NAME test_regions COMMAND test_regions)
    add_executable(test_extent test/test_extent.cpp)
    target_link_libraries(test_extent PRIVATE nxr_compute)
    add_test(NAME test_extent COMMAND test_extent)
    add_executable(test_scale test/test_scale.cpp)
    target_link_libraries(test_scale PRIVATE nxr_compute)
    add_test(NAME test_scale COMMAND test_scale)
    add_executable(test_risefall test/test_risefall.cpp)
    target_link_libraries(test_risefall PRIVATE nxr_compute)
    target_compile_definitions(test_risefall PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_risefall COMMAND test_risefall)
    add_executable(test_burst_stages test/test_burst_stages.cpp)
    target_link_libraries(test_burst_stages PRIVATE nxr_compute)
    target_compile_definitions(test_burst_stages PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_burst_stages COMMAND test_burst_stages)
    add_executable(test_stats test/test_stats.cpp)
    target_link_libraries(test_stats PRIVATE nxr_compute)
    target_compile_definitions(test_stats PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_stats COMMAND test_stats)
    add_executable(test_histogram test/test_histogram.cpp)
    target_link_libraries(test_histogram PRIVATE nxr_compute)
    target_compile_definitions(test_histogram PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_histogram COMMAND test_histogram)
    add_executable(test_scalespace test/test_scalespace.cpp)
    target_link_libraries(test_scalespace PRIVATE nxr_compute)
    add_test(NAME test_scalespace COMMAND test_scalespace)
    add_executable(test_scaling test/test_scaling.cpp)
    target_link_libraries(test_scaling PRIVATE nxr_compute)
    add_test(NAME test_scaling COMMAND test_scaling)
    add_executable(test_cores test/test_cores.cpp)
    target_link_libraries(test_cores PRIVATE nxr_compute)
    target_compile_definitions(test_cores PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_cores COMMAND test_cores)
    add_executable(test_tracks test/test_tracks.cpp)
    target_link_libraries(test_tracks PRIVATE nxr_compute)
    target_compile_definitions(test_tracks PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_tracks COMMAND test_tracks)
    add_executable(test_montage test/test_montage.cpp)   # the 10-05 rule: names, a sphere against MNE (golden/montage_golden.txt, golden/gen_montage_golden.py), an ellipsoid
    target_link_libraries(test_montage PRIVATE nxr_compute)
    target_compile_definitions(test_montage PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_montage COMMAND test_montage)
    add_executable(test_connectome test/test_connectome.cpp)
    target_link_libraries(test_connectome PRIVATE nxr_compute)
    add_test(NAME test_connectome COMMAND test_connectome)
    add_executable(test_connectome_golden test/test_connectome_golden.cpp)   # against the MATLAB reference (golden/connectome_golden.txt, golden/gen_connectome_golden.m)
    target_link_libraries(test_connectome_golden PRIVATE nxr_compute)
    target_compile_definitions(test_connectome_golden PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_connectome_golden COMMAND test_connectome_golden)
    add_executable(test_connectome_partition test/test_connectome_partition.cpp)
    target_link_libraries(test_connectome_partition PRIVATE nxr_compute)
    add_test(NAME test_connectome_partition COMMAND test_connectome_partition)
    add_executable(test_connectome_gate test/test_connectome_gate.cpp)
    target_link_libraries(test_connectome_gate PRIVATE nxr_compute)
    add_test(NAME test_connectome_gate COMMAND test_connectome_gate)
    add_executable(test_connectome_registration test/test_connectome_registration.cpp)
    target_link_libraries(test_connectome_registration PRIVATE nxr_compute)
    add_test(NAME test_connectome_registration COMMAND test_connectome_registration)
    add_executable(test_crossings test/test_crossings.cpp)
    target_link_libraries(test_crossings PRIVATE nxr_compute)
    target_compile_definitions(test_crossings PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_crossings COMMAND test_crossings)
    add_executable(test_surrogate test/test_surrogate.cpp)
    target_link_libraries(test_surrogate PRIVATE nxr_compute)
    target_compile_definitions(test_surrogate PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_surrogate COMMAND test_surrogate)
    add_executable(test_patterns test/test_patterns.cpp)
    target_link_libraries(test_patterns PRIVATE nxr_compute)
    add_test(NAME test_patterns COMMAND test_patterns)
    add_executable(test_spectral_scaling test/test_spectral_scaling.cpp)
    target_link_libraries(test_spectral_scaling PRIVATE nxr_compute)
    add_test(NAME test_spectral_scaling COMMAND test_spectral_scaling)
    add_executable(test_hodge_weights test/test_hodge_weights.cpp)
    target_link_libraries(test_hodge_weights PRIVATE nxr_compute)
    target_compile_definitions(test_hodge_weights PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_hodge_weights COMMAND test_hodge_weights)

    add_executable(test_fft test/test_fft.cpp)
    target_link_libraries(test_fft PRIVATE nxr_compute)
    target_compile_definitions(test_fft PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_fft COMMAND test_fft)

    add_executable(test_filterbank test/test_filterbank.cpp)
    target_link_libraries(test_filterbank PRIVATE nxr_compute)
    target_compile_definitions(test_filterbank PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_filterbank COMMAND test_filterbank)

    add_executable(test_eigen_complete test/test_eigen_complete.cpp)
    target_link_libraries(test_eigen_complete PRIVATE nxr_compute)
    add_test(NAME test_eigen_complete COMMAND test_eigen_complete)
    add_executable(test_eigenspectrum test/test_eigenspectrum.cpp)
    target_link_libraries(test_eigenspectrum PRIVATE nxr_compute)
    add_test(NAME test_eigenspectrum COMMAND test_eigenspectrum)

    add_executable(test_timebank test/test_timebank.cpp)
    target_link_libraries(test_timebank PRIVATE nxr_compute)
    target_compile_definitions(test_timebank PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_timebank COMMAND test_timebank)

    add_executable(test_jointbank test/test_jointbank.cpp)
    target_link_libraries(test_jointbank PRIVATE nxr_compute)
    target_compile_definitions(test_jointbank PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_jointbank COMMAND test_jointbank)

    add_executable(test_speedspectrum test/test_speedspectrum.cpp)
    target_link_libraries(test_speedspectrum PRIVATE nxr_compute)
    add_test(NAME test_speedspectrum COMMAND test_speedspectrum)

    add_executable(test_spacetime_kernels test/test_spacetime_kernels.cpp)
    target_link_libraries(test_spacetime_kernels PRIVATE nxr_compute)
    target_compile_definitions(test_spacetime_kernels PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_spacetime_kernels COMMAND test_spacetime_kernels)

    add_executable(test_welch test/test_welch.cpp)
    target_link_libraries(test_welch PRIVATE nxr_compute)
    target_compile_definitions(test_welch PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_welch COMMAND test_welch)

    add_executable(test_pca test/test_pca.cpp)
    target_link_libraries(test_pca PRIVATE nxr_compute)
    target_compile_definitions(test_pca PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_pca COMMAND test_pca)

    add_executable(test_quadrature test/test_quadrature.cpp)
    target_link_libraries(test_quadrature PRIVATE nxr_compute)
    target_compile_definitions(test_quadrature PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_quadrature COMMAND test_quadrature)

    add_executable(test_wt test/test_wt.cpp)
    target_link_libraries(test_wt PRIVATE nxr_compute)
    add_test(NAME test_wt COMMAND test_wt)
    set_tests_properties(test_wt PROPERTIES LABELS slow)   # minutes long: CI runs it on main and nightly, not on every pull request

    add_executable(test_analytic test/test_analytic.cpp)
    target_link_libraries(test_analytic PRIVATE nxr_compute)
    add_test(NAME test_analytic COMMAND test_analytic)

    add_executable(test_opticalflow test/test_opticalflow.cpp)
    target_link_libraries(test_opticalflow PRIVATE nxr_compute)
    target_compile_definitions(test_opticalflow PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_opticalflow COMMAND test_opticalflow)

    add_executable(test_opticalflow_ambient test/test_opticalflow_ambient.cpp)
    target_link_libraries(test_opticalflow_ambient PRIVATE nxr_compute)
    target_compile_definitions(test_opticalflow_ambient PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_opticalflow_ambient COMMAND test_opticalflow_ambient)

    add_executable(test_opticalflow_connection test/test_opticalflow_connection.cpp)
    target_link_libraries(test_opticalflow_connection PRIVATE nxr_compute)
    target_compile_definitions(test_opticalflow_connection PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_opticalflow_connection COMMAND test_opticalflow_connection)

    add_executable(test_chart test/test_chart.cpp)
    target_link_libraries(test_chart PRIVATE nxr_compute)
    target_compile_definitions(test_chart PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_chart COMMAND test_chart)

    add_executable(test_ladder test/test_ladder.cpp)
    target_link_libraries(test_ladder PRIVATE nxr_compute)
    target_compile_definitions(test_ladder PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_ladder COMMAND test_ladder)
    add_executable(test_group_tree test/test_group_tree.cpp)
    target_link_libraries(test_group_tree PRIVATE nxr_compute)
    target_compile_definitions(test_group_tree PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_group_tree COMMAND test_group_tree)

    add_executable(test_partition test/test_partition.cpp)
    target_link_libraries(test_partition PRIVATE nxr_compute)
    target_compile_definitions(test_partition PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_partition COMMAND test_partition)

    add_executable(test_time_tiles test/test_time_tiles.cpp)
    target_link_libraries(test_time_tiles PRIVATE nxr_compute)
    target_compile_definitions(test_time_tiles PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_time_tiles COMMAND test_time_tiles)
    add_executable(test_phase test/test_phase.cpp)
    target_link_libraries(test_phase PRIVATE nxr_compute)
    target_compile_definitions(test_phase PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_phase COMMAND test_phase)
    add_executable(test_monogenic test/test_monogenic.cpp)
    target_link_libraries(test_monogenic PRIVATE nxr_compute)
    add_test(NAME test_monogenic COMMAND test_monogenic)
    add_executable(test_magnetic test/test_magnetic.cpp)
    target_link_libraries(test_magnetic PRIVATE nxr_compute)
    target_compile_definitions(test_magnetic PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_magnetic COMMAND test_magnetic)
    add_executable(test_cross_spectrum test/test_cross_spectrum.cpp)
    target_link_libraries(test_cross_spectrum PRIVATE nxr_compute)
    add_test(NAME test_cross_spectrum COMMAND test_cross_spectrum)
    add_executable(test_band_moments test/test_band_moments.cpp)
    target_link_libraries(test_band_moments PRIVATE nxr_compute)
    add_test(NAME test_band_moments COMMAND test_band_moments)
    add_executable(test_lag test/test_lag.cpp)
    target_link_libraries(test_lag PRIVATE nxr_compute)
    add_test(NAME test_lag COMMAND test_lag)
    add_executable(test_region test/test_region.cpp)
    target_link_libraries(test_region PRIVATE nxr_compute)
    add_test(NAME test_region COMMAND test_region)
    add_executable(test_transport test/test_transport.cpp)
    target_link_libraries(test_transport PRIVATE nxr_compute)
    add_test(NAME test_transport COMMAND test_transport)
    add_executable(test_taps test/test_taps.cpp)
    target_link_libraries(test_taps PRIVATE nxr_compute)
    target_compile_definitions(test_taps PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_taps COMMAND test_taps)
    add_executable(test_morlet test/test_morlet.cpp)
    target_link_libraries(test_morlet PRIVATE nxr_compute)
    target_compile_definitions(test_morlet PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_morlet COMMAND test_morlet)
    add_executable(test_hat test/test_hat.cpp)
    target_link_libraries(test_hat PRIVATE nxr_compute)
    target_compile_definitions(test_hat PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_hat COMMAND test_hat)
    add_executable(test_tower test/test_tower.cpp)
    target_link_libraries(test_tower PRIVATE nxr_compute)
    target_compile_definitions(test_tower PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_tower COMMAND test_tower)
    add_executable(test_pyramid test/test_pyramid.cpp)
    target_link_libraries(test_pyramid PRIVATE nxr_compute)
    target_compile_definitions(test_pyramid PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_pyramid COMMAND test_pyramid)
    add_executable(test_pyramid_threads test/test_pyramid_threads.cpp)
    target_link_libraries(test_pyramid_threads PRIVATE nxr_compute)
    add_test(NAME test_pyramid_threads COMMAND test_pyramid_threads)
    add_executable(test_parallel_allocation test/test_parallel_allocation.cpp)
    target_link_libraries(test_parallel_allocation PRIVATE nxr_compute)
    add_test(NAME test_parallel_allocation COMMAND test_parallel_allocation)
    add_executable(test_integrals test/test_integrals.cpp)
    target_link_libraries(test_integrals PRIVATE nxr_compute)
    target_compile_definitions(test_integrals PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_integrals COMMAND test_integrals)
    add_executable(test_frame_tiling test/test_frame_tiling.cpp)
    target_link_libraries(test_frame_tiling PRIVATE nxr_compute)
    add_test(NAME test_frame_tiling COMMAND test_frame_tiling)
    add_executable(test_form_columns test/test_form_columns.cpp)
    target_link_libraries(test_form_columns PRIVATE nxr_compute)
    target_compile_definitions(test_form_columns PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_form_columns COMMAND test_form_columns)
    add_executable(test_forms test/test_forms.cpp)
    target_link_libraries(test_forms PRIVATE nxr_compute)
    target_compile_definitions(test_forms PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_forms COMMAND test_forms)
    add_executable(test_glue test/test_glue.cpp)
    target_link_libraries(test_glue PRIVATE nxr_compute)
    target_compile_definitions(test_glue PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_glue COMMAND test_glue)
    add_executable(test_sphere_registration test/test_sphere_registration.cpp)
    target_link_libraries(test_sphere_registration PRIVATE nxr_compute)
    add_test(NAME test_sphere_registration COMMAND test_sphere_registration)

    add_executable(test_frames test/test_frames.cpp)
    target_link_libraries(test_frames PRIVATE nxr_compute)
    target_compile_definitions(test_frames PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_frames COMMAND test_frames)

    add_executable(test_opticalflow_methods test/test_opticalflow_methods.cpp)
    target_link_libraries(test_opticalflow_methods PRIVATE nxr_compute)
    add_test(NAME test_opticalflow_methods COMMAND test_opticalflow_methods)

    add_executable(test_filter test/test_filter.cpp)
    target_link_libraries(test_filter PRIVATE nxr_compute)
    target_compile_definitions(test_filter PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME test_filter COMMAND test_filter)

    add_executable(test_vector_covariant_operators test/test_vector_covariant_operators.cpp)
    target_link_libraries(test_vector_covariant_operators PRIVATE nxr_compute)
    add_test(NAME test_vector_covariant_operators COMMAND test_vector_covariant_operators)
endif()

# CI runs a pull request only the tests of the libraries it touches (.github/workflows/ci.yml): every test added here carries this label.
set_property(DIRECTORY APPEND PROPERTY LABELS compute)
