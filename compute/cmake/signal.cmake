# nxr_compute: the signal toolbox (compute/include/nxr/signal/, compute/src/signal/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/signal/complex_signal.cpp
  src/signal/correlation.cpp
  src/signal/crossspectral_bins.cpp
  src/signal/crossspectral_decomposition.cpp
  src/signal/crossspectral_measures.cpp
  src/signal/decimation.cpp
  src/signal/filters.cpp
  src/signal/filters_fir.cpp
  src/signal/filters_fir_design.cpp
  src/signal/filters_iir.cpp
  src/signal/forms.cpp
  src/signal/fourier.cpp
  src/signal/gabor.cpp
  src/signal/graphfilterbank.cpp
  src/signal/graphtransforms.cpp
  src/signal/graphfilterbank_operator.cpp
  src/signal/graphwavelet.cpp
  src/signal/istft.cpp
  src/signal/jointtransforms.cpp
  src/signal/timefilterbank_kinds.cpp
  src/signal/wavelet_cross.cpp
  src/signal/windowed_graphtransforms.cpp
  src/signal/intrinsic_forms.cpp
  src/signal/hodge_decomposition.cpp
  src/signal/isolines.cpp
  src/signal/streamlines.cpp
  src/signal/jointfilterbank.cpp
  src/signal/jointwavelet.cpp
  src/signal/kernels.cpp
  src/signal/opticalflow.cpp
  src/signal/opticalflow_connection.cpp
  src/signal/opticalflow_methods.cpp
  src/signal/phase_gradient.cpp
  src/signal/phase_gradient_sums.cpp
  src/signal/polarisation.cpp
  src/signal/riesz.cpp
  src/signal/spectral.cpp
  src/signal/spectral_scaling.cpp
  src/signal/speedspectrum.cpp
  src/signal/timefilterbank.cpp
  src/signal/wavelet.cpp
)

set_source_files_properties(src/signal/filters_fir.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")    # nxr's taps.py (numpy: no FMA)
set_source_files_properties(src/signal/forms.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # a port of the Studio's fieldForms.ts / faceGradient.ts (no FMA in JavaScript)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_gabor test/test_gabor.cpp)
  target_link_libraries(test_gabor PRIVATE nxr_compute)
  add_test(NAME test_gabor COMMAND test_gabor)
  add_executable(test_intrinsic_forms test/test_intrinsic_forms.cpp)
  target_link_libraries(test_intrinsic_forms PRIVATE nxr_compute)
  add_test(NAME test_intrinsic_forms COMMAND test_intrinsic_forms)
  add_executable(test_patterns test/test_patterns.cpp)
  target_link_libraries(test_patterns PRIVATE nxr_compute)
  add_test(NAME test_patterns COMMAND test_patterns)
  add_executable(test_spectral_scaling test/test_spectral_scaling.cpp)
  target_link_libraries(test_spectral_scaling PRIVATE nxr_compute)
  add_test(NAME test_spectral_scaling COMMAND test_spectral_scaling)
  add_executable(test_signal_design test/test_signal_design.cpp)
  target_link_libraries(test_signal_design PRIVATE nxr_compute)
  target_compile_definitions(test_signal_design PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_signal_design COMMAND test_signal_design)
  add_executable(test_signal_grid test/test_signal_grid.cpp)
  target_link_libraries(test_signal_grid PRIVATE nxr_compute)
  target_compile_definitions(test_signal_grid PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_signal_grid COMMAND test_signal_grid)
  add_executable(test_fft test/test_fft.cpp)
  target_link_libraries(test_fft PRIVATE nxr_compute)
  target_compile_definitions(test_fft PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_fft COMMAND test_fft)
  add_executable(test_filterbank test/test_filterbank.cpp)
  target_link_libraries(test_filterbank PRIVATE nxr_compute)
  target_compile_definitions(test_filterbank PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_filterbank COMMAND test_filterbank)
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
  add_executable(test_quadrature test/test_quadrature.cpp)
  target_link_libraries(test_quadrature PRIVATE nxr_compute)
  target_compile_definitions(test_quadrature PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_quadrature COMMAND test_quadrature)
  add_executable(test_wt_at test/test_wt_at.cpp)
  target_link_libraries(test_wt_at PRIVATE nxr_compute)
  add_test(NAME test_wt_at COMMAND test_wt_at)
  add_executable(test_psd_linear_map test/test_psd_linear_map.cpp)
  target_link_libraries(test_psd_linear_map PRIVATE nxr_compute)
  add_test(NAME test_psd_linear_map COMMAND test_psd_linear_map)
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
  add_executable(test_phase test/test_phase.cpp)
  target_link_libraries(test_phase PRIVATE nxr_compute)
  target_compile_definitions(test_phase PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_phase COMMAND test_phase)
  add_executable(test_monogenic test/test_monogenic.cpp)
  target_link_libraries(test_monogenic PRIVATE nxr_compute)
  add_test(NAME test_monogenic COMMAND test_monogenic)
  add_executable(test_cross_spectrum test/test_cross_spectrum.cpp)
  target_link_libraries(test_cross_spectrum PRIVATE nxr_compute)
  add_test(NAME test_cross_spectrum COMMAND test_cross_spectrum)
  add_executable(test_band_moments test/test_band_moments.cpp)
  target_link_libraries(test_band_moments PRIVATE nxr_compute)
  add_test(NAME test_band_moments COMMAND test_band_moments)
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
  add_executable(test_pyramid test/test_pyramid.cpp)
  target_link_libraries(test_pyramid PRIVATE nxr_compute nxr_meeg)
  target_compile_definitions(test_pyramid PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_pyramid COMMAND test_pyramid)
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
  add_executable(test_isoline_methods test/test_isoline_methods.cpp)
  target_link_libraries(test_isoline_methods PRIVATE nxr_compute)
  add_test(NAME test_isoline_methods COMMAND test_isoline_methods)
  add_executable(test_opticalflow_methods test/test_opticalflow_methods.cpp)
  target_link_libraries(test_opticalflow_methods PRIVATE nxr_compute)
  add_test(NAME test_opticalflow_methods COMMAND test_opticalflow_methods)
  add_executable(test_filter test/test_filter.cpp)
  target_link_libraries(test_filter PRIVATE nxr_compute)
  target_compile_definitions(test_filter PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_filter COMMAND test_filter)
  add_executable(test_decimation test/test_decimation.cpp)   # decimate: chunk-size independence with the halo, zero phase, no alias
  target_link_libraries(test_decimation PRIVATE nxr_compute)
  add_test(NAME test_decimation COMMAND test_decimation)
  foreach(t test_crossspectrum_bins test_tensor test_wave)   # moved from meeg: they read compute's functions (and meeg's source power)
    add_executable(${t} test/${t}.cpp)
    target_link_libraries(${t} PRIVATE nxr_compute nxr_meeg)
    target_compile_definitions(${t} PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()
