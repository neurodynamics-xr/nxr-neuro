# Sources of nxr_compute owned by refactor batch B2 (signal), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/intrinsic_forms.cpp
  src/patterns.cpp
  src/spectral_scaling.cpp
  src/fft.cpp
  src/filterbank.cpp
  src/frame_tiling.cpp
  src/pyramid.cpp
  src/timebank.cpp
  src/jointbank.cpp
  src/spacetime_kernels.cpp
  src/welch.cpp
  src/quadrature.cpp
  src/filter.cpp
  src/wt.cpp
  src/analytic.cpp
  src/opticalflow.cpp
  src/opticalflow_connection.cpp
  src/speedspectrum.cpp
  src/opticalflow_methods.cpp
  src/forms.cpp
  src/taps.cpp
  src/phase.cpp
  src/monogenic.cpp
)

set_source_files_properties(src/taps.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")    # nxr's taps.py (numpy: no FMA)
set_source_files_properties(src/forms.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # a port of the Studio's fieldForms.ts / faceGradient.ts (no FMA in JavaScript)
