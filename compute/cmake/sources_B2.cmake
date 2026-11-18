# Sources of nxr_compute owned by refactor batch B2 (signal), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/signal/intrinsic_forms.cpp
  src/signal/patterns.cpp
  src/signal/spectral_scaling.cpp
  src/signal/fft.cpp
  src/signal/filterbank.cpp
  src/signal/frame_tiling.cpp
  src/pyramid.cpp
  src/signal/timebank.cpp
  src/signal/jointbank.cpp
  src/signal/spacetime_kernels.cpp
  src/signal/welch.cpp
  src/quadrature.cpp
  src/signal/filter.cpp
  src/signal/wt.cpp
  src/signal/analytic.cpp
  src/signal/opticalflow.cpp
  src/signal/opticalflow_connection.cpp
  src/signal/speedspectrum.cpp
  src/signal/opticalflow_methods.cpp
  src/signal/forms.cpp
  src/signal/taps.cpp
  src/signal/phase.cpp
  src/signal/monogenic.cpp
)

set_source_files_properties(src/signal/taps.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")    # nxr's taps.py (numpy: no FMA)
set_source_files_properties(src/signal/forms.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # a port of the Studio's fieldForms.ts / faceGradient.ts (no FMA in JavaScript)
