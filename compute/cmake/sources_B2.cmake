# Sources of nxr_compute owned by refactor batch B2 (signal), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/signal/intrinsic_forms.cpp
  src/signal/crossspectral_decomposition.cpp
  src/signal/spectral_scaling.cpp
  src/signal/fourier.cpp
  src/signal/graphfilterbank.cpp
  src/signal/resample.cpp
  src/pyramid.cpp
  src/signal/timefilterbank.cpp
  src/signal/jointfilterbank.cpp
  src/signal/kernels.cpp
  src/signal/spectral.cpp
  src/signal/crossspectral_measures.cpp
  src/signal/filters.cpp
  src/signal/wavelet.cpp
  src/signal/complex_signal.cpp
  src/signal/opticalflow.cpp
  src/signal/opticalflow_connection.cpp
  src/signal/speedspectrum.cpp
  src/signal/opticalflow_methods.cpp
  src/signal/forms.cpp
  src/signal/filters_fir.cpp
  src/signal/phase_gradient.cpp
  src/signal/riesz.cpp
)

set_source_files_properties(src/signal/filters_fir.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")    # nxr's taps.py (numpy: no FMA)
set_source_files_properties(src/signal/forms.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")   # a port of the Studio's fieldForms.ts / faceGradient.ts (no FMA in JavaScript)
