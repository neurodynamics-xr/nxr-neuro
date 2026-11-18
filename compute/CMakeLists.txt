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


# The pinned geometry-central commit (7 hex), read from the submodule at configure time: nxr::manifold::geometryCentralVersion() and the gauge identity
# (docs/specs/2026-11-17-geometry-central-switch.md section 7). "unknown" outside a git checkout (a source tarball).
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/deps/geometry-central rev-parse --short=7 HEAD
                OUTPUT_VARIABLE NXR_GC_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE NXR_GC_RC)
if(NOT NXR_GC_RC EQUAL 0 OR NXR_GC_COMMIT STREQUAL "")
  set(NXR_GC_COMMIT "unknown")
endif()
add_library(nxr_compute STATIC)
# the toolboxes (compute/cmake/<toolbox>.cmake): each lists its sources, their per-file flags and its tests; the layering between them is cmake/check_toolboxes.cmake
set(NXR_COMPUTE_TOOLBOXES core quantity optimisation stats geometry operators selection measurement select signal synthesis meeg pending_b6)
foreach(tb IN LISTS NXR_COMPUTE_TOOLBOXES)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${tb}.cmake)
    include(cmake/${tb}.cmake)
  endif()
endforeach()

target_include_directories(nxr_compute PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${spectra_SOURCE_DIR}/include
)

target_compile_features(nxr_compute PUBLIC cxx_std_17)

target_link_libraries(nxr_compute PUBLIC
  geometry-central
)
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

# (the tests of each toolbox are in its cmake file, above)

# CI runs a pull request only the tests of the libraries it touches (.github/workflows/ci.yml): every test added here carries this label.
set_property(DIRECTORY APPEND PROPERTY LABELS compute)

# the kind of every test of this folder (tests/kinds.tsv); a test missing from the table fails the configure
if(COMMAND nxr_test_kinds)
  nxr_test_kinds()
endif()
