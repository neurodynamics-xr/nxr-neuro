# nxr_core: the small library under nxr_compute and nxr_meeg (dated amendment 2026-11-17, ARCHITECTURE.md dependency table): core types and errors,
# cancellation, progress, parallel, general algebra (congruence) and optimisation's general solvers. Eigen and the platform's threads only.
# Included once, by compute/cmake/sources_B4.cmake or meeg/CMakeLists.txt, whichever comes first. Temporary file: consolidated in B5.
if(NOT TARGET nxr_core)
  add_library(nxr_core STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../src/core/congruence.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/core/parallel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/optimisation/fminsearch.cpp
  )
  # the headers are in compute/include/nxr/{core,optimisation}/
  target_include_directories(nxr_core PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
  target_compile_features(nxr_core PUBLIC cxx_std_17)
  target_link_libraries(nxr_core PUBLIC Eigen3::Eigen)
  if(NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
    target_link_libraries(nxr_core PUBLIC Threads::Threads)
  endif()
  set_target_properties(nxr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
