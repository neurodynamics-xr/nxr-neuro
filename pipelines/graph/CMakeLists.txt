# nxr_graph — the graph executor: nodes with typed ports, a graph of them checked on types, run block by block with state carried. Included by ../CMakeLists.txt.
# The spec: docs/specs/2026-11-17-compute-and-run.md. It may depend on compute, meeg, Eigen and nlohmann/json; NEVER on io (the layering guard,
# cmake/check_layering.cmake), so the bindings and the WebAssembly `compute` build get the graph without io. Nothing here reads a file, a subject or a database.
# nlohmann/json is a header, vendored in io/extern and shared by the build; including it is a build rule, not a dependency on the io library
if(NOT TARGET nlohmann_json)
  add_library(nlohmann_json INTERFACE)
  target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/io/extern)
endif()
add_library(nxr_graph
  src/node.cpp
  src/dispatch.cpp
  src/graph.cpp
  src/execution.cpp
)
target_include_directories(nxr_graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nxr_graph PUBLIC nlohmann_json)
if(MSVC)
  target_compile_options(nxr_graph PRIVATE /W4)
else()
  target_compile_options(nxr_graph PRIVATE -Wall -Wextra)
endif()

if(NXR_NEURO_BUILD_TESTS)
  add_executable(test_graph test/test_graph.cpp)
  target_link_libraries(test_graph PRIVATE nxr_graph)
  add_test(NAME test_graph COMMAND test_graph)
endif()
