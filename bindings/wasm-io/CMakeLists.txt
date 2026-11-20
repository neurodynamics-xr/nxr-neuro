# nxr-io in WebAssembly (Emscripten). For now the build and its smoke test; the binding (the JavaScript interface) comes once its target is chosen.
add_executable(nxr_io_wasm_smoke smoke.cpp)
target_link_libraries(nxr_io_wasm_smoke PRIVATE nxr_io)
# NODERAWFS: in Node the WASM reads and writes the real disk (the smoke test only; a browser build mounts its storage instead). ALLOW_MEMORY_GROWTH: arrays of any size.
target_link_options(nxr_io_wasm_smoke PRIVATE -sNODERAWFS=1 -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1 -sENVIRONMENT=node)

# THE MODULE (js/index.mjs loads it): Embind, an ES module factory. Node first: NODERAWFS gives it the real disk (the browser's file connection comes later).
# The commands of the command line's table as functions (src/binding.cpp: arrays, options and results converted, no logic; bindings/common/commands.h calls the library's
# function of each command, as the Python face does), and the geometry compute on a mesh in memory (bindings/wasm). The command line itself is not compiled in.
if(NOT TARGET nxr_pipelines)   # an io-only configure (the smoke test above): the module needs compute, meeg and the pipelines (-DNXR_NEURO_BUILD_PIPELINES=ON); its commands call them
  message(STATUS "nxr-io WebAssembly: the module is skipped (the pipelines are not built); the smoke test only")
  return()
endif()
add_executable(nxr_io_wasm src/binding.cpp)
target_compile_definitions(nxr_io_wasm PRIVATE NXR_NEURO_VERSION="${PROJECT_VERSION}")
target_include_directories(nxr_io_wasm PRIVATE ${PROJECT_SOURCE_DIR}/bindings/common)
set_target_properties(nxr_io_wasm PROPERTIES OUTPUT_NAME nxr_io SUFFIX ".mjs")
if(TARGET nxr_pipelines)   # compute, meeg and the pipelines: `run`, `compile`, the reads that compute (read crossspectrum, query measurements)
  target_link_libraries(nxr_io_wasm PRIVATE nxr_pipelines)
  # the GEOMETRY COMPUTE on a mesh in memory (the toolbox functions `<toolbox>.<function>` on the surface as data: operators, eigenmodes, frames, geodesics, ...) - the compute binding's Embind, in the same module
  target_sources(nxr_io_wasm PRIVATE ${PROJECT_SOURCE_DIR}/bindings/wasm/src/nxr_compute_wasm.cpp)
endif()
target_link_options(nxr_io_wasm PRIVATE -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createNxrIo -sNODERAWFS=1 -sENVIRONMENT=node -sALLOW_MEMORY_GROWTH=1 -sEMBIND_STD_STRING_IS_UTF8=1
  # the geometry compute's needs (bindings/wasm): an 8 MB stack (Eigen's SimplicialLLT on a ~14k-vertex Laplacian overflows the default 64 KB), the heap views
  # and malloc/free its JS wrapper passes arrays and cancel/progress slots through, and the message of a C++ exception thrown to JS
  -sSTACK_SIZE=8388608 -sEXPORT_EXCEPTION_HANDLING_HELPERS=1
  "-sEXPORTED_RUNTIME_METHODS=['HEAP8','HEAPU8','HEAP16','HEAPU16','HEAP32','HEAPU32','HEAPF32','HEAPF64','UTF8ToString','getValue','setValue','getExceptionMessage']"
  "-sEXPORTED_FUNCTIONS=['_malloc','_free']")
# the client next to the module, so the build folder is a loadable package (index.mjs imports ./nxr_io.mjs)
add_custom_command(TARGET nxr_io_wasm POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/js/index.mjs ${CMAKE_CURRENT_SOURCE_DIR}/js/index.d.ts ${CMAKE_CURRENT_SOURCE_DIR}/js/commands.mjs ${CMAKE_CURRENT_SOURCE_DIR}/js/commands.d.ts ${CMAKE_CURRENT_SOURCE_DIR}/js/arrays.mjs $<TARGET_FILE_DIR:nxr_io_wasm>
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/bindings/wasm/js/wrap.mjs $<TARGET_FILE_DIR:nxr_io_wasm>/compute.mjs)
