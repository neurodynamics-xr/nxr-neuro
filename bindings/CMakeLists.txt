# The bindings: shells around the nxr_compute library (nothing of io or pipelines is exposed through them yet).
# Each is its own CMakeLists.txt. Native: Node (only through cmake-js), the CLI smoke, MATLAB MEX (only when MATLAB is found).
# Under Emscripten: wasm.
if(EMSCRIPTEN)
  add_subdirectory(wasm)
else()
  add_subdirectory(node)
  add_subdirectory(cli)
  add_subdirectory(mex)
endif()
