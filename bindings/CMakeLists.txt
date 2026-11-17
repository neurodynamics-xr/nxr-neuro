# The bindings: faces, as the command line is. All of them carry nxr_compute; Python and MATLAB also the pure file formats and the pipeline level (nxr_pipelines, pipelines/face.h).
# Each is its own CMakeLists.txt. Native: Node (only through cmake-js), the CLI smoke, MATLAB MEX (only when MATLAB is found).
# Under Emscripten: wasm. Python (the wheel's nanobind module, bindings/python) only when NXR_NEURO_BUILD_PYTHON is ON.
if(EMSCRIPTEN)
  add_subdirectory(wasm)
elseif(NXR_NEURO_PYTHON_ONLY)
  add_subdirectory(python)   # the wheel build (scikit-build-core): the Python module alone; no MATLAB, Node or smoke shells
else()
  add_subdirectory(node)
  add_subdirectory(cli)
  add_subdirectory(mex)
  if(NXR_NEURO_BUILD_PYTHON)
    add_subdirectory(python)   # needs compute and io
  endif()
endif()
