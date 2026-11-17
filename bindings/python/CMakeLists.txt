# ── nxr_neuro._core: the nanobind module behind the Python package (the wheel) ───────────────────────────────────────────────────────────────
#
# A binding shell, as the MEX and Node ones are: it calls nxr_compute (the pure functions) and nxr_io_formats (the pure file formats) and nothing else
# (the dependency rule's 2026-11-17 amendment, docs/reference.md). Built only when asked (-DNXR_NEURO_BUILD_PYTHON=ON, which scikit-build-core does).
# The stable ABI (one wheel per platform for Python >= 3.12) is used when the Python is new enough.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(nxr_neuro_core NB_STATIC STABLE_ABI NB_DOMAIN nxr
  src/module.cpp
  src/compute.cpp
  src/formats.cpp)
set_target_properties(nxr_neuro_core PROPERTIES OUTPUT_NAME _core)
target_compile_features(nxr_neuro_core PRIVATE cxx_std_17)
target_link_libraries(nxr_neuro_core PRIVATE nxr_compute nxr_io_formats)
if(COMMAND nxr_io_hide_static_symbols)
  # the format readers use matio (EEGLAB), which carries HDF5 as static archives: neither is exported from the module (it may meet h5py's or MATLAB's)
  nxr_io_hide_static_symbols(nxr_neuro_core)
endif()

# Where scikit-build-core installs it: the package folder. The licence and the third-party notices go into the wheel's metadata folder.
# (a component of its own: the dependencies' install rules (HDF5, matio, geometry-central) stay out of the wheel; pyproject.toml installs this one)
install(TARGETS nxr_neuro_core LIBRARY DESTINATION nxr_neuro COMPONENT nxr_neuro_python)
if(DEFINED SKBUILD_METADATA_DIR)
  install(FILES ${CMAKE_SOURCE_DIR}/LICENSE ${CMAKE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md ${CMAKE_SOURCE_DIR}/AUTHORS
          DESTINATION ${SKBUILD_METADATA_DIR}/licenses COMPONENT nxr_neuro_python)
endif()
