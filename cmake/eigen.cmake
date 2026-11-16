# Eigen 3.4, fetched into the BUILD directory (the version nxr-compute is written for; an installed Eigen 5 is not what geometry-central and
# Spectra are tested with, and it fails geometry-central's version check). geometry-central would otherwise find an installed Eigen, or DOWNLOAD
# one INTO ITS SOURCE TREE (compute/deps/geometry-central): a target of this name keeps both from happening.
# -DNXR_EIGEN_DIR=<an Eigen 3 source folder, the one holding `Eigen/`> uses one you already have.
include(FetchContent)
if(NOT TARGET Eigen3::Eigen)
  if(NXR_EIGEN_DIR)
    set(_eigen_src "${NXR_EIGEN_DIR}")
  else()
    FetchContent_Declare(eigen34
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(eigen34)
    if(NOT eigen34_POPULATED)
      cmake_policy(PUSH)
      if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
      endif()
      FetchContent_Populate(eigen34)
      cmake_policy(POP)
    endif()
    set(_eigen_src "${eigen34_SOURCE_DIR}")
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_eigen_src}")
  set(EIGEN3_FOUND TRUE)
  set(EIGEN3_INCLUDE_DIR "${_eigen_src}")
endif()
