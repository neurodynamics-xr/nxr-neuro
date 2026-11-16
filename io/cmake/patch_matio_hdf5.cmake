# Run on matio's source (cmake -DMATIO_SRC=<dir> -P): matio's find_package(HDF5) cannot see an HDF5 added to the same build with
# FetchContent, so the call is replaced by the facts of that build (HDF5_VERSION is set by nxr-io's CMakeLists.txt). Idempotent.
set(f "${MATIO_SRC}/cmake/thirdParties.cmake")
file(READ "${f}" t)
string(REPLACE "        find_package(HDF5)\n" "        set(HDF5_FOUND TRUE)   # nxr-io: the HDF5 built in this tree\n" t "${t}")
file(WRITE "${f}" "${t}")
