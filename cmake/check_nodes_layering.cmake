# THE NODES LAYERING GUARD (docs/specs/2026-11-17-compute-and-run.md, PR C4): a node takes arrays and parameters, never a database, a subject, a dataset or a file.
# The node code is the registered steps' files: pipelines/src/registry.cpp, pipelines/src/*_steps.cpp and, from C5, pipelines/graph/nodes/. What a node needs beside its
# inputs is found and read by the runner and handed over through nxr::pipelines::graph::Bound (include/nxr/pipelines/bound.h, which the runner implements in src/bound.cpp).
# Run as a test:  cmake -DROOT=<repo> -P cmake/check_nodes_layering.cmake
#
# It fails on, in a node file (comment lines skipped):
#   - an #include of nxr/io/ (a node includes no io header);
#   - `SubjectWriter`, `DatasetWriter`, `SubjectPackage`, `Database`, `.db()`, `.store()`, `std::filesystem`, `fstream`: the handles a node would read or write through.
# There is no allowlist: every node file is clean (C4 emptied it).
cmake_minimum_required(VERSION 3.20)
if(NOT ROOT)
  message(FATAL_ERROR "ROOT is not set")
endif()
file(GLOB node_files ${ROOT}/pipelines/src/registry.cpp ${ROOT}/pipelines/src/*_steps.cpp ${ROOT}/pipelines/graph/nodes/*.cpp)
set(bad "")
set(scanned 0)
foreach(f ${node_files})
  math(EXPR scanned "${scanned} + 1")
  file(STRINGS ${f} lines)
  set(i 0)
  foreach(l IN LISTS lines)
    math(EXPR i "${i} + 1")
    if(l MATCHES "^[ \t]*(//|/\\*|\\*)")
      continue()
    endif()
    if(l MATCHES "#include *[<\"]nxr/io[/.]")
      string(APPEND bad "  ${f}:${i}: a node includes an io header: ${l}\n")
    elseif(l MATCHES "SubjectWriter|DatasetWriter|SubjectPackage|Database|\\.db\\(\\)|->db\\(\\)|\\.store\\(\\)|std::filesystem|fstream")
      string(APPEND bad "  ${f}:${i}: a node holds a database, subject, dataset or file handle (ask the runner: Bound): ${l}\n")
    endif()
  endforeach()
endforeach()
if(scanned EQUAL 0)
  message(FATAL_ERROR "no node file found under ${ROOT}")
endif()
if(bad)
  message(FATAL_ERROR "node code reaches io (docs/specs/2026-11-17-compute-and-run.md, C4):\n${bad}")
endif()
message(STATUS "layering: ${scanned} node files take no database, subject, dataset or file")
