# THE DISPATCH GUARD (ARCHITECTURE.md, "Backends and node traits"): a graph node's computation is called in ONE place, `dispatch` (pipelines/src/dispatch.cpp).
# Fails when any other file calls a registered node's functions (prepare, start, chunk, chunk_at, fold, finish, run) through a step: `step->chunk(`, `p->step->fold(`,
# `step("name").run(`, `registry()[i].finish(` and the like. Run as a test: cmake -DROOT=<repo> -P cmake/check_dispatch.cmake.
file(GLOB_RECURSE files
  ${ROOT}/pipelines/src/*.cpp ${ROOT}/pipelines/include/*.h ${ROOT}/pipelines/test/*.cpp
  ${ROOT}/cli/*.cpp ${ROOT}/cli/*.h
  ${ROOT}/bindings/*.cpp ${ROOT}/bindings/*.h)
set(bad "")
set(checked 0)
foreach(f ${files})
  if(f MATCHES "/deps/" OR f MATCHES "/pipelines/src/dispatch\\.cpp$" OR f MATCHES "/pipelines/test/test_node_traits\\.cpp$")
    continue()
  endif()
  math(EXPR checked "${checked} + 1")
  file(STRINGS ${f} lines)
  set(i 0)
  foreach(l IN LISTS lines)
    math(EXPR i "${i} + 1")
    # a step reached by pointer, reference or lookup, then one of its computation functions called
    if(l MATCHES "(step|registry\\(\\)|\\)|\\])(->|\\.)(prepare|start|chunk_at|chunk|fold|finish|run)\\(" AND NOT l MATCHES "^ *//")
      string(APPEND bad "  ${f}:${i}: a node's function is called outside the dispatch point (use NodeCall and dispatch): ${l}\n")
    endif()
  endforeach()
endforeach()
if(checked LESS 10)
  message(FATAL_ERROR "dispatch guard: only ${checked} files found under ${ROOT}: is ROOT right?")
endif()
if(bad)
  message(FATAL_ERROR "dispatch guard failed:\n${bad}")
endif()
message(STATUS "dispatch guard: ${checked} files, no node called outside dispatch")
