# THE LAYERING GUARD (the architecture policy, ARCHITECTURE.md: a single io service is responsible for chunk loading and looping over chunks; compute pipelines neither know nor care about either). Run as a test: cmake -DROOT=<repo> -P cmake/check_layering.cmake.
# THE STORE: a subject's arrays and node documents are zarr KEYS read and written through nxr::io::Store only
# (io/include/nxr/io/store.h; FolderStore is the one backend). Fails when the array reader, the array writer or the store-node removal use the filesystem
# (`std::filesystem`, `fs::`, `fstream`), or when pipelines, cli or bindings reach a subject's folder (`sub.folder()`) instead of its store.
# Fails when a file of pipelines/graph/ (the graph executor, nxr_graph) includes an io, bst or pipelines header.
# Fails when pipelines/ or cli/ (tests excluded) hold SQL text (a quoted SELECT, INSERT INTO, DELETE FROM, UPDATE .. SET, PRAGMA, json_extract) or call a database handle's `one`/`all`/`execute`/`query`/`insert`/`upsert`/`tx` directly: only io touches the database.
# Fails when code outside io/ (pipelines, cli, compute, meeg, bindings; tests excluded) reads a recording's samples or works out chunks itself:
#   a region read along a running sample index (`.read({..}, {.. + n})`), the chunk's arithmetic (`chunk_end`, `kChunk`, `chunk_span(`, `1 << 15`), the retired
#   read size (`chunk_samples`, `--chunk-samples`), or the chunk service built anywhere but the one door, pipelines/src/reading.cpp
#   (`span_of`, `source_of`: one `resolve_span(` and one `ChunkSource(`).
file(GLOB_RECURSE files
  ${ROOT}/pipelines/src/*.cpp ${ROOT}/pipelines/include/*.h
  ${ROOT}/pipelines/graph/src/*.cpp ${ROOT}/pipelines/graph/include/nxr/graph/*.h ${ROOT}/pipelines/graph/test/*.cpp
  ${ROOT}/cli/*.cpp ${ROOT}/cli/*.h
  ${ROOT}/compute/src/*.cpp ${ROOT}/compute/include/*.h
  ${ROOT}/meeg/src/*.cpp ${ROOT}/meeg/include/*.h
  ${ROOT}/bindings/*.cpp ${ROOT}/bindings/*.h)
set(bad "")
foreach(f ${ROOT}/io/src/array_reader.cpp ${ROOT}/io/src/array_writer.cpp ${ROOT}/io/src/named_paths.cpp)
  file(STRINGS ${f} lines)
  set(i 0)
  foreach(l IN LISTS lines)
    math(EXPR i "${i} + 1")
    if(l MATCHES "std::filesystem|fs::|<filesystem>|fstream|ifstream|ofstream")
      string(APPEND bad "  ${f}:${i}: the filesystem is used for store keys (use nxr::io::Store): ${l}\n")
    endif()
  endforeach()
endforeach()
set(n_span 0)
set(n_source 0)
foreach(f ${files})
  if(f MATCHES "/deps/")
    continue()
  endif()
  file(STRINGS ${f} lines)
  set(i 0)
  foreach(l IN LISTS lines)
    math(EXPR i "${i} + 1")
    if(l MATCHES "\\.read\\(\\{[^}]*\\}, \\{[^}]*\\+ " OR l MATCHES "chunk_end|kChunk|chunk_span\\(|1 << 15|chunk_samples|--chunk-samples")
      string(APPEND bad "  ${f}:${i}: ${l}\n")
    endif()
    if((f MATCHES "/compute/" OR f MATCHES "/meeg/") AND l MATCHES "#include *[<\"]nxr/(io|io/|bst/|pipelines/)")
      string(APPEND bad "  ${f}:${i}: compute includes io or pipelines (the architecture policy: compute is functions of matrices): ${l}\n")
    endif()
    if(f MATCHES "/pipelines/graph/" AND l MATCHES "#include *[<\"]nxr/(io|io/|bst/|pipelines/)")
      string(APPEND bad "  ${f}:${i}: the graph executor includes io or pipelines (nxr_graph depends on compute, meeg and nlohmann/json only): ${l}\n")
    endif()
    if((f MATCHES "/pipelines/" OR f MATCHES "/cli/") AND NOT f MATCHES "/test/" AND (l MATCHES "\"[^\"]*(SELECT |INSERT INTO|DELETE FROM|UPDATE [a-z_]+ SET|PRAGMA |json_extract)" OR l MATCHES "db(\\(\\))?\\.(one|all|execute|query|insert|upsert|tx)\\("))
      string(APPEND bad "  ${f}:${i}: SQL outside io (only io touches the database: use io's selection_store.h, selection_measurement.h, selection_reads.h, subject_reads.h or the task API): ${l}\n")
    endif()
    if(l MATCHES "(^|[^A-Za-z_])sub[A-Za-z0-9_]*\\.folder\\(\\)")
      string(APPEND bad "  ${f}:${i}: a subject's folder is reached outside io (use its store): ${l}\n")
    endif()
    if(l MATCHES "resolve_span\\(")
      math(EXPR n_span "${n_span} + 1")
    endif()
    if(l MATCHES "ChunkSource\\(")
      math(EXPR n_source "${n_source} + 1")
    endif()
  endforeach()
endforeach()
if(n_span GREATER 1 OR n_source GREATER 1)
  string(APPEND bad "  the chunk service is built outside span_of/source_of (resolve_span: ${n_span}, ChunkSource: ${n_source}; one each allowed)\n")
endif()
if(bad)
  message(FATAL_ERROR "layering violations (recordings are read only by the io chunk service, io/include/nxr/io/chunk_loop.h; SQL and the filesystem only in io):\n${bad}")
endif()
message(STATUS "layering: recordings are read only by the io chunk service")
