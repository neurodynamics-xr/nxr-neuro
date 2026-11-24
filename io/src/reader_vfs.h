// io/src only: the SQLite VFS a READ-ONLY connection opens with (database.cpp). A read-only open writes nothing.
//
// Natively the file locks tell a reader that another process is writing (its RESERVED or PENDING lock), and SQLite waits for it (the busy timeout). WebAssembly (Emscripten's file system
// under Node) has no file locks: a reader sees no writer, so the writer's rollback journal, once its header is written (a commit, or a transaction whose cache spilled), looks HOT (left by a
// writer that died) and SQLite tries to roll it back, which a read-only connection cannot: "attempt to write a readonly database". The reader VFS (WebAssembly only) wraps the default one so
// that a read-only connection never tries: while the journal beside the database is hot it answers BUSY to the shared lock (SQLite's busy timeout then waits, as a native reader waits for the
// writer's commit), and it reports the reserved lock held while a journal exists (the journal is the writer's, never this reader's to roll back). A journal left by a writer that died
// is rolled back by the next writable open.
// What it cannot do: without locks a writer does not see this reader either, so a commit that starts while a read is running can still be read half made; the reader then fails or reads
// rows of both states. Natively nothing changes (the default VFS, with its locks).
#pragma once

namespace nxr::io::detail {
// the name of the VFS a read-only connection opens with: "nxr-reader" in WebAssembly (registered on first use), nullptr (the default) natively
const char* reader_vfs();
}  // namespace nxr::io::detail
