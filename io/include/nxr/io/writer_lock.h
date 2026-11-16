// nxr-io — THE WRITER LOCK (OPEN-DEFECTS #26): ONE WRITER A DATASET, across processes. nxr.io.writer_lock, byte-compatible with it and with the
// Studio's `backend/src/service/writer-lock.ts`: the three read each other's files.
//
//   <dataset>/.writer.lock   JSON {pid, host, program, started_utc}, created atomically (O_CREAT|O_EXCL)
//
// ACQUIRE creates the file and writes its JSON at once; if it exists it is read: STALE (same host only) is a dead pid, a start before this
// machine's boot (a reused pid), or an unreadable file older than 30 s (taken over through `<lock>.break`); a live pid, a fresh unreadable file or
// another host is REFUSED (Error `writer_lock_held`). RELEASE removes the file, only while it still names this process. A crash leaves the file
// naming a dead pid: the next writer on this host takes it over. THE HANDSHAKE: NXR_WRITER_LOCK_HELD=<lock file>:<pid> borrows the holder's lock
// (neither created nor removed, and nothing recovered).
#pragma once
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "nxr/io/json_text.h"

namespace nxr::io {

constexpr const char* kWriterLockFile = ".writer.lock";
constexpr const char* kWriterLockEnv = "NXR_WRITER_LOCK_HELD";

class WriterLock {
 public:
  WriterLock() = default;
  WriterLock(WriterLock&&) noexcept;
  WriterLock& operator=(WriterLock&&) noexcept;
  WriterLock(const WriterLock&) = delete;
  WriterLock& operator=(const WriterLock&) = delete;
  ~WriterLock();  // releases

  explicit operator bool() const { return static_cast<bool>(s_); }
  bool borrowed() const { return s_ && s_->borrowed; }
  const std::filesystem::path& file() const { return s_->file; }
  void release();

 private:
  friend WriterLock acquire_writer_lock(const std::filesystem::path&, const std::string&);
  struct State {
    std::filesystem::path file;
    bool borrowed = false;
    bool released = false;
  };
  std::shared_ptr<State> s_;
};

// TAKE the writer lock of the dataset at `dataset_dir` (the folder must exist). Throws Error `writer_lock_held` while another live writer holds
// it, `not_writable` when the folder cannot be written.
WriterLock acquire_writer_lock(const std::filesystem::path& dataset_dir, const std::string& program = "nxr-io");
// The lock's holder as its file states it; nullopt when there is no file or it cannot be read.
std::optional<Json> read_writer_lock(const std::filesystem::path& dataset_dir);
// Can this process WRITE a dataset's database (the file, its -wal/-shm/-journal, the folder)? nullopt when it can; else the code (EROFS, EACCES)
// and the path that refused.
std::optional<std::pair<std::string, std::filesystem::path>> database_not_writable(const std::filesystem::path& file);

}  // namespace nxr::io
