// nxr-io — A DATASET OPENED ONCE (docs/specs/2026-11-19-io-service.md, rule 6, 6.1; issue #166): what `nxr_neuro.open(dataset, mode)` and `nxr.open(dataset, mode)` hold.
//
//   task::Handle h("/data/cohort", task::Handle::Mode::write, "nxr-neuro 0.1");   // write: the dataset's writer lock is taken now and held until close()
//   { task::HandleScope use(&h);  task::add_field(request); task::read_field(...); }   // the commands called inside borrow the handle
//
// What a handle keeps between calls:
// - A READER (the commands that only read: `read <class>`, `list recordings | kernels | results`, `show subject`, `query measurements`, ...) opens the subject's database ONCE,
//   read-only (#144: it never migrates, recovers or locks), and keeps it open across calls (`open_reader`). Opened lazily, one for each subject.
// - A WRITE command opens its own database for the length of its unit and closes it: a subject package's record and content hash are finished when its writer closes, so a
//   write is never left half-finished between calls. A write handle's commands borrow the lock the handle holds (io's handshake NXR_WRITER_LOCK_HELD), so another process's
//   write is refused (`writer_lock_held`) until `close()`.
// - A read handle takes no lock.
// A command called with no handle opens and closes what it needs, as before. A handle only applies to commands on ITS dataset (the same folder). A handle is used by one
// call at a time (a scope holds its mutex).
#pragma once
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/task/commands.h"
#include "nxr/io/writer.h"

namespace nxr::io::task {

class Lease;

class Handle {
 public:
  enum class Mode { read, write };
  /** open a handle on `dataset` (a catalogue, or a one-database dataset); `write` takes its writer lock (`writer_lock_held` when another live writer holds it) */
  Handle(const std::string& dataset, Mode mode, const std::string& program);
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle();

  const std::string& dataset() const;
  Mode mode() const;
  bool closed() const;
  /** a write handle's lock file (empty for a read handle) */
  std::string lock_file() const;
  /** close the open readers and release the writer lock; every later call through the handle is refused. Closing twice is not an error. */
  void close();

 private:
  friend class HandleScope;
  friend class Lease;
  friend std::optional<std::string> current_handle_token(const std::string&);
  friend Lease open_reader(const std::string&, const std::optional<std::string>&, const std::vector<std::string>&, const Progress&);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/** While alive, the commands called on this thread work through the handle (its dataset only). A null handle is no handle. Refuses a closed handle (`Refused`). */
class HandleScope {
 public:
  explicit HandleScope(Handle* handle);
  HandleScope(const HandleScope&) = delete;
  HandleScope& operator=(const HandleScope&) = delete;
  ~HandleScope();

 private:
  struct State;
  std::unique_ptr<State> s_;
};

/** A token naming the handle the commands of this thread work through on `dataset` (none: no handle, or a handle of another dataset): what a write in pieces records as its owner (commands.h `PieceOptions`). */
std::optional<std::string> current_handle_token(const std::string& dataset);

/** An opened dataset a command works on: its own (closed with the lease), or a handle's (kept open). Use `*lease` as the DatasetWriter. */
class Lease {
 public:
  explicit Lease(DatasetWriter own) : own_(std::move(own)), ref_(&*own_) {}
  explicit Lease(DatasetWriter* borrowed) : ref_(borrowed) {}
  Lease(Lease&& o) noexcept : own_(std::move(o.own_)), ref_(own_ ? &*own_ : o.ref_) {}
  Lease& operator=(Lease&&) = delete;
  DatasetWriter& operator*() { return *ref_; }
  DatasetWriter* operator->() { return ref_; }

 private:
  std::optional<DatasetWriter> own_;
  DatasetWriter* ref_;
};

/** The dataset a READ-ONLY command works on: a catalogue's subject package (found through the catalogue; `subject` required), or a one-database dataset (store formats 50 to 52),
 *  opened read-only, no lock, no recovery. Through a handle on `folder`, the handle's open reader (opened on first use); otherwise a reader of its own. */
Lease open_reader(const std::string& folder, const std::optional<std::string>& subject, const std::vector<std::string>& templates = {}, const Progress& progress = {});

}  // namespace nxr::io::task
