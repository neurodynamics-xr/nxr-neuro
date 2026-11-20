// A dataset opened once (nxr/io/task/handle.h; issue #166): the open readers a handle keeps, the writer lock a write handle holds and lends to its commands.
#include "nxr/io/task/handle.h"

#include <unistd.h>

#include <cstdlib>
#include <map>
#include <mutex>

#include "commands_open.h"
#include "nxr/io/writer_lock.h"

namespace nxr::io::task {

namespace fs = std::filesystem;

namespace {
thread_local Handle* current = nullptr;   // the handle the commands of this thread work through (set by a HandleScope)

std::string canonical(const std::string& p) {
  std::error_code ec;
  const fs::path c = fs::weakly_canonical(p, ec);
  return ec ? p : c.string();
}
}  // namespace

struct Handle::Impl {
  std::string dataset, where, program;   // `where`: the dataset's canonical path (what a command's folder is compared with)
  Mode mode = Mode::read;
  bool closed = false;
  WriterLock lock;
  std::mutex busy;                       // held by a HandleScope: one call at a time
  std::optional<bool> catalogue;
  std::map<std::string, std::unique_ptr<DatasetWriter>> readers;   // by subject (a catalogue) and templates; the one-database dataset: one
};

Handle::Handle(const std::string& dataset, Mode mode, const std::string& program) : impl_(std::make_unique<Impl>()) {
  impl_->dataset = dataset;
  impl_->where = canonical(dataset);
  impl_->program = program;
  impl_->mode = mode;
  if (mode == Mode::write) impl_->lock = acquire_writer_lock(dataset, program);
}
Handle::~Handle() { try { close(); } catch (...) {} }
const std::string& Handle::dataset() const { return impl_->dataset; }
Handle::Mode Handle::mode() const { return impl_->mode; }
bool Handle::closed() const { return impl_->closed; }
std::string Handle::lock_file() const { return impl_->lock ? impl_->lock.file().string() : std::string(); }
void Handle::close() {
  if (impl_->closed) return;
  impl_->closed = true;
  impl_->readers.clear();   // the databases first, the lock after them
  if (impl_->lock) impl_->lock.release();
}

struct HandleScope::State {
  Handle* handle = nullptr;
  Handle* previous = nullptr;
  std::unique_lock<std::mutex> turn;
  bool lent = false;                      // the writer lock lent through the environment
  std::optional<std::string> env_before;
};

HandleScope::HandleScope(Handle* handle) : s_(std::make_unique<State>()) {
  if (!handle) return;
  Handle::Impl& i = *handle->impl_;
  if (i.closed)
    throw Refused("the handle of " + i.dataset + " is closed" + (i.mode == Handle::Mode::write ? " (its writer lock released)" : "") + ": open it again", true);
  s_->turn = std::unique_lock<std::mutex>(i.busy);
  s_->handle = handle;
  s_->previous = current;
  current = handle;
  if (i.lock) {   // the commands borrow the lock the handle holds
    if (const char* e = std::getenv(kWriterLockEnv)) s_->env_before = e;
    // the pid the lock file names (this process's: in WebAssembly under Node, the host's process.pid, not Emscripten's stand-in)
    long long pid = ::getpid();
    if (const auto h = read_writer_lock(i.lock.file().parent_path()); h && h->contains("pid") && (*h)["pid"].is_number_integer()) pid = (*h)["pid"].get<long long>();
    const std::string held = i.lock.file().string() + ":" + std::to_string(pid);
    ::setenv(kWriterLockEnv, held.c_str(), 1);
    s_->lent = true;
  }
}
HandleScope::~HandleScope() {
  if (!s_ || !s_->handle) return;
  current = s_->previous;
  if (s_->lent) {
    if (s_->env_before) ::setenv(kWriterLockEnv, s_->env_before->c_str(), 1);
    else ::unsetenv(kWriterLockEnv);
  }
}

Lease open_reader(const std::string& folder, const std::optional<std::string>& subject, const std::vector<std::string>& templates, const Progress& progress) {
  using namespace commands_detail;
  Handle* h = current;
  if (h && !h->impl_->closed && canonical(folder) == h->impl_->where) {
    Handle::Impl& i = *h->impl_;
    if (!i.catalogue) i.catalogue = is_catalogue_dataset(folder);
    std::string key = *i.catalogue ? subject.value_or("") : std::string();
    for (const auto& t : templates) key += "\n" + t;
    const auto it = i.readers.find(key);
    if (it != i.readers.end()) return Lease(it->second.get());
    auto w = std::make_unique<DatasetWriter>(open_dataset(folder, subject, reader(), templates, progress));
    DatasetWriter* p = w.get();
    i.readers.emplace(key, std::move(w));
    return Lease(p);
  }
  return Lease(open_dataset(folder, subject, reader(), templates, progress));
}

}  // namespace nxr::io::task
