#include "nxr/io/writer_lock.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/time.h>
#endif

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "nxr/io/database.h"
#include "nxr/io/errors.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {

constexpr double kUnreadableStaleS = 30.0;
constexpr double kBootSlackS = 60.0;
constexpr double kBreakGraceS = 10.0;
#ifdef __EMSCRIPTEN__
constexpr double kProcessSlackS = 1.0;   // a lock written by this process is never earlier than its start; the slack absorbs the clocks' rounding
#endif

std::mutex g_mu;
std::set<fs::path>& held() { static std::set<fs::path> s; return s; }

#ifdef __EMSCRIPTEN__
// IN WEBASSEMBLY getpid() and gethostname() are Emscripten's fixed stand-ins (42, "emscripten"): a lock would name no real process, and no other writer
// could tell a live holder from a dead one. In Node the host's own are read (process.pid, os.hostname(), process.kill(pid, 0)), so the lock means what a
// native one does; elsewhere (a browser) they stay unknown (pid 0).
EM_JS(int, nxr_js_pid, (), { return (typeof process !== 'undefined' && process.pid) ? process.pid : 0; });
EM_JS(char*, nxr_js_host, (), {
  let h = 'unknown';
  try { const os = (typeof process !== 'undefined' && process.getBuiltinModule) ? process.getBuiltinModule('os') : (typeof require !== 'undefined' ? require('os') : null); if (os) h = os.hostname(); } catch (e) {}
  return stringToNewUTF8(h);
});
EM_JS(int, nxr_js_alive, (double pid), {
  if (typeof process === 'undefined' || !process.kill) return 1;
  try { process.kill(pid, 0); return 1; } catch (e) { return e && e.code === 'EPERM' ? 1 : 0; }
});
// THE LOCKS OF THIS PROCESS. Natively one process has one `held()`: a lock file naming this process's pid is live exactly while `held()` lists it. In WebAssembly a Node process
// can run several module instances, on its main thread and in its worker threads, each with its own memory (its own `held()`) and its own `globalThis`, so no record in either
// is the process's. The lock file's own owner fields are used instead: the pid, and an INSTANCE token naming the module instance that wrote it. A lock naming this pid that
// this instance does not hold is another instance's (on any thread), live while this process is, and refused; it is stale only when it carries this instance's own token
// (left behind), or when it was written before this process started (a dead process's, whose pid was reused). Where the process's start is unknown (no `process`, no
// `performance`), a lock naming this pid is stale unless this instance holds it.
EM_JS(double, nxr_js_process_start_s, (), {
  try {
    if (typeof process !== 'undefined' && process.uptime) return Date.now() / 1000 - process.uptime();   // the process's, the same on every thread
    if (typeof performance !== 'undefined' && performance.timeOrigin) return performance.timeOrigin / 1000;
  } catch (e) {}
  return 0;
});
#endif

#ifdef __EMSCRIPTEN__
// this module instance's token (random, made once): the lock files it writes carry it
const std::string& this_instance() {
  static const std::string token = [] {
    std::random_device rd;
    char b[40];
    std::snprintf(b, sizeof b, "%08x%08x%08x%08x", rd(), rd(), rd(), rd());
    return std::string(b);
  }();
  return token;
}
std::string instance_of(const std::optional<Json>& h) { return h && h->contains("instance") && (*h)["instance"].is_string() ? (*h)["instance"].get<std::string>() : std::string(); }
#endif

void process_took(const fs::path& file) { held().insert(file); }
void process_released(const fs::path& file) { held().erase(file); }

long long this_pid() {
#ifdef __EMSCRIPTEN__
  return nxr_js_pid();
#else
  return static_cast<long long>(::getpid());
#endif
}

std::string hostname() {
#ifdef __EMSCRIPTEN__
  char* h = nxr_js_host();
  std::string out(h ? h : "unknown");
  std::free(h);
  return out;
#else
  char b[256];
  if (gethostname(b, sizeof b) != 0) return "unknown";
  b[sizeof b - 1] = 0;
  return b;
#endif
}

fs::path lock_file(const fs::path& dir) { return fs::weakly_canonical(dir) / kWriterLockFile; }

std::string describe(const std::optional<Json>& h) {
  if (!h) return "an unreadable lock";
  auto s = [&](const char* k, const char* d) { return h->contains(k) && (*h)[k].is_string() ? (*h)[k].get<std::string>() : std::string(d); };
  return s("program", "?") + " (pid " + std::to_string((*h)["pid"].get<long long>()) + " on " + s("host", "?") + ", since " + s("started_utc", "?") + ")";
}

[[noreturn]] void held_error(const fs::path& file, const std::optional<Json>& h) {
  throw Error(code::writer_lock_held, file.parent_path().string() + ": another writer holds this dataset — " + describe(h) + " (" + file.string() +
                                          "); wait for it to finish, or stop it");
}

[[noreturn]] void not_writable(const fs::path& folder, int err) {
  const char* c = err == EROFS ? "EROFS" : err == EACCES ? "EACCES" : "EPERM";
  throw Error(code::not_writable, folder.string() + ": not writable (" + c +
                                      ") — nxr-io cannot write this dataset (read-only media, or no write permission); copy it somewhere writable");
}

// the O_EXCL create; a folder that cannot be written is `not_writable`, never a bare OSError
int create_excl(const fs::path& path, const fs::path& folder) {
  const int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
  if (fd >= 0) return fd;
  if (errno == EEXIST) return -EEXIST;
  if (errno == EROFS || errno == EACCES || errno == EPERM) not_writable(folder, errno);
  // Python's own wording: FileNotFoundError / OSError("[Errno N] text: 'path'")
  const std::string m = "[Errno " + std::to_string(errno) + "] " + std::strerror(errno) + ": '" + path.string() + "'";
  throw Error(errno == ENOENT ? code::not_found : code::io, m);
}

bool pid_alive(long long pid) {
  if (pid <= 0) return false;
#ifdef __EMSCRIPTEN__
  return nxr_js_alive(static_cast<double>(pid)) != 0;
#endif
  if (::kill(static_cast<pid_t>(pid), 0) == 0) return true;
  return errno != ESRCH;  // EPERM: alive, not ours
}

std::optional<double> boot_time() {
#ifdef __APPLE__
  struct timeval tv;
  size_t len = sizeof tv;
  int mib[2] = {CTL_KERN, KERN_BOOTTIME};
  if (sysctl(mib, 2, &tv, &len, nullptr, 0) == 0) return static_cast<double>(tv.tv_sec);
  return std::nullopt;
#else
  std::ifstream f("/proc/uptime");
  double up;
  if (f >> up) return static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()) - up;
  return std::nullopt;
#endif
}

double now_s() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

std::optional<double> parse_utc(const std::string& s) {
  std::tm tm{};
  int y, mo, d, h, mi;
  double sec = 0;
  if (std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%lf", &y, &mo, &d, &h, &mi, &sec) < 6) return std::nullopt;
  tm.tm_year = y - 1900; tm.tm_mon = mo - 1; tm.tm_mday = d; tm.tm_hour = h; tm.tm_min = mi; tm.tm_sec = 0;
  return static_cast<double>(timegm(&tm)) + sec;
}

struct Identity {
  bool present = false;
  std::uint64_t ino = 0;
  std::int64_t mtime_ns = 0;
  std::uint64_t size = 0;
  bool operator==(const Identity& o) const { return present == o.present && ino == o.ino && mtime_ns == o.mtime_ns && size == o.size; }
};
Identity identity(const fs::path& f) {
  struct stat st;
  if (::stat(f.c_str(), &st) != 0) return {};
#ifdef __APPLE__
  const std::int64_t ns = static_cast<std::int64_t>(st.st_mtimespec.tv_sec) * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
  const std::int64_t ns = static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
#endif
  return {true, static_cast<std::uint64_t>(st.st_ino), ns, static_cast<std::uint64_t>(st.st_size)};
}

double mtime_s(const fs::path& f) {
  struct stat st;
  if (::stat(f.c_str(), &st) != 0) return now_s();
  return static_cast<double>(st.st_mtime);
}

bool stale(const fs::path& file, const std::optional<Json>& h) {
  if (!h) return identity(file).present && now_s() - mtime_s(file) > kUnreadableStaleS;
  if ((*h)["host"].get<std::string>() != hostname()) return false;
  const long long pid = (*h)["pid"].get<long long>();
  if (pid == this_pid()) {
    {
      std::lock_guard<std::mutex> l(g_mu);
      if (held().count(file)) return false;   // this process (natively), this module instance (in WebAssembly), holds it
    }
#ifdef __EMSCRIPTEN__
    if (instance_of(h) == this_instance()) return true;   // this instance's own, left behind
    const double start = nxr_js_process_start_s();
    if (start <= 0) return true;                          // no process to compare with: as natively
    const auto started = h->contains("started_utc") && (*h)["started_utc"].is_string() ? parse_utc((*h)["started_utc"].get<std::string>()) : std::nullopt;
    return started && *started < start - kProcessSlackS;  // written before this process started (a reused pid); otherwise another instance of this process holds it
#else
    return true;
#endif
  }
  if (!pid_alive(pid)) return true;
  const auto boot = boot_time();
  const auto started = h->contains("started_utc") && (*h)["started_utc"].is_string() ? parse_utc((*h)["started_utc"].get<std::string>()) : std::nullopt;
  return boot && started && *started < *boot - kBootSlackS;
}

// does the lock name this holder: this process (natively), this module instance (in WebAssembly)? Only then is it removed on release.
bool names_this_holder(const std::optional<Json>& h) {
  if (!h || (*h)["pid"].get<long long>() != this_pid()) return false;
#ifdef __EMSCRIPTEN__
  return instance_of(h) == this_instance();
#else
  return true;
#endif
}

std::string holder_json() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char ts[40];
  std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tm);
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000;
  char frac[16] = "";
  if (us) std::snprintf(frac, sizeof frac, ".%06lld", static_cast<long long>(us));
  // json.dumps' own layout (", " and ": "), as nxr's Python writes it; in WebAssembly the module instance's token too (the readers ignore a field they do not use)
  std::string extra;
#ifdef __EMSCRIPTEN__
  extra = ", \"instance\": \"" + this_instance() + "\"";
#endif
  return "{\"pid\": " + std::to_string(this_pid()) + ", \"host\": " + js_json(Json(hostname())) + ", \"program\": \"nxr-io\", \"started_utc\": \"" + ts + frac + "Z\"" + extra + "}";
}

int take_over(const fs::path& file, const std::optional<Json>& stale_holder, const Identity& was) {
  const fs::path brk = file.parent_path() / (file.filename().string() + ".break");
  int b = -1;
  for (int attempt = 0; attempt < 2; ++attempt) {
    b = create_excl(brk, file.parent_path());
    if (b >= 0) break;
    if (!identity(brk).present) continue;
    const double age = now_s() - mtime_s(brk);
    if (attempt == 0 && age > kBreakGraceS) { ::unlink(brk.c_str()); continue; }
    held_error(file, stale_holder);
  }
  if (b < 0) held_error(file, read_writer_lock(file.parent_path()));
  auto done = [&] { ::close(b); ::unlink(brk.c_str()); };
  try {
    const auto now = read_writer_lock(file.parent_path());
    const Identity at = identity(file);
    auto key = [](const std::optional<Json>& h) {
      return h ? std::to_string((*h)["pid"].get<long long>()) + "|" + (h->contains("started_utc") ? (*h)["started_utc"].dump() : "") + "|" + (*h)["host"].get<std::string>() : std::string();
    };
    const bool same = !at.present || (at == was && key(now) == key(stale_holder));
    if (!same) held_error(file, now);  // a fresh holder's (or one being written this instant)
    ::unlink(file.c_str());
    const int fd = create_excl(file, file.parent_path());
    if (fd < 0) held_error(file, read_writer_lock(file.parent_path()));
    done();
    return fd;
  } catch (...) {
    done();
    throw;
  }
}

void release_all() {
  std::lock_guard<std::mutex> l(g_mu);
  for (const auto& f : held()) {
    const auto h = read_writer_lock(f.parent_path());
    if (names_this_holder(h)) ::unlink(f.c_str());
  }
  held().clear();
}

}  // namespace

std::optional<Json> read_writer_lock(const fs::path& dataset_dir) {
  std::ifstream f(lock_file(dataset_dir));
  if (!f) return std::nullopt;
  std::stringstream ss;
  ss << f.rdbuf();
  try {
    Json h = Json::parse(ss.str());
    if (h.is_object() && h.contains("pid") && h["pid"].is_number_integer() && h.contains("host") && h["host"].is_string()) return h;
  } catch (...) {}
  return std::nullopt;
}

std::optional<std::pair<std::string, fs::path>> database_not_writable(const fs::path& file) {
  const fs::path cands[] = {file, fs::path(file.string() + "-wal"), fs::path(file.string() + "-shm"), fs::path(file.string() + "-journal"), file.parent_path()};
  for (const auto& p : cands) {
    if (fs::exists(p) && ::access(p.c_str(), W_OK) != 0) {
      struct statvfs sv;
      const bool ro = ::statvfs(p.c_str(), &sv) == 0 && (sv.f_flag & ST_RDONLY);
      return std::make_pair(std::string(ro ? "EROFS" : "EACCES"), p);
    }
  }
  return std::nullopt;
}

WriterLock::WriterLock(WriterLock&& o) noexcept : s_(std::move(o.s_)) {}
WriterLock& WriterLock::operator=(WriterLock&& o) noexcept {
  if (this != &o) { release(); s_ = std::move(o.s_); }
  return *this;
}
WriterLock::~WriterLock() { release(); }

void WriterLock::release() {
  if (!s_) return;
  if (s_->released || s_->borrowed) { s_->released = true; return; }
  s_->released = true;
  { std::lock_guard<std::mutex> l(g_mu); process_released(s_->file); }
  const auto h = read_writer_lock(s_->file.parent_path());
  if (names_this_holder(h)) ::unlink(s_->file.c_str());
}

WriterLock acquire_writer_lock(const fs::path& dataset_dir, const std::string& program) {
  (void)program;
  static const bool registered = [] { std::atexit(release_all); return true; }();
  (void)registered;
  const fs::path file = lock_file(dataset_dir);
  WriterLock lock;
  // THE HANDSHAKE
  if (const char* env = std::getenv(kWriterLockEnv); env && *env) {
    const std::string v = env;
    const auto colon = v.rfind(':');
    if (colon != std::string::npos) {
      try {
        const long long pid = std::stoll(v.substr(colon + 1));
        std::error_code ec;
        if (fs::weakly_canonical(v.substr(0, colon), ec) == file) {
          const auto h = read_writer_lock(file.parent_path());
          if (h && (*h)["pid"].get<long long>() == pid && (*h)["host"].get<std::string>() == hostname()) {
            lock.s_ = std::make_shared<WriterLock::State>();
            lock.s_->file = file;
            lock.s_->borrowed = true;
            return lock;
          }
        }
      } catch (...) {}
    }
  }
  int fd = create_excl(file, file.parent_path());
  if (fd == -EEXIST) {
    const auto h = read_writer_lock(file.parent_path());
    const Identity was = identity(file);
    if (!stale(file, h)) held_error(file, h);
    fd = take_over(file, h, was);
  }
  // the JSON in the same call as the create; if it cannot be written, the empty file is REMOVED — never left to block (F2)
  const std::string body = holder_json();
  const ssize_t n = ::write(fd, body.data(), body.size());
  ::close(fd);
  if (n != static_cast<ssize_t>(body.size())) {
    ::unlink(file.c_str());
    throw Error(code::io, file.string() + ": cannot write the lock");
  }
  { std::lock_guard<std::mutex> l(g_mu); process_took(file); }
  lock.s_ = std::make_shared<WriterLock::State>();
  lock.s_->file = file;
  return lock;
}

}  // namespace nxr::io
