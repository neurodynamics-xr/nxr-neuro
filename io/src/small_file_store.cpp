#include "nxr/io/small_file_store.h"

#include <sqlite3.h>

#include <algorithm>
#include <cstring>
#include <set>

#include "nxr/io/database.h"
#include "nxr/io/errors.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {

// ── a prepared statement ──
class Stmt {
 public:
  Stmt(sqlite3* h, const char* sql) : h_(h) {
    if (sqlite3_prepare_v2(h, sql, -1, &s_, nullptr) != SQLITE_OK) throw Error(code::sqlite, std::string("small_files: ") + sqlite3_errmsg(h));
  }
  ~Stmt() { if (s_) sqlite3_finalize(s_); }
  Stmt(const Stmt&) = delete;
  Stmt& operator=(const Stmt&) = delete;
  Stmt& text(int i, const std::string& v) { sqlite3_bind_text(s_, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT); return *this; }
  Stmt& blob(int i, const std::uint8_t* d, std::size_t n) {
    static const std::uint8_t none = 0;
    sqlite3_bind_blob64(s_, i, n ? static_cast<const void*>(d) : &none, n, SQLITE_TRANSIENT);
    return *this;
  }
  Stmt& integer(int i, std::int64_t v) { sqlite3_bind_int64(s_, i, v); return *this; }
  bool row() {
    const int rc = sqlite3_step(s_);
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    throw Error(code::sqlite, std::string("small_files: ") + sqlite3_errmsg(h_));
  }
  void run() { while (row()) {} }
  std::string str(int c) const { return std::string(reinterpret_cast<const char*>(sqlite3_column_text(s_, c)), static_cast<std::size_t>(sqlite3_column_bytes(s_, c))); }
  Bytes bytes(int c) const {
    const auto n = static_cast<std::size_t>(sqlite3_column_bytes(s_, c));
    const auto* p = static_cast<const std::uint8_t*>(sqlite3_column_blob(s_, c));
    return n ? Bytes(p, p + n) : Bytes();
  }
  std::int64_t i64(int c) const { return sqlite3_column_int64(s_, c); }

 private:
  sqlite3* h_;
  sqlite3_stmt* s_ = nullptr;
};

void exec(sqlite3* h, const char* sql) {
  char* err = nullptr;
  if (sqlite3_exec(h, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    const std::string m = err ? err : "error";
    sqlite3_free(err);
    throw Error(code::sqlite, "small_files: " + m);
  }
}

// the keys under `prefix` as a range: [prefix + "/", prefix + "0") ('0' is the character after '/'); the whole table for the empty prefix
struct Range { std::string lo, hi; bool all; };
Range range_of(const std::string& prefix) { return prefix.empty() ? Range{"", "", true} : Range{prefix + "/", prefix + "0", false}; }

bool reserved(const std::string& name) {
  const std::string base = kSmallFilesFile;
  return name == base || name == base + "-wal" || name == base + "-shm" || name == base + "-journal";
}

std::string uri_of(const fs::path& p) {   // a file: URI with the characters SQLite's URI parser reads specially escaped
  std::string out = "file:";
  for (char c : p.generic_string()) {
    if (c == '%') out += "%25";
    else if (c == '?') out += "%3f";
    else if (c == '#') out += "%23";
    else out += c;
  }
  return out + "?immutable=1";
}

}  // namespace

// ── the connection ──
struct SmallFileStore::Db {
  sqlite3* h = nullptr;
  bool writable = false;
  bool has_table = false;
  int depth = 0;
  // At rest the database is ONE file: the journal goes back to the default (DELETE), which checkpoints the WAL and removes its files (some SQLite builds keep them
  // after the last close); a connection another process still holds makes the switch fail, and the WAL stays until that one closes. It is a WAL again on the next open.
  ~Db() {
    if (!h) return;
#ifndef __EMSCRIPTEN__
    bool at_rest = false;
    if (writable && depth == 0) at_rest = sqlite3_exec(h, "PRAGMA journal_mode = DELETE", nullptr, nullptr, nullptr) == SQLITE_OK;
    sqlite3_close(h);
    if (at_rest) {   // no connection holds the WAL now (the switch needs that); the shared-memory file some builds leave goes too
      std::error_code ec;
      fs::remove(path + "-wal", ec);
      fs::remove(path + "-shm", ec);
    }
    return;
#endif
    sqlite3_close(h);
  }
  std::string path;
};

SmallFileStore::SmallFileStore(Store& files, fs::path db_file, SmallFileOptions opt) : files_(&files), db_file_(std::move(db_file)), opt_(opt) {}
SmallFileStore::~SmallFileStore() = default;

void SmallFileStore::close() {
  std::lock_guard<std::recursive_mutex> g(mu_);
  db_.reset();
}

SmallFileStore::Db* SmallFileStore::db(bool create) const {
  std::lock_guard<std::recursive_mutex> g(mu_);
  if (db_) {
    if (!create || db_->writable) return db_.get();
    db_.reset();   // opened read-only before; a write wants it writable
  }
  std::error_code ec;
  const bool there = fs::is_regular_file(db_file_, ec);
  const bool want_write = create && !opt_.read_only;
  if (!there && !want_write) return nullptr;
  auto d = std::make_unique<Db>();
  d->path = db_file_.string();
  if (!there) fs::create_directories(db_file_.parent_path());
  const std::string p = db_file_.string();
  bool opened = false;
  if (!opt_.read_only) {
    opened = sqlite3_open_v2(p.c_str(), &d->h, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) == SQLITE_OK;
    d->writable = opened;
    if (!opened) { if (d->h) sqlite3_close(d->h); d->h = nullptr; }
  }
  if (!opened) {   // read-only: the file as a reader sees it (and a database in WAL mode on read-only media, as immutable)
    if (want_write) throw Error(code::not_writable, p + ": the small-file database cannot be written");
    if (sqlite3_open_v2(p.c_str(), &d->h, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) { if (d->h) sqlite3_close(d->h); d->h = nullptr; }
    bool ok = d->h != nullptr;
    if (ok) {
      sqlite3_busy_timeout(d->h, 5000);
      try { Stmt s(d->h, "SELECT count(*) FROM sqlite_master"); s.run(); } catch (const Error&) { ok = false; }
    }
    if (!ok) {
      if (d->h) { sqlite3_close(d->h); d->h = nullptr; }
      const std::string u = uri_of(db_file_);
      if (sqlite3_open_v2(u.c_str(), &d->h, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr) != SQLITE_OK) {
        const std::string m = d->h ? sqlite3_errmsg(d->h) : "cannot open";
        throw Error(code::sqlite, p + ": " + m);
      }
    }
  }
  sqlite3_busy_timeout(d->h, 5000);
  if (d->writable) {
#ifndef __EMSCRIPTEN__
    // WAL (readers while one writer); WebAssembly has one thread and no shared memory for it: the default journal there
    exec(d->h, "PRAGMA journal_mode = WAL");
    exec(d->h, "PRAGMA synchronous = NORMAL");
#else
    // WebAssembly has no shared memory, so no WAL: a database another build left in WAL mode (a Node or native process, or a copy of one) is put back to the default
    // journal, as at rest. Writes in a WAL this build cannot share would sit in a wal-index only this module sees: a commit that other connections (the Studio's Node
    // side) never read. When the switch is refused (a connection elsewhere holds the WAL) the write is refused, never made invisible.
    {
      std::string mode;
      { Stmt s(d->h, "PRAGMA journal_mode"); if (s.row()) mode = s.str(0); }   // finalized before the switch (an open statement is a transaction)
      if (mode == "wal") {
        std::string now;
        { Stmt t(d->h, "PRAGMA journal_mode = DELETE"); if (t.row()) now = t.str(0); }
        if (now != "delete") throw Error(code::not_writable, p + ": the small-file database is in WAL mode, held by another connection; it cannot be written from WebAssembly until that one closes");
      }
    }
#endif
    exec(d->h, "CREATE TABLE IF NOT EXISTS small_files (key TEXT PRIMARY KEY, value BLOB NOT NULL)");
    d->has_table = true;
  } else {
    Stmt s(d->h, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'small_files'");
    d->has_table = s.row();
  }
  db_ = std::move(d);
  return db_.get();
}

bool SmallFileStore::in_table(const std::string& key) const {
  std::lock_guard<std::recursive_mutex> g(mu_);
  Db* d = db(false);
  if (!d || !d->has_table) return false;
  check_store_key(key);
  Stmt s(d->h, "SELECT 1 FROM small_files WHERE key = ?1");
  return s.text(1, key).row();
}

std::vector<std::string> SmallFileStore::table_keys() const {
  std::lock_guard<std::recursive_mutex> g(mu_);
  std::vector<std::string> out;
  Db* d = db(false);
  if (!d || !d->has_table) return out;
  Stmt s(d->h, "SELECT key FROM small_files ORDER BY key");
  while (s.row()) out.push_back(s.str(0));
  return out;
}

void SmallFileStore::put_row(const std::string& key, const std::uint8_t* data, std::size_t n) {
  std::lock_guard<std::recursive_mutex> g(mu_);
  check_store_key(key);
  Db* d = db(true);
  Stmt s(d->h, "INSERT INTO small_files (key, value) VALUES (?1, ?2) ON CONFLICT (key) DO UPDATE SET value = excluded.value");
  s.text(1, key).blob(2, data, n).run();
}

void SmallFileStore::batch(const std::function<void()>& body) {
  std::lock_guard<std::recursive_mutex> g(mu_);
  Db* d = db(true);
  if (d->depth > 0) { body(); return; }
  exec(d->h, "BEGIN IMMEDIATE");
  d->depth = 1;
  try {
    body();
  } catch (...) {
    d->depth = 0;
    sqlite3_exec(d->h, "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
  d->depth = 0;
  exec(d->h, "COMMIT");
}

std::optional<Bytes> SmallFileStore::get(const std::string& key) const {
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    check_store_key(key);
    if (Db* d = db(false); d && d->has_table) {
      Stmt s(d->h, "SELECT value FROM small_files WHERE key = ?1");
      if (s.text(1, key).row()) return s.bytes(0);
    }
  }
  return files_->get(key);
}

std::optional<Bytes> SmallFileStore::get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const {
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    check_store_key(key);
    if (Db* d = db(false); d && d->has_table) {
      Stmt s(d->h, "SELECT length(value), substr(value, ?2, ?3) FROM small_files WHERE key = ?1");
      s.text(1, key).integer(2, static_cast<std::int64_t>(offset) + 1).integer(3, static_cast<std::int64_t>(length));
      if (s.row()) {
        if (offset + length > static_cast<std::uint64_t>(s.i64(0))) throw ZarrFormatError("[nxr/io] a chunk reaches past the end of its value");
        return length ? s.bytes(1) : Bytes();
      }
    }
  }
  return files_->get_range(key, offset, length);
}

std::optional<std::uint64_t> SmallFileStore::size(const std::string& key) const {
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    check_store_key(key);
    if (Db* d = db(false); d && d->has_table) {
      Stmt s(d->h, "SELECT length(value) FROM small_files WHERE key = ?1");
      if (s.text(1, key).row()) return static_cast<std::uint64_t>(s.i64(0));
    }
  }
  return files_->size(key);
}

bool SmallFileStore::exists(const std::string& key) const { return in_table(key) || files_->exists(key); }

bool SmallFileStore::exists_prefix(const std::string& prefix) const {
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (Db* d = db(false); d && d->has_table && !prefix.empty()) {
      const Range r = range_of(prefix);
      Stmt s(d->h, "SELECT 1 FROM small_files WHERE key = ?1 OR (key >= ?2 AND key < ?3) LIMIT 1");
      if (s.text(1, prefix).text(2, r.lo).text(3, r.hi).row()) return true;
    }
  }
  return files_->exists_prefix(prefix);
}

Listing SmallFileStore::list(const std::string& prefix) const {
  Listing out = files_->list(prefix);
  if (prefix.empty()) {
    auto bad = [](const std::string& n) { return reserved(n); };
    out.keys.erase(std::remove_if(out.keys.begin(), out.keys.end(), bad), out.keys.end());
  }
  std::lock_guard<std::recursive_mutex> g(mu_);
  Db* d = db(false);
  if (!d || !d->has_table) return out;
  std::set<std::string> keys(out.keys.begin(), out.keys.end()), prefixes(out.prefixes.begin(), out.prefixes.end());
  const Range r = range_of(prefix);
  Stmt s(d->h, r.all ? "SELECT key FROM small_files" : "SELECT key FROM small_files WHERE key >= ?1 AND key < ?2");
  if (!r.all) s.text(1, r.lo).text(2, r.hi);
  const std::size_t skip = r.all ? 0 : r.lo.size();
  while (s.row()) {
    const std::string rel = s.str(0).substr(skip);
    const auto slash = rel.find('/');
    if (slash == std::string::npos) keys.insert(rel);
    else prefixes.insert(rel.substr(0, slash));
  }
  out.keys.assign(keys.begin(), keys.end());
  out.prefixes.assign(prefixes.begin(), prefixes.end());
  return out;
}

bool SmallFileStore::is_link(const std::string& prefix) const { return files_->is_link(prefix); }
void SmallFileStore::check_inside(const std::string& key) const { files_->check_inside(key); }

void SmallFileStore::set(const std::string& key, const std::uint8_t* data, std::size_t n) {
  check_store_key(key);
  if (opt_.read_only) throw Error(code::not_writable, "the small-file store is read-only");
  files_->check_inside(key);   // a key under a link that leads out of the store is refused before a row is written or a file replaced
  if (static_cast<std::uint64_t>(n) < opt_.threshold) {
    put_row(key, data, n);
    if (files_->exists(key)) files_->erase(key);   // a key lives in one place
  } else {
    files_->set(key, data, n);
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (Db* d = db(false); d && d->has_table && d->writable) {
      Stmt s(d->h, "DELETE FROM small_files WHERE key = ?1");
      s.text(1, key).run();
    }
  }
}

// A streamed value: held until the threshold, then a file.
class SmallWriter : public StoreWriter {
 public:
  SmallWriter(SmallFileStore& st, std::string key) : st_(&st), key_(std::move(key)) {}
  void append(const std::uint8_t* data, std::size_t n) override {
    if (!n) return;
    if (!file_) {
      if (buf_.size() + n < st_->opt_.threshold) { buf_.insert(buf_.end(), data, data + n); return; }
      file_ = st_->files_->open_writer(key_);
      if (!buf_.empty()) file_->append(buf_.data(), buf_.size());
      Bytes().swap(buf_);
    }
    file_->append(data, n);
  }
  void close() override {
    if (file_) {
      file_->close();
      std::lock_guard<std::recursive_mutex> g(st_->mu_);
      if (auto* d = st_->db(false); d && d->has_table && d->writable) {
        Stmt s(d->h, "DELETE FROM small_files WHERE key = ?1");
        s.text(1, key_).run();
      }
      return;
    }
    st_->set(key_, buf_.data(), buf_.size());
  }

 private:
  SmallFileStore* st_;
  std::string key_;
  Bytes buf_;
  std::unique_ptr<StoreWriter> file_;
};

std::unique_ptr<StoreWriter> SmallFileStore::open_writer(const std::string& key) {
  check_store_key(key);
  if (opt_.read_only) throw Error(code::not_writable, "the small-file store is read-only");
  files_->check_inside(key);
  return std::make_unique<SmallWriter>(*this, key);
}

void SmallFileStore::erase(const std::string& key) {
  check_store_key(key);
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (Db* d = db(false); d && d->has_table) {
      if (!d->writable) throw Error(code::not_writable, "the small-file store is read-only");
      Stmt s(d->h, "DELETE FROM small_files WHERE key = ?1");
      s.text(1, key).run();
    }
  }
  files_->erase(key);
}

void SmallFileStore::erase_prefix(const std::string& prefix) {
  if (prefix.empty()) throw Error(code::invalid_argument, "the store's root is not erased");
  check_store_key(prefix);
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (Db* d = db(false); d && d->has_table) {
      if (!d->writable) throw Error(code::not_writable, "the small-file store is read-only");
      const Range r = range_of(prefix);
      Stmt s(d->h, "DELETE FROM small_files WHERE key = ?1 OR (key >= ?2 AND key < ?3)");
      s.text(1, prefix).text(2, r.lo).text(3, r.hi).run();
    }
  }
  if (files_->exists_prefix(prefix)) files_->erase_prefix(prefix);
}

void SmallFileStore::rename_prefix(const std::string& from, const std::string& to) {
  check_store_key(from);
  check_store_key(to);
  if (exists_prefix(to)) throw Error(code::already_exists, to + ": exists");
  files_->check_inside(to);
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (Db* d = db(false); d && d->has_table) {
      if (!d->writable) throw Error(code::not_writable, "the small-file store is read-only");
      const Range r = range_of(from);
      Stmt s(d->h, "UPDATE small_files SET key = ?4 || substr(key, ?5) WHERE key = ?1 OR (key >= ?2 AND key < ?3)");
      s.text(1, from).text(2, r.lo).text(3, r.hi).text(4, to).integer(5, static_cast<std::int64_t>(from.size()) + 1).run();
    }
  }
  if (files_->exists_prefix(from)) files_->rename_prefix(from, to);
}

// ══ DatastoreStore ═════════════════════════════════════════════════════════════════════════════════════

struct DatastoreStore::Sub {
  FolderStore files;
  std::unique_ptr<SmallFileStore> small;
  Sub(const fs::path& folder) : files(folder) {}
  Store& store() { return small ? static_cast<Store&>(*small) : static_cast<Store&>(files); }
};

namespace {
// `<dataset>/<subject>` and the rest of a key with three or more segments
bool split3(const std::string& key, std::string& subject, std::string& rest) {
  const auto a = key.find('/');
  if (a == std::string::npos) return false;
  const auto b = key.find('/', a + 1);
  if (b == std::string::npos || b + 1 >= key.size()) return false;
  subject = key.substr(0, b);
  rest = key.substr(b + 1);
  return true;
}
// `<dataset>/<subject>` exactly (the subject's folder as a prefix)
bool is_subject_prefix(const std::string& key) {
  const auto a = key.find('/');
  return a != std::string::npos && a + 1 < key.size() && key.find('/', a + 1) == std::string::npos;
}
}  // namespace

DatastoreStore::DatastoreStore(fs::path root, SmallFileOptions opt, std::function<bool(const std::string&)> format52)
    : root_(std::move(root)), opt_(opt), format52_(std::move(format52)), base_(root_) {}
DatastoreStore::~DatastoreStore() = default;

void DatastoreStore::set_small_files(const std::string& dataset, bool on) {
  std::lock_guard<std::recursive_mutex> g(mu_);
  on_[dataset] = on;
}

bool DatastoreStore::wants_table(const std::string& dataset) const {
  if (const auto it = on_.find(dataset); it != on_.end()) return it->second;
  if (format52_) return format52_(dataset);
  const fs::path file = root_ / dataset / kDatasetFile;
  std::error_code ec;
  if (!fs::is_regular_file(file, ec)) return false;   // not asked again until its database is there
  bool on = false;
  try {
    Database db = Database::open(file, false, "", true);
    on = store_format_number(db) >= 52;
  } catch (const Error&) {
    return false;
  }
  on_[dataset] = on;
  return on;
}

DatastoreStore::Sub& DatastoreStore::sub_of(const std::string& subject_key) const {
  auto it = subs_.find(subject_key);
  if (it != subs_.end()) return *it->second;
  auto sub = std::make_unique<Sub>(root_ / subject_key);
  const std::string dataset = subject_key.substr(0, subject_key.find('/'));
  std::error_code ec;
  if (wants_table(dataset) || fs::is_regular_file(root_ / subject_key / kSmallFilesFile, ec)) {
    SmallFileOptions o = opt_;
    sub->small = std::make_unique<SmallFileStore>(sub->files, root_ / subject_key / kSmallFilesFile, o);
  }
  return *subs_.emplace(subject_key, std::move(sub)).first->second;
}

SmallFileStore* DatastoreStore::small_files(const std::string& subject_key, bool create) {
  std::lock_guard<std::recursive_mutex> g(mu_);
  Sub& s = sub_of(subject_key);
  if (!s.small && create) s.small = std::make_unique<SmallFileStore>(s.files, root_ / subject_key / kSmallFilesFile, opt_);
  return s.small.get();
}

FolderStore& DatastoreStore::files(const std::string& subject_key) {
  std::lock_guard<std::recursive_mutex> g(mu_);
  return sub_of(subject_key).files;
}

void DatastoreStore::close_all() {
  std::lock_guard<std::recursive_mutex> g(mu_);
  subs_.clear();
}

DatastoreStore::Route DatastoreStore::route(const std::string& key, bool create) const {
  Route r;
  (void)create;
  check_store_key(key);
  if (!split3(key, r.subject, r.rest)) return r;
  r.sub = &sub_of(r.subject);
  r.routed = true;
  return r;
}

// The lock covers only the lookup of the subject's store: a read of a large shard is not held up by another thread's (the stores lock themselves)
#define NXR_ROUTE(key, call_small, call_base)                                          \
  Route r;                                                                             \
  { std::lock_guard<std::recursive_mutex> g(mu_); r = route(key, false); }             \
  if (r.routed) { Store& st = r.sub->store(); return call_small; }                    \
  return call_base;

std::optional<Bytes> DatastoreStore::get(const std::string& key) const { NXR_ROUTE(key, st.get(r.rest), base_.get(key)) }
std::optional<Bytes> DatastoreStore::get_range(const std::string& key, std::uint64_t o, std::uint64_t n) const { NXR_ROUTE(key, st.get_range(r.rest, o, n), base_.get_range(key, o, n)) }
std::optional<std::uint64_t> DatastoreStore::size(const std::string& key) const { NXR_ROUTE(key, st.size(r.rest), base_.size(key)) }
bool DatastoreStore::exists(const std::string& key) const { NXR_ROUTE(key, st.exists(r.rest), base_.exists(key)) }
bool DatastoreStore::exists_prefix(const std::string& prefix) const {
  Route r;
  { std::lock_guard<std::recursive_mutex> g(mu_); r = route(prefix, false); }
  if (r.routed) return r.sub->store().exists_prefix(r.rest);
  return base_.exists_prefix(prefix);
}
Listing DatastoreStore::list(const std::string& prefix) const {
  check_store_key(prefix);
  Route r;
  Sub* whole = nullptr;
  {
    std::lock_guard<std::recursive_mutex> g(mu_);
    if (is_subject_prefix(prefix)) whole = &sub_of(prefix);
    else r = route(prefix, false);
  }
  if (whole) return whole->store().list("");
  if (r.routed) return r.sub->store().list(r.rest);
  return base_.list(prefix);
}
bool DatastoreStore::is_link(const std::string& prefix) const {
  Route r;
  { std::lock_guard<std::recursive_mutex> g(mu_); r = route(prefix, false); }
  if (r.routed) return r.sub->store().is_link(r.rest);
  return base_.is_link(prefix);
}
void DatastoreStore::check_inside(const std::string& key) const { NXR_ROUTE(key, (st.check_inside(r.rest), void()), (base_.check_inside(key), void())) }
void DatastoreStore::set(const std::string& key, const std::uint8_t* data, std::size_t n) { NXR_ROUTE(key, (st.set(r.rest, data, n), void()), (base_.set(key, data, n), void())) }
std::unique_ptr<StoreWriter> DatastoreStore::open_writer(const std::string& key) { NXR_ROUTE(key, st.open_writer(r.rest), base_.open_writer(key)) }
void DatastoreStore::erase(const std::string& key) { NXR_ROUTE(key, (st.erase(r.rest), void()), (base_.erase(key), void())) }

void DatastoreStore::erase_prefix(const std::string& prefix) {
  std::lock_guard<std::recursive_mutex> g(mu_);
  check_store_key(prefix);
  if (is_subject_prefix(prefix)) {   // the whole subject: its database goes with its folder
    subs_.erase(prefix);
    base_.erase_prefix(prefix);
    return;
  }
  const Route r = route(prefix, false);
  if (r.routed) r.sub->store().erase_prefix(r.rest);
  else base_.erase_prefix(prefix);
}

void DatastoreStore::rename_prefix(const std::string& from, const std::string& to) {
  std::lock_guard<std::recursive_mutex> g(mu_);
  check_store_key(from);
  check_store_key(to);
  if (is_subject_prefix(from) || is_subject_prefix(to)) {   // a whole subject's folder moves with its database
    subs_.erase(from);
    subs_.erase(to);
    base_.rename_prefix(from, to);
    return;
  }
  const Route a = route(from, false), b = route(to, false);
  if (!a.routed && !b.routed) { base_.rename_prefix(from, to); return; }
  if (a.routed && b.routed && a.subject == b.subject) { a.sub->store().rename_prefix(a.rest, b.rest); return; }
  if (!a.routed || !b.routed) throw Error(code::invalid_argument, "a subject's key cannot be renamed into or out of the datastore's own keys");
  // another subject's folder (a node moved aside): the destination's own store takes each value
  if (b.sub->store().exists_prefix(b.rest)) throw Error(code::already_exists, to + ": exists");
  SmallFileStore* src_small = a.sub->small.get();
  // the values that are files move as files; the rows are written into the destination
  std::vector<std::string> row_keys;
  if (src_small) for (const auto& k : src_small->table_keys()) if (k == a.rest || k.rfind(a.rest + "/", 0) == 0) row_keys.push_back(k);
  std::function<void(const std::string&)> move_files = [&](const std::string& at) {
    const Listing l = a.sub->files.list(at);
    for (const auto& k : l.keys) {
      const std::string from_key = join_key(at, k);
      const std::string to_key = b.rest + from_key.substr(a.rest.size());
      if (b.sub->small) {   // the destination keeps small values as rows: only the large files move
        if (const auto n = a.sub->files.size(from_key); n && *n < b.sub->small->threshold()) { b.sub->small->set(to_key, *a.sub->files.get(from_key)); a.sub->files.erase(from_key); continue; }
      }
      b.sub->files.check_inside(to_key);
      const fs::path src = a.sub->files.path_of(from_key), dst = b.sub->files.path_of(to_key);
      fs::create_directories(dst.parent_path());
      fs::rename(src, dst);
    }
    for (const auto& p : l.prefixes) move_files(join_key(at, p));
  };
  if (a.sub->files.exists_prefix(a.rest)) {
    if (a.sub->files.exists(a.rest)) {
      b.sub->store().set(b.rest, *a.sub->files.get(a.rest));
      a.sub->files.erase(a.rest);
    } else {
      move_files(a.rest);
    }
  }
  for (const auto& k : row_keys) b.sub->store().set(b.rest + k.substr(a.rest.size()), *src_small->get(k));
  a.sub->store().erase_prefix(a.rest);
}

}  // namespace nxr::io
