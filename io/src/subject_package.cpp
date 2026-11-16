#include "nxr/io/subject_package.h"

#include <algorithm>
#include <map>
#include <set>

#include "nxr/io/bundle.h"
#include "nxr/io/hash.h"
#include "nxr/io/small_file_store.h"
#include "nxr/io/statistic.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {

const char* const kPackageTables =
    "CREATE TABLE IF NOT EXISTS key_digest (key TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL) STRICT;\n"
    "CREATE TABLE IF NOT EXISTS anchor_row (table_name TEXT NOT NULL, row_id TEXT NOT NULL, from_subject_id TEXT NOT NULL, PRIMARY KEY (table_name, row_id)) STRICT;\n";

constexpr const char* kRecordKey = "package_json";
constexpr std::uint64_t kReadSlab = 8ULL << 20;   // a key is hashed in slabs of this many bytes, never held whole

Error value_error(const std::string& m) { return Error(code::invalid_argument, m); }

// The tables whose rows are not the package's content: its own bookkeeping, and the outbox.
bool is_content_table(const std::string& t) {
  return t != "meta" && t != "sync_job" && t != "small_files" && t != "key_digest" && t.rfind("sqlite_", 0) != 0;
}

std::vector<std::string> content_tables(const Database& db) {
  std::vector<std::string> out;
  for (const auto& r : db.all("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"))
    if (is_content_table(text(r, "name"))) out.push_back(text(r, "name"));
  return out;
}

Json value_json(const Value& v) {
  if (std::holds_alternative<std::monostate>(v)) return nullptr;
  if (auto* n = std::get_if<std::int64_t>(&v)) return *n;
  if (auto* x = std::get_if<double>(&v)) return *x;
  return std::get<std::string>(v);
}

// ── the keys a package holds, with their digests ─────────────────────────────────────────────────────────────

void walk(const Store& store, const std::string& prefix, std::vector<std::string>& out) {
  const Listing l = store.list(prefix);
  for (const auto& k : l.keys) out.push_back(join_key(prefix, k));
  for (const auto& p : l.prefixes) walk(store, join_key(prefix, p), out);
}

// A key's size and SHA-256 from its bytes, read in slabs.
std::pair<std::uint64_t, std::string> digest_of(const Store& store, const std::string& key) {
  const auto size = store.size(key);
  if (!size) throw Error(code::not_found, key + ": no such key");
  Sha256 h;
  for (std::uint64_t at = 0; at < *size; at += kReadSlab) {
    const auto part = store.get_range(key, at, std::min<std::uint64_t>(kReadSlab, *size - at));
    if (!part) throw Error(code::not_found, key + ": vanished while it was hashed");
    h.update(part->data(), part->size());
  }
  return {*size, h.hex()};
}

void put_digest(Database& db, const std::string& key, std::uint64_t size, const std::string& sha) {
  db.execute("INSERT INTO key_digest (key, size_bytes, sha256) VALUES (?, ?, ?) ON CONFLICT (key) DO UPDATE SET size_bytes = excluded.size_bytes, sha256 = excluded.sha256",
             {key, V(static_cast<std::int64_t>(size)), sha});
}

class DigestWriter : public StoreWriter {
 public:
  DigestWriter(Database& db, std::unique_ptr<StoreWriter> inner, std::string key) : db_(&db), inner_(std::move(inner)), key_(std::move(key)) {}
  void append(const std::uint8_t* data, std::size_t n) override { hash_.update(data, n); inner_->append(data, n); }
  void close() override {
    inner_->close();
    const std::uint64_t size = hash_.size();
    put_digest(*db_, key_, size, hash_.hex());
  }

 private:
  Database* db_;
  std::unique_ptr<StoreWriter> inner_;
  std::string key_;
  Sha256 hash_;
};

// A store that records the size and SHA-256 of every value written (key_digest), in the connection of its database.
class DigestStore : public Store {
 public:
  DigestStore(Store& inner, Database& db) : inner_(&inner), db_(&db) {}
  std::optional<Bytes> get(const std::string& k) const override { return inner_->get(k); }
  std::optional<Bytes> get_range(const std::string& k, std::uint64_t o, std::uint64_t n) const override { return inner_->get_range(k, o, n); }
  std::optional<std::uint64_t> size(const std::string& k) const override { return inner_->size(k); }
  bool exists(const std::string& k) const override { return inner_->exists(k); }
  bool exists_prefix(const std::string& p) const override { return inner_->exists_prefix(p); }
  Listing list(const std::string& p) const override { return inner_->list(p); }
  bool is_link(const std::string& p) const override { return inner_->is_link(p); }
  void check_inside(const std::string& k) const override { inner_->check_inside(k); }
  using Store::set;
  void set(const std::string& key, const std::uint8_t* data, std::size_t n) override {
    inner_->set(key, data, n);
    Sha256 h;
    h.update(data, n);
    put_digest(*db_, key, n, h.hex());
  }
  std::unique_ptr<StoreWriter> open_writer(const std::string& key) override { return std::make_unique<DigestWriter>(*db_, inner_->open_writer(key), key); }
  void erase(const std::string& key) override {
    inner_->erase(key);
    db_->execute("DELETE FROM key_digest WHERE key = ?", {key});
  }
  void erase_prefix(const std::string& prefix) override {
    inner_->erase_prefix(prefix);
    db_->execute("DELETE FROM key_digest WHERE key = ? OR (key >= ? AND key < ?)", {prefix, prefix + "/", prefix + "0"});
  }
  void rename_prefix(const std::string& from, const std::string& to) override {
    inner_->rename_prefix(from, to);
    db_->execute("UPDATE key_digest SET key = ? || substr(key, ?) WHERE key = ? OR (key >= ? AND key < ?)",
                 {to, V(static_cast<std::int64_t>(from.size()) + 1), from, from + "/", from + "0"});
  }

 private:
  Store* inner_;
  Database* db_;
};

class DiscardWriter : public StoreWriter {
 public:
  void append(const std::uint8_t*, std::size_t) override {}
  void close() override {}
};

// The writer's view of a package: keys `<prefix>/…` (the home dataset's path and the subject's, as in a dataset's store) are the package's own keys. A package has no
// dataset node, so the group document of a folder ABOVE the prefix is accepted and dropped; any other key outside the prefix is refused.
class PackageKeyStore : public Store {
 public:
  PackageKeyStore(std::unique_ptr<FolderStore> files, std::unique_ptr<SmallFileStore> small, std::unique_ptr<DigestStore> digest, std::string prefix)
      : files_(std::move(files)), small_(std::move(small)), digest_(std::move(digest)), prefix_(std::move(prefix)) {}
  Store& keys() { return *digest_; }

  std::optional<Bytes> get(const std::string& k) const override { std::string r; return inside(k, r) && !r.empty() ? digest_->get(r) : std::nullopt; }
  std::optional<Bytes> get_range(const std::string& k, std::uint64_t o, std::uint64_t n) const override {
    std::string r;
    return inside(k, r) && !r.empty() ? digest_->get_range(r, o, n) : std::nullopt;
  }
  std::optional<std::uint64_t> size(const std::string& k) const override { std::string r; return inside(k, r) && !r.empty() ? digest_->size(r) : std::nullopt; }
  bool exists(const std::string& k) const override { std::string r; return inside(k, r) && !r.empty() && digest_->exists(r); }
  bool exists_prefix(const std::string& p) const override {
    std::string r;
    if (inside(p, r)) return r.empty() ? true : digest_->exists_prefix(r);
    return above(p);
  }
  Listing list(const std::string& p) const override {
    std::string r;
    if (inside(p, r)) return digest_->list(r);
    Listing out;
    if (above(p)) out.prefixes.push_back(next_segment(p));
    return out;
  }
  bool is_link(const std::string& p) const override { std::string r; return inside(p, r) && !r.empty() && digest_->is_link(r); }
  void check_inside(const std::string& k) const override { std::string r; if (inside(k, r) && !r.empty()) digest_->check_inside(r); }
  using Store::set;
  void set(const std::string& key, const std::uint8_t* data, std::size_t n) override {
    std::string r;
    if (inside(key, r) && !r.empty()) return digest_->set(r, data, n);
    if (!dropped(key)) throw value_error(key + ": outside the package (its keys are under " + prefix_ + "/)");
  }
  std::unique_ptr<StoreWriter> open_writer(const std::string& key) override {
    std::string r;
    if (inside(key, r) && !r.empty()) return digest_->open_writer(r);
    if (dropped(key)) return std::make_unique<DiscardWriter>();
    throw value_error(key + ": outside the package (its keys are under " + prefix_ + "/)");
  }
  void erase(const std::string& key) override {
    std::string r;
    if (inside(key, r) && !r.empty()) digest_->erase(r);
  }
  void erase_prefix(const std::string& prefix) override {
    std::string r;
    if (!inside(prefix, r)) return;
    if (!r.empty()) return digest_->erase_prefix(r);
    const Listing all = digest_->list("");   // the subject's whole store: every key, never the database or the lock
    for (const auto& k : all.keys) digest_->erase(k);
    for (const auto& p : all.prefixes) digest_->erase_prefix(p);
  }
  void rename_prefix(const std::string& from, const std::string& to) override {
    std::string a, b;
    if (!inside(from, a) || !inside(to, b) || a.empty() || b.empty()) throw value_error("a move reaches outside the package");
    digest_->rename_prefix(a, b);
  }

 private:
  // `key` under the prefix as the package's own key (empty for the prefix itself).
  bool inside(const std::string& key, std::string& rest) const {
    if (key == prefix_) { rest.clear(); return true; }
    if (key.size() > prefix_.size() + 1 && key.compare(0, prefix_.size(), prefix_) == 0 && key[prefix_.size()] == '/') { rest = key.substr(prefix_.size() + 1); return true; }
    return false;
  }
  // a folder strictly above the prefix
  bool above(const std::string& p) const { return p.empty() || (prefix_.size() > p.size() && prefix_.compare(0, p.size(), p) == 0 && prefix_[p.size()] == '/'); }
  std::string next_segment(const std::string& p) const {
    const std::string rest = p.empty() ? prefix_ : prefix_.substr(p.size() + 1);
    return rest.substr(0, rest.find('/'));
  }
  bool dropped(const std::string& key) const {
    const std::string tail = "zarr.json";
    if (key == tail) return false;
    if (key.size() <= tail.size() + 1 || key.compare(key.size() - tail.size(), tail.size(), tail) != 0 || key[key.size() - tail.size() - 1] != '/') return false;
    return above(key.substr(0, key.size() - tail.size() - 1));
  }
  std::unique_ptr<FolderStore> files_;
  std::unique_ptr<SmallFileStore> small_;
  std::unique_ptr<DigestStore> digest_;
  std::string prefix_;
};

std::unique_ptr<Store> make_package_store(const fs::path& folder, Database& db, const std::string& prefix) {
  auto files = std::make_unique<FolderStore>(folder);
  auto small = std::make_unique<SmallFileStore>(*files, db);
  auto digest = std::make_unique<DigestStore>(*small, db);
  return std::make_unique<PackageKeyStore>(std::move(files), std::move(small), std::move(digest), prefix);
}

// ── version checks and the record ────────────────────────────────────────────────────────────────────────────

void check_versions(const Database& db, const fs::path& folder) {
  const int format = store_format_number(db);
  if (format > kStoreFormatPackage)
    throw Error(code::schema_version, folder.string() + ": store format " + std::to_string(format) + " — this build reads subject packages up to " + std::to_string(kStoreFormatPackage));
  if (format < kStoreFormatPackage)
    throw Error(code::not_a_database, folder.string() + ": store format " + std::to_string(format) + " is a dataset database, not a subject package (`dataset split` makes packages)");
}

std::string package_prefix(const Database& db) {
  const auto r = db.one("SELECT ds.path AS d, s.path AS s FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = ?", {read_package_record(db).subject_id});
  if (!r) throw Error(code::not_a_database, db.path().string() + ": the package's subject row is missing");
  return text(*r, "d") + "/" + text(*r, "s");
}

}  // namespace

// ══ the record and the content hash ══════════════════════════════════════════════════════════════════════════════

Json PackageRecord::to_json() const {
  return Json{{"subject_id", subject_id}, {"name", name}, {"kind", kind}, {"schema_version", schema_version}, {"store_format", store_format},
              {"template_reference", template_reference}, {"members", members}, {"content_hash", content_hash}, {"hashed_utc", hashed_utc}};
}
PackageRecord PackageRecord::from_json(const Json& j) {
  PackageRecord r;
  auto str = [&](const char* k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); };
  r.subject_id = str("subject_id"); r.name = str("name"); r.kind = str("kind");
  r.schema_version = j.value("schema_version", 0);
  r.store_format = j.value("store_format", 0);
  r.template_reference = j.contains("template_reference") ? j["template_reference"] : Json(nullptr);
  r.members = j.contains("members") ? j["members"] : Json(nullptr);
  r.content_hash = str("content_hash"); r.hashed_utc = str("hashed_utc");
  return r;
}

PackageRecord read_package_record(const Database& db) {
  const auto r = db.one("SELECT value FROM meta WHERE key = ?", {std::string(kRecordKey)});
  if (!r) throw Error(code::not_a_database, db.path().string() + ": no package record (meta " + kRecordKey + ")");
  return PackageRecord::from_json(Json::parse(text(*r, "value")));
}

void write_package_record(Database& db, const PackageRecord& record) {
  db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value", {std::string(kRecordKey), jdump(record.to_json())});
}

void create_package_tables(Database& db) { db.exec_script(kPackageTables); }

std::string package_content_hash(const Database& db) {
  Sha256 h;
  for (const auto& r : db.all("SELECT key, size_bytes, sha256 FROM key_digest ORDER BY key"))
    h.update(text(r, "key") + " " + text(r, "size_bytes") + " " + text(r, "sha256") + "\n");
  h.update(std::string("rows\n"));
  for (const auto& table : content_tables(db)) {
    const auto info = db.all("SELECT name, pk FROM pragma_table_info(?) ORDER BY cid", {table});
    std::vector<std::string> cols, pk;
    std::vector<std::pair<std::int64_t, std::string>> pks;
    std::string list;
    for (const auto& c : info) {
      cols.push_back(text(c, "name"));
      list += (list.empty() ? "" : ", ") + std::string("\"") + text(c, "name") + "\"";
      if (integer(c, "pk") > 0) pks.push_back({integer(c, "pk"), text(c, "name")});
    }
    std::sort(pks.begin(), pks.end());
    std::string order;
    for (const auto& [i, n] : pks) order += (order.empty() ? "" : ", ") + std::string("\"") + n + "\"";
    const std::string sql = "SELECT " + list + " FROM \"" + table + "\" ORDER BY " + (order.empty() ? "rowid" : order);
    for (const auto& r : db.all(sql)) {
      Json a = Json::array();
      for (const auto& c : cols) a.push_back(value_json(r.at(c)));
      h.update(table + "\t" + jdump(a) + "\n");
    }
  }
  return h.hex();
}

bool refresh_package_record(Database& db) {
  PackageRecord rec = read_package_record(db);
  const std::string now_hash = package_content_hash(db);
  if (rec.content_hash == now_hash) return false;
  rec.content_hash = now_hash;
  rec.hashed_utc = now_utc();
  write_package_record(db, rec);
  return true;
}

std::vector<std::string> list_all_keys(const Store& store, const std::string& prefix) {
  std::vector<std::string> out;
  walk(store, prefix, out);
  std::sort(out.begin(), out.end());
  return out;
}

void rebuild_key_digests(Database& db, const Store& keys) {
  const auto all = list_all_keys(keys);
  db.tx([&] {
    db.execute("DELETE FROM key_digest");
    for (const auto& k : all) {
      const auto [size, sha] = digest_of(keys, k);
      put_digest(db, k, size, sha);
    }
  });
}

// ══ a package, read ══════════════════════════════════════════════════════════════════════════════════════════════

struct SubjectPackage::Impl {
  fs::path folder;
  std::unique_ptr<Database> db;
  std::unique_ptr<FolderStore> files;
  std::unique_ptr<SmallFileStore> small;
};

SubjectPackage SubjectPackage::open(const fs::path& folder_in) {
  const fs::path folder = folder_in;
  if (!fs::is_regular_file(folder / kPackageDatabaseFile)) throw Error(code::not_found, folder.string() + ": not a subject package (no " + kPackageDatabaseFile + ")");
  SubjectPackage p;
  p.impl_ = std::make_unique<Impl>();
  p.impl_->folder = folder;
  p.impl_->db = std::make_unique<Database>(Database::open(folder / kPackageDatabaseFile, false, "", true));
  check_versions(*p.impl_->db, folder);
  p.impl_->files = std::make_unique<FolderStore>(folder);
  p.impl_->small = std::make_unique<SmallFileStore>(*p.impl_->files, *p.impl_->db, SmallFileOptions{kSmallFileThreshold, true});
  return p;
}
SubjectPackage::SubjectPackage(SubjectPackage&&) noexcept = default;
SubjectPackage& SubjectPackage::operator=(SubjectPackage&&) noexcept = default;
SubjectPackage::~SubjectPackage() = default;

const fs::path& SubjectPackage::folder() const { return impl_->folder; }
const Database& SubjectPackage::db() const { return *impl_->db; }
const Store& SubjectPackage::store() const { return *impl_->small; }
PackageRecord SubjectPackage::record() const { return read_package_record(*impl_->db); }
Row SubjectPackage::subject_row() const {
  const auto r = impl_->db->one("SELECT * FROM subject WHERE id = ?", {record().subject_id});
  if (!r) throw Error(code::not_a_database, impl_->folder.string() + ": the package's subject row is missing");
  return *r;
}
std::optional<Json> SubjectPackage::bundle() const { return subject_bundle(*impl_->db, record().subject_id); }
std::string SubjectPackage::content_hash() const { return package_content_hash(*impl_->db); }

VerifyReport SubjectPackage::verify() const {
  VerifyReport rep;
  std::map<std::string, std::pair<std::int64_t, std::string>> recorded;
  for (const auto& r : impl_->db->all("SELECT key, size_bytes, sha256 FROM key_digest")) recorded[text(r, "key")] = {integer(r, "size_bytes"), text(r, "sha256")};
  Sha256 all;
  const auto keys = list_all_keys(*impl_->small);
  std::set<std::string> seen;
  for (const auto& k : keys) {
    seen.insert(k);
    const auto it = recorded.find(k);
    if (it == recorded.end()) { rep.problems.push_back(k + ": a key with no digest"); continue; }
    const auto [size, sha] = digest_of(*impl_->small, k);
    if (static_cast<std::int64_t>(size) != it->second.first) rep.problems.push_back(k + ": size " + std::to_string(size) + ", recorded " + std::to_string(it->second.first));
    else if (sha != it->second.second) rep.problems.push_back(k + ": SHA-256 differs from the recorded one");
  }
  for (const auto& [k, v] : recorded) if (!seen.count(k)) rep.problems.push_back(k + ": recorded but absent");
  // the hash of what is actually there: the digests just recomputed, then the rows
  Database& db = *impl_->db;
  const std::string recorded_hash = record().content_hash;
  rep.content_hash = package_content_hash(db);
  if (!recorded_hash.empty() && rep.content_hash != recorded_hash) rep.problems.push_back("the rows and digests hash to " + rep.content_hash + ", the record says " + recorded_hash);
  rep.ok = rep.problems.empty();
  return rep;
}

// ══ a package, written ═══════════════════════════════════════════════════════════════════════════════════════════

namespace {
void before_close(DatasetWriter& w) {
  if (w.db().read_only()) return;
  refresh_package_record(w.db());
}
}  // namespace

DatasetWriter create_package(const fs::path& folder_in, const std::string& subject_name, const CreatePackageOptions& opt) {
  if (opt.home_dataset.empty()) throw value_error("a package records the dataset it is made in: name it (home_dataset)");
  if (subject_name.empty() || subject_name.find('/') != std::string::npos) throw value_error("invalid subject name: '" + subject_name + "'");
  fs::path folder = fs::absolute(folder_in).lexically_normal();
  if (!folder.has_filename()) folder = folder.parent_path();
  if (fs::exists(folder / kPackageDatabaseFile)) throw Error(code::already_exists, folder.string() + ": already a subject package");
  const bool made = !fs::exists(folder);
  fs::create_directories(folder);
  WriterLock lock;
  try { lock = acquire_writer_lock(folder, opt.built_by.value_or("nxr-io")); } catch (...) { if (made) remove_folder(folder); throw; }
  std::optional<Database> db;
  auto undo = [&] {
    db.reset();
    lock.release();
    if (made) remove_folder(folder);
    else { std::error_code ec; fs::remove(folder / kPackageDatabaseFile, ec); }
  };
  Row home;
  try {
    db.emplace(Database::open(folder / kPackageDatabaseFile, true, opt.built_by.value_or("nxr-io")));
    db->execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat53)});
    db->exec_script(kPackageTables);
    const std::string now = now_utc();
    const std::string datastore_id = opt.datastore_id.value_or(uuid7());
    home = {{"id", opt.home_dataset_id.value_or(uuid7())}, {"datastore_id", datastore_id}, {"name", opt.home_dataset}, {"path", opt.home_dataset},
            {"source_tool", V(opt.subject.source_format)}, {"source_protocol", V(nullptr)}, {"source_path", V(opt.subject.source_path)}, {"created_utc", now},
            {"created_by", V(opt.subject.created_by)}, {"modified_utc", now}, {"modified_by", V(opt.subject.created_by)}};
    db->tx([&] {
      // the hint only: where the package was made; nothing reads it to find data
      db->insert("datastore", {{"id", datastore_id}, {"url", folder.parent_path().string()}, {"driver", "file"}, {"name", folder.parent_path().filename().string()}});
      db->insert("dataset", home);
    });
    ensure_band_dictionary(*db);   // the shared vocabulary a subject's rows name is copied into its package (section 2)
  } catch (...) {
    undo();
    throw;
  }
  Row dsrow = *db->read("dataset", text(home, "id"));
  const std::string prefix = opt.home_dataset + "/" + opt.subject.folder.value_or(subject_name + kPackageSuffix);
  DatasetWriter w = DatasetWriter::over_package(std::move(*db), folder, std::move(dsrow), std::move(lock),
                                                [&](Database& d) { return make_package_store(folder, d, prefix); }, before_close);
  try {
    SubjectOptions so = opt.subject;
    so.default_template = false;
    SubjectWriter sub = SubjectWriter::create(w, subject_name, so);
    sub.complete();
    PackageRecord rec;
    rec.subject_id = sub.id();
    rec.name = subject_name;
    rec.kind = so.kind;
    rec.schema_version = schema_version();
    rec.store_format = kStoreFormatPackage;
    rec.template_reference = nullptr;
    write_package_record(w.db(), rec);
  } catch (...) {
    // nothing is left behind: the writer is closed without a hook, then the folder (made here) or the database goes
    try { w.close(); } catch (...) {}
    if (made) { try { remove_folder(folder); } catch (...) {} }
    else { std::error_code ec; fs::remove(folder / kPackageDatabaseFile, ec); }
    throw;
  }
  return w;
}

DatasetWriter open_package_writer(const fs::path& folder_in, const OpenOptions& opt) {
  fs::path folder = fs::absolute(folder_in).lexically_normal();
  if (!folder.has_filename()) folder = folder.parent_path();
  if (!fs::is_regular_file(folder / kPackageDatabaseFile)) throw Error(code::not_found, folder.string() + ": not a subject package (no " + kPackageDatabaseFile + ")");
  WriterLock held;
  if (opt.lock) held = acquire_writer_lock(folder);
  const bool took = held && !held.borrowed();
  const bool do_recover = opt.recover.value_or(took);
  if (do_recover && !took) {
    const bool borrowed = static_cast<bool>(held);
    held.release();
    throw value_error(folder.string() + ": recovery rolls back pending rows — only the process that took the writer lock may run it" + (borrowed ? " (this one borrowed it)" : ""));
  }
  std::optional<Database> db;
  try {
    if (opt.lock) {
      if (const auto nw = database_not_writable(folder / kPackageDatabaseFile))
        throw Error(code::not_writable, nw->second.string() + ": not writable (" + nw->first + ") — copy the package somewhere writable");
    }
    db.emplace(Database::open(folder / kPackageDatabaseFile));
    check_versions(*db, folder);
  } catch (...) {
    db.reset();
    held.release();
    throw;
  }
  Row home;
  std::string prefix;
  try {
    prefix = package_prefix(*db);
    const auto ds = db->one("SELECT d.* FROM dataset d JOIN subject s ON s.dataset_id = d.id WHERE s.id = ?", {read_package_record(*db).subject_id});
    if (!ds) throw Error(code::not_a_database, folder.string() + ": the package's home dataset row is missing");
    home = *ds;
  } catch (...) {
    db.reset();
    held.release();
    throw;
  }
  DatasetWriter w = DatasetWriter::over_package(std::move(*db), folder, std::move(home), std::move(held),
                                                [&](Database& d) { return make_package_store(folder, d, prefix); }, before_close);
  if (do_recover) {
    try {
      recover(w.db(), w.node_store());
      drain(w.db(), w.node_store());
    } catch (...) {
      try { w.close(); } catch (...) {}
      throw;
    }
  }
  return w;
}

// ══ ids ══════════════════════════════════════════════════════════════════════════════════════════════════════════

std::vector<IdClash> find_id_clashes(const std::vector<fs::path>& packages) {
  std::vector<IdClash> out;
  struct Owner { std::size_t package; std::string shared_key; };   // shared_key: for a shared vocabulary row, its (scope, name)
  std::map<std::pair<std::string, std::string>, Owner> owner;      // (table, id) -> the package that holds it first
  for (std::size_t i = 0; i < packages.size(); ++i) {
    SubjectPackage p = SubjectPackage::open(packages[i]);
    const Database& db = p.db();
    std::set<std::pair<std::string, std::string>> anchors;
    for (const auto& a : db.all("SELECT table_name, row_id FROM anchor_row")) anchors.insert({text(a, "table_name"), text(a, "row_id")});
    for (const auto& table : content_tables(db)) {
      if (table == "dataset" || table == "datastore" || table == "anchor_row") continue;
      const auto info = db.all("SELECT name, pk FROM pragma_table_info(?)", {table});
      bool by_id = false;
      int pk_count = 0;
      for (const auto& c : info) { if (integer(c, "pk") > 0) ++pk_count; if (text(c, "name") == "id" && integer(c, "pk") == 1) by_id = true; }
      if (!by_id || pk_count != 1) continue;
      const bool vocabulary = table == "dictionary";
      for (const auto& r : db.all(vocabulary ? "SELECT id, scope || '/' || name AS shared FROM dictionary WHERE scope IN ('app', 'dataset', 'user')" + std::string(" UNION ALL SELECT id, NULL FROM dictionary WHERE scope = 'subject'")
                                              : "SELECT id, NULL AS shared FROM \"" + table + "\"")) {
        const std::string id = text(r, "id");
        if (anchors.count({table, id})) continue;
        const std::string shared = text(r, "shared");
        const auto [it, fresh] = owner.insert({{table, id}, Owner{i, shared}});
        // a shared vocabulary row copied unchanged into several packages (a split dataset's) is one row under one key
        if (!fresh && it->second.package != i && !(vocabulary && !shared.empty() && it->second.shared_key == shared)) out.push_back({table, id, packages[it->second.package], packages[i]});
      }
    }
  }
  return out;
}

}  // namespace nxr::io
