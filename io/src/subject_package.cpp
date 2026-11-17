#include "nxr/io/subject_package.h"

#include <algorithm>
#include <cctype>
#include <functional>
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
    "CREATE TABLE IF NOT EXISTS anchor_row (table_name TEXT NOT NULL, row_id TEXT NOT NULL, from_subject_id TEXT NOT NULL, PRIMARY KEY (table_name, row_id)) STRICT;\n"
    "CREATE TABLE IF NOT EXISTS origin (table_name TEXT NOT NULL, row_id TEXT NOT NULL, source TEXT NOT NULL, licence TEXT NOT NULL, attribution TEXT NOT NULL,"
    " redistributable INTEGER NOT NULL CHECK (redistributable IN (0, 1)), PRIMARY KEY (table_name, row_id)) STRICT;\n";

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
              {"template_reference", template_reference}, {"release", release}, {"description", description}, {"registration", registration}, {"members", members}, {"content_hash", content_hash}, {"hashed_utc", hashed_utc}};
}
PackageRecord PackageRecord::from_json(const Json& j) {
  PackageRecord r;
  auto str = [&](const char* k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); };
  r.subject_id = str("subject_id"); r.name = str("name"); r.kind = str("kind");
  r.schema_version = j.value("schema_version", 0);
  r.store_format = j.value("store_format", 0);
  r.template_reference = j.contains("template_reference") ? j["template_reference"] : Json(nullptr);
  r.release = j.contains("release") ? j["release"] : Json(nullptr);
  r.description = j.contains("description") ? j["description"] : Json(nullptr);
  r.registration = j.contains("registration") ? j["registration"] : Json(nullptr);
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

namespace {
// THE CONTENT HASH covers content only. A row is hashed by what it says and by what it points at, never by its id and never by when or by whom it was written:
//   - its own `id` is left out; a reference to another row (a declared foreign key, or a UUID inside a text or a `_json` value) is replaced by THAT ROW'S digest, so two
//     builds of the same content, whose rows got other ids, give the same digests;
//   - columns that record the time of writing (`*_utc`), the program that wrote (`created_by`, `modified_by`) and the keys of such times inside a `_json` value are left out;
//   - the build's own bookkeeping is not content: `job` (the log of jobs), `datastore` (where the package was made: a folder), and the tables the content hash never covered;
//   - the digests are sorted, so the order rows were written in does not matter.
// A package is hashed, then, by its keys' digests and the set of its rows' digests: the same content is the same hash, any changed value is another.
bool is_uuid_at(const std::string& t, std::size_t i) {
  if (i + 36 > t.size()) return false;
  for (std::size_t k = 0; k < 36; ++k) {
    const char c = t[i + k];
    if (k == 8 || k == 13 || k == 18 || k == 23) { if (c != '-') return false; }
    else if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  }
  return true;
}
bool left_out_column(const std::string& c) {
  return (c.size() > 4 && c.compare(c.size() - 4, 4, "_utc") == 0) || c == "created_by" || c == "modified_by" || c == "built_by";
}

// The rows reference each other in cycles (a field, its operator, a selection), so a digest cannot be made by following references once. Each row is first written as text in
// which every reference is a marker; the digests are then refined round by round, each marker replaced by the digest the row it names had in the round before (round 0: by
// the table it names), until no two rows are told apart that were not before. The result does not depend on any order, nor on any id.
struct RowDigests {
  struct Table { std::vector<std::string> cols; std::set<std::string> fk; std::vector<Row> rows; bool has_id = false; };
  std::map<std::string, Table> tables;
  std::map<std::string, std::size_t> index_of;               // a row id -> its slot
  std::vector<std::string> table_of, text_of, id_of;          // per slot

  static std::string marker(const std::string& id) { return "@@ref{" + id + "}@@"; }
  static std::string with_markers(const std::string& t) {
    std::string out;
    for (std::size_t i = 0; i < t.size();) {
      if (is_uuid_at(t, i)) { out += marker(t.substr(i, 36)); i += 36; } else out += t[i++];
    }
    return out;
  }
  static Json normalise(const Json& j) {
    if (j.is_string()) return with_markers(j.get<std::string>());
    if (j.is_array()) { Json a = Json::array(); for (const auto& x : j) a.push_back(normalise(x)); return a; }
    if (j.is_object()) {
      Json o = Json::object();
      for (auto it = j.begin(); it != j.end(); ++it) if (!left_out_column(it.key())) o[it.key()] = normalise(it.value());
      return o;
    }
    return j;
  }
  void build() {
    for (const auto& [name, t] : tables)
      if (t.has_id) for (std::size_t i = 0; i < t.rows.size(); ++i) { index_of[text(t.rows[i], "id")] = table_of.size(); table_of.push_back(name); id_of.push_back(text(t.rows[i], "id")); text_of.emplace_back(); }
    std::size_t slot_no_id = table_of.size();
    (void)slot_no_id;
    for (const auto& [name, t] : tables) {
      for (std::size_t i = 0; i < t.rows.size(); ++i) {
        const Row& r = t.rows[i];
        Json a = Json::array();
        for (const auto& c : t.cols) {
          if (left_out_column(c) || (c == "id" && t.has_id)) continue;
          const Value& v = r.at(c);
          if (std::holds_alternative<std::string>(v)) {
            const std::string& x = std::get<std::string>(v);
            if (t.fk.count(c)) { a.push_back(Json::array({c, marker(x)})); continue; }
            if (c.size() > 5 && c.compare(c.size() - 5, 5, "_json") == 0) {
              const Json p = Json::parse(x, nullptr, false);
              if (!p.is_discarded()) { a.push_back(Json::array({c, normalise(p)})); continue; }
            }
            a.push_back(Json::array({c, with_markers(x)}));
          } else {
            a.push_back(Json::array({c, value_json(v)}));
          }
        }
        const std::string line = name + "\t" + jdump(a);
        if (t.has_id) text_of[index_of.at(text(r, "id"))] = line; else rows_without_id.push_back(line);
      }
    }
  }
  std::vector<std::string> rows_without_id;
  // `text` with each marker replaced by `digest_of(id)`
  template <class F> static std::string resolve(const std::string& t, F digest_of) {
    std::string out;
    for (std::size_t i = 0; i < t.size();) {
      if (t.compare(i, 6, "@@ref{") == 0 && i + 6 + 36 + 3 <= t.size() && t.compare(i + 6 + 36, 3, "}@@") == 0) { out += digest_of(t.substr(i + 6, 36)); i += 6 + 36 + 3; }
      else out += t[i++];
    }
    return out;
  }
  std::vector<std::string> digests() {
    build();
    const std::size_t n = table_of.size();
    std::vector<std::string> cur(n), prev;
    auto sha = [](const std::string& s) { Sha256 h; h.update(s); return h.hex(); };
    for (std::size_t i = 0; i < n; ++i) cur[i] = sha(resolve(text_of[i], [&](const std::string& id) { const auto it = index_of.find(id); return it == index_of.end() ? std::string("unknown") : "table:" + table_of[it->second]; }));
    std::size_t distinct = std::set<std::string>(cur.begin(), cur.end()).size();
    for (int round = 0; round < 40; ++round) {
      prev = cur;
      for (std::size_t i = 0; i < n; ++i) cur[i] = sha(prev[i] + resolve(text_of[i], [&](const std::string& id) { const auto it = index_of.find(id); return it == index_of.end() ? std::string("unknown") : prev[it->second]; }));
      const std::size_t now = std::set<std::string>(cur.begin(), cur.end()).size();
      if (now == distinct) break;
      distinct = now;
    }
    std::vector<std::string> out;
    for (std::size_t i = 0; i < n; ++i) out.push_back(table_of[i] + "\t" + cur[i]);
    for (const auto& l : rows_without_id) out.push_back(resolve(l, [&](const std::string& id) { const auto it = index_of.find(id); return it == index_of.end() ? std::string("unknown") : cur[it->second]; }));
    std::sort(out.begin(), out.end());
    return out;
  }
};
}  // namespace

std::string package_content_hash(const Database& db) {
  Sha256 h;
  // the arrays' bytes: every key but the nodes' own texts (`zarr.json`: written from the rows, with their ids and times, so they say nothing the rows do not)
  for (const auto& r : db.all("SELECT key, size_bytes, sha256 FROM key_digest ORDER BY key")) {
    const std::string k = text(r, "key");
    if (k == "zarr.json" || (k.size() > 10 && k.compare(k.size() - 10, 10, "/zarr.json") == 0)) continue;
    h.update(k + " " + text(r, "size_bytes") + " " + text(r, "sha256") + "\n");
  }
  h.update(std::string("rows\n"));
  RowDigests rd;
  for (const auto& table : content_tables(db)) {
    if (table == "job" || table == "datastore") continue;
    RowDigests::Table t;
    for (const auto& c : db.all("SELECT name FROM pragma_table_info(?) ORDER BY cid", {table})) t.cols.push_back(text(c, "name"));
    for (const auto& f : db.all("SELECT \"from\" AS f FROM pragma_foreign_key_list(?)", {table})) t.fk.insert(text(f, "f"));
    t.has_id = std::find(t.cols.begin(), t.cols.end(), "id") != t.cols.end() && integer(*db.one("SELECT count(*) AS n FROM pragma_table_info(?) WHERE pk > 0", {table}), "n") == 1 &&
               integer(*db.one("SELECT pk FROM pragma_table_info(?) WHERE name = 'id'", {table}), "pk") == 1;
    std::string list;
    for (const auto& c : t.cols) list += (list.empty() ? "" : ", ") + std::string("\"") + c + "\"";
    t.rows = db.all("SELECT " + list + " FROM \"" + table + "\"");
    rd.tables[table] = std::move(t);
  }
  for (const auto& l : rd.digests()) h.update(l + "\n");
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
    if (opt.subject.kind == "registration") allow_subject_kind(*db, "registration");
    const std::string now = now_utc();
    const std::string datastore_id = opt.datastore_id.value_or(uuid7());
    home = {{"id", opt.home_dataset_id.value_or(uuid7())}, {"datastore_id", datastore_id}, {"name", opt.home_dataset}, {"path", opt.home_dataset},
            {"source_tool", V(opt.subject.source_format)}, {"source_protocol", V(nullptr)}, {"source_path", V(opt.subject.source_path)}, {"created_utc", now},
            {"created_by", V(opt.subject.created_by)}, {"modified_utc", now}, {"modified_by", V(opt.subject.created_by)}};
    db->tx([&] {
      // the hint only: where the package was made (its own folder, so two packages never share a url); nothing reads it to find data
      db->insert("datastore", {{"id", datastore_id}, {"url", folder.string()}, {"driver", "file"}, {"name", folder.filename().string()}});
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
    if (read_package_record(*db).release.is_object())
      throw Error(code::not_writable, folder.string() + ": a released package is frozen: a change is a new version (a new package)");
    db->exec_script(kPackageTables);   // a package made before a table of the package format existed gets it
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

// ══ origin, anchors, the registration kind ══════════════════════════════════════════════════════════════════════

void set_origin(Database& db, const std::string& table, const std::string& id, const OriginRecord& o) {
  if (o.source.empty() || o.licence.empty() || o.attribution.empty()) throw value_error("an origin record names its source, its licence and who is to be credited");
  if (!db.read(table, id)) throw Error(code::no_such_row, table + " " + id + ": no such row to give an origin");
  db.execute("INSERT INTO origin (table_name, row_id, source, licence, attribution, redistributable) VALUES (?, ?, ?, ?, ?, ?) "
             "ON CONFLICT (table_name, row_id) DO UPDATE SET source = excluded.source, licence = excluded.licence, attribution = excluded.attribution, redistributable = excluded.redistributable",
             {table, id, o.source, o.licence, o.attribution, V(o.redistributable)});
}

std::optional<OriginRecord> origin_of(const Database& db, const std::string& table, const std::string& id) {
  const auto r = db.one("SELECT * FROM origin WHERE table_name = ? AND row_id = ?", {table, id});
  if (!r) return std::nullopt;
  return OriginRecord{text(*r, "source"), text(*r, "licence"), text(*r, "attribution"), integer(*r, "redistributable") != 0};
}

std::vector<PackageItem> package_items(const Database& db) {
  std::vector<PackageItem> out;
  const std::string sid = read_package_record(db).subject_id;
  for (const char* t : {"manifold", "field", "operator", "selection", "statistic"}) {
    if (!db.one("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", {std::string(t)})) continue;
    for (const auto& r : db.all(std::string("SELECT id, name, path FROM ") + t + " WHERE subject_id = ? AND path IS NOT NULL AND status = 'complete' ORDER BY path", {sid}))
      out.push_back({t, text(r, "id"), text(r, "name"), text(r, "path")});
  }
  return out;
}

std::vector<PackageItem> items_without_origin(const Database& db) {
  std::vector<PackageItem> out;
  for (const auto& i : package_items(db)) if (!origin_of(db, i.table, i.id)) out.push_back(i);
  return out;
}

namespace {
struct ForeignKey { std::string from, parent; };
std::vector<ForeignKey> foreign_keys_for(const Database& db, const std::string& table) {
  std::vector<ForeignKey> out;
  for (const auto& f : db.all("SELECT \"from\" AS c, \"table\" AS p, \"to\" AS t FROM pragma_foreign_key_list(?)", {table})) {
    const std::string to = opt_text(f, "t").value_or("id");
    if (to != "id") throw Error(code::unsupported, "the foreign key " + table + "." + text(f, "c") + " names " + text(f, "p") + "." + to + ", not an id: anchor rows cannot follow it");
    out.push_back({text(f, "c"), text(f, "p")});
  }
  return out;
}
std::vector<std::string> column_names(const Database& db, const std::string& table) {
  std::vector<std::string> out;
  for (const auto& r : db.all("SELECT name FROM pragma_table_info(?) ORDER BY cid", {table})) out.push_back(text(r, "name"));
  return out;
}
std::string column_list(const std::vector<std::string>& cols) {
  std::string s;
  for (const auto& c : cols) s += (s.empty() ? "" : ", ") + std::string("\"") + c + "\"";
  return s;
}
bool is_bookkeeping(const std::string& t) {
  return t == "meta" || t == "sync_job" || t == "small_files" || t == "key_digest" || t == "anchor_row" || t == "origin" || t == "variable_dictionary" || t == "subject_variable" || t.rfind("sqlite_", 0) == 0;
}
std::vector<std::string> tables_in(const Database& db, const std::string& schema) {
  std::vector<std::string> out;
  for (const auto& r : db.all("SELECT name FROM " + schema + ".sqlite_master WHERE type = 'table' ORDER BY name")) out.push_back(text(r, "name"));
  return out;
}
bool contains(const std::vector<std::string>& v, const std::string& x) { return std::find(v.begin(), v.end(), x) != v.end(); }
}  // namespace

std::int64_t add_anchor_rows(Database& db, const fs::path& source, const std::vector<std::pair<std::string, std::string>>& roots) {
  const fs::path file = fs::is_directory(source) ? source / kPackageDatabaseFile : source;
  if (!fs::is_regular_file(file)) throw Error(code::not_found, source.string() + ": not a subject package");
  const bool fk_was_on = integer(*db.one("PRAGMA foreign_keys"), "foreign_keys") != 0;
  if (fk_was_on) db.execute("PRAGMA foreign_keys = OFF");   // parents and children arrive in any order; the check at the end says whether they are whole
  db.execute("ATTACH DATABASE ? AS anchor_src", {file.string()});
  std::int64_t added = 0;
  try {
    const std::int64_t jobs_before = integer(*db.one("SELECT coalesce(max(id), 0) AS m FROM sync_job"), "m");
    const auto src_tables = tables_in(db, "anchor_src");
    db.tx([&] {
      std::vector<std::pair<std::string, ForeignKey>> fks;
      for (const auto& t : tables_in(db, "main")) { if (is_bookkeeping(t)) continue; for (const auto& f : foreign_keys_for(db, t)) fks.push_back({t, f}); }
      std::set<std::pair<std::string, std::string>> seen;
      std::function<void(const std::string&, const std::string&)> anchor = [&](const std::string& table, const std::string& id) {
        if (db.one("SELECT 1 FROM main.\"" + table + "\" WHERE id = ?", {id})) return;
        if (!seen.insert({table, id}).second) return;
        if (!contains(src_tables, table)) throw value_error("a row names a row of " + table + ", which " + source.string() + " does not have");
        const auto cols = column_names(db, table);
        db.execute("INSERT INTO main.\"" + table + "\" (" + column_list(cols) + ") SELECT " + column_list(cols) + " FROM anchor_src.\"" + table + "\" WHERE id = ?", {id});
        if (integer(*db.one("SELECT changes() AS n"), "n") != 1) throw value_error(table + " " + id + " is not a row of " + source.string());
        std::string owner = id;
        if (contains(cols, "subject_id")) owner = text(*db.one("SELECT subject_id FROM main.\"" + table + "\" WHERE id = ?", {id}), "subject_id");
        db.execute("INSERT OR IGNORE INTO main.anchor_row (table_name, row_id, from_subject_id) VALUES (?, ?, ?)", {table, id, owner});
        ++added;
        struct Child { const char* table; const char* column; };
        const std::vector<Child> children = table == "topology" ? std::vector<Child>{{"topology_cell", "topology_id"}}
                                          : table == "geometry" ? std::vector<Child>{{"geometry_dimension", "geometry_id"}}
                                          : table == "dictionary" ? std::vector<Child>{{"dictionary_entry", "dictionary_id"}} : std::vector<Child>{};
        for (const auto& ch : children) {
          const auto cc = column_names(db, ch.table);
          db.execute("INSERT OR IGNORE INTO main.\"" + std::string(ch.table) + "\" (" + column_list(cc) + ") SELECT " + column_list(cc) + " FROM anchor_src.\"" + ch.table + "\" WHERE " + ch.column + " = ?", {id});
        }
        if (table == "manifold") for (const auto& c : db.all("SELECT id FROM anchor_src.manifold WHERE component_of_id = ?", {id})) anchor("manifold", text(c, "id"));
      };
      for (const auto& [t, id] : roots) anchor(t, id);
      for (bool grew = true; grew;) {
        grew = false;
        for (const auto& [t, f] : fks) {
          for (const auto& m : db.all("SELECT DISTINCT \"" + f.from + "\" AS v FROM main.\"" + t + "\" WHERE \"" + f.from + "\" IS NOT NULL AND \"" + f.from + "\" NOT IN (SELECT id FROM main.\"" + f.parent + "\")")) {
            anchor(f.parent, text(m, "v"));
            grew = true;
          }
        }
      }
      db.execute("DELETE FROM sync_job WHERE id > ?", {V(jobs_before)});   // the triggers' jobs for rows that are another package's, never this one's to write
    });
  } catch (...) {
    try { db.execute("DETACH DATABASE anchor_src"); } catch (...) {}
    if (fk_was_on) db.execute("PRAGMA foreign_keys = ON");
    throw;
  }
  db.execute("DETACH DATABASE anchor_src");
  if (fk_was_on) db.execute("PRAGMA foreign_keys = ON");
  return added;
}

std::vector<std::string> anchor_differences(const Database& db_in, const fs::path& source) {
  Database& db = const_cast<Database&>(db_in);
  const fs::path file = source / kPackageDatabaseFile;
  if (!fs::is_regular_file(file)) throw Error(code::not_found, source.string() + ": not a subject package");
  std::vector<std::string> out;
  db.execute("ATTACH DATABASE ? AS anchor_src", {file.string()});
  try {
    for (const auto& a : db.all("SELECT table_name, row_id FROM anchor_row ORDER BY table_name, row_id")) {
      const std::string t = text(a, "table_name"), l = "\"" + t + "\"";
      const auto here = db.one("SELECT count(*) AS n FROM (SELECT * FROM main." + l + " WHERE id = ? EXCEPT SELECT * FROM anchor_src." + l + " WHERE id = ?)", {text(a, "row_id"), text(a, "row_id")});
      const auto there = db.one("SELECT count(*) AS n FROM anchor_src." + l + " WHERE id = ?", {text(a, "row_id")});
      if (integer(*here, "n") != 0 || integer(*there, "n") != 1) out.push_back(t + " " + text(a, "row_id"));
    }
  } catch (...) {
    try { db.execute("DETACH DATABASE anchor_src"); } catch (...) {}
    throw;
  }
  db.execute("DETACH DATABASE anchor_src");
  return out;
}

bool allow_subject_kind(Database& db, const std::string& kind) {
  const std::string quoted = "'" + kind + "'";
  const auto r = db.one("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'subject'");
  if (!r) throw Error(code::not_a_database, "no subject table");
  if (text(*r, "sql").find(quoted) != std::string::npos) return false;
  static const std::string before = "kind IN ('subject', 'template', 'group')";
  if (text(*r, "sql").find(before) == std::string::npos) throw Error(code::schema_version, "the subject table's kind list is not the schema's; " + kind + " cannot be allowed in place");
  const std::int64_t v = integer(*db.one("PRAGMA schema_version"), "schema_version");
  db.with_writable_schema([&] {
    db.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name = 'subject'", {V(before), V("kind IN ('subject', 'template', 'group', " + quoted + ")")});
    db.execute("PRAGMA schema_version = " + std::to_string(v + 1));
  });
  const auto ok = db.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after " + kind + " was allowed");
  db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {V("subject_kind:" + kind), V(now_utc() + " allowed in place (additive; user_version unchanged)")});
  return true;
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
