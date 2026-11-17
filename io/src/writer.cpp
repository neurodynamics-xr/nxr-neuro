#include "nxr/io/writer.h"
#include "nxr/io/statistic.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "named_paths.h"
#include "nxr/io/layout.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {
constexpr int kFormatVersion = 1;  // nxr.io.schema.FORMAT_VERSION: nxr's own layer over the schema (the catalog, the folders, the array layout)

Error value_error(const std::string& m) { return Error(code::invalid_argument, m); }
std::string quote(const std::string& s) { return "'" + s + "'"; }  // a Python repr of a plain string

std::string read_file(const fs::path& f) {
  std::ifstream i(f, std::ios::binary);
  std::stringstream ss;
  ss << i.rdbuf();
  return ss.str();
}
}  // namespace

// ══ DATASETS AND THE CATALOG (D140) ══════════════════════════════════════════════════════════════════════════════

std::optional<Json> read_catalog(const fs::path& datastore) {
  const fs::path f = datastore / kCatalogFile;
  if (!fs::is_regular_file(f)) return std::nullopt;
  try {
    return Json::parse(read_file(f));
  } catch (...) {
    return std::nullopt;
  }
}

Json write_catalog(const fs::path& datastore_in, const std::optional<std::string>& id) {
  const fs::path datastore = datastore_in;
  const auto old = read_catalog(datastore);
  std::string cid = old && old->contains("id") && (*old)["id"].is_string() ? (*old)["id"].get<std::string>() : std::string();
  if (cid.empty()) cid = id ? *id : uuid7();
  std::vector<fs::path> files;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(datastore, ec)) {
    const fs::path f = e.path() / kDatasetFile;
    if (e.is_directory() && fs::is_regular_file(f)) files.push_back(f);
  }
  std::sort(files.begin(), files.end());
  std::vector<Json> rows;
  for (const auto& f : files) {
    const std::string folder = f.parent_path().filename().string();
    Database db = Database::open(f, false, "nxr-convert", true);
    const auto ds = db.one("SELECT * FROM dataset ORDER BY name LIMIT 1");
    if (!ds) continue;
    const auto n = integer(*db.one("SELECT count(*) AS n FROM subject WHERE dataset_id = ? AND status = 'complete'", {text(*ds, "id")}), "n");
    rows.push_back(Json{{"id", text(*ds, "id")}, {"name", text(*ds, "name")}, {"path", folder}, {"database", folder + "/" + kDatasetFile},
                        {"subjects", n}, {"modified_utc", now_utc()}});
  }
  std::sort(rows.begin(), rows.end(), [](const Json& a, const Json& b) { return a["path"].get<std::string>() < b["path"].get<std::string>(); });
  Json cat = Json::object();
  cat["schema_version"] = schema_version();
  cat["format_version"] = kFormatVersion;
  cat["id"] = cid;
  cat["datasets"] = Json::array();
  for (auto& r : rows) cat["datasets"].push_back(std::move(r));
  const fs::path tmp = datastore / (std::string(kCatalogFile) + ".tmp");
  {
    std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
    if (!o) throw Error(code::io, tmp.string() + ": cannot be written");
    o << cat.dump(2, ' ', false);
  }
  fs::rename(tmp, datastore / kCatalogFile);
  return cat;
}

Json list_datasets(const fs::path& datastore) {
  const auto cat = read_catalog(datastore);
  if (!cat) throw Error(code::not_found, datastore.string() + ": no catalog.json — not a datastore (create a dataset first)");
  return *cat;
}

DatasetWriter::DatasetWriter(Database db, fs::path root, Row row, WriterLock lock, bool package)
    : lock_(std::move(lock)), db_(std::make_unique<Database>(std::move(db))), root_(std::move(root)), row_(std::move(row)) {
  if (package) return;
  store_ = std::make_unique<DatastoreStore>(root_);
  store_->set_small_files(text(row_, "path"), store_format_number(*db_) >= 52);
}
DatasetWriter::DatasetWriter(DatasetWriter&&) noexcept = default;
DatasetWriter& DatasetWriter::operator=(DatasetWriter&& o) noexcept {
  if (this != &o) {
    try { close(); } catch (...) {}
    lock_ = std::move(o.lock_); db_ = std::move(o.db_); root_ = std::move(o.root_); store_ = std::move(o.store_); package_store_ = std::move(o.package_store_);
    before_close_ = std::move(o.before_close_); row_ = std::move(o.row_);
  }
  return *this;
}
DatasetWriter::~DatasetWriter() { try { close(); } catch (...) {} }
void DatasetWriter::close() {
  std::exception_ptr failed;
  if (before_close_ && db_ && db_->handle()) {
    auto hook = std::move(before_close_);
    before_close_ = nullptr;
    try { hook(*this); } catch (...) { failed = std::current_exception(); }
  }
  if (store_) store_->close_all();   // the subjects' small-file databases, their WAL files with them
  package_store_.reset();
  if (db_) db_->close();
  lock_.release();
  if (failed) std::rethrow_exception(failed);
}
DatastoreStore& DatasetWriter::store() {
  if (!store_) throw Error(code::invalid_argument, "this writer is over a subject package, which has no datastore: use node_store()");
  return *store_;
}
DatasetWriter DatasetWriter::over_package(Database db, fs::path folder, Row home, WriterLock lock, const std::function<std::unique_ptr<Store>(Database&)>& make_store,
                                          std::function<void(DatasetWriter&)> before_close) {
  DatasetWriter w(std::move(db), folder, std::move(home), std::move(lock), true);
  w.package_store_ = make_store(*w.db_);
  w.before_close_ = std::move(before_close);
  return w;
}
std::optional<Row> DatasetWriter::subject(const std::string& name) const {
  return db_->one("SELECT * FROM subject WHERE dataset_id = ? AND name = ?", {id(), name});
}
std::optional<Row> DatasetWriter::template_() const {
  return db_->one("SELECT * FROM subject WHERE dataset_id = ? AND kind = 'template' AND status = 'complete' ORDER BY created_utc LIMIT 1", {id()});
}

namespace {
const char* const kBandDictionary = "frequency bands";
struct Band { int code; const char* name; const char* sym; int lo, hi; };
const Band kBands[] = {{1, "delta", "δ", 1, 4}, {2, "theta", "θ", 4, 8}, {3, "alpha", "α", 8, 13}, {4, "beta", "β", 13, 30}, {5, "gamma", "γ", 30, 80}};

}  // namespace

// The app-wide `frequency bands` dictionary (D109) — each dataset carries its own copy (D140); `bands.ts`.
std::string ensure_band_dictionary(Database& db) {
  if (const auto have = db.one("SELECT id FROM dictionary WHERE scope = 'app' AND name = ?", {std::string(kBandDictionary)})) return text(*have, "id");
  const std::string did = uuid7();
  db.tx([&] {
    db.insert("dictionary", {{"id", did}, {"name", std::string(kBandDictionary)}, {"scope", "app"},
                             {"description", "the named frequency bands (D109): each entry a band's edges in Hz — a labelling of the frequency space, looked up on any Line"}});
    int i = 0;
    for (const auto& b : kBands) {
      const std::string label = std::string(b.sym) + " " + std::to_string(b.lo) + "–" + std::to_string(b.hi) + " Hz";
      db.insert("dictionary_entry", {{"dictionary_id", did}, {"code", V(b.code)}, {"name", std::string(b.name)}, {"measurement_unit", "Hz"}, {"ordinal", V(i++)},
                                     {"attributes_json", jdump(Json{{"lo_hz", b.lo}, {"hi_hz", b.hi}, {"symbol", b.sym}, {"label", label}, {"coordinate_system", "hz"}})}});
    }
  });
  return did;
}

namespace {

fs::path resolved(const fs::path& p) {
  std::error_code ec;
  fs::path r = fs::weakly_canonical(fs::absolute(p), ec);
  return ec ? fs::absolute(p) : r;
}

void check_dataset_name(const std::string& name) {
  if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos || name.find("..") != std::string::npos)
    throw value_error("invalid dataset name: " + quote(name));
}
}  // namespace

DatasetWriter create_dataset(const fs::path& datastore_in, const std::string& name, const CreateOptions& opt) {
  check_dataset_name(name);
  const fs::path datastore = resolved(datastore_in);
  const fs::path folder = datastore / name;
  const fs::path file = folder / kDatasetFile;
  if (fs::exists(file)) throw Error(code::already_exists, folder.string() + ": already a dataset");
  const bool made = !fs::exists(folder);
  fs::create_directories(folder);
  WriterLock lock;
  try {
    lock = acquire_writer_lock(folder);  // another live writer: refused, naming it (#26)
  } catch (...) {
    if (made) remove_folder(folder);
    throw;
  }
  const auto cat = read_catalog(datastore);
  std::string cid = cat && cat->contains("id") && (*cat)["id"].is_string() ? (*cat)["id"].get<std::string>() : std::string();
  if (cid.empty()) cid = uuid7();
  std::optional<Database> db;
  try {
    if (opt.store_format != 51 && opt.store_format != 52) throw value_error("store_format " + std::to_string(opt.store_format) + ": 51 or 52");
    db.emplace(Database::open(file, true, opt.built_by.value_or(opt.created_by.value_or("nxr-convert"))));
    if (opt.store_format == 52) db->execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat52)});
  } catch (...) {
    lock.release();
    throw;
  }
  Row row;
  try {
    const std::string now = now_utc();
    row = {{"id", uuid7()}, {"datastore_id", cid}, {"name", name}, {"path", name}, {"source_tool", V(opt.source_tool)}, {"source_protocol", V(opt.source_protocol)},
           {"source_path", V(opt.source_path)}, {"created_utc", now}, {"created_by", V(opt.created_by)}, {"modified_utc", now}, {"modified_by", V(opt.created_by)}};
    db->tx([&] {
      db->insert("datastore", {{"id", cid}, {"url", datastore.string()}, {"driver", "file"}, {"name", datastore.filename().string()}});
      db->insert("dataset", row);
    });
    ensure_band_dictionary(*db);
    DatastoreStore store(datastore);
    store.set_small_files(name, store_format_number(*db) >= 52);
    drain(*db, store);
  } catch (...) {
    db.reset();
    lock.release();
    if (made) remove_folder(folder);
    else { std::error_code ec; fs::remove(file, ec); }
    throw;
  }
  write_catalog(datastore, cid);
  Row dsrow = *db->read("dataset", text(row, "id"));
  return DatasetWriter(std::move(*db), datastore, std::move(dsrow), std::move(lock));
}

DatasetWriter open_writer(const fs::path& folder_in, const OpenOptions& opt) {
  const fs::path folder = resolved(folder_in);
  WriterLock held;
  if (opt.lock) held = acquire_writer_lock(folder);
  // RECOVERY ONLY BY THE TAKER (#26): a BORROWED lock recovers nothing
  const bool took = held && !held.borrowed();
  const bool do_recover = opt.recover.value_or(took);
  if (do_recover && !took) {
    const bool borrowed = static_cast<bool>(held);
    held.release();
    throw value_error(folder.string() + ": recovery rolls back pending rows — only the process that took the writer lock may run it" +
                      (borrowed ? " (this one borrowed it)" : ""));
  }
  // A WRITABLE FOLDER, A READ-ONLY DATABASE: refused here, clearly, the lock let go
  if (opt.lock) {
    if (const auto nw = database_not_writable(folder / kDatasetFile)) {
      held.release();
      throw Error(code::not_writable, nw->second.string() + ": not writable (" + nw->first +
                                          ") — nxr-convert cannot write this dataset (read-only media, or no write permission); copy it somewhere writable");
    }
  }
  std::optional<Database> db;
  try {
    db.emplace(Database::open(folder / kDatasetFile));
  } catch (...) {
    held.release();
    throw;
  }
  const auto ds = db->one("SELECT * FROM dataset ORDER BY name LIMIT 1");
  if (!ds) {
    db.reset();
    held.release();
    throw Error(code::not_a_database, folder.string() + ": its database holds no dataset row");
  }
  const fs::path root = folder.parent_path();
  if (store_format_number(*db) > kStoreFormatNewest) {
    const int v = store_format_number(*db);
    db.reset();
    held.release();
    throw Error(code::schema_version, folder.string() + ": store format " + std::to_string(v) + " — this build reads up to " + std::to_string(kStoreFormatNewest));
  }
  if (do_recover) {
    try {
      DatastoreStore store(root);
      store.set_small_files(text(*ds, "path"), store_format_number(*db) >= 52);
      recover(*db, store);
      drain(*db, store);
    } catch (...) {  // a failed recovery leaks neither the database nor the lock (F5)
      db.reset();
      held.release();
      throw;
    }
  }
  return DatasetWriter(std::move(*db), root, *ds, std::move(held));
}

void delete_dataset(const fs::path& datastore, const std::string& name) {
  check_dataset_name(name);
  const fs::path folder = datastore / name;
  WriterLock held;
  if (fs::exists(folder)) held = acquire_writer_lock(folder);
  try {
    remove_folder(folder);
  } catch (...) {
    held.release();
    throw;
  }
  held.release();
  write_catalog(datastore);
}

// ══ THE PROCEDURE ════════════════════════════════════════════════════════════════════════════════════════════════

void set_status(const Database& db, const std::string& table, const std::string& id, const std::string& status) {
  db.execute("UPDATE " + table + " SET status = ? WHERE id = ?", {status, id});
}

void create_node(Database& db, Store& store, const std::string& table, const std::string& id, const std::function<void()>& rows,
                 const Populate& populate) {
  db.tx([&] {
    rows();
    set_status(db, table, id, "pending");
  });
  drain(db, store);
  try {
    if (populate) {
      const auto at = location_of(db, table, id);
      populate(store, at ? *at : std::string());
    }
  } catch (...) {
    remove_node(db, store, table, id);
    throw;
  }
  db.tx([&] { set_status(db, table, id, "complete"); });
  drain(db, store);
}

void update_node(Database& db, Store& store, const std::function<void()>& rows) {
  db.tx(rows);
  drain(db, store);
}

namespace {
struct RollbackOnly {};

// A store-relative path removed from a subject's store: never outside it, never through a link.
void remove_store_path(Store& store, const std::string& rel) {
  std::stringstream ss(rel);
  std::string part, p;
  bool any = false;
  while (std::getline(ss, part, '/')) {
    if (part.empty()) continue;
    if (part == "..") return;  // outside the store: refused
    p = join_key(p, part);
    any = true;
    if (!store.exists_prefix(p)) return;
    if (store.is_link(p)) return;  // no link followed
  }
  if (any) store.erase_prefix(p);
}
}  // namespace

void delete_row(const Database& db, const std::string& table, const std::string& id) {
  if (table != "manifold") {
    db.execute("DELETE FROM " + table + " WHERE id = ?", {id});
    return;
  }
  const auto m = db.one("SELECT subject_id FROM manifold WHERE id = ?", {id});
  db.execute("DELETE FROM manifold WHERE id = ?", {id});   // its components cascade
  if (!m) return;
  // what no manifold of the subject stands on any more: the deleted ones' geometries and topologies (only manifolds point at them)
  const Value sid = m->at("subject_id");
  db.execute("DELETE FROM geometry WHERE subject_id = ? AND NOT EXISTS (SELECT 1 FROM manifold x WHERE x.geometry_id = geometry.id)", {sid});
  db.execute("DELETE FROM topology WHERE subject_id = ? AND NOT EXISTS (SELECT 1 FROM manifold x WHERE x.topology_id = topology.id)", {sid});
}

void remove_node(Database& db, Store& datastore, const std::string& table, const std::string& id) {
  const auto row = db.read(table, id);
  if (!row) return;
  const auto at = location_of(db, table, id);
  db.tx([&] { set_status(db, table, id, "deleting"); });
  if (table == "subject") {
    if (at) datastore.erase_prefix(*at);  // the whole store
  } else {
    // D164: the store paths this delete frees are the ones the subject's rows named before it and no remaining row names after it —
    // decided by deleting inside a transaction that is rolled back (the cascade says which rows go), so the bytes go BEFORE the row.
    // A node with no path of its own (a source estimate, a joint Selection) frees the arrays its rows name (measurement rows' array_path).
    const std::string sid = text(*row, "subject_id");
    const auto before = named_paths(db, sid);
    std::vector<std::pair<std::string, std::string>> after;
    try {
      db.tx([&] {
        delete_row(db, table, id);
        after = named_paths(db, sid);
        throw RollbackOnly{};
      });
    } catch (const RollbackOnly&) {}
    PrefixStore store(datastore, store_path_of(db, sid));
    for (const auto& [p, by] : before) {
      if (named_overlapping(after, p)) continue;
      (void)by;
      remove_store_path(store, p);
    }
  }
  db.tx([&] {
    if (table == "subject") {
      // the SELECTIONS first: a dictionary in use is RESTRICT, and the cascade may reach the subject's own first
      db.execute("DELETE FROM selection WHERE subject_id = ?", {id});
    }
    delete_row(db, table, id);
  });
  drain(db, datastore);
}

Recovered recover(Database& db, Store& store) {
  Recovered out;
  for (const char* table : kStatusTables) {
    if (!has_table(db, table)) continue;                     // a dataset made before the statistic table
    for (const auto& r : db.all(std::string("SELECT id, status FROM ") + table + " WHERE status <> 'complete'")) {
      if (!db.read(table, text(r, "id"))) continue;  // gone with a parent removed before it
      remove_node(db, store, table, text(r, "id"));
      (text(r, "status") == "pending" ? out.rolled_back : out.finished) += 1;
    }
  }
  return out;
}

}  // namespace nxr::io
