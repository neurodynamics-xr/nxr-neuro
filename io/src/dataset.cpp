#include "nxr/io/dataset.h"

#include <sqlite3.h>

namespace fs = std::filesystem;

namespace nxr::io {

Dataset::Dataset(Dataset&& o) noexcept : folder_(std::move(o.folder_)), db_(o.db_) { o.db_ = nullptr; }

Dataset& Dataset::operator=(Dataset&& o) noexcept {
  if (this != &o) {
    if (db_) sqlite3_close(static_cast<sqlite3*>(db_));
    folder_ = std::move(o.folder_);
    db_ = o.db_;
    o.db_ = nullptr;
  }
  return *this;
}

Dataset::~Dataset() {
  if (db_) sqlite3_close(static_cast<sqlite3*>(db_));
}

Dataset Dataset::open(const fs::path& folder) {
  const fs::path file = folder / "dataset.sqlite";
  const std::string p = file.string();
  if (!fs::is_regular_file(file) && fs::is_regular_file(folder / "catalogue.sqlite")) {
    throw DatasetError(folder.string() + ": a catalogue of subject packages has no dataset database; read a subject through its package (--dataset and --subject)", code::not_found);
  }
  if (!fs::is_regular_file(file)) {
    throw DatasetError(p + ": no dataset database here — create the dataset first (nxr dataset create)", code::not_found);
  }
  Dataset d;
  d.folder_ = folder;
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(p.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    const std::string msg = db ? sqlite3_errmsg(db) : "cannot open";
    if (db) sqlite3_close(db);
    throw DatasetError(p + ": " + msg);
  }
  d.db_ = db;
  try {
    const auto meta = d.query("SELECT count(*) AS n FROM sqlite_master WHERE type = 'table' AND name = 'meta'");
    if (integer(meta.at(0), "n") == 0) throw DatasetError(p + ": not a dataset database (no schema)");
    const std::int64_t v = integer(d.query("PRAGMA user_version").at(0), "user_version");
    if (v != kSchemaVersion) {
      const std::string how =
          v == kSchemaVersion - 1
              ? "run `nxr migrate --dataset <dataset folder>` (it keeps the old database and the arrays; add `--dry-run` to see what it will do first)"
              : "re-convert it from its source (D139/D144)";
      throw DatasetError(p + ": user_version " + std::to_string(v) + ", this build reads " +
                         std::to_string(kSchemaVersion) + " — " + how, code::schema_version);
    }
  } catch (...) {
    throw;  // d's destructor closes the handle
  }
  return d;
}

std::vector<Row> Dataset::query(const std::string& sql, const std::vector<Value>& params) const {
  sqlite3* db = static_cast<sqlite3*>(db_);
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) {
    throw DatasetError(std::string("sqlite: ") + sqlite3_errmsg(db) + " in: " + sql);
  }
  for (std::size_t i = 0; i < params.size(); ++i) {
    const int k = static_cast<int>(i) + 1;
    const Value& v = params[i];
    if (std::holds_alternative<std::monostate>(v)) sqlite3_bind_null(st, k);
    else if (auto* n = std::get_if<std::int64_t>(&v)) sqlite3_bind_int64(st, k, *n);
    else if (auto* x = std::get_if<double>(&v)) sqlite3_bind_double(st, k, *x);
    else {
      const std::string& s = std::get<std::string>(v);
      sqlite3_bind_text(st, k, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT);
    }
  }
  std::vector<Row> rows;
  int rc;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    Row r;
    for (int c = 0, n = sqlite3_column_count(st); c < n; ++c) {
      const std::string name = sqlite3_column_name(st, c);
      switch (sqlite3_column_type(st, c)) {
        case SQLITE_INTEGER: r[name] = static_cast<std::int64_t>(sqlite3_column_int64(st, c)); break;
        case SQLITE_FLOAT:   r[name] = sqlite3_column_double(st, c); break;
        case SQLITE_NULL:    r[name] = std::monostate{}; break;
        default: {  // TEXT or BLOB
          const auto* b = static_cast<const char*>(sqlite3_column_blob(st, c));
          r[name] = std::string(b ? b : "", static_cast<std::size_t>(sqlite3_column_bytes(st, c)));
        }
      }
    }
    rows.push_back(std::move(r));
  }
  const std::string err = rc == SQLITE_DONE ? "" : sqlite3_errmsg(db);
  sqlite3_finalize(st);
  if (!err.empty()) throw DatasetError("sqlite: " + err);
  return rows;
}

}  // namespace nxr::io
