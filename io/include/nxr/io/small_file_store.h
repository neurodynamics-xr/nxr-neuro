// nxr-io — the SMALL-FILE STORE of format 52.
//
// A subject's keys are the same zarr keys the folder tree holds, but the SMALL ones live as rows of ONE SQLite file in the subject's folder,
// `<subject>.nxr.zarr/small_files.sqlite` (table `small_files(key TEXT PRIMARY KEY, value BLOB)`): every `zarr.json` and every chunk or shard whose value is
// under a threshold (default 1 MiB). A value at or over the threshold stays a FILE of the subject's folder tree (the large shards), through a FolderStore.
//
//   SmallFileStore store(folder_store_of_the_subject, subject_folder / kSmallFilesFile);
//   store.set("a/zarr.json", bytes);        // a row
//   store.set("a/c.0.0", 40 MiB of bytes);  // a file
//
// The reads look at the table first, then at the files. A write routes by size and removes the key from the other place, so a key lives in one place; a
// streamed write (`open_writer`) buffers until the threshold, then spills to a file. `list` merges both; `erase_prefix` and `rename_prefix` act on both.
// The database is in WAL mode (synchronous NORMAL, a 5 s busy timeout): readers while one writer; the dataset's writer lock serialises writers. The keys of the
// table and of the files never overlap, and a reader of format 51 or 50 (files only) never meets this class.
//
// `DatastoreStore` is the dataset-level store: keys `<dataset>/<subject>/…`; the first two segments name the subject, whose SmallFileStore (format 52, or a
// subject that already has its small_files.sqlite) or plain FolderStore the key goes to. Anything with fewer than three segments is the folder tree's.
#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "nxr/io/database.h"
#include "nxr/io/store.h"

namespace nxr::io {

constexpr const char* kSmallFilesFile = "small_files.sqlite";                  // in the subject's folder; its -wal / -shm / -journal files come and go with it
constexpr std::uint64_t kSmallFileThreshold = 1024ULL * 1024ULL;               // a value under this many bytes is a row, at or over it a file

struct SmallFileOptions {
  std::uint64_t threshold = kSmallFileThreshold;
  bool read_only = false;   // never creates the database, never writes; an absent database is an empty table
};

class SmallFileStore : public Store {
 public:
  // `files`: the subject's folder tree (keys relative to the subject's folder; must outlive this store). `db_file`: the database's path (made on the first write).
  SmallFileStore(Store& files, std::filesystem::path db_file, SmallFileOptions opt = {});
  // The table in a database that is already open (a subject package's `subject.sqlite`, whose rows are in the same file): the SAME connection, so a node's rows and its
  // small keys are one transaction, and the file stays one file at rest. The store never opens, closes or switches the journal of this connection; read-only follows it.
  SmallFileStore(Store& files, Database& shared, SmallFileOptions opt = {});
  ~SmallFileStore() override;
  SmallFileStore(const SmallFileStore&) = delete;
  SmallFileStore& operator=(const SmallFileStore&) = delete;

  std::uint64_t threshold() const { return opt_.threshold; }
  const std::filesystem::path& db_file() const { return db_file_; }
  // Whether the database holds the key (the file tree not asked).
  bool in_table(const std::string& key) const;
  // Every key in the table (sorted), for a mover.
  std::vector<std::string> table_keys() const;
  // The row for `key` as it is, without touching a file of the same key (the mover verifies before it removes the file).
  void put_row(const std::string& key, const std::uint8_t* data, std::size_t n);
  // Run `body` in one transaction (the mover's batch); a throw rolls it back. Nested calls join the outer one.
  void batch(const std::function<void()>& body);
  // Close the connection (the WAL checkpointed and its files removed); it opens again on the next call.
  void close();

  std::optional<Bytes> get(const std::string& key) const override;
  std::optional<Bytes> get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const override;
  std::optional<std::uint64_t> size(const std::string& key) const override;
  bool exists(const std::string& key) const override;
  bool exists_prefix(const std::string& prefix) const override;
  Listing list(const std::string& prefix) const override;
  bool is_link(const std::string& prefix) const override;
  void check_inside(const std::string& key) const override;
  using Store::set;
  void set(const std::string& key, const std::uint8_t* data, std::size_t n) override;
  // The value comes into being at `close` (a row) or when it passes the threshold (a file): unlike a FolderStore's, it does not exist from the call.
  std::unique_ptr<StoreWriter> open_writer(const std::string& key) override;
  void erase(const std::string& key) override;
  void erase_prefix(const std::string& prefix) override;
  void rename_prefix(const std::string& from, const std::string& to) override;

 private:
  struct Db;
  Db* db(bool create) const;   // the connection, opened when the file exists (or `create`); null when there is none
  friend class SmallWriter;
  Store* files_;
  std::filesystem::path db_file_;
  SmallFileOptions opt_;
  void* shared_ = nullptr;   // the sqlite3 connection of a database that is not this store's (see the constructor); the handle survives a move of the Database
  mutable std::unique_ptr<Db> db_;
  mutable std::recursive_mutex mu_;
};

// The dataset-level store (and the datastore's, for a program that holds several datasets): `<dataset>/<subject>/<key>`.
class DatastoreStore : public Store {
 public:
  // `format52(dataset)`: whether a dataset's subjects keep their small keys in the table (default: the dataset's database, `meta.store_format`). A subject that
  // already has its small_files.sqlite is read and written through it either way (a dataset half way through `compact`).
  explicit DatastoreStore(std::filesystem::path root, SmallFileOptions opt = {}, std::function<bool(const std::string&)> format52 = nullptr);
  ~DatastoreStore() override;
  const std::filesystem::path& root() const { return root_; }
  // Say whether the dataset's subjects use the table (what the dataset's writer knows, and `compact` changes).
  void set_small_files(const std::string& dataset, bool on);
  // The subject `<dataset>/<subject>`'s tables: made when `create`, null when it has none and `create` is false.
  SmallFileStore* small_files(const std::string& subject_key, bool create);
  // The subject's folder tree on its own.
  FolderStore& files(const std::string& subject_key);
  // Close every subject's database (their WAL files go).
  void close_all();

  std::optional<Bytes> get(const std::string& key) const override;
  std::optional<Bytes> get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const override;
  std::optional<std::uint64_t> size(const std::string& key) const override;
  bool exists(const std::string& key) const override;
  bool exists_prefix(const std::string& prefix) const override;
  Listing list(const std::string& prefix) const override;
  bool is_link(const std::string& prefix) const override;
  void check_inside(const std::string& key) const override;
  using Store::set;
  void set(const std::string& key, const std::uint8_t* data, std::size_t n) override;
  std::unique_ptr<StoreWriter> open_writer(const std::string& key) override;
  void erase(const std::string& key) override;
  void erase_prefix(const std::string& prefix) override;
  void rename_prefix(const std::string& from, const std::string& to) override;

 private:
  struct Sub;
  struct Route { std::string subject, rest; Sub* sub = nullptr; bool routed = false; };
  Route route(const std::string& key, bool create) const;   // routed: three or more segments
  Sub& sub_of(const std::string& subject_key) const;
  bool wants_table(const std::string& dataset) const;
  std::filesystem::path root_;
  SmallFileOptions opt_;
  std::function<bool(const std::string&)> format52_;
  mutable FolderStore base_;
  mutable std::map<std::string, std::unique_ptr<Sub>> subs_;
  mutable std::map<std::string, bool> on_;
  mutable std::recursive_mutex mu_;
};

}  // namespace nxr::io
