// nxr-io — LEVEL 2 of the record API, the dataset's side (docs/specs/2026-11-17-io-record-api.md, section 0): what the command line asks of a whole one-database
// dataset (its subjects, the arrays its rows name, recovery, removal) without a row struct, a table name or a database handle. io answers inside.
//
//   task::Dataset d = task::Dataset::of(ds);                      // a writer's dataset
//   for (const SubjectEntry& s : d.subjects()) ...                // by name
//   task::Dataset r = task::Dataset::open_readonly(folder);       // a dataset folder, read only
#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/task/task.h"

namespace nxr::io::task {

// One subject of the dataset.
struct SubjectEntry {
  std::string id, name, kind, status, path;
};

// How many complete subjects of one kind.
struct KindCount {
  std::string kind;
  std::int64_t n = 0;
};

// An array a row of the dataset names: the record class that names it (field, selection, operator, geometry, topology_cell, selection_measurement), its subject's name
// and store path, and its own path inside the subject.
struct StoredArray {
  std::string record, subject_name, subject_path, path;
};

// What `recover` did: pending writes rolled back, deletions finished.
struct Healed {
  int rolled_back = 0, finished = 0;
};

class Dataset {
 public:
  static Dataset of(DatasetWriter& ds);                                  // a writer's dataset (or a package's): every call
  static Dataset of(const Database& db);                                 // reading only: no write call is made on it
  static Dataset open_readonly(const std::filesystem::path& folder);    // a dataset folder's database, opened read only (io's Dataset::open)

  // ---- subjects
  std::vector<SubjectEntry> subjects() const;                            // every subject, any status, by name
  std::vector<KindCount> subject_kinds() const;                          // the complete subjects counted by kind, by kind
  std::optional<SubjectEntry> subject_named(const std::string& name) const;
  std::optional<SubjectEntry> find_subject(const std::string& name_or_path) const;   // the first subject with that name or that store path

  // ---- arrays: every array the rows name, by record class (in the order above), each class in the database's order
  std::vector<StoredArray> stored_arrays() const;

  // ---- a writer's dataset only
  Healed recover() const;                                                // every pending row rolled back, every deleting row finished
  void drain() const;                                                    // the deferred deletions carried out
  void remove_subject(const std::string& subject_id) const;              // the subject with all it holds
  void remove(Cls cls, const std::string& id) const;                     // one node with what depends on it
  std::string package_subject_id() const;                                // a subject package's subject (its package record)

 private:
  struct Impl;
  Dataset() = default;
  std::shared_ptr<Impl> impl_;
};

}  // namespace nxr::io::task
