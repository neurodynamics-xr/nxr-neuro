// nxr-io — the SUBJECT PACKAGE (store format 53; docs/specs/2026-11-16-subject-centred-datasets.md, sections 2 and 6): everything about ONE subject in one folder,
// `<subject>.nxr.zarr/`, that can be copied, archived, shared and opened on its own.
//
//   <subject>.nxr.zarr/
//     subject.sqlite     the subject's rows (the model's tables, model.sql unchanged) AND its small zarr keys (the `small_files` table of format 52),
//                        in ONE database: a node's rows and its small keys are one transaction (one connection, no ATTACH)
//     .writer.lock       while a writer holds the package (the writer lock's file and rules, unchanged)
//     <node>/.../c.0.0   a large array's shards, one file each (format 51/52's sharding, unchanged)
//
// What is new in `subject.sqlite` besides the model's tables (created here, not in model.sql):
//   key_digest(key, size_bytes, sha256)   the size and SHA-256 of every key's value, written as the value is written (a streamed shard is hashed as it goes);
//   anchor_row(table_name, row_id, from_subject_id)   the rows of ANOTHER package this one's rows point at (a template's), by design not this package's ids;
//   meta `store_format` = 53 and meta `package_json`: the package's record (its subject, versions, the template reference and the CONTENT HASH).
//
// THE CONTENT HASH is the SHA-256 of the lines `<key> <size_bytes> <sha256>` of key_digest in key order, a line `rows`, then one line per row of every table
// (all but `meta`, `sync_job`, `small_files` and `key_digest`, tables in name order, rows in primary-key order): `<table>\t<JSON array of the row's values in the
// table's column order>`. A package rewritten is a different state; the hash is recomputed when a writer closes the package, from key_digest (no shard is re-read),
// and `SubjectPackage::verify` recomputes everything from the bytes.
//
// THE WRITER of a package is a DatasetWriter in package mode (`create_package`, `open_package_writer`): keys are `<home dataset path>/<subject path>/…` as in a
// dataset's store, mapped onto the package's own keys, so SubjectWriter and every pipeline written for a dataset write a package unchanged. One package, one
// writer lock: two jobs on two subjects share no file they write.
//
// A READER opens a package alone (`SubjectPackage::open`): no catalogue, no participants, no index, no other subject's file. It refuses a package at a store format
// or schema version it does not know, by name, and never opens it partly.
#pragma once
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
#include "nxr/io/store.h"
#include "nxr/io/writer.h"

namespace nxr::io {

constexpr const char* kPackageDatabaseFile = "subject.sqlite";
constexpr const char* kPackageSuffix = ".nxr.zarr";

// The package's record (meta `package_json`).
struct PackageRecord {
  std::string subject_id, name, kind;
  int schema_version = 0;
  int store_format = 0;
  Json template_reference;     // null, or {name, edition, version, content_hash, subject_id} of the template this subject is registered to (section 4.3)
  Json members;                // a group package: the subjects it was computed from, [{subject_id, name, content_hash}] (section 5); null otherwise
  std::string content_hash;    // empty until a writer has closed the package
  std::string hashed_utc;      // when it was computed
  Json to_json() const;
  static PackageRecord from_json(const Json& j);
};

struct VerifyReport {
  bool ok = true;
  std::string content_hash;               // recomputed from the bytes
  std::vector<std::string> problems;      // each names a key (or a row set) and what differs
};

// A package opened for reading, alone.
class SubjectPackage {
 public:
  // Refuses a folder that is not a package (no subject.sqlite: `not_found`; a database at store format below 53: `not_a_database`) and one at a version this
  // build does not know (`schema_version`).
  static SubjectPackage open(const std::filesystem::path& folder);
  SubjectPackage(SubjectPackage&&) noexcept;
  SubjectPackage& operator=(SubjectPackage&&) noexcept;
  ~SubjectPackage();

  const std::filesystem::path& folder() const;
  const Database& db() const;
  // The package's keys (`source/cortex/vertices/zarr.json`: what its rows' `path` columns name), read only.
  const Store& store() const;
  PackageRecord record() const;
  // The package's own subject row.
  Row subject_row() const;
  // The subject's bundle, as `subject_bundle` returns it from a dataset database: the same queries, on this package's database.
  std::optional<Json> bundle() const;
  // The content hash from key_digest and the rows (what a writer stores at close), without re-reading any key.
  std::string content_hash() const;
  // Every key's digest recomputed from its bytes, compared with key_digest; the rows' hash compared with the record's. Reports, never repairs.
  VerifyReport verify() const;

 private:
  SubjectPackage() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct CreatePackageOptions {
  std::string home_dataset;                    // the name of the dataset the package is made in (provenance, not membership): required
  std::optional<std::string> home_dataset_id;  // the dataset's id, so the packages of one dataset carry the same home row; made when absent
  std::optional<std::string> datastore_id;     // likewise the hint row's id
  SubjectOptions subject;                      // kind, source and creator of the subject row; `default_template` is ignored (a package has no sibling template)
  std::optional<std::string> built_by;         // the database's meta `built_by` (the program)
};

// CREATE a package at `folder` (which must not hold one) holding the subject `subject_name`, complete; returns the writer, whose lock is held until it closes.
// A refused creation leaves nothing behind.
DatasetWriter create_package(const std::filesystem::path& folder, const std::string& subject_name, const CreatePackageOptions& opt);
// OPEN a package for writing: its lock taken before it recovers (pending rows rolled back, deleting rows finished), as `open_writer` does for a dataset.
DatasetWriter open_package_writer(const std::filesystem::path& folder, const OpenOptions& opt = {});

// The content hash of a package's open database, and the record stored from it (what a writer's close does). Both are for the writer and the migration.
std::string package_content_hash(const Database& db);
PackageRecord read_package_record(const Database& db);
void write_package_record(Database& db, const PackageRecord& record);
// Recompute the record's hash and write it when the package's state differs from the recorded one; true when it wrote.
bool refresh_package_record(Database& db);
// Replace key_digest by the digests of the keys the store holds now (a package built by something that did not digest as it wrote: the migration).
void rebuild_key_digests(Database& db, const Store& keys);
// The package's keys, every one under `prefix` ("" for all), sorted.
std::vector<std::string> list_all_keys(const Store& store, const std::string& prefix = "");

// A row id that two packages both own (section 2.1): UUIDv7 ids do not collide in practice and nothing checks at write time, so registration and the index build do.
struct IdClash {
  std::string table, id;
  std::filesystem::path first, second;
};
// Every id of a table keyed by `id` found in two of the packages, except the home dataset and datastore rows (shared by the packages of one dataset by design) and
// the rows a package lists in anchor_row (another package's own ids by design). Each package is opened read-only; an unknown version is refused.
std::vector<IdClash> find_id_clashes(const std::vector<std::filesystem::path>& packages);

}  // namespace nxr::io
