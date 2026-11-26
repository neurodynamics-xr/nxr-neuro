// nxr-io — the task-level API, the dataset's side (io/include/nxr/io/task/dataset.h): the command line's questions about a whole dataset, answered here.
#include "nxr/io/task/dataset.h"

#include "nxr/io/database.h"
#include "nxr/io/dataset.h"
#include "nxr/io/selection_measurement.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::io::task {

struct Dataset::Impl {
  const Database* db = nullptr;                    // a writer's or a caller's database
  DatasetWriter* writer = nullptr;                 // set: the write calls are allowed
  std::unique_ptr<nxr::io::Dataset> readonly;      // open_readonly: the database it owns
  std::optional<std::string> dataset_id;           // a writer's: its subjects only

  std::vector<Row> all(const std::string& sql, const std::vector<Value>& params = {}) const { return readonly ? readonly->query(sql, params) : db->all(sql, params); }
  DatasetWriter& w(const char* what) const {
    if (!writer) throw Error(code::not_writable, std::string(what) + ": this Dataset was opened for reading");
    return *writer;
  }
  // `WHERE` on the subject table: the writer's dataset, or every subject of the database
  std::string subjects_where(std::vector<Value>& params, const std::string& extra = "") const {
    std::string w;
    if (dataset_id) { w = " WHERE dataset_id = ?"; params.push_back(V(*dataset_id)); }
    if (!extra.empty()) w += (w.empty() ? " WHERE " : " AND ") + extra;
    return w;
  }
};

namespace {
SubjectEntry entry_of(const Row& r) {
  return SubjectEntry{text(r, "id"), text(r, "name"), text(r, "kind"), text(r, "status"), text(r, "path")};
}
}  // namespace

Dataset Dataset::of(DatasetWriter& ds) {
  Dataset d;
  d.impl_ = std::make_shared<Impl>();
  d.impl_->db = &ds.db();
  d.impl_->writer = &ds;
  d.impl_->dataset_id = ds.id();
  return d;
}

Dataset Dataset::of(const Database& db) {
  Dataset d;
  d.impl_ = std::make_shared<Impl>();
  d.impl_->db = &db;
  return d;
}

Dataset Dataset::open_readonly(const std::filesystem::path& folder) {
  Dataset d;
  d.impl_ = std::make_shared<Impl>();
  d.impl_->readonly = std::make_unique<nxr::io::Dataset>(nxr::io::Dataset::open(folder));
  return d;
}

std::vector<SubjectEntry> Dataset::subjects() const {
  std::vector<Value> p;
  const std::string where = impl_->subjects_where(p);
  std::vector<SubjectEntry> out;
  for (const Row& r : impl_->all("SELECT id, name, kind, status, path FROM subject" + where + " ORDER BY name", p)) out.push_back(entry_of(r));
  return out;
}

std::vector<KindCount> Dataset::subject_kinds() const {
  std::vector<Value> p;
  const std::string where = impl_->subjects_where(p, "status = 'complete'");
  std::vector<KindCount> out;
  for (const Row& r : impl_->all("SELECT kind, count(*) AS n FROM subject" + where + " GROUP BY kind ORDER BY kind", p)) out.push_back(KindCount{text(r, "kind"), integer(r, "n")});
  return out;
}

std::optional<SubjectEntry> Dataset::subject_named(const std::string& name) const {
  std::vector<Value> p;
  const std::string where = impl_->subjects_where(p, "name = ?");
  p.push_back(V(name));
  const auto rows = impl_->all("SELECT id, name, kind, status, path FROM subject" + where, p);
  if (rows.empty()) return std::nullopt;
  return entry_of(rows[0]);
}

std::optional<SubjectEntry> Dataset::find_subject(const std::string& name_or_path) const {
  std::vector<Value> p;
  const std::string where = impl_->subjects_where(p, "(name = ? OR path = ?)");
  p.push_back(V(name_or_path));
  p.push_back(V(name_or_path));
  const auto rows = impl_->all("SELECT id, name, kind, status, path FROM subject" + where, p);
  if (rows.empty()) return std::nullopt;
  return entry_of(rows[0]);
}

std::vector<StoredArray> Dataset::stored_arrays() const {
  struct Named { const char* record; const char* sql; };
  static const Named named[] = {
      {"field", "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM field r JOIN subject s ON s.id = r.subject_id WHERE r.path IS NOT NULL"},
      {"operator", "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM operator r JOIN subject s ON s.id = r.subject_id WHERE r.path IS NOT NULL"},
      {"geometry", "SELECT s.path AS sp, s.name AS sn, r.positions_path AS p FROM geometry r JOIN subject s ON s.id = r.subject_id WHERE r.positions_path IS NOT NULL"},
      {"topology_cell", "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM topology_cell r JOIN topology t ON t.id = r.topology_id JOIN subject s ON s.id = t.subject_id WHERE r.path IS NOT NULL"},
  };
  std::vector<StoredArray> out;
  for (const auto& n : named)
    for (const Row& r : impl_->all(n.sql)) out.push_back(StoredArray{n.record, text(r, "sn"), text(r, "sp"), text(r, "p")});
  // the selections' and their measurements' arrays are listed by the selection module
  for (const auto& a : selection::array_paths_in_dataset([&](const std::string& sql) { return impl_->all(sql); })) out.push_back(StoredArray{"selection", a.subject_name, a.subject_path, a.path});
  for (const auto& a : selection::measurement_array_paths_in_dataset([&](const std::string& sql) { return impl_->all(sql); })) out.push_back(StoredArray{"selection_measurement", a.subject_name, a.subject_path, a.path});
  return out;
}

Healed Dataset::recover() const {
  DatasetWriter& ds = impl_->w("recover");
  const Recovered r = nxr::io::recover(ds.db(), ds.store());
  return Healed{r.rolled_back, r.finished};
}

void Dataset::drain() const {
  DatasetWriter& ds = impl_->w("drain");
  nxr::io::drain(ds.db(), ds.store());
}

void Dataset::remove_subject(const std::string& subject_id) const {
  DatasetWriter& ds = impl_->w("remove_subject");
  remove_node(ds.db(), ds.node_store(), "subject", subject_id);
}

void Dataset::remove(Cls cls, const std::string& id) const {
  DatasetWriter& ds = impl_->w("remove");
  remove_node(ds.db(), ds.node_store(), to_string(cls), id);
}

std::string Dataset::package_subject_id() const {
  return read_package_record(*impl_->db).subject_id;
}

}  // namespace nxr::io::task
