#include "nxr/io/selection_measurement.h"
#include "nxr/io/migrate.h"

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <deque>
#include <map>
#include <set>

#include "nxr/io/bundle.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/compact.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/hash.h"
#include "nxr/io/small_file_store.h"
#include "nxr/io/statistic.h"
#include "nxr/io/writer.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {

constexpr const char* kOriginalAside = "dataset.sqlite.format52";

Error refuse(const std::string& m) { return Error(code::invalid_argument, m); }

fs::path absolute_of(const fs::path& p) {
  std::error_code ec;
  fs::path r = fs::weakly_canonical(fs::absolute(p), ec);
  return ec ? fs::absolute(p).lexically_normal() : r;
}

// `path + suffix`, or the first unused `path + suffix-N`: a file kept aside is never overwritten.
fs::path aside_name(const fs::path& path, const std::string& suffix) {
  fs::path a = path.string() + suffix;
  std::error_code ec;
  for (int k = 2; fs::exists(a, ec); ++k) a = path.string() + suffix + "-" + std::to_string(k);
  return a;
}

// This process already holds the dataset folder's writer lock: say so through the lock's own handshake, so the dataset module's functions (which take it) borrow it.
struct BorrowedLock {
  std::string previous;
  bool had = false;
  explicit BorrowedLock(const fs::path& folder) {
    if (const char* e = std::getenv(kWriterLockEnv)) { had = true; previous = e; }
    ::setenv(kWriterLockEnv, (absolute_of(folder / kWriterLockFile).string() + ":" + std::to_string(::getpid())).c_str(), 1);
  }
  ~BorrowedLock() { if (had) ::setenv(kWriterLockEnv, previous.c_str(), 1); else ::unsetenv(kWriterLockEnv); }
};

std::vector<std::string> tables_of(const Database& db, const std::string& schema = "main") {
  std::vector<std::string> out;
  for (const auto& r : db.all("SELECT name FROM " + schema + ".sqlite_master WHERE type = 'table' ORDER BY name")) out.push_back(text(r, "name"));
  return out;
}

std::vector<std::string> columns_of(const Database& db, const std::string& table) {
  std::vector<std::string> out;
  for (const auto& r : db.all("SELECT name FROM pragma_table_info(?) ORDER BY cid", {table})) out.push_back(text(r, "name"));
  return out;
}

std::string quoted_list(const std::vector<std::string>& cols) {
  std::string s;
  for (const auto& c : cols) s += (s.empty() ? "" : ", ") + std::string("\"") + c + "\"";
  return s;
}

bool has(const std::vector<std::string>& v, const std::string& x) { return std::find(v.begin(), v.end(), x) != v.end(); }

// Tables that are not a subject's rows: the database's own, the outbox, the package's bookkeeping, and the participants (their own file).
bool bookkeeping(const std::string& t) {
  return t == "meta" || t == "sync_job" || t == "small_files" || t == "key_digest" || t == "anchor_row" || t == "variable_dictionary" || t == "subject_variable" || t.rfind("sqlite_", 0) == 0;
}

// The rows a subject OWNS, as a condition over the original's table (`?` is the subject's id). Ported from nxr's shared-dataset fragment: the owner tables carry the
// subject's id, the child tables follow their parent, the shared vocabulary is every app-, dataset- and user-scope dictionary. A table with no rule is refused, so a
// schema change cannot be migrated by guess.
std::string ownership(const std::string& table, const std::vector<std::string>& cols) {
  static const std::map<std::string, std::pair<std::string, std::string>> child = {
      {"topology_cell", {"topology_id", "topology"}}, {"coordinate_system", {"space_id", "space"}}, {"geometry_dimension", {"geometry_id", "geometry"}},
      {"selection_element", {"selection_id", "selection"}}, {"selection_extent", {"selection_id", "selection"}}, {"selection_measurement", {"selection_id", "selection"}},
      {"selection_span", {"selection_id", "selection"}}, {"selection_factor", {"selection_id", "selection"}}, {"operator_factor", {"operator_id", "operator"}}};
  if (table == "subject") return "id = ?";
  if (table == "dictionary") return "scope IN ('app', 'dataset', 'user') OR subject_id = ?";
  if (table == "dictionary_entry") return "dictionary_id IN (SELECT id FROM src.dictionary WHERE scope IN ('app', 'dataset', 'user') OR subject_id = ?)";
  if (table == "measure_declaration")   // a declaration goes with the subject whose measurements use the name (one database's declarations are its subjects')
    return "name IN (SELECT m.measure FROM src.selection_measurement m JOIN src.selection s ON s.id = m.selection_id WHERE s.subject_id = ?)";
  if (table == "field_contribution") return "field_id IN (SELECT id FROM src.field WHERE subject_id = ?)";   // the contributor (subject_id) is another subject: the owner is the field's
  if (const auto it = child.find(table); it != child.end()) return it->second.first + " IN (SELECT id FROM src." + it->second.second + " WHERE subject_id = ?)";
  if (has(cols, "subject_id")) return "subject_id = ?";
  throw refuse("the table '" + table + "' has no rule for which subject owns a row: this build cannot split it");
}

// The columns of `table` in a schema ("main", "src", "orig"...), in that table's own order.
std::vector<std::string> columns_in(const Database& db, const std::string& schema, const std::string& table) {
  std::vector<std::string> out;
  for (const auto& r : db.all("SELECT name FROM pragma_table_info(?, ?) ORDER BY cid", {table, schema})) out.push_back(text(r, "name"));
  return out;
}

// Rows are compared BY COLUMN NAME: a database whose table was extended in place (a column added at the end) has its columns in another order than the model's, and the
// same rows. Both tables must have the same columns (a column the model lacks, or one the original lacks, is refused: nothing is dropped or invented).
std::string named_columns(const Database& db, const std::string& table, const std::string& here, const std::string& there, const std::string& what) {
  auto a = columns_in(db, here, table), b = columns_in(db, there, table);
  std::vector<std::string> sa = a, sb = b;
  std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
  if (sa != sb) throw refuse(what + ": the table " + table + " has other columns than the original's; this build cannot move it without changing it");
  return quoted_list(a);
}

// a table of the original has the same rows as the table here, value for value, in this direction
void require_same_rows(const Database& db, const std::string& table, const std::string& here, const std::string& there, const std::string& what) {
  const std::string l = "\"" + table + "\"";
  std::string cols = named_columns(db, table, here, there, what);
  if (table == "subject")   // format 56: a package's subject row names no template; the original's template_id is not compared
    for (std::string::size_type at; (at = cols.find("\"template_id\"")) != std::string::npos;) cols.replace(at, 13, "NULL");
  const auto a = integer(*db.one("SELECT count(*) AS n FROM (SELECT " + cols + " FROM " + here + "." + l + " EXCEPT SELECT " + cols + " FROM " + there + "." + l + ")"), "n");
  if (a) throw refuse(what + ": " + std::to_string(a) + " row(s) of " + table + " are not the original's");
}

struct SubjectOut {
  std::string name, outcome;
  std::int64_t rows = 0, anchor_rows = 0, keys = 0, small_keys_verified = 0;
};

}  // namespace

Json split_dataset(const fs::path& folder_in, const SplitOptions& opt) {
  if (opt.registered_by.empty()) throw refuse("split: say who runs it (registered_by)");
  const fs::path folder = absolute_of(folder_in);
  const fs::path original = folder / kDatasetFile;
  Json report = Json::object();
  if (!fs::is_regular_file(original)) {
    if (fs::is_regular_file(folder / kCatalogueFile) && fs::exists(folder / kOriginalAside)) { report["already_split"] = true; return report; }
    throw Error(code::not_found, folder.string() + ": no " + kDatasetFile + " here: not a dataset of formats 50 to 52");
  }
  {   // the selections are converted first: a package made from unconverted rows would carry the new marker over the old form
    Database db = Database::open(original, false, "", true);
    const auto m = db.one("SELECT value FROM meta WHERE key = 'selection_storage'");
    if (!m || text(*m, "value") != kSelectionStorage) throw refuse("split: convert the selections first (migrate_selection_storage: `nxr-neuro migrate selections --dataset <folder>`)");
  }
  if (opt.dry_run) {
    Database db = Database::open(original, false, "", true);
    Json subjects = Json::array();
    for (const auto& s : db.all("SELECT name, kind FROM subject ORDER BY name")) subjects.push_back(Json{{"name", text(s, "name")}, {"kind", text(s, "kind")}});
    return Json{{"dry_run", true}, {"store_format", store_format_number(db)}, {"needs_compact", store_format_number(db) < 52}, {"subjects", subjects}};
  }

  DatasetWriter ds = open_writer(folder);   // the dataset's writer lock, taken before anything is recovered or written
  BorrowedLock borrowed(folder);
  if (store_format_number(ds.db()) < 52) {
    compact_dataset(ds, CompactOptions{});
    if (store_format_number(ds.db()) < 52) throw refuse("split: the dataset could not be brought to format 52 (see `dataset compact`): " + folder.string());
  }
  const Row dataset_row = ds.row();
  const auto datastore_row = ds.db().read("datastore", text(dataset_row, "datastore_id"));
  if (!datastore_row) throw refuse("split: the dataset's datastore row is missing");
  const std::string dataset_id = text(dataset_row, "id");

  // the subjects: templates first, then by name
  const auto subjects = ds.db().all("SELECT * FROM subject WHERE dataset_id = ? ORDER BY (kind = 'template') DESC, created_utc, name", {dataset_id});
  for (const auto& s : subjects) if (text(s, "status") != "complete") throw refuse("split: subject " + text(s, "name") + " is " + text(s, "status") + ": recover the dataset first");

  if (!fs::exists(folder / kCatalogueFile)) create_catalogue_from(folder, dataset_row, *datastore_row, "nxr-io split", false);
  std::vector<SubjectOut> outs;

  Database src_ro = Database::open(original, false, "", true);
  // format 56: a template keeps its group results (the template holds the cohort); no group package is made, and a subject names no template
  for (const auto& srow : subjects) {
    SubjectOut out;
    const std::string sid = text(srow, "id"), name = text(srow, "name"), kind = text(srow, "kind");
    out.name = name;
    const fs::path sdir = folder / text(srow, "path");
    if (!fs::is_directory(sdir)) throw Error(code::not_found, "split: " + name + ": its store folder is missing: " + sdir.string());
    const fs::path pkg_file = sdir / kPackageDatabaseFile, building = sdir / (std::string(kPackageDatabaseFile) + ".building"), old_small = sdir / kSmallFilesFile;
    std::error_code ec;
    fs::remove(building, ec);   // a build a crash left half-made starts again

    const auto entry = Catalogue::open(folder).find(sid);
    bool built = false;
    if (fs::is_regular_file(pkg_file)) {
      // finished earlier up to some stage: it must still be this subject's package, whole
      SubjectPackage p = SubjectPackage::open(sdir);
      const PackageRecord rec = p.record();
      if (rec.subject_id != sid || p.content_hash() != rec.content_hash) throw refuse("split: " + name + ": a package is here that is not this subject's as it was migrated; move it aside and run again");
      out.outcome = entry && entry->content_hash == rec.content_hash && !fs::exists(old_small) ? "already" : "migrated";
    } else {
      built = true;
      out.outcome = "migrated";
      WriterLock sublock = acquire_writer_lock(sdir, "nxr-io split");
      {
        Database pkg = Database::open(building, true, "nxr-io split");
        pkg.execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat53)});
        create_package_tables(pkg);
        pkg.execute("PRAGMA foreign_keys = OFF");
        pkg.execute("ATTACH DATABASE ? AS src", {original.string()});
        const std::vector<std::string> model_tables = tables_of(pkg);
        pkg.tx([&] {
          for (const char* t : {"datastore", "dataset"}) {
            const std::string cl = named_columns(pkg, t, "main", "src", "split");
            pkg.execute(std::string("INSERT INTO main.") + t + " (" + cl + ") SELECT " + cl + " FROM src." + t + " WHERE id = ?", {std::string(t) == "datastore" ? text(dataset_row, "datastore_id") : dataset_id});
          }
          // the rows it owns, parents before children
          static const char* const first[] = {"subject", "acquisition", "topology", "topology_cell", "space", "coordinate_system", "geometry", "geometry_dimension", "manifold",
                                              "dictionary", "dictionary_entry", "selection", "selection_element", "selection_extent", "selection_measurement", "selection_span", "selection_factor",
                                              "field", "field_contribution", "operator", "operator_factor", "statistic", "job"};
          std::vector<std::string> order(std::begin(first), std::end(first));
          for (const auto& t : model_tables) if (!bookkeeping(t) && t != "dataset" && t != "datastore" && !has(order, t)) order.push_back(t);
          const auto src_tables = tables_of(pkg, "src");
          for (const auto& t : order) {
            if (!has(model_tables, t) || !has(src_tables, t)) continue;   // a table the original lacks (a database from before it)
            const auto cols = columns_of(pkg, t);
            const std::string cond = ownership(t, cols);
            std::vector<Value> args;
            for (char c : cond) if (c == '?') args.push_back(sid);
            std::string select = quoted_list(cols);
            if (t == "subject")   // format 56: a subject names no template
              for (std::string::size_type at; (at = select.find("\"template_id\"")) != std::string::npos;) select.replace(at, 13, "NULL");
            pkg.execute("INSERT INTO main.\"" + t + "\" (" + quoted_list(cols) + ") SELECT " + select + " FROM src.\"" + t + "\" WHERE " + cond, args);
          }
        });
        // the anchor rows: what its rows point at that another subject owns (the template's manifolds, a group's members), with their parents and what a bundle follows
        out.anchor_rows = add_anchor_rows(pkg, original);
        pkg.execute("DELETE FROM main.sync_job");
        // the subject's small keys, as rows of the package's own table
        pkg.execute("CREATE TABLE IF NOT EXISTS small_files (key TEXT PRIMARY KEY, value BLOB NOT NULL)");
        if (fs::is_regular_file(old_small)) {
          pkg.execute("ATTACH DATABASE ? AS sf", {old_small.string()});
          pkg.tx([&] { pkg.execute("INSERT INTO main.small_files (key, value) SELECT key, value FROM sf.small_files WHERE " + std::string("1")); });
          pkg.execute("DETACH DATABASE sf");
        }
        // verification, before anything is marked done: the foreign keys hold, every row is the original's value for value, and the bundle is the original's
        if (!pkg.all("PRAGMA foreign_key_check").empty()) throw refuse("split: " + name + ": a row of the package names a row it does not hold (foreign key check)");
        for (const auto& t : tables_of(pkg)) if (!bookkeeping(t) && has(tables_of(pkg, "src"), t)) require_same_rows(pkg, t, "main", "src", "split: " + name);
        for (const auto& t : model_tables) if (!bookkeeping(t) && !has(tables_of(pkg, "src"), t) && integer(*pkg.one("SELECT count(*) AS n FROM main.\"" + t + "\""), "n") != 0) throw refuse("split: " + name + ": rows of " + t + ", which the original lacks");
        out.rows = 0;
        for (const auto& t : model_tables) if (!bookkeeping(t)) out.rows += integer(*pkg.one("SELECT count(*) AS n FROM main.\"" + t + "\""), "n");
        {
          auto a = subject_bundle(pkg, sid), b = subject_bundle(src_ro, sid);
          for (auto* x : {&a, &b}) if (*x && x->value().contains("subject") && (*x)->at("subject").is_object()) (**x)["subject"].erase("template_id");   // format 56: not compared
          if (!a || !b || *a != *b) throw refuse("split: " + name + ": the bundle read from the package is not the bundle read from the original database");
        }
        pkg.execute("DETACH DATABASE src");
        // the keys: digested from what is there now, and the small ones compared with the old table byte for byte
        {
          FolderStore files(sdir);
          SmallFileStore small(files, pkg, SmallFileOptions{});
          rebuild_key_digests(pkg, small);
          out.keys = integer(*pkg.one("SELECT count(*) AS n FROM key_digest"), "n");
          if (fs::is_regular_file(old_small)) {
            SmallFileStore old(files, old_small, SmallFileOptions{kSmallFileThreshold, true});
            auto before = list_all_keys(old);
            const auto after = list_all_keys(small);
            if (before != after) throw refuse("split: " + name + ": the package's keys are not the subject's keys");
            for (const auto& k : old.table_keys()) {
              const auto x = old.get(k), y = small.get(k);
              if (!x || !y || *x != *y) throw refuse("split: " + name + ": the key " + k + " differs after the move");
              ++out.small_keys_verified;
            }
          }
          small.close();
        }
        // the record, and the content hash
        PackageRecord rec;
        rec.subject_id = sid; rec.name = name; rec.kind = kind;
        rec.schema_version = schema_version(); rec.store_format = kStoreFormatPackage;
        write_package_record(pkg, rec);
        refresh_package_record(pkg);
      }
      sublock.release();
      if (opt.on_stage) opt.on_stage("built", name);
      if (opt.on_stage) opt.on_stage("verified", name);
      // the commit of the package: the new database in place, the old table kept aside
      fs::rename(building, pkg_file);
      if (fs::exists(old_small)) fs::rename(old_small, aside_name(old_small, ".format52"));
      for (const char* extra : {"small_files.sqlite-wal", "small_files.sqlite-shm", "small_files.sqlite-journal"}) fs::remove(sdir / extra, ec);
      if (opt.on_stage) opt.on_stage("committed", name);
    }
    if (!built && fs::exists(old_small)) fs::rename(old_small, aside_name(old_small, ".format52"));   // a crash between the two renames

    RegisterOptions ro;
    ro.registered_by = opt.registered_by;
    const auto res = register_packages(folder, {sdir}, ro);
    if (res.empty() || (res[0].outcome != "registered" && res[0].outcome != "unchanged" && res[0].outcome != "updated"))
      throw refuse("split: " + name + ": the package was not registered: " + (res.empty() ? std::string() : res[0].reason));
    if (opt.on_stage) opt.on_stage("registered", name);
    if (out.outcome == "migrated" && res[0].outcome == "unchanged" && !built) out.outcome = "already";
    outs.push_back(out);
  }

  // participants, if the original ever held any: their own file
  {
    const auto n = integer(*src_ro.one("SELECT count(*) AS n FROM variable_dictionary"), "n");
    if (n > 0 && !fs::exists(folder / kParticipantsFile)) {
      Database p = Database::open_plain(folder / (std::string(kParticipantsFile) + ".building"), true);
      p.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT");
      p.execute("CREATE TABLE variable_dictionary (id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, name TEXT NOT NULL, type TEXT NOT NULL CHECK (type IN ('number', 'string', 'boolean')), unit TEXT, description TEXT, levels_json TEXT, UNIQUE (dataset_id, name)) STRICT");
      p.execute("CREATE TABLE subject_variable (subject_id TEXT NOT NULL, variable_dictionary_id TEXT NOT NULL REFERENCES variable_dictionary (id) ON DELETE CASCADE, value_num REAL, value_text TEXT, PRIMARY KEY (subject_id, variable_dictionary_id), CHECK ((value_num IS NULL) <> (value_text IS NULL))) STRICT");
      p.execute("INSERT INTO meta (key, value) VALUES ('participants_version', '1')");
      p.execute("ATTACH DATABASE ? AS src", {original.string()});
      p.tx([&] {
        p.execute("INSERT INTO variable_dictionary SELECT * FROM src.variable_dictionary");
        p.execute("INSERT INTO subject_variable SELECT * FROM src.subject_variable");
      });
      p.execute("DETACH DATABASE src");
      p.close();
      fs::rename(folder / (std::string(kParticipantsFile) + ".building"), folder / kParticipantsFile);
    }
  }

  // the whole: the index from the packages against the original, table by table, row for row
  build_index(folder, true);
  {
    Database idx = Database::open(folder / kIndexFile, false, "", true);
    idx.execute("ATTACH DATABASE ? AS orig", {original.string()});
    for (const auto& t : tables_of(idx)) {
      if (bookkeeping(t)) continue;
      if (!has(tables_of(idx, "orig"), t)) continue;
      const std::string l = "\"" + t + "\"";
      std::string named = named_columns(idx, t, "main", "orig", "split: the index built from the packages");
      if (t == "subject")   // format 56: a package's subject row names no template; the original's template_id is not compared
        for (std::string::size_type at; (at = named.find("\"template_id\"")) != std::string::npos;) named.replace(at, 13, "NULL");
      const std::string here = "SELECT " + named + " FROM main." + l;
      const auto extra = integer(*idx.one("SELECT count(*) AS n FROM (" + here + " EXCEPT SELECT " + named + " FROM orig." + l + ")"), "n");
      if (extra) throw refuse("split: the index built from the packages: " + std::to_string(extra) + " row(s) of " + t + " are not the original's");
      const auto missing = integer(*idx.one("SELECT count(*) AS n FROM (SELECT " + named + " FROM orig." + l + " EXCEPT " + here + ")"), "n");
      if (missing) throw refuse("split: " + std::to_string(missing) + " row(s) of " + t + " in the original database are in no package (nothing was renamed; the packages are kept)");
      // every row exactly once: the same number of rows, none of them twice
      const auto have = integer(*idx.one("SELECT count(*) AS n FROM (" + here + ")"), "n"), want = integer(*idx.one("SELECT count(*) AS n FROM orig." + l), "n");
      if (have != want) throw refuse("split: " + t + " has " + std::to_string(have) + " row(s) in the packages and " + std::to_string(want) + " in the original");
    }
    idx.execute("DETACH DATABASE orig");
  }

  // the end: the original is kept, renamed
  src_ro.close();
  Json listed = Json::array();
  for (const auto& o : outs) listed.push_back(Json{{"name", o.name}, {"outcome", o.outcome}, {"rows", o.rows}, {"anchor_rows", o.anchor_rows}, {"keys", o.keys}, {"small_keys_verified", o.small_keys_verified}});
  report["subjects"] = listed;
  ds.close();
  fs::rename(original, aside_name(original, ".format52"));
  report["original_kept_as"] = kOriginalAside;
  return report;
}

}  // namespace nxr::io

// ══ join ═════════════════════════════════════════════════════════════════════════════════════════════════════════

namespace nxr::io {

Json join_dataset(const fs::path& folder_in, const JoinOptions& opt) {
  const fs::path folder = absolute_of(folder_in);
  const fs::path target = folder / kDatasetFile, aside = folder / kOriginalAside;
  Json report = Json::object();
  if (!fs::is_regular_file(folder / kCatalogueFile)) {
    if (fs::is_regular_file(target)) { report["already_joined"] = true; return report; }
    throw Error(code::not_found, folder.string() + ": no catalogue here: not a dataset of format 53");
  }
  WriterLock lock = acquire_writer_lock(folder, "nxr-io join");
  BorrowedLock borrowed(folder);
  Catalogue cat = Catalogue::open(folder);

  const auto members = [&] { std::vector<CatalogueEntry> m; for (const auto& e : cat.entries()) if (e.status != "replaced") m.push_back(e); return m; }();
  for (const auto& e : members) {
    const fs::path p = cat.package_path(e);
    if (fs::path(e.path).is_absolute()) throw refuse("join: " + e.name + " is a package outside the dataset folder (" + p.string() + "): it cannot be joined into this dataset's database");
    if (!fs::is_regular_file(p / kPackageDatabaseFile) && !fs::is_regular_file(p / kSmallFilesFile)) throw Error(code::not_found, "join: " + e.name + ": its package is missing at " + p.string());
  }
  std::error_code ec;

  // 1. the database: the index build, written as the dataset database (committed first: from then on a rerun only finishes the subjects)
  if (!fs::is_regular_file(target)) {
    build_index(folder, true);
    const fs::path tmp = folder / (std::string(kDatasetFile) + ".joining");
    fs::remove(tmp, ec);
    fs::copy_file(folder / kIndexFile, tmp);
    {
      Database db = Database::open(tmp);
      db.execute("PRAGMA foreign_keys = OFF");
      db.tx([&] {
        db.execute("DELETE FROM meta WHERE key = 'index_json'");
        db.execute("DELETE FROM sync_job");
        db.execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat52)});
        // the participants table, back into the model's two tables
        if (fs::is_regular_file(folder / kParticipantsFile)) {
          db.execute("ATTACH DATABASE ? AS part", {(folder / kParticipantsFile).string()});
          db.execute("INSERT INTO variable_dictionary SELECT * FROM part.variable_dictionary");
          db.execute("INSERT INTO subject_variable SELECT * FROM part.subject_variable");
        }
      });
      if (fs::exists(folder / kParticipantsFile)) db.execute("DETACH DATABASE part");
      // the original's own meta (when its copy is kept): what the database said about itself
      if (fs::is_regular_file(aside)) {
        Database old = Database::open(aside, false, "", true);
        db.tx([&] {
          db.execute("DELETE FROM meta");
          for (const auto& m : old.all("SELECT key, value FROM meta")) db.execute("INSERT INTO meta (key, value) VALUES (?, ?)", {text(m, "key"), text(m, "value")});
        });
      }
      if (!db.all("PRAGMA foreign_key_check").empty()) throw refuse("join: the database built from the packages has a row naming a row it does not hold");
    }
    if (opt.on_stage) opt.on_stage("database", "");
    fs::rename(tmp, target);
  }

  // 2. each subject: its small keys back into its small_files.sqlite, byte for byte; its package database kept aside
  Json subjects = Json::array();
  for (const auto& e : members) {
    const fs::path sdir = cat.package_path(e), pkg_file = sdir / kPackageDatabaseFile, small = sdir / kSmallFilesFile;
    if (fs::is_regular_file(pkg_file)) {
      std::int64_t keys = 0;
      {
        Database pkg = Database::open(pkg_file, false, "", true);
        FolderStore files(sdir);
        if (fs::exists(small)) fs::rename(small, aside_name(small, ".format52"));   // an earlier small-file table is not overwritten
        {
          SmallFileStore out(files, small, SmallFileOptions{});
          out.batch([&] {
            for (const auto& r : pkg.all("SELECT key, value FROM small_files ORDER BY key")) {
              const std::string& v = std::get<std::string>(r.at("value"));
              out.put_row(text(r, "key"), reinterpret_cast<const std::uint8_t*>(v.data()), v.size());
              ++keys;
            }
          });
          // verified byte for byte against the package's table
          SmallFileStore in(files, pkg, SmallFileOptions{kSmallFileThreshold, true});
          for (const auto& k : in.table_keys()) {
            const auto x = in.get(k), y = out.get(k);
            if (!x || !y || *x != *y) throw refuse("join: " + e.name + ": the key " + k + " differs after the move");
          }
          auto expected = in.table_keys();
          if (expected != out.table_keys()) throw refuse("join: " + e.name + ": the keys of the table are not the package's");
          out.close();
        }
      }
      if (opt.on_stage) opt.on_stage("keys", e.name);
      fs::rename(pkg_file, aside_name(pkg_file, ".format53"));
      if (opt.on_stage) opt.on_stage("aside", e.name);
      subjects.push_back(Json{{"name", e.name}, {"small_keys", keys}});
    }
  }

  // 3. the dataset module's files, kept aside; the index is a cache and goes
  for (const char* f : {kCatalogueFile, kParticipantsFile}) if (fs::is_regular_file(folder / f)) fs::rename(folder / f, aside_name(folder / f, ".format53"));
  fs::remove(folder / kIndexFile, ec);
  report["subjects"] = subjects;
  report["database"] = kDatasetFile;
  return report;
}

// ══ format 56: the template holds the cohort ══════════════════════════════════════════════════════════════════════

namespace {
// FORMAT 56 FOR A ONE-DATABASE DATASET (`dataset.sqlite`): every subject's `template_id` cleared, and each subject of kind `group` FOLDED INTO THE TEMPLATE: its fields
// re-owned by the template, their arrays moved from the group's folder to the template's (same paths; refused when one is there already), the group's row removed and
// its emptied folder kept aside (`<folder>.format55-group`), never deleted. A group that owns anything but fields is refused, naming the table. Then the database is
// opened to write (55 -> 56 in place).
Json template_links_one_database(const fs::path& folder, const TemplateLinksOptions& opt) {
  const fs::path file = folder / kDatasetFile;
  if (!fs::is_regular_file(file)) throw refuse(folder.string() + ": neither a catalogue nor a one-database dataset");
  Json report = Json{{"dataset", folder.string()}, {"form", "one database"}, {"dry_run", opt.dry_run}};
  struct Fold { std::string id, name, path; std::vector<std::pair<std::string, std::string>> fields; };   // field id, its array path
  std::vector<Fold> folds;
  std::string template_id, template_path;
  {
    Database d = Database::open_plain(file, false, true);
    const auto templates = d.all("SELECT id, path FROM subject WHERE kind = 'template'");
    const auto groups = d.all("SELECT id, name, path FROM subject WHERE kind = 'group'");
    report["template_id_set"] = integer(*d.one("SELECT count(*) AS n FROM subject WHERE template_id IS NOT NULL"), "n");
    if (!groups.empty()) {
      if (templates.size() != 1) throw refuse(folder.string() + ": " + std::to_string(templates.size()) + " template subjects; a group subject is folded into the dataset's ONE template");
      template_id = text(templates.front(), "id"); template_path = text(templates.front(), "path");
    }
    const auto tables = tables_of(d);
    for (const auto& g : groups) {
      Fold f{text(g, "id"), text(g, "name"), text(g, "path"), {}};
      for (const auto& t : tables) {
        if (t == "subject" || t == "field" || t == "field_contribution" || bookkeeping(t)) continue;   // a contribution's subject_id is its contributor
        if (!has(columns_of(d, t), "subject_id")) continue;
        if (integer(*d.one("SELECT count(*) AS n FROM \"" + t + "\" WHERE subject_id = ?", {f.id}), "n") > 0)
          throw refuse(folder.string() + ": the group subject " + f.name + " owns rows of " + t + "; only fields are folded into the template");
      }
      for (const auto& r : d.all("SELECT id, path FROM field WHERE subject_id = ?", {f.id})) {
        const std::string P = opt_text(r, "path").value_or("");
        if (!P.empty() && fs::exists(folder / template_path / P)) throw refuse(folder.string() + ": " + (folder / template_path / P).string() + " exists: the group's array " + P + " would overwrite it");
        f.fields.emplace_back(text(r, "id"), P);
      }
      folds.push_back(f);
    }
  }
  Json groups_json = Json::array();
  for (const auto& f : folds) groups_json.push_back(Json{{"group", f.name}, {"fields", static_cast<std::int64_t>(f.fields.size())}, {"into", template_path}});
  report["groups_folded"] = groups_json;
  if (opt.dry_run) return report;
  // the arrays first, COPIED key by key through each subject's store (plain files, or its small_files.sqlite) and compared byte for byte; on a failure of the rows the
  // copies are erased. The group's originals stay where they are, in its folder, which is then kept aside whole.
  struct SubjectStore {
    FolderStore files;
    std::unique_ptr<SmallFileStore> small;
    explicit SubjectStore(const fs::path& dir) : files(dir) {
      if (fs::is_regular_file(dir / kSmallFilesFile)) small = std::make_unique<SmallFileStore>(files, dir / kSmallFilesFile);
    }
    Store& store() { return small ? static_cast<Store&>(*small) : static_cast<Store&>(files); }
  };
  std::vector<std::string> copied;
  std::int64_t moved = 0;
  SubjectStore to(folder / template_path);
  try {
    for (const auto& f : folds) {
      SubjectStore from(folder / f.path);
      for (const auto& [id, P] : f.fields) {
        if (P.empty()) continue;
        const auto keys = list_all_keys(from.store(), P);
        for (const auto& k : keys) {
          const auto bytes = from.store().get(k);
          if (!bytes) throw refuse(folder.string() + ": the group's key " + k + " cannot be read");
          to.store().set(k, bytes->data(), bytes->size());
          copied.push_back(k);
          const auto back = to.store().get(k);
          if (!back || *back != *bytes) throw refuse(folder.string() + ": the copy of " + k + " in the template differs from the group's");
        }
        if (!keys.empty()) ++moved;
      }
      if (from.small) from.small->close();
    }
    Database d = Database::open_plain(file);
    d.execute("PRAGMA foreign_keys = OFF");
    d.execute("BEGIN");
    try {
      for (const auto& f : folds) {
        d.execute("UPDATE field SET subject_id = ? WHERE subject_id = ?", {template_id, f.id});
        d.execute("DELETE FROM subject WHERE id = ?", {f.id});
      }
      d.execute("UPDATE subject SET template_id = NULL");
      if (const auto bad = d.all("PRAGMA foreign_key_check"); !bad.empty()) throw refuse(folder.string() + ": a row of " + text(bad.front(), "table") + " no longer finds the row it names");
      d.execute("COMMIT");
    } catch (...) { try { d.execute("ROLLBACK"); } catch (...) {} throw; }
  } catch (...) {
    for (const auto& k : copied) { try { to.store().erase(k); } catch (...) {} }
    throw;
  }
  if (to.small) to.small->close();
  for (const auto& f : folds) if (fs::exists(folder / f.path)) fs::rename(folder / f.path, aside_name(folder / f.path, ".format55-group"));
  { Database d = Database::open(file); }   // 55 -> 56 in place (the schema's CHECKs, the views)
  report["moved_arrays"] = moved;
  return report;
}
}  // namespace

Json migrate_template_links(const fs::path& folder, const TemplateLinksOptions& opt) {
  if (opt.registered_by.empty() && !opt.dry_run) throw refuse("migrate template-links: say who registers the packages again (registered_by)");
  if (!is_catalogue_dataset(folder)) return template_links_one_database(folder, opt);
  const fs::path cfile = folder / kCatalogueFile;
  struct Member { std::string id, name, kind; fs::path package; };
  std::vector<Member> members;
  std::set<std::string> template_ids;
  {
    Database c = Database::open_plain(cfile, false, true);
    for (const auto& r : c.all("SELECT s.id, s.name, s.kind, s.path FROM subject s JOIN subject_entry e ON e.subject_id = s.id WHERE e.status <> 'replaced' ORDER BY s.name")) {
      const fs::path p(text(r, "path"));
      members.push_back(Member{text(r, "id"), text(r, "name"), text(r, "kind"), p.is_absolute() ? p : folder / p});
      if (text(r, "kind") == "template") template_ids.insert(text(r, "id"));
    }
  }
  std::string groups;
  for (const auto& m : members) if (m.kind == "group") groups += (groups.empty() ? "" : ", ") + m.name;
  if (!groups.empty()) throw refuse("migrate template-links: the dataset holds group packages (" + groups + "): a cohort's results belong in the dataset's template (format 56); move them there first");
  Json report = Json{{"dataset", folder.string()}, {"dry_run", opt.dry_run}};
  Json subjects = Json::array();
  // each subject package: its rows' link to the template undone, in one transaction, verified by the foreign keys before it commits
  for (const auto& m : members) {
    Json one = Json{{"name", m.name}, {"kind", m.kind}};
    Database p = Database::open_plain(m.package / kPackageDatabaseFile, false, opt.dry_run);
    for (const auto& r : p.all("SELECT id FROM subject WHERE kind = 'template'")) template_ids.insert(text(r, "id"));
    std::string in_templates;
    for (const auto& t : template_ids) in_templates += (in_templates.empty() ? "" : ", ") + std::string("'") + t + "'";
    if (in_templates.empty()) in_templates = "''";
    const auto anchors = p.all("SELECT table_name, row_id FROM anchor_row WHERE from_subject_id IN (" + in_templates + ")");
    const std::int64_t linked = integer(*p.one("SELECT count(*) AS n FROM subject WHERE template_id IS NOT NULL"), "n");
    one["template_id_set"] = linked;
    one["anchor_rows_from_template"] = static_cast<std::int64_t>(anchors.size());
    if (!opt.dry_run && (linked > 0 || !anchors.empty() || read_package_record(p).template_reference.is_object())) {
      p.execute("PRAGMA foreign_keys = OFF");
      p.execute("BEGIN");
      try {
        p.execute("UPDATE subject SET template_id = NULL");
        for (const auto& a : anchors) {
          p.execute("DELETE FROM \"" + text(a, "table_name") + "\" WHERE id = ?", {text(a, "row_id")});
          p.execute("DELETE FROM anchor_row WHERE table_name = ? AND row_id = ?", {text(a, "table_name"), text(a, "row_id")});
        }
        if (const auto bad = p.all("PRAGMA foreign_key_check"); !bad.empty())
          throw refuse("migrate template-links: " + m.name + ": a row of " + text(bad.front(), "table") + " still points at a row it carried from the template; this package is left as it was");
        PackageRecord rec = read_package_record(p);
        rec.template_reference = nullptr;
        write_package_record(p, rec);
        p.execute("COMMIT");
      } catch (...) { try { p.execute("ROLLBACK"); } catch (...) {} throw; }
    }
    p.close();
    subjects.push_back(one);
  }
  report["subjects"] = subjects;
  if (opt.dry_run) return report;
  // the catalogue's copies of the subject rows
  {
    Database c = Database::open_plain(cfile);
    c.execute("UPDATE subject SET template_id = NULL");
  }
  // every database to 56 in place (Database::open's step), each package's content hash recomputed by its writer, the packages registered again, the index rebuilt
  std::vector<fs::path> packages;
  for (const auto& m : members) {
    DatasetWriter w = open_package_writer(m.package);
    w.close();
    packages.push_back(m.package);
  }
  RegisterOptions ro;
  ro.registered_by = opt.registered_by;
  Json registered = Json::array();
  for (const auto& r : register_packages(folder, packages, ro)) {
    if (r.outcome == "refused") throw refuse("migrate template-links: " + r.package.string() + " was not registered again: " + r.reason);
    registered.push_back(Json{{"package", r.package.string()}, {"outcome", r.outcome}});
  }
  report["registered"] = registered;
  build_index(folder, true);
  return report;
}

}  // namespace nxr::io
