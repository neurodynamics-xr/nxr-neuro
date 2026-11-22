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
std::string ownership(const std::string& table, const std::vector<std::string>& cols, bool exclude_moved = false) {
  // `exclude_moved`: the group fields that move to a group package (the ids are in the package database's temp table `moved_field`) are not this subject's
  static const std::string kNotMoved = " AND NOT EXISTS (SELECT 1 FROM temp.moved_field m WHERE m.id = ";
  if (exclude_moved) {
    if (table == "field") return "subject_id = ?" + kNotMoved + "field.id)";
    if (table == "field_contribution") return "field_id IN (SELECT id FROM src.field WHERE subject_id = ?)" + kNotMoved + "field_contribution.field_id)";
    if (table == "selection_measurement") return "selection_id IN (SELECT id FROM src.selection WHERE subject_id = ?) AND (of_field_id IS NULL OR NOT EXISTS (SELECT 1 FROM temp.moved_field m WHERE m.id = selection_measurement.of_field_id))";
  }
  static const std::map<std::string, std::pair<std::string, std::string>> child = {
      {"topology_cell", {"topology_id", "topology"}}, {"coordinate_system", {"space_id", "space"}}, {"geometry_dimension", {"geometry_id", "geometry"}},
      {"selection_element", {"selection_id", "selection"}}, {"selection_extent", {"selection_id", "selection"}}, {"selection_measurement", {"selection_id", "selection"}},
      {"selection_span", {"selection_id", "selection"}}, {"selection_factor", {"selection_id", "selection"}}, {"operator_factor", {"operator_id", "operator"}}};
  if (table == "subject") return "id = ?";
  if (table == "dictionary") return "scope IN ('app', 'dataset', 'user') OR subject_id = ?";
  if (table == "dictionary_entry") return "dictionary_id IN (SELECT id FROM src.dictionary WHERE scope IN ('app', 'dataset', 'user') OR subject_id = ?)";
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
  const std::string l = "\"" + table + "\"", cols = named_columns(db, table, here, there, what);
  const auto a = integer(*db.one("SELECT count(*) AS n FROM (SELECT " + cols + " FROM " + here + "." + l + " EXCEPT SELECT " + cols + " FROM " + there + "." + l + ")"), "n");
  if (a) throw refuse(what + ": " + std::to_string(a) + " row(s) of " + table + " are not the original's");
}

// ── group results ────────────────────────────────────────────────────────────────────────────────────────────

std::string sql_quote(const std::string& v) {
  std::string o = "'";
  for (char c : v) { if (c == '\'') o += "''"; else o += c; }
  return o + "'";
}

// A key is the prefix itself or below it.
bool under(const std::string& key, const std::string& prefix) { return key == prefix || (key.size() > prefix.size() && key.compare(0, prefix.size(), prefix) == 0 && key[prefix.size()] == '/'); }
bool under_any(const std::string& key, const std::vector<std::string>& prefixes) { for (const auto& p : prefixes) if (under(key, p)) return true; return false; }
// the SQL condition "the key is under none of the prefixes"
std::string not_under_sql(const std::vector<std::string>& prefixes, const std::string& column = "key") {
  std::string c = "1";
  for (const auto& p : prefixes) c += " AND NOT (" + column + " = " + sql_quote(p) + " OR substr(" + column + ", 1, " + std::to_string(p.size() + 1) + ") = " + sql_quote(p + "/") + ")";
  return c;
}
std::string under_sql(const std::vector<std::string>& prefixes, const std::string& column = "key") {
  std::string c = "0";
  for (const auto& p : prefixes) c += " OR " + column + " = " + sql_quote(p) + " OR substr(" + column + ", 1, " + std::to_string(p.size() + 1) + ") = " + sql_quote(p + "/");
  return "(" + c + ")";
}

std::string file_sha256(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  Sha256 h;
  std::vector<char> buf(1 << 20);
  while (in) { in.read(buf.data(), static_cast<std::streamsize>(buf.size())); h.update(reinterpret_cast<const std::uint8_t*>(buf.data()), static_cast<std::size_t>(in.gcount())); }
  return h.hex();
}

// Copy a folder tree, then compare every file's SHA-256 with its copy's (a file already there with the same bytes is left; one that differs is refused).
std::int64_t copy_tree_verified(const fs::path& from, const fs::path& to, const std::string& what) {
  std::int64_t files = 0;
  for (const auto& e : fs::recursive_directory_iterator(from)) {
    if (!e.is_regular_file()) continue;
    const fs::path dest = to / fs::relative(e.path(), from);
    fs::create_directories(dest.parent_path());
    if (fs::exists(dest)) {
      if (file_sha256(dest) != file_sha256(e.path())) throw refuse(what + ": " + dest.string() + " exists and differs from " + e.path().string());
    } else {
      fs::copy_file(e.path(), dest);
    }
    if (file_sha256(dest) != file_sha256(e.path())) throw refuse(what + ": " + dest.string() + " differs from " + e.path().string() + " after the copy");
    ++files;
  }
  return files;
}

// Every file of `a` is in `b` with the same bytes, and the other way round.
bool same_tree(const fs::path& a, const fs::path& b) {
  std::map<std::string, std::string> x, y;
  for (const auto& e : fs::recursive_directory_iterator(a)) if (e.is_regular_file()) x[fs::relative(e.path(), a).generic_string()] = file_sha256(e.path());
  for (const auto& e : fs::recursive_directory_iterator(b)) if (e.is_regular_file()) y[fs::relative(e.path(), b).generic_string()] = file_sha256(e.path());
  return x == y;
}

// The group results of one template (default subject) subject: the fields, and where their arrays are.
struct MovedGroup {
  std::string template_id, template_name, template_path;
  std::string group_id, group_name, group_path;
  std::vector<std::string> field_ids;
  std::vector<std::string> prefixes;                 // keys under the template's folder: `<node>/maps`, `<node>/time`
  std::int64_t n_fields = 0, n_contributions = 0, n_measurements = 0, files_copied = 0;
};

std::vector<MovedGroup> find_moved_groups(const Database& src, const std::vector<Row>& subjects, const fs::path& folder) {
  std::vector<MovedGroup> out;
  std::set<std::string> names;
  for (const auto& s : subjects) names.insert(text(s, "name"));
  for (const auto& s : subjects) {
    if (text(s, "kind") != "template") continue;
    MovedGroup g;
    g.template_id = text(s, "id"); g.template_name = text(s, "name"); g.template_path = text(s, "path");
    for (const auto& f : src.all("SELECT id FROM field WHERE subject_id = ? AND path IS NULL AND json_extract(producer_json, '$.atlas') IN ('spatial', 'time') ORDER BY id", {g.template_id}))
      g.field_ids.push_back(text(f, "id"));
    if (g.field_ids.empty()) continue;
    g.n_fields = static_cast<std::int64_t>(g.field_ids.size());
    std::set<std::string> prefixes;
    for (const auto& id : g.field_ids) {
      g.n_contributions += integer(*src.one("SELECT count(*) AS n FROM field_contribution WHERE field_id = ?", {id}), "n");
      for (const std::string& path : selection::measurement_array_paths_of_field(src, id)) {
        ++g.n_measurements;
        for (const char* marker : {"/maps/", "/time/"}) {
          const auto at = path.find(marker);
          if (at != std::string::npos) { prefixes.insert(path.substr(0, at + 5)); break; }
        }
      }
      g.n_measurements += 0;
    }
    g.prefixes.assign(prefixes.begin(), prefixes.end());
    std::string name = "group";
    for (int k = 2; names.count(name); ++k) name = "group-" + std::to_string(k);
    names.insert(name);
    g.group_name = name; g.group_path = name + kPackageSuffix;
    g.group_id = uuid7();
    // a group package a crashed run already committed keeps its identity
    for (const auto& e : fs::directory_iterator(folder)) {
      const fs::path pf = e.path() / kPackageDatabaseFile;
      if (!e.is_directory() || !fs::is_regular_file(pf)) continue;
      const PackageRecord rec = SubjectPackage::open(e.path()).record();
      if (rec.kind == "group" && rec.template_reference.is_object() && rec.template_reference.value("subject_id", "") == g.template_id && rec.name.rfind("group", 0) == 0) {
        g.group_id = rec.subject_id; g.group_name = rec.name; g.group_path = e.path().filename().string();
      }
    }
    out.push_back(g);
  }
  return out;
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
    Json groups = Json::array();
    {
      std::vector<Row> srows = db.all("SELECT * FROM subject ORDER BY name");
      for (const auto& g : find_moved_groups(db, srows, folder)) groups.push_back(Json{{"from", g.template_name}, {"to", g.group_name}, {"fields", g.n_fields}, {"keys_under", g.prefixes}});
    }
    return Json{{"dry_run", true}, {"store_format", store_format_number(db)}, {"needs_compact", store_format_number(db) < 52}, {"subjects", subjects}, {"group_results_move", groups}};
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
  const std::string dataset_id = text(dataset_row, "id"), dataset_name = text(dataset_row, "name");

  // the subjects: templates first, then by name
  const auto subjects = ds.db().all("SELECT * FROM subject WHERE dataset_id = ? ORDER BY (kind = 'template') DESC, created_utc, name", {dataset_id});
  for (const auto& s : subjects) if (text(s, "status") != "complete") throw refuse("split: subject " + text(s, "name") + " is " + text(s, "status") + ": recover the dataset first");
  std::string default_template_id;
  for (const auto& s : subjects) if (text(s, "kind") == "template") { default_template_id = text(s, "id"); break; }

  if (!fs::exists(folder / kCatalogueFile)) create_catalogue_from(folder, dataset_row, *datastore_row, "nxr-io split", false);
  std::map<std::string, Json> template_reference_of;   // a template subject's id -> {name, edition, version, content_hash, subject_id}
  std::vector<SubjectOut> outs;

  Database src_ro = Database::open(original, false, "", true);
  std::vector<MovedGroup> groups = find_moved_groups(src_ro, subjects, folder);
  std::map<std::string, std::string> member_name;   // for a group package's record
  for (const auto& srow : subjects) member_name[text(srow, "id")] = text(srow, "name");
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
      MovedGroup* mg = nullptr;
      for (auto& g : groups) if (g.template_id == sid) mg = &g;
      if (mg) {
        // THE GROUP RESULTS' ARRAYS: copied into the group package's folder (every file compared), the originals then moved aside, never deleted; a rerun after a crash
        // here finds the copy made and the originals either still in place (copied again, compared) or already aside
        fs::create_directories(folder / mg->group_path);
        for (const auto& P : mg->prefixes) {
          const fs::path from = sdir / P;
          if (!fs::exists(from)) continue;
          mg->files_copied += copy_tree_verified(from, folder / mg->group_path / P, "split: " + name + ": the group results " + P);
          if (opt.on_stage) opt.on_stage("group-arrays", name);
          const fs::path aside_dir = aside_name(folder / "_moved-aside" / mg->template_path / P, "");
          fs::create_directories(aside_dir.parent_path());
          fs::rename(from, aside_dir);
        }
      }
      WriterLock sublock = acquire_writer_lock(sdir, "nxr-io split");
      {
        Database pkg = Database::open(building, true, "nxr-io split");
        pkg.execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat53)});
        create_package_tables(pkg);
        pkg.execute("PRAGMA foreign_keys = OFF");
        pkg.execute("ATTACH DATABASE ? AS src", {original.string()});
        const std::vector<std::string> model_tables = tables_of(pkg);
        if (mg) {
          pkg.execute("CREATE TEMP TABLE moved_field (id TEXT PRIMARY KEY)");
          pkg.tx([&] { for (const auto& id : mg->field_ids) pkg.execute("INSERT INTO temp.moved_field (id) VALUES (?)", {id}); });
        }
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
            const std::string cond = ownership(t, cols, mg != nullptr);
            std::vector<Value> args;
            for (char c : cond) if (c == '?') args.push_back(sid);
            pkg.execute("INSERT INTO main.\"" + t + "\" (" + quoted_list(cols) + ") SELECT " + quoted_list(cols) + " FROM src.\"" + t + "\" WHERE " + cond, args);
          }
        });
        // the anchor rows: what its rows point at that another subject owns (the template's manifolds, a group's members), with their parents and what a bundle follows
        out.anchor_rows = add_anchor_rows(pkg, original);
        pkg.execute("DELETE FROM main.sync_job");
        // the subject's small keys, as rows of the package's own table
        pkg.execute("CREATE TABLE IF NOT EXISTS small_files (key TEXT PRIMARY KEY, value BLOB NOT NULL)");
        if (fs::is_regular_file(old_small)) {
          pkg.execute("ATTACH DATABASE ? AS sf", {old_small.string()});
          pkg.tx([&] { pkg.execute("INSERT INTO main.small_files (key, value) SELECT key, value FROM sf.small_files WHERE " + (mg ? not_under_sql(mg->prefixes) : std::string("1"))); });
          pkg.execute("DETACH DATABASE sf");
        }
        // verification, before anything is marked done: the foreign keys hold, every row is the original's value for value, and the bundle is the original's
        if (!pkg.all("PRAGMA foreign_key_check").empty()) throw refuse("split: " + name + ": a row of the package names a row it does not hold (foreign key check)");
        for (const auto& t : tables_of(pkg)) if (!bookkeeping(t) && has(tables_of(pkg, "src"), t)) require_same_rows(pkg, t, "main", "src", "split: " + name);
        for (const auto& t : model_tables) if (!bookkeeping(t) && !has(tables_of(pkg, "src"), t) && integer(*pkg.one("SELECT count(*) AS n FROM main.\"" + t + "\""), "n") != 0) throw refuse("split: " + name + ": rows of " + t + ", which the original lacks");
        out.rows = 0;
        for (const auto& t : model_tables) if (!bookkeeping(t)) out.rows += integer(*pkg.one("SELECT count(*) AS n FROM main.\"" + t + "\""), "n");
        if (!mg) {   // (a template that gave its group results to a group package has fewer rows than the original's bundle: its rows are verified one by one above)
          const auto a = subject_bundle(pkg, sid), b = subject_bundle(src_ro, sid);
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
            if (mg) before.erase(std::remove_if(before.begin(), before.end(), [&](const std::string& k) { return under_any(k, mg->prefixes); }), before.end());
            if (before != after) throw refuse("split: " + name + ": the package's keys are not the subject's keys");
            for (const auto& k : old.table_keys()) {
              if (mg && under_any(k, mg->prefixes)) continue;
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
        rec.template_reference = nullptr;
        const std::string tid = opt_text(srow, "template_id").value_or("");
        if (!tid.empty()) {
          const auto ref = template_reference_of.find(tid);
          if (ref == template_reference_of.end()) throw refuse("split: " + name + ": its template " + tid + " is not a template subject of this dataset");
          rec.template_reference = ref->second;
        }
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

    if (kind == "template") {
      const PackageRecord rec = SubjectPackage::open(sdir).record();
      template_reference_of[sid] = Json{{"name", sid == default_template_id ? dataset_name + "-default" : dataset_name + "-" + name}, {"edition", "full"}, {"version", "local"},
                                        {"content_hash", rec.content_hash}, {"subject_id", sid}};
    }
    RegisterOptions ro;
    ro.registered_by = opt.registered_by;
    const auto res = register_packages(folder, {sdir}, ro);
    if (res.empty() || (res[0].outcome != "registered" && res[0].outcome != "unchanged" && res[0].outcome != "updated"))
      throw refuse("split: " + name + ": the package was not registered: " + (res.empty() ? std::string() : res[0].reason));
    if (opt.on_stage) opt.on_stage("registered", name);
    if (out.outcome == "migrated" && res[0].outcome == "unchanged" && !built) out.outcome = "already";
    outs.push_back(out);
  }

  // THE GROUP PACKAGES: the group results the default subjects gave up, owned by a group subject made here (the mapping is in the header of this file)
  Json group_report = Json::array();
  for (MovedGroup& g : groups) {
    const fs::path gdir = folder / g.group_path, gpkg = gdir / kPackageDatabaseFile, gbuilding = gdir / (std::string(kPackageDatabaseFile) + ".building");
    const fs::path tdir = folder / g.template_path;
    SubjectOut gout;
    gout.name = g.group_name;
    gout.outcome = "migrated";
    std::error_code gec;
    fs::remove(gbuilding, gec);
    if (fs::is_regular_file(gpkg)) {
      SubjectPackage p = SubjectPackage::open(gdir);
      if (p.record().subject_id != g.group_id || p.content_hash() != p.record().content_hash) throw refuse("split: " + g.group_name + ": a package is here that is not the group results as they were migrated; move it aside and run again");
      gout.outcome = "already";
    } else {
      fs::create_directories(gdir);
      WriterLock glock = acquire_writer_lock(gdir, "nxr-io split");
      {
        Database pkg = Database::open(gbuilding, true, "nxr-io split");
        pkg.execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat53)});
        create_package_tables(pkg);
        pkg.execute("PRAGMA foreign_keys = OFF");
        pkg.execute("ATTACH DATABASE ? AS src", {original.string()});
        const std::vector<std::string> model_tables = tables_of(pkg);
        pkg.execute("CREATE TEMP TABLE moved_field (id TEXT PRIMARY KEY)");
        pkg.tx([&] { for (const auto& id : g.field_ids) pkg.execute("INSERT INTO temp.moved_field (id) VALUES (?)", {id}); });
        const std::string now = now_utc();
        pkg.tx([&] {
          for (const char* t : {"datastore", "dataset"}) {
            const std::string cl = named_columns(pkg, t, "main", "src", "split");
            pkg.execute(std::string("INSERT INTO main.") + t + " (" + cl + ") SELECT " + cl + " FROM src." + t + " WHERE id = ?", {std::string(t) == "datastore" ? text(dataset_row, "datastore_id") : dataset_id});
          }
          pkg.insert("subject", Row{{"id", g.group_id}, {"dataset_id", dataset_id}, {"name", g.group_name}, {"path", g.group_path}, {"kind", "group"},
                                    {"created_utc", now}, {"created_by", "nxr-io split"}, {"modified_utc", now}, {"modified_by", "nxr-io split"}});
          // the shared vocabulary every package carries a copy of
          for (const char* t : {"dictionary", "dictionary_entry"}) {
            const auto cols = columns_of(pkg, t);
            const std::string cond = ownership(t, cols);
            std::vector<Value> args;
            for (char c : cond) if (c == '?') args.push_back(g.group_id);
            pkg.execute(std::string("INSERT INTO main.\"") + t + "\" (" + quoted_list(cols) + ") SELECT " + quoted_list(cols) + " FROM src.\"" + t + "\" WHERE " + cond, args);
          }
          // the moved rows: the field's owner changes (its `subject_id`), nothing else does
          {
            const auto cols = columns_of(pkg, "field");
            std::string select;
            for (const auto& c : cols) select += (select.empty() ? "" : ", ") + (c == "subject_id" ? sql_quote(g.group_id) + " AS subject_id" : std::string("\"") + c + "\"");
            pkg.execute("INSERT INTO main.field (" + quoted_list(cols) + ") SELECT " + select + " FROM src.field WHERE id IN (SELECT id FROM temp.moved_field)");
          }
          for (const auto& [t, col] : {std::pair<const char*, const char*>{"field_contribution", "field_id"}, {"selection_measurement", "of_field_id"}}) {
            const auto cols = columns_of(pkg, t);
            pkg.execute(std::string("INSERT INTO main.\"") + t + "\" (" + quoted_list(cols) + ") SELECT " + quoted_list(cols) + " FROM src.\"" + t + "\" WHERE " + col + " IN (SELECT id FROM temp.moved_field)");
          }
        });
        gout.anchor_rows = add_anchor_rows(pkg, original);   // the default subject's partition and what it holds, the contributing subjects' rows and the fields they name
        pkg.execute("DELETE FROM main.sync_job");
        // the small keys under the moved prefixes, from the default subject's old table
        pkg.execute("CREATE TABLE IF NOT EXISTS small_files (key TEXT PRIMARY KEY, value BLOB NOT NULL)");
        const fs::path t_small = fs::exists(tdir / kSmallFilesFile) ? tdir / kSmallFilesFile : fs::path((tdir / kSmallFilesFile).string() + ".format52");
        if (fs::is_regular_file(t_small)) {
          pkg.execute("ATTACH DATABASE ? AS sf", {t_small.string()});
          pkg.tx([&] { pkg.execute("INSERT INTO main.small_files (key, value) SELECT key, value FROM sf.small_files WHERE " + under_sql(g.prefixes)); });
          pkg.execute("DETACH DATABASE sf");
        }
        // verification: the foreign keys hold; every row is the original's, the moved fields with their owner changed back; the counts are the original's
        if (!pkg.all("PRAGMA foreign_key_check").empty()) throw refuse("split: " + g.group_name + ": a row of the group package names a row it does not hold (foreign key check)");
        for (const auto& t : {std::string("field_contribution"), std::string("selection_measurement"), std::string("dictionary"), std::string("dictionary_entry")}) require_same_rows(pkg, t, "main", "src", "split: " + g.group_name);
        {
          const auto cols = columns_of(pkg, "field");
          std::string back;
          for (const auto& c : cols) back += (back.empty() ? "" : ", ") + (c == "subject_id" ? sql_quote(g.template_id) + " AS subject_id" : std::string("\"") + c + "\"");
          // (the field rows the package holds as anchors, the members' source fields, are the owners' own and are not looked at here)
          const auto diff = integer(*pkg.one("SELECT count(*) AS n FROM (SELECT " + back + " FROM main.field WHERE id IN (SELECT id FROM temp.moved_field) EXCEPT SELECT " + quoted_list(cols) + " FROM src.field)"), "n");
          if (diff) throw refuse("split: " + g.group_name + ": " + std::to_string(diff) + " group field(s) are not the original's with the owner changed");
          if (integer(*pkg.one("SELECT count(*) AS n FROM main.field WHERE subject_id = ?", {g.group_id}), "n") != g.n_fields) throw refuse("split: " + g.group_name + ": the group package does not own exactly the group fields");
        }
        const auto n_contrib = integer(*pkg.one("SELECT count(*) AS n FROM main.field_contribution WHERE field_id IN (SELECT id FROM temp.moved_field)"), "n"),
                   n_meas = integer(*pkg.one("SELECT count(*) AS n FROM main.selection_measurement WHERE of_field_id IN (SELECT id FROM temp.moved_field)"), "n");
        if (n_contrib != integer(*pkg.one("SELECT count(*) AS n FROM src.field_contribution WHERE field_id IN (SELECT id FROM temp.moved_field)"), "n") ||
            n_meas != integer(*pkg.one("SELECT count(*) AS n FROM src.selection_measurement WHERE of_field_id IN (SELECT id FROM temp.moved_field)"), "n"))
          throw refuse("split: " + g.group_name + ": the contributions or measurements are not the group fields' all");
        g.n_contributions = n_contrib; g.n_measurements = n_meas;
        gout.rows = g.n_fields + n_contrib + n_meas + 1;
        pkg.execute("DETACH DATABASE src");
        {
          FolderStore files(gdir);
          SmallFileStore small(files, pkg, SmallFileOptions{});
          rebuild_key_digests(pkg, small);
          gout.keys = integer(*pkg.one("SELECT count(*) AS n FROM key_digest"), "n");
          small.close();
        }
        PackageRecord rec;
        rec.subject_id = g.group_id; rec.name = g.group_name; rec.kind = "group";
        rec.schema_version = schema_version(); rec.store_format = kStoreFormatPackage;
        const auto tref = template_reference_of.find(g.template_id);
        if (tref == template_reference_of.end()) throw refuse("split: " + g.group_name + ": its template " + g.template_name + " was not migrated");
        rec.template_reference = tref->second;
        rec.members = Json::array();
        {
          const Catalogue cat = Catalogue::open(folder);
          Database q = Database::open(gbuilding, false, "", true);
          for (const auto& m : q.all("SELECT DISTINCT subject_id FROM main.field_contribution WHERE field_id IN (SELECT id FROM main.field WHERE subject_id = ?) ORDER BY subject_id", {g.group_id})) {
            const std::string msid = text(m, "subject_id");
            const auto e = cat.find(msid);
            rec.members.push_back(Json{{"subject_id", msid}, {"name", member_name.count(msid) ? member_name[msid] : std::string()}, {"content_hash", e ? e->content_hash : std::string()}});
          }
        }
        write_package_record(pkg, rec);
        refresh_package_record(pkg);
      }
      glock.release();
      if (opt.on_stage) opt.on_stage("built", g.group_name);
      fs::rename(gbuilding, gpkg);
      if (opt.on_stage) opt.on_stage("committed", g.group_name);
    }
    RegisterOptions gro;
    gro.registered_by = opt.registered_by;
    const auto gres = register_packages(folder, {gdir}, gro);
    if (gres.empty() || (gres[0].outcome != "registered" && gres[0].outcome != "unchanged" && gres[0].outcome != "updated"))
      throw refuse("split: " + g.group_name + ": the group package was not registered: " + (gres.empty() ? std::string() : gres[0].reason));
    if (opt.on_stage) opt.on_stage("registered", g.group_name);
    outs.push_back(gout);
    group_report.push_back(Json{{"from", g.template_name}, {"to", g.group_name}, {"group_subject_id", g.group_id}, {"template_subject_id", g.template_id},
                                {"field", Json{{"rows", g.n_fields}, {"change", "subject_id: " + g.template_id + " -> " + g.group_id}}},
                                {"field_contribution", Json{{"rows", g.n_contributions}, {"change", "owner follows the field"}}},
                                {"selection_measurement", Json{{"rows", g.n_measurements}, {"change", "owner follows the field"}}},
                                {"keys_under", g.prefixes}, {"originals_kept_aside", (folder / "_moved-aside" / g.template_path).string()}});
  }
  if (!groups.empty()) {
    Json mapping = Json::array();
    for (const MovedGroup& g : groups) mapping.push_back(Json{{"group_subject_id", g.group_id}, {"group_name", g.group_name}, {"group_path", g.group_path}, {"template_subject_id", g.template_id}, {"template_path", g.template_path}, {"prefixes", g.prefixes}, {"fields", g.field_ids}});
    Database cdb = Database::open(folder / kCatalogueFile);
    cdb.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('split_groups', ?)", {mapping.dump()});
    cdb.close();
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
    // THE OWNER MAPPING: the moved group fields are the original's with `subject_id` the group subject's, and each group subject is the one row the original lacks
    std::string group_ids = "''", to_template = "";
    for (const MovedGroup& g : groups) { group_ids += ", " + sql_quote(g.group_id); to_template += " WHEN " + sql_quote(g.group_id) + " THEN " + sql_quote(g.template_id); }
    for (const auto& t : tables_of(idx)) {
      if (bookkeeping(t)) continue;
      if (!has(tables_of(idx, "orig"), t)) continue;
      const std::string l = "\"" + t + "\"", named = named_columns(idx, t, "main", "orig", "split: the index built from the packages");
      std::string here = "SELECT " + named + " FROM main." + l;
      if (!groups.empty() && t == "subject") here = "SELECT " + named + " FROM main.subject WHERE id NOT IN (" + group_ids + ")";
      if (!groups.empty() && t == "field") {
        std::string cols;
        for (const auto& c : columns_of(idx, "field")) cols += (cols.empty() ? "" : ", ") + (c == "subject_id" ? "CASE subject_id" + to_template + " ELSE subject_id END AS subject_id" : "\"" + c + "\"");
        here = "SELECT " + cols + " FROM main.field";
      }
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
  if (!group_report.empty()) report["group_results"] = group_report;
  report["template_references"] = Catalogue::open(folder).template_references();
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
  // the group results `split` moved out of the default subjects: they go back (the mapping split wrote, reversed)
  Json groups = Json::array();
  if (const auto r = cat.db().one("SELECT value FROM meta WHERE key = 'split_groups'")) groups = Json::parse(text(*r, "value"));
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
        for (const Json& g : groups) {
          db.execute("UPDATE field SET subject_id = ? WHERE subject_id = ?", {g["template_subject_id"].get<std::string>(), g["group_subject_id"].get<std::string>()});
          db.execute("DELETE FROM subject WHERE id = ?", {g["group_subject_id"].get<std::string>()});
        }
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
  //    a group package's keys go back to the default subject it came from: the arrays (the originals `split` moved aside, else the group package's copy, compared), the small keys
  Json subjects = Json::array();
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> returning;   // default subject's folder -> small keys coming back
  std::set<std::string> group_ids;
  for (const Json& g : groups) {
    const std::string gid = g["group_subject_id"].get<std::string>();
    group_ids.insert(gid);
    const fs::path gdir = folder / g["group_path"].get<std::string>(), tdir = folder / g["template_path"].get<std::string>();
    if (!fs::is_regular_file(gdir / kPackageDatabaseFile)) continue;   // already returned
    for (const auto& r : Database::open(gdir / kPackageDatabaseFile, false, "", true).all("SELECT key, value FROM small_files ORDER BY key"))
      returning[g["template_path"].get<std::string>()].emplace_back(text(r, "key"), std::get<std::string>(r.at("value")));
    for (const Json& pj : g["prefixes"]) {
      const std::string P = pj.get<std::string>();
      if (!fs::exists(gdir / P)) continue;
      if (!fs::exists(tdir / P)) {
        const fs::path aside_dir = folder / "_moved-aside" / g["template_path"].get<std::string>() / P;
        if (fs::exists(aside_dir) && same_tree(aside_dir, gdir / P)) { fs::create_directories((tdir / P).parent_path()); fs::rename(aside_dir, tdir / P); }
        else copy_tree_verified(gdir / P, tdir / P, "join: the group results " + P);
      } else if (!same_tree(tdir / P, gdir / P)) {
        throw refuse("join: " + (tdir / P).string() + " is not the group package's copy of it");
      }
    }
  }
  for (const auto& e : members) {
    const fs::path sdir = cat.package_path(e), pkg_file = sdir / kPackageDatabaseFile, small = sdir / kSmallFilesFile;
    if (group_ids.count(e.subject_id)) {
      if (fs::is_regular_file(pkg_file)) {
        fs::rename(sdir, aside_name(sdir, ".format53"));   // the whole group package, kept: its arrays are copies of what went back
        subjects.push_back(Json{{"name", e.name}, {"returned_to_default_subject", true}});
        if (opt.on_stage) opt.on_stage("aside", e.name);
      }
      continue;
    }
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
            if (const auto it = returning.find(e.path); it != returning.end())
              for (const auto& [k, v] : it->second) { out.put_row(k, reinterpret_cast<const std::uint8_t*>(v.data()), v.size()); ++keys; }
          });
          // verified byte for byte against the package's table
          SmallFileStore in(files, pkg, SmallFileOptions{kSmallFileThreshold, true});
          for (const auto& k : in.table_keys()) {
            const auto x = in.get(k), y = out.get(k);
            if (!x || !y || *x != *y) throw refuse("join: " + e.name + ": the key " + k + " differs after the move");
          }
          auto expected = in.table_keys();
          if (const auto it = returning.find(e.path); it != returning.end()) { for (const auto& kv : it->second) expected.push_back(kv.first); std::sort(expected.begin(), expected.end()); }
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

}  // namespace nxr::io
