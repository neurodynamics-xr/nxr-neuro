#include "nxr/io/dataset_index.h"

#include <map>
#include <set>

#include "nxr/io/hash.h"
#include "nxr/io/writer_lock.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {

Error refuse(const std::string& m) { return Error(code::invalid_argument, m); }

// the members the index is built from: registered, in name order
std::vector<CatalogueEntry> index_members(const Catalogue& cat) {
  std::vector<CatalogueEntry> out;
  for (const auto& e : cat.entries()) if (e.status == "registered") out.push_back(e);
  return out;
}

Json built_from(const std::vector<CatalogueEntry>& members) {
  Json j = Json::object();
  for (const auto& m : members) j[m.subject_id] = m.content_hash;
  return j;
}

std::string entries_digest(const Database& db, const std::string& schema, const std::string& dictionary_id) {
  const auto cols = db.all("SELECT name FROM pragma_table_info('dictionary_entry') WHERE name <> 'dictionary_id' ORDER BY cid");
  std::string list;
  for (const auto& c : cols) list += (list.empty() ? "" : ", ") + std::string("\"") + text(c, "name") + "\"";
  Json all = Json::array();
  for (const auto& e : db.all("SELECT " + list + " FROM " + schema + ".dictionary_entry WHERE dictionary_id = ? ORDER BY code", {dictionary_id})) {
    Json a = Json::array();
    for (const auto& c : cols) {
      const Value& v = e.at(text(c, "name"));
      if (std::holds_alternative<std::monostate>(v)) a.push_back(nullptr);
      else if (auto* n = std::get_if<std::int64_t>(&v)) a.push_back(*n);
      else if (auto* x = std::get_if<double>(&v)) a.push_back(*x);
      else a.push_back(std::get<std::string>(v));
    }
    all.push_back(a);
  }
  return sha256_hex(jdump(all));
}

}  // namespace

bool index_is_current(const fs::path& folder) {
  Catalogue cat = Catalogue::open(folder);
  if (!fs::is_regular_file(cat.folder() / kIndexFile)) return false;
  try {
    Database db = Database::open(cat.folder() / kIndexFile, false, "", true);
    const auto r = db.one("SELECT value FROM meta WHERE key = 'index_json'");
    if (!r) return false;
    return Json::parse(text(*r, "value")).value("built_from", Json::object()) == built_from(index_members(cat));
  } catch (const Error&) {
    return false;
  }
}

IndexReport build_index(const fs::path& folder_in, bool force) {
  Catalogue cat = Catalogue::open(folder_in);
  const fs::path folder = cat.folder();
  const auto members = index_members(cat);
  IndexReport rep;
  for (const auto& m : members) rep.subjects.push_back(m.name);
  rep.n_subjects = static_cast<std::int64_t>(members.size());
  if (!force && index_is_current(folder)) return rep;

  std::vector<fs::path> paths;
  for (const auto& m : members) {
    const fs::path p = cat.package_path(m);
    if (!fs::is_regular_file(p / kPackageDatabaseFile)) throw refuse("index: " + m.name + ": its package is missing at " + p.string());
    // an unknown version refuses the whole build, naming the package (SubjectPackage::open does)
    const std::string have = SubjectPackage::open(p).record().content_hash;
    if (have != m.content_hash) throw refuse("index: " + m.name + " is stale: its package has changed since it was registered (register it again)");
    paths.push_back(p);
  }
  IndexProblems problems;
  for (const auto& c : find_id_clashes(paths)) problems.clashes.push_back("the " + c.table + " id " + c.id + " is in both " + c.first.string() + " and " + c.second.string());

  WriterLock lock = acquire_writer_lock(folder, "nxr-io");
  const fs::path tmp = folder / (std::string(kIndexFile) + ".tmp");
  std::error_code ec;
  fs::remove(tmp, ec);
  try {
    {
      Database db = Database::open(tmp, true, "nxr-io");
      db.execute("PRAGMA foreign_keys = OFF");
      const Row dataset = cat.dataset_row();
      const std::string dataset_id = text(dataset, "id");
      db.tx([&] {
        db.insert("datastore", *Database::open(folder / kCatalogueFile, false, "", true).read("datastore", text(dataset, "datastore_id")));
        db.insert("dataset", dataset);
        db.execute("CREATE TEMP TABLE dict_remap (old TEXT PRIMARY KEY, new TEXT NOT NULL)");
      });
      // the tables the packages fold into the index (the model's, but the home rows, the outbox and the package's own bookkeeping)
      std::vector<std::string> tables;
      for (const auto& r : db.all("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")) {
        const std::string t = text(r, "name");
        if (t == "meta" || t == "dataset" || t == "datastore" || t == "sync_job" || t.rfind("sqlite_", 0) == 0) continue;
        tables.push_back(t);
      }
      std::map<std::string, std::vector<std::string>> columns;
      std::map<std::string, std::vector<std::string>> dictionary_refs;   // table -> its columns that reference dictionary(id)
      for (const auto& t : tables) {
        for (const auto& c : db.all("SELECT name FROM pragma_table_info(?) ORDER BY cid", {t})) columns[t].push_back(text(c, "name"));
        for (const auto& f : db.all("SELECT \"from\" AS col, \"table\" AS parent FROM pragma_foreign_key_list(?)", {t}))
          if (text(f, "parent") == "dictionary") dictionary_refs[t].push_back(text(f, "col"));
      }
      for (std::size_t i = 0; i < members.size(); ++i) {
        db.execute("ATTACH DATABASE ? AS pk", {paths[i].string() + "/" + kPackageDatabaseFile});
        try {
          db.tx([&] {
            db.execute("DELETE FROM dict_remap");
            // the shared vocabulary: matched by (scope, name), the first copy kept, a different copy a conflict
            for (const auto& d : db.all("SELECT * FROM pk.dictionary WHERE scope IN ('app', 'dataset', 'user') ORDER BY name")) {
              const auto kept = db.one("SELECT id FROM main.dictionary WHERE scope = ? AND name = ?", {text(d, "scope"), text(d, "name")});
              if (kept) {
                if (entries_digest(db, "main", text(*kept, "id")) != entries_digest(db, "pk", text(d, "id")))
                  problems.conflicts.push_back("'" + text(d, "name") + "' (" + text(d, "scope") + " scope) in " + paths[i].string() + " has entries that differ from the copy already in the index");
                db.execute("INSERT INTO dict_remap (old, new) VALUES (?, ?)", {text(d, "id"), text(*kept, "id")});
              } else {
                Row r = d;
                if (text(d, "scope") == "dataset") r["dataset_id"] = dataset_id;
                db.insert("dictionary", r);
                db.execute("INSERT OR IGNORE INTO main.dictionary_entry SELECT * FROM pk.dictionary_entry WHERE dictionary_id = ?", {text(d, "id")});
              }
            }
            // a subject's own dictionaries (and a template's, anchored in the subjects that name them): each is one row, once
            for (const auto& d : db.all("SELECT * FROM pk.dictionary WHERE scope = 'subject'")) {
              db.execute("INSERT OR IGNORE INTO main.dictionary SELECT * FROM pk.dictionary WHERE id = ?", {text(d, "id")});
              db.execute("INSERT OR IGNORE INTO main.dictionary_entry SELECT * FROM pk.dictionary_entry WHERE dictionary_id = ?", {text(d, "id")});
            }
            for (const auto& t : tables) {
              if (t == "dictionary" || t == "dictionary_entry" || t == "anchor_row" || t == "key_digest" || t == "small_files") continue;
              std::string cols, sel;
              for (const auto& c : columns[t]) {
                cols += (cols.empty() ? "" : ", ") + std::string("\"") + c + "\"";
                std::string expr = "x.\"" + c + "\"";
                if (c == "dataset_id") expr = "'" + dataset_id + "'";
                for (const auto& ref : dictionary_refs[t]) if (ref == c) expr = "COALESCE((SELECT new FROM dict_remap WHERE old = x.\"" + c + "\"), x.\"" + c + "\")";
                sel += (sel.empty() ? "" : ", ") + expr;
              }
              const bool has_id = !columns[t].empty() && std::find(columns[t].begin(), columns[t].end(), "id") != columns[t].end();
              std::string where;
              if (has_id) where = " WHERE NOT EXISTS (SELECT 1 FROM pk.anchor_row a WHERE a.table_name = '" + t + "' AND a.row_id = x.id)";
              // the subject rows of a member name no template in this file's rows: a catalogue's template_id is set from the packages' own
              db.execute("INSERT OR IGNORE INTO main.\"" + t + "\" (" + cols + ") SELECT " + sel + " FROM pk.\"" + t + "\" x" + where);   // a child row anchored in a subject is the template's own, once
            }
            // anchor rows no member supplied itself (a template that is not a member): taken from the package that names them
            for (const auto& a : db.all("SELECT table_name, row_id FROM pk.anchor_row")) {
              const std::string t = text(a, "table_name");
              if (!columns.count(t)) continue;
              std::string cols;
              for (const auto& c : columns[t]) cols += (cols.empty() ? "" : ", ") + std::string("\"") + c + "\"";
              db.execute("INSERT OR IGNORE INTO main.\"" + t + "\" (" + cols + ") SELECT " + cols + " FROM pk.\"" + t + "\" WHERE id = ?", {text(a, "row_id")});
            }
          });
        } catch (...) {
          db.execute("DETACH DATABASE pk");
          throw;
        }
        db.execute("DETACH DATABASE pk");
      }
      if (problems.conflicts.empty() && problems.clashes.empty()) {
        for (const auto& r : db.all("PRAGMA foreign_key_check"))
          problems.conflicts.push_back("a row of " + text(r, "table") + " (rowid " + text(r, "rowid") + ") names a row of " + text(r, "parent") + " that no member holds");
      }
      if (problems.conflicts.empty() && problems.clashes.empty()) {
        db.tx([&] {
          db.execute("DELETE FROM sync_job");
          db.execute("INSERT INTO meta (key, value) VALUES ('index_json', ?)", {jdump(Json{{"dataset_id", dataset_id}, {"built_from", built_from(members)}, {"built_utc", now_utc()}})});
        });
      }
    }
    if (!problems.clashes.empty() || !problems.conflicts.empty()) {
      std::string msg = "index: not built";
      for (const auto& c : problems.clashes) msg += "\n  id clash: " + c;
      for (const auto& c : problems.conflicts) msg += "\n  conflict: " + c;
      throw Error(problems.clashes.empty() ? code::invalid_argument : code::id_clash, msg);
    }
    fs::rename(tmp, folder / kIndexFile);
  } catch (...) {
    fs::remove(tmp, ec);
    throw;
  }
  rep.rebuilt = true;
  return rep;
}

}  // namespace nxr::io
