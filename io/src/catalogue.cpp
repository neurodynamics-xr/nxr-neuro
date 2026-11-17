#include "nxr/io/catalogue.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

#include "nxr/io/hash.h"
#include "nxr/io/bundle.h"
#include "nxr/io/participants.h"
#include "nxr/io/statistic.h"
#include "nxr/io/writer_lock.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {

// (`subject_entry` and `template_reference` are the model's since schema 51)
const char* const kCatalogueTables =
    // the shared vocabulary the members' copies are held to by key (section 2.1): the first registrant's entries, as a digest
    "CREATE TABLE vocabulary (\n"
    "  scope TEXT NOT NULL, name TEXT NOT NULL, sha256 TEXT NOT NULL, first_subject_id TEXT NOT NULL,\n"
    "  PRIMARY KEY (scope, name)\n"
    ") STRICT;\n";

Error value_error(const std::string& m) { return Error(code::invalid_argument, m); }

void check_name(const std::string& what, const std::string& n) {
  if (n.empty() || n.find('/') != std::string::npos || n.find('\\') != std::string::npos || n.find("..") != std::string::npos) throw value_error("invalid " + what + " name: '" + n + "'");
}

fs::path absolute_of(const fs::path& p) {
  std::error_code ec;
  fs::path r = fs::weakly_canonical(fs::absolute(p), ec);
  return ec ? fs::absolute(p).lexically_normal() : r;
}

CatalogueEntry entry_of(const Row& r) {
  CatalogueEntry e;
  e.subject_id = text(r, "id"); e.name = text(r, "name"); e.path = text(r, "path"); e.kind = text(r, "kind");
  e.schema_version = static_cast<int>(integer(r, "schema_version")); e.store_format = static_cast<int>(integer(r, "store_format"));
  e.content_hash = text(r, "content_hash"); e.registered_utc = text(r, "registered_utc"); e.registered_by = text(r, "registered_by"); e.status = text(r, "status");
  e.n_acquisitions = integer(r, "n_acquisitions"); e.n_recordings = integer(r, "n_recordings"); e.n_surfaces = integer(r, "n_surfaces");
  e.n_fields = integer(r, "n_fields"); e.n_operators = integer(r, "n_operators");
  return e;
}

const char* const kEntrySelect =
    "SELECT s.id, s.name, s.path, s.kind, e.schema_version, e.store_format, e.content_hash, e.registered_utc, e.registered_by, e.status, "
    "e.n_acquisitions, e.n_recordings, e.n_surfaces, e.n_fields, e.n_operators FROM subject s JOIN subject_entry e ON e.subject_id = s.id ";

}  // namespace

Json CatalogueEntry::to_json() const {
  return Json{{"subject_id", subject_id}, {"name", name}, {"path", path}, {"kind", kind}, {"schema_version", schema_version}, {"store_format", store_format},
              {"content_hash", content_hash}, {"registered_utc", registered_utc}, {"registered_by", registered_by}, {"status", status},
              {"n_acquisitions", n_acquisitions}, {"n_recordings", n_recordings}, {"n_surfaces", n_surfaces}, {"n_fields", n_fields}, {"n_operators", n_operators}};
}

// ══ the catalogue ═══════════════════════════════════════════════════════════════════════════════════════════════

Catalogue Catalogue::open(const fs::path& folder_in) {
  const fs::path folder = absolute_of(folder_in);
  if (!fs::is_regular_file(folder / kCatalogueFile)) throw Error(code::not_found, folder.string() + ": not a dataset (no " + kCatalogueFile + ")");
  Catalogue c;
  c.folder_ = folder;
  c.db_ = std::make_unique<Database>(Database::open(folder / kCatalogueFile, false, "", true));
  const auto v = c.db_->one("SELECT value FROM meta WHERE key = 'catalogue_version'");
  const int version = v ? std::atoi(text(*v, "value").c_str()) : 0;
  if (version != kCatalogueVersion)
    throw Error(code::schema_version, folder.string() + ": catalogue version " + std::to_string(version) + ", this build reads " + std::to_string(kCatalogueVersion));
  if (store_format_number(*c.db_) > kStoreFormatPackage)
    throw Error(code::schema_version, folder.string() + ": store format " + std::to_string(store_format_number(*c.db_)) + " — this build reads up to " + std::to_string(kStoreFormatPackage));
  return c;
}
Catalogue::Catalogue(Catalogue&&) noexcept = default;
Catalogue& Catalogue::operator=(Catalogue&&) noexcept = default;
Catalogue::~Catalogue() = default;

Row Catalogue::dataset_row() const { return *db_->one("SELECT * FROM dataset ORDER BY name LIMIT 1"); }

std::vector<CatalogueEntry> Catalogue::entries() const {
  std::vector<CatalogueEntry> out;
  for (const auto& r : db_->all(std::string(kEntrySelect) + "ORDER BY (e.status = 'replaced'), s.name, s.id")) out.push_back(entry_of(r));
  return out;
}

std::optional<CatalogueEntry> Catalogue::find(const std::string& name_or_id) const {
  if (const auto r = db_->one(std::string(kEntrySelect) + "WHERE s.id = ?", {name_or_id})) return entry_of(*r);
  if (const auto r = db_->one(std::string(kEntrySelect) + "WHERE s.name = ? AND e.status <> 'replaced'", {name_or_id})) return entry_of(*r);
  return std::nullopt;
}

fs::path Catalogue::package_path(const CatalogueEntry& e) const {
  const fs::path p(e.path);
  return p.is_absolute() ? p : folder_ / p;
}

std::vector<Json> Catalogue::template_references() const {
  std::vector<Json> out;
  for (const auto& r : db_->all("SELECT * FROM template_reference ORDER BY name, version, edition"))
    out.push_back(Json{{"name", text(r, "name")}, {"edition", text(r, "edition")}, {"version", text(r, "version")}, {"subject_id", text(r, "subject_id")},
                       {"content_hash", text(r, "content_hash")}, {"location_hint", opt_text(r, "location_hint") ? Json(text(r, "location_hint")) : Json(nullptr)}});
  return out;
}

bool Catalogue::has_participants() const { return fs::is_regular_file(folder_ / kParticipantsFile); }

CreatePackageOptions Catalogue::package_options() const {
  const Row ds = dataset_row();
  CreatePackageOptions o;
  o.home_dataset = text(ds, "name");
  o.home_dataset_id = text(ds, "id");
  o.datastore_id = text(ds, "datastore_id");
  return o;
}

Catalogue create_catalogue_from(const fs::path& folder_in, const Row& dataset, const Row& datastore, const std::string& built_by, bool lock_it) {
  const fs::path folder = absolute_of(folder_in);
  if (fs::exists(folder / kCatalogueFile)) throw Error(code::already_exists, folder.string() + ": already a dataset");
  WriterLock lock;
  if (lock_it) lock = acquire_writer_lock(folder, built_by);
  const fs::path file = folder / kCatalogueFile;
  try {
    Database db = Database::open(file, true, built_by);
    db.execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat53)});
    db.execute("INSERT INTO meta (key, value) VALUES ('catalogue_version', ?)", {std::to_string(kCatalogueVersion)});
    db.exec_script(kCatalogueTables);
    db.tx([&] {
      db.insert("datastore", datastore);
      db.insert("dataset", dataset);
      db.execute("DELETE FROM sync_job");
    });
  } catch (...) {
    std::error_code ec;
    fs::remove(file, ec);
    throw;
  }
  return Catalogue::open(folder);
}

Catalogue create_catalogue(const fs::path& folder_in, const CreateCatalogueOptions& opt) {
  check_name("dataset", opt.name);
  const fs::path folder = absolute_of(folder_in);
  if (fs::exists(folder / kCatalogueFile)) throw Error(code::already_exists, folder.string() + ": already a dataset");
  const bool made = !fs::exists(folder);
  fs::create_directories(folder);
  WriterLock lock;
  try { lock = acquire_writer_lock(folder, opt.built_by.value_or("nxr-io")); } catch (...) { if (made) remove_folder(folder); throw; }
  try {
    {
      Database db = Database::open(folder / kCatalogueFile, true, opt.built_by.value_or("nxr-io"));
      db.execute("UPDATE meta SET value = ? WHERE key = 'store_format'", {std::string(kStoreFormat53)});
      db.execute("INSERT INTO meta (key, value) VALUES ('catalogue_version', ?)", {std::to_string(kCatalogueVersion)});
      db.exec_script(kCatalogueTables);
      const std::string now = now_utc(), datastore_id = uuid7();
      db.tx([&] {
        // the hint only: where the dataset was made; nothing reads it to find data
        db.insert("datastore", {{"id", datastore_id}, {"url", folder.parent_path().string()}, {"driver", "file"}, {"name", folder.parent_path().filename().string()}});
        db.insert("dataset", {{"id", uuid7()}, {"datastore_id", datastore_id}, {"name", opt.name}, {"path", folder.filename().string()}, {"source_tool", V(opt.source_tool)},
                              {"source_protocol", V(opt.source_protocol)}, {"source_path", V(opt.source_path)}, {"created_utc", now}, {"created_by", V(opt.created_by)},
                              {"modified_utc", now}, {"modified_by", V(opt.created_by)}});
        db.execute("DELETE FROM sync_job");
      });
    }
    lock.release();
  } catch (...) {
    lock.release();
    if (made) remove_folder(folder);
    else { std::error_code ec; fs::remove(folder / kCatalogueFile, ec); }
    throw;
  }
  return Catalogue::open(folder);
}

// ══ registering ═════════════════════════════════════════════════════════════════════════════════════════════════

namespace {

struct Vocab { std::string scope, name, sha256; };

struct Candidate {
  fs::path path;
  std::size_t result = 0;
  PackageRecord rec;
  Row subject, counts;
  std::vector<Vocab> vocab;
};

std::vector<Vocab> vocabulary_of(const Database& db) {
  std::vector<Vocab> out;
  const auto cols = db.all("SELECT name FROM pragma_table_info('dictionary_entry') WHERE name <> 'dictionary_id' ORDER BY cid");
  std::string list;
  for (const auto& c : cols) list += (list.empty() ? "" : ", ") + std::string("\"") + text(c, "name") + "\"";
  for (const auto& d : db.all("SELECT id, scope, name FROM dictionary WHERE scope IN ('app', 'dataset') ORDER BY scope, name")) {
    Json entries = Json::array();
    for (const auto& e : db.all("SELECT " + list + " FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", {text(d, "id")})) {
      Json a = Json::array();
      for (const auto& c : cols) {
        const Value& v = e.at(text(c, "name"));
        if (std::holds_alternative<std::monostate>(v)) a.push_back(nullptr);
        else if (auto* n = std::get_if<std::int64_t>(&v)) a.push_back(*n);
        else if (auto* x = std::get_if<double>(&v)) a.push_back(*x);
        else a.push_back(std::get<std::string>(v));
      }
      entries.push_back(a);
    }
    out.push_back({text(d, "scope"), text(d, "name"), sha256_hex(jdump(entries))});
  }
  return out;
}

// A package read for registration, or the reason it cannot be.
std::optional<std::string> examine(const fs::path& path, Candidate& c, const std::optional<TemplateSearch>& templates) {
  if (!fs::is_directory(path) || !fs::is_regular_file(path / kPackageDatabaseFile)) return path.string() + ": not a subject package (no " + kPackageDatabaseFile + ")";
  std::error_code ec;
  if (fs::exists(path / kWriterLockFile, ec)) return path.string() + ": a writer holds the package (or left its lock): close it, or open and close it with a writer to recover";
  try {
    SubjectPackage p = SubjectPackage::open(path);
    c.rec = p.record();
    if (c.rec.kind != "subject" && c.rec.kind != "template" && c.rec.kind != "group")
      return path.string() + ": a package of kind '" + c.rec.kind + "' is not a member of a dataset";
    for (const char* t : kStatusTables) {
      if (!has_table(p.db(), t)) continue;
      if (const auto r = p.db().one(std::string("SELECT name, status FROM ") + t + " WHERE status <> 'complete' LIMIT 1"))
        return path.string() + ": incomplete: " + t + " '" + text(*r, "name") + "' is " + text(*r, "status") + " (open it with a writer to recover)";
    }
    if (c.rec.content_hash.empty()) return path.string() + ": no content hash: a writer has never closed this package";
    if (p.content_hash() != c.rec.content_hash)
      return path.string() + ": its rows and digests no longer hash to its record (changed outside a writer): open and close it with a writer";
    c.subject = p.subject_row();
    c.counts = *p.db().one("SELECT n_acquisitions, n_recordings, n_surfaces, n_fields, n_operators FROM subject_status WHERE id = ?", {c.rec.subject_id});
    c.vocab = vocabulary_of(p.db());
    if (templates && c.rec.template_reference.is_object() && c.rec.template_reference.contains("content_hash")) {
      const ResolvedTemplate t = resolve_template(TemplateRef::from_json(c.rec.template_reference), *templates);
      const auto differ = anchor_differences(p.db(), t.folder);
      if (!differ.empty()) {
        std::string list;
        for (std::size_t i = 0; i < differ.size() && i < 5; ++i) list += (list.empty() ? "" : ", ") + differ[i];
        return path.string() + ": its anchor rows differ from template " + t.record.release.value("name", "") + "@" + t.record.release.value("version", "") + "'s: " + list + (differ.size() > 5 ? ", ..." : "");
      }
    }
  } catch (const Error& e) {
    return path.string() + ": " + e.what();
  }
  return std::nullopt;
}

std::string stored_path(const fs::path& folder, const fs::path& package) {
  const fs::path rel = package.lexically_relative(folder);
  if (!rel.empty() && *rel.begin() != "..") return rel.generic_string();
  return package.generic_string();
}

}  // namespace

std::vector<RegisterResult> register_packages(const fs::path& folder_in, const std::vector<fs::path>& packages, const RegisterOptions& opt) {
  if (opt.registered_by.empty()) throw value_error("register: say who registers (registered_by)");
  Catalogue cat = Catalogue::open(folder_in);
  const fs::path folder = cat.folder();
  std::vector<RegisterResult> results;
  std::vector<Candidate> cands;
  std::set<std::string> seen;
  for (const auto& p : packages) {
    RegisterResult r;
    r.package = absolute_of(p);
    results.push_back(r);
    Candidate c;
    c.path = absolute_of(p);
    c.result = results.size() - 1;
    if (!seen.insert(c.path.string()).second) { results.back().outcome = "refused"; results.back().reason = c.path.string() + ": named twice"; continue; }
    if (const auto why = examine(c.path, c, opt.templates)) { results.back().outcome = "refused"; results.back().reason = *why; continue; }
    results.back().name = c.rec.name;
    cands.push_back(std::move(c));
  }
  if (cands.empty()) return results;

  WriterLock lock = acquire_writer_lock(folder, opt.registered_by);
  Database db = Database::open(folder / kCatalogueFile);
  const std::string dataset_id = text(*db.one("SELECT id FROM dataset LIMIT 1"), "id");

  // ids: the candidates against the members they do not replace, and against each other (one pass over all of them)
  {
    std::set<std::string> cand_ids;
    for (const auto& c : cands) cand_ids.insert(c.rec.subject_id);
    std::vector<fs::path> all;
    for (const auto& e : cat.entries()) {
      if (e.status == "replaced" || e.status == "missing" || cand_ids.count(e.subject_id)) continue;
      const fs::path pp = cat.package_path(e);
      if (fs::is_regular_file(pp / kPackageDatabaseFile)) all.push_back(pp);
    }
    const std::size_t n_members = all.size();
    for (const auto& c : cands) all.push_back(c.path);
    std::map<std::string, std::size_t> index_of;
    for (std::size_t i = 0; i < all.size(); ++i) index_of[all[i].string()] = i;
    for (const auto& clash : find_id_clashes(all)) {
      const std::size_t a = index_of[clash.first.string()], b = index_of[clash.second.string()];
      const std::size_t loser = std::max(a, b);   // the later one: a candidate, never a member
      if (loser < n_members) continue;
      Candidate& c = cands[loser - n_members];
      auto& res = results[c.result];
      if (res.outcome == "refused") continue;
      res.outcome = "refused";
      res.reason = c.path.string() + ": the " + clash.table + " id " + clash.id + " is also in " + (a == loser ? clash.second : clash.first).string();
    }
  }

  for (auto& c : cands) {
    RegisterResult& res = results[c.result];
    if (res.outcome == "refused") continue;
    try {
      std::string outcome;
      db.tx([&] {
        const std::string rel = stored_path(folder, c.path);
        const auto existing = db.one(std::string(kEntrySelect) + "WHERE s.id = ?", {c.rec.subject_id});
        Row srow = c.subject;
        srow["dataset_id"] = dataset_id;
        srow["path"] = rel;
        srow["template_id"] = std::monostate{};
        srow["status"] = std::string("complete");
        const std::string now = now_utc();
        Row ent = {{"subject_id", c.rec.subject_id}, {"schema_version", V(c.rec.schema_version)}, {"store_format", V(c.rec.store_format)}, {"content_hash", c.rec.content_hash},
                   {"registered_utc", now}, {"registered_by", opt.registered_by}, {"status", std::string("registered")},
                   {"n_acquisitions", c.counts.at("n_acquisitions")}, {"n_recordings", c.counts.at("n_recordings")}, {"n_surfaces", c.counts.at("n_surfaces")},
                   {"n_fields", c.counts.at("n_fields")}, {"n_operators", c.counts.at("n_operators")}};
        if (existing && text(*existing, "status") != "replaced") {
          const CatalogueEntry old = entry_of(*existing);
          const fs::path old_path = cat.package_path(old);
          std::error_code ec;
          if (old.path != rel && fs::is_directory(old_path, ec) && fs::is_regular_file(old_path / kPackageDatabaseFile, ec) && !fs::equivalent(old_path, c.path, ec))
            throw value_error(c.path.string() + ": subject " + old.name + " (" + old.subject_id + ") is already registered at " + old_path.string() + "; a copy elsewhere has the same id");
          if (old.content_hash == c.rec.content_hash && old.path == rel && old.status == "registered" && old.name == c.rec.name) { outcome = "unchanged"; return; }
          db.upsert("subject", srow);
          db.upsert("subject_entry", ent);
          outcome = "updated";
        } else {
          // another subject under this name or path
          std::vector<Row> clash = db.all(std::string(kEntrySelect) + "WHERE (s.name = ? OR s.path = ?) AND e.status <> 'replaced' AND s.id <> ?", {c.rec.name, rel, c.rec.subject_id});
          if (!clash.empty() && !opt.replace)
            throw value_error(c.path.string() + ": the catalogue already holds a subject '" + text(clash[0], "name") + "' (" + text(clash[0], "id") + ") under that name or path; replace it on purpose with --replace");
          for (const auto& o : clash) {
            const CatalogueEntry old = entry_of(o);
            std::string moved = old.path;
            const fs::path old_path = cat.package_path(old);
            std::error_code ec;
            const bool inside = !fs::path(old.path).is_absolute();
            if (inside && fs::exists(old_path, ec)) {
              fs::path aside = old_path.string() + ".replaced-" + now_utc().substr(0, 10);
              for (int k = 2; fs::exists(aside, ec); ++k) aside = old_path.string() + ".replaced-" + now_utc().substr(0, 10) + "-" + std::to_string(k);
              if (old_path != c.path) fs::rename(old_path, aside);   // the package being registered may be the one in place: then there is nothing to move
              moved = stored_path(folder, aside);
            }
            db.execute("UPDATE subject SET path = ? WHERE id = ?", {moved == old.path && old.path == rel ? old.path + ".replaced" : moved, old.subject_id});
            db.execute("UPDATE subject_entry SET status = 'replaced' WHERE subject_id = ?", {old.subject_id});
          }
          if (existing) { db.upsert("subject", srow); db.upsert("subject_entry", ent); }   // a replaced entry registered again
          else { db.insert("subject", srow); db.insert("subject_entry", ent); }
          outcome = clash.empty() ? "registered" : "replaced";
        }
        // the template this subject is registered to: one reference a dataset, one hash a name, edition and version
        if (c.rec.template_reference.is_object()) {
          const Json& t = c.rec.template_reference;
          const std::string tname = t.value("name", ""), ted = t.value("edition", "full"), tver = t.value("version", ""), thash = t.value("content_hash", "");
          if (const auto have = db.one("SELECT content_hash FROM template_reference WHERE name = ? AND edition = ? AND version = ?", {tname, ted, tver})) {
            if (text(*have, "content_hash") != thash)
              throw value_error(c.path.string() + ": registered to template " + tname + "@" + tver + " with content hash " + thash + ", the dataset's subjects use " + text(*have, "content_hash") + " (a cohort is never pooled across template versions)");
          } else {
            db.insert("template_reference", {{"name", tname}, {"edition", ted}, {"version", tver}, {"subject_id", t.value("subject_id", "")}, {"content_hash", thash},
                                             {"location_hint", t.contains("location_hint") && t["location_hint"].is_string() ? Value(t["location_hint"].get<std::string>()) : Value(std::monostate{})}});
          }
        }
        // the shared vocabulary: copies under one key must agree
        for (const auto& v : c.vocab) {
          if (const auto have = db.one("SELECT sha256, first_subject_id FROM vocabulary WHERE scope = ? AND name = ?", {v.scope, v.name})) {
            if (text(*have, "sha256") != v.sha256)
              throw value_error(c.path.string() + ": its '" + v.name + "' vocabulary (" + v.scope + " scope) differs from the dataset's (first registered by subject " + text(*have, "first_subject_id") + "); two copies of one vocabulary are never merged silently");
          } else {
            db.insert("vocabulary", {{"scope", v.scope}, {"name", v.name}, {"sha256", v.sha256}, {"first_subject_id", c.rec.subject_id}});
          }
        }
        db.execute("DELETE FROM sync_job");
      });
      res.outcome = outcome;
    } catch (const Error& e) {
      res.outcome = "refused";
      res.reason = e.what();
    }
  }
  return results;
}

std::vector<fs::path> scan_packages(const fs::path& folder) {
  std::vector<fs::path> out;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(folder, ec)) {
    const std::string n = e.path().filename().string();
    if (e.is_directory() && n.size() > std::strlen(kPackageSuffix) && n.compare(n.size() - std::strlen(kPackageSuffix), std::string::npos, kPackageSuffix) == 0 &&
        fs::is_regular_file(e.path() / kPackageDatabaseFile))
      out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<CatalogueEntry> refresh_member_status(const fs::path& folder_in) {
  Catalogue cat = Catalogue::open(folder_in);
  std::vector<std::pair<std::string, std::string>> changes;
  std::vector<CatalogueEntry> out = cat.entries();
  for (auto& e : out) {
    if (e.status == "replaced") continue;
    const fs::path p = cat.package_path(e);
    std::string now = "registered";
    if (!fs::is_regular_file(p / kPackageDatabaseFile)) now = "missing";
    else {
      try { if (SubjectPackage::open(p).record().content_hash != e.content_hash) now = "stale"; } catch (const Error&) { now = "stale"; }
    }
    if (now != e.status) { changes.push_back({e.subject_id, now}); e.status = now; }
  }
  if (!changes.empty()) {
    WriterLock lock = acquire_writer_lock(cat.folder());
    Database db = Database::open(cat.folder() / kCatalogueFile);
    db.tx([&] { for (const auto& [id, st] : changes) db.execute("UPDATE subject_entry SET status = ? WHERE subject_id = ?", {st, id}); });
  }
  return out;
}

// ══ selecting subjects ══════════════════════════════════════════════════════════════════════════════════════════

Json SelectedSubjects::provenance() const {
  Json members_json = Json::array();
  for (const auto& m : members) members_json.push_back(Json{{"subject_id", m.subject_id}, {"name", m.name}, {"content_hash", m.content_hash}});
  Json j = Json{{"how", how}, {"members", members_json}};
  if (how == "where") { j["filter"] = filter_text; j["filter_language_version"] = filter_language_version; }
  return j;
}

SelectedSubjects select_subjects(const fs::path& folder, const SubjectSelection& sel) {
  if (sel.subjects.empty() == !sel.where.has_value()) {
    throw value_error(!sel.where && sel.subjects.empty()
                          ? "select subjects: name them (--subjects NAME,NAME,...) or filter them (--where FILTER); there is no 'all subjects' default"
                          : "select subjects: give --subjects or --where, not both");
  }
  Catalogue cat = Catalogue::open(folder);
  SelectedSubjects out;
  if (!sel.subjects.empty()) {
    out.how = "subjects";
    std::string unknown;
    std::set<std::string> seen;
    for (const auto& n : sel.subjects) {
      const auto e = cat.find(n);
      if (!e) { unknown += (unknown.empty() ? "" : ", ") + n; continue; }
      if (seen.insert(e->subject_id).second) out.members.push_back(*e);
    }
    if (!unknown.empty()) throw value_error("select subjects: not in this dataset: " + unknown);
  } else {
    out.how = "where";
    out.filter_text = *sel.where;
    out.filter_language_version = kFilterLanguageVersion;
    const auto variables = participant_variables(cat.folder());
    const SubjectFilter filter = SubjectFilter::parse(*sel.where, variables);
    const auto values = participant_values(cat.folder());
    for (const auto& e : cat.entries()) {
      if (e.status == "replaced" || e.kind != "subject") continue;
      const auto it = values.find(e.subject_id);
      if (filter.selects(it == values.end() ? ParticipantValues{} : it->second)) out.members.push_back(e);
    }
    std::sort(out.members.begin(), out.members.end(), [](const CatalogueEntry& a, const CatalogueEntry& b) { return a.name < b.name; });
  }
  if (out.members.empty()) throw value_error("select subjects: the selection matches no subject");
  std::map<std::string, std::string> template_of;   // "name@version (edition, hash)" -> a member that uses it
  for (const auto& m : out.members) {
    if (m.kind != "subject") throw value_error("select subjects: " + m.name + " is a " + m.kind + ", not a subject: a cohort holds subjects only");
    if (m.status != "registered") throw value_error("select subjects: " + m.name + " is " + m.status + " (run the status check, then register again)");
    const fs::path p = cat.package_path(m);
    if (!fs::is_regular_file(p / kPackageDatabaseFile)) throw value_error("select subjects: " + m.name + ": its package is missing at " + p.string());
    std::string have;
    try { have = SubjectPackage::open(p).record().content_hash; } catch (const Error& e) { throw value_error("select subjects: " + m.name + ": " + e.what()); }
    if (have != m.content_hash) throw value_error("select subjects: " + m.name + " is stale: its package has changed since it was registered (register it again)");
    const Json t = SubjectPackage::open(p).record().template_reference;
    const std::string key = t.is_object() ? t.value("name", "") + "@" + t.value("version", "") + (t.value("edition", "full") == "full" ? "" : ":" + t.value("edition", "")) + " (" + t.value("content_hash", "") + ")" : "no template";
    template_of.emplace(key, m.name);
  }
  if (template_of.size() > 1) {
    std::string list;
    for (const auto& [k, who] : template_of) list += (list.empty() ? "" : "; ") + k + " by " + who;
    throw value_error("select subjects: the members are registered to different template versions (a cohort is never pooled across them): " + list);
  }
  return out;
}

// ══ group packages ══════════════════════════════════════════════════════════════════════════════════════════════

fs::path create_group_package(const fs::path& folder_in, const std::string& group_name, const SelectedSubjects& members, const RegisterOptions& opt) {
  check_name("group", group_name);
  if (members.members.empty()) throw value_error("a group package names the subjects it was computed from");
  Catalogue cat = Catalogue::open(folder_in);
  if (cat.find(group_name)) throw value_error(group_name + ": the dataset already holds a member of that name");
  const fs::path path = cat.folder() / (group_name + kPackageSuffix);
  if (fs::exists(path)) throw Error(code::already_exists, path.string() + ": exists");
  CreatePackageOptions po = cat.package_options();
  po.subject.kind = "group";
  po.subject.created_by = opt.registered_by;
  Json list = Json::array();
  for (const auto& m : members.members) list.push_back(Json{{"subject_id", m.subject_id}, {"name", m.name}, {"content_hash", m.content_hash}});
  {
    DatasetWriter w = create_package(path, group_name, po);
    PackageRecord r = read_package_record(w.db());
    r.members = list;
    write_package_record(w.db(), r);
  }
  const auto res = register_packages(cat.folder(), {path}, opt);
  if (res.empty() || (res[0].outcome != "registered" && res[0].outcome != "unchanged")) throw value_error("the group package was not registered: " + (res.empty() ? std::string() : res[0].reason));
  return path;
}

std::vector<GroupMemberStatus> group_member_status(const fs::path& folder, const std::string& group_name) {
  Catalogue cat = Catalogue::open(folder);
  const auto g = cat.find(group_name);
  if (!g || g->kind != "group") throw Error(code::no_such_row, group_name + ": no group package in this dataset");
  const PackageRecord rec = SubjectPackage::open(cat.package_path(*g)).record();
  std::vector<GroupMemberStatus> out;
  for (const auto& m : rec.members) {
    GroupMemberStatus s;
    s.subject_id = m.value("subject_id", ""); s.name = m.value("name", ""); s.recorded_hash = m.value("content_hash", "");
    if (const auto now = cat.find(s.subject_id); now && now->status != "replaced") s.current_hash = now->content_hash;
    s.current = !s.current_hash.empty() && s.current_hash == s.recorded_hash;
    out.push_back(s);
  }
  return out;
}

}  // namespace nxr::io

namespace nxr::io {

bool is_catalogue_dataset(const std::filesystem::path& dataset_folder) { return std::filesystem::is_regular_file(dataset_folder / kCatalogueFile); }

DatasetWriter open_subject_package(const std::filesystem::path& dataset_folder, const std::string& subject, const OpenOptions& opt, const TemplateSearch& templates) {
  namespace fs = std::filesystem;
  const Catalogue cat = Catalogue::open(dataset_folder);
  const auto entry = cat.find(subject);
  if (!entry) throw Error(code::no_such_row, subject + ": no such subject in the dataset " + dataset_folder.string());
  if (entry->status != "registered") throw Error(code::invalid_argument, subject + ": the catalogue lists it as `" + entry->status + "`; register its package again");
  const fs::path folder = cat.package_path(*entry);
  if (!fs::is_regular_file(folder / kPackageDatabaseFile)) throw Error(code::not_found, subject + ": its package " + folder.string() + " is missing");
  DatasetWriter w = open_package_writer(folder, opt);
  // the template and the registration its rows are anchored to: their arrays are read from their own packages
  const PackageRecord rec = read_package_record(w.db());
  std::vector<std::filesystem::path> members;
  std::map<std::string, std::filesystem::path> member_of;   // subject id -> its package in this dataset
  for (const CatalogueEntry& e : cat.entries()) member_of[e.subject_id] = cat.package_path(e);
  for (const Json* ref : {&rec.template_reference, &rec.registration}) {
    if (!ref->is_object() || !ref->contains("subject_id") || !ref->contains("content_hash")) continue;
    const std::string sid = (*ref)["subject_id"].get<std::string>();
    const auto row = w.db().one("SELECT s.path AS s, d.path AS d FROM subject s JOIN dataset d ON d.id = s.dataset_id WHERE s.id = ?", {sid});
    if (!row || sid == rec.subject_id) continue;
    const Json r = *ref;
    const fs::path local = member_of.count(sid) ? member_of[sid] : fs::path();
    route_keys_to_package(w, text(*row, "d") + "/" + text(*row, "s"), [r, local, templates]() -> std::shared_ptr<SubjectPackage> {
      const std::string hash = r["content_hash"].get<std::string>();
      if (!local.empty() && fs::is_regular_file(local / kPackageDatabaseFile)) {
        auto p = std::make_shared<SubjectPackage>(SubjectPackage::open(local));
        if (p->record().content_hash == hash) return p;
      }
      const ResolvedTemplate t = resolve_template(TemplateRef::from_json(r), templates);
      return std::make_shared<SubjectPackage>(SubjectPackage::open(t.folder));
    });
  }
  return w;
}


ResolvedTemplate resolve_dataset_template(const std::filesystem::path& dataset_folder, const TemplateRef& ref, const TemplateSearch& search) {
  namespace fs = std::filesystem;
  const Catalogue cat = Catalogue::open(dataset_folder);
  for (const CatalogueEntry& e : cat.entries()) {
    if (e.kind != "template" || e.status == "replaced" || e.content_hash != ref.content_hash) continue;
    const fs::path p = cat.package_path(e);
    if (!fs::is_regular_file(p / kPackageDatabaseFile)) continue;
    SubjectPackage pk = SubjectPackage::open(p);
    return ResolvedTemplate{p, pk.record()};
  }
  return resolve_template(ref, search);
}

namespace {
// A package made new holds a copy of the app-wide vocabulary under ids of its own; the rows it is about to carry from another package (its anchors) name that package's copy
// of the same dictionary. The one nothing here refers to yet goes, so the other arrives under its own name and id (a dictionary is found by scope and name).
void drop_unreferenced_vocabulary_held_elsewhere(Database& db, const std::filesystem::path& source) {
  db.execute("ATTACH DATABASE ? AS vsrc", {(source / kPackageDatabaseFile).string()});
  try {
    db.execute("DELETE FROM main.dictionary WHERE id IN (SELECT m.id FROM main.dictionary m JOIN vsrc.dictionary s ON s.scope = m.scope AND coalesce(s.dataset_id, '') = coalesce(m.dataset_id, '') "
               "AND coalesce(s.subject_id, '') = coalesce(m.subject_id, '') AND s.name = m.name AND s.id <> m.id "
               "WHERE NOT EXISTS (SELECT 1 FROM main.selection x WHERE x.dictionary_id = m.id))");
  } catch (...) { try { db.execute("DETACH DATABASE vsrc"); } catch (...) {} throw; }
  db.execute("DETACH DATABASE vsrc");
}
}  // namespace

struct GroupAnalysis::Impl {
  std::filesystem::path dataset, group_folder;
  std::string group_name, registered_by, template_name;
  TemplateSearch search;
  SelectedSubjects selected;
  std::map<std::string, std::unique_ptr<DatasetWriter>> members;
  std::map<std::string, std::filesystem::path> member_folder;   // subject id -> its package folder
  std::unique_ptr<DatasetWriter> group;
};
GroupAnalysis::GroupAnalysis(GroupAnalysis&&) noexcept = default;
GroupAnalysis& GroupAnalysis::operator=(GroupAnalysis&&) noexcept = default;
GroupAnalysis::~GroupAnalysis() = default;

GroupAnalysis GroupAnalysis::open(const std::filesystem::path& dataset_folder, const std::string& group_name, const std::vector<std::string>& subjects, const TemplateSearch& templates,
                                  const std::string& registered_by) {
  namespace fs = std::filesystem;
  GroupAnalysis a;
  a.impl_ = std::make_unique<Impl>();
  Impl& m = *a.impl_;
  m.dataset = dataset_folder; m.group_name = group_name; m.search = templates; m.registered_by = registered_by;
  SubjectSelection sel;
  sel.subjects = subjects;
  m.selected = select_subjects(dataset_folder, sel);
  const Catalogue cat = Catalogue::open(dataset_folder);
  Json tref;
  for (const CatalogueEntry& e : m.selected.members) {
    const fs::path p = cat.package_path(e);
    m.member_folder[e.subject_id] = p;
    const Json r = SubjectPackage::open(p).record().template_reference;
    if (!r.is_object()) throw Error(code::invalid_argument, e.name + ": it is registered to no template, so a group analysis has no common anatomy to put its results on");
    if (tref.is_null()) tref = r;
    else if (r.value("content_hash", "") != tref.value("content_hash", ""))
      throw Error(code::invalid_argument, e.name + ": registered to " + r.value("name", "") + "@" + r.value("version", "") + ", the others to " + tref.value("name", "") + "@" + tref.value("version", "") +
                                          ": members of different template versions are never pooled");
  }
  const ResolvedTemplate tmpl = resolve_dataset_template(dataset_folder, TemplateRef::from_json(tref), templates);
  m.template_name = tmpl.record.name;
  // the group package: made, or the one of this name (a group) opened to replace its results
  if (const auto have = cat.find(group_name)) {
    if (have->kind != "group") throw Error(code::invalid_argument, group_name + ": the dataset holds a member of that name that is not a group package");
    m.group_folder = cat.package_path(*have);
  } else {
    RegisterOptions ro;
    ro.registered_by = registered_by;
    m.group_folder = create_group_package(dataset_folder, group_name, m.selected, ro);
  }
  m.group = std::make_unique<DatasetWriter>(open_package_writer(m.group_folder));
  DatasetWriter& g = *m.group;
  {
    PackageRecord rec = read_package_record(g.db());
    Json list = Json::array();
    for (const auto& e : m.selected.members) list.push_back(Json{{"subject_id", e.subject_id}, {"name", e.name}, {"content_hash", e.content_hash}});
    rec.members = list;
    rec.template_reference = tref;
    write_package_record(g.db(), rec);
  }
  // the default subject's rows the results are on, as anchors: its manifolds, its selections, its canonical frames
  {
    const SubjectPackage tp = SubjectPackage::open(tmpl.folder);
    const std::string tid = tmpl.record.subject_id;
    std::vector<std::pair<std::string, std::string>> roots = {{"subject", tid}};
    for (const auto& r : tp.db().all("SELECT id FROM manifold WHERE subject_id = ? AND status = 'complete'", {tid})) roots.emplace_back("manifold", text(r, "id"));
    for (const auto& r : tp.db().all("SELECT id FROM selection WHERE subject_id = ? AND status = 'complete'", {tid})) roots.emplace_back("selection", text(r, "id"));
    for (const auto& r : tp.db().all("SELECT id FROM field WHERE subject_id = ? AND status = 'complete' AND name LIKE '%_canonical_frames'", {tid})) roots.emplace_back("field", text(r, "id"));
    drop_unreferenced_vocabulary_held_elsewhere(g.db(), tmpl.folder);
    add_anchor_rows(g.db(), tmpl.folder, roots);
    // the default subject's arrays are read from its own package
    if (const auto row = g.db().one("SELECT s.path AS s, d.path AS d FROM subject s JOIN dataset d ON d.id = s.dataset_id WHERE s.id = ?", {tid})) {
      const fs::path tfolder = tmpl.folder;
      route_keys_to_package(g, text(*row, "d") + "/" + text(*row, "s"), [tfolder]() { return std::make_shared<SubjectPackage>(SubjectPackage::open(tfolder)); });
    }
  }
  return a;
}

DatasetWriter& GroupAnalysis::member(const std::string& name) {
  Impl& m = *impl_;
  auto it = m.members.find(name);
  if (it != m.members.end()) return *it->second;
  bool listed = false;
  for (const auto& e : m.selected.members) if (e.name == name) listed = true;
  if (!listed) throw Error(code::no_such_row, name + ": not a member of this group analysis");
  OpenOptions oo;
  oo.lock = false;
  auto w = std::make_unique<DatasetWriter>(open_subject_package(m.dataset, name, oo, m.search));
  return *(m.members[name] = std::move(w));
}
DatasetWriter& GroupAnalysis::group() { return *impl_->group; }
const std::string& GroupAnalysis::template_name() const { return impl_->template_name; }
const SelectedSubjects& GroupAnalysis::selected() const { return impl_->selected; }

void GroupAnalysis::anchor_contributions(const std::vector<std::pair<std::string, std::string>>& subject_and_field) {
  Impl& m = *impl_;
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> by_subject;
  for (const auto& [sid, fid] : subject_and_field) {
    if (!m.member_folder.count(sid)) throw Error(code::invalid_argument, "a contribution names subject " + sid + ", which is not a member of this group analysis");
    by_subject[sid].emplace_back("field", fid);
  }
  for (auto& [sid, roots] : by_subject) {
    roots.insert(roots.begin(), {"subject", sid});
    drop_unreferenced_vocabulary_held_elsewhere(m.group->db(), m.member_folder.at(sid));
    add_anchor_rows(m.group->db(), m.member_folder.at(sid), roots);
  }
}

std::string GroupAnalysis::finish() {
  Impl& m = *impl_;
  m.members.clear();
  m.group->close();
  m.group.reset();
  RegisterOptions ro;
  ro.registered_by = m.registered_by;
  const auto res = register_packages(m.dataset, {m.group_folder}, ro);
  if (res.empty() || res[0].outcome == "refused") throw Error(code::invalid_argument, m.group_name + ": the group package was not registered: " + (res.empty() ? std::string() : res[0].reason));
  return res[0].outcome;
}

}  // namespace nxr::io
