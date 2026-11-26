#include "nxr/io/templates.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

#include "nxr/io/statistic.h"
#include "nxr/io/writer.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {
Error refuse(const std::string& m) { return Error(code::invalid_argument, m); }

void check_component(const std::string& what, const std::string& v) {
  if (v.empty() || v.find_first_of("/\\@:+") != std::string::npos || v == "." || v == "..") throw refuse("invalid template " + what + ": '" + v + "' (no / \\ @ : + and not empty)");
}

fs::path absolute_of(const fs::path& p) {
  std::error_code ec;
  fs::path r = fs::weakly_canonical(fs::absolute(p), ec);
  return ec ? fs::absolute(p).lexically_normal() : r;
}

std::vector<fs::path> split_path_list(const std::string& s) {
#ifdef _WIN32
  const char sep = ';';
#else
  const char sep = ':';
#endif
  std::vector<fs::path> out;
  std::string cur;
  for (char c : s) { if (c == sep) { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

void set_record(const fs::path& package, const std::function<void(PackageRecord&)>& edit) {
  Database db = Database::open(package / kPackageDatabaseFile);
  PackageRecord r = read_package_record(db);
  edit(r);
  write_package_record(db, r);
}

// the keys of a package under a prefix with their digests
std::map<std::string, std::pair<std::int64_t, std::string>> digests_under(const Database& db, const std::string& prefix) {
  std::map<std::string, std::pair<std::int64_t, std::string>> out;
  for (const auto& r : db.all("SELECT key, size_bytes, sha256 FROM key_digest WHERE key = ? OR (key >= ? AND key < ?)", {prefix, prefix + "/", prefix + "0"}))
    out[text(r, "key")] = {integer(r, "size_bytes"), text(r, "sha256")};
  return out;
}

std::string item_label(const PackageItem& i) { return i.table + " " + i.path; }
}  // namespace

Json TemplateRef::to_json() const {
  Json j = Json{{"name", name}, {"edition", edition}, {"version", version}, {"content_hash", content_hash}};
  if (location_hint) j["location_hint"] = *location_hint;
  return j;
}
TemplateRef TemplateRef::from_json(const Json& j) {
  TemplateRef r;
  r.name = j.value("name", ""); r.version = j.value("version", ""); r.edition = j.value("edition", "full"); r.content_hash = j.value("content_hash", "");
  if (j.contains("location_hint") && j["location_hint"].is_string()) r.location_hint = j["location_hint"].get<std::string>();
  return r;
}

TemplateRef parse_template_ref(const std::string& text) {
  TemplateRef r;
  const auto at = text.rfind('@');
  if (at == std::string::npos || at == 0) throw refuse("a template is named NAME@VERSION (or NAME@VERSION:redistributable), not '" + text + "'");
  r.name = text.substr(0, at);
  std::string rest = text.substr(at + 1);
  if (const auto colon = rest.find(':'); colon != std::string::npos) {
    r.edition = rest.substr(colon + 1);
    rest = rest.substr(0, colon);
    if (r.edition != "redistributable" && r.edition != "full") throw refuse("the edition of '" + text + "' is 'full' or 'redistributable', not '" + r.edition + "'");
  }
  r.version = rest;
  check_component("name", r.name);
  check_component("version", r.version);
  return r;
}

std::string template_package_name(const std::string& name, const std::string& version, const std::string& edition) {
  check_component("name", name);
  check_component("version", version);
  if (edition != "full" && edition != "redistributable") throw refuse("the edition is 'full' or 'redistributable', not '" + edition + "'");
  return name + "@" + version + (edition == "redistributable" ? "+redistributable" : "") + kPackageSuffix;
}

fs::path user_settings_file() {
#ifdef _WIN32
  const char* a = std::getenv("APPDATA");
  return fs::path(a ? a : ".") / "nxr" / "settings.json";
#elif defined(__APPLE__)
  const char* h = std::getenv("HOME");
  return fs::path(h ? h : ".") / "Library" / "Application Support" / "nxr" / "settings.json";
#else
  if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x) return fs::path(x) / "nxr" / "settings.json";
  const char* h = std::getenv("HOME");
  return fs::path(h ? h : ".") / ".config" / "nxr" / "settings.json";
#endif
}

std::vector<fs::path> template_search_path(const TemplateSearch& s) {
  std::vector<fs::path> out;
  auto add = [&](const fs::path& p) { if (std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p); };
  for (const auto& p : s.command_paths) add(p);
  if (s.use_environment) if (const char* e = std::getenv(kTemplatePathEnv); e && *e) for (const auto& p : split_path_list(e)) add(p);
  const fs::path settings = s.settings_file.value_or(user_settings_file());
  std::error_code ec;
  if (fs::is_regular_file(settings, ec)) {
    std::ifstream in(settings);
    std::stringstream ss;
    ss << in.rdbuf();
    try {
      const Json j = Json::parse(ss.str());
      if (j.contains("template_path") && j["template_path"].is_array()) for (const auto& p : j["template_path"]) if (p.is_string()) add(p.get<std::string>());
    } catch (const std::exception& e) {
      throw refuse(settings.string() + ": the settings file is not JSON: " + e.what());
    }
  }
  return out;
}

ResolvedTemplate resolve_template(const TemplateRef& ref, const TemplateSearch& search) {
  if (ref.content_hash.empty()) throw refuse("a template reference names its content hash as well as its name and version: " + ref.name + "@" + ref.version);
  std::vector<fs::path> where = template_search_path(search);
  if (ref.location_hint) where.push_back(*ref.location_hint);
  const std::string file = template_package_name(ref.name, ref.version, ref.edition);
  std::string report;
  for (const auto& dir : where) {
    const fs::path cand = dir / file;
    if (!fs::is_regular_file(cand / kPackageDatabaseFile)) { report += "\n  " + dir.string() + ": no " + file; continue; }
    try {
      SubjectPackage p = SubjectPackage::open(cand);
      const PackageRecord rec = p.record();
      if (!rec.release.is_object()) { report += "\n  " + cand.string() + ": not a released package"; continue; }
      if (rec.release.value("name", "") != ref.name || rec.release.value("version", "") != ref.version || rec.release.value("edition", "full") != ref.edition) {
        report += "\n  " + cand.string() + ": released as " + rec.release.value("name", "") + "@" + rec.release.value("version", "") + " (" + rec.release.value("edition", "full") + ")";
        continue;
      }
      if (rec.content_hash != ref.content_hash) { report += "\n  " + cand.string() + ": content hash " + rec.content_hash + ", the reference names " + ref.content_hash; continue; }
      if (p.content_hash() != rec.content_hash) { report += "\n  " + cand.string() + ": its rows no longer hash to its record"; continue; }
      return {absolute_of(cand), rec};
    } catch (const Error& e) {
      report += "\n  " + cand.string() + ": " + e.what();
    }
  }
  std::string searched;
  for (const auto& dir : where) searched += (searched.empty() ? "" : ", ") + dir.string();
  throw Error(code::not_found, "template " + ref.name + "@" + ref.version + (ref.edition == "full" ? "" : ":" + ref.edition) + " with content hash " + ref.content_hash + " not found; searched: " +
                                   (searched.empty() ? "(nowhere: name --templates, set NXR_TEMPLATE_PATH or the user setting template_path)" : searched) + report);
}

std::vector<TemplateListing> list_templates(const TemplateSearch& search) {
  std::vector<TemplateListing> out;
  for (const auto& dir : template_search_path(search)) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) continue;
    std::vector<fs::path> entries;
    for (const auto& e : fs::directory_iterator(dir, ec)) entries.push_back(e.path());
    std::sort(entries.begin(), entries.end());
    for (const auto& p : entries) {
      if (!fs::is_regular_file(p / kPackageDatabaseFile, ec)) continue;
      try {
        const PackageRecord rec = SubjectPackage::open(p).record();
        if (!rec.release.is_object()) continue;
        out.push_back({p, rec.release.value("name", ""), rec.release.value("version", ""), rec.release.value("edition", "full"), rec.kind, rec.content_hash});
      } catch (const Error&) {}
    }
  }
  return out;
}

// ══ release ═════════════════════════════════════════════════════════════════════════════════════════════════════

ReleaseReport release_package(const fs::path& package_in, const ReleaseOptions& opt) {
  check_component("name", opt.name);
  check_component("version", opt.version);
  if (opt.released_by.empty()) throw refuse("release: say who releases it (released_by)");
  const fs::path package = absolute_of(package_in);
  PackageRecord rec;
  {
    SubjectPackage p = SubjectPackage::open(package);
    rec = p.record();
    if (rec.kind != "template" && rec.kind != "registration") throw refuse(package.string() + ": a package of kind '" + rec.kind + "' is not released; only a template or a registration is");
    if (rec.release.is_object()) throw refuse(package.string() + ": already released as " + rec.release.value("name", "") + "@" + rec.release.value("version", ""));
    std::error_code ec;
    if (fs::exists(package / kWriterLockFile, ec)) throw refuse(package.string() + ": a writer holds it");
    if (rec.content_hash.empty() || p.content_hash() != rec.content_hash) throw refuse(package.string() + ": its rows and digests do not hash to its record: open and close it with a writer first");
    const auto missing = items_without_origin(p.db());
    if (!missing.empty()) {
      std::string list;
      for (std::size_t i = 0; i < missing.size() && i < 10; ++i) list += (list.empty() ? "" : ", ") + item_label(missing[i]);
      throw refuse(package.string() + ": " + std::to_string(missing.size()) + " item(s) have no origin record (source, licence, attribution): " + list + (missing.size() > 10 ? ", ..." : ""));
    }
  }
  fs::create_directories(opt.into);
  const fs::path final_dir = absolute_of(opt.into) / template_package_name(opt.name, opt.version, "full");
  if (fs::exists(final_dir)) {
    const PackageRecord have = SubjectPackage::open(final_dir).record();
    if (have.content_hash == rec.content_hash && have.release.is_object()) return {final_dir, rec.content_hash, Json::array()};   // the same release again
    throw refuse(final_dir.string() + ": " + opt.name + "@" + opt.version + " is already released with content hash " + have.content_hash + ": a released version is frozen, a change is a new version");
  }
  const fs::path tmp = final_dir.string() + ".building";
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::copy(package, tmp, fs::copy_options::recursive);
  fs::remove(tmp / kWriterLockFile, ec);
  try {
    set_record(tmp, [&](PackageRecord& r) {
      r.release = Json{{"name", opt.name}, {"edition", "full"}, {"version", opt.version}, {"released_by", opt.released_by}, {"released_utc", now_utc()}, {"program", "nxr-io"}};
    });
    SubjectPackage copy = SubjectPackage::open(tmp);
    const VerifyReport v = copy.verify();
    if (!v.ok || v.content_hash != rec.content_hash) throw Error(code::hash_mismatch, tmp.string() + ": the released copy does not verify against the package it was made from");
  } catch (...) {
    fs::remove_all(tmp, ec);
    throw;
  }
  fs::rename(tmp, final_dir);
  return {final_dir, rec.content_hash, Json::array()};
}

ReleaseReport release_redistributable(const fs::path& full_in, const fs::path& into, const std::string& released_by) {
  if (released_by.empty()) throw refuse("release: say who releases it (released_by)");
  const fs::path full = absolute_of(full_in);
  PackageRecord frec;
  {
    SubjectPackage p = SubjectPackage::open(full);
    frec = p.record();
    if (!frec.release.is_object() || frec.release.value("edition", "full") != "full") throw refuse(full.string() + ": the redistributable edition is made from a released full edition");
    if (frec.kind != "template") throw refuse(full.string() + ": only a template has a redistributable edition");
    const auto missing = items_without_origin(p.db());
    if (!missing.empty()) throw refuse(full.string() + ": " + std::to_string(missing.size()) + " item(s) have no origin record: " + item_label(missing[0]));
  }
  const std::string name = frec.release.value("name", ""), version = frec.release.value("version", "");
  fs::create_directories(into);
  const fs::path final_dir = absolute_of(into) / template_package_name(name, version, "redistributable");
  if (fs::exists(final_dir)) throw refuse(final_dir.string() + ": the redistributable edition of " + name + "@" + version + " is already released: a released version is frozen");
  const fs::path tmp = final_dir.string() + ".building";
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::copy(full, tmp, fs::copy_options::recursive);
  fs::remove(tmp / kWriterLockFile, ec);
  Json left_out = Json::array();
  try {
    set_record(tmp, [&](PackageRecord& r) { r.release = nullptr; });   // being made: a writer may open it
    std::vector<PackageItem> before;
    {
      DatasetWriter w = open_package_writer(tmp);
      before = package_items(w.db());
      SubjectWriter sub = SubjectWriter::open(w, frec.name);
      for (const auto& it : before) {
        const auto o = origin_of(w.db(), it.table, it.id);
        if (!o) throw refuse(item_label(it) + ": no origin record");
        if (!o->redistributable && w.db().read(it.table, it.id)) sub.remove(it.table, it.id);
      }
      for (const char* t : {"manifold", "field", "operator", "selection", "statistic"})
        w.db().execute(std::string("DELETE FROM origin WHERE table_name = '") + t + "' AND row_id NOT IN (SELECT id FROM " + t + ")");
      std::set<std::string> kept;
      for (const auto& it : package_items(w.db())) kept.insert(it.table + "/" + it.id);
      for (const auto& it : before) if (!kept.count(it.table + "/" + it.id)) left_out.push_back(item_label(it));
    }
    PackageRecord kept_rec = SubjectPackage::open(tmp).record();
    // what is kept is the full edition's, byte for byte: every key under a kept item has the same size and SHA-256
    {
      SubjectPackage a = SubjectPackage::open(full), b = SubjectPackage::open(tmp);
      for (const auto& it : package_items(b.db())) {
        const auto x = digests_under(a.db(), it.path), y = digests_under(b.db(), it.path);
        if (x != y) throw Error(code::hash_mismatch, "the redistributable edition's item " + item_label(it) + " is not the full edition's, byte for byte");
      }
    }
    set_record(tmp, [&](PackageRecord& r) {
      r.release = Json{{"name", name}, {"edition", "redistributable"}, {"version", version}, {"released_by", released_by}, {"released_utc", now_utc()}, {"program", "nxr-io"},
                       {"derived_from", frec.content_hash}, {"left_out", left_out}};
    });
    const VerifyReport v = SubjectPackage::open(tmp).verify();
    if (!v.ok) throw Error(code::hash_mismatch, tmp.string() + ": the redistributable edition does not verify");
  } catch (...) {
    fs::remove_all(tmp, ec);
    throw;
  }
  fs::rename(tmp, final_dir);
  return {final_dir, SubjectPackage::open(final_dir).record().content_hash, left_out};
}

// ══ a subject and its template ══════════════════════════════════════════════════════════════════════════════════

std::string template_manifold_id(const ResolvedTemplate& t, const std::string& manifold_name) {
  SubjectPackage p = SubjectPackage::open(t.folder);
  const auto r = p.db().one("SELECT id FROM manifold WHERE subject_id = ? AND name = ? AND status = 'complete'", {t.record.subject_id, manifold_name});
  if (!r) throw Error(code::no_such_row, "template " + t.record.release.value("name", "") + "@" + t.record.release.value("version", "") + " has no manifold '" + manifold_name + "'");
  return text(*r, "id");
}

// ══ a registration between two templates ═══════════════════════════════════════════════════════════════════════

fs::path create_registration_package(const fs::path& path, const RegistrationSpec& spec, const std::function<void(SubjectWriter&, const RegistrationIds&)>& write_operators) {
  if (spec.method.empty()) throw refuse("a registration says what its operators are (method)");
  if (spec.from.name == spec.to.name && spec.from.version == spec.to.version && spec.from.edition == spec.to.edition) throw refuse("a registration links two different templates");
  const ResolvedTemplate from = resolve_template(spec.from, spec.search), to = resolve_template(spec.to, spec.search);
  const std::string from_id = template_manifold_id(from, spec.from_manifold), to_id = template_manifold_id(to, spec.to_manifold);
  CreatePackageOptions po = spec.package;
  po.subject.kind = "registration";
  DatasetWriter w = create_package(path, spec.name, po);
  try {
    add_anchor_rows(w.db(), from.folder, {{"subject", from.record.subject_id}, {"manifold", from_id}});
    add_anchor_rows(w.db(), to.folder, {{"subject", to.record.subject_id}, {"manifold", to_id}});
    PackageRecord rec = read_package_record(w.db());
    TemplateRef a = spec.from, b = spec.to;
    a.content_hash = from.record.content_hash; b.content_hash = to.record.content_hash;
    Json ja = a.to_json(), jb = b.to_json();
    ja["subject_id"] = from.record.subject_id; jb["subject_id"] = to.record.subject_id;
    rec.template_reference = Json{{"from", ja}, {"to", jb}};
    rec.description = Json{{"method", spec.method}, {"warnings", spec.warnings}};
    write_package_record(w.db(), rec);
    SubjectWriter sub = SubjectWriter::open(w, spec.name);
    write_operators(sub, RegistrationIds{from_id, to_id, spec.name});
    sub.complete();
    for (const auto& it : package_items(w.db())) set_origin(w.db(), it.table, it.id, spec.origin);
  } catch (...) {
    const fs::path folder = w.root();
    try { w.close(); } catch (...) {}
    std::error_code ec;
    fs::remove_all(folder, ec);
    throw;
  }
  w.close();
  return path;
}

}  // namespace nxr::io
