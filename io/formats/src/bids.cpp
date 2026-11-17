// BIDS: entities and file names, the folder source, the dataset reader and the inheritance principle. The rules are those of the BIDS specification
// (v1.10); see include/nxr/formats/bids.h.
#include "nxr/formats/bids.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>

#include "nxr/formats/errors.h"

namespace fs = std::filesystem;

namespace nxr::formats::bids {

// ---- entities and file names --------------------------------------------------------------------------------------------------------------

const std::vector<EntityDef>& entity_definitions() {
  static const std::vector<EntityDef> defs = {
      {"sub", "subject"}, {"ses", "session"}, {"sample", "sample"}, {"task", "task"}, {"tissue", "tissue"}, {"acq", "acquisition"}, {"ce", "ceagent"},
      {"trc", "tracer"}, {"stain", "stain"}, {"rec", "reconstruction"}, {"dir", "direction"}, {"run", "run"}, {"mod", "modality"}, {"echo", "echo"},
      {"flip", "flip"}, {"inv", "inversion"}, {"mt", "mtransfer"}, {"part", "part"}, {"proc", "processing"}, {"hemi", "hemisphere"}, {"space", "space"},
      {"split", "split"}, {"recording", "recording"}, {"chunk", "chunk"}, {"atlas", "atlas"}, {"res", "resolution"}, {"den", "density"}, {"label", "label"},
      {"desc", "description"}};
  return defs;
}

int entity_rank(const std::string& key) {
  const auto& d = entity_definitions();
  for (std::size_t i = 0; i < d.size(); ++i) if (key == d[i].key) return static_cast<int>(i);
  return -1;
}

std::optional<std::string> Entities::get(const std::string& key) const {
  for (const auto& kv : items) if (kv.first == key) return kv.second;
  return std::nullopt;
}

void Entities::set(const std::string& key, const std::string& value) {
  for (auto& kv : items) if (kv.first == key) { kv.second = value; return; }
  items.emplace_back(key, value);
}

bool Entities::in_canonical_order() const {
  int last = -1;
  for (const auto& kv : items) {
    const int r = entity_rank(kv.first);
    if (r < 0 || r <= last) return false;
    last = r;
  }
  return true;
}

bool Entities::subset_of(const Entities& other, const std::set<std::string>& ignore) const {
  for (const auto& kv : items) {
    if (ignore.count(kv.first)) continue;
    const auto v = other.get(kv.first);
    if (!v || *v != kv.second) return false;
  }
  return true;
}

static bool alnum(const std::string& s) {
  if (s.empty()) return false;
  for (unsigned char c : s) if (!std::isalnum(c)) return false;
  return true;
}

Entities parse_name(const std::string& name) {
  Entities e;
  const std::size_t dot = name.find('.');
  const std::string stem = name.substr(0, dot);
  if (dot != std::string::npos) e.extension = name.substr(dot);
  if (stem.empty()) throw format_error("BIDS: the file name '" + name + "' has no suffix");
  std::vector<std::string> parts;
  std::size_t st = 0;
  for (;;) {
    const std::size_t u = stem.find('_', st);
    parts.push_back(stem.substr(st, u == std::string::npos ? std::string::npos : u - st));
    if (u == std::string::npos) break;
    st = u + 1;
  }
  const std::string& suffix = parts.back();
  if (!alnum(suffix)) throw format_error("BIDS: the file name '" + name + "' has no valid suffix (letters and digits only)");
  e.suffix = suffix;
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    const std::size_t h = parts[i].find('-');
    if (h == std::string::npos) throw format_error("BIDS: '" + parts[i] + "' in '" + name + "' is not a key-value entity");
    const std::string key = parts[i].substr(0, h), value = parts[i].substr(h + 1);
    if (!alnum(key)) throw format_error("BIDS: the entity key '" + key + "' in '" + name + "' is not letters and digits");
    if (!alnum(value)) throw format_error("BIDS: the value of entity '" + key + "' in '" + name + "' is empty or has a character other than letters and digits");
    if (e.has(key)) throw format_error("BIDS: the entity '" + key + "' is repeated in '" + name + "'");
    e.items.emplace_back(key, value);
  }
  return e;
}

std::string build_name(const Entities& e, bool canonical) {
  std::vector<std::pair<std::string, std::string>> items = e.items;
  if (canonical)
    std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
      const int ra = entity_rank(a.first), rb = entity_rank(b.first);
      return (ra < 0 ? 1000 : ra) < (rb < 0 ? 1000 : rb);
    });
  std::string s;
  for (const auto& kv : items) s += kv.first + "-" + kv.second + "_";
  return s + e.suffix + e.extension;
}

// ---- sources ------------------------------------------------------------------------------------------------------------------------------

DirectorySource::DirectorySource(std::string root) : root_(std::move(root)) {
  std::error_code ec;
  if (!fs::is_directory(root_, ec)) throw io_error("BIDS: '" + root_ + "' is not a folder");
}

static std::string join(const std::string& a, const std::string& b) { return a.empty() ? b : a + "/" + b; }

std::vector<Entry> DirectorySource::list(const std::string& dir) const {
  std::vector<Entry> out;
  std::error_code ec;
  for (const auto& de : fs::directory_iterator(fs::path(root_) / dir, ec)) {
    Entry e;
    e.name = de.path().filename().string();
    e.is_dir = fs::is_directory(de.path(), ec);
    out.push_back(std::move(e));
  }
  if (ec) throw io_error("BIDS: cannot list '" + join(root_, dir) + "': " + ec.message());
  std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) { return a.name < b.name; });
  return out;
}

bool DirectorySource::exists(const std::string& path) const {
  std::error_code ec;
  return fs::exists(fs::path(root_) / path, ec);
}

std::shared_ptr<const ByteSource> DirectorySource::open(const std::string& path) const {
  return std::make_shared<FileSource>((fs::path(root_) / path).string());
}

void MemoryTree::add(const std::string& path, const std::string& content) { files_.emplace_back(path, content); }
void MemoryTree::add_dir(const std::string& path) { dirs_.insert(path); }

std::vector<Entry> MemoryTree::list(const std::string& dir) const {
  std::map<std::string, bool> names;
  const std::string prefix = dir.empty() ? "" : dir + "/";
  auto take = [&](const std::string& p, bool leaf_is_dir) {
    if (p.compare(0, prefix.size(), prefix) != 0 || p.size() == prefix.size()) return;
    const std::string rest = p.substr(prefix.size());
    const std::size_t slash = rest.find('/');
    if (slash == std::string::npos) names[rest] = names[rest] || leaf_is_dir;
    else names[rest.substr(0, slash)] = true;
  };
  for (const auto& f : files_) take(f.first, false);
  for (const auto& d : dirs_) take(d, true);
  std::vector<Entry> out;
  for (const auto& n : names) out.push_back({n.first, n.second});
  return out;
}

bool MemoryTree::exists(const std::string& path) const {
  if (dirs_.count(path)) return true;
  for (const auto& f : files_) if (f.first == path || f.first.compare(0, path.size() + 1, path + "/") == 0) return true;
  return false;
}

std::shared_ptr<const ByteSource> MemoryTree::open(const std::string& path) const {
  for (const auto& f : files_) if (f.first == path) return std::make_shared<MemorySource>(f.second);
  throw io_error("BIDS: no file '" + path + "' in the tree");
}

// ---- the dataset description --------------------------------------------------------------------------------------------------------------

DatasetDescription parse_dataset_description(const nlohmann::json& j) {
  if (!j.is_object()) throw format_error("BIDS: dataset_description.json is not a JSON object");
  DatasetDescription d;
  d.raw = j;
  auto str = [&](const char* k) -> std::optional<std::string> {
    if (!j.contains(k)) return std::nullopt;
    if (!j[k].is_string()) throw format_error(std::string("BIDS: dataset_description.json: '") + k + "' is not a string");
    return j[k].get<std::string>();
  };
  auto list = [&](const char* k) {
    std::vector<std::string> v;
    if (!j.contains(k)) return v;
    if (j[k].is_string()) { v.push_back(j[k].get<std::string>()); return v; }
    if (!j[k].is_array()) throw format_error(std::string("BIDS: dataset_description.json: '") + k + "' is not a list");
    for (const auto& x : j[k]) if (x.is_string()) v.push_back(x.get<std::string>());
    return v;
  };
  const auto name = str("Name"), version = str("BIDSVersion");
  if (!name) throw format_error("BIDS: dataset_description.json has no Name");
  if (!version) throw format_error("BIDS: dataset_description.json has no BIDSVersion");
  d.name = *name;
  d.bids_version = *version;
  d.dataset_type = str("DatasetType");
  d.license = str("License");
  d.how_to_acknowledge = str("HowToAcknowledge");
  d.dataset_doi = str("DatasetDOI");
  d.acknowledgements = str("Acknowledgements");
  if (j.contains("Funding") && j["Funding"].is_string()) d.funding_text = j["Funding"].get<std::string>();   // a string in older datasets; the list form is below
  d.authors = list("Authors");
  d.funding = list("Funding");
  d.ethics_approvals = list("EthicsApprovals");
  d.references_and_links = list("ReferencesAndLinks");
  if (j.contains("GeneratedBy")) d.generated_by = j["GeneratedBy"];
  if (j.contains("SourceDatasets")) d.source_datasets = j["SourceDatasets"];
  return d;
}

// ---- inheritance --------------------------------------------------------------------------------------------------------------------------

static std::vector<std::string> folders_to(const std::string& target) {   // "", "sub-01", "sub-01/ses-a", ... down to the target's folder
  std::vector<std::string> out{""};
  const std::size_t slash = target.rfind('/');
  if (slash == std::string::npos) return out;
  const std::string dir = target.substr(0, slash);
  std::size_t st = 0;
  for (;;) {
    const std::size_t s = dir.find('/', st);
    out.push_back(dir.substr(0, s));
    if (s == std::string::npos) break;
    st = s + 1;
  }
  return out;
}

static std::string base_name(const std::string& path) {
  const std::size_t s = path.rfind('/');
  return s == std::string::npos ? path : path.substr(s + 1);
}

std::vector<std::string> applicable_files(const Source& src, const std::string& target, const std::string& suffix, const std::string& extension, const std::set<std::string>& ignore) {
  Entities te;
  try { te = parse_name(base_name(target)); } catch (const nxr::io::Error&) { return {}; }
  struct Cand { std::size_t depth, n; std::string path; };
  std::vector<Cand> found;
  const auto dirs = folders_to(target);
  for (std::size_t d = 0; d < dirs.size(); ++d) {
    for (const Entry& e : src.list(dirs[d])) {
      if (e.is_dir) continue;
      Entities ce;
      try { ce = parse_name(e.name); } catch (const nxr::io::Error&) { continue; }
      if (ce.suffix != suffix || ce.extension != extension) continue;
      if (!ce.subset_of(te, ignore)) continue;
      found.push_back({d, ce.items.size(), join(dirs[d], e.name)});
    }
  }
  std::stable_sort(found.begin(), found.end(), [](const Cand& a, const Cand& b) { return a.depth != b.depth ? a.depth < b.depth : a.n < b.n; });
  std::vector<std::string> out;
  for (const Cand& c : found) out.push_back(c.path);
  return out;
}

nlohmann::json merged_sidecar(const Source& src, const std::string& target, const std::string& suffix, std::vector<std::string>* used) {
  nlohmann::json merged = nlohmann::json::object();
  for (const std::string& p : applicable_files(src, target, suffix, ".json")) {
    const nlohmann::json j = tsv::read_sidecar(*src.open(p));
    if (!j.is_object()) throw format_error("BIDS: the sidecar '" + p + "' is not a JSON object");
    for (auto it = j.begin(); it != j.end(); ++it) merged[it.key()] = it.value();   // a deeper file's value replaces the whole value
    if (used) used->push_back(p);
  }
  return merged;
}

// ---- the dataset --------------------------------------------------------------------------------------------------------------------------

namespace {

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

bool is_ephys_suffix(const std::string& s) { return s == "meg" || s == "eeg" || s == "ieeg"; }

const std::set<std::string>& primary_extensions() {
  static const std::set<std::string> s = {".fif", ".ds", ".con", ".sqd", ".edf", ".bdf", ".vhdr", ".set", ".cnt", ".CNT", ".cdt", ".nwb", ".mefd", ".pdf", ".lay"};
  return s;
}
const std::set<std::string>& companion_extensions() {
  static const std::set<std::string> s = {".vmrk", ".eeg", ".EEG", ".dat", ".fdt", ".bin"};
  return s;
}

struct Reader {
  const Source& src;
  const ReadOptions& opt;
  Dataset& ds;

  void problem(const std::string& path, const std::string& message) {
    if (opt.strict) throw format_error("BIDS: " + path + ": " + message);
    ds.issues.push_back({path, message});
  }

  TableFile table(const std::string& path) { return {path, tsv::read(*src.open(path))}; }

  std::optional<File> parse_file(const std::string& path, const std::string& datatype) {
    File f;
    f.path = path;
    f.datatype = datatype;
    try {
      f.entities = parse_name(base_name(path));
    } catch (const nxr::io::Error& e) {
      problem(path, e.what());
      return std::nullopt;
    }
    if (!f.entities.in_canonical_order()) problem(path, "the entities are not in the order the specification gives, or one is not a BIDS entity");
    return f;
  }

  Recording make_recording(const std::string& datatype, std::vector<File> parts, std::vector<std::string> companions) {
    Recording r;
    r.datatype = datatype;
    std::stable_sort(parts.begin(), parts.end(), [](const File& a, const File& b) {
      const auto sa = a.entities.get("split").value_or(""), sb = b.entities.get("split").value_or("");
      if (sa.size() != sb.size()) return sa.size() < sb.size();   // numeric order for equal-width and unequal-width counters alike
      return sa < sb;
    });
    r.entities = parts[0].entities;
    r.entities.items.erase(std::remove_if(r.entities.items.begin(), r.entities.items.end(), [](const auto& kv) { return kv.first == "split"; }), r.entities.items.end());
    for (const File& p : parts) r.data_files.push_back(p.path);
    r.companions = std::move(companions);
    const std::string target = parts[0].path;
    r.sidecar = merged_sidecar(src, target, r.entities.suffix, &r.sidecar_files);
    auto last_table = [&](const char* suffix) -> std::optional<TableFile> {
      const auto v = applicable_files(src, target, suffix, ".tsv");
      if (v.empty()) return std::nullopt;
      return table(v.back());
    };
    auto merged_json = [&](const char* suffix) -> std::optional<JsonFile> {
      std::vector<std::string> used;
      nlohmann::json j = merged_sidecar(src, target, suffix, &used);
      if (used.empty()) return std::nullopt;
      return JsonFile{used.back(), std::move(j)};
    };
    r.channels = last_table("channels");
    r.events = last_table("events");
    r.channels_sidecar = merged_json("channels");
    r.events_sidecar = merged_json("events");
    // electrodes and coordinate systems: one per `space` (the most specific of each)
    auto per_space = [&](const char* suffix, const char* ext) {
      std::map<std::string, std::string> by_space;
      for (const std::string& p : applicable_files(src, target, suffix, ext, {"space"})) by_space[parse_name(base_name(p)).get("space").value_or("")] = p;
      std::vector<std::string> out;
      for (const auto& kv : by_space) out.push_back(kv.second);
      return out;
    };
    for (const std::string& p : per_space("electrodes", ".tsv")) r.electrodes.push_back(table(p));
    for (const std::string& p : per_space("coordsystem", ".json")) r.coordsystems.push_back({p, tsv::read_sidecar(*src.open(p))});
    return r;
  }

  Session read_session(const std::string& dir, const std::string& label) {
    Session s;
    s.label = label;
    std::vector<File> data, rest;
    std::vector<std::string> data_dirs;
    for (const Entry& e : src.list(dir)) {
      const std::string path = join(dir, e.name);
      if (e.is_dir && !(e.name.size() > 3 && e.name.compare(e.name.size() - 3, 3, ".ds") == 0)) {
        data_dirs.push_back(e.name);
        continue;
      }
      if (auto f = parse_file(path, "")) {
        if (f->entities.suffix == "scans" && f->entities.extension == ".tsv") s.scans = table(path);
        else rest.push_back(*f);
      }
    }
    std::vector<File> in_folders;
    for (const std::string& dt : data_dirs) {
      for (const Entry& e : src.list(join(dir, dt))) {
        const std::string path = join(join(dir, dt), e.name);
        if (e.is_dir && !(e.name.size() > 3 && e.name.compare(e.name.size() - 3, 3, ".ds") == 0)) {
          problem(path, "a folder inside a datatype folder that is not a data folder (`.ds`)");
          continue;
        }
        if (auto f = parse_file(path, dt)) in_folders.push_back(*f);
      }
    }
    // recordings: the data files of the electrophysiology datatypes, grouped by name without `split`
    std::set<std::string> consumed;
    std::map<std::string, std::vector<File>> groups;   // key: name without `split`
    std::vector<std::string> order;
    for (const File& f : in_folders) {
      if (!is_ephys_suffix(f.entities.suffix) || f.entities.suffix != f.datatype || !primary_extensions().count(f.entities.extension)) continue;
      Entities k = f.entities;
      k.items.erase(std::remove_if(k.items.begin(), k.items.end(), [](const auto& kv) { return kv.first == "split"; }), k.items.end());
      const std::string key = f.datatype + "/" + build_name(k);
      if (!groups.count(key)) order.push_back(key);
      groups[key].push_back(f);
      consumed.insert(f.path);
    }
    for (const std::string& key : order) {
      std::vector<File>& parts = groups[key];
      const File& first = parts[0];
      std::vector<std::string> companions;
      for (const File& f : in_folders) {
        if (consumed.count(f.path) || f.datatype != first.datatype || f.entities.suffix != first.entities.suffix || !companion_extensions().count(f.entities.extension)) continue;
        for (const File& p : parts) {
          Entities a = f.entities, b = p.entities;
          a.extension = b.extension = "";
          if (build_name(a, false) == build_name(b, false)) { companions.push_back(f.path); consumed.insert(f.path); break; }
        }
      }
      s.recordings.push_back(make_recording(first.datatype, parts, companions));
    }
    // the rest of the folder's files: associated metadata is consumed by the recordings; the others are the session's files
    for (const File& f : in_folders) {
      if (consumed.count(f.path)) continue;
      const std::string& sfx = f.entities.suffix;
      const std::string& ext = f.entities.extension;
      const bool metadata = (ext == ".json" && (is_ephys_suffix(sfx) || sfx == "channels" || sfx == "events" || sfx == "electrodes" || sfx == "coordsystem")) ||
                            (ext == ".tsv" && (sfx == "channels" || sfx == "events" || sfx == "electrodes"));
      if (metadata) continue;
      bool attached = false;
      if (sfx == "headshape" || sfx == "photo" || sfx == "markers") {
        for (Recording& r : s.recordings) {
          Entities probe = f.entities;
          Entities rec = r.entities;
          if (f.datatype == r.datatype && probe.subset_of(rec, {"space", "acq"})) { r.other_files.push_back(f.path); attached = true; }
        }
      }
      if (!attached) s.files.push_back(f);
    }
    for (const File& f : rest) s.files.push_back(f);
    return s;
  }

  Subject read_subject(const std::string& label) {
    Subject sub;
    sub.label = label;
    const std::string dir = "sub-" + label;
    std::vector<std::string> sessions;
    bool plain_datatypes = false;
    for (const Entry& e : src.list(dir)) {
      if (e.is_dir && starts_with(e.name, "ses-")) sessions.push_back(e.name);
      else if (e.is_dir) plain_datatypes = true;
    }
    if (!sessions.empty() && plain_datatypes) problem(dir, "datatype folders beside session folders");
    if (sessions.empty()) {
      sub.sessions.push_back(read_session(dir, ""));
    } else {
      for (const std::string& s : sessions) sub.sessions.push_back(read_session(join(dir, s), s.substr(4)));
      for (const Entry& e : src.list(dir))   // the subject's own files (`_sessions.tsv`, a subject-level sidecar)
        if (!e.is_dir)
          if (auto f = parse_file(join(dir, e.name), "")) sub.files.push_back(*f);
    }
    return sub;
  }
};

}  // namespace

Dataset read_dataset(const Source& src, const ReadOptions& options) {
  Dataset ds;
  if (!src.exists("dataset_description.json")) throw format_error("BIDS: not a BIDS dataset: no dataset_description.json at the root");
  ds.description = parse_dataset_description(tsv::read_sidecar(*src.open("dataset_description.json")));
  Reader R{src, options, ds};
  std::vector<std::string> labels;
  for (const Entry& e : src.list("")) {
    if (e.is_dir) {
      if (starts_with(e.name, "sub-")) labels.push_back(e.name.substr(4));
      else if (e.name == "derivatives") {
        for (const Entry& d : src.list("derivatives")) if (d.is_dir) ds.derivatives.push_back("derivatives/" + d.name);
      } else if (e.name != "sourcedata" && e.name != "code" && e.name != "stimuli" && e.name != "phenotype" && e.name != "logs" && e.name[0] != '.') {
        ds.issues.push_back({e.name, "an unexpected folder at the dataset root"});
      }
      continue;
    }
    if (e.name == "dataset_description.json") continue;
    if (e.name == "participants.tsv") ds.participants = R.table("participants.tsv");
    else if (e.name == "participants.json") ds.participants_sidecar = JsonFile{"participants.json", tsv::read_sidecar(*src.open("participants.json"))};
    else ds.top_level_files.push_back(e.name);
  }
  if (!ds.participants_sidecar && ds.participants && src.exists("participants.json")) ds.participants_sidecar = JsonFile{"participants.json", tsv::read_sidecar(*src.open("participants.json"))};
  for (const std::string& l : labels) {
    if (!options.subjects.empty() && std::find(options.subjects.begin(), options.subjects.end(), l) == options.subjects.end()) continue;
    ds.subjects.push_back(R.read_subject(l));
  }
  return ds;
}

}  // namespace nxr::formats::bids
