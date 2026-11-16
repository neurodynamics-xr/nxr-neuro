// nxr::bst — a Brainstorm protocol: nxr.convert.protocol, nxr.convert.naming (the stems) and nxr.convert.sources (the source view).
#include "nxr/bst/protocol.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <set>

#include "nxr/bst/mat.h"
#include "nxr/io/errors.h"
#include "nxr/io/surface.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;

namespace {

// ── Python's pathlib, as far as the readers use it ──
bool is_dir(const fs::path& p) { std::error_code ec; return fs::is_directory(p, ec); }
bool is_file(const fs::path& p) { std::error_code ec; return fs::is_regular_file(p, ec); }
bool starts(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

std::string strip(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const auto a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

}  // namespace

std::string py_path(const std::string& p) {
  if (p.empty()) return ".";
  const bool abs = p[0] == '/';
  std::vector<std::string> parts;
  std::size_t i = 0;
  while (i <= p.size()) {
    const auto j = p.find('/', i);
    const std::string c = p.substr(i, j == std::string::npos ? std::string::npos : j - i);
    if (!c.empty() && c != ".") parts.push_back(c);
    if (j == std::string::npos) break;
    i = j + 1;
  }
  std::string out = abs ? "/" : "";
  for (std::size_t k = 0; k < parts.size(); ++k) out += (k ? "/" : "") + parts[k];
  return out.empty() ? "." : out;
}

namespace {

// Path(p).parent.name
std::string parent_name(const std::string& p) {
  const std::string n = py_path(p);
  const auto slash = n.rfind('/');
  if (slash == std::string::npos) return "";
  return name_of(n.substr(0, slash));
}

std::vector<std::string> dir_names(const fs::path& dir, bool dirs_only) {
  std::vector<std::string> out;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    if (dirs_only && !is_dir(it->path())) continue;
    out.push_back(it->path().filename().string());
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> sorted_union(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  std::set<std::string> s(a.begin(), a.end());
  s.insert(b.begin(), b.end());
  return {s.begin(), s.end()};
}

// ── MATLAB values, the way nxr's readers take them ──
std::string sval(const Value* v) {   // _s: the text, trimmed; "" for nothing. A one-cell array is its cell (pymatreader squeezes it)
  if (!v) return "";
  if (v->is_cell() && v->numel() == 1) return sval(&v->cell(0));
  if (!v->is_char()) return "";
  try { return strip(v->string()); } catch (const Error&) { return ""; }
}

// the first element of a numeric value as a number; nullopt for anything else or an empty array
std::optional<double> first_number(const Value* v) {
  if (!v || !v->is_numeric() || v->re.empty()) return std::nullopt;
  return v->scalar();
}
bool truthy(const Value* v) { auto x = first_number(v); return x && *x != 0; }

struct Rec {   // one element of a struct array
  const Value* st = nullptr;
  std::size_t e = 0;
  const Value* get(const char* k) const { return st ? st->field(k, e) : nullptr; }
  explicit operator bool() const { return st != nullptr; }
};
// _records: a MATLAB struct array as its elements (anything else has none)
std::vector<Rec> records(const Value* st) {
  std::vector<Rec> out;
  if (!st || !st->is_struct()) return out;
  for (std::size_t e = 0; e < st->numel(); ++e) out.push_back({st, e});
  return out;
}
const Value* get(const std::map<std::string, Value>& m, const char* k) { auto it = m.find(k); return it == m.end() ? nullptr : &it->second; }

// ── the view ──
struct Recording {
  std::string file, comment, data_type;
  bool bad_trial = false, is_raw_link = false;
  std::optional<std::string> raw_format;
  std::optional<double> duration_s, sfreq;
  std::optional<std::int64_t> n_samples;
};
struct Surface {
  std::string file, comment, type;
  bool present = true;
};
struct Condition {
  std::string name;
  std::vector<Recording> recordings;
  std::vector<std::string> kernels, head_models;
  std::optional<std::string> channel_file;
  std::optional<std::int64_t> n_channels;
  std::vector<std::string> modalities;
  std::optional<std::string> primary_surface;
  std::string importability = "ready";
  std::optional<std::string> discrepancy;
};
struct Subject {
  std::string name;
  std::vector<Condition> conditions;
  std::vector<Surface> surfaces;
  bool plain_surfaces = false;   // the walk lists file names, the index lists entries
  bool anatomy = false;
  std::optional<std::string> cortex, scalp;
  bool uses_default_anat = false;
  std::optional<std::string> discrepancy;
};
struct View {
  std::string root, kind;
  std::vector<Subject> subjects;
  std::optional<double> db_version;
  std::string scanned_at;
  std::optional<double> index_mtime;
  std::optional<std::string> protocol_comment;
};

std::string importability(const Condition& c, const std::string& subject) {
  if (subject == "Group_analysis") return "group-analysis";
  if (c.recordings.empty()) return "no-channels";
  if (!c.channel_file || c.channel_file->empty()) return "no-channels";
  if (c.kernels.empty()) return "missing-kernel";
  if (std::all_of(c.recordings.begin(), c.recordings.end(), [](const Recording& r) { return r.is_raw_link; })) return "ready-raw";
  return "ready";
}

std::optional<double> mtime_of(const fs::path& p) {
  struct stat st;
  if (::stat(p.string().c_str(), &st) != 0) return std::nullopt;
#ifdef __APPLE__
  return static_cast<double>(st.st_mtimespec.tv_sec) + static_cast<double>(st.st_mtimespec.tv_nsec) * 1e-9;
#else
  return static_cast<double>(st.st_mtim.tv_sec) + static_cast<double>(st.st_mtim.tv_nsec) * 1e-9;
#endif
}

std::string now_iso_seconds() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
  gmtime_r(&t, &tm);
  char b[40];
  std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S+00:00", &tm);
  return b;
}

// round(x, 6) as Python rounds: the decimal of the exact binary value
double round6(double x) {
  char b[64];
  std::snprintf(b, sizeof b, "%.6f", x);
  return std::strtod(b, nullptr);
}

// _kernel_surface: the one surface file name the condition's kernels name, nullopt when none or they disagree; an unreadable kernel is unknown
std::optional<std::string> kernel_surface(const fs::path& root, const std::vector<std::string>& kernel_files) {
  std::set<std::string> found;
  for (const auto& rel : kernel_files) {
    const fs::path p = root / "data" / rel;
    if (!is_file(p)) continue;
    try {
      const auto sf = read_variable(p, "SurfaceFile");
      if (sf && sf->is_char()) {
        const std::string s = sf->string();
        if (!s.empty()) found.insert(name_of(s));
      }
    } catch (const Error&) {
      continue;
    }
  }
  if (found.size() == 1) return *found.begin();
  return std::nullopt;
}

// _raw_link_facts: format, sfreq, duration and sample count from the raw link's `F`
struct RawFacts { std::optional<std::string> raw_format; std::optional<double> sfreq, duration_s; std::optional<std::int64_t> n_samples; };
RawFacts raw_link_facts(const fs::path& root, const std::string& rel) {
  RawFacts out;
  const fs::path p = root / "data" / rel;
  if (!is_file(p)) return out;
  std::optional<Value> F;
  try { F = read_variable(p, "F"); } catch (const Error&) { return out; }
  if (!F || !F->is_struct()) return out;
  std::string fmt = sval(F->field("format"));
  if (!fmt.empty()) out.raw_format = fmt;
  const Value* prop = F->field("prop");
  if (prop && prop->is_struct()) {
    std::optional<double> sfreq = first_number(prop->field("sfreq"));
    if (prop->field("sfreq") && prop->field("sfreq")->is_numeric() && prop->field("sfreq")->numel() != 1) sfreq = std::nullopt;   // float(array) of several values fails
    if (sfreq && *sfreq > 0) out.sfreq = sfreq;
    const Value* times = prop->field("times");
    if (times && times->is_numeric() && times->numel() >= 2 && sfreq && *sfreq != 0) {
      const std::vector<double> t = times->doubles();
      const double t0 = t.front(), t1 = t.back();
      if (t1 > t0) {
        out.duration_s = round6(t1 - t0);
        out.n_samples = static_cast<std::int64_t>(std::nearbyint((t1 - t0) * *sfreq)) + 1;
      }
    }
  }
  return out;
}

// The subject's surfaces from the index: list, cortex, scalp, discrepancy (_surfaces_of)
struct SurfacesOf { std::vector<Surface> out; std::optional<std::string> cortex, scalp, discrepancy; };
SurfacesOf surfaces_of(const Rec& rec, const fs::path& fallback_dir) {
  SurfacesOf r;
  const Value* surf = rec.get("Surface");
  std::vector<std::string> files, comments, types;
  if (surf && surf->is_struct()) {
    if (surf->has_field("FileName"))
      for (std::size_t e = 0; e < surf->numel(); ++e) { auto s = sval(surf->field("FileName", e)); if (!s.empty()) files.push_back(s); }
    for (std::size_t e = 0; e < surf->numel(); ++e) {
      if (surf->has_field("Comment")) comments.push_back(sval(surf->field("Comment", e)));
      if (surf->has_field("SurfaceType")) types.push_back(sval(surf->field("SurfaceType", e)));
    }
  }
  if (files.empty()) {
    if (!is_dir(fallback_dir)) return r;
    for (const auto& n : glob_names(fallback_dir, "tess_", ".mat")) r.out.push_back({n, "", "", true});
    return r;
  }
  std::vector<std::string> missing;
  for (std::size_t i = 0; i < files.size(); ++i) {
    const std::string name = name_of(files[i]);
    const bool present = is_file(fallback_dir / name);
    if (!present) missing.push_back(name);
    r.out.push_back({name, i < comments.size() ? comments[i] : "", i < types.size() ? types[i] : "", present});
  }
  auto at = [&](const char* key) -> std::optional<std::string> {
    const Value* v = rec.get(key);
    if (!v || !v->is_numeric() || v->numel() != 1) return std::nullopt;
    const double x = v->scalar();
    if (std::isnan(x) || std::isinf(x)) return std::nullopt;
    const auto idx = static_cast<std::int64_t>(x);
    if (idx >= 1 && idx <= static_cast<std::int64_t>(r.out.size())) return r.out[static_cast<std::size_t>(idx - 1)].file;
    return std::nullopt;
  };
  r.cortex = at("iCortex");
  r.scalp = at("iScalp");
  if (!missing.empty()) {
    std::string d = std::to_string(missing.size()) + " surface(s) listed by the index are not on disk: ";
    for (std::size_t k = 0; k < missing.size() && k < 4; ++k) d += (k ? ", " : "") + missing[k];
    if (missing.size() > 4) d += "\xe2\x80\xa6";
    r.discrepancy = d;
  }
  return r;
}

View read_protocol_mat(const fs::path& root) {
  const auto m = read_variables(root / "data" / "protocol.mat", {"ProtocolInfo", "ProtocolSubjects", "ProtocolStudies", "DbVersion"});
  const Value* subjects_st = get(m, "ProtocolSubjects");
  const Value* subject_arr = subjects_st && subjects_st->is_struct() ? subjects_st->field("Subject") : nullptr;
  const Value* studies_st = get(m, "ProtocolStudies");
  const Value* study_arr = studies_st && studies_st->is_struct() ? studies_st->field("Study") : nullptr;
  const fs::path anat = root / "anat";
  Rec default_rec;
  if (subjects_st && subjects_st->is_struct()) {
    const Value* d = subjects_st->field("DefaultSubject");
    if (d && d->is_struct() && d->numel() >= 1) default_rec = {d, 0};
  }

  std::map<std::string, Subject> by_subject;
  for (const Rec& rec : records(subject_arr)) {
    std::string name = sval(rec.get("Name"));
    if (name.empty()) name = parent_name(sval(rec.get("FileName")));
    if (name.empty()) continue;
    const fs::path sdir = anat / name;
    const bool use_default = [&] { return truthy(rec.get("UseDefaultAnat")); }();
    const Rec src_rec = use_default ? default_rec : rec;
    const fs::path src_dir = use_default ? anat / "@default_subject" : sdir;
    SurfacesOf so = surfaces_of(src_rec, src_dir);
    Subject s;
    s.name = name;
    s.anatomy = !so.out.empty();
    s.surfaces = std::move(so.out);
    s.cortex = so.cortex;
    s.scalp = so.scalp;
    s.uses_default_anat = use_default;
    s.discrepancy = so.discrepancy;
    by_subject[name] = std::move(s);
  }

  for (const Rec& st : records(study_arr)) {
    std::string cond_name = sval(st.get("Condition"));
    if (cond_name.empty()) cond_name = sval(st.get("Name"));
    const std::string subj_file = sval(st.get("BrainStormSubject"));
    const std::string subj = subj_file.empty() ? "" : parent_name(subj_file);
    if (cond_name.empty() || subj.empty() || cond_name == "@default_study" || cond_name == "@intra" || cond_name == "@inter") continue;
    if (starts(cond_name, "@raw")) cond_name = cond_name.substr(4);
    else if (starts(cond_name, "@")) continue;
    if (!by_subject.count(subj)) by_subject[subj].name = subj;
    Subject& sj = by_subject[subj];

    const std::vector<Rec> chan = records(st.get("Channel"));
    const Rec ch0 = chan.empty() ? Rec{} : chan[0];
    std::vector<std::string> mods;
    if (ch0) {
      const Value* mo = ch0.get("Modalities");
      if (mo && mo->is_char()) { try { mods.push_back(mo->string()); } catch (const Error&) {} }
      else if (mo && mo->is_cell()) {
        const bool one = mo->numel() == 1 && mo->cell(0).is_char();
        for (std::size_t i = 0; i < mo->numel(); ++i) {
          if (one) { try { mods.push_back(mo->cell(0).string()); } catch (const Error&) {} }
          else mods.push_back(sval(&mo->cell(i)));
        }
      }
    }

    std::vector<Recording> recs;
    for (const Rec& d : records(st.get("Data"))) {
      Recording r;
      r.file = sval(d.get("FileName"));
      const bool is_raw = r.file.find("data_0raw") != std::string::npos || starts(parent_name(r.file), "@raw");
      r.comment = sval(d.get("Comment"));
      if (r.comment.empty()) r.comment = name_of(r.file);
      r.data_type = sval(d.get("DataType"));
      r.bad_trial = truthy(d.get("BadTrial"));
      r.is_raw_link = is_raw;
      if (is_raw) {
        const RawFacts f = raw_link_facts(root, r.file);
        r.raw_format = f.raw_format;
        r.sfreq = f.sfreq;
        r.duration_s = f.duration_s;
        r.n_samples = f.n_samples;
      }
      recs.push_back(std::move(r));
    }

    std::vector<std::string> kernels, kernel_files, heads;
    for (const Rec& r : records(st.get("Result"))) {
      std::string fn = sval(r.get("FileName"));
      std::string up = fn;
      std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
      if (up.find("KERNEL") == std::string::npos) continue;
      std::string comment = sval(r.get("Comment"));
      kernels.push_back(comment.empty() ? name_of(fn) : comment);
      if (r.get("FileName") && r.get("FileName")->is_char() && !r.get("FileName")->string().empty()) kernel_files.push_back(fn);
    }
    for (const Rec& h : records(st.get("HeadModel"))) {
      std::string comment = sval(h.get("Comment"));
      heads.push_back(comment.empty() ? name_of(sval(h.get("FileName"))) : comment);
    }

    auto it = std::find_if(sj.conditions.begin(), sj.conditions.end(), [&](const Condition& c) { return c.name == cond_name; });
    const bool existing = it != sj.conditions.end();
    Condition fresh;
    fresh.name = cond_name;
    Condition& c = existing ? *it : fresh;
    c.recordings.insert(c.recordings.end(), recs.begin(), recs.end());
    c.kernels = sorted_union(c.kernels, kernels);
    c.head_models = sorted_union(c.head_models, heads);
    if (!c.primary_surface || c.primary_surface->empty()) c.primary_surface = kernel_surface(root, kernel_files);
    if (ch0) {
      if (!c.channel_file || c.channel_file->empty()) c.channel_file = sval(ch0.get("FileName"));
      if (!c.n_channels || *c.n_channels == 0) {
        auto n = first_number(ch0.get("nbChannels"));
        c.n_channels = (n && *n != 0) ? std::optional<std::int64_t>(static_cast<std::int64_t>(*n)) : std::nullopt;
      }
      std::vector<std::string> nz;
      for (const auto& x : mods) if (!x.empty()) nz.push_back(x);
      c.modalities = sorted_union(c.modalities, nz);
    }
    if (!existing) sj.conditions.push_back(std::move(fresh));
  }

  View v;
  v.root = py_path(root.string());
  v.kind = "protocol.mat";
  for (auto& kv : by_subject) {
    for (auto& c : kv.second.conditions) c.importability = importability(c, kv.second.name);
    v.subjects.push_back(std::move(kv.second));
  }
  if (const Value* dv = get(m, "DbVersion")) if (auto x = first_number(dv)) v.db_version = *x;
  if (const Value* info = get(m, "ProtocolInfo")) {
    const std::string comment = info->is_struct() ? sval(info->field("Comment")) : "";
    if (!comment.empty()) v.protocol_comment = comment;
  }
  v.scanned_at = now_iso_seconds();
  v.index_mtime = mtime_of(root / "data" / "protocol.mat");
  return v;
}

View read_walk(const fs::path& root) {
  const fs::path anat = root / "anat";
  View v;
  v.root = py_path(root.string());
  v.kind = "walk";
  for (const Json& s : list_subjects(root)) {
    Subject sj;
    sj.name = s["subject"].get<std::string>();
    for (const Json& c : s["conditions"]) {
      Condition cond;
      cond.name = c["name"].get<std::string>();
      for (const Json& r : c["recordings"]) {
        Recording rec;
        rec.file = r.get<std::string>();
        rec.comment = fs::path(rec.file).stem().string();
        cond.recordings.push_back(rec);
      }
      for (const Json& r : c["raw_links"]) {
        Recording rec;
        rec.file = r.get<std::string>();
        rec.comment = fs::path(rec.file).stem().string();
        rec.is_raw_link = true;
        if (!c["raw_format"].is_null()) rec.raw_format = c["raw_format"].get<std::string>();
        cond.recordings.push_back(rec);
      }
      for (const Json& k : c["kernels"]) cond.kernels.push_back(k.get<std::string>());
      if (!c["channel_files"].empty()) cond.channel_file = c["channel_files"][0].get<std::string>();
      cond.importability = importability(cond, sj.name);
      sj.conditions.push_back(std::move(cond));
    }
    sj.anatomy = is_dir(anat / sj.name);
    sj.plain_surfaces = true;
    for (const Json& f : s["surfaces"]) sj.surfaces.push_back({f.get<std::string>(), "", "", true});
    if (sj.surfaces.empty() && sj.anatomy)
      for (const auto& n : glob_names(anat / sj.name, "tess_", ".mat")) sj.surfaces.push_back({n, "", "", true});
    v.subjects.push_back(std::move(sj));
  }
  v.scanned_at = now_iso_seconds();
  v.index_mtime = mtime_of(root / "data" / "protocol.mat");
  return v;
}

View read_view(const fs::path& root, Reader prefer) {
  if (!is_dir(root / "data")) throw Error(code::not_found, "not a Brainstorm protocol (no data/): " + py_path(root.string()));
  if (prefer == Reader::Walk) return read_walk(root);
  if (prefer == Reader::ProtocolMat) return read_protocol_mat(root);
  if (is_file(root / "data" / "protocol.mat")) return read_protocol_mat(root);
  return read_walk(root);
}

template <class T>
Json opt(const std::optional<T>& v) { return v ? Json(*v) : Json(nullptr); }

Json to_json(const View& v) {
  Json subjects = Json::array();
  for (const Subject& s : v.subjects) {
    Json conds = Json::array();
    for (const Condition& c : s.conditions) {
      Json recs = Json::array();
      for (const Recording& r : c.recordings)
        recs.push_back(Json{{"file", r.file}, {"comment", r.comment}, {"data_type", r.data_type}, {"bad_trial", r.bad_trial}, {"is_raw_link", r.is_raw_link},
                            {"raw_format", opt(r.raw_format)}, {"duration_s", opt(r.duration_s)}, {"sfreq", opt(r.sfreq)}, {"n_samples", opt(r.n_samples)}});
      conds.push_back(Json{{"name", c.name}, {"recordings", recs}, {"kernels", c.kernels}, {"head_models", c.head_models}, {"channel_file", opt(c.channel_file)},
                           {"n_channels", opt(c.n_channels)}, {"modalities", c.modalities}, {"primary_surface", opt(c.primary_surface)},
                           {"importability", c.importability}, {"discrepancy", opt(c.discrepancy)}});
    }
    Json surfs = Json::array();
    for (const Surface& f : s.surfaces) {
      if (s.plain_surfaces) surfs.push_back(f.file);
      else surfs.push_back(Json{{"file", f.file}, {"comment", f.comment}, {"type", f.type}, {"present", f.present}});
    }
    subjects.push_back(Json{{"name", s.name}, {"conditions", conds}, {"surfaces", surfs}, {"anatomy", s.anatomy}, {"cortex", opt(s.cortex)}, {"scalp", opt(s.scalp)},
                            {"uses_default_anat", s.uses_default_anat}, {"discrepancy", opt(s.discrepancy)}});
  }
  return Json{{"root", v.root}, {"kind", v.kind}, {"subjects", subjects}, {"db_version", opt(v.db_version)}, {"scanned_at", v.scanned_at},
              {"index_mtime", opt(v.index_mtime)}, {"protocol_comment", opt(v.protocol_comment)}, {"schema", "nxr.source@1.0"}};
}
}  // namespace

// ── public ──
std::string name_of(const std::string& path) {
  std::string p = py_path(path);
  if (p == "." ) return "";
  const auto slash = p.rfind('/');
  return slash == std::string::npos ? p : p.substr(slash + 1);
}

std::string surface_node_name(const fs::path& surface_file) {
  std::string stem = fs::path(name_of(surface_file.string())).stem().string();
  if (starts(stem, "tess_")) stem = stem.substr(5);
  return nxr::io::sanitize_node_name(stem);
}

std::vector<std::string> glob_names(const fs::path& dir, const std::string& prefix, const std::string& suffix, const std::string& middle) {
  std::vector<std::string> out;
  if (!is_dir(dir)) return out;
  for (const auto& n : dir_names(dir, false)) {
    if (n.size() < prefix.size() + middle.size() + suffix.size()) continue;
    if (!starts(n, prefix) || n.compare(n.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
    if (!middle.empty()) {
      const auto at = n.find(middle, prefix.size());
      if (at == std::string::npos || at + middle.size() + suffix.size() > n.size()) continue;
    }
    out.push_back(n);
  }
  return out;
}

Json list_subjects(const fs::path& root) {
  const fs::path data = root / "data", anat = root / "anat";
  if (!is_dir(data)) throw Error(code::not_found, "not a Brainstorm protocol (no data/): " + py_path(root.string()));
  Json out = Json::array();
  for (const auto& subj : dir_names(data, true)) {
    if (starts(subj, "@")) continue;
    struct Cond { std::string name; std::vector<std::string> recordings, kernels, channel_files, raw_links; std::optional<std::string> raw_format; };
    std::map<std::string, Cond> bases;
    for (const auto& cname : dir_names(data / subj, true)) {
      const bool is_raw = starts(cname, "@raw");
      if (starts(cname, "@") && !is_raw) continue;
      const std::string base = is_raw ? cname.substr(4) : cname;
      Cond& e = bases[base];
      e.name = base;
      const fs::path cond = data / subj / cname;
      const auto kerns = glob_names(cond, "results_", ".mat", "KERNEL");
      const auto chans = glob_names(cond, "channel_", ".mat");
      if (is_raw) {
        const auto raws = glob_names(cond, "data_0raw_", ".mat");
        e.raw_links = raws;
        e.raw_format.reset();
        if (!raws.empty()) {
          try {
            const auto F = read_variable(cond / raws[0], "F");
            if (F && F->is_struct()) {
              const Value* f = F->field("format");
              if (!f) e.raw_format = "None";
              else if (f->is_char()) e.raw_format = f->string();
              else e.raw_format = "[]";
            }
          } catch (const Error&) {
          }
        }
        e.kernels = sorted_union(e.kernels, kerns);
        if (e.channel_files.empty()) e.channel_files = chans;
      } else {
        e.recordings = glob_names(cond, "data_", ".mat");
        e.kernels = sorted_union(e.kernels, kerns);
        if (!chans.empty()) e.channel_files = chans;
      }
    }
    Json conditions = Json::array();
    for (const auto& kv : bases) {
      const Cond& e = kv.second;
      if (e.recordings.empty() && e.kernels.empty() && e.raw_links.empty()) continue;
      conditions.push_back(Json{{"name", e.name}, {"recordings", e.recordings}, {"kernels", e.kernels}, {"channel_files", e.channel_files},
                                {"raw_links", e.raw_links}, {"raw_format", opt(e.raw_format)}});
    }
    if (conditions.empty()) continue;
    out.push_back(Json{{"subject", subj}, {"conditions", conditions}, {"surfaces", glob_names(anat / subj, "tess_", ".mat")}});
  }
  return out;
}

Json read_source(const fs::path& root, Reader prefer) { return to_json(read_view(root, prefer)); }

Json source_counts(const Json& view) {
  std::int64_t conds = 0, recs = 0, kernels = 0, importable = 0;
  for (const Json& s : view["subjects"])
    for (const Json& c : s["conditions"]) {
      ++conds;
      recs += static_cast<std::int64_t>(c["recordings"].size());
      kernels += static_cast<std::int64_t>(c["kernels"].size());
      const std::string i = c["importability"].get<std::string>();
      if (i == "ready" || i == "ready-raw") ++importable;
    }
  return Json{{"subjects", static_cast<std::int64_t>(view["subjects"].size())}, {"conditions", conds}, {"recordings", recs}, {"kernels", kernels}, {"importable", importable}};
}

std::optional<std::string> index_cortex(const fs::path& root, const std::string& subject) {
  try {
    const View v = read_view(root, Reader::ProtocolMat);
    for (const Subject& s : v.subjects) if (s.name == subject) return s.cortex;
  } catch (const std::exception&) {
  }
  return std::nullopt;
}

}  // namespace nxr::bst
