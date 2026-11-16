#include "nxr/io/compact.h"

#include <algorithm>
#include <cstring>
#include <filesystem>

#include "chunking.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/database.h"
#include "nxr/io/errors.h"
#include "nxr/io/small_file_store.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {

constexpr const char* kStaging = ".compact";
constexpr const char* kVerified = ".verified";
constexpr std::int64_t kSlabBytes = 128LL * 1024 * 1024;   // a slab of the array held at a time (three copies at most: old, new, read back)

struct Counts {
  std::int64_t files = 0, folders = 0;
  Counts& operator+=(const Counts& o) { files += o.files; folders += o.folders; return *this; }
};

// the files and folders strictly under `prefix`
Counts count_under(const Store& store, const std::string& prefix) {
  Counts c;
  const Listing l = store.list(prefix);
  c.files += static_cast<std::int64_t>(l.keys.size());
  for (const auto& p : l.prefixes) {
    ++c.folders;
    c += count_under(store, join_key(prefix, p));
  }
  return c;
}

struct Layout {
  ArrayMeta meta;      // as the file states it
  Json doc;
  bool sharded = false;
};

ArrayMeta meta_of(const Json& doc) {
  ArrayMeta m;
  m.shape = doc["shape"].get<Shape>();
  m.data_type = doc["data_type"].get<std::string>();
  m.grid = doc["chunk_grid"]["configuration"]["chunk_shape"].get<Shape>();
  m.codecs = doc["codecs"];
  m.fill_value = doc["fill_value"];
  m.separator = doc.contains("chunk_key_encoding") ? doc["chunk_key_encoding"].value("configuration", Json::object()).value("separator", "/") : "/";
  m.in_pieces = doc.contains("nxr_written_in_pieces");
  m.chunks = m.grid;
  if (m.codecs.size() == 1 && m.codecs[0]["name"] == "sharding_indexed") m.chunks = m.codecs[0]["configuration"]["chunk_shape"].get<Shape>();
  return m;
}

bool compressed(const ArrayMeta& m) {
  const Json* chain = &m.codecs;
  if (m.codecs.size() == 1 && m.codecs[0]["name"] == "sharding_indexed") chain = &m.codecs[0]["configuration"]["codecs"];
  for (const auto& c : *chain) if (c["name"] == "blosc") return true;
  return false;
}

// the attributes text of an array's zarr.json, verbatim (it is the canonical `...,"attributes":<text>}`)
std::string attributes_of(const std::string& raw, const Json& doc) {
  static const std::string key = "\"attributes\":";
  const auto at = raw.find(key);
  if (at != std::string::npos && !raw.empty() && raw.back() == '}') return raw.substr(at + key.size(), raw.size() - 1 - (at + key.size()));
  return doc.contains("attributes") ? js_json(doc["attributes"]) : std::string("{}");
}

// the new layout of an array: the same values, the separator ".", an array over 4 MiB sharded by the default policy
ArrayMeta new_meta_of(const ArrayMeta& old, bool shard) {
  ArrayMeta m = old;
  m.separator = ".";
  const bool was_sharded = old.chunks != old.grid;
  if (shard && !was_sharded && !old.in_pieces) {   // an array written in pieces stays unsharded: `write_region` refuses a sharded one
    if (const auto sh = default_shards(old.shape, old.chunks, old.itemsize())) {
      m.grid = *sh;
      ArrayOptions o;
      o.chunks = old.chunks;
      o.compress = compressed(old);
      o.shards = sh;
      const ArrayMeta made = array_meta(dtype_from_string(old.data_type), old.shape, o);
      m.codecs = made.codecs;
    }
  }
  return m;
}

std::int64_t grid_cells(const ArrayMeta& m) {
  std::int64_t n = 1;
  for (std::size_t d = 0; d < m.shape.size(); ++d) n *= (m.shape[d] + m.grid[d] - 1) / m.grid[d];
  return m.shape.empty() ? 1 : n;
}

std::int64_t files_in(const Store& store, const std::string& at) {   // the chunk files of an array, of either separator
  std::int64_t n = 0;
  const Listing l = store.list(at);
  for (const auto& k : l.keys) if (k.rfind("c.", 0) == 0 || k == "c") ++n;
  if (store.exists_prefix(join_key(at, "c")) && !store.exists(join_key(at, "c"))) n += count_under(store, join_key(at, "c")).files;
  return n;
}

struct Row_ {
  std::string table, id;
  std::string codecs_json, chunk_shape_json;
};

struct Work {
  DatasetWriter& ds;
  const CompactOptions& opt;
  Store& store;
};

void stage(const Work& w, const char* s, const std::string& at) {
  if (w.opt.on_stage) w.opt.on_stage(s, at);
}

// write the array's new layout into the staging key, slab by slab, reading each slab back; true when every byte equals the old array's
bool write_and_verify(const Work& w, const std::string& at, const std::string& staging, const ArrayMeta& nm, bool dense, std::int64_t old_files) {
  Store& store = w.store;
  layout_array(store, staging, nm);
  const ArrayReader oldr = ArrayReader::open(store, at);
  if (nm.shape.empty() || detail::product(nm.shape) == 0) return true;
  const ArrayReader newr = ArrayReader::open(store, staging);
  SlabWriter sw(store, staging, 0);
  const std::int64_t rowbytes = std::max<std::int64_t>(1, detail::product(Shape(nm.shape.begin() + 1, nm.shape.end())) * static_cast<std::int64_t>(nm.itemsize()));
  const std::int64_t cell = nm.grid[0];
  const std::int64_t per = std::max<std::int64_t>(1, kSlabBytes / std::max<std::int64_t>(1, cell * rowbytes));
  const std::int64_t step = cell * per;
  const DType dt = dtype_from_string(nm.data_type);
  Shape lo(nm.shape.size(), 0), hi(nm.shape);
  for (std::int64_t start = 0; start < nm.shape[0]; start += step) {
    const std::int64_t stop = std::min(nm.shape[0], start + step);
    lo[0] = start;
    hi[0] = stop;
    const Bytes want = oldr.read(lo, hi);
    Shape sh = nm.shape;
    sh[0] = stop - start;
    sw.write(start, ArrayView{want.data(), dt, sh}, dense);
    const Bytes got = newr.read(lo, hi);
    if (got.size() != want.size() || (!want.empty() && std::memcmp(got.data(), want.data(), want.size()) != 0)) return false;
  }
  // an old array that kept its all-fill chunks (written dense) keeps them: fewer files than the old layout had means the chunks it lacked were stored
  if (!dense && files_in(store, staging) < old_files && nm.chunks == nm.grid) return false;
  return true;
}

enum class Done { Compacted, Already, Skipped };

struct ArrayResult {
  Done done = Done::Skipped;
  Counts before, after;
};

ArrayResult compact_array(const Work& w, const std::string& at, const std::optional<Row_>& row, bool& stale_row) {
  Store& store = w.store;
  const std::string zkey = join_key(at, "zarr.json");
  const std::string raw = store.get_text(zkey).value_or("");
  const Json doc = Json::parse(raw);
  const ArrayMeta old = meta_of(doc);
  const std::string attrs = attributes_of(raw, doc);
  const std::string staging = join_key(at, kStaging);
  ArrayResult r;
  r.before = count_under(store, at);
  const bool scalar = old.shape.empty();

  if (old.separator == ".") {
    // already format 51; remove what a crash after the commit left (the old c/ folder: an array of rank 0 has the file `c` instead), and bring the row to the file
    r.done = Done::Already;
    if (!w.opt.dry_run) {
      if (!scalar && store.exists_prefix(join_key(at, "c"))) store.erase_prefix(join_key(at, "c"));
      if (store.exists_prefix(staging)) store.erase_prefix(staging);
    }
    r.after = w.opt.dry_run ? r.before : count_under(store, at);
    if (row && (row->codecs_json != jdump(old.codecs) || row->chunk_shape_json != jdump(Json(old.chunks)))) stale_row = true;
    return r;
  }

  const ArrayMeta nm = new_meta_of(old, w.opt.shard);
  r.done = Done::Compacted;
  if (w.opt.dry_run) {
    r.after = Counts{1 + (scalar ? 1 : grid_cells(nm)), 0};
    return r;
  }

  if (scalar) {
    // its one chunk is the file `c` under either separator: only the declaration changes
    const ArrayReader oldr = ArrayReader::open(store, at);
    const Bytes want = oldr.read();
    Json nd = doc;
    nd["chunk_key_encoding"]["configuration"]["separator"] = ".";
    store.set(zkey, zarr_json_text(nd, attrs));
    const Bytes got = ArrayReader::open(store, at).read();
    if (got != want) {
      store.set(zkey, raw);
      throw Error(code::io, at + ": the values differ after the layout changed (kept as it was)");
    }
    r.after = count_under(store, at);
    return r;
  }

  const std::int64_t old_files = files_in(store, at);
  const bool resume = store.exists(join_key(staging, kVerified));
  if (!resume) {
    // a first or restarted attempt: what an earlier one left is not trusted
    if (store.exists_prefix(staging)) store.erase_prefix(staging);
    for (const auto& k : store.list(at).keys) if (k.rfind("c.", 0) == 0) store.erase(join_key(at, k));   // the old layout's chunks are c/…: these are orphans
    bool ok = write_and_verify(w, at, staging, nm, false, old_files);
    if (!ok) {
      store.erase_prefix(staging);
      ok = write_and_verify(w, at, staging, nm, true, old_files);   // the chunks that hold only the fill value are stored too
    }
    if (!ok) {
      store.erase_prefix(staging);
      throw Error(code::io, at + ": the new layout's values differ from the old (the array is kept as it was)");
    }
    stage(w, "written", at);
    store.set(join_key(staging, kVerified), std::string("ok"));
    stage(w, "verified", at);
  }

  // the swap: the new chunk files move up beside the old c/ folder (other keys), then zarr.json is written (the commit), then the old layout goes
  const Json ndoc = Json::parse(store.get_text(join_key(staging, "zarr.json")).value());
  for (const auto& k : store.list(staging).keys) {
    if (k.rfind("c.", 0) != 0 && k != "c") continue;
    store.rename_prefix(join_key(staging, k), join_key(at, k));
  }
  stage(w, "moved", at);
  store.set(zkey, zarr_json_text(ndoc, attrs));
  stage(w, "committed", at);
  store.erase_prefix(join_key(at, "c"));
  store.erase_prefix(staging);
  r.after = count_under(store, at);
  const ArrayMeta now = meta_of(ndoc);
  if (row && (row->codecs_json != jdump(now.codecs) || row->chunk_shape_json != jdump(Json(now.chunks)))) stale_row = true;
  return r;
}

// the arrays under `prefix`: a folder with a zarr.json of node_type "array"; links are not followed
void walk(const Store& store, const std::string& prefix, std::vector<std::string>& arrays, std::vector<std::string>& links) {
  const Listing l = store.list(prefix);
  if (std::find(l.keys.begin(), l.keys.end(), "zarr.json") != l.keys.end()) {
    const auto raw = store.get_text(join_key(prefix, "zarr.json"));
    if (raw) {
      Json d = Json::parse(*raw, nullptr, false);
      if (!d.is_discarded() && d.value("node_type", "") == "array") {
        arrays.push_back(prefix);
        return;
      }
    }
  }
  for (const auto& p : l.prefixes) {
    if (!p.empty() && p[0] == '.') continue;
    const std::string child = join_key(prefix, p);
    if (store.is_link(child)) { links.push_back(child); continue; }
    walk(store, child, arrays, links);
  }
}

// ── format 52: a subject's small keys move into its small_files.sqlite ──

// the files and folders on the disk under `dir` (what a cluster's file system counts; the small-file database is one file)
Counts disk_counts(const fs::path& dir) {
  Counts c;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return c;
  for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec), end; it != end; it.increment(ec)) {
    if (ec) break;
    if (it->is_symlink(ec) || !it->is_directory(ec)) ++c.files;
    else ++c.folders;
  }
  return c;
}

// the keys of a subject that are files and under the threshold (links are left as they are), relative to the subject's folder
void small_file_keys(const Store& files, const std::string& at, std::uint64_t threshold, std::vector<std::string>& out) {
  const Listing l = files.list(at);
  for (const auto& k : l.keys) {
    if (at.empty() && std::string(k).rfind(kSmallFilesFile, 0) == 0) continue;
    const std::string key = join_key(at, k);
    if (const auto n = files.size(key); n && *n < threshold) out.push_back(key);
  }
  for (const auto& p : l.prefixes) {
    const std::string child = join_key(at, p);
    if (files.is_link(child)) continue;
    small_file_keys(files, child, threshold, out);
  }
}

bool remove_empty_folders(const fs::path& dir, bool keep_self) {   // true when `dir` held nothing (and is gone, unless kept)
  std::error_code ec;
  bool empty = true;
  std::vector<fs::path> children;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) children.push_back(it->path());
  for (const auto& c : children) {
    if (!fs::is_symlink(fs::symlink_status(c, ec)) && fs::is_directory(c, ec)) { if (!remove_empty_folders(c, false)) empty = false; }
    else empty = false;
  }
  if (empty && !keep_self) fs::remove(dir, ec);
  return empty;
}

struct SmallMove { std::int64_t keys = 0, bytes = 0; };

// MOVE: every small file key of the subject goes into its table, each batch read back and compared with its file BYTE FOR BYTE before the file is removed;
// a crash leaves a key in both places (the same bytes: the table is read first), and the next run does the rest.
SmallMove move_small_keys(const Work& w, const std::string& base, bool dry_run) {
  DatastoreStore& ds = w.ds.store();
  FolderStore& files = ds.files(base);
  std::vector<std::string> keys;
  small_file_keys(files, "", kSmallFileThreshold, keys);
  SmallMove r;
  if (dry_run) {
    for (const auto& k : keys) { ++r.keys; r.bytes += static_cast<std::int64_t>(files.size(k).value_or(0)); }
    return r;
  }
  if (keys.empty()) { remove_empty_folders(files.root(), true); return r; }
  SmallFileStore& small = *ds.small_files(base, true);
  constexpr std::int64_t kBatchBytes = 64LL * 1024 * 1024;
  constexpr std::size_t kBatchKeys = 1024;
  bool first = true;
  for (std::size_t at = 0; at < keys.size();) {
    std::vector<std::pair<std::string, Bytes>> batch;
    std::int64_t held = 0;
    while (at < keys.size() && batch.size() < kBatchKeys && held < kBatchBytes) {
      Bytes b = files.get(keys[at]).value();
      held += static_cast<std::int64_t>(b.size());
      batch.emplace_back(keys[at], std::move(b));
      ++at;
    }
    small.batch([&] { for (const auto& [k, b] : batch) small.put_row(k, b.data(), b.size()); });
    for (const auto& [k, b] : batch) {
      const auto got = small.get(k);
      const auto file = files.get(k);
      if (!got || !file || *got != b || *file != b) throw Error(code::io, join_key(base, k) + ": the small-file store holds other bytes than its file (the file is kept)");
    }
    if (first) { stage(w, "small_files_written", base); first = false; }
    for (const auto& [k, b] : batch) { files.erase(k); ++r.keys; r.bytes += static_cast<std::int64_t>(b.size()); }
  }
  stage(w, "small_files_removed", base);
  remove_empty_folders(files.root(), true);
  small.close();   // at rest: the database is one file (no WAL), as the report's counts want
  return r;
}

bool has_table(const Database& db, const std::string& t) {
  return db.one("SELECT 1 AS x FROM sqlite_master WHERE type = 'table' AND name = ?", {t}).has_value();
}

}  // namespace

Json compact_dataset(DatasetWriter& ds, const CompactOptions& opt) {
  Database& db = ds.db();
  Store& store = ds.store();
  const Work w{ds, opt, store};
  std::vector<Row> subjects = db.all("SELECT id, name FROM subject WHERE dataset_id = ? ORDER BY name", {ds.id()});
  if (opt.subject) {
    subjects.erase(std::remove_if(subjects.begin(), subjects.end(), [&](const Row& r) { return text(r, "name") != *opt.subject; }), subjects.end());
    if (subjects.empty()) throw Error(code::no_such_row, "compact: no subject named '" + *opt.subject + "'");
  }
  Json out = Json{{"stage", "result"}, {"dry_run", opt.dry_run}, {"subjects", Json::array()}};
  Counts before, after, disk_before, disk_after;
  std::int64_t small_moved = 0;
  std::int64_t n_arrays = 0, n_compacted = 0, n_already = 0, n_links = 0;
  for (const Row& s : subjects) {
    const std::string base = store_path_of(db, text(s, "id"));
    std::vector<std::string> arrays, links;
    walk(store, base, arrays, links);
    // the rows that name an array (a node's own array: its codecs and chunk shape are columns of the row)
    std::map<std::string, Row_> rows;
    for (const char* t : {"field", "operator", "statistic"}) {
      if (!has_table(db, t)) continue;
      for (const Row& r : db.all(std::string("SELECT id, path, codecs_json, chunk_shape_json FROM ") + t + " WHERE subject_id = ? AND path IS NOT NULL AND data_type IS NOT NULL", {text(s, "id")}))
        rows[join_key(base, text(r, "path"))] = Row_{t, text(r, "id"), text(r, "codecs_json"), text(r, "chunk_shape_json")};
    }
    Counts sb, sa;
    const Counts db_ = disk_counts(ds.root() / base);
    std::int64_t compacted = 0, already = 0;
    std::vector<std::pair<std::string, Row_>> stale;
    for (const std::string& at : arrays) {
      const auto it = rows.find(at);
      const std::optional<Row_> row = it == rows.end() ? std::nullopt : std::optional<Row_>(it->second);
      bool stale_row = false;
      const ArrayResult r = compact_array(w, at, row, stale_row);
      sb += r.before;
      sa += r.after;
      (r.done == Done::Compacted ? compacted : already) += 1;
      if (stale_row) stale.emplace_back(at, *row);
    }
    if (!opt.dry_run && !stale.empty()) {
      // the rows follow the files: codecs and chunk shape from the array's zarr.json, then the node's zarr.json as the view writes it
      db.tx([&] {
        for (const auto& [at, row] : stale) {
          const ArrayMeta m = meta_of(Json::parse(store.get_text(join_key(at, "zarr.json")).value()));
          db.execute(std::string("UPDATE ") + row.table + " SET codecs_json = ?, chunk_shape_json = ? WHERE id = ?", {jdump(m.codecs), jdump(Json(m.chunks)), row.id});
        }
      });
      for (const auto& [at, row] : stale) {
        const std::string zkey = join_key(at, "zarr.json");
        const Json doc = Json::parse(store.get_text(zkey).value());
        if (const auto txt = attributes_text(db, row.table, row.id)) store.set(zkey, zarr_json_text(doc, *txt));
      }
    }
    // format 52: the small keys into the subject's table
    SmallMove sm;
    if (opt.small_files) sm = move_small_keys(w, base, opt.dry_run);
    const Counts da = opt.dry_run ? Counts{db_.files - sm.keys + (sm.keys > 0 && !fs::exists(ds.root() / base / kSmallFilesFile) ? 1 : 0), 0} : disk_counts(ds.root() / base);
    small_moved += sm.keys;
    disk_before += db_;
    disk_after += da;
    n_arrays += static_cast<std::int64_t>(arrays.size());
    n_compacted += compacted;
    n_already += already;
    n_links += static_cast<std::int64_t>(links.size());
    before += sb;
    after += sa;
    out["subjects"].push_back(Json{{"subject", text(s, "name")}, {"arrays", arrays.size()}, {"to_compact", compacted}, {"already_format_51", already}, {"links_skipped", links},
                                   {"files_before", sb.files}, {"folders_before", sb.folders}, {"files_after", sa.files}, {"folders_after", sa.folders},
                                   {"small_keys_moved", sm.keys}, {"small_bytes_moved", sm.bytes},
                                   {"files_on_disk_before", db_.files}, {"folders_on_disk_before", db_.folders}, {"files_on_disk_after", da.files}, {"folders_on_disk_after", opt.dry_run ? Json() : Json(da.folders)}});
  }
  // meta.store_format: set when no array of the dataset is left in format 50 (every subject, not only the ones this run named)
  bool all51 = !opt.dry_run;
  if (all51) {
    for (const Row& s : db.all("SELECT id FROM subject WHERE dataset_id = ?", {ds.id()})) {
      std::vector<std::string> arrays, links;
      walk(store, store_path_of(db, text(s, "id")), arrays, links);
      for (const auto& at : arrays) if (meta_of(Json::parse(store.get_text(join_key(at, "zarr.json")).value())).separator != ".") { all51 = false; break; }
      if (!all51) break;
    }
    if (all51 && store_format_number(db) < 51)
      db.execute("INSERT INTO meta (key, value) VALUES ('store_format', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value", {std::string(kStoreFormat51)});
  }
  // ... and format 52 when, besides, no small key of any subject is left as a file (a dataset made at 52 is there already)
  bool all52 = false;
  if (all51 && opt.small_files) {
    all52 = true;
    for (const Row& s : db.all("SELECT id FROM subject WHERE dataset_id = ?", {ds.id()})) {
      std::vector<std::string> keys;
      small_file_keys(ds.store().files(store_path_of(db, text(s, "id"))), "", kSmallFileThreshold, keys);
      if (!keys.empty()) { all52 = false; break; }
    }
    if (all52 && store_format_number(db) < 52) {
      db.execute("INSERT INTO meta (key, value) VALUES ('store_format', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value", {std::string(kStoreFormat52)});
      ds.store().set_small_files(text(ds.row(), "path"), true);
    }
  }
  out["arrays"] = n_arrays;
  out["compacted"] = n_compacted;
  out["already_format_51"] = n_already;
  out["links_skipped"] = n_links;
  out["files_before"] = before.files;
  out["folders_before"] = before.folders;
  out["files_after"] = after.files;
  out["folders_after"] = after.folders;
  out["files_after_is_an_upper_bound"] = opt.dry_run;
  out["store_format_set"] = all51;
  out["small_keys_moved"] = small_moved;
  out["files_on_disk_before"] = disk_before.files;
  out["folders_on_disk_before"] = disk_before.folders;
  out["files_on_disk_after"] = disk_after.files;
  out["folders_on_disk_after"] = opt.dry_run ? Json() : Json(disk_after.folders);
  out["store_format"] = store_format_number(db);
  return out;
}

}  // namespace nxr::io
