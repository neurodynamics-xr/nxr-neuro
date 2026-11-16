// nxr-neuro read — read an nxr dataset: list its subjects and every array its rows name (shape · dtype · CRC-32 of the
// decoded little-endian bytes), or read one array / region. (Step 1's `nxr-read`, folded into the one command.)
//
//   nxr-neuro read <dataset folder>
//   nxr-neuro read <dataset folder> --array <subject> <path> [--region a:b,c:d,...] [--out FILE]
#include <zlib.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include "nxr/io/array_reader.h"
#include "nxr/io/dataset.h"
#include "nxr/io/small_file_store.h"
#include "nxr/io/store.h"

namespace fs = std::filesystem;
using namespace nxr::io;

namespace {

std::string shape_text(const std::vector<std::int64_t>& s) {
  std::string t = "[";
  for (std::size_t i = 0; i < s.size(); ++i) t += (i ? "," : "") + std::to_string(s[i]);
  return t + "]";
}

std::string crc_text(const std::vector<std::uint8_t>& b) {
  uLong c = crc32(0L, Z_NULL, 0);
  for (std::size_t o = 0; o < b.size(); o += 1u << 30) c = crc32(c, b.data() + o, static_cast<uInt>(std::min<std::size_t>(1u << 30, b.size() - o)));
  char buf[16];
  std::snprintf(buf, sizeof buf, "%08lx", static_cast<unsigned long>(c));
  return buf;
}

// The dataset's keys (`<subject>/<array>/…`) as its format keeps them: files, or (format 52) a subject's small keys in its small_files.sqlite. Read only.
struct DatasetKeys {
  static fs::path normal(const fs::path& p) { fs::path q = fs::absolute(p).lexically_normal(); if (!q.has_filename()) q = q.parent_path(); return q; }
  explicit DatasetKeys(const fs::path& dataset_folder)
      : folder(normal(dataset_folder)), datastore(folder.parent_path(), SmallFileOptions{kSmallFileThreshold, true}), keys(datastore, folder.filename().string()) {}
  fs::path folder;
  DatastoreStore datastore;
  PrefixStore keys;
};


bool is_array(const Store& store, const std::string& key) {
  const std::string t = store.get_text(join_key(key, "zarr.json")).value_or("");
  return t.find("\"node_type\":\"array\"") != std::string::npos || t.find("\"node_type\": \"array\"") != std::string::npos;
}

// An array, or the arrays inside a group (a sparse operator or field is a group of data · indices · indptr).
void collect(const Store& store, const std::string& key, std::vector<std::string>& out) {
  if (!store.exists(join_key(key, "zarr.json"))) return;
  if (is_array(store, key)) { out.push_back(key); return; }
  for (const auto& k : store.list(key).prefixes) collect(store, join_key(key, k), out);
}

void print_array(const Store& store, const std::string& subject, const std::string& table, const std::string& key, const std::string& rel) {
  const ArrayReader a = ArrayReader::open(store, key);
  const auto bytes = a.read();
  std::cout << subject << '\t' << table << '\t' << rel << '\t' << shape_text(a.shape()) << '\t' << dtype_to_string(a.dtype())
            << '\t' << crc_text(bytes) << '\n';
}

}  // namespace

int read_command(int argc, char** argv) {
  try {
    if (argc < 2) { std::cerr << "usage: nxr-neuro read <dataset folder> [--array <subject> <path> [--region a:b,c:d] [--out FILE]]\n"; return 2; }
    const Dataset ds = Dataset::open(argv[1]);
    std::string subject, path, region, out;
    bool one = false;
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--array" && i + 2 < argc) { one = true; subject = argv[i + 1]; path = argv[i + 2]; i += 2; }
      else if (a == "--region" && i + 1 < argc) region = argv[++i];
      else if (a == "--out" && i + 1 < argc) out = argv[++i];
      else { std::cerr << "unknown argument: " << a << '\n'; return 2; }
    }
    if (one) {
      const auto s = ds.query("SELECT path FROM subject WHERE name = ? OR path = ?", {subject, subject});
      if (s.empty()) { std::cerr << "no subject \"" << subject << "\"\n"; return 1; }
      const DatasetKeys dk(ds.folder());
      const Store& store = dk.keys;
      const ArrayReader a = ArrayReader::open(store, join_key(text(s[0], "path"), path));
      std::vector<std::int64_t> lo(a.shape().size(), 0), hi = a.shape();
      if (!region.empty()) {
        std::stringstream ss(region);
        std::string part;
        for (std::size_t d = 0; std::getline(ss, part, ','); ++d) {
          if (d >= lo.size()) { std::cerr << "region has more axes than the array\n"; return 1; }
          const auto c = part.find(':');
          lo[d] = std::stoll(part.substr(0, c));
          hi[d] = std::stoll(part.substr(c + 1));
        }
      }
      const auto bytes = a.read(lo, hi);
      std::vector<std::int64_t> ext(lo.size());
      for (std::size_t d = 0; d < lo.size(); ++d) ext[d] = hi[d] - lo[d];
      std::cout << shape_text(ext) << '\t' << dtype_to_string(a.dtype()) << '\t' << crc_text(bytes) << '\n';
      if (!out.empty()) std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      return 0;
    }
    for (const auto& s : ds.query("SELECT name, path, kind FROM subject ORDER BY name"))
      std::cout << "subject\t" << text(s, "name") << '\t' << text(s, "path") << '\t' << text(s, "kind") << '\n';
    struct Named { const char* table; const char* sql; };
    const Named named[] = {
        {"field", "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM field r JOIN subject s ON s.id = r.subject_id WHERE r.path IS NOT NULL"},
        {"selection", "SELECT s.path AS sp, s.name AS sn, r.array_path AS p FROM selection r JOIN subject s ON s.id = r.subject_id WHERE r.array_path IS NOT NULL"},
        {"operator", "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM operator r JOIN subject s ON s.id = r.subject_id WHERE r.path IS NOT NULL"},
        {"geometry", "SELECT s.path AS sp, s.name AS sn, r.positions_path AS p FROM geometry r JOIN subject s ON s.id = r.subject_id WHERE r.positions_path IS NOT NULL"},
        {"topology_cell", "SELECT s.path AS sp, s.name AS sn, r.path AS p FROM topology_cell r JOIN topology t ON t.id = r.topology_id JOIN subject s ON s.id = t.subject_id WHERE r.path IS NOT NULL"},
        {"selection_measurement", "SELECT s.path AS sp, s.name AS sn, r.array_path AS p FROM selection_measurement r JOIN selection x ON x.id = r.selection_id JOIN subject s ON s.id = x.subject_id WHERE r.array_path IS NOT NULL"},
    };
    std::set<std::string> seen;
    const DatasetKeys dk(ds.folder());
      const Store& store = dk.keys;
    for (const auto& n : named) {
      for (const auto& r : ds.query(n.sql)) {
        std::vector<std::string> keys;
        const std::string sp = text(r, "sp");
        collect(store, join_key(sp, text(r, "p")), keys);
        for (const auto& key : keys) {
          const std::string rel = key.substr(sp.size() + 1);   // the subject's own key, as the rows state it
          if (seen.insert(sp + "\n" + rel).second) print_array(store, text(r, "sn"), n.table, key, rel);
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
