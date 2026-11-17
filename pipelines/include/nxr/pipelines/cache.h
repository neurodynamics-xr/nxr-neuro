// nxr-pipelines — THE CACHE, once, for every pipeline: a pipeline's result is FOUND BY ITS ROWS, never by guessing paths, and reused.
//
// A stored result carries the identity of the job that made it in its node's `producer_json`, under the key "cache":
//
//     "cache": { "pipeline": "eigen_solve",            the pipeline's name
//                "params":   { "modes": 3000 },        its parameters (the choices; defaults written out, so a default is a value)
//                "input":    { "mesh": "<nF>:<fnv1a>", the identity of what it was run on (for a surface: nxr's mesh key,
//                              "positions": "..." },   the Studio's `meshKey`, and the positions' own hash: pial and white share faces)
//                "compute":  "0.1.0+26f6347000" }      nxr-compute's version and commit (a new version is a new result)
//
// `find_cached` returns the complete node of a subject whose "cache" equals a key, or nothing. `cached` is the whole rule:
//
//     HIT   the stored arrays are read and returned (nothing is computed);
//     MISS  the job is computed, then its rows and arrays are stored TOGETHER in one composition (all or nothing, nxr-io's `compose`).
//
// A pipeline supplies three functions (`load`, `compute`, `store`) and a CacheKey; it writes `stamp(key)` into the `producer_json` of the node
// `find_cached` will look at (the pipeline's main product). Nothing else in a pipeline knows about the cache.
#pragma once
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"
#include "nxr/pipelines/task.h"

namespace nxr::pipelines {

using nxr::io::Json;
using nxr::io::Row;

struct CacheKey {
  std::string pipeline;
  Json params = Json::object();
  Json input = Json::object();
  std::string compute;                       // nxr-compute's version (NXR_COMPUTE_VERSION: package version + commit)
  Json json() const;                         // {"pipeline","params","input","compute"}
  bool operator==(const CacheKey& o) const { return json() == o.json(); }
};

// What the producer record carries: {"cache": key.json()}. Merge it into the main product's producer_json.
Json stamp(const CacheKey& key);

// nxr's mesh key (`nxr.geom.glue.mesh_key`, the Studio's `meshKey`): FNV-1a over the faces' int32 bytes (little-endian), "<nF>:<hash:08x>".
std::string mesh_key(const std::vector<std::int32_t>& faces);
// The same hash over the vertex positions as float32 (the precision the Studio receives), 64-bit: "<nV>:<hash:016x>".
std::string positions_key(const std::vector<double>& vertices);

// The complete node of `table` (manifold · field · operator · selection) of the subject whose producer record holds exactly `key` (a Selection's: its
// params_json, which a Selection has instead of producer_json); with `path`, only that node.
std::optional<Row> find_cached(const nxr::io::Database& db, const std::string& subject_id, const std::string& table, const CacheKey& key,
                               const std::optional<std::string>& path = std::nullopt);

template <class R>
struct Outcome {
  R value;
  bool hit = false;
  double seconds = 0;                        // the time of the load (a hit) or of the compute and the store (a miss)
  std::string node_id;                       // the node found (a hit)
};

// The rule. `table`/`path` name the node that carries the stamp; `load(row)` reads the stored arrays; `compute()` makes the result;
// `store(result)` writes it (rows and arrays) and runs inside `sub.compose`, so it is all or nothing.
template <class R>
Outcome<R> cached(nxr::io::SubjectWriter& sub, const std::string& table, const std::optional<std::string>& path, const CacheKey& key,
                  const std::function<R(const Row&)>& load, const std::function<R()>& compute, const std::function<void(const R&)>& store) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  auto secs = [&] { return std::chrono::duration<double>(clock::now() - t0).count(); };
  if (const auto row = find_cached(sub.db(), sub.id(), table, key, path)) {
    R v = load(*row);
    return {std::move(v), true, secs(), nxr::io::text(*row, "id")};
  }
  R v = compute();
  sub.compose([&] { store(v); });
  return {std::move(v), false, secs(), {}};
}

// The same rule over the task-level API: `load(node)` reads the stored arrays of the node found; `store(result)` runs inside one unit (`Subject::commit`): the nodes it
// makes, their arrays and every row it writes are stored together or not at all.
template <class R>
Outcome<R> cached(nxr::io::SubjectWriter& sub, const std::string& table, const std::optional<std::string>& path, const CacheKey& key,
                  const std::function<R(const task::Node&)>& load, const std::function<R()>& compute, const std::function<void(const R&)>& store) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  auto secs = [&] { return std::chrono::duration<double>(clock::now() - t0).count(); };
  const task::Subject subj = task::Subject::of(sub);
  const task::Cls cls = table == "manifold" ? task::Cls::manifold : table == "field" ? task::Cls::field : table == "operator" ? task::Cls::operator_
                      : table == "selection" ? task::Cls::selection : task::Cls::statistic;
  if (const auto node = subj.cached(cls, key.json(), path)) {
    R v = load(*node);
    return {std::move(v), true, secs(), node->id};
  }
  R v = compute();
  subj.commit([&] { store(v); });
  return {std::move(v), false, secs(), {}};
}

// Arrays and rows of a name already stored (`nodes`: {table, path}): refused (already_exists), or with `replace` moved aside into the datastore's
// `_replaced-<date>/` (never deleted: the data rule) and the rows removed. Returns where they were moved ("" when nothing was); `replaced` says whether any was.
// WHERE A REPLACED NODE'S ARRAYS ARE MOVED ASIDE (never deleted): `base` is the key folder (`n` is 1 for the first replacement of the day, then 2, 3, ...), `on_disk` the
// folder it is on. In a one-database dataset it is `_replaced-<date>/<subject folder>` at the datastore's top; in a subject package, which holds nothing outside itself,
// `_replaced-<date>` inside the package.
struct AsideFolder { std::string base; std::string on_disk; };
AsideFolder replaced_aside(const nxr::io::DatasetWriter& ds, const nxr::io::SubjectWriter& sub, const std::string& date, int n);
std::string clear_stored(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const std::vector<std::pair<const char*, std::string>>& nodes, bool replace,
                         const std::string& what, bool& replaced);

}  // namespace nxr::pipelines
