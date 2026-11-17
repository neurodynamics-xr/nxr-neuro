// nxr-io — FIBER BUNDLES in a subject or template package (docs/specs/2026-11-16-structural-connectome.md, sections 3, 4.1, 6 and 7).
//
// A bundle is nF fibers of nP points each (fiber-major), in the subject's SCS, in metres: a `curve` manifold `<name>` at `fibre/<name>` (io's folder for curves) with an array `points` [nF·nP x 3]. Its
// producer record names the points per fiber and, for a bundle that came from somewhere, its `origin` (source, citation, licence, terms): a result derived from a
// bundle carries that origin (`derived_origin`), so its licence and attribution travel with it (a template's HCP-1065 bundle is CC BY-SA 4.0 together with the WU-Minn
// HCP Data Use Terms; `template release` refuses an item without an origin).
//
// THE ENDS (the pointwise object): each fiber's two ends on an end surface, stored per bundle and surface:
//   operator `<bundle>_ends_<surface>`        kind `fibre ends`, CSR [surface vertices x bundle points]: the first and last point of every fiber on the surface vertex
//                                             nearest to it (what the Studio's `ends` reads). It does not depend on any gate.
//   array     `<curve node>/end_distance_<surface>`   [2nF] the distance of each end to the surface, in metres, in the precision of the fibers.
// Both arrays of the pair are in the stored order: fiber f's first end at 2f, its last at 2f + 1.
//
// Precision follows the input: a bundle is stored in float32 or float64 as given; the distances have the same precision. All of it through io's task API: a
// `task::Subject` finds, and the writer's calls run inside the caller's `Unit` (or composition), so a bundle is stored with its ends or not at all.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"
#include "nxr/io/store.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

namespace nxr::io::fibers {

template <class S>
struct Bundle {
  std::string name;                         // the node's name (sanitised for its path)
  std::int64_t n_fibers = 0, points_per_fiber = 0;
  std::vector<S> points;                    // [n_fibers · points_per_fiber x 3], row-major, fiber-major, SCS metres
  std::vector<std::uint8_t> colors;         // optional [rows x 3]
  std::string source;                       // the file or program the fibers came from
  std::string comment;
  Json origin;                              // the origin document (null: none); copied into the producer record
  Json producer = Json::object();          // further facts for the producer record (e.g. {"processing": [...]})
};

struct Stored {
  std::string id, name, path;               // the curve manifold
  std::int64_t n_fibers = 0, points_per_fiber = 0;
  std::string precision;                    // "float32" | "float64"
};

/** Stores the bundle as a `curve` manifold. std::invalid_argument/Error(invalid_argument) for nP < 2, a points array of the wrong size, or a name already in the subject. */
template <class S>
Stored store_bundle(SubjectWriter& sub, const Bundle<S>& bundle);

/** The stored bundle `name` (the curve manifold) of a subject. Error(no_such_row) when there is none; Error(invalid_argument) when it is stored in the other precision. */
template <class S>
Bundle<S> read_bundle(const task::Subject& subject, const Store& store, const std::string& name);
/** "float32" or "float64": how the bundle `name` is stored. */
std::string bundle_precision(const task::Subject& subject, const Store& store, const std::string& name);

template <class S>
struct Ends {
  std::vector<std::int32_t> vertex;         // [2nF] the end surface's vertex of each end (stored order)
  std::vector<S> distance;                  // [2nF] metres
};

/** Stores the ends of bundle `bundle` on the surface `end_surface` (the subject's `source/<end_surface>`). A different surface or gate is another call; the same pair twice
 *  is refused. `producer` goes into the operator's producer record (the method, the cache key). */
template <class S>
void store_ends(SubjectWriter& sub, const std::string& bundle, const std::string& end_surface, const Ends<S>& ends, const Json& producer = Json::object());
template <class S>
Ends<S> read_ends(const task::Subject& subject, const Store& store, const std::string& bundle, const std::string& end_surface);

/** What a result derived from a bundle carries in its provenance: the bundle's `origin` document as it is, plus `derived_from` (the bundle's name) and, when the licence
 *  is ShareAlike, a notice that the result is shared under the same terms. Null when the bundle has no origin. `bundle_producer` is the producer record of its curve node. */
Json derived_origin(const Json& bundle_producer, const std::string& bundle_name);

}  // namespace nxr::io::fibers
