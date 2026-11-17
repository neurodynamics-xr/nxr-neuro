// nxr-graph — VALUES AND TYPES: what flows along the arrows of a graph of pure nodes, and what is known about it before any data flows.
// The spec: docs/specs/2026-11-17-compute-and-run.md, section 3.1. This library (`nxr_graph`, pipelines/graph/) knows arrays and parameters only: no file, no
// subject, no database, never `io` (the layering guard). The runner (pipelines/) finds and stores; the executor here computes.
//
//   PortType  what a port carries, known BEFORE any data: the arrows are checked on these (`check`, graph.h).
//   Block     a value INSIDE the loop: a block's columns with the halo still round them; float32, row-major.
//   Array     a value OUTSIDE the loop: float64 [rows x cols x comps].
//   Place     where a block's own columns lie on the time axis, and which of them are good.
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace nxr::graph {

using Json = nlohmann::ordered_json;   // keys keep the order they were written in (as nxr::io::Json: the same type, no io needed)

enum class ValueType { real, complex, vector3, quaternion, hermitian3 };
enum class PortClass { field, operator_, selection, partition, constant };
/** what a field's columns are: samples (streamed, inside the loop), frames (a retimed stream), or none (an array outside the loop) */
enum class Axis { none, samples, frames, members };

const char* to_string(ValueType v);
const char* to_string(PortClass c);
const char* to_string(Axis a);

/** What a port carries. No database and no row: the runner builds one from a found node, `compute` from the array handed to it and the caller's metadata. */
struct PortType {
  PortClass cls = PortClass::field;
  ValueType value = ValueType::real;
  std::int64_t rows = 0;                 // elements (channels, vertices, members), a vertex's components consecutive
  Axis axis = Axis::none;                // samples: streamed (inside the loop); frames: retimed; none: an array
  std::optional<double> sfreq;           // Hz, for samples (frames: the frame rate)
  std::string unit;                      // SI, as stored
  std::string on, to;                    // the manifold an element lies on (a label the caller gives: "sensors", "cortex"); an operator's target
  bool nonlinear = false;                // made by a non-linear node: a linear map may not follow (D106)
  Json facts = Json::object();           // what a node's plan may read beside the shape: a ladder's tiling, a selection's parameters, a channel list
  /** a value that flows block by block (samples or frames) rather than being an array */
  bool streamed() const { return axis == Axis::samples || axis == Axis::frames; }
};
using Types = std::map<std::string, PortType>;

/** A VALUE INSIDE THE LOOP: a block's columns, the halo still round them. Row-major [rows x width], float32; a complex value carries its imaginary part.
 *  The own columns are [left, width - right). Rows are the value's elements, a vector's components consecutive. */
struct Block {
  std::int64_t rows = 0, width = 0, left = 0, right = 0;
  std::vector<float> re, im;
  std::int64_t own() const { return width - left - right; }
  bool complex() const { return !im.empty(); }
};
/** A VALUE OUTSIDE THE LOOP: [rows x cols x comps] doubles, the components of an element consecutive (comps 1 for a real map; 3 for a vector; 2 for a complex
 *  value handed to or from `apply`: re, im). `meta` is what a node records about it for the nodes after it. */
struct Array {
  std::int64_t rows = 0, cols = 1, comps = 1;
  std::vector<double> data;
  Json meta = Json::object();
};
using Blocks = std::map<std::string, Block>;
using Arrays = std::map<std::string, Array>;
/** where a block's own columns lie: the first sample (or frame), the count, and which are good (outside the bad samples; null: every one) */
struct Place {
  std::int64_t first = 0, n = 0;
  const std::vector<std::uint8_t>* good = nullptr;
};
/** where a RETIMING node (the frame tiling) puts a block's own columns: its first frame, their count, and which frames are good (empty: all) */
struct FramePlace {
  std::int64_t first = 0, n = 0;
  std::vector<std::uint8_t> good;
};

/** a refusal: invalid_argument with the message, raised before any data flows (the plan) or when a block breaks a rule */
[[noreturn]] void refuse(const std::string& message);

}  // namespace nxr::graph
