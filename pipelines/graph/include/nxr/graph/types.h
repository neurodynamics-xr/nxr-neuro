// nxr-graph — VALUES AND TYPES: what flows along the arrows of a graph of pure nodes, and what is known about it before any data flows.
// The spec: docs/specs/2026-11-17-compute-and-run.md, section 3.1. This library (`nxr_graph`, pipelines/graph/) knows arrays and parameters only: no file, no
// subject, no database, never `io` (the layering guard). The runner (pipelines/) finds and stores; the executor here computes.
//
//   PortType  what a port carries, known BEFORE any data: the arrows are checked on these (`check`, graph.h).
//   Block     a value INSIDE the loop: a block's columns with the halo still round them; float32, row-major.
//   Array     a value OUTSIDE the loop: float64 [rows x cols x comps].
//   Place     where a block's own columns lie on the time axis, and which of them are good.
//
// THE TYPE SYSTEM (spec docs/specs/2026-11-17-type-system.md, T1): a PortType also says WHICH manifold its values lie on (`domain`: a product of
// `Manifold` terms, with restrictions and member sets), what the values ARE (`quantity`, a closed vocabulary), at what resolution (`res`: the band and
// the spatial level), the shape of a selection (`sel`), what a statistic is OF (`of`), and G5's `spin` and `gauge`. The old members (`on`, `to`, `facts`)
// stay for the C1 plans; the typed members are filled where a type is built with them (`typed`), and the checker (checker.h) reads only those.
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace nxr::graph {

using Json = nlohmann::ordered_json;   // keys keep the order they were written in (as nxr::io::Json: the same type, no io needed)

enum class ValueType { real, complex, vector3, quaternion, hermitian3, monogenic, integer };
enum class PortClass {
  field, operator_, selection, partition, constant,                 // C1's
  measurement, statistic, manifold                                  // the type system's: one value per member of a partition; what is computed from members; a line or space made
};
/** what a field's columns are: samples (streamed, inside the loop), frames (a retimed stream), or none (an array outside the loop) */
enum class Axis { none, samples, frames, members };

const char* to_string(ValueType v);
const char* to_string(PortClass c);
const char* to_string(Axis a);

// ───────────────────────────────────────────── the type system: manifolds as terms ─────────────────────────────────────────────

enum class ManifoldKind { points, line, surface, volume, curve, graph, members };
/** what a line or member set is FOR, as far as the checker cares: the factor a stream walks, and what a measurement's member set came from */
enum class Role { none, time, frames, time_tiles, frequency, modes, levels };
const char* to_string(ManifoldKind k);
const char* to_string(Role r);

/** a chart on a space, with a unit (the schema's coordinate_system): {"head", "scs", "m"} · {"time", "clock", "s"} · {"frequency", "hz", "Hz"}.
 *  Two manifolds are comparable in position only when their systems are of one space. */
struct CoordinateSystem {
  std::string space, name, unit;
  bool operator==(const CoordinateSystem& o) const { return space == o.space && name == o.name && unit == o.unit; }
};

struct Manifold;
using ManifoldRef = std::shared_ptr<const Manifold>;

/** A MANIFOLD AS A TERM: which manifold, known or not yet known. Equality is structural (`same`): two ids are equal when their labels and ids are; a member set
 *  when its partitions are; a restriction when the manifold and the selection are. Printed `label(key)` (a made one `label(~key)`, a variable by
 *  its name, an anonymous one `#n`). */
struct Manifold {
  enum class Form {
    id,          // a stored manifold, by id (run: resolved when the inputs are found)
    anonymous,   // an array's manifold in `compute`: equal only to itself (or to another the caller declares the same)
    variable,    // a node signature's M, T: bound by unification, never left in a checked graph
    made,        // made by a node in this graph (a frequency line, a mode axis): keyed by the use and the output
    restricted,  // `of` restricted to the elements of the selection `by` (a kernel's channels, a source space)
    members,     // the member set of the partition `by`; it is NOT a stored manifold (the partition's dictionary and count are its storage)
    derived      // a stored manifold related to `of` by the relation `key` (`glued`: the glued surface of a cortex)
  };
  Form form = Form::id;
  std::string key;                                    // the id · the anonymous number · the variable's name · "<use>.<output>" · the relation's name
  std::string label;                                  // what kind of thing it is, for a person: "cortex", "sensors", "time", "frames"
  ManifoldKind kind = ManifoldKind::points;
  Role role = Role::none;                             // a variable: the role it accepts (time: a time line only; frames: a time line or a frame line)
  std::optional<CoordinateSystem> cs;                 // the system its positions are in; none for a manifold without geometry (members, an index line)
  std::int64_t n = 0;                                 // its vertices (0: not known before data, as a stream's time line)
  std::optional<double> spacing;                      // a closed-form line's spacing
  ManifoldRef of;                                     // restricted, derived: the manifold it is of
  std::string by;                                     // restricted, members: the selection's identity (a stored id, or "<use>.<output>"; a signature's "$port")
  bool rest = false;                                  // variable: stands for all the other factors of a domain (R)
  bool subsume = false;                               // variable at an operator's source: also accepts a restriction of what it is bound to (`apply`'s one implicit move)
};

// the builders. A stored manifold, by label and id; an anonymous one; the variables of a signature; one a node makes; a restriction; a member set; a derived one.
ManifoldRef manifold_id(const std::string& label, const std::string& key, ManifoldKind kind = ManifoldKind::points, Role role = Role::none,
                        std::optional<CoordinateSystem> cs = std::nullopt, std::int64_t n = 0);
ManifoldRef time_line(const std::string& key, double sfreq = 0);                       // a `time/clock` line: role time
ManifoldRef frame_line(const std::string& key);                                        // a `time/clock` line of frames: role frames
ManifoldRef anonymous(int number, ManifoldKind kind = ManifoldKind::points, Role role = Role::none);
ManifoldRef var(const std::string& name);
ManifoldRef time_var(const std::string& name);       // binds only a factor whose role is time
ManifoldRef frames_var(const std::string& name);     // binds a factor whose role is time or frames
ManifoldRef var_rest(const std::string& name);       // the other factors, carried through unchanged
ManifoldRef from_or_restriction_of(const ManifoldRef& v);   // an operator's source: a variable that also accepts a restriction of the manifold it is bound to
ManifoldRef made(const std::string& label, const std::string& key, ManifoldKind kind = ManifoldKind::points, Role role = Role::none, std::optional<CoordinateSystem> cs = std::nullopt);
ManifoldRef restricted(const ManifoldRef& of, const std::string& by);
ManifoldRef members(const std::string& partition);
/** the member set of `partition`, with the line it partitions (`of`: a time tile's line; the time line of a joint partition), the role it has (frames for a frame tiling,
 *  time_tiles for a time ladder's level) and a label (`joint` for the cells of a joint partition of space x time). Equality ignores all three: it is the partition's. */
ManifoldRef members(const std::string& partition, const ManifoldRef& of, Role role = Role::time_tiles, const std::string& label = "");
ManifoldRef members_of_port(const std::string& port);      // in a signature: the member set of the selection read at `port` (instantiated to members(<its identity>))
ManifoldRef derived(const std::string& relation, const ManifoldRef& of);
bool same(const ManifoldRef& a, const ManifoldRef& b);                 // structural equality; the coordinate system is part of it when both manifolds record one (T5)
bool same_cs(const std::optional<CoordinateSystem>& a, const std::optional<CoordinateSystem>& b);   // true when either is not recorded
std::string print_cs(const std::optional<CoordinateSystem>& c);        // "head/scs (m)"
std::string print(const ManifoldRef& m);
ManifoldRef parse_manifold(const std::string& text);

/** where values lie: a product's factors in axis order (one factor when it is not a product), and the cells (the schema's key: "0", "1", "2", "1,0") */
struct Domain {
  std::vector<ManifoldRef> factors;
  std::string cells = "0";
  /** the factor a stream walks: the last, when its role is time or frames; none for an array */
  const Manifold* streamed() const;
  bool empty() const { return factors.empty(); }
};
bool same(const Domain& a, const Domain& b);
std::string print(const Domain& d);

/** what a field is band-limited to, or what a ladder's level resolves (the level IS the band) */
struct Resolution {
  std::optional<std::array<double, 2>> band;          // Hz
  std::string band_shape;                             // "morlet", "meyer"
  std::string ladder;                                 // the spatial ladder's identity, and its level(s): one level, or a stack (the value's components)
  std::vector<int> levels;
  std::string spatial_shape;                          // "mexhat" · "meyer"
  bool empty() const { return !band && ladder.empty() && levels.empty(); }
  bool same_band(const Resolution& o) const { return band == o.band; }
  bool same_level(const Resolution& o) const { return ladder == o.ladder && levels == o.levels; }
  bool operator==(const Resolution& o) const { return same_band(o) && same_level(o) && band_shape == o.band_shape && spatial_shape == o.spatial_shape; }
};
std::string print(const Resolution& r);   // "band 8-12 Hz morlet; level tiles@5" (the parts that are set)

/** a selection's shape: what the members are */
struct SelectionShape {
  std::string type;                                   // the schema's: point · span · spans · indices · kernel · set · ladder · tiling (closed-form spans)
  bool partition = false;                             // its members are disjoint: a measurement may be taken over it
  std::string identity;                               // a stored id, or "<use>.<output>" for one this graph makes
  std::string detected_over;                          // detected events: the identity of the partition whose members they were found on
};

/** what a statistic is OF: the quantity, the resolution and the domain of the field or measurement before the reduction */
struct OfType { std::string quantity; Resolution res; Domain on; };

/** the closed vocabulary of quantities (a field's kind, a measurement's measure, a statistic's measure, an operator's kind): a quantity outside it is refused */
const std::vector<std::string>& quantities();
bool is_quantity(const std::string& q);

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

  // ── the type system's members (T1). `typed`: they are filled; a type without them is checked by its node's plan alone, as before.
  bool typed = false;
  bool complex_parts = false;            // vector3, quaternion, ... carrying an imaginary part ("complex vector3"); a complex scalar is `value` = complex
  int components = 1;                    // a vertex's components (3 for vector3) times a stack of levels
  Domain domain;                         // field, measurement: where the values lie · selection: what it indexes · statistic: its axes · manifold: itself · operator: its source
  Domain target;                         // operator: its target
  std::string quantity;                  // field: what the values are · measurement, statistic: the measure · operator: its kind · selection: its type name
  Resolution res;                        // field, measurement: the band and the spatial level · selection: a ladder level's or a joint partition's
  ValueType from_value = ValueType::real;   // operator: the value type it reads (`value` is the one it makes)
  int spin = 0;                          // G5: 0 scalar, 1 tangent vector, n an n-RoSy, -1 a conjugate
  std::string gauge;                     // G5: "" for spin 0 and for vector3 (an embedded value is gauge-free); else the gauge id
  std::optional<SelectionShape> sel;
  std::optional<OfType> of;              // statistic: what it is of
  std::optional<std::pair<CoordinateSystem, CoordinateSystem>> transition;   // operator of kind `transition`: from -> to, one space
  /** a value that flows block by block (samples or frames) rather than being an array */
  bool streamed() const { return axis == Axis::samples || axis == Axis::frames; }
};
using Types = std::map<std::string, PortType>;

/** the printed form of a type (`--dry-run`, messages, composites' declared ports): one printer and one parser, spec section 3.3
 *    field<cortex(kernel) × frames(tiling); complex vector3; current; band 8-12 Hz morlet; level tiles@5>
 *    measurement<members(cells_l5); amplitude; band 8-16 Hz; level tiles@5>
 *    selection<spans on time(rest); partition; detected over ladder@8-16>
 *    statistic<summaries; axes groups(cells_l5); of amplitude>
 *    operator<sensors(rest)|channels(kernel) → cortex(kernel); real → vector3; inverse kernel>
 *    operator<transition head: device:run1 → head>
 *  `parse_type(print(t))` prints as `t` (what is printed is what is parsed; kinds, units and counts of a manifold are not printed). */
std::string print(const PortType& t);
PortType parse_type(const std::string& text);

// ── unification (spec section 4): the variables of a signature bound to the terms they meet
struct Bindings {
  std::map<std::string, ManifoldRef> vars;
  std::map<std::string, std::vector<ManifoldRef>> rests;
};
/** `pattern` against `actual`, extending `b`; false with `why` (naming both terms) when they do not unify. The rules, and no others:
 *  a variable binds any term its role allows (time: a time line; frames: time or frames) and a second meeting must be the same term; ids and anonymous
 *  manifolds unify when equal; members(P) with members(Q) when P = Q; restricted(M, S) with restricted(N, U) when M = N and S = U; derived(rel, M) with
 *  derived(rel, N) when M = N; and, at an operator's source only, the one subsumption: a source restricted to S accepts the whole manifold (the kernel
 *  reads its channels of the recording), and a restriction to the same S; a restriction to another U is refused. */
bool unify(const ManifoldRef& pattern, const ManifoldRef& actual, Bindings& b, std::string& why);
/** a pattern's factors (rest variables splice) against a domain's; the cells must be equal when `pattern.cells` is not empty */
bool unify(const Domain& pattern, const Domain& actual, Bindings& b, std::string& why);
/** a pattern with the bindings substituted: refuses a variable left unbound. `ports` resolves members_of_port placeholders (port -> the selection's identity). */
ManifoldRef instantiate(const ManifoldRef& pattern, const Bindings& b, const std::map<std::string, std::string>& ports = {});
Domain instantiate(const Domain& pattern, const Bindings& b, const std::map<std::string, std::string>& ports = {});

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
  std::int64_t rows = 0, cols = 1;
  std::vector<double> data;
  Json meta = Json::object();
  std::int64_t comps = 1;
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
