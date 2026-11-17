// nxr-graph — NODE SIGNATURES AND THE ONE SHARED CHECKER (spec docs/specs/2026-11-17-type-system.md, sections 4 to 6; step T2).
//
// A node declares a SIGNATURE: its ports' types written with manifold variables, and its MOVE. The checker (`type_use`) solves a use by unification: it binds
// each variable to the term it meets, builds the outputs' types from the signature (so nothing a node's own code does can get around the rules), applies the
// move's rule, then runs the node's small `refine` (what the signature cannot say: a parameter that picks a band, a level, a form), which may only narrow
// the outputs or refuse. Signatures are written with a C++ builder (checked by the compiler, printed by the printer):
//
//   auto M = var("M"); auto T = time_var("T");                                       // T: a time line only
//   sig(Move::pointwise)
//     .in("x", field({M, T}).streamed().value(ValueType::real))
//     .out("y", like("x").value(ValueType::complex));                                // like(x): x's type with the changes listed
//
// THE MOVES fix what a node may do to the domain, and that is where the cardinal rule lives (reductions only as measurements over partitions):
//   pointwise   the domain is unchanged                  apply      one factor mapped by an operator (its source a variable that may be a restriction)
//   restrict    one factor narrowed                      compose    two operators meet: A -> B, B -> C gives A -> C
//   measure     the ONLY move that removes a factor: a partition's members replace the factors it partitions
//   summarize   across a partition's members: a statistic that remembers what it is OF; a field is measured first
//   detect      events found on a scalar field or measurement: a selection with an identity, and measurements over its members
//   make        a selection, a manifold or an operator made from parameters and geometry; no field input
// NO TRANSITION IS INSERTED by the checker: it refuses, names the node to add (`positions.transition`), and stops. A node that reads POSITIONS declares `positions_share({ports})` (one
// coordinate system) or `positions_in(port, "scs")`; the coordinate system is part of every manifold's type (T5).
#pragma once
#include <functional>
#include <stdexcept>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/graph/types.h"

namespace nxr::graph {

enum class Move { pointwise, apply, compose, restrict, measure, summarize, detect, make };
const char* to_string(Move m);

/** the pattern of an input port: its class, the domain it lies on (manifold variables), and what it requires of the type */
struct InPat {
  std::optional<PortClass> cls;                         // none: any class (a port the checker does not type yet)
  Domain dom;                                           // the pattern the domain unifies with (an operator: its source)
  bool has_dom = false;
  Domain target;                                        // an operator's target
  bool has_target = false;
  std::vector<ValueType> values;                        // allowed value types (empty: any); a `complex_parts` value counts as complex
  bool is_linear = false;                               // the value must be a linear function of the recording (D106)
  bool is_streamed = false;                             // a value inside the loop (samples or frames)
  bool is_optional = false;
  bool is_over = false;                                 // measure: the partition the measurement is taken over
  std::vector<std::string> quantity_in;                 // allowed quantities (empty: any)
  std::string hint;                                     // what to do, appended to the port's refusals

  InPat& value(ValueType v) { values = {v}; return *this; }
  InPat& value_in(std::vector<ValueType> v) { values = std::move(v); return *this; }
  InPat& linear() { is_linear = true; return *this; }
  InPat& streamed() { is_streamed = true; return *this; }
  InPat& optional() { is_optional = true; return *this; }
  InPat& over() { is_over = true; return *this; }
  InPat& quantity(std::vector<std::string> q) { quantity_in = std::move(q); return *this; }
  InPat& because(std::string h) { hint = std::move(h); return *this; }
};
InPat field(std::vector<ManifoldRef> factors);
InPat measurement(std::vector<ManifoldRef> factors);
InPat measurement_any();
InPat selection(std::vector<ManifoldRef> factors);
InPat selection_any();
InPat statistic_any();
InPat operator_(const ManifoldRef& from, const ManifoldRef& to);
InPat any_port();
InPat manifold_port(const ManifoldRef& m);               // a manifold as a value (a positions input): class manifold, lying on `m`
InPat transition_port();                                 // an operator of kind transition (a change of coordinate system of one space)

/** the pattern of an output: a copy of an input's type with changes, or a blank of a class; the domain is a pattern instantiated with what the inputs bound */
struct OutPat {
  std::string like_port;
  std::optional<PortClass> cls;
  std::optional<Domain> dom, target;
  std::optional<ValueType> value;
  std::optional<bool> complex_parts, nonlinear;
  std::optional<std::string> quantity;
  OutPat& on(std::vector<ManifoldRef> factors) { Domain d; d.factors = std::move(factors); dom = d; return *this; }
  OutPat& onto(std::vector<ManifoldRef> factors) { Domain d; d.factors = std::move(factors); target = d; return *this; }
  OutPat& as(PortClass c) { cls = c; return *this; }
  OutPat& with_value(ValueType v) { value = v; complex_parts = false; return *this; }
  OutPat& with_quantity(std::string q) { quantity = std::move(q); return *this; }
  OutPat& non_linear() { nonlinear = true; return *this; }
};
OutPat like(const std::string& port);
OutPat make_of(PortClass c);

/** what refine sees and may narrow: the inputs' types, the outputs the signature built, the bindings, and where the use sits (for its refusals) */
struct Unified {
  Types in, out;
  Bindings bind;
  std::string label;                                    // "chunks[3] measure"
  bool chunk_node = false;                              // a node that runs on every block (a partition of the streamed factor needs an accumulate node)
  std::string first_out;                                // the first output's value name, for messages
  std::map<std::string, std::string> in_names, out_names;   // port -> value name; an output's value name is the identity of a selection it makes
  std::map<std::string, std::string> made_by;           // value name -> the use that made it ("chunks[1] power"), for messages
  /** a refusal at a port: "<label> -> <first output>: port `<port>`: <message>" (the port's type and the use that made it are appended) */
  [[noreturn]] void fail(const std::string& port, const std::string& message) const;
  const PortType& type(const std::string& port) const;               // an input that is present (refuses when absent)
  bool has(const std::string& port) const { return in.count(port) != 0; }
};

/** a refusal of the checker: invalid_argument whose message is complete (the use, the port, the two types, the rule, what to do) */
struct TypeRefusal : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

struct Signature {
  Move move = Move::pointwise;
  std::vector<std::pair<std::string, InPat>> inputs;
  std::vector<std::pair<std::string, OutPat>> outputs;
  std::function<void(Unified&, const Json&)> refine_fn;
  bool accumulates = false;                             // the node is an accumulate node (a fold over the blocks in time order): it may measure a partition of the streamed factor
  std::vector<std::vector<std::string>> position_groups;   // ports whose positions must be in ONE coordinate system (T5)
  std::vector<std::pair<std::string, std::string>> positions_named;   // (port, system name): the port's positions must be in that system ("scs")
  Signature& positions_share(std::vector<std::string> ports) { position_groups.push_back(std::move(ports)); return *this; }
  Signature& positions_in(const std::string& port, const std::string& system) { positions_named.emplace_back(port, system); return *this; }
  std::string reduces;                                  // a fused statistic estimation (measure over a declared partition of the stream, then summarize): the partition it reduces over, in words
  Signature& in(const std::string& port, InPat p) { inputs.emplace_back(port, std::move(p)); return *this; }
  Signature& out(const std::string& port, OutPat p) { outputs.emplace_back(port, std::move(p)); return *this; }
  Signature& refine(std::function<void(Unified&, const Json&)> f) { refine_fn = std::move(f); return *this; }
  Signature& accumulating() { accumulates = true; return *this; }
  Signature& reduces_over(std::string partition) { reduces = std::move(partition); return *this; }
  /** the quantities the signature names must be in the closed vocabulary; every pattern's port is read or made once. Refuses naming `node`. */
  void validate(const std::string& node) const;
};
Signature sig(Move m);

/** one use to be checked: its place, its inputs' types and where they came from */
struct TypeUse {
  std::string label;                                    // "chunks[3] measure"
  bool chunk_node = false;
  std::map<std::string, std::string> in_names, out_names;
  Types in;
  std::map<std::string, std::string> made_by;
};
/** THE SHARED CHECKER's one function: unify the inputs with the signature, make the outputs' types, apply the move's rule, run refine. Refuses (TypeRefusal) with the use, the
 *  port, the two types in the printed form, the rule and what to do. Returns the outputs' types by port. */
Types type_use(const std::string& node, const Signature& signature, const TypeUse& use, const Json& params);

/** the signatures of the registered nodes the shipped definitions use (signatures.cpp); null when a node has none yet (it is then checked by its own `plan`) */
const Signature* builtin_signature(const std::string& node);
std::vector<std::string> builtin_signature_names();

/** a graph typed by signatures alone: every use in order (before, chunks, after), its outputs' types; for a graph whose nodes all have signatures. The inputs must be typed.
 *  `signatures` finds a node's signature (default: the built-in ones); a use of a node without one is refused. Parameters: `$name` values are taken from `params`. */
struct Graph;
struct TypedGraph {
  Types values;                                         // every value: the inputs and what each use makes
  std::map<std::string, std::string> made_by;           // value -> the use
};
TypedGraph type_check(const Graph& g, const Types& inputs, const Json& params = Json::object(),
                      const std::function<const Signature*(const std::string&)>& signatures = builtin_signature);
/** the declared kinds of a graph's outputs against what it makes: a statistic declared a measurement is refused (kinds: field, operator, selection, spans, labelling, measurement, statistic, manifold) */
void check_output_kinds(const Types& values, const std::map<std::string, std::string>& declared);

}  // namespace nxr::graph
