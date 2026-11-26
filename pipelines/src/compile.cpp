// nxr-pipelines — `compile` (face.h): a pipeline definition expanded and type-checked without running it, the compiled graph handed back. Orchestration only: the definition is
// read and expanded (definition.h), the inputs are found and typed by io's finder and the runner's plan (or given by the caller), the graph is checked by the executor's own
// check (or by the signatures alone), and the result is written by nxr::graph::compiled_json. Nothing is read beyond metadata; nothing is written.
#include <set>

#include "nxr/graph/type_json.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/errors.h"
#include "nxr/io/finder.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/face.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/runner.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace g = nxr::pipelines::graph;
namespace ng = nxr::graph;

namespace nxr::pipelines::face {

namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }

// a refusal of the graph library (std::invalid_argument) is the command's: an io Error with the checker's message
template <class F>
auto refusing(F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const Error&) {
    throw;
  } catch (const std::invalid_argument& e) {
    refuse(e.what());
  }
}

// a signature looked up as `run` would: the node's own (a registered node of the graph registry), the built-in table's, then the caller's
const ng::Signature* signature_of(const std::string& name) {
  if (const ng::NodeSpec* n = ng::registry().find(name); n && n->signature) return &*n->signature;
  if (const ng::Signature* s = ng::builtin_signature(name)) return s;
  if (const g::ExternalNodes* x = g::ExternalNodes::current()) return x->signature(name);
  return nullptr;
}

Json params_of(const Json& p) { Json o = Json::object(); for (const auto& [k, v] : p.items()) o[k] = v; return o; }

}  // namespace

CommandResult compile(const CompileRequest& rq) {
  if (rq.dataset.has_value() != rq.subject.has_value()) refuse("compile: give the dataset and the subject together (the inputs typed from their stored records), or neither (the inputs' types given)");
  if (!rq.parameters.is_object()) refuse("compile: the parameters are a set of names and values");
  if (!rq.inputs.is_object()) refuse("compile: the inputs' types are a set of names and types");
  if (rq.dataset && !rq.inputs.empty()) refuse("compile: the inputs are typed from the dataset's records; inputs' types are given only without a dataset");

  const std::vector<ng::NodeSpec> declared = rq.signatures.is_null() ? std::vector<ng::NodeSpec>() : refusing([&] { return ng::external_nodes(rq.signatures); });
  const g::ExternalNodes scope(declared);                                  // the declared nodes are found by the definition reader, the plan and the check, for this call only
  const g::Definition d = refusing([&] { return g::load(rq.definition); });
  // a face that carries parameters only names a preset as the parameter `preset`, as `run`
  Json set = rq.parameters;
  std::string preset = rq.preset;
  if (set.contains("preset") && !d.parameters.contains("preset")) { preset = set["preset"].get<std::string>(); set.erase("preset"); }

  Json plans = Json::array();
  if (rq.dataset) {
    // TYPES FROM THE STORED RECORDS: the subject opened to be read (no lock; nothing is migrated or written), every run planned as `run --dry-run` plans it
    OpenOptions o;
    o.lock = false;
    DatasetWriter ds = is_catalogue_dataset(*rq.dataset) ? open_subject_package(*rq.dataset, *rq.subject, o) : open_writer(*rq.dataset, o);
    SubjectWriter sub = SubjectWriter::open(ds, *rq.subject);
    const auto runs = refusing([&] { return g::plan_runs(d, sub, set, preset); });
    for (const auto& P : runs) {
      Json bind = Json::array(), streams = Json::array();
      const Json found = finder::describe(P.found);
      for (const auto& [k, f] : found.items()) {
        Json e = f;
        e["input"] = k;
        (P.types.count(k) && P.types.at(k).streamed() ? streams : bind).push_back(e);
      }
      const auto anchor = d.anchor();
      plans.push_back(Json{{"parameters", params_of(P.parameters)}, {"anchor", anchor && P.found.count(*anchor) ? Json(text(*P.found.at(*anchor).row, "name")) : Json()},
                           {"graph", ng::compiled_json(*P.checked)}, {"bind", bind}, {"streams", streams}});
    }
  } else {
    // TYPES GIVEN BY THE CALLER: each plan's inputs typed from `inputs`, the graph checked by the node signatures alone (no plan reads a record: there is none)
    std::map<std::string, ng::PortType> given;
    for (const auto& [k, t] : rq.inputs.items()) given[k] = refusing([&] { return ng::port_type_from_json(t); });
    const Json base = refusing([&] { return g::parameters_of(d, set, preset); });
    for (const Json& each : refusing([&] { return g::plan_parameter_sets(d, base); })) {
      const g::Definition e = refusing([&] { return g::expand(d, each); });
      std::vector<std::string> present, missing;
      ng::Types types;
      for (const auto& [k, spec] : e.inputs.items()) {
        if (given.count(k)) { present.push_back(k); types[k] = given.at(k); continue; }
        if (spec.value("required", true)) missing.push_back(k);
      }
      for (const auto& [k, _] : given)
        if (!e.inputs.contains(k)) refuse("compile: '" + k + "' is not an input of " + e.name + " (its inputs: " + [&] { std::string s; for (const auto& [n, _2] : e.inputs.items()) s += (s.empty() ? "" : ", ") + n; return s; }() + ")");
      if (!missing.empty()) {
        std::string m;
        for (const auto& k : missing) m += (m.empty() ? "" : ", ") + k;
        refuse("compile: the input(s) " + m + " of " + e.name + " have no type: give each its type (inputs), or compile against a dataset and a subject (their stored records type them)");
      }
      const ng::Graph graph = refusing([&] { return g::graph_of(e, each, present); });
      const ng::TypedGraph typed = refusing([&] { return ng::type_check(graph, types, Json::object(), signature_of); });
      std::map<std::string, std::string> kinds;
      for (const auto& [o, spec] : e.outputs.items()) if (spec.is_object() && spec.contains("kind")) kinds[o] = spec["kind"].get<std::string>();
      refusing([&] { ng::check_output_kinds(typed.values, kinds); return 0; });
      Json bind = Json::array(), streams = Json::array();
      for (const auto& k : present) (types.at(k).streamed() ? streams : bind).push_back(Json{{"input", k}, {"name", nullptr}});
      plans.push_back(Json{{"parameters", params_of(each)}, {"anchor", nullptr}, {"graph", ng::compiled_json(graph, typed, Json::object(), signature_of)}, {"bind", bind}, {"streams", streams}});
    }
  }
  CommandResult r;
  r.result = Json{{"definition", d.name + "@" + std::to_string(d.version)}, {"subject", rq.subject ? Json(*rq.subject) : Json()}, {"types", rq.dataset ? "stored" : "given"}, {"plans", plans}};
  return r;
}

}  // namespace nxr::pipelines::face
