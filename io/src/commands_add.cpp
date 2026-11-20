// The `add <class>` commands (nxr/io/task/commands.h; docs/specs/2026-11-19-io-service.md, 4.2): what a user computed, read from .npy files, checked against what it
// lies on, and stored as ONE unit through io's typed writes (task::Unit), with a producer record naming the command and its input files (by name, never by path).
#include "commands_open.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <type_traits>

#include "nxr/io/errors.h"
#include "nxr/io/task/handle.h"
#include "pieces.h"
#include "nxr/io/layout.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

namespace fs = std::filesystem;

namespace nxr::io::task {

using namespace commands_detail;

namespace {

std::string shape_text(const std::vector<std::int64_t>& shape) {
  std::string t;
  for (const auto d : shape) t += (t.empty() ? "" : " x ") + std::to_string(d);
  return "[" + t + "]";
}

std::string file_name(const ArrayInput& a) { return a.label(); }

// An array input (a .npy file or in memory) as io's Values, in its own precision: float32 and float64 kept; integers as int32 (refused when a value does not fit); uint8 and bool as uint8.
Values values_of(const ArrayInput& in, const std::string& what) {
  const std::string dtype = array_dtype(in);
  Values v;
  v.shape = array_shape(in);
  if (dtype == "float32") v.f32 = array_all<float>(in);
  else if (dtype == "float64") v.f64 = array_all<double>(in);
  else if (dtype == "uint8" || dtype == "bool") v.u8 = array_all<std::uint8_t>(in);
  else {
    const std::vector<std::int64_t> all = array_all<std::int64_t>(in);
    if (dtype == "uint64") {   // read through int64: a value above its range wraps negative
      const std::vector<double> check = array_all<double>(in);
      for (const double x : check) if (x > static_cast<double>(std::numeric_limits<std::int32_t>::max())) throw Refused(what + " (" + file_name(in) + "): a value does not fit int32");
    }
    v.i32.reserve(all.size());
    for (const auto x : all) {
      if (x < std::numeric_limits<std::int32_t>::min() || x > std::numeric_limits<std::int32_t>::max()) throw Refused(what + " (" + file_name(in) + "): the value " + std::to_string(x) + " does not fit int32");
      v.i32.push_back(static_cast<std::int32_t>(x));
    }
  }
  return v;
}
std::string dtype_of(const Values& v) { return !v.f32.empty() ? "float32" : !v.f64.empty() ? "float64" : !v.u8.empty() ? "uint8" : !v.i32.empty() ? "int32" : "float64"; }
// float32 or float64 values, refused otherwise (a statistic, an operator, measurements, positions)
void need_real(const Values& v, const std::string& what, const ArrayInput& path) {
  if (v.f32.empty() && v.f64.empty() && v.size() > 0) throw Refused(what + " (" + file_name(path) + ") holds " + dtype_of(v) + " values: give float32 or float64");
}
Values as_f64(const Values& v) {
  Values out;
  out.shape = v.shape;
  if (!v.f64.empty()) out.f64 = v.f64;
  else if (!v.f32.empty()) out.f64.assign(v.f32.begin(), v.f32.end());
  return out;
}

// each value's own shape for a value type (the writer's: io/src/subject_writer.cpp)
std::vector<std::int64_t> element_shape(const std::string& value_type) {
  if (value_type == "scalar" || value_type == "integer") return {};
  if (value_type == "complex" || value_type == "vector2") return {2};
  if (value_type == "vector3") return {3};
  if (value_type == "quaternion") return {4};
  if (value_type == "matrix3") return {3, 3};
  if (value_type == "hermitian3") return {3, 3, 2};
  throw Refused("--value-type " + value_type + " is not a value type (scalar, integer, complex, vector2, vector3, matrix3, quaternion, hermitian3)");
}

// the subject opened for writing, one unit committed when `body` returns
template <class F>
CommandResult in_unit(const std::string& dataset, const std::string& subject, const std::vector<std::string>& templates, const std::string& created_by, const Json& producer,
                      const Progress& progress, F&& body) {
  DatasetWriter ds = open_dataset(dataset, subject, OpenOptions{}, templates, progress);
  OpenSubjectOptions oo;
  oo.created_by = created_by;
  SubjectWriter sub = SubjectWriter::open(ds, subject, oo);
  const Subject ts = Subject::of(sub);
  CommandResult r;
  {
    Unit u = ts.unit(By{created_by, producer});
    if constexpr (std::is_invocable_v<F, const Subject&, Unit&, DatasetWriter&>) r.result = body(ts, u, ds);   // a body that looks in other subjects of the dataset
    else r.result = body(ts, u);
    u.commit();
  }
  return r;
}

// a JSON option (`--producer`, `--params`): its text parsed, refused with the option's name when it is not JSON (or not an object)
Json json_of(const std::string& text, const std::string& option, bool object = true) {
  Json j = Json::parse(text, nullptr, false);
  if (j.is_discarded()) throw Refused(option + ": not JSON (" + text.substr(0, 40) + ")");
  if (object && !j.is_object()) throw Refused(option + ": a JSON object ({...})");
  return j;
}
// what a maker says of how a node was made (`--producer`): keys of its own, never the ones the record keeps for the command itself
Json maker_producer(const std::optional<std::string>& text) {
  if (!text) return Json();
  Json j = json_of(*text, "--producer");
  for (const char* k : {"command", "program", "inputs", "output", "cache"})
    if (j.contains(k)) throw Refused(std::string("--producer: the key '") + k + "' is the command's own record; name the maker's facts under other keys");
  return j;
}

// the producer record of an added node: the command, the program, the input files' names, the user's description
Json producer_of(const std::string& command, const std::string& created_by, const Json& inputs, const std::optional<std::string>& description) {
  Json p = Json{{"command", command}, {"program", created_by}, {"inputs", inputs}};
  if (description) p["description"] = *description;
  return p;
}

Node require_manifold(const Subject& ts, const std::string& name, const std::string& option) {
  const auto m = ts.by_id_or_name(Cls::manifold, name);
  if (!m) throw Error(code::no_such_row, "the subject holds no manifold named '" + name + "' (" + option + ")");
  return *m;
}

// a node of the same name already in the subject: refused unless `replace`, then moved aside (never deleted); returns where its arrays went
std::string clear_name(const Subject& ts, Unit& u, Cls cls, const std::string& name, const std::string& fallback_path, bool replace) {
  const auto have = ts.by_id_or_name(cls, name);
  const std::string what = std::string(to_string(cls)) + " " + name;
  if (have && !replace) throw Error(code::already_exists, "the subject already holds a " + std::string(to_string(cls)) + " named '" + name + "': pass --replace to move it aside");
  if (have && !have->path) {   // a node with no arrays (an operator stored by its method): nothing to move aside, its row goes
    u.remove_unstored(cls, have->id);
    return {};
  }
  return u.replace(cls, have && have->path ? *have->path : fallback_path, replace, what);
}

Json stored(const char* cls, const std::string& name, const std::string& id, const std::string& path, const std::string& moved) {
  Json j = Json{{"class", cls}, {"name", name}, {"id", id}, {"path", path}};
  if (!moved.empty()) j["replaced"] = moved;
  return j;
}

// the number of cells of a manifold its values sit on: "0" vertices, "1" edges, "2" faces
std::int64_t cell_count(const Node& m, const std::string& cell) {
  if (cell == "0") return m.n_vertices.value_or(-1);
  const Json n = m.fact(cell == "1" ? "n_edges" : cell == "2" ? "n_faces" : "");
  if (cell != "1" && cell != "2") throw Refused("--cell " + cell + " (0 vertices, 1 edges, 2 faces)");
  return n.is_number_integer() ? n.get<std::int64_t>() : -1;
}


// what a field's values must be: the manifold's cells (and a time line's samples), then the value type's components
struct FieldShape {
  Node m;
  std::int64_t n = 0, samples = -1;
  std::optional<Node> line;
  std::vector<std::int64_t> want;
};
FieldShape plan_field_shape(const Subject& ts, const AddFieldRequest& rq, const std::vector<std::int64_t>& elem, const std::string& cell) {
  FieldShape p{require_manifold(ts, rq.manifold, "--manifold"), 0, -1, std::nullopt, {}};
  p.n = cell_count(p.m, cell);
  p.want = {p.n};
  if (rq.time) {
    p.line = require_manifold(ts, *rq.time, "--time");
    if (p.line->type != std::optional<std::string>("line")) throw Refused("--time " + *rq.time + " is a " + p.line->type.value_or("manifold") + ", not a line");
    p.samples = p.line->n_vertices.value_or(-1);
    p.want.push_back(p.samples);
  }
  p.want.insert(p.want.end(), elem.begin(), elem.end());
  return p;
}
std::string field_shape_message(const FieldShape& p, const AddFieldRequest& rq, const std::string& what, const std::vector<std::int64_t>& got, const std::string& cell) {
  return what + " is " + shape_text(got) + "; a " + rq.value_type + " field on '" + p.m.name + "' (" + std::to_string(p.n) + " " + (cell == "0" ? "vertices" : cell == "1" ? "edges" : "faces") +
         (rq.time ? ", " + std::to_string(p.samples) + " samples of " + *rq.time : std::string()) + ") is " + shape_text(p.want);
}

// a statistic's values: one axis a --axes entry, then the value type's components; each axis the size of the manifold it runs along
void check_statistic_rank(const AddStatisticRequest& rq, const std::vector<std::int64_t>& shape, const std::string& what) {
  if (rq.axes.empty()) throw Refused("name each axis of the values and the manifold it runs along: --axes NAME:MANIFOLD ...");
  const std::vector<std::int64_t> elem = element_shape(rq.value_type);
  if (shape.size() != rq.axes.size() + elem.size())
    throw Refused(what + " is " + shape_text(shape) + ": " + std::to_string(rq.axes.size()) + " axes and a " + rq.value_type + " value make " + std::to_string(rq.axes.size() + elem.size()) + " dimensions");
}
void check_statistic_axis(const Node& m, const std::string& axis, std::int64_t size, const std::string& what) {
  const std::string name = axis.substr(0, axis.find(':'));
  if (m.n_vertices && *m.n_vertices != size)
    throw Refused(what + " has " + std::to_string(size) + " along axis '" + name + "'; '" + m.name + "' has " + std::to_string(*m.n_vertices) + " elements");
}

// the options the one-shot form of add field | statistic requires (the write in pieces takes fewer: its begin names them once)
void need_options(const std::vector<std::pair<const char*, const std::string&>>& given, const ArrayInput& values, const std::string& command) {
  std::string missing;
  for (const auto& [option, v] : given) if (v.empty()) missing += (missing.empty() ? "" : ", ") + std::string(option);
  if (!values.in_memory && values.path.empty()) missing += (missing.empty() ? "" : ", ") + std::string("--values");
  if (!missing.empty()) throw Refused("the following arguments are required: " + missing + (command.empty() ? "" : " (or --begin, to write in pieces)"));
}
}  // namespace

CommandResult add_field(const AddFieldRequest& rq, const Progress& progress) {
  if (rq.piece.active()) return pieces::field(rq, progress);
  need_options({{"--subject", rq.subject}, {"--name", rq.name}, {"--manifold", rq.manifold}}, rq.values, "add field");
  const Values v = values_of(rq.values, "--values");
  const std::vector<std::int64_t> elem = element_shape(rq.value_type);
  const std::string cell = rq.cell.value_or("0");
  if (rq.time && cell != "0") throw Refused("a time-resolved field (--time) lies on the manifold's vertices (--cell 0)");
  Json inputs = Json{{"values", file_name(rq.values)}};
  const Json maker = maker_producer(rq.producer);
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add field", rq.created_by, inputs, rq.description), progress, [&](const Subject& ts, Unit& u, DatasetWriter& ds) {
    const FieldShape plan = plan_field_shape(ts, rq, elem, cell);
    const Node& m = plan.m;
    const std::optional<Node>& line = plan.line;
    std::vector<std::int64_t> shape = v.shape;
    if (!rq.time && elem.empty() && shape.size() == 2 && shape[1] == 1) shape.pop_back();   // [n x 1] is a scalar a cell
    if (shape != plan.want) throw Refused(field_shape_message(plan, rq, "--values " + file_name(rq.values), v.shape, cell));
    FieldOut f;
    f.name = rq.name;
    f.output = rq.name;
    f.kind = rq.kind;
    f.manifold_id = rq.time ? u.product(ProductOut{{m.id, line->id}, std::nullopt}) : m.id;
    f.value_type = rq.value_type;
    f.unit = rq.unit;
    f.description = rq.description;
    if (rq.cell) f.cell = cell;
    if (rq.derived_from) {
      const auto d = ts.by_id_or_name(Cls::field, *rq.derived_from);
      if (!d) throw Error(code::no_such_row, "the subject holds no field named '" + *rq.derived_from + "' (--derived-from)");
      f.derived_from_id = d->id;
    }
    if (rq.by_operator) {
      const auto o = ts.by_id_or_name(Cls::operator_, *rq.by_operator);
      if (!o) throw Error(code::no_such_row, "the subject holds no operator named '" + *rq.by_operator + "' (--by-operator)");
      f.by_operator_id = o->id;
    }
    std::set<std::pair<std::string, std::string>> seen;
    for (const std::string& c : rq.contribution) {   // SUBJECT:FIELD[:WEIGHT]
      std::string rest = c, weight_text;
      const auto first = rest.find(':'), last = rest.rfind(':');
      if (first == std::string::npos || first == 0 || first + 1 == rest.size()) throw Refused("--contribution " + c + ": a contribution is SUBJECT:FIELD or SUBJECT:FIELD:WEIGHT");
      double weight = 1;
      if (last != first) {   // a trailing number is the weight
        try {
          std::size_t used = 0;
          const std::string tail = rest.substr(last + 1);
          const double w = std::stod(tail, &used);
          if (used == tail.size()) { weight = w; rest = rest.substr(0, last); }
        } catch (const std::exception&) {}
      }
      if (!std::isfinite(weight)) throw Refused("--contribution " + c + ": the weight is a finite number");
      const std::string subject_key = rest.substr(0, first), field_key = rest.substr(first + 1);
      const auto owner = ds.db().one("SELECT id FROM subject WHERE id = ? OR name = ? ORDER BY (id = ?) DESC", {V(subject_key), V(subject_key), V(subject_key)});
      if (!owner) throw Error(code::no_such_row, "the dataset holds no subject named '" + subject_key + "' (--contribution " + c + ")");
      const std::string owner_id = text(*owner, "id");
      const auto source = Subject::of(ds, owner_id).by_id_or_name(Cls::field, field_key);
      if (!source) throw Error(code::no_such_row, "the subject '" + subject_key + "' holds no field named '" + field_key + "' (--contribution " + c + ")");
      if (!seen.insert({owner_id, source->id}).second) throw Refused("--contribution " + c + ": that field is named twice");
      f.contributions.push_back(FieldOut::Contribution{owner_id, source->id, weight});
    }
    f.producer = maker;
    f.data = v;
    f.data.shape = shape;
    const std::string path = u.path_of(f);
    const std::string moved = clear_name(ts, u, Cls::field, rq.name, path, rq.replace);
    const std::string id = u.field(f);
    Json out = stored("field", rq.name, id, path, moved);
    out["manifold"] = m.name;
    if (rq.time) out["time"] = *rq.time;
    out["cell"] = cell;
    out["shape"] = shape;
    out["dtype"] = dtype_of(v);
    if (rq.by_operator) out["by_operator"] = *rq.by_operator;
    if (!f.contributions.empty()) out["contributions"] = f.contributions.size();
    return out;
  });
}

CommandResult add_selection(const AddSelectionRequest& rq, const Progress& progress) {
  const std::map<std::string, std::optional<ArrayInput>> files = {{"partition", rq.members}, {"set", rq.members}, {"indices", rq.indices}, {"spans", rq.spans}};
  const auto it = files.find(rq.type);
  if (it == files.end()) throw Refused("--type " + rq.type + " (partition, set, indices or spans)");
  const char* option = rq.type == "indices" ? "--indices" : rq.type == "spans" ? "--spans" : "--members";
  if (!it->second) throw Refused("a selection of type " + rq.type + " takes its arrays from " + option + " FILE.npy");
  if ((rq.members ? 1 : 0) + (rq.indices ? 1 : 0) + (rq.spans ? 1 : 0) != 1) throw Refused("give one of --members, --indices and --spans: the one its --type takes (" + std::string(option) + ")");
  const ArrayInput path_in = *it->second;
  const Values v = values_of(path_in, option);
  Json inputs = Json{{option + 2, file_name(path_in)}};
  std::optional<Values> weights;
  if (rq.weights) {
    weights = values_of(*rq.weights, "--weights");
    need_real(*weights, "--weights", *rq.weights);
    if (weights->shape.size() != 1) throw Refused("--weights " + file_name(*rq.weights) + " is " + shape_text(weights->shape) + ": one weight a member [members]");
    for (const float x : weights->f32) if (!std::isfinite(x)) throw Refused("--weights " + file_name(*rq.weights) + ": a weight is not a finite number");
    for (const double x : weights->f64) if (!std::isfinite(x)) throw Refused("--weights " + file_name(*rq.weights) + ": a weight is not a finite number");
    inputs["weights"] = file_name(*rq.weights);
  }
  const Json maker = maker_producer(rq.producer);
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add selection", rq.created_by, inputs, rq.description), progress, [&](const Subject& ts, Unit& u) {
    const Node m = require_manifold(ts, rq.manifold, "--manifold");
    const std::int64_t n = m.n_vertices.value_or(-1);
    SelectionOut s;
    s.name = rq.name;
    s.output = rq.name;
    s.type = rq.type;
    s.manifold_id = m.id;
    s.description = rq.description;
    s.entries = rq.entries;
    s.producer = maker;
    if (rq.of_field) {
      const auto f = ts.by_id_or_name(Cls::field, *rq.of_field);
      if (!f) throw Error(code::no_such_row, "the subject holds no field named '" + *rq.of_field + "' (--of-field)");
      s.of_field_id = f->id;
    }
    if (rq.type == "partition" || rq.type == "set") {
      if (!v.f32.empty() || !v.f64.empty()) throw Refused("--members " + file_name(path_in) + " holds " + dtype_of(v) + " values: a label per element is an integer (-1: none)");
      if (v.shape.size() != 1 || v.shape[0] != n)
        throw Refused("--members " + file_name(path_in) + " is " + shape_text(v.shape) + "; a label per element of '" + m.name + "' is [" + std::to_string(n) + "]");
      // the labels as io stores a set: each element's member NUMBER (0 .. k-1, -1: none) and each member's label, ascending
      std::vector<std::int64_t> labels;
      if (!v.u8.empty()) labels.assign(v.u8.begin(), v.u8.end()); else labels.assign(v.i32.begin(), v.i32.end());
      std::set<std::int64_t> distinct;
      for (const auto l : labels) if (l >= 0) distinct.insert(l);
      s.member_codes.assign(distinct.begin(), distinct.end());
      std::map<std::int64_t, std::int32_t> number;
      for (std::size_t k = 0; k < s.member_codes.size(); ++k) number[s.member_codes[k]] = static_cast<std::int32_t>(k);
      for (const auto l : labels) s.member_of.push_back(l < 0 ? -1 : number.at(l));
      if (s.member_codes.empty()) throw Refused("--members " + file_name(path_in) + " labels no element (every label is negative)");
    } else if (rq.type == "indices") {
      if (!v.f32.empty() || !v.f64.empty() || v.shape.size() != 1) throw Refused("--indices " + file_name(path_in) + " is " + shape_text(v.shape) + " " + dtype_of(v) + ": the elements picked are integers [k]");
      std::vector<std::int64_t> picks;
      if (!v.u8.empty()) picks.assign(v.u8.begin(), v.u8.end()); else picks.assign(v.i32.begin(), v.i32.end());
      for (const auto i : picks) if (i < 0 || i >= n) throw Refused("--indices " + file_name(path_in) + ": element " + std::to_string(i) + " is not one of the " + std::to_string(n) + " of '" + m.name + "'");
      s.codes = picks;
    } else {
      if (m.type != std::optional<std::string>("line")) throw Refused("spans lie on a line: '" + m.name + "' is a " + m.type.value_or("manifold"));
      if (v.shape.size() != 2 || (v.shape[1] != 2 && v.shape[1] != 3)) throw Refused("--spans " + file_name(path_in) + " is " + shape_text(v.shape) + ": spans are [k x 2] (start, stop) or [k x 3] (start, stop, code)");
      const Values d = v.i32.empty() && v.u8.empty() ? as_f64(v) : Values{};
      if (d.f64.empty() && v.shape[0] > 0) throw Refused("--spans " + file_name(path_in) + ": start and stop are float32 or float64, in the line's unit");
      const std::int64_t w = v.shape[1];
      for (std::int64_t k = 0; k < v.shape[0]; ++k) {
        Span sp;
        sp.ordinal = k;
        sp.start = d.f64[static_cast<std::size_t>(k * w)];
        sp.stop = d.f64[static_cast<std::size_t>(k * w + 1)];
        if (!(sp.stop >= sp.start)) throw Refused("--spans " + file_name(path_in) + ": span " + std::to_string(k) + " stops before it starts");
        if (w == 3 && d.f64[static_cast<std::size_t>(k * w + 2)] >= 0) sp.code = static_cast<std::int64_t>(d.f64[static_cast<std::size_t>(k * w + 2)]);
        s.spans.push_back(sp);
      }
    }
    if (weights) {   // one weight a member: a set's labelled members, the picks, the spans
      const std::int64_t members = rq.type == "indices" ? static_cast<std::int64_t>(s.codes.size()) : rq.type == "spans" ? static_cast<std::int64_t>(s.spans.size()) : static_cast<std::int64_t>(s.member_codes.size());
      if (weights->shape[0] != members)
        throw Refused("--weights " + file_name(*rq.weights) + " has " + std::to_string(weights->shape[0]) + " values; '" + rq.name + "' has " + std::to_string(members) + " members (one weight a member)");
      s.weights = *weights;
    }
    const std::string path = u.path_of(s);
    const std::string moved = clear_name(ts, u, Cls::selection, rq.name, path, rq.replace);
    const std::string id = u.selection(s);
    Json out = stored("selection", rq.name, id, path, moved);
    out["manifold"] = m.name;
    out["type"] = rq.type;
    out["shape"] = v.shape;
    if (!s.member_codes.empty()) out["members"] = s.member_codes.size();
    if (rq.of_field) out["of_field"] = *rq.of_field;
    if (weights) out["weights"] = weights->shape[0];
    return out;
  });
}

CommandResult add_statistic(const AddStatisticRequest& rq, const Progress& progress) {
  if (rq.piece.active()) return pieces::statistic(rq, progress);
  need_options({{"--subject", rq.subject}, {"--name", rq.name}, {"--measure", rq.measure}, {"--method", rq.method}}, rq.values, "add statistic");
  if (rq.axes.empty()) throw Refused("the following arguments are required: --axes");
  const Values v = values_of(rq.values, "--values");
  need_real(v, "--values", rq.values);
  if (rq.axes.empty()) throw Refused("name each axis of the values and the manifold it runs along: --axes NAME:MANIFOLD ...");
  check_statistic_rank(rq, v.shape, "--values " + file_name(rq.values));
  Json inputs = Json{{"values", file_name(rq.values)}};
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add statistic", rq.created_by, inputs, rq.description), progress, [&](const Subject& ts, Unit& u) {
    StatisticOut st;
    st.name = rq.name;
    st.output = rq.name;
    st.measure = rq.measure;
    st.method = rq.method;
    st.value_type = rq.value_type;
    st.unit = rq.unit;
    st.description = rq.description;
    st.axes = Json::array();
    for (std::size_t i = 0; i < rq.axes.size(); ++i) {
      const std::string& a = rq.axes[i];
      const auto colon = a.find(':');
      if (colon == std::string::npos || colon == 0 || colon + 1 == a.size()) throw Refused("--axes " + a + ": an axis is NAME:MANIFOLD");
      const Node m = require_manifold(ts, a.substr(colon + 1), "--axes " + a);
      check_statistic_axis(m, a, v.shape[i], "--values " + file_name(rq.values));
      st.axes.push_back(Json{{"name", a.substr(0, colon)}, {"size", v.shape[i]}, {"manifold_id", m.id}});
    }
    if (rq.of_field) {
      const auto f = ts.by_id_or_name(Cls::field, *rq.of_field);
      if (!f) throw Error(code::no_such_row, "the subject holds no field named '" + *rq.of_field + "' (--of-field)");
      st.of_field_id = f->id;
    } else {
      st.detached = true;                               // computed elsewhere: of no field of the subject, its provenance in its options
      st.options = Json{{"computed_elsewhere", true}};
    }
    if (rq.partition) {
      const auto p = ts.by_id_or_name(Cls::selection, *rq.partition);
      if (!p) throw Error(code::no_such_row, "the subject holds no selection named '" + *rq.partition + "' (--partition)");
      st.partition_selection_id = p->id;
    }
    st.data = v;
    const std::string path = u.path_of(st);
    const std::string moved = clear_name(ts, u, Cls::statistic, rq.name, path, rq.replace);
    const std::string id = u.statistic(st);
    Json out = stored("statistic", rq.name, id, path, moved);
    out["shape"] = v.shape;
    out["dtype"] = dtype_of(v);
    return out;
  });
}

CommandResult add_measurements(const AddMeasurementsRequest& rq, const Progress& progress) {
  if (rq.measure.empty() && rq.measure_arrays.empty()) throw Refused("give each measure and its values: --measure NAME=FILE.npy ...");
  std::vector<std::pair<std::string, ArrayInput>> given;
  Json inputs = Json::object();
  for (const auto& m : rq.measure) {
    const auto eq = m.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == m.size()) throw Refused("--measure " + m + ": a measure is NAME=FILE.npy");
    given.emplace_back(m.substr(0, eq), ArrayInput(m.substr(eq + 1)));
    inputs[m.substr(0, eq)] = file_name(given.back().second);
  }
  for (const auto& [name, array] : rq.measure_arrays) {
    given.emplace_back(name, array);
    inputs[name] = file_name(array);
  }
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add measurements", rq.created_by, inputs, std::nullopt), progress, [&](const Subject& ts, Unit& u) {
    const auto sel = ts.by_id_or_name(Cls::selection, rq.selection);
    if (!sel) throw Error(code::no_such_row, "the subject holds no selection named '" + rq.selection + "' (--selection)");
    std::optional<Node> field;
    if (rq.field) {
      field = ts.by_id_or_name(Cls::field, *rq.field);
      if (!field) throw Error(code::no_such_row, "the subject holds no field named '" + *rq.field + "' (--field)");
    }
    // the members' codes, in member order: a set's labels (as its measurements are keyed), else the members' ordinals
    std::vector<std::int64_t> codes;
    try {
      codes = ts.element_members(sel->id).member_codes;
    } catch (const Error&) {}
    if (codes.empty() && sel->n_members) for (std::int64_t k = 0; k < *sel->n_members; ++k) codes.push_back(k);
    if (codes.empty()) throw Refused("'" + sel->name + "' has no members to measure (a set, partition, pick or spans has)");
    Json rows = Json::array();
    Json replaced = Json::array();
    for (const auto& [measure, file] : given) {
      const Values v = values_of(file, "--measure " + measure);
      need_real(v, "--measure " + measure, file);
      if (v.shape.size() != 1 || v.shape[0] != static_cast<std::int64_t>(codes.size()))
        throw Refused("--measure " + measure + "=" + file_name(file) + " is " + shape_text(v.shape) + "; one value a member of '" + sel->name + "' is [" + std::to_string(codes.size()) + "]");
      if (ts.has_measure(sel->id, measure, field ? std::optional<std::string>(field->id) : std::nullopt)) {
        if (!rq.replace) throw Error(code::already_exists, "'" + sel->name + "' already holds the measure '" + measure + "'" + (field ? " of " + field->name : std::string()) + ": pass --replace");
        u.remove_measurements(sel->id, field ? std::optional<std::string>(field->id) : std::nullopt, measure);
        replaced.push_back(measure);
      }
      const Values d = as_f64(v);                       // a measurement row's value is a float64
      for (std::size_t k = 0; k < codes.size(); ++k) {
        Json row = Json{{"selection_id", sel->id}, {"code", codes[k]}, {"measure", measure}, {"value", d.f64[k]}};
        if (field) row["of_field_id"] = field->id;
        if (rq.unit) row["unit"] = *rq.unit;
        rows.push_back(std::move(row));
      }
    }
    const std::int64_t written = u.write_measurements(rows);
    Json out = Json{{"class", "measurements"}, {"selection", sel->name}, {"field", field ? Json(field->name) : Json()}, {"measures", static_cast<std::int64_t>(given.size())}, {"rows", written}};
    if (!replaced.empty()) out["replaced"] = replaced;
    return out;
  });
}

CommandResult add_operator(const AddOperatorRequest& rq, const Progress& progress) {
  if (rq.layout && *rq.layout != "dense" && *rq.layout != "csr" && *rq.layout != "analytic") throw Refused("--layout " + *rq.layout + " (dense, csr or analytic)");
  const bool analytic = rq.layout && *rq.layout == "analytic";
  if (analytic) {
    if (!rq.method || rq.method->empty()) throw Refused("--layout analytic takes --method NAME: how the operator is evaluated");
    if (rq.values || rq.data || rq.indices || rq.indptr) throw Refused("an analytic operator is stored by its method and holds no matrix: leave out --values, --data, --indices and --indptr");
  } else if (rq.method || rq.params) {
    throw Refused("--method and --params belong to --layout analytic");
  }
  const bool sparse = !analytic && (rq.data || rq.indices || rq.indptr);
  if (!analytic) {
    if (sparse == rq.values.has_value()) throw Refused("give --values FILE.npy (dense) or --data, --indices and --indptr (sparse, CSR): one of the two (or --layout analytic with --method)");
    if (sparse && !(rq.data && rq.indices && rq.indptr)) throw Refused("a sparse operator takes all three of --data, --indices and --indptr (scipy.sparse.csr_matrix's arrays)");
    if (rq.layout && *rq.layout == "dense" && sparse) throw Refused("--layout dense takes --values");
    if (rq.layout && *rq.layout == "csr" && !sparse) throw Refused("--layout csr takes --data, --indices and --indptr");
  }
  Json inputs = analytic ? Json::object() : sparse ? Json{{"data", file_name(*rq.data)}, {"indices", file_name(*rq.indices)}, {"indptr", file_name(*rq.indptr)}} : Json{{"values", file_name(*rq.values)}};
  Values v;
  if (!analytic) {
    v = values_of(sparse ? *rq.data : *rq.values, sparse ? "--data" : "--values");
    need_real(v, sparse ? "--data" : "--values", sparse ? *rq.data : *rq.values);
  }
  const Json parameters = rq.params ? json_of(*rq.params, "--params") : Json();
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add operator", rq.created_by, inputs, rq.description), progress, [&](const Subject& ts, Unit& u) {
    const Node from = require_manifold(ts, rq.from_manifold, "--from-manifold"), to = require_manifold(ts, rq.to_manifold, "--to-manifold");
    const std::int64_t rows = to.n_vertices.value_or(-1), cols = from.n_vertices.value_or(-1);
    OperatorOut o;
    o.name = rq.name;
    o.output = rq.name;
    o.kind = rq.kind;
    o.from_manifold_id = from.id;
    o.to_manifold_id = to.id;
    o.description = rq.description;
    o.data = v;
    o.params = parameters;
    if (rq.from_selection) {
      const auto sel = ts.by_id_or_name(Cls::selection, *rq.from_selection);
      if (!sel) throw Error(code::no_such_row, "the subject holds no selection named '" + *rq.from_selection + "' (--from-selection)");
      if (sel->manifold_id != std::optional<std::string>(from.id)) throw Refused("--from-selection " + *rq.from_selection + " does not lie on '" + from.name + "' (the --from-manifold)");
      o.from_selection_id = sel->id;
    }
    if (analytic) {
      o.analytic = OperatorOut::Analytic{*rq.method, rows, cols};
    } else if (sparse) {
      if (v.shape.size() != 1) throw Refused("--data " + file_name(*rq.data) + " is " + shape_text(v.shape) + ": the nonzero values [nnz]");
      const Values ix = values_of(*rq.indices, "--indices"), ip = values_of(*rq.indptr, "--indptr");
      if (ix.i32.empty() && ix.u8.empty() && v.shape[0] > 0) throw Refused("--indices " + file_name(*rq.indices) + ": the column of each value, an integer");
      OperatorOut::Sparse sp;
      sp.layout = "csr";
      sp.n_rows = rows;
      sp.n_cols = cols;
      sp.indices = ix.i32;
      if (!ix.u8.empty()) sp.indices.assign(ix.u8.begin(), ix.u8.end());
      if (!ip.i32.empty()) sp.indptr.assign(ip.i32.begin(), ip.i32.end());
      else if (!ip.u8.empty()) sp.indptr.assign(ip.u8.begin(), ip.u8.end());
      else throw Refused("--indptr " + file_name(*rq.indptr) + ": the row offsets, integers");
      if (static_cast<std::int64_t>(sp.indptr.size()) != rows + 1)
        throw Refused("--indptr " + file_name(*rq.indptr) + " has " + std::to_string(sp.indptr.size()) + " offsets; an operator to '" + to.name + "' (" + std::to_string(rows) + " rows) has " + std::to_string(rows + 1));
      for (const auto c : sp.indices) if (c < 0 || c >= cols) throw Refused("--indices " + file_name(*rq.indices) + ": column " + std::to_string(c) + " is not one of the " + std::to_string(cols) + " of '" + from.name + "'");
      o.sparse = sp;
    } else if (v.shape != std::vector<std::int64_t>{rows, cols}) {
      throw Refused("--values " + file_name(*rq.values) + " is " + shape_text(v.shape) + "; an operator from '" + from.name + "' (" + std::to_string(cols) + ") to '" + to.name + "' (" +
                    std::to_string(rows) + ") is [" + std::to_string(rows) + " x " + std::to_string(cols) + "]");
    }
    const std::string path = u.path_of(o);
    const std::string moved = clear_name(ts, u, Cls::operator_, rq.name, path, rq.replace);
    const std::string id = u.operator_(o);
    Json out = stored("operator", rq.name, id, analytic ? std::string() : path, moved);
    out["layout"] = analytic ? "analytic" : sparse ? "csr" : "dense";
    out["shape"] = Json::array({rows, cols});
    if (analytic) out["method"] = *rq.method; else out["dtype"] = dtype_of(v);
    return out;
  });
}

CommandResult add_points(const AddPointsRequest& rq, const Progress& progress) {
  const Values v = values_of(rq.positions, "--positions");
  need_real(v, "--positions", rq.positions);
  const bool xyz = v.shape.size() == 2 && v.shape[1] == 3;
  if (v.shape.size() != 1 && !xyz) throw Refused("--positions " + file_name(rq.positions) + " is " + shape_text(v.shape) + ": positions [n x 3] or one coordinate each [n]");
  Json inputs = Json{{"positions", file_name(rq.positions)}};
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add points", rq.created_by, inputs, rq.description), progress, [&](const Subject& ts, Unit& u) {
    PointsOut p;
    p.name = rq.name;
    p.output = rq.name;
    p.cs_key = rq.coordinate_system.value_or(xyz ? "scs" : rq.name);
    p.description = rq.description;
    p.positions = as_f64(v);                            // io stores a manifold's coordinates in float64
    const std::string path = u.path_of(p);
    const std::string moved = clear_name(ts, u, Cls::manifold, rq.name, path, rq.replace);
    const std::string id = u.points(p);
    Json out = stored("manifold", rq.name, id, path, moved);
    out["type"] = "points";
    out["n"] = v.shape[0];
    out["coordinate_system"] = p.cs_key;
    return out;
  });
}

CommandResult add_line(const AddLineRequest& rq, const Progress& progress) {
  if (rq.unit.empty()) throw Refused("--unit: the unit of the line's coordinates (s for a time line)");
  Values coordinates, kinds;
  Json inputs = Json::object();
  if (rq.coordinates) {   // a line whose points are not evenly spaced: its coordinates, and a code a point
    if (rq.n || rq.spacing || rq.origin) throw Refused("--coordinates gives every point: leave out --n, --spacing and --origin");
    if (rq.dimension != "eigenvalue" && rq.dimension.rfind("eigenvalue:", 0) != 0)
      throw Refused("--coordinates stores a line whose points are not evenly spaced under the dimension it runs along: --dimension eigenvalue (an operator's spectrum), the one the layout places (a time or frequency line is evenly spaced: --n and --spacing)");
    coordinates = values_of(*rq.coordinates, "--coordinates");
    need_real(coordinates, "--coordinates", *rq.coordinates);
    if (coordinates.shape.size() != 1 || coordinates.shape[0] < 1) throw Refused("--coordinates " + file_name(*rq.coordinates) + " is " + shape_text(coordinates.shape) + ": one coordinate a point [n]");
    coordinates = as_f64(coordinates);
    inputs["coordinates"] = file_name(*rq.coordinates);
    if (rq.kinds) {
      kinds = values_of(*rq.kinds, "--kinds");
      if (!kinds.f32.empty() || !kinds.f64.empty()) throw Refused("--kinds " + file_name(*rq.kinds) + " holds " + dtype_of(kinds) + " values: one integer code a point");
      if (kinds.shape != coordinates.shape) throw Refused("--kinds " + file_name(*rq.kinds) + " is " + shape_text(kinds.shape) + "; one code a point is " + shape_text(coordinates.shape));
      if (kinds.i32.empty()) kinds.i32.assign(kinds.u8.begin(), kinds.u8.end());
      inputs["kinds"] = file_name(*rq.kinds);
    }
  } else {
    if (rq.kinds) throw Refused("--kinds goes with --coordinates");
    if (!rq.n) throw Refused("give --n and --spacing (an evenly spaced line), or --coordinates (points as given)");
    if (!rq.spacing) throw Refused("--spacing: the line's step, in --unit");
    if (*rq.n < 1) throw Refused("--n: a line has at least one point");
    if (!(*rq.spacing > 0) || !std::isfinite(*rq.spacing)) throw Refused("--spacing: a positive number (the line's step, in --unit)");
    if (rq.dimension == "time" && rq.unit != "s") throw Refused("a time line is in seconds (--unit s), the clock every recording is read on");
  }
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add line", rq.created_by, inputs, std::nullopt), progress, [&](const Subject& ts, Unit& u) {
    LineOut l;
    l.name = rq.name;
    l.n = rq.n.value_or(0);
    l.spacing = rq.spacing.value_or(1);
    l.origin = rq.origin.value_or(0);
    l.unit = rq.unit;
    l.dimension = rq.dimension;
    if (rq.coordinates) { l.coordinates = coordinates.f64; l.kinds = kinds.i32; l.n = static_cast<std::int64_t>(l.coordinates.size()); l.folder = "eigen"; l.cs_key = rq.dimension; }
    else if (rq.dimension == "time") { l.folder = "time"; l.cs_key = "clock"; }
    else if (rq.dimension != "frequency") { l.folder = "line"; l.cs_key = rq.dimension; }
    std::string moved;
    if (const auto have = ts.by_id_or_name(Cls::manifold, rq.name)) {
      if (!rq.replace) throw Error(code::already_exists, "the subject already holds a manifold named '" + rq.name + "': pass --replace to move it aside");
      moved = u.replace(Cls::manifold, have->path.value_or(""), true, "manifold " + rq.name);
    }
    const std::string id = u.line(l);
    Json out = stored("manifold", rq.name, id, ts.by_id(Cls::manifold, id) ? ts.by_id(Cls::manifold, id)->path.value_or("") : std::string(), moved);
    out["type"] = "line";
    out["n"] = l.n;
    out["dimension"] = rq.dimension;
    if (rq.coordinates) out["spacing"] = "uneven";
    return out;
  });
}

CommandResult add_product(const AddProductRequest& rq, const Progress& progress) {
  if (rq.name.empty()) throw Refused("--name: the name the product is stored under");
  if (rq.factors.size() < 2) throw Refused("--factors A,B: the manifolds the product is of, two or more, in axis order");
  return in_unit(rq.dataset, rq.subject, rq.templates, rq.created_by, producer_of("add product", rq.created_by, Json{{"factors", rq.factors}}, std::nullopt), progress, [&](const Subject& ts, Unit& u) {
    if (ts.by_id_or_name(Cls::manifold, rq.name)) throw Error(code::already_exists, "the subject already holds a manifold named '" + rq.name + "'");
    ProductOut p;
    Json names = Json::array();
    for (const std::string& f : rq.factors) {
      const Node m = require_manifold(ts, f, "--factors");
      if (m.type == std::optional<std::string>("product")) throw Refused("--factors " + f + " is a product: the factors of a product are not products themselves");
      p.factors.push_back(m.id);
      names.push_back(m.name);
    }
    p.name = rq.name;
    const std::string id = u.product(p);
    const auto made = ts.by_id(Cls::manifold, id);
    if (made && made->name != rq.name) throw Error(code::already_exists, "the factors already make the product '" + made->name + "'");
    Json out = stored("manifold", rq.name, id, std::string(), std::string());
    out["type"] = "product";
    out["factors"] = names;
    out["n"] = made && made->n_vertices ? Json(*made->n_vertices) : Json();
    return out;
  });
}


// ---- what `begin` of a write in pieces checks (pieces.h): the one-shot command's refusals that do not need the values ----
namespace pieces {

void plan_field(const AddFieldRequest& rq, const std::vector<std::int64_t>& shape, const Progress& progress) {
  need_options({{"--subject", rq.subject}, {"--name", rq.name}, {"--manifold", rq.manifold}}, ArrayInput("-"), "");
  const std::vector<std::int64_t> elem = element_shape(rq.value_type);
  const std::string cell = rq.cell.value_or("0");
  if (rq.time && cell != "0") throw Refused("a time-resolved field (--time) lies on the manifold's vertices (--cell 0)");
  Lease lease = open_reader(rq.dataset, rq.subject, rq.templates, progress);
  SubjectWriter sub = SubjectWriter::open(*lease, rq.subject);
  const Subject ts = Subject::of(sub);
  const FieldShape plan = plan_field_shape(ts, rq, elem, cell);
  if (shape != plan.want) throw Refused(field_shape_message(plan, rq, "--shape", shape, cell));
  if (ts.by_id_or_name(Cls::field, rq.name) && !rq.replace) throw Error(code::already_exists, "the subject already holds a field named '" + rq.name + "': pass --replace to move it aside");
  if (rq.derived_from && !ts.by_id_or_name(Cls::field, *rq.derived_from)) throw Error(code::no_such_row, "the subject holds no field named '" + *rq.derived_from + "' (--derived-from)");
  if (rq.by_operator && !ts.by_id_or_name(Cls::operator_, *rq.by_operator)) throw Error(code::no_such_row, "the subject holds no operator named '" + *rq.by_operator + "' (--by-operator)");
  (void)maker_producer(rq.producer);
}

void plan_statistic(const AddStatisticRequest& rq, const std::vector<std::int64_t>& shape, const Progress& progress) {
  need_options({{"--subject", rq.subject}, {"--name", rq.name}, {"--measure", rq.measure}, {"--method", rq.method}}, ArrayInput("-"), "");
  check_statistic_rank(rq, shape, "--shape");
  Lease lease = open_reader(rq.dataset, rq.subject, rq.templates, progress);
  SubjectWriter sub = SubjectWriter::open(*lease, rq.subject);
  const Subject ts = Subject::of(sub);
  for (std::size_t i = 0; i < rq.axes.size(); ++i) {
    const std::string& a = rq.axes[i];
    const auto colon = a.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == a.size()) throw Refused("--axes " + a + ": an axis is NAME:MANIFOLD");
    check_statistic_axis(require_manifold(ts, a.substr(colon + 1), "--axes " + a), a, shape[i], "--shape " + shape_text(shape));
  }
  if (ts.by_id_or_name(Cls::statistic, rq.name) && !rq.replace) throw Error(code::already_exists, "the subject already holds a statistic named '" + rq.name + "': pass --replace to move it aside");
  if (rq.of_field && !ts.by_id_or_name(Cls::field, *rq.of_field)) throw Error(code::no_such_row, "the subject holds no field named '" + *rq.of_field + "' (--of-field)");
  if (rq.partition && !ts.by_id_or_name(Cls::selection, *rq.partition)) throw Error(code::no_such_row, "the subject holds no selection named '" + *rq.partition + "' (--partition)");
}

}  // namespace pieces

}  // namespace nxr::io::task
