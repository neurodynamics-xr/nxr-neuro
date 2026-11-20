// THE COMMANDS IN A FACE (docs/specs/2026-11-19-io-service.md, sections 2 and 5): `nxr-neuro <verb> <noun> --option VALUE` called from a binding as
// command(verb, noun, {option: value}). The options arrive as a JSON object keyed by the command's long options with the dashes as underscores (`--from-sphere` is
// "from_sphere"; `--from` is "from"; the positional argument by its placeholder in lower case: "protocol", "folder", "file", "definition", "templates_dir"). This header only
// CONVERTS: each command's options become its library request (nxr/io/task/commands.h, nxr/pipelines/face.h) and the library function is called once. No maths, no SQL.
//
// Values: a string, a number, a boolean (a flag), null (not given), or a list (a repeatable option, or an option the command line takes comma-separated). An option the
// command does not take is refused, naming it; a required option missing is refused as the command line refuses it. Shared by the Python binding (and the WebAssembly one).
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/handle.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/face.h"

namespace nxr::bindings {

using nxr::io::Json;
namespace task = nxr::io::task;
namespace face = nxr::pipelines::face;

/** The arrays of one call, in memory, by the option they are the value of: `values`, `members`, ... (a path to a .npy file goes in the options instead); for `add measurements`,
 *  `measure.NAME` for the measure NAME. A face hands its arrays over here (no temporary file); the name each carries is what the producer records. */
using Arrays = std::map<std::string, task::ArrayInput>;

/** The options of one call, read once each: what is left unread when the command is called is an option the command does not take. */
class CommandOptions {
 public:
  CommandOptions(std::string prog, const Json& options, const Arrays& arrays = {}) : prog_(std::move(prog)), j_(options.is_null() ? Json::object() : options), arrays_(arrays) {
    if (!j_.is_object()) throw task::Refused(prog_ + ": the options are a set of names and values");
  }
  /** an option that takes an array: in memory (`arrays`), or the path of a .npy file (the options); not both */
  std::optional<task::ArrayInput> array(const std::string& k) {
    used_.insert(k);
    const auto it = arrays_.find(k);
    if (it != arrays_.end()) {
      if (has(k)) throw task::Refused(prog_ + ": " + k + " is given twice, as a file and as an array");
      return it->second;
    }
    if (!has(k)) return std::nullopt;
    return task::ArrayInput(text(k, j_[k]));
  }
  task::ArrayInput array_required(const std::string& k) {
    auto a = array(k);
    if (!a) throw task::Refused(prog_ + ": the following arguments are required: " + k);
    return *a;
  }
  /** the measures given as arrays (`add measurements`): NAME and its values, in name order */
  std::vector<std::pair<std::string, task::ArrayInput>> measure_arrays() {
    std::vector<std::pair<std::string, task::ArrayInput>> out;
    for (const auto& [k, a] : arrays_)
      if (k.rfind("measure.", 0) == 0) {
        used_.insert(k);
        out.emplace_back(k.substr(8), a);
      }
    return out;
  }
  bool has(const std::string& k) const { return j_.contains(k) && !j_[k].is_null(); }
  std::string str(const std::string& k) {
    if (!has(k)) throw task::Refused(prog_ + ": the following arguments are required: " + k);
    return text(k, j_[k]);
  }
  std::optional<std::string> maybe(const std::string& k) { return has(k) ? std::optional<std::string>(text(k, j_[k])) : std::nullopt; }
  bool flag(const std::string& k) {
    used_.insert(k);
    if (!has(k)) return false;
    if (!j_[k].is_boolean()) throw task::Refused(prog_ + ": " + k + " is a flag (True or False)");
    return j_[k].get<bool>();
  }
  std::optional<double> number(const std::string& k) {
    used_.insert(k);
    if (!has(k)) return std::nullopt;
    if (!j_[k].is_number() || j_[k].is_boolean()) throw task::Refused(prog_ + ": " + k + " is a number (got " + j_[k].dump() + ")");
    return j_[k].get<double>();
  }
  std::optional<std::int64_t> integer(const std::string& k) {
    const auto v = number(k);
    if (!v) return std::nullopt;
    if (std::floor(*v) != *v) throw task::Refused(prog_ + ": " + k + " is a whole number (got " + j_[k].dump() + ")");
    return static_cast<std::int64_t>(*v);
  }
  /** a repeatable option: a list of values, or one value */
  std::vector<std::string> list(const std::string& k) {
    used_.insert(k);
    std::vector<std::string> out;
    if (!has(k)) return out;
    if (j_[k].is_array()) for (const auto& v : j_[k]) out.push_back(text(k, v));
    else out.push_back(text(k, j_[k]));
    return out;
  }
  std::optional<std::vector<std::string>> maybe_list(const std::string& k) {
    if (!has(k)) return std::nullopt;
    return list(k);
  }
  /** an option the command line takes comma-separated (`--subjects A,B`): a list, or the comma-separated text */
  std::vector<std::string> commas(const std::string& k) {
    used_.insert(k);
    std::vector<std::string> out;
    if (!has(k)) return out;
    if (j_[k].is_array()) {
      for (const auto& v : j_[k]) out.push_back(text(k, v));
      return out;
    }
    std::stringstream ss(text(k, j_[k]));
    for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) out.push_back(x);
    return out;
  }
  std::optional<std::vector<std::string>> maybe_commas(const std::string& k) {
    if (!has(k)) return std::nullopt;
    return commas(k);
  }
  /** a JSON value as given (an object or a list) */
  Json value(const std::string& k) {
    used_.insert(k);
    return has(k) ? j_[k] : Json();
  }
  /** a write in pieces (commands.h `PieceOptions`): `begin`, `append`, `commit` flags; `write` the id; `shape` a list of whole numbers (or comma-separated text); `dtype`; `region` START:STOP for each axis, as a list of
   *  [start, stop] pairs, a list of "START:STOP" texts, or the comma-separated text */
  task::PieceOptions piece() {
    task::PieceOptions p;
    p.begin = flag("begin");
    p.append = flag("append");
    p.commit = flag("commit");
    p.write = maybe("write");
    p.dtype = maybe("dtype");
    if (has("shape")) {
      std::vector<std::int64_t> shape;
      const Json& j = j_["shape"];
      if (j.is_array()) { for (const auto& v : j) shape.push_back(whole("shape", v)); }
      else if (j.is_number_integer()) shape.push_back(j.get<std::int64_t>());
      else for (const auto& t : commas("shape")) shape.push_back(whole("shape", t));
      p.shape = shape;
    }
    used_.insert("shape");
    if (has("region")) {
      std::vector<std::array<std::int64_t, 2>> region;
      const Json& j = j_["region"];
      auto pair_of = [&](const std::string& t) {
        const auto colon = t.find(':');
        if (colon == std::string::npos) throw task::Refused(prog_ + ": region is START:STOP for each axis (got '" + t + "')");
        return std::array<std::int64_t, 2>{whole("region", t.substr(0, colon)), whole("region", t.substr(colon + 1))};
      };
      if (j.is_array()) {
        for (const auto& v : j) {
          if (v.is_array() && v.size() == 2) region.push_back({whole("region", v[0]), whole("region", v[1])});
          else if (v.is_string()) region.push_back(pair_of(v.get<std::string>()));
          else throw task::Refused(prog_ + ": region is a list of [start, stop] for each axis");
        }
      } else {
        for (const auto& t : commas("region")) region.push_back(pair_of(t));
      }
      p.region = region;
    }
    used_.insert("region");
    return p;
  }
  /** every option given was read: one left is an option the command does not take (an option given as null is an option not given) */
  void done() const {
    for (auto it = j_.begin(); it != j_.end(); ++it)
      if (!it.value().is_null() && !used_.count(it.key())) throw task::Refused(prog_ + ": unrecognized argument: " + it.key());
    for (const auto& [k, a] : arrays_)
      if (!used_.count(k)) throw task::Refused(prog_ + ": unrecognized argument: " + k);
  }
  const std::string& prog() const { return prog_; }

 private:
  std::int64_t whole(const std::string& k, const Json& v) {
    if (v.is_number_integer()) return v.get<std::int64_t>();
    throw task::Refused(prog_ + ": " + k + " holds whole numbers (got " + v.dump() + ")");
  }
  std::int64_t whole(const std::string& k, const std::string& t) {
    try {
      std::size_t used = 0;
      const long long x = std::stoll(t, &used);
      if (used == t.size()) return x;
    } catch (const std::exception&) {}
    throw task::Refused(prog_ + ": " + k + " holds whole numbers (got '" + t + "')");
  }
  std::string text(const std::string& k, const Json& v) {
    used_.insert(k);
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number_integer()) return std::to_string(v.get<std::int64_t>());
    if (v.is_number()) {
      std::ostringstream s;
      s.precision(17);
      s << v.get<double>();
      return s.str();
    }
    throw task::Refused(prog_ + ": " + k + " is text (got " + v.dump() + ")");
  }
  std::string prog_;
  Json j_;
  Arrays arrays_;
  std::set<std::string> used_;
};

/** The command `verb noun` with its options: the library's result and arrays. `created_by` is the producer stamp (the face's: "nxr-neuro <version> / C++17"). */
inline task::CommandResult command(const std::string& verb, const std::string& noun, const Json& options, const task::Progress& progress, const std::string& created_by,
                                   const Arrays& arrays = {}, task::Handle* handle = nullptr) {
  const task::HandleScope through(handle);   // a handle: the call works through its open readers and borrows its writer lock (nxr/io/task/handle.h)
  CommandOptions o("nxr-neuro " + verb + (noun.empty() ? "" : " " + noun), options, arrays);
  const std::string c = verb + " " + noun;
  task::CommandResult r;
  auto call = [&](auto&& f) {
    o.done();
    r = f();
  };
  auto range = [&]() {
    task::ExportRange x;
    x.from = o.number("from");
    x.to = o.number("to");
    x.all = o.flag("all");
    return x;
  };
  auto provenance = [&]() {   // what an export records of the call that made it: the version, the call as a command line, its options
    task::ExportProvenance p;
    p.version = NXR_NEURO_VERSION;
    p.command = o.prog();
    for (auto it = options.begin(); it != options.end(); ++it) {
      if (it.value().is_null() || (it.value().is_boolean() && !it.value().get<bool>())) continue;
      std::string name = it.key();
      for (char& ch : name) if (ch == '_') ch = '-';
      p.command += " --" + name + (it.value().is_boolean() ? "" : " " + (it.value().is_string() ? it.value().get<std::string>() : it.value().dump()));
      p.options[name] = it.value();
    }
    return p;
  };
  auto origin = [&]() {
    task::FileOrigin x;
    x.source_root = o.maybe("source_root");
    x.licence = o.maybe("licence");
    x.attribution = o.maybe("attribution");
    x.anonymize = o.flag("anonymize");
    return x;
  };
  auto node = [&]() {   // `read field | statistic | operator | selection | manifold`
    face::ReadNodeRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    q.out = o.maybe("out");
    q.templates = o.list("templates");
    return q;
  };

  // ---- datasets ----
  if (c == "create dataset") {
    task::CreateDatasetRequest q;
    q.dataset = o.maybe("dataset");
    q.datastore = o.maybe("datastore");
    q.name = o.maybe("name");
    q.source_tool = o.maybe("source_tool");
    q.source_protocol = o.maybe("source_protocol");
    q.source_path = o.maybe("source_path");
    q.templates = o.maybe("templates");
    q.default_ = o.maybe("default");
    q.atlas_finest = o.maybe("atlas_finest");
    q.lc_scales = o.maybe_commas("lc_scales");
    if (const auto m = o.integer("modes")) q.modes = static_cast<int>(*m);
    q.store_format = o.maybe("store_format");
    q.created_by = created_by;
    call([&] { return face::create_dataset(q, progress); });
  } else if (c == "list datasets") {
    task::ListDatasetsRequest q{o.str("datastore")};
    call([&] { return task::list_datasets(q); });
  } else if (c == "show dataset") {
    task::ShowDatasetRequest q{o.str("dataset")};
    call([&] { return task::show_dataset(q, progress); });
  } else if (c == "delete dataset") {
    task::DeleteDatasetRequest q{o.str("datastore"), o.str("name")};
    call([&] { return task::delete_dataset(q); });
  } else if (c == "recover dataset") {
    task::RecoverDatasetRequest q{o.str("dataset")};
    call([&] { return task::recover_dataset(q); });
  } else if (c == "compact dataset") {
    task::CompactDatasetRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.maybe("subject");
    q.dry_run = o.flag("dry_run");
    q.no_shard = o.flag("no_shard");
    q.no_small_files = o.flag("no_small_files");
    call([&] { return task::compact_dataset(q); });
  } else if (c == "split dataset") {
    task::SplitDatasetRequest q{o.str("dataset"), o.str("registered_by"), o.flag("dry_run")};
    call([&] { return task::split_dataset(q, progress); });
  } else if (c == "join dataset") {
    task::JoinDatasetRequest q{o.str("dataset")};
    call([&] { return task::join_dataset(q); });
  } else if (c == "migrate measurements") {
    task::MigrateMeasurementsRequest q{o.str("dataset"), o.flag("dry_run"), o.flag("remove_statistics")};
    call([&] { return task::migrate_measurements(q, progress); });

    // ---- subjects ----
  } else if (c == "create subject") {
    task::CreateSubjectRequest q;
    q.dataset = o.str("dataset");
    q.name = o.str("name");
    if (const auto k = o.maybe("kind")) q.kind = *k;
    q.created_by = created_by;
    call([&] { return task::create_subject(q, progress); });
  } else if (c == "list subjects") {
    task::ListSubjectsRequest q;
    q.dataset = o.str("dataset");
    q.subjects = o.maybe_commas("subjects");
    q.where = o.maybe("where");
    call([&] { return task::list_subjects(q, progress); });
  } else if (c == "show subject") {
    task::ShowSubjectRequest q{o.str("dataset"), o.str("subject"), o.list("templates")};
    call([&] { return task::show_subject(q, progress); });
  } else if (c == "read subject") {
    task::ReadSubjectRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.classes = o.maybe_commas("classes");
    q.related = o.flag("related");
    q.templates = o.list("templates");
    call([&] { return face::read_subject(q, progress); });
  } else if (c == "list manifolds" || c == "list fields" || c == "list selections" || c == "list operators" || c == "list statistics" || c == "list dictionaries" || c == "list spaces" ||
             c == "list acquisitions" || c == "list jobs") {
    task::ListClassRequest q;
    q.cls = noun;
    q.dataset = o.str("dataset");
    q.subject = o.maybe("subject");
    q.name = o.maybe("name");
    q.kind = o.maybe("kind");
    q.on = o.maybe("on");
    q.after = o.maybe("after");
    q.limit = o.integer("limit");
    q.columns = o.maybe_commas("columns");
    q.templates = o.list("templates");
    call([&] { return face::list_class(q, progress); });
  } else if (c == "delete subject") {
    task::DeleteSubjectRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.node = o.maybe("node");
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return task::delete_subject(q); });
  } else if (c == "add surface") {
    task::AddSurfaceRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.file = o.maybe("file").value_or("");
    q.vertices = o.array("vertices");
    q.faces = o.array("faces");
    q.name = o.maybe("name");
    q.scale = o.number("scale").value_or(1.0);
    q.primary = o.flag("primary");
    q.no_primary = o.flag("no_primary");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_surface(q, progress); });

    // ---- add <class> ----
  } else if (c == "add field") {
    task::AddFieldRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.maybe("subject").value_or("");
    q.name = o.maybe("name").value_or("");
    q.manifold = o.maybe("manifold").value_or("");
    if (const auto v = o.array("values")) q.values = *v;
    q.piece = o.piece();
    q.time = o.maybe("time");
    if (const auto v = o.maybe("value_type")) q.value_type = *v;
    if (const auto v = o.maybe("kind")) q.kind = *v;
    q.unit = o.maybe("unit");
    q.cell = o.maybe("cell");
    q.derived_from = o.maybe("derived_from");
    q.description = o.maybe("description");
    q.by_operator = o.maybe("by_operator");
    q.contribution = o.list("contribution");
    q.producer = o.maybe("producer");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_field(q, progress); });
  } else if (c == "add selection") {
    task::AddSelectionRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    q.manifold = o.str("manifold");
    q.type = o.str("type");
    q.members = o.array("members");
    q.indices = o.array("indices");
    q.spans = o.array("spans");
    q.entries = o.commas("entries");
    q.description = o.maybe("description");
    q.of_field = o.maybe("of_field");
    q.weights = o.array("weights");
    q.producer = o.maybe("producer");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_selection(q, progress); });
  } else if (c == "add statistic") {
    task::AddStatisticRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.maybe("subject").value_or("");
    q.name = o.maybe("name").value_or("");
    q.measure = o.maybe("measure").value_or("");
    q.method = o.maybe("method").value_or("");
    if (const auto v = o.array("values")) q.values = *v;
    q.piece = o.piece();
    q.axes = o.list("axes");
    if (const auto v = o.maybe("value_type")) q.value_type = *v;
    q.of_field = o.maybe("of_field");
    q.partition = o.maybe("partition");
    q.unit = o.maybe("unit");
    q.description = o.maybe("description");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_statistic(q, progress); });
  } else if (c == "add measurements") {
    task::AddMeasurementsRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.selection = o.str("selection");
    q.field = o.maybe("field");
    q.unit = o.maybe("unit");
    q.measure_arrays = o.measure_arrays();
    if (!o.has("measure") && q.measure_arrays.empty()) throw task::Refused(o.prog() + ": the following arguments are required: measure");
    q.measure = o.list("measure");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_measurements(q, progress); });
  } else if (c == "add operator") {
    task::AddOperatorRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    q.from_manifold = o.str("from_manifold");
    q.to_manifold = o.str("to_manifold");
    if (const auto v = o.maybe("kind")) q.kind = *v;
    q.values = o.array("values");
    q.data = o.array("data");
    q.indices = o.array("indices");
    q.indptr = o.array("indptr");
    q.description = o.maybe("description");
    q.layout = o.maybe("layout");
    q.method = o.maybe("method");
    q.params = o.maybe("params");
    q.from_selection = o.maybe("from_selection");
    q.producer = o.maybe("producer");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_operator(q, progress); });
  } else if (c == "add points") {
    task::AddPointsRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    q.positions = o.array_required("positions");
    q.coordinate_system = o.maybe("coordinate_system");
    q.description = o.maybe("description");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_points(q, progress); });
  } else if (c == "add line") {
    task::AddLineRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    q.n = o.integer("n");
    q.spacing = o.number("spacing");
    q.origin = o.number("origin");
    q.unit = o.str("unit");
    if (const auto v = o.maybe("dimension")) q.dimension = *v;
    q.coordinates = o.array("coordinates");
    q.kinds = o.array("kinds");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_line(q, progress); });
  } else if (c == "add product") {
    task::AddProductRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    q.factors = o.commas("factors");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_product(q, progress); });
  } else if (c == "add dictionary") {
    task::AddDictionaryRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.maybe("subject");
    q.name = o.str("name");
    q.scope = o.str("scope");
    q.entries = o.str("entries");
    q.description = o.maybe("description");
    q.templates = o.list("templates");
    q.by = o.maybe("by");
    q.created_by = created_by;
    call([&] { return task::add_dictionary(q, progress); });
  } else if (c == "edit manifold") {
    task::EditManifoldRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    q.primary = o.maybe("primary");
    q.label = o.maybe("label");
    q.description = o.maybe("description");
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return task::edit_manifold(q, progress); });
  } else if (c == "edit dictionary") {
    task::EditDictionaryRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.maybe("subject");
    q.name = o.str("name");
    q.entry = o.integer("entry");
    q.name_of = o.maybe("name_of");
    q.label = o.maybe("label");
    q.colour = o.maybe("colour");
    q.description = o.maybe("description");
    q.params = o.maybe("params");
    q.entries = o.maybe("entries");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return task::edit_dictionary(q, progress); });

    // ---- subject packages and the catalogue ----
  } else if (c == "create package") {
    task::CreatePackageRequest q{o.str("package"), o.str("name"), o.str("home_dataset"), o.maybe("kind").value_or("subject"), created_by};
    call([&] { return task::create_package(q); });
  } else if (c == "verify package") {
    task::VerifyPackageRequest q{o.str("package")};
    call([&] { return task::verify_package(q); });
  } else if (c == "show package") {
    task::ShowPackageRequest q{o.str("package")};
    call([&] { return task::show_package(q); });
  } else if (c == "register package") {
    task::RegisterPackageRequest q;
    q.dataset = o.str("dataset");
    q.registered_by = o.str("registered_by");
    q.package = o.list("package");
    q.scan = o.flag("scan");
    q.replace = o.flag("replace");
    q.templates = o.maybe_list("templates");
    call([&] { return task::register_package(q); });
  } else if (c == "list members") {
    task::ListMembersRequest q{o.str("dataset")};
    call([&] { return task::list_members(q); });
  } else if (c == "create index") {
    task::CreateIndexRequest q{o.str("dataset"), o.flag("force")};
    call([&] { return task::create_index(q); });
  } else if (c == "release template") {
    task::ReleaseTemplateRequest q{o.str("package"), o.str("name"), o.str("template_version"), o.str("released_by"), o.str("into")};
    call([&] { return task::release_template(q); });
  } else if (c == "release redistributable") {
    task::ReleaseRedistributableRequest q{o.str("package"), o.str("into"), o.str("released_by")};
    call([&] { return task::release_redistributable(q); });
  } else if (c == "list templates") {
    task::ListTemplatesRequest q{o.list("templates")};
    call([&] { return task::list_templates(q); });
  } else if (c == "create template") {
    task::CreateTemplateRequest q;
    q.recipe = o.str("recipe");
    q.sources = o.str("sources");
    q.package = o.str("package");
    q.name = o.maybe("name");
    q.home_dataset = o.maybe("home_dataset");
    q.hcp_trk = o.maybe("hcp_trk");
    q.hcp_end_surface = o.maybe("hcp_end_surface");
    q.max_end_distance = o.number("max_end_distance");
    if (const auto n = o.integer("hcp_points")) q.hcp_points = static_cast<int>(*n);
    q.cortex_mask = o.flag("cortex_mask");
    q.scalp = o.flag("scalp");
    q.accept_unlisted_sources = o.flag("accept_unlisted_sources");
    q.surface = o.maybe_list("surface");
    q.parcellation = o.maybe_list("parcellation");
    q.created_by = created_by;
    call([&] { return task::create_template(q); });
  } else if (c == "create registration") {
    face::CreateRegistrationRequest q;
    q.package = o.str("package");
    q.name = o.str("name");
    q.from = o.str("from");
    q.to = o.str("to");
    q.from_sphere = o.str("from_sphere");
    q.to_sphere = o.str("to_sphere");
    q.method = o.maybe("method");
    if (const auto k = o.integer("neighbours")) q.neighbours = static_cast<int>(*k);
    q.one_direction = o.flag("one_direction");
    q.templates = o.list("templates");
    q.home_dataset = o.maybe("home_dataset");
    q.source = o.maybe("source");
    q.licence = o.maybe("licence");
    q.attribution = o.maybe("attribution");
    q.redistributable = o.maybe("redistributable");
    q.created_by = created_by;
    call([&] { return face::create_registration(q, progress); });
  } else if (c == "create recording") {
    task::CreateRecordingRequest q;
    q.package = o.str("package");
    q.synthetic = o.flag("synthetic");
    q.name = o.maybe("name");
    q.acquisition = o.maybe("acquisition");
    q.channels = o.integer("channels");
    q.sfreq = o.number("sfreq");
    q.duration = o.number("duration");
    q.amplitude = o.number("amplitude");
    q.noise = o.number("noise");
    if (const auto s = o.integer("seed")) q.seed = static_cast<std::uint64_t>(*s);
    if (o.has("frequency")) {
      q.frequency.emplace();
      const Json f = o.value("frequency");
      for (const Json& x : f.is_array() ? f : Json::array({f})) {
        if (!x.is_number()) throw task::Refused(o.prog() + ": frequency is a list of numbers (Hz)");
        q.frequency->push_back(x.get<double>());
      }
    } else {
      o.value("frequency");
    }
    call([&] { return task::create_recording(q, progress); });

    // ---- the lists that need the pipelines ----
  } else if (c == "list recordings" || c == "list kernels" || c == "list results") {
    face::ListNodesRequest q{o.str("dataset"), o.str("subject"), o.list("templates")};
    call([&] {
      if (noun == "recordings") return face::list_recordings(q, progress);
      if (noun == "kernels") return face::list_kernels(q, progress);
      return face::list_results(q, progress);
    });

    // ---- reading and querying ----
  } else if (c == "read dataset") {
    task::ReadDatasetRequest q;
    q.folder = o.str("folder");
    if (o.has("array")) {
      const auto a = o.list("array");
      if (a.size() != 2) throw task::Refused(o.prog() + ": array is the subject and the array's path inside it (two values)");
      q.array = std::array<std::string, 2>{a[0], a[1]};
    } else {
      o.list("array");
    }
    if (o.has("region")) {
      const Json reg = o.value("region");
      if (reg.is_string()) {   // the command line's form a:b,c:d
        std::stringstream ss(reg.get<std::string>());
        for (std::string part; std::getline(ss, part, ',');) {
          const auto colon = part.find(':');
          if (colon == std::string::npos) throw task::Refused(o.prog() + ": region is LO:HI per axis, comma-separated (or a list of [lo, hi])");
          q.region.push_back({std::stoll(part.substr(0, colon)), std::stoll(part.substr(colon + 1))});
        }
      } else {
        for (const Json& ax : reg) {
          if (!ax.is_array() || ax.size() != 2) throw task::Refused(o.prog() + ": region is a list of [lo, hi] per axis");
          q.region.push_back({ax[0].get<std::int64_t>(), ax[1].get<std::int64_t>()});
        }
      }
    } else {
      o.value("region");
    }
    q.out = o.maybe("out");
    call([&] { return task::read_dataset(q); });
  } else if (c == "read field") {
    face::ReadNodeRequest q = node();
    q.from = o.number("from");
    q.to = o.number("to");
    q.all = o.flag("all");
    for (const auto& r : o.commas("rows")) q.rows.push_back(static_cast<std::int32_t>(std::stol(r)));
    if (const auto p = o.maybe("projectors")) q.projectors = *p;
    q.shape_only = o.flag("shape_only");
    call([&] { return face::read_field(q); });
  } else if (c == "read statistic") {
    face::ReadNodeRequest q = node();
    call([&] { return face::read_statistic(q); });
  } else if (c == "read operator") {
    face::ReadNodeRequest q = node();
    call([&] { return face::read_operator(q); });
  } else if (c == "read selection") {
    face::ReadNodeRequest q = node();
    q.measure = o.maybe("measure");
    q.field = o.maybe("field");
    call([&] { return face::read_selection(q); });
  } else if (c == "read manifold") {
    face::ReadNodeRequest q = node();
    q.family = o.maybe("family");
    call([&] { return face::read_manifold(q); });
  } else if (c == "read crossspectrum") {
    face::ReadCrossspectrumRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.name = o.str("name");
    if (o.has("pair")) {
      const auto ab = o.commas("pair");
      if (ab.size() != 2) throw task::Refused(o.prog() + ": pair is two channel indices A, B");
      q.pair = std::array<std::int32_t, 2>{static_cast<std::int32_t>(std::stol(ab[0])), static_cast<std::int32_t>(std::stol(ab[1]))};
    } else {
      o.value("pair");
    }
    if (const auto s = o.integer("seed")) q.seed = static_cast<std::int32_t>(*s);
    if (const auto v = o.maybe("reading")) q.reading = *v;
    if (const auto v = o.maybe("weights")) q.weights = *v;
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return face::read_crossspectrum(q, progress); });
  } else if (c == "query measurements") {
    face::QueryMeasurementsRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.selection = o.str("selection");
    q.measure = o.str("measure");
    q.field = o.maybe("field");
    q.band = o.maybe("band");
    q.spatial_scale = o.maybe("spatial_scale");
    q.tile = o.maybe("tile");
    q.factor = o.maybe("factor");
    q.where = o.maybe("where");
    q.subjects = o.maybe_commas("subjects");
    q.min_value = o.number("min_value");
    q.max_value = o.number("max_value");
    q.limit = o.integer("limit");
    q.order = o.maybe("order");
    o.list("templates");   // accepted as every subject command accepts it; the query opens the subject as `run` does
    call([&] { return face::query_measurements(q); });
  } else if (c == "compile ") {
    face::CompileRequest q;
    q.definition = o.str("definition");
    q.dataset = o.maybe("dataset");
    q.subject = o.maybe("subject");
    if (q.dataset.has_value() != q.subject.has_value())
      throw task::Refused(o.prog() + ": dataset and subject go together (the inputs typed from the subject's records), or neither (inputs gives their types)");
    for (const auto& s : o.list("set")) {   // NAME=VALUE (VALUE read as YAML), or a preset's NAME
      const auto eq = s.find('=');
      if (eq == 0) throw task::Refused(o.prog() + ": set " + s + ": NAME=VALUE, or a preset's NAME");
      if (eq == std::string::npos) {
        if (!q.preset.empty()) throw task::Refused(o.prog() + ": set " + s + ": one preset (already " + q.preset + ")");
        q.preset = s;
        continue;
      }
      q.parameters[s.substr(0, eq)] = nxr::io::parse_yaml(s.substr(eq + 1), "set " + s.substr(0, eq));
    }
    q.signatures = o.value("signatures");
    if (const Json in = o.value("inputs"); !in.is_null()) q.inputs = in;
    call([&] { return face::compile(q); });

    // ---- import ----
  } else if (c == "import list") {
    task::ImportListRequest q{o.str("protocol")};
    call([&] { return task::import_list(q); });
  } else if (c == "import scan") {
    task::ImportScanRequest q{o.str("protocol"), o.maybe("prefer"), o.maybe("out"), false};
    q.summary = o.flag("summary") || q.out.has_value();
    call([&] { return task::import_scan(q); });
  } else if (c == "import subject") {
    task::ImportSubjectRequest q{o.str("protocol"), o.str("subject"), o.str("dataset"), o.number("resample"), created_by};
    o.list("templates");
    call([&] { return task::import_subject(q, progress); });
  } else if (c == "import template") {
    task::ImportTemplateRequest q{o.str("protocol"), o.str("dataset"), o.maybe_list("surface"), created_by};
    call([&] { return task::import_template(q, progress); });
  } else if (c == "import group") {
    task::ImportGroupRequest q{o.str("protocol"), o.str("dataset"), created_by};
    call([&] { return task::import_group(q, progress); });
  } else if (c == "import templates") {
    task::ImportTemplatesRequest q{o.str("templates_dir"), o.str("dataset"), o.maybe("subject"), o.maybe("trk"), o.maybe("surface"), created_by};
    call([&] { return task::import_templates(q); });
  } else if (c == "import condition") {
    task::ImportConditionRequest q;
    q.protocol = o.str("protocol");
    q.subject = o.str("subject");
    q.condition = o.str("condition");
    q.dataset = o.str("dataset");
    q.surface = o.maybe("surface");
    q.recording = o.maybe("recording");
    q.channel_file = o.maybe("channel_file");
    q.resample = o.number("resample");
    q.no_gain = o.flag("no_gain");
    q.all_surfaces = o.flag("all_surfaces");
    q.include_imported = o.flag("include_imported");
    q.no_raw = o.flag("no_raw");
    q.extra_surface = o.maybe_list("extra_surface");
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return task::import_condition(q, progress); });
  } else if (c == "import surface") {
    if (!o.has("file")) throw task::Refused(o.prog() + ": give file (tess_*.mat files)");
    task::ImportSurfaceRequest q{o.str("dataset"), o.str("subject"), o.list("file"), o.maybe("bst_root"), o.list("templates"), created_by};
    call([&] { return task::import_surface(q, progress); });
  } else if (c == "import mri") {
    task::ImportMriRequest q{o.str("dataset"), o.str("subject"), o.str("anat"), o.maybe_list("only"), o.list("templates"), created_by};
    call([&] { return task::import_mri(q, progress); });
  } else if (c == "import inverse") {
    task::ImportInverseRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.file = o.str("file");
    q.acquisition = o.str("acquisition");
    q.bst_root = o.str("bst_root");
    q.channels_id = o.maybe("channels_id");
    q.surface_id = o.maybe("surface_id");
    q.forward_id = o.maybe("forward_id");
    q.name = o.maybe("name");
    q.channel_file = o.maybe("channel_file");
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return task::import_inverse(q, progress); });
  } else if (c == "import map") {
    task::ImportMapRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.bst_root = o.str("bst_root");
    if (!o.has("file")) throw task::Refused(o.prog() + ": the following arguments are required: file");
    q.file = o.list("file");
    q.acquisition = o.maybe("acquisition");
    q.provenance = o.maybe("provenance");
    q.replace = o.flag("replace");
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return task::import_map(q, progress); });
  } else if (c == "import resample") {
    task::ImportResampleRequest q;
    q.dataset = o.str("dataset");
    q.subject = o.str("subject");
    q.recording = o.str("recording");
    if (!o.has("to")) throw task::Refused(o.prog() + ": the following arguments are required: to");
    q.to = *o.number("to");
    q.name = o.maybe("name");
    q.templates = o.list("templates");
    q.created_by = created_by;
    call([&] { return task::import_resample(q, progress); });
  } else if (c == "import badsegments" || c == "import events") {
    task::ImportBadsegmentsRequest q{o.str("dataset"), o.str("subject"), o.str("from"), o.flag("replace"), o.list("templates"), created_by};
    call([&] { return noun == "badsegments" ? task::import_badsegments(q, progress) : task::import_events(q, progress); });
  } else if (c == "import recording") {
    task::ImportRecordingRequest q;
    q.from = o.str("from");
    q.package = o.str("package");
    q.name = o.maybe("name");
    q.acquisition = o.maybe("acquisition");
    q.format = o.maybe("format");
    q.montage = o.maybe("montage");
    q.origin = origin();
    q.in_place = o.flag("in_place");
    call([&] { return task::import_recording(q, progress); });
  } else if (c == "import fif") {
    task::ImportFifRequest q;
    q.from = o.str("from");
    q.package = o.str("package");
    q.kind = o.maybe("kind");
    q.name = o.maybe("name");
    q.acquisition = o.maybe("acquisition");
    q.channels = o.maybe("channels");
    q.recording = o.maybe("recording");
    q.surface = o.maybe("surface");
    q.origin = origin();
    call([&] { return task::import_fif(q, progress); });
  } else if (c == "import transform") {
    task::ImportTransformRequest q{o.str("from"), o.str("package"), o.maybe("name"), o.maybe("channels")};
    call([&] { return task::import_transform(q, progress); });
  } else if (c == "import nifti") {
    task::ImportNiftiRequest q;
    q.from = o.str("from");
    q.package = o.str("package");
    q.name = o.maybe("name");
    q.kind = o.maybe("kind");
    q.parcellation = o.maybe("parcellation");
    q.label_names = o.maybe("label_names");
    q.field_kind = o.maybe("field_kind");
    q.unit = o.maybe("unit");
    q.origin = origin();
    call([&] { return task::import_nifti(q, progress); });
  } else if (c == "import freesurfer") {
    task::ImportFreesurferRequest q;
    q.from = o.str("from");
    q.package = o.str("package");
    q.name = o.maybe("name");
    q.orig = o.maybe("orig");
    q.field_kind = o.maybe("field_kind");
    q.surfaces = o.maybe_list("surfaces");
    q.annotations = o.maybe_list("annotations");
    q.overlays = o.maybe_list("overlays");
    q.origin = origin();
    call([&] { return task::import_freesurfer(q, progress); });
  } else if (c == "import gifti") {
    task::ImportGiftiRequest q;
    q.from = o.str("from");
    q.package = o.str("package");
    q.name = o.maybe("name");
    q.kind = o.maybe("kind");
    q.surface = o.maybe("surface");
    q.parcellation = o.maybe("parcellation");
    q.field_kind = o.maybe("field_kind");
    q.unit = o.maybe("unit");
    q.origin = origin();
    call([&] { return task::import_gifti(q, progress); });
  } else if (c == "import cifti") {
    task::ImportCiftiRequest q;
    q.from = o.str("from");
    q.package = o.str("package");
    q.name = o.maybe("name");
    q.left = o.maybe("left");
    q.right = o.maybe("right");
    q.parcellation = o.maybe("parcellation");
    q.field_kind = o.maybe("field_kind");
    q.unit = o.maybe("unit");
    q.origin = origin();
    call([&] { return task::import_cifti(q, progress); });
  } else if (c == "import bids") {
    task::ImportBidsRequest q;
    q.from = o.str("from");
    q.dataset = o.str("dataset");
    q.subject = o.list("subject");
    q.in_place = o.flag("in_place");
    q.anonymize = o.flag("anonymize");
    q.no_participants = o.flag("no_participants");
    q.registered_by = o.maybe("registered_by");
    q.created_by = created_by;
    call([&] { return task::import_bids(q, progress); });
  } else if (c == "import participants" || c == "export participants") {
    task::ParticipantsRequest q{o.str("dataset"), o.str("tsv"), o.maybe("sidecar")};
    call([&] { return verb == "import" ? task::import_participants(q) : task::export_participants(q); });

    // ---- export ----
  } else if (c == "export gifti") {
    task::ExportGiftiRequest q;
    q.provenance = provenance();
    q.package = o.str("package");
    q.out = o.str("out");
    q.field = o.maybe("field");
    q.labelling = o.maybe("labelling");
    q.kind = o.maybe("kind");
    q.surface_unit = o.maybe("surface_unit");
    q.with_surface = o.flag("with_surface");
    q.hemispheres = o.flag("hemispheres");
    q.range = range();
    call([&] { return task::export_gifti(q, progress); });
  } else if (c == "export tsv") {
    task::ExportTsvRequest q;
    q.provenance = provenance();
    q.package = o.str("package");
    q.out = o.str("out");
    q.statistic = o.maybe("statistic");
    q.partition = o.maybe("partition");
    q.field = o.maybe("field");
    q.measures = o.maybe_commas("measures");
    call([&] { return task::export_tsv(q, progress); });
  } else if (c == "export fif") {
    task::ExportFifRequest q;
    q.provenance = provenance();
    q.package = o.str("package");
    q.out = o.str("out");
    q.field = o.maybe("field");
    q.covariance = o.maybe("covariance");
    q.forward = o.maybe("forward");
    q.inverse = o.maybe("inverse");
    q.range = range();
    call([&] { return task::export_fif(q, progress); });
  } else if (c == "export bids-derivatives") {
    task::ExportBidsDerivativesRequest q;
    q.provenance = provenance();
    q.package = o.str("package");
    q.out = o.str("out");
    q.results = o.str("results");
    q.space = o.maybe("space");
    q.session = o.maybe("session");
    q.task = o.maybe("task");
    q.run = o.maybe("run");
    q.measurements = o.maybe("measurements");
    q.with_surface = o.flag("with_surface");
    q.stc = o.flag("stc");
    q.range = range();
    call([&] { return task::export_bids_derivatives(q, progress); });
  } else if (c == "export measurements") {
    face::ExportMeasurementsRequest q;
    q.dataset = o.str("dataset");
    q.labelling = o.str("labelling");
    q.fields = o.str("fields");
    q.table = o.str("table");
    q.out = o.str("out");
    q.statistics = o.maybe_commas("statistics");
    q.subjects = o.maybe_commas("subjects");
    q.recipe = o.maybe("recipe");
    q.exclude = o.maybe_commas("exclude");
    q.keep_all = o.flag("keep_all");
    q.layout = o.maybe("layout");
    q.scaling = o.maybe("scaling");
    call([&] { return face::export_measurements(q); });
  } else {
    throw task::Refused("nxr-neuro has no command '" + verb + (noun.empty() ? "" : " " + noun) + "'");
  }
  return r;
}

/** `run PIPELINE` on a subject, or a group: `args` is {pipeline, dataset, subject, parameters, from, to, all, replace, dry_run, preset, subjects, into}; the result as JSON.
 *  A handle: the same as `command`. */
inline Json run(const Json& args, const task::Progress& progress, task::Handle* handle = nullptr) {
  const task::HandleScope through(handle);
  auto text = [&](const char* k) { return args.contains(k) && args[k].is_string() ? args[k].get<std::string>() : std::string(); };
  auto number = [&](const char* k) { return args.contains(k) && args[k].is_number() ? std::optional<double>(args[k].get<double>()) : std::nullopt; };
  auto flag = [&](const char* k) { return args.contains(k) && args[k].is_boolean() && args[k].get<bool>(); };
  const Json parameters = args.contains("parameters") && args["parameters"].is_object() ? args["parameters"] : Json::object();
  std::vector<std::string> subjects;
  if (args.contains("subjects") && args["subjects"].is_array()) for (const auto& x : args["subjects"]) subjects.push_back(x.get<std::string>());
  const std::string dataset = text("dataset"), subject = text("subject"), into = text("into");
  if (!subjects.empty() || !into.empty()) {   // a group run: the members and the group package are the caller's
    face::GroupRunRequest rq;
    rq.pipeline = text("pipeline");
    rq.parameters = parameters;
    rq.subjects = subjects;
    rq.into = into;
    rq.replace = flag("replace");
    rq.preset = text("preset");
    rq.dry_run = flag("dry_run");
    return face::run_group(dataset, rq, progress);
  }
  face::RunRequest rq;
  rq.pipeline = text("pipeline");
  rq.parameters = parameters;
  rq.from = number("from");
  rq.to = number("to");
  rq.all = flag("all");
  rq.replace = flag("replace");
  rq.preset = text("preset");
  rq.dry_run = flag("dry_run");
  return face::run(dataset, subject, rq, progress);
}

}  // namespace nxr::bindings
