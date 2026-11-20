// The legacy parser of the commands over subject packages (`create package`, `register package`, `split dataset`, ...: cli/command_table.cpp maps each new
// command to one action here). The subject packages and the dataset module (store format 53), a thin front on io/include/nxr/io/{subject_package,catalogue,participants,
// dataset_index,migrate,templates}.h: it parses and prints; every choice of data is named on the command line.
//
//   package create  --package P --name S --home-dataset N [--kind subject|template|group]
//   package verify  --package P
//   package show    --package P
//   package register --dataset D (--package P ... | --scan) --registered-by WHO [--replace] [--templates DIR ...]
//   package status  --dataset D
//   package select  --dataset D (--subjects A,B,... | --where FILTER)
//   package index   --dataset D [--force]
//   package participants-import --dataset D --tsv TSV [--sidecar JSON]    package participants-export --dataset D --tsv TSV [--sidecar JSON]
//   package split   --dataset D --registered-by WHO [--dry-run]            package join --dataset D
//   package migrate-selections --dataset D [--dry-run]                     package migrate-measurements --dataset D [--dry-run] [--remove-statistics]
//   package release --package P --name N --template-version V --released-by WHO --into DIR    package release-redistributable --package P --into DIR --released-by WHO
//   package templates [--templates DIR ...]
//   package template-create --recipe fsaverage|icbm152 --sources DIR --package P [--name N] [--surface S ...] [--parcellation NAME ...] [--cortex-mask] [--hcp-trk FILE --hcp-end-surface S
//                            [--max-end-distance MM] [--hcp-points N]] [--accept-unlisted-sources] [--home-dataset N]
//   package registration-create --package P --name N --from NAME@VERSION --to NAME@VERSION --from-sphere S --to-sphere S [--method shepard|nearest|barycentric]
//                               [--neighbours K] [--one-direction] [--templates DIR ...] [--home-dataset N] [--source TEXT --licence TEXT --attribution TEXT --redistributable yes|no]
#include "cli.h"
#include "nxr/io/participants.h"
#include "nxr/io/migrate.h"
#ifdef NXR_NEURO_WITH_PIPELINES
#include "nxr/pipelines/face.h"
#endif

using namespace nxr::io;
using namespace cli;
namespace task = nxr::io::task;

int package_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro package: error: the following arguments are required: action");
  const std::string action = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro package " + action;
  if (action == "create") {
    const Args a = parse(prog, rest, {"--package", "--name", "--home-dataset", "--kind"}, {}, {"--package", "--name", "--home-dataset"});
    const task::CommandResult r = task::create_package({a.get("--package"), a.get("--name"), a.get("--home-dataset"), a.maybe("--kind").value_or("subject"), stamp()});
    emit(Json{{"stage", "result"}, {"package", r.result["package"]}, {"subject_id", r.result["subject_id"]}});   // two result lines, as it always printed
    emit(Json{{"stage", "result"}, {"content_hash", r.result["content_hash"]}});
  } else if (action == "verify") {
    const Args a = parse(prog, rest, {"--package"}, {}, {"--package"});
    const task::CommandResult r = task::verify_package({a.get("--package")});
    emit_result(r.result);
    return r.result["ok"].get<bool>() ? 0 : 1;
  } else if (action == "show") {
    const Args a = parse(prog, rest, {"--package"}, {}, {"--package"});
    emit_result(task::show_package({a.get("--package")}).result);
  } else if (action == "register") {
    const Args a = parse(prog, rest, {"--dataset", "--registered-by"}, {"--scan", "--replace"}, {"--dataset", "--registered-by"}, 0, {"--package", "--templates"});
    task::RegisterPackageRequest rq;
    rq.dataset = a.get("--dataset");
    rq.registered_by = a.get("--registered-by");
    rq.package = values_of(a, "--package");
    rq.scan = a.flag.count("--scan") > 0;
    rq.replace = a.flag.count("--replace") > 0;
    rq.templates = maybe_values(a, "--templates");
    const task::CommandResult r = call(prog, [&] { return task::register_package(rq); });
    emit_result(r.result);
    for (const auto& x : r.result["results"]) if (x["outcome"] == "refused") return 1;
    return 0;
  } else if (action == "status") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    emit_result(task::list_members({a.get("--dataset")}).result);
  } else if (action == "select") {
    const Args a = parse(prog, rest, {"--dataset", "--subjects", "--where"}, {}, {"--dataset"});
    SubjectSelection sel;
    if (const auto s = a.maybe("--subjects")) sel.subjects = split_commas(*s);
    sel.where = a.maybe("--where");
    emit(Json{{"stage", "result"}, {"selection", select_subjects(a.get("--dataset"), sel).provenance()}});
  } else if (action == "index") {
    const Args a = parse(prog, rest, {"--dataset"}, {"--force"}, {"--dataset"});
    emit_result(task::create_index({a.get("--dataset"), a.flag.count("--force") > 0}).result);
  } else if (action == "participants-import") {
    const Args a = parse(prog, rest, {"--dataset", "--tsv", "--sidecar"}, {}, {"--dataset", "--tsv"});
    std::optional<std::filesystem::path> side;
    if (const auto s = a.maybe("--sidecar")) side = *s;
    const auto r = participants_import(a.get("--dataset"), a.get("--tsv"), side);
    emit(Json{{"stage", "result"}, {"variables", r.variables}, {"subjects", r.n_subjects}, {"values", r.n_values}});
  } else if (action == "participants-export") {
    const Args a = parse(prog, rest, {"--dataset", "--tsv", "--sidecar"}, {}, {"--dataset", "--tsv"});   // written only where the command names
    std::optional<std::filesystem::path> side;
    if (const auto s = a.maybe("--sidecar")) side = *s;
    participants_export(a.get("--dataset"), a.get("--tsv"), side);
    emit(Json{{"stage", "result"}, {"tsv", a.get("--tsv")}});
  } else if (action == "split") {
    const Args a = parse(prog, rest, {"--dataset", "--registered-by"}, {"--dry-run"}, {"--dataset", "--registered-by"});
    emit_result(task::split_dataset({a.get("--dataset"), a.get("--registered-by"), a.flag.count("--dry-run") > 0}, progress()).result);
  } else if (action == "migrate-selections") {
    const Args a = parse(prog, rest, {"--dataset"}, {"--dry-run"}, {"--dataset"});
    SelectionStorageOptions o;
    o.dry_run = a.flag.count("--dry-run") > 0;
    o.on_stage = [](const std::string& stage, const std::string& subject) { emit(Json{{"stage", stage}, {"subject", subject}}); };
    emit(Json{{"stage", "result"}, {"report", migrate_selection_storage(a.get("--dataset"), o)}});
  } else if (action == "migrate-measurements") {
    // THE MEASUREMENTS' STORAGE (schema 53, io/migrate.h): one table for a value over a selection's members. Subject by subject, resumable, verified; the old statistics are KEPT unless the explicit second step is asked.
    const Args a = parse(prog, rest, {"--dataset"}, {"--dry-run", "--remove-statistics"}, {"--dataset"});
    emit_result(task::migrate_measurements({a.get("--dataset"), a.flag.count("--dry-run") > 0, a.flag.count("--remove-statistics") > 0}, progress()).result);
  } else if (action == "join") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    emit_result(task::join_dataset({a.get("--dataset")}).result);
  } else if (action == "release") {
    const Args a = parse(prog, rest, {"--package", "--name", "--template-version", "--released-by", "--into"}, {}, {"--package", "--name", "--template-version", "--released-by", "--into"});
    emit_result(task::release_template({a.get("--package"), a.get("--name"), a.get("--template-version"), a.get("--released-by"), a.get("--into")}).result);
  } else if (action == "release-redistributable") {
    const Args a = parse(prog, rest, {"--package", "--into", "--released-by"}, {}, {"--package", "--into", "--released-by"});
    emit_result(task::release_redistributable({a.get("--package"), a.get("--into"), a.get("--released-by")}).result);
  } else if (action == "templates") {
    const Args a = parse(prog, rest, {}, {}, {}, 0, {"--templates"});
    emit_result(task::list_templates({values_of(a, "--templates")}).result);
  } else if (action == "template-create") {
    const Args a = parse(prog, rest, {"--recipe", "--sources", "--package", "--name", "--hcp-trk", "--hcp-end-surface", "--max-end-distance", "--hcp-points", "--home-dataset"},
                         {"--cortex-mask", "--scalp", "--accept-unlisted-sources"}, {"--recipe", "--sources", "--package"}, 0, {"--surface", "--parcellation"});
    task::CreateTemplateRequest rq;
    rq.recipe = a.get("--recipe");
    rq.sources = a.get("--sources");
    rq.package = a.get("--package");
    rq.name = a.maybe("--name");
    rq.home_dataset = a.maybe("--home-dataset");
    rq.created_by = stamp();
    rq.accept_unlisted_sources = a.flag.count("--accept-unlisted-sources") > 0;
    rq.cortex_mask = a.flag.count("--cortex-mask") > 0;
    rq.scalp = a.flag.count("--scalp") > 0;
    rq.surface = maybe_values(a, "--surface");
    rq.parcellation = maybe_values(a, "--parcellation");
    rq.hcp_trk = a.maybe("--hcp-trk");
    rq.hcp_end_surface = a.maybe("--hcp-end-surface");
    try {
      if (const auto m = a.maybe("--max-end-distance")) rq.max_end_distance = std::stod(*m);
      if (const auto n = a.maybe("--hcp-points")) rq.hcp_points = std::stoi(*n);
    } catch (const std::exception&) { throw Usage(prog + ": error: --max-end-distance and --hcp-points take numbers"); }
    Json out = task::create_template(rq).result;   // the library's report, "stage" last as it always printed
    out["stage"] = "result";
    emit(out);
  } else if (action == "registration-create") {
#ifdef NXR_NEURO_WITH_PIPELINES
    const Args a = parse(prog, rest, {"--package", "--name", "--from", "--to", "--from-sphere", "--to-sphere", "--method", "--neighbours", "--home-dataset", "--source", "--licence", "--attribution", "--redistributable"},
                         {"--one-direction"}, {"--package", "--name", "--from", "--to", "--from-sphere", "--to-sphere"}, 0, {"--templates"});
    nxr::pipelines::face::CreateRegistrationRequest rq;
    rq.package = a.get("--package");
    rq.name = a.get("--name");
    rq.from = a.get("--from");
    rq.to = a.get("--to");
    rq.from_sphere = a.get("--from-sphere");
    rq.to_sphere = a.get("--to-sphere");
    rq.method = a.maybe("--method");
    if (const auto k = a.maybe("--neighbours")) {
      try { rq.neighbours = std::stoi(*k); } catch (const std::exception&) { throw Usage(prog + ": error: argument --neighbours: invalid integer value: '" + *k + "'"); }
    }
    rq.one_direction = a.flag.count("--one-direction") > 0;
    rq.templates = values_of(a, "--templates");
    rq.home_dataset = a.maybe("--home-dataset");
    rq.created_by = stamp();
    rq.source = a.maybe("--source");
    rq.licence = a.maybe("--licence");
    rq.attribution = a.maybe("--attribution");
    rq.redistributable = a.maybe("--redistributable");
    const task::CommandResult r = call(prog, [&] { return nxr::pipelines::face::create_registration(rq, progress()); });
    Json out;                                      // the registration's info, then "stage", the package and its content hash, as it always printed
    for (auto it = r.result.begin(); it != r.result.end(); ++it) if (it.key() != "package" && it.key() != "content_hash") out[it.key()] = it.value();
    out["stage"] = "result";
    out["package"] = r.result["package"];
    out["content_hash"] = r.result["content_hash"];
    emit(out);
#else
    throw Error(code::unsupported, "nxr-neuro create registration needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
  } else {
    throw Usage("nxr-neuro package: error: argument action: invalid choice: '" + action +
                "' (choose from 'create', 'verify', 'show', 'register', 'status', 'select', 'index', 'participants-import', 'participants-export', 'split', 'join', 'migrate-selections', 'migrate-measurements', 'release', 'release-redistributable', 'templates', 'registration-create', 'template-create')");
  }
  return 0;
}
