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
//   package release --package P --name N --template-version V --released-by WHO --into DIR    package release-redistributable --package P --into DIR --released-by WHO
//   package templates [--templates DIR ...]
#include <sstream>

#include "cli.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/migrate.h"
#include "nxr/io/participants.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/templates.h"

using namespace nxr::io;
using namespace cli;

namespace {
std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) out.push_back(x);
  return out;
}
TemplateSearch search_of(const Args& a) {
  TemplateSearch s;
  if (const auto it = a.multi.find("--templates"); it != a.multi.end()) for (const auto& p : it->second) s.command_paths.push_back(p);
  return s;
}
}  // namespace

int package_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro package: error: the following arguments are required: action");
  const std::string action = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro package " + action;
  if (action == "create") {
    const Args a = parse(prog, rest, {"--package", "--name", "--home-dataset", "--kind"}, {}, {"--package", "--name", "--home-dataset"});
    CreatePackageOptions o;
    o.home_dataset = a.get("--home-dataset");
    o.subject.kind = a.maybe("--kind").value_or("subject");
    o.subject.created_by = stamp();
    o.built_by = stamp();
    DatasetWriter w = create_package(a.get("--package"), a.get("--name"), o);
    emit(Json{{"stage", "result"}, {"package", a.get("--package")}, {"subject_id", read_package_record(w.db()).subject_id}});
    w.close();
    emit(Json{{"stage", "result"}, {"content_hash", SubjectPackage::open(a.get("--package")).record().content_hash}});
  } else if (action == "verify") {
    const Args a = parse(prog, rest, {"--package"}, {}, {"--package"});
    const VerifyReport v = SubjectPackage::open(a.get("--package")).verify();
    Json problems = Json::array();
    for (const auto& p : v.problems) problems.push_back(p);
    emit(Json{{"stage", "result"}, {"ok", v.ok}, {"content_hash", v.content_hash}, {"problems", problems}});
    return v.ok ? 0 : 1;
  } else if (action == "show") {
    const Args a = parse(prog, rest, {"--package"}, {}, {"--package"});
    emit(Json{{"stage", "result"}, {"record", SubjectPackage::open(a.get("--package")).record().to_json()}});
  } else if (action == "register") {
    const Args a = parse(prog, rest, {"--dataset", "--registered-by"}, {"--scan", "--replace"}, {"--dataset", "--registered-by"}, 0, {"--package", "--templates"});
    std::vector<std::filesystem::path> pk;
    if (const auto it = a.multi.find("--package"); it != a.multi.end()) for (const auto& p : it->second) pk.push_back(p);
    if (a.flag.count("--scan")) { if (!pk.empty()) throw Usage(prog + ": error: --scan and --package are alternatives"); pk = scan_packages(a.get("--dataset")); }
    if (pk.empty()) throw Usage(prog + ": error: name the packages (--package P ...) or --scan the dataset folder");
    RegisterOptions ro;
    ro.registered_by = a.get("--registered-by");
    ro.replace = a.flag.count("--replace") > 0;
    if (a.multi.count("--templates")) ro.templates = search_of(a);
    Json out = Json::array();
    bool refused = false;
    for (const auto& r : register_packages(a.get("--dataset"), pk, ro)) {
      out.push_back(Json{{"package", r.package.string()}, {"name", r.name}, {"outcome", r.outcome}, {"reason", r.reason}});
      refused = refused || r.outcome == "refused";
    }
    emit(Json{{"stage", "result"}, {"results", out}});
    return refused ? 1 : 0;
  } else if (action == "status") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    Json out = Json::array();
    for (const auto& e : refresh_member_status(a.get("--dataset"))) out.push_back(e.to_json());
    emit(Json{{"stage", "result"}, {"subjects", out}});
  } else if (action == "select") {
    const Args a = parse(prog, rest, {"--dataset", "--subjects", "--where"}, {}, {"--dataset"});
    SubjectSelection sel;
    if (const auto s = a.maybe("--subjects")) sel.subjects = split_commas(*s);
    sel.where = a.maybe("--where");
    emit(Json{{"stage", "result"}, {"selection", select_subjects(a.get("--dataset"), sel).provenance()}});
  } else if (action == "index") {
    const Args a = parse(prog, rest, {"--dataset"}, {"--force"}, {"--dataset"});
    const IndexReport r = build_index(a.get("--dataset"), a.flag.count("--force") > 0);
    emit(Json{{"stage", "result"}, {"rebuilt", r.rebuilt}, {"subjects", r.subjects}});
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
    SplitOptions o;
    o.registered_by = a.get("--registered-by");
    o.dry_run = a.flag.count("--dry-run") > 0;
    o.on_stage = [](const std::string& stage, const std::string& subject) { emit(Json{{"stage", stage}, {"subject", subject}}); };
    emit(Json{{"stage", "result"}, {"report", split_dataset(a.get("--dataset"), o)}});
  } else if (action == "join") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    emit(Json{{"stage", "result"}, {"report", join_dataset(a.get("--dataset"))}});
  } else if (action == "release") {
    const Args a = parse(prog, rest, {"--package", "--name", "--template-version", "--released-by", "--into"}, {}, {"--package", "--name", "--template-version", "--released-by", "--into"});
    const ReleaseReport r = release_package(a.get("--package"), ReleaseOptions{a.get("--name"), a.get("--template-version"), a.get("--released-by"), a.get("--into")});
    emit(Json{{"stage", "result"}, {"folder", r.folder.string()}, {"content_hash", r.content_hash}});
  } else if (action == "release-redistributable") {
    const Args a = parse(prog, rest, {"--package", "--into", "--released-by"}, {}, {"--package", "--into", "--released-by"});
    const ReleaseReport r = release_redistributable(a.get("--package"), a.get("--into"), a.get("--released-by"));
    emit(Json{{"stage", "result"}, {"folder", r.folder.string()}, {"content_hash", r.content_hash}, {"left_out", r.left_out}});
  } else if (action == "templates") {
    const Args a = parse(prog, rest, {}, {}, {}, 0, {"--templates"});
    Json out = Json::array();
    for (const auto& l : list_templates(search_of(a)))
      out.push_back(Json{{"folder", l.folder.string()}, {"name", l.name}, {"version", l.version}, {"edition", l.edition}, {"kind", l.kind}, {"content_hash", l.content_hash}});
    emit(Json{{"stage", "result"}, {"templates", out}});
  } else {
    throw Usage("nxr-neuro package: error: argument action: invalid choice: '" + action +
                "' (choose from 'create', 'verify', 'show', 'register', 'status', 'select', 'index', 'participants-import', 'participants-export', 'split', 'join', 'release', 'release-redistributable', 'templates')");
  }
  return 0;
}
