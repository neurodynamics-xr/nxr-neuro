// The legacy parser of the commands over subject packages (`create package`, `register package`, `split dataset`, ...: cli/command_table.cpp maps each new
// command to one action here). The subject packages and the dataset module (store format 53), a thin front on io/include/nxr/io/{subject_package,catalogue,participants,
// dataset_index,migrate,templates}.h: it parses and prints; every choice of data is named on the command line.
//
//   package catalogue-create --dataset D --name N
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
//   package template-create --recipe fsaverage|icbm152 --sources DIR --package P [--name N] [--surface S ...] [--parcellation NAME ...] [--cortex-mask] [--hcp-trk FILE --hcp-end-surface S
//                            [--max-end-distance MM] [--hcp-points N]] [--accept-unlisted-sources] [--home-dataset N]
//   package registration-create --package P --name N --from NAME@VERSION --to NAME@VERSION --from-sphere S --to-sphere S [--method shepard|nearest|barycentric]
//                               [--neighbours K] [--one-direction] [--templates DIR ...] [--home-dataset N] [--source TEXT --licence TEXT --attribution TEXT --redistributable yes|no]
#include <sstream>

#include "cli.h"
#include "nxr/convert/templates.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/migrate.h"
#include "nxr/io/participants.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/templates.h"
#ifdef NXR_NEURO_WITH_PIPELINES
#include "nxr/pipelines/registration.h"
#endif

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
  } else if (action == "catalogue-create") {
    const Args a = parse(prog, rest, {"--dataset", "--name"}, {}, {"--dataset", "--name"});
    CreateCatalogueOptions o;
    o.name = a.get("--name");
    o.created_by = stamp();
    o.built_by = stamp();
    const Catalogue c = create_catalogue(a.get("--dataset"), o);
    emit(Json{{"stage", "result"}, {"dataset", text(c.dataset_row(), "name")}, {"id", text(c.dataset_row(), "id")}, {"folder", c.folder().string()}});
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
  } else if (action == "template-create") {
    const Args a = parse(prog, rest, {"--recipe", "--sources", "--package", "--name", "--hcp-trk", "--hcp-end-surface", "--max-end-distance", "--hcp-points", "--home-dataset"},
                         {"--cortex-mask", "--accept-unlisted-sources"}, {"--recipe", "--sources", "--package"}, 0, {"--surface", "--parcellation"});
    nxr::convert::BuildRequest r;
    r.recipe = a.get("--recipe");
    r.sources = a.get("--sources");
    r.package = a.get("--package");
    if (const auto n = a.maybe("--name")) r.name = *n;
    if (const auto h = a.maybe("--home-dataset")) r.home_dataset = *h;
    r.created_by = stamp();
    r.accept_unlisted_sources = a.flag.count("--accept-unlisted-sources") > 0;
    r.cortex_mask = a.flag.count("--cortex-mask") > 0;
    if (a.multi.count("--surface")) r.surfaces = a.multi.at("--surface");
    if (a.multi.count("--parcellation")) r.annotations = a.multi.at("--parcellation");
    if (const auto t = a.maybe("--hcp-trk")) r.hcp_trk = std::filesystem::path(*t);
    if (const auto s = a.maybe("--hcp-end-surface")) r.hcp_end_surface = *s;
    try {
      if (const auto m = a.maybe("--max-end-distance")) r.hcp_max_end_distance_mm = std::stod(*m);
      if (const auto n = a.maybe("--hcp-points")) r.hcp_points = std::stoi(*n);
    } catch (const std::exception&) { throw Usage(prog + ": error: --max-end-distance and --hcp-points take numbers"); }
    emit([&] { Json j = nxr::convert::build_template(r); j["stage"] = "result"; return j; }());
  } else if (action == "registration-create") {
#ifdef NXR_NEURO_WITH_PIPELINES
    const Args a = parse(prog, rest, {"--package", "--name", "--from", "--to", "--from-sphere", "--to-sphere", "--method", "--neighbours", "--home-dataset", "--source", "--licence", "--attribution", "--redistributable"},
                         {"--one-direction"}, {"--package", "--name", "--from", "--to", "--from-sphere", "--to-sphere"}, 0, {"--templates"});
    nxr::pipelines::RegistrationCreateOptions o;
    o.package = a.get("--package");
    o.name = a.get("--name");
    o.from = a.get("--from");
    o.to = a.get("--to");
    o.from_sphere = a.get("--from-sphere");
    o.to_sphere = a.get("--to-sphere");
    if (const auto m = a.maybe("--method")) o.method = *m;
    if (const auto k = a.maybe("--neighbours")) {
      try { o.neighbours = std::stoi(*k); } catch (const std::exception&) { throw Usage(prog + ": error: argument --neighbours: invalid integer value: '" + *k + "'"); }
    }
    o.both_directions = a.flag.count("--one-direction") == 0;
    if (a.multi.count("--templates")) for (const auto& p : a.multi.at("--templates")) o.templates.push_back(p);
    if (const auto h = a.maybe("--home-dataset")) o.home_dataset = *h;
    o.created_by = stamp();
    o.source = a.maybe("--source");
    o.licence = a.maybe("--licence");
    o.attribution = a.maybe("--attribution");
    if (const auto r = a.maybe("--redistributable")) {
      if (*r != "yes" && *r != "no") throw Usage(prog + ": error: argument --redistributable: invalid choice: '" + *r + "' (choose from 'yes', 'no')");
      o.redistributable = *r == "yes";
    }
    const auto res = nxr::pipelines::registration_create(o, [](const Json& j) { emit(j); });
    Json out = res.info;
    out["stage"] = "result";
    out["package"] = res.package.string();
    out["content_hash"] = SubjectPackage::open(res.package).record().content_hash;
    emit(out);
#else
    throw Error(code::unsupported, "nxr-neuro create registration needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
  } else {
    throw Usage("nxr-neuro package: error: argument action: invalid choice: '" + action +
                "' (choose from 'catalogue-create', 'create', 'verify', 'show', 'register', 'status', 'select', 'index', 'participants-import', 'participants-export', 'split', 'join', 'release', 'release-redistributable', 'templates', 'registration-create', 'template-create')");
  }
  return 0;
}
