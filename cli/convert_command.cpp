// nxr-neuro convert — the importers' verbs, nxr convert's: what a Brainstorm protocol holds (list · scan) and its anatomy and images into a dataset (subject · surface · mri · inverse · map).
//
//   nxr-neuro convert list <protocol>
//   nxr-neuro convert scan <protocol> [--prefer protocol.mat|walk] [--out FILE] [--summary]
//   nxr-neuro convert subject <protocol> --subject S --dataset <folder> [--resample HZ]
//   nxr-neuro convert template <protocol> --dataset D [--surface tess_*.mat ...]
//   nxr-neuro convert group <protocol> --dataset D
//   nxr-neuro convert convert <protocol> --subject S --condition C --dataset D [--resample HZ] ...
//   nxr-neuro convert resample|badsegments|events ...
//   nxr-neuro convert templates <references-fsaverage-ico7 dir> --dataset D [--subject fsaverage] [--trk hcp1065.trk] [--surface NODE]
//   nxr-neuro convert surface --dataset D --subject S --file tess_*.mat ... [--bst-root P]
//   nxr-neuro convert inverse --dataset D --subject S --file results_*KERNEL*.mat --acquisition C --bst-root P [--channels-id ..] [--surface-id ..] [--forward-id ..] [--name N] [--channel-file F]
//   nxr-neuro convert map --dataset D --subject S --file results_*.mat|matrix_*.mat ... --bst-root P [--acquisition C] [--replace] [--provenance FILE]
//
// Progress and results are JSON lines on stdout (`stage` keyed), as nxr's; an error is a `stage: error` line and exit 1. The other importers (sessions, volumes,
// templates builder, MNE, BIDS) are not implemented: their commands say so.
#include <iostream>
#include <optional>
#include <set>

#include "cli.h"

using namespace nxr::io;
using namespace cli;
namespace task = nxr::io::task;

// Each command parses, calls its library function (nxr/io/task/commands.h) once and prints.
namespace {

int list_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert list", rest, {}, {}, {}, 1, {}, "root");
  const task::CommandResult r = task::import_list({a.pos[0]});
  emit(Json{{"stage", "protocol"}, {"root", r.result["root"]}, {"subjects", r.result["subjects"]}});
  return 0;
}

int scan_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert scan";
  const Args a = parse(prog, rest, {"--prefer", "--out"}, {"--summary"}, {}, 1, {}, "root");
  task::ImportScanRequest rq{a.pos[0], a.maybe("--prefer"), a.maybe("--out"), a.flag.count("--summary") > 0 || a.maybe("--out").has_value()};
  const task::CommandResult r = call(prog, [&] { return task::import_scan(rq); });
  if (rq.summary) {           // --summary, or the view written by --out: the counts line
    Json line = Json{{"stage", "scan"}};
    for (auto it = r.result.begin(); it != r.result.end(); ++it) line[it.key()] = it.value();
    emit(line);
  } else {
    std::cout << pyjson::dumps(r.result["view"], 2) << std::endl;
  }
  return 0;
}

std::optional<double> resample_of(const Args& a) {
  const auto r = a.maybe("--resample");
  if (!r) return std::nullopt;
  try { std::size_t n = 0; const double x = std::stod(*r, &n); if (n == r->size()) return x; } catch (const std::exception&) {}
  throw Usage("error: argument --resample: invalid float value: '" + *r + "'");
}

int subject_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert subject";
  const Args a = parse(prog, rest, {"--subject", "--dataset", "--resample"}, {}, {"--subject", "--dataset"}, 1, {}, "root");
  const task::ImportSubjectRequest rq{a.pos[0], a.get("--subject"), a.get("--dataset"), resample_of(a), stamp()};
  emit_result(call(prog, [&] { return task::import_subject(rq, progress()); }).result);
  return 0;
}

int template_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert template";
  const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"}, 1, {"--surface"}, "root");
  const task::ImportTemplateRequest rq{a.pos[0], a.get("--dataset"), maybe_values(a, "--surface"), stamp()};
  emit_result(call(prog, [&] { return task::import_template(rq, progress()); }).result);
  return 0;
}

int group_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert group";
  const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"}, 1, {}, "root");
  const task::ImportGroupRequest rq{a.pos[0], a.get("--dataset"), stamp()};
  emit_result(call(prog, [&] { return task::import_group(rq, progress()); }).result);
  return 0;
}

int templates_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert templates", rest, {"--dataset", "--subject", "--trk", "--surface"}, {}, {"--dataset"}, 1, {}, "source");
  emit_result(task::import_templates({a.pos[0], a.get("--dataset"), a.maybe("--subject"), a.maybe("--trk"), a.maybe("--surface"), stamp()}).result);
  return 0;
}

int surface_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert surface";
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--bst-root"}, {}, {"--dataset", "--subject"}, 0, {"--file"});
  if (!a.multi.count("--file")) throw Usage(prog + ": give --file tess_*.mat ... (--fs-dir is not ported yet)");
  const task::ImportSurfaceRequest rq{a.get("--dataset"), a.get("--subject"), values_of(a, "--file"), a.maybe("--bst-root"), values_of(a, "--templates"), stamp()};
  emit_result(call(prog, [&] { return task::import_surface(rq, progress()); }).result);
  return 0;
}

// nxr convert mri --dataset D --subject S --anat <anat dir> [--only N ...]
int mri_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert mri";
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--anat"}, {}, {"--dataset", "--subject", "--anat"}, 0, {"--only"});
  const task::ImportMriRequest rq{a.get("--dataset"), a.get("--subject"), a.get("--anat"), maybe_values(a, "--only"), values_of(a, "--templates"), stamp()};
  emit_result(call(prog, [&] { return task::import_mri(rq, progress()); }).result);
  return 0;
}

// nxr convert convert <protocol> --subject S --condition C --dataset D [...]: one condition as a session (the subject made if absent)
int convert_condition_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert convert";
  const Args a = parse(prog, rest, {"--subject", "--condition", "--dataset", "--surface", "--recording", "--channel-file", "--resample"},
                       {"--no-gain", "--all-surfaces", "--include-imported", "--no-raw"}, {"--subject", "--condition", "--dataset"}, 1, {"--extra-surface"}, "root");
  task::ImportConditionRequest rq;
  rq.protocol = a.pos[0];
  rq.subject = a.get("--subject");
  rq.condition = a.get("--condition");
  rq.dataset = a.get("--dataset");
  rq.surface = a.maybe("--surface");
  rq.recording = a.maybe("--recording");
  rq.channel_file = a.maybe("--channel-file");
  if (const auto r = a.maybe("--resample")) rq.resample = std::stod(*r);
  rq.no_gain = a.flag.count("--no-gain") > 0;
  rq.all_surfaces = a.flag.count("--all-surfaces") > 0;
  rq.include_imported = a.flag.count("--include-imported") > 0;
  rq.no_raw = a.flag.count("--no-raw") > 0;
  rq.extra_surface = maybe_values(a, "--extra-surface");
  rq.templates = values_of(a, "--templates");
  rq.created_by = stamp();
  emit_result(call(prog, [&] { return task::import_condition(rq, progress()); }).result);
  return 0;
}

// nxr convert inverse: ONE inverse kernel (results_*KERNEL*.mat) into a subject; never overwrites
int inverse_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert inverse";
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--file", "--acquisition", "--bst-root", "--channels-id", "--surface-id", "--forward-id", "--name", "--channel-file"},
                       {}, {"--dataset", "--subject", "--file", "--acquisition", "--bst-root"});
  task::ImportInverseRequest rq;
  rq.dataset = a.get("--dataset");
  rq.subject = a.get("--subject");
  rq.file = a.get("--file");
  rq.acquisition = a.get("--acquisition");
  rq.bst_root = a.get("--bst-root");
  rq.channels_id = a.maybe("--channels-id");
  rq.surface_id = a.maybe("--surface-id");
  rq.forward_id = a.maybe("--forward-id");
  rq.name = a.maybe("--name");
  rq.channel_file = a.maybe("--channel-file");
  rq.templates = values_of(a, "--templates");
  rq.created_by = stamp();
  emit_result(call(prog, [&] { return task::import_inverse(rq, progress()); }).result);
  return 0;
}

// nxr convert map: static maps into a subject, a source map (results_*.mat, one value per vertex of its SurfaceFile) or a region map (matrix_*.mat, one value per region of its Atlas)
int map_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert map";
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--bst-root", "--acquisition", "--provenance"}, {"--replace"},
                       {"--dataset", "--subject", "--file", "--bst-root"}, 0, {"--file"});
  task::ImportMapRequest rq;
  rq.dataset = a.get("--dataset");
  rq.subject = a.get("--subject");
  rq.bst_root = a.get("--bst-root");
  rq.file = values_of(a, "--file");
  rq.acquisition = a.maybe("--acquisition");
  rq.provenance = a.maybe("--provenance");
  rq.replace = a.flag.count("--replace") > 0;
  rq.templates = values_of(a, "--templates");
  rq.created_by = stamp();
  emit_result(call(prog, [&] { return task::import_map(rq, progress()); }).result);
  return 0;
}

// nxr convert resample --dataset D --subject S --recording NAME --to HZ [--name N]
int resample_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert resample";
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--recording", "--to", "--name"}, {}, {"--dataset", "--subject", "--recording", "--to"});
  const task::ImportResampleRequest rq{a.get("--dataset"), a.get("--subject"), a.get("--recording"), std::stod(a.get("--to")), a.maybe("--name"), values_of(a, "--templates"), stamp()};
  emit_result(call(prog, [&] { return task::import_resample(rq, progress()); }).result);
  return 0;
}

// nxr convert badsegments --dataset D --subject S --from JSON [--replace]; nxr convert events --dataset D --subject S --from <protocol data folder> [--replace]
int events_command(const std::string& cmd, const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro convert " + cmd;
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--from"}, {"--replace"}, {"--dataset", "--subject", "--from"});
  const task::ImportBadsegmentsRequest rq{a.get("--dataset"), a.get("--subject"), a.get("--from"), a.flag.count("--replace") > 0, values_of(a, "--templates"), stamp()};
  if (cmd == "badsegments") emit_result(call(prog, [&] { return task::import_badsegments(rq, progress()); }).result);
  else emit_result(call(prog, [&] { return task::import_events(rq, progress()); }).result);
  return 0;
}

}  // namespace

int convert_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro convert: error: the following arguments are required: cmd");
  const std::string cmd = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  if (cmd == "list") return list_command(rest);
  if (cmd == "scan") return scan_command(rest);
  if (cmd == "subject") return subject_command(rest);
  if (cmd == "template") return template_command(rest);
  if (cmd == "group") return group_command(rest);
  if (cmd == "templates") return templates_command(rest);
  if (cmd == "surface") return surface_command(rest);
  if (cmd == "mri") return mri_command(rest);
  if (cmd == "convert") return convert_condition_command(rest);
  if (cmd == "inverse") return inverse_command(rest);
  if (cmd == "map") return map_command(rest);
  if (cmd == "resample") return resample_command(rest);
  if (cmd == "badsegments" || cmd == "events") return events_command(cmd, rest);
  static const std::set<std::string> later = {"qc", "migrate", "fragment", "merge", "mne-meg", "mne-eeg", "bids-ieeg", "atlas"};
  if (later.count(cmd)) {
    emit(Json{{"stage", "error"}, {"message", "NotImplementedError: nxr-neuro convert " + cmd + " is not ported yet (ported: list, scan, subject, convert, template, group, templates [the atlas/nibabel readers excepted], surface, mri, inverse, map, resample, badsegments, events)"}});
    return 1;
  }
  throw Usage("nxr-neuro convert: error: argument cmd: invalid choice: '" + cmd + "' (choose from 'list', 'scan', 'subject', 'convert', 'template', 'group', 'templates', 'surface', 'mri', 'inverse', 'map', 'resample', 'badsegments', 'events')");
}
