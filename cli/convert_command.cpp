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
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

#include "cli.h"
#include "nxr/bst/inverse.h"
#include "nxr/bst/maps.h"
#include "nxr/bst/mri.h"
#include "nxr/bst/protocol.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/bst/session.h"
#include "nxr/bst/subject.h"
#include "nxr/bst/fibers.h"
#include "nxr/io/layout.h"
#include "nxr/bst/templates.h"

using namespace nxr::io;
using namespace cli;
namespace fs = std::filesystem;

namespace {

int list_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert list", rest, {}, {}, {}, 1, {}, "root");
  emit(Json{{"stage", "protocol"}, {"root", a.pos[0]}, {"subjects", nxr::bst::list_subjects(a.pos[0])}});
  return 0;
}

int scan_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert scan", rest, {"--prefer", "--out"}, {"--summary"}, {}, 1, {}, "root");
  nxr::bst::Reader prefer = nxr::bst::Reader::Auto;
  if (const auto p = a.maybe("--prefer")) {
    if (*p == "protocol.mat") prefer = nxr::bst::Reader::ProtocolMat;
    else if (*p == "walk") prefer = nxr::bst::Reader::Walk;
    else throw Usage("nxr-neuro convert scan: error: argument --prefer: invalid choice: '" + *p + "' (choose from 'protocol.mat', 'walk')");
  }
  const Json view = nxr::bst::read_source(a.pos[0], prefer);
  // `--out` WRITES, whatever --summary says: --summary chooses what is PRINTED, --out whether the view is persisted
  std::optional<std::string> wrote;
  if (const auto out = a.maybe("--out")) {
    wrote = nxr::bst::py_path(*out);
    const fs::path p = *out;
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    if (!f) throw Error(code::io, "cannot write " + *out);
    f << pyjson::dumps(view, 2) << "\n";
  }
  if (a.flag.count("--summary") || wrote) {
    Json line = Json{{"stage", "scan"}, {"kind", view["kind"]}, {"root", view["root"]}};
    if (wrote) line["wrote"] = *wrote;
    const Json counts = nxr::bst::source_counts(view);
    for (auto it = counts.begin(); it != counts.end(); ++it) line[it.key()] = it.value();
    emit(line);
  } else {
    std::cout << pyjson::dumps(view, 2) << std::endl;
  }
  return 0;
}

std::optional<double> resample_of(const Args& a) {
  const auto r = a.maybe("--resample");
  if (!r) return std::nullopt;
  try { std::size_t n = 0; const double x = std::stod(*r, &n); if (n == r->size()) return x; } catch (const std::exception&) {}
  throw Usage("error: argument --resample: invalid float value: '" + *r + "'");
}

Json result_line(const Json& r) {
  Json out = Json{{"stage", "result"}};
  for (auto it = r.begin(); it != r.end(); ++it) out[it.key()] = it.value();
  return out;
}

int subject_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert subject", rest, {"--subject", "--dataset", "--resample"}, {}, {"--subject", "--dataset"}, 1, {}, "root");
  DatasetWriter ds = open_one_database(a, "import subject");
  emit(result_line(nxr::bst::convert_subject(ds, a.pos[0], a.get("--subject"), [](const Json& line) { emit(line); }, stamp(), resample_of(a))));
  return 0;
}

int template_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert template", rest, {"--dataset"}, {}, {"--dataset"}, 1, {"--surface"}, "root");
  DatasetWriter ds = open_one_database(a, "import template");
  std::optional<std::vector<std::string>> surfaces;
  if (a.multi.count("--surface")) surfaces = a.multi.at("--surface");
  emit(result_line(nxr::bst::convert_template(ds, a.pos[0], surfaces, [](const Json& line) { emit(line); }, stamp())));
  return 0;
}

int group_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert group", rest, {"--dataset"}, {}, {"--dataset"}, 1, {}, "root");
  DatasetWriter ds = open_one_database(a, "import group");
  emit(result_line(nxr::bst::convert_group(ds, a.pos[0], [](const Json& line) { emit(line); }, stamp())));
  return 0;
}

int templates_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert templates", rest, {"--dataset", "--subject", "--trk", "--surface"}, {}, {"--dataset"}, 1, {}, "source");
  nxr::bst::TemplatesOptions o;
  if (const auto s = a.maybe("--subject")) o.subject = *s;
  if (const auto t = a.maybe("--trk")) o.trk = *t;
  o.surface = a.maybe("--surface");
  emit(result_line(nxr::bst::build_templates(a.pos[0], a.get("--dataset"), o, stamp())));
  return 0;
}

int surface_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert surface", rest, {"--dataset", "--subject", "--bst-root"}, {}, {"--dataset", "--subject"}, 0, {"--file"});
  if (!a.multi.count("--file")) throw Usage("nxr-neuro convert surface: give --file tess_*.mat ... (--fs-dir is not ported yet)");
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  const std::string bst_root = a.maybe("--bst-root").value_or("");
  Json wrote = Json::array();
  sub.compose([&] {
    for (const auto& f : a.multi.at("--file")) {
      std::error_code ec;
      if (!fs::is_regular_file(f, ec)) throw Error(code::not_found, "not a file: " + f);
      const SurfaceInfo info = nxr::bst::export_surface(sub, f, bst_root, false);
      wrote.push_back(info.name);
      Json line = Json{{"stage", "surface"}};
      const Json body = nxr::bst::surface_json(info);
      for (auto it = body.begin(); it != body.end(); ++it) line[it.key()] = it.value();
      emit(line);
    }
  });
  emit(Json{{"stage", "result"}, {"surfaces", wrote}});
  return 0;
}

// nxr convert mri --dataset D --subject S --anat <anat dir> [--only N ...]
int mri_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert mri", rest, {"--dataset", "--subject", "--anat"}, {}, {"--dataset", "--subject", "--anat"}, 0, {"--only"});
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  std::optional<std::vector<std::string>> only;
  if (a.multi.count("--only")) only = a.multi.at("--only");
  Json names = Json::array();
  sub.compose([&] {
    for (const Json& r : nxr::bst::import_subjectimages(sub, a.get("--anat"), only, [](const Json& line) { emit(line); })) names.push_back(r["name"]);
  });
  emit(Json{{"stage", "result"}, {"volumes", names}});
  return 0;
}

// nxr convert convert <protocol> --subject S --condition C --dataset D [...]: one condition as a session (the subject made if absent)
int convert_condition_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert convert", rest, {"--subject", "--condition", "--dataset", "--surface", "--recording", "--channel-file", "--resample"},
                       {"--no-gain", "--all-surfaces", "--include-imported", "--no-raw"}, {"--subject", "--condition", "--dataset"}, 1, {"--extra-surface"}, "root");
  const fs::path root = a.pos[0];
  DatasetWriter ds = open_dataset(a);
  nxr::bst::ConditionPlanOptions po;
  po.surface = a.maybe("--surface");
  if (a.multi.count("--extra-surface")) po.extra_surfaces = a.multi.at("--extra-surface");
  po.recording = a.maybe("--recording");
  po.channel_file = a.maybe("--channel-file");
  po.include_imported = a.flag.count("--include-imported") > 0;
  po.no_raw = a.flag.count("--no-raw") > 0;
  const nxr::bst::ConditionPlan plan = nxr::bst::condition_plan(root, a.get("--subject"), a.get("--condition"), po);
  nxr::bst::ConvertOptions co;
  co.write_gain = !a.flag.count("--no-gain");
  co.all_surfaces = a.flag.count("--all-surfaces") > 0;
  if (const auto r = a.maybe("--resample")) co.resample = std::stod(*r);
  std::optional<SubjectWriter> sub;
  if (ds.subject(a.get("--subject"))) {
    OpenSubjectOptions oo;
    oo.created_by = stamp();
    sub.emplace(SubjectWriter::open(ds, a.get("--subject"), oo));
  } else {
    SubjectOptions so;
    so.source_format = "brainstorm";
    so.source_path = nxr::bst::py_path(nxr::bst::anat_dir(root, a.get("--subject")).first.string());
    so.created_by = stamp();
    sub.emplace(SubjectWriter::create(ds, a.get("--subject"), so));
  }
  Json r;
  sub->compose([&] { r = nxr::bst::convert_condition(*sub, root, a.get("--condition"), plan, co, [](const Json& line) { emit(line); }); });
  if (!ds.is_package()) write_catalog(ds.root());
  Json out = Json{{"stage", "result"}};
  for (auto it = r.begin(); it != r.end(); ++it) out[it.key()] = it.value();
  emit(out);
  return 0;
}

// nxr convert inverse: ONE inverse kernel (results_*KERNEL*.mat) into a subject; never overwrites
int inverse_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert inverse", rest, {"--dataset", "--subject", "--file", "--acquisition", "--bst-root", "--channels-id", "--surface-id", "--forward-id", "--name", "--channel-file"},
                       {}, {"--dataset", "--subject", "--file", "--acquisition", "--bst-root"});
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  nxr::bst::AddInverseOptions ao;
  ao.acquisition = a.get("--acquisition");
  ao.bst_root = a.get("--bst-root");
  ao.channels_id = a.maybe("--channels-id");
  ao.surface_id = a.maybe("--surface-id");
  ao.forward_id = a.maybe("--forward-id");
  ao.name = a.maybe("--name");
  ao.channel_file = a.maybe("--channel-file");
  nxr::bst::InverseInfo info;
  sub.compose([&] { info = nxr::bst::add_inverse(sub, a.get("--file"), ao, [](const Json& line) { emit(line); }); });
  Json out = Json{{"stage", "result"}};
  const Json body = nxr::bst::inverse_json(info);
  for (auto it = body.begin(); it != body.end(); ++it) out[it.key()] = it.value();
  emit(out);
  return 0;
}

// nxr convert map: static maps into a subject, a source map (results_*.mat, one value per vertex of its SurfaceFile) or a region map (matrix_*.mat, one value per region of its Atlas)
int map_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert map", rest, {"--dataset", "--subject", "--bst-root", "--acquisition", "--provenance"}, {"--replace"},
                       {"--dataset", "--subject", "--file", "--bst-root"}, 0, {"--file"});
  std::optional<Json> prov;
  if (const auto pf = a.maybe("--provenance")) {
    std::ifstream in(*pf, std::ios::binary);
    if (!in) throw Error(code::not_found, "[Errno 2] No such file or directory: '" + *pf + "'");
    std::stringstream buf;
    buf << in.rdbuf();
    prov = Json::parse(buf.str());
  }
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  const nxr::bst::MapHooks& hooks = nxr::bst::recording_map_hooks();   // a map over time is written as a recording is
  Json wrote = Json::array();
  sub.compose([&] {
    for (const auto& f : a.multi.at("--file")) {
      std::error_code ec;
      if (!fs::is_regular_file(f, ec)) throw Error(code::not_found, "not a file: " + f);
      const Json info = nxr::bst::export_map(sub, f, a.get("--bst-root"), a.maybe("--acquisition"), a.flag.count("--replace") > 0, prov, &hooks);
      wrote.push_back(info["name"]);
      Json line = Json{{"stage", "map"}};
      for (auto it = info.begin(); it != info.end(); ++it) if (it.key() != "id") line[it.key()] = it.value();
      emit(line);
    }
  });
  emit(Json{{"stage", "result"}, {"maps", wrote}});
  return 0;
}

// nxr convert resample --dataset D --subject S --recording NAME --to HZ [--name N]
int resample_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert resample", rest, {"--dataset", "--subject", "--recording", "--to", "--name"}, {}, {"--dataset", "--subject", "--recording", "--to"});
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  Json info;
  sub.compose([&] { info = nxr::bst::add_resampled(sub, a.get("--recording"), std::stod(a.get("--to")), a.maybe("--name"), [](const Json& line) { emit(line); }); });
  Json out = Json{{"stage", "result"}};
  for (auto it = info.begin(); it != info.end(); ++it) out[it.key()] = it.value();
  emit(out);
  return 0;
}

// nxr convert badsegments --dataset D --subject S --from JSON [--replace]; nxr convert events --dataset D --subject S --from <protocol data folder> [--replace]
int events_command(const std::string& cmd, const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro convert " + cmd, rest, {"--dataset", "--subject", "--from"}, {"--replace"}, {"--dataset", "--subject", "--from"});
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  Json info;
  const bool replace = a.flag.count("--replace") > 0;
  if (cmd == "badsegments") {
    std::ifstream f(a.get("--from"), std::ios::binary);
    if (!f) throw Error(code::not_found, "[Errno 2] No such file or directory: " + nxr::bst::str_repr(nxr::bst::py_path(a.get("--from"))));
    const Json src = Json::parse(f);
    sub.compose([&] { info = nxr::bst::import_bad_segments(sub, src, replace); });
  } else {
    sub.compose([&] { info = nxr::bst::import_events(sub, a.get("--from"), replace); });
  }
  Json out = Json{{"stage", "result"}};
  for (auto it = info.begin(); it != info.end(); ++it) out[it.key()] = it.value();
  emit(out);
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
