// The import commands' library functions (nxr/io/task/commands.h, "import"): Brainstorm protocols, files in the field's formats, BIDS and the participants, moved
// out of cli/ so the command line and the faces call one function. Behaviour, refusals and results are the command line's, unchanged.
#include <fstream>
#include <functional>
#include <sstream>

#include "commands_open.h"
#include "nxr/bst/fibers.h"
#include "nxr/bst/covariance.h"
#include "nxr/bst/inverse.h"
#include "nxr/bst/maps.h"
#include "nxr/bst/mat.h"
#include "nxr/bst/mri.h"
#include "nxr/bst/protocol.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/bst/session.h"
#include "nxr/bst/subject.h"
#include "nxr/bst/templates.h"
#include "nxr/convert/bids.h"
#include "nxr/convert/cifti_import.h"
#include "nxr/convert/fif_objects.h"
#include "nxr/convert/freesurfer_import.h"
#include "nxr/convert/gifti_import.h"
#include "nxr/convert/nifti_import.h"
#include "nxr/convert/recording.h"
#include "nxr/convert/synthetic.h"
#include "nxr/io/errors.h"
#include "nxr/io/layout.h"
#include "nxr/io/participants.h"
#include "nxr/io/py_json.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/jobs.h"

namespace nxr::io::task {

using namespace commands_detail;

namespace {

CommandResult result_of(Json body) {
  CommandResult r;
  r.result = std::move(body);
  return r;
}

// the subject of a dataset an import writes into, opened as every command opens it (built in place: the subject keeps a reference to its dataset)
struct Opened {
  Opened(const std::string& dataset, const std::string& subject, const std::vector<std::string>& templates, const std::string& created_by, const Progress& progress)
      : ds(open_dataset(dataset, subject, OpenOptions{}, templates, progress)), sub(SubjectWriter::open(ds, subject, options(created_by))) {}
  static OpenSubjectOptions options(const std::string& created_by) {
    OpenSubjectOptions oo;
    oo.created_by = created_by;
    return oo;
  }
  DatasetWriter ds;
  SubjectWriter sub;
};

// the subject package a file importer writes into (made first by `create package`)
struct OpenedPackage {
  explicit OpenedPackage(const std::string& package) : name(SubjectPackage::open(package).record().name), w(open_package_writer(package)), sub(SubjectWriter::open(w, name)) {}
  std::string name;
  DatasetWriter w;
  SubjectWriter sub;
};


// the program a job row names when the request carries none (the file importers': they take no producer stamp)
const char* const kProgram = "nxr-neuro";
template <class T>
Json oj(const std::optional<T>& v) { return v ? Json(*v) : Json(); }

// the recorded source of a file: relative to the root the caller names (default: the file's folder), never an absolute path
std::string source_of(const std::string& file, const std::optional<std::string>& source_root, const fs::path& default_root) {
  const fs::path root = source_root ? fs::path(*source_root) : default_root;
  return fs::relative(fs::absolute(file), fs::absolute(root.empty() ? fs::path(".") : root)).generic_string();
}

// the progress handed to an importer: the caller's, or one that says nothing (the importers call it unconditionally)
Progress said(const Progress& progress) { return progress ? progress : Progress([](const Json&) {}); }

// THE FILES A BRAINSTORM IMPORT SKIPPED because this build cannot read them (MATLAB v7.3 in the WebAssembly build; nxr/bst/mat.h): each named with its reason in the result's
// `skipped` [{what, reason}] (beside what the import skips for its own reasons), and a `message` naming them. The job row keeps the result, so both are in it too.
Json with_skipped(Json r, const nxr::bst::SkippedFiles& s) {
  if (s.files().empty()) return r;
  if (!r.contains("skipped") || !r["skipped"].is_array()) r["skipped"] = Json::array();
  std::string names;
  for (const auto& f : s.files()) {
    bool listed = false;
    for (const auto& x : r["skipped"]) listed = listed || (x.is_object() && x.value("what", std::string()) == f.file);
    if (!listed) r["skipped"].push_back(Json{{"what", f.file}, {"reason", f.reason}});
    names += (names.empty() ? "" : ", ") + fs::path(f.file).filename().string();
  }
  r["message"] = std::to_string(s.files().size()) + (s.files().size() == 1 ? " file" : " files") + " skipped (" + names + "): " + nxr::bst::kUnreadableV73;
  return r;
}

// THE NEW MEMBER OF A CATALOGUE AN IMPORT MAKES (a subject, the template, the group: a catalogue is the default form): its package made in the dataset's folder as `create subject`
// makes one, with the subject row `so`, filled by `fill`, closed, and only then registered. A failed import removes the package it made, and registers nothing, as a one-database
// dataset removes a subject whose import failed. A name the catalogue lists, or a package of that name already in the folder, is refused (a subject is imported whole).
Json into_new_member(const std::string& dataset, const std::string& name, const SubjectOptions& so, const std::string& created_by, const std::function<Json(DatasetWriter&)>& fill) {
  pieces::discard_orphans(dataset);
  const Catalogue cat = Catalogue::open(dataset);
  if (cat.find(name)) throw Error(code::already_exists, name + ": already a subject of " + dataset + " - delete it first (a subject is imported whole)");
  const fs::path pkg = member_package_path(cat, name);
  if (fs::exists(pkg)) throw Error(code::already_exists, pkg.string() + ": a package of that name is already in the dataset's folder (`register package` lists it)");
  Json r;
  try {
    DatasetWriter w = create_member_package(cat, name, so, created_by);
    r = fill(w);
    w.close();
  } catch (...) {
    std::error_code ec;
    fs::remove_all(pkg, ec);   // the package this import made (nothing of it was registered)
    throw;
  }
  register_member(dataset, pkg, name, created_by);
  return r;
}
bool listed_in_catalogue(const std::string& dataset, const std::string& name) { return is_catalogue_dataset(dataset) && Catalogue::open(dataset).find(name).has_value(); }
// where the job row of an import of the template or the group goes: a catalogue's member of that name (its package), or the one-database dataset itself
std::optional<std::string> member_of(const std::string& dataset, const char* name) {
  return is_catalogue_dataset(dataset) ? std::optional<std::string>(name) : std::nullopt;
}

void require_file(const std::string& file) {
  if (!fs::exists(file)) throw Error(code::not_found, "no such file: " + file);
}

}  // namespace

// ---- Brainstorm protocols ----

CommandResult import_list(const ImportListRequest& rq) { return result_of(Json{{"root", rq.protocol}, {"subjects", nxr::bst::list_subjects(rq.protocol)}}); }

CommandResult import_scan(const ImportScanRequest& rq) {
  nxr::bst::Reader prefer = nxr::bst::Reader::Auto;
  if (rq.prefer) {
    if (*rq.prefer == "protocol.mat") prefer = nxr::bst::Reader::ProtocolMat;
    else if (*rq.prefer == "walk") prefer = nxr::bst::Reader::Walk;
    else throw Refused("argument --prefer: invalid choice: '" + *rq.prefer + "' (choose from 'protocol.mat', 'walk')");
  }
  const Json view = nxr::bst::read_source(rq.protocol, prefer);
  // `out` WRITES, whatever `summary` says: `summary` chooses what is given back, `out` whether the view is persisted
  Json r = Json{{"kind", view["kind"]}, {"root", view["root"]}};
  if (rq.out) {
    r["wrote"] = nxr::bst::py_path(*rq.out);
    const fs::path p = *rq.out;
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    if (!f) throw Error(code::io, "cannot write " + *rq.out);
    f << pyjson::dumps(view, 2) << "\n";
  }
  const Json counts = nxr::bst::source_counts(view);
  for (auto it = counts.begin(); it != counts.end(); ++it) r[it.key()] = it.value();
  if (!rq.summary) r["view"] = view;
  return result_of(std::move(r));
}

static CommandResult import_subject_work(const ImportSubjectRequest& rq, const Progress& progress);
CommandResult import_subject(const ImportSubjectRequest& rq, const Progress& progress) {
  return recorded("import:subject", JobTarget{rq.dataset, "", rq.subject}, Json{{"protocol", rq.protocol}, {"subject", rq.subject}, {"resample", oj(rq.resample)}}, rq.created_by, [&] { return import_subject_work(rq, progress); });
}

static CommandResult import_subject_work(const ImportSubjectRequest& rq, const Progress& progress) {
  const nxr::bst::SkippedFiles skipped;
  const auto convert = [&](DatasetWriter& ds) { return nxr::bst::convert_subject(ds, rq.protocol, rq.subject, said(progress), rq.created_by, rq.resample); };
  if (is_catalogue_dataset(rq.dataset))   // the subject into a package of its own
    return result_of(with_skipped(into_new_member(rq.dataset, rq.subject, nxr::bst::subject_options(rq.protocol, rq.subject, rq.created_by), rq.created_by, convert), skipped));
  DatasetWriter ds = open_one_database(rq.dataset, "import subject");
  return result_of(with_skipped(convert(ds), skipped));
}

static CommandResult import_template_work(const ImportTemplateRequest& rq, const Progress& progress);
CommandResult import_template(const ImportTemplateRequest& rq, const Progress& progress) {
  return recorded("import:template", JobTarget{rq.dataset, "", member_of(rq.dataset, nxr::bst::kTemplateSubject)}, Json{{"protocol", rq.protocol}, {"surface", oj(rq.surface)}}, rq.created_by, [&] { return import_template_work(rq, progress); });
}

static CommandResult import_template_work(const ImportTemplateRequest& rq, const Progress& progress) {
  const nxr::bst::SkippedFiles skipped;
  const auto convert = [&](DatasetWriter& ds) { return nxr::bst::convert_template(ds, rq.protocol, rq.surface, said(progress), rq.created_by); };
  if (is_catalogue_dataset(rq.dataset))   // the template into a package of its own (kind `template`), a member of the dataset
    return result_of(with_skipped(into_new_member(rq.dataset, nxr::bst::kTemplateSubject, nxr::bst::template_options(rq.protocol, rq.created_by), rq.created_by, convert), skipped));
  DatasetWriter ds = open_one_database(rq.dataset, "import template");
  return result_of(with_skipped(convert(ds), skipped));
}

static CommandResult import_group_work(const ImportGroupRequest& rq, const Progress& progress);
CommandResult import_group(const ImportGroupRequest& rq, const Progress& progress) {
  const std::string tname = is_catalogue_dataset(rq.dataset) ? dataset_template(rq.dataset).name : std::string();
  return recorded("import:group", JobTarget{rq.dataset, "", member_of(rq.dataset, tname.c_str())}, Json{{"protocol", rq.protocol}}, rq.created_by, [&] { return import_group_work(rq, progress); });
}

static CommandResult import_group_work(const ImportGroupRequest& rq, const Progress& progress) {
  const nxr::bst::SkippedFiles skipped;
  const auto convert = [&](DatasetWriter& ds) { return nxr::bst::convert_group(ds, rq.protocol, said(progress), rq.created_by); };
  if (is_catalogue_dataset(rq.dataset)) {   // the protocol's group results into the dataset's template (format 56: the template holds the cohort), registered again
    const CatalogueEntry t = dataset_template(rq.dataset);
    Json out;
    fs::path package;
    {
      DatasetWriter ds = open_subject_package(rq.dataset, t.name);
      out = convert(ds);
      package = ds.root();
    }
    register_member(rq.dataset, package, t.name, rq.created_by);
    return result_of(with_skipped(out, skipped));
  }
  DatasetWriter ds = open_one_database(rq.dataset, "import group");
  return result_of(with_skipped(convert(ds), skipped));
}

CommandResult import_templates(const ImportTemplatesRequest& rq) {
  nxr::bst::TemplatesOptions o;
  if (rq.subject) o.subject = *rq.subject;
  if (rq.trk) o.trk = *rq.trk;
  o.surface = rq.surface;
  return result_of(nxr::bst::build_templates(rq.source, rq.dataset, o, rq.created_by));
}

static CommandResult import_surface_work(const ImportSurfaceRequest& rq, const Progress& progress);
CommandResult import_surface(const ImportSurfaceRequest& rq, const Progress& progress) {
  return recorded("import:surface", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"file", rq.file}, {"bst_root", oj(rq.bst_root)}}, rq.created_by, [&] { return import_surface_work(rq, progress); });
}

static CommandResult import_surface_work(const ImportSurfaceRequest& rq, const Progress& progress) {
  if (rq.file.empty()) throw Refused("give --file tess_*.mat ... (--fs-dir is not ported yet)");
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  const std::string bst_root = rq.bst_root.value_or("");
  Json wrote = Json::array();
  o.sub.compose([&] {
    for (const auto& f : rq.file) {
      std::error_code ec;
      if (!fs::is_regular_file(f, ec)) throw Error(code::not_found, "not a file: " + f);
      const SurfaceInfo info = nxr::bst::export_surface(o.sub, f, bst_root, false);
      wrote.push_back(info.name);
      Json line = Json{{"stage", "surface"}};
      const Json body = nxr::bst::surface_json(info);
      for (auto it = body.begin(); it != body.end(); ++it) line[it.key()] = it.value();
      if (progress) progress(line);
    }
  });
  return result_of(Json{{"surfaces", wrote}});
}

static CommandResult import_mri_work(const ImportMriRequest& rq, const Progress& progress);
CommandResult import_mri(const ImportMriRequest& rq, const Progress& progress) {
  return recorded("import:mri", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"anat", rq.anat}, {"only", oj(rq.only)}}, rq.created_by, [&] { return import_mri_work(rq, progress); });
}

static CommandResult import_mri_work(const ImportMriRequest& rq, const Progress& progress) {
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  Json names = Json::array();
  o.sub.compose([&] {
    for (const Json& r : nxr::bst::import_subjectimages(o.sub, rq.anat, rq.only, said(progress))) names.push_back(r["name"]);
  });
  return result_of(Json{{"volumes", names}});
}

static CommandResult import_condition_work(const ImportConditionRequest& rq, const Progress& progress);
CommandResult import_condition(const ImportConditionRequest& rq, const Progress& progress) {
  return recorded("import:condition", JobTarget{rq.dataset, "", rq.subject}, Json{{"protocol", rq.protocol}, {"subject", rq.subject}, {"condition", rq.condition}, {"resample", oj(rq.resample)}}, rq.created_by, [&] { return import_condition_work(rq, progress); });
}

static CommandResult import_condition_work(const ImportConditionRequest& rq, const Progress& progress) {
  const fs::path root = rq.protocol;
  nxr::bst::ConditionPlanOptions po;
  po.surface = rq.surface;
  if (rq.extra_surface) po.extra_surfaces = *rq.extra_surface;
  po.recording = rq.recording;
  po.channel_file = rq.channel_file;
  po.include_imported = rq.include_imported;
  po.no_raw = rq.no_raw;
  const nxr::bst::ConditionPlan plan = nxr::bst::condition_plan(root, rq.subject, rq.condition, po);
  nxr::bst::ConvertOptions co;
  co.write_gain = !rq.no_gain;
  co.all_surfaces = rq.all_surfaces;
  if (rq.resample) co.resample = *rq.resample;
  const nxr::bst::SkippedFiles skipped;
  // the condition into the subject, which is made when the dataset has no such subject yet
  const auto convert = [&](DatasetWriter& ds) {
    std::optional<SubjectWriter> sub;
    if (ds.subject(rq.subject)) {
      OpenSubjectOptions oo;
      oo.created_by = rq.created_by;
      sub.emplace(SubjectWriter::open(ds, rq.subject, oo));
    } else {
      sub.emplace(SubjectWriter::create(ds, rq.subject, nxr::bst::subject_options(root, rq.subject, rq.created_by)));
    }
    Json r;
    sub->compose([&] { r = nxr::bst::convert_condition(*sub, root, rq.condition, plan, co, said(progress)); });
    if (!ds.is_package()) write_catalog(ds.root());
    return r;
  };
  if (is_catalogue_dataset(rq.dataset) && !listed_in_catalogue(rq.dataset, rq.subject))   // a catalogue without the subject: its package is made, filled, then registered
    return result_of(with_skipped(into_new_member(rq.dataset, rq.subject, nxr::bst::subject_options(root, rq.subject, rq.created_by), rq.created_by, convert), skipped));
  DatasetWriter ds = open_dataset(rq.dataset, rq.subject, OpenOptions{}, rq.templates, progress);
  return result_of(with_skipped(convert(ds), skipped));
}

static CommandResult import_inverse_work(const ImportInverseRequest& rq, const Progress& progress);
CommandResult import_inverse(const ImportInverseRequest& rq, const Progress& progress) {
  return recorded("import:inverse", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"file", rq.file}, {"acquisition", rq.acquisition}, {"name", oj(rq.name)}}, rq.created_by, [&] { return import_inverse_work(rq, progress); });
}

static CommandResult import_inverse_work(const ImportInverseRequest& rq, const Progress& progress) {
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  nxr::bst::AddInverseOptions ao;
  ao.acquisition = rq.acquisition;
  ao.bst_root = rq.bst_root;
  ao.channels_id = rq.channels_id;
  ao.surface_id = rq.surface_id;
  ao.forward_id = rq.forward_id;
  ao.name = rq.name;
  ao.channel_file = rq.channel_file;
  nxr::bst::InverseInfo info;
  o.sub.compose([&] { info = nxr::bst::add_inverse(o.sub, rq.file, ao, said(progress)); });
  return result_of(nxr::bst::inverse_json(info));
}

static CommandResult import_covariance_work(const ImportCovarianceRequest& rq, const Progress& progress);
CommandResult import_covariance(const ImportCovarianceRequest& rq, const Progress& progress) {
  return recorded("import:covariance", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"condition", rq.condition}, {"kind", rq.kind}, {"name", oj(rq.name)}, {"replace", rq.replace}}, rq.created_by,
                  [&] { return import_covariance_work(rq, progress); });
}

static CommandResult import_covariance_work(const ImportCovarianceRequest& rq, const Progress& progress) {
  const fs::path data = fs::path(rq.protocol) / "data" / rq.subject;
  std::error_code ec;
  fs::path dir = data / rq.condition;
  if (!fs::is_directory(dir, ec)) dir = data / ("@raw" + rq.condition);     // a raw file's study
  if (!fs::is_directory(dir, ec)) throw Error(code::not_found, "no study " + rq.condition + " of " + rq.subject + " in the protocol");
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  nxr::bst::CovarianceOptions co;
  co.kind = rq.kind;
  co.bst_root = rq.protocol;
  co.name = rq.name;
  co.replace = rq.replace;
  Json body;
  o.sub.compose([&] { body = nxr::bst::import_covariance(o.sub, dir, rq.condition, co, said(progress)); });
  return result_of(body);
}

static CommandResult import_map_work(const ImportMapRequest& rq, const Progress& progress);
CommandResult import_map(const ImportMapRequest& rq, const Progress& progress) {
  return recorded("import:map", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"file", rq.file}, {"acquisition", oj(rq.acquisition)}, {"replace", rq.replace}}, rq.created_by, [&] { return import_map_work(rq, progress); });
}

static CommandResult import_map_work(const ImportMapRequest& rq, const Progress& progress) {
  std::optional<Json> prov;
  if (rq.provenance) {
    std::ifstream in(*rq.provenance, std::ios::binary);
    if (!in) throw Error(code::not_found, "[Errno 2] No such file or directory: '" + *rq.provenance + "'");
    std::stringstream buf;
    buf << in.rdbuf();
    prov = Json::parse(buf.str());
  }
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  const nxr::bst::MapHooks& hooks = nxr::bst::recording_map_hooks();   // a map over time is written as a recording is
  Json wrote = Json::array();
  o.sub.compose([&] {
    for (const auto& f : rq.file) {
      std::error_code ec;
      if (!fs::is_regular_file(f, ec)) throw Error(code::not_found, "not a file: " + f);
      const Json info = nxr::bst::export_map(o.sub, f, rq.bst_root, rq.acquisition, rq.replace, prov, &hooks);
      wrote.push_back(info["name"]);
      Json line = Json{{"stage", "map"}};
      for (auto it = info.begin(); it != info.end(); ++it) if (it.key() != "id") line[it.key()] = it.value();
      if (progress) progress(line);
    }
  });
  return result_of(Json{{"maps", wrote}});
}

static CommandResult import_resample_work(const ImportResampleRequest& rq, const Progress& progress);
CommandResult import_resample(const ImportResampleRequest& rq, const Progress& progress) {
  return recorded("import:resample", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"recording", rq.recording}, {"to", rq.to}, {"name", oj(rq.name)}}, rq.created_by, [&] { return import_resample_work(rq, progress); });
}

static CommandResult import_resample_work(const ImportResampleRequest& rq, const Progress& progress) {
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  Json info;
  o.sub.compose([&] { info = nxr::bst::add_resampled(o.sub, rq.recording, rq.to, rq.name, said(progress)); });
  return result_of(std::move(info));
}

static CommandResult import_badsegments_work(const ImportBadsegmentsRequest& rq, const Progress& progress);
CommandResult import_badsegments(const ImportBadsegmentsRequest& rq, const Progress& progress) {
  return recorded("import:badsegments", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"from", rq.from}, {"replace", rq.replace}}, rq.created_by, [&] { return import_badsegments_work(rq, progress); });
}

static CommandResult import_badsegments_work(const ImportBadsegmentsRequest& rq, const Progress& progress) {
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  std::ifstream f(rq.from, std::ios::binary);
  if (!f) throw Error(code::not_found, "[Errno 2] No such file or directory: " + nxr::bst::str_repr(nxr::bst::py_path(rq.from)));
  const Json src = Json::parse(f);
  Json info;
  o.sub.compose([&] { info = nxr::bst::import_bad_segments(o.sub, src, rq.replace); });
  return result_of(std::move(info));
}

static CommandResult import_events_work(const ImportEventsRequest& rq, const Progress& progress);
CommandResult import_events(const ImportEventsRequest& rq, const Progress& progress) {
  return recorded("import:events", JobTarget{rq.dataset, "", rq.subject}, Json{{"subject", rq.subject}, {"from", rq.from}, {"replace", rq.replace}}, rq.created_by, [&] { return import_events_work(rq, progress); });
}

static CommandResult import_events_work(const ImportEventsRequest& rq, const Progress& progress) {
  Opened o(rq.dataset, rq.subject, rq.templates, rq.created_by, progress);
  Json info;
  o.sub.compose([&] { info = nxr::bst::import_events(o.sub, rq.from, rq.replace); });
  return result_of(std::move(info));
}

// ---- files in the field's formats, into a subject package ----

static CommandResult import_recording_work(const ImportRecordingRequest& rq, const Progress& progress);
CommandResult import_recording(const ImportRecordingRequest& rq, const Progress& progress) {
  return recorded("import:recording", JobTarget{"", rq.package, std::nullopt}, Json{{"from", rq.from}, {"name", oj(rq.name)}, {"acquisition", oj(rq.acquisition)}, {"format", oj(rq.format)}, {"in_place", rq.in_place}}, kProgram, [&] { return import_recording_work(rq, progress); });
}

static CommandResult import_recording_work(const ImportRecordingRequest& rq, const Progress& progress) {
  const std::string& file = rq.from;
  require_file(file);
  nxr::convert::RecordingImport opt;
  opt.format = rq.format.value_or(nxr::convert::format_of(file));
  opt.name = rq.name.value_or(fs::path(file).stem().string());
  opt.acquisition = rq.acquisition.value_or("rest");
  opt.anonymize = rq.origin.anonymize;
  opt.in_place = rq.in_place;
  opt.licence = rq.origin.licence.value_or("unknown");
  opt.attribution = rq.origin.attribution.value_or("unknown");
  opt.redistributable = false;
  opt.montage = rq.montage.value_or("");
  opt.source = source_of(file, rq.origin.source_root, fs::path(file).parent_path());
  opt.source_file = fs::is_directory(file) ? "" : file;   // a CTF .ds folder is not one file: no hash
  opt.progress = said(progress);
  auto reader = nxr::convert::open_recording(opt.format, file);
  OpenedPackage p(rq.package);
  return result_of(nxr::convert::import_recording(p.w, p.sub, *reader, opt));
}

namespace {
CommandResult import_fif_object(const std::string& file, const std::string& package, nxr::convert::FifObjectImport opt, const fs::path& root, const Progress& progress) {
  opt.source = fs::relative(fs::absolute(file), fs::absolute(root.empty() ? fs::path(".") : root)).generic_string();
  opt.source_file = file;
  opt.progress = said(progress);
  OpenedPackage p(package);
  return result_of(nxr::convert::import_fif_object(p.w, p.sub, file, opt));
}
}  // namespace

static CommandResult import_fif_work(const ImportFifRequest& rq, const Progress& progress);
CommandResult import_fif(const ImportFifRequest& rq, const Progress& progress) {
  return recorded("import:fif", JobTarget{"", rq.package, std::nullopt}, Json{{"from", rq.from}, {"kind", oj(rq.kind)}, {"name", oj(rq.name)}}, kProgram, [&] { return import_fif_work(rq, progress); });
}

static CommandResult import_fif_work(const ImportFifRequest& rq, const Progress& progress) {
  require_file(rq.from);
  nxr::convert::FifObjectImport opt;
  opt.kind = rq.kind.value_or("");
  opt.name = rq.name.value_or("");
  opt.channels = rq.channels.value_or("");
  opt.acquisition = rq.acquisition.value_or("rest");
  opt.recording = rq.recording.value_or("");
  opt.surface = rq.surface.value_or("");
  opt.licence = rq.origin.licence.value_or("unknown");
  opt.attribution = rq.origin.attribution.value_or("unknown");
  opt.anonymize = rq.origin.anonymize;
  const fs::path root = rq.origin.source_root ? fs::path(*rq.origin.source_root) : fs::path(rq.from).parent_path();
  return import_fif_object(rq.from, rq.package, opt, root, progress);
}

static CommandResult import_transform_work(const ImportTransformRequest& rq, const Progress& progress);
CommandResult import_transform(const ImportTransformRequest& rq, const Progress& progress) {
  return recorded("import:transform", JobTarget{"", rq.package, std::nullopt}, Json{{"from", rq.from}, {"name", oj(rq.name)}}, kProgram, [&] { return import_transform_work(rq, progress); });
}

static CommandResult import_transform_work(const ImportTransformRequest& rq, const Progress& progress) {
  require_file(rq.from);
  nxr::convert::FifObjectImport opt;
  opt.kind = "trans";
  opt.name = rq.name.value_or("");
  opt.channels = rq.channels.value_or("");
  return import_fif_object(rq.from, rq.package, opt, fs::path(rq.from).parent_path(), progress);
}

static CommandResult import_nifti_work(const ImportNiftiRequest& rq, const Progress& progress);
CommandResult import_nifti(const ImportNiftiRequest& rq, const Progress& progress) {
  return recorded("import:nifti", JobTarget{"", rq.package, std::nullopt}, Json{{"from", rq.from}, {"name", oj(rq.name)}, {"kind", oj(rq.kind)}}, kProgram, [&] { return import_nifti_work(rq, progress); });
}

static CommandResult import_nifti_work(const ImportNiftiRequest& rq, const Progress& progress) {
  const std::string& file = rq.from;
  require_file(file);
  nxr::convert::NiftiImport opt;
  opt.name = rq.name.value_or("");
  opt.kind = rq.kind.value_or("");
  opt.parcellation = rq.parcellation.value_or("");
  opt.label_names = rq.label_names.value_or("");
  opt.field_kind = rq.field_kind.value_or("intensity");
  opt.unit = rq.unit.value_or("");
  opt.licence = rq.origin.licence.value_or("unknown");
  opt.attribution = rq.origin.attribution.value_or("unknown");
  opt.anonymize = rq.origin.anonymize;
  opt.source = source_of(file, rq.origin.source_root, fs::path(file).parent_path());
  opt.source_file = file;
  opt.progress = said(progress);
  OpenedPackage p(rq.package);
  return result_of(nxr::convert::import_nifti(p.w, p.sub, file, opt));
}

static CommandResult import_freesurfer_work(const ImportFreesurferRequest& rq, const Progress& progress);
CommandResult import_freesurfer(const ImportFreesurferRequest& rq, const Progress& progress) {
  return recorded("import:freesurfer", JobTarget{"", rq.package, std::nullopt}, Json{{"from", rq.from}, {"name", oj(rq.name)}}, kProgram, [&] { return import_freesurfer_work(rq, progress); });
}

static CommandResult import_freesurfer_work(const ImportFreesurferRequest& rq, const Progress& progress) {
  const std::string& dir = rq.from;
  if (!fs::is_directory(dir)) throw Error(code::not_found, "no such folder: " + dir);
  nxr::convert::FreeSurferImport opt;
  opt.name = rq.name.value_or("cortex");
  if (rq.surfaces) opt.surfaces = *rq.surfaces;
  if (rq.annotations) opt.annotations = *rq.annotations;
  if (rq.overlays) opt.overlays = *rq.overlays;
  opt.orig = rq.orig.value_or("");
  opt.field_kind = rq.field_kind.value_or("overlay");
  opt.licence = rq.origin.licence.value_or("unknown");
  opt.attribution = rq.origin.attribution.value_or("unknown");
  opt.anonymize = rq.origin.anonymize;
  opt.source = source_of(dir, rq.origin.source_root, fs::path(dir).parent_path());
  opt.progress = said(progress);
  OpenedPackage p(rq.package);
  return result_of(nxr::convert::import_freesurfer_subject(p.w, p.sub, dir, opt));
}

static CommandResult import_gifti_work(const ImportGiftiRequest& rq, const Progress& progress);
CommandResult import_gifti(const ImportGiftiRequest& rq, const Progress& progress) {
  return recorded("import:gifti", JobTarget{"", rq.package, std::nullopt}, Json{{"from", rq.from}, {"name", oj(rq.name)}, {"kind", oj(rq.kind)}}, kProgram, [&] { return import_gifti_work(rq, progress); });
}

static CommandResult import_gifti_work(const ImportGiftiRequest& rq, const Progress& progress) {
  const std::string& file = rq.from;
  require_file(file);
  nxr::convert::GiftiImport opt;
  opt.name = rq.name.value_or("");
  opt.kind = rq.kind.value_or("");
  opt.surface = rq.surface.value_or("");
  opt.parcellation = rq.parcellation.value_or("");
  opt.field_kind = rq.field_kind.value_or("overlay");
  opt.unit = rq.unit.value_or("");
  opt.licence = rq.origin.licence.value_or("unknown");
  opt.attribution = rq.origin.attribution.value_or("unknown");
  opt.anonymize = rq.origin.anonymize;
  opt.source = source_of(file, rq.origin.source_root, fs::path(file).parent_path());
  opt.source_file = file;
  opt.progress = said(progress);
  OpenedPackage p(rq.package);
  return result_of(nxr::convert::import_gifti(p.w, p.sub, file, opt));
}

static CommandResult import_cifti_work(const ImportCiftiRequest& rq, const Progress& progress);
CommandResult import_cifti(const ImportCiftiRequest& rq, const Progress& progress) {
  return recorded("import:cifti", JobTarget{"", rq.package, std::nullopt}, Json{{"from", rq.from}, {"name", oj(rq.name)}}, kProgram, [&] { return import_cifti_work(rq, progress); });
}

static CommandResult import_cifti_work(const ImportCiftiRequest& rq, const Progress& progress) {
  const std::string& file = rq.from;
  require_file(file);
  nxr::convert::CiftiImport opt;
  opt.name = rq.name.value_or("");
  opt.left = rq.left.value_or("");
  opt.right = rq.right.value_or("");
  opt.parcellation = rq.parcellation.value_or("");
  opt.field_kind = rq.field_kind.value_or("overlay");
  opt.unit = rq.unit.value_or("");
  opt.licence = rq.origin.licence.value_or("unknown");
  opt.attribution = rq.origin.attribution.value_or("unknown");
  opt.anonymize = rq.origin.anonymize;
  opt.source = source_of(file, rq.origin.source_root, fs::path(file).parent_path());
  opt.source_file = file;
  opt.progress = said(progress);
  OpenedPackage p(rq.package);
  return result_of(nxr::convert::import_cifti(p.w, p.sub, file, opt));
}

CommandResult create_recording(const CreateRecordingRequest& rq, const Progress& progress) {
  if (!rq.synthetic) throw Refused("say what to make: --synthetic (sine waves plus noise from a seed); a recording of a file is `import recording`");
  nxr::convert::SyntheticRecording opt;
  opt.name = rq.name.value_or("synthetic");
  opt.acquisition = rq.acquisition.value_or("rest");
  if (rq.channels) opt.n_channels = *rq.channels;
  if (rq.sfreq) opt.sfreq = *rq.sfreq;
  if (rq.duration) opt.duration = *rq.duration;
  if (rq.amplitude) opt.amplitude = *rq.amplitude;
  if (rq.noise) opt.noise = *rq.noise;
  if (rq.seed) opt.seed = *rq.seed;
  if (rq.frequency) opt.frequencies = *rq.frequency;
  opt.progress = said(progress);
  OpenedPackage p(rq.package);
  return result_of(nxr::convert::create_synthetic(p.w, p.sub, opt));
}

CommandResult import_bids(const ImportBidsRequest& rq, const Progress& progress) {
  if (!fs::is_directory(rq.from)) throw Error(code::not_found, "no such BIDS folder: " + rq.from);
  nxr::convert::BidsImport opt;
  opt.registered_by = rq.registered_by.value_or(rq.created_by);
  opt.in_place = rq.in_place;
  opt.anonymize = rq.anonymize;
  opt.participants = !rq.no_participants;
  for (const auto& s : rq.subject) opt.subjects.push_back(s.rfind("sub-", 0) == 0 ? s.substr(4) : s);
  opt.progress = said(progress);
  return result_of(nxr::convert::import_bids(rq.from, rq.dataset, opt));
}

// ---- the participants ----

CommandResult import_participants(const ParticipantsRequest& rq) {
  std::optional<fs::path> side;
  if (rq.sidecar) side = *rq.sidecar;
  const auto r = participants_import(rq.dataset, rq.tsv, side);
  return result_of(Json{{"variables", r.variables}, {"subjects", r.n_subjects}, {"values", r.n_values}});
}

CommandResult export_participants(const ParticipantsRequest& rq) {
  std::optional<fs::path> side;   // written only where the command names
  if (rq.sidecar) side = *rq.sidecar;
  participants_export(rq.dataset, rq.tsv, side);
  return result_of(Json{{"tsv", rq.tsv}});
}

}  // namespace nxr::io::task
