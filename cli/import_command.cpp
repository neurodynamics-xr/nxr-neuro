// nxr-neuro import bids: a BIDS dataset into a dataset of subject packages (nxr/convert/bids.h):
//
//   nxr-neuro import bids --from DIR --dataset PATH [--subject LABEL ...] [--in-place] [--anonymize] [--no-participants] [--registered-by WHO]
//
// nxr-neuro import fif: an MNE object file (source space, covariance, forward solution, inverse operator, events, annotations, head-to-MRI transform) into a subject package (nxr/convert/fif_objects.h):
//
//   nxr-neuro import fif --from FILE --package PATH [--kind K] [--name N] [--acquisition A] [--channels SENSORS] [--recording REC] [--surface S] [--licence T --attribution T] [--anonymize]
//   nxr-neuro import transform --from FILE-trans.fif --package PATH [--name N] [--channels SENSORS]
//
// nxr-neuro import freesurfer: a FreeSurfer subject folder (surfaces in freesurfer-tkr, c_ras to scanner RAS, named annotations, overlays); import gifti: a GIFTI surface, maps or labels.
//
//   nxr-neuro import freesurfer --from SUBJECT_DIR --package PATH [--name N] [--surfaces white pial ...] [--annotations aparc ...] [--overlays thickness ...] [--orig FILE.mgz] [--anonymize]
//   nxr-neuro import gifti --from FILE --package PATH [--name N] [--surface S] [--parcellation NAME] [--unit U] [--anonymize]
//
// nxr-neuro import nifti: a NIfTI volume into a subject package (nxr/convert/nifti_import.h).
//
//   nxr-neuro import nifti --from FILE --package PATH [--name N] [--parcellation NAME [--label-names FILE]] [--unit U] [--licence T --attribution T] [--anonymize]
//
// nxr-neuro import recording: a recording file (FIF, EDF/BDF, BrainVision, EEGLAB, CTF, KIT) copied into a subject package (docs/specs/2026-11-17-interoperability.md, 3.2).
//
//   nxr-neuro import recording --from FILE --package PATH [--name REC] [--acquisition NAME] [--format F] [--source-root DIR] [--licence T --attribution T] [--anonymize] [--in-place]
//
// The package is made first (`create package`). The recording is COPIED, block by block, never loaded whole (or, with --in-place, LINKED: the chunk service streams from the original file, whose path and SHA-256 the package records; a changed file is refused); the file's header fields are kept as the file has them unless
// --anonymize drops the identifying ones. The command parses and prints; io's importer (nxr/convert/recording.h) does the work. Progress and results are JSON lines.
#include <filesystem>

#include "cli.h"
#include "nxr/convert/bids.h"
#include "nxr/convert/fif_objects.h"
#include "nxr/convert/freesurfer_import.h"
#include "nxr/convert/gifti_import.h"
#include "nxr/convert/nifti_import.h"
#include "nxr/convert/recording.h"
#include "nxr/convert/synthetic.h"
#include "nxr/io/subject_package.h"

using namespace nxr::io;
using namespace cli;
namespace fs = std::filesystem;

namespace {
int recording_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import recording", rest, {"--from", "--package", "--name", "--acquisition", "--format", "--source-root", "--licence", "--attribution"}, {"--anonymize", "--in-place"},
                       {"--from", "--package"});
  const std::string file = a.get("--from");
  if (!fs::exists(file)) throw Error(code::not_found, "no such file: " + file);
  nxr::convert::RecordingImport opt;
  opt.format = a.maybe("--format").value_or(nxr::convert::format_of(file));
  opt.name = a.maybe("--name").value_or(fs::path(file).stem().string());
  opt.acquisition = a.maybe("--acquisition").value_or("rest");
  opt.anonymize = a.flag.count("--anonymize") > 0;
  opt.in_place = a.flag.count("--in-place") > 0;
  opt.licence = a.maybe("--licence").value_or("unknown");
  opt.attribution = a.maybe("--attribution").value_or("unknown");
  opt.redistributable = false;
  // the recorded source is relative to the root the caller names, never an absolute path
  const fs::path root = a.maybe("--source-root").value_or(fs::path(file).parent_path().string());
  opt.source = fs::relative(fs::absolute(file), fs::absolute(root.empty() ? fs::path(".") : root)).generic_string();
  opt.source_file = fs::is_directory(file) ? "" : file;   // a CTF .ds folder is not one file: no hash
  opt.progress = [](const Json& j) { emit(j); };
  auto reader = nxr::convert::open_recording(opt.format, file);
  const PackageRecord rec = SubjectPackage::open(a.get("--package")).record();
  DatasetWriter w = open_package_writer(a.get("--package"));
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  Json out = nxr::convert::import_recording(w, sub, *reader, opt);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

// import fif / import transform: an MNE object file (nxr/convert/fif_objects.h)
int fif_command(const std::string& prog, const std::vector<std::string>& rest, const std::string& fixed_kind) {
  const bool transform = !fixed_kind.empty();
  const Args a = transform ? parse(prog, rest, {"--from", "--package", "--name", "--channels"}, {}, {"--from", "--package"})
                           : parse(prog, rest, {"--from", "--package", "--kind", "--name", "--acquisition", "--channels", "--recording", "--surface", "--source-root", "--licence", "--attribution"},
                                   {"--anonymize"}, {"--from", "--package"});
  const std::string file = a.get("--from");
  if (!fs::exists(file)) throw Error(code::not_found, "no such file: " + file);
  nxr::convert::FifObjectImport opt;
  opt.kind = transform ? fixed_kind : a.maybe("--kind").value_or("");
  opt.name = a.maybe("--name").value_or("");
  opt.channels = a.maybe("--channels").value_or("");
  if (!transform) {
    opt.acquisition = a.maybe("--acquisition").value_or("rest");
    opt.recording = a.maybe("--recording").value_or("");
    opt.surface = a.maybe("--surface").value_or("");
    opt.licence = a.maybe("--licence").value_or("unknown");
    opt.attribution = a.maybe("--attribution").value_or("unknown");
    opt.anonymize = a.flag.count("--anonymize") > 0;
  }
  const fs::path root = transform ? fs::path(file).parent_path() : fs::path(a.maybe("--source-root").value_or(fs::path(file).parent_path().string()));
  opt.source = fs::relative(fs::absolute(file), fs::absolute(root.empty() ? fs::path(".") : root)).generic_string();
  opt.source_file = file;
  opt.progress = [](const Json& j) { emit(j); };
  const PackageRecord rec = SubjectPackage::open(a.get("--package")).record();
  DatasetWriter w = open_package_writer(a.get("--package"));
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  Json out = nxr::convert::import_fif_object(w, sub, file, opt);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

// import nifti: a NIfTI volume (a lattice in its voxel chart, its intensities or a named parcellation, the voxel-to-world transition)
int nifti_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import nifti", rest, {"--from", "--package", "--name", "--kind", "--parcellation", "--label-names", "--field-kind", "--unit", "--source-root", "--licence", "--attribution"},
                       {"--anonymize"}, {"--from", "--package"});
  const std::string file = a.get("--from");
  if (!fs::exists(file)) throw Error(code::not_found, "no such file: " + file);
  nxr::convert::NiftiImport opt;
  opt.name = a.maybe("--name").value_or("");
  opt.kind = a.maybe("--kind").value_or("");
  opt.parcellation = a.maybe("--parcellation").value_or("");
  opt.label_names = a.maybe("--label-names").value_or("");
  opt.field_kind = a.maybe("--field-kind").value_or("intensity");
  opt.unit = a.maybe("--unit").value_or("");
  opt.licence = a.maybe("--licence").value_or("unknown");
  opt.attribution = a.maybe("--attribution").value_or("unknown");
  opt.anonymize = a.flag.count("--anonymize") > 0;
  const fs::path root = a.maybe("--source-root").value_or(fs::path(file).parent_path().string());
  opt.source = fs::relative(fs::absolute(file), fs::absolute(root.empty() ? fs::path(".") : root)).generic_string();
  opt.source_file = file;
  opt.progress = [](const Json& j) { emit(j); };
  const PackageRecord rec = SubjectPackage::open(a.get("--package")).record();
  DatasetWriter w = open_package_writer(a.get("--package"));
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  Json out = nxr::convert::import_nifti(w, sub, file, opt);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

// import freesurfer: a FreeSurfer subject folder (surfaces in freesurfer-tkr with c_ras to scanner RAS; named annotations; overlays)
int freesurfer_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import freesurfer", rest, {"--from", "--package", "--name", "--orig", "--field-kind", "--source-root", "--licence", "--attribution"}, {"--anonymize"},
                       {"--from", "--package"}, 0, {"--surfaces", "--annotations", "--overlays"});
  const std::string dir = a.get("--from");
  if (!fs::is_directory(dir)) throw Error(code::not_found, "no such folder: " + dir);
  nxr::convert::FreeSurferImport opt;
  opt.name = a.maybe("--name").value_or("cortex");
  if (a.multi.count("--surfaces")) opt.surfaces = a.multi.at("--surfaces");
  if (a.multi.count("--annotations")) opt.annotations = a.multi.at("--annotations");
  if (a.multi.count("--overlays")) opt.overlays = a.multi.at("--overlays");
  opt.orig = a.maybe("--orig").value_or("");
  opt.field_kind = a.maybe("--field-kind").value_or("overlay");
  opt.licence = a.maybe("--licence").value_or("unknown");
  opt.attribution = a.maybe("--attribution").value_or("unknown");
  opt.anonymize = a.flag.count("--anonymize") > 0;
  const fs::path root = a.maybe("--source-root").value_or(fs::path(dir).parent_path().string());
  opt.source = fs::relative(fs::absolute(dir), fs::absolute(root.empty() ? fs::path(".") : root)).generic_string();
  opt.progress = [](const Json& j) { emit(j); };
  const PackageRecord rec = SubjectPackage::open(a.get("--package")).record();
  DatasetWriter w = open_package_writer(a.get("--package"));
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  Json out = nxr::convert::import_freesurfer_subject(w, sub, dir, opt);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

// import gifti: a surface, maps or labels from a GIFTI file
int gifti_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import gifti", rest, {"--from", "--package", "--name", "--kind", "--surface", "--parcellation", "--field-kind", "--unit", "--source-root", "--licence", "--attribution"},
                       {"--anonymize"}, {"--from", "--package"});
  const std::string file = a.get("--from");
  if (!fs::exists(file)) throw Error(code::not_found, "no such file: " + file);
  nxr::convert::GiftiImport opt;
  opt.name = a.maybe("--name").value_or("");
  opt.kind = a.maybe("--kind").value_or("");
  opt.surface = a.maybe("--surface").value_or("");
  opt.parcellation = a.maybe("--parcellation").value_or("");
  opt.field_kind = a.maybe("--field-kind").value_or("overlay");
  opt.unit = a.maybe("--unit").value_or("");
  opt.licence = a.maybe("--licence").value_or("unknown");
  opt.attribution = a.maybe("--attribution").value_or("unknown");
  opt.anonymize = a.flag.count("--anonymize") > 0;
  const fs::path root = a.maybe("--source-root").value_or(fs::path(file).parent_path().string());
  opt.source = fs::relative(fs::absolute(file), fs::absolute(root.empty() ? fs::path(".") : root)).generic_string();
  opt.source_file = file;
  opt.progress = [](const Json& j) { emit(j); };
  const PackageRecord rec = SubjectPackage::open(a.get("--package")).record();
  DatasetWriter w = open_package_writer(a.get("--package"));
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  Json out = nxr::convert::import_gifti(w, sub, file, opt);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

// create recording --synthetic: sine waves plus noise from a seed, written into a subject package through the recording importer
int synthetic_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro create recording", rest, {"--package", "--name", "--acquisition", "--channels", "--sfreq", "--duration", "--seed", "--amplitude", "--noise"}, {"--synthetic"},
                       {"--package"}, 0, {"--frequency"});
  if (!a.flag.count("--synthetic")) throw Usage("nxr-neuro create recording: error: say what to make: --synthetic (sine waves plus noise from a seed); a recording of a file is `import recording`");
  nxr::convert::SyntheticRecording opt;
  auto num = [&](const char* k, auto& into) {
    if (const auto v = a.maybe(k)) {
      try { std::size_t used = 0; const double x = std::stod(*v, &used); if (used != v->size()) throw std::invalid_argument(*v); into = static_cast<std::remove_reference_t<decltype(into)>>(x); }
      catch (const std::exception&) { throw Usage(std::string("nxr-neuro create recording: error: argument ") + k + ": '" + *v + "' is not a number"); }
    }
  };
  opt.name = a.maybe("--name").value_or("synthetic");
  opt.acquisition = a.maybe("--acquisition").value_or("rest");
  num("--channels", opt.n_channels);
  num("--sfreq", opt.sfreq);
  num("--duration", opt.duration);
  num("--amplitude", opt.amplitude);
  num("--noise", opt.noise);
  if (const auto v = a.maybe("--seed")) {
    try { opt.seed = std::stoull(*v); } catch (const std::exception&) { throw Usage("nxr-neuro create recording: error: argument --seed: '" + *v + "' is not a whole number"); }
  }
  if (const auto it = a.multi.find("--frequency"); it != a.multi.end()) {
    opt.frequencies.clear();
    for (const auto& f : it->second) {
      try { opt.frequencies.push_back(std::stod(f)); } catch (const std::exception&) { throw Usage("nxr-neuro create recording: error: argument --frequency: '" + f + "' is not a number"); }
    }
  }
  opt.progress = [](const Json& j) { emit(j); };
  const PackageRecord rec = SubjectPackage::open(a.get("--package")).record();
  DatasetWriter w = open_package_writer(a.get("--package"));
  SubjectWriter sub = SubjectWriter::open(w, rec.name);
  Json out = nxr::convert::create_synthetic(w, sub, opt);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}

// import bids: a BIDS dataset (MEG, EEG, iEEG) into a new dataset of subject packages, with the participant variables
int bids_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import bids", rest, {"--from", "--dataset", "--registered-by"}, {"--in-place", "--anonymize", "--no-participants"}, {"--from", "--dataset"}, 0, {"--subject"});
  if (!fs::is_directory(a.get("--from"))) throw Error(code::not_found, "no such BIDS folder: " + a.get("--from"));
  nxr::convert::BidsImport opt;
  opt.registered_by = a.maybe("--registered-by").value_or(stamp());
  opt.in_place = a.flag.count("--in-place") > 0;
  opt.anonymize = a.flag.count("--anonymize") > 0;
  opt.participants = a.flag.count("--no-participants") == 0;
  if (const auto it = a.multi.find("--subject"); it != a.multi.end()) for (const auto& s : it->second) opt.subjects.push_back(s.rfind("sub-", 0) == 0 ? s.substr(4) : s);
  opt.progress = [](const Json& j) { emit(j); };
  Json out = nxr::convert::import_bids(a.get("--from"), a.get("--dataset"), opt);
  Json line = Json{{"stage", "result"}};
  for (auto it = out.begin(); it != out.end(); ++it) line[it.key()] = it.value();
  emit(line);
  return 0;
}
}  // namespace

int import_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro import: error: the following arguments are required: what");
  const std::string what = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  if (what == "recording") return recording_command(rest);
  if (what == "bids") return bids_command(rest);
  if (what == "fif") return fif_command("nxr-neuro import fif", rest, "");
  if (what == "transform") return fif_command("nxr-neuro import transform", rest, "trans");
  if (what == "nifti") return nifti_command(rest);
  if (what == "freesurfer") return freesurfer_command(rest);
  if (what == "gifti") return gifti_command(rest);
  if (what == "synthetic") return synthetic_command(rest);
  throw Usage("nxr-neuro import: error: argument what: invalid choice: '" + what + "' (choose from 'recording', 'bids', 'fif', 'transform', 'nifti', 'freesurfer', 'gifti', 'synthetic')");
}
