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
// --anonymize drops the identifying ones. The command parses, calls its library function (nxr/io/task/commands.h) and prints; io's importer (nxr/convert/recording.h) does the work. Progress and results are JSON lines.
#include "cli.h"

using namespace nxr::io;
using namespace cli;
namespace task = nxr::io::task;

namespace {
// the origin options every file importer takes
task::FileOrigin origin_of(const Args& a) {
  task::FileOrigin o;
  o.source_root = a.maybe("--source-root");
  o.licence = a.maybe("--licence");
  o.attribution = a.maybe("--attribution");
  o.anonymize = a.flag.count("--anonymize") > 0;
  return o;
}

int recording_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import recording", rest, {"--from", "--package", "--name", "--acquisition", "--format", "--source-root", "--licence", "--attribution", "--montage"}, {"--anonymize", "--in-place"},
                       {"--from", "--package"});
  task::ImportRecordingRequest rq;
  rq.from = a.get("--from");
  rq.package = a.get("--package");
  rq.name = a.maybe("--name");
  rq.acquisition = a.maybe("--acquisition");
  rq.format = a.maybe("--format");
  rq.montage = a.maybe("--montage");
  rq.origin = origin_of(a);
  rq.in_place = a.flag.count("--in-place") > 0;
  emit_result(task::import_recording(rq, progress()).result);
  return 0;
}

// import fif / import transform: an MNE object file (nxr/convert/fif_objects.h)
int fif_command(const std::string& prog, const std::vector<std::string>& rest, bool transform) {
  if (transform) {
    const Args a = parse(prog, rest, {"--from", "--package", "--name", "--channels"}, {}, {"--from", "--package"});
    emit_result(task::import_transform({a.get("--from"), a.get("--package"), a.maybe("--name"), a.maybe("--channels")}, progress()).result);
    return 0;
  }
  const Args a = parse(prog, rest, {"--from", "--package", "--kind", "--name", "--acquisition", "--channels", "--recording", "--surface", "--source-root", "--licence", "--attribution"},
                       {"--anonymize"}, {"--from", "--package"});
  task::ImportFifRequest rq;
  rq.from = a.get("--from");
  rq.package = a.get("--package");
  rq.kind = a.maybe("--kind");
  rq.name = a.maybe("--name");
  rq.acquisition = a.maybe("--acquisition");
  rq.channels = a.maybe("--channels");
  rq.recording = a.maybe("--recording");
  rq.surface = a.maybe("--surface");
  rq.origin = origin_of(a);
  emit_result(task::import_fif(rq, progress()).result);
  return 0;
}

// import nifti: a NIfTI volume (a lattice in its voxel chart, its intensities or a named parcellation, the voxel-to-world transition)
int nifti_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import nifti", rest, {"--from", "--package", "--name", "--kind", "--parcellation", "--label-names", "--field-kind", "--unit", "--source-root", "--licence", "--attribution"},
                       {"--anonymize"}, {"--from", "--package"});
  task::ImportNiftiRequest rq;
  rq.from = a.get("--from");
  rq.package = a.get("--package");
  rq.name = a.maybe("--name");
  rq.kind = a.maybe("--kind");
  rq.parcellation = a.maybe("--parcellation");
  rq.label_names = a.maybe("--label-names");
  rq.field_kind = a.maybe("--field-kind");
  rq.unit = a.maybe("--unit");
  rq.origin = origin_of(a);
  emit_result(task::import_nifti(rq, progress()).result);
  return 0;
}

// import freesurfer: a FreeSurfer subject folder (surfaces in freesurfer-tkr with c_ras to scanner RAS; named annotations; overlays)
int freesurfer_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import freesurfer", rest, {"--from", "--package", "--name", "--orig", "--field-kind", "--source-root", "--licence", "--attribution"}, {"--anonymize"},
                       {"--from", "--package"}, 0, {"--surfaces", "--annotations", "--overlays"});
  task::ImportFreesurferRequest rq;
  rq.from = a.get("--from");
  rq.package = a.get("--package");
  rq.name = a.maybe("--name");
  rq.orig = a.maybe("--orig");
  rq.field_kind = a.maybe("--field-kind");
  rq.surfaces = maybe_values(a, "--surfaces");
  rq.annotations = maybe_values(a, "--annotations");
  rq.overlays = maybe_values(a, "--overlays");
  rq.origin = origin_of(a);
  emit_result(task::import_freesurfer(rq, progress()).result);
  return 0;
}

// import gifti: a surface, maps or labels from a GIFTI file
int gifti_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import gifti", rest, {"--from", "--package", "--name", "--kind", "--surface", "--parcellation", "--field-kind", "--unit", "--source-root", "--licence", "--attribution"},
                       {"--anonymize"}, {"--from", "--package"});
  task::ImportGiftiRequest rq;
  rq.from = a.get("--from");
  rq.package = a.get("--package");
  rq.name = a.maybe("--name");
  rq.kind = a.maybe("--kind");
  rq.surface = a.maybe("--surface");
  rq.parcellation = a.maybe("--parcellation");
  rq.field_kind = a.maybe("--field-kind");
  rq.unit = a.maybe("--unit");
  rq.origin = origin_of(a);
  emit_result(task::import_gifti(rq, progress()).result);
  return 0;
}

// import cifti: the cortical brain models of a dense CIFTI map on named surfaces of the package
int cifti_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import cifti", rest, {"--from", "--package", "--name", "--left", "--right", "--parcellation", "--field-kind", "--unit", "--source-root", "--licence", "--attribution"},
                       {"--anonymize"}, {"--from", "--package"});
  task::ImportCiftiRequest rq;
  rq.from = a.get("--from");
  rq.package = a.get("--package");
  rq.name = a.maybe("--name");
  rq.left = a.maybe("--left");
  rq.right = a.maybe("--right");
  rq.parcellation = a.maybe("--parcellation");
  rq.field_kind = a.maybe("--field-kind");
  rq.unit = a.maybe("--unit");
  rq.origin = origin_of(a);
  emit_result(task::import_cifti(rq, progress()).result);
  return 0;
}

// create recording --synthetic: sine waves plus noise from a seed, written into a subject package through the recording importer
int synthetic_command(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro create recording";
  const Args a = parse(prog, rest, {"--package", "--name", "--acquisition", "--channels", "--sfreq", "--duration", "--seed", "--amplitude", "--noise"}, {"--synthetic"},
                       {"--package"}, 0, {"--frequency"});
  // refused before the numbers are read, as it always was (the library refuses it too, for the faces)
  if (!a.flag.count("--synthetic")) throw Usage(prog + ": error: say what to make: --synthetic (sine waves plus noise from a seed); a recording of a file is `import recording`");
  task::CreateRecordingRequest rq;
  rq.package = a.get("--package");
  rq.synthetic = true;
  auto num = [&](const char* k) -> std::optional<double> {
    const auto v = a.maybe(k);
    if (!v) return std::nullopt;
    try { std::size_t used = 0; const double x = std::stod(*v, &used); if (used != v->size()) throw std::invalid_argument(*v); return x; }
    catch (const std::exception&) { throw Usage(prog + ": error: argument " + k + ": '" + *v + "' is not a number"); }
  };
  rq.name = a.maybe("--name");
  rq.acquisition = a.maybe("--acquisition");
  if (const auto x = num("--channels")) rq.channels = static_cast<std::int64_t>(*x);
  rq.sfreq = num("--sfreq");
  rq.duration = num("--duration");
  rq.amplitude = num("--amplitude");
  rq.noise = num("--noise");
  if (const auto v = a.maybe("--seed")) {
    try { rq.seed = std::stoull(*v); } catch (const std::exception&) { throw Usage(prog + ": error: argument --seed: '" + *v + "' is not a whole number"); }
  }
  if (const auto it = a.multi.find("--frequency"); it != a.multi.end()) {
    rq.frequency.emplace();
    for (const auto& f : it->second) {
      try { rq.frequency->push_back(std::stod(f)); } catch (const std::exception&) { throw Usage(prog + ": error: argument --frequency: '" + f + "' is not a number"); }
    }
  }
  emit_result(call(prog, [&] { return task::create_recording(rq, progress()); }).result);
  return 0;
}

// import bids: a BIDS dataset (MEG, EEG, iEEG) into a new dataset of subject packages, with the participant variables
int bids_command(const std::vector<std::string>& rest) {
  const Args a = parse("nxr-neuro import bids", rest, {"--from", "--dataset", "--registered-by"}, {"--in-place", "--anonymize", "--no-participants"}, {"--from", "--dataset"}, 0, {"--subject"});
  task::ImportBidsRequest rq;
  rq.from = a.get("--from");
  rq.dataset = a.get("--dataset");
  rq.subject = values_of(a, "--subject");
  rq.in_place = a.flag.count("--in-place") > 0;
  rq.anonymize = a.flag.count("--anonymize") > 0;
  rq.no_participants = a.flag.count("--no-participants") > 0;
  rq.registered_by = a.maybe("--registered-by");
  rq.created_by = stamp();
  emit_result(task::import_bids(rq, progress()).result);
  return 0;
}
}  // namespace

int import_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro import: error: the following arguments are required: what");
  const std::string what = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  if (what == "recording") return recording_command(rest);
  if (what == "bids") return bids_command(rest);
  if (what == "fif") return fif_command("nxr-neuro import fif", rest, false);
  if (what == "transform") return fif_command("nxr-neuro import transform", rest, true);
  if (what == "nifti") return nifti_command(rest);
  if (what == "freesurfer") return freesurfer_command(rest);
  if (what == "gifti") return gifti_command(rest);
  if (what == "cifti") return cifti_command(rest);
  if (what == "synthetic") return synthetic_command(rest);
  throw Usage("nxr-neuro import: error: argument what: invalid choice: '" + what + "' (choose from 'recording', 'bids', 'fif', 'transform', 'nifti', 'freesurfer', 'gifti', 'cifti', 'synthetic')");
}
