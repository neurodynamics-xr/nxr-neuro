#include "nxr/pipelines/meeg.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/subject_reads.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <limits>
#include <map>
#include <set>

#include "nxr/bst/frequency.h"
#include "nxr/intrinsic_surface.h"
#include "nxr/filterbank.h"
#include "nxr/intrinsic_forms.h"
#include <sstream>
#include "nxr/hodge_weights.h"
#include "nxr/opticalflow.h"
#include "nxr/ladder.h"
#include "nxr/fft.h"
#include "nxr/rotary.h"
#include "nxr/timebank.h"
#include "nxr/patterns.h"
#include "nxr/pipelines/spectral_record.h"
#include "nxr/congruence.h"
#include "nxr/wt.h"
#include "nxr/bst/inverse.h"
#include "nxr/bst/session.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/reading.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/selection_measurement.h"
#include "nxr/io/statistic.h"
#include "nxr/meeg/tensor.h"
#include "nxr/meeg/wave.h"
#include "nxr/hodge_weights.h"

#ifndef NXR_COMPUTE_VERSION
#define NXR_COMPUTE_VERSION "unknown"
#endif
#ifndef NXR_IO_VERSION
#define NXR_IO_VERSION "0.0.0"
#endif
#ifndef NXR_NEURO_VERSION
#define NXR_NEURO_VERSION "0.0.0"
#endif

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;

std::vector<std::string> imaging_channel_types() { return {"MEG", "MEG MAG", "MEG GRAD", "EEG", "SEEG", "ECOG"}; }

namespace {

[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }

std::string producer_name(const char* verb) { return std::string("nxr-neuro ") + NXR_NEURO_VERSION + " " + verb + " (nxr-io " + NXR_IO_VERSION + ")"; }

template <class T>
std::vector<T> as_vector(const std::vector<std::uint8_t>& raw, DType t) {
  const std::size_t n = raw.size() / dtype_size(t);
  std::vector<T> out(n);
  auto conv = [&](auto tag) {
    using S = decltype(tag);
    for (std::size_t i = 0; i < n; ++i) { S v; std::memcpy(&v, raw.data() + i * sizeof(S), sizeof(S)); out[i] = static_cast<T>(v); }
  };
  switch (t) {
    case DType::Float32: conv(float{}); break;
    case DType::Float64: conv(double{}); break;
    case DType::Int32: conv(std::int32_t{}); break;
    case DType::Int64: conv(std::int64_t{}); break;
    case DType::Int16: conv(std::int16_t{}); break;
    case DType::UInt8: conv(std::uint8_t{}); break;
    default: throw Error(code::unsupported, std::string("an array of type ") + dtype_to_string(t) + " is not read here");
  }
  return out;
}
template <class T>
std::vector<T> read_all(const nxr::io::Store& store, const std::string& key, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(store, key);
  if (shape) *shape = r.shape();
  return as_vector<T>(r.read(), r.dtype());
}

// one level name per element of a labelling Selection of `manifold_id` called `family` (a set's partition: 'type', 'quality'); empty when there is none
std::vector<std::string> read_labelling(SubjectWriter& sub, const Row& selection) {
  const std::string array_path = opt_text(selection, "array_path").value_or("");
  const std::string dict = opt_text(selection, "dictionary_id").value_or("");
  if (array_path.empty() || dict.empty()) return {};
  std::map<std::int64_t, std::string> names;
  names = nxr::io::selection::lookup_names(sub.db(), dict);
  // the array holds each element's member NUMBER; the member's label (the dictionary's code) is beside it
  const std::vector<std::int64_t> labels = nxr::io::selection::member_labels(sub.db(), sub.store(), text(selection, "id"));
  std::vector<std::string> out;
  for (std::int32_t n : read_all<std::int32_t>(sub.store(), array_path)) {
    const std::int64_t c = n >= 0 && static_cast<std::size_t>(n) < labels.size() ? labels[static_cast<std::size_t>(n)] : -1;
    out.push_back(names.count(c) ? names[c] : std::string());
  }
  return out;
}
std::vector<std::string> family_of(SubjectWriter& sub, const std::string& channels_id, const std::string& family) {
  const auto row = nxr::io::selection::labelling_row(sub.db(), sub.id(), family, channels_id);
  return row ? read_labelling(sub, *row) : std::vector<std::string>();
}

struct RecordingInput {
  Row field;
  std::string id, name, channels_id, time_id;
  std::optional<std::string> acquisition_id;
  std::int64_t n_channels = 0, n_samples = 0;
  double sfreq = 0, t0 = 0;
  std::vector<std::string> types, quality;              // per channel; quality from the recording's own flags, else the montage's
  std::vector<std::string> units;                       // per channel, the montage's 'unit' family (T, T/m, V ...); empty when the montage has none
};

// The unit of a power the recording's channels give: the recording's own unit, else the montage's per-channel unit when every channel has the same one;
// else empty (channels of several types, each in its own unit: the caller stores "channel unit^2" and the per-type units in the provenance, as the sensor cross-spectrum does).
// the power unit of each channel type present (type -> unit^2 [/Hz]), for a recording whose channels do not share one unit
RecordingInput read_recording(SubjectWriter& sub, const std::string& name, bool require_types = true) {   // require_types false: a recording whose channels carry no type labelling is read with none (a node that needs them says so)
  const auto f = find_field(sub.db(), sub.id(), FieldQuery{name, "recording"});
  if (!f) missing(sub.name() + ": no recording named '" + name + "'");
  RecordingInput r;
  r.field = *f;
  r.id = text(*f, "id");
  r.name = name;
  r.acquisition_id = opt_text(*f, "acquisition_id");
  const auto dims = factor_manifolds(sub.db(), text(*f, "manifold_id"));
  if (dims.size() != 2) invalid(name + ": a recording lies on channels x time");
  r.channels_id = dims[0];
  r.time_id = dims[1];
  const auto td = line_factor(sub.db(), r.time_id);
  if (!td || !(td->spacing > 0)) invalid(name + ": its time line has no sampling interval");
  r.sfreq = 1.0 / td->spacing;
  r.t0 = td->origin.value_or(0.0);
  ArrayReader ar = ArrayReader::open(sub.store(), text(*f, "path"));
  if (ar.shape().size() != 2) invalid(name + ": the recording array is not channels x time");
  r.n_channels = ar.shape()[0];
  r.n_samples = ar.shape()[1];
  r.types = family_of(sub, r.channels_id, "type");
  if (static_cast<std::int64_t>(r.types.size()) != r.n_channels) {
    if (require_types) invalid(name + ": its channels have no 'type' labelling of the right size");
    r.types.clear();
  }
  // the recording's own flags, else the montage's quality; none: every channel good
  if (const auto fl = nxr::io::selection::row_over_field(sub.db(), sub.id(), r.id, name + "_flags"))
    r.quality = read_labelling(sub, *fl);
  if (static_cast<std::int64_t>(r.quality.size()) != r.n_channels) r.quality = family_of(sub, r.channels_id, "quality");
  r.units = family_of(sub, r.channels_id, "unit");
  if (static_cast<std::int64_t>(r.units.size()) != r.n_channels) r.units.clear();
  if (static_cast<std::int64_t>(r.quality.size()) != r.n_channels) r.quality.assign(static_cast<std::size_t>(r.n_channels), "good");
  return r;
}

}  // namespace

// the type of each channel of a channels manifold (its 'type' labelling); empty when there is none
std::vector<std::string> channel_types_of(SubjectWriter& sub, const std::string& channels_id) { return family_of(sub, channels_id, "type"); }

// THE SENSOR GEOMETRY of a channels manifold (what a head model reads; a node is handed it): positions [M x 3], the 'type' of each channel, the orientations
// `<name>_orientation` when stored, and the coils `<name>_coils` (positions, `<name>_coils_orientation`) with their weights `<name>_coil_weights` (CSR [channels x coils])
SensorGeometry sensor_geometry(SubjectWriter& sub, const std::string& channels_id) {
  SensorGeometry g;
  const auto chm = manifold_with_geometry(sub.db(), channels_id);
  if (!chm) missing("channels manifold " + channels_id + ": not among the subject's rows");
  g.id = channels_id; g.name = text(*chm, "name"); g.cs_id = opt_text(*chm, "cs_id").value_or("");
  g.types = family_of(sub, channels_id, "type");
  std::vector<std::int64_t> sh;
  g.positions = read_all<double>(sub.store(), text(*chm, "positions_path"), &sh);
  const std::int64_t M = sh.empty() ? 0 : sh[0];
  if (sh.size() != 2 || sh[1] != 3 || static_cast<std::int64_t>(g.types.size()) != M) invalid(g.name + ": its positions are not [channels x 3] with a 'type' for each channel");
  auto vec3 = [&](const std::string& manifold_id, const std::string& name) -> std::optional<std::vector<double>> {
    const auto f = find_field(sub.db(), sub.id(), FieldQuery{name, std::nullopt, manifold_id});
    if (!f) return std::nullopt;
    std::vector<std::int64_t> s2;
    std::vector<double> d = read_all<double>(sub.store(), text(*f, "path"), &s2);
    if (s2.size() != 2 || s2[1] != 3) invalid(name + ": not [n x 3]");
    return d;
  };
  g.orientation = vec3(channels_id, g.name + "_orientation");
  if (const auto cm = manifold_with_geometry_named(sub.db(), sub.id(), g.name + "_coils"))
    if (const auto wop = find_operator(sub.db(), sub.id(), "coil weights", g.name + "_coil_weights")) {
      std::vector<std::int64_t> s3;
      g.coil_positions = read_all<double>(sub.store(), text(*cm, "positions_path"), &s3);
      const auto co = vec3(text(*cm, "id"), g.name + "_coils_orientation");
      g.coil_orientation = co ? *co : std::vector<double>(g.coil_positions.size(), std::nan(""));
      g.indptr = read_all<std::int32_t>(sub.store(), text(*wop, "path") + "/indptr");
      g.indices = read_all<std::int32_t>(sub.store(), text(*wop, "path") + "/indices");
      g.weights = read_all<float>(sub.store(), text(*wop, "path") + "/data");
      if (static_cast<std::int64_t>(g.indptr.size()) != M + 1) invalid(g.name + "_coil_weights: not [channels x coils]");
      g.coils = true;
    }
  return g;
}

// the length unit of a manifold's coordinate system ("" when it records none)
std::string coordinate_unit_of(SubjectWriter& sub, const std::string& manifold_id) {
  return manifold_coordinate_unit(sub.db(), manifold_id);
}

// the coordinate system a manifold's geometry is in ("" when it records none)
std::string coordinate_system_of(SubjectWriter& sub, const std::string& manifold_id) {
  return manifold_coordinate_system(sub.db(), manifold_id);
}

// what a node may know of a recording's channels (a fact on its port type, not a store read): their types, quality (the recording's flags, else the montage's) and units
Json channel_facts(SubjectWriter& sub, const std::string& recording) {
  const RecordingInput r = read_recording(sub, recording, false);
  Json f = Json{{"quality", r.quality}, {"units", r.units}, {"n_channels", r.n_channels}};
  if (!r.types.empty()) f["types"] = r.types;                              // a recording whose channels carry no type labelling hands over none: the nodes that need types refuse
  return f;
}

namespace {
std::string acquisition_label(SubjectWriter& sub, const std::optional<std::string>& id) {
  if (!id) return "";
  return nxr::io::acquisition_label_of(sub.db(), *id);
}

// THE CHUNK SERVICE's side of a pipeline (nxr/io/chunk_loop.h): a pipeline DECLARES what it reads (the recording, its rows, its span) and folds each chunk
// handed to it; it never opens a recording's array or loops over its samples.
nxr::io::Span span_of(const RecordingInput& rec, const std::optional<std::string>& chunks, const std::optional<double>& start, const std::optional<double>& stop, int level = 0) {
  return nxr::pipelines::span_of(rec.field, rec.n_samples, rec.sfreq, rec.t0, chunks, start, stop, level);   // the one door (reading.h)
}
// THE WELCH SEGMENT (the time-tile amendment of docs/specs/2026-11-13-cycle-pattern.md, §3). Under the default set `tiles` a segment is a TILE: 2^k samples of
// the recording's line, k = `level`, or log2 of `samples` (a power of two), or round(log2(seconds · fs)) (seconds snap to the nearest level, as constantq's
// anchorsFor rounds; default 4 s: level 12 at 1200 Hz), and the hop must be a tile too (overlap 0, ½, ¾, ...); the span then snaps to level k, so segments
// start on the line's grid. Every other set keeps its reference's rule (brainstorm: round(seconds · fs) from the span's start; nxr-chain: even). Returns the
// level the span snaps to (0 for the other sets: no snap).
int welch_segment(const std::string& who, const std::string& set, const std::optional<std::int64_t>& samples, const std::optional<double>& seconds, const std::optional<int>& level,
                  double sfreq, nxr::welch::WelchOptions& wo) {
  namespace cq = nxr::io::constantq;
  if (set != "tiles") {
    if (level) invalid(who + ": --segment-level applies to --set tiles (the " + set + " set states its segment in seconds or samples)");
    if (samples) wo.segmentSamples = static_cast<std::size_t>(*samples);
    else if (seconds) wo.segmentSamples = nxr::welch::segmentSamplesFor(*seconds, sfreq, set == "nxr-chain");
    return 0;
  }
  if (level && samples) invalid(who + ": --segment-level and --segment-samples are alternatives");
  int k = 0;
  if (level) k = *level;
  else if (samples) {
    k = cq::levelOfWidth(*samples);
    if (k < 0) invalid(who + ": under --set tiles a segment is a tile of 2^k samples (got " + std::to_string(*samples) + "): give --segment-level k, or --set brainstorm|scipy|nxr-chain");
  } else {
    if (seconds && !(*seconds > 0)) invalid(who + ": --segment-seconds is a positive duration");
    k = cq::levelOf(seconds.value_or(4.0), sfreq);
  }
  if (k < 1 || k > 24) invalid(who + ": a segment's tile level is 1 to 24 (got " + std::to_string(k) + ")");
  const std::int64_t n = std::int64_t(1) << k, hop = n - static_cast<std::int64_t>(std::floor(static_cast<double>(n) * wo.overlap));
  if (cq::levelOfWidth(hop) < 0)
    invalid(who + ": under --set tiles the hop must be a tile too (" + std::to_string(hop) + " samples at overlap " + std::to_string(wo.overlap) + "): overlap 0, 0.5, 0.75, 0.875, ...");
  wo.segmentSamples = static_cast<std::size_t>(n);
  return k;
}
nxr::io::ChunkSource source_of(SubjectWriter& sub, const RecordingInput& rec, std::vector<std::int32_t> rows, const std::optional<std::vector<std::pair<double, double>>>& bad) {
  return nxr::pipelines::source_of(sub, rec.field, rec.sfreq, rec.t0, std::move(rows), bad);
}
// a chunk handed over, as the matrix the computation takes [rows x n]
Eigen::MatrixXd matrix_of(const nxr::io::Chunk& c, std::int64_t rows) {
  return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(c.data.data(), rows, c.n);
}

}  // namespace

// ── the noise covariance ─────────────────────────────────────────────────────────────────────────────────────────

// ── the Welch PSD ────────────────────────────────────────────────────────────────────────────────────────────────

namespace {

const char* window_name(nxr::welch::Window w) {
  using nxr::welch::Window;
  return w == Window::Hann ? "hann" : w == Window::Hamming ? "hamming" : w == Window::Blackman ? "blackman" : "rectangular";
}
const char* detrend_name(nxr::welch::Detrend d) {
  using nxr::welch::Detrend;
  return d == Detrend::None ? "none" : d == Detrend::Constant ? "constant" : "linear";
}
const char* scaling_name(nxr::welch::Scaling s) {
  using nxr::welch::Scaling;
  return s == Scaling::Density ? "density" : s == Scaling::Spectrum ? "spectrum" : "chain";
}

// The bad spans [start, stop) in seconds of the recording's `<name>_bad_segments` Selection (none: no such Selection)
std::optional<std::vector<std::pair<double, double>>> bad_spans(SubjectWriter& sub, const RecordingInput& rec) {
  const auto sel = nxr::io::selection::named_row(sub.db(), sub.id(), rec.name + "_bad_segments", "spans");
  if (!sel) return std::nullopt;
  return nxr::io::selection::span_bounds(sub.db(), text(*sel, "id"));
}


// what the recording's provenance says about projectors (nxr 1.13 / D168): the entries as recorded, and which are in the samples
Json projector_record(const Row& field) {
  Json out = Json{{"projectors", Json::array()}, {"applied_in_samples", Json::array()}, {"not_applied", Json::array()}, {"ctf_compensation", Json()}};
  const Json pj = Json::parse(opt_text(field, "producer_json").value_or("{}"), nullptr, false);
  if (!pj.is_object()) return out;
  if (pj.contains("projectors") && pj["projectors"].is_array()) {
    out["projectors"] = pj["projectors"];
    for (const Json& p : pj["projectors"]) {
      const int status = p.value("status", 0);
      const std::string who = p.value("operator", std::string());
      (status == 1 || status == 2 ? out["applied_in_samples"] : out["not_applied"]).push_back(who);
    }
  }
  if (pj.contains("ctf_compensation")) out["ctf_compensation"] = pj["ctf_compensation"];
  return out;
}

}  // namespace

namespace {
// THE STATISTICS' COMMON PIECES (docs/specs/2026-11-13-statistics.md): where a statistic of a recording lives, and the Welch segment partition a
// Welch statistic is computed from (its segments as a closed-form spans Selection on the time line: overlapping when the hop is under the width)
std::int64_t welch_hop(const nxr::welch::WelchOptions& wo) {
  const std::int64_t n = static_cast<std::int64_t>(wo.segmentSamples);
  return n - static_cast<std::int64_t>(std::floor(static_cast<double>(n) * wo.overlap));
}
std::string statistic_path(const RecordingInput& rec, const std::string& name) { return nxr::io::text(rec.field, "path") + "_statistics/" + nxr::io::safe_name(name); }
std::string segments_path(SubjectWriter& sub, const RecordingInput& rec, const std::string& stem) { return nxr::io::node_path(sub.db(), rec.time_id, stem + "_segments"); }
std::string welch_segments(SubjectWriter& sub, const RecordingInput& rec, const std::string& stem, const std::string& acq, std::int64_t s0, std::int64_t s1,
                           const nxr::welch::WelchOptions& wo, std::int64_t segments, std::int64_t skipped) {
  const std::int64_t n_seg = static_cast<std::int64_t>(wo.segmentSamples), hop = welch_hop(wo);
  // the segments are a closed-form tiling of the recording's time line, stored by the selection store (it writes the tiling into the parameters)
  nxr::io::selection::Description tiles;
  tiles.type = nxr::io::selection::Type::set;
  tiles.unit = "s";
  tiles.tiling = nxr::io::selection::Tiling{rec.t0 + static_cast<double>(s0) / rec.sfreq, static_cast<double>(n_seg) / rec.sfreq, static_cast<double>(hop) / rec.sfreq, segments + skipped};
  nxr::io::selection::Placement at;
  at.name = stem + "_segments"; at.path = segments_path(sub, rec, stem); at.manifold_id = rec.time_id;
  if (!acq.empty()) at.acquisition = acq;
  at.cell = "1"; at.of_field_id = rec.id;
  at.cols.set("n_elements", rec.n_samples);
  at.description = "the " + std::to_string(segments + skipped) + " Welch segments of " + rec.name + " behind " + stem + " (" + std::to_string(n_seg) + " samples, a new one every " +
                   std::to_string(hop) + "; " + std::to_string(skipped) + " left out for a bad sample)";
  at.params = Json{{"end", rec.t0 + static_cast<double>(s1) / rec.sfreq}, {"segments_skipped", skipped}};
  return nxr::io::selection::save(sub, tiles, at).id;
}
// every Welch option, as a statistic's options record them
Json welch_options(const nxr::welch::WelchOptions& wo, const std::string& option_set, bool exclude_bad) {
  return Json{{"option_set", option_set}, {"segment_samples", static_cast<std::int64_t>(wo.segmentSamples)}, {"hop_samples", welch_hop(wo)}, {"overlap", wo.overlap},
              {"window", window_name(wo.window)}, {"window_form", wo.periodic ? "periodic" : "symmetric"}, {"detrend", detrend_name(wo.detrend)},
              {"scaling", scaling_name(wo.scaling)}, {"exclude_bad_segments", exclude_bad}};
}
}  // namespace

namespace internal {
std::string statistic_path_of(SubjectWriter& sub, const std::string& recording, const std::string& name) { return statistic_path(read_recording(sub, recording), name); }
std::string segments_path_of(SubjectWriter& sub, const std::string& recording, const std::string& stem) { return segments_path(sub, read_recording(sub, recording), stem); }
std::string welch_segments_of(SubjectWriter& sub, const std::string& recording, const std::string& stem, std::int64_t s0, std::int64_t s1, const nxr::welch::WelchOptions& wo,
                              std::int64_t segments, std::int64_t skipped) {
  const RecordingInput rec = read_recording(sub, recording);
  return welch_segments(sub, rec, stem, acquisition_label(sub, rec.acquisition_id), s0, s1, wo, segments, skipped);
}
Json welch_options_of(const nxr::welch::WelchOptions& wo, const std::string& option_set, bool exclude_bad) { return welch_options(wo, option_set, exclude_bad); }
}  // namespace internal

// ── the sensor cross-spectrum: its readers (the definitions `crossspectrum` and `crossspectrum_analytic` make it) ──────────────────

namespace {
// A STORED WELCH CROSS-SPECTRUM, as a statistic (2026-11-13: cross_spectral_density · cross_spectrum by welch, axes channel x channel, value_type complex) or, made
// before, an Operator of kind `cross-spectral density` (still read: no migration): its row, the channels in order, their manifold, its array, its producer
struct CrossSpectrumNode {
  Row row;
  std::vector<std::int32_t> channels;
  std::string channels_manifold, path;
  Json producer;
  std::optional<std::vector<double>> values;   // a band sliced from a statistic with a band axis: its values (the band's shape is the old per-band statistic's)
  std::vector<std::int64_t> shape;
};
// the values of a node: its array, or the band sliced from it
std::vector<double> node_values(SubjectWriter& sub, const CrossSpectrumNode& n, std::vector<std::int64_t>* shape) {
  if (n.values) { *shape = n.shape; return *n.values; }
  return read_all<double>(sub.store(), n.path, shape);
}
// A BAND OF A STATISTIC WITH A BAND AXIS (the definitions' form: `<stem>` with axes band x ...): found by the old per-band name `<stem>_<band>` + `suffix`. The band's slice,
// its channels (the axis after the band's, or `channel_axis`), and the producer with the band's name, edges and recording added as the per-band statistics recorded them.
std::optional<CrossSpectrumNode> band_slice(SubjectWriter& sub, const std::string& name, const std::string& method, const std::string& measure_prefix, const std::string& suffix,
                                            std::size_t channel_axis) {
  for (const Row& r : nxr::io::statistics_of(sub.db(), sub.id())) {
    if (text(r, "method") != method || text(r, "measure").rfind(measure_prefix, 0) != 0) continue;
    const Json axes = Json::parse(text(r, "axes_json"), nullptr, false);
    if (!axes.is_array() || axes.empty() || axes[0].value("name", std::string()) != "band" || !axes[0].contains("values")) continue;
    const Json& bands = axes[0]["values"];
    for (std::size_t b = 0; b < bands.size(); ++b) {
      if (nxr::bst::band_field_name(text(r, "name"), bands[b].value("name", std::string())) + suffix != name) continue;
      const auto sel = sub.db().read("selection", axes[channel_axis].value("selection_id", std::string()));
      if (!sel || is_null(*sel, "array_path")) invalid(name + ": its channels (the channel axis's Selection) are not stored");
      CrossSpectrumNode n{r, read_all<std::int32_t>(sub.store(), text(*sel, "array_path")), text(*sel, "manifold_id"), text(r, "path"),
                          Json::parse(opt_text(r, "producer_json").value_or("{}"), nullptr, false), std::nullopt, {}};
      std::vector<std::int64_t> shape;
      const std::vector<double> all = read_all<double>(sub.store(), text(r, "path"), &shape);
      if (shape.empty() || shape[0] != static_cast<std::int64_t>(bands.size())) invalid(text(r, "name") + ": its array's first axis is not its bands");
      const std::size_t per = all.size() / bands.size();
      n.values = std::vector<double>(all.begin() + static_cast<std::ptrdiff_t>(b * per), all.begin() + static_cast<std::ptrdiff_t>((b + 1) * per));
      n.shape.assign(shape.begin() + 1, shape.end());
      if (!n.producer.is_object()) n.producer = Json::object();
      n.producer["band_name"] = bands[b].value("name", std::string());
      n.producer["band_lo_hz"] = bands[b].value("lo_hz", 0.0);
      n.producer["band_hi_hz"] = bands[b].value("hi_hz", 0.0);
      if (bands[b].contains("window_samples")) n.producer["window_samples"] = bands[b]["window_samples"];
      if (bands[b].contains("spectral_scaling")) n.producer["spectral_scaling"] = bands[b]["spectral_scaling"];   // the band's own scaling record (its bandwidth)
      if (!is_null(r, "of_field_id"))
        if (const auto f = sub.db().read("field", text(r, "of_field_id"))) n.producer["recording"] = text(*f, "name");
      return n;
    }
  }
  return std::nullopt;
}
std::optional<CrossSpectrumNode> find_cross_spectrum(SubjectWriter& sub, const std::string& name) {
  for (const Row& r : nxr::io::statistics_of(sub.db(), sub.id())) {
    if (text(r, "name") != name || text(r, "method") != "welch" || text(r, "measure").rfind("cross_spectr", 0) != 0) continue;
    const Json axes = Json::parse(text(r, "axes_json"));
    const auto sel = sub.db().read("selection", axes[0].value("selection_id", std::string()));
    if (!sel || is_null(*sel, "array_path")) invalid(name + ": its channels (the channel axis's Selection) are not stored");
    return CrossSpectrumNode{r, read_all<std::int32_t>(sub.store(), text(*sel, "array_path")), text(*sel, "manifold_id"), text(r, "path"),
                             Json::parse(opt_text(r, "producer_json").value_or("{}"), nullptr, false), std::nullopt, {}};
  }
  if (auto n = band_slice(sub, name, "welch", "cross_spectr", "", 1)) return n;
  const auto op = find_operator(sub.db(), sub.id(), "cross-spectral density", name);
  if (!op) return std::nullopt;
  const auto sel = is_null(*op, "from_selection_id") ? std::nullopt : sub.db().read("selection", text(*op, "from_selection_id"));
  if (!sel || is_null(*sel, "array_path")) invalid(name + ": its channels (from_selection) are not stored");
  return CrossSpectrumNode{*op, read_all<std::int32_t>(sub.store(), text(*sel, "array_path")), text(*op, "from_manifold_id"), text(*op, "path"),
                           Json::parse(opt_text(*op, "producer_json").value_or("{}"), nullptr, false), std::nullopt, {}};
}
}  // namespace

StoredCrossSpectrum crossspectrum_read(SubjectWriter& sub, const std::string& name, const std::optional<std::vector<std::int32_t>>& channels, const std::string& scaling) {
  const auto node = find_cross_spectrum(sub, name);
  if (!node) missing(sub.name() + ": no cross-spectral density named '" + name + "'");
  const std::vector<std::int32_t>& have = node->channels;
  const std::int64_t M = static_cast<std::int64_t>(have.size());
  std::vector<std::int64_t> shape;
  const std::vector<double> c = node_values(sub, *node, &shape);
  if (shape.size() != 3 || shape[0] != M || shape[1] != M || shape[2] != 2) invalid(name + ": its array is not [" + std::to_string(M) + " x " + std::to_string(M) + " x 2]");
  StoredCrossSpectrum out;
  out.producer = node->producer;
  std::vector<std::int64_t> at;                                 // the rows of C, in the order wanted
  if (channels) {
    std::map<std::int32_t, std::int64_t> where;
    for (std::int64_t i = 0; i < M; ++i) where[have[static_cast<std::size_t>(i)]] = i;
    for (std::int32_t ch : *channels) {
      const auto it = where.find(ch);
      if (it == where.end()) invalid(name + " does not cover channel " + std::to_string(ch) + ": compute it on these channels (`crossspectrum compute --channels-of`)");
      at.push_back(it->second);
    }
    out.channels = *channels;
  } else {
    for (std::int64_t i = 0; i < M; ++i) at.push_back(i);
    out.channels = have;
  }
  const std::int64_t m = static_cast<std::int64_t>(at.size());
  out.S.resize(m, m);
  out.Sq.resize(m, m);
  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < m; ++j) {
      const std::size_t k = static_cast<std::size_t>(2 * (at[static_cast<std::size_t>(i)] * M + at[static_cast<std::size_t>(j)]));
      out.S(i, j) = c[k];
      out.Sq(i, j) = c[k + 1];
    }
  // the scaling chosen at read (a factor: a cross-spectrum takes only the linear targets)
  nxr::spectral::Target t;
  try { t = nxr::spectral::parseTarget(scaling); } catch (const std::invalid_argument& e) { invalid(e.what()); }
  const auto rec = spectral_record(out.producer);
  if (t != nxr::spectral::Target::Stored) {
    if (!rec) invalid(name + ": its provenance has no spectral scaling record (computed before 2026-11-13, or nxr's chain scaling): read it as stored, or --replace it");
    if (!nxr::spectral::isLinear(t)) invalid(name + ": a cross-spectrum is converted only to density or power (" + scaling + " applies to a real power)");
    double f = 0;
    try { f = nxr::spectral::linearFactor(rec->scaling, t); } catch (const std::invalid_argument& e) { invalid(name + ": " + e.what()); }
    out.S *= f;
    out.Sq *= f;
  }
  out.scaling = scaling;
  out.unit = rec ? nxr::spectral::unitOf(rec->power_unit, rec->scaling, t) : out.producer.value("units", std::string());
  return out;
}

namespace {
// THE BINS OF ONE BAND from the definitions' form (`<stem>_bins`, axes frequency x channel x channel, its frequency axis naming each band's bins), by the old name
// `<stem>_<band>_bins`, rearranged as the per-band statistic was: [channels x channels x frequency x 2], its frequency axis the band's bins
std::optional<nxr::io::StoredStatistic> bins_slice(SubjectWriter& sub, const std::string& name) {
  for (const Row& r : nxr::io::statistics_of(sub.db(), sub.id())) {
    const std::string sn = text(r, "name");
    if (text(r, "method") != "welch" || text(r, "measure").rfind("cross_spectr", 0) != 0 || sn.size() < 5 || sn.compare(sn.size() - 5, 5, "_bins") != 0) continue;
    const Json axes = Json::parse(text(r, "axes_json"), nullptr, false);
    if (!axes.is_array() || axes.size() != 3 || axes[0].value("name", std::string()) != "frequency" || !axes[0].contains("bands")) continue;
    const std::string stem = sn.substr(0, sn.size() - 5);
    for (const auto& band : axes[0]["bands"]) {
      if (nxr::bst::band_field_name(stem, band.value("name", std::string()) + "_bins") != name) continue;
      nxr::io::StoredStatistic st = nxr::io::read_statistic(sub, sn);
      const std::int64_t F = st.shape[0], M = st.shape[1], first = band["first"].get<std::int64_t>(), n = band["count"].get<std::int64_t>();
      std::vector<double> v(static_cast<std::size_t>(M * M * n * 2));
      for (std::int64_t k = 0; k < n; ++k)
        for (std::int64_t i = 0; i < M; ++i)
          for (std::int64_t j = 0; j < M; ++j)
            for (int c = 0; c < 2; ++c) v[static_cast<std::size_t>(((i * M + j) * n + k) * 2 + c)] = st.values[static_cast<std::size_t>((((first + k) * M + i) * M + j) * 2 + c)];
      (void)F;
      Json hz = Json::array();
      for (std::int64_t k = 0; k < n; ++k) hz.push_back(axes[0]["values"][static_cast<std::size_t>(first + k)]);
      st.values = std::move(v);
      st.shape = {M, M, n, 2};
      st.axes = Json::array({axes[1], axes[2], Json{{"name", "frequency"}, {"size", n}, {"unit", "Hz"}, {"values", hz}}});
      return st;
    }
  }
  return std::nullopt;
}
}  // namespace

StoredCrossSpectrumBins crossspectrum_bins_read(SubjectWriter& sub, const std::string& name, const std::optional<std::vector<std::int32_t>>& channels) {
  const auto sliced = bins_slice(sub, name);
  const nxr::io::StoredStatistic st = sliced ? *sliced : nxr::io::read_statistic(sub, name);
  if (text(st.row, "method") != "welch" || text(st.row, "measure").rfind("cross_spectr", 0) != 0 || st.axes.size() != 3 || st.axes[2].value("name", std::string()) != "frequency" || !st.axes[2].contains("values"))
    invalid(name + " is not a band's cross-spectrum bin by bin (`crossspectrum compute --resolve bins`, `<stem>_<band>_bins`)");
  const auto sel = sub.db().read("selection", st.axes[0].value("selection_id", std::string()));
  if (!sel || is_null(*sel, "array_path")) invalid(name + ": its channels (the channel axis's Selection) are not stored");
  const std::vector<std::int32_t> have = read_all<std::int32_t>(sub.store(), text(*sel, "array_path"));
  const std::int64_t M = static_cast<std::int64_t>(have.size()), F = static_cast<std::int64_t>(st.axes[2]["values"].size());
  if (st.shape != std::vector<std::int64_t>{M, M, F, 2}) invalid(name + ": its array is not [channels x channels x frequency x 2]");
  StoredCrossSpectrumBins out;
  out.producer = st.producer;
  out.unit = opt_text(st.row, "unit").value_or("");
  for (const auto& v : st.axes[2]["values"]) out.frequencies.push_back(v.get<double>());
  std::vector<std::int64_t> at;
  if (channels) {
    std::map<std::int32_t, std::int64_t> where;
    for (std::int64_t i = 0; i < M; ++i) where[have[static_cast<std::size_t>(i)]] = i;
    for (std::int32_t ch : *channels) {
      const auto it = where.find(ch);
      if (it == where.end()) invalid(name + " does not cover channel " + std::to_string(ch));
      at.push_back(it->second);
    }
    out.channels = *channels;
  } else {
    for (std::int64_t i = 0; i < M; ++i) at.push_back(i);
    out.channels = have;
  }
  const std::int64_t m = static_cast<std::int64_t>(at.size());
  for (std::int64_t k = 0; k < F; ++k) {
    Eigen::MatrixXcd c(m, m);
    for (std::int64_t i = 0; i < m; ++i)
      for (std::int64_t j = 0; j < m; ++j) {
        const std::size_t q = static_cast<std::size_t>(((at[static_cast<std::size_t>(i)] * M + at[static_cast<std::size_t>(j)]) * F + k) * 2);
        c(i, j) = std::complex<double>(st.values[q], st.values[q + 1]);
      }
    out.C.push_back(std::move(c));
  }
  return out;
}

namespace {
// A STORED ANALYTIC CROSS-SPECTRUM SERIES, as a statistic (2026-11-13: cross_spectrum_sum by analytic, axes window x channel x channel, value_type complex)
// or, made before, an Operator of kind `cross-spectrum series` (still read: no migration): its row, the channels in order, their manifold, its array, its producer
std::optional<CrossSpectrumNode> find_cross_spectrum_series(SubjectWriter& sub, const std::string& name) {
  for (const Row& r : nxr::io::statistics_of(sub.db(), sub.id())) {
    if (text(r, "name") != name || text(r, "method") != "analytic" || text(r, "measure") != "cross_spectrum_sum") continue;
    const Json axes = Json::parse(text(r, "axes_json"));
    const auto sel = sub.db().read("selection", axes[1].value("selection_id", std::string()));
    if (!sel || is_null(*sel, "array_path")) invalid(name + ": its channels (the channel axis's Selection) are not stored");
    return CrossSpectrumNode{r, read_all<std::int32_t>(sub.store(), text(*sel, "array_path")), text(*sel, "manifold_id"), text(r, "path"),
                             Json::parse(opt_text(r, "producer_json").value_or("{}"), nullptr, false), std::nullopt, {}};
  }
  if (auto n = band_slice(sub, name, "analytic", "cross_spectrum_sum", "_windows", 2)) return n;
  const auto op = find_operator(sub.db(), sub.id(), "cross-spectrum series", name);
  if (!op) return std::nullopt;
  const auto sel = is_null(*op, "from_selection_id") ? std::nullopt : sub.db().read("selection", text(*op, "from_selection_id"));
  if (!sel || is_null(*sel, "array_path")) invalid(name + ": its channels (from_selection) are not stored");
  return CrossSpectrumNode{*op, read_all<std::int32_t>(sub.store(), text(*sel, "array_path")), text(*op, "from_manifold_id"), text(*op, "path"),
                           Json::parse(opt_text(*op, "producer_json").value_or("{}"), nullptr, false), std::nullopt, {}};
}
}  // namespace

CrossSpectrumWindows crossspectrum_windows_read(SubjectWriter& sub, const std::string& name, const std::optional<std::vector<std::int32_t>>& channels, const std::string& scaling,
                                                std::optional<int> level) {
  const auto node = find_cross_spectrum_series(sub, name);
  if (!node) missing(sub.name() + ": no cross-spectrum series named '" + name + "' (crossspectrum compute --estimator analytic)");
  const std::vector<std::int32_t>& have = node->channels;
  const std::int64_t M = static_cast<std::int64_t>(have.size());
  std::vector<std::int64_t> shape;
  const std::vector<double> c = node_values(sub, *node, &shape);
  if (shape.size() != 4 || shape[1] != M || shape[2] != M || shape[3] != 2) invalid(name + ": its array is not [windows x " + std::to_string(M) + " x " + std::to_string(M) + " x 2]");
  CrossSpectrumWindows out;
  out.producer = node->producer;
  std::vector<std::int64_t> at;
  if (channels) {
    std::map<std::int32_t, std::int64_t> where;
    for (std::int64_t i = 0; i < M; ++i) where[have[static_cast<std::size_t>(i)]] = i;
    for (std::int32_t ch : *channels) {
      const auto it = where.find(ch);
      if (it == where.end()) invalid(name + " does not cover channel " + std::to_string(ch) + ": compute it on these channels (`crossspectrum compute --channels-of`)");
      at.push_back(it->second);
    }
    out.channels = *channels;
  } else {
    for (std::int64_t i = 0; i < M; ++i) at.push_back(i);
    out.channels = have;
  }
  const std::int64_t m = static_cast<std::int64_t>(at.size()), K = shape[0];
  const std::int64_t win = out.producer.is_object() ? out.producer.value("window_samples", std::int64_t(0)) : 0;
  const Json idx = out.producer.is_object() && out.producer.contains("window_index") ? out.producer["window_index"] : Json();
  for (std::int64_t k = 0; k < K; ++k) {
    Eigen::MatrixXcd S(m, m);
    for (std::int64_t i = 0; i < m; ++i)
      for (std::int64_t j = 0; j < m; ++j) {
        const std::size_t p = static_cast<std::size_t>(2 * ((k * M + at[static_cast<std::size_t>(i)]) * M + at[static_cast<std::size_t>(j)]));
        S(i, j) = {c[p], c[p + 1]};
      }
    out.sum.push_back(std::move(S));
    out.count.push_back(win);
    out.index.push_back(idx.is_array() && static_cast<std::int64_t>(idx.size()) == K ? idx[static_cast<std::size_t>(k)].get<std::int64_t>() : k);
  }
  out.level = out.producer.is_object() ? out.producer.value("window_level", -1) : -1;
  // THE ROLL-UP to a coarser level (the Studio's envelope, D96): window M of `level` is the sum of the stored tiles m with m >> shift = M, kept only when all
  // 2^shift of them are stored
  if (level && *level != out.level) {
    if (out.level < 0) invalid(name + ": stored before the time tiles (no window_level): its windows are not tiles, so they roll up to no level; --replace it");
    if (*level < out.level) invalid(name + ": stored at level " + std::to_string(out.level) + "; a finer level (" + std::to_string(*level) + ") cannot be read from it");
    if (*level > 40) invalid(name + ": a tile level is at most 40");
    const int shift = *level - out.level;
    CrossSpectrumWindows up;
    up.channels = out.channels; up.producer = out.producer; up.level = *level;
    for (std::size_t k = 0; k < out.sum.size();) {
      const std::int64_t M2 = out.index[k] >> shift;
      std::size_t e = k;
      Eigen::MatrixXcd S = Eigen::MatrixXcd::Zero(m, m);
      std::int64_t n = 0;
      while (e < out.sum.size() && (out.index[e] >> shift) == M2) { S += out.sum[e]; n += out.count[e]; ++e; }
      if (static_cast<std::int64_t>(e - k) == (std::int64_t(1) << shift)) { up.sum.push_back(std::move(S)); up.count.push_back(n); up.index.push_back(M2); }
      k = e;
    }
    if (up.sum.empty()) invalid(name + ": no whole window of level " + std::to_string(*level) + " (all " + std::to_string(std::int64_t(1) << shift) + " of its tiles stored)");
    out = std::move(up);
  }
  // the scaling chosen at read: each window's value from its sum (the band power sum / (2 count)), then the factor
  nxr::spectral::Target t;
  try { t = nxr::spectral::parseTarget(scaling); } catch (const std::invalid_argument& e) { invalid(e.what()); }
  out.scaling = scaling;
  if (t != nxr::spectral::Target::Stored) {
    const auto rec = spectral_record(out.producer);
    if (!rec || !rec->per_window_sum) invalid(name + ": its provenance has no analytic scaling record (computed before 2026-11-13): read it as stored, or --replace it");
    if (!nxr::spectral::isLinear(t)) invalid(name + ": a cross-spectrum is converted only to density or power (" + scaling + " applies to a real power)");
    double f = 0;
    try { f = nxr::spectral::linearFactor(rec->scaling, t); } catch (const std::invalid_argument& e) { invalid(name + ": " + e.what()); }
    for (std::size_t w = 0; w < out.sum.size(); ++w) out.value.push_back(out.sum[w] * (f / (2.0 * static_cast<double>(out.count[w]))));
    out.unit = nxr::spectral::unitOf(rec->power_unit, rec->scaling, t);
  } else {
    out.unit = "channel unit\xc2\xb2 (summed over a window's samples)";
  }
  return out;
}
// ── Hodge–Dirac eigenmode power of source maps ──────────────────────────────────────────────────────────────────

namespace {

// The subject's stored Hodge–Dirac package of a surface (the eigenvectors operator, the axis, the kinds, the support) and its glued surface.
struct HodgePackage {
  SurfaceInput in, glued;
  std::string stem, axis_id;
  Row opr, axis;
  std::int64_t Vin = 0, Vg = 0, Ktot = 0, K = 0, kept = 0;
  std::vector<double> mu_all;
  std::vector<std::uint8_t> kind_all;
  std::vector<std::int32_t> support;
};

HodgePackage resolve_hodge_package(SubjectWriter& sub, const Row& smf, const std::optional<std::string>& package, const std::optional<std::int64_t>& modes, const std::string& who) {
  HodgePackage pk;
  pk.Vin = integer(smf, "n_vertices");
  const std::int64_t Vin = pk.Vin;
  const std::optional<std::int64_t>& opt_modes = modes;
  const SurfaceInput in = read_surface(sub, text(smf, "name"));
  const std::string stem_pkg = package ? *package : (in.primary ? std::string("hodgeDirac") : "hodgeDirac__" + in.name);
  const auto opr = find_operator(sub.db(), sub.id(), "eigenvectors", stem_pkg + "_eigenvectors");
  if (!opr) missing(sub.name() + ": no stored Hodge-Dirac package '" + stem_pkg + "' (nxr-neuro run eigenmodes, or nxr eigen solve)");
  const auto axis = sub.db().read("manifold", text(*opr, "from_manifold_id"));
  // by NAME, never by path: a migrated dataset keeps the support in its manifold's folder (source/<stem>_support), a fresh solve at the root
  const auto kindf = find_field(sub.db(), sub.id(), FieldQuery{stem_pkg + "_kind"});
  const auto supp = nxr::io::selection::named_row(sub.db(), sub.id(), stem_pkg + "_support");
  if (!axis || !kindf || !supp) missing(sub.name() + ": the stored package '" + stem_pkg + "' is incomplete (axis, kinds or support)");
  const std::string axis_id = text(*axis, "id");
  const auto gm = sub.db().read("manifold", text(*opr, "to_manifold_id"));
  if (!gm) missing(stem_pkg + ": its eigenvectors' target surface is not stored");
  const SurfaceInput glued = read_surface(sub, text(*gm, "name"));      // the glued surface (the input itself for a surface of one piece)
  const std::int64_t Vg = glued.n_vertices();
  const std::vector<double> mu_all = read_all<double>(sub.store(), text(*axis, "path"));
  const std::vector<std::uint8_t> kind_all = read_all<std::uint8_t>(sub.store(), text(*kindf, "path"));
  const std::int64_t Ktot = static_cast<std::int64_t>(mu_all.size());
  const std::int64_t K = opt_modes ? std::min<std::int64_t>(*opt_modes, Ktot) : Ktot;
  if (K < 2) invalid(who + ": at least 2 modes");
  const std::vector<std::int32_t> support = read_all<std::int32_t>(sub.store(), text(*supp, "array_path"));
  const std::int64_t kept = static_cast<std::int64_t>(support.size());
  if (kept > Vg || kept > Vin) invalid(stem_pkg + ": its support does not fit the surfaces");
  for (auto s : support) if (s < 0 || s >= Vin) invalid(stem_pkg + ": its support indexes outside " + in.name);
  pk.in = in; pk.glued = glued; pk.stem = stem_pkg; pk.axis_id = axis_id; pk.opr = *opr; pk.axis = *axis;
  pk.Vg = Vg; pk.Ktot = Ktot; pk.K = K; pk.kept = kept; pk.mu_all = mu_all; pk.kind_all = kind_all; pk.support = support;
  return pk;
}

// The basis of the first pk.K modes: the stored potentials (the operator is [V_glued x Ktot], row-major), the vertex mass of the glued mesh as the solve took it.
nxr::hodge::Basis load_hodge_basis(SubjectWriter& sub, const HodgePackage& pk) {
  const std::int64_t Vg = pk.Vg, Ktot = pk.Ktot, K = pk.K, kept = pk.kept;
  const SurfaceInput& glued = pk.glued;
  const std::string& stem_pkg = pk.stem;
  const std::vector<std::int32_t>& support = pk.support;
  const std::vector<std::uint8_t>& kind_all = pk.kind_all;
  const std::vector<double>& mu_all = pk.mu_all;
  const std::optional<Row> opr = pk.opr;
  std::vector<std::int64_t> eshape;
  const std::vector<float> ev = read_all<float>(sub.store(), text(*opr, "path"), &eshape);
  if (eshape.size() != 2 || eshape[0] != Vg || eshape[1] != Ktot) invalid(stem_pkg + ": the stored eigenvectors are not [" + std::to_string(Vg) + " x " + std::to_string(Ktot) + "]");
  nxr::hodge::Basis basis;
  basis.P.resize(K, Vg);
  for (std::int64_t v = 0; v < Vg; ++v) for (std::int64_t k = 0; k < K; ++k) basis.P(k, v) = static_cast<double>(ev[static_cast<std::size_t>(v * Ktot + k)]);
  basis.mu = Eigen::Map<const Eigen::VectorXd>(mu_all.data(), K);
  for (std::int64_t k = 0; k < K; ++k) basis.kind.push_back(static_cast<int>(kind_all[static_cast<std::size_t>(k)]));
  basis.positions = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(glued.vertices.data(), Vg, 3);
  basis.faces = Eigen::Map<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>>(glued.faces.data(), glued.n_faces(), 3);
  basis.keep.assign(static_cast<std::size_t>(Vg), -1);
  for (std::int64_t i = 0; i < kept; ++i) basis.keep[static_cast<std::size_t>(i)] = support[static_cast<std::size_t>(i)];
  {
    nxr::manifold::DelaunayOptions dopt;
    const Json prov = Json::parse(opt_text(*opr, "producer_json").value_or("{}"), nullptr, false);
    if (prov.is_object() && prov.contains("solve") && prov["solve"].is_object() && prov["solve"].contains("options") && prov["solve"]["options"].is_object()) {
      dopt.mollify = prov["solve"]["options"].value("mollify", dopt.mollify);
      dopt.flipEps = prov["solve"]["options"].value("flip_eps", dopt.flipEps);
    }
    nxr::manifold::IntrinsicOptions io;                                   // the mollified input's mixed-Voronoi mass, as the solve took it
    io.mollify = dopt.mollify; io.flipEps = dopt.flipEps;
    nxr::manifold::IntrinsicSurface isurf(glued.vertices.data(), static_cast<int>(Vg), glued.faces.data(), static_cast<int>(glued.n_faces()), io);
    const std::vector<double> dm_mass = isurf.inputMixedVoronoiMass();
    basis.mass = Eigen::Map<const Eigen::VectorXd>(dm_mass.data(), static_cast<Eigen::Index>(dm_mass.size()));
  }
  return basis;
}

}  // namespace

// A HODGE-DIRAC PACKAGE as a node reads it (the runner hands it over: a node reads no store): the basis of its first `modes` modes (all: null), the input surface's
// positions (the kernel's sources) and the package's identity
HodgePackageData hodge_package_data(SubjectWriter& sub, const std::string& eigenvectors_name, const std::optional<std::int64_t>& modes) {
  const std::string suffix = "_eigenvectors";
  if (eigenvectors_name.size() <= suffix.size() || eigenvectors_name.compare(eigenvectors_name.size() - suffix.size(), suffix.size(), suffix) != 0)
    invalid(eigenvectors_name + ": not a Hodge-Dirac package's eigenvectors (<stem>_eigenvectors)");
  const std::string stem = eigenvectors_name.substr(0, eigenvectors_name.size() - suffix.size());
  const auto opr = find_operator(sub.db(), sub.id(), "eigenvectors", eigenvectors_name);
  if (!opr) missing(sub.name() + ": no stored Hodge-Dirac package '" + stem + "'");
  const auto gm = sub.db().read("manifold", text(*opr, "to_manifold_id"));
  if (!gm) missing(stem + ": its eigenvectors' target surface is not stored");
  const Json gp = Json::parse(opt_text(*gm, "producer_json").value_or("{}"), nullptr, false);
  const std::string input = gp.is_object() && gp.contains("glued_from") ? gp["glued_from"].get<std::string>() : text(*gm, "name");   // the glued surface's input, or itself
  const auto smf = find_manifold(sub.db(), sub.id(), input, "surface");
  if (!smf) missing(stem + ": its input surface '" + input + "' is not stored");
  const HodgePackage pk = resolve_hodge_package(sub, *smf, stem, modes, "hodge package");
  HodgePackageData d;
  d.basis = load_hodge_basis(sub, pk);
  d.input_positions = pk.in.vertices;
  d.input_surface_id = pk.in.id;
  d.stem = stem;
  d.axis_id = pk.axis_id;
  d.id = text(pk.opr, "id");
  return d;
}

// ── the stored package's scalar (exact) modes, for the spatial band of a tile level (spatial_band.cpp) ─────────────────

namespace internal {

StoredLaplace stored_laplace_info(SubjectWriter& sub, const SurfaceInput& S) {
  StoredLaplace out;
  const std::string stem = S.primary ? std::string("hodgeDirac") : "hodgeDirac__" + S.name;
  if (!find_operator(sub.db(), sub.id(), "eigenvectors", stem + "_eigenvectors")) return out;
  const HodgePackage pk = resolve_hodge_package(sub, S.manifold, std::nullopt, std::nullopt, "spatial band");
  out.found = true;
  out.package = pk.stem;
  out.glued_surface = pk.glued.name;
  out.modes_total = pk.Ktot;
  out.glued_vertices = pk.Vg;
  out.kept = pk.kept;
  out.lambda_min = std::numeric_limits<double>::infinity();
  for (std::int64_t k = 0; k < pk.Ktot; ++k) {
    if (pk.kind_all[static_cast<std::size_t>(k)] != 0) { ++out.n_coexact; continue; }
    const double lam = pk.mu_all[static_cast<std::size_t>(k)] * pk.mu_all[static_cast<std::size_t>(k)];
    ++out.n_exact;
    out.lambda_max = std::max(out.lambda_max, lam);
    out.lambda_min = std::min(out.lambda_min, lam);
  }
  return out;
}

GluedDomain glued_domain(SubjectWriter& sub, const SurfaceInput& S) {
  GluedDomain d;
  const std::string stem = S.primary ? std::string("hodgeDirac") : "hodgeDirac__" + S.name;
  if (find_operator(sub.db(), sub.id(), "eigenvectors", stem + "_eigenvectors")) {
    const HodgePackage pk = resolve_hodge_package(sub, S.manifold, std::nullopt, std::nullopt, "glued domain");
    d.found = true; d.surface = pk.glued; d.support = pk.support;
    return d;
  }
  if (connected_pieces(S.faces, S.n_vertices()) == 1) {
    d.found = true; d.surface = S;
    d.support.resize(static_cast<std::size_t>(S.n_vertices()));
    for (std::int64_t i = 0; i < S.n_vertices(); ++i) d.support[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
    return d;
  }
  d.why = S.name + " is a cortex of several pieces and no glued surface is stored: run `nxr-neuro run eigenmodes --glued` (it glues the hemispheres by anatomy and stores the glued surface)";
  return d;
}

LaplaceModes read_laplace_modes(SubjectWriter& sub, const SurfaceInput& S, const std::function<bool(double)>& want) {
  const HodgePackage pk = resolve_hodge_package(sub, S.manifold, std::nullopt, std::nullopt, "spatial band");
  LaplaceModes out;
  out.package = pk.stem;
  const std::int64_t Vg = pk.Vg, Ktot = pk.Ktot;
  std::vector<std::int64_t> cols;                       // the exact modes the caller wants, in the package's order
  for (std::int64_t k = 0; k < Ktot; ++k) {
    if (pk.kind_all[static_cast<std::size_t>(k)] != 0) continue;
    const double mu = pk.mu_all[static_cast<std::size_t>(k)];
    if (want(mu * mu)) cols.push_back(k);
  }
  // the glued surface's vertex mass as the solve took it (the package's provenance gives its Delaunay options), as load_hodge_basis
  nxr::manifold::DelaunayOptions dopt;
  const Json prov = Json::parse(opt_text(pk.opr, "producer_json").value_or("{}"), nullptr, false);
  if (prov.is_object() && prov.contains("solve") && prov["solve"].is_object() && prov["solve"].contains("options") && prov["solve"]["options"].is_object()) {
    dopt.mollify = prov["solve"]["options"].value("mollify", dopt.mollify);
    dopt.flipEps = prov["solve"]["options"].value("flip_eps", dopt.flipEps);
  }
  nxr::manifold::IntrinsicOptions io;                                   // the mollified input's mixed-Voronoi mass, as the solve took it
    io.mollify = dopt.mollify; io.flipEps = dopt.flipEps;
    nxr::manifold::IntrinsicSurface isurf(pk.glued.vertices.data(), static_cast<int>(Vg), pk.glued.faces.data(), static_cast<int>(pk.glued.n_faces()), io);
    const std::vector<double> dm_mass = isurf.inputMixedVoronoiMass();
  out.mass = Eigen::Map<const Eigen::VectorXd>(dm_mass.data(), static_cast<Eigen::Index>(dm_mass.size()));
  out.support = pk.support;
  // the constant mode (lambda = 0) is never stored; the glued surface is one connected piece, so it is 1/sqrt(area) on every vertex
  const bool constant = want(0.0);
  const Eigen::Index k = static_cast<Eigen::Index>(cols.size()) + (constant ? 1 : 0);
  out.phi.resize(Vg, k);
  out.lambda.resize(k);
  Eigen::Index j = 0;
  if (constant) { out.phi.col(0).setConstant(1.0 / std::sqrt(out.mass.sum())); out.lambda(0) = 0.0; j = 1; }
  if (!cols.empty()) {
    std::vector<std::int64_t> eshape;
    const std::vector<float> ev = read_all<float>(sub.store(), text(pk.opr, "path"), &eshape);
    if (eshape.size() != 2 || eshape[0] != Vg || eshape[1] != Ktot) invalid(pk.stem + ": the stored eigenvectors are not [" + std::to_string(Vg) + " x " + std::to_string(Ktot) + "]");
    for (const std::int64_t c : cols) {
      for (std::int64_t v = 0; v < Vg; ++v) out.phi(v, j) = static_cast<double>(ev[static_cast<std::size_t>(v * Ktot + c)]);
      const double mu = pk.mu_all[static_cast<std::size_t>(c)];
      out.lambda(j) = mu * mu;
      // unit norm in the vertex mass: the stored potentials are psi/sqrt2 (norm^2 1/2); the norm is read off the data, as the package's reader does
      const double n2 = out.phi.col(j).cwiseAbs2().dot(out.mass);
      if (!(n2 > 0.1 && n2 < 2.0)) invalid(pk.stem + ": exact mode " + std::to_string(c) + " has mass norm^2 " + std::to_string(n2) + " (expected 1/2 or 1): not a vertex potential of this glued surface");
      out.phi.col(j) /= std::sqrt(n2);
      ++j;
    }
  }
  return out;
}

}  // namespace internal

// ── the mode cross-spectra, for pipelines that read them (speedspectrum) ────────────────────────────────────────────

namespace internal {
namespace {

}  // namespace



std::string clear_stored_nodes(DatasetWriter& ds, SubjectWriter& sub, const std::vector<std::pair<const char*, std::string>>& nodes, bool replace, const std::string& what, bool& replaced) {
  return clear_stored(ds, sub, nodes, replace, what, replaced);
}
std::string producer_string(const char* verb) { return producer_name(verb); }
const char* window_label(nxr::welch::Window w) { return window_name(w); }
const char* detrend_label(nxr::welch::Detrend d) { return detrend_name(d); }
const char* scaling_label(nxr::welch::Scaling s) { return scaling_name(s); }
}  // namespace internal

// ── temporal filters ─────────────────────────────────────────────────────────────────────────────────────────────


}  // namespace nxr::pipelines
