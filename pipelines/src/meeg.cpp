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
#include "nxr/pipelines/tiles.h"
#include "nxr/pipelines/pyramid.h"
#include "nxr/pipelines/frame_tiling.h"
#include "nxr/ladder.h"
#include "nxr/fft.h"
#include "nxr/rotary.h"
#include "nxr/timebank.h"
#include "nxr/partition.h"
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
RecordingInput read_recording(SubjectWriter& sub, const std::string& name) {
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
  if (static_cast<std::int64_t>(r.types.size()) != r.n_channels) invalid(name + ": its channels have no 'type' labelling of the right size");
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
  const RecordingInput r = read_recording(sub, recording);
  return Json{{"types", r.types}, {"quality", r.quality}, {"units", r.units}, {"n_channels", r.n_channels}};
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

// ── the phase gradient ───────────────────────────────────────────────────────────────────────────────────────────

namespace {
// What the plain and the spatial-band routes of phasegradient share: the cross-spectrum, the kernel, its surface.
struct PhaseInputs {
  Row crow, kern, smf;
  Json cpj;
  std::string band, recording, kname, surface, surface_id, unit;
  double lo = 0, hi = 0, omega = 0, metres = 1;
  std::vector<std::int32_t> chans;
  std::vector<float> kf;
  std::vector<std::int64_t> kshape;
  std::int64_t M = 0, N = 0;
};
PhaseGradientResult phasegradient_band(DatasetWriter& ds, SubjectWriter& sub, const PhaseGradientOptions& opt, const PhaseInputs& in, const std::function<void(const Json&)>& say);
}  // namespace

PhaseGradientResult phasegradient_compute(DatasetWriter& ds, SubjectWriter& sub, const PhaseGradientOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.signal != "normal" && opt.signal != "ambient") invalid("phasegradient: signal must be normal or ambient");
  const auto cnode = find_cross_spectrum(sub, opt.cross_spectrum);
  if (!cnode) missing(sub.name() + ": no cross-spectral density named '" + opt.cross_spectrum + "' (`nxr-neuro crossspectrum compute` first)");
  const std::optional<Row> crow = cnode->row;                     // the statistic's (or an older Operator's) row: its id and modified time key the result
  const Json cpj = cnode->producer;
  if (!cpj.is_object() || !cpj.contains("band_name") || !cpj.contains("band_lo_hz") || !cpj.contains("band_hi_hz") || !cpj.contains("recording"))
    invalid(opt.cross_spectrum + ": its producer record does not name its band and recording (computed by an older `crossspectrum compute`: --replace it)");
  const std::string band = cpj["band_name"].get<std::string>(), recording = cpj["recording"].get<std::string>();
  const double lo = cpj["band_lo_hz"].get<double>(), hi = cpj["band_hi_hz"].get<double>(), omega = 2 * M_PI * 0.5 * (lo + hi);
  Row kern;
  if (opt.kernel) {
    const auto r = find_operator(sub.db(), sub.id(), "inverse kernel", *opt.kernel);
    if (!r) missing(sub.name() + ": no inverse kernel named '" + *opt.kernel + "'");
    kern = *r;
  } else {
    const auto rs = stored_operators(sub.db(), sub.id(), "inverse kernel");
    if (rs.size() != 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " inverse kernels; name one with --kernel");
    kern = rs[0];
  }
  const std::string kname = text(kern, "name"), surface_id = text(kern, "to_manifold_id");
  if (opt_text(kern, "to_value_type").value_or("vector3") == "scalar") invalid(kname + ": a fixed-orientation kernel: the phase gradient reads the current's components (a free-orientation kernel)");
  if (text(kern, "from_manifold_id") != cnode->channels_manifold) invalid(kname + " is on other channels than " + opt.cross_spectrum);
  const auto ksel = sub.db().read("selection", text(kern, "from_selection_id"));
  if (!ksel || is_null(*ksel, "array_path")) invalid(kname + ": its channels (from_selection) are not stored");
  const std::vector<std::int32_t> chans = read_all<std::int32_t>(sub.store(), text(*ksel, "array_path"));
  const std::int64_t M = static_cast<std::int64_t>(chans.size());
  std::vector<std::int64_t> kshape;
  const std::vector<float> kf = read_all<float>(sub.store(), text(kern, "path"), &kshape);
  if (kshape.size() != 2 || kshape[1] != M || kshape[0] % 3 != 0) invalid(kname + ": its array is not [3 x sources rows x " + std::to_string(M) + " channels]");
  const std::int64_t N = kshape[0] / 3;
  const auto smf = sub.db().read("manifold", surface_id);
  if (!smf || integer(*smf, "n_vertices") != N) invalid(kname + ": " + std::to_string(N) + " sources, its target surface has another vertex count");
  const std::string surface = text(*smf, "name");
  // the length unit of the surface's coordinates
  const auto urow = manifold_coordinate_unit(sub.db(), surface_id);
  const std::string unit = urow;
  if (unit != "m" && unit != "mm") invalid(surface + ": its coordinates are in '" + unit + "', not m or mm: the speed needs a length unit");
  const double metres = unit == "mm" ? 1e-3 : 1.0;
  {
    if (!opt.spatial_band) invalid("phasegradient: without --spatial-band it is the definition `nxr-neuro run phasegradient` (measurements of the leaves)");
    PhaseInputs pin;
    pin.crow = *crow; pin.kern = kern; pin.smf = *smf; pin.cpj = cpj; pin.band = band; pin.recording = recording; pin.kname = kname; pin.surface = surface;
    pin.surface_id = surface_id; pin.unit = unit; pin.lo = lo; pin.hi = hi; pin.omega = omega; pin.metres = metres; pin.chans = chans; pin.kf = kf; pin.kshape = kshape;
    pin.M = M; pin.N = N;
    return phasegradient_band(ds, sub, opt, pin, say);
  }
  invalid("phasegradient: without --spatial-band it is the definition `nxr-neuro run phasegradient` (measurements of the leaves); this command keeps the spatial-band route");
}

// the quadrature spectrum of the sources is the definition `quadrature` (the node `quadrature.source_pairs`, pipelines/src/quadrature_steps.cpp)

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

namespace {


// ── phasegradient at a spatial scale: the band's current kernel, its phase sums, the reading of its phase field in the modes ──
// ── SHARED BY THE SPATIAL-BAND VERBS (phasegradient --spatial-band, cyclepattern) ──
// The spatial bands of a package: gains on λ = μ² of its modes ("modes": every mode at gain 1; "weyl:J" | "weyl:all": the Weyl Meyer octave bank, THE ONE the surface's `pyramid spatial-bands` stores — λ_l = 4π·2^l / A with A
// the ladder's mean piece area, one band per ladder level, code = the level; a member with no gain above 1e-3 on the package's modes is left out).
struct SpatialBand { std::string label; std::vector<double> gain; std::int64_t level = -1; };   // level: the stored bank's code (-1 for "modes": no spatial band, no within key)
struct SpatialBank { std::string selection_id; std::vector<nxr::io::SpatialBandEdge> edges; double piece_area = 0; };
SpatialBank spatial_bank_of(DatasetWriter& ds, SubjectWriter& sub, const std::string& surface) {
  SpatialBandsOptions o; o.surface = surface;
  const SpatialBandsResult r = pyramid_spatial_bands(ds, sub, o);
  return {r.selection_id, r.edges, r.piece_area};
}
std::vector<SpatialBand> spatial_bands(const HodgePackage& pk, const std::string& spec, const std::string& who, const SpatialBank& stored) {
  const std::int64_t K = pk.K;
  std::vector<SpatialBand> bands;
  if (spec == "modes") {
    bands.push_back({"modes", std::vector<double>(static_cast<std::size_t>(K), 1.0), -1});
  } else if (spec.rfind("weyl:", 0) == 0) {
    const int levels = static_cast<int>(stored.edges.size());
    const nxr::filterbank::Bank bank = nxr::filterbank::weyl(stored.piece_area, levels);
    for (int j = 0; j < levels; ++j) {   // the pairing, as spatial_band.cpp checks it: member j of the bank IS level j of the stored bank
      const auto close = [](double x, double y) { return std::abs(x - y) <= 1e-9 * std::max(std::abs(x), std::abs(y)); };
      if (!close(bank.members[static_cast<std::size_t>(j)].hi, stored.edges[static_cast<std::size_t>(j)].hi) || !close(bank.members[static_cast<std::size_t>(j)].lo, stored.edges[static_cast<std::size_t>(j)].lo))
        invalid(who + ": the stored spatial bands are not Weyl's bank of the ladder's mean piece area: the ladder was made with another area");
    }
    const std::string which = spec.substr(5);
    std::vector<int> members;
    if (which == "all") { for (int j = 0; j < levels; ++j) members.push_back(j); }
    else {
      int j = -1;
      try { std::size_t used = 0; j = std::stoi(which, &used); if (used != which.size()) j = -1; } catch (...) { j = -1; }
      if (j < 0 || j >= levels) invalid(who + ": --spatial-band weyl:J takes J in 0 … " + std::to_string(levels - 1) + " (the surface's stored spatial bands, one per ladder level) or all");
      members.push_back(j);
    }
    for (int j : members) {
      SpatialBand b{"weyl" + std::to_string(j), {}, j};
      double peak = 0;
      for (std::int64_t k = 0; k < K; ++k) { const double m2 = pk.mu_all[static_cast<std::size_t>(k)] * pk.mu_all[static_cast<std::size_t>(k)]; b.gain.push_back(bank.members[static_cast<std::size_t>(j)].gain(m2)); peak = std::max(peak, std::abs(b.gain.back())); }
      if (peak > 1e-3) bands.push_back(std::move(b));
    }
  } else {
    invalid(who + ": --spatial-band must be modes, weyl:J or weyl:all");
  }
  return bands;
}

// THE KERNEL IN THE FULL FORMS: the intrinsic complex the package's modes are solved on (stored by `run eigenmodes`), the modes' full forms, and a free kernel
// on the glued surface read into them once: its normal part n·J as a 2-form (A2 [K x M], coexact modes) and its tangent part as a 1-form (A1 [K x M]).
struct KernelForms {
  nxr::intrinsic::Complex c;
  nxr::intrinsic::Modes md;
  Eigen::MatrixXd A2, A1;                                       // the kernel's normal (2-form) and tangent (1-form) coefficients [K x M]
  Eigen::MatrixXd N3;                                           // the vertex normals [Vg x 3]
};
KernelForms kernel_forms(SubjectWriter& sub, const HodgePackage& pk, const std::vector<float>& kf, const std::vector<std::int64_t>& kshape, std::int64_t M, const std::string& kname,
                         const std::function<void(const Json&)>& say) {
  const std::int64_t Vg = pk.Vg, K = pk.K;
  const std::string iname = pk.glued.name + "_intrinsic";
  const auto im = sub.find("manifold", "source/" + iname);
  const auto f2 = sub.find("operator", pk.stem + "_forms2");
  if (!im || !f2) missing(sub.name() + ": the package '" + pk.stem + "' has no full forms (the intrinsic complex " + iname + ", " + pk.stem + "_forms2): `run eigenmodes --replace` stores them");
  const std::string iid = text(*im, "id");
  KernelForms out;
  nxr::intrinsic::Complex& c = out.c;
  nxr::intrinsic::Modes& md = out.md;
  c.nV = static_cast<int>(Vg); c.nE = static_cast<int>(integer(*im, "n_edges")); c.nF = static_cast<int>(integer(*im, "n_faces"));
  auto fieldv = [&](const std::string& n) { const auto r = sub.find("field", nxr::io::node_path(sub.db(), iid, iname + "_" + n)); if (!r) missing(iname + "_" + n + " is not stored"); return read_all<double>(sub.store(), text(*r, "path")); };
  auto vec = [](const std::vector<double>& v) { return Eigen::VectorXd(Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(v.size()))); };
  auto csr = [&](const std::string& n, int rows, int cols) {
    const auto r = sub.find("operator", nxr::io::node_path(sub.db(), iid, iname + "_" + n));
    if (!r) missing(iname + "_" + n + " is not stored");
    const std::string at = text(*r, "path");
    const auto ip = read_all<std::int64_t>(sub.store(), at + "/indptr"); const auto ix = read_all<std::int32_t>(sub.store(), at + "/indices");
    const auto dv = read_all<double>(sub.store(), at + "/data");
    std::vector<Eigen::Triplet<double>> t;
    for (int i = 0; i < rows; ++i) for (std::int64_t k = ip[static_cast<std::size_t>(i)]; k < ip[static_cast<std::size_t>(i + 1)]; ++k) t.emplace_back(i, ix[static_cast<std::size_t>(k)], dv[static_cast<std::size_t>(k)]);
    Eigen::SparseMatrix<double> Mx(rows, cols);
    Mx.setFromTriplets(t.begin(), t.end());
    return Mx;
  };
  c.d0 = csr("d0", c.nE, c.nV);
  c.d1 = csr("d1", c.nF, c.nE);
  c.edges = read_all<std::int32_t>(sub.store(), "source/" + iname + "/edges");
  c.length = vec(fieldv("length")); c.star0 = vec(fieldv("star0")); c.star1 = vec(fieldv("star1")); c.star2 = vec(fieldv("star2"));
  c.angleTail = vec(fieldv("angle_tail")); c.angleTip = vec(fieldv("angle_tip")); c.coneAngle = vec(fieldv("angle_sum"));
  { const auto r = fieldv("reference"); c.reference.assign(r.begin(), r.end()); }
  // the embedding: the glued surface the complex was made from (the same vertices, as the solve took them: float32-rounded positions)
  nxr::intrinsic::attachEmbedding(c, pk.glued.vertices.data(), pk.glued.faces.data(), static_cast<int>(pk.glued.n_faces()));
  // the modes' full forms
  const nxr::hodge::Basis basis = load_hodge_basis(sub, pk);
  md.P = basis.P.transpose();
  std::vector<std::int64_t> sh;
  const auto x2 = read_all<float>(sub.store(), text(*f2, "path"), &sh);
  if (sh.size() != 2 || sh[0] != c.nF || sh[1] < K) invalid(pk.stem + "_forms2 is not [" + std::to_string(c.nF) + " x K]");
  md.X2.resize(c.nF, K);
  for (std::int64_t f = 0; f < c.nF; ++f) for (std::int64_t k = 0; k < K; ++k) md.X2(f, k) = x2[static_cast<std::size_t>(f * sh[1] + k)];
  md.mu = basis.mu; md.kind = basis.kind;
  // the kernel on the glued surface, analysed once: its normal part n·J as a 2-form, its tangent part as a 1-form, both read into the modes
  const Eigen::MatrixXd Kd = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(kf.data(), kshape[0], M).cast<double>();
  const nxr::hodge::RowMat3 Pin = Eigen::Map<const nxr::hodge::RowMat3>(pk.in.vertices.data(), pk.in.n_vertices(), 3);
  if (say) say(Json{{"stage", "forms"}, {"kernel", kname}, {"package", pk.stem}, {"modes", K}, {"glued_vertices", Vg}, {"intrinsic_edges", c.nE}});
  const Eigen::MatrixXd Wg = nxr::hodge::gluedKernel(Kd, basis, Pin);          // [3 Vg x M]
  out.N3 = c.normals;
  Eigen::MatrixXd rho(Vg, M), Jt = Wg;
  for (std::int64_t v = 0; v < Vg; ++v) {
    const Eigen::Vector3d n = out.N3.row(v).transpose();
    rho.row(v) = n.transpose() * Wg.middleRows(3 * v, 3);
    Jt.middleRows(3 * v, 3) -= n * rho.row(v);
  }
  out.A2 = nxr::intrinsic::read2(c, md, nxr::intrinsic::area2form(c, rho));
  out.A1 = nxr::intrinsic::read1(c, md, nxr::intrinsic::flat(c, Jt));
  return out;
}

PhaseGradientResult phasegradient_band(DatasetWriter& ds, SubjectWriter& sub, const PhaseGradientOptions& opt, const PhaseInputs& in, const std::function<void(const Json&)>& say) {
  if (opt.signal != "ambient") invalid("phasegradient: --spatial-band reads the band's current by its x y z components (no orientation assumed): pass --signal ambient");
  if (opt.phase_of != "current" && opt.phase_of != "stream") invalid("phasegradient: --phase-of must be current or stream");
  if (opt.fold_edges != "keep" && opt.fold_edges != "axial" && opt.fold_edges != "all") invalid("phasegradient: --fold-edges must be keep, axial (the edges with opposed normals) or all");
  if (opt.defects != "keep" && opt.defects != "exclude") invalid("phasegradient: --defects must be keep or exclude");
  if (opt.defects == "exclude" && opt.fit == "inner") invalid("phasegradient: --defects exclude weights edges out: it goes with a weighted fit (--fit mesh or modes)");
  if (opt.fit != "inner" && opt.fit != "mesh" && opt.fit != "modes") invalid("phasegradient: --fit must be inner, mesh or modes");
  if (opt.weight != "power" && opt.weight != "coherence") invalid("phasegradient: --weight must be power or coherence");
  if (opt.edge_lag != "frame" && opt.edge_lag != "frame-normal" && opt.edge_lag != "frame-tangent" && opt.edge_lag != "axial" && opt.edge_lag != "ambient")
    invalid("phasegradient: --edge-lag must be frame, frame-normal, frame-tangent, axial or ambient");
  if (opt.field != "gradient") invalid("phasegradient: --spatial-band reads the phase 1-form gradient (each intrinsic edge's lag arg T): a power-weighted timing fit is a weighted solve, not inner products in the modes, and is not built");
  const HodgePackage pk = resolve_hodge_package(sub, in.smf, opt.package, std::nullopt, "phasegradient");
  const std::int64_t Vg = pk.Vg, K = pk.K, kept = pk.kept;
  const std::string spec = *opt.spatial_band, iname = pk.glued.name + "_intrinsic";
  const SpatialBank bank = spec == "modes" ? SpatialBank{} : spatial_bank_of(ds, sub, in.surface);   // THE ONE bank: stored by `pyramid spatial-bands` (made, with the ladder, when absent)
  const std::vector<SpatialBand> bands = spatial_bands(pk, spec, "phasegradient", bank);
  // ── built once, on the first miss: THE INTRINSIC COMPLEX the modes are solved on and their FULL FORMS (stored by `run eigenmodes`), the kernel's analysis ──
  // Everything below is on that one complex: the band filter reads the current into the modes by inner products with their forms (the
  // tangent part as a 1-form on the intrinsic edges, the normal part as a 2-form on its faces), the phase field is a 1-form on its edges (each edge's
  // lag from the cross-spectrum between its ends), and its split is inner products with the modes' 1-forms.
  std::optional<KernelForms> kfm;
  std::optional<StoredCrossSpectrum> C;
  auto prepare = [&]() {
    if (kfm) return;
    kfm = kernel_forms(sub, pk, in.kf, in.kshape, in.M, in.kname, say);
    C = crossspectrum_read(sub, opt.cross_spectrum, in.chans);
  };
  const std::string stem0 = (opt.name ? *opt.name : in.recording + "_" + in.kname + "_phasegradient");
  const std::string gid = pk.glued.id;
  Json results = Json::array();
  PhaseGradientResult res;
  // THE LEAVES (the measurement pyramid, spec docs/specs/2026-11-14-measurement-pyramid.md, decision 1): the sums that ARE additive (the band's power) and the tiles' exact integrals of the propagation's flux and of
  // the phase field's circulation (nxr measure/integrals.py) are array-backed measurements [leaf] of a byte-less Field on the surface's leaf partition; the global fits (timing, stream function, propagation,
  // inconsistency) have no additive form and are ONE statistic of vertex x quantity on the glued surface, of the cross-spectrum statistic they are computed from; the mode coefficients another on the eigen axis
  const SurfaceInput Sin = read_surface(sub, in.surface, true);                       // as stored (float64): the ladder's input
  if (Sin.n_vertices() != pk.Vin) invalid(in.surface + ": another vertex count than the package's surface");
  PyramidLeaves PL = pyramid_leaves(ds, sub, Sin, opt.depth, "phasegradient", say);
  std::vector<std::int32_t> gcodes(static_cast<std::size_t>(Vg), -1);                // the leaf of each glued vertex: its input vertex's (support order); the bridge is in no leaf
  for (std::int64_t v = 0; v < kept; ++v) gcodes[static_cast<std::size_t>(v)] = PL.codes[static_cast<std::size_t>(pk.support[static_cast<std::size_t>(v)])];
  std::vector<double> in_area;
  {
    nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(Sin.n_vertices()); mesh.positions = Sin.vertices; mesh.faces = Sin.faces;
    in_area = nxr::ladder::vertexAreas(mesh);
  }
  for (const SpatialBand& bd : bands) {
    const std::string stem = stem0 + "_" + bd.label;
    const std::string fname = nxr::bst::band_field_name(stem, in.band);                // the byte-less Field the leaf measurements are of
    const std::string map_name = nxr::bst::band_field_name(stem + "_map", in.band);   // the statistic of the global fits
    const std::string modes_name = nxr::bst::band_field_name(stem + "_modes", in.band);
    const std::string map_path = "statistics/" + nxr::io::safe_name(map_name), modes_path = "statistics/" + nxr::io::safe_name(modes_name);
    CacheKey key;
    key.pipeline = "phasegradient";
    key.params = Json{{"name", stem}, {"signal", opt.signal}, {"spatial_band", bd.label}, {"weyl_piece_area", bank.piece_area}, {"field", opt.field}, {"edge_lag", opt.edge_lag}, {"fit", opt.fit}, {"phase_of", opt.phase_of}, {"defects", opt.defects}, {"fold_edges", opt.fold_edges}, {"weight", opt.fit == "inner" ? Json() : Json(opt.weight)},
                      {"route", "the intrinsic complex: the band by the full forms, the phase 1-form on its edges, its split by inner products"}, {"store", "statistic + leaf measurements"}, {"depth", PL.depth}};
    key.input = Json{{"cross_spectrum", text(in.crow, "id")}, {"cross_spectrum_modified", opt_text(in.crow, "modified_utc").value_or("")}, {"kernel", text(in.kern, "id")},
                     {"kernel_modified", opt_text(in.kern, "modified_utc").value_or("")}, {"surface", in.surface_id}, {"package", text(pk.opr, "id")}, {"leaves", text(*PL.part, "id")}};
    key.compute = NXR_COMPUTE_VERSION;
    Eigen::VectorXd power, timing, speed, stream, incons, coef, leaf_power, leaf_flux, leaf_circ;
    Eigen::MatrixXd prop;
    double share_ex = NAN, share_co = NAN, share_res = NAN, wide = NAN, wide_energy = NAN, weak_energy = NAN, leak_band = NAN, leak_modes = NAN, excluded_energy = NAN;
    int excluded = 0, fold_count = 0;
    std::string moved_aside;
    bool replaced = false;
    auto load = [&](const Row&) { return 0; };
    auto compute = [&]() {
      prepare();
      const nxr::intrinsic::Complex& c = kfm->c; const nxr::intrinsic::Modes& md = kfm->md; const Eigen::MatrixXd& A2 = kfm->A2; const Eigen::MatrixXd& A1 = kfm->A1; const Eigen::MatrixXd& N3 = kfm->N3;
      say(Json{{"stage", "cache"}, {"hit", false}, {"cross_spectrum", opt.cross_spectrum}, {"kernel", in.kname}, {"spatial_band", bd.label}});
      // the band: the kernel's coefficients weighted by the band's gains, synthesised back to a vertex current (tangent + normal)
      Eigen::VectorXd g(K);
      for (std::int64_t k = 0; k < K; ++k) g(k) = bd.gain[static_cast<std::size_t>(k)];
      power = Eigen::VectorXd::Zero(Vg);
      Eigen::VectorXcd T = Eigen::VectorXcd::Zero(c.nE);
      if (opt.phase_of == "stream") {
        // THE STREAM FUNCTION, a scalar at each vertex: S = Σ_coexact (g_k c_k / μ_k) P_k, c the tangent current's coefficients (its coexact part:
        // −n × ∇S); no vector, no frame. Each edge's lag: arg of S's cross-spectrum between its ends (a pure 1-form)
        Eigen::VectorXd gs = Eigen::VectorXd::Zero(K);
        for (std::int64_t k = 0; k < K; ++k) if (md.kind[static_cast<std::size_t>(k)] == 2) gs(k) = g(k) / md.mu(k);
        const Eigen::MatrixXd Ks = md.P * (gs.asDiagonal() * A1);          // [Vg x M]
        const Eigen::MatrixXcd Cc = C->S.cast<std::complex<double>>() + std::complex<double>(0, 1) * C->Sq.cast<std::complex<double>>();
        const Eigen::MatrixXcd Z = Ks.cast<std::complex<double>>() * Cc;
        for (std::int64_t v = 0; v < Vg; ++v) power(v) = (Z.row(v) * Ks.row(v).transpose().cast<std::complex<double>>())(0, 0).real();
        for (int e = 0; e < c.nE; ++e)
          T(e) = (Z.row(c.edges[2 * static_cast<std::size_t>(e) + 1]) * Ks.row(c.edges[2 * static_cast<std::size_t>(e)]).transpose().cast<std::complex<double>>())(0, 0);
      } else {
        const Eigen::MatrixXd rhoB = nxr::intrinsic::vertexDensity(c, nxr::intrinsic::synth2(c, md, g.asDiagonal() * A2));
        const Eigen::MatrixXd Jt = nxr::intrinsic::sharp(c, nxr::intrinsic::synth1(c, md, g.asDiagonal() * A1));   // the tangent part [3 Vg x M]
        Eigen::MatrixXd J = Jt;
        for (std::int64_t v = 0; v < Vg; ++v) J.middleRows(3 * v, 3) += N3.row(v).transpose() * rhoB.row(v);
        {   // (diagnostic) the band's tangent current read back into the modes: the share of its energy outside the band, and outside the modes
          const Eigen::MatrixXd wt = nxr::intrinsic::flat(c, Jt);
          const Eigen::MatrixXd cb = nxr::intrinsic::read1(c, md, wt);
          double inb = 0, outb = 0;
          for (std::int64_t k = 0; k < K; ++k) { const double e = 0.5 * cb.row(k).squaredNorm(); (std::abs(bd.gain[static_cast<std::size_t>(k)]) > 0.1 ? inb : outb) += e; }
          const double tot = (wt.transpose() * c.star1.asDiagonal() * wt).trace();
          leak_band = outb / tot; leak_modes = std::max(0.0, 1 - (inb + outb) / tot);
        }
        // the power at each vertex, and each intrinsic edge's cross-spectrum between its ends, T(tip, tail) = ⟨z_tip conj z_tail⟩ (Z = K C: row u is k(u) C)
        const Eigen::MatrixXcd Cc = C->S.cast<std::complex<double>>() + std::complex<double>(0, 1) * C->Sq.cast<std::complex<double>>();
        auto rowsOf = [&](const Eigen::MatrixXd& X, int comp) { Eigen::MatrixXd R(Vg, in.M); for (std::int64_t v = 0; v < Vg; ++v) R.row(v) = X.row(3 * v + comp); return R; };
        auto dot = [](const Eigen::MatrixXcd& Z, int a, const Eigen::MatrixXd& K, int b) { return (Z.row(a) * K.row(b).transpose().cast<std::complex<double>>())(0, 0); };
        const bool framed = opt.edge_lag == "frame" || opt.edge_lag == "frame-normal" || opt.edge_lag == "frame-tangent";
        const bool use_n = opt.edge_lag != "frame-tangent", use_t = opt.edge_lag != "frame-normal";
        if (framed) {
          // IN THE SURFACE'S OWN FRAME: the normal parts as scalars (n·J, oriented by the outward normal), plus the tangent parts with the tip's carried to the
          // tail along the edge (the transport mapping the tip's frame (d_tip, n_tip × d_tip) onto the tail's, d the edge's direction at each end): the
          // surface's folding is not read as phase
          const Eigen::MatrixXcd Zr = rhoB.cast<std::complex<double>>() * Cc;
          Eigen::MatrixXd Kt[3];
          Eigen::MatrixXcd Zt[3];
          for (int a = 0; a < 3; ++a) { Kt[a] = rowsOf(Jt, a); Zt[a] = Kt[a].cast<std::complex<double>>() * Cc; }
          for (std::int64_t v = 0; v < Vg; ++v) {
            power(v) += dot(Zr, static_cast<int>(v), rhoB, static_cast<int>(v)).real();
            for (int a = 0; a < 3; ++a) power(v) += dot(Zt[a], static_cast<int>(v), Kt[a], static_cast<int>(v)).real();
          }
          for (int e = 0; e < c.nE; ++e) {
            const int tail = c.edges[2 * static_cast<std::size_t>(e)], tip = c.edges[2 * static_cast<std::size_t>(e) + 1];
            const Eigen::Vector3d du = nxr::intrinsic::tangentAt(c, e, false), dv = nxr::intrinsic::tangentAt(c, e, true);
            const Eigen::Vector3d nu = N3.row(tail).transpose(), nv = N3.row(tip).transpose();
            const Eigen::Matrix3d R = du * dv.transpose() + nu.cross(du) * nv.cross(dv).transpose();   // tip's tangent plane → tail's
            // T = ρ(tip) C ρ(tail)ᵀ + Σ_a,b R_ab t_b(tip) C t_a(tail)ᵀ: the tip's tangent current carried to the tail, then compared
            std::complex<double> t = use_n ? dot(Zr, tip, rhoB, tail) : std::complex<double>(0, 0);
            if (use_t) for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) if (R(a, b) != 0) t += R(a, b) * dot(Zt[b], tip, Kt[a], tail);
            T(e) = t;
          }
        } else {
          for (int comp = 0; comp < 3; ++comp) {                   // AMBIENT: the x y z components compared as they are
            const Eigen::MatrixXd Kc = rowsOf(J, comp);
            const Eigen::MatrixXcd Z = Kc.cast<std::complex<double>>() * Cc;
            for (std::int64_t v = 0; v < Vg; ++v) power(v) += dot(Z, static_cast<int>(v), Kc, static_cast<int>(v)).real();
            for (int e = 0; e < c.nE; ++e) T(e) += dot(Z, c.edges[2 * static_cast<std::size_t>(e) + 1], Kc, c.edges[2 * static_cast<std::size_t>(e)]);
          }
        }
      }
      // the phase 1-form: φ(tip) − φ(tail) = arg T(tip, tail); AXIAL: modulo π (a reversal counts as in phase), ½ arg T²
      Eigen::VectorXd w(c.nE);
      // FOLD EDGES (`fold_edges` axial): an edge whose ends' normals are opposed spans a fold the mesh does not resolve (a sulcal fundus: the two ends on
      // facing walls). The estimate is smooth in space, so the surface frame reverses across it while the current does not: read its lag modulo π
      // (a reversal is no lag; a lag below π/2 is kept — a real one across one edge is ~0.01 rad)
      int folds = 0;
      for (int e = 0; e < c.nE; ++e) {
        const bool fold = opt.fold_edges == "all" ||
                          (opt.fold_edges == "axial" && N3.row(c.edges[2 * static_cast<std::size_t>(e)]).dot(N3.row(c.edges[2 * static_cast<std::size_t>(e) + 1])) < 0);
        folds += fold;
        w(e) = (opt.edge_lag == "axial" || fold) ? 0.5 * std::arg(T(e) * T(e)) : std::arg(T(e));
      }
      fold_count = folds;
      {   // the edges whose comparison is beyond ±π/2 (Re T < 0): their share, and the share of the phase field's ⋆1 energy they carry
        double n = 0, ew = 0, et = 0;
        for (int e = 0; e < c.nE; ++e) { const double x = c.star1(e) * w(e) * w(e); et += x; if (T(e).real() < 0) { ++n; ew += x; } }
        wide = n / c.nE; wide_energy = et > 0 ? ew / et : NAN;
        std::vector<double> mag(static_cast<std::size_t>(c.nE));
        for (int e = 0; e < c.nE; ++e) mag[static_cast<std::size_t>(e)] = std::abs(T(e));
        std::vector<double> sorted = mag; std::nth_element(sorted.begin(), sorted.begin() + c.nE / 10, sorted.end());
        const double q10 = sorted[static_cast<std::size_t>(c.nE / 10)];
        double el = 0; for (int e = 0; e < c.nE; ++e) if (mag[static_cast<std::size_t>(e)] <= q10) el += c.star1(e) * w(e) * w(e);
        weak_energy = et > 0 ? el / et : NAN;
      }
      // its split. INNER: inner products with the modes' 1-forms (the exact part = d(timing), the coexact part the rotation, the rest outside the modes);
      // MESH / MODES: the WEIGHTED fit (each edge's lag counted by its weight: its cross-power |T| or its coherence |T|/√(P_tail P_tip); the linearised
      // magnetic Laplacian), on the whole complex or within the package's exact modes; the rotation is then not separated (rest = rotation + residual)
      coef = nxr::intrinsic::read1(c, md, w).col(0);
      Eigen::VectorXd ce = coef, cc = coef;
      for (std::int64_t k = 0; k < K; ++k) (md.kind[static_cast<std::size_t>(k)] == 0 ? cc : ce)(k) = 0;
      Eigen::VectorXd wex, wco = Eigen::VectorXd::Zero(c.nE), phi, ew = Eigen::VectorXd::Ones(c.nE);
      if (opt.fit == "inner") {
        wex = nxr::intrinsic::synth1(c, md, ce).col(0); wco = nxr::intrinsic::synth1(c, md, cc).col(0);
        phi = nxr::intrinsic::potential(md, ce).col(0);
        const double E = w.dot(c.star1.asDiagonal() * w), Eex = wex.dot(c.star1.asDiagonal() * wex), Eco = wco.dot(c.star1.asDiagonal() * wco);
        share_ex = Eex / E; share_co = Eco / E; share_res = std::max(0.0, 1 - share_ex - share_co);
      } else {
        for (int e = 0; e < c.nE; ++e) {
          const double mag = std::abs(T(e));
          const int tail = c.edges[2 * static_cast<std::size_t>(e)], tip = c.edges[2 * static_cast<std::size_t>(e) + 1];
          const double pp = power(tail) * power(tip);
          ew(e) = opt.weight == "coherence" ? (pp > 0 ? mag / std::sqrt(pp) : 0.0) : mag;
        }
        // DEFECTS (`defects` exclude): edges whose ends' normals are opposed (a fold-over of the mesh) or shorter than 0.1 × the median edge (degenerate):
        // weight 0, counted and reported with the share of the phase field's ⋆1 energy they carried
        if (opt.defects == "exclude") {
          std::vector<double> lens(c.length.data(), c.length.data() + c.nE);
          std::nth_element(lens.begin(), lens.begin() + c.nE / 2, lens.end());
          const double floor_len = 0.1 * lens[static_cast<std::size_t>(c.nE / 2)];
          double et = 0, ex = 0; int nx = 0;
          for (int e = 0; e < c.nE; ++e) {
            const int tail = c.edges[2 * static_cast<std::size_t>(e)], tip = c.edges[2 * static_cast<std::size_t>(e) + 1];
            const double en = c.star1(e) * w(e) * w(e);
            et += en;
            if (N3.row(tail).dot(N3.row(tip)) < 0 || c.length(e) < floor_len) { ew(e) = 0; ++nx; ex += en; }
          }
          excluded = nx; excluded_energy = et > 0 ? ex / et : NAN;
        }
        if (opt.fit == "mesh") phi = nxr::intrinsic::weightedPotentialMesh(c, ew, w);
        else {
          // every exact mode of the package: the phase of a band-filtered field is not confined to the band's own scales (a large-scale timing gradient
          // under a fine band), so the timing map is band-limited by the package, not by the band
          const std::vector<char> use(static_cast<std::size_t>(K), 1);
          phi = nxr::intrinsic::weightedPotentialModes(c, md, use, ew, w);
        }
        wex = c.d0 * phi;
        const Eigen::VectorXd W = c.star1.cwiseProduct(ew);
        const double E = w.dot(W.asDiagonal() * w), R = (w - wex).dot(W.asDiagonal() * (w - wex));
        share_ex = 1 - R / E; share_co = NAN; share_res = R / E;           // in the weights' own metric; the rest is rotation and residual together
      }
      timing = phi / in.omega;                                       // s; a larger value LEADS (z ∝ e^{+iωt})
      // the propagation: −∇φ at the vertices (the exact part, sharpened), rad/m; the speed ω / |∇φ|
      const Eigen::MatrixXd pv = nxr::intrinsic::sharp(c, -wex);
      prop.resize(Vg, 3); speed.resize(Vg);
      for (std::int64_t v = 0; v < Vg; ++v) { prop.row(v) = pv.middleRows(3 * v, 3).transpose() / in.metres; const double k = prop.row(v).norm(); speed(v) = k > 0 ? in.omega / k : NAN; }
      // the stream function (the coexact part's 2-form potential at the vertices), the divergence of the propagation and the curl of the phase field
      Eigen::VectorXd cs = cc;
      for (std::int64_t k = 0; k < K; ++k) if (md.kind[static_cast<std::size_t>(k)] == 2) cs(k) /= md.mu(k);
      stream = opt.fit == "inner" ? Eigen::VectorXd(nxr::intrinsic::vertexDensity(c, md.X2 * cs).col(0)) : Eigen::VectorXd::Constant(Vg, NAN);
      // THE TILES' EXACT INTEGRALS (decision 1 of the measurement pyramid; nxr's measure/integrals.py is the extrinsic form, nxr::integrals, for a vertex field): on the complex the fits are made on, the propagation's FLUX out
      // of each leaf (the net source of -grad(phi): the discrete divergence theorem on the exact part, rad) and the CIRCULATION of the phase field round it (the discrete Stokes theorem on its 1-form, rad: a winding
      // number x 2 pi round a phase defect, which a vertex vector field cannot carry). Exact on the complex, additive up the tile tree; the mean divergence / curl of a tile is the integral over its area.
      // The per-vertex densities are no longer stored (a vertex's divergence is not a measurement; a tile's is).
      {
        leaf_flux = nxr::intrinsic::tileFlux(c, -wex, gcodes.data(), static_cast<int>(PL.leaves));
        leaf_circ = nxr::intrinsic::tileCirculation(c, w, gcodes.data(), static_cast<int>(PL.leaves));
        leaf_power = Eigen::VectorXd::Zero(PL.leaves);                                                   // the band's power, summed over each leaf with the input surface's vertex areas
        for (std::int64_t v = 0; v < kept; ++v) if (gcodes[static_cast<std::size_t>(v)] >= 0) leaf_power(gcodes[static_cast<std::size_t>(v)]) += in_area[static_cast<std::size_t>(pk.support[static_cast<std::size_t>(v)])] * power(v);
      }
      // the inconsistency per vertex: over its edges, the share of ⋆1 |ω|² (× the weight in a weighted fit) the timing map's gradient leaves
      Eigen::VectorXd num = Eigen::VectorXd::Zero(Vg), den = Eigen::VectorXd::Zero(Vg);
      for (int e = 0; e < c.nE; ++e) {
        const double r2 = c.star1(e) * ew(e) * std::pow(w(e) - wex(e), 2), w2 = c.star1(e) * ew(e) * w(e) * w(e);
        for (int end = 0; end < 2; ++end) { const int v = c.edges[2 * static_cast<std::size_t>(e) + static_cast<std::size_t>(end)]; num(v) += r2; den(v) += w2; }
      }
      incons = num.cwiseQuotient(den);
      return 0;
    };
    auto store = [&](const int&) {
      moved_aside = clear_stored(ds, sub, {{"statistic", map_path}, {"statistic", modes_path}}, opt.replace, stem, replaced);
      if (const auto old_field = find_field(sub.db(), sub.id(), FieldQuery{fname, "phase gradient", std::nullopt, false})) sub.remove("field", text(*old_field, "id"));
      const std::string density = in.cpj.value("scaling", std::string("density")) == "density" ? "/Hz" : "";
      const float nan = std::numeric_limits<float>::quiet_NaN();
      const std::vector<std::string> qn = {"timing", "propagation_x", "propagation_y", "propagation_z", "speed", "stream", "inconsistency"};
      const std::int64_t nq = static_cast<std::int64_t>(qn.size());
      std::vector<float> data(static_cast<std::size_t>(Vg * nq), nan);
      for (std::int64_t v = 0; v < kept; ++v) {                     // the bridge (v >= kept) is not cortex: NaN
        const double vals[7] = {timing(v), prop(v, 0), prop(v, 1), prop(v, 2), speed(v) * in.metres, stream(v), incons(v)};
        for (int i = 0; i < 7; ++i) data[static_cast<std::size_t>(v * nq + i)] = static_cast<float>(vals[i]);
      }
      Json pj = stamp(key);
      pj["stage"] = "phasegradient"; pj["by"] = "nxr-neuro phasegradient compute"; pj["producer"] = producer_name("phasegradient compute");
      pj["cross_spectrum"] = opt.cross_spectrum; pj["cross_spectrum_id"] = text(in.crow, "id"); pj["recording"] = in.recording; pj["kernel"] = in.kname; pj["kernel_id"] = text(in.kern, "id");
      pj["band"] = Json{{"name", in.band}, {"lo_hz", in.lo}, {"hi_hz", in.hi}}; pj["omega_rad_s"] = in.omega; pj["signal"] = opt.signal; pj["surface_unit"] = in.unit;
      pj["spatial_band"] = Json{{"spec", spec}, {"label", bd.label}, {"package", pk.stem}, {"modes", K}, {"bank", bank.selection_id}, {"level", bd.level}, {"weyl_piece_area", bank.piece_area}};
      pj["field"] = opt.field;
      pj["edge_lag"] = opt.edge_lag;
      pj["phase_of"] = opt.phase_of;
      pj["fit"] = opt.fit;
      if (opt.fit != "inner") pj["weight"] = opt.weight;
      pj["fold_edges"] = Json{{"rule", opt.fold_edges}, {"criteria", "the ends' normals opposed: the lag read modulo π"}, {"edges", fold_count}};
      pj["defects"] = Json{{"rule", opt.defects}, {"criteria", "the ends' normals opposed (a fold-over), or shorter than 0.1 × the median edge"}, {"excluded_edges", excluded},
                           {"their_energy_share", std::isfinite(excluded_energy) ? Json(excluded_energy) : Json()}};
      pj["shares"] = Json{{"timing", share_ex}, {"rotation", std::isfinite(share_co) ? Json(share_co) : Json()}, {"residual", share_res}};
      pj["edges_beyond_half_pi"] = Json{{"share", wide}, {"energy_share", wide_energy}};
      pj["inconsistency"] = 1 - share_ex;
      pj["complex"] = iname;
      pj["leaves"] = Json{{"selection", PL.name}, {"depth", PL.depth}, {"count", PL.leaves}, {"tree", PL.tree}, {"weights", "the input surface's lumped vertex area (power); the integrals are over the glued surface's faces"}};
      pj["stored"] = Json{{"statistic", map_name}, {"modes_statistic", modes_name}, {"field", fname},
                          {"measures", Json::array({"phase_power", "phase_flux", "phase_circulation"})},
                          {"quantities", Json::array({"timing (s)", "propagation_x/y/z (rad/m)", "speed (m/s)", "stream (rad)", "inconsistency (share)"})}};
      pj["derived_at_read"] = "the mean divergence of a tile = phase_flux / its area, the mean curl = phase_circulation / its area (the leaf areas `w` of the partition); the speed is omega over the propagation's length";
      pj["warnings"] = Json::array({"the speed takes omega at the band's centre frequency",
                                    "a time average over the cross-spectrum's span: waves going opposite ways, or turning, cancel",
                                    "kinematic: the movement of the phase, not of the current; the gradient field is arg T, linear in the lags (no wrapping beyond ±π per edge)",
                                    "band-limited by the package's modes",
                                    "the bridge of the glued surface is not cortex: NaN in every vertex quantity, and in no leaf",
                                    "the timing map, stream function, propagation and inconsistency are global fits over the modes (no additive form): a statistic, not leaf sums; the flux and circulation are exact tile integrals of the propagation and of the phase field on the glued surface's faces (the tangent flow's: the normal part is dropped)",
                                    "a leaf's flux and circulation count only its vertices' faces; a leaf on the edge of the cut (the medial wall) has an open boundary there, so its integrals include the flow across the cut"});
      // what the fits are OF: the cross-spectrum statistic (an older Operator series: the recording's Field)
      std::optional<std::string> of_stat, of_field;
      if (in.crow.count("measure")) of_stat = text(in.crow, "id");
      else if (const auto rf = find_field(sub.db(), sub.id(), FieldQuery{in.recording, "recording"})) of_field = text(*rf, "id");
      else of_stat = text(in.crow, "id");
      const std::string what_all = " of the " + bd.label + " band of the current of " + in.recording + " by " + in.kname + " \xe2\x80\x94 " + in.band;
      {   // the fits: [vertex, quantity] on the glued surface (the bridge NaN)
        StatisticSpec st;
        st.name = map_name; st.path = map_path; st.measure = "phase_lag_split"; st.method = "phase_split"; st.value_type = "scalar";
        st.of_field_id = of_field; st.of_statistic_id = of_stat;
        Json qa = Json::array(); for (const auto& q : qn) qa.push_back(q);
        st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", gid}, {"size", Vg}}, Json{{"name", "quantity"}, {"values", qa}, {"size", nq}}});
        st.data = view(data, {Vg, nq});
        st.label = "phase lag split";
        st.description = "the timing map (s; a larger value leads), its propagation -grad(phi) (rad/m), the phase speed (m/s), the stream function of the phase field's rotation (rad) and the inconsistency (the share of the phase field the timing map leaves)" + what_all;
        st.options = Json{{"spatial_band", bd.label}, {"fit", opt.fit}, {"edge_lag", opt.edge_lag}, {"phase_of", opt.phase_of}, {"fold_edges", opt.fold_edges}, {"defects", opt.defects}};
        st.producer = pj;
        sub.statistic(std::move(st));
      }
      {   // the phase 1-form's coefficients in the modes' 1-forms, on the eigen axis
        std::vector<float> cf(static_cast<std::size_t>(K));
        for (std::int64_t k = 0; k < K; ++k) cf[static_cast<std::size_t>(k)] = static_cast<float>(coef(k));
        StatisticSpec st;
        st.name = modes_name; st.path = modes_path; st.measure = "phase_lag_split"; st.method = "phase_split"; st.value_type = "scalar";
        st.of_field_id = of_field; st.of_statistic_id = of_stat;
        st.axes = Json::array({Json{{"name", "mode"}, {"manifold_id", pk.axis_id}, {"size", K}}});
        st.data = view(cf, {K});
        st.label = "phase lag split coefficients";
        st.description = "the phase 1-form's coefficients in the modes' 1-forms" + what_all;
        st.options = Json{{"spatial_band", bd.label}, {"fit", opt.fit}, {"edge_lag", opt.edge_lag}};
        st.producer = pj;
        sub.statistic(std::move(st));
      }
      // the leaves: a byte-less Field and its array-backed measurements on the leaf partition, within the frequency band
      pyramid_leaves_record(sub, PL, Json(nullptr));
      const nxr::io::Json band_ref = nxr::bst::band_ref(sub.db(), in.band, in.lo, in.hi);
      FieldSpec fs;
      fs.name = fname; fs.kind = "phase gradient"; fs.manifold_id = in.surface_id; fs.value_type = "scalar";
      fs.cols.set("function", "phase gradient").set("by_operator_id", text(in.kern, "id"))
          .set("description", "the exact tile integrals and the power of the phase field of the " + bd.label + " band of the current of " + in.recording + " by " + in.kname + " \xe2\x80\x94 " + in.band + ", stored on the leaves of " + PL.name + " only");
      fs.cols.json("producer_json", pj);
      const std::string field_id = sub.field(std::move(fs));
      // the within key: the frequency band, the spatial band (the stored bank's level) or both as the joint bands Selection (frequency bank x spatial bank, code = band * n_spatial + level)
      std::string within_joint;
      std::int64_t n_spatial = 1;
      if (bd.level >= 0 && !band_ref["selection_id"].is_null()) {
        const std::string fbank = band_ref["selection_id"].get<std::string>();
        within_joint = nxr::io::band_key_selection(sub, fbank, bank.selection_id, text(*sub.db().read("selection", fbank), "name") + " \xc3\x97 " + nxr::io::spatial_bands_name(in.surface));
        n_spatial = static_cast<std::int64_t>(bank.edges.size());
      }
      const std::string group = PL.node + "/phase/" + nxr::io::safe_name(fname);
      struct Out { const char* rel; const char* measure; const Eigen::VectorXd* v; std::string unit; };
      const Out outs[3] = {{"power", "phase_power", &leaf_power, std::string("A\xc2\xb2\xc2\xb7m\xc2\xb2") + density}, {"flux", "phase_flux", &leaf_flux, "rad"}, {"circulation", "phase_circulation", &leaf_circ, "rad"}};
      for (const Out& o : outs) {
        const std::string path = group + "/" + o.rel;
        std::vector<double> vals(o.v->data(), o.v->data() + o.v->size());
        sub.write_array(path, view(vals, {static_cast<std::int64_t>(PL.leaves)}));
        Cols cm;
        cm.set("measure", o.measure).set("of_field_id", field_id).set("array_path", path).set("unit", o.unit).set("computed_by", "nxr-neuro phasegradient compute");
        if (bd.level >= 0 && !band_ref["selection_id"].is_null()) cm.set("within_selection_id", within_joint).set("within_code", nxr::io::band_key_code(band_ref["code"].get<std::int64_t>(), bd.level, n_spatial));
        else if (bd.level >= 0) cm.set("within_selection_id", bank.selection_id).set("within_code", bd.level);   // the spatial band alone (the frequency band is not in the stored bank)
        else if (!band_ref["selection_id"].is_null()) cm.set("within_selection_id", band_ref["selection_id"].get<std::string>()).set("within_code", band_ref["code"].get<std::int64_t>());
        nxr::io::selection::save_measurement(sub, text(*PL.part, "id"), cm);
      }
    };
    const Outcome<int> o = cached<int>(sub, "statistic", map_path, key, load, compute, store);
    res.hit = results.empty() ? o.hit : (res.hit && o.hit);
    res.seconds += o.seconds;
    if (o.hit) say(Json{{"stage", "cache"}, {"hit", true}, {"cross_spectrum", opt.cross_spectrum}, {"spatial_band", bd.label}});
    results.push_back(Json{{"spatial_band", bd.label}, {"name", stem}, {"statistic", map_name}, {"modes_statistic", modes_name}, {"field", fname}, {"leaves", PL.name}, {"depth", PL.depth}, {"measures", Json::array({"phase_power", "phase_flux", "phase_circulation"})}, {"cache", o.hit ? "hit" : "miss"},
                           {"shares", std::isfinite(share_ex) ? Json{{"timing", share_ex}, {"rotation", std::isfinite(share_co) ? Json(share_co) : Json()}, {"residual", share_res}} : Json()},
                           {"edges_beyond_half_pi", std::isfinite(wide) ? Json{{"share", wide}, {"energy_share", wide_energy}} : Json()},
                           {"fold_edges", Json{{"rule", opt.fold_edges}, {"edges", fold_count}}},
                           {"defects", Json{{"rule", opt.defects}, {"excluded_edges", excluded}, {"their_energy_share", std::isfinite(excluded_energy) ? Json(excluded_energy) : Json()}}},
                           {"weakest_tenth_energy_share", std::isfinite(weak_energy) ? Json(weak_energy) : Json()},
                           {"band_current", std::isfinite(leak_band) ? Json{{"outside_band", leak_band}, {"outside_modes", leak_modes}} : Json()},
                           {"inconsistency", std::isfinite(share_ex) ? Json(1 - share_ex) : Json()}, {"replaced", replaced},
                           {"moved_aside", moved_aside.empty() ? Json() : Json(moved_aside)}});
  }
  res.info = Json{{"cross_spectrum", opt.cross_spectrum}, {"band", in.band}, {"kernel", in.kname}, {"signal", opt.signal}, {"package", pk.stem}, {"glued_surface", pk.glued.name},
                  {"complex", iname}, {"spatial_bands", results}};
  return res;
}

}  // namespace

CyclePatternResult cyclepattern_compute(DatasetWriter& ds, SubjectWriter& sub, const CyclePatternOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.patterns < 1) invalid("cyclepattern: --patterns is at least 1");
  // ── preparation: the series, the kernel (free, on the series' channels), the package; every check before any maths ──
  const auto snode = find_cross_spectrum_series(sub, opt.cross_spectrum);
  if (!snode) missing(sub.name() + ": no cross-spectrum series named '" + opt.cross_spectrum + "' (`nxr-neuro crossspectrum compute --estimator analytic` first)");
  const std::optional<Row> crow = snode->row;                     // the statistic's (or an older Operator's) row: its id and modified time key the result
  const Json cpj = snode->producer;
  if (!cpj.is_object() || !cpj.contains("band_name") || !cpj.contains("recording")) invalid(opt.cross_spectrum + ": its producer record does not name its band and recording");
  const std::string band = cpj["band_name"].get<std::string>(), recording = cpj["recording"].get<std::string>();
  for (const auto& t : opt.timing_of) if (t != "stream" && t != "potential" && t != "normal") invalid("cyclepattern: --timing-of names stream, potential or normal (got '" + t + "')");
  const double omega = cpj.contains("band_lo_hz") && cpj.contains("band_hi_hz") ? M_PI * (cpj["band_lo_hz"].get<double>() + cpj["band_hi_hz"].get<double>()) : NAN;
  if (!opt.timing_of.empty() && !std::isfinite(omega)) invalid(opt.cross_spectrum + ": its producer record does not give the band's edges (the timing needs its centre frequency)");
  Row kern;
  if (opt.kernel) {
    const auto r = find_operator(sub.db(), sub.id(), "inverse kernel", *opt.kernel);
    if (!r) missing(sub.name() + ": no inverse kernel named '" + *opt.kernel + "'");
    kern = *r;
  } else {
    const auto rs = stored_operators(sub.db(), sub.id(), "inverse kernel");
    if (rs.size() != 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " inverse kernels; name one with --kernel");
    kern = rs[0];
  }
  const std::string kname = text(kern, "name"), surface_id = text(kern, "to_manifold_id");
  double metres = NAN;
  if (!opt.timing_of.empty()) {
    const auto urow = manifold_coordinate_unit(sub.db(), surface_id);
    const std::string unit = urow;
    if (unit != "m" && unit != "mm") invalid("cyclepattern: the surface's coordinates are in '" + unit + "', not m or mm: the speed needs a length unit");
    metres = unit == "mm" ? 1e-3 : 1.0;
  }
  if (opt_text(kern, "to_value_type").value_or("vector3") == "scalar") invalid(kname + ": a fixed-orientation kernel: the cycle patterns read the current's components (a free-orientation kernel; no orientation is assumed)");
  if (text(kern, "from_manifold_id") != snode->channels_manifold) invalid(kname + " is on other channels than " + opt.cross_spectrum);
  const auto ksel = sub.db().read("selection", text(kern, "from_selection_id"));
  if (!ksel || is_null(*ksel, "array_path")) invalid(kname + ": its channels (from_selection) are not stored");
  const std::vector<std::int32_t> chans = read_all<std::int32_t>(sub.store(), text(*ksel, "array_path"));
  const std::int64_t M = static_cast<std::int64_t>(chans.size());
  std::vector<std::int64_t> kshape;
  const std::vector<float> kf = read_all<float>(sub.store(), text(kern, "path"), &kshape);
  if (kshape.size() != 2 || kshape[1] != M || kshape[0] % 3 != 0) invalid(kname + ": its array is not [3 x sources rows x " + std::to_string(M) + " channels]");
  const auto smf = sub.db().read("manifold", surface_id);
  if (!smf || integer(*smf, "n_vertices") != kshape[0] / 3) invalid(kname + ": its sources and its target surface's vertices differ in number");
  const HodgePackage pk = resolve_hodge_package(sub, *smf, opt.package, std::nullopt, "cyclepattern");
  const std::int64_t Vg = pk.Vg, K = pk.K, kept = pk.kept;
  const SpatialBank bank = opt.spatial_band == "modes" ? SpatialBank{} : spatial_bank_of(ds, sub, text(*smf, "name"));   // THE ONE bank, as phasegradient reads it
  const std::vector<SpatialBand> bands = spatial_bands(pk, opt.spatial_band, "cyclepattern", bank);
  // THE WINDOW read (the time-tile amendment, 2026-11-13): the stored tiles rolled up to `level` (or `window_seconds` snapped to the nearest level)
  const double cs_sfreq = cpj.value("sfreq", 0.0);
  int level = -1;
  if (opt.level) level = *opt.level;
  else if (opt.window_seconds) {
    if (!(*opt.window_seconds > 0) || !(cs_sfreq > 0)) invalid("cyclepattern: --window-seconds is a positive duration, on a series that records its sampling rate");
    level = nxr::io::constantq::levelOf(*opt.window_seconds, cs_sfreq);
  }
  const CrossSpectrumWindows win = crossspectrum_windows_read(sub, opt.cross_spectrum, chans, "stored", level >= 0 ? std::optional<int>(level) : std::nullopt);   // refuses a channel the series does not cover
  const std::size_t W = win.sum.size();
  if (W == 0) invalid(opt.cross_spectrum + ": the series holds no window");
  std::string base = opt.cross_spectrum;
  if (base.size() > 8 && base.compare(base.size() - 8, 8, "_windows") == 0) base.resize(base.size() - 8);
  const std::string stem0 = opt.name ? *opt.name : base + "_cycle";
  const std::string gid = pk.glued.id, iname = pk.glued.name + "_intrinsic";
  std::optional<KernelForms> kfm;
  Json results = Json::array();
  CyclePatternResult res;
  res.hit = true;
  for (const SpatialBand& bd : bands) {
    const std::string stem = stem0 + "_" + bd.label;
    // the band's rows: (mode, part), part 2 the normal current of a coexact mode, part 1 the tangent current of any mode
    std::vector<std::pair<int, int>> rows;
    std::vector<double> rgain;
    for (std::int64_t k = 0; k < K; ++k) {
      const double g = bd.gain[static_cast<std::size_t>(k)];
      if (std::abs(g) < 1e-3) continue;
      if (pk.kind_all[static_cast<std::size_t>(k)] == 2) { rows.push_back({static_cast<int>(k), 2}); rgain.push_back(g); }
      rows.push_back({static_cast<int>(k), 1}); rgain.push_back(g);
    }
    const std::int64_t n = static_cast<std::int64_t>(rows.size());
    const int q = static_cast<int>(std::min<std::int64_t>(opt.patterns, n));
    // THE STATISTICS of this decomposition (step 5 of the measurement pyramid): the patterns, ONE statistic of axes vertex x pattern x quantity (stream, potential, normal), complex; with their coefficients
    // on the eigen axis (`<stem>_modes`) and, when asked, their lags (`<stem>_lags`): never one Field per pattern and quantity
    const std::string stat_name = stem, modes_name = stem + "_modes", lags_name = stem + "_lags";
    const std::string stat_path = "statistics/" + nxr::io::safe_name(stat_name), modes_path = "statistics/" + nxr::io::safe_name(modes_name), lags_path = "statistics/" + nxr::io::safe_name(lags_name);
    CacheKey key;
    key.pipeline = "cyclepattern";
    key.params = Json{{"name", stem}, {"store", "statistic"}, {"spatial_band", bd.label}, {"weyl_piece_area", bank.piece_area}, {"patterns", q}, {"rows", "coexact normal (2-form) and every tangent (1-form) coefficient, times the gain, |g| >= 1e-3"},
                      {"phase", "each pattern's frame of largest variance at phase 0"}, {"timing_of", opt.timing_of}, {"window_level", win.level}};
    key.input = Json{{"cross_spectrum", text(*crow, "id")}, {"cross_spectrum_modified", opt_text(*crow, "modified_utc").value_or("")}, {"kernel", text(kern, "id")},
                     {"kernel_modified", opt_text(kern, "modified_utc").value_or("")}, {"package", text(pk.opr, "id")}};
    key.compute = NXR_COMPUTE_VERSION;
    nxr::patterns::Leading span;
    std::vector<nxr::patterns::Leading> per;
    std::vector<Eigen::VectorXd> sim_span, sim_first;
    std::vector<nxr::intrinsic::PatternFields> fields;
    std::vector<nxr::intrinsic::PhaseSplit> splits;                       // [q x timing_of]
    std::string moved_aside;
    bool replaced = false;
    auto load = [&](const Row&) { return 0; };
    auto compute = [&]() {
      if (!kfm) kfm = kernel_forms(sub, pk, kf, kshape, M, kname, say);
      say(Json{{"stage", "cache"}, {"hit", false}, {"cross_spectrum", opt.cross_spectrum}, {"kernel", kname}, {"spatial_band", bd.label}, {"coefficients", n}});
      Eigen::MatrixXd Bb(n, M);
      for (std::int64_t i = 0; i < n; ++i) {
        const auto [k, part] = rows[static_cast<std::size_t>(i)];
        Bb.row(i) = rgain[static_cast<std::size_t>(i)] * (part == 2 ? kfm->A2.row(k) : kfm->A1.row(k));
      }
      // the span: the window partition's sum over its counts (a measurement over the partition, never a re-read)
      Eigen::MatrixXcd Cs = Eigen::MatrixXcd::Zero(M, M);
      std::int64_t total = 0;
      for (std::size_t w = 0; w < W; ++w) { Cs += win.sum[w]; total += win.count[w]; }
      span = nxr::patterns::leading(nxr::algebra::congruence(Bb, Cs / static_cast<double>(total)), q);
      for (std::size_t w = 0; w < W; ++w) {
        per.push_back(nxr::patterns::leading(nxr::algebra::congruence(Bb, win.sum[w] / static_cast<double>(win.count[w])), q));
        sim_span.push_back(nxr::patterns::similarity(per.back().vectors, span.vectors));
        sim_first.push_back(nxr::patterns::similarity(per.back().vectors, per.front().vectors));
      }
      for (int i = 0; i < q; ++i) fields.push_back(nxr::intrinsic::patternFields(kfm->c, kfm->md, rows, span.vectors.col(i)));
      // each pattern's lags: the phase 1-form of the field asked for, split in the modes
      for (int i = 0; i < q; ++i)
        for (const auto& t : opt.timing_of) {
          const auto& pf = fields[static_cast<std::size_t>(i)];
          const Eigen::VectorXcd& x = t == "stream" ? pf.S : t == "potential" ? pf.A : pf.rho;
          splits.push_back(nxr::intrinsic::splitPhase(kfm->c, kfm->md, nxr::intrinsic::phaseOneForm(kfm->c, x)));
        }
      return 0;
    };
    auto store = [&](const int&) {
      std::vector<std::pair<const char*, std::string>> have = {{"statistic", stat_path}, {"statistic", modes_path}};
      if (!opt.timing_of.empty()) have.push_back({"statistic", lags_path});
      moved_aside = clear_stored(ds, sub, have, opt.replace, stem, replaced);
      Json eig = Json::array(), tr = Json::array(), ss = Json::array(), sf = Json::array();
      for (std::size_t w = 0; w < W; ++w) {
        eig.push_back(std::vector<double>(per[w].values.data(), per[w].values.data() + q)); tr.push_back(per[w].trace);
        ss.push_back(std::vector<double>(sim_span[w].data(), sim_span[w].data() + q)); sf.push_back(std::vector<double>(sim_first[w].data(), sim_first[w].data() + q));
      }
      Json standing = Json::array();
      for (int i = 0; i < q; ++i) { const Eigen::VectorXcd m = span.vectors.col(i); standing.push_back(m.imag().squaredNorm() / m.real().squaredNorm()); }
      Json pj = stamp(key);
      pj["stage"] = "cyclepattern"; pj["by"] = "nxr-neuro cyclepattern compute"; pj["producer"] = producer_name("cyclepattern compute");
      pj["cross_spectrum"] = opt.cross_spectrum; pj["cross_spectrum_id"] = text(*crow, "id"); pj["recording"] = recording; pj["band"] = cpj.value("band", Json()); pj["band_name"] = band;
      pj["kernel"] = kname; pj["kernel_id"] = text(kern, "id");
      pj["spatial_band"] = Json{{"spec", opt.spatial_band}, {"label", bd.label}, {"package", pk.stem}, {"modes", K}, {"bank", bank.selection_id}, {"level", bd.level}, {"weyl_piece_area", bank.piece_area}, {"coefficients", n}};
      pj["complex"] = iname;
      pj["span"] = Json{{"eigenvalues", std::vector<double>(span.values.data(), span.values.data() + q)}, {"trace", span.trace}, {"windows", static_cast<std::int64_t>(W)}};
      pj["window_level"] = win.level; pj["window_level_stored"] = cpj.value("window_level", -1);
      pj["window_s"] = win.level >= 0 && cs_sfreq > 0 ? Json(static_cast<double>(std::int64_t(1) << win.level) / cs_sfreq) : Json();
      pj["window_rule"] = opt.level ? "--level" : opt.window_seconds ? "--window-seconds snapped to the nearest level (round(log2(seconds x sfreq)))" : "the stored level";
      pj["windows"] = Json{{"index", win.index}, {"eigenvalues", eig}, {"trace", tr}, {"similarity_to_span", ss}, {"similarity_to_first", sf}};
      pj["standing_share"] = standing;
      if (!opt.timing_of.empty()) {
        Json lags = Json::array();
        std::size_t si = 0;
        for (int i = 0; i < q; ++i)
          for (const auto& t : opt.timing_of) { const auto& sp = splits[si++]; lags.push_back(Json{{"pattern", i + 1}, {"of", t}, {"timing", sp.shareTiming}, {"rotation", sp.shareRotation}, {"residual", sp.shareResidual}}); }
        pj["lags"] = Json{{"omega_rad_s", omega}, {"edge_lag", "arg(x_tip conj x_tail) of the pattern's field on each intrinsic edge"}, {"shares", lags}};
      }
      pj["phase"] = "each pattern's phase origin is free: phase 0 is its frame of largest variance (m e^{i theta}, theta = -arg(sum m_k^2) / 2)";
      pj["layout"] = "per pattern i the span's i-th eigenvector m of C_Phi = B C B^T; shares lambda / trace are derived at read; similarity |<m_w, m>| matches patterns by rank (meaningful for the leading pattern, fragile when two eigenvalues are close)";
      pj["warnings"] = Json::array({"a pattern's phase origin is arbitrary: phase 0 is its frame of largest variance",
                                    "a window holds about the series' independent_estimates_per_window independent complex samples: a band with more coefficients than that has at most that many patterns per window, so one window's shares are bounded below by about 1 / (B T), not 1 / n",
                                    "the patterns are of the inverse's current: what the kernel spreads is in them",
                                    "the bridge of the glued surface is not cortex: NaN in every vertex field"});
      const float nan = std::numeric_limits<float>::quiet_NaN();
      // what the decomposition is OF: the cross-spectrum statistic it decomposes (an older Operator series: the recording's Field)
      std::optional<std::string> of_stat, of_field;
      if (crow->count("measure")) of_stat = text(*crow, "id");
      else if (const auto rf = find_field(sub.db(), sub.id(), FieldQuery{recording, "recording"})) of_field = text(*rf, "id");
      else of_stat = text(*crow, "id");
      const Json pattern_axis = [&] { Json a = Json::array(); for (int i = 1; i <= q; ++i) a.push_back(i); return a; }();
      const std::string what_all = " of the " + bd.label + " band's consistent cycle patterns, of the current of " + recording + " by " + kname + " \xe2\x80\x94 " + band;
      {   // the patterns: [vertex, pattern, quantity] complex on the glued surface (the bridge NaN)
        std::vector<float> data(static_cast<std::size_t>(Vg * q * 3 * 2), nan);
        for (int i = 0; i < q; ++i) {
          const auto& pf = fields[static_cast<std::size_t>(i)];
          const Eigen::VectorXcd* parts[3] = {&pf.S, &pf.A, &pf.rho};
          for (int j = 0; j < 3; ++j)
            for (std::int64_t x = 0; x < kept; ++x) {
              data[static_cast<std::size_t>(((x * q + i) * 3 + j) * 2)] = static_cast<float>((*parts[j])(x).real());
              data[static_cast<std::size_t>(((x * q + i) * 3 + j) * 2 + 1)] = static_cast<float>((*parts[j])(x).imag());
            }
        }
        StatisticSpec st;
        st.name = stat_name; st.path = stat_path; st.measure = "cycle_pattern"; st.method = "eigendecomposition"; st.value_type = "complex";
        st.of_field_id = of_field; st.of_statistic_id = of_stat;
        st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", gid}, {"size", Vg}}, Json{{"name", "pattern"}, {"values", pattern_axis}, {"size", q}},
                               Json{{"name", "quantity"}, {"values", Json::array({"stream", "potential", "normal"})}, {"size", 3}}});
        st.data = view(data, {Vg, static_cast<std::int64_t>(q), 3, 2});
        st.label = "cycle pattern";
        st.description = "the stream function S (the tangent current's divergence-free part), the potential A (its curl-free part) and the normal current of the leading patterns" + what_all;
        st.options = Json{{"patterns", q}, {"spatial_band", bd.label}, {"window_level", win.level}};
        st.producer = pj;
        sub.statistic(std::move(st));
      }
      {   // their coefficients on the package's eigen axis: [mode, pattern, part (tangent, normal)] complex, 0 outside the band
        std::vector<float> data(static_cast<std::size_t>(K * q * 2 * 2), 0.0f);
        for (int i = 0; i < q; ++i)
          for (std::int64_t r = 0; r < n; ++r) {
            const auto [k, part] = rows[static_cast<std::size_t>(r)];
            const std::complex<double> c = span.vectors(r, i);
            const std::int64_t o = ((k * q + i) * 2 + (part == 2 ? 1 : 0)) * 2;
            data[static_cast<std::size_t>(o)] = static_cast<float>(c.real()); data[static_cast<std::size_t>(o + 1)] = static_cast<float>(c.imag());
          }
        StatisticSpec st;
        st.name = modes_name; st.path = modes_path; st.measure = "cycle_pattern"; st.method = "eigendecomposition"; st.value_type = "complex";
        st.of_field_id = of_field; st.of_statistic_id = of_stat;
        st.axes = Json::array({Json{{"name", "mode"}, {"manifold_id", pk.axis_id}, {"size", K}}, Json{{"name", "pattern"}, {"values", pattern_axis}, {"size", q}},
                               Json{{"name", "part"}, {"values", Json::array({"tangent", "normal"})}, {"size", 2}}});
        st.data = view(data, {K, static_cast<std::int64_t>(q), 2, 2});
        st.label = "cycle pattern coefficients";
        st.description = "the coefficients of the tangent current in the modes' 1-forms and of the normal current in the coexact modes' 2-forms (times the band's gain)" + what_all;
        st.options = Json{{"patterns", q}, {"spatial_band", bd.label}, {"window_level", win.level}};
        st.producer = pj;
        sub.statistic(std::move(st));
      }
      if (!opt.timing_of.empty()) {   // the lags of each pattern's fields: [vertex, pattern, field, (timing s, speed m/s, inconsistency)] on the glued surface (the bridge NaN)
        const std::int64_t nt = static_cast<std::int64_t>(opt.timing_of.size());
        std::vector<float> data(static_cast<std::size_t>(Vg * q * nt * 3), nan);
        std::size_t si = 0;
        for (int i = 0; i < q; ++i)
          for (std::int64_t t = 0; t < nt; ++t) {
            const auto& sp = splits[si++];
            for (std::int64_t x = 0; x < kept; ++x) {
              const double k = sp.propagation.middleRows(3 * x, 3).norm() / metres;
              const std::size_t o = static_cast<std::size_t>(((x * q + i) * nt + t) * 3);
              data[o] = static_cast<float>(sp.potential(x) / omega);
              data[o + 1] = static_cast<float>(k > 0 ? omega / k : NAN);
              data[o + 2] = static_cast<float>(sp.inconsistency(x));
            }
          }
        StatisticSpec st;
        st.name = lags_name; st.path = lags_path; st.measure = "cycle_pattern_lags"; st.method = "eigendecomposition"; st.value_type = "scalar";
        st.of_field_id = of_field; st.of_statistic_id = of_stat;
        Json tn = Json::array(); for (const auto& t : opt.timing_of) tn.push_back(t);
        st.axes = Json::array({Json{{"name", "vertex"}, {"manifold_id", gid}, {"size", Vg}}, Json{{"name", "pattern"}, {"values", pattern_axis}, {"size", q}},
                               Json{{"name", "of"}, {"values", tn}, {"size", nt}}, Json{{"name", "lag"}, {"values", Json::array({"timing", "speed", "inconsistency"})}, {"size", 3}}});
        st.data = view(data, {Vg, static_cast<std::int64_t>(q), nt, 3});
        st.label = "cycle pattern lags";
        st.description = "the timing map (s; a larger value leads), the phase speed (m/s) and the inconsistency (the share of the lags the timing map leaves) of each pattern's fields" + what_all;
        st.options = Json{{"patterns", q}, {"spatial_band", bd.label}, {"timing_of", opt.timing_of}};
        st.producer = pj;
        sub.statistic(std::move(st));
      }
    };
    const Outcome<int> o = cached<int>(sub, "statistic", stat_path, key, load, compute, store);
    res.hit = res.hit && o.hit;
    res.seconds += o.seconds;
    if (o.hit) say(Json{{"stage", "cache"}, {"hit", true}, {"cross_spectrum", opt.cross_spectrum}, {"spatial_band", bd.label}});
    Json r{{"spatial_band", bd.label}, {"name", stem}, {"coefficients", n}, {"patterns", q}, {"statistic", stat_name}, {"coefficients_statistic", modes_name}, {"lags_statistic", opt.timing_of.empty() ? Json() : Json(lags_name)}, {"of", crow->count("measure") ? "the cross-spectrum statistic" : "the recording (an older cross-spectrum Operator)"}, {"cache", o.hit ? "hit" : "miss"},
           {"replaced", replaced}, {"moved_aside", moved_aside.empty() ? Json() : Json(moved_aside)}};
    if (!o.hit) {
      Json shares = Json::array(), lead = Json::array(), sspan = Json::array(), sfirst = Json::array();
      for (int i = 0; i < q; ++i) shares.push_back(span.values(i) / span.trace);
      for (std::size_t w = 0; w < W; ++w) { lead.push_back(per[w].values(0) / per[w].trace); sspan.push_back(sim_span[w](0)); sfirst.push_back(sim_first[w](0)); }
      if (!opt.timing_of.empty()) {
        Json lags = Json::array();
        std::size_t si = 0;
        for (int i = 0; i < q; ++i) for (const auto& t : opt.timing_of) { const auto& sp = splits[si++]; lags.push_back(Json{{"pattern", i + 1}, {"of", t}, {"timing", sp.shareTiming}, {"rotation", sp.shareRotation}, {"residual", sp.shareResidual}}); }
        r["lags"] = lags;
      }
      r["span_shares"] = shares; r["window_leading_share"] = lead; r["window_similarity_to_span"] = sspan; r["window_similarity_to_first"] = sfirst;
    }
    results.push_back(r);
  }
  res.info = Json{{"cross_spectrum", opt.cross_spectrum}, {"band", band}, {"kernel", kname}, {"package", pk.stem}, {"glued_surface", pk.glued.name}, {"complex", iname}, {"windows", static_cast<std::int64_t>(W)},
                  {"spatial_bands", results}};
  return res;
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

// ── the apparent motion of a source map ──────────────────────────────────────────────────────────────────────────

namespace {
const char* family_name(nxr::timebank::Family f) {
  using F = nxr::timebank::Family;
  return f == F::Itersine ? "itersine" : f == F::Meyer ? "meyer" : f == F::Morlet ? "morlet" : f == F::Morse ? "morse" : f == F::Amor ? "amor" : "bump";
}
}  // namespace

OpticalflowComputeResult opticalflow_compute(DatasetWriter& ds, SubjectWriter& sub, const OpticalflowComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  const RecordingInput rec = read_recording(sub, opt.recording);
  if (!std::isnan(opt.alpha) && !(opt.alpha > 0)) invalid("opticalflow: alpha must be positive");
  if (!opt.family) invalid("opticalflow: the filterbank family is required (itersine, meyer, morlet, morse, amor or bump)");
  if (!(opt.gain > 0)) invalid("opticalflow: the filterbank gain is required and positive (2 makes |c| the physical amplitude)");
  if (!(opt.freq > 0) || !(opt.freq < rec.sfreq / 2)) invalid("opticalflow: freq must satisfy 0 < freq < Nyquist (" + std::to_string(rec.sfreq / 2) + " Hz)");
  if (!(opt.resolution > 0)) invalid("opticalflow: resolution must be positive");
  Row kern;
  if (opt.kernel) {
    const auto r = find_operator(sub.db(), sub.id(), "inverse kernel", *opt.kernel);
    if (!r) missing(sub.name() + ": no inverse kernel named '" + *opt.kernel + "'");
    kern = *r;
  } else {
    const auto rs = stored_operators(sub.db(), sub.id(), "inverse kernel");
    if (rs.size() != 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " inverse kernels; name one with --kernel");
    kern = rs[0];
  }
  const std::string kname = text(kern, "name"), surface_id = text(kern, "to_manifold_id");
  if (text(kern, "from_manifold_id") != rec.channels_id) invalid(kname + " is on other channels than " + opt.recording);
  const auto ksel = sub.db().read("selection", text(kern, "from_selection_id"));
  if (!ksel || is_null(*ksel, "array_path")) invalid(kname + ": its channels (from_selection) are not stored");
  const std::vector<std::int32_t> chans = read_all<std::int32_t>(sub.store(), text(*ksel, "array_path"));
  const std::int64_t M = static_cast<std::int64_t>(chans.size());
  for (std::int32_t c : chans) if (c < 0 || c >= rec.n_channels) invalid(kname + ": a channel index is outside the recording's channels");
  std::vector<std::int64_t> kshape;
  const std::vector<float> kf = read_all<float>(sub.store(), text(kern, "path"), &kshape);
  if (kshape.size() != 2 || kshape[1] != M) invalid(kname + ": its array is not [rows x " + std::to_string(M) + " channels]");
  const int orientations = opt_text(kern, "to_value_type").value_or("vector3") == "scalar" ? 1 : 3;
  const auto smf = sub.db().read("manifold", surface_id);
  if (!smf) missing(kname + ": its target surface is not stored");
  const std::int64_t N = integer(*smf, "n_vertices");
  if (kshape[0] != orientations * N) invalid(kname + ": " + std::to_string(kshape[0]) + " rows for " + std::to_string(N) + " sources");
  const HodgePackage pk = resolve_hodge_package(sub, *smf, opt.package, std::nullopt, "opticalflow");
  const std::int64_t K = pk.K;
  // the surface's frames, when stored (nxr-neuro frames compute): the time-mean velocity is then also reported in ANATOMICAL components (anterior, superior-or-medial)
  const StoredFrames sfr = read_stored_frames(sub, text(*smf, "name"), N);
  const bool anatomical = sfr.canonical && sfr.anatomical && sfr.anatomy.reference == "canonical frames";

  const nxr::io::Span span = span_of(rec, opt.chunks, opt.start, opt.stop);   // what is loaded: one chunk by default (the chunk service)
  const std::int64_t s0 = span.s0, s1 = span.s1;
  [[maybe_unused]] const double start_s = rec.t0 + static_cast<double>(s0) / rec.sfreq, stop_s = rec.t0 + static_cast<double>(s1) / rec.sfreq;
  // the time filterbank (analytic, a real recording), designed once on a grid of the stated resolution; the member nearest to freq
  namespace tb = nxr::timebank;
  tb::Design bd(*opt.family, opt.gain, tb::Sides::Analytic);
  bd.centres = opt.centres;
  if (*opt.family == tb::Family::Morlet && bd.centres.empty()) bd.centres = {opt.freq};
  bd.cycles = opt.cycles; bd.voices = opt.voices; bd.anchor = opt.anchor; bd.limits = opt.limits; bd.halfWidth = opt.half_width;
  bd.gamma = opt.gamma; bd.timeBandwidth = opt.time_bandwidth;
  const std::size_t nfft = nxr::fft::nextFastLen(static_cast<std::size_t>(std::ceil(rec.sfreq / opt.resolution)), true);
  const tb::TimeBank bank = tb::design(bd, nfft, rec.sfreq);
  nxr::wt::Options wo;
  wo.oversample = opt.oversample;
  {
    const nxr::io::DType rd = ArrayReader::open(sub.store(), text(rec.field, "path")).dtype();
    wo.tolerance = opt.tolerance ? *opt.tolerance : (rd == nxr::io::DType::Float64 ? 1e-12 : 1e-7);
    if (!(wo.tolerance > 0) || !(wo.tolerance < 1)) invalid("opticalflow: tolerance must be in (0, 1)");
  }
  nxr::wt::Wt wt(bank, static_cast<int>(M), wo);
  const nxr::wt::Plan& plan = wt.plan();
  const Eigen::VectorXd centres_hz = tb::centerFrequencies(bank);
  int member = 0;
  for (int m = 1; m < bank.size(); ++m) if (std::abs(centres_hz[m] - opt.freq) < std::abs(centres_hz[member] - opt.freq)) member = m;
  const double fc = centres_hz[member], rate = plan.rate[member];     // the member's OWN rate: its coefficients are the frames
  const std::int64_t mp = plan.p[member], mq = plan.q[member];        // frame k sits at k * mp / mq samples of the fed stream
  const std::int64_t lb = plan.margin, la = plan.margin;
  const std::int64_t r0 = std::max<std::int64_t>(0, s0 - lb), r1 = std::min<std::int64_t>(rec.n_samples, s1 + la);
  const bool margins_complete = r0 == s0 - lb && r1 == s1 + la;
  const std::int64_t ceilk0 = (((s0 - r0) * mq) + mp - 1) / mp, ceilk1 = (((s1 - r0) * mq) + mp - 1) / mp;   // frames k with s0 <= r0 + k mp/mq < s1
  const std::int64_t frames = ceilk1 - ceilk0;
  if (frames < 8) invalid(opt.recording + ": the window holds " + std::to_string(frames) + " frames at the member's rate (" + std::to_string(rate) + " Hz); the flow needs at least 8");
  const std::int64_t steps = frames - 1;
  const std::string stem = opt.name ? *opt.name : opt.recording + "_" + kname + "_opticalflow";
  // THE PYRAMID'S LEAVES AND FRAMES (spec docs/specs/2026-11-14-measurement-pyramid.md, step 5): the flow's time means are stored as additive SUMS per leaf of the surface's tile partition and per frame of the
  // recording's tower tiling, never as per-vertex time means; the divergence, the curl and the speed of a leaf and its mean over time are read from the sums
  const std::string surface_name = text(*smf, "name");
  const SurfaceInput leaf_surface = read_surface(sub, surface_name, true);
  if (leaf_surface.n_vertices() != N) invalid(surface_name + ": another vertex count than " + kname);
  PyramidLeaves PL = pyramid_leaves(ds, sub, leaf_surface, opt.depth, "opticalflow", say);
  const FrameCells fcells = frame_cells(ds, sub, opt.recording, opt.frame_level, opt.day_start_s, "opticalflow");
  const auto gauge_row = find_field(sub.db(), sub.id(), FieldQuery{surface_name + "_canonical_frames"});
  const Json leaf_gauge = gauge_row ? Json(text(*gauge_row, "id")) : Json(nullptr);
  nxr::ladder::Mesh leaf_mesh; leaf_mesh.n = static_cast<int>(N); leaf_mesh.positions = leaf_surface.vertices; leaf_mesh.faces = leaf_surface.faces;
  const std::vector<double> vertex_area = nxr::ladder::vertexAreas(leaf_mesh);
  auto step_sample = [&](std::int64_t j) {         // the sample the velocity step j sits at: between activation frames j and j+1 of the member's own rate
    const double smp = static_cast<double>(r0) + (static_cast<double>(ceilk0 + j) + 0.5) * static_cast<double>(mp) / static_cast<double>(mq);
    return std::min<std::int64_t>(rec.n_samples - 1, std::max<std::int64_t>(0, static_cast<std::int64_t>(std::floor(smp))));
  };
  const std::int64_t flow_frame0 = fcells.ft.frameOfSample(step_sample(0)), flow_frames_span = fcells.ft.frameOfSample(step_sample(steps - 1)) - flow_frame0 + 1;
  nxr::flow::FlowMethodOptions fo;
  try { fo.method = nxr::flow::parseFlowMethod(opt.method); } catch (const std::invalid_argument& e) { invalid(e.what()); }
  try { fo.regulariser = nxr::flow::parseRegulariser(opt.regulariser); fo.dataFit = nxr::flow::resolveDataFit(fo.regulariser, nxr::flow::parseDataFit(opt.datafit)); } catch (const std::invalid_argument& e) { invalid(e.what()); }
  // the smoothness weight: given, or the regulariser's own default (connection: the Studio's 0.01, ruled 2026-10-04; ambient: 1)
  const double flow_alpha = std::isnan(opt.alpha) ? (fo.regulariser == nxr::flow::Regulariser::Connection ? 0.01 : 1.0) : opt.alpha;
  if (opt.input != "amplitude" && opt.input != "signed") invalid("opticalflow: input is amplitude or signed");
  const bool signed_input = opt.input == "signed";
  if (signed_input && orientations != 1) invalid("opticalflow: input signed needs a fixed-orientation kernel (one value per source); " + kname + " has free orientations, whose signed sum has no meaning. Use a fixed-orientation kernel or input amplitude");
  fo.tolerance = opt.flow_tolerance; fo.maxIterations = opt.max_iterations;
  if (opt.preconditioner == "ldlt") fo.preconditioner = nxr::flow::PcgOptions::Preconditioner::Ldlt;
  else if (opt.preconditioner == "ic") fo.preconditioner = nxr::flow::PcgOptions::Preconditioner::IncompleteCholesky;
  else invalid("opticalflow: preconditioner is ldlt or ic");
  if (!(fo.tolerance > 0)) invalid("opticalflow: the pcg tolerance must be positive");
  if (fo.maxIterations < 1) invalid("opticalflow: the pcg maximum iterations must be at least 1");
  fo.scaleFamily = opt.scale_family; fo.scaleVoices = opt.scale_voices; fo.scaleFilters = opt.scale_filters;
  if (fo.method == nxr::flow::FlowMethod::Perscale && fo.scaleFamily != "mexhat" && fo.scaleFamily != "itersine" && fo.scaleFamily != "logitersine") invalid("opticalflow: the perscale method needs --scale-family (mexhat, itersine or logitersine)");
  const bool per_scale = fo.method == nxr::flow::FlowMethod::Perscale;
  // the series of fields: one, or one per band-pass member for perscale (`<stem>_m00_...`, the member's centre wavenumber in producer_json)
  auto fname = [&](const std::string& tag, const char* what) { return stem + (tag.empty() ? "" : "_" + tag) + "_" + what; };
  const std::string n_check = fname(per_scale ? "m00" : "", "flow_modes");
  std::vector<std::string> tags = {""};
  std::vector<double> tag_centres;
  CacheKey key;
  key.pipeline = "opticalflow";
  key.params = Json{{"name", stem}, {"window_samples", Json::array({s0, s1})}, {"family", family_name(*opt.family)}, {"gain", opt.gain}, {"centres", bd.centres}, {"cycles", opt.cycles}, {"voices", opt.voices}, {"anchor", opt.anchor},
                    {"limits", opt.limits ? Json::array({opt.limits->first, opt.limits->second}) : Json()}, {"half_width", opt.half_width}, {"gamma", opt.gamma}, {"time_bandwidth", opt.time_bandwidth},
                    {"resolution", opt.resolution}, {"oversample", opt.oversample}, {"tolerance", wo.tolerance}, {"member", member}, {"alpha", flow_alpha}, {"modes", K}, {"filter", "timebank + wt"}};
  key.params["input"] = opt.input;
  key.params["store"] = "leaf sums"; key.params["depth"] = PL.depth; key.params["frame_level"] = opt.frame_level; key.params["day_start_s"] = opt.day_start_s ? Json(*opt.day_start_s) : Json();
  key.params["regulariser"] = nxr::flow::regulariserName(fo.regulariser); key.params["datafit"] = nxr::flow::dataFitName(fo.dataFit);
  {
    key.params["method"] = opt.method;
    key.params["flow_tolerance"] = fo.tolerance; key.params["max_iterations"] = fo.maxIterations; key.params["preconditioner"] = opt.preconditioner;
    if (per_scale) { key.params["scale_family"] = fo.scaleFamily; key.params["scale_voices"] = fo.scaleVoices; key.params["scale_filters"] = fo.scaleFilters; }
  }
  key.input = Json{{"recording", rec.id}, {"kernel", text(kern, "id")}, {"kernel_modified", opt_text(kern, "modified_utc").value_or("")}, {"package", text(pk.opr, "id")},
                   {"package_modified", opt_text(pk.opr, "modified_utc").value_or("")}, {"sfreq", rec.sfreq}, {"channels", chans}};
  key.input["leaves"] = text(*PL.part, "id"); key.input["gauge"] = leaf_gauge; key.input["frames"] = fcells.frames_id;
  if (anatomical) key.input["anatomical_frame"] = Json{{"id", sfr.anatomical_id}, {"modified", sfr.anatomical_modified}};
  key.compute = NXR_COMPUTE_VERSION;

  OpticalflowComputeResult res;
  std::string moved_aside;
  bool replaced = false;
  Json summary = Json::object();
  // perscale: the run's summary holds no powers of its own, only the run (method, timing, fields) and `series`, one entry per member (name, centre wavenumber, exact/coexact
  // power, exact fraction, mean speed, ... each the member's own). A member's fields carry that member's entry beside the run's.
  auto run_summary = [](const Json& full) {
    Json r = Json::object();
    for (const char* k : {"method", "solve_seconds_per_frame", "fields", "series"}) if (full.contains(k)) r[k] = full[k];
    return r;
  };
  auto load = [&](const Row& f) {
    const Json pj = Json::parse(opt_text(f, "producer_json").value_or("{}"), nullptr, false);
    if (pj.is_object() && pj.contains("summary")) summary = pj["summary"];
    if (per_scale) summary = run_summary(summary);                  // the field read is member m00's: keep only what belongs to the whole run
    return 0;
  };
  auto compute = [&]() {
    say(Json{{"stage", "cache"}, {"hit", false}, {"recording", rec.name}, {"kernel", kname}, {"package", pk.stem}, {"channels", M}, {"sources", N}, {"modes", K}, {"frames", frames}, {"rate", rate}});
    std::optional<nxr::hodge::Basis> basis_holder(load_hodge_basis(sub, pk));      // released once the carry and the reader are built (it is the size of the modes x vertices)
    const std::vector<int> basis_kind = basis_holder->kind;
    const Eigen::MatrixXd Kmat = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(kf.data(), kshape[0], M).cast<double>();
    // STREAMED: no [sources x frames] or [3V x frames] array is ever held. The window with the filterbank's margin and the kernel's channels is fed to the chunked wt; each
    // block of the member's coefficients inside the window is turned to amplitude maps (the kernel applied, then |J|; the baseband shift does not change it) and handed to
    // `on_frame` one frame at a time. Memory is the constant pieces plus a few maps, whatever the window's length.
    auto stream = [&](const std::function<void(std::int64_t, const Eigen::VectorXd&)>& on_frame) {
      std::int64_t have = 0;
      nxr::wt::Wt w(bank, static_cast<int>(M), wo);
      auto take = [&](const std::vector<nxr::wt::Block>& bs) {
        const nxr::wt::Block& b = bs[static_cast<std::size_t>(member)];
        const std::int64_t k_lo = std::max<std::int64_t>(0, ceilk0 - b.first), k_hi = std::min<std::int64_t>(b.coef.rows(), ceilk0 + frames - b.first);
        for (std::int64_t i0 = k_lo; i0 < k_hi; i0 += 16) {
          const std::int64_t nf = std::min<std::int64_t>(16, k_hi - i0);
          const Eigen::MatrixXcd Xs = b.coef.middleRows(i0, nf).transpose();
          const Eigen::MatrixXcd Jc = nxr::meeg::applyKernel(Kmat, Xs);
          const Eigen::MatrixXd amp = signed_input ? Eigen::MatrixXd(Jc.real()) : nxr::meeg::amplitude(Jc, orientations);
          for (std::int64_t j = 0; j < nf; ++j) { on_frame(b.first + i0 + j - ceilk0, amp.col(j)); ++have; }
        }
      };
      const nxr::io::ChunkSource src = source_of(sub, rec, chans, std::nullopt);     // the chunk service loads; this folds
      // the filterbank's margins are the HALO of the span: read by the chunk service from the neighbouring chunks, clipped where the recording ends
      const nxr::io::Span read = nxr::io::with_halo(span, s0 - r0, r1 - s1, rec.n_samples, src.width());
      nxr::io::chunk_loop(src, read, [&](const nxr::io::Chunk& c) { take(w.add(matrix_of(c, M))); });
      take(w.finish());
      if (have != frames) throw std::runtime_error("opticalflow: the filterbank returned " + std::to_string(have) + " of " + std::to_string(frames) + " frames");
    };
    // pass 1: the window's max amplitude (Rheome scales the map to unit amplitude over the whole window, so the solve needs it first); only the running max is kept
    const nxr::hodge::RowMat3 Pin = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(pk.in.vertices.data(), N, 3);
    const nxr::hodge::FaceMat Fin = Eigen::Map<const Eigen::Matrix<std::int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>>(pk.in.faces.data(), pk.in.n_faces(), 3);
    nxr::flow::FlowEngine engine(fo, Pin, Fin, nxr::flow::vertexNormals(Pin, Fin), flow_alpha);
    stream([&](std::int64_t, const Eigen::VectorXd& a) { engine.observe(a); });
    const double max_amp = engine.maxAbs();
    say(Json{{"stage", "wavelet"}, {"member", member}, {"centre_hz", fc}, {"frames", frames}, {"max_amplitude", max_amp}, {"method", opt.method}});
    // pass 2: the flow on the kernel's own surface frame by frame; each velocity is carried to the glued surface, read in the modes and dropped; the time means are running sums
    const int nS = engine.series();
    tags = engine.seriesNames();
    tag_centres = engine.seriesCentres();
    std::optional<nxr::hodge::GluedCarry> carry;
    std::optional<nxr::hodge::WeightsReader> reader;
    carry.emplace(*basis_holder, Pin); reader.emplace(*basis_holder);
    const nxr::flow::FlowReader readout(Pin, Fin);
    basis_holder.reset();
    const std::size_t S = static_cast<std::size_t>(nS);
    res.series_coefficients.assign(S, Eigen::MatrixXd(K, steps));
    std::vector<Eigen::VectorXd> div_sum(S, Eigen::VectorXd::Zero(N)), curl_sum(S, Eigen::VectorXd::Zero(N)), speed_sum(S, Eigen::VectorXd::Zero(N));
    std::vector<Eigen::VectorXd> vel_sum((anatomical || opt.tiles) ? S : 0, Eigen::VectorXd::Zero(3 * N));      // the velocity itself (embedded 3V), for its anatomical components
    Eigen::VectorXd prev, dv, cv, sp;
    // the leaf x frame sums (the stored form): per series the rows div, curl, speed, [velocity_1, velocity_2: the canonical frames], [anterior, second: the anatomical frame], each [leaf x frames of the span], and the steps a frame holds
    const std::int64_t LL = PL.leaves;
    const bool canon = sfr.canonical;
    std::vector<std::string> row_measure = {"flow_divergence", "flow_curl", "flow_speed"};
    std::vector<std::string> row_rel = {"divergence", "curl", "speed"};
    std::vector<std::string> row_unit = {"1/s", "1/s", "m/s"};
    if (canon) { row_measure.insert(row_measure.end(), {"flow_velocity_1", "flow_velocity_2"}); row_rel.insert(row_rel.end(), {"velocity_1", "velocity_2"}); row_unit.insert(row_unit.end(), {"m/s", "m/s"}); }
    if (anatomical) { row_measure.insert(row_measure.end(), {"flow_velocity_anterior", "flow_velocity_second"}); row_rel.insert(row_rel.end(), {"velocity_anterior", "velocity_second"}); row_unit.insert(row_unit.end(), {"m/s", "m/s"}); }
    const std::size_t NR = row_measure.size();
    res.frame_sums.assign(S, std::vector<std::vector<double>>(NR, std::vector<double>(static_cast<std::size_t>(LL * flow_frames_span), 0.0)));
    res.frame_counts.assign(S, std::vector<double>(static_cast<std::size_t>(flow_frames_span), 0.0));
    res.frame_rows = row_measure; res.frame_rel = row_rel; res.frame_units = row_unit;
    std::vector<double> w_all(vertex_area), w_frame(vertex_area), w_anat(vertex_area), xv(static_cast<std::size_t>(N));
    if (canon) for (std::int64_t v = 0; v < N; ++v) if (!(std::isfinite(sfr.e1(v, 0)) && std::isfinite(sfr.e1(v, 1)) && std::isfinite(sfr.e1(v, 2)))) w_frame[static_cast<std::size_t>(v)] = 0.0;   // a vertex with no frame (the poles) is out of the frame sums
    auto leaf_add = [&](std::vector<double>& into, std::int64_t f, const std::vector<double>& x, const std::vector<double>& w, double scale) {
      const auto sums = nxr::partition::accumulate(PL.codes.data(), PL.codes.size(), static_cast<int>(LL), x.data(), w.data()).sum;
      for (std::int64_t l = 0; l < LL; ++l) into[static_cast<std::size_t>(l * flow_frames_span + f)] += scale * sums[static_cast<std::size_t>(l)];
    };
    std::vector<double> iter_sum(S, 0.0);
    std::vector<int> iter_max(S, 0);
    bool have_iter = false;
    double engine_seconds = 0, block_seconds = 0, block_iterations = 0;
    std::int64_t block_frames = 0;
    const std::int64_t every = std::max<std::int64_t>(1, opt.progress_frames);
    stream([&](std::int64_t k, const Eigen::VectorXd& a) {
      if (k > 0) {
        const nxr::flow::FlowStep st = engine.step(prev, a);
        engine_seconds += st.seconds; block_seconds += st.seconds; ++block_frames;
        for (std::size_t i = 0; i < S; ++i) {
          const Eigen::VectorXd& v = st.velocity[i];
          readout.read(v, dv, cv, sp);
          div_sum[i] += dv; curl_sum[i] += cv; speed_sum[i] += sp;
          if (anatomical || opt.tiles) vel_sum[i] += v;
          {   // THE LEAF x FRAME SUMS of this step (per second: x rate), weighted by the vertex's area, in the tower frame the step sits in
            const std::int64_t f = fcells.ft.frameOfSample(step_sample(k - 1)) - flow_frame0;
            res.frame_counts[i][static_cast<std::size_t>(f)] += 1.0;
            auto vec_of = [&](const Eigen::VectorXd& a) { for (std::int64_t q = 0; q < N; ++q) xv[static_cast<std::size_t>(q)] = std::isfinite(a[q]) ? a[q] : 0.0; };
            vec_of(dv); leaf_add(res.frame_sums[i][0], f, xv, w_all, rate);
            vec_of(cv); leaf_add(res.frame_sums[i][1], f, xv, w_all, rate);
            vec_of(sp); leaf_add(res.frame_sums[i][2], f, xv, w_all, rate);
            std::size_t r = 3;
            if (canon) {
              for (int c = 0; c < 2; ++c) {
                const nxr::frames::Mat3& e = c == 0 ? sfr.e1 : sfr.e2;
                for (std::int64_t q = 0; q < N; ++q) xv[static_cast<std::size_t>(q)] = w_frame[static_cast<std::size_t>(q)] > 0 ? v[3 * q] * e(q, 0) + v[3 * q + 1] * e(q, 1) + v[3 * q + 2] * e(q, 2) : 0.0;
                leaf_add(res.frame_sums[i][r++], f, xv, w_frame, rate);
              }
            }
            if (anatomical) {
              nxr::frames::Mat3 vm(N, 3);
              for (std::int64_t q = 0; q < N; ++q) for (int d = 0; d < 3; ++d) vm(q, d) = v[3 * q + d];
              const auto comp = nxr::frames::anatomicalComponents(sfr.anatomy, sfr.e1, sfr.e2, vm);
              for (int c = 0; c < 2; ++c) {
                for (std::int64_t q = 0; q < N; ++q) { const bool ok = std::isfinite(comp(q, c)); xv[static_cast<std::size_t>(q)] = ok ? comp(q, c) : 0.0; w_anat[static_cast<std::size_t>(q)] = ok ? vertex_area[static_cast<std::size_t>(q)] : 0.0; }
                leaf_add(res.frame_sums[i][r++], f, xv, w_anat, rate);
              }
            }
          }
          res.series_coefficients[i].col(k - 1) = reader->flow(carry->apply(v)).col(0);
          if (!st.iterations.empty()) { have_iter = true; iter_sum[i] += st.iterations[i]; iter_max[i] = std::max(iter_max[i], st.iterations[i]); if (i == 0) block_iterations += st.iterations[0]; }
        }
        if (k % every == 0 || k == steps) {
          Json ev{{"stage", "flow"}, {"frame", k}, {"frames", steps}, {"method", opt.method}, {"seconds_per_frame", block_seconds / static_cast<double>(block_frames)}};
          if (!st.iterations.empty()) ev["iterations"] = block_iterations / static_cast<double>(block_frames);
          say(ev);
          block_seconds = 0; block_iterations = 0; block_frames = 0;
        }
      }
      prev = a;
    });
    const double inv = 1.0 / static_cast<double>(steps);
    res.series_divergence.resize(S); res.series_curl.resize(S); res.series_speed.resize(S);
    if (anatomical) { res.series_anterior.resize(S); res.series_second.resize(S); }
    // a summary: how much of the flow's power is exact (sources and sinks) and coexact (rotation), and the strongest modes
    auto summarise = [&](const Eigen::MatrixXd& co) {
      Eigen::VectorXd pw = co.cwiseAbs2().rowwise().mean();
      double pe = 0, pc = 0;
      for (std::int64_t k = 0; k < K; ++k) (basis_kind[static_cast<std::size_t>(k)] == 2 ? pc : pe) += pw[k];
      auto top = [&](bool coexact) {
        std::vector<std::int64_t> idx;
        for (std::int64_t k = 0; k < K; ++k) if ((basis_kind[static_cast<std::size_t>(k)] == 2) == coexact) idx.push_back(k);
        std::sort(idx.begin(), idx.end(), [&](std::int64_t x, std::int64_t y) { return pw[x] > pw[y]; });
        Json t = Json::array();
        for (std::size_t i = 0; i < std::min<std::size_t>(3, idx.size()); ++i) t.push_back(Json{{"mode", idx[i]}, {"power", pw[idx[i]]}});
        return t;
      };
      return Json{{"exact_power", pe}, {"coexact_power", pc}, {"exact_fraction", pe + pc > 0 ? pe / (pe + pc) : 0.0}, {"top_exact_modes", top(false)}, {"top_coexact_modes", top(true)}};
    };
    Json fields = Json::array(), series = Json::array();
    for (std::size_t i = 0; i < S; ++i) {
      res.series_divergence[i] = (div_sum[i] * inv) * rate;                    // per frame -> per second
      res.series_curl[i] = (curl_sum[i] * inv) * rate;
      res.series_speed[i] = (speed_sum[i] * rate) * inv;
      if (opt.tiles) {
        if (res.series_velocity.size() != S) res.series_velocity.resize(S);
        res.series_velocity[i].resize(N, 3);
        for (std::int64_t v = 0; v < N; ++v) for (int d = 0; d < 3; ++d) res.series_velocity[i](v, d) = vel_sum[i][3 * v + d] * inv * rate;
      }
      if (anatomical) {
        // the time-mean velocity (per second) read in the anatomical frame: column 0 anterior, column 1 the second axis (superior or medial per the stored label); NaN at the unreliable vertices
        nxr::frames::Mat3 vm(N, 3);
        for (std::int64_t v = 0; v < N; ++v) for (int d = 0; d < 3; ++d) vm(v, d) = vel_sum[i][3 * v + d] * inv * rate;
        const auto comp = nxr::frames::anatomicalComponents(sfr.anatomy, sfr.e1, sfr.e2, vm);
        res.series_anterior[i] = comp.col(0); res.series_second[i] = comp.col(1);
      }
      for (const char* w : {"flow_modes", "flow"}) fields.push_back(fname(tags[i], w));
      Json one = summarise(res.series_coefficients[i]);
      one["mean_speed"] = res.series_speed[i].mean();
      one["max_mean_speed"] = res.series_speed[i].maxCoeff();
      if (per_scale) { one["name"] = tags[i]; one["centre_wavenumber"] = tag_centres[i]; }
      if (have_iter) { one["iterations_per_frame_mean"] = iter_sum[i] * inv; one["iterations_per_frame_max"] = iter_max[i]; }
      series.push_back(one);
    }
    if (anatomical) { res.anterior = res.series_anterior[0]; res.second = res.series_second[0]; }
    res.coefficients = res.series_coefficients[0]; res.divergence = res.series_divergence[0]; res.curl = res.series_curl[0]; res.speed = res.series_speed[0];
    res.seconds_per_frame = engine_seconds * inv;
    summary = per_scale ? Json::object() : series[0];       // perscale: no member speaks for the run (the members' own entries are in `series`)
    summary["method"] = opt.method;
    summary["solve_seconds_per_frame"] = res.seconds_per_frame;
    summary["fields"] = fields;
    if (per_scale) summary["series"] = series;
    return 0;
  };
  auto store = [&](const int&) {
    // the window's time line: frame j sits between activation frames j and j+1
    const double dt = 1.0 / rate, origin = rec.t0 + (static_cast<double>(r0) + static_cast<double>(ceilk0 * mp) / static_cast<double>(mq)) / rec.sfreq + 0.5 * dt;
    auto line_named = [&](const std::string& lname) { return find_manifold(sub.db(), sub.id(), lname, "line"); };
    auto fits = [&](const std::string& lid) {
      const auto d = line_factor(sub.db(), lid);
      return d && d->count == steps && std::abs(d->spacing - dt) <= 1e-12 * dt && d->origin && std::abs(*d->origin - origin) <= 1e-9;
    };
    std::string lname = stem + "_time", line_id;
    if (const auto l = line_named(lname)) {
      if (fits(text(*l, "id"))) line_id = text(*l, "id");
      else lname = stem + "_time_" + std::to_string(s0) + "_" + std::to_string(steps);
    }
    if (line_id.empty()) if (const auto l = line_named(lname)) line_id = text(*l, "id");
    if (line_id.empty()) {
      ManifoldSpec ms;
      ms.name = lname;
      ms.path = nxr::io::folder_for("line", "", std::string("time")) + "/" + lname;
      ms.type = "line";
      ms.n_vertices = steps;
      ms.topology = {"lattice", 1, 1, std::nullopt, {}};
      const auto lc = nxr::io::lattice_counts({steps});
      if (rec.acquisition_id) ms.cols.set("acquisition_id", *rec.acquisition_id); else ms.cols.null("acquisition_id");
      if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
      if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
      if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
      GeometrySpec g;
      g.form = "closed_form";
      g.cs_key = "clock";
      DimensionSpec d;
      d.name = "time";
      d.n_vertices = steps;
      d.spacing = dt;
      d.origin = origin;
      d.unit = "s";
      g.dimensions.push_back(d);
      ms.geometry = g;
      line_id = sub.manifold(std::move(ms));
    }
    const std::string product = sub.product({pk.axis_id, line_id}, rec.acquisition_id);
    std::vector<std::pair<const char*, std::string>> have;
    for (const std::string& tag : tags) {
      have.push_back({"field", nxr::io::node_path(sub.db(), product, fname(tag, "flow_modes"))});
    }
    moved_aside = clear_stored(ds, sub, have, opt.replace, stem, replaced);
    // the byte-less Fields of the leaf sums and their joint Selection of an earlier run (the run is a replacement: clear_stored refused otherwise)
    const std::string joint_name = opt.recording + " \xc3\x97 " + stem + " " + PL.name + " \xc3\x97 " + [&] { std::ostringstream o; o << fcells.ft.frame_s(); return o.str(); }() + " s";
    if (const auto oj = nxr::io::selection::named_row(sub.db(), sub.id(), joint_name)) sub.remove("selection", text(*oj, "id"));
    for (const std::string& tag : tags)
      if (const auto of = find_field(sub.db(), sub.id(), FieldQuery{fname(tag, "flow"), "optical flow", std::nullopt, false})) sub.remove("field", text(*of, "id"));
    pyramid_leaves_record(sub, PL, leaf_gauge);
    std::vector<FrameSumsSeries> sums_series;
    const Json pr = projector_record(rec.field);
    Json warnings = Json::array({
        "kinematic, not physical: the flow is the apparent velocity of the amplitude pattern; where activity is created or destroyed rather than moved the residual appears as divergence",
        "aperture-limited: Horn-Schunck recovers the velocity along the intensity gradient only, the rest is the smoothness prior; alpha sets how much of v is data (vary it and report the range)",
        "per frame, not per second: the map is differenced between adjacent frames; the stored divergence, curl, speed and velocity sums are multiplied by the frame rate (per second), the coefficients are per frame",
        "a leaf's sums are of its vertices' values weighted by area: a leaf larger than the flow's spatial scale averages a turning or opposing flow out; a tower frame holds as many flow steps as the member's rate gives it (none when the rate is below the frame rate: no mean there)",
        "the filterbank's margin: the record's margins are read when it has them, else the first and last frames are transients",
        "the record's bad segments are not excluded: choose the window free of them"});
    if (per_scale) warnings.push_back("perscale: a measurement, not a solver; each series is the flow of one band-pass member of the graph filterbank applied to the map (scaled to its own unit amplitude), not of the activation itself");
    Json pj = stamp(key);
    pj["stage"] = "opticalflow";
    pj["by"] = "nxr-neuro opticalflow compute";
    pj["producer"] = producer_name("opticalflow compute");
    pj["recording"] = rec.name;
    pj["recording_id"] = rec.id;
    pj["kernel"] = kname;
    pj["kernel_id"] = text(kern, "id");
    pj["package"] = pk.stem;
    pj["package_id"] = text(pk.opr, "id");
    pj["modes"] = K;
    pj["flow_method"] = Json{{"name", opt.method}, {"tolerance", fo.tolerance}, {"max_iterations", fo.maxIterations}, {"preconditioner", opt.preconditioner},
                             {"scale_family", fo.scaleFamily}, {"scale_voices", fo.scaleVoices}, {"scale_filters", fo.scaleFilters}};
    pj["flow_functional"] = Json{{"regulariser", nxr::flow::regulariserName(fo.regulariser)}, {"datafit", nxr::flow::dataFitName(fo.dataFit)}};
    pj["input"] = opt.input;
    pj["method"] = "Rheome flow.apparent: amplitude |J| of the kernel applied to the complex wavelet coefficients of one filterbank member (own rate) -> manifold Horn-Schunck optical flow (opticalflow_scalar) -> Hodge-Dirac flow weights of the velocity (fused weights; exact = divergence potential, coexact = stream function)";
    pj["activation"] = "amplitude per source of the kernel applied to the complex coefficients (|J|; sqrt of the orientations' sum of squares for a free kernel)";
    pj["orientations"] = orientations;
    pj["bank"] = Json{{"family", family_name(*opt.family)}, {"gain", opt.gain}, {"sides", "analytic"}, {"nfft", nfft}, {"resolution_hz", opt.resolution}, {"members", bank.size()}, {"centres", bd.centres},
                      {"cycles", opt.cycles}, {"voices", opt.voices}, {"anchor", opt.anchor}, {"half_width", opt.half_width}, {"gamma", opt.gamma}, {"time_bandwidth", opt.time_bandwidth}};
    pj["member"] = Json{{"index", member}, {"centre_hz", fc}, {"requested_hz", opt.freq}, {"rate_hz", rate}, {"step_samples", static_cast<double>(mp) / static_cast<double>(mq)}};
    pj["filter"] = Json{{"method", "nxr::wt (chunked, overlap-save)"}, {"margin_samples", lb}, {"block_samples", plan.block}, {"tolerance", wo.tolerance}, {"achieved_tolerance", plan.achieved}, {"margins_complete", margins_complete}, {"oversample", opt.oversample}};
    pj["alpha"] = flow_alpha;
    pj["sfreq"] = rec.sfreq;
    pj["rate"] = rate;
    pj["frames"] = frames;
    pj["velocity_frames"] = steps;
    pj["analysis_window"] = Json{{"start_sample", s0}, {"stop_sample", s1}, {"start_s", rec.t0 + static_cast<double>(s0) / rec.sfreq}, {"stop_s", rec.t0 + static_cast<double>(s1) / rec.sfreq}, {"chunks", span.what}, {"chunks_read", Json::array({span.first_chunk, span.last_chunk})}};
    pj["time_line"] = lname;
    pj["warnings"] = warnings;
    pj["summary"] = summary;
    pj["velocity_stored"] = "as leaf x frame sums only (flow_velocity_*), never per frame per vertex";
    if (anatomical) pj["anatomical_frame"] = Json{{"definition", sfr.anatomy.definition}, {"reference", sfr.anatomy.reference}, {"canonical_frames", text(*smf, "name") + "_canonical_frames"},
                                                   {"note", "velocity_anterior / velocity_second: the time-mean tangent velocity along anterior and along the second axis (superior on lateral and medial cortex, medial on dorsal and ventral, per the stored label); NaN where the frame is unreliable"}};
    pj["projectors"] = pr["projectors"];
    pj["projectors_applied_in_samples"] = pr["applied_in_samples"];
    pj["projectors_not_applied"] = pr["not_applied"];
    pj["ctf_compensation"] = pr["ctf_compensation"];
    auto put = [&](const Json& own_summary, const std::string& fname, const std::string& manifold, const std::string& fkind, const std::string& unit, const std::string& description, const std::vector<float>& data, const std::vector<std::int64_t>& shape, const char* part) {
      FieldSpec f;
      f.name = fname;
      f.path = nxr::io::node_path(sub.db(), manifold, fname);
      f.kind = fkind;
      f.manifold_id = manifold;
      f.data = view(data, shape);
      if (rec.acquisition_id) f.cols.set("acquisition_id", *rec.acquisition_id); else f.cols.null("acquisition_id");
      f.cols.set("function", "opticalflow").set("derived_from_id", rec.id);
      if (!unit.empty()) f.cols.set("unit", unit);
      f.cols.set("description", description);
      Json j = pj;
      j["summary"] = own_summary;
      j["part"] = part;
      f.cols.json("producer_json", j);
      sub.field(std::move(f));
    };
    char bandtxt[80];
    std::snprintf(bandtxt, sizeof bandtxt, "%.3g Hz wavelet", fc);
    const std::string of = std::string(" of the apparent motion of the ") + bandtxt + " envelope of the sources of " + rec.name + " by " + kname;
    for (std::size_t si = 0; si < tags.size(); ++si) {
      const std::string& tag = tags[si];
      std::string ofs = of;
      if (!tag.empty()) {
        char bt[160];
        std::snprintf(bt, sizeof bt, " (scale %s of the %s graph filterbank, centre wavenumber %.4g)", tag.c_str(), fo.scaleFamily.c_str(), si < tag_centres.size() ? tag_centres[si] : 0.0);
        ofs += bt;
      }
      Json own = summary;                                    // this member's entry first, then the run's
      if (per_scale) { own = summary["series"][si]; for (const char* k : {"method", "solve_seconds_per_frame", "fields", "series"}) own[k] = summary[k]; }
      const Eigen::MatrixXd& cm = res.series_coefficients[si];
      std::vector<float> co(static_cast<std::size_t>(K * steps));
      for (std::int64_t k = 0; k < K; ++k) for (std::int64_t t = 0; t < steps; ++t) co[static_cast<std::size_t>(k * steps + t)] = static_cast<float>(cm(k, t));
      put(own, fname(tag, "flow_modes"), product, "optical flow", "", "the Hodge-Dirac flow weights (exact modes: divergence potential, sources and sinks; coexact: stream function, rotation) per frame" + ofs, co, {K, steps}, "flow_modes");
      {   // the byte-less Field the leaf x frame sums are of; the sums themselves are written below, once every series has its Field
        FieldSpec fs;
        fs.name = fname(tag, "flow"); fs.kind = "optical flow"; fs.manifold_id = surface_id; fs.value_type = "vector3";
        if (rec.acquisition_id) fs.cols.set("acquisition_id", *rec.acquisition_id); else fs.cols.null("acquisition_id");
        fs.cols.set("function", "opticalflow").set("derived_from_id", rec.id).set("by_operator_id", text(kern, "id"));
        fs.cols.set("description", "the divergence, curl, speed and velocity" + ofs + ", stored as additive sums per leaf of " + PL.name + " and per frame of the recording's tower tiling only, never as per-vertex time means");
        Json j = pj;
        j["summary"] = own; j["part"] = "flow";
        j["leaves"] = Json{{"selection", PL.name}, {"depth", PL.depth}, {"count", PL.leaves}, {"tree", PL.tree}, {"weights", "the vertex's lumped area"}, {"node", PL.node}};
        j["frames"] = Json{{"tiling", frame_tiling_name(opt.recording, opt.frame_level)}, {"frame_s", fcells.ft.frame_s()}, {"frame_level", opt.frame_level}, {"steps_per_frame", "the flow steps whose midpoint lies in the frame (the `count` of the tiling for this Field)"}};
        j["stored"] = "per leaf and frame, sums over the flow steps of the vertex values weighted by vertex area, per second: flow_divergence, flow_curl, flow_speed (|v|, the tangent speed), flow_velocity_1 / _2 (v . e1 / e2 in the canonical frames: e1 north, e2 west)" + std::string(anatomical ? ", flow_velocity_anterior / _second (the anatomical frame)" : "");
        j["derived_at_read"] = "the time mean of a measure over a leaf or any roll-up: `<measure>_density` = sum / (area x steps); the speed of the mean velocity is read from its components";
        j["studio"] = Json{{"type", "complex"}};                       // the velocity is a tangent vector a vertex: the Studio's complex (backend valueType.ts algebraOf)
        fs.cols.json("producer_json", j);
        const std::string fid = sub.field(std::move(fs));
        FrameSumsSeries ss;
        ss.field_id = fid; ss.group = "flow/" + nxr::io::safe_name(stem) + (tag.empty() ? std::string() : "_" + tag);
        ss.counts = res.frame_counts[si];
        for (std::size_t r = 0; r < res.frame_rows.size(); ++r) ss.rows.push_back({res.frame_rows[r], res.frame_rel[r], res.frame_units[r], res.frame_sums[si][r]});
        sums_series.push_back(std::move(ss));
      }
    }
    frame_sums_write(sub, PL, fcells, joint_name, "the optical flow's leaf sums over the joint cells of " + PL.name + " x the " + std::to_string(fcells.n_frames) + " frames of " + opt.recording + ": code = (tile << " + std::to_string(fcells.bits) + ") | frame",
                     flow_frame0, flow_frames_span, sums_series, "nxr-neuro opticalflow compute", rec.acquisition_id);
  };
  // the cache marker: the flow's coefficients (the one Field of the run with bytes); a run of another producer or key is a miss, refused by clear_stored unless --replace
  const auto marker_row = find_field(sub.db(), sub.id(), FieldQuery{n_check});
  const std::optional<std::string> marker = marker_row ? opt_text(*marker_row, "path") : std::optional<std::string>("-");
  const Outcome<int> o = cached<int>(sub, "field", marker, key, load, compute, store);
  res.hit = o.hit;
  res.seconds = o.seconds;
  if (o.hit) { say(Json{{"stage", "cache"}, {"hit", true}, {"recording", rec.name}}); res.seconds_per_frame = summary.value("solve_seconds_per_frame", 0.0); }
  res.info = Json{{"name", stem}, {"fields", summary.contains("fields") ? summary["fields"] : Json::array({fname("", "flow_modes"), fname("", "flow")})}, {"leaves", PL.name}, {"depth", PL.depth}, {"frame_level", opt.frame_level}, {"measures", Json::array({"flow_divergence", "flow_curl", "flow_speed", "flow_velocity_1", "flow_velocity_2", "flow_velocity_anterior", "flow_velocity_second"})}, {"method", opt.method}, {"recording", rec.name}, {"kernel", kname}, {"package", pk.stem}, {"modes", K}, {"sources", N},
                  {"family", family_name(*opt.family)}, {"gain", opt.gain}, {"member", member}, {"centre_hz", fc}, {"alpha", flow_alpha}, {"rate", rate}, {"frames", frames}, {"velocity_frames", steps},
                  {"window", Json::array({s0, s1})}, {"solve_seconds_per_frame", res.seconds_per_frame}, {"summary", summary}, {"replaced", replaced}, {"moved_aside", moved_aside.empty() ? Json() : Json(moved_aside)}, {"cache", o.hit ? "hit" : "miss"}};
  if (opt.tiles) {
    // the maps' tile summaries per level (nxr/pipelines/tiles.h): the STORED divergence, curl and speed as area-weighted scalars; the time-mean velocity (not stored) in the frame and Levi-Civita
    // forms when the flow was computed in this run
    const auto srow = sub.db().read("manifold", surface_id);
    TileSummaryOptions to;
    to.surface = srow ? std::optional<std::string>(text(*srow, "name")) : std::nullopt;
    to.levels = opt.tile_levels; to.max_level = opt.tile_max_level; to.levi_civita = opt.tile_levi_civita; to.replace = opt.replace;
    Json tj = Json::array();
    for (std::size_t si = 0; si < tags.size(); ++si) {
      TileMaps tm;
      tm.stem = stem + (tags[si].empty() ? "" : "_" + tags[si]);
      // the time-mean divergence, curl and speed are the leaf sums on the pyramid (rolled up to any ladder level at read): only the velocity's frame and Levi-Civita forms are summarised here
      if (!o.hit && si < res.series_velocity.size()) tm.vectors.push_back({"velocity", res.series_velocity[si]});
      Json one = tile_summaries_write(ds, sub, tm, to, progress);
      if (o.hit) one["note"] = "the velocity's tile summaries are written when the flow is computed; the time-mean maps are the leaf sums on the pyramid (`measure query`)";
      tj.push_back(one);
    }
    res.info["tiles"] = tj;
  }
  return res;
}

// ── temporal filters ─────────────────────────────────────────────────────────────────────────────────────────────


}  // namespace nxr::pipelines
