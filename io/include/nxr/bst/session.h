// nxr-io — a Brainstorm condition as a SESSION of a subject: nxr.convert.condition (convert_condition, condition_plan, resolve_raw_binary), .raw (the BST-BIN reader),
// .timeseries (the recording on channels x time, its grid and envelope, resampling), .events (events and bad segments) and the naming of session-scoped nodes.
//
// COPY, NEVER LINK: the samples are read from the `.bst` (or the `.mat`'s F) and written into the dataset's own Zarr array, one 2^17-sample block (all channels) at a time.
// Peak memory of a recording of C channels: about 2 blocks of C x 2^17 float32 (the block and its envelope tee), i.e. ~1 MB per channel; a resample holds the resampled recording
// whole (as nxr does) plus 16 channels of input as float64.
#pragma once
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/mat.h"
#include "nxr/bst/maps.h"
#include "nxr/bst/naming.h"
#include "nxr/bst/subject.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

// ── the BST-BIN binary (nxr.convert.raw) ──
// Brainstorm's own format: a hdrsize header, then EPOCHS of [nchannels x epochsize] float32, each channel contiguous within its epoch and `epochsize` samples
// apart in every epoch, the last one included (in_fread_bst.m, out_fwrite_bst.m): the file ends after the last channel's declared samples ("epoch stride") or
// holds the last epoch whole ("whole epochs"). A last epoch re-laid with its channels their declared length apart ("cut") is read too; a file shorter than its
// header declares is read in Brainstorm's layout (truncated()); a size that fits none of them is refused.
class BstReader {
 public:
  BstReader(const Value& sfile, const std::filesystem::path& path);
  // samples [s0, s0+n) of channels [c0, c1) as [c1-c0, n] float32, row after row
  void read(std::int64_t s0, std::int64_t n, std::int64_t c0, std::int64_t c1, std::vector<float>& out);
  std::int64_t n_channels() const { return n_chan_; }
  std::int64_t n_samples() const { return n_total_; }   // the samples the file holds in full (< declared_samples() when it is cut short)
  std::int64_t declared_samples() const { return declared_; }
  bool truncated() const { return n_total_ < declared_; }
  // the layout found from the file's size: "epoch stride", "whole epochs", "cut" or "epoch stride, cut short" (empty when the file was not there to measure)
  const std::string& layout() const { return layout_; }

 private:
  std::filesystem::path path_;
  std::int64_t n_chan_, epoch_, hdr_, n_total_, declared_, last_stride_;
  std::string layout_;
};

// the binary a raw link reads (D122); emits `relinked` when it is beside the link
std::filesystem::path resolve_raw_binary(const std::filesystem::path& raw_link, const Value& sfile, const Emit& progress);

// ── events (nxr.convert.events) ──
struct EventTable {
  std::vector<std::int32_t> type;
  std::vector<double> onset, duration;
  std::vector<std::int32_t> channel_offsets, channel_index;
  std::vector<std::string> type_names;
  std::vector<std::vector<double>> type_colors;
  std::size_t n_events() const { return onset.size(); }
};
EventTable parse_events(const Value* events, const std::vector<std::string>& channel_names);
Json export_events(nxr::io::SubjectWriter& sub, const std::string& rec_path, const EventTable& table, const std::string& line_id, const std::string& of_id,
                   const std::optional<std::string>& acquisition = std::nullopt, const std::optional<std::string>& acquisition_id = std::nullopt);
Json import_bad_segments(nxr::io::SubjectWriter& sub, const Json& src, bool replace);
Json import_events(nxr::io::SubjectWriter& sub, const std::filesystem::path& bst_data, bool replace);

// ── recordings (nxr.convert.timeseries) ──
// resample_poly (scipy), `step` channels at a time: read(c0, c1) -> [c1-c0, n_t] float64 row-major; returns [n_chan, ceil(n_t*to/fs)] float32
std::vector<float> resample_rows(const std::function<std::vector<double>(std::int64_t, std::int64_t)>& read, std::int64_t n_chan, std::int64_t n_t, double fs, double to,
                                 std::int64_t& n_new, std::int64_t step = 16);
// D168: what the recording's provenance keeps of the channel file's projectors (`projectors`, `ctf_compensation`). The samples are stored as recorded: no projector is
// applied by an import; the chunk service applies the status-1 ones on load.
struct Projection {
  Json record = Json::object();
};
Json add_resampled(nxr::io::SubjectWriter& sub, const std::string& recording, double to, const std::optional<std::string>& name, const Emit& progress);
// A recording made from another one, on the same channels and the same time line (same length and rate): `fill` is handed a writer and writes the samples in blocks of
// `recording_block_samples()` (all channels; float32 [channels x n] row after row; the last block shorter), each block starting where the last ended. The new recording is stored as the
// importer stores one (the grid of chunks and shards, the envelope, the chunk tiling), named `name`, its Field derived from `source`; the source's channel flags, events and bad segments
// (Selections on the time line) are copied under the new name unchanged, and its provenance (projectors, CTF compensation, ...) carried into `producer_json` beneath `producer`.
using BlockWriter = std::function<void(std::int64_t s0, const std::vector<float>& x, std::int64_t n)>;
std::int64_t recording_block_samples();
Json add_derived_recording(nxr::io::SubjectWriter& sub, const std::string& source, const std::string& name, const std::string& comment, const std::string& source_tag, const Json& producer,
                           const std::function<void(const BlockWriter&)>& fill, const Emit& progress);
Json export_raw_timeseries(nxr::io::SubjectWriter& sub, const std::string& name, const Value& sfile, const Value* channel_flag, const std::string& acquisition,
                           const std::string& channels_id, const Value* comment, const std::filesystem::path& bst_path, const std::vector<std::string>& channel_names,
                           const Emit& progress, std::optional<double> resample, const Projection& projection = {});
// REPLACE THE SAMPLES of a raw recording already imported, in place (`import condition --replace-recording`): the raw link read again as the import reads it (the BST-BIN
// reader, `resample`), its samples and envelope rewritten while the recording's row, id, links, events, channel flags, projectors and every other node stay. Refused, with
// nothing written: no such recording of the acquisition, other channels (count, and the channel file and flags' sha1 `channels_fingerprint` when the channels recorded one),
// another shape or rate, no note. The stored samples and envelope are moved aside (`_replaced-<date>`), never deleted; the recording's producer_json gains an entry of
// `replaced_samples` {utc, by, note, command, source {file, layout, declared_samples, read_samples}, channels_checked, from {samples digest, moved_aside}, to {samples digest}}.
// The result lists the stored results whose provenance names the recording (`derived`): they were computed from the old samples.
struct ReplaceRecording {
  std::string note, by;
};
Json replace_raw_timeseries(nxr::io::SubjectWriter& sub, const std::string& name, const Value& sfile, const std::string& acquisition, const std::filesystem::path& bst_path,
                            const std::string& channels_fingerprint, const Emit& progress, std::optional<double> resample, const ReplaceRecording& why);
Json export_timeseries(nxr::io::SubjectWriter& sub, const std::string& name, const std::filesystem::path& data_file, const std::string& acquisition,
                       const std::string& channels_id, const std::vector<std::string>& channel_names, std::optional<double> resample, const Projection& projection = {});

// A recording copied from any block source (the importers of other formats, `nxr/convert/recording.h`): `read(s0, n, out)` fills `out` with samples [s0, s0 + n) of all channels,
// float32, row after row. Stored exactly as the importer stores a Brainstorm recording (the grid of chunks and shards, the envelope, the events and bad segments, the channel flags).
struct BlockRecording {
  std::string name, channels_id, source = "file";
  std::optional<std::string> acquisition;
  std::int64_t n_channels = 0, n_samples = 0;
  double sfreq = 0, t0 = 0;
  std::vector<std::int64_t> flags;               // 1 good, -1 bad, one per channel (empty: none)
  EventTable events;
  std::vector<std::string> channel_names;
  std::string comment;
  Json producer = Json::object();
  Emit progress;
  bool linked = false;   // the samples stay in the file they are read from (an in-place import): the array is laid out, the envelope made, no samples stored
  std::function<void(std::int64_t s0, std::int64_t n, std::vector<float>& out)> read;
};
Json write_recording_blocks(nxr::io::SubjectWriter& sub, const BlockRecording& rec);

// ── the condition (nxr.convert.condition, nxr.convert.subject.condition_plan) ──
struct ConvertOptions {
  bool write_gain = true, all_surfaces = false;
  std::optional<double> resample;
  std::optional<ReplaceRecording> replace_recording;   // replace the samples of the condition's raw recording, already imported, in place (nothing else is imported)
};
// The writers a map over time and a confounds matrix need (nxr.convert.timeseries `_recording` with `_write_blocks`, and `_time_line`): MapHooks over this file's recording writer.
const MapHooks& recording_map_hooks();
// Mirror one condition into `sub` as the session `condition` (nxr.convert.condition.convert_condition): the channels, the recordings, then the kernels and head models
// (`import_condition_kernels`) and the source power by band (`export_condition_timefreqs`).
Json convert_condition(nxr::io::SubjectWriter& sub, const std::filesystem::path& bst_root, const std::string& condition, const ConditionPlan& plan, const ConvertOptions& opt,
                       const Emit& progress);

}  // namespace nxr::bst
