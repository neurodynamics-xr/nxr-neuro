// nxr::bst — a recording on channels x time: nxr.convert.timeseries with nxr.io.grid (the TIMESERIES GRID and the ENVELOPE TEE), and the BST-BIN reader (nxr.convert.raw).
//
//   raw        [channels x samples] float32, inner chunk [1 x 2^15], shard [channels x 2^17], UNCOMPRESSED      (clipped for a recording shorter than a chunk)
//   envelope   [channels x windows x 2] float32 (min, max per 2^5 samples), inner [1 x 2^12 x 2], shard [channels x 2^12 x 2]
// window 2^5 < raw chunk 2^15 < block 2^17: every boundary shared. The envelope is reduced from each block AS IT IS WRITTEN (one pass over the raw).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>

#include "np.h"
#include "nxr/bst/entities.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/bst/projectors.h"
#include "nxr/bst/session.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/layout.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::ArrayMeta;
using nxr::io::ArrayOptions;
using nxr::io::Cols;
using nxr::io::SlabWriter;
using nxr::io::SubjectWriter;
using nxr::io::view;

namespace {
constexpr std::int64_t kRawChunk = 1 << 15, kBlock = 1 << 17, kWindow = 1 << 5, kEnvChunk = kBlock / kWindow;

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
std::string i2s(std::int64_t x) { return std::to_string(x); }

ArrayMeta recording_meta(std::int64_t n_chan, std::int64_t n_samples) {
  const std::int64_t c = std::min(kRawChunk, std::max<std::int64_t>(1, n_samples));
  const std::int64_t s = n_samples >= kRawChunk ? kBlock : c;
  ArrayOptions o;
  o.chunks = nxr::io::Shape{1, c};
  o.shards = nxr::io::Shape{std::max<std::int64_t>(1, n_chan), s};
  o.compress = false;
  return nxr::io::array_meta(nxr::io::DType::Float32, {n_chan, n_samples}, o);
}

// the array at key `at` laid out (its zarr.json written): `at` back
const std::string& laid(nxr::io::Store& store, const std::string& at, const ArrayMeta& m) {
  nxr::io::layout_array(store, at, m);
  return at;
}

// The recording's array and its ENVELOPE (at `dest`), the envelope reduced from each block as it is written.
class EnvelopeTee {
 public:
  EnvelopeTee(nxr::io::Store& store, const std::string& arr_at, const std::string& dest, std::int64_t n_chan, std::int64_t n_samples)
      : arr_(store, arr_at), env_(store, laid(store, dest, env_meta(n_chan, n_samples))), n_chan_(n_chan), n_samples_(n_samples) {}

  static ArrayMeta env_meta(std::int64_t n_chan, std::int64_t n_samples) {
    const std::int64_t n_win = (n_samples + kWindow - 1) / kWindow;
    const std::int64_t c = std::min(kEnvChunk, std::max<std::int64_t>(1, n_win));
    ArrayOptions o;
    o.chunks = nxr::io::Shape{1, c, 2};
    o.shards = nxr::io::Shape{std::max<std::int64_t>(1, n_chan), c, 2};
    o.compress = false;
    return nxr::io::array_meta(nxr::io::DType::Float32, {n_chan, n_win, 2}, o);
  }

  // x: [n_chan, n] float32 row after row, the samples [s0, s0 + n): a whole-channel block on window boundaries, whole windows long except at the end
  void write(std::int64_t s0, const std::vector<float>& x, std::int64_t n) {
    if (s0 % kWindow || (n % kWindow && s0 + n != n_samples_))
      value_error("envelope: a write at (slice(None, None, None), slice(" + i2s(s0) + ", " + i2s(s0 + n) + ", None)) is not a whole-channel block on window boundaries");
    arr_.write(s0, view(x.data(), {n_chan_, n}));
    const std::int64_t nw = (n + kWindow - 1) / kWindow;
    std::vector<float> e(static_cast<std::size_t>(n_chan_ * nw * 2));
    for (std::int64_t c = 0; c < n_chan_; ++c)
      for (std::int64_t w = 0; w < nw; ++w) {
        const float* p = x.data() + c * n + w * kWindow;
        const std::int64_t len = std::min(kWindow, n - w * kWindow);
        float lo = p[0], hi = p[0];
        for (std::int64_t k = 1; k < len; ++k) {   // np.minimum / np.maximum: a NaN propagates, a tie keeps the first
          if (!(lo <= p[k] || std::isnan(lo))) lo = p[k];
          if (!(hi >= p[k] || std::isnan(hi))) hi = p[k];
        }
        e[static_cast<std::size_t>((c * nw + w) * 2)] = lo;
        e[static_cast<std::size_t>((c * nw + w) * 2 + 1)] = hi;
      }
    env_.write(s0 / kWindow, view(e.data(), {n_chan_, nw, 2}));
  }

 private:
  SlabWriter arr_, env_;
  std::int64_t n_chan_, n_samples_;
};

using Write = std::function<void(EnvelopeTee&)>;

std::string str_of(const Value* v) { return v && v->is_char() ? v->string() : ""; }

// _write_flags: the recording's own channel flags, a partition of the channels
void write_flags(SubjectWriter& sub, const std::string& rec_path, const std::vector<std::int64_t>& flags, const std::optional<std::string>& acquisition_id, const std::string& channels_id,
                 const std::string& of_id) {
  if (flags.empty()) return;
  FlagLevels fl = flag_levels(flags, rec_path + ": ");
  LabellingOptions lo;
  lo.manifold_id = channels_id;
  lo.dictionary = rec_path.substr(rec_path.rfind('/') + 1) + "_flags";
  lo.scope = "subject";
  lo.colors = fl.colors;
  lo.acquisition_id = acquisition_id;
  lo.description = "the recording's own channel flags";
  lo.of_field_id = of_id;
  labelling(sub, rec_path + "_flags", fl.codes, fl.levels, lo);
}

struct Recording {
  std::string name;
  std::int64_t n_chan = 0, n_samples = 0;
  double sfreq = 0, t0 = 0;
  std::optional<std::string> acquisition, acquisition_id;
  std::string channels_id;
  Write write;
  std::vector<std::int64_t> flags;
  const Value* events = nullptr;
  std::optional<EventTable> table;   // events given as a table (an importer of another format); `events` then unused
  std::vector<std::string> channel_names;
  std::optional<std::string> comment;
  std::string kind, source;
  Json producer = Json::object();
  Emit progress;
  // a time-resolved MAP's (nxr's `field_kind`, `description`, `unit`, `time_id`): `channels_id` is then the manifold of its rows, `time_id` a Line another field lives on
  std::string field_kind = "recording";
  std::optional<std::string> description, unit, time_id;
  std::optional<std::string> derived_from_id;   // the recording this one was made from (a filtered or resampled copy)
};

// nxr.convert.timeseries._time_line: the Time Line, three numbers, exactly as a Volume's voxel positions are - a closed form, no array
std::string write_time_line(SubjectWriter& sub, const std::string& path, const std::optional<std::string>& aid, std::int64_t n_samples, double sfreq, double t0) {
  nxr::io::ManifoldSpec ms;
  ms.name = path.substr(path.rfind('/') + 1);
  ms.path = path;
  ms.type = "line";
  ms.n_vertices = n_samples;
  ms.topology = {"lattice", 1, 1, std::nullopt, {}};
  const auto lc = nxr::io::lattice_counts({n_samples});
  if (aid) ms.cols.set("acquisition_id", *aid); else ms.cols.null("acquisition_id");
  if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
  if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
  if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
  nxr::io::GeometrySpec g;
  g.form = "closed_form";
  g.cs_key = "clock";
  nxr::io::DimensionSpec d;
  d.name = "time";
  d.n_vertices = n_samples;
  d.spacing = 1.0 / sfreq;
  d.origin = t0;
  d.unit = "s";
  g.dimensions.push_back(d);
  ms.geometry = g;
  return sub.manifold(std::move(ms));
}

// everything after the bytes (nxr.convert.timeseries._recording)
Json write_recording(SubjectWriter& sub, const Recording& r) {
  auto& db = sub.db();
  const std::optional<std::string> aid = r.acquisition_id ? r.acquisition_id : sub.acquisition(r.acquisition);
  std::string label;
  if (r.acquisition) label = *r.acquisition;
  else if (const auto row = aid ? db.one("SELECT label FROM acquisition WHERE id = ?", {*aid}) : std::nullopt) label = nxr::io::text(*row, "label");
  const std::string tpath = nxr::io::folder_for("line", "", std::string("time")) + "/" + r.name;

  const std::string time_id = r.time_id ? *r.time_id : write_time_line(sub, tpath + "_time", aid, r.n_samples, r.sfreq, r.t0);

  const std::string product = sub.product({r.channels_id, time_id}, aid);
  const std::string rec = nxr::io::node_path(db, product, r.name);
  const ArrayMeta meta = recording_meta(r.n_chan, r.n_samples);
  const std::string env = rec + "_envelope";

  nxr::io::FieldSpec f;
  f.name = r.name;
  f.path = rec;
  f.kind = r.field_kind;
  f.manifold_id = product;
  f.meta = meta;
  f.populate = [&](nxr::io::Store& store, const std::string& at) {
    nxr::io::ensure_groups(sub.store(), env, false);
    EnvelopeTee tee(store, at, nxr::io::join_key(sub.prefix(), env), r.n_chan, r.n_samples);
    r.write(tee);
    if (r.progress) r.progress(Json{{"stage", "envelope"}, {"name", r.name}});
  };
  if (aid) f.cols.set("acquisition_id", *aid); else f.cols.null("acquisition_id");
  f.cols.set("description", r.description ? *r.description : label + " \xe2\x80\x94 the " + r.kind + " recording").set("source", r.source);
  if (r.unit) f.cols.set("unit", *r.unit);
  if (r.derived_from_id) f.cols.set("derived_from_id", *r.derived_from_id);
  if (r.comment && !r.comment->empty()) f.cols.set("comment", *r.comment);
  Json producer = Json::object();
  producer["kind"] = r.kind;
  for (auto it = r.producer.begin(); it != r.producer.end(); ++it) producer[it.key()] = it.value();
  f.cols.json("producer_json", producer);
  const std::string rid = sub.field(std::move(f));

  const double dt = 1.0 / r.sfreq;
  // THE CHUNK GRID (D46): closed form - four numbers state a regular grid, `at(k)` derives every member
  const std::int64_t chunk = meta.chunks[1];
  const std::int64_t n = (r.n_samples + chunk - 1) / chunk;
  {
    nxr::io::SelectionSpec s;
    s.name = r.name + "_chunks";
    s.type = "spans";
    s.manifold_id = time_id;
    s.cols.set("cell", "1").set("of_field_id", rid).set("acquisition_id", nxr::io::V(aid))
        .set("description", "the " + i2s(n) + " stored chunks of " + i2s(chunk) + " samples").set("n_elements", r.n_samples).set("n_members", n);
    s.cols.json("params_json", Json{{"tiling", Json{{"origin", 0}, {"width", static_cast<double>(chunk) * dt}, {"hop", static_cast<double>(chunk) * dt}, {"count", n}}},
                                    {"end", static_cast<double>(r.n_samples) * dt}});
    sub.selection(std::move(s));
  }
  // THE ENVELOPE (D96): a closed-form tiling of the time Line with two array-backed measurements
  {
    const std::int64_t w = (r.n_samples + kWindow - 1) / kWindow;
    nxr::io::SelectionSpec s;
    s.name = r.name + "_envelope";
    s.type = "spans";
    s.manifold_id = time_id;
    s.cols.set("cell", "1").set("of_field_id", rid).set("acquisition_id", nxr::io::V(aid))
        .set("description", "the " + i2s(w) + " windows of " + i2s(kWindow) + " samples the recording's min and max are measured over")
        .set("n_elements", r.n_samples).set("n_members", w);
    s.cols.json("params_json", Json{{"tiling", Json{{"origin", r.t0}, {"width", static_cast<double>(kWindow) * dt}, {"hop", static_cast<double>(kWindow) * dt}, {"count", w}}},
                                    {"end", r.t0 + static_cast<double>(r.n_samples) * dt}});
    for (int comp = 0; comp < 2; ++comp) {
      Cols m;
      m.set("measure", comp == 0 ? "min" : "max").set("of_field_id", rid).set("array_path", env).set("component", comp);
      s.measurements.push_back(m);
    }
    sub.selection(std::move(s));
  }
  if (!r.flags.empty()) write_flags(sub, nxr::io::node_path(db, r.channels_id, r.name), r.flags, aid, r.channels_id, rid);
  const EventTable table = r.table ? *r.table : parse_events(r.events, r.channel_names);
  const Json ev = export_events(sub, tpath, table, time_id, rid, std::nullopt, aid);
  return Json{{"name", r.name}, {"id", rid}, {"time_id", time_id}, {"n_channels", r.n_chan}, {"n_samples", r.n_samples}, {"sfreq", r.sfreq},
              {"events", ev["n_events"]}, {"event_types", ev["n_types"]}, {"bad_segments", ev["bad_segments"]}};
}

std::vector<std::int64_t> flags_of(const Value* v) {
  std::vector<std::int64_t> out;
  if (v && v->is_numeric()) for (double x : v->doubles()) out.push_back(static_cast<std::int64_t>(x));
  return out;
}

}  // namespace

// ── the BST-BIN binary ──

BstReader::BstReader(const Value& sfile, const fs::path& path) : path_(path) {
  const Value* h = sfile.field("header");
  auto num = [&](const char* k) {
    const Value* v = h ? h->field(k) : nullptr;
    if (!v || !v->is_numeric() || v->numel() == 0) throw nxr::io::Error(code::no_such_row, k);
    return static_cast<std::int64_t>(v->scalar());
  };
  n_chan_ = num("nchannels");
  epoch_ = num("epochsize");
  hdr_ = num("hdrsize");
  n_total_ = num("nsamples");
}

void BstReader::read(std::int64_t s0, std::int64_t n, std::int64_t c0, std::int64_t c1, std::vector<float>& out) {
  if (s0 < 0 || s0 + n > n_total_) value_error("read [" + i2s(s0) + ", " + i2s(s0 + n) + ") outside [0, " + i2s(n_total_) + ")");
  const std::int64_t nc = c1 - c0;
  out.assign(static_cast<std::size_t>(nc * n), 0.0f);
  std::ifstream f(path_, std::ios::binary);
  if (!f) throw nxr::io::Error(code::not_found, "[Errno 2] No such file or directory: " + str_repr(py_path(path_.string())));
  std::vector<float> block(static_cast<std::size_t>(epoch_ * nc));
  std::int64_t got = 0;
  const std::int64_t e0 = s0 / epoch_, e1 = (s0 + n - 1) / epoch_;
  for (std::int64_t e = e0; e <= e1; ++e) {
    const std::int64_t lo = std::max(e * epoch_, s0), hi = std::min((e + 1) * epoch_, s0 + n), count = hi - lo;
    f.clear();
    f.seekg(static_cast<std::streamoff>(hdr_ + (e * n_chan_ + c0) * epoch_ * 4));
    f.read(reinterpret_cast<char*>(block.data()), static_cast<std::streamsize>(block.size() * 4));
    const std::int64_t have = static_cast<std::int64_t>(f.gcount()) / 4;
    if (have < epoch_ * nc) value_error("cannot reshape array of size " + i2s(have) + " into shape (" + i2s(nc) + "," + i2s(epoch_) + ")");
    for (std::int64_t c = 0; c < nc; ++c)
      std::copy(block.begin() + c * epoch_ + (lo - e * epoch_), block.begin() + c * epoch_ + (hi - e * epoch_), out.begin() + c * n + got);
    got += count;
  }
}

// ── the recordings ──

Json export_raw_timeseries(SubjectWriter& sub, const std::string& name, const Value& sfile, const Value* channel_flag, const std::string& acquisition,
                           const std::string& channels_id, const Value* comment, const fs::path& bst_path, const std::vector<std::string>& channel_names,
                           const Emit& progress, std::optional<double> resample, const Projection& projection) {
  const std::string fmt = str_of(sfile.field("format")).empty() ? "unknown" : str_of(sfile.field("format"));
  if (fmt != "BST-BIN")
    value_error("raw link format '" + fmt + "' is not supported \xe2\x80\x94 only a Brainstorm-preprocessed continuous (BST-BIN) imports byte-faithfully. "
                "Run the recording through a Brainstorm process first, which writes one.");
  const Value* prop = sfile.field("prop");
  const Value* cur = prop ? prop->field("currCtfComp") : nullptr;
  const Value* dest = prop ? prop->field("destCtfComp") : nullptr;
  if (cur && dest && cur->is_numeric() && dest->is_numeric() && cur->numel() == 1 && dest->numel() == 1 &&
      static_cast<std::int64_t>(cur->scalar()) != static_cast<std::int64_t>(dest->scalar())) {
    auto show = [](const Value* v) { const bool fl = v->cls == Class::Double || v->cls == Class::Single; return fl ? float_repr(v->scalar()) : i2s(static_cast<std::int64_t>(v->scalar())); };
    value_error("raw link's CTF compensation is not at destination order (curr=" + show(cur) + ", dest=" + show(dest) + ") \xe2\x80\x94 apply compensation in Brainstorm first.");
  }
  BstReader reader(sfile, bst_path);
  const std::int64_t n_chan = reader.n_channels();
  std::int64_t n_t = reader.n_samples();
  const Value* sf = prop ? prop->field("sfreq") : nullptr;
  const Value* tv = prop ? prop->field("times") : nullptr;
  if (!sf) throw nxr::io::Error(code::no_such_row, "sfreq");
  if (!tv) throw nxr::io::Error(code::no_such_row, "times");
  double sfreq = sf->scalar();
  const double t0 = tv->doubles().at(0);

  Json producer = Json{{"format", fmt}, {"device", str_of(sfile.field("device"))}};
  if (!bst_path.empty()) producer["bst_file"] = py_path(bst_path.string());
  for (auto it = projection.record.begin(); it != projection.record.end(); ++it) producer[it.key()] = it.value();
  const Value* events = sfile.field("events");

  Recording r;
  r.name = name;
  r.n_chan = n_chan;
  r.acquisition = acquisition;
  r.channels_id = channels_id;
  r.t0 = t0;
  r.flags = flags_of(channel_flag);
  r.events = events;
  r.channel_names = channel_names;
  if (comment && comment->is_char() && !comment->string().empty()) r.comment = comment->string();
  r.kind = "sensor";
  r.source = "raw-link";
  r.progress = progress;
  std::vector<float> resampled;
  if (resample && *resample != sfreq) {
    std::int64_t n_new = 0;
    resampled = resample_rows([&](std::int64_t c0, std::int64_t c1) {
      std::vector<float> b;
      reader.read(0, n_t, c0, c1, b);
      return std::vector<double>(b.begin(), b.end());
    }, n_chan, n_t, sfreq, *resample, n_new);
    producer["resample"] = Json{{"from", sfreq}, {"to", *resample}, {"method", "resample_poly"}};
    n_t = n_new;
    sfreq = *resample;
    r.write = [&resampled, n_chan, n_t, &projection](EnvelopeTee& tee) {
      std::vector<float> blk, out;
      for (std::int64_t s0 = 0; s0 < n_t; s0 += kBlock) {
        const std::int64_t n = std::min(kBlock, n_t - s0);
        blk.resize(static_cast<std::size_t>(n_chan * n));
        for (std::int64_t c = 0; c < n_chan; ++c) std::copy(resampled.begin() + c * n_t + s0, resampled.begin() + c * n_t + s0 + n, blk.begin() + c * n);
        if (!projection.matrix.empty()) { project(projection.matrix, static_cast<std::size_t>(n_chan), blk, static_cast<std::size_t>(n), out); blk.swap(out); }
        tee.write(s0, blk, n);
      }
    };
  } else {
    r.write = [&reader, &name, n_t, n_chan, &progress, &projection](EnvelopeTee& tee) {
      std::vector<float> blk, out;
      for (std::int64_t s0 = 0; s0 < n_t; s0 += kBlock) {
        const std::int64_t n = std::min(kBlock, n_t - s0);
        reader.read(s0, n, 0, n_chan, blk);
        if (!projection.matrix.empty()) { project(projection.matrix, static_cast<std::size_t>(n_chan), blk, static_cast<std::size_t>(n), out); blk.swap(out); }
        tee.write(s0, blk, n);
        if (progress) progress(Json{{"stage", "raw-stream"}, {"name", name}, {"written", s0 + n}, {"of", n_t}});
      }
    };
  }
  r.n_samples = n_t;
  r.sfreq = sfreq;
  r.producer = producer;
  return write_recording(sub, r);
}

Json export_timeseries(SubjectWriter& sub, const std::string& name, const fs::path& data_file, const std::string& acquisition, const std::string& channels_id,
                       const std::vector<std::string>& channel_names, std::optional<double> resample, const Projection& projection) {
  MatFile mf = MatFile::open(data_file);
  for (const auto& v : mf.list())
    if (v.name == "F" && v.cls == Class::Struct)
      throw nxr::io::Error(code::unsupported, "timeseries " + str_repr(name) + " is a RAW LINK (F is an sFile struct) \xe2\x80\x94 pass it to export_raw_timeseries");
  const auto info = mf.matrix_info("F");
  if (!info) throw nxr::io::Error(code::no_such_row, "F");
  const auto time = mf.read("Time");
  const std::vector<double> t = time && time->is_numeric() ? time->doubles() : std::vector<double>{};
  if (t.size() < 2) value_error(name + ": not a Brainstorm data file (need .Time with >=2 samples)");
  std::vector<double> diff(t.size() - 1);
  for (std::size_t i = 0; i + 1 < t.size(); ++i) diff[i] = t[i + 1] - t[i];
  double sfreq = 1.0 / (np_sum(diff.data(), diff.size()) / static_cast<double>(diff.size()));   // 1 / np.mean(np.diff(t))
  const std::int64_t n_chan = static_cast<std::int64_t>(info->rows);
  std::int64_t n_t = static_cast<std::int64_t>(info->cols);
  const auto dt = mf.read("DataType");
  Json producer = Json{{"data_type", str_of(dt ? &*dt : nullptr)}};
  for (auto it = projection.record.begin(); it != projection.record.end(); ++it) producer[it.key()] = it.value();
  const auto flags_v = mf.read("ChannelFlag");
  const auto events = mf.read("Events");
  const auto comment = mf.read("Comment");

  Recording r;
  r.name = name;
  r.n_chan = n_chan;
  r.acquisition = acquisition;
  r.channels_id = channels_id;
  r.t0 = t[0];
  r.flags = flags_of(flags_v ? &*flags_v : nullptr);
  r.events = events ? &*events : nullptr;
  r.channel_names = channel_names;
  if (comment && comment->is_char() && !comment->string().empty()) r.comment = comment->string();
  r.kind = "sensor";
  r.source = "imported";
  std::vector<float> resampled;
  if (resample && *resample != sfreq) {
    producer["resample"] = Json{{"from", sfreq}, {"to", *resample}, {"method", "resample_poly"}};
    std::int64_t n_new = 0;
    resampled = resample_rows([&](std::int64_t c0, std::int64_t c1) {
      std::vector<float> all;
      mf.read_columns_f32("F", 0, static_cast<std::size_t>(n_t), all);   // an excerpt: the .mat holds at most 2 GB
      return std::vector<double>(all.begin() + c0 * n_t, all.begin() + c1 * n_t);
    }, n_chan, n_t, sfreq, *resample, n_new);
    n_t = n_new;
    sfreq = *resample;
    r.write = [&resampled, n_chan, n_t, &projection](EnvelopeTee& tee) {
      std::vector<float> blk, out;
      for (std::int64_t s0 = 0; s0 < n_t; s0 += kBlock) {
        const std::int64_t n = std::min(kBlock, n_t - s0);
        blk.resize(static_cast<std::size_t>(n_chan * n));
        for (std::int64_t c = 0; c < n_chan; ++c) std::copy(resampled.begin() + c * n_t + s0, resampled.begin() + c * n_t + s0 + n, blk.begin() + c * n);
        if (!projection.matrix.empty()) { project(projection.matrix, static_cast<std::size_t>(n_chan), blk, static_cast<std::size_t>(n), out); blk.swap(out); }
        tee.write(s0, blk, n);
      }
    };
  } else {
    r.write = [&mf, n_t, n_chan, &projection](EnvelopeTee& tee) {
      std::vector<float> blk, out;
      for (std::int64_t s0 = 0; s0 < n_t; s0 += kBlock) {
        const std::int64_t n = std::min(kBlock, n_t - s0);
        mf.read_columns_f32("F", static_cast<std::size_t>(s0), static_cast<std::size_t>(n), blk);
        if (!projection.matrix.empty()) { project(projection.matrix, static_cast<std::size_t>(n_chan), blk, static_cast<std::size_t>(n), out); blk.swap(out); }
        tee.write(s0, blk, n);
      }
    };
  }
  r.n_samples = n_t;
  r.sfreq = sfreq;
  r.producer = producer;
  return write_recording(sub, r);
}

Json add_resampled(SubjectWriter& sub, const std::string& recording, double to, const std::optional<std::string>& name, const Emit& progress) {
  auto& db = sub.db();
  const std::string sname = nxr::io::text(sub.row(), "name");
  const auto rec = db.one("SELECT * FROM field WHERE subject_id = ? AND kind = 'recording' AND name = ? AND status = 'complete'", {sub.id(), recording});
  if (!rec) throw nxr::io::Error(code::no_such_row, sname + ": no recording " + str_repr(recording));
  std::vector<std::string> fac;
  for (const auto& r : db.all("SELECT d.manifold_id FROM manifold p JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE p.id = ? ORDER BY d.ordinal", {nxr::io::text(*rec, "manifold_id")}))
    fac.push_back(nxr::io::text(r, "manifold_id"));
  const auto dim = db.one("SELECT d.* FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.id = ?", {fac.at(1)});
  const double spacing = std::get<double>(dim->at("spacing"));
  const double fs = 1.0 / spacing;
  const double t0 = nxr::io::is_null(*dim, "origin") ? 0.0 : std::get<double>(dim->at("origin"));
  char g[64];
  std::snprintf(g, sizeof g, "%g", to);
  const std::string rname = nxr::io::text(*rec, "name");
  const std::string nu = name ? *name : rname + "_" + (to == std::floor(to) ? std::to_string(static_cast<std::int64_t>(to)) : float_repr(to)) + "Hz";
  if (db.one("SELECT id FROM field WHERE subject_id = ? AND name = ?", {sub.id(), nu})) throw nxr::io::Error(code::already_exists, sname + ": a field " + str_repr(nu) + " exists");
  const auto reader = nxr::io::ArrayReader::open(sub.store(), nxr::io::text(*rec, "path"));
  const std::int64_t n_chan = reader.shape().at(0), n_t = reader.shape().at(1);
  std::int64_t n_new = 0;
  std::vector<float> x = resample_rows([&](std::int64_t c0, std::int64_t c1) {
    const auto bytes = reader.read({c0, 0}, {c1, n_t});
    std::vector<float> f(bytes.size() / 4);
    std::memcpy(f.data(), bytes.data(), bytes.size());
    return std::vector<double>(f.begin(), f.end());
  }, n_chan, n_t, fs, to, n_new);
  if (progress) progress(Json{{"stage", "resampled"}, {"name", nu}, {"from", fs}, {"to", to}, {"samples", n_new}});
  std::string kind = "sensor";
  if (const auto pj = nxr::io::opt_text(*rec, "producer_json")) {
    try {
      const Json j = Json::parse(*pj);
      if (j.is_object() && j.contains("kind")) kind = j["kind"].get<std::string>();
    } catch (...) {}
  }
  Recording r;
  r.name = nu;
  r.n_chan = n_chan;
  r.n_samples = n_new;
  r.sfreq = to;
  r.t0 = t0;
  if (!nxr::io::is_null(*rec, "acquisition_id")) r.acquisition_id = nxr::io::text(*rec, "acquisition_id");
  r.channels_id = fac.at(0);
  r.comment = rname + " resampled to " + g + " Hz";
  r.kind = kind;
  r.source = "resampled";
  r.producer = Json{{"resampled_from", Json{{"field", nxr::io::text(*rec, "id")}, {"name", rname}}}, {"resample", Json{{"from", fs}, {"to", to}, {"method", "resample_poly"}}}};
  r.progress = progress;
  r.write = [&x, n_chan, n_new](EnvelopeTee& tee) {
    std::vector<float> blk;
    for (std::int64_t s0 = 0; s0 < n_new; s0 += kBlock) {
      const std::int64_t n = std::min(kBlock, n_new - s0);
      blk.resize(static_cast<std::size_t>(n_chan * n));
      for (std::int64_t c = 0; c < n_chan; ++c) std::copy(x.begin() + c * n_new + s0, x.begin() + c * n_new + s0 + n, blk.begin() + c * n);
      tee.write(s0, blk, n);
    }
  };
  return write_recording(sub, r);
}

std::int64_t recording_block_samples() { return kBlock; }

namespace {

// A spans Selection of the source recording (`<source>_events`, `<source>_bad_segments`) as the same spans on the new recording: its dictionary, colours, description and parameters kept.
void copy_spans(SubjectWriter& sub, const std::string& from, const std::string& to, const std::string& time_id, const std::string& of_field_id, const std::optional<std::string>& aid) {
  auto& db = sub.db();
  const auto sel = db.one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'spans' AND status = 'complete'", {sub.id(), from});
  if (!sel) return;
  std::vector<std::string> types;
  std::map<std::int64_t, std::int64_t> index;
  TimeSpansOptions o;
  if (!nxr::io::is_null(*sel, "dictionary_id"))
    for (const auto& e : db.all("SELECT code, name, color_json FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal, code", {nxr::io::text(*sel, "dictionary_id")})) {
      index[nxr::io::integer(e, "code")] = static_cast<std::int64_t>(types.size());
      types.push_back(nxr::io::text(e, "name"));
      if (!nxr::io::is_null(e, "color_json")) {
        const Json c = Json::parse(nxr::io::text(e, "color_json"), nullptr, false);
        if (c.is_array() && c.size() >= 3) o.colors.push_back({c[0].get<double>() / 255.0, c[1].get<double>() / 255.0, c[2].get<double>() / 255.0});
      }
    }
  if (o.colors.size() != types.size()) o.colors.clear();
  if (types.empty()) types.push_back("event");
  std::vector<TimeSpan> spans;
  for (const auto& r : db.all("SELECT start, stop, code FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {nxr::io::text(*sel, "id")})) {
    const std::int64_t code = nxr::io::is_null(r, "code") ? 0 : nxr::io::integer(r, "code");
    const auto it = index.find(code);
    spans.push_back({std::stod(nxr::io::text(r, "start")), std::stod(nxr::io::text(r, "stop")), it == index.end() ? 0 : it->second});
  }
  o.line_id = time_id;
  o.of_field_id = of_field_id;
  o.acquisition_id = aid;
  o.description = nxr::io::opt_text(*sel, "description").value_or("");
  if (const auto pj = nxr::io::opt_text(*sel, "params_json")) {
    const Json p = Json::parse(*pj, nullptr, false);
    if (p.is_object()) o.params = p;
  }
  time_spans(sub, nxr::io::node_path(db, time_id, to), spans, types, o);
}

}  // namespace

Json add_derived_recording(SubjectWriter& sub, const std::string& source, const std::string& name, const std::string& comment, const std::string& source_tag, const Json& producer,
                           const std::function<void(const BlockWriter&)>& fill, const Emit& progress) {
  auto& db = sub.db();
  const std::string sname = nxr::io::text(sub.row(), "name");
  const auto rec = db.one("SELECT * FROM field WHERE subject_id = ? AND kind = 'recording' AND name = ? AND status = 'complete'", {sub.id(), source});
  if (!rec) throw nxr::io::Error(code::no_such_row, sname + ": no recording " + str_repr(source));
  if (db.one("SELECT id FROM field WHERE subject_id = ? AND name = ?", {sub.id(), name}))
    throw nxr::io::Error(code::already_exists, sname + ": a field " + str_repr(name) + " exists (choose another --name, or remove it first)");
  std::vector<std::string> fac;
  for (const auto& r : db.all("SELECT d.manifold_id FROM manifold p JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE p.id = ? ORDER BY d.ordinal", {nxr::io::text(*rec, "manifold_id")}))
    fac.push_back(nxr::io::text(r, "manifold_id"));
  if (fac.size() != 2) throw nxr::io::Error(code::invalid_argument, source + ": a recording lies on channels x time");
  const auto dim = db.one("SELECT d.* FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.id = ?", {fac.at(1)});
  const double fs = 1.0 / std::get<double>(dim->at("spacing"));
  const double t0 = nxr::io::is_null(*dim, "origin") ? 0.0 : std::get<double>(dim->at("origin"));
  const auto reader = nxr::io::ArrayReader::open(sub.store(), nxr::io::text(*rec, "path"));
  Json src = Json::object();
  if (const auto pj = nxr::io::opt_text(*rec, "producer_json")) {
    const Json j = Json::parse(*pj, nullptr, false);
    if (j.is_object()) src = j;
  }
  Recording r;
  r.name = name;
  r.n_chan = reader.shape().at(0);
  r.n_samples = reader.shape().at(1);
  r.sfreq = fs;
  r.t0 = t0;
  if (!nxr::io::is_null(*rec, "acquisition_id")) r.acquisition_id = nxr::io::text(*rec, "acquisition_id");
  r.channels_id = fac.at(0);
  r.time_id = fac.at(1);
  r.derived_from_id = nxr::io::text(*rec, "id");
  r.comment = comment;
  r.kind = src.value("kind", std::string("sensor"));
  r.source = source_tag;
  if (const auto u = nxr::io::opt_text(*rec, "unit")) r.unit = *u;
  // what the source's provenance says (the projectors and the CTF compensation among it) is carried over; the new record's own keys win
  Json merged = Json::object();
  for (auto it = src.begin(); it != src.end(); ++it)
    if (it.key() != "kind" && it.key() != "cache" && it.key() != "resample" && it.key() != "resampled_from" && it.key() != "filter" && it.key() != "filtered_from") merged[it.key()] = it.value();
  for (auto it = producer.begin(); it != producer.end(); ++it) merged[it.key()] = it.value();
  r.producer = merged;
  // the recording's own channel flags (1 good / -1 bad), read from its `<source>_flags` partition
  if (const auto fl = db.one("SELECT * FROM selection WHERE subject_id = ? AND of_field_id = ? AND name = ? AND status = 'complete'", {sub.id(), nxr::io::text(*rec, "id"), source + "_flags"});
      fl && !nxr::io::is_null(*fl, "array_path") && !nxr::io::is_null(*fl, "dictionary_id")) {
    std::map<std::int64_t, std::int64_t> sign;
    for (const auto& e : db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", {nxr::io::text(*fl, "dictionary_id")})) sign[nxr::io::integer(e, "code")] = nxr::io::text(e, "name") == "good" ? 1 : -1;
    const nxr::io::ArrayReader cr = nxr::io::ArrayReader::open(sub.store(), nxr::io::text(*fl, "array_path"));
    const auto raw = cr.read();
    if (cr.dtype() == nxr::io::DType::Int32 && raw.size() == static_cast<std::size_t>(4 * r.n_chan)) {
      for (std::int64_t c = 0; c < r.n_chan; ++c) {
        std::int32_t code32;
        std::memcpy(&code32, raw.data() + 4 * c, 4);
        r.flags.push_back(sign.count(code32) ? sign[code32] : 1);
      }
    }
  }
  r.progress = progress;
  r.write = [&fill](EnvelopeTee& tee) { fill([&tee](std::int64_t s0, const std::vector<float>& x, std::int64_t n) { tee.write(s0, x, n); }); };
  Json out = write_recording(sub, r);
  const std::string rid = out.at("id").get<std::string>();
  const std::string tid = out.at("time_id").get<std::string>();
  copy_spans(sub, source + "_events", name + "_events", tid, rid, r.acquisition_id);
  copy_spans(sub, source + "_bad_segments", name + "_bad_segments", tid, rid, r.acquisition_id);
  return out;
}

// The MapHooks of maps.h over the recording writer: `results_map._series` (`_recording` with `_write_blocks`) and `_time_line`.
const MapHooks& recording_map_hooks() {
  static const MapHooks hooks = {
      [](SubjectWriter& sub, const SeriesRequest& q) -> std::string {
        Recording r;
        r.name = q.stem;
        r.n_chan = q.rows;
        r.n_samples = q.samples;
        r.sfreq = 1.0 / q.step;
        r.t0 = q.t0;
        r.acquisition = q.acquisition;
        r.channels_id = q.rows_id;
        r.comment = q.comment;
        r.kind = "map";
        r.source = q.source;
        r.producer = q.producer;
        r.field_kind = q.comment;
        r.description = q.description;
        r.unit = q.unit;
        const std::vector<float>& x = q.x;
        r.write = [&x, &q](EnvelopeTee& tee) {   // _write_blocks
          std::vector<float> blk;
          for (std::int64_t s0 = 0; s0 < q.samples; s0 += kBlock) {
            const std::int64_t n = std::min(kBlock, q.samples - s0);
            blk.resize(static_cast<std::size_t>(q.rows * n));
            for (std::int64_t c = 0; c < q.rows; ++c)
              std::copy(x.begin() + c * q.samples + s0, x.begin() + c * q.samples + s0 + n, blk.begin() + c * n);
            tee.write(s0, blk, n);
          }
        };
        return write_recording(sub, r)["id"].get<std::string>();
      },
      [](SubjectWriter& sub, const TimeLineRequest& q) -> std::string { return write_time_line(sub, q.path, q.acquisition_id, q.samples, q.sfreq, q.t0); }};
  return hooks;
}

// ── a recording from any block source (the importers of other formats) ──
Json write_recording_blocks(SubjectWriter& sub, const BlockRecording& b) {
  Recording r;
  r.name = b.name;
  r.n_chan = b.n_channels;
  r.n_samples = b.n_samples;
  r.sfreq = b.sfreq;
  r.t0 = b.t0;
  r.acquisition = b.acquisition;
  r.channels_id = b.channels_id;
  r.flags = b.flags;
  r.table = b.events;
  r.channel_names = b.channel_names;
  if (!b.comment.empty()) r.comment = b.comment;
  r.kind = "sensor";
  r.source = b.source;
  r.producer = b.producer;
  r.progress = b.progress;
  r.write = [&b](EnvelopeTee& tee) {
    std::vector<float> blk;
    for (std::int64_t s0 = 0; s0 < b.n_samples; s0 += kBlock) {
      const std::int64_t n = std::min(kBlock, b.n_samples - s0);
      b.read(s0, n, blk);
      if (static_cast<std::int64_t>(blk.size()) != b.n_channels * n) value_error("a block of the source has the wrong size");
      tee.write(s0, blk, n);
      if (b.progress) b.progress(Json{{"stage", "raw-stream"}, {"name", b.name}, {"written", s0 + n}, {"of", b.n_samples}});
    }
  };
  return write_recording(sub, r);
}

}  // namespace nxr::bst
