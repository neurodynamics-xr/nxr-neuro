// The importers of continuous recordings: see nxr/convert/recording.h.
#include "nxr/convert/recording.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

#include "nxr/bst/sensors.h"
#include "nxr/convert/sensor_frames.h"
#include "nxr/formats/brainvision.h"
#include "nxr/formats/ctf.h"
#include "nxr/formats/edf.h"
#include "nxr/formats/eeglab.h"
#include "nxr/formats/fif.h"
#include "nxr/formats/kit.h"
#include "nxr/io/hash.h"

namespace nxr::convert {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::Json;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string ends_with_any(const std::string& path, std::initializer_list<const char*> ext) {
  const std::string p = lower(path);
  for (const char* e : ext) if (p.size() >= std::strlen(e) && p.compare(p.size() - std::strlen(e), std::strlen(e), e) == 0) return e;
  return "";
}

// Brainstorm's channel type of a reader channel (the vocabulary the Brainstorm importer writes)
std::string bst_type(const nxr::formats::Channel& c) {
  const std::string k = lower(c.kind);
  if (k == "meg") return c.unit == "T/m" ? "MEG GRAD" : "MEG";
  if (k == "eeg" || k == "eog" || k == "ecg" || k == "emg" || k == "seeg" || k == "ecog") {
    std::string u = k;
    for (auto& ch : u) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    return u;
  }
  if (k == "stim") return "Stim";
  if (k.empty()) return "Misc";
  std::string t = k;
  t[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(t[0])));
  return t;
}

nxr::bst::Value text_value(const std::string& s) {
  nxr::bst::Value v;
  v.cls = nxr::bst::Class::Char;
  v.dims = {1, s.size()};
  v.text = {s};
  return v;
}

nxr::bst::Value doubles_value(const std::array<double, 3>& a) {
  nxr::bst::Value v;
  v.cls = nxr::bst::Class::Double;
  v.dims = {3, 1};
  v.re.resize(3 * sizeof(double));
  std::memcpy(v.re.data(), a.data(), 3 * sizeof(double));
  return v;
}

// the digitised points as the channel file's `HeadPoints` struct: Loc [3 x H] (column-major), Label and Type (cells of text)
nxr::bst::Value headpoints_value(const std::vector<SensorPlacement::DigitisedPoint>& pts) {
  const std::size_t H = pts.size();
  nxr::bst::Value loc;
  loc.cls = nxr::bst::Class::Double;
  loc.dims = {3, H};
  loc.re.resize(3 * H * sizeof(double));
  nxr::bst::Value labels, types;
  labels.cls = types.cls = nxr::bst::Class::Cell;
  labels.dims = types.dims = {1, H};
  for (std::size_t j = 0; j < H; ++j) {
    std::memcpy(loc.re.data() + 3 * j * sizeof(double), pts[j].r.data(), 3 * sizeof(double));
    labels.elems.push_back(text_value(pts[j].label));
    types.elems.push_back(text_value(pts[j].type));
  }
  nxr::bst::Value hp;
  hp.cls = nxr::bst::Class::Struct;
  hp.dims = {1, 1};
  hp.fields = {"Loc", "Label", "Type"};
  hp.elems = {std::move(loc), std::move(labels), std::move(types)};
  return hp;
}

// the channel file as the sensor writer reads it: Channel(Name, Type, Loc, Orient)
std::map<std::string, nxr::bst::Value> channel_mat(const nxr::formats::RecordingHeader& h, const SensorPlacement& place, std::map<std::string, std::string>& units) {
  nxr::bst::Value ch;
  ch.cls = nxr::bst::Class::Struct;
  ch.dims = {1, h.channels.size()};
  ch.fields = {"Name", "Type", "Loc", "Orient"};
  std::size_t ci = 0;
  for (const auto& c : h.channels) {
    const std::size_t i = ci++;
    const std::string type = bst_type(c);
    if (!c.unit.empty()) {
      const auto it = units.find(type);
      if (it != units.end() && it->second != c.unit) refuse("channels of type " + type + " have different units (" + it->second + ", " + c.unit + ")");
      units[type] = c.unit;
    }
    ch.elems.push_back(text_value(c.name));
    ch.elems.push_back(text_value(type));
    // positions and orientations as `place_sensors` placed them: in one named coordinate system, in metres
    if (place.location[i]) ch.elems.push_back(doubles_value(*place.location[i]));
    else ch.elems.emplace_back();
    if (place.orientation[i]) ch.elems.push_back(doubles_value(*place.orientation[i]));
    else ch.elems.emplace_back();
  }
  return {{"Channel", std::move(ch)}};
}

std::string file_sha256(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw nxr::io::Error(code::not_found, "cannot read " + path);
  nxr::io::Sha256 h;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    h.update(reinterpret_cast<const std::uint8_t*>(buf.data()), static_cast<std::size_t>(f.gcount()));
  }
  return h.hex();
}

// the recording's time of its first sample, seconds: a FIF file counts samples from the acquisition's start
double first_time(const nxr::formats::RecordingReader& r) {
  if (const auto* fif = dynamic_cast<const nxr::formats::fif::RawReader*>(&r)) return static_cast<double>(fif->info().first_samp) / r.header().sampling_rate;
  return 0.0;
}

nxr::bst::EventTable event_table(const nxr::formats::RecordingHeader& h, double t0) {
  nxr::bst::EventTable t;
  auto type_of = [&](const std::string& label) {
    const auto it = std::find(t.type_names.begin(), t.type_names.end(), label);
    if (it != t.type_names.end()) return static_cast<std::int32_t>(it - t.type_names.begin());
    t.type_names.push_back(label);
    t.type_colors.push_back({0.5, 0.5, 0.5});
    return static_cast<std::int32_t>(t.type_names.size() - 1);
  };
  auto add = [&](double onset, double duration, const std::string& label) {
    t.type.push_back(type_of(label));
    t.onset.push_back(t0 + onset);
    t.duration.push_back(duration);
  };
  for (const auto& a : h.annotations) add(a.onset, a.duration, a.label.empty() ? "event" : a.label);
  // samples the file does not hold are zeros here, and a bad segment labelled as a skip: never interpolated
  for (const auto& g : h.gaps) add(static_cast<double>(g.first) / h.sampling_rate, static_cast<double>(g.count) / h.sampling_rate, "BAD_skip");
  t.channel_offsets.assign(t.onset.size() + 1, 0);
  return t;
}

// the header fields that identify a person or a session, by the readers' own names
bool identifying(const std::string& key) {
  const std::string k = lower(key);
  for (const char* w : {"patient", "subject", "experimenter", "recording_id", "recording id", "startdate", "start_date", "start_time", "date", "time", "birth", "name", "sex", "hand",
                        "description", "comment", "operator", "id"}) {
    if (k.find(w) != std::string::npos) return true;
  }
  return false;
}
}  // namespace

std::string format_of(const std::string& path) {
  if (!ends_with_any(path, {".fif", ".fif.gz"}).empty()) return "fif";
  if (!ends_with_any(path, {".edf"}).empty()) return "edf";
  if (!ends_with_any(path, {".bdf"}).empty()) return "bdf";
  if (!ends_with_any(path, {".vhdr"}).empty()) return "brainvision";
  if (!ends_with_any(path, {".set"}).empty()) return "eeglab";
  if (!ends_with_any(path, {".ds"}).empty() || (fs::is_directory(path) && lower(path).find(".ds") != std::string::npos)) return "ctf";
  if (!ends_with_any(path, {".con", ".sqd"}).empty()) return "kit";
  refuse("'" + path + "' is not a recording this build imports (choose from: .fif, .edf, .bdf, .vhdr, .set, .ds, .con/.sqd)");
}

std::unique_ptr<nxr::formats::RecordingReader> open_recording(const std::string& format, const std::string& path) {
  namespace nf = nxr::formats;
  if (format == "fif") return nf::fif::open_file(path);
  if (format == "edf" || format == "bdf") return nf::edf::open_file(path);
  if (format == "brainvision") return nf::brainvision::open_file(path);
  if (format == "eeglab") return nf::eeglab::open_file(path);
  if (format == "ctf") return nf::ctf::open(nf::ctf::DirectoryFolder(path));
  if (format == "kit") return nf::kit::open_file(path);
  refuse("'" + format + "' is not a recording format (choose from: fif, edf, bdf, brainvision, eeglab, ctf, kit)");
}

Json format_record(const std::string& format, const nxr::formats::RecordingReader& reader, bool anonymize) {
  namespace nf = nxr::formats;
  Json rec = Json::object();
  if (const auto* fif = dynamic_cast<const nf::fif::RawReader*>(&reader)) {
    const nf::fif::Info& in = fif->info();
    if (in.line_freq) rec["line_freq"] = *in.line_freq;
    if (in.highpass) rec["highpass"] = *in.highpass;
    if (in.lowpass) rec["lowpass"] = *in.lowpass;
    rec["first_samp"] = in.first_samp;
    rec["compensation_grade"] = in.compensation_grade;
    Json tf = Json::object();
    auto trans = [](const nf::fif::CoordTrans& t) { return Json{{"from", t.from}, {"to", t.to}, {"matrix", std::vector<double>(t.trans.begin(), t.trans.end())}, {"unit", "m"}}; };
    if (in.dev_head_t) tf["dev_head_t"] = trans(*in.dev_head_t);
    if (in.ctf_head_t) tf["ctf_head_t"] = trans(*in.ctf_head_t);
    if (in.dev_ctf_t) tf["dev_ctf_t"] = trans(*in.dev_ctf_t);
    if (!tf.empty()) rec["transforms"] = tf;
    if (in.has_dig) {
      Json dig = Json::array();
      for (const auto& d : in.dig) dig.push_back(Json{{"kind", d.kind}, {"ident", d.ident}, {"r", std::vector<double>(d.r.begin(), d.r.end())}, {"coord_frame", d.coord_frame}});
      rec["digitisation"] = dig;
    }
    Json proj = Json::array();
    for (const auto& p : in.projectors) proj.push_back(Json{{"description", p.desc}, {"active", p.active}, {"n_vectors", p.nvec}});
    if (!proj.empty()) rec["projectors"] = proj;   // recorded; the samples are copied as the file stores them (no projector is applied)
    if (!anonymize) {
      if (in.meas_date) rec["meas_date"] = Json::array({(*in.meas_date)[0], (*in.meas_date)[1]});
      if (in.experimenter) rec["experimenter"] = *in.experimenter;
      if (in.description) rec["description"] = *in.description;
    }
  } else if (const auto* ctf = dynamic_cast<const nf::ctf::Reader*>(&reader)) {
    rec["compensation_grade"] = ctf->compensation_grade();
    if (!anonymize) rec["start_epoch_seconds"] = ctf->start_epoch_seconds();
  } else if (const auto* kit = dynamic_cast<const nf::kit::Reader*>(&reader)) {
    rec["system_name"] = kit->info().system_name;
    rec["model_name"] = kit->info().model_name;
    if (!anonymize) rec["comment"] = kit->info().comment;
  }
  (void)format;
  return rec;
}

Json import_recording(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const nxr::formats::RecordingReader& reader, const RecordingImport& opt) {
  const nxr::formats::RecordingHeader& h = reader.header();
  if (h.channels.empty() || h.n_samples == 0 || h.sampling_rate <= 0) refuse("the recording has no channels, no samples or no sampling rate");
  const std::int64_t n_chan = static_cast<std::int64_t>(h.channels.size()), n_samp = static_cast<std::int64_t>(h.n_samples);
  const double t0 = first_time(reader);

  Json producer = Json::object();
  producer["format"] = opt.format;
  producer["reader"] = "nxr-neuro";
  producer["precision"] = h.precision == nxr::formats::Precision::float64 ? "float64" : "float32";
  producer["sampling_rate"] = h.sampling_rate;
  Json units = Json::array();
  for (const auto& c : h.channels) units.push_back(Json{{"name", c.name}, {"file_unit", c.file_unit}, {"unit", c.unit}, {"scale_to_si", c.scale_to_si}});
  producer["channel_units"] = units;
  producer["anonymized"] = opt.anonymize;
  Json info = Json::object();
  for (const auto& [k, v] : h.info) if (!(opt.anonymize && identifying(k))) info[k] = v;
  producer["header"] = info;
  if (!opt.anonymize && !h.start_time.empty()) producer["start_time"] = h.start_time;
  Json rec = format_record(opt.format, reader, opt.anonymize);
  for (auto it = rec.begin(); it != rec.end(); ++it) producer[it.key()] = it.value();
  if (opt.extra.is_object()) for (auto it = opt.extra.begin(); it != opt.extra.end(); ++it) producer[it.key()] = it.value();
  Json origin = Json{{"schema", "nxr-origin/1"}, {"source", opt.format + ":" + opt.source}, {"licence", opt.licence}, {"attribution", opt.attribution}, {"redistributable", opt.redistributable}};
  std::string sha;
  if (!opt.source_file.empty()) {
    sha = file_sha256(opt.source_file);
    origin["source"] = opt.format + ":" + opt.source + "#sha256=" + sha;
  }
  producer["origin"] = origin;

  Json out;
  sub.compose([&] {
    std::map<std::string, std::string> units_by_type;
    const SensorPlacement place = place_sensors(reader, opt.name + "_channels", opt.acquisition);
    make_frames(sub, place);
    auto mat = channel_mat(h, place, units_by_type);
    if (!place.digitisation.empty() && !place.headpoints_cs_key.empty()) mat["HeadPoints"] = headpoints_value(place.digitisation);
    std::vector<std::int64_t> flags;
    bool any_bad = false;
    for (const auto& c : h.channels) {
      const bool bad = c.bad || std::find(h.bad_channels.begin(), h.bad_channels.end(), c.name) != h.bad_channels.end();
      flags.push_back(bad ? -1 : 1);
      any_bad = any_bad || bad;
    }
    nxr::bst::SensorOptions so;
    so.name = opt.name + "_channels";
    so.acquisition = opt.acquisition;
    so.units = units_by_type;
    so.cs_key = place.cs_key.empty() ? "scs" : place.cs_key;
    so.placed = !place.cs_key.empty();
    if (!place.headpoints_cs_key.empty()) so.headpoints_cs_key = place.headpoints_cs_key;
    so.producer = Json{{"format", opt.format}, {"origin", origin}, {"frames", place.record}};
    const nxr::bst::SensorInfo si = nxr::bst::export_sensors(sub, mat, flags, so);
    write_links(sub, place, si.id, opt.acquisition.empty() ? std::nullopt : std::optional<std::string>(opt.acquisition));

    nxr::bst::BlockRecording b;
    b.name = opt.name;
    b.channels_id = si.id;
    b.source = "file";
    b.acquisition = opt.acquisition;
    b.n_channels = n_chan;
    b.n_samples = n_samp;
    b.sfreq = h.sampling_rate;
    b.t0 = t0;
    b.flags = flags;
    b.events = event_table(h, t0);
    b.channel_names = si.names;
    b.producer = producer;
    b.progress = opt.progress;
    b.read = [&](std::int64_t s0, std::int64_t n, std::vector<float>& blk) {
      blk.resize(static_cast<std::size_t>(n_chan * n));
      reader.read(static_cast<std::uint64_t>(s0), static_cast<std::uint64_t>(n), 0, static_cast<std::size_t>(n_chan), blk.data());
    };
    out = nxr::bst::write_recording_blocks(sub, b);
    out["channels"] = si.name;
    out["channels_id"] = si.id;
    out["n_bad_channels"] = std::count(flags.begin(), flags.end(), -1);
  });
  // an origin record on every item the import made
  int n_items = 0;
  for (const auto& it : nxr::io::items_without_origin(w.db())) {
    nxr::io::set_origin(w.db(), it.table, it.id, {origin["source"].get<std::string>(), opt.licence, opt.attribution, opt.redistributable});
    ++n_items;
  }
  out["origin"] = origin;
  out["origin_items"] = n_items;
  return out;
}

}  // namespace nxr::convert
