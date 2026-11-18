// Where an imported recording's channel positions are: see nxr/convert/sensor_frames.h.
#include "nxr/convert/sensor_frames.h"

#include <cmath>
#include <map>

#include "nxr/formats/ctf.h"
#include "nxr/formats/fif.h"
#include "nxr/formats/geometry.h"
#include "nxr/formats/kit.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/errors.h"

namespace nxr::convert {

namespace nf = nxr::formats;
namespace geom = nxr::formats::geom;
using nxr::io::Json;
namespace code = nxr::io::code;

namespace {
using Point = geom::Point;
using Matrix4 = geom::Matrix4;

[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

// the metres in one unit of `unit`; none for an unstated one (never assumed to be metres)
std::optional<double> metres_per(const std::string& unit) {
  if (unit == "m") return 1.0;
  if (unit == "cm") return 1e-2;
  if (unit == "mm") return 1e-3;
  if (unit.empty()) return std::nullopt;
  refuse("a channel position unit '" + unit + "' is not one of m, cm, mm");
}

std::array<double, 12> three_by_four(const Matrix4& m) {
  std::array<double, 12> o{};
  for (std::size_t i = 0; i < 12; ++i) o[i] = m[i];
  return o;
}

struct Chain {
  std::string head_key;                                  // "head" · "ctf-head"; empty: the format has none
  std::optional<Matrix4> device_to_head;
  std::optional<std::array<Point, 3>> landmarks;         // nasion, left, right, in the head frame
  std::string device_to_head_source;
};

std::optional<std::array<Point, 3>> fif_landmarks(const nf::fif::Info& in) {
  std::optional<Point> nas, lpa, rpa;
  for (const auto& d : in.dig) {
    if (d.kind != 1 || d.coord_frame != 4) continue;
    if (d.ident == 2) nas = d.r; else if (d.ident == 1) lpa = d.r; else if (d.ident == 3) rpa = d.r;
  }
  if (nas && lpa && rpa) return std::array<Point, 3>{*nas, *lpa, *rpa};
  return std::nullopt;
}

std::string cardinal_label(int ident) { return ident == 1 ? "LPA" : ident == 2 ? "NAS" : ident == 3 ? "RPA" : ident == 4 ? "INI" : "CARDINAL" + std::to_string(ident); }

std::vector<SensorPlacement::DigitisedPoint> digitisation_of(const nf::RecordingReader& reader) {
  using D = SensorPlacement::DigitisedPoint;
  std::vector<D> out;
  if (const auto* fif = dynamic_cast<const nf::fif::RawReader*>(&reader)) {
    for (const auto& d : fif->info().dig) {
      if (d.coord_frame != 4) continue;
      const std::string id = std::to_string(d.ident);
      if (d.kind == 1) out.push_back({d.r, "CARDINAL", cardinal_label(d.ident)});
      else if (d.kind == 2) out.push_back({d.r, "HPI", "HPI" + id});
      else if (d.kind == 3) out.push_back({d.r, "EEG", "EEG" + id});
      else out.push_back({d.r, "EXTRA", "EXTRA" + id});
    }
  } else if (const auto* ctf = dynamic_cast<const nf::ctf::Reader*>(&reader)) {
    for (const auto& c : ctf->coils()) {
      if (!c.valid || c.coord_frame != 1004) continue;
      if (c.kind == 1 || c.kind == 2 || c.kind == 3) out.push_back({c.r, "CARDINAL", c.kind == 1 ? "LPA" : c.kind == 2 ? "RPA" : "NAS"});
      else out.push_back({c.r, "HPI", c.descriptor.empty() ? "HPI" : c.descriptor});
    }
    for (const auto& e : ctf->electrodes()) out.push_back({e.r, "EEG", e.label.empty() ? "EEG" + std::to_string(e.id) : e.label});
    for (const auto& q : ctf->polhemus()) out.push_back({q.r, "EXTRA", q.label.empty() ? q.id : q.label});
  } else if (const auto* kit = dynamic_cast<const nf::kit::Reader*>(&reader)) {
    const auto& dg = kit->info().digitization;
    if (dg && dg->elp.size() >= 3) {
      const Matrix4 to_head = geom::neuromag_head_from_landmarks(dg->elp[0], dg->elp[1], dg->elp[2]);
      static const char* fid[3] = {"NAS", "LPA", "RPA"};
      for (std::size_t k = 0; k < dg->elp.size(); ++k)
        out.push_back({geom::apply(to_head, dg->elp[k]), k < 3 ? "CARDINAL" : "HPI", k < 3 ? std::string(fid[k]) : "HPI" + std::to_string(k - 2)});
      for (const auto& q : dg->head_shape) out.push_back({geom::apply(to_head, q), "EXTRA", "EXTRA"});
    }
  }
  return out;
}

Chain chain_of(const nf::RecordingReader& reader) {
  Chain c;
  if (const auto* fif = dynamic_cast<const nf::fif::RawReader*>(&reader)) {
    const nf::fif::Info& in = fif->info();
    c.head_key = "head";
    if (in.dev_head_t && in.dev_head_t->from == 1 && in.dev_head_t->to == 4) c.device_to_head = in.dev_head_t->trans;
    c.device_to_head_source = "the file's dev_head_t";
    c.landmarks = fif_landmarks(in);
  } else if (const auto* ctf = dynamic_cast<const nf::ctf::Reader*>(&reader)) {
    c.head_key = "ctf-head";
    c.device_to_head = ctf->transforms().ctf_dev_to_ctf_head;
    c.device_to_head_source = "the fit of the device-frame head coils to their head-frame positions (.hc)";
    std::optional<Point> nas, lpa, rpa;
    for (const auto& p : ctf->coils()) {
      if (!p.valid || p.coord_frame != 1004) continue;
      if (p.kind == 3 && !nas) nas = p.r; else if (p.kind == 1 && !lpa) lpa = p.r; else if (p.kind == 2 && !rpa) rpa = p.r;
    }
    if (nas && lpa && rpa) c.landmarks = std::array<Point, 3>{*nas, *lpa, *rpa};
  } else if (const auto* kit = dynamic_cast<const nf::kit::Reader*>(&reader)) {
    c.head_key = "head";
    c.device_to_head_source = "the fit of the marker coils (device frame) to the digitised HPI points (Neuromag head)";
    const auto& dg = kit->info().digitization;
    if (dg && dg->elp.size() >= 3) {
      const Matrix4 to_head = geom::neuromag_head_from_landmarks(dg->elp[0], dg->elp[1], dg->elp[2]);
      c.landmarks = std::array<Point, 3>{geom::apply(to_head, dg->elp[0]), geom::apply(to_head, dg->elp[1]), geom::apply(to_head, dg->elp[2])};
      std::vector<Point> dev, hd;
      for (std::size_t k = 0; k < dg->marker_coils.size() && 3 + k < dg->elp.size(); ++k) {
        if (k < dg->marker_done.size() && !dg->marker_done[k]) continue;
        dev.push_back(dg->marker_coils[k]);
        hd.push_back(geom::apply(to_head, dg->elp[3 + k]));
      }
      if (dev.size() >= 3) c.device_to_head = geom::fit_rigid(dev, hd);
    }
  }
  return c;
}
}  // namespace

SensorPlacement place_sensors(const nf::RecordingReader& reader, const std::string& base, const std::string& acquisition, const nf::RecordingHeader* header) {
  const nf::RecordingHeader& h = header ? *header : reader.header();
  const std::size_t n = h.channels.size();
  SensorPlacement p;
  p.location.assign(n, std::nullopt);
  p.orientation.assign(n, std::nullopt);
  p.record = Json::object();

  std::map<std::string, std::size_t> frames;       // the frames the channels' positions are in, and how many each
  for (const auto& c : h.channels) if (c.location) ++frames[c.position_frame];
  if (frames.empty()) {
    p.record["positions"] = "none in the file";
    return p;
  }
  const std::optional<double> to_m = metres_per(h.position_unit);

  // formats whose frame is their own, with no head to relate it to
  const bool bids_frame = frames.size() == 1 && frames.begin()->first.rfind("bids-", 0) == 0;      // a BIDS electrodes table: the coordinate system it names, with no transform to a head
  if (frames.size() == 1 && (frames.begin()->first == "sphere" || frames.begin()->first == "eeglab" || bids_frame)) {
    const std::string key = bids_frame ? frames.begin()->first : frames.begin()->first == "sphere" ? "brainvision-sphere" : "eeglab";
    p.cs_key = key;
    const double s = to_m.value_or(1.0);
    for (std::size_t i = 0; i < n; ++i) {
      const auto& c = h.channels[i];
      if (c.location) p.location[i] = Point{(*c.location)[0] * s, (*c.location)[1] * s, (*c.location)[2] * s};
      if (c.orientation) p.orientation[i] = *c.orientation;
    }
    SensorPlacement::System sy;
    sy.space = key;
    sy.name = key;
    if (to_m) sy.unit = "m";
    sy.description = bids_frame ? "the coordinate system the BIDS electrodes table names (its `space` entity and coordsystem file), in metres"
              : key == "eeglab" ? "EEGLAB's own axes: x toward the nose, y left, z up; the file's origin and unit"
                                : "BrainVision's coordinates: the electrodes on a sphere of the file's radius (a unit sphere unless it says otherwise)";
    sy.space_description = "a frame of its own: no transition relates it to the subject's head space, so its positions are not comparable to the anatomy";
    p.systems.push_back(sy);
    p.record["frame"] = key;
    p.record["position_unit"] = to_m ? Json(h.position_unit) : Json("unknown");
    p.record["comparable"] = false;
    return p;
  }

  if (!to_m) refuse("channel positions in a device or head frame with no stated unit: the unit is never assumed");
  const double s = *to_m;
  const Chain ch = chain_of(reader);
  p.head_key = ch.head_key;
  if (ch.landmarks) {
    p.landmarks = *ch.landmarks;
    p.head_to_scs = geom::scs_from_landmarks((*ch.landmarks)[0], (*ch.landmarks)[1], (*ch.landmarks)[2]);
  }
  const bool has_device = frames.count("device") > 0, has_head = frames.count("head") > 0;
  const std::string device_key = "device:" + (acquisition.empty() ? base : acquisition);

  // the frame the manifold is written in: scs when the chain reaches it, else the head frame, else the device frame
  Matrix4 id{};
  id[0] = id[5] = id[10] = id[15] = 1.0;
  Matrix4 dev = id, hd = id;                     // device -> target, head -> target
  std::string target;
  std::size_t dropped = 0;
  const bool dev_ok = ch.device_to_head.has_value();
  if (ch.head_key.empty() || (has_device && !has_head && !dev_ok)) {
    target = device_key;                         // only device positions, and no way to the head
  } else {
    target = ch.head_key;
    if (p.head_to_scs) { target = "scs"; hd = *p.head_to_scs; }
    if (dev_ok) dev = geom::multiply(hd, *ch.device_to_head);
  }
  p.cs_key = target;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& c = h.channels[i];
    if (!c.location) continue;
    const bool is_dev = c.position_frame == "device";
    if (!is_dev && c.position_frame != "head") { ++dropped; continue; }           // a frame the file does not name
    if (is_dev && target != device_key && !dev_ok) { ++dropped; continue; }      // a device position with no way to the frame
    if (!is_dev && target == device_key) { ++dropped; continue; }
    const Matrix4& m = is_dev ? dev : hd;
    const Point loc{(*c.location)[0] * s, (*c.location)[1] * s, (*c.location)[2] * s};
    p.location[i] = geom::apply(m, loc);
    if (c.orientation) p.orientation[i] = geom::rotate(m, *c.orientation);
  }
  p.digitisation = digitisation_of(reader);
  p.record["frame"] = target;
  p.record["file_frames"] = Json::object();
  for (const auto& [f, k] : frames) p.record["file_frames"][f] = k;
  p.record["position_unit"] = h.position_unit;
  if (dropped) p.record["positions_not_placed"] = dropped;

  // the systems and transitions the placement rests on
  if (target == device_key) {
    SensorPlacement::System sy;
    sy.space = "device:" + (acquisition.empty() ? base : acquisition);
    sy.name = device_key;
    sy.unit = "m";
    sy.description = "the sensor array's own frame in this acquisition; the file gives no transform to a head frame, so the positions are not comparable to the anatomy";
    sy.space_description = "a frame of its own: no transition relates it to the subject's head space";
    p.systems.push_back(sy);
    p.record["comparable"] = false;
    return p;
  }
  p.headpoints_cs_key = ch.head_key;
  const bool in_head_space = p.head_to_scs.has_value();
  const std::string space = in_head_space ? "head" : "unreferenced:" + ch.head_key;
  {
    SensorPlacement::System sy;
    sy.space = space;
    sy.name = ch.head_key;
    sy.unit = "m";
    sy.description = ch.head_key == "ctf-head" ? "CTF's head frame: origin between the ears, x toward the nasion (defined by the head coils)"
                                               : "the Neuromag head frame: x toward the right ear, y toward the nasion, z up (defined by the digitised landmarks)";
    sy.space_description = in_head_space ? "" : "a head frame with no landmarks to relate it to the subject coordinate system: its positions are not comparable to the anatomy";
    p.systems.push_back(sy);
  }
  if (has_device && dev_ok) {
    SensorPlacement::System sy;
    sy.space = space;
    sy.name = device_key;
    sy.unit = "m";
    sy.description = "the sensor array's own frame in this acquisition (the head moves between acquisitions); its transition to the head frame is the file's";
    p.systems.push_back(sy);
    SensorPlacement::Link l;
    l.from = device_key;
    l.to = ch.head_key;
    l.name = base + "_device_to_" + ch.head_key;
    l.matrix = three_by_four(*ch.device_to_head);
    l.description = "device frame \xe2\x86\x92 " + ch.head_key + " (3x4 affine, metres): " + ch.device_to_head_source;
    l.producer = Json{{"source", ch.device_to_head_source}};
    p.links.push_back(std::move(l));
  }
  if (const auto* fif = dynamic_cast<const nf::fif::RawReader*>(&reader); fif && ch.head_key == "head") {
    // the file's CTF frames (a CTF recording converted to FIF): ctf_head_t (CTF head <-> Neuromag head) and dev_ctf_t (CTF device <-> CTF head), as transitions, each in the
    // direction it is stated for (4 head, 1001 CTF device, 1004 CTF head) or inverted to it
    auto system = [&](const std::string& key, const std::string& description) {
      for (const auto& x : p.systems) if (x.name == key) return;
      SensorPlacement::System sy;
      sy.space = space;
      sy.name = key;
      sy.unit = "m";
      sy.description = description;
      p.systems.push_back(sy);
    };
    auto link = [&](const std::optional<nf::fif::CoordTrans>& t, int a, int b, const std::string& from, const std::string& to, const std::string& name, const std::string& what) {
      if (!t) return;
      Matrix4 m;
      if (t->from == a && t->to == b) m = t->trans;
      else if (t->from == b && t->to == a) m = geom::invert(t->trans);
      else return;
      SensorPlacement::Link l;
      l.from = from;
      l.to = to;
      l.name = base + "_" + name;
      l.matrix = three_by_four(m);
      l.description = from + " \xe2\x86\x92 " + to + " (3x4 affine, metres): the file's " + what;
      l.producer = Json{{"source", "the file's " + what}};
      p.links.push_back(std::move(l));
    };
    const nf::fif::Info& in = fif->info();
    if (in.ctf_head_t || in.dev_ctf_t) system("ctf-head", "CTF's head frame: origin between the ears, x toward the nasion (defined by the head coils)");
    const std::string ctf_device_key = "ctf-device:" + (acquisition.empty() ? base : acquisition);
    if (in.dev_ctf_t) system(ctf_device_key, "the CTF device frame of this acquisition (the sensor array in the CTF recording the file was converted from)");
    link(in.ctf_head_t, 1004, 4, "ctf-head", "head", "ctf_head_to_head", "ctf_head_t");
    link(in.dev_ctf_t, 1001, 1004, ctf_device_key, "ctf-head", "ctf_device_to_ctf_head", "dev_ctf_t");
  }
  if (in_head_space) {
    SensorPlacement::Link l;
    l.from = ch.head_key;
    l.to = "scs";
    l.name = base + "_" + ch.head_key + "_to_scs";
    l.matrix = three_by_four(*p.head_to_scs);
    l.description = ch.head_key + " \xe2\x86\x92 SCS (3x4 affine, metres), from the digitised nasion, left and right ear points: origin midway between the ears, x toward the nasion";
    l.producer = Json{{"landmarks", Json{{"nasion", (*ch.landmarks)[0]}, {"lpa", (*ch.landmarks)[1]}, {"rpa", (*ch.landmarks)[2]}}}, {"convention", "Brainstorm's SCS"}};
    p.links.push_back(std::move(l));
  } else {
    p.record["comparable"] = false;
  }
  return p;
}

void make_frames(nxr::io::SubjectWriter& sub, const SensorPlacement& p) {
  for (const auto& s : p.systems) sub.frame({s.space, s.name, s.unit, s.description, s.space_description});
}

void write_links(nxr::io::SubjectWriter& sub, const SensorPlacement& p, const std::string& from_manifold_id, const std::optional<std::string>& acquisition) {
  for (const auto& l : p.links) {
    nxr::io::OperatorSpec op;
    op.name = l.name;
    op.path = l.name;
    op.kind = "transition";
    op.from_manifold_id = from_manifold_id;
    op.cols.set("from_coordinate_system_id", sub.cs(l.from));
    op.cols.set("to_coordinate_system_id", sub.cs(l.to));
    op.n_rows = 3;
    op.n_cols = 4;
    op.data = nxr::io::view(l.matrix.data(), {3, 4});
    op.acquisition = acquisition;
    op.cols.set("description", l.description);
    op.cols.json("producer_json", l.producer);
    sub.operator_(std::move(op));
  }
}

std::string import_head_mri_transition(nxr::io::SubjectWriter& sub, const nf::fif::CoordTrans& t, const std::string& name, const std::string& from_manifold_id,
                                       const std::string& head_key) {
  Matrix4 m = t.trans;
  if (t.from == 5 && t.to == 4) m = geom::invert(m);
  else if (!(t.from == 4 && t.to == 5)) refuse("the transform is not between the head (4) and the MRI (5) frames");
  std::string id;
  sub.compose([&] {
    sub.frame({"head", head_key, std::string("m"), head_key == "ctf-head" ? "CTF's head frame" : "the Neuromag head frame: x toward the right ear, y toward the nasion, z up", ""});
    sub.frame({"head", "freesurfer-tkr", std::string("m"), "FreeSurfer's surface RAS (tkr) in metres, as its own surfaces are stored", ""});
  const std::array<double, 12> three = three_by_four(m);
  nxr::io::OperatorSpec op;
  op.name = name;
  op.path = name;
  op.kind = "transition";
  op.from_manifold_id = from_manifold_id;
  op.cols.set("from_coordinate_system_id", sub.cs(head_key));
  op.cols.set("to_coordinate_system_id", sub.cs("freesurfer-tkr"));
  op.n_rows = 3;
  op.n_cols = 4;
  op.data = nxr::io::view(three.data(), {3, 4});
  op.cols.set("description", head_key + " \xe2\x86\x92 the MRI's surface RAS (3x4 affine, metres), from an MNE head-to-MRI transform");
  op.cols.json("producer_json", Json{{"source", "head-to-MRI transform"}});
  id = sub.operator_(std::move(op));
  });
  return id;
}

}  // namespace nxr::convert
