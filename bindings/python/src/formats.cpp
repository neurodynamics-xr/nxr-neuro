// nxr_neuro._core, the file formats (io/formats): each reader opens a file and gives arrays and plain dictionaries. A recording is opened by its
// header (small) and read by blocks the caller states; nothing is loaded whole unless the caller asks for the whole span.
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>

#include "arrays.h"
#include "nxr/formats/brainvision.h"
#include "nxr/formats/edf.h"
#include "nxr/formats/eeglab.h"
#include "nxr/formats/fif.h"
#include "nxr/formats/gifti.h"
#include "nxr/formats/nifti.h"
#include "nxr/formats/recording.h"
#include "nxr/formats/tsv.h"

namespace nxrpy {

namespace {

namespace fm = nxr::formats;

nb::object triple(const std::optional<std::array<double, 3>>& v) {
  if (!v) return nb::none();
  return nb::make_tuple((*v)[0], (*v)[1], (*v)[2]);
}

nb::dict channel_dict(const fm::Channel& c) {
  nb::dict d;
  d["name"] = c.name;
  d["kind"] = c.kind;
  d["unit"] = c.unit;
  d["file_unit"] = c.file_unit;
  d["scale_to_si"] = c.scale_to_si;
  d["gain"] = c.gain;
  d["offset"] = c.offset;
  d["sampling_rate"] = c.sampling_rate;
  d["transducer"] = c.transducer;
  d["prefilter"] = c.prefilter;
  d["reference"] = c.reference;
  d["location"] = triple(c.location);
  d["orientation"] = triple(c.orientation);
  d["coil_type"] = c.coil_type;
  d["bad"] = c.bad;
  return d;
}

nb::dict header_dict(const fm::RecordingHeader& h) {
  nb::dict d;
  d["format"] = h.format;
  nb::list channels;
  for (const auto& c : h.channels) channels.append(channel_dict(c));
  d["channels"] = channels;
  d["sampling_rate"] = h.sampling_rate;
  d["n_samples"] = h.n_samples;
  d["precision"] = h.precision == fm::Precision::float32 ? "float32" : "float64";
  d["start_time"] = h.start_time;
  nb::list annotations;
  for (const auto& a : h.annotations) {
    nb::dict e;
    e["onset"] = a.onset;
    e["duration"] = a.duration;
    e["label"] = a.label;
    e["channels"] = a.channels;
    annotations.append(e);
  }
  d["annotations"] = annotations;
  d["bad_channels"] = h.bad_channels;
  nb::list gaps;
  for (const auto& g : h.gaps) gaps.append(nb::make_tuple(g.first, g.count));
  d["gaps"] = gaps;
  d["position_unit"] = h.position_unit;
  d["info"] = h.info;
  return d;
}

/** An open recording: the header now, samples by block. */
class Recording {
 public:
  explicit Recording(std::shared_ptr<fm::RecordingReader> r) : r_(std::move(r)) {}
  nb::dict header() const { return header_dict(r_->header()); }
  // samples [start, stop) of channels [first_channel, last_channel): [channels, samples], C-ordered, in SI units, float32 or float64 as the file holds them
  nb::object read(uint64_t start, uint64_t stop, size_t first_channel, std::optional<size_t> last_channel) const {
    const auto& h = r_->header();
    const size_t c1 = last_channel ? *last_channel : h.channels.size();
    if (stop < start) throw fm::range_error("stop is before start");
    const uint64_t n = stop - start;
    if (c1 < first_channel) throw fm::range_error("last_channel is before first_channel");
    const size_t rows = c1 - first_channel;
    if (h.precision == fm::Precision::float32) {
      std::vector<float> buf(rows * n);
      { nb::gil_scoped_release release; r_->read(start, n, first_channel, c1, buf.data()); }
      return own_array<float>(std::move(buf), {rows, static_cast<size_t>(n)});
    }
    std::vector<double> buf(rows * n);
    { nb::gil_scoped_release release; r_->read(start, n, first_channel, c1, buf.data()); }
    return own_array<double>(std::move(buf), {rows, static_cast<size_t>(n)});
  }

 private:
  std::shared_ptr<fm::RecordingReader> r_;
};

Recording open_edf(const std::string& path, double sampling_rate) {
  fm::edf::OpenOptions o;
  o.sampling_rate = sampling_rate;
  std::unique_ptr<fm::RecordingReader> r;
  { nb::gil_scoped_release release; r = fm::edf::open_file(path, o); }
  return Recording(std::move(r));
}
Recording open_brainvision(const std::string& path) {
  std::unique_ptr<fm::RecordingReader> r;
  { nb::gil_scoped_release release; r = fm::brainvision::open_file(path); }
  return Recording(std::move(r));
}
Recording open_eeglab(const std::string& path) {
  std::unique_ptr<fm::RecordingReader> r;
  { nb::gil_scoped_release release; r = fm::eeglab::open_file(path); }
  return Recording(std::move(r));
}

Recording open_fif(const std::string& path, bool allow_maxshield) {
  fm::fif::OpenOptions o;
  o.allow_maxshield = allow_maxshield;
  std::unique_ptr<fm::RecordingReader> r;
  { nb::gil_scoped_release release; r = fm::fif::open_file(path, o); }
  return Recording(std::move(r));
}

// ---- NIfTI ----

nb::tuple read_nifti(const std::string& path) {
  fm::nifti::Volume v;
  { nb::gil_scoped_release release; v = fm::nifti::read_file(path); }
  const auto& h = v.header;
  nb::dict d;
  d["version"] = h.version;
  d["little_endian"] = h.little_endian;
  d["datatype"] = h.datatype;
  d["bitpix"] = h.bitpix;
  d["shape"] = h.shape;
  d["pixdim"] = h.pixdim;
  d["scl_slope"] = h.scl_slope;
  d["scl_inter"] = h.scl_inter;
  d["cal_min"] = h.cal_min;
  d["cal_max"] = h.cal_max;
  d["toffset"] = h.toffset;
  d["slice_duration"] = h.slice_duration;
  d["dim_info"] = h.dim_info;
  d["xyzt_units"] = h.xyzt_units;
  d["intent_code"] = h.intent_code;
  d["intent_name"] = h.intent_name;
  d["descrip"] = h.descrip;
  d["aux_file"] = h.aux_file;
  d["qform_code"] = h.qform_code;
  d["sform_code"] = h.sform_code;
  d["spatial_unit"] = h.spatial_unit();
  d["time_unit"] = h.time_unit();
  d["unit_to_metres"] = h.unit_to_metres();
  const char* src = h.affine_source() == fm::nifti::Header::AffineSource::sform ? "sform"
                    : h.affine_source() == fm::nifti::Header::AffineSource::qform ? "qform" : "pixdim";
  d["affine_source"] = src;
  const auto a = h.affine();   // 3 x 4 row-major; the 4 x 4 adds the row 0 0 0 1
  d["affine"] = own_array<double>({a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], 0, 0, 0, 1}, {4, 4});
  nb::list extensions;
  for (const auto& e : h.extensions) extensions.append(nb::make_tuple(e.code, nb::bytes(reinterpret_cast<const char*>(e.bytes.data()), e.bytes.size())));
  d["extensions"] = extensions;
  std::vector<size_t> shape(h.shape.begin(), h.shape.end());
  // the file stores x fastest: a Fortran-ordered array of the file's shape, no transpose and no copy
  nb::object data = v.precision == fm::Precision::float32 ? own_array<float>(std::move(v.f32), shape, true) : own_array<double>(std::move(v.f64), shape, true);
  return nb::make_tuple(data, d);
}

// ---- GIFTI ----

nb::object gifti_values(fm::gifti::DataArray& a) {
  std::vector<size_t> shape(a.dims.begin(), a.dims.end());
  if (a.datatype == "NIFTI_TYPE_INT32") return own_array<int32_t>(std::move(a.i32), shape);
  if (a.datatype == "NIFTI_TYPE_UINT8") return own_array<uint8_t>(std::move(a.u8), shape);
  return own_array<float>(std::move(a.f32), shape);
}

nb::dict read_gifti(const std::string& path) {
  fm::gifti::Image img;
  { nb::gil_scoped_release release; img = fm::gifti::read_file(path); }
  nb::dict out;
  out["version"] = img.version;
  nb::dict meta;
  for (const auto& kv : img.metadata) meta[nb::str(kv.first.c_str())] = kv.second;
  out["metadata"] = meta;
  nb::list labels;
  for (const auto& l : img.labels) {
    nb::dict e;
    e["key"] = l.key;
    e["name"] = l.name;
    e["rgba"] = l.has_colour ? nb::object(nb::make_tuple(l.rgba[0], l.rgba[1], l.rgba[2], l.rgba[3])) : nb::none();
    labels.append(e);
  }
  out["labels"] = labels;
  nb::list arrays;
  for (auto& a : img.arrays) {
    nb::dict e;
    e["intent"] = a.intent;
    e["datatype"] = a.datatype;
    nb::dict am;
    for (const auto& kv : a.metadata) am[nb::str(kv.first.c_str())] = kv.second;
    e["metadata"] = am;
    nb::list transforms;
    for (const auto& t : a.transforms) {
      nb::dict td;
      td["dataspace"] = t.dataspace;
      td["xformspace"] = t.xformspace;
      td["matrix"] = own_array<double>(std::vector<double>(t.matrix.begin(), t.matrix.end()), {4, 4});
      transforms.append(td);
    }
    e["transforms"] = transforms;
    e["data"] = gifti_values(a);
    arrays.append(e);
  }
  out["arrays"] = arrays;
  return out;
}

// ---- TSV ----

nb::dict read_tsv(const std::string& path) {
  fm::tsv::Table t;
  { nb::gil_scoped_release release; t = fm::tsv::read_file(path); }
  nb::dict d;
  for (size_t c = 0; c < t.columns.size(); ++c) {
    nb::list col;
    for (const auto& row : t.rows) col.append(row[c]);
    d[nb::str(t.columns[c].c_str())] = col;
  }
  return d;
}

}  // namespace

void bind_formats(nb::module_& m) {
  nb::class_<Recording>(m, "Recording")
      .def("header", &Recording::header)
      .def("read", &Recording::read, nb::arg("start"), nb::arg("stop"), nb::arg("first_channel"), nb::arg("last_channel").none());
  m.def("open_edf", &open_edf, nb::arg("path"), nb::arg("sampling_rate"));
  m.def("open_brainvision", &open_brainvision, nb::arg("path"));
  m.def("open_eeglab", &open_eeglab, nb::arg("path"));
  m.def("open_fif", &open_fif, nb::arg("path"), nb::arg("allow_maxshield"));
  m.def("read_nifti", &read_nifti, nb::arg("path"));
  m.def("read_gifti", &read_gifti, nb::arg("path"));
  m.def("read_tsv", &read_tsv, nb::arg("path"));
}

}  // namespace nxrpy
