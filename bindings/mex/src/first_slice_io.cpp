// The first slice's readers (`io.*` commands): each opens a file with io/formats and gives arrays and structs. A recording is opened by its header
// (small) and read by blocks the caller states; nothing is loaded whole unless the caller asks for the whole span. The MATLAB class
// nxr.io.Recording holds the handle this file keeps.
//
// Indices at this boundary are MATLAB's: samples and channels are 1-based and inclusive (first, last); the gaps' first sample and a GIFTI
// triangle's vertex indices come back 1-based.
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "first_slice_internal.h"
#include "nxr/formats/brainvision.h"
#include "nxr/formats/edf.h"
#include "nxr/formats/eeglab.h"
#include "nxr/formats/fif.h"
#include "nxr/formats/gifti.h"
#include "nxr/formats/nifti.h"
#include "nxr/formats/recording.h"
#include "nxr/formats/tsv.h"

namespace nxr::mex {

namespace {

using namespace nxr::mex::slice;
namespace fm = nxr::formats;

std::map<std::uint64_t, std::shared_ptr<fm::RecordingReader>> sRecordings;
std::uint64_t sNextHandle = 1;
bool sAtExit = false;

std::shared_ptr<fm::RecordingReader> recording(const mxArray* a) {
  if (!mxIsUint64(a) || mxGetNumberOfElements(a) != 1) throw std::invalid_argument("not a recording handle");
  const std::uint64_t h = *static_cast<const std::uint64_t*>(mxGetData(a));
  auto it = sRecordings.find(h);
  if (it == sRecordings.end()) throw std::invalid_argument("the recording is closed (or not one)");
  return it->second;
}

mxArray* triple(const std::optional<std::array<double, 3>>& v) {
  if (!v) return mxCreateDoubleMatrix(0, 0, mxREAL);
  mxArray* a = mxCreateDoubleMatrix(1, 3, mxREAL);
  for (int i = 0; i < 3; ++i) mxGetPr(a)[i] = (*v)[static_cast<std::size_t>(i)];
  return a;
}

// a std::map<string,string> as a struct (the file's own names, as MATLAB field names)
mxArray* map_struct(const std::map<std::string, std::string>& m) {
  mxArray* s = mxCreateStructMatrix(1, 1, 0, nullptr);
  for (const auto& kv : m) {
    const std::string f = field_name(kv.first);
    if (mxGetFieldNumber(s, f.c_str()) >= 0) continue;   // two names that become one: the first stays
    mxAddField(s, f.c_str());
    mxSetField(s, 0, f.c_str(), give_string(kv.second));
  }
  return s;
}

mxArray* struct_array(const std::vector<std::string>& fields, std::size_t n) {
  std::vector<const char*> names;
  for (const auto& f : fields) names.push_back(f.c_str());
  return mxCreateStructMatrix(static_cast<mwSize>(n), 1, static_cast<int>(names.size()), names.data());
}

mxArray* header_struct(const fm::RecordingHeader& h) {
  const std::vector<std::string> chFields = {"name", "kind", "unit", "file_unit", "scale_to_si", "gain", "offset", "sampling_rate", "transducer", "prefilter",
                                             "reference", "location", "orientation", "coil_type", "bad"};
  mxArray* channels = struct_array(chFields, h.channels.size());
  for (std::size_t i = 0; i < h.channels.size(); ++i) {
    const fm::Channel& c = h.channels[i];
    const mwIndex k = static_cast<mwIndex>(i);
    mxSetField(channels, k, "name", give_string(c.name));
    mxSetField(channels, k, "kind", give_string(c.kind));
    mxSetField(channels, k, "unit", give_string(c.unit));
    mxSetField(channels, k, "file_unit", give_string(c.file_unit));
    mxSetField(channels, k, "scale_to_si", give_scalar(c.scale_to_si));
    mxSetField(channels, k, "gain", give_scalar(c.gain));
    mxSetField(channels, k, "offset", give_scalar(c.offset));
    mxSetField(channels, k, "sampling_rate", give_scalar(c.sampling_rate));
    mxSetField(channels, k, "transducer", give_string(c.transducer));
    mxSetField(channels, k, "prefilter", give_string(c.prefilter));
    mxSetField(channels, k, "reference", give_string(c.reference));
    mxSetField(channels, k, "location", triple(c.location));
    mxSetField(channels, k, "orientation", triple(c.orientation));
    mxSetField(channels, k, "coil_type", give_string(c.coil_type));
    mxSetField(channels, k, "bad", give_bool(c.bad));
  }
  mxArray* annotations = struct_array({"onset", "duration", "label", "channels"}, h.annotations.size());
  for (std::size_t i = 0; i < h.annotations.size(); ++i) {
    const fm::Annotation& a = h.annotations[i];
    const mwIndex k = static_cast<mwIndex>(i);
    mxSetField(annotations, k, "onset", give_scalar(a.onset));
    mxSetField(annotations, k, "duration", give_scalar(a.duration));
    mxSetField(annotations, k, "label", give_string(a.label));
    mxSetField(annotations, k, "channels", give_string(a.channels));
  }
  mxArray* gaps = mxCreateDoubleMatrix(static_cast<mwSize>(h.gaps.size()), 2, mxREAL);
  for (std::size_t i = 0; i < h.gaps.size(); ++i) {   // [first sample (1-based), count]
    mxGetPr(gaps)[i] = static_cast<double>(h.gaps[i].first) + 1.0;
    mxGetPr(gaps)[i + h.gaps.size()] = static_cast<double>(h.gaps[i].count);
  }
  return make_struct({{"format", give_string(h.format)},
                      {"channels", channels},
                      {"sampling_rate", give_scalar(h.sampling_rate)},
                      {"n_samples", give_scalar(static_cast<double>(h.n_samples))},
                      {"precision", give_string(h.precision == fm::Precision::float32 ? "single" : "double")},
                      {"start_time", give_string(h.start_time)},
                      {"annotations", annotations},
                      {"bad_channels", give_cellstr(h.bad_channels)},
                      {"gaps", gaps},
                      {"position_unit", give_string(h.position_unit)},
                      {"info", map_struct(h.info)}});
}

// io.open(format, path, option) -> handle. option: edf the sampling rate (Hz; 0 = the file's only rate); fif allow_maxshield (0 or 1); others unused.
void ioOpen(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 4) throw std::invalid_argument("nxr_compute('io.open') takes 3 arguments (use the nxr.io functions)");
  const std::string format = get_string(prhs[1], "format"), path = get_string(prhs[2], "path");
  const double option = get_scalar(prhs[3], "option");
  std::unique_ptr<fm::RecordingReader> r;
  if (format == "edf") {
    fm::edf::OpenOptions o;
    o.sampling_rate = option;
    r = fm::edf::open_file(path, o);
  } else if (format == "brainvision") {
    r = fm::brainvision::open_file(path);
  } else if (format == "eeglab") {
    r = fm::eeglab::open_file(path);
  } else if (format == "fif") {
    fm::fif::OpenOptions o;
    o.allow_maxshield = option != 0.0;
    r = fm::fif::open_file(path, o);
  } else {
    throw std::invalid_argument("format '" + format + "' is not one of edf, brainvision, eeglab, fif");
  }
  if (!sAtExit) {
    mexAtExit([]() { sRecordings.clear(); });
    sAtExit = true;
  }
  const std::uint64_t h = sNextHandle++;
  sRecordings[h] = std::move(r);
  mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *static_cast<std::uint64_t*>(mxGetData(out)) = h;
  plhs[0] = out;
}

void ioClose(int, mxArray**, int nrhs, const mxArray** prhs) {
  if (nrhs != 2) throw std::invalid_argument("nxr_compute('io.close') takes 1 argument");
  if (mxIsUint64(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1) sRecordings.erase(*static_cast<const std::uint64_t*>(mxGetData(prhs[1])));
}

void ioHeader(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 2) throw std::invalid_argument("nxr_compute('io.header') takes 1 argument");
  plhs[0] = header_struct(recording(prhs[1])->header());
}

// x = io.read(handle, first, last, first_channel, last_channel): samples first..last of channels first_channel..last_channel, 1-based, inclusive;
// [channels x samples], SI units, single for a file that holds integers or 32-bit floats, double for a 64-bit one.
void ioRead(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 6) throw std::invalid_argument("nxr_compute('io.read') takes 5 arguments");
  auto r = recording(prhs[1]);
  const auto& h = r->header();
  const double first = get_scalar(prhs[2], "first"), last = get_scalar(prhs[3], "last");
  const double cfirst = get_scalar(prhs[4], "first_channel"), clast = get_scalar(prhs[5], "last_channel");
  if (first < 1 || last < first - 1 || first != std::floor(first) || last != std::floor(last))
    throw nxr::formats::range_error("the span is first..last, 1-based and inclusive, with first >= 1 and last >= first - 1");
  if (cfirst < 1 || clast < cfirst - 1 || cfirst != std::floor(cfirst) || clast != std::floor(clast))
    throw nxr::formats::range_error("the channels are first_channel..last_channel, 1-based and inclusive");
  const std::uint64_t s0 = static_cast<std::uint64_t>(first) - 1, n = static_cast<std::uint64_t>(last - first + 1);
  const std::size_t c0 = static_cast<std::size_t>(cfirst) - 1, c1 = static_cast<std::size_t>(clast);
  (void)h;
  const std::size_t rows = c1 - c0;
  const bool single = r->header().precision == fm::Precision::float32;
  // the reader gives channel after channel (row-major [rows x n]); MATLAB wants the same matrix column-major: one transposing copy
  mxArray* out = mxCreateNumericMatrix(static_cast<mwSize>(rows), static_cast<mwSize>(n), single ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
  if (single) {
    std::vector<float> buf(rows * n);
    r->read(s0, n, c0, c1, buf.data());
    float* p = static_cast<float*>(mxGetData(out));
    for (std::size_t c = 0; c < rows; ++c)
      for (std::size_t i = 0; i < n; ++i) p[c + rows * i] = buf[c * n + i];
  } else {
    std::vector<double> buf(rows * n);
    r->read(s0, n, c0, c1, buf.data());
    double* p = static_cast<double*>(mxGetData(out));
    for (std::size_t c = 0; c < rows; ++c)
      for (std::size_t i = 0; i < n; ++i) p[c + rows * i] = buf[c * n + i];
  }
  plhs[0] = out;
}

// ---- N-D arrays ----

mxClassID class_of(const std::int32_t*) { return mxINT32_CLASS; }
mxClassID class_of(const std::uint8_t*) { return mxUINT8_CLASS; }
mxClassID class_of(const float*) { return mxSINGLE_CLASS; }
mxClassID class_of(const double*) { return mxDOUBLE_CLASS; }

std::vector<mwSize> matlab_dims(const std::vector<std::size_t>& shape) {
  std::vector<mwSize> d(shape.begin(), shape.end());
  if (d.empty()) d = {1, 1};
  if (d.size() == 1) d.push_back(1);   // a vector is a column
  return d;
}

// an array stored with the first index slowest (C order) as MATLAB's column-major array of the same shape (indices added to each value: 1 makes 0-based ones 1-based)
template <class T>
mxArray* from_row_major(const std::vector<T>& v, const std::vector<std::size_t>& shape, int add = 0) {
  const auto dims = matlab_dims(shape);
  mxArray* a = mxCreateNumericArray(static_cast<mwSize>(dims.size()), dims.data(), class_of(static_cast<const T*>(nullptr)), mxREAL);
  T* out = static_cast<T*>(mxGetData(a));
  const std::size_t n = v.size();
  std::vector<std::size_t> d(dims.begin(), dims.end());
  std::vector<std::size_t> idx(d.size(), 0);
  for (std::size_t flat = 0; flat < n; ++flat) {   // flat = C-order position of idx
    std::size_t at = 0, stride = 1;
    for (std::size_t k = 0; k < d.size(); ++k) { at += idx[k] * stride; stride *= d[k]; }
    out[at] = static_cast<T>(v[flat] + add);
    for (std::size_t k = d.size(); k-- > 0;) { if (++idx[k] < d[k]) break; idx[k] = 0; }
  }
  return a;
}

// an array stored with the first index fastest (Fortran order) is MATLAB's layout already
template <class T>
mxArray* from_column_major(const std::vector<T>& v, const std::vector<std::size_t>& shape) {
  const auto dims = matlab_dims(shape);
  mxArray* a = mxCreateNumericArray(static_cast<mwSize>(dims.size()), dims.data(), class_of(static_cast<const T*>(nullptr)), mxREAL);
  std::memcpy(mxGetData(a), v.data(), v.size() * sizeof(T));
  return a;
}

// [data, header] = io.read_nifti(path)
void ioReadNifti(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 2) throw std::invalid_argument("nxr_compute('io.read_nifti') takes 1 argument");
  fm::nifti::Volume v = fm::nifti::read_file(get_string(prhs[1], "path"));
  const auto& h = v.header;
  auto vec = [](const auto& c) { Eigen::VectorXd e(static_cast<Eigen::Index>(c.size())); for (std::size_t i = 0; i < c.size(); ++i) e[static_cast<Eigen::Index>(i)] = static_cast<double>(c[i]); return e; };
  const char* src = h.affine_source() == fm::nifti::Header::AffineSource::sform ? "sform" : h.affine_source() == fm::nifti::Header::AffineSource::qform ? "qform" : "pixdim";
  const auto a = h.affine();   // 3 x 4 row-major; the 4 x 4 adds the row 0 0 0 1
  Eigen::MatrixXd affine(4, 4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) affine(i, j) = a[static_cast<std::size_t>(4 * i + j)];
  affine.row(3) << 0, 0, 0, 1;
  mxArray* extensions = mxCreateCellMatrix(static_cast<mwSize>(h.extensions.size()), 1);
  for (std::size_t i = 0; i < h.extensions.size(); ++i) {
    mxArray* bytes = mxCreateNumericMatrix(static_cast<mwSize>(h.extensions[i].bytes.size()), 1, mxUINT8_CLASS, mxREAL);
    if (!h.extensions[i].bytes.empty()) std::memcpy(mxGetData(bytes), h.extensions[i].bytes.data(), h.extensions[i].bytes.size());
    mxSetCell(extensions, static_cast<mwIndex>(i), make_struct({{"code", give_scalar(h.extensions[i].code)}, {"bytes", bytes}}));
  }
  mxArray* header = make_struct({{"version", give_scalar(h.version)},
                                 {"little_endian", give_bool(h.little_endian)},
                                 {"datatype", give_scalar(h.datatype)},
                                 {"bitpix", give_scalar(h.bitpix)},
                                 {"shape", give_vector(vec(h.shape), false, true)},
                                 {"pixdim", give_vector(vec(h.pixdim), false, true)},
                                 {"scl_slope", give_scalar(h.scl_slope)},
                                 {"scl_inter", give_scalar(h.scl_inter)},
                                 {"cal_min", give_scalar(h.cal_min)},
                                 {"cal_max", give_scalar(h.cal_max)},
                                 {"toffset", give_scalar(h.toffset)},
                                 {"slice_duration", give_scalar(h.slice_duration)},
                                 {"dim_info", give_scalar(h.dim_info)},
                                 {"xyzt_units", give_scalar(h.xyzt_units)},
                                 {"intent_code", give_scalar(h.intent_code)},
                                 {"intent_name", give_string(h.intent_name)},
                                 {"descrip", give_string(h.descrip)},
                                 {"aux_file", give_string(h.aux_file)},
                                 {"qform_code", give_scalar(h.qform_code)},
                                 {"sform_code", give_scalar(h.sform_code)},
                                 {"spatial_unit", give_string(h.spatial_unit())},
                                 {"time_unit", give_string(h.time_unit())},
                                 {"unit_to_metres", give_scalar(h.unit_to_metres())},
                                 {"affine_source", give_string(src)},
                                 {"affine", give(affine, false)},
                                 {"extensions", extensions}});
  std::vector<std::size_t> shape(h.shape.begin(), h.shape.end());
  // the file stores x fastest: MATLAB's own order, no transpose
  plhs[0] = v.precision == fm::Precision::float32 ? from_column_major<float>(v.f32, shape) : from_column_major<double>(v.f64, shape);
  plhs[1] = header;
}

// image = io.read_gifti(path)
mxArray* gifti_values(fm::gifti::DataArray& a) {
  std::vector<std::size_t> shape(a.dims.begin(), a.dims.end());
  if (a.datatype == "NIFTI_TYPE_INT32") return from_row_major<std::int32_t>(a.i32, shape, a.intent == "NIFTI_INTENT_TRIANGLE" ? 1 : 0);   // triangles: 1-based vertex indices
  if (a.datatype == "NIFTI_TYPE_UINT8") return from_row_major<std::uint8_t>(a.u8, shape);
  return from_row_major<float>(a.f32, shape);
}

void ioReadGifti(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 2) throw std::invalid_argument("nxr_compute('io.read_gifti') takes 1 argument");
  fm::gifti::Image img = fm::gifti::read_file(get_string(prhs[1], "path"));
  mxArray* labels = struct_array({"key", "name", "rgba"}, img.labels.size());
  for (std::size_t i = 0; i < img.labels.size(); ++i) {
    const auto& l = img.labels[i];
    const mwIndex k = static_cast<mwIndex>(i);
    mxSetField(labels, k, "key", give_scalar(l.key));
    mxSetField(labels, k, "name", give_string(l.name));
    mxArray* rgba = mxCreateDoubleMatrix(l.has_colour ? 1 : 0, l.has_colour ? 4 : 0, mxREAL);
    if (l.has_colour)
      for (int j = 0; j < 4; ++j) mxGetPr(rgba)[j] = l.rgba[static_cast<std::size_t>(j)];
    mxSetField(labels, k, "rgba", rgba);
  }
  mxArray* arrays = struct_array({"intent", "datatype", "metadata", "transforms", "data"}, img.arrays.size());
  for (std::size_t i = 0; i < img.arrays.size(); ++i) {
    auto& a = img.arrays[i];
    const mwIndex k = static_cast<mwIndex>(i);
    mxSetField(arrays, k, "intent", give_string(a.intent));
    mxSetField(arrays, k, "datatype", give_string(a.datatype));
    mxSetField(arrays, k, "metadata", map_struct(std::map<std::string, std::string>(a.metadata.begin(), a.metadata.end())));
    mxArray* transforms = struct_array({"dataspace", "xformspace", "matrix"}, a.transforms.size());
    for (std::size_t t = 0; t < a.transforms.size(); ++t) {
      const auto& tr = a.transforms[t];
      Eigen::MatrixXd m(4, 4);
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) m(r, c) = tr.matrix[static_cast<std::size_t>(4 * r + c)];   // row-major in the file's object
      mxSetField(transforms, static_cast<mwIndex>(t), "dataspace", give_string(tr.dataspace));
      mxSetField(transforms, static_cast<mwIndex>(t), "xformspace", give_string(tr.xformspace));
      mxSetField(transforms, static_cast<mwIndex>(t), "matrix", give(m, false));
    }
    mxSetField(arrays, k, "transforms", transforms);
    mxSetField(arrays, k, "data", gifti_values(a));
  }
  plhs[0] = make_struct({{"version", give_string(img.version)},
                         {"metadata", map_struct(std::map<std::string, std::string>(img.metadata.begin(), img.metadata.end()))},
                         {"labels", labels},
                         {"arrays", arrays}});
}

// [names, columns] = io.read_tsv(path): the column names as the file has them, and one cell array of text per column
void ioReadTsv(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 2) throw std::invalid_argument("nxr_compute('io.read_tsv') takes 1 argument");
  fm::tsv::Table t = fm::tsv::read_file(get_string(prhs[1], "path"));
  mxArray* columns = mxCreateCellMatrix(static_cast<mwSize>(t.columns.size()), 1);
  for (std::size_t c = 0; c < t.columns.size(); ++c) {
    mxArray* col = mxCreateCellMatrix(static_cast<mwSize>(t.rows.size()), 1);
    for (std::size_t r = 0; r < t.rows.size(); ++r) mxSetCell(col, static_cast<mwIndex>(r), give_string(t.rows[r][c]));
    mxSetCell(columns, static_cast<mwIndex>(c), col);
  }
  plhs[0] = give_cellstr(t.columns);
  plhs[1] = columns;
}

}  // namespace

bool dispatchIo(const std::string& cmd, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (cmd == "io.open") ioOpen(nlhs, plhs, nrhs, prhs);
  else if (cmd == "io.close") ioClose(nlhs, plhs, nrhs, prhs);
  else if (cmd == "io.header") ioHeader(nlhs, plhs, nrhs, prhs);
  else if (cmd == "io.read") ioRead(nlhs, plhs, nrhs, prhs);
  else if (cmd == "io.read_nifti") ioReadNifti(nlhs, plhs, nrhs, prhs);
  else if (cmd == "io.read_gifti") ioReadGifti(nlhs, plhs, nrhs, prhs);
  else if (cmd == "io.read_tsv") ioReadTsv(nlhs, plhs, nrhs, prhs);
  else return false;
  return true;
}

}  // namespace nxr::mex
