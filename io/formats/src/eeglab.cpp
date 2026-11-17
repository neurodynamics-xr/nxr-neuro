#include "nxr/formats/eeglab.h"

#include <matio.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/endian.h"

namespace nxr::formats::eeglab {

namespace {

struct MatDeleter { void operator()(mat_t* m) const { if (m) Mat_Close(m); } };
struct VarDeleter { void operator()(matvar_t* v) const { if (v) Mat_VarFree(v); } };
using MatPtr = std::unique_ptr<mat_t, MatDeleter>;
using VarPtr = std::unique_ptr<matvar_t, VarDeleter>;

std::size_t numel(const matvar_t* v) {
  std::size_t n = 1;
  for (int i = 0; i < v->rank; ++i) n *= v->dims[i];
  return v->rank == 0 ? 0 : n;
}

// A (small) variable's data, read when it is not yet. Only ever called on scalars, strings and the small struct arrays (chanlocs, event): never on `data`.
void load(mat_t* m, matvar_t* v) {
  if (v->data != nullptr || numel(v) == 0) return;
  if (Mat_VarReadDataAll(m, v) != 0) throw format_error("EEGLAB: cannot read a field of the .set file");
}

// all of a leaf variable's data, as doubles
std::vector<double> numbers(mat_t* m, matvar_t* v) {
  const std::size_t n = numel(v);
  if (n == 0) return {};
  if (v->isComplex) throw format_error("EEGLAB: a complex field");
  load(m, v);
  std::vector<double> out(n);
  const unsigned char* base = static_cast<const unsigned char*>(v->data);
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned char* p = base + i * v->data_size;
    switch (v->class_type) {
      case MAT_C_DOUBLE: { double d; std::memcpy(&d, p, 8); out[i] = d; break; }
      case MAT_C_SINGLE: { float f; std::memcpy(&f, p, 4); out[i] = f; break; }
      case MAT_C_INT8: out[i] = static_cast<std::int8_t>(*p); break;
      case MAT_C_UINT8: out[i] = *p; break;
      case MAT_C_INT16: { std::int16_t x; std::memcpy(&x, p, 2); out[i] = x; break; }
      case MAT_C_UINT16: { std::uint16_t x; std::memcpy(&x, p, 2); out[i] = x; break; }
      case MAT_C_INT32: { std::int32_t x; std::memcpy(&x, p, 4); out[i] = x; break; }
      case MAT_C_UINT32: { std::uint32_t x; std::memcpy(&x, p, 4); out[i] = x; break; }
      case MAT_C_INT64: { std::int64_t x; std::memcpy(&x, p, 8); out[i] = static_cast<double>(x); break; }
      case MAT_C_UINT64: { std::uint64_t x; std::memcpy(&x, p, 8); out[i] = static_cast<double>(x); break; }
      default: throw format_error("EEGLAB: a numeric field of an unexpected class");
    }
  }
  return out;
}

std::string text(mat_t* m, matvar_t* v) {
  if (!v || v->class_type != MAT_C_CHAR) return "";
  const std::size_t n = numel(v);
  if (n == 0) return "";
  load(m, v);
  const unsigned char* raw = static_cast<const unsigned char*>(v->data);
  std::string s;
  if (v->data_size == 2) {   // UTF-16 code units
    for (std::size_t i = 0; i < n; ++i) {
      unsigned int c = load_u16(raw + 2 * i, true);
      if (c == 0) break;
      if (c < 0x80) s += static_cast<char>(c);
      else if (c < 0x800) { s += static_cast<char>(0xC0 | (c >> 6)); s += static_cast<char>(0x80 | (c & 0x3F)); }
      else { s += static_cast<char>(0xE0 | (c >> 12)); s += static_cast<char>(0x80 | ((c >> 6) & 0x3F)); s += static_cast<char>(0x80 | (c & 0x3F)); }
    }
  } else {   // bytes (UTF-8)
    for (std::size_t i = 0; i < v->nbytes && raw[i]; ++i) s += static_cast<char>(raw[i]);
  }
  return s;
}

std::string num_text(double v) {
  char b[40];
  std::snprintf(b, sizeof b, "%.10g", v);
  return b;
}

// a field of EEG as text, whatever it holds: a string as is, a scalar as a number, an array as its size
std::string describe(mat_t* m, matvar_t* v) {
  if (!v) return "";
  if (v->class_type == MAT_C_CHAR) return text(m, v);
  if (v->class_type == MAT_C_STRUCT || v->class_type == MAT_C_CELL) {
    std::string d;
    for (int i = 0; i < v->rank; ++i) d += (i ? "x" : "") + std::to_string(v->dims[i]);
    return d;
  }
  const std::size_t n = numel(v);
  if (n == 1) return num_text(numbers(m, v)[0]);
  std::string d;
  for (int i = 0; i < v->rank; ++i) d += (i ? "x" : "") + std::to_string(v->dims[i]);
  return d;
}

class Reader final : public RecordingReader {
 public:
  using RecordingReader::read;

  explicit Reader(const std::string& path) : path_(path) {
    {
      std::FILE* f = std::fopen(path.c_str(), "rb");
      if (!f) throw io_error("cannot open " + path + " for reading");
      std::fclose(f);
    }
    mat_.reset(Mat_Open(path.c_str(), MAT_ACC_RDONLY));
    if (!mat_) throw format_error(path + ": not a MATLAB file (an EEGLAB .set is a MATLAB .mat file)");
    parse();
  }

  const RecordingHeader& header() const override { return h_; }

  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const override {
    check_range(s0, n, c0, c1);
    if (h_.precision != Precision::float32) throw refused("EEGLAB: this dataset's data are stored in double precision: read them with the double overload");
    if (n == 0 || c0 == c1) return;
    if (fdt_) {
      const std::size_t nch = h_.channels.size();
      std::vector<unsigned char> buf(static_cast<std::size_t>(n) * nch * 4);
      fdt_->read(s0 * nch * 4, buf.size(), buf.data());
      for (std::size_t c = c0; c < c1; ++c)
        for (std::uint64_t i = 0; i < n; ++i) out[(c - c0) * n + i] = static_cast<float>(load_f32(buf.data() + (i * nch + c) * 4, true) * kMicro);
    } else {
      std::vector<float> col(static_cast<std::size_t>((c1 - c0) * n));
      slab(s0, n, c0, c1, col.data());
      for (std::size_t c = 0; c < c1 - c0; ++c)
        for (std::uint64_t i = 0; i < n; ++i) out[c * n + i] = static_cast<float>(static_cast<double>(col[c + (c1 - c0) * i]) * kMicro);
    }
  }

  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, double* out) const override {
    check_range(s0, n, c0, c1);
    if (h_.precision == Precision::float32) { RecordingReader::read(s0, n, c0, c1, out); return; }
    if (n == 0 || c0 == c1) return;
    std::vector<double> col(static_cast<std::size_t>((c1 - c0) * n));
    slab(s0, n, c0, c1, col.data());
    for (std::size_t c = 0; c < c1 - c0; ++c)
      for (std::uint64_t i = 0; i < n; ++i) out[c * n + i] = col[c + (c1 - c0) * i] * kMicro;
  }

 private:
  static constexpr double kMicro = 1e-6;

  // The samples [s0, s0 + n) of all channels, as the variable holds them (channel fastest) into `col`, as one linear read. (matio's 2-D slab read
  // ignores the start column when it reads whole columns, so the linear read is the one that is right.)
  template <class T>
  void slab(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, T* col) const {
    const std::size_t nch = h_.channels.size();
    if ((s0 + n) * nch > static_cast<std::uint64_t>(INT_MAX))
      throw refused("EEGLAB: block reads of data inside the .set reach past 2^31 values (a MATLAB variable is indexed by 32-bit integers here); keep the data in a .fdt file");
    std::vector<T> all(static_cast<std::size_t>(n) * nch);
    if (Mat_VarReadDataLinear(mat_.get(), data_var_, all.data(), static_cast<int>(s0 * nch), 1, static_cast<int>(n * nch)) != 0)
      throw io_error(path_ + ": reading a block of the data variable failed");
    const std::size_t w = c1 - c0;
    for (std::uint64_t i = 0; i < n; ++i)
      for (std::size_t c = 0; c < w; ++c) col[c + w * i] = all[c0 + c + nch * i];
  }

  // the field `name` of EEG (element 0), or the top-level variable of that name; null when absent. The variable is owned by this reader.
  matvar_t* field(const std::string& name) {
    if (eeg_) {
      if (eeg_->class_type != MAT_C_STRUCT) return nullptr;
      return Mat_VarGetStructFieldByName(eeg_.get(), name.c_str(), 0);
    }
    auto it = top_.find(name);
    if (it != top_.end()) return it->second.get();
    matvar_t* v = Mat_VarReadInfo(mat_.get(), name.c_str());
    if (v) top_[name] = VarPtr(v);
    return v;
  }

  double scalar(const std::string& name, double dflt) {
    matvar_t* v = field(name);
    if (!v || v->class_type == MAT_C_CHAR || numel(v) != 1) return dflt;
    return numbers(mat_.get(), v)[0];
  }

  void parse() {
    matvar_t* e = Mat_VarReadInfo(mat_.get(), "EEG");
    if (e) {
      eeg_.reset(e);
      if (e->class_type != MAT_C_STRUCT) throw format_error(path_ + ": the variable EEG is not a struct");
    } else {
      if (field("nbchan") == nullptr) throw format_error(path_ + ": not an EEGLAB dataset (no variable EEG and no nbchan)");
    }
    const double trials = scalar("trials", 1);
    const double nbchan = scalar("nbchan", -1), pnts = scalar("pnts", -1), srate = scalar("srate", -1);
    if (nbchan < 1 || pnts < 0 || !(srate > 0)) throw format_error(path_ + ": nbchan, pnts or srate missing or out of range");
    if (trials > 1) throw refused("EEGLAB: this dataset is epoched (" + num_text(trials) + " trials); only continuous datasets are read. Export the continuous data, or read the epochs from their source");
    const std::size_t nch = static_cast<std::size_t>(nbchan);

    h_.format = "eeglab";
    h_.sampling_rate = srate;
    h_.n_samples = static_cast<std::uint64_t>(pnts);
    h_.position_unit = "";
    h_.info["unit"] = "uV (EEGLAB stores no unit; microvolts by convention)";

    // channels
    matvar_t* cl = field("chanlocs");
    const bool have_locs = cl && cl->class_type == MAT_C_STRUCT && numel(cl) == nch;
    if (have_locs) load(mat_.get(), cl);
    for (std::size_t i = 0; i < nch; ++i) {
      Channel c;
      c.name = "Ch" + std::to_string(i + 1);
      c.file_unit = "uV";
      c.unit = "V";
      c.scale_to_si = kMicro;
      c.sampling_rate = srate;
      if (have_locs) {
        auto get = [&](const char* f) { return Mat_VarGetStructFieldByName(cl, f, static_cast<int>(i)); };
        matvar_t* lab = get("labels");
        if (lab && lab->class_type == MAT_C_CHAR) { std::string s = text(mat_.get(), lab); if (!s.empty()) c.name = s; }
        matvar_t* ty = get("type");
        if (ty && ty->class_type == MAT_C_CHAR) { c.kind = text(mat_.get(), ty); for (char& ch : c.kind) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }
        matvar_t* rf = get("ref");
        if (rf && rf->class_type == MAT_C_CHAR) c.reference = text(mat_.get(), rf);
        matvar_t *x = get("X"), *y = get("Y"), *z = get("Z");
        if (x && y && z && numel(x) == 1 && numel(y) == 1 && numel(z) == 1 && x->class_type != MAT_C_CHAR) {
          const double X = numbers(mat_.get(), x)[0], Y = numbers(mat_.get(), y)[0], Z = numbers(mat_.get(), z)[0];
          if (std::isfinite(X) && std::isfinite(Y) && std::isfinite(Z)) c.location = std::array<double, 3>{X, Y, Z};
        }
      }
      h_.channels.push_back(c);
    }

    // events
    matvar_t* ev = field("event");
    if (ev && ev->class_type == MAT_C_STRUCT) {
      const std::size_t ne = numel(ev);
      load(mat_.get(), ev);
      for (std::size_t i = 0; i < ne; ++i) {
        matvar_t* ty = Mat_VarGetStructFieldByName(ev, "type", static_cast<int>(i));
        matvar_t* la = Mat_VarGetStructFieldByName(ev, "latency", static_cast<int>(i));
        matvar_t* du = Mat_VarGetStructFieldByName(ev, "duration", static_cast<int>(i));
        if (!la || la->class_type == MAT_C_CHAR || numel(la) != 1) continue;
        Annotation a;
        a.onset = (numbers(mat_.get(), la)[0] - 1.0) / srate;
        if (du && du->class_type != MAT_C_CHAR && numel(du) == 1) { const double d = numbers(mat_.get(), du)[0]; a.duration = std::isfinite(d) ? d / srate : 0.0; }
        if (ty) a.label = ty->class_type == MAT_C_CHAR ? text(mat_.get(), ty) : (numel(ty) == 1 ? num_text(numbers(mat_.get(), ty)[0]) : "");
        h_.annotations.push_back(a);
      }
    }

    // the other header fields, as text
    for (const char* f : {"setname", "filename", "filepath", "subject", "group", "condition", "session", "comments", "ref", "xmin", "xmax", "trials", "nbchan", "pnts", "srate"}) {
      matvar_t* v = field(f);
      if (v) h_.info[f] = describe(mat_.get(), v);
    }
    for (const char* f : {"icaweights", "icasphere", "icawinv", "icaact", "icachansind", "urchanlocs", "chaninfo", "urevent", "epoch", "times"}) {
      matvar_t* v = field(f);
      if (v && numel(v) > 0) h_.info[f] = describe(mat_.get(), v);   // ICA operators and the like: recorded by size, not applied
    }

    // the data
    matvar_t* d = field("data");
    if (!d) throw format_error(path_ + ": the dataset has no data field");
    if (d->class_type == MAT_C_CHAR) {
      std::string fname = text(mat_.get(), d);
      const std::size_t slash = path_.find_last_of("/\\");
      const std::string dir = slash == std::string::npos ? "" : path_.substr(0, slash + 1);
      // the name inside the file may carry a folder of the machine that wrote it: use its last part, beside the .set
      const std::size_t s2 = fname.find_last_of("/\\");
      if (s2 != std::string::npos) fname = fname.substr(s2 + 1);
      fdt_ = std::make_unique<FileSource>(dir + fname);
      const std::uint64_t want = static_cast<std::uint64_t>(nch) * h_.n_samples * 4;
      if (fdt_->size() != want)
        throw format_error("EEGLAB: " + fname + " holds " + std::to_string(fdt_->size()) + " bytes but " + std::to_string(nch) + " channels x " + std::to_string(h_.n_samples) + " samples of float32 are " + std::to_string(want) + ": the file is truncated or the header is wrong");
      h_.precision = Precision::float32;
      h_.info["data_file"] = fname;
    } else {
      if (d->rank != 2) throw refused("EEGLAB: the data have " + std::to_string(d->rank) + " dimensions (epoched or multi-trial); only continuous channels x samples are read");
      if (d->dims[0] != nch || d->dims[1] != h_.n_samples) throw format_error("EEGLAB: the data variable is " + std::to_string(d->dims[0]) + " x " + std::to_string(d->dims[1]) + ", the header says " + std::to_string(nch) + " x " + std::to_string(h_.n_samples));
      if (d->class_type != MAT_C_SINGLE && d->class_type != MAT_C_DOUBLE) throw refused("EEGLAB: the data are stored as an integer class; only single and double are read");
      if (d->compression == MAT_COMPRESSION_ZLIB)
        throw refused("EEGLAB: the data are inside a zlib-compressed MATLAB variable (-v7), which cannot be read in blocks (reading the end inflates everything before it). Re-save the dataset in EEGLAB with its data in a separate .fdt file");
      h_.precision = d->class_type == MAT_C_DOUBLE ? Precision::float64 : Precision::float32;
      data_var_ = d;
      h_.info["data_file"] = "";
    }
  }

  std::string path_;
  MatPtr mat_;
  VarPtr eeg_;
  std::map<std::string, VarPtr> top_;
  matvar_t* data_var_ = nullptr;
  std::unique_ptr<FileSource> fdt_;
  RecordingHeader h_;
};

}  // namespace

std::unique_ptr<RecordingReader> open_file(const std::string& set_path) { return std::make_unique<Reader>(set_path); }

}  // namespace nxr::formats::eeglab
