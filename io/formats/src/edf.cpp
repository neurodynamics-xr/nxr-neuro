#include "nxr/formats/edf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nxr/formats/endian.h"
#include "nxr/formats/units.h"

namespace nxr::formats::edf {

namespace {

std::string rtrim(const std::string& s) {
  std::size_t e = s.size();
  while (e > 0 && (s[e - 1] == ' ' || s[e - 1] == '\0')) --e;
  return s.substr(0, e);
}
std::string trim(const std::string& s) {
  const std::string r = rtrim(s);
  std::size_t b = 0;
  while (b < r.size() && r[b] == ' ') ++b;
  return r.substr(b);
}

double parse_num(const std::string& field, const std::string& what) {
  const std::string t = trim(field);
  char* end = nullptr;
  const double v = std::strtod(t.c_str(), &end);
  if (t.empty() || *end != '\0') throw format_error("EDF: the header field " + what + " (\"" + t + "\") is not a number");
  return v;
}

std::string infer_kind(const std::string& label) {
  static const char* kinds[] = {"EEG", "ECG", "EKG", "EOG", "EMG", "MEG", "ERG", "EGG"};
  const std::string first = label.substr(0, label.find(' '));
  for (const char* k : kinds) {
    if (first == k) {
      std::string l = k;
      for (char& c : l) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      return l == "ekg" ? "ecg" : l;
    }
  }
  if (label == "Status") return "stim";
  return "";
}

std::string iso_start(const std::string& date, const std::string& time, const std::string& recording) {
  auto two = [](const std::string& s, std::size_t i, int& out) {
    if (i + 2 > s.size() || !std::isdigit(static_cast<unsigned char>(s[i])) || !std::isdigit(static_cast<unsigned char>(s[i + 1]))) return false;
    out = (s[i] - '0') * 10 + (s[i + 1] - '0');
    return true;
  };
  int d, mo, y, h, mi, s;
  if (date.size() < 8 || time.size() < 8 || !two(date, 0, d) || !two(date, 3, mo) || !two(date, 6, y) || !two(time, 0, h) || !two(time, 3, mi) || !two(time, 6, s)) return "";
  int year = y >= 85 ? 1900 + y : 2000 + y;
  // EDF+: the recording field may carry "Startdate dd-MMM-yyyy": its four-digit year wins
  if (recording.compare(0, 10, "Startdate ") == 0) {
    int dd;
    if (recording.size() >= 21 && two(recording, 10, dd) && dd == d) {
      const int yy = std::atoi(recording.substr(17, 4).c_str());
      if (yy >= 1900 && yy <= 2999 && yy % 100 == y) year = yy;
    }
  }
  if (d < 1 || d > 31 || mo < 1 || mo > 12 || h > 23 || mi > 59 || s > 59) return "";
  char b[32];
  std::snprintf(b, sizeof b, "%04d-%02d-%02dT%02d:%02d:%02d", year, mo, d, h, mi, s);
  return b;
}

struct Signal {
  std::string label, transducer, unit, prefilter;
  double pmin = 0, pmax = 0, dmin = 0, dmax = 0;
  std::uint32_t nsamp = 0;
  std::uint64_t byte_offset = 0;   // within a record
  bool annotations = false;
};

struct Tal {
  double onset = 0;
  bool has_duration = false;
  double duration = 0;
  std::vector<std::string> labels;
};

// The TALs of one annotation signal's bytes for a record.
std::vector<Tal> parse_tals(const unsigned char* p, std::size_t n) {
  std::vector<Tal> out;
  std::size_t i = 0;
  while (i < n) {
    if (p[i] == 0) { ++i; continue; }
    std::size_t e = i;
    while (e < n && p[e] != 0) ++e;
    const std::string s(reinterpret_cast<const char*>(p + i), e - i);
    i = e + 1;
    const std::size_t first = s.find('\x14');
    if (first == std::string::npos) continue;   // not a TAL
    Tal t;
    std::string time = s.substr(0, first);
    const std::size_t d = time.find('\x15');
    if (d != std::string::npos) { t.has_duration = true; t.duration = std::atof(time.substr(d + 1).c_str()); time.resize(d); }
    if (time.empty() || (time[0] != '+' && time[0] != '-')) continue;
    t.onset = std::atof(time.c_str());
    std::size_t k = first + 1;
    while (k < s.size()) {
      const std::size_t f = s.find('\x14', k);
      const std::size_t end = f == std::string::npos ? s.size() : f;
      if (end > k) t.labels.emplace_back(s.substr(k, end - k));
      k = end + 1;
    }
    out.push_back(std::move(t));
  }
  return out;
}

class Reader final : public RecordingReader {
 public:
  Reader(std::shared_ptr<const ByteSource> src, const OpenOptions& opt) : src_(std::move(src)) { parse(opt); }
  const RecordingHeader& header() const override { return h_; }

  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const override {
    check_range(s0, n, c0, c1);
    const std::size_t nc = c1 - c0;
    std::fill(out, out + nc * n, 0.0f);   // gaps stay zero
    if (n == 0 || nc == 0) return;
    // the first record that ends after s0
    std::size_t k = std::upper_bound(rec_start_.begin(), rec_start_.end(), s0) - rec_start_.begin();
    if (k > 0) --k;
    std::vector<unsigned char> buf;
    for (; k < rec_start_.size() && rec_start_[k] < s0 + n; ++k) {
      const std::uint64_t rs = rec_start_[k], re = rs + group_nsamp_;
      const std::uint64_t a = std::max(rs, s0), b = std::min(re, s0 + n);
      if (a >= b) continue;
      const std::uint64_t la = a - rs, lb = b - rs;   // samples within the record
      // the byte range of the record covering the wanted channels
      std::uint64_t lo = UINT64_MAX, hi = 0;
      for (std::size_t c = c0; c < c1; ++c) {
        const Signal& sg = sig_[group_[c]];
        lo = std::min<std::uint64_t>(lo, sg.byte_offset + la * bps_);
        hi = std::max<std::uint64_t>(hi, sg.byte_offset + lb * bps_);
      }
      buf.resize(static_cast<std::size_t>(hi - lo));
      src_->read(header_bytes_ + k * record_bytes_ + lo, buf.size(), buf.data());
      for (std::size_t c = c0; c < c1; ++c) {
        const Signal& sg = sig_[group_[c]];
        const unsigned char* p = buf.data() + (sg.byte_offset + la * bps_ - lo);
        float* o = out + (c - c0) * n + (a - s0);
        const double mul = mul_[c], add = add_[c];
        for (std::uint64_t i = 0; i < lb - la; ++i) {
          const double v = bps_ == 2 ? load_i16(p + i * 2, true) : load_i24(p + i * 3, true);
          o[i] = static_cast<float>(v * mul + add);
        }
      }
    }
  }

 private:
  void parse(const OpenOptions& opt) {
    if (src_->size() < 256) throw format_error("EDF: fewer than 256 bytes, not an EDF/BDF header");
    const std::string fixed = src_->string(0, 256);
    const std::string version = fixed.substr(0, 8);
    bool bdf;
    if (version == "0       ") bdf = false;
    else if (static_cast<unsigned char>(version[0]) == 0xFF && version.compare(1, 7, "BIOSEMI") == 0) bdf = true;
    else throw format_error("EDF: the version field is not \"0\" (EDF) or 0xFF BIOSEMI (BDF): not an EDF/BDF file");
    bps_ = bdf ? 3 : 2;
    const std::string patient = rtrim(fixed.substr(8, 80)), recording = rtrim(fixed.substr(88, 80));
    const std::string startdate = fixed.substr(168, 8), starttime = fixed.substr(176, 8);
    header_bytes_ = static_cast<std::uint64_t>(parse_num(fixed.substr(184, 8), "header bytes"));
    const std::string reserved = rtrim(fixed.substr(192, 44));
    double nrec = parse_num(fixed.substr(236, 8), "number of data records");
    const double rec_dur = parse_num(fixed.substr(244, 8), "duration of a data record");
    const int ns = static_cast<int>(parse_num(fixed.substr(252, 4), "number of signals"));
    if (ns < 1 || ns > 9999) throw format_error("EDF: " + std::to_string(ns) + " signals");
    if (header_bytes_ != 256u * (static_cast<std::uint64_t>(ns) + 1)) throw format_error("EDF: the header says " + std::to_string(header_bytes_) + " bytes for " + std::to_string(ns) + " signals; expected " + std::to_string(256 * (ns + 1)));
    if (!(rec_dur > 0)) throw format_error("EDF: a data record duration of " + std::to_string(rec_dur) + " s");
    if (src_->size() < header_bytes_) throw format_error("EDF: the file is shorter than its header");

    const std::string sh = src_->string(256, static_cast<std::size_t>(header_bytes_ - 256));
    std::size_t pos = 0;
    auto field = [&](std::size_t width, int i) { (void)i; return sh.substr(pos + static_cast<std::size_t>(i) * width, width); };
    sig_.resize(ns);
    auto column = [&](std::size_t width, auto&& assign) {
      for (int i = 0; i < ns; ++i) assign(sig_[i], field(width, i));
      pos += width * static_cast<std::size_t>(ns);
    };
    column(16, [](Signal& s, const std::string& f) { s.label = rtrim(f); });
    column(80, [](Signal& s, const std::string& f) { s.transducer = rtrim(f); });
    column(8, [](Signal& s, const std::string& f) { s.unit = trim(f); });
    column(8, [](Signal& s, const std::string& f) { s.pmin = parse_num(f, "physical minimum"); });
    column(8, [](Signal& s, const std::string& f) { s.pmax = parse_num(f, "physical maximum"); });
    column(8, [](Signal& s, const std::string& f) { s.dmin = parse_num(f, "digital minimum"); });
    column(8, [](Signal& s, const std::string& f) { s.dmax = parse_num(f, "digital maximum"); });
    column(80, [](Signal& s, const std::string& f) { s.prefilter = rtrim(f); });
    column(8, [](Signal& s, const std::string& f) { const double v = parse_num(f, "samples per record"); s.nsamp = v < 0 ? 0 : static_cast<std::uint32_t>(v); });
    std::vector<std::string> reserved_sig;
    for (int i = 0; i < ns; ++i) reserved_sig.push_back(rtrim(field(32, i)));

    record_bytes_ = 0;
    for (Signal& s : sig_) {
      s.byte_offset = record_bytes_;
      record_bytes_ += static_cast<std::uint64_t>(s.nsamp) * bps_;
      s.annotations = s.label == "EDF Annotations" || s.label == "BDF Annotations";
    }
    if (record_bytes_ == 0) throw format_error("EDF: a data record of no bytes");
    const std::uint64_t avail = src_->size() - header_bytes_;
    if (nrec < 0) nrec = static_cast<double>(avail / record_bytes_);   // unknown count (-1): as many whole records as the file holds
    n_records_ = static_cast<std::uint64_t>(nrec);
    if (n_records_ > avail / record_bytes_)
      throw format_error("EDF: the header counts " + std::to_string(n_records_) + " records of " + std::to_string(record_bytes_) + " bytes but the file holds only " + std::to_string(avail / record_bytes_) + ": the file is truncated");

    // which rate to read
    std::vector<double> rates;
    for (const Signal& s : sig_) {
      if (s.annotations) continue;
      if (s.nsamp == 0) throw format_error("EDF: signal \"" + s.label + "\" has no samples per record");
      const double r = s.nsamp / rec_dur;
      bool seen = false;
      for (double q : rates) seen = seen || std::fabs(q - r) <= 1e-9 * r;
      if (!seen) rates.push_back(r);
    }
    if (rates.empty()) throw format_error("EDF: the file has no signals besides annotations");
    double rate = 0;
    if (opt.sampling_rate > 0) {
      for (double q : rates) if (std::fabs(q - opt.sampling_rate) <= 1e-6 * q) rate = q;
      if (rate == 0) throw refused("EDF: no signal at " + std::to_string(opt.sampling_rate) + " Hz in this file");
    } else if (rates.size() == 1) {
      rate = rates[0];
    } else {
      std::string list;
      for (double q : rates) list += (list.empty() ? "" : ", ") + trim_num(q);
      throw refused("EDF: the signals have " + std::to_string(rates.size()) + " sampling rates (" + list + " Hz); name the rate to read (OpenOptions::sampling_rate), each rate is its own recording");
    }
    for (std::size_t i = 0; i < sig_.size(); ++i) {
      const Signal& s = sig_[i];
      if (s.annotations) continue;
      if (std::fabs(s.nsamp / rec_dur - rate) > 1e-9 * rate) continue;
      group_.push_back(i);
      group_nsamp_ = s.nsamp;
    }

    // the annotation signals: time-keeping and annotations
    std::vector<std::size_t> ann_sigs;
    for (std::size_t i = 0; i < sig_.size(); ++i) if (sig_[i].annotations) ann_sigs.push_back(i);
    std::vector<double> rec_time(n_records_, 0.0);
    std::vector<std::vector<Tal>> rec_tals(n_records_);
    const bool plus = reserved.compare(0, 4, bdf ? "BDF+" : "EDF+") == 0 || !ann_sigs.empty();
    const bool discontinuous = reserved.size() >= 5 && reserved[4] == 'D';
    bool have_time = false;
    if (!ann_sigs.empty()) {
      have_time = true;
      for (std::uint64_t k = 0; k < n_records_; ++k) {
        for (std::size_t j = 0; j < ann_sigs.size(); ++j) {
          const Signal& s = sig_[ann_sigs[j]];
          const std::vector<unsigned char> b = src_->bytes(header_bytes_ + k * record_bytes_ + s.byte_offset, static_cast<std::size_t>(s.nsamp) * bps_);
          std::vector<Tal> tals = parse_tals(b.data(), b.size());
          if (j == 0) {
            if (tals.empty()) throw format_error("EDF+: record " + std::to_string(k) + " has no time-keeping annotation");
            rec_time[k] = tals[0].onset;
            tals.erase(tals.begin());
          }
          for (Tal& t : tals) rec_tals[k].push_back(std::move(t));
        }
      }
    }
    // placing the records on one axis
    rec_start_.resize(n_records_);
    for (std::uint64_t k = 0; k < n_records_; ++k) {
      const double t = have_time ? rec_time[k] - rec_time[0] : k * rec_dur;
      const double sf = t * rate;
      const double r = std::nearbyint(sf);
      if (std::fabs(sf - r) > 1e-3) throw format_error("EDF+: record " + std::to_string(k) + " starts at " + trim_num(t) + " s, which is not on the sample grid of " + trim_num(rate) + " Hz");
      rec_start_[k] = static_cast<std::uint64_t>(r < 0 ? 0 : r);
      if (r < 0 || (k > 0 && rec_start_[k] < rec_start_[k - 1] + group_nsamp_)) throw format_error("EDF+: data records overlap or run backwards at record " + std::to_string(k));
    }
    (void)discontinuous;

    // header
    h_.format = std::string(bdf ? "bdf" : "edf") + (plus ? "+" : "");
    h_.sampling_rate = rate;
    h_.n_samples = n_records_ ? rec_start_.back() + group_nsamp_ : 0;
    h_.start_time = iso_start(startdate, starttime, recording);
    for (std::uint64_t k = 1; k < n_records_; ++k) {
      const std::uint64_t prev_end = rec_start_[k - 1] + group_nsamp_;
      if (rec_start_[k] > prev_end) h_.gaps.push_back({prev_end, rec_start_[k] - prev_end});
    }
    h_.info["version"] = bdf ? "BIOSEMI" : "0";
    h_.info["patient"] = patient;
    h_.info["recording"] = recording;
    h_.info["startdate"] = rtrim(startdate);
    h_.info["starttime"] = rtrim(starttime);
    h_.info["reserved"] = reserved;
    h_.info["data_records"] = std::to_string(n_records_);
    h_.info["record_duration"] = trim_num(rec_dur);
    h_.info["signals"] = std::to_string(ns);
    h_.info["rates"] = [&] { std::string l; for (double q : rates) l += (l.empty() ? "" : ",") + trim_num(q); return l; }();
    for (std::size_t j = 0; j < group_.size(); ++j) {
      const Signal& s = sig_[group_[j]];
      Channel c;
      c.name = s.label;
      c.kind = infer_kind(s.label);
      c.file_unit = s.unit;
      const SiUnit u = si_unit(s.unit);
      c.unit = u.unit;
      c.scale_to_si = u.scale;
      if (s.dmax == s.dmin) throw format_error("EDF: signal \"" + s.label + "\" has equal digital minimum and maximum");
      c.gain = (s.pmax - s.pmin) / (s.dmax - s.dmin);
      c.offset = s.pmin - s.dmin * c.gain;
      c.sampling_rate = rate;
      c.transducer = s.transducer;
      c.prefilter = s.prefilter;
      h_.channels.push_back(c);
      mul_.push_back(c.gain * c.scale_to_si);
      add_.push_back(c.offset * c.scale_to_si);
    }
    for (std::uint64_t k = 0; k < n_records_; ++k) {
      for (const Tal& t : rec_tals[k]) {
        for (const std::string& lab : t.labels) {
          Annotation a;
          a.onset = t.onset - (have_time ? rec_time[0] : 0.0);
          a.duration = t.has_duration ? t.duration : 0.0;
          a.label = lab;
          h_.annotations.push_back(a);
        }
      }
    }
    (void)reserved_sig;
  }

  static std::string trim_num(double v) {
    char b[40];
    std::snprintf(b, sizeof b, "%.10g", v);
    return b;
  }

  std::shared_ptr<const ByteSource> src_;
  RecordingHeader h_;
  std::vector<Signal> sig_;
  std::vector<std::size_t> group_;   // the data signals read (indices into sig_), one per channel
  std::vector<double> mul_, add_;
  std::vector<std::uint64_t> rec_start_;   // first sample of each record on the continuous axis
  std::uint64_t header_bytes_ = 0, record_bytes_ = 0, n_records_ = 0;
  std::uint32_t group_nsamp_ = 0;
  int bps_ = 2;
};

}  // namespace

std::unique_ptr<RecordingReader> open(std::shared_ptr<const ByteSource> src, const OpenOptions& options) {
  return std::make_unique<Reader>(std::move(src), options);
}

std::unique_ptr<RecordingReader> open_file(const std::string& path, const OpenOptions& options) {
  return open(std::make_shared<FileSource>(path), options);
}

}  // namespace nxr::formats::edf
