// CTF data sets. Ported from MNE-Python's mne/io/ctf/{res4,ctf,info,hc,trans,eeg,markers}.py (BSD-3-Clause, Copyright the MNE-Python contributors;
// see THIRD_PARTY_NOTICES.md).
#include "nxr/formats/ctf.h"
#include "nxr/formats/geometry.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>

#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"

namespace fs = std::filesystem;

namespace nxr::formats::ctf {

namespace {

constexpr std::uint64_t kHeaderSize = 8;       // the ".meg4" header
constexpr std::size_t kFunnyPos = 1844;        // where the run description starts in the .res4
constexpr std::int32_t kFrameDevice = 1001, kFrameHead = 1004;
constexpr std::int32_t kLpa = 1, kRpa = 2, kNas = 3, kSpare = 4;

// ---- a cursor over a big-endian buffer -------------------------------------------------------------------------------------------------------

class Cur {
 public:
  explicit Cur(const std::vector<unsigned char>& b) : b_(b) {}
  std::size_t pos() const { return p_; }
  void seek(std::size_t p) { p_ = p; }
  void align(std::size_t k) { if (p_ % k) p_ = p_ - p_ % k + k; }
  void need(std::size_t n) const { if (p_ + n > b_.size()) throw format_error("CTF: the .res4 file is truncated (needs " + std::to_string(p_ + n) + " bytes, has " + std::to_string(b_.size()) + ")"); }
  std::string str(std::size_t n) {
    need(n);
    std::size_t e = 0;
    while (e < n && b_[p_ + e] != 0) ++e;
    std::string s(reinterpret_cast<const char*>(&b_[p_]), e);
    p_ += n;
    return s;
  }
  std::int16_t i16() { need(2); const auto v = load_i16(&b_[p_], false); p_ += 2; return v; }
  std::int32_t i32() { need(4); const auto v = load_i32(&b_[p_], false); p_ += 4; return v; }
  double f64() { need(8); const auto v = load_f64(&b_[p_], false); p_ += 8; return v; }
  unsigned char u8() { need(1); return b_[p_++]; }
  std::vector<unsigned char> raw(std::size_t n) { need(n); std::vector<unsigned char> v(b_.begin() + static_cast<std::ptrdiff_t>(p_), b_.begin() + static_cast<std::ptrdiff_t>(p_ + n)); p_ += n; return v; }

 private:
  const std::vector<unsigned char>& b_;
  std::size_t p_ = 0;
};

Coil read_coil(Cur& c) {
  Coil k;
  for (double& x : k.pos) x = c.f64() / 100.0;   // cm -> m
  c.f64();
  for (double& x : k.norm) x = c.f64();
  c.f64();
  k.turns = c.i16();
  c.i32();
  c.i16();
  k.area = c.f64() * 1e-4;                       // cm^2 -> m^2
  return k;
}

Res4 parse_res4(const std::vector<unsigned char>& bytes) {
  Cur c(bytes);
  Res4 r;
  r.head = c.str(8);
  r.appname = c.str(256);
  r.origin = c.str(256);
  r.desc = c.str(256);
  r.nave = c.i16();
  r.data_time = c.str(255);
  r.data_date = c.str(255);
  if (r.data_time.find('/') != std::string::npos && r.data_date.find(':') != std::string::npos) std::swap(r.data_time, r.data_date);   // entered by hand the wrong way round
  r.nsamp = c.i32();
  r.nchan = c.i16();
  c.align(8);
  r.sfreq = c.f64();
  r.epoch_time = c.f64();
  r.no_trials = c.i16();
  c.align(4);
  r.pre_trig_pts = c.i32();
  r.no_trials_done = c.i16();
  r.no_trials_bst_message_windowlay = c.i16();
  c.align(4);
  r.save_trials = c.i32();
  r.primary_trigger = c.u8();
  for (auto& s : r.secondary_trigger) s = c.u8();
  r.trigger_polarity_mask = c.u8();
  r.trigger_mode = c.i16();
  c.align(4);
  r.accept_reject = c.i32();
  r.run_time_bst_message_windowlay = c.i16();
  c.align(4);
  r.zero_head = c.i32();
  c.align(4);
  r.artifact_mode = c.i32();
  c.i32();   // padding
  r.nf_run_name = c.str(32);
  r.nf_run_title = c.str(256);
  r.nf_instruments = c.str(32);
  r.nf_collect_descriptor = c.str(32);
  r.nf_subject_id = c.str(32);
  r.nf_operator = c.str(32);
  r.nf_sensor_file_name = c.raw(60);
  c.align(4);
  r.rdlen = c.i32();
  c.seek(kFunnyPos);
  if (r.rdlen > 0) r.run_desc = c.str(static_cast<std::size_t>(r.rdlen));
  if (r.nchan <= 0) throw format_error("CTF: the .res4 file declares " + std::to_string(r.nchan) + " channels");
  const std::int16_t nfilt = c.i16();
  for (std::int16_t i = 0; i < nfilt; ++i) {
    Filter f;
    f.freq = c.f64();
    f.klass = c.i32();
    f.type = c.i32();
    f.npar = c.i16();
    for (std::int16_t k = 0; k < f.npar; ++k) f.pars.push_back(c.f64());
    r.filters.push_back(std::move(f));
  }
  std::vector<std::string> names;
  for (std::int16_t i = 0; i < r.nchan; ++i) names.push_back(c.str(32));
  for (std::int16_t i = 0; i < r.nchan; ++i) {
    ChannelRecord ch;
    ch.name = names[static_cast<std::size_t>(i)];
    ch.sensor_type_index = c.i16();
    ch.original_run_no = c.i16();
    ch.coil_type = c.i32();
    ch.proper_gain = c.f64();
    ch.qgain = c.f64();
    ch.io_gain = c.f64();
    ch.io_offset = c.f64();
    ch.num_coils = c.i16();
    ch.grad_order_no = c.i16();
    c.i32();
    for (Coil& k : ch.coil) k = read_coil(c);
    for (Coil& k : ch.head_coil) k = read_coil(c);
    r.channels.push_back(std::move(ch));
  }
  const std::int16_t ncomp = c.i16();
  for (std::int16_t i = 0; i < ncomp; ++i) {
    CompensationRecord k;
    k.sensor_name = c.str(32);
    k.coeff_type = c.i32();
    c.i32();
    k.ncoeff = c.i16();
    std::vector<std::string> sensors;
    for (int s = 0; s < 50; ++s) sensors.push_back(c.str(31));
    std::vector<double> coeffs;
    for (int s = 0; s < 50; ++s) coeffs.push_back(c.f64());
    if (k.ncoeff < 0 || k.ncoeff > 50) throw format_error("CTF: a compensation row has " + std::to_string(k.ncoeff) + " coefficients (at most 50)");
    k.sensors.assign(sensors.begin(), sensors.begin() + k.ncoeff);
    k.coeffs.assign(coeffs.begin(), coeffs.begin() + k.ncoeff);
    r.compensation.push_back(std::move(k));
  }
  return r;
}

// ---- text files ------------------------------------------------------------------------------------------------------------------------------

std::vector<std::string> lines_of(const ByteSource& src) {
  const std::string text = src.string(0, static_cast<std::size_t>(src.size()));
  std::vector<std::string> out;
  std::size_t st = 0;
  while (st < text.size()) {
    std::size_t e = text.find('\n', st);
    if (e == std::string::npos) e = text.size();
    std::string l = text.substr(st, e - st);
    if (!l.empty() && l.back() == '\r') l.pop_back();
    out.push_back(std::move(l));
    st = e + 1;
  }
  return out;
}

std::string trim(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

std::vector<std::string> split_ws(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream is(s);
  std::string w;
  while (is >> w) out.push_back(w);
  return out;
}

std::string lower(std::string s) { for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; }

double to_double(const std::string& s, const char* what) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || *end != '\0') throw format_error(std::string("CTF: '") + s + "' in " + what + " is not a number");
  return v;
}

std::vector<CoilPoint> parse_hc(const ByteSource& src) {
  const std::vector<std::string> lines = lines_of(src);
  std::vector<CoilPoint> out;
  std::size_t i = 0;
  for (;;) {
    std::string one = "#";
    while (i < lines.size() && one.compare(0, 1, "#") == 0 && !one.empty()) one = lines[i++];
    if (i >= lines.size() && (one.empty() || one[0] == '#')) break;   // end of the file
    const std::string d = trim(one), dl = lower(d);
    if (dl.find("unable") != std::string::npos) throw refused("CTF: HPI information not available (the .hc file says it was unable to measure)");
    CoilPoint p;
    p.descriptor = d;
    p.valid = dl.find("measured") != std::string::npos;
    if (dl.find("relative to dewar") != std::string::npos) p.coord_frame = kFrameDevice;
    else if (dl.find("relative to head") != std::string::npos) p.coord_frame = kFrameHead;
    if (dl.find("nasion") != std::string::npos) p.kind = kNas;
    else if (dl.find("left ear") != std::string::npos || dl.find("lpa") != std::string::npos) p.kind = kLpa;
    else if (dl.find("right ear") != std::string::npos || dl.find("rpa") != std::string::npos) p.kind = kRpa;
    else if (dl.find("spare") != std::string::npos) p.kind = kSpare;
    const char axes[3] = {'x', 'y', 'z'};
    for (int a = 0; a < 3; ++a) {
      const std::string sp = i < lines.size() ? trim(lines[i++]) : std::string();
      if (sp.empty()) continue;   // a blank line
      std::vector<std::string> w;
      std::size_t st = 0;
      for (;;) { const std::size_t k = sp.find(' ', st); w.push_back(sp.substr(st, k == std::string::npos ? std::string::npos : k - st)); if (k == std::string::npos) break; st = k + 1; }
      if (w.size() != 3 || w[0] != std::string(1, axes[a]) || w[1] != "=") throw format_error("CTF: a bad line in the .hc file: '" + d + "'");
      p.r[static_cast<std::size_t>(a)] = to_double(w[2], "the .hc file") / 100.0;
    }
    out.push_back(std::move(p));
    if (i >= lines.size()) break;
  }
  return out;
}

std::vector<Marker> parse_markers(const ByteSource& src) {
  const std::vector<std::string> lines = lines_of(src);
  std::size_t i = 0;
  auto consume_until = [&](const char* prefix) {
    while (i < lines.size() && lines[i].compare(0, std::strlen(prefix), prefix) != 0) ++i;
    if (i >= lines.size()) throw format_error(std::string("CTF: the marker file has no '") + prefix + "' line");
    ++i;
  };
  auto next = [&]() -> const std::string& {
    if (i >= lines.size()) throw format_error("CTF: the marker file ends early");
    return lines[i++];
  };
  std::vector<Marker> out;
  consume_until("NUMBER OF MARKERS:");
  const int n = static_cast<int>(to_double(trim(next()), "the marker count"));
  for (int m = 0; m < n; ++m) {
    Marker mk;
    consume_until("NAME:");
    mk.label = next();
    consume_until("NUMBER OF SAMPLES:");
    const int ns = static_cast<int>(to_double(trim(next()), "a marker's sample count"));
    consume_until("LIST OF SAMPLES:");
    next();   // the column header
    for (int k = 0; k < ns; ++k) {
      const std::vector<std::string> w = split_ws(next());
      if (w.size() < 2) throw format_error("CTF: a marker sample line needs a trial and a time");
      MarkerEntry e;
      e.trial = static_cast<std::int32_t>(to_double(w[0], "a marker trial"));
      e.sync = to_double(w[1], "a marker time");
      mk.samples.push_back(e);
    }
    out.push_back(std::move(mk));
  }
  return out;
}

// ---- dates -------------------------------------------------------------------------------------------------------------------------------------

std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

int month_of(const std::string& s) {
  static const char* names[12] = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"};
  const std::string l = lower(s).substr(0, 3);
  for (int i = 0; i < 12; ++i) if (l == names[i]) return i + 1;
  return 0;
}

// MNE's _convert_time: the date as one of four layouts, the time as H:M:S or H:M, read as UTC
std::int64_t convert_time(const std::string& date_in, const std::string& time_in, int* y_out, int* mo_out, int* d_out, int* h_out, int* mi_out, int* s_out) {
  std::string date = date_in, time = time_in;
  if (date.empty() && time.empty()) { date = "01/01/1970"; time = "00:00:00"; }
  date = trim(date);
  int y = 0, mo = 0, d = 0;
  bool ok = false;
  {
    int a, b, c;
    char t1, t2;
    std::istringstream is(date);
    if ((is >> a >> t1 >> b >> t2 >> c) && t1 == '/' && t2 == '/' && is.peek() == std::istringstream::traits_type::eof()) {
      if (c > 31) { y = c; d = a; mo = b; ok = true; }   // %d/%m/%Y
    }
    if (!ok) {
      std::istringstream is2(date);
      if ((is2 >> a >> t1 >> b >> t2 >> c) && t1 == '/' && t2 == '/' && a > 31) { y = a; mo = b; d = c; ok = true; }   // %Y/%m/%d
    }
  }
  if (!ok) {   // %d-%b-%Y
    const std::size_t d1 = date.find('-'), d2 = d1 == std::string::npos ? d1 : date.find('-', d1 + 1);
    if (d2 != std::string::npos) {
      const int mm = month_of(date.substr(d1 + 1, d2 - d1 - 1));
      if (mm) { d = std::atoi(date.substr(0, d1).c_str()); mo = mm; y = std::atoi(date.substr(d2 + 1).c_str()); ok = d > 0 && y > 0; }
    }
  }
  if (!ok) {   // %a, %b %d, %Y
    const std::size_t comma = date.find(',');
    if (comma != std::string::npos) {
      const std::vector<std::string> w = split_ws(date.substr(comma + 1));   // "Mar 05, 2026"
      if (w.size() == 3 && month_of(w[0])) { mo = month_of(w[0]); d = std::atoi(w[1].c_str()); y = std::atoi(w[2].c_str()); ok = d > 0 && y > 0; }
    }
  }
  if (!ok) throw format_error("CTF: illegal date '" + date_in + "'");
  int h = 0, mi = 0, s = 0;
  {
    char c1, c2;
    std::istringstream is(trim(time));
    if (is >> h >> c1 >> mi && c1 == ':') {
      if (is >> c2 >> s) { if (c2 != ':') throw format_error("CTF: illegal time '" + time_in + "'"); }
    } else {
      throw format_error("CTF: illegal time '" + time_in + "'");
    }
  }
  if (y_out) { *y_out = y; *mo_out = mo; *d_out = d; *h_out = h; *mi_out = mi; *s_out = s; }
  return days_from_civil(y, static_cast<unsigned>(mo), static_cast<unsigned>(d)) * 86400 + h * 3600 + mi * 60 + s;
}

}  // namespace

// the 4 x 4 geometry lives in geometry.h (shared with the importers); these are its public forms in this namespace
std::array<double, 3> apply(const Matrix4& m, const std::array<double, 3>& p) { return geom::apply(m, p); }
Matrix4 multiply(const Matrix4& a, const Matrix4& b) { return geom::multiply(a, b); }
Matrix4 invert(const Matrix4& m) { return geom::invert(m); }
Matrix4 neuromag_head_from_landmarks(const std::array<double, 3>& nas, const std::array<double, 3>& lpa, const std::array<double, 3>& rpa) { return geom::neuromag_head_from_landmarks(nas, lpa, rpa); }
Matrix4 fit_rigid(const std::vector<std::array<double, 3>>& from, const std::vector<std::array<double, 3>>& to) { return geom::fit_rigid(from, to); }

std::string clean_name(const std::string& name) {
  const std::size_t d = name.find('-');
  return d == std::string::npos ? name : name.substr(0, d);
}

// ---- folders -------------------------------------------------------------------------------------------------------------------------------------

DirectoryFolder::DirectoryFolder(std::string path) : path_(std::move(path)) {
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  std::error_code ec;
  if (!fs::is_directory(path_, ec)) throw io_error("CTF: '" + path_ + "' is not a folder");
  const std::string base = fs::path(path_).filename().string();
  if (base.size() < 3 || base.compare(base.size() - 3, 3, ".ds") != 0) throw refused("CTF: the folder '" + base + "' does not end in .ds");
  name_ = base.substr(0, base.size() - 3);
}

std::vector<std::string> DirectoryFolder::list() const {
  std::vector<std::string> out;
  std::error_code ec;
  for (const auto& de : fs::directory_iterator(path_, ec)) if (de.is_regular_file(ec)) out.push_back(de.path().filename().string());
  std::sort(out.begin(), out.end());
  return out;
}

std::shared_ptr<const ByteSource> DirectoryFolder::open(const std::string& file) const {
  const fs::path p = fs::path(path_) / file;
  std::error_code ec;
  if (!fs::is_regular_file(p, ec)) return nullptr;
  return std::make_shared<FileSource>(p.string());
}

std::vector<std::string> MemoryFolder::list() const {
  std::vector<std::string> out;
  for (const auto& f : files_) out.push_back(f.first);
  std::sort(out.begin(), out.end());
  return out;
}

std::shared_ptr<const ByteSource> MemoryFolder::open(const std::string& file) const {
  for (const auto& f : files_) if (f.first == file) return std::make_shared<MemorySource>(f.second);
  return nullptr;
}

// ---- opening ---------------------------------------------------------------------------------------------------------------------------------------

std::unique_ptr<Reader> open(const Folder& folder, const OpenOptions& options) {
  auto rd = std::unique_ptr<Reader>(new Reader());
  Reader& r = *rd;
  const std::string name = folder.name();
  r.files_ = folder.list();
  auto standard = [&](const std::string& ext, bool required) -> std::shared_ptr<const ByteSource> {
    auto s = folder.open(name + "." + ext);
    if (!s && required) throw io_error("CTF: standard file " + name + "." + ext + " not found");
    return s;
  };
  {
    const auto src = standard("res4", true);
    r.res4_ = parse_res4(src->bytes(0, static_cast<std::size_t>(src->size())));
  }
  const Res4& h = r.res4_;
  if (h.pre_trig_pts > 0) throw refused("CTF: an epoched data set (" + std::to_string(h.pre_trig_pts) + " pre-trigger points) is not read; only continuous recordings are");
  if (h.nave > 1) throw refused("CTF: an averaged data set (" + std::to_string(h.nave) + " averages) is not read; only continuous recordings are");
  if (h.nsamp <= 0) throw format_error("CTF: the .res4 file declares " + std::to_string(h.nsamp) + " samples per trial");
  if (h.sfreq <= 0.0) throw format_error("CTF: the .res4 file declares a sampling rate of " + std::to_string(h.sfreq));

  // the other files
  if (auto hc = standard("hc", false)) r.coils_ = parse_hc(*hc);
  if (auto eeg = standard("eeg", false)) {
    for (const std::string& line : lines_of(*eeg)) {
      const std::string l = trim(line);
      if (l.empty()) continue;
      const std::vector<std::string> w = split_ws(l);
      if (w.size() != 5) throw format_error("CTF: illegal data in the EEG position file: '" + l + "'");
      ElectrodePoint e;
      e.id = static_cast<std::int32_t>(to_double(w[0], "the EEG position file"));
      e.label = w[1];
      for (int k = 0; k < 3; ++k) e.r[static_cast<std::size_t>(k)] = to_double(w[static_cast<std::size_t>(2 + k)], "the EEG position file") / 100.0;
      r.electrodes_.push_back(std::move(e));
    }
  }
  {
    std::vector<std::string> pos;
    for (const std::string& f : r.files_) if (f.size() > 4 && f.compare(f.size() - 4, 4, ".pos") == 0) pos.push_back(f);
    if (pos.size() == 1) {
      for (const std::string& line : lines_of(*folder.open(pos[0]))) {
        const std::vector<std::string> w = split_ws(trim(line));
        if (w.size() != 4 && w.size() != 5) continue;
        PolhemusPoint p;
        p.id = w[0];
        if (w.size() == 5) p.label = w[1];
        for (int k = 0; k < 3; ++k) p.r[static_cast<std::size_t>(k)] = to_double(w[w.size() - 3 + static_cast<std::size_t>(k)], "the .pos file") / 100.0;
        r.polhemus_.push_back(std::move(p));
      }
    }
  }
  if (auto mk = folder.open("MarkerFile.mrk")) r.markers_ = parse_markers(*mk);
  std::vector<std::string> bad_names;
  if (auto bc = folder.open("BadChannels")) {
    std::map<std::string, std::string> by_clean;
    for (const auto& c : h.channels) by_clean[clean_name(c.name)] = c.name;
    for (const std::string& line : lines_of(*bc)) {
      const std::string l = trim(line);
      if (l.empty()) continue;
      const auto it = by_clean.find(l);
      if (it == by_clean.end()) throw format_error("CTF: BadChannels names '" + l + "', which is not a channel");
      bad_names.push_back(it->second);
    }
  }
  if (auto bs = folder.open("bad.segments")) {
    for (const std::string& line : lines_of(*bs)) {
      const std::vector<std::string> w = split_ws(line);
      if (w.empty()) continue;
      if (w.size() < 3) throw format_error("CTF: a bad.segments line needs a label, a start and an end");
      r.bad_segments_.push_back({w[0], to_double(w[1], "bad.segments"), to_double(w[2], "bad.segments")});
    }
  }

  // time
  int y, mo, d, hh, mi, ss;
  r.start_seconds_ = convert_time(h.data_date, h.data_time, &y, &mo, &d, &hh, &mi, &ss);

  // transforms from the head coils
  if (!r.coils_.empty()) {
    const CoilPoint *lpa = nullptr, *rpa = nullptr, *nas = nullptr;
    for (const CoilPoint& p : r.coils_) {
      if (!p.valid || p.coord_frame != kFrameHead) continue;
      if (!lpa && p.kind == kLpa) lpa = &p;
      else if (!rpa && p.kind == kRpa) rpa = &p;
      else if (!nas && p.kind == kNas) nas = &p;
    }
    if (!lpa || !rpa || !nas) throw format_error("CTF: some of the mandatory HPI head-coordinate coil positions are not in the .hc file");
    Matrix4 ras = neuromag_head_from_landmarks(nas->r, lpa->r, rpa->r);
    r.transforms_.ctf_head_to_head = invert(invert(ras));   // CTF head -> head is the landmark transform itself
    std::map<std::int32_t, std::array<double, 3>> hp, dp;
    for (const CoilPoint& p : r.coils_) {
      if (!p.valid) continue;
      auto& m = p.coord_frame == kFrameHead ? hp : (p.coord_frame == kFrameDevice ? dp : hp);
      if (p.coord_frame != kFrameHead && p.coord_frame != kFrameDevice) continue;
      for (std::int32_t k : {kLpa, kRpa, kNas, kSpare}) if (!m.count(k) && p.kind == k) m[k] = p.r;
    }
    for (std::int32_t k : {kLpa, kRpa, kNas}) {
      if (!hp.count(k)) throw format_error("CTF: some of the mandatory HPI head-coordinate coil positions are not in the .hc file");
      if (!dp.count(k)) throw format_error("CTF: some of the mandatory HPI device-coordinate coil positions are not in the .hc file");
    }
    std::vector<std::array<double, 3>> rh, rdv;
    for (std::int32_t k : {kLpa, kRpa, kNas, kSpare}) if (hp.count(k) && dp.count(k)) { rh.push_back(hp[k]); rdv.push_back(dp[k]); }
    r.transforms_.ctf_dev_to_ctf_head = fit_rigid(rdv, rh);
    r.transforms_.ctf_dev_to_head = multiply(*r.transforms_.ctf_head_to_head, *r.transforms_.ctf_dev_to_ctf_head);
  }
  {
    Matrix4 R{};
    const double v = 0.5 * std::sqrt(2.0);
    R = {v, -v, 0, 0, v, v, 0, 0, 0, 0, 1, 0.19 * 0 + 0, 0, 0, 0, 1};
    R[11] = 0.19;
    r.transforms_.ctf_dev_to_dev = R;
    if (r.transforms_.ctf_dev_to_head) r.transforms_.dev_head = multiply(*r.transforms_.ctf_dev_to_head, invert(R));
  }

  // channels
  RecordingHeader& hd = r.header_;
  hd.format = "ctf";
  hd.sampling_rate = h.sfreq;
  hd.position_unit = "m";
  std::int32_t grade = -1;
  for (const ChannelRecord& c : h.channels) {
    Channel ch;
    ch.name = c.name;
    const double cal = 1.0 / (c.proper_gain * c.qgain);
    ch.gain = cal;
    ch.sampling_rate = h.sfreq;
    ch.location = std::array<double, 3>{c.coil[0].pos[0], c.coil[0].pos[1], c.coil[0].pos[2]};
    ch.position_frame = "device";
    ch.orientation = std::array<double, 3>{c.coil[0].norm[0], c.coil[0].norm[1], c.coil[0].norm[2]};
    const bool meg_like = c.sensor_type_index == 0 || c.sensor_type_index == 1 || c.sensor_type_index == 5;
    if (meg_like) {
      const double p2 = c.coil[0].pos[0] * c.coil[0].pos[0] + c.coil[0].pos[1] * c.coil[0].pos[1] + c.coil[0].pos[2] * c.coil[0].pos[2];
      const double n2 = c.coil[0].norm[0] * c.coil[0].norm[0] + c.coil[0].norm[1] * c.coil[0].norm[1] + c.coil[0].norm[2] * c.coil[0].norm[2];
      if (p2 < 1e-6 || n2 < 1e-6) {   // no position: MNE makes it a misc channel
        ch.kind = "misc"; ch.unit = "V"; ch.file_unit = "V";
        ch.location.reset(); ch.orientation.reset();
      } else {
        ch.unit = "T"; ch.file_unit = "T";
        if (c.sensor_type_index == 0) { ch.kind = "meg_ref"; ch.coil_type = "CTF_REF_MAG"; }
        else if (c.sensor_type_index == 1) {
          ch.kind = "meg_ref";
          double dx[3], dn = 0;
          for (int k = 0; k < 3; ++k) { dx[k] = c.coil[0].pos[static_cast<std::size_t>(k)] - c.coil[1].pos[static_cast<std::size_t>(k)]; dn += dx[k] * dx[k]; }
          dn = std::sqrt(dn);
          double dotv = 0;
          std::array<double, 3> ez = c.coil[0].norm;
          if (c.proper_gain > 0.0) for (double& x : ez) x = -x;
          const double enorm = std::sqrt(ez[0] * ez[0] + ez[1] * ez[1] + ez[2] * ez[2]);
          if (dn > 0.0) { for (int k = 0; k < 3; ++k) dotv += dx[k] / dn * ez[static_cast<std::size_t>(k)] / enorm; }
          ch.coil_type = (dn > 0.0 && std::fabs(dotv) < 1e-3) ? "CTF_OFFDIAG_REF_GRAD" : "CTF_REF_GRAD";
        } else {
          ch.kind = "meg"; ch.coil_type = "CTF_GRAD";
          if (grade == -1) grade = c.grad_order_no;
          else if (grade != c.grad_order_no)
            throw format_error("CTF: gradiometer channel " + c.name + " with inconsistent compensation grade " + std::to_string(c.grad_order_no) + ", should be " + std::to_string(grade));
        }
      }
    } else if (c.sensor_type_index == 9) { ch.kind = "eeg"; ch.unit = "V"; ch.file_unit = "V"; ch.coil_type = "EEG"; }
    else if (c.sensor_type_index == 11) { ch.kind = "stim"; ch.unit = "V"; ch.file_unit = "V"; ch.location.reset(); ch.orientation.reset(); }
    else { ch.kind = "misc"; ch.unit = "V"; ch.file_unit = "V"; ch.location.reset(); ch.orientation.reset(); }
    if (std::find(bad_names.begin(), bad_names.end(), c.name) != bad_names.end()) ch.bad = true;
    r.cal_.push_back(cal);
    hd.channels.push_back(std::move(ch));
  }
  r.grade_ = grade < 0 ? 0 : grade;
  hd.bad_channels = bad_names;

  // data files: <name>.meg4, <name>.1_meg4, ...
  int clock = -1;
  for (std::size_t i = 0; i < h.channels.size(); ++i) if (h.channels[i].name == "SCLK01-177") { clock = static_cast<int>(i); break; }
  std::uint64_t offset = 0;
  const std::uint64_t nsamp = static_cast<std::uint64_t>(h.nsamp), nchan = static_cast<std::uint64_t>(h.nchan);
  std::vector<std::string> missing, empty;
  for (int idx = 0;; ++idx) {
    const std::string suffix = idx == 0 ? "meg4" : std::to_string(idx) + "_meg4";
    auto src = standard(suffix, false);
    if (!src) { missing.push_back(name + "." + suffix); break; }
    const std::uint64_t size = src->size();
    if (size < kHeaderSize) throw format_error("CTF: " + name + "." + suffix + " is shorter than its header");
    unsigned char magic[4];
    src->read(0, 4, magic);
    if (std::memcmp(magic, "MEG4", 4) != 0) throw format_error("CTF: " + name + "." + suffix + " does not start with MEG4 (not a CTF data file)");
    if ((size - kHeaderSize) % (4 * nsamp * nchan) != 0) throw format_error("CTF: the number of samples in " + name + "." + suffix + " is not an even multiple of the trial size");
    Reader::Part p;
    p.src = src;
    p.n_samp_tot = (size - kHeaderSize) / (4 * nchan);
    p.n_samp = p.n_samp_tot;
    const std::uint64_t n_trial = p.n_samp_tot / nsamp;
    if (clock >= 0 && options.system_clock == OpenOptions::SystemClock::truncate) {
      for (std::uint64_t t = 0; t < n_trial; ++t) {
        const std::uint64_t at = kHeaderSize + (t * nsamp * nchan + static_cast<std::uint64_t>(clock) * nsamp) * 4;
        std::vector<unsigned char> row(static_cast<std::size_t>(nsamp) * 4);
        src->read(at, row.size(), row.data());
        bool stopped = false;
        for (std::uint64_t s = 0; s < nsamp; ++s) if (load_i32(&row[static_cast<std::size_t>(s) * 4], false) == 0) { p.n_samp = t * nsamp + s; stopped = true; break; }
        if (stopped) break;
      }
    }
    if (p.n_samp == 0) { empty.push_back(name + "." + suffix); break; }
    p.offset = offset;
    offset += p.n_samp;
    r.parts_.push_back(std::move(p));
  }
  if (r.parts_.empty()) {
    std::string m;
    for (const auto& s : missing) m += (m.empty() ? "" : ", ") + s;
    std::string e;
    for (const auto& s : empty) e += (e.empty() ? "" : ", ") + s;
    throw io_error("CTF: no data: could not find " + (m.empty() ? std::string("(none missing)") : m) + "; no valid samples in " + (e.empty() ? std::string("(none)") : e));
  }
  hd.n_samples = offset;
  {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d", y, mo, d, hh, mi, ss);
    hd.start_time = buf;
  }
  // annotations: bad segments, then markers
  const double pre = static_cast<double>(h.pre_trig_pts) / h.sfreq;
  for (const BadSegment& b : r.bad_segments_) hd.annotations.push_back({b.start + pre, b.end - b.start, "bad_" + b.label, ""});
  const double trial_duration = static_cast<double>(h.nsamp) / h.sfreq;
  for (const Marker& m : r.markers_)
    for (const MarkerEntry& e : m.samples) hd.annotations.push_back({e.sync + static_cast<double>(e.trial) * trial_duration + pre, 0.0, m.label, ""});
  // every header field by its own name
  auto put = [&](const char* k, const std::string& v) { hd.info[k] = v; };
  put("head", h.head); put("appname", h.appname); put("origin", h.origin); put("desc", h.desc);
  put("nave", std::to_string(h.nave)); put("data_time", h.data_time); put("data_date", h.data_date);
  put("nsamp", std::to_string(h.nsamp)); put("nchan", std::to_string(h.nchan)); put("epoch_time", std::to_string(h.epoch_time));
  put("no_trials", std::to_string(h.no_trials)); put("pre_trig_pts", std::to_string(h.pre_trig_pts));
  put("no_trials_done", std::to_string(h.no_trials_done)); put("save_trials", std::to_string(h.save_trials));
  put("trigger_mode", std::to_string(h.trigger_mode)); put("accept_reject", std::to_string(h.accept_reject));
  put("zero_head", std::to_string(h.zero_head)); put("artifact_mode", std::to_string(h.artifact_mode));
  put("nf_run_name", h.nf_run_name); put("nf_run_title", h.nf_run_title); put("nf_instruments", h.nf_instruments);
  put("nf_collect_descriptor", h.nf_collect_descriptor); put("nf_subject_id", h.nf_subject_id); put("nf_operator", h.nf_operator);
  put("run_desc", h.run_desc); put("compensation_grade", std::to_string(r.grade_));
  put("trials", std::to_string(h.no_trials));
  return rd;
}

std::unique_ptr<Reader> open_directory(const std::string& path, const OpenOptions& options) {
  DirectoryFolder f(path);
  return open(f, options);
}

void Reader::read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const {
  check_range(s0, n, c0, c1);
  const std::uint64_t nsamp = static_cast<std::uint64_t>(res4_.nsamp), nchan = static_cast<std::uint64_t>(res4_.nchan);
  std::vector<unsigned char> buf;
  for (const Part& p : parts_) {
    const std::uint64_t a = std::max(s0, p.offset), b = std::min(s0 + n, p.offset + p.n_samp);
    if (a >= b) continue;
    std::uint64_t local = a - p.offset;
    while (a + (local - (a - p.offset)) < b) {
      const std::uint64_t t = local / nsamp, o = local % nsamp;
      const std::uint64_t run = std::min(nsamp - o, b - (p.offset + local));
      for (std::size_t c = c0; c < c1; ++c) {
        buf.resize(static_cast<std::size_t>(run) * 4);
        p.src->read(kHeaderSize + ((t * nchan + c) * nsamp + o) * 4, buf.size(), buf.data());
        float* dst = out + (c - c0) * n + (p.offset + local - s0);
        const double cal = cal_[c];
        for (std::uint64_t i = 0; i < run; ++i) dst[i] = static_cast<float>(static_cast<double>(load_i32(&buf[static_cast<std::size_t>(i) * 4], false)) * cal);
      }
      local += run;
    }
  }
}

}  // namespace nxr::formats::ctf
