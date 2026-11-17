#include "nxr/formats/brainvision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "nxr/formats/endian.h"
#include "nxr/formats/units.h"

namespace nxr::formats::brainvision {

namespace {

struct Ini {
  std::string first_line;
  std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> sections;   // in file order
  const std::vector<std::pair<std::string, std::string>>* find(const std::string& name) const {
    for (const auto& s : sections) if (s.first == name) return &s.second;
    return nullptr;
  }
  std::string get(const std::string& section, const std::string& key, const std::string& dflt = "") const {
    const auto* s = find(section);
    if (s) for (const auto& kv : *s) if (kv.first == key) return kv.second;
    return dflt;
  }
};

std::string latin1_to_utf8(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c < 0x80) o += static_cast<char>(c);
    else { o += static_cast<char>(0xC0 | (c >> 6)); o += static_cast<char>(0x80 | (c & 0x3F)); }
  }
  return o;
}

std::string trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
  return s.substr(b, e - b);
}

std::string lower(std::string s) { for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return s; }

Ini parse_ini(std::string text, const char* expect_prefix, const char* what) {
  if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF && static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) text.erase(0, 3);
  Ini ini;
  std::size_t pos = 0;
  bool first = true;
  while (pos <= text.size()) {
    std::size_t e = text.find('\n', pos);
    if (e == std::string::npos) e = text.size();
    std::string line = trim(text.substr(pos, e - pos));
    pos = e + 1;
    if (first) {
      first = false;
      ini.first_line = line;
      if (line.compare(0, std::char_traits<char>::length(expect_prefix), expect_prefix) != 0)
        throw format_error(std::string("BrainVision: not a ") + what + " (the first line is \"" + line.substr(0, 60) + "\")");
      continue;
    }
    if (line.empty() || line[0] == ';') continue;
    if (line[0] == '[' && line.back() == ']') { ini.sections.emplace_back(line.substr(1, line.size() - 2), std::vector<std::pair<std::string, std::string>>{}); continue; }
    if (ini.sections.empty()) continue;
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
      if (ini.sections.back().first == "Comment") ini.sections.back().second.emplace_back("", line);
      continue;
    }
    ini.sections.back().second.emplace_back(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
  }
  return ini;
}

std::string unescape(std::string s) {
  std::size_t p = 0;
  while ((p = s.find("\\1", p)) != std::string::npos) { s.replace(p, 2, ","); ++p; }
  return s;
}

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::size_t b = 0;
  while (true) {
    const std::size_t c = s.find(',', b);
    if (c == std::string::npos) { out.push_back(s.substr(b)); break; }
    out.push_back(s.substr(b, c - b));
    b = c + 1;
  }
  return out;
}

double to_double(const std::string& s, const std::string& what) {
  const std::string t = trim(s);
  char* end = nullptr;
  const double v = std::strtod(t.c_str(), &end);
  if (t.empty() || *end != '\0') throw format_error("BrainVision: " + what + " (\"" + t + "\") is not a number");
  return v;
}

class Reader final : public RecordingReader {
 public:
  Reader(std::shared_ptr<const ByteSource> hdr, std::shared_ptr<const ByteSource> mrk, std::shared_ptr<const ByteSource> data) : data_(std::move(data)) {
    parse(*hdr, mrk.get());
  }
  const RecordingHeader& header() const override { return h_; }

  void read(std::uint64_t s0, std::uint64_t n, std::size_t c0, std::size_t c1, float* out) const override {
    check_range(s0, n, c0, c1);
    if (n == 0 || c0 == c1) return;
    const std::size_t nch = h_.channels.size();
    if (multiplexed_) {
      std::vector<unsigned char> buf(static_cast<std::size_t>(n) * nch * bps_);   // bounded by the call
      data_->read(s0 * nch * bps_, buf.size(), buf.data());
      for (std::size_t c = c0; c < c1; ++c)
        for (std::uint64_t i = 0; i < n; ++i) out[(c - c0) * n + i] = value(buf.data() + (i * nch + c) * bps_, c);
    } else {
      std::vector<unsigned char> buf(static_cast<std::size_t>(n) * bps_);
      for (std::size_t c = c0; c < c1; ++c) {
        data_->read((c * h_.n_samples + s0) * bps_, buf.size(), buf.data());
        for (std::uint64_t i = 0; i < n; ++i) out[(c - c0) * n + i] = value(buf.data() + i * bps_, c);
      }
    }
  }

 private:
  float value(const unsigned char* p, std::size_t c) const {
    double v;
    switch (kind_) {
      case 0: v = load_i16(p, true); break;
      case 1: v = load_i32(p, true); break;
      default: v = load_f32(p, true); break;
    }
    return static_cast<float>(v * mul_[c]);
  }

  void parse(const ByteSource& hsrc, const ByteSource* msrc) {
    std::string text = hsrc.string(0, static_cast<std::size_t>(hsrc.size()));
    Ini ini = parse_ini(text, "Brain Vision Data Exchange Header File", "BrainVision header (.vhdr)");
    const std::string codepage = lower(ini.get("Common Infos", "Codepage", "ANSI"));
    const bool utf8 = codepage == "utf-8" || codepage == "utf8";
    auto text_of = [&](const std::string& s) { return utf8 ? s : latin1_to_utf8(s); };

    const std::string fmt = ini.get("Common Infos", "DataFormat", "BINARY");
    if (lower(fmt) != "binary") throw refused("BrainVision: DataFormat=" + fmt + " (ASCII data) is not read; only BINARY. Export the recording as binary");
    const std::string type = ini.get("Common Infos", "DataType", "TIME");
    if (lower(type) != "time") throw refused("BrainVision: DataType=" + type + " is not read; only TIME-domain data");
    const std::string orient = ini.get("Common Infos", "DataOrientation", "MULTIPLEXED");
    if (lower(orient) == "multiplexed") multiplexed_ = true;
    else if (lower(orient) == "vectorized") multiplexed_ = false;
    else throw format_error("BrainVision: DataOrientation=" + orient);
    const std::string bf = ini.get("Binary Infos", "BinaryFormat", "INT_16");
    if (bf == "INT_16") { kind_ = 0; bps_ = 2; }
    else if (bf == "INT_32") { kind_ = 1; bps_ = 4; }
    else if (bf == "IEEE_FLOAT_32") { kind_ = 2; bps_ = 4; }
    else throw refused("BrainVision: BinaryFormat=" + bf + " is not read (supported: INT_16, INT_32, IEEE_FLOAT_32)");
    const int nch = static_cast<int>(to_double(ini.get("Common Infos", "NumberOfChannels"), "NumberOfChannels"));
    const double interval = to_double(ini.get("Common Infos", "SamplingInterval"), "SamplingInterval");
    if (nch < 1 || !(interval > 0)) throw format_error("BrainVision: NumberOfChannels / SamplingInterval out of range");
    const double rate = 1e6 / interval;

    const auto* chs = ini.find("Channel Infos");
    std::map<int, std::vector<std::string>> chmap;
    if (chs) for (const auto& kv : *chs) if (kv.first.size() > 2 && kv.first.compare(0, 2, "Ch") == 0) chmap[std::atoi(kv.first.c_str() + 2)] = split_commas(kv.second);
    const auto* coords = ini.find("Coordinates");
    std::map<int, std::vector<std::string>> cmap;
    if (coords) for (const auto& kv : *coords) if (kv.first.size() > 2 && kv.first.compare(0, 2, "Ch") == 0) cmap[std::atoi(kv.first.c_str() + 2)] = split_commas(kv.second);
    for (int i = 1; i <= nch; ++i) {
      if (!chmap.count(i)) throw format_error("BrainVision: no [Channel Infos] entry Ch" + std::to_string(i) + " for NumberOfChannels=" + std::to_string(nch));
      const auto& f = chmap[i];
      Channel c;
      c.name = text_of(unescape(f.empty() ? "" : f[0]));
      c.reference = f.size() > 1 ? text_of(unescape(f[1])) : "";
      const double res = (f.size() > 2 && !trim(f[2]).empty()) ? to_double(f[2], "the resolution of Ch" + std::to_string(i)) : 1.0;
      c.file_unit = (f.size() > 3 && !trim(f[3]).empty()) ? text_of(trim(f[3])) : "\xC2\xB5V";   // µV when absent
      const SiUnit u = si_unit(c.file_unit);
      c.unit = u.unit;
      c.scale_to_si = u.scale;
      c.gain = res;
      c.offset = 0.0;
      c.sampling_rate = rate;
      if (cmap.count(i)) {
        const auto& p = cmap[i];
        if (p.size() >= 3) {
          const double r = to_double(p[0], "a radius"), th = to_double(p[1], "a theta") * 3.14159265358979323846 / 180.0, ph = to_double(p[2], "a phi") * 3.14159265358979323846 / 180.0;
          if (r != 0.0 || th != 0.0 || ph != 0.0) c.location = std::array<double, 3>{r * std::sin(th) * std::cos(ph), r * std::sin(th) * std::sin(ph), r * std::cos(th)};
        }
      }
      h_.channels.push_back(c);
      mul_.push_back(c.gain * c.scale_to_si);
    }

    // data size -> samples
    const std::uint64_t bytes = data_->size();
    const std::uint64_t frame = static_cast<std::uint64_t>(nch) * bps_;
    if (bytes % frame != 0)
      throw format_error("BrainVision: the data file holds " + std::to_string(bytes) + " bytes, not a whole number of samples of " + std::to_string(nch) + " channels x " + std::to_string(bps_) + " bytes: the file is truncated or the header is wrong");
    h_.format = "brainvision";
    h_.sampling_rate = rate;
    h_.n_samples = bytes / frame;
    h_.position_unit = "";
    h_.info["version"] = ini.first_line.substr(std::string("Brain Vision Data Exchange Header File ").size());
    h_.info["codepage"] = ini.get("Common Infos", "Codepage");
    h_.info["data_file"] = ini.get("Common Infos", "DataFile");
    h_.info["marker_file"] = ini.get("Common Infos", "MarkerFile");
    h_.info["data_format"] = fmt;
    h_.info["data_orientation"] = orient;
    h_.info["binary_format"] = bf;
    h_.info["sampling_interval_us"] = ini.get("Common Infos", "SamplingInterval");
    std::string comment;
    if (const auto* cm = ini.find("Comment")) for (const auto& kv : *cm) comment += (comment.empty() ? "" : "\n") + (kv.first.empty() ? kv.second : kv.first + "=" + kv.second);
    h_.info["comment"] = text_of(comment);

    if (msrc) parse_markers(*msrc, rate);
  }

  void parse_markers(const ByteSource& src, double rate) {
    const Ini ini = parse_ini(src.string(0, static_cast<std::size_t>(src.size())), "Brain Vision Data Exchange Marker File", "BrainVision marker file (.vmrk)");
    const std::string codepage = lower(ini.get("Common Infos", "Codepage", "ANSI"));
    const bool utf8 = codepage == "utf-8" || codepage == "utf8";
    const auto* mk = ini.find("Marker Infos");
    if (!mk) return;
    std::string segments;
    for (const auto& kv : *mk) {
      if (kv.first.size() < 3 || kv.first.compare(0, 2, "Mk") != 0) continue;
      std::vector<std::string> f = split_commas(kv.second);
      if (f.size() < 3) throw format_error("BrainVision: marker " + kv.first + " has fewer than three fields");
      for (std::string& s : f) { s = unescape(s); if (!utf8) s = latin1_to_utf8(s); }
      const double pos = to_double(f[2], "the position of " + kv.first);
      const double size = f.size() > 3 && !trim(f[3]).empty() ? to_double(f[3], "the size of " + kv.first) : 1.0;
      const int chn = f.size() > 4 && !trim(f[4]).empty() ? static_cast<int>(to_double(f[4], "the channel of " + kv.first)) : 0;
      if (pos < 1) throw format_error("BrainVision: marker " + kv.first + " has position " + trim(f[2]) + " (positions count from 1)");
      Annotation a;
      a.onset = (pos - 1.0) / rate;
      a.duration = size / rate;
      a.label = f[0] + "/" + f[1];
      if (chn >= 1 && static_cast<std::size_t>(chn) <= h_.channels.size()) a.channels = h_.channels[static_cast<std::size_t>(chn) - 1].name;
      h_.annotations.push_back(a);
      if (f[0] == "New Segment") {
        segments += (segments.empty() ? "" : ",") + std::to_string(static_cast<std::uint64_t>(pos) - 1);
        if (f.size() > 5 && f[5].size() >= 14 && h_.start_time.empty()) {
          const std::string& d = f[5];
          h_.start_time = d.substr(0, 4) + "-" + d.substr(4, 2) + "-" + d.substr(6, 2) + "T" + d.substr(8, 2) + ":" + d.substr(10, 2) + ":" + d.substr(12, 2) + (d.size() > 14 ? "." + d.substr(14) : "");
        }
      }
    }
    if (!segments.empty()) h_.info["segment_starts"] = segments;
  }

  std::shared_ptr<const ByteSource> data_;
  RecordingHeader h_;
  std::vector<double> mul_;
  bool multiplexed_ = true;
  int kind_ = 0;
  int bps_ = 2;
};

}  // namespace

std::unique_ptr<RecordingReader> open(std::shared_ptr<const ByteSource> header, std::shared_ptr<const ByteSource> markers, std::shared_ptr<const ByteSource> data) {
  return std::make_unique<Reader>(std::move(header), std::move(markers), std::move(data));
}

std::unique_ptr<RecordingReader> open_file(const std::string& vhdr_path) {
  auto h = std::make_shared<FileSource>(vhdr_path);
  const std::string text = h->string(0, static_cast<std::size_t>(h->size()));
  auto named = [&](const char* key) {
    std::size_t p = 0;
    const std::string k = std::string(key) + "=";
    while ((p = text.find(k, p)) != std::string::npos) {
      if (p == 0 || text[p - 1] == '\n') {
        const std::size_t e = text.find_first_of("\r\n", p);
        return trim(text.substr(p + k.size(), e == std::string::npos ? std::string::npos : e - p - k.size()));
      }
      ++p;
    }
    return std::string();
  };
  const std::size_t slash = vhdr_path.find_last_of("/\\");
  const std::string dir = slash == std::string::npos ? "" : vhdr_path.substr(0, slash + 1);
  const std::string data_name = named("DataFile"), marker_name = named("MarkerFile");
  if (data_name.empty()) throw format_error("BrainVision: the header names no DataFile");
  auto data = std::make_shared<FileSource>(dir + data_name);
  std::shared_ptr<const ByteSource> markers;
  if (!marker_name.empty()) {
    std::FILE* f = std::fopen((dir + marker_name).c_str(), "rb");
    if (f) { std::fclose(f); markers = std::make_shared<FileSource>(dir + marker_name); }
  }
  return open(h, markers, data);
}

}  // namespace nxr::formats::brainvision
