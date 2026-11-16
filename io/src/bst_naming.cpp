// nxr::bst — nxr.convert.naming: the names of session-scoped nodes, and the sha1 fingerprint of a source.
#include "nxr/bst/naming.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "nxr/bst/protocol.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::SubjectWriter;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
std::string stem_of(const fs::path& p) { return p.stem().string(); }

// ── sha1 ──
struct Sha1 {
  std::uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
  std::uint64_t len = 0;
  unsigned char buf[64];
  std::size_t fill = 0;
  static std::uint32_t rol(std::uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
  void block(const unsigned char* p) {
    std::uint32_t w[80];
    for (int i = 0; i < 16; ++i) w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) | (std::uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
    for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
      std::uint32_t f, k;
      if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else { f = b ^ c ^ d; k = 0xCA62C1D6; }
      const std::uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
  }
  void update(const unsigned char* p, std::size_t n) {
    len += n;
    while (n) {
      const std::size_t k = std::min<std::size_t>(64 - fill, n);
      std::memcpy(buf + fill, p, k);
      fill += k; p += k; n -= k;
      if (fill == 64) { block(buf); fill = 0; }
    }
  }
  std::string hex() {
    const std::uint64_t bits = len * 8;
    unsigned char pad = 0x80;
    update(&pad, 1);
    pad = 0;
    while (fill != 56) update(&pad, 1);
    unsigned char l[8];
    for (int i = 0; i < 8; ++i) l[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    update(l, 8);
    char out[41];
    for (int i = 0; i < 5; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    return out;
  }
};

}  // namespace

std::string sha1_hex(const std::string& bytes) {
  Sha1 s;
  s.update(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
  return s.hex();
}

std::string source_fingerprint(const std::vector<fs::path>& files, const std::string& extra) {
  Sha1 s;
  std::vector<char> b(1 << 20);
  for (const auto& f : files) {
    std::ifstream in(f, std::ios::binary);
    if (!in) throw nxr::io::Error(code::not_found, "[Errno 2] No such file or directory: " + str_repr(py_path(f.string())));
    while (in) {
      in.read(b.data(), static_cast<std::streamsize>(b.size()));
      s.update(reinterpret_cast<const unsigned char*>(b.data()), static_cast<std::size_t>(in.gcount()));
    }
  }
  s.update(reinterpret_cast<const unsigned char*>(extra.data()), extra.size());
  return s.hex();
}

std::string recording_node_name(const fs::path& data_file) {
  std::string s = stem_of(data_file);
  if (s.rfind("data_0raw_", 0) == 0) s = s.substr(10);
  else if (s.rfind("data_", 0) == 0) s = s.substr(5);
  return nxr::io::sanitize_node_name(s);
}

std::string channel_node_name(const fs::path& channel_file) { return nxr::io::sanitize_node_name(stem_of(channel_file)); }

std::string acquisition_label(const std::string& condition) {
  auto alnum = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; };
  std::string joined;
  for (const char* e : {"ses", "task", "acq", "run", "rec"}) {
    const std::string key = std::string(e) + "-";
    for (std::size_t p = condition.find(key); p != std::string::npos; p = condition.find(key, p + 1)) {
      if (p > 0 && alnum(condition[p - 1])) continue;
      std::size_t q = p + key.size();
      while (q < condition.size() && alnum(condition[q])) ++q;
      if (q == p + key.size()) continue;
      joined += (joined.empty() ? "" : "_") + condition.substr(p, q - p);
      break;
    }
  }
  return nxr::io::sanitize_node_name(joined.empty() ? condition : joined);
}

std::pair<std::string, bool> claim_node_name(SubjectWriter& sub, const std::string& folder, const std::string& stem, const std::string& acquisition, const std::string& fingerprint) {
  const std::string label = acquisition_label(acquisition);
  std::vector<std::string> candidates = {stem, stem + "__" + label};
  for (int k = 2; k < 100; ++k) candidates.push_back(stem + "__" + label + "_" + std::to_string(k));
  for (const auto& name : candidates) {
    const std::string path = folder.empty() ? name : folder + "/" + name;
    std::optional<nxr::io::Row> row;
    for (const char* t : {"manifold", "operator", "field", "selection"}) if ((row = sub.find(t, path))) break;
    if (!row) return {name, false};
    const auto pj = nxr::io::opt_text(*row, "producer_json");
    const Json j = Json::parse(pj && !pj->empty() ? *pj : "{}");
    if (j.contains("source_sha1") && j["source_sha1"] == fingerprint) return {name, true};
  }
  value_error(stem + ": 100 different sources claim this name in " + sub.name());
}


std::string kernel_node_name(const fs::path& results_file) {
  std::string s = stem_of(results_file);
  if (s.rfind("results_", 0) == 0) s = s.substr(8);
  return nxr::io::sanitize_node_name(s);
}
std::string headmodel_node_name(const fs::path& headmodel_file) { return nxr::io::sanitize_node_name(stem_of(headmodel_file)); }

}  // namespace nxr::bst
