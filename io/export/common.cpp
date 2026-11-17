// The exporters' shared part: see nxr/exporters/common.h.
#include "nxr/exporters/common.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>

#include "nxr/io/errors.h"
#include "nxr/io/hash.h"

namespace nxr::exporters {

namespace code = nxr::io::code;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}
}  // namespace

bool is_share_alike(const std::string& licence) {
  const std::string l = lower(licence);
  return l.find("-sa") != std::string::npos || l.find(" sa") != std::string::npos || l.find("sharealike") != std::string::npos || l.find("share-alike") != std::string::npos ||
         l.find("share alike") != std::string::npos || l.find("odbl") != std::string::npos;
}

Json Terms::to_json() const {
  return Json{{"licence", licence}, {"attribution", attribution}, {"redistributable", redistributable}, {"share_alike", share_alike}, {"origins", origins}};
}

Terms terms_of(const task::Subject& subject, const std::vector<task::Node>& inputs) {
  Terms t;
  std::set<std::string> licences, attributions, sa;
  bool all_redistributable = !inputs.empty();
  for (const auto& n : inputs) {
    const auto o = subject.origin(n);
    Json rec = Json{{"name", n.name}, {"class", task::to_string(n.cls)}};
    if (o) {
      rec["source"] = o->source;
      rec["licence"] = o->licence;
      rec["attribution"] = o->attribution;
      rec["redistributable"] = o->redistributable;
      licences.insert(o->licence);
      if (o->attribution != "unknown") attributions.insert(o->attribution);
      if (is_share_alike(o->licence)) sa.insert(o->licence);
      if (!o->redistributable) all_redistributable = false;
    } else {
      rec["licence"] = "unknown";
      licences.insert("unknown");
      all_redistributable = false;
    }
    t.origins.push_back(rec);
  }
  if (sa.size() > 1) {
    std::string both;
    for (const auto& l : sa) both += (both.empty() ? "" : ", ") + std::string("'") + l + "'";
    refuse("the inputs carry different ShareAlike licences (" + both + "), which cannot be combined in one export; export them separately");
  }
  t.share_alike = !sa.empty();
  if (t.share_alike) t.licence = *sa.begin();
  else if (licences.size() == 1) t.licence = *licences.begin();
  else if (!licences.empty()) {
    std::string all;
    for (const auto& l : licences) all += (all.empty() ? "" : " + ") + l;
    t.licence = all;                                   // several different licences, none ShareAlike: all are listed, none is picked
  }
  if (!attributions.empty()) {
    std::string a;
    for (const auto& s : attributions) a += (a.empty() ? "" : "; ") + s;
    t.attribution = a;
  }
  t.redistributable = all_redistributable;
  return t;
}

Json input_digest(const task::Subject& subject, const task::Node& node) {
  std::vector<std::string> prefixes;
  if (node.path) prefixes.push_back(*node.path);
  if (node.array_path) prefixes.push_back(*node.array_path);
  nxr::io::Sha256 h;
  std::int64_t keys = 0, bytes = 0;
  for (const auto& p : prefixes)
    for (const auto& d : subject.key_digests(p)) {
      h.update(d.key + ":" + std::to_string(d.size_bytes) + ":" + d.sha256 + "\n");
      ++keys;
      bytes += d.size_bytes;
    }
  return Json{{"keys", keys}, {"bytes", bytes}, {"sha256", keys ? Json(h.hex()) : Json()}};
}

Json provenance(const nxr::io::SubjectPackage& package, const task::Subject& subject, const Context& ctx, const std::vector<task::Node>& inputs, const Json& span, const Json& extra) {
  Json p = Json::object();
  p["schema"] = "nxr-export/1";
  p["producer"] = Json{{"name", ctx.tool}, {"version", ctx.version}};
  p["command"] = ctx.command;
  p["options"] = ctx.options;
  p["span"] = span;
  const nxr::io::PackageRecord rec = package.record();
  p["package"] = Json{{"name", rec.name}, {"content_hash", package.content_hash()}};
  Json ins = Json::array();
  for (const auto& n : inputs) {
    Json i = Json{{"class", task::to_string(n.cls)}, {"name", n.name}, {"id", n.id}, {"kind", n.kind ? Json(*n.kind) : Json()}, {"unit", n.unit ? Json(*n.unit) : Json()},
                  {"producer", n.producer}, {"arrays", input_digest(subject, n)}};
    ins.push_back(i);
  }
  p["inputs"] = ins;
  p["terms"] = terms_of(subject, inputs).to_json();
  p["details"] = extra;
  return p;
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

std::string write_sidecar(const std::string& first_file, Json prov, const std::vector<std::string>& files) {
  Json outs = Json::array();
  for (const auto& f : files) outs.push_back(Json{{"file", f.substr(f.find_last_of("/\\") == std::string::npos ? 0 : f.find_last_of("/\\") + 1)}, {"sha256", file_sha256(f)}});
  prov["outputs"] = outs;
  const std::string path = first_file + ".json";
  std::ofstream o(path, std::ios::binary);
  if (!o) throw nxr::io::Error(code::not_writable, "cannot write " + path);
  o << prov.dump(2, ' ', false) << "\n";
  return path;
}

std::vector<double> widen(const std::vector<std::uint8_t>& raw, nxr::io::DType dtype) {
  using nxr::io::DType;
  const std::size_t sz = nxr::io::dtype_size(dtype);
  const std::size_t n = raw.size() / sz;
  std::vector<double> out(n);
  auto conv = [&](auto tag) {
    using T = decltype(tag);
    for (std::size_t i = 0; i < n; ++i) {
      T v;
      std::memcpy(&v, raw.data() + i * sizeof(T), sizeof(T));
      out[i] = static_cast<double>(v);
    }
  };
  switch (dtype) {
    case DType::Float64: std::memcpy(out.data(), raw.data(), raw.size()); break;
    case DType::Float32: conv(float{}); break;
    case DType::Int8: conv(std::int8_t{}); break;
    case DType::Int16: conv(std::int16_t{}); break;
    case DType::Int32: conv(std::int32_t{}); break;
    case DType::Int64: conv(std::int64_t{}); break;
    case DType::UInt8: conv(std::uint8_t{}); break;
    case DType::UInt16: conv(std::uint16_t{}); break;
    case DType::UInt32: conv(std::uint32_t{}); break;
    case DType::UInt64: conv(std::uint64_t{}); break;
    case DType::Bool: conv(std::uint8_t{}); break;
    default: throw nxr::io::Error(code::unsupported, std::string("an export writes real numeric arrays; this one is ") + nxr::io::dtype_to_string(dtype));
  }
  return out;
}

}  // namespace nxr::exporters
