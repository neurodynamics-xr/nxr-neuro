// nxr-io-bst-dump FILE — every variable of a .mat as JSON, through nxr::bst (for test/parity: compared with the values scipy wrote).
#include <cstring>
#include <iostream>

#include "nxr/bst/mat.h"
#include "nxr/io/py_json.h"   // the JSON writer of the command line (Python-style number text)

using namespace nxr::bst;
using nxr::io::Json;

namespace {
Json describe(const Value& v) {
  Json j = Json::object();
  j["class"] = class_name(v.cls);
  Json dims = Json::array();
  for (auto d : v.dims) dims.push_back(d);
  j["dims"] = dims;
  if (v.is_complex) j["complex"] = true;
  if (v.is_logical) j["logical"] = true;
  if (v.is_numeric() || v.cls == Class::Sparse) {
    Json data = Json::array();
    if (v.cls == Class::Int64 || v.cls == Class::UInt64 || v.cls == Class::Int32 || v.cls == Class::UInt32 || v.cls == Class::Int16 || v.cls == Class::UInt16 ||
        v.cls == Class::Int8 || v.cls == Class::UInt8) {
      if (v.cls == Class::UInt64) {
        for (std::size_t i = 0; i + 8 <= v.re.size(); i += 8) { std::uint64_t x; std::memcpy(&x, v.re.data() + i, 8); data.push_back(x); }
      } else {
        for (auto x : v.int64s()) data.push_back(x);
      }
    } else {
      for (auto x : v.doubles()) data.push_back(x);
    }
    j["data"] = data;
    if (v.is_complex) { Json im = Json::array(); for (auto x : v.imag()) im.push_back(x); j["imag"] = im; }
  } else if (v.is_char()) {
    j["text"] = v.text;
  } else if (v.is_struct()) {
    j["fields"] = v.fields;
    Json elems = Json::array();
    for (std::size_t e = 0; e < v.numel(); ++e) {
      Json o = Json::object();
      for (const auto& f : v.fields) o[f] = describe(*v.field(f, e));
      elems.push_back(o);
    }
    j["elements"] = elems;
  } else if (v.is_cell()) {
    Json cells = Json::array();
    for (const auto& c : v.elems) cells.push_back(describe(c));
    j["cells"] = cells;
  }
  return j;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: nxr-io-bst-dump FILE\n"; return 2; }
  try {
    MatFile f = MatFile::open(argv[1]);
    Json out = Json::object();
    for (const auto& [name, v] : f.read_all()) out[name] = describe(v);
    std::cout << pyjson::dumps(out) << std::endl;
  } catch (const std::exception& e) {
    std::cout << pyjson::dumps(Json{{"error", e.what()}}) << std::endl;
    return 1;
  }
  return 0;
}
