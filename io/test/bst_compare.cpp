// nxr-io-bst-compare A.mat B.mat — reads every variable of both files through nxr::bst and requires them equal, class by class, value by
// value (the v7.3 smoke test: a file saved with -v7.3 must read as its -v7 twin). Exit 0 and one line when equal; exit 1 naming the first
// difference. `--only NAME,NAME` limits it to those variables.
#include <cstring>
#include <iostream>
#include <sstream>

#include "nxr/bst/mat.h"

using namespace nxr::bst;

namespace {
std::string diff(const Value& a, const Value& b, const std::string& where) {
  auto at = [&](const std::string& what) { return where + ": " + what; };
  if (a.cls != b.cls) return at(std::string("class ") + class_name(a.cls) + " vs " + class_name(b.cls));
  if (a.dims != b.dims) return at("dims differ");
  if (a.is_complex != b.is_complex) return at("complexity differs");
  if (a.is_logical != b.is_logical) return at("logical flag differs");
  if (a.re != b.re) return at("values differ");
  if (a.im != b.im) return at("imaginary values differ");
  if (a.ir != b.ir || a.jc != b.jc) return at("sparse structure differs");
  if (a.text != b.text) return at("text differs");
  if (a.fields != b.fields) return at("field names differ");
  if (a.elems.size() != b.elems.size()) return at("element count differs");
  for (std::size_t i = 0; i < a.elems.size(); ++i) {
    std::string w = where;
    if (a.cls == Class::Struct) w += "(" + std::to_string(i / a.fields.size() + 1) + ")." + a.fields[i % a.fields.size()];
    else w += "{" + std::to_string(i + 1) + "}";
    if (auto d = diff(a.elems[i], b.elems[i], w); !d.empty()) return d;
  }
  return {};
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> only;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--only") && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      for (std::string n; std::getline(ss, n, ',');) only.push_back(n);
    } else files.push_back(argv[i]);
  }
  if (files.size() != 2) { std::cerr << "usage: nxr-io-bst-compare [--only NAME,..] A.mat B.mat\n"; return 2; }
  try {
    const auto a = only.empty() ? read_all_variables(files[0]) : read_variables(files[0], only);
    const auto b = only.empty() ? read_all_variables(files[1]) : read_variables(files[1], only);
    if (a.size() != b.size()) { std::cout << "variable count " << a.size() << " vs " << b.size() << "\n"; return 1; }
    for (const auto& [name, v] : a) {
      auto it = b.find(name);
      if (it == b.end()) { std::cout << name << ": missing in the second file\n"; return 1; }
      if (auto d = diff(v, it->second, name); !d.empty()) { std::cout << d << "\n"; return 1; }
    }
    std::cout << "equal: " << a.size() << " variables\n";
    return 0;
  } catch (const std::exception& e) { std::cout << "error: " << e.what() << "\n"; return 1; }
}
