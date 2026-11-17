// nxr::selection (compute) — lookups (lookup.h) and labelled spans (spans.h).
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

#include "nxr/selection/lookup.h"
#include "nxr/selection/spans.h"

namespace nxr::selection {

const Entry& entry(const Lookup& lookup, std::int64_t code) {
  for (const Entry& e : lookup) if (e.code == code) return e;
  throw std::invalid_argument("lookup: no entry for code " + std::to_string(code));
}

void check(const Lookup& lookup) {
  std::set<std::int64_t> codes;
  for (const Entry& e : lookup) {
    if (e.name.empty()) throw std::invalid_argument("lookup: code " + std::to_string(e.code) + " has no name");
    if (!codes.insert(e.code).second) throw std::invalid_argument("lookup: code " + std::to_string(e.code) + " appears twice");
  }
}

Spans spans(std::string on, std::string unit, std::vector<Span> list, Lookup lookup) {
  check(lookup);
  for (const Span& s : list) {
    if (!std::isfinite(s.start) || !std::isfinite(s.stop) || !(s.stop > s.start))
      throw std::invalid_argument("spans: [" + std::to_string(s.start) + ", " + std::to_string(s.stop) + ") is not an interval");
    if (!lookup.empty()) entry(lookup, s.code);                       // throws for a code the lookup lacks
  }
  std::stable_sort(list.begin(), list.end(), [](const Span& a, const Span& b) { return a.start < b.start || (a.start == b.start && a.stop < b.stop); });
  return Spans{std::move(on), std::move(unit), std::move(list), std::move(lookup)};
}

std::vector<std::size_t> spans_at(const Spans& s, double value) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < s.spans.size() && s.spans[i].start <= value; ++i)
    if (value < s.spans[i].stop) out.push_back(i);
  return out;
}

}  // namespace nxr::selection
