// nxr::measurement — `reduce` (measurement/reduce.h). The accumulation of one value into a member's ingredients is the Studio's measureOverPartition, line for line (the same products
// in the same order); compiled with -ffp-contract=off (see CMakeLists.txt).
#include "nxr/measurement/reduce.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::measurement {

namespace {
using selection::Elements;
using selection::MemberOfElement;
using selection::ProductForm;
using selection::Rule;

[[noreturn]] void refuse(const std::string& what) { throw std::invalid_argument("reduce: " + what); }

// one value into one (row, member) cell's ingredients: component 0 carries the whole set, a later component only its sum
struct Folder {
  Ingredients& a;
  bool first, hold;
  std::int64_t comp;
  void value(std::size_t k, double v, double w) {
    a.sum[k * static_cast<std::size_t>(a.components) + static_cast<std::size_t>(comp)] += v * w;
    if (!first) return;
    a.weight[k] += w; ++a.count[k];
    if (!std::isfinite(v)) ++a.nonfinite[k];
    a.sum_sq[k] += v * v * w; a.sum_cube[k] += v * v * v * w; a.sum_fourth[k] += v * v * v * v * w;
    if (v > a.max[k]) a.max[k] = v;
    if (v < a.min[k]) a.min[k] = v;
    const double ab = std::abs(v);
    if (ab > a.abs_max[k]) a.abs_max[k] = ab;
    a.sum_abs[k] += ab * w; a.sum_sqrt_abs[k] += std::sqrt(ab) * w;
    if (hold && std::isfinite(v)) a.held[k].push_back(v);
  }
  void geometry(std::size_t k, double w) {                       // a member's own measures: only the weight and the count
    if (!first) return;
    a.weight[k] += w; ++a.count[k];
  }
};

Folder prepare(Ingredients& acc, const Request& req, std::int64_t rows, std::int64_t members, const double* weights, bool has_values) {
  if (acc.rows != rows || acc.members != members)
    refuse("the ingredients hold " + std::to_string(acc.rows) + " rows x " + std::to_string(acc.members) + " members, the call needs " + std::to_string(rows) + " x " + std::to_string(members));
  if (req.component < 0 || req.component >= acc.components) refuse("component " + std::to_string(req.component) + " of " + std::to_string(acc.components));
  if (req.weighting == Weighting::given && !weights) refuse("Weighting::given needs the weights");
  if (req.weighting == Weighting::none && weights) refuse("weights were handed with Weighting::none: state the weighting");
  const bool wants_median = std::find(req.measures.begin(), req.measures.end(), "median") != req.measures.end();
  if (wants_median && acc.held.empty()) refuse("a median needs the values held (Ingredients(..., hold = true))");
  (void)has_values;
  return Folder{acc, req.component == 0, !acc.held.empty(), req.component};
}

// the members each element belongs to, as CSR (a member number per element: at most one)
struct Memberships {
  std::vector<std::int64_t> offsets, members;
  std::pair<const std::int64_t*, const std::int64_t*> of(std::int64_t e) const { return {members.data() + offsets[static_cast<std::size_t>(e)], members.data() + offsets[static_cast<std::size_t>(e) + 1]}; }
};
Memberships memberships_of(const MemberOfElement& m, std::int64_t& unassigned) {
  Memberships out;
  out.offsets.assign(m.member_of.size() + 1, 0);
  for (std::size_t i = 0; i < m.member_of.size(); ++i) {
    const std::int32_t c = m.member_of[i];
    if (c >= m.members) refuse("element " + std::to_string(i) + " is in member " + std::to_string(c) + ", outside [0, " + std::to_string(m.members) + ")");
    if (c < 0) ++unassigned;
    else out.members.push_back(c);
    out.offsets[i + 1] = static_cast<std::int64_t>(out.members.size());
  }
  return out;
}
Memberships memberships_of(const Elements& m) {
  Memberships out;
  const std::size_t n = static_cast<std::size_t>(m.size);
  std::vector<std::vector<std::int64_t>> per(n);
  for (std::int64_t k = 0; k < m.members(); ++k)
    for (std::int64_t j = m.offsets[static_cast<std::size_t>(k)]; j < m.offsets[static_cast<std::size_t>(k) + 1]; ++j) {
      const std::int64_t e = m.elements[static_cast<std::size_t>(j)];
      if (e < 0 || e >= m.size) refuse("a member holds element " + std::to_string(e) + ", outside [0, " + std::to_string(m.size) + ")");
      per[static_cast<std::size_t>(e)].push_back(k);
    }
  out.offsets.assign(n + 1, 0);
  for (std::size_t e = 0; e < n; ++e) { out.members.insert(out.members.end(), per[e].begin(), per[e].end()); out.offsets[e + 1] = static_cast<std::int64_t>(out.members.size()); }
  return out;
}

// the members holding each sample of a block: a rule's [first, stop) member numbers
void rule_members(const Rule& r, std::int64_t first, std::int64_t n, std::vector<std::int64_t>& m0, std::vector<std::int64_t>& m1) {
  m0.assign(static_cast<std::size_t>(n), 0);
  m1.assign(static_cast<std::size_t>(n), 0);
  for (std::int64_t s = 0; s < n; ++s) {
    const auto [a, b] = selection::rule_members_at(r, first + s);
    m0[static_cast<std::size_t>(s)] = a;
    m1[static_cast<std::size_t>(s)] = b;
  }
}

template <class T>
void over_members(Ingredients& acc, const T* values, std::int64_t rows, std::int64_t elements, const MemberOfElement& mem, const double* weights, const Request& req) {
  if (mem.size() != elements) refuse("the members cover " + std::to_string(mem.size()) + " elements, the values " + std::to_string(elements));
  Folder f = prepare(acc, req, rows, mem.members, weights, values != nullptr);
  std::int64_t unassigned = 0;
  for (std::size_t i = 0; i < mem.member_of.size(); ++i) {                       // a member number past the count is refused; a negative one is in no member, counted once
    const std::int32_t c = mem.member_of[i];
    if (c >= mem.members) refuse("element " + std::to_string(i) + " is in member " + std::to_string(c) + ", outside [0, " + std::to_string(mem.members) + ")");
    if (c < 0) ++unassigned;
  }
  if (f.first) acc.unassigned += unassigned;
  const std::size_t M = static_cast<std::size_t>(mem.members);
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t i = 0; i < elements; ++i) {
      const std::int32_t c = mem.member_of[static_cast<std::size_t>(i)];
      if (c < 0) continue;
      const std::size_t k = static_cast<std::size_t>(r) * M + static_cast<std::size_t>(c);
      const double w = weights ? weights[i] : 1.0;
      if (values) f.value(k, static_cast<double>(values[static_cast<std::size_t>(r) * static_cast<std::size_t>(elements) + static_cast<std::size_t>(i)]), w);
      else f.geometry(k, w);
    }
}

template <class T>
void over_elements(Ingredients& acc, const T* values, std::int64_t rows, std::int64_t elements, const Elements& mem, const double* weights, const Request& req) {
  if (mem.size != elements) refuse("the members lie on " + std::to_string(mem.size) + " elements, the values hold " + std::to_string(elements));
  Folder f = prepare(acc, req, rows, mem.members(), weights, values != nullptr);
  const std::size_t M = static_cast<std::size_t>(mem.members());
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::size_t m = 0; m < M; ++m)
      for (std::int64_t j = mem.offsets[m]; j < mem.offsets[m + 1]; ++j) {
        const std::int64_t e = mem.elements[static_cast<std::size_t>(j)];
        if (e < 0 || e >= elements) refuse("a member holds element " + std::to_string(e) + ", outside [0, " + std::to_string(elements) + ")");
        const std::size_t k = static_cast<std::size_t>(r) * M + m;
        const double w = weights ? weights[e] : 1.0;
        if (values) f.value(k, static_cast<double>(values[static_cast<std::size_t>(r) * static_cast<std::size_t>(elements) + static_cast<std::size_t>(e)]), w);
        else f.geometry(k, w);
      }
}

template <class T>
void over_rule(Ingredients& acc, const T* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const Rule& tiles, const Request& req) {
  if (req.weighting != Weighting::none) refuse("a reduction over a time rule takes no weights (every sample counts once)");
  Folder f = prepare(acc, req, req.per_row ? rows : 1, tiles.count, nullptr, values != nullptr);
  std::vector<std::int64_t> m0, m1;
  rule_members(tiles, first_sample, samples, m0, m1);
  const std::size_t M = static_cast<std::size_t>(tiles.count);
  auto one = [&](std::int64_t r, std::int64_t s) {
    const std::size_t row = req.per_row ? static_cast<std::size_t>(r) : 0;
    for (std::int64_t m = m0[static_cast<std::size_t>(s)]; m < m1[static_cast<std::size_t>(s)]; ++m) {
      const std::size_t k = row * M + static_cast<std::size_t>(m);
      if (values) f.value(k, static_cast<double>(values[static_cast<std::size_t>(r) * static_cast<std::size_t>(samples) + static_cast<std::size_t>(s)]), 1.0);
      else f.geometry(k, 1.0);
    }
  };
  if (req.per_row) { for (std::int64_t r = 0; r < rows; ++r) for (std::int64_t s = 0; s < samples; ++s) one(r, s); }
  else { for (std::int64_t s = 0; s < samples; ++s) for (std::int64_t r = 0; r < rows; ++r) one(r, s); }      // sample first: a cell's order is the sample order whatever the blocks
}

template <class T>
void over_product(Ingredients& acc, const T* values, std::int64_t rows, std::int64_t samples, std::int64_t first_sample, const ProductForm& cells, const double* weights, const Request& req) {
  if (cells.factors.size() != 2) refuse("a product here has two factors, time then space (it has " + std::to_string(cells.factors.size()) + ")");
  Folder f = prepare(acc, req, 1, cells.members(), weights, values != nullptr);
  // factor 0: the members of each sample of the block
  std::vector<std::int64_t> t0(static_cast<std::size_t>(samples)), t1(static_cast<std::size_t>(samples));
  if (const Rule* rule = std::get_if<Rule>(&cells.factors[0])) rule_members(*rule, first_sample, samples, t0, t1);
  else if (const MemberOfElement* mo = std::get_if<MemberOfElement>(&cells.factors[0])) {
    for (std::int64_t s = 0; s < samples; ++s) {
      const std::int64_t g = first_sample + s;
      const std::int32_t c = g >= 0 && g < mo->size() ? mo->member_of[static_cast<std::size_t>(g)] : -1;
      t0[static_cast<std::size_t>(s)] = c < 0 ? 0 : c;
      t1[static_cast<std::size_t>(s)] = c < 0 ? 0 : c + 1;
    }
  } else refuse("factor 0 of a product is a rule or a member number per sample on the line (resolve intervals to member numbers first)");
  // factor 1: the members of each row's element
  std::int64_t unassigned = 0;
  Memberships sp;
  if (const MemberOfElement* mo = std::get_if<MemberOfElement>(&cells.factors[1])) { if (mo->size() != rows) refuse("factor 1 covers " + std::to_string(mo->size()) + " elements, the values hold " + std::to_string(rows) + " rows"); sp = memberships_of(*mo, unassigned); }
  else if (const Elements* el = std::get_if<Elements>(&cells.factors[1])) { if (el->size != rows) refuse("factor 1 lies on " + std::to_string(el->size) + " elements, the values hold " + std::to_string(rows) + " rows"); sp = memberships_of(*el); }
  else refuse("factor 1 of a product is a member number per element or listed elements");
  if (f.first) acc.unassigned += unassigned;
  const std::int64_t radix0 = cells.radices[0], count0 = cells.counts[0];
  for (std::int64_t s = 0; s < samples; ++s)
    for (std::int64_t m_t = t0[static_cast<std::size_t>(s)]; m_t < t1[static_cast<std::size_t>(s)]; ++m_t) {
      if (m_t >= count0) refuse("a sample is in time member " + std::to_string(m_t) + " of " + std::to_string(count0));
      for (std::int64_t r = 0; r < rows; ++r) {
        const auto [b, e] = sp.of(r);
        for (const std::int64_t* m_s = b; m_s != e; ++m_s) {
          const std::size_t k = static_cast<std::size_t>(m_t + radix0 * *m_s);
          const double w = weights ? weights[r] : 1.0;
          if (values) f.value(k, static_cast<double>(values[static_cast<std::size_t>(r) * static_cast<std::size_t>(samples) + static_cast<std::size_t>(s)]), w);
          else f.geometry(k, w);
        }
      }
    }
}

std::int64_t floor_div(std::int64_t a, std::int64_t b) { std::int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q; }
}  // namespace

void reduce(Ingredients& acc, const float* v, std::int64_t rows, std::int64_t elements, const MemberOfElement& m, const double* w, const Request& r) { over_members(acc, v, rows, elements, m, w, r); }
void reduce(Ingredients& acc, const double* v, std::int64_t rows, std::int64_t elements, const MemberOfElement& m, const double* w, const Request& r) { over_members(acc, v, rows, elements, m, w, r); }
void reduce(Ingredients& acc, const float* v, std::int64_t rows, std::int64_t elements, const Elements& m, const double* w, const Request& r) { over_elements(acc, v, rows, elements, m, w, r); }
void reduce(Ingredients& acc, const double* v, std::int64_t rows, std::int64_t elements, const Elements& m, const double* w, const Request& r) { over_elements(acc, v, rows, elements, m, w, r); }
void reduce(Ingredients& acc, const float* v, std::int64_t rows, std::int64_t samples, std::int64_t first, const Rule& t, const Request& r) { over_rule(acc, v, rows, samples, first, t, r); }
void reduce(Ingredients& acc, const double* v, std::int64_t rows, std::int64_t samples, std::int64_t first, const Rule& t, const Request& r) { over_rule(acc, v, rows, samples, first, t, r); }
void reduce(Ingredients& acc, const float* v, std::int64_t rows, std::int64_t samples, std::int64_t first, const ProductForm& c, const double* w, const Request& r) { over_product(acc, v, rows, samples, first, c, w, r); }
void reduce(Ingredients& acc, const double* v, std::int64_t rows, std::int64_t samples, std::int64_t first, const ProductForm& c, const double* w, const Request& r) { over_product(acc, v, rows, samples, first, c, w, r); }

Eigen::MatrixXcd MatrixIngredients::mean(std::int64_t member) const {
  if (member < 0 || member >= members()) refuse("member " + std::to_string(member) + " outside the matrix ingredients");
  const auto k = static_cast<std::size_t>(member);
  if (count[k] == 0) refuse("member " + std::to_string(member) + " holds no value (its mean is undefined)");
  return sum[k] / std::complex<double>(static_cast<double>(count[k]), 0.0);
}

void add(MatrixIngredients& acc, std::int64_t member, const Eigen::MatrixXcd& value) {
  if (member < 0 || member >= acc.members()) refuse("a value is added to member " + std::to_string(member) + " of " + std::to_string(acc.members()));
  const auto k = static_cast<std::size_t>(member);
  if (acc.count[k] == 0) acc.sum[k] = Eigen::MatrixXcd::Zero(value.rows(), value.cols());
  else if (acc.sum[k].rows() != value.rows() || acc.sum[k].cols() != value.cols()) refuse("the values of a member have different shapes");
  acc.sum[k] += value;
  acc.count[k] += 1;
}

void reduce(MatrixIngredients& acc, const Eigen::MatrixXcd& Z, std::int64_t first_sample, const Rule& tiles) {
  if (acc.members() != tiles.count) refuse("the matrix ingredients hold " + std::to_string(acc.members()) + " members, the rule " + std::to_string(tiles.count));
  if (tiles.width < 1 || tiles.hop < 1) refuse("a rule of width and hop >= 1");
  const std::int64_t n = Z.cols(), last = first_sample + n - 1, C = Z.rows();
  if (n == 0) return;
  // the members that touch the block: member k = [origin + k hop, origin + k hop + width), clipped to the line
  const std::int64_t k_hi = std::min<std::int64_t>(tiles.count - 1, floor_div(last - tiles.origin, tiles.hop));
  const std::int64_t k_lo = std::max<std::int64_t>(0, floor_div(first_sample - tiles.origin - tiles.width, tiles.hop) + 1);
  for (std::int64_t k = k_lo; k <= k_hi; ++k) {
    const auto [a, b] = selection::rule_member(tiles, k);
    const std::int64_t from = std::max(a, first_sample), to = std::min(b, last + 1);
    if (to <= from) continue;
    const auto block = Z.middleCols(from - first_sample, to - from);
    const auto idx = static_cast<std::size_t>(k);
    if (acc.count[idx] == 0) acc.sum[idx] = Eigen::MatrixXcd::Zero(C, C);
    else if (acc.sum[idx].rows() != C) refuse("the channels differ between blocks");
    acc.sum[idx].noalias() += block * block.adjoint();               // Σ_t z zᴴ over the block's part of the member: one complex product
    acc.count[idx] += to - from;
  }
}

}  // namespace nxr::measurement
