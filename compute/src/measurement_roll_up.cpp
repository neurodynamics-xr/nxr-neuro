// nxr::measurement — `roll_up` (measurement/roll_up.h).
#include "nxr/measurement/roll_up.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::measurement {

namespace {
using selection::Elements;
constexpr double kInf = std::numeric_limits<double>::infinity();

[[noreturn]] void refuse(const std::string& what) { throw std::invalid_argument("roll_up: " + what); }

// groups of `n` fine members by a shift: member m goes to m >> shift (shift < 0: one group)
Elements shifted(std::int64_t n, int shift) {
  Elements out;
  const std::int64_t groups = shift < 0 ? 1 : ((n + (std::int64_t(1) << shift) - 1) >> shift);
  out.size = n;
  out.offsets.assign(static_cast<std::size_t>(groups) + 1, 0);
  for (std::int64_t m = 0; m < n; ++m) ++out.offsets[static_cast<std::size_t>(shift < 0 ? 0 : (m >> shift)) + 1];
  for (std::size_t g = 0; g < static_cast<std::size_t>(groups); ++g) out.offsets[g + 1] += out.offsets[g];
  out.elements.resize(static_cast<std::size_t>(n));
  std::vector<std::int64_t> at(out.offsets.begin(), out.offsets.end() - 1);
  for (std::int64_t m = 0; m < n; ++m) out.elements[static_cast<std::size_t>(at[static_cast<std::size_t>(shift < 0 ? 0 : (m >> shift))]++)] = m;
  return out;
}

bool is_dyadic(const selection::Selection& g) { return g.grid && g.grid->kind == selection::Grid::Kind::dyadic; }

// the shift a grid's roll-up from level `a` to level `b` is, and the fine member count (b == kWhole: shift -1)
struct Move { std::int64_t fine; int shift; };
Move move_of(const selection::Selection& grid, int a, int b) {
  if (auto why = selection::rollup_refusal(grid)) refuse(*why);
  if (is_dyadic(grid)) {
    if (a < 0 || a > grid.grid->depth) refuse("level " + std::to_string(a) + " is outside the dyadic grid's levels 0 .. " + std::to_string(grid.grid->depth));
    if (b != kWhole && (b < 0 || b >= a)) refuse("a dyadic grid rolls up to a coarser level: " + std::to_string(b) + " is not below " + std::to_string(a) + " (or " + std::to_string(kWhole) + ", the whole)");
    return Move{grid.grid->slots << a, b == kWhole ? -1 : a - b};
  }
  if (grid.tiling) {
    if (b != kWhole && b < a) refuse("a time grid rolls up to a coarser level: " + std::to_string(b) + " is below " + std::to_string(a) + " (tiles of 2^k samples, k larger is coarser)");
    return Move{grid.tiling->count, b == kWhole ? -1 : b - a};
  }
  refuse("only a dyadic surface grid or a dyadic time tiling rolls up by shifts");
}

int whole_shift(const selection::Selection& g) {
  // the shift that sends every member of the factor to member 0
  std::int64_t n = is_dyadic(g) ? (g.grid->slots << g.grid->depth) : (g.tiling ? g.tiling->count : 1);
  int s = 0;
  while ((std::int64_t(1) << s) < n) ++s;
  return s;
}
}  // namespace

Elements coarser_groups(const selection::Selection& grid, int from_level, int to_level) {
  const Move m = move_of(grid, from_level, to_level);
  return shifted(m.fine, m.shift);
}

Elements coarser_groups(const selection::Selection& product, std::size_t factor, int from_level, int to_level) {
  if (product.factors.empty() || factor >= product.factors.size()) refuse("a factor outside the product (" + std::to_string(product.factors.size()) + " factors)");
  if (auto why = selection::rollup_refusal(product, static_cast<int>(factor))) refuse(*why);
  const selection::Selection& f = product.factors[factor];
  int shift;
  if (to_level == kWhole) shift = whole_shift(f);
  else shift = move_of(f, from_level, to_level).shift;
  const std::int64_t fine = selection::count(product);
  std::vector<std::int64_t> radices = selection::rolled_radices(product, static_cast<int>(factor), shift);
  std::int64_t coarse = 1;
  for (std::int64_t r : radices) coarse *= r;
  Elements out;
  out.size = fine;
  std::vector<std::int64_t> to(static_cast<std::size_t>(fine)), n(static_cast<std::size_t>(coarse), 0);
  for (std::int64_t c = 0; c < fine; ++c) {
    to[static_cast<std::size_t>(c)] = selection::coarser_code(product, c, static_cast<int>(factor), shift);
    ++n[static_cast<std::size_t>(to[static_cast<std::size_t>(c)])];
  }
  out.offsets.assign(static_cast<std::size_t>(coarse) + 1, 0);
  for (std::int64_t g = 0; g < coarse; ++g) out.offsets[static_cast<std::size_t>(g) + 1] = out.offsets[static_cast<std::size_t>(g)] + n[static_cast<std::size_t>(g)];
  out.elements.resize(static_cast<std::size_t>(fine));
  std::vector<std::int64_t> at(out.offsets.begin(), out.offsets.end() - 1);
  for (std::int64_t c = 0; c < fine; ++c) out.elements[static_cast<std::size_t>(at[static_cast<std::size_t>(to[static_cast<std::size_t>(c)])]++)] = c;
  return out;
}

namespace {
void check_rollable(const Request& request) {
  const Registry& reg = Registry::standard();
  for (const std::string& name : request.measures) {
    const MeasureSpec* s = reg.find(name);
    if (!s) continue;                                                     // a measure of the caller's own registry
    if (s->merge == Merge::none && s->derived_from.empty()) refuse("'" + name + "' is never rolled up (merge none)");
  }
}

void merge_into(Ingredients& out, std::size_t o, const Ingredients& in, std::size_t i, std::int64_t components) {
  out.count[o] += in.count[i]; out.nonfinite[o] += in.nonfinite[i];
  out.weight[o] += in.weight[i]; out.sum_sq[o] += in.sum_sq[i]; out.sum_cube[o] += in.sum_cube[i]; out.sum_fourth[o] += in.sum_fourth[i];
  out.sum_abs[o] += in.sum_abs[i]; out.sum_sqrt_abs[o] += in.sum_sqrt_abs[i];
  if (in.max[i] > out.max[o]) out.max[o] = in.max[i];
  if (in.min[i] < out.min[o]) out.min[o] = in.min[i];
  if (in.abs_max[i] > out.abs_max[o]) out.abs_max[o] = in.abs_max[i];
  for (std::int64_t c = 0; c < components; ++c) out.sum[o * static_cast<std::size_t>(components) + static_cast<std::size_t>(c)] += in.sum[i * static_cast<std::size_t>(components) + static_cast<std::size_t>(c)];
}
}  // namespace

Ingredients merge(const Ingredients& in, const Elements& groups, const Request& request) {
  check_rollable(request);
  if (!in.held.empty()) refuse("ingredients that hold values for a median cannot be rolled up (a median is never rolled up)");
  Ingredients out(in.rows, groups.members(), in.components, false);
  out.unassigned = in.unassigned;
  for (std::int64_t g = 0; g < groups.members(); ++g)
    for (std::int64_t j = groups.offsets[static_cast<std::size_t>(g)]; j < groups.offsets[static_cast<std::size_t>(g) + 1]; ++j) {
      const std::int64_t m = groups.elements[static_cast<std::size_t>(j)];
      if (m < 0 || m >= in.members) refuse("group " + std::to_string(g) + " holds member " + std::to_string(m) + ", outside the " + std::to_string(in.members) + " fine members");
      for (std::int64_t r = 0; r < in.rows; ++r) merge_into(out, out.at(r, g), in, in.at(r, m), in.components);
    }
  return out;
}

Ingredients roll_up(const Ingredients& fine, const Elements& groups, const Request& request) { return merge(fine, groups, request); }

Ingredients across(const std::vector<const Ingredients*>& subjects, const Request& request) {
  check_rollable(request);
  if (subjects.empty() || !subjects.front()) refuse("across needs at least one subject's ingredients");
  const Ingredients& a = *subjects.front();
  Ingredients out(a.rows, a.members, a.components, false);
  for (const Ingredients* s : subjects) {
    if (!s) refuse("a subject's ingredients are missing");
    if (s->rows != a.rows || s->members != a.members || s->components != a.components)
      refuse("the subjects' ingredients differ in shape (" + std::to_string(s->rows) + " x " + std::to_string(s->members) + " against " + std::to_string(a.rows) + " x " + std::to_string(a.members) + "): they must lie on the same codes");
    if (!s->held.empty()) refuse("ingredients that hold values for a median cannot be rolled up across subjects");
    for (std::int64_t r = 0; r < a.rows; ++r)
      for (std::int64_t m = 0; m < a.members; ++m) merge_into(out, out.at(r, m), *s, s->at(r, m), a.components);
    out.unassigned += s->unassigned;
  }
  return out;
}

MatrixIngredients merge(const MatrixIngredients& in, const Elements& groups) {
  MatrixIngredients out(groups.members());
  for (std::int64_t g = 0; g < groups.members(); ++g)
    for (std::int64_t j = groups.offsets[static_cast<std::size_t>(g)]; j < groups.offsets[static_cast<std::size_t>(g) + 1]; ++j) {
      const std::int64_t m = groups.elements[static_cast<std::size_t>(j)];
      if (m < 0 || m >= in.members()) refuse("group " + std::to_string(g) + " holds member " + std::to_string(m) + ", outside the " + std::to_string(in.members()) + " fine members");
      if (in.count[static_cast<std::size_t>(m)] == 0) continue;                    // a member with no value adds nothing; a group of none stays empty (0 x 0)
      const auto o = static_cast<std::size_t>(g);
      if (out.count[o] == 0) out.sum[o] = in.sum[static_cast<std::size_t>(m)];
      else {
        if (out.sum[o].rows() != in.sum[static_cast<std::size_t>(m)].rows()) refuse("the values of a group have different shapes");
        out.sum[o] += in.sum[static_cast<std::size_t>(m)];
      }
      out.count[o] += in.count[static_cast<std::size_t>(m)];
    }
  return out;
}

Eigen::MatrixXd roll_values(const Registry& registry, const std::string& measure, const Eigen::MatrixXd& values, const Eigen::MatrixXd* weights, const Elements& groups) {
  const MeasureSpec& s = registry.at(measure);
  if (!s.derived_from.empty()) refuse("'" + measure + "' is derived: roll up its ingredients and derive it at read");
  if (s.merge == Merge::none) refuse("'" + measure + "' is never rolled up (merge none)");
  const bool mean = s.merge == Merge::mean;
  if (mean && !weights) refuse("'" + measure + "' is a weighted mean: the values of its weight measure ('" + s.weight + "') are needed");
  if (mean && weights && ((weights->rows() != values.rows() && weights->rows() != 1) || weights->cols() != values.cols())) refuse("the weights must be rows x members (or 1 x members) like the values");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd out(values.rows(), groups.members());
  for (Eigen::Index r = 0; r < values.rows(); ++r)
    for (std::int64_t g = 0; g < groups.members(); ++g) {
      double acc = s.merge == Merge::max ? -kInf : (s.merge == Merge::min ? kInf : 0.0), w = 0;
      bool bad = false, any = false;
      for (std::int64_t j = groups.offsets[static_cast<std::size_t>(g)]; j < groups.offsets[static_cast<std::size_t>(g) + 1]; ++j) {
        const std::int64_t m = groups.elements[static_cast<std::size_t>(j)];
        if (m < 0 || m >= values.cols()) refuse("group " + std::to_string(g) + " holds member " + std::to_string(m) + ", outside the " + std::to_string(values.cols()) + " fine members");
        const double v = values(r, m);
        if (std::isnan(v)) { bad = true; break; }
        any = true;
        switch (s.merge) {
          case Merge::sum: acc += v; break;
          case Merge::max: acc = std::max(acc, v); break;
          case Merge::min: acc = std::min(acc, v); break;
          case Merge::mean: { const double wm = (*weights)(weights->rows() == 1 ? 0 : r, m); acc += v * wm; w += wm; break; }
          case Merge::none: break;
        }
      }
      if (bad) out(r, g) = nan;
      else if (mean) out(r, g) = any && w > 0 ? acc / w : nan;
      else out(r, g) = acc;
    }
  return out;
}

}  // namespace nxr::measurement
