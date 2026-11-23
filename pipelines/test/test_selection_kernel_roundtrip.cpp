// KERNEL SELECTIONS stored by io (synthetic only): a set of two gaussian log2 kernel members and one box member on a frequency line survives save -> load unchanged; resolved on a line of
// fs = 600 Hz, n = 32768 (bins k fs / n) the gaussian members' weights peak at the bin nearest their centre with the stated FWHM (in octaves), the box member holds the bins of the equivalent span;
// a single kernel round-trips as a row of type `kernel`; the frequency line has a `log2` chart; a missing parameter, or the unstated edge-bin rule, is refused with its name.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/selection_store.h"
#include "nxr/selection/kernel.h"
#include "nxr/selection/resolved.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace sel = nxr::selection;

namespace {
template <class F> std::string what_of(F f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }

sel::Selection gaussian(double centre, double fwhm_octaves, double gain, std::int64_t code) {
  sel::Selection m = sel::span({sel::Extent{0, centre, centre}}, std::nullopt, "Hz");
  (void)code;
  sel::Profile p;
  p.family = "gaussian"; p.params["fwhm"] = fwhm_octaves; p.chart = "log2"; p.gain = gain;
  m.profile = p;
  return m;
}

bool same(const sel::Selection& a, const sel::Selection& b) {
  if (a.type != b.type || a.unit != b.unit || a.extents.size() != b.extents.size() || a.profile.has_value() != b.profile.has_value()) return false;
  for (std::size_t i = 0; i < a.extents.size(); ++i)
    if (a.extents[i].axis != b.extents[i].axis || a.extents[i].start != b.extents[i].start || a.extents[i].stop != b.extents[i].stop) return false;
  if (a.profile) return a.profile->family == b.profile->family && a.profile->chart == b.profile->chart && a.profile->gain == b.profile->gain && a.profile->params == b.profile->params;
  return true;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_kernel_roundtrip_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  const double fs_hz = 600.0;
  const std::int64_t n = 32768, bins = n / 2 + 1;
  const double df = fs_hz / static_cast<double>(n);
  std::vector<double> line(static_cast<std::size_t>(bins));
  for (std::int64_t k = 0; k < bins; ++k) line[static_cast<std::size_t>(k)] = static_cast<double>(k) * df;

  DatasetWriter ds = create_dataset(root, "demo");
  SubjectWriter sub = SubjectWriter::create(ds, "S1");
  Database& db = sub.db();
  std::string line_id, set_id, single_id;
  sub.compose([&] {
    ManifoldSpec t;
    t.name = "frequency"; t.path = "frequency/line"; t.type = "line"; t.n_vertices = bins;
    t.topology = {"lattice", 1, 1, std::nullopt, {}};
    GeometrySpec g; g.form = "closed_form"; g.cs_key = "frequency:line";
    DimensionSpec d; d.name = "frequency"; d.n_vertices = bins; d.spacing = df; d.origin = 0.0; d.unit = "Hz";
    g.dimensions.push_back(d); t.geometry = g;
    t.cols.set("n_edges", bins - 1);
    line_id = sub.manifold(t);

    sel::Selection box = sel::span({sel::Extent{0, 10.0, 20.0}}, std::nullopt, "Hz");
    sel::Selection s = sel::set({gaussian(10.0, 1.0, 1.0, 0), box, gaussian(40.0, 0.5, 2.0, 1)});
    selection::Placement at;
    at.name = "kernels"; at.manifold_id = line_id;
    set_id = selection::save(sub, s, at).id;

    selection::Placement one;
    one.name = "one kernel"; one.manifold_id = line_id;
    single_id = selection::save(sub, gaussian(25.0, 1.0, 1.0, 0), one).id;
  });

  // the rows: the kernel members are type `kernel` (family in weight_fn, anchor extent, params_json); the box member is a plain span
  CHECK_EQ(integer(*db.one("SELECT count(*) AS c FROM selection WHERE type = 'kernel'"), "c"), std::int64_t(3));
  const auto krow = *db.one("SELECT weight_fn, params_json FROM selection WHERE id = ?", {V(single_id)});
  CHECK_EQ(text(krow, "weight_fn"), std::string("gaussian"));
  CHECK(text(krow, "params_json").find("\"chart\":\"log2\"") != std::string::npos);
  // the frequency line has a log2 chart beside linear and log10
  const std::string subject_id = text(*db.one("SELECT id FROM subject"), "id");
  sub.compose([&] { coordinate_system(db, subject_id, "frequency:log2"); });
  CHECK(db.one("SELECT 1 FROM coordinate_system WHERE name = 'log2' AND unit = 'log2 Hz'").has_value());

  // save -> load
  const auto loaded = selection::load(db, sub.store(), set_id).description;
  CHECK_EQ(loaded.members.size(), std::size_t(3));
  const sel::Selection expect[3] = {gaussian(10.0, 1.0, 1.0, 0), sel::span({sel::Extent{0, 10.0, 20.0}}, std::nullopt, "Hz"), gaussian(40.0, 0.5, 2.0, 1)};
  // a set's members come back in their stored order; match each
  for (const auto& e : expect) {
    bool found = false;
    for (const auto& m : loaded.members) found = found || same(e, m);
    CHECK(found);
  }
  CHECK(same(selection::load(db, sub.store(), single_id).description, gaussian(25.0, 1.0, 1.0, 0)));

  // resolution
  sel::WeightOptions drop;
  drop.edge_bins = sel::EdgeBins::drop;
  drop.nyquist_bin = bins - 1;
  sel::Selection ordered = sel::set({gaussian(10.0, 1.0, 1.0, 0), sel::span({sel::Extent{0, 10.0, 20.0}}, std::nullopt, "Hz"), gaussian(40.0, 0.5, 2.0, 1)});
  const sel::WeightedElements w = sel::resolve_weights(ordered, line, drop);
  CHECK_EQ(w.members(), std::int64_t(3));
  const double sigma_factor = 2.0 * std::sqrt(2.0 * std::log(2.0));
  const double centres[2] = {10.0, 40.0}, fwhm[2] = {1.0, 0.5}, gain[2] = {1.0, 2.0};
  const int which[2] = {0, 2};
  for (int q = 0; q < 2; ++q) {
    const std::int64_t a = w.offsets[which[q]], b = w.offsets[which[q] + 1];
    CHECK(b > a);
    std::int64_t arg = a;
    for (std::int64_t i = a; i < b; ++i) if (w.weights[i] > w.weights[arg]) arg = i;
    CHECK_EQ(w.elements[arg], static_cast<std::int64_t>(std::llround(centres[q] / df)));
    CHECK(std::abs(w.weights[arg] - gain[q]) < 1e-3 * gain[q]);
    CHECK(w.elements[a] > 0);   // DC has no log chart value: weight 0, not listed
    // the half-maximum bins span fwhm octaves
    std::int64_t lo = -1, hi = -1;
    for (std::int64_t i = a; i < b; ++i) if (w.weights[i] >= 0.5 * gain[q]) { if (lo < 0) lo = w.elements[i]; hi = w.elements[i]; }
    CHECK(std::abs(std::log2(static_cast<double>(hi) / static_cast<double>(lo)) - fwhm[q]) < 0.01);
    // g at one sigma away in octaves: exp(-1/2)
    const double f1 = centres[q] * std::exp2(fwhm[q] / sigma_factor);
    const std::int64_t k1 = std::llround(f1 / df);
    for (std::int64_t i = a; i < b; ++i)
      if (w.elements[i] == k1) CHECK(std::abs(w.weights[i] - gain[q] * std::exp(-0.5)) < 5e-3 * gain[q]);
  }
  // the box member: the bins the equivalent span selects (resolved.h), weight 1
  sel::Line ln; ln.unit = "Hz"; ln.count = bins; ln.origin = 0.0; ln.spacing = df;
  const sel::Intervals iv = sel::resolve_intervals(sel::span({sel::Extent{0, 10.0, 20.0}}, std::nullopt, "Hz"), ln);
  std::vector<std::int64_t> expected_bins;
  for (std::int64_t i = iv.start[0]; i < iv.stop[0]; ++i) expected_bins.push_back(i);
  const std::vector<std::int64_t> got(w.elements.begin() + w.offsets[1], w.elements.begin() + w.offsets[2]);
  CHECK(got == expected_bins);
  CHECK(std::all_of(w.weights.begin() + w.offsets[1], w.weights.begin() + w.offsets[2], [](double g) { return g == 1.0; }));
  // the quadratic field takes g^2
  CHECK(std::abs(sel::quadratic(w).weights[w.offsets[1]] - 1.0) < 1e-15 && std::abs(sel::quadratic(w).weights[w.offsets[2] + 0] - w.weights[w.offsets[2]] * w.weights[w.offsets[2]]) < 1e-15);
  // edge bins: `single` keeps the Nyquist bin of a box that reaches it, `drop` leaves it out
  sel::Selection top = sel::span({sel::Extent{0, 290.0, 300.0}}, std::nullopt, "Hz");
  sel::WeightOptions single = drop;
  single.edge_bins = sel::EdgeBins::single;
  CHECK_EQ(sel::resolve_weights(top, line, single).elements.back(), bins - 1);
  CHECK(sel::resolve_weights(top, line, drop).elements.back() < bins - 1);

  // refusals name what is missing
  sel::Selection bad = gaussian(10.0, 1.0, 1.0, 0);
  bad.profile->params.clear();
  CHECK(what_of([&] { sel::resolve_weights(bad, line, drop); }).find("fwhm") != std::string::npos);
  CHECK(what_of([&] { sel::resolve_weights(ordered, line, sel::WeightOptions{}); }).find("edge_bins") != std::string::npos);
  sel::Selection meyer = gaussian(10.0, 1.0, 1.0, 0);
  meyer.profile->family = "meyer";
  CHECK(what_of([&] { sel::resolve_weights(meyer, line, drop); }).find("meyer") != std::string::npos);
  return nxrtest::finish("test_selection_kernel_roundtrip");
}
