// nxr::signal::kernelGains on synthetic lines (a frequency line with a negative point and DC, an eigenvalue line from 0):
//   the families gaussian (linear, log2, proportional width), box and cos2_partition equal, bit for bit, the evaluation the flow nodes used before (kept here
//   as the reference); the bank families equal the banks' own functions; every coordinate <= 0 has gain 0 (the cos2 partition: its value at -infinity);
//   the centres and full widths reported; the refusals name their cause.
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/selection/selection.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/kernel_gains.h"
#include "nxr/signal/spectral_flow.h"
#include "nxr/signal/timefilterbank.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

namespace {
namespace sel = nxr::selection;
namespace ns = nxr::signal;

struct Member { double start, stop; };

sel::Selection kernelSet(const std::string& family, const std::string& chart, const std::vector<Member>& members, std::map<std::string, double> params, double gain = 1.0) {
  std::vector<sel::Selection> listed;
  for (std::size_t i = 0; i < members.size(); ++i) {
    sel::Selection m = sel::span({{0, members[i].start, members[i].stop}}, static_cast<std::int64_t>(i));
    m.profile = sel::Profile{family, params, chart, gain};
    listed.push_back(m);
  }
  return sel::set(listed);
}
std::vector<Member> at(const std::vector<double>& centres) {
  std::vector<Member> m;
  for (double c : centres) m.push_back({c, c});
  return m;
}

// THE REFERENCE: the flow nodes' evaluation before kernelGains (pipelines/src/sensor_flow_steps.cpp gains_of, the same arithmetic in the same order).
Eigen::MatrixXd referenceGains(const std::string& family, const std::string& chart, const std::vector<double>& centres, double width, bool proportional, double gain,
                               const Eigen::VectorXd& x) {
  const Eigen::Index M = static_cast<Eigen::Index>(centres.size());
  if (family == "cos2_partition") {
    std::vector<double> log2Centres;
    for (double c : centres) log2Centres.push_back(std::log2(c));
    Eigen::VectorXd lx(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) lx(i) = x(i) > 0 ? std::log2(x(i)) : -std::numeric_limits<double>::infinity();
    return ns::cosineSquaredPartition(lx, log2Centres);
  }
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(M, x.size());
  for (Eigen::Index m = 0; m < M; ++m) {
    const double c = centres[static_cast<std::size_t>(m)];
    double fwhm = width;
    if (proportional) fwhm *= c;
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      if (!(x(i) > 0)) continue;
      const double u = chart == "log2" ? std::log2(x(i)) - std::log2(c) : x(i) - c;
      if (family == "gaussian") { const double sd = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0))); g(m, i) = gain * std::exp(-u * u / (2.0 * sd * sd)); }
      else g(m, i) = std::abs(u) <= fwhm / 2.0 ? gain : 0.0;
    }
  }
  return g;
}

bool refuses(const std::function<void()>& f, const std::string& word) {
  try { f(); } catch (const std::invalid_argument& e) { return std::string(e.what()).find(word) != std::string::npos; }
  return false;
}
}  // namespace

int main() {
  std::cout << "test_kernel_gains\n";
  const auto zero = ns::NonPositiveCoordinates::zero;
  Eigen::VectorXd frequency(122);
  frequency(0) = -1.0;
  for (Eigen::Index i = 1; i < frequency.size(); ++i) frequency(i) = 0.5 * static_cast<double>(i - 1);   // 0, 0.5, ..., 60 Hz
  Eigen::VectorXd mu(201);
  for (Eigen::Index i = 0; i < mu.size(); ++i) mu(i) = 0.01 * static_cast<double>(i);                     // 0 .. 2

  const std::vector<double> centres = {2.0, 4.0, 8.0, 16.0, 32.0};
  {  // matches_old_gains_of
    struct Case { std::string family, chart; std::string widthKey; double width; bool proportional; double gain; };
    for (const Case& c : std::vector<Case>{{"gaussian", "log2", "fwhm", 0.5, false, 1.0}, {"gaussian", "linear", "fwhm", 1.5, false, 2.0},
                                            {"gaussian", "linear", "fwhm_per_centre", 0.25, true, 1.0}, {"box", "log2", "fwhm", 0.5, false, 1.0},
                                            {"box", "linear", "fwhm", 3.0, false, 0.5}}) {
      const auto got = ns::kernelGains(kernelSet(c.family, c.chart, at(centres), {{c.widthKey, c.width}}, c.gain), frequency, zero);
      const Eigen::MatrixXd want = referenceGains(c.family, c.chart, centres, c.width, c.proportional, c.gain, frequency);
      EXPECT(got.gains.rows() == want.rows() && got.gains.cols() == want.cols() && (got.gains.array() == want.array()).all(),
             "matches_old_gains_of: " + c.family + " " + c.chart + " " + c.widthKey + " bit for bit");
    }
    const auto partition = ns::kernelGains(kernelSet("cos2_partition", "log2", at(centres), {}), frequency, zero);
    EXPECT((partition.gains.array() == referenceGains("cos2_partition", "log2", centres, 0, false, 1.0, frequency).array()).all(),
           "matches_old_gains_of: cos2_partition bit for bit");
  }
  {  // bank_families
    const std::vector<Member> bands = {{4.0, 8.0}, {8.0, 12.0}, {13.0, 30.0}};
    const auto morlet = ns::kernelGains(kernelSet("morlet_band", "linear", bands, {}, 2.0), frequency, zero);
    const auto meyer = ns::kernelGains(kernelSet("meyer_band", "log2", bands, {{"half_width", 1.0 / 3.0}}, 2.0), frequency, zero);
    bool same = true;
    for (std::size_t b = 0; b < bands.size(); ++b)
      for (Eigen::Index i = 0; i < frequency.size(); ++i) {
        const double f = frequency(i);
        const double wantMorlet = f > 0 ? 2.0 * ns::timefilterbank::morletBand(f, bands[b].start, bands[b].stop) : 0.0;
        const double wantMeyer = f > 0 ? 2.0 * ns::timefilterbank::meyerBand(f, bands[b].start, bands[b].stop, 1.0 / 3.0) : 0.0;
        same = same && morlet.gains(static_cast<Eigen::Index>(b), i) == wantMorlet && meyer.gains(static_cast<Eigen::Index>(b), i) == wantMeyer;
      }
    EXPECT(same, "bank_families: morlet_band and meyer_band are the time bank's own band shapes times the gain");
    EXPECT(morlet.centres(1) == 10.0, "bank_families: a band's centre is the midpoint of its span");
    const auto hat = ns::kernelGains(kernelSet("mexican_hat", "linear", at({0.2, 0.8}), {}), mu, zero);
    bool hatSame = true;
    for (Eigen::Index i = 0; i < mu.size(); ++i)
      for (Eigen::Index m = 0; m < 2; ++m) hatSame = hatSame && hat.gains(m, i) == (mu(i) > 0 ? ns::graphfilterbank::mexicanHat(mu(i), m == 0 ? 0.2 : 0.8) : 0.0);
    EXPECT(hatSame, "bank_families: mexican_hat is the graph bank's own hat");
  }
  {  // zero_and_negative_coordinates
    bool zeroAtEdge = true;
    for (const auto& family : {std::string("gaussian"), std::string("box")}) {
      const auto g = ns::kernelGains(kernelSet(family, "linear", at({0.5, 1.0}), {{"fwhm", 4.0}}), frequency, zero).gains;
      zeroAtEdge = zeroAtEdge && (g.col(0).array() == 0.0).all() && (g.col(1).array() == 0.0).all();
    }
    EXPECT(zeroAtEdge, "zero_and_negative_coordinates: gaussian and box give 0 at -1 Hz and at DC, even when wide enough to reach them");
    const auto partition = ns::kernelGains(kernelSet("cos2_partition", "log2", at(centres), {}), frequency, zero).gains;
    Eigen::VectorXd minusInfinity(1);
    minusInfinity(0) = -std::numeric_limits<double>::infinity();
    std::vector<double> log2Centres;
    for (double c : centres) log2Centres.push_back(std::log2(c));
    const Eigen::MatrixXd atMinusInfinity = ns::cosineSquaredPartition(minusInfinity, log2Centres);
    EXPECT((partition.col(1).array() == atMinusInfinity.col(0).array()).all(), "zero_and_negative_coordinates: cos2_partition at DC is its value at -infinity (the first member flat)");
  }
  {  // proportional_width
    const auto g = ns::kernelGains(kernelSet("gaussian", "linear", at(centres), {{"fwhm_per_centre", 0.25}}), frequency, zero);
    bool widths = true;
    for (Eigen::Index m = 0; m < g.fullWidths.size(); ++m) widths = widths && g.fullWidths(m) == 0.25 * centres[static_cast<std::size_t>(m)];
    EXPECT(widths && g.centres(2) == 8.0, "proportional_width: the full width is fwhm_per_centre times the centre");
    const auto partition = ns::kernelGains(kernelSet("cos2_partition", "log2", at(centres), {}), frequency, zero);
    EXPECT(std::isnan(partition.fullWidths(0)), "proportional_width: a partition reports no width");
  }
  {  // refusals
    sel::Selection mixed = kernelSet("gaussian", "linear", at({2.0, 4.0}), {{"fwhm", 1.0}});
    mixed.members[1].profile->family = "box";
    EXPECT(refuses([&] { ns::kernelGains(mixed, frequency, zero); }, "one family"), "refusals: members of different families");
    EXPECT(refuses([&] { ns::kernelGains(kernelSet("gaussian", "linear", at({2.0}), {{"fwhm", 1.0}, {"fwhm_per_centre", 0.1}}), frequency, zero); }, "fwhm"),
           "refusals: both fwhm and fwhm_per_centre");
    EXPECT(refuses([&] { ns::kernelGains(kernelSet("gaussian", "linear", at({2.0}), {}), frequency, zero); }, "fwhm"), "refusals: neither fwhm nor fwhm_per_centre");
    EXPECT(refuses([&] { ns::kernelGains(kernelSet("cos2_partition", "linear", at(centres), {}), frequency, zero); }, "log2"), "refusals: cos2_partition in a linear chart");
    EXPECT(refuses([&] { ns::kernelGains(kernelSet("meyer_band", "log2", {{8.0, 12.0}}, {}), frequency, zero); }, "half_width"), "refusals: meyer_band without half_width");
    EXPECT(refuses([&] { ns::kernelGains(kernelSet("gaussian", "linear", {{8.0, 12.0}}, {{"fwhm", 1.0}}), frequency, zero); }, "centre"),
           "refusals: a span anchor for a family anchored at its centre");
    EXPECT(refuses([&] { ns::kernelGains(kernelSet("heat", "linear", at({2.0}), {}), frequency, zero); }, "heat"), "refusals: an unknown family, named");
    EXPECT(refuses([&] { ns::kernelGains(sel::set({}), frequency, zero); }, "no member"), "refusals: a set with no member");
  }
  std::cout << (failures ? "FAILED" : "ALL PASSED") << " (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
