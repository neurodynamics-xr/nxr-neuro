// nxr-pipelines — the Wishart null (wishart_null.h).
#include "nxr/pipelines/wishart_null.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "nxr/filterbank.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/surrogate.h"

namespace nxr::pipelines {

using nxr::io::Error;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::text;
using nxr::io::V;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(nxr::io::code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(nxr::io::code::no_such_row, m); }
double lagged_share(const Eigen::MatrixXcd& Q) { const double t = Q.squaredNorm(); return t > 0 ? Q.imag().squaredNorm() / t : std::numeric_limits<double>::quiet_NaN(); }
}  // namespace

WishartNullResult wishart_null_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const WishartNullOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.recording.empty() || opt.windows.empty() || opt.bursts.empty() || opt.psd.empty()) invalid("wishart compute: name the recording, its analytic windows (--windows), the bursts and the PSD (--psd)");
  if (opt.band.size() != 2 || !(opt.band[0] >= 0 && opt.band[0] < opt.band[1])) invalid("wishart compute: --band is LO,HI in Hz, LO < HI");
  if (opt.draws < 20) invalid("wishart compute: --draws is at least 20 (the project's 1000)");
  if (opt.shape != "morlet" && opt.shape != "meyer") invalid("wishart compute: --shape is morlet or meyer");
  const auto rec = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(opt.recording)});
  if (!rec) missing(sub.name() + ": no recording (Field) named '" + opt.recording + "'");
  const CrossSpectrumWindows win = crossspectrum_windows_read(sub, opt.windows, std::nullopt, "stored");
  if (win.level < 0 || win.sum.empty()) invalid("wishart compute: " + opt.windows + " is not a stored analytic cross-spectrum series on time tiles (`crossspectrum compute --estimator analytic`)");
  // the bursts and the time line
  const auto brow = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(opt.bursts)});
  if (!brow) missing(sub.name() + ": no Selection named '" + opt.bursts + "' (the bursts)");
  if (text(*brow, "type") != "spans") invalid(opt.bursts + " is a " + text(*brow, "type") + " Selection, not spans");
  std::vector<std::pair<double, double>> spans;
  for (const auto& r : sub.db().all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(*brow, "id"))})) spans.emplace_back(std::stod(text(r, "start")), std::stod(text(r, "stop")));
  if (spans.empty()) invalid("wishart compute: " + opt.bursts + " holds no burst");
  const auto dims = sub.db().all("SELECT d.manifold_id AS mid FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? ORDER BY d.ordinal", {V(text(*rec, "manifold_id"))});
  if (dims.size() != 2) invalid(opt.recording + ": a recording lies on channels x time");
  const std::string tid = text(dims[1], "mid");
  if (text(*brow, "manifold_id") != tid) invalid(opt.bursts + " is on another time line than " + opt.recording + "'s");
  const auto ld = sub.db().one("SELECT d.spacing AS spacing, d.origin AS origin FROM geometry_dimension d JOIN manifold mm ON mm.geometry_id = d.geometry_id WHERE mm.id = ? AND d.ordinal = 0", {V(tid)});
  if (!ld || nxr::io::is_null(*ld, "spacing")) invalid(opt.recording + ": its time line has no sampling step");
  const double dt = std::stod(text(*ld, "spacing")), t0 = nxr::io::is_null(*ld, "origin") ? 0.0 : std::stod(text(*ld, "origin")), rate = 1.0 / dt;
  const double width = static_cast<double>(std::int64_t(1) << win.level) * dt, eps = 0.5 * dt;
  // the bandwidth: the PSD's mean over the windows' channels through the band's analytic gain
  const nxr::io::StoredStatistic P = nxr::io::read_statistic(sub, opt.psd);
  if (P.shape.size() != 2) invalid("wishart compute: " + opt.psd + " is not [channels x frequencies]");
  const double df = P.producer.value("frequency_resolution_hz", 0.0);
  if (!(df > 0)) invalid("wishart compute: " + opt.psd + " records no frequency resolution");
  const std::int64_t nf = P.shape[1];
  Eigen::VectorXd freq(nf), psd = Eigen::VectorXd::Zero(nf), gain(nf);
  for (std::int64_t k = 0; k < nf; ++k) freq[k] = static_cast<double>(k) * df;
  for (const std::int32_t ch : win.channels) {
    if (ch < 0 || ch >= P.shape[0]) invalid("wishart compute: the PSD has no channel " + std::to_string(ch));
    for (std::int64_t k = 0; k < nf; ++k) psd[k] += P.values[static_cast<std::size_t>(ch * nf + k)] / static_cast<double>(win.channels.size());
  }
  for (std::int64_t k = 0; k < nf; ++k) gain[k] = opt.shape == "morlet" ? nxr::filterbank::morletBand(freq[k], opt.band[0], opt.band[1]) : nxr::filterbank::meyerBand(freq[k], opt.band[0], opt.band[1], 1.0 / 3.0);
  double Beff;
  try { Beff = nxr::surrogate::effectiveBandwidth(freq, psd, gain); } catch (const std::invalid_argument& e) { invalid(std::string("wishart compute: ") + e.what()); }
  // THE CACHE
  const std::string base = "wishart." + opt.windows + "." + opt.bursts;
  CacheKey key;
  key.pipeline = "wishart compute";
  key.params = Json{{"band", opt.band}, {"shape", opt.shape}, {"draws", opt.draws}, {"seed", opt.seed}};
  key.input = Json{{"recording", text(*rec, "id")}, {"windows", opt.windows}, {"windows_level", win.level}, {"windows_n", win.sum.size()}, {"bursts", text(*brow, "id")}, {"bursts_n", spans.size()}, {"psd", text(P.row, "id")}};
  key.compute = NXR_COMPUTE_VERSION;
  const std::string pb = "statistics/" + nxr::io::safe_name(base), pr = "statistics/" + nxr::io::safe_name(base + ".run");
  WishartNullResult res;
  if (!opt.replace) if (find_cached(sub.db(), sub.id(), "statistic", key, pb)) { res.hit = true; res.info = Json{{"name", base}, {"cache", "hit"}}; say(Json{{"stage", "cache"}, {"hit", true}}); return res; }
  // THE BURSTS' SECOND MOMENTS: the windows WHOLLY inside each burst, summed
  std::map<std::int64_t, std::size_t> at;
  for (std::size_t w = 0; w < win.index.size(); ++w) at[win.index[w]] = w;
  const std::int64_t B = static_cast<std::int64_t>(spans.size());
  std::vector<Eigen::MatrixXcd> Q(static_cast<std::size_t>(B));
  std::vector<double> dof(static_cast<std::size_t>(B), std::numeric_limits<double>::quiet_NaN());
  std::int64_t used = 0;
  for (std::int64_t b = 0; b < B; ++b) {
    const double a = spans[static_cast<std::size_t>(b)].first, z = spans[static_cast<std::size_t>(b)].second;
    std::int64_t k0 = std::max<std::int64_t>(static_cast<std::int64_t>(std::ceil((a - t0 - eps) / width)), 0), k1 = static_cast<std::int64_t>(std::floor((z - t0 + eps) / width)) - 1;
    Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(win.channels.size()), static_cast<Eigen::Index>(win.channels.size()));
    bool whole = k1 >= k0;
    for (std::int64_t k = k0; whole && k <= k1; ++k) { const auto it = at.find(k); if (it == at.end()) whole = false; else sum += win.sum[it->second]; }
    if (!whole) continue;
    Q[static_cast<std::size_t>(b)] = sum;
    dof[static_cast<std::size_t>(b)] = static_cast<double>(k1 - k0 + 1) * width * Beff;
    ++used;
  }
  if (used == 0) invalid("wishart compute: no burst is whole in the stored windows (tile-aligned bursts inside the windows' span)");
  // THE DRAWS
  Eigen::VectorXd share(B), nullmean = Eigen::VectorXd::Zero(B), exceed = Eigen::VectorXd::Zero(B), p(B), nd = Eigen::VectorXd::Zero(B);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> run_null(static_cast<std::size_t>(opt.draws), 0.0);
  for (std::int64_t b = 0; b < B; ++b) {
    share[b] = Q[static_cast<std::size_t>(b)].size() ? lagged_share(Q[static_cast<std::size_t>(b)]) : nan;
    if (!std::isfinite(share[b])) { p[b] = nan; continue; }
    const Eigen::MatrixXd R = Q[static_cast<std::size_t>(b)].real();
    for (int j = 0; j < opt.draws; ++j) {
      const double s = lagged_share(nxr::surrogate::wishartDraw(R, dof[static_cast<std::size_t>(b)], opt.seed + static_cast<std::uint64_t>(b) * 7919u, static_cast<std::uint64_t>(j)));
      nullmean[b] += s / opt.draws; run_null[static_cast<std::size_t>(j)] += s;
      if (s >= share[b]) exceed[b] += 1;
    }
    p[b] = static_cast<double>(exceed[b] + 1) / static_cast<double>(opt.draws + 1);
    say(Json{{"stage", "bursts"}, {"burst", b}, {"share", share[b]}, {"p", p[b]}, {"dof", dof[static_cast<std::size_t>(b)]}});
  }
  const Eigen::VectorXd q = nxr::surrogate::benjaminiHochberg(p);
  double mean_share = 0, nmean = 0, nsd = 0;
  for (std::int64_t b = 0; b < B; ++b) if (std::isfinite(share[b])) mean_share += share[b] / static_cast<double>(used);
  for (int j = 0; j < opt.draws; ++j) { run_null[static_cast<std::size_t>(j)] /= static_cast<double>(used); nmean += run_null[static_cast<std::size_t>(j)] / opt.draws; }
  for (int j = 0; j < opt.draws; ++j) nsd += std::pow(run_null[static_cast<std::size_t>(j)] - nmean, 2) / (opt.draws - 1);
  nsd = std::sqrt(nsd);
  const double prun = static_cast<double>(1 + std::count_if(run_null.begin(), run_null.end(), [&](double s) { return s >= mean_share; })) / static_cast<double>(opt.draws + 1);
  bool replaced = false;
  clear_stored(ds, sub, {{"statistic", pb}, {"statistic", pr}}, opt.replace, base, replaced);
  const Json warnings = Json::array({"the null is a zero-lag Gaussian process with the burst's own covariance and `dof` independent samples: a burst beyond it has lags beyond sampling noise, not a wave",
                                     "dof = the burst's duration x the band's effective bandwidth (the PSD's, through the analytic gain): a spectrum with structure inside the band changes it",
                                     "the lagged share is a SENSOR-level quantity (the share of the second moment in quadrature between the sensors); the cortical quadrature and phase current need the operator applied to each draw (not built)"});
  Json opts = Json{{"draws", opt.draws}, {"seed", opt.seed}, {"band", opt.band}, {"shape", opt.shape}, {"effective_bandwidth_hz", Beff}, {"windows", opt.windows}, {"bursts", opt.bursts}, {"psd", opt.psd}, {"windows_level", win.level}, {"bursts_used", used}, {"bursts_total", B}};
  Json names = Json::array();
  for (std::int64_t b = 0; b < B; ++b) names.push_back(Json{{"name", "burst" + std::to_string(b)}, {"start", spans[static_cast<std::size_t>(b)].first}, {"stop", spans[static_cast<std::size_t>(b)].second}});
  std::vector<double> vb(static_cast<std::size_t>(B * 5));
  for (std::int64_t b = 0; b < B; ++b) { vb[static_cast<std::size_t>(5 * b)] = share[b]; vb[static_cast<std::size_t>(5 * b + 1)] = std::isfinite(share[b]) ? nullmean[b] : nan; vb[static_cast<std::size_t>(5 * b + 2)] = p[b]; vb[static_cast<std::size_t>(5 * b + 3)] = q[b]; vb[static_cast<std::size_t>(5 * b + 4)] = dof[static_cast<std::size_t>(b)]; }
  std::vector<double> vr = {mean_share, nmean, nsd, prun};
  Json pj = stamp(key);
  pj["by"] = "nxr-neuro wishart compute"; pj["warnings"] = warnings; pj["reference"] = "nxr vortex.wishart, dof_fraction";
  sub.compose([&] {
    nxr::io::StatisticSpec st;
    st.name = base; st.path = pb; st.measure = "null_p"; st.method = "wishart_null"; st.of_field_id = text(*rec, "id"); st.n_members = used;
    st.axes = Json::array({Json{{"name", "burst"}, {"values", names}, {"size", B}}, Json{{"name", "test"}, {"size", 5}}});
    st.components = Json::array({Json{{"name", "lagged_share"}}, Json{{"name", "null_mean"}}, Json{{"name", "p"}}, Json{{"name", "q"}}, Json{{"name", "dof"}}});
    st.data = nxr::io::view(vb, {B, 5});
    st.description = "per burst of " + opt.bursts + ": the share of the second moment in quadrature between the sensors, its mean under the Wishart null (" + std::to_string(opt.draws) + " draws), the p, BH q and degrees of freedom";
    st.options = opts; st.producer = pj;
    sub.statistic(std::move(st));
    nxr::io::StatisticSpec sr;
    sr.name = base + ".run"; sr.path = pr; sr.measure = "null_p"; sr.method = "wishart_null"; sr.of_field_id = text(*rec, "id"); sr.n_members = used;
    sr.axes = Json::array({Json{{"name", "test"}, {"values", Json::array({Json{{"name", "mean_share"}}, Json{{"name", "null_mean"}}, Json{{"name", "null_sd"}}, Json{{"name", "p"}}})}, {"size", 4}}});
    sr.data = nxr::io::view(vr, {4});
    sr.description = "the run's mean lagged share over the bursts of " + opt.bursts + " against the Wishart null's mean over the same bursts (" + std::to_string(opt.draws) + " draws)";
    sr.options = opts; sr.producer = pj;
    sub.statistic(std::move(sr));
  });
  std::int64_t sig = 0;
  for (std::int64_t b = 0; b < B; ++b) sig += std::isfinite(q[b]) && q[b] < 0.05;
  res.info = Json{{"name", base}, {"bursts", B}, {"bursts_used", used}, {"effective_bandwidth_hz", Beff}, {"mean_share", mean_share}, {"null_mean", nmean}, {"null_sd", nsd}, {"p_run", prun}, {"bursts_q05", sig}, {"replaced", replaced}, {"cache", "miss"}};
  return res;
}

}  // namespace nxr::pipelines
