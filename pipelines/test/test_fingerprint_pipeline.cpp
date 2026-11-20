// THE `fingerprint` DEFINITION as a GROUP run (pipelines/definitions/fingerprint.yaml): a synthetic cohort, each member holding two stored features (region x bin "spectra"), identified across the
// members: the runner reads every member's two features by name and stacks them (preparation), the nodes stats.featureTable, stats.identification, stats.pca and stats.identificationCurve make
// the statistics, stored in the dataset's default subject. Checks the wiring and the behaviour, not the maths (test_identification holds that to the release):
//   a planted cohort is identified both ways (accuracy 1); a cohort whose two measurements are drawn independently is identified near chance; every statistic is stored with its axes;
//   a rerun is a cache hit; refused: a member missing its feature (named), a member whose feature has another length (named).
// Synthetic data only.
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/runner.h"

using namespace nxr::io;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;

namespace {
template <class F> std::string refusal(F f) { try { f(); } catch (const std::exception& e) { return e.what(); } return std::string(); }

void put_feature(SubjectWriter& sub, const std::string& name, const std::vector<double>& v, std::int64_t regions, std::int64_t bins) {
  Json reg = Json::array(), fr = Json::array();
  for (std::int64_t r = 0; r < regions; ++r) reg.push_back(Json{{"name", "region " + std::to_string(r)}});
  for (std::int64_t b = 0; b < bins; ++b) fr.push_back(Json{{"name", std::to_string(0.5 * static_cast<double>(b)) + " Hz"}});
  StatisticSpec st;
  st.name = name; st.measure = "power_density"; st.method = "welch"; st.detached = true;
  st.axes = Json::array({Json{{"name", "region"}, {"size", regions}, {"values", reg}}, Json{{"name", "frequency"}, {"size", bins}, {"values", fr}}});
  st.options = Json{{"synthetic", "a planted region x bin spectrum (test)"}};
  st.data = view(v, {regions, bins});
  sub.statistic(std::move(st));
}
std::vector<double> values_of(SubjectWriter& sub, const std::string& name, std::vector<std::int64_t>* shape = nullptr) {
  const StoredStatistic s = read_statistic(sub, name);
  if (shape) *shape = s.shape;
  return s.values;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_fingerprint_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  const int n = 10, regions = 6, bins = 10, p = regions * bins;
  DatasetWriter ds = create_dataset(root, "cohort");
  SubjectOptions tpl;
  tpl.kind = "template";
  SubjectWriter group = SubjectWriter::create(ds, "group", tpl);
  group.complete();
  std::vector<std::string> names;
  std::vector<std::unique_ptr<SubjectWriter>> subs;
  std::vector<g::GroupMember> members;
  std::mt19937_64 rng(20261120);
  std::normal_distribution<double> z;
  for (int i = 0; i < n; ++i) {
    const std::string s = "S" + std::to_string(10 + i);
    names.push_back(s);
    subs.push_back(std::make_unique<SubjectWriter>(SubjectWriter::create(ds, s)));
    SubjectWriter& sw = *subs.back();
    members.push_back(g::GroupMember{s, &ds, &sw});
    std::vector<double> own(p), a(p), b(p), x(p), y(p);
    for (int k = 0; k < p; ++k) own[k] = z(rng);
    for (int k = 0; k < p; ++k) {
      a[k] = std::exp(0.5 * own[k] + 0.15 * z(rng)); b[k] = std::exp(0.5 * own[k] + 0.15 * z(rng));   // the person's spectrum, twice, with independent noise
      x[k] = std::exp(0.5 * z(rng)); y[k] = std::exp(0.5 * z(rng));                                    // two independent draws: no person in them
    }
    put_feature(sw, "fp_A", a, regions, bins);
    put_feature(sw, "fp_B", b, regions, bins);
    put_feature(sw, "null_A", x, regions, bins);
    put_feature(sw, "null_B", y, regions, bins);
    if (i == 3) put_feature(sw, "odd_B", std::vector<double>(p - bins, 1.0), regions - 1, bins);   // another length in one member
    else put_feature(sw, "odd_B", b, regions, bins);
    sw.complete();                                                       // the subject and its features: complete
  }
  const g::Definition d = g::load("fingerprint");
  g::RunOptions opt;

  // a planted cohort: identified both ways
  const Json set = Json{{"target", "fp_B"}, {"database", "fp_A"}, {"name", "planted"}};
  const auto r = g::run_group(ds, group, members, d, set, opt);
  CHECK(r.size() == 1 && !r.at(0).value("hit", true));
  const std::vector<double> acc = values_of(group, "planted_accuracy");
  CHECK(acc.size() == 3 && acc[0] == 1.0 && acc[1] == 1.0 && std::abs(acc[2] - 1.0 / n) < 1e-15);
  std::vector<std::int64_t> shape;
  CHECK(values_of(group, "planted_correlation", &shape).size() == static_cast<std::size_t>(n * n) && shape == std::vector<std::int64_t>({n, n}));
  const std::vector<double> self = values_of(group, "planted_self", &shape);
  CHECK((shape == std::vector<std::int64_t>{n, 2}));
  bool high = true;
  for (int i = 0; i < n; ++i) high = high && self[static_cast<std::size_t>(2 * i + 1)] > 2;
  CHECK(high);
  values_of(group, "planted_edgewise", &shape);
  CHECK((shape == std::vector<std::int64_t>{p, 3}));
  const std::vector<double> curve = values_of(group, "planted_curve", &shape);
  CHECK((shape == std::vector<std::int64_t>{3, 5}) && curve[0] == 2 && curve[5] == 5 && curve[10] == 10);   // the notebook's steps for 2n = 20: 2, 5, 10
  values_of(group, "planted_reconstructed_edgewise", &shape);
  CHECK((shape == std::vector<std::int64_t>{p, 3}));
  values_of(group, "planted_pca", &shape);
  CHECK(shape.size() == 2 && shape[1] == p);
  // a rerun of the same members and options is a cache hit
  CHECK(g::run_group(ds, group, members, d, set, opt).at(0).value("hit", false));

  // two independent draws per member: identified near chance
  g::run_group(ds, group, members, d, Json{{"target", "null_B"}, {"database", "null_A"}, {"name", "null"}}, opt);
  const std::vector<double> nacc = values_of(group, "null_accuracy");
  CHECK(nacc[0] <= 0.3 && nacc[1] <= 0.3);

  // refusals, each naming the member
  std::vector<g::GroupMember> with_missing = members;
  SubjectWriter s99 = SubjectWriter::create(ds, "S99");
  put_feature(s99, "fp_A", std::vector<double>(p, 1.0), regions, bins);
  s99.complete();
  with_missing.push_back(g::GroupMember{"S99", &ds, &s99});
  const std::string missing = refusal([&] { g::run_group(ds, group, with_missing, d, Json{{"target", "fp_B"}, {"database", "fp_A"}, {"name", "x"}}, opt); });
  CHECK(missing.find("S99") != std::string::npos && missing.find("target") != std::string::npos);
  const std::string odd = refusal([&] { g::run_group(ds, group, members, d, Json{{"target", "odd_B"}, {"database", "fp_A"}, {"name", "y"}}, opt); });
  CHECK(odd.find(names[3]) != std::string::npos && odd.find("shape") != std::string::npos);
  const std::string none = refusal([&] { g::run_group(ds, group, {}, d, set, opt); });
  CHECK(none.find("subjects") != std::string::npos);
  return nxrtest::finish("test_fingerprint_pipeline");
}
