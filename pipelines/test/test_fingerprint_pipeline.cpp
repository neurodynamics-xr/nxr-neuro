// THE `fingerprint` DEFINITION as a GROUP run (pipelines/definitions/fingerprint.yaml), through the library's group run (face.h, what `run fingerprint --subjects ...` and the bindings call), on a
// synthetic cohort in BOTH dataset forms: a CATALOGUE of subject packages with a group package (the default form, `create dataset --dataset`) and a one-database dataset with a default subject.
// Each member holds two stored features (region x bin "spectra"); the runner reads every member's two features by name from the member's own package and stacks them (preparation); the nodes
// stats.featureTable, stats.identification, stats.pca and stats.identificationCurve make the statistics, stored in the group package (the default subject) with their own rows: no row links
// into a member's package; the provenance lists every member's inputs. Checks the wiring and the behaviour, not the maths (test_identification holds that to the release):
//   a planted cohort is identified both ways (accuracy 1); a cohort whose two measurements are drawn independently is identified near chance; every statistic is stored with its axes;
//   the provenance names each member's target and database; a rerun is a cache hit; refused: a member missing its feature (named), a member whose feature has another length (named),
//   no members. Synthetic data only.
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/statistic.h"
#include "nxr/io/task/commands.h"
#include "nxr/pipelines/face.h"

using namespace nxr::io;
namespace face = nxr::pipelines::face;
namespace fs = std::filesystem;

namespace {
template <class F> std::string refusal(F f) { try { f(); } catch (const std::exception& e) { return e.what(); } return std::string(); }

DatasetWriter open_member(const fs::path& folder, const std::string& subject) { return is_catalogue_dataset(folder) ? open_subject_package(folder, subject) : open_writer(folder); }

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
void put_features(const fs::path& folder, const std::string& subject, const std::vector<std::pair<std::string, std::vector<double>>>& features, std::int64_t bins) {
  DatasetWriter ds = open_member(folder, subject);
  SubjectWriter sub = SubjectWriter::open(ds, subject);
  sub.compose([&] { for (const auto& [name, v] : features) put_feature(sub, name, v, static_cast<std::int64_t>(v.size()) / bins, bins); });
}
Json run(const fs::path& folder, const std::vector<std::string>& subjects, const std::string& into, Json parameters) {
  face::GroupRunRequest rq;
  rq.pipeline = "fingerprint";
  rq.subjects = subjects;
  rq.into = into;
  rq.parameters = std::move(parameters);
  return face::run_group(folder, rq);
}
std::vector<double> values_of(const fs::path& folder, const std::string& group, const std::string& name, std::vector<std::int64_t>* shape = nullptr, Json* row = nullptr) {
  DatasetWriter ds = open_member(folder, group);
  SubjectWriter sub = SubjectWriter::open(ds, group);
  const StoredStatistic s = read_statistic(sub, name);
  if (shape) *shape = s.shape;
  if (row) *row = Json{{"options", s.options}, {"of_statistic_id", opt_text(s.row, "of_statistic_id") ? Json(*opt_text(s.row, "of_statistic_id")) : Json()},
                       {"of_field_id", opt_text(s.row, "of_field_id") ? Json(*opt_text(s.row, "of_field_id")) : Json()}, {"producer", s.producer}};
  return s.values;
}

// one cohort in one dataset form; `catalogue`: the default form (packages and a group package), else a one-database dataset whose default subject holds the results
void cohort(const fs::path& root, bool catalogue) {
  const int n = 10, regions = 6, bins = 10, p = regions * bins;
  const fs::path folder = root / (catalogue ? "catalogue" : "onedb");
  task::CreateDatasetRequest cd;
  if (catalogue) cd.dataset = folder.string();
  else { cd.datastore = root.string(); cd.name = "onedb"; }
  task::create_dataset(cd);
  const std::string group = "group";
  task::create_subject({folder.string(), group, catalogue ? "group" : "template", "test"});
  std::vector<std::string> names;
  std::mt19937_64 rng(20261120);
  std::normal_distribution<double> z;
  for (int i = 0; i < n; ++i) {
    const std::string s = "S" + std::to_string(10 + i);
    names.push_back(s);
    task::create_subject({folder.string(), s, "subject", "test"});
    std::vector<double> own(p), a(p), b(p), x(p), y(p);
    for (int k = 0; k < p; ++k) own[k] = z(rng);
    for (int k = 0; k < p; ++k) {
      a[k] = std::exp(0.5 * own[k] + 0.15 * z(rng)); b[k] = std::exp(0.5 * own[k] + 0.15 * z(rng));   // the person's spectrum, twice, with independent noise
      x[k] = std::exp(0.5 * z(rng)); y[k] = std::exp(0.5 * z(rng));                                    // two independent draws: no person in them
    }
    put_features(folder, s, {{"fp_A", a}, {"fp_B", b}, {"null_A", x}, {"null_B", y}, {"odd_B", i == 3 ? std::vector<double>(p - bins, 1.0) : b}}, bins);   // odd_B: another length in one member
  }
  const std::string into = catalogue ? group : std::string();

  // a planted cohort: identified both ways
  const Json set = Json{{"target", "fp_B"}, {"database", "fp_A"}, {"name", "planted"}};
  const Json r = run(folder, names, into, set);
  CHECK(r.value("cache", std::string()) == "computed");
  Json row;
  const std::vector<double> acc = values_of(folder, group, "planted_accuracy", nullptr, &row);
  CHECK(acc.size() == 3 && acc[0] == 1.0 && acc[1] == 1.0 && std::abs(acc[2] - 1.0 / n) < 1e-15);
  // its own row in the group package: no link into a member's package; every member's two inputs in the provenance
  CHECK(row["of_statistic_id"].is_null() && row["of_field_id"].is_null());
  const Json members = row["producer"].value("group", Json::object()).value("members", Json::array());
  CHECK(members.size() == static_cast<std::size_t>(n));
  if (members.size() == static_cast<std::size_t>(n))
    CHECK(members[0].value("subject", std::string()) == names[0] && members[0]["inputs"]["target"].value("name", std::string()) == "fp_B" &&
          members[0]["inputs"]["database"].value("name", std::string()) == "fp_A" && members[0]["inputs"]["target"].contains("id"));
  // the feature axes, in their stored order, for a reader to map each feature (the edgewise statistics' options)
  Json erow;
  values_of(folder, group, "planted_edgewise", nullptr, &erow);
  std::vector<std::int64_t> shape;
  CHECK(values_of(folder, group, "planted_correlation", &shape).size() == static_cast<std::size_t>(n * n) && shape == std::vector<std::int64_t>({n, n}));
  const std::vector<double> self = values_of(folder, group, "planted_self", &shape);
  CHECK((shape == std::vector<std::int64_t>{n, 2}));
  bool high = true;
  for (int i = 0; i < n; ++i) high = high && self[static_cast<std::size_t>(2 * i + 1)] > 2;
  CHECK(high);
  values_of(folder, group, "planted_edgewise", &shape);
  CHECK((shape == std::vector<std::int64_t>{p, 3}));
  const Json fax = erow["options"].value("inputs", Json::object()).value("feature_axes", Json::array());
  CHECK(fax.size() == 2 && fax[0].value("name", std::string()) == "region" && fax[0]["values"][2].value("name", std::string()) == "region 2");
  const std::vector<double> curve = values_of(folder, group, "planted_curve", &shape);
  CHECK((shape == std::vector<std::int64_t>{3, 5}) && curve[0] == 2 && curve[5] == 5 && curve[10] == 10);   // the notebook's steps for 2n = 20: 2, 5, 10
  values_of(folder, group, "planted_reconstructed_edgewise", &shape);
  CHECK((shape == std::vector<std::int64_t>{p, 3}));
  values_of(folder, group, "planted_pca", &shape);
  CHECK(shape.size() == 2 && shape[1] == p);
  // a rerun of the same members and options is a cache hit
  CHECK(run(folder, names, into, set).value("cache", std::string()) == "hit");

  // two independent draws per member: identified near chance
  run(folder, names, into, Json{{"target", "null_B"}, {"database", "null_A"}, {"name", "null"}});
  const std::vector<double> nacc = values_of(folder, group, "null_accuracy");
  CHECK(nacc[0] <= 0.3 && nacc[1] <= 0.3);

  // refusals, each naming the member
  task::create_subject({folder.string(), "S99", "subject", "test"});
  put_features(folder, "S99", {{"fp_A", std::vector<double>(p, 1.0)}}, bins);
  std::vector<std::string> with_missing = names;
  with_missing.push_back("S99");
  const std::string missing = refusal([&] { run(folder, with_missing, into, Json{{"target", "fp_B"}, {"database", "fp_A"}, {"name", "x"}}); });
  CHECK(missing.find("S99") != std::string::npos && missing.find("target") != std::string::npos);
  const std::string odd = refusal([&] { run(folder, names, into, Json{{"target", "odd_B"}, {"database", "fp_A"}, {"name", "y"}}); });
  CHECK(odd.find(names[3]) != std::string::npos && odd.find("shape") != std::string::npos);
  const std::string none = refusal([&] { run(folder, {}, into, set); });
  CHECK(none.find("subjects") != std::string::npos);
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_fingerprint_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  std::printf("a catalogue of subject packages, the results in a group package\n");
  try { cohort(root, true); } catch (const std::exception& e) { std::printf("  catalogue: %s\n", e.what()); CHECK(false); }
  std::printf("a one-database dataset, the results in its default subject\n");
  try { cohort(root, false); } catch (const std::exception& e) { std::printf("  one database: %s\n", e.what()); CHECK(false); }
  return nxrtest::finish("test_fingerprint_pipeline");
}
