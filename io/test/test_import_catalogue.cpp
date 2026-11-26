// The Brainstorm importers on a CATALOGUE dataset (the default form, `create dataset --dataset`) as on a one-database one: `import subject` puts the subject in a package of its own,
// `import template` and `import group` put the template and the group results in theirs, `import condition` makes the subject's package when the catalogue has none, each package
// registered only once it is filled. A subject the catalogue lists is refused; a failed import leaves no package and registers nothing; the job row goes into the member's package.
// Synthetic data only: a protocol of small octahedra (a default anatomy and a subject's own cortex) and one imported recording of three EEG channels, written with matio.
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <functional>

extern "C" {
#include "matio.h"
}
#include "bst_protocol.h"
#include "check.h"
#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/task/commands.h"

using namespace nxr::io;
namespace fs = std::filesystem;

using namespace bst_protocol;

namespace {
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
std::string catalogue(const fs::path& folder) {
  task::CreateDatasetRequest d;
  d.dataset = folder.string();
  d.created_by = "test";
  task::create_dataset(d);
  return folder.string();
}
std::string one_database(const fs::path& store, const std::string& name) {
  task::CreateDatasetRequest d;
  d.datastore = store.string();
  d.name = name;
  d.created_by = "test";
  task::create_dataset(d);
  return (store / name).string();
}
// the catalogue's members, "name:kind:status"
std::vector<std::string> members(const std::string& D) {
  std::vector<std::string> out;
  for (const auto& e : Catalogue::open(D).entries()) out.push_back(e.name + ":" + e.kind + ":" + e.status);
  return out;
}
Json jobs_of(const std::string& D, const std::optional<std::string>& subject) {
  task::ListClassRequest q;
  q.cls = "jobs";
  q.dataset = D;
  q.subject = subject;
  return task::list_class(q).result["rows"];
}
std::size_t n_manifolds(const std::string& D, const std::string& subject) {
  task::ListClassRequest q;
  q.cls = "manifolds";
  q.dataset = D;
  q.subject = subject;
  return task::list_class(q).result["rows"].size();
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_import_catalogue_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "store");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const std::string P = protocol(root / "protocol").string();

  // ---- the one-database form, as before: the reference for what each import gives
  const std::string O = one_database(root / "store", "one");
  task::ImportSubjectRequest is;
  is.protocol = P;
  is.subject = "S01";
  is.created_by = "test";
  is.dataset = O;
  const Json one_subject = task::import_subject(is).result;
  CHECK_EQ(one_subject["sessions"], Json::array({"rest"}));
  task::ImportTemplateRequest it;
  it.protocol = P;
  it.created_by = "test";
  it.dataset = O;
  const Json one_template = task::import_template(it).result;
  CHECK_EQ(one_template["surfaces"], Json::array({"cortex_pial_low"}));

  // ---- a catalogue: the subject in a package of its own, registered once filled; the same nodes as the one-database form
  const std::string C = catalogue(root / "study");
  is.dataset = C;
  const Json cat_subject = task::import_subject(is).result;
  CHECK_EQ(cat_subject["sessions"], one_subject["sessions"]);
  CHECK_EQ(cat_subject["rows"], one_subject["rows"]);
  CHECK(fs::is_regular_file(root / "study" / "S01.nxr.zarr" / kPackageDatabaseFile));
  CHECK(members(C) == std::vector<std::string>({"S01:subject:registered"}));
  CHECK_EQ(n_manifolds(C, "S01"), n_manifolds(O, "S01"));
  {
    const Json jobs = jobs_of(C, std::string("S01"));   // the job row is in the subject's package
    CHECK_EQ(jobs.size(), std::size_t(1));
    if (jobs.size() == 1) CHECK(jobs[0]["kind"] == "import:subject" && jobs[0]["status"] == "done");
  }
  CHECK(!cat_subject.contains("job_error"));
  // imported again: refused, as a one-database dataset refuses it (a subject is imported whole)
  CHECK(has(thrown([&] { task::import_subject(is); }), "already_exists"));
  is.dataset = O;
  CHECK(has(thrown([&] { task::import_subject(is); }), "already_exists"));

  // a subject with nothing to import: the import fails, no package is left and nothing is registered
  is.dataset = C;
  is.subject = "S02";
  CHECK(!thrown([&] { task::import_subject(is); }).empty());
  CHECK(!fs::exists(root / "study" / "S02.nxr.zarr"));
  CHECK(members(C) == std::vector<std::string>({"S01:subject:registered"}));
  // a package of that name already in the folder, not registered: refused, and left as it is
  fs::create_directories(root / "study" / "S02.nxr.zarr");
  CHECK(has(thrown([&] { task::import_subject(is); }), "already in the dataset's folder"));
  CHECK(fs::is_directory(root / "study" / "S02.nxr.zarr"));
  fs::remove_all(root / "study" / "S02.nxr.zarr");

  // ---- the template: its own package (kind template), a member of the dataset; its job row there
  it.dataset = C;
  const Json cat_template = task::import_template(it).result;
  CHECK_EQ(cat_template["surfaces"], one_template["surfaces"]);
  CHECK(!cat_template.contains("job_error"));
  CHECK(members(C) == std::vector<std::string>({"S01:subject:registered", "template:template:registered"}));
  CHECK_EQ(jobs_of(C, std::string("template")).size(), std::size_t(1));
  CHECK(has(thrown([&] { task::import_template(it); }), "already_exists"));

  // ---- the group: a protocol without group results fails as on a one-database dataset, and leaves no group package
  task::ImportGroupRequest ig;
  ig.protocol = P;
  ig.created_by = "test";
  ig.dataset = O;
  const std::string one_group = thrown([&] { task::import_group(ig); });
  CHECK(has(one_group, "not_found"));
  ig.dataset = C;
  CHECK_EQ(thrown([&] { task::import_group(ig); }), one_group);
  CHECK(!fs::exists(root / "study" / "group.nxr.zarr"));
  CHECK_EQ(members(C).size(), std::size_t(2));

  // ---- a condition into a catalogue without the subject: its package is made, filled and registered; into a listed subject: its package opened
  {
    const std::string C2 = catalogue(root / "study2");
    task::ImportConditionRequest ic;
    ic.protocol = P;
    ic.subject = "S01";
    ic.condition = "rest";
    ic.created_by = "test";
    ic.dataset = C2;
    const Json r = task::import_condition(ic).result;
    CHECK_EQ(r["subject"].get<std::string>(), std::string("S01"));
    CHECK(members(C2) == std::vector<std::string>({"S01:subject:registered"}));
    CHECK(n_manifolds(C2, "S01") > 0);
    CHECK_EQ(jobs_of(C2, std::string("S01")).size(), std::size_t(1));
    task::CreateSubjectRequest cs;
    cs.dataset = C2;
    cs.name = "S03";
    cs.created_by = "test";
    task::create_subject(cs);
    ic.subject = "S03";
    CHECK(!thrown([&] { task::import_condition(ic); }).empty());   // S03 has no such condition in the protocol: refused, the listed package stays
    CHECK(members(C2) == std::vector<std::string>({"S01:subject:registered", "S03:subject:registered"}));
  }
  return nxrtest::finish("test_import_catalogue");
}
