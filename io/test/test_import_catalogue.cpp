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
#include "check.h"
#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/task/commands.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
matvar_t* numeric(const char* name, const std::vector<double>& row_major, size_t rows, size_t cols) {
  std::vector<double> col(row_major.size());
  for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < cols; ++c) col[c * rows + r] = row_major[r * cols + c];
  size_t d[2] = {rows, cols};
  return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d, col.data(), 0);
}
matvar_t* empty(const char* name) {
  size_t d[2] = {0, 0};
  return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d, nullptr, 0);
}
matvar_t* chars(const char* name, const std::string& s) {
  size_t d[2] = {1, s.size()};
  return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, d, const_cast<char*>(s.data()), 0);
}
void write(mat_t* m, matvar_t* v) {
  Mat_VarWrite(m, v, MAT_COMPRESSION_NONE);
  Mat_VarFree(v);
}
// a Brainstorm cortex: two octahedra (12 vertices, metres)
void write_tess(const fs::path& p) {
  std::vector<double> v, faces;
  const std::vector<int> f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  for (const double cx : {-0.03, 0.03}) {
    const double r = 0.01;
    for (const double x : {cx + r, 0.0, 0.0, cx - r, 0.0, 0.0, cx, r, 0.0, cx, -r, 0.0, cx, 0.0, r, cx, 0.0, -r}) v.push_back(x);
  }
  for (int h = 0; h < 2; ++h) for (const int i : f) faces.push_back(i + 1 + 6 * h);
  fs::create_directories(p.parent_path());
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, MAT_FT_MAT5);
  write(m, numeric("Vertices", v, 12, 3));
  write(m, numeric("Faces", faces, 16, 3));
  write(m, chars("Comment", "cortex_synthetic"));
  Mat_Close(m);
}
// a subject image: a 4 x 4 x 4 uint8 Cube, its voxel size and its registration frames
void write_image(const fs::path& p) {
  std::vector<std::uint8_t> cube(64);   // an anatomical image is uint8, as Brainstorm stores one
  for (std::size_t i = 0; i < cube.size(); ++i) cube[i] = static_cast<std::uint8_t>(i % 7);
  size_t d[3] = {4, 4, 4};
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, MAT_FT_MAT5);
  write(m, Mat_VarCreate("Cube", MAT_C_UINT8, MAT_T_UINT8, 3, d, cube.data(), 0));
  write(m, numeric("Voxsize", {1, 1, 1}, 1, 3));
  write(m, chars("Comment", "MRI"));
  {   // the registration frames an import requires, as Brainstorm writes them: SCS with its landmarks (voxel units), NCS (identity axes, shifted so that the nasion is at +x, the left ear at +y, the right ear at -y, all three at z = 0)
    const char* fn[] = {"R", "T", "NAS", "LPA", "RPA"};
    size_t one[2] = {1, 1};
    matvar_t* sc = Mat_VarCreateStruct("SCS", 2, one, fn, 5);
    Mat_VarSetStructFieldByName(sc, "R", 0, numeric(nullptr, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3));
    Mat_VarSetStructFieldByName(sc, "T", 0, numeric(nullptr, {0, -50, -2}, 3, 1));
    Mat_VarSetStructFieldByName(sc, "NAS", 0, numeric(nullptr, {50, 25, 2}, 1, 3));
    Mat_VarSetStructFieldByName(sc, "LPA", 0, numeric(nullptr, {2, 99, 2}, 1, 3));
    Mat_VarSetStructFieldByName(sc, "RPA", 0, numeric(nullptr, {2, 1, 2}, 1, 3));
    write(m, sc);
    const char* nf[] = {"R", "T"};
    matvar_t* nc = Mat_VarCreateStruct("NCS", 2, one, nf, 2);
    Mat_VarSetStructFieldByName(nc, "R", 0, numeric(nullptr, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3));
    Mat_VarSetStructFieldByName(nc, "T", 0, numeric(nullptr, {-100, -120, -80}, 3, 1));
    write(m, nc);
  }
  Mat_Close(m);
}
// a condition with one imported recording: three EEG channels (channel.mat) and their values (data_rest.mat: F, Time, ChannelFlag)
void write_condition(const fs::path& dir) {
  fs::create_directories(dir);
  const std::vector<std::string> names = {"Fz", "Cz", "Pz"};
  {
    mat_t* m = Mat_CreateVer((dir / "channel.mat").string().c_str(), nullptr, MAT_FT_MAT5);
    const char* fields[] = {"Name", "Type", "Loc", "Orient", "Weight", "Comment"};
    size_t d[2] = {1, names.size()};
    matvar_t* ch = Mat_VarCreateStruct("Channel", 2, d, fields, 6);
    for (size_t i = 0; i < names.size(); ++i) {
      Mat_VarSetStructFieldByName(ch, "Name", i, chars(nullptr, names[i]));
      Mat_VarSetStructFieldByName(ch, "Type", i, chars(nullptr, "EEG"));
      Mat_VarSetStructFieldByName(ch, "Loc", i, numeric(nullptr, {0.01 * double(i), 0.02, 0.08}, 3, 1));
      Mat_VarSetStructFieldByName(ch, "Orient", i, empty(nullptr));
      Mat_VarSetStructFieldByName(ch, "Weight", i, numeric(nullptr, {1.0}, 1, 1));
      Mat_VarSetStructFieldByName(ch, "Comment", i, chars(nullptr, ""));
    }
    write(m, ch);
    write(m, chars("Comment", "EEG"));
    Mat_Close(m);
  }
  const std::size_t n = 200;
  std::vector<double> F(names.size() * n), t(n);
  for (std::size_t c = 0; c < names.size(); ++c) for (std::size_t k = 0; k < n; ++k) F[c * n + k] = 1e-6 * double((c + 1) * (k % 17));
  for (std::size_t k = 0; k < n; ++k) t[k] = double(k) / 100.0;
  mat_t* m = Mat_CreateVer((dir / "data_rest.mat").string().c_str(), nullptr, MAT_FT_MAT5);
  write(m, numeric("F", F, names.size(), n));
  write(m, numeric("Time", t, 1, n));
  write(m, numeric("ChannelFlag", {1, 1, 1}, 3, 1));
  write(m, chars("Comment", "rest"));
  write(m, chars("DataType", "recordings"));
  Mat_Close(m);
}
// the protocol: a default anatomy (the template), subject S01 with its own cortex and one condition, and subject S02 with no condition (an import that fails)
fs::path protocol(const fs::path& root) {
  write_tess(root / "anat" / "@default_subject" / "tess_cortex_pial_low.mat");
  write_image(root / "anat" / "@default_subject" / "subjectimage_T1.mat");
  write_tess(root / "anat" / "S01" / "tess_cortex_pial_low.mat");
  write_condition(root / "data" / "S01" / "rest");
  write_tess(root / "anat" / "S02" / "tess_cortex_pial_low.mat");
  fs::create_directories(root / "data" / "S02");
  return root;
}
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
