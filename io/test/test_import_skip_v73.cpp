// A Brainstorm import in a build that cannot read MATLAB v7.3 (HDF5) files, the WebAssembly one: each v7.3 file is SKIPPED and the rest imported, the file named with its reason in
// the command's result (`skipped`, `message`) and in its job row; any other unreadable file is still refused; a build that reads v7.3 (the native one) imports everything as before.
// The WebAssembly path is forced natively (nxr::bst::set_v73_readable(false)). Synthetic data only: a default anatomy of small octahedra (v5), one cortex written as v7.3 by matio.
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

extern "C" {
#include "matio.h"
}
#include "check.h"
#include "nxr/bst/mat.h"
#include "nxr/io.h"
#include "nxr/io/task/commands.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}
matvar_t* numeric(const char* name, const std::vector<double>& row_major, size_t rows, size_t cols) {
  std::vector<double> col(row_major.size());
  for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < cols; ++c) col[c * rows + r] = row_major[r * cols + c];
  size_t d[2] = {rows, cols};
  return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d, col.data(), 0);
}
matvar_t* chars(const char* name, const std::string& s) {
  size_t d[2] = {1, s.size()};
  return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, d, const_cast<char*>(s.data()), 0);
}
void write(mat_t* m, matvar_t* v) {
  Mat_VarWrite(m, v, MAT_COMPRESSION_NONE);
  Mat_VarFree(v);
}
// a Brainstorm cortex: two octahedra (12 vertices, metres), Vertices, Faces and Comment, in the MAT-file version given
void write_tess(const fs::path& p, mat_ft version, double scale) {
  std::vector<double> v, faces;
  const std::vector<int> f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  for (const double cx : {-0.03, 0.03}) {
    const double r = 0.01 * scale;
    for (const double x : {cx + r, 0.0, 0.0, cx - r, 0.0, 0.0, cx, r, 0.0, cx, -r, 0.0, cx, 0.0, r, cx, 0.0, -r}) v.push_back(x);
  }
  for (int h = 0; h < 2; ++h) for (const int i : f) faces.push_back(i + 1 + 6 * h);
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, version);
  write(m, numeric("Vertices", v, 12, 3));
  write(m, numeric("Faces", faces, 16, 3));
  write(m, chars("Comment", "cortex_synthetic"));
  Mat_Close(m);
}
// a subject image: a 4 x 4 x 4 uint8 Cube, its voxel size and its registration frames
void write_image(const fs::path& p, mat_ft version) {
  std::vector<std::uint8_t> cube(64);   // an anatomical image is uint8, as Brainstorm stores one
  for (std::size_t i = 0; i < cube.size(); ++i) cube[i] = static_cast<std::uint8_t>(i % 7);
  size_t d[3] = {4, 4, 4};
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, version);
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
// a protocol whose default anatomy has a v5 cortex and image (both platforms' matio read them) and ONE v7.3 file, the second cortex
fs::path protocol(const fs::path& root) {
  const fs::path a = root / "anat" / "@default_subject";
  fs::create_directories(a);
  write_tess(a / "tess_cortex_pial_low.mat", MAT_FT_MAT5, 1.0);
  write_tess(a / "tess_cortex_white_low.mat", MAT_FT_MAT73, 0.9);
  write_image(a / "subjectimage_T1.mat", MAT_FT_MAT5);
  return root;
}
std::string dataset(const fs::path& store, const std::string& name) {
  task::CreateDatasetRequest d;
  d.datastore = store.string();
  d.name = name;
  d.created_by = "test";
  task::create_dataset(d);
  return (store / name).string();
}
bool named(const Json& list, const std::string& part) {
  for (const auto& x : list) if (x.is_object() && x.value("what", std::string()).find(part) != std::string::npos && x.value("reason", std::string()) == nxr::bst::kUnreadableV73) return true;
  return false;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_import_skip_v73_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "store");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path proto = protocol(root / "protocol");
  CHECK(nxr::bst::is_v73(proto / "anat" / "@default_subject" / "tess_cortex_white_low.mat"));
  CHECK(!nxr::bst::is_v73(proto / "anat" / "@default_subject" / "tess_cortex_pial_low.mat"));
  CHECK(!nxr::bst::is_v73(proto / "anat" / "@default_subject" / "subjectimage_T1.mat"));
  {   // the image's registration frame reads back (the import needs it)
    const auto scs = nxr::bst::read_variable(proto / "anat" / "@default_subject" / "subjectimage_T1.mat", "SCS");
    CHECK(scs && scs->is_struct() && scs->field("R") != nullptr && scs->field("NAS") != nullptr);
  }

  // ---- a build that cannot read v7.3 (the WebAssembly one, forced): the two v7.3 files skipped, the rest imported, both named in the result and the job row
  {
    nxr::bst::set_v73_readable(false);
    const std::string D = dataset(root / "store", "skipping");
    task::ImportTemplateRequest rq;
    rq.protocol = proto.string();
    rq.dataset = D;
    rq.created_by = "test";
    const Json r = task::import_template(rq).result;
    CHECK(r["surfaces"] == Json::array({"cortex_pial_low"}));
    CHECK_EQ(r["volumes"].size(), std::size_t(1));
    CHECK(named(r["skipped"], "tess_cortex_white_low.mat"));
    CHECK(r["message"].is_string() && r["message"].get<std::string>().find("1 file skipped") != std::string::npos &&
          r["message"].get<std::string>().find("tess_cortex_white_low.mat") != std::string::npos);
    task::ListClassRequest jobs;
    jobs.cls = "jobs";
    jobs.dataset = D;
    const Json rows = task::list_class(jobs).result["rows"];
    CHECK_EQ(rows.size(), std::size_t(1));
    if (rows.size() == 1) {
      CHECK(rows[0]["status"] == "done");
      CHECK(named(rows[0]["result"]["skipped"], "tess_cortex_white_low.mat"));
      CHECK(rows[0]["result"]["message"] == r["message"]);
    }
    // a file this build cannot read for another reason is refused as before: nothing is skipped silently
    fs::path bad = root / "protocol_bad";
    protocol(bad);
    std::ofstream(bad / "anat" / "@default_subject" / "tess_cortex_mid_low.mat", std::ios::binary) << "MATLAB 5.0 MAT-file, not really";
    rq.protocol = bad.string();
    rq.dataset = dataset(root / "store", "refusing");
    CHECK(!code_of([&] { task::import_template(rq); }).empty());
    // the same refusal by name when a v7.3 file is the only cortex there is
    const fs::path only = root / "protocol_only";
    fs::create_directories(only / "anat" / "@default_subject");
    write_tess(only / "anat" / "@default_subject" / "tess_cortex_white_low.mat", MAT_FT_MAT73, 1.0);
    rq.protocol = only.string();
    rq.dataset = dataset(root / "store", "only");
    CHECK_EQ(code_of([&] { task::import_template(rq); }), std::string(code::unsupported));
  }

  // ---- a build that reads v7.3 (native): everything imported, nothing skipped
  {
    nxr::bst::set_v73_readable(true);
    task::ImportTemplateRequest rq;
    rq.protocol = proto.string();
    rq.dataset = dataset(root / "store", "reading");
    rq.created_by = "test";
    const Json r = task::import_template(rq).result;
    CHECK_EQ(r["surfaces"].size(), std::size_t(2));
    CHECK_EQ(r["volumes"].size(), std::size_t(1));
    CHECK(r["skipped"].empty());
    CHECK(!r.contains("message"));
  }
  return nxrtest::finish("test_import_skip_v73");
}
