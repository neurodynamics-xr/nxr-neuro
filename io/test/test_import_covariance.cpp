// `import covariance` (nxr/bst/covariance.h): a Brainstorm study's noise covariance (noisecov_*.mat) stored as a statistic of the condition's channels: the rows with a non-zero value,
// in the channel file's order, the values of those rows, the file, comment, history and the recording it was computed from in its options; a missing data covariance, a repeat and
// a size that disagrees with the channel file are refused. Synthetic data only: a protocol of one condition with four EEG channels (one with a zero row), written with matio.
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
#include "nxr/io/statistic.h"
#include "nxr/io/subject_package.h"
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
matvar_t* cell_of_chars(const char* name, const std::vector<std::string>& col_major, size_t rows, size_t cols) {
  std::vector<matvar_t*> v;
  for (const std::string& s : col_major) v.push_back(chars(nullptr, s));
  v.push_back(nullptr);
  size_t d[2] = {rows, cols};
  return Mat_VarCreate(name, MAT_C_CELL, MAT_T_CELL, 2, d, v.data(), 0);
}
// a condition: four EEG channels (channel.mat), their values (data_rest.mat) and a noise covariance (noisecov_full.mat) whose second channel has a zero row and column
void write_condition(const fs::path& dir) {
  fs::create_directories(dir);
  const std::vector<std::string> names = {"Fz", "Cz", "Pz", "Oz"};
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
  {
    mat_t* m = Mat_CreateVer((dir / "data_rest.mat").string().c_str(), nullptr, MAT_FT_MAT5);
    write(m, numeric("F", F, names.size(), n));
    write(m, numeric("Time", t, 1, n));
    write(m, numeric("ChannelFlag", {1, 1, 1, 1}, 4, 1));
    write(m, chars("Comment", "rest"));
    write(m, chars("DataType", "recordings"));
    Mat_Close(m);
  }
  std::vector<double> cov(16, 0.0), ns(16, 0.0);
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
    if (r == 1 || c == 1) continue;
    cov[r * 4 + c] = r == c ? 4.0 + 0.1 * r : 0.5;
    ns[r * 4 + c] = 1000;
  }
  mat_t* m = Mat_CreateVer((dir / "noisecov_full.mat").string().c_str(), nullptr, MAT_FT_MAT5);
  write(m, numeric("NoiseCov", cov, 4, 4));
  write(m, numeric("nSamples", ns, 4, 4));
  write(m, numeric("FourthMoment", cov, 4, 4));
  write(m, chars("Comment", "Noise covariance"));
  // History: rows {date, type, text}, as a cell stored column after column
  write(m, cell_of_chars("History", {"26-Jun-2026", "26-Jun-2026", "import", "src", "from file", "emptyroom_raw"}, 2, 3));
  Mat_Close(m);
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_import_covariance_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path P = root / "protocol";
  write_tess(P / "anat" / "@default_subject" / "tess_cortex_pial_low.mat");
  write_image(P / "anat" / "@default_subject" / "subjectimage_T1.mat");
  write_tess(P / "anat" / "S01" / "tess_cortex_pial_low.mat");
  write_condition(P / "data" / "S01" / "rest");
  task::CreateDatasetRequest cd;
  cd.dataset = (root / "study").string();
  cd.created_by = "test";
  task::create_dataset(cd);
  task::ImportConditionRequest ic;
  ic.protocol = P.string();
  ic.subject = "S01";
  ic.condition = "rest";
  ic.dataset = (root / "study").string();
  ic.created_by = "test";
  task::import_condition(ic);

  task::ImportCovarianceRequest rq;
  rq.protocol = P.string();
  rq.dataset = (root / "study").string();
  rq.subject = "S01";
  rq.condition = "rest";
  rq.created_by = "test";
  const Json r = task::import_covariance(rq).result;
  CHECK_EQ(r["name"].get<std::string>(), std::string("rest_noise_covariance"));
  CHECK_EQ(r["n_channels"].get<std::int64_t>(), std::int64_t(3));
  CHECK_EQ(r["source_recording"].get<std::string>(), std::string("emptyroom_raw"));

  {
    DatasetWriter w = open_package_writer(root / "study" / "S01.nxr.zarr");
    SubjectWriter sub = SubjectWriter::open(w, "S01");
    const StoredStatistic st = read_statistic(sub, "rest_noise_covariance");
    CHECK_EQ(text(st.row, "measure"), std::string("covariance"));
    CHECK_EQ(text(st.row, "method"), std::string("empirical"));
    CHECK(st.shape == std::vector<std::int64_t>({3, 3}));
    // the covered channels 0, 2, 3 (channel 1 has a zero row): the values of those rows
    const double c0 = 4.0, c2 = 4.2, c3 = 4.3;
    const std::vector<double> want = {c0, 0.5, 0.5, 0.5, c2, 0.5, 0.5, 0.5, c3};
    CHECK(st.values.size() == 9);
    for (std::size_t i = 0; i < 9 && i < st.values.size(); ++i) CHECK(std::abs(st.values[i] - want[i]) < 1e-12);
    CHECK_EQ(integer(st.row, "n_members"), std::int64_t(1000));
    CHECK_EQ(st.options["comment"].get<std::string>(), std::string("Noise covariance"));
    CHECK_EQ(st.options["source_file"].get<std::string>(), std::string("data/S01/rest/noisecov_full.mat"));
    CHECK_EQ(st.options["source_recording"].get<std::string>(), std::string("emptyroom_raw"));
    CHECK(st.options["history"].size() == 2);
    // its axes are the channels Selection (3 indices on the condition's channel manifold), twice
    CHECK(st.axes.size() == 2 && st.axes[0]["selection_id"] == st.axes[1]["selection_id"] && st.axes[0]["size"] == 3);
    const auto sel = sub.find("selection", "rest_noise_covariance_channels");
    CHECK(sel.has_value());
    if (sel) {
      CHECK_EQ(text(*sel, "id"), st.axes[0]["selection_id"].get<std::string>());
      CHECK_EQ(integer(*sel, "n_elements"), std::int64_t(3));
    }
  }
  // refused: a repeat (until --replace), a covariance the study does not have, a kind that is not one
  CHECK(thrown([&] { task::import_covariance(rq); }).find("--replace") != std::string::npos);
  rq.replace = true;
  CHECK(thrown([&] { task::import_covariance(rq); }).empty());
  rq.replace = false;
  rq.kind = "data";
  CHECK(thrown([&] { task::import_covariance(rq); }).find("no data covariance") != std::string::npos);
  return nxrtest::finish("test_import_covariance");
}
