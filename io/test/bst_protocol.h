// A synthetic Brainstorm protocol, written with matio, for the tests of the importers (io) and of the import face (pipelines): a default anatomy (small octahedra),
// subject S01 with its own cortex and one condition `rest` (one imported recording of three EEG channels: channel.mat, data_rest.mat), and subject S02 with no condition.
// Synthetic data only. The test that includes it links matio.
#pragma once
#include <filesystem>
#include <string>
#include <vector>

extern "C" {
#include "matio.h"
}

namespace bst_protocol {
namespace fs = std::filesystem;

inline matvar_t* numeric(const char* name, const std::vector<double>& row_major, size_t rows, size_t cols) {
  std::vector<double> col(row_major.size());
  for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < cols; ++c) col[c * rows + r] = row_major[r * cols + c];
  size_t d[2] = {rows, cols};
  return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d, col.data(), 0);
}
inline matvar_t* empty(const char* name) {
  size_t d[2] = {0, 0};
  return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d, nullptr, 0);
}
inline matvar_t* chars(const char* name, const std::string& s) {
  size_t d[2] = {1, s.size()};
  return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, d, const_cast<char*>(s.data()), 0);
}
inline void write(mat_t* m, matvar_t* v) {
  Mat_VarWrite(m, v, MAT_COMPRESSION_NONE);
  Mat_VarFree(v);
}
// a Brainstorm cortex: two octahedra (12 vertices, metres)
inline void write_tess(const fs::path& p) {
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
inline void write_image(const fs::path& p) {
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
inline void write_condition(const fs::path& dir) {
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
inline fs::path protocol(const fs::path& root) {
  write_tess(root / "anat" / "@default_subject" / "tess_cortex_pial_low.mat");
  write_image(root / "anat" / "@default_subject" / "subjectimage_T1.mat");
  write_tess(root / "anat" / "S01" / "tess_cortex_pial_low.mat");
  write_condition(root / "data" / "S01" / "rest");
  write_tess(root / "anat" / "S02" / "tess_cortex_pial_low.mat");
  fs::create_directories(root / "data" / "S02");
  return root;
}

}  // namespace bst_protocol
