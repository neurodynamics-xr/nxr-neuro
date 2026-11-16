// The `bst` reading layer over matio: every class, real and complex, logical, char (UTF-8), struct arrays, cells, empty values — written here with matio
// itself (v5, compressed and not) and read back through nxr::bst; a v7.3 file (written by MATLAB, test/data/mat-v73) reads as its v7 twin. (test/parity compares with scipy-written files.)
#include "nxr/bst/mat.h"
#include "check.h"

extern "C" {
#include "matio.h"
}

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace nxr::bst;

namespace {
matvar_t* num(const char* name, matio_classes cls, matio_types type, std::vector<size_t> dims, void* data, int flags = 0) {
  return Mat_VarCreate(name, cls, type, static_cast<int>(dims.size()), dims.data(), data, flags);
}
matvar_t* chars(const char* name, const std::string& s) {
  size_t d[2] = {1, s.size()};
  return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, d, const_cast<char*>(s.data()), MAT_F_DONT_COPY_DATA);
}

void write_file(const fs::path& p, matio_compression comp) {
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, MAT_FT_MAT5);
  // a 3x2 double matrix, column-major: [1 4; 2 5; 3 6]
  double a[6] = {1, 2, 3, 4, 5, 6};
  Mat_VarWrite(m, num("A", MAT_C_DOUBLE, MAT_T_DOUBLE, {3, 2}, a), comp);
  std::int8_t i8[2] = {-5, 7}; std::uint8_t u8[2] = {200, 3}; std::int16_t i16[2] = {-300, 300}; std::uint16_t u16[2] = {65535, 1};
  std::int32_t i32[2] = {-70000, 70000}; std::uint32_t u32[2] = {4294967295u, 1}; std::int64_t i64[2] = {-4611686018427387905LL, 4611686018427387905LL};
  std::uint64_t u64[2] = {18446744073709551615ULL, 1}; float f32[2] = {1.5f, -2.25f};
  Mat_VarWrite(m, num("i8", MAT_C_INT8, MAT_T_INT8, {1, 2}, i8), comp);
  Mat_VarWrite(m, num("u8", MAT_C_UINT8, MAT_T_UINT8, {1, 2}, u8), comp);
  Mat_VarWrite(m, num("i16", MAT_C_INT16, MAT_T_INT16, {1, 2}, i16), comp);
  Mat_VarWrite(m, num("u16", MAT_C_UINT16, MAT_T_UINT16, {1, 2}, u16), comp);
  Mat_VarWrite(m, num("i32", MAT_C_INT32, MAT_T_INT32, {1, 2}, i32), comp);
  Mat_VarWrite(m, num("u32", MAT_C_UINT32, MAT_T_UINT32, {1, 2}, u32), comp);
  Mat_VarWrite(m, num("i64", MAT_C_INT64, MAT_T_INT64, {1, 2}, i64), comp);
  Mat_VarWrite(m, num("u64", MAT_C_UINT64, MAT_T_UINT64, {1, 2}, u64), comp);
  Mat_VarWrite(m, num("f32", MAT_C_SINGLE, MAT_T_SINGLE, {1, 2}, f32), comp);
  double re[2] = {1, 2}, im[2] = {-1, 0.5};
  mat_complex_split_t z = {re, im};
  Mat_VarWrite(m, num("Z", MAT_C_DOUBLE, MAT_T_DOUBLE, {1, 2}, &z, MAT_F_COMPLEX), comp);
  std::uint8_t lg[3] = {1, 0, 1};
  Mat_VarWrite(m, num("L", MAT_C_UINT8, MAT_T_UINT8, {1, 3}, lg, MAT_F_LOGICAL), comp);
  Mat_VarWrite(m, chars("S", "Cortex L"), comp);
  Mat_VarWrite(m, chars("E", ""), comp);
  size_t zero[2] = {0, 0};
  Mat_VarWrite(m, Mat_VarCreate("N", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, zero, nullptr, 0), comp);

  // a 1x2 struct array {Name, V, Sub}
  const char* fn[3] = {"Name", "V", "Sub"};
  size_t d12[2] = {1, 2};
  matvar_t* st = Mat_VarCreateStruct("St", 2, d12, fn, 3);
  double v0 = 10, v1 = 20;
  Mat_VarSetStructFieldByName(st, "Name", 0, chars(nullptr, "first"));
  Mat_VarSetStructFieldByName(st, "Name", 1, chars(nullptr, "second"));
  Mat_VarSetStructFieldByName(st, "V", 0, num(nullptr, MAT_C_DOUBLE, MAT_T_DOUBLE, {1, 1}, &v0));
  Mat_VarSetStructFieldByName(st, "V", 1, num(nullptr, MAT_C_DOUBLE, MAT_T_DOUBLE, {1, 1}, &v1));
  const char* sf[1] = {"X"};
  size_t one[2] = {1, 1};
  matvar_t* sub = Mat_VarCreateStruct(nullptr, 2, one, sf, 1);
  double xx = 3;
  Mat_VarSetStructFieldByName(sub, "X", 0, num(nullptr, MAT_C_DOUBLE, MAT_T_DOUBLE, {1, 1}, &xx));
  Mat_VarSetStructFieldByName(st, "Sub", 0, sub);
  Mat_VarWrite(m, st, comp);

  // a 1x2 cell {'MEG', [1 2 3]}
  matvar_t** cells = static_cast<matvar_t**>(std::calloc(3, sizeof(matvar_t*)));
  double three[3] = {1, 2, 3};
  cells[0] = chars(nullptr, "MEG");
  cells[1] = num(nullptr, MAT_C_DOUBLE, MAT_T_DOUBLE, {1, 3}, three);
  Mat_VarWrite(m, Mat_VarCreate("C", MAT_C_CELL, MAT_T_CELL, 2, d12, cells, 0), comp);
  Mat_Close(m);
}
}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() / "nxr_bst_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  for (const bool compressed : {true, false}) {
    const fs::path p = dir / (compressed ? "v7.mat" : "v6.mat");
    write_file(p, compressed ? MAT_COMPRESSION_ZLIB : MAT_COMPRESSION_NONE);
    MatFile f = MatFile::open(p);
    CHECK_EQ(f.version(), 5);
    const auto names = f.list();
    CHECK_EQ(names.size(), static_cast<std::size_t>(17));
    CHECK_EQ(names.front().name, std::string("A"));
    CHECK(names.front().cls == Class::Double && names.front().dims == (std::vector<std::size_t>{3, 2}));
    CHECK(!f.read("nothing").has_value());

    const Value A = *f.read("A");
    CHECK_EQ(A.rows(), static_cast<std::size_t>(3));
    CHECK_EQ(A.cols(), static_cast<std::size_t>(2));
    CHECK((A.doubles() == std::vector<double>{1, 2, 3, 4, 5, 6}));             // as stored: column after column
    CHECK((A.row_major() == std::vector<double>{1, 4, 2, 5, 3, 6}));           // row after row
    CHECK((f.read("i8")->int64s() == std::vector<std::int64_t>{-5, 7}) && f.read("i8")->cls == Class::Int8);
    CHECK((f.read("u8")->int64s() == std::vector<std::int64_t>{200, 3}) && f.read("u8")->cls == Class::UInt8);
    CHECK((f.read("i16")->int64s() == std::vector<std::int64_t>{-300, 300}) && f.read("i16")->cls == Class::Int16);
    CHECK((f.read("u16")->int64s() == std::vector<std::int64_t>{65535, 1}) && f.read("u16")->cls == Class::UInt16);
    CHECK((f.read("i32")->int64s() == std::vector<std::int64_t>{-70000, 70000}) && f.read("i32")->cls == Class::Int32);
    CHECK((f.read("u32")->int64s() == std::vector<std::int64_t>{4294967295LL, 1}) && f.read("u32")->cls == Class::UInt32);
    CHECK((f.read("i64")->int64s() == std::vector<std::int64_t>{-4611686018427387905LL, 4611686018427387905LL}) && f.read("i64")->cls == Class::Int64);
    CHECK(f.read("u64")->cls == Class::UInt64 && f.read("u64")->re.size() == 16);
    CHECK((f.read("f32")->doubles() == std::vector<double>{1.5, -2.25}) && f.read("f32")->cls == Class::Single);
    const Value Z = *f.read("Z");
    CHECK(Z.is_complex && (Z.doubles() == std::vector<double>{1, 2}) && (Z.imag() == std::vector<double>{-1, 0.5}));
    const Value L = *f.read("L");
    CHECK(L.is_logical && L.cls == Class::UInt8 && (L.int64s() == std::vector<std::int64_t>{1, 0, 1}));
    CHECK_EQ(f.read("S")->string(), std::string("Cortex L"));
    CHECK(f.read("E")->is_char() && f.read("E")->string().empty());
    CHECK(f.read("N")->empty() && f.read("N")->cls == Class::Double);

    const Value St = *f.read("St");
    CHECK(St.is_struct() && St.numel() == 2 && St.fields == (std::vector<std::string>{"Name", "V", "Sub"}));
    CHECK_EQ(St.field("Name", 1)->string(), std::string("second"));
    CHECK(St.field("V", 0)->scalar() == 10 && St.field("V", 1)->scalar() == 20);
    CHECK(St.field("Sub", 0)->is_struct() && St.field("Sub", 0)->field("X")->scalar() == 3);
    CHECK(St.field("Nope") == nullptr && St.field("Name", 2) == nullptr);

    const Value C = *f.read("C");
    CHECK(C.is_cell() && C.numel() == 2 && C.cell(0).string() == "MEG" && (C.cell(1).doubles() == std::vector<double>{1, 2, 3}));

    const auto all = f.read_all();
    CHECK_EQ(all.size(), static_cast<std::size_t>(17));
    const auto some = read_variables(p, {"A", "S", "missing"});
    CHECK(some.size() == 2 && some.count("A") && some.count("S"));
  }

  // v7.3 (HDF5), written by MATLAB itself (test/data/mat-v73/make_v73_smoke.m): read through the same Values; the whole twin comparison is the
  // ctest nxr-io-bst-compare
  {
    const fs::path d = fs::path(NXR_IO_DATA_DIR) / "mat-v73";
    CHECK(is_v73(d / "t73.mat"));
    CHECK(!is_v73(d / "t7.mat"));
    MatFile f = MatFile::open(d / "t73.mat");
    CHECK_EQ(f.version(), 73);
    CHECK_EQ(f.list().size(), static_cast<std::size_t>(24));
    const Value tess = *f.read("tess");
    CHECK(tess.is_struct());
    CHECK_EQ(tess.field("Vertices")->dims[0], static_cast<std::size_t>(500));
    const Value* atlas = tess.field("Atlas");
    CHECK_EQ(atlas->numel(), static_cast<std::size_t>(3));
    CHECK_EQ(atlas->field("Name", 0)->string(), std::string("Desikan"));
    CHECK_EQ(atlas->field("Scouts", 0)->numel(), static_cast<std::size_t>(2));
    CHECK_EQ(atlas->field("Scouts", 1)->numel(), static_cast<std::size_t>(0));
    CHECK_EQ(atlas->field("Scouts", 0)->field("Label", 1)->string(), std::string("B"));
    CHECK(f.read("lg")->is_logical);
    CHECK(f.read("cx")->is_complex);
    CHECK_EQ(f.read("ec")->string(), std::string(""));
    CHECK_EQ(f.read("big")->cols(), static_cast<std::size_t>(200));
    CHECK_EQ(f.read("Comment") ? 0 : 1, 1);   // absent in this file
    // a MATLAB string / datetime / table is stored opaquely: refused by name
    MatFile o = MatFile::open(d / "opaque73.mat");
    CHECK(o.read("sc").has_value());
    for (const char* name : {"s", "dt", "tb"}) {
      try { o.read(name); CHECK(false); } catch (const Error& e) {
        CHECK_EQ(e.code(), std::string("unsupported"));
        CHECK(std::string(e.what()).find(std::string("'") + name + "'") != std::string::npos);
        CHECK(std::string(e.what()).find("opaque") != std::string::npos);
      }
    }
    CHECK(o.read_all().size() == 4);
  }

  // refusals: an HDF5 signature without MATLAB's variables, a missing file, something that is no MAT-file
  {
    const fs::path q = dir / "h5.mat";
    std::ofstream(q, std::ios::binary) << std::string("\x89HDF\r\n\x1a\n", 8) << std::string(200, '\0');
    CHECK(is_v73(q));
    try { MatFile::open(q); CHECK(false); } catch (const Error& e) { CHECK_EQ(e.code(), std::string("invalid_argument")); }
    try { MatFile::open(dir / "none.mat"); CHECK(false); } catch (const Error& e) { CHECK_EQ(e.code(), std::string("not_found")); }
    const fs::path r = dir / "text.mat";
    std::ofstream(r) << "this is not a MAT-file\n";
    CHECK(!is_v73(r));
    try { MatFile::open(r); CHECK(false); } catch (const Error& e) { CHECK_EQ(e.code(), std::string("invalid_argument")); }
  }
  fs::remove_all(dir);
  return nxrtest::finish("bst");
}
