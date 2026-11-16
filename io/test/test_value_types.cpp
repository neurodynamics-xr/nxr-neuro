// The value type `hermitian3` (a complex 3x3 Hermitian matrix a cell, float32 [.., 3, 3, 2]), added to the schema ADDITIVELY on 2026-11-12: a new
// dataset allows it; a dataset made BEFORE it (its stored CHECK lists without it — made here by putting the older text back) refuses such a row
// until the writer relaxes the lists in place on its first one: then the row is written, the older rows are untouched, the user_version is the
// same, the database passes its check, and `meta` notes it. A second write relaxes nothing.
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

#include "check.h"
#include "nxr/io.h"

using namespace nxr::io;
namespace fs = std::filesystem;

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_value_types_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const std::vector<double> P = {1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1};          // a tetrahedron: a closed surface of 4 vertices
  const std::vector<std::int32_t> F = {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2};
  std::vector<float> herm(4 * 3 * 3 * 2, 0.0f);
  for (int v = 0; v < 4; ++v) for (int i = 0; i < 3; ++i) herm[static_cast<size_t>(((v * 3 + i) * 3 + i) * 2)] = 1.0f + static_cast<float>(v);   // a real diagonal: Hermitian
  // a dataset with a subject, a surface and one scalar Field
  auto make = [&](const std::string& name, std::string& mid) {
    DatasetWriter ds = create_dataset(root, name);
    SubjectWriter sub = SubjectWriter::create(ds, "S");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; mid = add_surface(sub, so, P, F).id;
      const std::vector<float> m = {1, 2, 3, 4};
      FieldSpec f; f.name = "m"; f.path = "source/m"; f.kind = "map"; f.manifold_id = mid; f.data = view(m, {4});
      sub.field(std::move(f));
    });
    return ds;
  };
  auto field = [&](SubjectWriter& sub, const std::string& mid, const std::string& name) {
    FieldSpec f; f.name = name; f.path = "source/" + name; f.kind = "cross-spectral tensor"; f.manifold_id = mid; f.value_type = "hermitian3"; f.data = view(herm, {4, 3, 3, 2});
    return sub.field(std::move(f));
  };
  // A NEW dataset allows it: written, nothing relaxed
  {
    std::string mid;
    DatasetWriter ds = make("fresh", mid);
    SubjectWriter sub = SubjectWriter::open(ds, "S");
    const std::string id = field(sub, mid, "t");
    const auto row = sub.db().read("field", id);
    CHECK(row.has_value() && text(*row, "value_type") == "hermitian3" && integer(*row, "n_components") == 18);
    CHECK(!sub.db().one("SELECT value FROM meta WHERE key = 'value_type:hermitian3'").has_value());
    CHECK(!allow_value_type(sub.db(), "hermitian3"));
  }
  // A dataset made BEFORE it: no hermitian3 row, and the older CHECK text put back, as such a database stores it
  std::string mid;
  DatasetWriter ds = make("older", mid);
  {
    Database& db = ds.db();
    const std::int64_t v = integer(*db.one("PRAGMA schema_version"), "schema_version");
    db.with_writable_schema([&] {
      db.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name IN ('field', 'operator')", {V(std::string("'matrix3', 'quaternion', 'hermitian3')")), V(std::string("'matrix3', 'quaternion')"))});
      db.execute("PRAGMA schema_version = " + std::to_string(v + 1));
    });
    CHECK(text(*db.one("PRAGMA quick_check"), "quick_check") == "ok");
    CHECK(text(*db.one("SELECT sql FROM sqlite_master WHERE name = 'field'"), "sql").find("'hermitian3'") == std::string::npos);
  }
  const std::int64_t version = integer(*ds.db().one("PRAGMA user_version"), "user_version");
  const std::int64_t fields_before = integer(*ds.db().one("SELECT COUNT(*) AS n FROM field"), "n");
  // the older constraint bites: a row made hermitian3 around the writer is refused
  bool refused = false;
  try { ds.db().execute("UPDATE field SET value_type = 'hermitian3' WHERE name = 'm'"); } catch (const std::exception&) { refused = true; }
  CHECK(refused);
  // the writer relaxes it in place on its first hermitian3 row
  {
    SubjectWriter sub = SubjectWriter::open(ds, "S");
    const std::string id = field(sub, mid, "t");
    CHECK(sub.db().read("field", id).has_value());
    const auto note = sub.db().one("SELECT value FROM meta WHERE key = 'value_type:hermitian3'");
    CHECK(note.has_value() && text(*note, "value").find("allowed in place") != std::string::npos);
    CHECK(text(*sub.db().one("SELECT sql FROM sqlite_master WHERE name = 'operator'"), "sql").find("'quaternion', 'hermitian3')") != std::string::npos);
    CHECK(integer(*sub.db().one("PRAGMA user_version"), "user_version") == version);
    CHECK(integer(*sub.db().one("SELECT COUNT(*) AS n FROM field"), "n") == fields_before + 1);
    CHECK(text(*sub.db().one("SELECT value_type FROM field WHERE name = 'm'"), "value_type") == "scalar");   // the older rows untouched
    CHECK(text(*sub.db().one("PRAGMA quick_check"), "quick_check") == "ok");
    CHECK(!allow_value_type(sub.db(), "hermitian3"));                    // once: nothing left to relax
  }
  fs::remove_all(root);
  return nxrtest::finish("test_value_types");
}
