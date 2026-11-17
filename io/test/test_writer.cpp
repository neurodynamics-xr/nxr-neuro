// Writer unit tests on synthetic data: the procedure, errors with their codes, the layout rule, the delete rule (D164), the array policy.
// (The comparison with nxr's Python is test/parity/test_write_parity.py.)
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "check.h"
#include "store_at.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/dataset.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}

ManifoldSpec tri(const std::string& name) {
  ManifoldSpec m;
  m.name = name;
  m.path = "source/" + name;
  m.type = "surface";
  m.n_vertices = 4;
  m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/" + name + "/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = "source/" + name + "/vertices";
  m.geometry = g;
  return m;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_writer_" + std::to_string(::getpid()));
  fs::remove_all(root);

  // errors: one type, a stable code, nxr's wording
  CHECK_EQ(code_of([&] { create_dataset(root, "../x"); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { open_writer(root / "none"); }), std::string(code::not_found));
  {
    DatasetWriter ds = create_dataset(root, "demo");
    CHECK_EQ(code_of([&] { create_dataset(root, "demo"); }), std::string(code::already_exists));
    CHECK_EQ(code_of([&] { open_writer(root / "demo"); }), std::string(code::writer_lock_held));   // we hold it
    CHECK_EQ(code_of([&] { SubjectWriter::open(ds, "nobody"); }), std::string(code::no_such_row));
    CHECK_EQ(code_of([&] { delete_dataset(root, "demo"); }), std::string(code::writer_lock_held));
    try { SubjectWriter::open(ds, "nobody"); } catch (const Error& e) { CHECK_EQ(std::string(e.what()), std::string("nobody: no subject of demo")); CHECK_EQ(e.python_name(), std::string("KeyError")); }

    // the procedure: bytes first, the row last; pending -> complete
    SubjectWriter sub = SubjectWriter::create(ds, "S");
    CHECK_EQ(text(*ds.db().one("SELECT status FROM subject"), "status"), std::string("pending"));
    const std::vector<double> v(12, 1.0);
    const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
    ManifoldSpec m = tri("m");
    m.populate = [&](Store&, const std::string&) {
      sub.write_array("source/m/vertices", view(v, {4, 3}));
      sub.write_array("source/m/faces", view(f, {2, 3}));
    };
    const std::string mid = sub.manifold(m);
    CHECK_EQ(text(*ds.db().read("manifold", mid), "status"), std::string("pending"));
    CHECK(sub.store().exists("source/m/vertices/zarr.json"));
    sub.complete();
    CHECK_EQ(text(*ds.db().read("manifold", mid), "status"), std::string("complete"));
    CHECK_EQ(text(*ds.db().read("subject", sub.id()), "status"), std::string("complete"));
    CHECK_EQ(text_parity(ds.db(), ds.store()).mismatched, 0);
    CHECK_EQ(folder_of(ds.db(), mid), std::string("source"));

    // a failed composition rolls back what it created
    {
      SubjectWriter s2 = SubjectWriter::open(ds, "S");
      const std::string before = text(*ds.db().one("SELECT count(*) AS n FROM manifold"), "n");
      bool threw = false;
      try {
        s2.compose([&] {
          ManifoldSpec x = tri("x");
          x.populate = [&](Store&, const std::string&) { s2.write_array("source/x/vertices", view(v, {4, 3})); };
          s2.manifold(x);
          ManifoldSpec y = tri("y");
          y.populate = [](Store&, const std::string&) { throw Error(code::io, "boom"); };
          s2.manifold(y);
        });
      } catch (const Error&) { threw = true; }
      CHECK(threw);
      CHECK_EQ(text(*ds.db().one("SELECT count(*) AS n FROM manifold"), "n"), before);
      CHECK(!s2.store().exists_prefix("source/x"));
    }

    // D164: an array another row names stays; a field nobody else names goes
    SubjectWriter s3 = SubjectWriter::open(ds, "S");
    const std::vector<float> a(4, 2.0f);
    for (const char* n : {"a", "c"}) {
      FieldSpec fs_;
      fs_.name = n;
      fs_.path = std::string("source/") + n;
      fs_.kind = "map";
      fs_.manifold_id = mid;
      fs_.data = view(a, {4});
      s3.field(fs_);
    }
    SelectionSpec w;
    w.name = "w";
    w.path = "source/w";
    w.type = "kernel";
    w.manifold_id = mid;
    w.cols.set("weight_fn", "heat").set("params_json", "{\"weights\":\"source/a\"}");
    s3.selection(w);
    s3.complete();
    remove_node(ds.db(), ds.store(), "field", text(*s3.find("field", "source/c"), "id"));
    CHECK(!s3.store().exists_prefix("source/c"));
    remove_node(ds.db(), ds.store(), "field", text(*s3.find("field", "source/a"), "id"));
    CHECK(s3.store().exists("source/a/zarr.json"));
    CHECK(!s3.find("field", "source/a").has_value());

    // recovery: a pending row is rolled back, a deleting one finished
    {
      SubjectWriter s4 = SubjectWriter::open(ds, "S");
      FieldSpec p;
      p.name = "p";
      p.path = "source/p";
      p.kind = "map";
      p.manifold_id = mid;
      p.data = view(a, {4});
      s4.field(p);  // never completed
      CHECK_EQ(recover(ds.db(), ds.store()).rolled_back, 1);
      CHECK(!s4.store().exists_prefix("source/p"));
    }
    ds.close();
  }
  CHECK(!fs::exists(root / "demo" / kWriterLockFile));

  // the array policy
  CHECK_EQ(default_chunks({100, 10}, 4), (Shape{100, 10}));
  CHECK_EQ(default_chunks({2562, 1000}, 4), (Shape{513, 1000}));
  CHECK_EQ(default_chunks({0, 3}, 4), (Shape{1, 3}));
  const ArrayMeta big = array_meta(DType::Float32, {2562, 1000}, {.unsharded = true});
  const ArrayMeta big_default = array_meta(DType::Float32, {2562, 1000});   // format 51: over 4 MiB a shard of the 513-row chunks (one file here)
  CHECK_EQ(big_default.chunks, (Shape{513, 1000}));
  CHECK_EQ(big_default.grid, (Shape{2565, 1000}));
  CHECK_EQ(std::string(big_default.codecs[0]["name"]), std::string("sharding_indexed"));
  CHECK_EQ(big_default.separator, std::string("."));
  CHECK_EQ(js_json(big.codecs), std::string("[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},{\"name\":\"blosc\",\"configuration\":{\"typesize\":4,\"cname\":\"zstd\",\"clevel\":3,\"shuffle\":\"shuffle\",\"blocksize\":0}}]"));
  CHECK_EQ(js_json(array_meta(DType::UInt8, {3}).codecs), std::string("[{\"name\":\"bytes\"}]"));
  CHECK_EQ(js_json(array_meta(DType::Bool, {3}).fill_value), std::string("false"));
  CHECK_EQ(js_json(array_meta(DType::Complex64, {3}).fill_value), std::string("[0,0]"));
  CHECK_EQ(code_of([&] { array_meta(DType::Float32, {10, 10}, {.chunks = Shape{3, 3}, .shards = Shape{10, 10}}); }), std::string(code::invalid_argument));

  // numbers and layout
  CHECK_EQ(js_number(1e-7), std::string("1e-7"));
  CHECK_EQ(js_number(1e21), std::string("1e+21"));
  CHECK_EQ(js_number(0.1 + 0.2), std::string("0.30000000000000004"));
  CHECK_EQ(js_number(-0.0), std::string("0"));
  CHECK_EQ(folder_for("surface", "scalp"), std::string("mri"));
  CHECK_EQ(folder_for("points", "", std::nullopt, std::string("laplace")), std::string("eigenvalue"));
  CHECK_EQ(safe_name("  cortex pial/tiles L8 "), std::string("cortex_pial_tiles_L8"));
  CHECK_EQ(sanitize_node_name("tess cortex.pial"), std::string("tess_cortex.pial"));
  const FVector pc = product_counts({{2562, 7680, 5120, std::nullopt, 2}, {100, 99, std::nullopt, std::nullopt, 1}});
  CHECK_EQ(pc.rank, 3);
  CHECK_EQ(*pc.n_vertices, 256200);
  CHECK_EQ(bids_entities("ses-02_task-rest_run-02")["run"].value(), std::string("02"));
  CHECK(!bids_entities("N2")["ses"].has_value());

  // the schema is the one embedded
  CHECK_EQ(schema_version(), 53);

  fs::remove_all(root);
  // ── WRITE A REGION (write_region): read-modify-write of the chunks a region touches, against the whole array built in memory ──
  {
    const fs::path at = root / "region.zarr";
    FolderStore fst(root);
    ArrayOptions ao;
    ao.chunks = Shape{2, 3, 2};
    layout_array(fst, "region.zarr", array_meta(DType::Float32, {5, 7, 3}, ao));
    std::vector<float> want(5 * 7 * 3, 0.0f);
    auto put = [&](Shape s, Shape n, float base) {
      std::vector<float> v(static_cast<std::size_t>(n[0] * n[1] * n[2]));
      for (std::size_t i = 0; i < v.size(); ++i) v[i] = base + static_cast<float>(i);
      write_region(fst, "region.zarr", s, view(v, n));
      for (std::int64_t a = 0; a < n[0]; ++a) for (std::int64_t b = 0; b < n[1]; ++b) for (std::int64_t c = 0; c < n[2]; ++c)
        want[static_cast<std::size_t>(((s[0] + a) * 7 + (s[1] + b)) * 3 + (s[2] + c))] = v[static_cast<std::size_t>((a * n[1] + b) * n[2] + c)];
    };
    auto got = [&] {
      const auto raw = nxrtest::open_array(at).read();
      std::vector<float> v(raw.size() / 4);
      std::memcpy(v.data(), raw.data(), raw.size());
      return v;
    };
    put({0, 0, 0}, {5, 7, 3}, 1.0f);                // the whole array (populate_array)
    CHECK(got() == want);
    put({1, 2, 1}, {3, 4, 2}, 500.0f);              // across chunk boundaries on every axis
    CHECK(got() == want);
    put({4, 6, 0}, {1, 1, 3}, 900.0f);              // the edge chunks (partial)
    CHECK(got() == want);
    put({0, 0, 0}, {5, 2, 3}, 0.0f);                // a span along axis 1 (the Studio's [[0,K],[f0,f1],[0,4]] form), first value 0
    CHECK(got() == want);
    // a chunk made all fill (zero) is removed, not left stale: chunk (0,0,0) covers [0,2)x[0,3)x[0,2)
    std::vector<float> z(2 * 3 * 2, 0.0f);
    write_region(fst, "region.zarr", {0, 0, 0}, view(z, {2, 3, 2}));
    for (std::int64_t a = 0; a < 2; ++a) for (std::int64_t b = 0; b < 3; ++b) for (std::int64_t c = 0; c < 2; ++c) want[static_cast<std::size_t>((a * 7 + b) * 3 + c)] = 0.0f;
    CHECK(got() == want);
    CHECK(!fs::exists(at / "c.0.0.0"));
    // refusals: outside the array; another type
    std::vector<float> one(1, 1.0f);
    CHECK_EQ(code_of([&] { write_region(fst, "region.zarr", {5, 0, 0}, view(one, {1, 1, 1})); }), std::string(code::invalid_argument));
    std::vector<double> d1(1, 1.0);
    CHECK_EQ(code_of([&] { write_region(fst, "region.zarr", {0, 0, 0}, view(d1, {1, 1, 1})); }), std::string(code::invalid_argument));
    // a sharded array: refused
    const std::string sh = "sharded.zarr";
    ArrayOptions so;
    so.chunks = Shape{2, 2};
    so.shards = Shape{4, 4};
    layout_array(fst, sh, array_meta(DType::Float32, {8, 8}, so));
    std::vector<float> two(4, 1.0f);
    CHECK_EQ(code_of([&] { write_region(fst, sh, {0, 0}, view(two, {2, 2})); }), std::string(code::unsupported));
  }

  return nxrtest::finish("test_writer");
}
