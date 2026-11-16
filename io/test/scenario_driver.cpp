// nxr-io-scenario — the conformance scenario written through nxr-io's API, with deterministic ids and clock, to be compared with the same
// scenario written through nxr's Python (test/parity/test_write_parity.py). Also holds a writer lock and leaves an interrupted write, for the lock
// and recovery checks.
//
//   nxr-io-scenario scenario  <datastore> <mesh.obj>      dataset -> subject -> surface -> fields -> selection -> operator
//   nxr-io-scenario interrupt <dataset folder> <subject>  a field written and never completed (rows pending, bytes on disk)
//   nxr-io-scenario hold      <dataset folder>            take the writer lock, print "locked", wait for a line on stdin
//   nxr-io-scenario jsnum                                 js_number of each double read from stdin (one per line)
//   nxr-io-scenario remove    <dataset folder> <subject> <table> <path>   delete one node by its path
#include <cstdlib>
#include <iostream>

#include "nxr/io/surface.h"
#include "nxr/io/compact.h"
#include "nxr/io/writer.h"

using namespace nxr::io;

namespace {

std::vector<float> ramp(std::int64_t n, std::int64_t zero_from = -1, std::int64_t zero_to = -1, std::int64_t row_len = 1) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    const std::uint64_t x = (static_cast<std::uint64_t>(i) * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    v[static_cast<std::size_t>(i)] = static_cast<float>(x >> 20) / 8.0f;
    const std::int64_t row = i / row_len;
    if (zero_from >= 0 && row >= zero_from && row < zero_to) v[static_cast<std::size_t>(i)] = 0.0f;
  }
  return v;
}

int scenario(const std::string& datastore, const std::string& obj) {
  testing::deterministic(true);
  CreateOptions co;
  co.source_tool = "parity-tool";
  co.source_protocol = "p";
  co.source_path = "s";
  co.created_by = "parity";
  // the parity tests and the byte-identity goldens are format 51 (the folder tree: what nxr in Python and the Studio read until they take format 52); NXR_SCENARIO_STORE_FORMAT=52 writes format 52
  const char* fmt = std::getenv("NXR_SCENARIO_STORE_FORMAT");
  co.store_format = fmt ? std::atoi(fmt) : 51;
  DatasetWriter ds = create_dataset(datastore, "demo", co);
  SubjectOptions so;
  so.created_by = "parity";
  SubjectWriter sub = SubjectWriter::create(ds, "S1", so);

  const Mesh mesh = read_mesh(obj);
  const std::int64_t nv = mesh.n_vertices();
  SurfaceOptions sfo;
  sfo.name = "sphere";
  sfo.source = "sphere.obj";
  const SurfaceInfo info = add_surface(sub, sfo, mesh.vertices, mesh.faces);
  const std::string sid = info.id;

  const std::string label = "ses-01_task-rest_run-01";
  sub.acquisition(label);

  // a time line (closed form) and the product sphere x time
  ManifoldSpec t;
  t.name = "t";
  t.path = "time/t";
  t.type = "line";
  t.n_vertices = 100;
  t.topology = {"path", 1, 1, std::nullopt, {}};
  GeometrySpec tg;
  tg.form = "closed_form";
  tg.cs_key = "clock";
  DimensionSpec td;
  td.name = "time";
  td.n_vertices = 100;
  td.spacing = 0.001;
  td.origin = 0.0;
  td.unit = "s";
  tg.dimensions.push_back(td);
  t.geometry = tg;
  t.cols.set("n_edges", 99);
  const std::string tid = sub.manifold(t);
  sub.product({sid, tid});

  // fields
  const auto small = ramp(nv);
  FieldSpec f;
  f.name = "small";
  f.path = "source/small";
  f.kind = "map";
  f.manifold_id = sid;
  f.data = view(small, {nv});
  f.acquisition = label;
  const std::string small_id = sub.field(f);

  auto big = ramp(nv * 1000, 1000, 1500, 1000);
  FieldSpec b;
  b.name = "big";
  b.path = "source/big";
  b.kind = "map";
  b.manifold_id = sid;
  b.data = view(big, {nv, 1000});
  sub.field(b);

  FieldSpec bs = b;
  bs.name = "big_sharded";
  bs.path = "source/big_sharded";
  bs.meta = array_meta(DType::Float32, {nv, 1000}, {.chunks = Shape{256, 500}, .shards = Shape{1024, 1000}});
  sub.field(bs);

  std::vector<double> sm(300 * 200);
  for (std::size_t i = 0; i < sm.size(); ++i) sm[i] = (i % 61 == 0) ? 0.0 : static_cast<double>(i) * 0.5;
  FieldSpec ss;
  ss.name = "small_sharded";
  ss.path = "source/small_sharded";
  ss.kind = "map";
  ss.manifold_id = sid;
  ss.data = view(sm, {300, 200});
  ss.meta = array_meta(DType::Float64, {300, 200}, {.chunks = Shape{100, 100}, .shards = Shape{300, 200}});
  sub.field(ss);

  std::vector<std::uint8_t> zeros(50, 0);
  FieldSpec dz;
  dz.name = "dense_zeros";
  dz.path = "source/dense_zeros";
  dz.kind = "map";
  dz.manifold_id = sid;
  dz.value_type = "integer";
  dz.data = view(zeros, {50});
  dz.array.dense = true;
  sub.field(dz);

  // a sparse (CSC) field
  const std::vector<std::int32_t> indptr = {0, 2, 3, 3, 6, 7}, indices = {0, 3, 1, 0, 2, 4, 3};
  const std::vector<float> values = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f};
  FieldSpec sp;
  sp.name = "sparse";
  sp.path = "source/sparse";
  sp.kind = "map";
  sp.manifold_id = sid;
  sp.sparse = SparseArrays{{{"data", view(values, {7})}, {"indices", view(indices, {7})}, {"indptr", view(indptr, {6})}}};
  sub.field(sp);

  // a dictionary and a selection coded by it, with a measurement
  std::vector<std::int32_t> codes(static_cast<std::size_t>(nv));
  for (std::int64_t i = 0; i < nv; ++i) codes[static_cast<std::size_t>(i)] = static_cast<std::int32_t>((i / 100) % 5);
  DictionarySpec dsp;
  dsp.name = "parcels";
  dsp.names = {"alpha", "beta", "gamma", "delta", "epsilon"};
  const DictionaryResult dict = sub.dictionary(dsp);
  SelectionSpec sel;
  sel.name = "parcels";
  sel.path = "source/parcels";
  sel.type = "set";
  sel.manifold_id = sid;
  sel.data = view(codes, {nv});
  sel.cols.set("dictionary_id", dict.id).set("cell", "0");
  sel.measurements.push_back(Cols().set("code", 1).set("of_field_id", small_id).set("measure", "mean").set("value", 12.5).set("unit", "a.u."));
  sub.selection(sel);

  // a dense operator
  std::vector<double> op(12);
  for (std::size_t i = 0; i < op.size(); ++i) op[i] = static_cast<double>(i) * 0.5;
  OperatorSpec o;
  o.name = "affine";
  o.path = "affine";
  o.kind = "transition";
  o.from_manifold_id = sid;
  o.n_rows = 3;
  o.n_cols = 4;
  o.data = view(op, {3, 4});
  sub.operator_(o);

  sub.complete();

  // a second composition on the existing subject
  SubjectWriter again = SubjectWriter::open(ds, "S1", {.created_by = "parity"});
  FieldSpec ex;
  ex.name = "extra";
  ex.path = "source/extra";
  ex.kind = "map";
  ex.manifold_id = sid;
  ex.data = view(small, {nv});
  again.field(ex);
  again.complete();
  ds.close();
  return 0;
}

int interrupt(const std::string& folder, const std::string& subject) {
  testing::deterministic(true, 1000);
  DatasetWriter ds = open_writer(folder);
  SubjectWriter sub = SubjectWriter::open(ds, subject, {.created_by = "parity"});
  const std::string sid = text(*sub.find("manifold", "source/sphere"), "id");
  const auto v = ramp(100);
  FieldSpec f;
  f.name = "interrupted";
  f.path = "source/interrupted";
  f.kind = "map";
  f.manifold_id = sid;
  f.data = view(v, {100});
  sub.field(f);
  ds.close();  // never completed
  return 0;
}

int hold(const std::string& folder) {
  WriterLock lock = acquire_writer_lock(folder);
  std::cout << "locked" << std::endl;
  std::string line;
  std::getline(std::cin, line);
  return 0;
}

int jsnum() {
  std::string line;
  while (std::getline(std::cin, line)) std::cout << js_number(std::strtod(line.c_str(), nullptr)) << '\n';
  return 0;
}

int remove(const std::string& folder, const std::string& subject, const std::string& table, const std::string& path) {
  DatasetWriter ds = open_writer(folder);
  SubjectWriter sub = SubjectWriter::open(ds, subject);
  const auto r = sub.find(table, path);
  if (!r) { std::cerr << "no such node\n"; return 1; }
  remove_node(ds.db(), ds.store(), table, text(*r, "id"));
  return 0;
}

}  // namespace

// `dataset compact` on the dataset folder, with nothing recovered (the interrupted node of the scenario stays as it is)
int compact(const std::string& dataset_folder) {
  OpenOptions oo;
  oo.recover = false;
  DatasetWriter ds = open_writer(dataset_folder, oo);
  CompactOptions o;
  o.small_files = false;   // format 50 -> 51 (the goldens are format 51 trees)
  const Json r = compact_dataset(ds, o);
  if (r["compacted"].get<int>() == 0) return 3;
  return 0;
}

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "scenario" && argc == 4) return scenario(argv[2], argv[3]);
    if (mode == "interrupt" && argc == 4) return interrupt(argv[2], argv[3]);
    if (mode == "compact" && argc == 3) return compact(argv[2]);
    if (mode == "jsnum" && argc == 2) return jsnum();
    if (mode == "hold" && argc == 3) return hold(argv[2]);
    if (mode == "remove" && argc == 6) return remove(argv[2], argv[3], argv[4], argv[5]);
    std::cerr << "usage: nxr-io-scenario scenario|interrupt|hold|remove ...\n";
    return 2;
  } catch (const Error& e) {
    std::cerr << e.python_name() << ": " << e.what() << '\n';
    return 1;
  }
}
