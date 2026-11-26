// Writes in pieces, deletes by class and jobs (docs/specs/2026-11-20-data-layer-queries.md, step 3: 3.7 and 3.8), through the commands' library functions on a synthetic dataset.
//   - `add field | statistic` in pieces: a round trip byte for byte against the one-shot command, invisible before the commit, rolled back by `recover dataset` and by the close of the handle that began it,
//     carried by a handle, and every refusal (a region outside the shape, a region twice or overlapping, a commit with regions missing, an unknown write, another handle's write, a wrong dtype or shape);
//   - `delete subject --node CLASS:NAME`: what a node owns goes (its arrays), what depends on it refuses the delete and is named, an ambiguous name is refused naming the CLASS:NAME forms;
//   - `run` and `import` leave a job row (done, and failed), `list jobs` reads it back.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <functional>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/migrate.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/handle.h"
#include "nxr/pipelines/face.h"

using namespace nxr::io;
namespace face = nxr::pipelines::face;
namespace fs = std::filesystem;

namespace {
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
template <class T>
std::vector<std::uint8_t> raw(const std::vector<T>& v) {
  std::vector<std::uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}
template <class T>
task::ArrayInput mem(const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<T>& v) {
  return task::ArrayInput::memory("values.npy", dtype, shape, raw(v));
}
template <class T>
std::string npy(const fs::path& file, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<T>& v) {
  write_npy(file, dtype, shape, raw(v));
  return file.string();
}
const Json* named(const Json& rows, const std::string& name) {
  for (const auto& r : rows) if (r["name"] == name) return &r;
  return nullptr;
}
// the rows [r0, r1) x columns [c0, c1) of the 3 x 50 values
std::vector<float> block(const std::vector<float>& all, int r0, int r1, int c0, int c1) {
  std::vector<float> out;
  for (int r = r0; r < r1; ++r) for (int c = c0; c < c1; ++c) out.push_back(all[static_cast<std::size_t>(r * 50 + c)]);
  return out;
}
int exit_of(const std::string& command) {
  const int r = std::system(command.c_str());
  return WIFEXITED(r) ? WEXITSTATUS(r) : -1;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_pieces_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "in");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);
  const std::string D = folder.string(), by = face::stamp();
  auto base = [&](auto& r, const std::string& name) { r.dataset = D; r.subject = "S1"; r.name = name; r.created_by = by; };
  auto read = [&]() { task::ReadSubjectRequest q; q.dataset = D; q.subject = "S1"; return face::read_subject(q).result; };
  auto list = [&](const std::string& cls, const std::function<void(task::ListClassRequest&)>& set = {}) {
    task::ListClassRequest q;
    q.cls = cls;
    q.dataset = D;
    q.subject = "S1";
    if (set) set(q);
    return face::list_class(q).result["rows"];
  };
  auto reading = [&](const std::string& name, bool all = true) { face::ReadNodeRequest r; r.dataset = D; r.subject = "S1"; r.name = name; r.all = all; return r; };
  auto del = [&](const std::string& node) { task::DeleteSubjectRequest q; q.dataset = D; q.subject = "S1"; q.node = node; q.created_by = by; return task::delete_subject(q); };
  auto arrays_of = [&]() { task::ReadDatasetRequest q; q.folder = D; return task::read_dataset(q).result["arrays"]; };
  auto count_arrays = [&](const std::string& part) { std::size_t n = 0; for (const auto& a : arrays_of()) if (a["path"].get<std::string>().find(part) != std::string::npos) ++n; return n; };
  auto array_there = [&](const std::string& path) { task::ReadDatasetRequest q; q.folder = D; q.array = std::array<std::string, 2>{"S1", path}; return thrown([&] { task::read_dataset(q); }).empty(); };
  auto writes = [&]() { std::size_t n = 0; std::error_code ec; for (const auto& e : fs::directory_iterator(folder / ".writes", ec)) n += e.is_directory(); return n; };

  // ---- what the writes lie on: a time line (50 samples), the values of a time-resolved field on the 3 sensors
  {
    task::AddLineRequest l;
    base(l, "epoch_time");
    l.n = 50;
    l.spacing = 0.01;
    l.unit = "s";
    task::add_line(l);
  }
  std::vector<float> values(150);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) * 0.5f - 7.0f;
  auto one_shot = [&](const std::string& name) {
    task::AddFieldRequest f;
    base(f, name);
    f.manifold = "sensors";
    f.time = "epoch_time";
    f.values = mem("float32", {3, 50}, values);
    f.unit = "uV";
    return task::add_field(f);
  };
  auto begin_field = [&](const std::string& name) {
    task::AddFieldRequest f;
    base(f, name);
    f.manifold = "sensors";
    f.time = "epoch_time";
    f.unit = "uV";
    f.piece.begin = true;
    f.piece.shape = std::vector<std::int64_t>{3, 50};
    f.piece.dtype = "float32";
    return f;
  };
  auto append_field = [&](const std::string& write, std::array<std::int64_t, 4> r, const std::vector<float>& v) {
    task::AddFieldRequest f;
    f.dataset = D;
    f.piece.append = true;
    f.piece.write = write;
    f.piece.region = std::vector<std::array<std::int64_t, 2>>{{r[0], r[1]}, {r[2], r[3]}};
    f.values = mem("float32", {r[1] - r[0], r[3] - r[2]}, v);
    return f;
  };
  auto commit_field = [&](const std::string& write) {
    task::AddFieldRequest f;
    f.dataset = D;
    f.created_by = by;
    f.piece.commit = true;
    f.piece.write = write;
    return f;
  };

  // ---- a write in pieces round trips byte for byte against the one-shot command; invisible before the commit
  {
    one_shot("whole");
    const task::CommandResult began = task::add_field(begin_field("pieces"));
    const std::string w = began.result["write"].get<std::string>();
    CHECK(!w.empty());
    CHECK_EQ(began.result["shape"], Json::array({3, 50}));
    CHECK(named(read()["fields"], "pieces") == nullptr);                          // not in the subject
    CHECK(named(list("fields"), "pieces") == nullptr);
    CHECK_EQ(writes(), std::size_t(1));
    const task::CommandResult a1 = task::add_field(append_field(w, {0, 2, 0, 50}, block(values, 0, 2, 0, 50)));
    CHECK_EQ(a1.result["written"].get<std::int64_t>(), std::int64_t(100));
    CHECK(named(read()["fields"], "pieces") == nullptr);                          // still not
    task::add_field(append_field(w, {2, 3, 20, 50}, block(values, 2, 3, 20, 50)));  // out of order: the last row's second part first
    // a commit with a region missing is refused and names it; the write stays
    const std::string missing = thrown([&] { task::add_field(commit_field(w)); });
    CHECK(has(missing, "is not complete") && has(missing, "[2:3, 0:20]") && has(missing, "20 of 150 values"));
    CHECK(named(read()["fields"], "pieces") == nullptr);
    task::add_field(append_field(w, {2, 3, 0, 20}, block(values, 2, 3, 0, 20)));
    const task::CommandResult done = task::add_field(commit_field(w));
    CHECK_EQ(done.result["name"].get<std::string>(), std::string("pieces"));
    CHECK_EQ(done.result["regions"].get<std::int64_t>(), std::int64_t(3));
    CHECK(named(read()["fields"], "pieces") != nullptr);
    CHECK_EQ(writes(), std::size_t(0));                                            // the staged pieces are gone
    const task::CommandResult a = face::read_field(reading("whole")), b = face::read_field(reading("pieces"));
    CHECK(!a.arrays.empty() && !b.arrays.empty());
    if (!a.arrays.empty() && !b.arrays.empty()) {
      CHECK(a.arrays[0].bytes == b.arrays[0].bytes && a.arrays[0].dtype == b.arrays[0].dtype && a.arrays[0].shape == b.arrays[0].shape);
      CHECK(a.arrays[0].bytes == raw(values));
    }
    CHECK(has(thrown([&] { task::add_field(commit_field(w)); }), "no unfinished write"));   // committed once
    const Json fields = read()["fields"];
    const Json* pf = named(fields, "pieces");
    CHECK(pf != nullptr && (*pf)["type"]["unit"] == "uV");
  }

  // ---- the refusals of begin and append; none stores anything
  {
    task::AddFieldRequest bad = begin_field("bad");
    bad.piece.shape = std::vector<std::int64_t>{3, 49};
    CHECK(has(thrown([&] { task::add_field(bad); }), "--shape is [3 x 49]"));
    bad = begin_field("bad");
    bad.manifold = "nowhere";
    CHECK(has(thrown([&] { task::add_field(bad); }), "no manifold named 'nowhere'"));
    bad = begin_field("whole");
    CHECK(has(thrown([&] { task::add_field(bad); }), "already_exists"));           // a name used, without --replace
    bad = begin_field("bad");
    bad.piece.dtype.reset();
    CHECK(has(thrown([&] { task::add_field(bad); }), "--dtype"));
    bad.piece.dtype = "complex64";
    CHECK(has(thrown([&] { task::add_field(bad); }), "float32, float64"));
    bad = begin_field("bad");
    bad.values = mem("float32", {1}, std::vector<float>{1});
    CHECK(has(thrown([&] { task::add_field(bad); }), "not the values"));
    CHECK_EQ(writes(), std::size_t(0));

    const std::string w = task::add_field(begin_field("refused")).result["write"].get<std::string>();
    auto region = [&](std::array<std::int64_t, 4> r) { return append_field(w, r, std::vector<float>(static_cast<std::size_t>((r[1] - r[0]) * (r[3] - r[2])), 1.0f)); };
    CHECK(has(thrown([&] { task::add_field(region({0, 4, 0, 50})); }), "is not inside the shape"));            // outside the shape
    CHECK(has(thrown([&] { task::add_field(region({0, 1, 40, 51})); }), "is not inside the shape"));
    CHECK(has(thrown([&] { task::add_field(region({1, 1, 0, 5})); }), "start < stop"));
    task::add_field(region({0, 2, 0, 10}));
    CHECK(has(thrown([&] { task::add_field(region({0, 2, 0, 10})); }), "was already written"));                // twice
    CHECK(has(thrown([&] { task::add_field(region({1, 3, 5, 15})); }), "overlaps region [0:2, 0:10]"));       // overlapping
    task::AddFieldRequest rank = region({0, 1, 20, 25});
    rank.piece.region = std::vector<std::array<std::int64_t, 2>>{{0, 1}};
    CHECK(has(thrown([&] { task::add_field(rank); }), "has 1 axes"));
    task::AddFieldRequest shape = region({0, 1, 20, 25});
    shape.values = mem("float32", {1, 4}, std::vector<float>(4, 1.0f));
    CHECK(has(thrown([&] { task::add_field(shape); }), "is [1 x 4]; region [0:1, 20:25] is [1 x 5]"));
    task::AddFieldRequest dtype = region({0, 1, 20, 25});
    dtype.values = mem("float64", {1, 5}, std::vector<double>(5, 1.0));
    CHECK(has(thrown([&] { task::add_field(dtype); }), "stores float32"));
    task::AddFieldRequest unknown = region({0, 1, 20, 25});
    unknown.piece.write = "w-000000000000";
    CHECK(has(thrown([&] { task::add_field(unknown); }), "no unfinished write 'w-000000000000'"));
    unknown.piece.write = "../etc";
    CHECK(has(thrown([&] { task::add_field(unknown); }), "is not a write id"));
    task::AddStatisticRequest wrong;                                                                            // a write of a field is not a statistic's
    wrong.dataset = D;
    wrong.piece.commit = true;
    wrong.piece.write = w;
    CHECK(has(thrown([&] { task::add_statistic(wrong); }), "is a write of a field, not a statistic"));
    task::AddFieldRequest none = commit_field(w);
    none.piece.write.reset();
    CHECK(has(thrown([&] { task::add_field(none); }), "name the write"));                                       // no handle, no id
    // an abandoned write is rolled back by `recover dataset`; the write id is unknown after it
    CHECK_EQ(writes(), std::size_t(1));
    const task::CommandResult rec = task::recover_dataset({D});
    CHECK_EQ(rec.result["writes_rolled_back"].get<int>(), 1);
    CHECK_EQ(writes(), std::size_t(0));
    CHECK(has(thrown([&] { task::add_field(commit_field(w)); }), "no unfinished write"));
    CHECK(named(read()["fields"], "refused") == nullptr);
  }

  // ---- a handle carries the write id, owns its writes, and rolls an unfinished one back when it closes
  {
    std::string foreign;
    {
      task::Handle h(D, task::Handle::Mode::write, "test");
      task::HandleScope use(&h);
      const std::string w = task::add_field(begin_field("viaHandle")).result["write"].get<std::string>();
      task::AddFieldRequest a = append_field(w, {0, 3, 0, 50}, values);
      a.piece.write.reset();                                                    // the handle's one unfinished write
      task::add_field(a);
      task::AddFieldRequest c = commit_field(w);
      c.piece.write.reset();
      task::add_field(c);
      CHECK(named(read()["fields"], "viaHandle") != nullptr);
      foreign = task::add_field(begin_field("closedWithHandle")).result["write"].get<std::string>();
      task::AddFieldRequest second = begin_field("second");
      task::add_field(second);
      task::AddFieldRequest ambiguous = commit_field("");
      ambiguous.piece.write.reset();
      CHECK(has(thrown([&] { task::add_field(ambiguous); }), "holds several unfinished writes"));
      CHECK_EQ(writes(), std::size_t(2));
    }
    CHECK_EQ(writes(), std::size_t(0));                                            // closing the handle rolled both back
    CHECK(has(thrown([&] { task::add_field(commit_field(foreign)); }), "no unfinished write"));
    // a write a handle began belongs to it: outside the handle (another process, another handle) it is refused
    task::Handle h(D, task::Handle::Mode::write, "test");
    std::string owned;
    {
      task::HandleScope use(&h);
      owned = task::add_field(begin_field("owned")).result["write"].get<std::string>();
    }
    CHECK(has(thrown([&] { task::add_field(commit_field(owned)); }), "belongs to another handle"));
    h.close();
    CHECK_EQ(writes(), std::size_t(0));
  }

  // ---- `add statistic` in pieces: the same unit, the same bytes as the one-shot command
  {
    std::vector<double> psd(3 * 50);
    for (std::size_t i = 0; i < psd.size(); ++i) psd[i] = 1.0 / static_cast<double>(i + 1);
    task::AddStatisticRequest s;
    base(s, "psd_whole");
    s.measure = "power_density";
    s.method = "welch";
    s.axes = {"channel:sensors", "t:epoch_time"};
    s.values = mem("float64", {3, 50}, psd);
    task::add_statistic(s);
    task::AddStatisticRequest b = s;
    b.name = "psd_pieces";
    b.values = {};
    b.piece.begin = true;
    b.piece.shape = std::vector<std::int64_t>{3, 50};
    b.piece.dtype = "float64";
    const std::string w = task::add_statistic(b).result["write"].get<std::string>();
    CHECK(named(read()["statistics"], "psd_pieces") == nullptr);
    for (int part = 0; part < 3; ++part) {
      task::AddStatisticRequest a;
      a.dataset = D;
      a.piece.append = true;
      a.piece.write = w;
      a.piece.region = std::vector<std::array<std::int64_t, 2>>{{part, part + 1}, {0, 50}};
      a.values = mem("float64", {1, 50}, std::vector<double>(psd.begin() + part * 50, psd.begin() + (part + 1) * 50));
      task::add_statistic(a);
    }
    task::AddStatisticRequest c;
    c.dataset = D;
    c.created_by = by;
    c.piece.commit = true;
    c.piece.write = w;
    task::add_statistic(c);
    CHECK(named(read()["statistics"], "psd_pieces") != nullptr);
    const task::CommandResult x = face::read_statistic(reading("psd_whole", false)), y = face::read_statistic(reading("psd_pieces", false));
    CHECK(!x.arrays.empty() && !y.arrays.empty());
    if (!x.arrays.empty() && !y.arrays.empty()) CHECK(x.arrays[0].bytes == y.arrays[0].bytes && x.arrays[0].dtype == y.arrays[0].dtype);
    task::AddStatisticRequest bad = b;
    bad.name = "psd_bad";
    bad.piece.shape = std::vector<std::int64_t>{3, 49};
    CHECK(has(thrown([&] { task::add_statistic(bad); }), "has 49 along axis 't'"));
    CHECK_EQ(writes(), std::size_t(0));
  }

  // ---- `add statistic --producer --declare`: a measure and a method the registry does not know, declared; refused without the declaration; the producer kept
  {
    std::vector<double> centroids(3 * 50, 0.25);
    task::AddStatisticRequest s;
    base(s, "states");
    s.measure = "flow_state_centroid";
    s.method = "kmeans";
    s.axes = {"channel:sensors", "t:epoch_time"};
    s.values = mem("float64", {3, 50}, centroids);
    s.producer = R"J({"members": [{"subject": "S1", "sha256": "abc"}]})J";
    CHECK(has(thrown([&] { task::add_statistic(s); }), "declare what it is"));
    s.declare = R"J({"measure": {"description": "a state's centroid in the modes", "unit_rule": "(the coefficients' unit)"}, "method": {"description": "k-means on unit vectors"}})J";
    task::add_statistic(s);
    const Json rec = face::read_statistic(reading("states", false)).result;
    CHECK(rec.dump().find("abc") != std::string::npos && rec.dump().find("k-means on unit vectors") != std::string::npos);
    task::AddStatisticRequest odd = s;
    odd.name = "states_odd";
    odd.declare = R"J({"measure": {"description": "x", "unit_rule": "s"}, "method": {"description": "k-means"}})J";
    CHECK(has(thrown([&] { task::add_statistic(odd); }), "one name keeps one meaning"));
  }

  // ---- `add measurements`: declared measures (refused at roll-up when `none`), the producer each row keeps, and ONE measure in pieces equal to the one-shot write
  {
    task::AddSelectionRequest g;
    g.dataset = D; g.subject = "S1"; g.name = "channel_groups"; g.manifold = "sensors"; g.type = "partition"; g.created_by = by;
    g.members = mem("int32", {3}, std::vector<std::int32_t>{0, 1, 2});
    task::add_selection(g);
    task::AddMeasurementsRequest m;
    m.dataset = D; m.subject = "S1"; m.selection = "channel_groups"; m.created_by = by;
    m.measure_arrays = {{"p_of_test", mem("float64", {3}, std::vector<double>{0.01, 0.5, 0.2})}};
    m.declare = {R"J({"name": "p_of_test", "merge": "none", "description": "a p value"})J"};
    m.producer = R"J({"parameters": {"draws": 1000}})J";
    const Json r = task::add_measurements(m).result;
    CHECK(r.contains("declared") && r["declared"][0]["merge"] == "none");
    face::ReadNodeRequest rd = reading("channel_groups", false);
    rd.measure = "p_of_test";
    const Json meta = face::read_selection(rd).result;
    CHECK(meta["declared"] == true && meta["declaration"]["merge"] == "none");
    CHECK(meta["producers"].size() == 1 && meta["producers"][0]["parameters"]["draws"] == 1000 && meta["producers"][0]["command"] == "add measurements");
    // a declaration of a measure not given, and one that contradicts the registry, are refused and store nothing
    task::AddMeasurementsRequest bad = m;
    bad.measure_arrays = {{"q_of_test", mem("float64", {3}, std::vector<double>{1, 2, 3})}};
    CHECK(has(thrown([&] { task::add_measurements(bad); }), "not among the measures given"));
    bad.measure_arrays = {{"median", mem("float64", {3}, std::vector<double>{1, 2, 3})}};
    bad.declare = {R"J({"name": "median", "merge": "sum"})J"};
    CHECK(has(thrown([&] { task::add_measurements(bad); }), "the registry 'none'"));

    // one measure [members x rows] in pieces, against the one-shot write of the same values
    std::vector<double> coefficients(3 * 4);
    for (std::size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = 0.5 * static_cast<double>(i) - 1.0;
    task::AddMeasurementsRequest whole = m;
    whole.field = "rest";
    whole.measure_arrays = {{"coefficient_whole", mem("float64", {3, 4}, coefficients)}};
    whole.declare = {R"J({"name": "coefficient_whole", "merge": "sum", "form": "mode"})J"};
    task::add_measurements(whole);
    task::AddMeasurementsRequest b;
    b.dataset = D; b.subject = "S1"; b.selection = "channel_groups"; b.field = "rest"; b.created_by = by;
    b.measure = {"coefficient_pieces"};
    b.declare = {R"J({"name": "coefficient_pieces", "merge": "sum", "form": "mode"})J"};
    b.piece.begin = true;
    b.piece.shape = std::vector<std::int64_t>{3, 4};
    b.piece.dtype = "float64";
    const std::string w = task::add_measurements(b).result["write"].get<std::string>();
    for (int part = 0; part < 3; ++part) {
      task::AddMeasurementsRequest a;
      a.dataset = D;
      a.piece.append = true;
      a.piece.write = w;
      a.piece.region = std::vector<std::array<std::int64_t, 2>>{{part, part + 1}, {0, 4}};
      a.measure_arrays = {{"", mem("float64", {1, 4}, std::vector<double>(coefficients.begin() + part * 4, coefficients.begin() + (part + 1) * 4))}};
      task::add_measurements(a);
    }
    rd.measure = "coefficient_pieces";
    rd.field = "rest";
    CHECK(face::read_selection(rd).result["rows"] == 0);   // invisible until the commit
    task::AddMeasurementsRequest c;
    c.dataset = D; c.created_by = by;
    c.piece.commit = true;
    c.piece.write = w;
    const Json cr = task::add_measurements(c).result;
    CHECK(cr["step"] == "commit" && cr["regions"] == 3);
    const task::CommandResult x = face::read_selection(rd);
    rd.measure = "coefficient_whole";
    const task::CommandResult y = face::read_selection(rd);
    CHECK(x.result["declaration"]["merge"] == "sum" && x.arrays.size() == y.arrays.size());
    if (x.arrays.size() == 2 && y.arrays.size() == 2) CHECK(x.arrays[1].bytes == y.arrays[1].bytes);
    // ONE MEASURE ON TWO MANIFOLDS (issue #264): the manifold the rows lie on is part of the key; read without it, the two are named; with it, each its own
    task::AddMeasurementsRequest on_a = whole, on_b = whole;
    on_a.measure_arrays = {{"flow_on", mem("float64", {3, 4}, coefficients)}};
    on_a.declare = {R"J({"name": "flow_on", "merge": "sum"})J"};
    on_a.rows_on = "sensors";
    std::vector<double> other(coefficients.size(), 7.0);
    on_b.measure_arrays = {{"flow_on", mem("float64", {3, 4}, other)}};
    on_b.declare.clear();
    on_b.rows_on = "epoch_time";
    task::add_measurements(on_a);
    task::add_measurements(on_b);
    CHECK(has(thrown([&] { task::add_measurements(on_b); }), "on epoch_time: pass --replace"));
    face::ReadNodeRequest ro = reading("channel_groups", false);
    ro.measure = "flow_on"; ro.field = "rest";
    CHECK(has(thrown([&] { face::read_selection(ro); }), "stored on several manifolds"));
    ro.rows_on = "sensors";
    const task::CommandResult ra = face::read_selection(ro);
    ro.rows_on = "epoch_time";
    const task::CommandResult rb = face::read_selection(ro);
    CHECK(ra.arrays.size() == 2 && rb.arrays.size() == 2);
    if (ra.arrays.size() == 2 && rb.arrays.size() == 2 && y.arrays.size() == 2) CHECK(ra.arrays[1].bytes == y.arrays[1].bytes && rb.arrays[1].bytes != ra.arrays[1].bytes);
    // a begin of two measures, or of a shape that is not [members ...], is refused
    task::AddMeasurementsRequest two = b;
    two.measure = {"x1", "x2"};
    two.declare.clear();
    CHECK(has(thrown([&] { task::add_measurements(two); }), "ONE measure"));
    task::AddMeasurementsRequest shape = b;
    shape.measure = {"x3"};
    shape.declare.clear();
    shape.piece.shape = std::vector<std::int64_t>{4, 4};
    CHECK(has(thrown([&] { task::add_measurements(shape); }), "are [3]"));
    CHECK_EQ(writes(), std::size_t(0));
  }

#ifdef NXR_TEST_CLI
  // ---- the command line: the write id works across separate processes (begin, append, append, commit), and `recover dataset` rolls an abandoned one back
  {
    const std::string cli = std::string(NXR_TEST_CLI) + " add field --dataset " + D + " ", quiet = " 2>&1";
    const fs::path out = root / "cli_out.txt";
    auto run_cli = [&](const std::string& args) {
      const int rc = exit_of(cli + args + " > " + out.string() + quiet);
      std::ifstream f(out);
      std::stringstream ss;
      ss << f.rdbuf();
      return std::make_pair(rc, ss.str());
    };
    const auto began = run_cli("--subject S1 --name cli_pieces --manifold sensors --time epoch_time --begin --shape 3,50 --dtype float32");
    CHECK_EQ(began.first, 0);
    const auto at = began.second.find("\"write\": \"");
    std::string w = at == std::string::npos ? "" : began.second.substr(at + 10, began.second.find('"', at + 10) - (at + 10));
    CHECK(w.rfind("w-", 0) == 0);
    const std::string top = npy(root / "in" / "top.npy", "float32", {2, 50}, block(values, 0, 2, 0, 50)), bottom = npy(root / "in" / "bottom.npy", "float32", {1, 50}, block(values, 2, 3, 0, 50));
    CHECK_EQ(run_cli("--append --write " + w + " --region 0:2,0:50 --values " + top).first, 0);
    const auto early = run_cli("--commit --write " + w);
    CHECK(early.first != 0 && has(early.second, "is not complete") && has(early.second, "[2:3, 0:50]"));    // one process later: still there, still unfinished
    const auto twice = run_cli("--append --write " + w + " --region 1:2,0:50 --values " + bottom);
    CHECK(twice.first != 0 && has(twice.second, "overlaps"));
    CHECK_EQ(run_cli("--append --write " + w + " --region 2:3,0:50 --values " + bottom).first, 0);
    CHECK_EQ(run_cli("--commit --write " + w).first, 0);
    const task::CommandResult a = face::read_field(reading("whole")), b = face::read_field(reading("cli_pieces"));
    CHECK(!a.arrays.empty() && !b.arrays.empty() && a.arrays[0].bytes == b.arrays[0].bytes);
    const auto lost = run_cli("--subject S1 --name cli_lost --manifold sensors --time epoch_time --begin --shape 3,50 --dtype float32");
    CHECK_EQ(lost.first, 0);
    CHECK_EQ(writes(), std::size_t(1));
    CHECK_EQ(exit_of(std::string(NXR_TEST_CLI) + " recover dataset --dataset " + D + " > /dev/null 2>&1"), 0);
    CHECK_EQ(writes(), std::size_t(0));
    CHECK_EQ(exit_of(std::string(NXR_TEST_CLI) + " delete subject --dataset " + D + " --subject S1 --node field:cli_pieces > /dev/null 2>&1"), 0);
    CHECK(named(list("fields"), "cli_pieces") == nullptr);
  }
#endif

  // ---- delete by class: what the node owns goes, what depends on it refuses the delete
  {
    task::AddFieldRequest f;
    base(f, "f1");
    f.manifold = "sensors";
    f.values = mem("float64", {3}, std::vector<double>{1, 2, 3});
    task::add_field(f);
    task::AddFieldRequest d2 = f;
    d2.name = "f2";
    d2.derived_from = "f1";
    task::add_field(d2);
    task::AddOperatorRequest o;
    base(o, "mix");
    o.from_manifold = "sensors";
    o.to_manifold = "sensors";
    o.values = mem("float64", {3, 3}, std::vector<double>{1, 0, 0, 0, 2, 0, 0.5, 0, 3});
    task::add_operator(o);
    task::AddFieldRequest est = f;
    est.name = "est";
    est.by_operator = "mix";
    task::add_field(est);
    task::AddSelectionRequest p;
    base(p, "parts");
    p.manifold = "sensors";
    p.type = "partition";
    p.members = mem("int32", {3}, std::vector<std::int32_t>{0, 1, 0});
    p.entries = {"left", "right"};
    p.weights = mem("float32", {2}, std::vector<float>{0.5f, 2.0f});   // one weight a member
    task::add_selection(p);
    // a name in two classes
    task::AddFieldRequest dup = f;
    dup.name = "dup";
    task::add_field(dup);
    task::AddSelectionRequest dups = p;
    dups.name = "dup";
    dups.weights.reset();
    task::add_selection(dups);

    CHECK(has(thrown([&] { del("field:f1"); }), "field f1 is used by field f2"));                      // another field derives from it
    CHECK(has(thrown([&] { del("field:f1"); }), "delete those first"));
    CHECK(named(list("fields"), "f1") != nullptr && named(list("fields"), "f2") != nullptr);          // nothing was deleted
    const std::string on_manifold = thrown([&] { del("manifold:sensors"); });
    CHECK(has(on_manifold, "manifold sensors is used by") && has(on_manifold, "field f1") && has(on_manifold, "selection parts") && has(on_manifold, "operator mix"));
    CHECK(has(thrown([&] { del("operator:mix"); }), "operator mix is used by field est"));
    CHECK(has(thrown([&] { del("field:nothing"); }), "holds no field named 'nothing'"));
    // an ambiguous name is refused, naming the forms; a unique one still works as before
    const std::string amb = thrown([&] { del("dup"); });
    CHECK(has(amb, "ambiguous") && has(amb, "field:dup") && has(amb, "selection:dup"));
    CHECK(named(list("fields"), "dup") != nullptr && named(list("selections"), "dup") != nullptr);

    // a field's arrays go with it
    const std::size_t before_f2 = count_arrays("f2");
    CHECK(before_f2 >= 1);
    const task::CommandResult r2 = del("field:f2");
    CHECK_EQ(r2.result["class"].get<std::string>(), std::string("field"));
    CHECK_EQ(count_arrays("f2"), std::size_t(0));
    CHECK(named(list("fields"), "f2") == nullptr);
    CHECK_EQ(del("field:dup").result["by"].get<std::string>(), std::string("class"));
    CHECK(named(list("fields"), "dup") == nullptr && named(list("selections"), "dup") != nullptr);       // the selection of that name stays
    CHECK(named(list("fields"), "f1") != nullptr);
    // a selection's weights array goes with it (issue 182's sibling: delete)
    CHECK(array_there("sensor/parts") && array_there("sensor/parts.weights"));                         // its elements and its weights are in the store
    del("selection:parts");
    CHECK(!array_there("sensor/parts") && !array_there("sensor/parts.weights"));                       // the weights array went with the selection
    CHECK(named(list("selections"), "parts") == nullptr);
    // issue 182: replacing a selection that had weights, by one without, moves the weights array aside with its values (never left under the old name, never lost)
    task::AddSelectionRequest w1 = p;
    w1.name = "weighted";
    task::add_selection(w1);
    CHECK(array_there("sensor/weighted.weights"));
    w1.weights.reset();
    w1.replace = true;
    const task::CommandResult swapped = task::add_selection(w1);
    CHECK(!swapped.result["replaced"].get<std::string>().empty());
    CHECK(!array_there("sensor/weighted.weights"));
    CHECK(array_there("sensor/weighted"));
    {   // where it went: beside the selection's values, in the day's `_replaced-` folder of the subject
      OpenOptions ro;
      ro.lock = false;
      DatasetWriter ds = open_writer(folder, ro);
      const std::string aside = "_replaced-" + now_utc(false).substr(0, 10) + "/" + text(*ds.subject("S1"), "path") + "/sensor/";
      CHECK(ds.node_store().exists_prefix(aside + "weighted") && ds.node_store().exists_prefix(aside + "weighted.weights"));
    }
    // the operator a deleted field was made by stays: it is a stored node of its own (an inverse kernel), deleted only when named
    const task::CommandResult re = del("field:est");
    CHECK(!re.result.contains("also_removed"));
    CHECK(named(list("operators"), "mix") != nullptr);
    del("operator:mix");
    CHECK(named(list("operators"), "mix") == nullptr);
    CHECK_EQ(count_arrays("mix"), std::size_t(0));
    // an operator two fields were made by stays after one is deleted
    task::AddOperatorRequest o2 = o;
    o2.name = "mix2";
    task::add_operator(o2);
    task::AddFieldRequest a = f, b = f;
    a.name = "ea"; a.by_operator = "mix2";
    b.name = "eb"; b.by_operator = "mix2";
    task::add_field(a);
    task::add_field(b);
    CHECK(!del("field:ea").result.contains("also_removed"));
    CHECK(named(list("operators"), "mix2") != nullptr);
    del("field:eb");
    CHECK(named(list("operators"), "mix2") != nullptr);   // still there with no field left: only a delete that names it removes it
    del("operator:mix2");
    CHECK(named(list("operators"), "mix2") == nullptr);
    // a manifold with nothing on it goes with its geometry and topology
    task::AddPointsRequest pts;
    base(pts, "pts");
    pts.positions = mem("float64", {2, 3}, std::vector<double>{0, 0, 0, 1, 0, 0});
    task::add_points(pts);
    CHECK(count_arrays("pts") >= 1);
    del("manifold:pts");
    CHECK_EQ(count_arrays("pts"), std::size_t(0));
    CHECK(named(list("manifolds"), "pts") == nullptr);
    // a subject's dictionary, deleted unless a selection reads it
    task::AddDictionaryRequest dict;
    dict.dataset = D;
    dict.subject = "S1";
    dict.name = "grade";
    dict.scope = "subject";
    dict.entries = R"([{"code": 0, "name": "good", "colour": "#00ff00"}, {"code": 1, "name": "bad", "colour": "#ff0000"}])";
    dict.created_by = by;
    task::add_dictionary(dict);
    CHECK(named(list("dictionaries"), "grade") != nullptr);
    del("dictionary:grade");
    CHECK(named(list("dictionaries"), "grade") == nullptr);
  }

  // ---- jobs: a run leaves a done row, a failed run a failed one, a cache hit none
  {
    CHECK_EQ(list("jobs").size(), std::size_t(0));
    face::RunRequest rq;
    rq.pipeline = "psd";
    rq.parameters = {{"recording", "rest"}, {"segment_seconds", 2.0}};
    rq.preset = "brainstorm";
    rq.all = true;
    CHECK_EQ(face::run(folder, "S1", rq)["cache"].get<std::string>(), std::string("computed"));
    const Json jobs = list("jobs");
    CHECK_EQ(jobs.size(), std::size_t(1));
    if (jobs.size() == 1) {
      const Json& j = jobs[0];
      CHECK_EQ(j["kind"].get<std::string>(), std::string("run:psd"));
      CHECK_EQ(j["status"].get<std::string>(), std::string("done"));
      CHECK_EQ(j["subject"].get<std::string>(), std::string("S1"));
      CHECK(j["error"].is_null());
      CHECK(has(j["requested_by"].get<std::string>(), "nxr-neuro"));
      CHECK(!j["started_utc"].is_null() && !j["finished_utc"].is_null());
      CHECK_EQ(j["parameters"]["pipeline"].get<std::string>().rfind("psd@", 0), std::size_t(0));
      CHECK_EQ(j["parameters"]["parameters"]["recording"].get<std::string>(), std::string("rest"));
      CHECK_EQ(j["result"]["stored"][0]["name"].get<std::string>(), std::string("rest_psd_welch"));
      CHECK(j["inputs"].is_object() || j["inputs"].is_array());
    }
    CHECK_EQ(face::run(folder, "S1", rq)["cache"].get<std::string>(), std::string("hit"));
    CHECK_EQ(list("jobs").size(), std::size_t(1));                               // nothing ran: no row
    // a run that fails while it executes (a span outside the recording): a failed row with the message, and the error goes on
    face::RunRequest late = rq;
    late.all = false;
    late.from = 1.0e6;
    late.to = 1.0e6 + 4.0;
    late.parameters["segment_seconds"] = 1.0;
    const std::string why = thrown([&] { face::run(folder, "S1", late); });
    CHECK(!why.empty());
    const Json after = list("jobs");
    CHECK_EQ(after.size(), std::size_t(2));
    int failed = 0;
    for (const auto& j : after)
      if (j["status"] == "failed") {
        ++failed;
        CHECK(!j["error"].is_null() && !j["error"].get<std::string>().empty());
        CHECK_EQ(j["kind"].get<std::string>(), std::string("run:psd"));
        CHECK(j["result"].is_null());
      }
    CHECK_EQ(failed, 1);
    CHECK_EQ(list("jobs", [](task::ListClassRequest& q) { q.kind = "run:psd"; }).size(), std::size_t(2));
    face::RunRequest dry = rq;
    dry.parameters["segment_seconds"] = 1.5;
    dry.dry_run = true;
    face::run(folder, "S1", dry);
    CHECK_EQ(list("jobs").size(), std::size_t(2));                               // a dry run leaves none
  }

  // ---- jobs: an import into a subject package leaves a row, a failed import a failed row
  {
    const fs::path cohort = root / "cohort";
    task::CreateDatasetRequest cd;
    cd.dataset = cohort.string();
    cd.created_by = by;
    task::create_dataset(cd);
    task::CreateSubjectRequest cs;
    cs.dataset = cohort.string();
    cs.name = "P1";
    cs.created_by = by;
    task::create_subject(cs);
    std::string package;
    task::ListSubjectsRequest ls;
    ls.dataset = cohort.string();
    const task::CommandResult listed = task::list_subjects(ls);
    for (const auto& s : listed.result["subjects"]) if (s["name"] == "P1") package = (cohort / s["path"].get<std::string>()).string();
    CHECK(!package.empty() && fs::exists(package));
    task::ImportRecordingRequest ir;
    ir.from = std::string(NXR_FORMATS_DATA_DIR) + "/bdf_plain.bdf";
    ir.package = package;
    ir.name = "rest";
    task::import_recording(ir);
    task::ListClassRequest q;
    q.cls = "jobs";
    q.dataset = cohort.string();
    q.subject = "P1";
    const Json jobs = face::list_class(q).result["rows"];
    CHECK_EQ(jobs.size(), std::size_t(1));
    if (jobs.size() == 1) {
      CHECK_EQ(jobs[0]["kind"].get<std::string>(), std::string("import:recording"));
      CHECK_EQ(jobs[0]["status"].get<std::string>(), std::string("done"));
      CHECK_EQ(jobs[0]["parameters"]["name"].get<std::string>(), std::string("rest"));
      CHECK(!jobs[0]["finished_utc"].is_null());
    }
    // a file that is not there, and a file that is not a recording: each import failed while it ran, a failed row with its message
    task::ImportRecordingRequest missing = ir;
    missing.from = (root / "nothing.edf").string();
    CHECK(has(thrown([&] { task::import_recording(missing); }), "no such file"));
    task::ImportRecordingRequest broken = ir;
    broken.from = npy(root / "in" / "notarecording.edf", "float64", {2}, std::vector<double>{1, 2});
    broken.name = "broken";
    CHECK(!thrown([&] { task::import_recording(broken); }).empty());
    const Json after = face::list_class(q).result["rows"];
    int failed = 0;
    for (const auto& j : after)
      if (j["status"] == "failed") {
        ++failed;
        CHECK(!j["error"].is_null() && !j["error"].get<std::string>().empty() && j["result"].is_null());
        CHECK_EQ(j["kind"].get<std::string>(), std::string("import:recording"));
      }
    CHECK_EQ(failed, 2);
    CHECK_EQ(after.size(), std::size_t(3));
  }

  // ---- `delete subject --node --by`: a client deletes only what it saved (the node's created_by, as `add --by` records it); without --by any node goes, as before
  {
    auto line_by = [&](const std::string& name, const std::optional<std::string>& author) {
      task::AddLineRequest l;
      base(l, name);
      l.n = 4;
      l.spacing = 1.0;
      l.unit = "s";
      l.by = author;
      task::add_line(l);
    };
    auto del_by = [&](const std::string& node, const std::optional<std::string>& author) {
      task::DeleteSubjectRequest q;
      q.dataset = D;
      q.subject = "S1";
      q.node = node;
      q.by = author;
      q.created_by = by;
      return task::delete_subject(q);
    };
    line_by("saved_by_client", std::string("a-client"));
    line_by("saved_by_program", std::nullopt);
    const std::string other = thrown([&] { del_by("manifold:saved_by_client", std::string("another-client")); });
    CHECK(has(other, "invalid_argument") && has(other, "'a-client'") && has(other, "was not deleted"));
    CHECK(named(list("manifolds"), "saved_by_client") != nullptr);                                  // still there
    const std::string program = thrown([&] { del_by("manifold:saved_by_program", std::string("a-client")); });
    CHECK(has(program, "invalid_argument") && has(program, "'" + by + "'"));                        // the program's node: named by its author, the program
    CHECK(named(list("manifolds"), "saved_by_program") != nullptr);
    CHECK_EQ(del_by("manifold:saved_by_client", std::string("a-client")).result["removed"].get<std::string>(), std::string("saved_by_client"));
    CHECK(named(list("manifolds"), "saved_by_client") == nullptr);
    CHECK_EQ(del_by("manifold:saved_by_program", std::nullopt).result["removed"].get<std::string>(), std::string("saved_by_program"));   // no --by: unchanged
    task::DeleteSubjectRequest whole;
    whole.dataset = D;
    whole.subject = "S1";
    whole.by = std::string("a-client");
    whole.created_by = by;
    CHECK(has(thrown([&] { task::delete_subject(whole); }), "refused: --by goes with --node"));      // --by names a node's author: never the whole subject
    CHECK(!read()["manifolds"].empty());                                                               // the subject is still there
  }

  // ---- `delete subject --node` on a catalogue of subject packages (a package has no datastore: its arrays go through the package's own node store); a used node is still refused
  {
    const fs::path pk = face::synthetic_dataset(root, "pk", "S1", "rest", 3, 100.0, 40.0);
    {
      auto basep = [&](auto& r, const std::string& name) { r.dataset = pk.string(); r.subject = "S1"; r.name = name; r.created_by = by; };
      task::AddFieldRequest f;
      basep(f, "pf1");
      f.manifold = "sensors";
      f.values = mem("float64", {3}, std::vector<double>{1, 2, 3});
      task::add_field(f);
      task::AddFieldRequest d2 = f;
      d2.name = "pf2";
      d2.derived_from = "pf1";
      task::add_field(d2);
    }
    SplitOptions sp;
    sp.registered_by = "test";
    split_dataset(pk, sp);
    auto fields_of = [&]() { task::ListClassRequest q; q.cls = "fields"; q.dataset = pk.string(); q.subject = "S1"; return face::list_class(q).result["rows"]; };
    // the package's own node store holds a field's arrays: `pf2`'s path, and whether anything is stored under it (the writer is closed again before a command opens the package)
    auto stored = [&](const std::string& name, std::string& path) {
      DatasetWriter w = open_subject_package(pk, "S1");
      SubjectWriter sw = SubjectWriter::open(w, "S1");
      if (path.empty()) { const auto r = sw.by_name("field", name); if (r) path = text(*r, "path"); }
      const bool there = !path.empty() && w.node_store().exists_prefix(pk.filename().string() + "/" + text(sw.row(), "path") + "/" + path);   // the package store's key: the dataset folder, the subject's folder, the node's path
      w.close();
      return there;
    };
    auto delp = [&](const std::string& node) { task::DeleteSubjectRequest q; q.dataset = pk.string(); q.subject = "S1"; q.node = node; q.created_by = by; return task::delete_subject(q); };
    CHECK(named(fields_of(), "pf1") != nullptr && named(fields_of(), "pf2") != nullptr);
    CHECK(has(thrown([&] { delp("field:pf1"); }), "delete those first"));                          // used by pf2: refused, nothing deleted
    CHECK(named(fields_of(), "pf1") != nullptr);
    std::string pf2_path;
    CHECK(stored("pf2", pf2_path));                                                                  // its arrays are in the package
    CHECK_EQ(delp("field:pf2").result["removed"].get<std::string>(), std::string("pf2"));
    CHECK(named(fields_of(), "pf2") == nullptr);
    CHECK(!stored("pf2", pf2_path));                                                                 // its arrays are gone from the package
    CHECK_EQ(delp("field:pf1").result["removed"].get<std::string>(), std::string("pf1"));            // free now
  }

  return nxrtest::finish("test_pieces_deletes_jobs");
}
