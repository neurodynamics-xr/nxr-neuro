// The options of `add`, the dictionaries and `edit manifold` (docs/specs/2026-11-20-data-layer-queries.md, step 2: 3.3 to 3.6). One round trip each, through the commands' library functions, read back
// through `read subject`, `list <class>` and `read <class>`: `add product`, `add line --coordinates --kinds`, `add field --by-operator --contribution --producer`, `add selection --of-field --weights
// --producer`, `add operator --layout analytic`, `add dictionary`, `edit dictionary`, `edit manifold --primary`; the refusals (a missing factor, coordinates not increasing, a duplicate code, a bad
// colour, a non-surface made primary, ...) leave nothing stored. Synthetic data only (face::synthetic_dataset).
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/commands.h"
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
std::string npy(const fs::path& file, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<T>& v) {
  write_npy(file, dtype, shape, raw(v));
  return file.string();
}
const Json* named(const Json& rows, const std::string& name) {
  for (const auto& r : rows) if (r["name"] == name) return &r;
  return nullptr;
}
const task::CommandArray& array_named(const task::CommandResult& r, const std::string& name) {
  for (const auto& a : r.arrays) if (a.name == name) return a;
  static task::CommandArray none;
  std::printf("  no array '%s'\n", name.c_str());
  return none;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_add_options_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "in");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);
  const std::string D = folder.string(), by = face::stamp();
  auto base = [&](auto& r, const std::string& name, const std::string& subject = "S1") { r.dataset = D; r.subject = subject; r.name = name; r.created_by = by; };
  auto read = [&](const std::string& subject = "S1") { task::ReadSubjectRequest q; q.dataset = D; q.subject = subject; return face::read_subject(q).result; };
  auto list = [&](const std::string& cls, const std::function<void(task::ListClassRequest&)>& set = {}) {
    task::ListClassRequest q;
    q.cls = cls;
    q.dataset = D;
    q.subject = "S1";
    if (set) set(q);
    return face::list_class(q).result["rows"];
  };
  auto reading = [&](const std::string& name) { face::ReadNodeRequest r; r.dataset = D; r.subject = "S1"; r.name = name; return r; };

  // ---- what the checks lie on: a time line, a second subject with a field, an operator, a partition
  {
    task::AddLineRequest l;
    base(l, "epoch_time");
    l.n = 50;
    l.spacing = 0.01;
    l.unit = "s";
    task::add_line(l);
    task::CreateSubjectRequest cs;
    cs.dataset = D;
    cs.name = "S2";
    cs.created_by = by;
    task::create_subject(cs);
    task::AddPointsRequest pts;
    base(pts, "pts", "S2");
    pts.positions = npy(root / "in" / "pts.npy", "float64", {2, 3}, std::vector<double>{0, 0, 0, 1, 0, 0});
    task::add_points(pts);
    task::AddFieldRequest pf;
    base(pf, "pf", "S2");
    pf.manifold = "pts";
    pf.values = npy(root / "in" / "pf.npy", "float64", {2}, std::vector<double>{1, 2});
    task::add_field(pf);
    task::AddFieldRequest w;
    base(w, "w");
    w.manifold = "sensors";
    w.values = npy(root / "in" / "w.npy", "float32", {3}, std::vector<float>{0.5f, -1.25f, 3.0f});
    task::add_field(w);
    task::AddOperatorRequest o;
    base(o, "mix");
    o.from_manifold = "sensors";
    o.to_manifold = "sensors";
    o.values = npy(root / "in" / "mix.npy", "float64", {3, 3}, std::vector<double>{1, 0, 0, 0, 2, 0, 0.5, 0, 3});
    task::add_operator(o);
    task::AddSelectionRequest p;
    base(p, "parts");
    p.manifold = "sensors";
    p.type = "partition";
    p.members = npy(root / "in" / "parts.npy", "int32", {3}, std::vector<std::int32_t>{0, 1, 0});
    p.entries = {"left", "right"};
    task::add_selection(p);
  }

  // ---- add line --coordinates --kinds, add product (3.3)
  {
    const std::vector<double> lambda = {0.5, 1.5, 4.0, 9.0};
    task::AddLineRequest l;
    base(l, "eigs");
    l.unit = "1/m2";
    l.dimension = "eigenvalue";
    l.coordinates = npy(root / "in" / "lambda.npy", "float64", {4}, lambda);
    l.kinds = npy(root / "in" / "kinds.npy", "int32", {4}, std::vector<std::int32_t>{0, 1, 1, 2});
    const task::CommandResult made = task::add_line(l);
    CHECK_EQ(made.result["n"].get<std::int64_t>(), std::int64_t(4));
    const task::CommandResult back = face::read_manifold(reading("eigs"));
    CHECK(array_named(back, "positions").bytes == raw(lambda));                                       // the coordinates as given, float64
    CHECK(array_named(back, "kinds").bytes == raw(std::vector<std::int32_t>{0, 1, 1, 2}));            // and the code of each point
    const Json sub = read();
    const Json* eigs = named(sub["manifolds"], "eigs");
    CHECK(eigs != nullptr);
    if (eigs) {
      CHECK_EQ((*eigs)["kind"].get<std::string>(), std::string("line"));
      CHECK_EQ((*eigs)["size"]["vertices"].get<std::int64_t>(), std::int64_t(4));
      CHECK_EQ((*eigs)["coordinates"]["system"].get<std::string>(), std::string("eigenvalue"));
      CHECK_EQ((*eigs)["coordinates"]["unit"].get<std::string>(), std::string("1/m2"));
      CHECK((*eigs)["coordinates"]["kinds"].get<bool>());
    }
    // refusals: coordinates not strictly increasing, an even spacing as well, a dimension the layout cannot place, kinds of another length; none leaves a node
    task::AddLineRequest bad = l;
    bad.name = "bad";
    bad.coordinates = npy(root / "in" / "down.npy", "float64", {3}, std::vector<double>{1.0, 3.0, 2.0});
    bad.kinds.reset();
    CHECK(has(thrown([&] { task::add_line(bad); }), "not strictly increasing"));
    bad.coordinates = npy(root / "in" / "flat.npy", "float64", {3}, std::vector<double>{1.0, 1.0, 2.0});
    CHECK(has(thrown([&] { task::add_line(bad); }), "not strictly increasing"));
    bad.coordinates = npy(root / "in" / "ok.npy", "float64", {3}, std::vector<double>{1.0, 2.0, 3.0});
    bad.n = 3;
    CHECK(has(thrown([&] { task::add_line(bad); }), "leave out --n, --spacing and --origin"));
    bad.n.reset();
    bad.dimension = "time";
    CHECK(has(thrown([&] { task::add_line(bad); }), "--dimension eigenvalue"));
    bad.dimension = "eigenvalue";
    bad.kinds = npy(root / "in" / "kinds2.npy", "int32", {2}, std::vector<std::int32_t>{0, 1});
    CHECK(has(thrown([&] { task::add_line(bad); }), "one code a point"));
    bad.kinds = npy(root / "in" / "kindsf.npy", "float64", {3}, std::vector<double>{0, 1, 2});
    CHECK(has(thrown([&] { task::add_line(bad); }), "one integer code a point"));
    CHECK(has(thrown([&] { task::add_line(l); }), "already_exists"));                                   // the name is used
    CHECK(named(read()["manifolds"], "bad") == nullptr);

    // add product: the product of a stored line and a time line, then a field on it (the layout places the eigenvalue line)
    task::AddProductRequest pr;
    base(pr, "eig_by_epoch");
    pr.factors = {"eigs", "epoch_time"};
    const task::CommandResult product = task::add_product(pr);
    CHECK_EQ(product.result["type"].get<std::string>(), std::string("product"));
    CHECK_EQ(product.result["n"].get<std::int64_t>(), std::int64_t(200));
    const Json after = read();
    const Json* made_product = named(after["manifolds"], "eig_by_epoch");
    CHECK(made_product != nullptr);
    if (made_product) {
      CHECK_EQ((*made_product)["kind"].get<std::string>(), std::string("product"));
      CHECK_EQ((*made_product)["factors"].size(), std::size_t(2));
      CHECK_EQ((*made_product)["factors"][0].get<std::string>(), std::string("eigs"));
      CHECK_EQ((*made_product)["factors"][1].get<std::string>(), std::string("epoch_time"));
    }
    CHECK_EQ(list("manifolds", [](task::ListClassRequest& q) { q.on = "eigs"; }).size(), std::size_t(1));   // the manifolds built on the line
    task::AddFieldRequest f;
    base(f, "spectrum");
    f.manifold = "eigs";
    f.time = "epoch_time";
    f.values = npy(root / "in" / "spectrum.npy", "float32", {4, 50}, std::vector<float>(200, 0.25f));
    task::add_field(f);
    const Json snap1 = read();
    const Json* spectrum = named(snap1["fields"], "spectrum");
    CHECK(spectrum != nullptr);
    CHECK_EQ(list("fields", [](task::ListClassRequest& q) { q.on = "eigs"; }).size(), std::size_t(1));
    // refusals: a factor that does not exist, one factor, a name in use, the same factors under another name
    task::AddProductRequest nope = pr;
    nope.name = "nope";
    nope.factors = {"eigs", "missing"};
    CHECK(has(thrown([&] { task::add_product(nope); }), "no_such_row"));
    nope.factors = {"eigs"};
    CHECK(has(thrown([&] { task::add_product(nope); }), "two or more"));
    nope.factors = {"eigs", "epoch_time"};
    CHECK(has(thrown([&] { task::add_product(nope); }), "already make the product 'eig_by_epoch'"));
    nope.name = "eig_by_epoch";
    CHECK(has(thrown([&] { task::add_product(nope); }), "already_exists"));
    CHECK(named(read()["manifolds"], "nope") == nullptr);
  }

  // ---- add field --by-operator --contribution --producer (3.4)
  {
    task::AddFieldRequest f;
    base(f, "applied");
    f.manifold = "sensors";
    f.values = npy(root / "in" / "applied.npy", "float32", {3}, std::vector<float>{1, 2, 3});
    f.by_operator = "mix";
    f.contribution = {"S2:pf:0.5", "S1:w"};
    f.producer = R"({"maker": "a script", "settings": {"n": 3}})";
    const task::CommandResult made = task::add_field(f);
    CHECK_EQ(made.result["by_operator"].get<std::string>(), std::string("mix"));
    CHECK_EQ(made.result["contributions"].get<std::int64_t>(), std::int64_t(2));
    const Json snap2 = read();
    const Json* applied = named(snap2["fields"], "applied");
    CHECK(applied != nullptr);
    if (applied) {
      CHECK_EQ((*applied)["lineage"]["by_operator"].get<std::string>(), std::string("mix"));           // the lineage `read subject` shows
      const Json& c = (*applied)["lineage"]["contributions"];
      CHECK_EQ(c.size(), std::size_t(2));
      if (c.size() == 2) {
        CHECK_EQ(c[0]["field"].get<std::string>(), std::string("S2:pf"));
        CHECK_EQ(c[0]["weight"].get<double>(), 0.5);
        CHECK_EQ(c[1]["field"].get<std::string>(), std::string("w"));
        CHECK_EQ(c[1]["weight"].get<double>(), 1.0);
      }
    }
    const task::CommandResult back = face::read_field(reading("applied"));
    CHECK_EQ(back.result["producer"]["maker"].get<std::string>(), std::string("a script"));            // the maker's facts, beside the command's own
    CHECK_EQ(back.result["producer"]["command"].get<std::string>(), std::string("add field"));
    // refusals; nothing is stored
    task::AddFieldRequest bad = f;
    bad.name = "bad";
    bad.contribution.clear();
    bad.by_operator = "nothing";
    CHECK(has(thrown([&] { task::add_field(bad); }), "no_such_row"));
    bad.by_operator.reset();
    bad.contribution = {"S2"};
    CHECK(has(thrown([&] { task::add_field(bad); }), "SUBJECT:FIELD"));
    bad.contribution = {"S3:pf"};
    CHECK(has(thrown([&] { task::add_field(bad); }), "no subject named 'S3'"));
    bad.contribution = {"S2:nothing"};
    CHECK(has(thrown([&] { task::add_field(bad); }), "holds no field named 'nothing'"));
    bad.contribution = {"S2:pf", "S2:pf:2"};
    CHECK(has(thrown([&] { task::add_field(bad); }), "named twice"));
    bad.contribution.clear();
    bad.producer = R"({"command": "forged"})";
    CHECK(has(thrown([&] { task::add_field(bad); }), "the command's own record"));
    bad.producer = "[1]";
    CHECK(has(thrown([&] { task::add_field(bad); }), "a JSON object"));
    CHECK(named(read()["fields"], "bad") == nullptr);
  }

  // ---- add selection --of-field --weights --producer (3.4)
  {
    task::AddSelectionRequest s;
    base(s, "picked");
    s.manifold = "sensors";
    s.type = "indices";
    s.indices = npy(root / "in" / "picks.npy", "int32", {2}, std::vector<std::int32_t>{2, 0});
    s.of_field = "rest";
    s.weights = npy(root / "in" / "weights.npy", "float32", {2}, std::vector<float>{0.75f, 0.25f});
    s.producer = R"({"maker": "a script"})";
    const task::CommandResult made = task::add_selection(s);
    CHECK_EQ(made.result["of_field"].get<std::string>(), std::string("rest"));
    const Json snap3 = read();
    const Json* picked = named(snap3["selections"], "picked");
    CHECK(picked != nullptr);
    if (picked) {
      CHECK_EQ((*picked)["of_field"].get<std::string>(), std::string("rest"));
      CHECK((*picked)["weights"].get<bool>());
    }
    const task::CommandResult back = face::read_selection(reading("picked"));
    CHECK(array_named(back, "weights").bytes == raw(std::vector<float>{0.75f, 0.25f}));                  // one weight a member, in the stored precision
    CHECK(array_named(back, "values").bytes == raw(std::vector<std::int32_t>{2, 0}));
    task::AddSelectionRequest q;                                                                        // a partition: one weight a member (a label)
    base(q, "weighted_parts");
    q.manifold = "sensors";
    q.type = "partition";
    q.members = npy(root / "in" / "parts2.npy", "int32", {3}, std::vector<std::int32_t>{0, 1, 0});
    q.weights = npy(root / "in" / "w2.npy", "float64", {2}, std::vector<double>{3.0, 5.0});
    task::add_selection(q);
    CHECK(array_named(face::read_selection(reading("weighted_parts")), "weights").bytes == raw(std::vector<double>{3.0, 5.0}));
    // refusals: a weight too few, an unknown field, a weight that is not a number
    task::AddSelectionRequest bad = s;
    bad.name = "bad";
    bad.weights = npy(root / "in" / "w3.npy", "float32", {3}, std::vector<float>{1, 2, 3});
    CHECK(has(thrown([&] { task::add_selection(bad); }), "one weight a member"));
    bad.weights = npy(root / "in" / "w4.npy", "float32", {2}, std::vector<float>{1.0f, std::nanf("")});
    CHECK(has(thrown([&] { task::add_selection(bad); }), "not a finite number"));
    bad.weights.reset();
    bad.of_field = "missing";
    CHECK(has(thrown([&] { task::add_selection(bad); }), "no_such_row"));
    CHECK(named(read()["selections"], "bad") == nullptr);
  }

  // ---- add operator --layout analytic (3.4)
  {
    task::AddOperatorRequest o;
    base(o, "smooth");
    o.from_manifold = "sensors";
    o.to_manifold = "sensors";
    o.layout = "analytic";
    o.method = "gaussian_smoothing";
    o.params = R"({"sigma": 2.5})";
    o.from_selection = "parts";
    const task::CommandResult made = task::add_operator(o);
    CHECK_EQ(made.result["layout"].get<std::string>(), std::string("analytic"));
    const Json snap4 = read();
    const Json* smooth = named(snap4["operators"], "smooth");
    CHECK(smooth != nullptr);
    if (smooth) {
      CHECK_EQ((*smooth)["method"].get<std::string>(), std::string("gaussian_smoothing"));
      CHECK_EQ((*smooth)["parameters"]["sigma"].get<double>(), 2.5);
      CHECK_EQ((*smooth)["from"]["selection"].get<std::string>(), std::string("parts"));
      CHECK_EQ((*smooth)["shape"]["rows"].get<std::int64_t>(), std::int64_t(3));
      CHECK(!(*smooth)["stored"].get<bool>());                                                          // no matrix is stored
    }
    CHECK(has(thrown([&] { face::read_operator(reading("smooth")); }), "no stored array"));
    // refusals: a matrix with it, a method without it, a missing method, a selection of another manifold
    task::AddOperatorRequest bad = o;
    bad.name = "bad";
    bad.values = npy(root / "in" / "m.npy", "float64", {3, 3}, std::vector<double>(9, 1.0));
    CHECK(has(thrown([&] { task::add_operator(bad); }), "holds no matrix"));
    bad.values.reset();
    bad.method.reset();
    CHECK(has(thrown([&] { task::add_operator(bad); }), "takes --method"));
    bad.method = "x";
    bad.layout.reset();
    bad.values = npy(root / "in" / "m2.npy", "float64", {3, 3}, std::vector<double>(9, 1.0));
    CHECK(has(thrown([&] { task::add_operator(bad); }), "belong to --layout analytic"));
    bad.values.reset();
    bad.method.reset();
    bad.params.reset();
    CHECK(has(thrown([&] { task::add_operator(bad); }), "--values"));
    bad.layout = "analytic";
    bad.method = "x";
    bad.from_selection = "picked";
    task::AddOperatorRequest other = bad;
    other.from_manifold = "eigs";
    other.to_manifold = "eigs";
    CHECK(has(thrown([&] { task::add_operator(other); }), "does not lie on 'eigs'"));
    CHECK(named(read()["operators"], "bad") == nullptr);
    // replacing it by a stored one (no arrays to move aside)
    o.layout.reset();
    o.method.reset();
    o.params.reset();
    o.from_selection.reset();
    o.values = npy(root / "in" / "m3.npy", "float64", {3, 3}, std::vector<double>(9, 1.0));
    CHECK(has(thrown([&] { task::add_operator(o); }), "already_exists"));
    o.replace = true;
    task::add_operator(o);
    CHECK(array_named(face::read_operator(reading("smooth")), "values").shape == (std::vector<std::int64_t>{3, 3}));
    int count = 0;
    const Json operators = read();
    for (const auto& r : operators["operators"]) count += r["name"] == "smooth";
    CHECK_EQ(count, 1);
  }

  // ---- dictionaries (3.5)
  {
    const fs::path file = root / "in" / "segment_quality.json";
    {
      std::ofstream out(file);
      out << R"([{"code": 0, "name": "good", "label": "Good", "colour": "#00ff00", "description": "no artefact"},
                 {"code": 2, "name": "bad", "colour": "#FF0000", "params": {"threshold": 3}}])";
    }
    task::AddDictionaryRequest d;
    d.dataset = D;
    d.subject = "S1";
    d.name = "segment_quality";
    d.scope = "dataset";
    d.entries = file.string();
    d.description = "the quality of a segment";
    d.created_by = by;
    const task::CommandResult made = task::add_dictionary(d);
    CHECK_EQ(made.result["entries"].size(), std::size_t(2));
    const Json rows = list("dictionaries", [](task::ListClassRequest& q) { q.name = "segment_quality"; });
    if (rows.size() != 1) std::printf("  rows: %s\n", rows.dump().c_str());
    CHECK_EQ(rows.size(), std::size_t(1));
    if (rows.size() == 1) {
      CHECK_EQ(rows[0]["scope"].get<std::string>(), std::string("dataset"));
      CHECK_EQ(rows[0]["description"].get<std::string>(), std::string("the quality of a segment"));
      const Json& e = rows[0]["entries"];
      CHECK_EQ(e.size(), std::size_t(2));
      if (e.size() == 2) {
        CHECK_EQ(e[0]["code"].get<std::int64_t>(), std::int64_t(0));
        CHECK_EQ(e[0]["label"].get<std::string>(), std::string("Good"));
        CHECK_EQ(e[0]["colour"].get<std::string>(), std::string("#00ff00"));
        CHECK_EQ(e[0]["description"].get<std::string>(), std::string("no artefact"));
        CHECK_EQ(e[1]["code"].get<std::int64_t>(), std::int64_t(2));
        CHECK_EQ(e[1]["colour"].get<std::string>(), std::string("#ff0000"));
        CHECK_EQ(e[1]["params"]["threshold"].get<int>(), 3);
        CHECK(e[1]["label"].is_null());
      }
    }
    CHECK(named(read()["dictionaries"], "segment_quality") != nullptr);                                          // `read subject` shows it too
    // a subject's own, from entries given as text; the same name in another scope is another dictionary
    task::AddDictionaryRequest s = d;
    s.scope = "subject";
    s.entries = R"([{"code": 1, "name": "only"}])";
    task::add_dictionary(s);
    CHECK_EQ(list("dictionaries", [](task::ListClassRequest& q) { q.name = "segment_quality"; }).size(), std::size_t(2));
    // refusals; nothing is stored
    task::AddDictionaryRequest bad = d;
    bad.name = "bad";
    bad.entries = R"([{"code": 1, "name": "a"}, {"code": 1, "name": "b"}])";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "the code 1 is used by an earlier entry"));
    bad.entries = R"([{"code": 1, "name": "a", "colour": "red"}])";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "not #rrggbb"));
    bad.entries = R"([{"code": 1, "name": "a", "colour": "#ff00"}])";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "not #rrggbb"));
    bad.entries = R"([{"code": 1, "name": "a", "colour": "#ff00gg"}])";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "not #rrggbb"));
    bad.entries = R"([{"name": "a"}])";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "a code"));
    bad.entries = R"([{"code": 1, "name": "a", "shade": 1}])";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "not a field of an entry"));
    bad.entries = "no_such_file.json";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "no such file"));
    bad.entries = R"([{"code": 1, "name": "a"}])";
    CHECK(has(thrown([&] { task::add_dictionary(d); }), "already_exists"));
    bad.scope = "subject";
    bad.subject.reset();
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "--subject S"));
    bad.scope = "world";
    CHECK(has(thrown([&] { task::add_dictionary(bad); }), "dataset or subject"));
    CHECK(named(read()["dictionaries"], "bad") == nullptr);

    // a dictionary of the subject is found before the dataset's of the same name: entry 1 exists only in the subject's, entry 0 only in the dataset's
    task::EditDictionaryRequest e;
    e.dataset = D;
    e.subject = "S1";
    e.name = "segment_quality";
    e.entry = 1;
    e.name_of = "single";
    e.created_by = by;
    task::edit_dictionary(e);
    CHECK_EQ(list("dictionaries", [](task::ListClassRequest& q) { q.name = "segment_quality"; q.kind = "subject"; })[0]["entries"][0]["name"].get<std::string>(), std::string("single"));
    CHECK_EQ(list("dictionaries", [](task::ListClassRequest& q) { q.name = "segment_quality"; q.kind = "dataset"; })[0]["entries"][0]["name"].get<std::string>(), std::string("good"));
  }
  {
    // ... so the entry of the DATASET's dictionary is edited when the subject has none of that name: a dictionary only the dataset has
    task::AddDictionaryRequest d;
    d.dataset = D;
    d.subject = "S1";
    d.name = "kind";
    d.scope = "dataset";
    d.entries = R"([{"code": 0, "name": "meg"}, {"code": 1, "name": "eeg", "colour": "#112233"}])";
    d.created_by = by;
    task::add_dictionary(d);
    task::EditDictionaryRequest e;
    e.dataset = D;
    e.subject = "S1";
    e.name = "kind";
    e.entry = 1;
    e.name_of = "EEG";
    e.colour = "#445566";
    e.label = "Electroencephalography";
    e.description = "scalp electrodes";
    e.params = R"({"ref": "average"})";
    e.created_by = by;
    task::edit_dictionary(e);
    const Json rows = list("dictionaries", [](task::ListClassRequest& q) { q.name = "kind"; });
    CHECK_EQ(rows.size(), std::size_t(1));
    if (rows.size() == 1) {
      const Json& x = rows[0]["entries"][1];
      CHECK_EQ(x["name"].get<std::string>(), std::string("EEG"));
      CHECK_EQ(x["colour"].get<std::string>(), std::string("#445566"));
      CHECK_EQ(x["label"].get<std::string>(), std::string("Electroencephalography"));
      CHECK_EQ(x["description"].get<std::string>(), std::string("scalp electrodes"));
      CHECK_EQ(x["params"]["ref"].get<std::string>(), std::string("average"));
      CHECK_EQ(rows[0]["entries"][0]["name"].get<std::string>(), std::string("meg"));                    // the other entry is as it was
    }
    // the entry's label cleared by an empty text
    task::EditDictionaryRequest clear = e;
    clear.name_of.reset(); clear.colour.reset(); clear.description.reset(); clear.params.reset();
    clear.label = "";
    task::edit_dictionary(clear);
    CHECK(list("dictionaries", [](task::ListClassRequest& q) { q.name = "kind"; })[0]["entries"][1]["label"].is_null());
    // the whole list replaced: codes unique, colours checked; the codes a selection reads stay
    task::EditDictionaryRequest all;
    all.dataset = D;
    all.subject = "S1";
    all.name = "kind";
    all.entries = R"([{"code": 0, "name": "MEG"}, {"code": 5, "name": "ecog", "colour": "#abcdef"}])";
    all.replace = true;
    all.created_by = by;
    task::edit_dictionary(all);
    const Json replaced = list("dictionaries", [](task::ListClassRequest& q) { q.name = "kind"; })[0]["entries"];
    CHECK_EQ(replaced.size(), std::size_t(2));
    CHECK_EQ(replaced[1]["code"].get<std::int64_t>(), std::int64_t(5));
    CHECK_EQ(replaced[0]["name"].get<std::string>(), std::string("MEG"));
    // a selection's own dictionary (`parts`: left, right): an entry renamed; the list replaced without a code in use is refused, nothing changes
    task::EditDictionaryRequest rn;
    rn.dataset = D;
    rn.subject = "S1";
    rn.name = "parts";
    rn.entry = 1;
    rn.name_of = "RIGHT";
    rn.created_by = by;
    task::edit_dictionary(rn);
    const Json snap5 = read();
    const Json* parts = named(snap5["selections"], "parts");
    CHECK(parts != nullptr);
    if (parts) CHECK_EQ((*parts)["entries"][1]["name"].get<std::string>(), std::string("RIGHT"));
    task::EditDictionaryRequest drop = all;
    drop.name = "parts";
    drop.entries = R"([{"code": 0, "name": "left"}])";
    CHECK(has(thrown([&] { task::edit_dictionary(drop); }), "read by a selection"));
    CHECK_EQ((*named(read()["selections"], "parts"))["entries"].size(), std::size_t(2));
    // refusals: nothing to change, both modes, an entry that does not exist, a bad colour, an unknown dictionary, an app dictionary
    task::EditDictionaryRequest bad = rn;
    bad.name_of.reset();
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "nothing to change"));
    bad.entry.reset();
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "name the entry to change"));
    bad = rn;
    bad.entries = R"([{"code": 0, "name": "a"}])";
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "replaces the whole list"));
    bad = rn;
    bad.entry = 9;
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "no entry with the code 9"));
    bad = rn;
    bad.colour = "blue";
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "not #rrggbb"));
    bad = rn;
    bad.name = "nothing";
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "no_such_row"));
    bad = rn;
    bad.name = "frequency bands";
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "the program's own"));
    bad = rn;
    bad.entries = R"([{"code": 0, "name": "a"}])";
    bad.entry.reset();
    bad.name_of.reset();
    bad.replace = false;
    CHECK(has(thrown([&] { task::edit_dictionary(bad); }), "--replace"));
  }

  // ---- edit manifold (3.6): the primary flag moves in one unit
  {
    task::AddSurfaceRequest sa;
    sa.dataset = D; sa.subject = "S1"; sa.created_by = by;
    sa.name = "tetra_a";
    sa.vertices = task::ArrayInput::memory("vertices.npy", "float64", {4, 3}, raw(std::vector<double>{0, 0, 1, 0, 1, 0, 1, 0, 0, 0.5, 0.5, 0.5}));
    sa.faces = task::ArrayInput::memory("faces.npy", "int32", {4, 3}, raw(std::vector<std::int32_t>{0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3}));
    task::add_surface(sa);                                                                              // primary (the default)
    task::AddSurfaceRequest sb = sa;
    sb.name = "tetra_b";
    sb.no_primary = true;
    task::add_surface(sb);
    auto primaries = [&] {
      std::vector<std::string> out;
      const Json surfaces = list("manifolds", [](task::ListClassRequest& q) { q.kind = "surface"; });
      for (const auto& r : surfaces) if (r["primary"].get<bool>()) out.push_back(r["name"].get<std::string>());
      return out;
    };
    CHECK(primaries() == (std::vector<std::string>{"tetra_a"}));
    task::EditManifoldRequest e;
    e.dataset = D; e.subject = "S1"; e.name = "tetra_b"; e.primary = "yes"; e.created_by = by;
    const task::CommandResult made = task::edit_manifold(e);
    CHECK(made.result["primary"].get<bool>());
    CHECK(primaries() == (std::vector<std::string>{"tetra_b"}));                                        // the flag moved: exactly one primary surface
    e.primary = "yes";
    e.label = "the second tetrahedron";
    e.description = "a test surface";
    task::edit_manifold(e);                                                                             // already primary: still one
    CHECK(primaries() == (std::vector<std::string>{"tetra_b"}));
    const Json snap6 = read();
    const Json* b = named(snap6["manifolds"], "tetra_b");
    CHECK(b != nullptr);
    if (b) {
      CHECK_EQ((*b)["label"].get<std::string>(), std::string("the second tetrahedron"));
      CHECK_EQ((*b)["description"].get<std::string>(), std::string("a test surface"));
    }
    e.name = "tetra_a";
    e.primary = "yes";
    e.label.reset();
    e.description.reset();
    task::edit_manifold(e);
    CHECK(primaries() == (std::vector<std::string>{"tetra_a"}));
    e.primary = "no";
    task::edit_manifold(e);
    CHECK(primaries().empty());
    e.primary = "yes";
    task::edit_manifold(e);
    // refusals: a non-surface made primary (and its label, in the same call, not applied), a value that is not yes or no, nothing to change, an unknown manifold
    task::EditManifoldRequest bad = e;
    bad.name = "sensors";
    bad.label = "should not be stored";
    CHECK(has(thrown([&] { task::edit_manifold(bad); }), "only a surface is primary"));
    const Json snap7 = read();
    const Json* sensors = named(snap7["manifolds"], "sensors");
    CHECK(sensors != nullptr && (*sensors)["label"].is_null());
    CHECK(primaries() == (std::vector<std::string>{"tetra_a"}));
    bad = e;
    bad.primary = "maybe";
    CHECK(has(thrown([&] { task::edit_manifold(bad); }), "yes or no"));
    bad = e;
    bad.primary.reset();
    CHECK(has(thrown([&] { task::edit_manifold(bad); }), "nothing to change"));
    bad.name = "missing";
    bad.primary = "yes";
    CHECK(has(thrown([&] { task::edit_manifold(bad); }), "no_such_row"));
    CHECK(primaries() == (std::vector<std::string>{"tetra_a"}));
  }

  return nxrtest::finish("test_add_options");
}
