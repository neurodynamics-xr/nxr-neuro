// `read subject` and `list <class>` (docs/specs/2026-11-20-data-layer-queries.md, steps 1): the description of every node of a subject as nxr-neuro's typed model, the lists of each class
// (paged by a cursor, filtered by name, kind and manifold, with only the columns asked for), the counts of `show dataset`; no key of any result is a column of the schema (`*_id`, `*_json`);
// a reader leaves every file of the dataset as it was. Synthetic data only (face::synthetic_dataset and the `add <class>` commands).
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <set>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/commands.h"
#include "nxr/pipelines/face.h"

using namespace nxr::io;
namespace face = nxr::pipelines::face;
namespace fs = std::filesystem;

namespace {
std::string bytes_of(const fs::path& f) { std::ifstream in(f, std::ios::binary); return std::string(std::istreambuf_iterator<char>(in), {}); }
std::map<std::string, std::string> tree(const fs::path& folder) {
  std::map<std::string, std::string> out;
  for (const auto& e : fs::recursive_directory_iterator(folder)) if (e.is_regular_file()) out[fs::relative(e.path(), folder).string()] = bytes_of(e.path());
  return out;
}
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
template <class T>
fs::path npy(const fs::path& file, const std::string& dtype, const std::vector<std::int64_t>& shape, const std::vector<T>& v) {
  std::vector<std::uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  write_npy(file, dtype, shape, b);
  return file;
}
// no key anywhere in a result is a column of the schema
bool clean(const Json& j, std::string path = "") {
  bool ok = true;
  if (j.is_object())
    for (auto it = j.begin(); it != j.end(); ++it) {
      const std::string k = it.key();
      auto ends = [&](const char* suffix) { const std::string s = suffix; return k.size() >= s.size() && k.compare(k.size() - s.size(), s.size(), s) == 0; };
      if (ends("_id") || ends("_json")) { std::printf("  schema column '%s' at %s\n", k.c_str(), path.c_str()); ok = false; }
      ok = clean(it.value(), path + "/" + k) && ok;
    }
  else if (j.is_array())
    for (const auto& x : j) ok = clean(x, path + "[]") && ok;
  return ok;
}
const Json* named(const Json& rows, const std::string& name) {
  for (const auto& r : rows) if (r["name"] == name) return &r;
  return nullptr;
}
std::vector<std::string> names(const Json& rows) {
  std::vector<std::string> v;
  for (const auto& r : rows) v.push_back(r["name"].get<std::string>());
  return v;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_read_subject_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root / "in");
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);
  const std::string D = folder.string(), by = face::stamp();
  auto base = [&](auto& r, const std::string& name, const std::string& subject = "S1") { r.dataset = D; r.subject = subject; r.name = name; r.created_by = by; };

  // ---- what the subject holds: the synthetic recording, plus one of each class made by `add <class>`
  {
    task::AddFieldRequest f;
    base(f, "w");
    f.manifold = "sensors";
    f.values = npy(root / "in" / "w.npy", "float32", {3}, std::vector<float>{0.5f, -1.25f, 3.0f}).string();
    f.unit = "mm";
    f.description = "a test map";
    task::add_field(f);
    task::AddLineRequest l;
    base(l, "epoch_time");
    l.n = 50;
    l.spacing = 0.01;
    l.unit = "s";
    task::add_line(l);
    std::vector<double> series(3 * 50, 1.0);
    task::AddFieldRequest s;
    base(s, "series");
    s.manifold = "sensors";
    s.time = "epoch_time";
    s.values = npy(root / "in" / "series.npy", "float64", {3, 50}, series).string();
    task::add_field(s);
    task::AddSelectionRequest p;
    base(p, "parts");
    p.manifold = "sensors";
    p.type = "partition";
    p.members = npy(root / "in" / "parts.npy", "int32", {3}, std::vector<std::int32_t>{0, 1, 0}).string();
    p.entries = {"left", "right"};
    task::add_selection(p);
    task::AddOperatorRequest o;
    base(o, "mix");
    o.from_manifold = "sensors";
    o.to_manifold = "sensors";
    o.values = npy(root / "in" / "mix.npy", "float64", {3, 3}, std::vector<double>{1, 0, 0, 0, 2, 0, 0.5, 0, 3}).string();
    task::add_operator(o);
    task::AddLineRequest fl;
    base(fl, "freq");
    fl.n = 4;
    fl.spacing = 1.0;
    fl.origin = 1.0;
    fl.unit = "Hz";
    fl.dimension = "frequency";
    task::add_line(fl);
    task::AddStatisticRequest st;
    base(st, "psd");
    st.measure = "power_density";
    st.method = "welch";
    st.axes = {"channel:sensors", "frequency:freq"};
    st.of_field = "rest";
    st.values = npy(root / "in" / "psd.npy", "float32", {3, 4}, std::vector<float>(12, 0.5f)).string();
    task::add_statistic(st);
    // a second subject with its own points and a map on them
    task::CreateSubjectRequest cs;
    cs.dataset = D;
    cs.name = "S2";
    cs.created_by = by;
    task::create_subject(cs);
    task::AddPointsRequest pts;
    base(pts, "pts", "S2");
    pts.positions = npy(root / "in" / "pts.npy", "float64", {2, 3}, std::vector<double>{0, 0, 0, 1, 0, 0}).string();
    task::add_points(pts);
    task::AddFieldRequest pf;
    base(pf, "pf", "S2");
    pf.manifold = "pts";
    pf.values = npy(root / "in" / "pf.npy", "float64", {2}, std::vector<double>{1, 2}).string();
    task::add_field(pf);
  }

  const auto before = tree(folder);
  auto read = [&](const std::string& subject) { task::ReadSubjectRequest q; q.dataset = D; q.subject = subject; return face::read_subject(q).result; };
  auto list = [&](const std::string& cls, const std::function<void(task::ListClassRequest&)>& set = {}) {
    task::ListClassRequest q;
    q.cls = cls;
    q.dataset = D;
    if (set) set(q);
    return face::list_class(q).result;
  };

  // ---- read subject
  const Json sub = read("S1");
  CHECK_EQ(sub["schema"].get<std::string>(), std::string("nxr-neuro-subject@1"));
  CHECK_EQ(sub["subject"]["name"].get<std::string>(), std::string("S1"));
  CHECK(clean(sub));
  for (const char* k : {"manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions"}) CHECK(sub[k].is_array());
  CHECK(!sub.contains("related"));
  {
    const Json* sensors = named(sub["manifolds"], "sensors");
    CHECK(sensors != nullptr);
    if (sensors) {
      CHECK_EQ((*sensors)["kind"].get<std::string>(), std::string("points"));
      CHECK_EQ((*sensors)["size"]["vertices"].get<std::int64_t>(), std::int64_t(3));
      CHECK_EQ((*sensors)["type"]["kind"].get<std::string>(), std::string("points"));          // a manifold's type is its term in the type system
      CHECK_EQ((*sensors)["type"]["coordinate_system"]["space"].get<std::string>(), std::string("head"));
      CHECK_EQ((*sensors)["type"]["coordinate_system"]["name"].get<std::string>(), std::string("scs"));
      CHECK_EQ((*sensors)["type"]["n"].get<std::int64_t>(), std::int64_t(3));
    }
    const Json* rest_time = named(sub["manifolds"], "rest_time");
    CHECK(rest_time != nullptr);
    if (rest_time) CHECK_EQ((*rest_time)["type"]["kind"].get<std::string>(), std::string("line"));
    // the storage words are gone: a node is described by its type
    for (const char* k : {"value_type", "cells", "data_type", "layout", "domain", "components", "unit", "time", "axes", "coordinate_system", "topology"}) {
      for (const auto& f : sub["fields"]) CHECK(!f.contains(k));
      for (const auto& m : sub["manifolds"]) CHECK(!m.contains(k));
    }
    for (const auto& o : sub["operators"]) CHECK(!o.contains("layout") && !o.contains("data_type") && !o["from"].contains("value_type") && !o["from"].contains("cells"));
    for (const auto& st : sub["statistics"]) CHECK(!st.contains("value_type") && !st.contains("components") && !st.contains("unit"));
    const Json* w = named(sub["fields"], "w");
    CHECK(w != nullptr);
    if (w) {
      CHECK_EQ((*w)["type"]["class"].get<std::string>(), std::string("field"));
      CHECK_EQ((*w)["type"]["domain"]["factors"][0]["label"].get<std::string>(), std::string("sensors"));
      CHECK_EQ((*w)["type"]["unit"].get<std::string>(), std::string("mm"));
      CHECK_EQ((*w)["type"]["value"].get<std::string>(), std::string("real"));
      CHECK((*w)["stored"].get<bool>());
      CHECK_EQ((*w)["description"].get<std::string>(), std::string("a test map"));
    }
    const Json* series = named(sub["fields"], "series");
    CHECK(series != nullptr);
    if (series) CHECK_EQ((*series)["type"]["domain"]["factors"].size(), std::size_t(2));
    const Json* rest = named(sub["fields"], "rest");
    CHECK(rest != nullptr);
    if (rest) {
      CHECK_EQ((*rest)["type"]["sfreq"].get<double>(), 100.0);
      // the same type `compile` gives the same stored field (the recording input of psd)
      face::CompileRequest cq;
      cq.definition = "psd";
      cq.dataset = D;
      cq.subject = "S1";
      cq.parameters = {{"recording", "rest"}};
      const Json compiled = face::compile(cq).result;
      CHECK((*rest)["type"] == compiled["plans"][0]["graph"]["values"]["recording"]);
    }
    const Json* parts = named(sub["selections"], "parts");
    CHECK(parts != nullptr);
    if (parts) {
      CHECK_EQ((*parts)["kind"].get<std::string>(), std::string("set"));   // a partition is stored as a set: a member per element
      CHECK_EQ((*parts)["type"]["class"].get<std::string>(), std::string("selection"));
      CHECK(!parts->contains("domain"));
      CHECK_EQ((*parts)["members"].get<std::int64_t>(), std::int64_t(2));
      CHECK_EQ((*parts)["entries"].size(), std::size_t(2));
      CHECK_EQ((*parts)["entries"][1]["name"].get<std::string>(), std::string("right"));
    }
    CHECK(named(sub["selections"], "rest_bad_segments") != nullptr);
    const Json* mix = named(sub["operators"], "mix");
    CHECK(mix != nullptr);
    if (mix) {
      CHECK_EQ((*mix)["type"]["class"].get<std::string>(), std::string("operator"));
      CHECK_EQ((*mix)["from"]["manifold"].get<std::string>(), std::string("sensors"));
      CHECK_EQ((*mix)["to"]["manifold"].get<std::string>(), std::string("sensors"));
      CHECK_EQ((*mix)["shape"]["rows"].get<std::int64_t>(), std::int64_t(3));
      CHECK_EQ((*mix)["kind"].get<std::string>(), std::string("added"));
    }
    const Json* psd = named(sub["statistics"], "psd");
    CHECK(psd != nullptr);
    if (psd) {
      CHECK_EQ((*psd)["measure"].get<std::string>(), std::string("power_density"));
      CHECK_EQ((*psd)["of_field"].get<std::string>(), std::string("rest"));
      CHECK_EQ((*psd)["axes"].size(), std::size_t(2));
      CHECK_EQ((*psd)["axes"][0]["manifold"].get<std::string>(), std::string("sensors"));
    }
    CHECK(named(sub["spaces"], "head") != nullptr);
    CHECK(!sub["dictionaries"].empty());
  }
  {
    task::ReadSubjectRequest q;
    q.dataset = D;
    q.subject = "S1";
    q.classes = std::vector<std::string>{"fields", "manifolds"};
    const Json some = face::read_subject(q).result;
    CHECK(some.contains("fields") && some.contains("manifolds") && !some.contains("operators") && !some.contains("statistics"));
    CHECK_EQ(some["fields"].size(), sub["fields"].size());
    q.classes = std::vector<std::string>{"fields", "rows"};
    CHECK(has(thrown([&] { face::read_subject(q); }), "refused: --classes: 'rows' is not a class"));
    q.classes.reset();
    q.related = true;
    const Json rel = face::read_subject(q).result;
    CHECK(rel["related"].is_array());
    CHECK(clean(rel));
    q.subject = "nobody";
    CHECK(has(thrown([&] { face::read_subject(q); }), "no_such_row"));
  }

  // ---- list <class>: the same descriptions, a row each, the subject first
  for (const char* cls : {"manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions"}) {
    const Json l = list(cls, [](task::ListClassRequest& q) { q.subject = "S1"; });
    CHECK(clean(l));
    CHECK_EQ(l["schema"].get<std::string>(), std::string("nxr-neuro-list@1"));
    CHECK(l["next"].is_null());
    CHECK_EQ(l["rows"].size(), sub[cls].size());
    for (std::size_t i = 0; i < l["rows"].size() && i < sub[cls].size(); ++i) {
      Json row = l["rows"][i];
      CHECK_EQ(row["subject"].get<std::string>(), std::string("S1"));
      row.erase("subject");
      CHECK(row == sub[cls][i]);   // a row of a list is the node's description, as `read subject` gives it
    }
  }
  CHECK(list("jobs")["rows"].empty());
  CHECK(list("jobs", [](task::ListClassRequest& q) { q.subject = "S1"; })["rows"].empty());

  // ---- a one-database dataset lists every subject's nodes, each row naming its own
  {
    const Json all = list("fields");
    std::set<std::string> subjects;
    for (const auto& r : all["rows"]) subjects.insert(r["subject"].get<std::string>());
    CHECK(subjects == (std::set<std::string>{"S1", "S2"}));
    CHECK_EQ(all["rows"].size(), sub["fields"].size() + 1);
    // paging: every row once, in the same order, whatever the page
    for (const std::int64_t page : {1, 2, 3}) {
      std::vector<std::string> seen;
      std::optional<std::string> after;
      int pages = 0;
      for (;; ++pages) {
        const Json p = list("fields", [&](task::ListClassRequest& q) { q.limit = page; q.after = after; });
        CHECK(p["rows"].size() <= static_cast<std::size_t>(page));
        for (const auto& r : p["rows"]) seen.push_back(r["subject"].get<std::string>() + "/" + r["name"].get<std::string>());
        if (p["next"].is_null()) break;
        after = p["next"].get<std::string>();
        CHECK(pages < 100);
      }
      std::vector<std::string> expect;
      for (const auto& r : all["rows"]) expect.push_back(r["subject"].get<std::string>() + "/" + r["name"].get<std::string>());
      CHECK(seen == expect);
    }
    CHECK(has(thrown([&] { list("fields", [](task::ListClassRequest& q) { q.after = "not a cursor"; }); }), "refused: --after: not a cursor"));
    CHECK(has(thrown([&] { list("fields", [](task::ListClassRequest& q) { q.limit = 0; }); }), "refused: --limit"));
    CHECK(has(thrown([&] { list("beans"); }), "is not a class of nodes"));
    // a page that ends exactly at the end gives no cursor
    const Json exact = list("fields", [&](task::ListClassRequest& q) { q.limit = static_cast<std::int64_t>(all["rows"].size()); });
    CHECK(exact["next"].is_null());
  }

  // ---- the filters: named options
  {
    const Json byname = list("fields", [](task::ListClassRequest& q) { q.subject = "S1"; q.name = "SER"; });
    CHECK(names(byname["rows"]) == (std::vector<std::string>{"series"}));    // contains, case-insensitive
    const Json none = list("fields", [](task::ListClassRequest& q) { q.name = "100%_"; });
    CHECK(none["rows"].empty());                                              // % and _ are not wildcards
    const Json rest = *named(sub["fields"], "rest");
    const std::string kind = rest["kind"].get<std::string>();
    const Json bykind = list("fields", [&](task::ListClassRequest& q) { q.subject = "S1"; q.kind = kind; });
    CHECK(named(bykind["rows"], "rest") != nullptr);
    for (const auto& r : bykind["rows"]) CHECK_EQ(r["kind"].get<std::string>(), kind);
    CHECK(bykind["rows"].size() < sub["fields"].size());
    const Json on = list("fields", [](task::ListClassRequest& q) { q.subject = "S1"; q.on = "sensors"; });
    CHECK(named(on["rows"], "w") && named(on["rows"], "series") && named(on["rows"], "rest"));   // on the manifold, or on a product of it
    const Json time_on = list("fields", [](task::ListClassRequest& q) { q.subject = "S1"; q.on = "epoch_time"; });
    CHECK(names(time_on["rows"]) == (std::vector<std::string>{"series"}));
    CHECK(has(thrown([&] { list("fields", [](task::ListClassRequest& q) { q.subject = "S1"; q.on = "nowhere"; }); }), "no_such_row"));
    CHECK(names(list("selections", [](task::ListClassRequest& q) { q.subject = "S1"; q.on = "sensors"; q.kind = "set"; })["rows"]).size() >= 1);
    CHECK(names(list("operators", [](task::ListClassRequest& q) { q.subject = "S1"; q.on = "sensors"; })["rows"]) == (std::vector<std::string>{"mix"}));
    CHECK(names(list("statistics", [](task::ListClassRequest& q) { q.subject = "S1"; q.on = "freq"; })["rows"]) == (std::vector<std::string>{"psd"}));
    CHECK(names(list("statistics", [](task::ListClassRequest& q) { q.subject = "S1"; q.kind = "count"; })["rows"]).empty());
    const Json products = list("manifolds", [](task::ListClassRequest& q) { q.subject = "S1"; q.on = "epoch_time"; });
    for (const auto& r : products["rows"]) CHECK_EQ(r["kind"].get<std::string>(), std::string("product"));
    CHECK(names(list("manifolds", [](task::ListClassRequest& q) { q.subject = "S2"; })["rows"]) == (std::vector<std::string>{"pts"}));
    CHECK(has(thrown([&] { list("acquisitions", [](task::ListClassRequest& q) { q.kind = "x"; }); }), "refused: list acquisitions"));
    CHECK(has(thrown([&] { list("jobs", [](task::ListClassRequest& q) { q.name = "x"; }); }), "a job has no name"));
    CHECK(has(thrown([&] { list("spaces", [](task::ListClassRequest& q) { q.on = "sensors"; }); }), "refused: list spaces"));
  }

  // ---- --columns: only these, in this order; a name that is not a column is refused with the valid ones; the same columns whether or not a row exists
  {
    const Json c = list("fields", [](task::ListClassRequest& q) { q.subject = "S1"; q.columns = std::vector<std::string>{"kind", "name"}; });
    CHECK(!c["rows"].empty());
    for (const auto& r : c["rows"]) {
      CHECK_EQ(r.size(), std::size_t(2));
      CHECK(r.contains("kind") && r.contains("name"));
      CHECK_EQ(r.begin().key(), std::string("kind"));
    }
    CHECK(has(thrown([&] { list("fields", [](task::ListClassRequest& q) { q.columns = std::vector<std::string>{"geometry_id"}; }); }), "refused: --columns: 'geometry_id' is not a column of fields"));
    for (const char* cls : {"manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions"}) {
      const Json full = list(cls, [](task::ListClassRequest& q) { q.subject = "S1"; });
      std::vector<std::string> keys;
      if (!full["rows"].empty()) for (auto it = full["rows"][0].begin(); it != full["rows"][0].end(); ++it) keys.push_back(it.key());
      if (keys.empty()) keys = {"subject", "id", "name"};
      const Json empty = list(cls, [&](task::ListClassRequest& q) { q.subject = "S1"; q.name = "zzzz-no-such-node"; q.columns = keys; });
      CHECK(empty["rows"].empty());                                           // the class's own column list accepts every key a description has
    }
  }

  // ---- nothing is written by a read or a list
  CHECK(tree(folder) == before);

  // ---- show dataset: the count of each class (what `list` returns, summed over the subjects)
  {
    task::ShowDatasetRequest q{D};
    const Json shown = task::show_dataset(q).result;
    CHECK(clean(shown));
    for (const char* cls : {"manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions", "jobs"}) {
      std::int64_t sum = 0;
      for (const char* s : {"S1", "S2"}) sum += static_cast<std::int64_t>(list(cls, [&](task::ListClassRequest& r) { r.subject = s; })["rows"].size());
      CHECK_EQ(shown["nodes"][cls].get<std::int64_t>(), sum);
    }
    CHECK_EQ(shown["nodes"]["fields"].get<std::int64_t>(), static_cast<std::int64_t>(sub["fields"].size() + 1));
    CHECK_EQ(shown["nodes"]["jobs"].get<std::int64_t>(), std::int64_t(0));

    // the same dataset as a catalogue of subject packages: the same counts, and a package is listed one subject at a time
    task::SplitDatasetRequest split;
    split.dataset = D;
    split.registered_by = by;
    task::split_dataset(split);
    const Json cat = task::show_dataset(q).result;
    CHECK(clean(cat));
    CHECK(cat["nodes"] == shown["nodes"]);
    CHECK(has(thrown([&] { list("fields"); }), "a catalogue of subject packages is listed one subject at a time"));
    const Json packaged = list("fields", [](task::ListClassRequest& r) { r.subject = "S1"; });
    CHECK_EQ(packaged["rows"].size(), sub["fields"].size());
    const Json again = read("S1");
    CHECK(again["fields"] == sub["fields"] && again["manifolds"] == sub["manifolds"] && again["statistics"] == sub["statistics"]);
  }
  return nxrtest::finish("test_read_subject");
}
