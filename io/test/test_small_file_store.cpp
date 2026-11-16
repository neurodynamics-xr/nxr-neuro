// The small-file store (format 52): SmallFileStore over a FolderStore: every Store operation, the threshold routing, the spill of a streamed value, the list that
// merges rows and files, erase and rename across both; and DatastoreStore, the dataset-level store that gives each subject its own. Synthetic bytes only.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "check.h"
#include "nxr/io/errors.h"
#include "nxr/io/small_file_store.h"

namespace fs = std::filesystem;
using namespace nxr::io;

namespace {
Bytes bytes(const std::string& s) { return Bytes(s.begin(), s.end()); }
Bytes filled(std::size_t n, std::uint8_t v) { return Bytes(n, v); }
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
bool is_file(const fs::path& p) { return fs::is_regular_file(p); }
}  // namespace

int main() {
  const fs::path tmp = fs::temp_directory_path() / "nxr_io_small_file_store";
  fs::remove_all(tmp);
  const fs::path folder = tmp / "sub";
  SmallFileOptions opt;
  opt.threshold = 64;
  {
    FolderStore files(folder);
    SmallFileStore st(files, folder / kSmallFilesFile, opt);

    // nothing yet: no database is made by a read
    CHECK(!st.get("a/zarr.json").has_value() && !st.exists("a/zarr.json") && !st.exists_prefix("a"));
    CHECK(!fs::exists(folder / kSmallFilesFile));

    // a small value is a row (no file), a large one a file (no row)
    st.set("a/zarr.json", std::string("{}"));
    CHECK(st.in_table("a/zarr.json") && !is_file(folder / "a/zarr.json"));
    st.set("a/c.0", filled(100, 7));
    CHECK(!st.in_table("a/c.0") && is_file(folder / "a/c.0"));
    CHECK(fs::exists(folder / kSmallFilesFile));
    CHECK(st.get_text("a/zarr.json") == std::string("{}"));
    CHECK(*st.size("a/zarr.json") == 2 && *st.size("a/c.0") == 100);
    CHECK(st.exists("a/zarr.json") && st.exists("a/c.0") && !st.exists("a") && st.exists_prefix("a") && !st.exists_prefix("b"));
    CHECK(!st.get("a/none").has_value() && !st.size("a/none").has_value() && !st.get_range("a/none", 0, 1).has_value());
    st.set("empty", Bytes{});   // an empty value is small
    CHECK(st.in_table("empty") && st.exists("empty") && *st.size("empty") == 0);
    // the threshold: 63 bytes is a row, 64 a file
    st.set("t63", filled(63, 1));
    st.set("t64", filled(64, 1));
    CHECK(st.in_table("t63") && !st.in_table("t64") && is_file(folder / "t64"));

    // ranges, from a row and from a file; past the end is refused (the code a FolderStore gives)
    st.set("r/s", bytes("0123456789"));
    CHECK(*st.get_range("r/s", 3, 4) == bytes("3456"));
    CHECK(st.get_range("r/s", 10, 0)->empty());
    CHECK_EQ(code_of([&] { st.get_range("r/s", 8, 4); }), std::string(code::zarr_format));
    Bytes big(200);
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<std::uint8_t>(i);
    st.set("r/l", big);
    CHECK(*st.get_range("r/l", 100, 50) == Bytes(big.begin() + 100, big.begin() + 150));
    CHECK_EQ(code_of([&] { st.get_range("r/l", 190, 20); }), std::string(code::zarr_format));

    // a key lives in one place: small -> large -> small
    st.set("m", filled(10, 1));
    CHECK(st.in_table("m") && !is_file(folder / "m"));
    st.set("m", filled(100, 2));
    CHECK(!st.in_table("m") && is_file(folder / "m") && *st.get("m") == filled(100, 2));
    st.set("m", filled(10, 3));
    CHECK(st.in_table("m") && !is_file(folder / "m") && *st.get("m") == filled(10, 3));
    st.set("m", bytes("xy"));   // replaced, not appended
    CHECK(*st.get("m") == bytes("xy"));

    // a streamed value: held while small, a file once it passes the threshold
    {
      auto w = st.open_writer("w/small");
      w->append(bytes("abc").data(), 3);
      w->append(bytes("defg").data(), 4);
      CHECK(!st.exists("w/small"));   // it comes into being at close
      w->close();
      CHECK(st.in_table("w/small") && *st.get("w/small") == bytes("abcdefg"));
    }
    {
      auto w = st.open_writer("w/large");
      const Bytes p1 = filled(40, 5), p2 = filled(40, 6), p3 = filled(10, 7);
      w->append(p1.data(), p1.size());
      w->append(p2.data(), p2.size());   // 80 > 64: spilled to a file
      CHECK(is_file(folder / "w/large"));
      w->append(p3.data(), p3.size());
      w->close();
      CHECK(!st.in_table("w/large") && st.size("w/large") == std::optional<std::uint64_t>(90));
      Bytes want = p1;
      want.insert(want.end(), p2.begin(), p2.end());
      want.insert(want.end(), p3.begin(), p3.end());
      CHECK(*st.get("w/large") == want);
    }
    {   // a streamed value replacing a row (small) and a file (large)
      st.set("w/rep", bytes("old"));
      auto w = st.open_writer("w/rep");
      const Bytes p = filled(100, 9);
      w->append(p.data(), p.size());
      w->close();
      CHECK(!st.in_table("w/rep") && *st.get("w/rep") == p);
      auto w2 = st.open_writer("w/rep");
      w2->append(bytes("new").data(), 3);
      w2->close();
      CHECK(st.in_table("w/rep") && !is_file(folder / "w/rep") && *st.get("w/rep") == bytes("new"));
    }

    // list merges rows and files: names only, sorted; folder prefixes come from the keys; the database file is not a key
    st.set("l/x/zarr.json", std::string("{}"));
    st.set("l/y", bytes("1"));
    st.set("l/z/big", filled(100, 1));
    st.set("l/z/zarr.json", std::string("{}"));
    const Listing l = st.list("l");
    CHECK(l.keys == (std::vector<std::string>{"y"}));
    CHECK(l.prefixes == (std::vector<std::string>{"x", "z"}));
    const Listing lz = st.list("l/z");
    CHECK(lz.keys == (std::vector<std::string>{"big", "zarr.json"}) && lz.prefixes.empty());
    const Listing root = st.list("");
    for (const auto& k : root.keys) CHECK(k.rfind(kSmallFilesFile, 0) != 0);
    CHECK(std::find(root.prefixes.begin(), root.prefixes.end(), "l") != root.prefixes.end());
    CHECK(std::find(root.keys.begin(), root.keys.end(), "empty") != root.keys.end() && std::find(root.keys.begin(), root.keys.end(), "t64") != root.keys.end());
    CHECK(st.list("none").keys.empty() && st.list("none").prefixes.empty());

    // erase a key from either place; erase_prefix from both
    st.erase("l/y");
    st.erase("l/z/big");
    CHECK(!st.exists("l/y") && !st.exists("l/z/big") && !is_file(folder / "l/z/big"));
    st.erase("l/none");   // nothing
    st.set("e/a/zarr.json", std::string("{}"));
    st.set("e/a/c.0", filled(100, 1));
    st.set("e/b", bytes("x"));
    st.set("ee", bytes("x"));   // not under "e"
    st.erase_prefix("e");
    CHECK(!st.exists_prefix("e") && !st.exists("e/a/zarr.json") && !st.exists("e/a/c.0") && !st.exists("e/b") && st.exists("ee"));
    CHECK_EQ(code_of([&] { st.erase_prefix(""); }), std::string(code::invalid_argument));

    // rename_prefix moves rows and files; refused when the target exists
    st.set("from/n/zarr.json", std::string("{}"));
    st.set("from/n/c.0", filled(100, 4));
    st.set("from/m", bytes("v"));
    st.set("fromage", bytes("untouched"));
    st.rename_prefix("from", "to/deep");
    CHECK(!st.exists_prefix("from") && st.exists("fromage"));
    CHECK(st.get_text("to/deep/n/zarr.json") == std::string("{}") && *st.get("to/deep/n/c.0") == filled(100, 4) && st.get_text("to/deep/m") == std::string("v"));
    CHECK(st.in_table("to/deep/m") && is_file(folder / "to/deep/n/c.0"));
    st.set("other", bytes("o"));
    CHECK_EQ(code_of([&] { st.rename_prefix("other", "to"); }), std::string(code::already_exists));
    st.rename_prefix("other", "other2");   // one value
    CHECK(!st.exists("other") && st.get_text("other2") == std::string("o"));

    // a key may not leave the store
    CHECK_EQ(code_of([&] { st.set("../x", bytes("x")); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { st.get("/abs"); }), std::string(code::invalid_argument));

    // a batch is one transaction; a throw inside rolls it back
    st.batch([&] { st.put_row("b/1", bytes("1").data(), 1); st.put_row("b/2", bytes("2").data(), 1); });
    CHECK(st.in_table("b/1") && st.in_table("b/2"));
    try { st.batch([&] { st.put_row("b/3", bytes("3").data(), 1); throw std::runtime_error("stop"); }); } catch (const std::runtime_error&) {}
    CHECK(!st.in_table("b/3"));
  }

  // the data survives the store: a new store over the same folder, and a read-only one
  {
    FolderStore files(folder);
    SmallFileStore st(files, folder / kSmallFilesFile, opt);
    CHECK(st.get_text("a/zarr.json") == std::string("{}") && *st.get("a/c.0") == filled(100, 7) && st.get_text("to/deep/m") == std::string("v"));
    SmallFileOptions ro = opt;
    ro.read_only = true;
    SmallFileStore rd(files, folder / kSmallFilesFile, ro);
    CHECK(rd.get_text("a/zarr.json") == std::string("{}") && *rd.get("a/c.0") == filled(100, 7) && rd.list("a").keys.size() == 2);
    CHECK_EQ(code_of([&] { rd.set("a/new", bytes("x")); }), std::string(code::not_writable));
    CHECK_EQ(code_of([&] { rd.erase("a/zarr.json"); }), std::string(code::not_writable));
    // read-only over a folder with no database: empty, and none made
    FolderStore nofiles(tmp / "none");
    SmallFileStore none(nofiles, tmp / "none" / kSmallFilesFile, ro);
    CHECK(!none.exists("x") && none.list("").keys.empty());
    CHECK(!fs::exists(tmp / "none"));
  }
  // closed cleanly: no WAL files are left beside the database
  CHECK(!fs::exists(folder / (std::string(kSmallFilesFile) + "-wal")));

  // ── DatastoreStore: `<dataset>/<subject>/<key>` ──
  {
    const fs::path root = tmp / "datastore";
    fs::create_directories(root / "ds");
    DatastoreStore ds(root, opt);
    ds.set_small_files("ds", true);
    ds.set("ds/zarr.json", std::string("{\"dataset\":1}"));   // two segments: the folder tree's
    CHECK(is_file(root / "ds/zarr.json"));
    ds.set("ds/S.nxr.zarr/zarr.json", std::string("{}"));
    ds.set("ds/S.nxr.zarr/x/zarr.json", std::string("{}"));
    ds.set("ds/S.nxr.zarr/x/c.0", filled(100, 3));
    CHECK(is_file(root / "ds/S.nxr.zarr" / kSmallFilesFile));
    CHECK(!is_file(root / "ds/S.nxr.zarr/zarr.json") && !is_file(root / "ds/S.nxr.zarr/x/zarr.json") && is_file(root / "ds/S.nxr.zarr/x/c.0"));
    CHECK(ds.get_text("ds/S.nxr.zarr/x/zarr.json") == std::string("{}") && ds.exists_prefix("ds/S.nxr.zarr") && ds.exists_prefix("ds/S.nxr.zarr/x"));
    const Listing sl = ds.list("ds/S.nxr.zarr");
    CHECK(sl.keys == (std::vector<std::string>{"zarr.json"}) && sl.prefixes == (std::vector<std::string>{"x"}));
    const Listing dl = ds.list("ds");
    CHECK(std::find(dl.prefixes.begin(), dl.prefixes.end(), "S.nxr.zarr") != dl.prefixes.end());
    // a subject of a dataset not at format 52 is plain files
    ds.set_small_files("old", false);
    ds.set("old/S/zarr.json", std::string("{}"));
    CHECK(is_file(root / "old/S/zarr.json") && !fs::exists(root / "old/S" / kSmallFilesFile));
    // ... unless it already has its database (a dataset half way through compact)
    ds.small_files("old/S", true)->put_row("half/zarr.json", bytes("{}").data(), 2);
    CHECK(ds.get_text("old/S/half/zarr.json") == std::string("{}") && ds.get_text("old/S/zarr.json") == std::string("{}"));
    ds.set("old/S/half/small", bytes("q"));
    CHECK(ds.small_files("old/S", false)->in_table("half/small"));
    // a node moved aside into another folder (the replaced basis): rows become the plain files there
    ds.rename_prefix("ds/S.nxr.zarr/x", "_replaced-2026-11-15/S.nxr.zarr/x");
    CHECK(!ds.exists_prefix("ds/S.nxr.zarr/x") && ds.get_text("_replaced-2026-11-15/S.nxr.zarr/x/zarr.json") == std::string("{}"));
    CHECK(is_file(root / "_replaced-2026-11-15/S.nxr.zarr/x/zarr.json") && is_file(root / "_replaced-2026-11-15/S.nxr.zarr/x/c.0"));
    CHECK(*ds.get("_replaced-2026-11-15/S.nxr.zarr/x/c.0") == filled(100, 3));
    // erase_prefix inside a subject, and of a whole subject (its database goes with it)
    ds.set("ds/S.nxr.zarr/y/zarr.json", std::string("{}"));
    ds.erase_prefix("ds/S.nxr.zarr/y");
    CHECK(!ds.exists_prefix("ds/S.nxr.zarr/y"));
    ds.erase_prefix("ds/S.nxr.zarr");
    CHECK(!fs::exists(root / "ds/S.nxr.zarr") && !ds.exists_prefix("ds/S.nxr.zarr"));
    // a subject renamed whole carries its database
    ds.set("ds/T/zarr.json", std::string("{}"));
    ds.rename_prefix("ds/T", "ds/U");
    CHECK(ds.get_text("ds/U/zarr.json") == std::string("{}") && !ds.exists_prefix("ds/T") && is_file(root / "ds/U" / kSmallFilesFile));
    ds.close_all();
    CHECK(!fs::exists(root / "ds/U" / (std::string(kSmallFilesFile) + "-wal")));
  }

  // a second connection to the same file sees a created key at once (each write is committed when it returns): the Studio's Node side reads a subject the WASM module writes
  {
    const fs::path f2 = tmp / "two";
    FolderStore fa(f2), fb(f2);
    SmallFileStore a(fa, f2 / kSmallFilesFile, opt), b(fb, f2 / kSmallFilesFile, opt);
    CHECK(!b.exists("n/zarr.json"));
    a.set("n/zarr.json", std::string("{}"));
    CHECK(b.exists("n/zarr.json") && b.get_text("n/zarr.json") == std::string("{}") && b.exists_prefix("n"));
    CHECK(!b.exists("m/zarr.json"));
    a.set("m/zarr.json", std::string("{\"x\":1}"));
    CHECK(b.exists("m/zarr.json") && b.get_text("m/zarr.json") == std::string("{\"x\":1}"));
    a.erase("m/zarr.json");
    CHECK(!b.exists("m/zarr.json"));
  }

  // nothing is written, spilled or erased outside the store through a link: a row, a spill (a large value) and a streamed value under a folder link that leads
  // out are refused, and the file there stays
  {
    const fs::path outside = tmp / "outside", f3 = tmp / "linked";
    fs::create_directories(outside / "d");
    { std::ofstream o(outside / "d/victim"); o << "keep"; }
    FolderStore fl(f3);
    SmallFileStore st(fl, f3 / kSmallFilesFile, opt);
    st.set("in/zarr.json", std::string("{}"));
    fs::create_directory_symlink(outside, f3 / "esc");
    auto untouched = [&] { std::ifstream i(outside / "d/victim"); std::string t; i >> t; return t == "keep"; };
    CHECK_EQ(code_of([&] { st.set("esc/d/victim", std::string("{}")); }), std::string(code::invalid_argument));          // a small value: no row, no erase of the file
    CHECK_EQ(code_of([&] { st.set("esc/d/victim", filled(100, 1)); }), std::string(code::invalid_argument));              // a large value: no spill
    CHECK_EQ(code_of([&] { st.open_writer("esc/d/victim"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { st.erase("esc/d/victim"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { st.erase_prefix("esc/d"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { st.rename_prefix("in", "esc/d/moved"); }), std::string(code::invalid_argument));
    CHECK(untouched() && !st.in_table("esc/d/victim") && !fs::exists(outside / "d/moved") && st.exists("in/zarr.json"));
    // a link AT the key (a file link) is not written through either
    fs::create_symlink(outside / "d/victim", f3 / "flink");
    CHECK_EQ(code_of([&] { st.set("flink", filled(100, 2)); }), std::string(code::invalid_argument));
    CHECK(untouched());
    // the dataset-level store routes the same refusals
    DatastoreStore ds(tmp / "dsroot", opt, [](const std::string&) { return true; });
    ds.set("d/S/in/zarr.json", std::string("{}"));
    fs::create_directory_symlink(outside, tmp / "dsroot/d/S/esc");
    CHECK_EQ(code_of([&] { ds.set("d/S/esc/d/victim", std::string("{}")); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { ds.erase("d/S/esc/d/victim"); }), std::string(code::invalid_argument));
    CHECK(untouched());
    ds.close_all();
  }

  fs::remove_all(tmp);
  return nxrtest::finish("small_file_store");
}
