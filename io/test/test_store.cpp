// The store interface on its one backend, the folder tree: get · ranged get · size · set · a streamed set · exists · list · erase · rename, and a prefix view.
#include <fstream>
#include "nxr/io/store.h"

#include <filesystem>
#include <string>

#include "check.h"
#include "nxr/io/errors.h"
#include "nxr/io/writer.h"

namespace fs = std::filesystem;
using namespace nxr::io;

namespace {
Bytes bytes(const std::string& s) { return Bytes(s.begin(), s.end()); }
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
}  // namespace

int main() {
  const fs::path tmp = fs::temp_directory_path() / "nxr_io_store";
  fs::remove_all(tmp);
  FolderStore st(tmp / "root");

  // set makes the keys above it; the key is the file root/key
  st.set("a/b/zarr.json", std::string("{}"));
  CHECK(fs::is_regular_file(tmp / "root/a/b/zarr.json"));
  CHECK(st.exists("a/b/zarr.json"));
  CHECK(!st.exists("a/b"));            // a prefix is not a value
  CHECK(st.exists_prefix("a/b") && st.exists_prefix("a") && !st.exists_prefix("a/c"));
  CHECK(st.get_text("a/b/zarr.json") == std::string("{}"));
  CHECK(!st.get("a/b/none").has_value() && !st.size("a/b/none").has_value() && !st.get_range("a/b/none", 0, 1).has_value());

  // a value, whole, by range, its size; a range past the end is refused
  st.set("a/c/0/1", bytes("0123456789"));
  CHECK(*st.size("a/c/0/1") == 10);
  CHECK(*st.get("a/c/0/1") == bytes("0123456789"));
  CHECK(*st.get_range("a/c/0/1", 3, 4) == bytes("3456"));
  CHECK(st.get_range("a/c/0/1", 10, 0)->empty());
  CHECK_EQ(code_of([&] { st.get_range("a/c/0/1", 8, 4); }), std::string(code::zarr_format));
  st.set("a/c/0/1", bytes("xy"));     // replaced, not appended
  CHECK(*st.get("a/c/0/1") == bytes("xy"));
  st.set("empty", Bytes{});
  CHECK(st.exists("empty") && *st.size("empty") == 0);

  // a streamed value, piece by piece
  {
    auto w = st.open_writer("s/shard");
    const Bytes p1 = bytes("abc"), p2 = bytes("defg");
    w->append(p1.data(), p1.size());
    w->append(p2.data(), p2.size());
    w->close();
  }
  CHECK(*st.get("s/shard") == bytes("abcdefg"));

  // list: the immediate children, names only, sorted
  const Listing l = st.list("a");
  CHECK(l.keys.empty() && l.prefixes == (std::vector<std::string>{"b", "c"}));
  const Listing r = st.list("");
  CHECK(r.keys == (std::vector<std::string>{"empty"}) && r.prefixes == (std::vector<std::string>{"a", "s"}));
  CHECK(st.list("a/b").keys == (std::vector<std::string>{"zarr.json"}));
  CHECK(st.list("nothing").keys.empty() && st.list("nothing").prefixes.empty());

  // keys cannot leave the store
  CHECK_EQ(code_of([&] { st.get("../x"); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { st.set("a/../../x", std::string("")); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { st.get("/etc/passwd"); }), std::string(code::invalid_argument));
  CHECK_EQ(code_of([&] { st.erase_prefix(""); }), std::string(code::invalid_argument));

  // rename moves a whole prefix, refuses an existing target
  st.rename_prefix("a/c", "moved/deep/c");
  CHECK(!st.exists_prefix("a/c") && *st.get("moved/deep/c/0/1") == bytes("xy"));
  CHECK_EQ(code_of([&] { st.rename_prefix("a/b", "moved"); }), std::string(code::already_exists));

  // erase a value, erase a prefix (and what is not there is nothing)
  st.erase("moved/deep/c/0/1");
  CHECK(!st.exists("moved/deep/c/0/1"));
  st.erase("moved/deep/c/0/1");
  st.erase_prefix("a");
  CHECK(!st.exists_prefix("a") && !fs::exists(tmp / "root/a"));
  st.erase_prefix("a");
  CHECK(st.exists("empty") && st.exists("s/shard"));

  // a prefix view: the keys of a subject inside its dataset's store
  st.set("ds/sub/x/zarr.json", std::string("{\"k\":1}"));
  st.set("ds/sub/x/c/0", bytes("zz"));
  PrefixStore sub(st, "ds/sub");
  CHECK(sub.get_text("x/zarr.json") == std::string("{\"k\":1}"));
  CHECK(*sub.get_range("x/c/0", 1, 1) == bytes("z") && *sub.size("x/c/0") == 2);
  CHECK(sub.list("x").prefixes == (std::vector<std::string>{"c"}));
  sub.set("y/zarr.json", std::string("{}"));
  CHECK(st.exists("ds/sub/y/zarr.json"));
  sub.rename_prefix("y", "z");
  CHECK(st.exists("ds/sub/z/zarr.json") && !sub.exists_prefix("y"));
  sub.erase_prefix("x");
  CHECK(!st.exists_prefix("ds/sub/x") && st.exists_prefix("ds/sub/z"));
  CHECK(join_key("", "a") == "a" && join_key("a", "") == "a" && join_key("a", "b/c") == "a/b/c");

  // the Studio's node removal over a store: exact names, never through a link
  {
    FolderStore base(tmp / "base");
    PrefixStore subj(base, "ds/s.nxr.zarr");
    subj.set("source/x/zarr.json", std::string("{}"));
    subj.set("source/x/c/0", bytes("1"));
    subj.set("source/y/zarr.json", std::string("{}"));
    fs::create_directory_symlink(tmp / "base/ds/s.nxr.zarr/source/y", tmp / "base/ds/s.nxr.zarr/source/link");
    CHECK(subj.is_link("source/link") && !subj.is_link("source/x"));
    CHECK(!remove_store_node(subj, "source/none"));
    CHECK(remove_store_node(subj, "source/x") && !subj.exists_prefix("source/x") && subj.exists_prefix("source"));
    CHECK_EQ(code_of([&] { remove_store_node(subj, "source/link"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { remove_store_node(subj, "source/../x"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { remove_store_node(subj, "source/y/zarr.json"); }), std::string(code::invalid_argument));   // a value, not a group
    CHECK(subj.exists("source/y/zarr.json"));
  }

  // a key that resolves outside the root (through a link inside the store) is refused by every delete; the outside stays
  {
    fs::create_directories(tmp / "outside/victim");
    { std::ofstream f(tmp / "outside/victim/keep"); f << "x"; }
    FolderStore base(tmp / "jail");
    base.set("in/zarr.json", std::string("{}"));
    fs::create_directory_symlink(tmp / "outside", tmp / "jail/esc");
    CHECK_EQ(code_of([&] { base.erase_prefix("esc/victim"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { base.erase("esc/victim/keep"); }), std::string(code::invalid_argument));
    CHECK(fs::exists(tmp / "outside/victim/keep"));
    // the writes are refused the same way: a set, a streamed value and a rename into a link that leads out; a link at the key itself is not written through
    CHECK_EQ(code_of([&] { base.set("esc/victim/keep", std::string("changed")); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { base.open_writer("esc/victim/new"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { base.rename_prefix("in", "esc/victim/moved"); }), std::string(code::invalid_argument));
    fs::create_symlink(tmp / "outside/victim/keep", tmp / "jail/flink");
    CHECK_EQ(code_of([&] { base.set("flink", std::string("changed")); }), std::string(code::invalid_argument));
    { std::ifstream i(tmp / "outside/victim/keep"); std::string t; i >> t; CHECK(t == "x"); }
    CHECK(!fs::exists(tmp / "outside/victim/new") && !fs::exists(tmp / "outside/victim/moved") && base.exists_prefix("in"));
    fs::remove(tmp / "jail/flink");
    base.erase_prefix("esc");   // the link itself goes, never its target
    CHECK(!fs::exists(fs::symlink_status(tmp / "jail/esc")) && fs::exists(tmp / "outside/victim/keep"));
    base.erase_prefix("in");
    CHECK(!base.exists_prefix("in"));
  }

  fs::remove_all(tmp);
  return nxrtest::finish("test_store");
}
