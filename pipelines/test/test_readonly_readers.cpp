// A READER never writes (issue 144): the face's reads, the command line's `show dataset` and `list` verbs, and io's `open_writer` / `open_package_writer` with `lock = false` open SQLite
// read-only. A dataset at schema 50 (made here by putting 50 back) is read AS IT IS: the file is byte for byte and mtime for mtime the same afterwards, the rows are returned, the
// tables 51 to 54 added are absent and the answer says so (no product factors, no statistics). Only a writable open, under the writer lock, migrates. A subject package made before
// a table of its format (`origin`) is read as it is too. A database in WAL mode that another connection holds (its commits still in the -wal) is read leaving the database file
// and its -wal unchanged; only SQLite's shared-memory index (-shm), which WAL readers write too, may change. Synthetic data only.
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/pipelines/face.h"

using namespace nxr::io;
namespace face = nxr::pipelines::face;
namespace fs = std::filesystem;

namespace {
std::string bytes_of(const fs::path& f) { std::ifstream in(f, std::ios::binary); return std::string(std::istreambuf_iterator<char>(in), {}); }
struct Stamp {
  std::string bytes;
  fs::file_time_type mtime;
  bool operator==(const Stamp& o) const { return bytes == o.bytes && mtime == o.mtime; }
};
Stamp stamp(const fs::path& f) { return {bytes_of(f), fs::last_write_time(f)}; }
// the same for every file beside the database (a reader leaves no -wal, -shm or journal behind)
std::map<std::string, Stamp> tree(const fs::path& folder) {
  std::map<std::string, Stamp> out;
  for (const auto& e : fs::recursive_directory_iterator(folder)) if (e.is_regular_file()) out[fs::relative(e.path(), folder).string()] = stamp(e.path());
  return out;
}
std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }

// schema 50 and the version read back: io's test-only support (the SQL stays in io)
void put_back_to_50(const fs::path& file) { nxr::io::testing::rewrite_as_schema_50(file); }
int version_of(const fs::path& file) { return nxr::io::testing::user_version_of(file); }
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_readonly_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};

  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);
  const fs::path file = folder / "dataset.sqlite";
  put_back_to_50(file);
  CHECK_EQ(version_of(file), 50);
  const auto before = tree(folder);

  // the face's readers
  {
    const face::Json subs = face::subjects(folder);
    CHECK_EQ(subs["subjects"].size(), std::size_t(1));
    CHECK_EQ(subs["subjects"][0]["name"].get<std::string>(), std::string("S1"));
    const face::Json rec = face::results(folder, "S1", "recordings");
    CHECK(rec.size() >= 1);
    CHECK(face::results(folder, "S1", "results").is_array());      // no statistic table: no statistics, no error
    const face::Json ds = face::datasets(root);
    CHECK(!ds.is_null());
  }
  CHECK(tree(folder) == before);                                   // byte for byte, mtime for mtime, no side file
  CHECK_EQ(version_of(file), 50);

  // io's readers: lock = false opens read-only, and the answers of the tables an older version lacks are empty
  {
    OpenOptions o;
    o.lock = false;
    DatasetWriter ds = open_writer(folder, o);
    CHECK(nxr::io::testing::Probe(ds).read_only());
    CHECK(!ds.locked());
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    task::Want want;
    want.cls = task::Cls::statistic;
    CHECK(task::Subject::of(sub).find_all(want).empty());
    CHECK_EQ(code_of([&] { nxr::io::testing::Probe(ds).tamper("CREATE TABLE x (a TEXT)"); }), std::string(code::sqlite));   // a write is refused by SQLite itself
    ds.close();
  }
  CHECK(tree(folder) == before);

  // the command line's read verbs
#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = folder.string(), log = (root / "cli.log").string();
    CHECK_EQ(std::system((cli + " show dataset --dataset " + d + " > " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " list subjects --dataset " + d + " >> " + log).c_str()), 0);
    CHECK_EQ(std::system((cli + " list recordings --dataset " + d + " --subject S1 >> " + log).c_str()), 0);
    CHECK(bytes_of(log).find("S1") != std::string::npos);
    CHECK(tree(folder) == before);
    CHECK_EQ(version_of(file), 50);
  }
#endif

  // below 50 is still refused to a reader
  {
    Database d = Database::open_plain(file);   // (no version check: put it below 50 by hand)
    nxr::io::testing::Probe(d).tamper("PRAGMA user_version = 49");
    d.close();
    OpenOptions o;
    o.lock = false;
    CHECK_EQ(code_of([&] { open_writer(folder, o); }), std::string(code::schema_version));
  }

  // only a writable open (the writer lock) migrates
  {
    Database d = Database::open_plain(file);
    nxr::io::testing::Probe(d).tamper("PRAGMA user_version = 50");
    d.close();
    CHECK_EQ(version_of(file), 50);
    DatasetWriter ds = open_writer(folder);
    CHECK(ds.locked());
    ds.close();
    CHECK_EQ(version_of(file), 54);
  }

  // a subject package made before the table `origin`: a reader reads it, and leaves it byte for byte
  {
    const fs::path pkg = root / "A.nxr.zarr";
    {
      CreatePackageOptions opt;
      opt.home_dataset = "home";
      DatasetWriter w = create_package(pkg, "A", opt);
      w.close();
    }
    {
      Database d = Database::open(pkg / kPackageDatabaseFile);
      nxr::io::testing::Probe(d).tamper("DROP TABLE origin");
      nxr::io::testing::Probe(d).tamper("DROP TABLE anchor_row");
      d.close();
    }
    const auto pbefore = tree(pkg);
    OpenOptions o;
    o.lock = false;
    {
      DatasetWriter r = open_package_writer(pkg, o);
      const nxr::io::testing::Probe pr(r);
      CHECK(pr.read_only());
      CHECK_EQ(pr.count("subject", {{"name", Value(std::string("A"))}}), 1);   // read, though the package lacks `origin` (origin_of answers none there)
      r.close();
    }
    CHECK(tree(pkg) == pbefore);
    {   // a writer gets the table back
      DatasetWriter w = open_package_writer(pkg);
      CHECK_EQ(nxr::io::testing::Probe(w).count("sqlite_master", {{"name", Value(std::string("origin"))}}), 1);
      w.close();
    }
  }
  // A DATABASE IN WAL MODE WITH A LIVE WRITER (a subject's small_files.sqlite another connection holds, its last commits still in the -wal): every read leaves the database file
  // and its -wal byte for byte and mtime for mtime as they were. Only the -shm (SQLite's shared-memory wal-index, which WAL readers write too) may change: it holds no data.
  {
    fs::path small;
    for (const auto& e : fs::recursive_directory_iterator(folder))
      if (e.is_regular_file() && e.path().filename() == "small_files.sqlite") small = e.path();
    CHECK(!small.empty());
    if (!small.empty()) {
      Database live = Database::open_plain(small);   // the other connection: WAL, no automatic checkpoint, a commit left in the -wal
      const nxr::io::testing::Probe lp(live);
      lp.tamper("PRAGMA journal_mode = WAL");
      lp.tamper("PRAGMA wal_autocheckpoint = 0");
      lp.tamper("CREATE TABLE IF NOT EXISTS wal_probe (a INTEGER)");
      lp.tamper("INSERT INTO wal_probe VALUES (1)");
      const fs::path wal = small.string() + "-wal";
      CHECK(fs::is_regular_file(wal) && fs::file_size(wal) > 0);
      const Stamp db_before = stamp(small), wal_before = stamp(wal);
      {   // the face's readers and io's (lock = false), reading the subject's small keys
        CHECK_EQ(face::subjects(folder)["subjects"].size(), std::size_t(1));
        CHECK(face::results(folder, "S1", "recordings").size() >= 1);
        OpenOptions o;
        o.lock = false;
        DatasetWriter ds = open_writer(folder, o);
        SubjectWriter sub = SubjectWriter::open(ds, "S1");
        task::Want want;
        want.cls = task::Cls::manifold;
        CHECK(!task::Subject::of(sub).find_all(want).empty());
        ds.close();
      }
#ifdef NXR_TEST_CLI
      {   // the command line's reads: a subject, a recording's time range, a surface region past the end (what the Studio's check did)
        const std::string cli = NXR_TEST_CLI, d = folder.string(), log = (root / "wal.log").string();
        CHECK_EQ(std::system((cli + " read subject --dataset " + d + " --subject S1 > " + log + " 2>&1").c_str()), 0);
        CHECK_EQ(std::system((cli + " read field --dataset " + d + " --subject S1 --name rest --from 0 --to 1 >> " + log + " 2>&1").c_str()), 0);
        std::system((cli + " read manifold --dataset " + d + " --subject S1 --name sensors --region 4294967295:4294967296 >> " + log + " 2>&1").c_str());
        CHECK_EQ(std::system((cli + " list fields --dataset " + d + " --subject S1 >> " + log + " 2>&1").c_str()), 0);
      }
#endif
      CHECK(stamp(small) == db_before);   // the database file: not a byte, not its mtime
      CHECK(stamp(wal) == wal_before);    // the -wal: likewise
      live.close();
    }
  }
  return nxrtest::finish("test_readonly_readers");
}
