// `list datasets` (the datastore's folders) and the writer lock within one process. Every dataset folder of either form is listed, a catalogue dataset (`create dataset --dataset`) as
// well as a one-database one, each with its form; an empty folder is an empty list and only a folder that does not exist is `not_found`. A second writer of the same dataset in the same
// process is refused while the first holds the lock, and the borrowed-lock handshake (NXR_WRITER_LOCK_HELD) still lends it. Synthetic data only.
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <functional>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/writer_lock.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception& e) { return std::string("other: ") + e.what(); }
  return "";
}
const Json* named(const Json& rows, const std::string& name) {
  for (const auto& r : rows) if (r["name"] == name) return &r;
  return nullptr;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_list_datasets_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path store = root / "store";
  fs::create_directories(store);

  // ---- G5: an empty folder is an empty list; a folder that does not exist is not_found
  {
    const Json r = task::list_datasets({store.string()}).result;
    CHECK(r["datasets"].is_array() && r["datasets"].empty());
    CHECK(r["datastore"].is_null());
    CHECK_EQ(code_of([&] { task::list_datasets({(root / "missing").string()}); }), std::string(code::not_found));
  }

  // ---- G4: both forms, each with its form; a folder that holds no dataset is not one
  {
    task::CreateDatasetRequest one;
    one.datastore = store.string();
    one.name = "one";
    one.created_by = "test";
    task::create_dataset(one);
    task::CreateSubjectRequest s1;
    s1.dataset = (store / "one").string();
    s1.name = "S1";
    s1.created_by = "test";
    task::create_subject(s1);
    task::CreateDatasetRequest cat;
    cat.dataset = (store / "cat").string();
    cat.created_by = "test";
    task::create_dataset(cat);
    for (const char* n : {"A", "B"}) {
      task::CreateSubjectRequest s;
      s.dataset = (store / "cat").string();
      s.name = n;
      s.created_by = "test";
      task::create_subject(s);
    }
    fs::create_directories(store / "notes");
    const Json r = task::list_datasets({store.string()}).result;
    CHECK_EQ(r["datasets"].size(), std::size_t(2));
    const Json* a = named(r["datasets"], "one");
    const Json* b = named(r["datasets"], "cat");
    CHECK(a && (*a)["form"] == "one-database" && (*a)["path"] == "one" && (*a)["subjects"] == 1 && (*a)["database"] == "one/dataset.sqlite" && (*a)["id"].is_string());
    CHECK(b && (*b)["form"] == "catalogue" && (*b)["path"] == "cat" && (*b)["subjects"] == 2 && (*b)["database"] == "cat/catalogue.sqlite" && (*b)["id"].is_string());
    CHECK(r["datastore"].is_string());   // the datastore's id, from catalog.json, which the one-database form keeps
    CHECK(named(r["datasets"], "notes") == nullptr);
    // a catalogue dataset alone in its folder is listed too (no catalog.json)
    const fs::path alone = root / "alone";
    fs::create_directories(alone);
    task::CreateDatasetRequest c2;
    c2.dataset = (alone / "only").string();
    c2.created_by = "test";
    task::create_dataset(c2);
    const Json r2 = task::list_datasets({alone.string()}).result;
    CHECK(r2["datasets"].size() == 1 && r2["datasets"][0]["form"] == "catalogue" && r2["datasets"][0]["subjects"] == 0);
  }

  // ---- G11: one writer a dataset within a process too; the handshake still lends the lock
  {
    const fs::path d = store / "one";
    WriterLock first = acquire_writer_lock(d, "test");
    CHECK(static_cast<bool>(first));
    CHECK_EQ(code_of([&] { acquire_writer_lock(d, "test"); }), std::string(code::writer_lock_held));
    ::setenv(kWriterLockEnv, (first.file().string() + ":" + std::to_string(::getpid())).c_str(), 1);
    {
      WriterLock lent = acquire_writer_lock(d, "test");
      CHECK(lent.borrowed());
    }   // a borrowed lock is not removed
    ::unsetenv(kWriterLockEnv);
    CHECK(read_writer_lock(d).has_value());
    CHECK_EQ(code_of([&] { acquire_writer_lock(d, "test"); }), std::string(code::writer_lock_held));
    first.release();
    WriterLock again = acquire_writer_lock(d, "test");
    CHECK(static_cast<bool>(again) && !again.borrowed());
  }
  return nxrtest::finish("test_list_datasets");
}
