// What the Python and MATLAB faces call (nxr/pipelines/face.h), on a small synthetic dataset made in a temp directory: the pipelines listed with their parameters, the
// refusal of a run without a time range, `run psd` through the face, its stored result read back (a statistic, and the recording as a field), the lists, and, when the command is built,
// the SAME result as `nxr-neuro run psd` (the same options: a cache hit on the face's own result, and the same bytes under another name).
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

#include "check.h"
#include "nxr/io/errors.h"
#include "nxr/pipelines/face.h"

using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code(); }
  return "";
}
bool has(const face::Json& list, const char* key, const std::string& name) {
  for (const auto& e : list) if (e.value(key, "") == name) return true;
  return false;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_face_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const fs::path folder = face::synthetic_dataset(root, "demo", "S1", "rest", 3, 100.0, 40.0);

  // what `run` takes: psd and the shipped definitions, each parameter with its meaning
  const face::Json pl = face::pipelines();
  CHECK(has(pl, "name", "psd") && has(pl, "name", "global_bursts"));
  for (const auto& p : pl) {
    if (p["name"] == "psd") {
      CHECK_EQ(p["form"].get<std::string>(), std::string("definition"));      // psd is a shipped definition like the others
      CHECK(has(p["parameters"], "name", "segment_seconds") && has(p["parameters"], "name", "recording"));
      for (const auto& q : p["parameters"]) CHECK(!q["meaning"].get<std::string>().empty());
    }
    if (p["name"] == "global_bursts") CHECK_EQ(p["form"].get<std::string>(), std::string("definition"));
  }

  // the time range is required: none, half of one, both kinds, and an unknown parameter are refused before anything is read
  face::RunRequest rq;
  rq.pipeline = "psd";
  rq.parameters = {{"recording", "rest"}, {"segment_seconds", 2.0}};
  rq.preset = "brainstorm";
  CHECK_EQ(code_of([&] { face::run(folder, "S1", rq); }), std::string("invalid_argument"));
  face::RunRequest half = rq;
  half.from = 0.0;
  CHECK_EQ(code_of([&] { face::run(folder, "S1", half); }), std::string("invalid_argument"));
  face::RunRequest both = rq;
  both.from = 0.0; both.to = 10.0; both.all = true;
  CHECK_EQ(code_of([&] { face::run(folder, "S1", both); }), std::string("invalid_argument"));
  face::RunRequest bogus = rq;
  bogus.all = true;
  bogus.parameters["bogus"] = 1;
  CHECK_EQ(code_of([&] { face::run(folder, "S1", bogus); }), std::string("invalid_argument"));

  // run psd over the whole recording; a rerun is a hit
  rq.all = true;
  int lines = 0;
  const face::Json a = face::run(folder, "S1", rq, [&](const face::Json&) { ++lines; });
  CHECK_EQ(a["cache"].get<std::string>(), std::string("computed"));
  CHECK(lines > 0);
  CHECK_EQ(a["stored"].size(), static_cast<std::size_t>(1));
  CHECK_EQ(a["stored"][0]["name"].get<std::string>(), std::string("rest_psd_welch"));
  CHECK_EQ(face::run(folder, "S1", rq)["cache"].get<std::string>(), std::string("hit"));

  // read it back: values [3 channels x 101 frequencies] in float64, the statistic's record
  const face::Stored s = face::read(folder, "S1", "rest_psd_welch");
  CHECK_EQ(s.meta["class"].get<std::string>(), std::string("statistic"));
  CHECK_EQ(s.meta["measure"].get<std::string>(), std::string("power_density"));
  CHECK(s.arrays.size() == 1 && s.arrays[0].shape == (std::vector<std::int64_t>{3, 101}) && s.arrays[0].dtype == "float64");
  CHECK_EQ(s.arrays[0].bytes.size(), static_cast<std::size_t>(3 * 101 * 8));
  // the recording as a field: [3 x 4000] float32
  const face::Stored r = face::read(folder, "S1", "rest");
  CHECK_EQ(r.meta["class"].get<std::string>(), std::string("field"));
  CHECK(r.arrays.size() == 1 && r.arrays[0].shape == (std::vector<std::int64_t>{3, 4000}) && r.arrays[0].dtype == "float32");
  CHECK_EQ(code_of([&] { face::read(folder, "S1", "nothing_of_that_name"); }), std::string("no_such_row"));

  // the lists
  CHECK(has(face::datasets(root)["datasets"], "name", "demo"));
  CHECK(has(face::subjects(folder)["subjects"], "name", "S1"));
  CHECK(has(face::results(folder, "S1"), "name", "rest_psd_welch"));
  CHECK(has(face::results(folder, "S1", "recordings"), "name", "rest"));
  CHECK_EQ(code_of([&] { face::subjects(folder, {}, std::string("age > 3")); }), std::string("invalid_argument"));

#ifdef NXR_TEST_CLI
  {   // the command line with the same options: the face's result is found (a cache hit), and under another name the same bytes
    const std::string cli = NXR_TEST_CLI, d = folder.string(), log = (root / "cli.log").string();
    const std::string same = cli + " run psd --dataset " + d + " --subject S1 --recording rest --set brainstorm --segment-seconds 2 --all";
    CHECK_EQ(std::system((same + " > " + log).c_str()), 0);
    std::ifstream in(log);
    std::string line, all;
    while (std::getline(in, line)) all += line + "\n";
    CHECK(all.find("\"hit\": true") != std::string::npos || all.find("\"hit\":true") != std::string::npos);   // the run report of a cache hit
    CHECK_EQ(std::system((same + " --name cli_psd >> " + log).c_str()), 0);
    const face::Stored c = face::read(folder, "S1", "cli_psd");
    CHECK(c.arrays.size() == 1 && c.arrays[0].bytes == s.arrays[0].bytes);
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_face");
}
