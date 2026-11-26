// What the Python and MATLAB faces call (nxr/pipelines/face.h), on a small synthetic dataset made in a temp directory: the pipelines listed with their parameters, the
// refusal of a run without a time range, `run psd` through the face, its stored result read back (a statistic, and the recording as a field), the lists, and, when the command is built,
// the SAME result as `nxr-neuro run psd` (the same options: a cache hit on the face's own result, and the same bytes under another name). `import subject` through the face on a
// synthetic Brainstorm protocol (three EEG channels): the import stands, its EEG sensor mesh is refused (fewer than four channels) and listed in `skipped`; with
// sensor_mesh false nothing is run and the result says so.
#include <map>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

#include "bst_protocol.h"
#include "check.h"
#include "nxr/io/task/commands.h"
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

  // read it back: values [3 channels x 101 frequencies] in the stored precision (float32, as the recording), the statistic's record
  const face::Stored s = face::read(folder, "S1", "rest_psd_welch");
  CHECK_EQ(s.meta["class"].get<std::string>(), std::string("statistic"));
  CHECK_EQ(s.meta["measure"].get<std::string>(), std::string("power_density"));
  CHECK(s.arrays.size() == 1 && s.arrays[0].shape == (std::vector<std::int64_t>{3, 101}) && s.arrays[0].dtype == "float32");
  CHECK_EQ(s.arrays[0].bytes.size(), static_cast<std::size_t>(3 * 101 * 4));
  // the recording as a field: [3 x 4000] float32
  const face::Stored r = face::read(folder, "S1", "rest");
  CHECK_EQ(r.meta["class"].get<std::string>(), std::string("field"));
  CHECK(r.arrays.size() == 1 && r.arrays[0].shape == (std::vector<std::int64_t>{3, 4000}) && r.arrays[0].dtype == "float32");
  CHECK_EQ(code_of([&] { face::read(folder, "S1", "nothing_of_that_name"); }), std::string("no_such_row"));
  CHECK_EQ(r.meta["rate"].get<double>(), 100.0);
  CHECK_EQ(r.meta["channels"].get<std::string>(), std::string("sensors"));
  CHECK_EQ(r.meta["n_samples"].get<std::int64_t>(), static_cast<std::int64_t>(4000));
  {   // a span and rows, through the chunk service (stored in chunks of 700 samples: the span crosses two chunk edges): the same values as the whole read's
    face::ReadRequest rq2;
    rq2.start = 1000;
    rq2.stop = 2500;
    rq2.rows = {2, 0};
    const face::Stored p = face::read(folder, "S1", "rest", rq2);
    CHECK(p.arrays.size() == 1 && p.arrays[0].shape == (std::vector<std::int64_t>{2, 1500}) && p.arrays[0].dtype == "float32");
    const float* whole = reinterpret_cast<const float*>(r.arrays[0].bytes.data());
    const float* part = reinterpret_cast<const float*>(p.arrays[0].bytes.data());
    bool same = true;
    for (int i = 0; i < 2; ++i)
      for (int t = 0; t < 1500; ++t) same = same && part[i * 1500 + t] == whole[(i == 0 ? 2 : 0) * 4000 + 1000 + t];
    CHECK(same);
    CHECK_EQ(p.meta["span"]["start_sample"].get<std::int64_t>(), static_cast<std::int64_t>(1000));
    face::ReadRequest shape;
    shape.reading = "shape";
    const face::Stored d = face::read(folder, "S1", "rest", shape);
    CHECK(d.arrays.empty() && d.meta["n_channels"].get<std::int64_t>() == 3 && d.meta["origin"].get<double>() == 0.0);
    face::ReadRequest bad = rq2;
    bad.stop = 5000;
    CHECK_EQ(code_of([&] { face::read(folder, "S1", "rest", bad); }), std::string("invalid_argument"));
    bad = rq2;
    bad.rows = {3};
    CHECK_EQ(code_of([&] { face::read(folder, "S1", "rest", bad); }), std::string("invalid_argument"));
  }
  {   // the channel manifold: positions [3 x 3], the names; a labelling family: codes and the dictionary
    const face::Stored m = face::read(folder, "S1", "sensors");
    CHECK_EQ(m.meta["class"].get<std::string>(), std::string("manifold"));
    CHECK(m.arrays.size() == 1 && m.arrays[0].name == "positions" && m.arrays[0].shape == (std::vector<std::int64_t>{3, 3}));
    CHECK(m.meta["names"] == face::Json::array({"C0", "C1", "C2"}));
    face::ReadRequest fam;
    fam.family = "quality";
    const face::Stored q = face::read(folder, "S1", "sensors", fam);
    CHECK(q.arrays.size() == 1 && q.arrays[0].name == "codes" && q.arrays[0].dtype == "int64" && q.arrays[0].shape == (std::vector<std::int64_t>{3}));
    std::map<std::int64_t, std::string> dict;
    for (const auto& e : q.meta["dictionary"]) dict[e["code"].get<std::int64_t>()] = e["name"].get<std::string>();
    const std::int64_t* codes = reinterpret_cast<const std::int64_t*>(q.arrays[0].bytes.data());
    CHECK(dict[codes[0]] == "good" && dict[codes[1]] == "good" && dict[codes[2]] == "bad");
    fam.family = "nothing";
    CHECK_EQ(code_of([&] { face::read(folder, "S1", "sensors", fam); }), std::string("no_such_row"));
  }
  {   // a selection of spans: start, stop (s), code, the dictionary
    const face::Stored b = face::read(folder, "S1", "rest_bad_segments");
    CHECK(b.arrays.size() == 3 && b.arrays[0].name == "start" && b.arrays[1].name == "stop" && b.arrays[2].name == "code");
    const double* start = reinterpret_cast<const double*>(b.arrays[0].bytes.data());
    const double* stop = reinterpret_cast<const double*>(b.arrays[1].bytes.data());
    CHECK(b.arrays[0].shape == (std::vector<std::int64_t>{1}) && start[0] == 12.0 && stop[0] == 15.0);
    CHECK(b.meta["dictionary"].size() == 1 && b.meta["dictionary"][0]["name"] == "BAD");
  }

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
  // import subject: the import, then the sensor meshes; a refused mesh is reported, never silent, and the import stands
  {
    const fs::path protocol = bst_protocol::protocol(root / "protocol");
    auto import_into = [&](const std::string& name, bool sensor_mesh) {
      nxr::io::task::CreateDatasetRequest create;
      create.datastore = (root / "imports").string();
      create.name = name;
      create.created_by = "test";
      nxr::io::task::create_dataset(create);
      face::ImportSubjectRequest request;
      request.protocol = protocol.string();
      request.subject = "S01";
      request.dataset = (root / "imports" / name).string();
      request.sensor_mesh = sensor_mesh;
      return face::import_subject(request);
    };
    const face::Json meshed = import_into("with_mesh", true);
    CHECK(meshed.contains("sensor_meshes") && meshed["sensor_meshes"].empty());
    bool reported = false;
    for (const face::Json& skipped : meshed.value("skipped", face::Json::array()))
      if (skipped.value("what", std::string()).find("sensor mesh") != std::string::npos && skipped.value("what", std::string()).find("EEG") != std::string::npos) {
        std::printf("  import subject, EEG sensor mesh: %s\n", skipped.value("reason", std::string()).substr(0, 140).c_str());
        reported = skipped.value("reason", std::string()).find("fewer than the four") != std::string::npos;
      }
    CHECK(reported);
    const face::Json unmeshed = import_into("without_mesh", false);
    CHECK(unmeshed.value("sensor_meshes_skipped", std::string()) == "--no-sensor-mesh");
    CHECK(!unmeshed.contains("sensor_meshes"));
  }
  fs::remove_all(root);
  return nxrtest::finish("test_face");
}
