// `import condition --replace-recording` (nxr::bst::replace_raw_timeseries): the samples of a raw recording already imported are read again from its BST-BIN file and rewritten in
// place, the recording's row, id, events, channel flags and every other node kept, the old samples moved aside, the replacement recorded in its provenance; the stored results
// computed from it are listed. Refused with nothing written: other channels, another shape, no note, an acquisition not imported. Synthetic: a protocol of one condition with a raw
// link of four EEG channels whose length is not a whole number of epochs (Brainstorm's layout, issue #276), written with matio.
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

extern "C" {
#include "matio.h"
}
#include "check.h"
#include "nxr/io.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/task/commands.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string thrown(const std::function<void()>& f) {
  try { f(); } catch (const task::Refused& e) { return std::string("refused: ") + e.what(); } catch (const Error& e) { return e.code() + std::string(": ") + e.what(); }
  return "";
}
bool has(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }
matvar_t* numeric(const char* name, const std::vector<double>& row_major, size_t rows, size_t cols) {
  std::vector<double> col(row_major.size());
  for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < cols; ++c) col[c * rows + r] = row_major[r * cols + c];
  size_t d[2] = {rows, cols};
  return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d, col.data(), 0);
}
matvar_t* empty(const char* name) {
  size_t d[2] = {0, 0};
  return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, d, nullptr, 0);
}
matvar_t* chars(const char* name, const std::string& s) {
  size_t d[2] = {1, s.size()};
  return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, d, const_cast<char*>(s.data()), 0);
}
void write(mat_t* m, matvar_t* v) {
  Mat_VarWrite(m, v, MAT_COMPRESSION_NONE);
  Mat_VarFree(v);
}
matvar_t* record(const char* name, const std::vector<std::pair<const char*, matvar_t*>>& fields) {
  std::vector<const char*> names;
  for (const auto& f : fields) names.push_back(f.first);
  size_t one[2] = {1, 1};
  matvar_t* s = Mat_VarCreateStruct(name, 2, one, names.data(), static_cast<unsigned>(names.size()));
  for (const auto& f : fields) Mat_VarSetStructFieldByName(s, f.first, 0, f.second);
  return s;
}
// a Brainstorm cortex: two octahedra (12 vertices, metres)
void write_tess(const fs::path& p) {
  std::vector<double> v, faces;
  const std::vector<int> f = {0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5};
  for (const double cx : {-0.03, 0.03}) {
    const double r = 0.01;
    for (const double x : {cx + r, 0.0, 0.0, cx - r, 0.0, 0.0, cx, r, 0.0, cx, -r, 0.0, cx, 0.0, r, cx, 0.0, -r}) v.push_back(x);
  }
  for (int h = 0; h < 2; ++h) for (const int i : f) faces.push_back(i + 1 + 6 * h);
  fs::create_directories(p.parent_path());
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, MAT_FT_MAT5);
  write(m, numeric("Vertices", v, 12, 3));
  write(m, numeric("Faces", faces, 16, 3));
  write(m, chars("Comment", "cortex_synthetic"));
  Mat_Close(m);
}
// a subject image: a 4 x 4 x 4 uint8 Cube, its voxel size and its registration frames
void write_image(const fs::path& p) {
  std::vector<std::uint8_t> cube(64);   // an anatomical image is uint8, as Brainstorm stores one
  for (std::size_t i = 0; i < cube.size(); ++i) cube[i] = static_cast<std::uint8_t>(i % 7);
  size_t d[3] = {4, 4, 4};
  mat_t* m = Mat_CreateVer(p.string().c_str(), nullptr, MAT_FT_MAT5);
  write(m, Mat_VarCreate("Cube", MAT_C_UINT8, MAT_T_UINT8, 3, d, cube.data(), 0));
  write(m, numeric("Voxsize", {1, 1, 1}, 1, 3));
  write(m, chars("Comment", "MRI"));
  {   // the registration frames an import requires, as Brainstorm writes them: SCS with its landmarks (voxel units), NCS (identity axes, shifted so that the nasion is at +x, the left ear at +y, the right ear at -y, all three at z = 0)
    const char* fn[] = {"R", "T", "NAS", "LPA", "RPA"};
    size_t one[2] = {1, 1};
    matvar_t* sc = Mat_VarCreateStruct("SCS", 2, one, fn, 5);
    Mat_VarSetStructFieldByName(sc, "R", 0, numeric(nullptr, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3));
    Mat_VarSetStructFieldByName(sc, "T", 0, numeric(nullptr, {0, -50, -2}, 3, 1));
    Mat_VarSetStructFieldByName(sc, "NAS", 0, numeric(nullptr, {50, 25, 2}, 1, 3));
    Mat_VarSetStructFieldByName(sc, "LPA", 0, numeric(nullptr, {2, 99, 2}, 1, 3));
    Mat_VarSetStructFieldByName(sc, "RPA", 0, numeric(nullptr, {2, 1, 2}, 1, 3));
    write(m, sc);
    const char* nf[] = {"R", "T"};
    matvar_t* nc = Mat_VarCreateStruct("NCS", 2, one, nf, 2);
    Mat_VarSetStructFieldByName(nc, "R", 0, numeric(nullptr, {1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3));
    Mat_VarSetStructFieldByName(nc, "T", 0, numeric(nullptr, {-100, -120, -80}, 3, 1));
    write(m, nc);
  }
  Mat_Close(m);
}
constexpr std::int64_t kChannels = 4, kEpoch = 64, kHeader = 16;
// the channel file: four EEG channels named `names`
void write_channels(const fs::path& dir, const std::vector<std::string>& names) {
  mat_t* m = Mat_CreateVer((dir / "channel.mat").string().c_str(), nullptr, MAT_FT_MAT5);
  const char* fields[] = {"Name", "Type", "Loc", "Orient", "Weight", "Comment"};
  size_t d[2] = {1, names.size()};
  matvar_t* ch = Mat_VarCreateStruct("Channel", 2, d, fields, 6);
  for (size_t i = 0; i < names.size(); ++i) {
    Mat_VarSetStructFieldByName(ch, "Name", i, chars(nullptr, names[i]));
    Mat_VarSetStructFieldByName(ch, "Type", i, chars(nullptr, "EEG"));
    Mat_VarSetStructFieldByName(ch, "Loc", i, numeric(nullptr, {0.01 * double(i), 0.02, 0.08}, 3, 1));
    Mat_VarSetStructFieldByName(ch, "Orient", i, empty(nullptr));
    Mat_VarSetStructFieldByName(ch, "Weight", i, numeric(nullptr, {1.0}, 1, 1));
    Mat_VarSetStructFieldByName(ch, "Comment", i, chars(nullptr, ""));
  }
  write(m, ch);
  write(m, chars("Comment", "EEG"));
  Mat_Close(m);
}
float sample(std::int64_t c, std::int64_t t, float scale) { return scale * static_cast<float>((c + 1) * 1000 + t); }
// the raw link (data_0raw_rest.mat) and its binary rest.bst, laid out as Brainstorm lays it out: every channel's block kEpoch apart in every epoch, the last channel
// ending at its declared samples; `scale` sets the values
void write_raw(const fs::path& dir, std::int64_t nsamples, float scale) {
  const fs::path bst = dir / "rest.bst";
  {
    std::vector<float> data;
    for (std::int64_t e = 0; e * kEpoch < nsamples; ++e) {
      const std::int64_t len = std::min(kEpoch, nsamples - e * kEpoch);
      for (std::int64_t c = 0; c < kChannels; ++c) {
        for (std::int64_t t = 0; t < len; ++t) data.push_back(sample(c, e * kEpoch + t, scale));
        if (c + 1 < kChannels) for (std::int64_t t = len; t < kEpoch; ++t) data.push_back(0.0f);
      }
    }
    std::ofstream f(bst, std::ios::binary);
    f << std::string(static_cast<std::size_t>(kHeader), 'h');
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 4));
  }
  mat_t* m = Mat_CreateVer((dir / "data_0raw_rest.mat").string().c_str(), nullptr, MAT_FT_MAT5);
  matvar_t* header = record(nullptr, {{"nchannels", numeric(nullptr, {double(kChannels)}, 1, 1)}, {"epochsize", numeric(nullptr, {double(kEpoch)}, 1, 1)},
                                      {"hdrsize", numeric(nullptr, {double(kHeader)}, 1, 1)}, {"nsamples", numeric(nullptr, {double(nsamples)}, 1, 1)}});
  matvar_t* prop = record(nullptr, {{"sfreq", numeric(nullptr, {100.0}, 1, 1)}, {"times", numeric(nullptr, {0.0, double(nsamples - 1) / 100.0}, 1, 2)}});
  matvar_t* events = record(nullptr, {{"label", chars(nullptr, "blink")}, {"color", numeric(nullptr, {1, 0, 0}, 1, 3)}, {"epochs", numeric(nullptr, {1, 1}, 1, 2)},
                                      {"times", numeric(nullptr, {0.5, 1.2}, 1, 2)}, {"channels", empty(nullptr)}});
  write(m, record("F", {{"filename", chars(nullptr, bst.string())}, {"format", chars(nullptr, "BST-BIN")}, {"device", chars(nullptr, "EEG")}, {"header", header}, {"prop", prop},
                        {"events", events}}));
  write(m, numeric("ChannelFlag", {1, 1, -1, 1}, 4, 1));
  write(m, chars("Comment", "Link to raw file"));
  Mat_Close(m);
}
struct Stored { std::string id, path; std::vector<float> samples; std::vector<float> envelope; Json producer; std::int64_t events = 0; };
Stored stored(const fs::path& package) {
  DatasetWriter w = open_package_writer(package);
  SubjectWriter sub = SubjectWriter::open(w, "S01");
  Stored s;
  const auto f = sub.db().one("SELECT * FROM field WHERE name = 'rest' AND kind = 'recording'");
  if (!f) return s;
  s.id = text(*f, "id");
  s.path = text(*f, "path");
  s.producer = Json::parse(opt_text(*f, "producer_json").value_or("{}"));
  auto floats = [&](const std::string& at) {
    const auto bytes = ArrayReader::open(sub.store(), at).read();
    std::vector<float> v(bytes.size() / 4);
    std::memcpy(v.data(), bytes.data(), bytes.size());
    return v;
  };
  s.samples = floats(s.path);
  s.envelope = floats(s.path + "_envelope");
  s.events = integer(*sub.db().one("SELECT count(*) AS n FROM selection WHERE of_field_id = ?", {s.id}), "n");
  return s;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_import_replace_recording_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path P = root / "protocol";
  const fs::path raw = P / "data" / "S01" / "@rawrest";
  write_tess(P / "anat" / "@default_subject" / "tess_cortex_pial_low.mat");
  write_image(P / "anat" / "@default_subject" / "subjectimage_T1.mat");
  write_tess(P / "anat" / "S01" / "tess_cortex_pial_low.mat");
  fs::create_directories(raw);
  fs::create_directories(P / "data" / "S01" / "rest");
  const std::vector<std::string> names = {"Fz", "Cz", "Pz", "Oz"};
  write_channels(raw, names);
  const std::int64_t n = 150;   // 2 epochs of 64 and 22 samples: the last epoch is partial
  write_raw(raw, n, 1.0f);
  task::CreateDatasetRequest cd;
  cd.dataset = (root / "study").string();
  cd.created_by = "test";
  task::create_dataset(cd);
  task::ImportConditionRequest ic;
  ic.protocol = P.string();
  ic.subject = "S01";
  ic.condition = "rest";
  ic.dataset = (root / "study").string();
  ic.created_by = "test";
  task::import_condition(ic);
  const fs::path pkg = root / "study" / "S01.nxr.zarr";
  const Stored first = stored(pkg);
  CHECK(!first.id.empty());
  CHECK_EQ(first.samples.size(), std::size_t(kChannels * n));
  for (std::int64_t c = 0; c < kChannels; ++c)   // read at Brainstorm's layout, the last epoch's channels included
    CHECK(first.samples[c * n + n - 1] == sample(c, n - 1, 1.0f) && first.samples[c * n + 2 * kEpoch] == sample(c, 2 * kEpoch, 1.0f));
  {   // a stored result computed from the recording: its provenance names the recording (as a run's cache key does)
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "S01");
    const auto ch = sub.db().one("SELECT id FROM manifold WHERE type = 'points' AND n_vertices = 4");
    CHECK(ch.has_value());
    sub.compose([&] {
      std::vector<float> values = {1, 2, 3, 4};
      FieldSpec f;
      f.name = "rest_power";
      f.path = "sensor/rest_power";
      f.kind = "map";
      f.manifold_id = text(*ch, "id");
      f.data = view(values.data(), {4});
      f.cols.json("producer_json", Json{{"cache", Json{{"input", Json{{"inputs", Json{{"recording", first.id}}}}}}}});
      sub.field(std::move(f));
    });
  }

  // the file gives other samples (as the fixed reader gives a file the old one misread): the samples are replaced in place
  write_raw(raw, n, 2.0f);
  task::ImportConditionRequest rr = ic;
  rr.replace_recording = true;
  CHECK(has(thrown([&] { task::import_condition(rr); }), "--note"));   // why is required
  rr.note = "issue #276: the last epoch was read at the wrong stride";
  const Json res = task::import_condition(rr).result;
  const Json& rep = res["replaced"];
  CHECK_EQ(rep["name"].get<std::string>(), std::string("rest"));
  CHECK_EQ(rep["id"].get<std::string>(), first.id);
  CHECK_EQ(rep["layout"].get<std::string>(), std::string("epoch stride"));
  CHECK(rep["changed"].get<bool>());
  CHECK(rep["derived"].size() == 1 && rep["derived"][0]["name"] == "rest_power");
  CHECK(rep["from"]["samples"]["sha256"] != rep["to"]["samples"]["sha256"]);
  const Stored second = stored(pkg);
  CHECK_EQ(second.id, first.id);                   // the same row
  CHECK_EQ(second.path, first.path);
  CHECK_EQ(second.events, first.events);           // its events, flags, chunk and envelope tilings kept
  CHECK(second.samples.size() == first.samples.size());
  bool all = true;
  for (std::int64_t c = 0; c < kChannels; ++c) for (std::int64_t t = 0; t < n; ++t) all = all && second.samples[c * n + t] == sample(c, t, 2.0f);
  CHECK(all);
  CHECK(second.envelope.size() == first.envelope.size() && second.envelope[1] == 2.0f * first.envelope[1]);   // the envelope follows the samples
  {   // the provenance: what replaced what, when, why
    const Json& e = second.producer["replaced_samples"];
    CHECK(e.is_array() && e.size() == 1);
    CHECK_EQ(e[0]["note"].get<std::string>(), *rr.note);
    CHECK_EQ(e[0]["command"].get<std::string>(), std::string("import condition --replace-recording"));
    CHECK(e[0]["from"]["samples"] == rep["from"]["samples"] && e[0]["to"]["samples"] == rep["to"]["samples"]);
    CHECK(e[0].contains("utc") && e[0]["source"]["layout"] == "epoch stride" && e[0]["source"]["declared_samples"] == n);
    CHECK_EQ(second.producer["format"].get<std::string>(), std::string("BST-BIN"));   // the import's record stays
    // the old samples kept aside, never deleted
    DatasetWriter w = open_package_writer(pkg);
    SubjectWriter sub = SubjectWriter::open(w, "S01");
    const std::string aside = e[0]["from"]["moved_aside"].get<std::string>();
    const auto bytes = ArrayReader::open(sub.store(), aside).read();
    CHECK(bytes.size() == first.samples.size() * 4 && std::memcmp(bytes.data(), first.samples.data(), bytes.size()) == 0);
    CHECK(!sub.store().exists_prefix(first.path + "__replacing"));
  }
  CHECK(SubjectPackage::open(pkg).verify().ok);   // every digest and the content hash agree with the bytes

  // refused, nothing written: another length, other channels, an acquisition never imported
  write_raw(raw, n - 10, 3.0f);
  CHECK(has(thrown([&] { task::import_condition(rr); }), "only samples of the same shape are replaced"));
  write_raw(raw, n, 3.0f);
  write_channels(raw, {"Fz", "Cz", "Pz", "O1"});
  CHECK(has(thrown([&] { task::import_condition(rr); }), "differ from those the recording was imported with"));
  write_channels(raw, names);
  {   // a condition of the protocol never imported into the subject
    const fs::path raw2 = P / "data" / "S01" / "@rawother";
    fs::create_directories(raw2);
    write_channels(raw2, names);
    write_raw(raw2, n, 3.0f);
    task::ImportConditionRequest other = rr;
    other.condition = "other";
    CHECK(has(thrown([&] { task::import_condition(other); }), "nothing to replace"));
  }
  const Stored third = stored(pkg);
  CHECK(third.samples == second.samples && third.producer["replaced_samples"].size() == 1);
  // --note alone is refused
  task::ImportConditionRequest note_only = ic;
  note_only.note = "x";
  CHECK(has(thrown([&] { task::import_condition(note_only); }), "--note is kept only with --replace-recording"));
  return nxrtest::finish("test_import_replace_recording");
}
