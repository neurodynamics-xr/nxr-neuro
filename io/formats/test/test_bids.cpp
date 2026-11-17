// BIDS against a synthetic dataset written by MNE-BIDS (test/golden/gen_bids.py): file names and entities, the folder tree, recordings with their
// merged sidecars and tables, the dataset description, participants, and the inheritance principle; and the refusals.
#include <cmath>
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/bids.h"
#include "nxr/formats/errors.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kTree = std::string(NXR_FORMATS_TEST_DIR) + "/data/bids";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

static json golden() { std::ifstream f(kGold + "bids_golden.json"); return json::parse(f); }

// MNE-BIDS names entities in full (`subject`); the specification's short keys and these long names are one table
static std::string long_name(const std::string& key) {
  for (const auto& d : bids::entity_definitions()) if (key == d.key) return d.name;
  return key;
}
static std::string short_key(const std::string& name) {
  for (const auto& d : bids::entity_definitions()) if (name == d.name) return d.key;
  return name;
}

static void check_table(const std::optional<bids::TableFile>& got, const json& want, const char* what) {
  if (want.is_null()) { CHECK(!got); return; }
  CHECK(got.has_value());
  if (!got) return;
  CHECK_EQ(got->path, want["path"].get<std::string>());
  CHECK_EQ(got->table.columns.size(), want["columns"].size());
  for (std::size_t c = 0; c < got->table.columns.size() && c < want["columns"].size(); ++c) CHECK_EQ(got->table.columns[c], want["columns"][c].get<std::string>());
  CHECK_EQ(got->table.rows.size(), want["rows"].size());
  bool ok = true;
  for (std::size_t r = 0; r < got->table.rows.size() && r < want["rows"].size(); ++r)
    for (std::size_t c = 0; c < got->table.rows[r].size() && c < want["rows"][r].size(); ++c) ok = ok && got->table.rows[r][c] == want["rows"][r][c].get<std::string>();
  if (!ok) std::printf("  table %s differs\n", what);
  CHECK(ok);
}

int main() {
  const json G = golden();

  // ---- file names: parsed against MNE-BIDS where it reads them, the specification elsewhere ----
  {
    std::size_t i = 0;
    for (const auto& c : G["names"]) {
      const std::string name = c["name"].get<std::string>();
      const bool want_valid = i < 12;   // the first twelve cases of the generator are valid names, the rest are refused
      ++i;
      bool valid = true;
      bids::Entities e;
      try { e = bids::parse_name(name); } catch (const nxr::io::Error& err) { valid = false; CHECK(err.code() == std::string(code::format)); }
      if (valid != want_valid) std::printf("  %s: parse %s\n", name.c_str(), valid ? "accepted" : "refused");
      CHECK_EQ(valid, want_valid);
      if (valid && !c["mne_bids"].is_null()) {
        CHECK_EQ(e.items.size(), c["mne_bids"].size());
        for (auto it = c["mne_bids"].begin(); it != c["mne_bids"].end(); ++it) {
          const auto v = e.get(short_key(it.key()));
          CHECK(v.has_value());
          if (v) CHECK_EQ(*v, it.value().get<std::string>());
        }
      }
    }
    // MNE-BIDS reads long names back to the one table
    CHECK_EQ(bids::entity_rank("sub"), 0);
    CHECK(bids::entity_rank("split") > bids::entity_rank("space"));
    CHECK_EQ(bids::entity_rank("notanentity"), -1);
    for (const auto& b : G["build"]) {
      bids::Entities e;
      for (auto it = b["entities"].begin(); it != b["entities"].end(); ++it) {
        if (it.key() == "suffix") e.suffix = it.value().get<std::string>();
        else if (it.key() == "extension") e.extension = it.value().get<std::string>();
        else e.items.emplace_back(short_key(it.key()), it.value().get<std::string>());
      }
      // shuffled input order: the name is built in the specification's order, as MNE-BIDS builds it
      std::reverse(e.items.begin(), e.items.end());
      CHECK_EQ(bids::build_name(e), b["name"].get<std::string>());
    }
    bids::Entities a = bids::parse_name("sub-01_task-rest_eeg.edf");
    CHECK(a.in_canonical_order());
    CHECK(!bids::parse_name("sub-01_acq-a_task-rest_eeg.edf").in_canonical_order());
    CHECK(!bids::parse_name("sub-01_foo-a_eeg.edf").in_canonical_order());
    CHECK_EQ(bids::parse_name("sub-01_T1w.nii.gz").extension, std::string(".nii.gz"));
    CHECK(bids::parse_name("sub-01_task-rest_eeg").extension.empty());
    CHECK(bids::parse_name("sub-01_ses-a_eeg").items.size() == 2);
  }

  // ---- the dataset ----
  bids::DirectorySource src(kTree);
  const bids::Dataset ds = bids::read_dataset(src);
  {
    const json& d = G["description"];
    CHECK_EQ(ds.description.name, d["Name"].get<std::string>());
    CHECK_EQ(ds.description.bids_version, d["BIDSVersion"].get<std::string>());
    CHECK(ds.description.license == d["License"].get<std::string>());
    CHECK(ds.description.dataset_type == d["DatasetType"].get<std::string>());
    CHECK(ds.description.dataset_doi == d["DatasetDOI"].get<std::string>());
    CHECK_EQ(ds.description.authors.size(), d["Authors"].size());
    CHECK_EQ(ds.description.authors[0], d["Authors"][0].get<std::string>());
    CHECK_EQ(ds.description.funding.size(), std::size_t(1));
    CHECK_EQ(ds.description.ethics_approvals.size(), std::size_t(1));
    CHECK(ds.description.raw == d);
    check_table(ds.participants, G["participants"], "participants");
    CHECK(ds.participants_sidecar.has_value());
    if (ds.participants_sidecar) CHECK(ds.participants_sidecar->content == G["participants_json"]);
    CHECK_EQ(ds.derivatives.size(), G["derivatives"].size());
    CHECK_EQ(ds.derivatives[0], G["derivatives"][0].get<std::string>());
    CHECK_EQ(ds.top_level_files.size(), G["top_level_files"].size());
    for (std::size_t i = 0; i < ds.top_level_files.size() && i < G["top_level_files"].size(); ++i) CHECK_EQ(ds.top_level_files[i], G["top_level_files"][i].get<std::string>());
    // subjects and sessions
    CHECK_EQ(ds.subjects.size(), G["subjects"].size());
    for (const bids::Subject& s : ds.subjects) {
      const json& want = G["subjects"][s.label];
      if (want.empty()) {
        CHECK_EQ(s.sessions.size(), std::size_t(1));
        CHECK(s.sessions[0].label.empty());
      } else {
        CHECK_EQ(s.sessions.size(), want.size());
        for (std::size_t i = 0; i < s.sessions.size() && i < want.size(); ++i) CHECK_EQ(s.sessions[i].label, want[i].get<std::string>());
      }
    }
    // the file that is not a BIDS name is reported, not dropped
    bool found = false;
    for (const auto& is : ds.issues) found = found || is.path == "sub-02/eeg/notes_misc.txt";
    CHECK(found);
    // scans
    for (const bids::Subject& s : ds.subjects)
      for (const bids::Session& ss : s.sessions) {
        if (!ss.scans) continue;
        const json& w = G["scans"][ss.scans->path];
        check_table(ss.scans, w, "scans");
      }
  }
  // ---- recordings ----
  std::vector<const bids::Recording*> recs;
  for (const auto& s : ds.subjects) for (const auto& ss : s.sessions) for (const auto& r : ss.recordings) recs.push_back(&r);
  CHECK_EQ(recs.size(), G["recordings"].size());
  for (const auto& w : G["recordings"]) {
    const bids::Recording* got = nullptr;
    for (const auto* r : recs) if (r->data_files[0] == w["data_files"][0].get<std::string>()) got = r;
    if (!got) { std::printf("  recording %s not found\n", w["data_files"][0].get<std::string>().c_str()); CHECK(false); continue; }
    std::printf("  case %s\n", got->data_files[0].c_str());
    CHECK_EQ(got->datatype, w["datatype"].get<std::string>());
    CHECK_EQ(got->entities.items.size(), w["entities"].size());
    for (auto it = w["entities"].begin(); it != w["entities"].end(); ++it) CHECK(got->entities.get(short_key(it.key())) == it.value().get<std::string>());
    CHECK_EQ(got->data_files.size(), w["data_files"].size());
    CHECK_EQ(got->companions.size(), w["companions"].size());
    for (std::size_t i = 0; i < got->companions.size() && i < w["companions"].size(); ++i) CHECK_EQ(got->companions[i], w["companions"][i].get<std::string>());
    CHECK_EQ(got->sidecar_files.size(), w["sidecar_files"].size());
    for (std::size_t i = 0; i < got->sidecar_files.size() && i < w["sidecar_files"].size(); ++i) CHECK_EQ(got->sidecar_files[i], w["sidecar_files"][i].get<std::string>());
    if (got->sidecar != w["sidecar"]) std::printf("  merged sidecar differs: %s\n  want %s\n", got->sidecar.dump().c_str(), w["sidecar"].dump().c_str());
    CHECK(got->sidecar == w["sidecar"]);
    check_table(got->channels, w["channels"], "channels");
    check_table(got->events, w["events"], "events");
    CHECK(got->channels.has_value() && got->channels->path == w["mne_bids_channels"].get<std::string>());   // the nearest channels file, as MNE-BIDS finds it
    CHECK_EQ(got->electrodes.size(), w["electrodes"].size());
    for (std::size_t i = 0; i < got->electrodes.size() && i < w["electrodes"].size(); ++i) check_table(got->electrodes[i], w["electrodes"][i], "electrodes");
    CHECK_EQ(got->coordsystems.size(), w["coordsystems"].size());
    for (std::size_t i = 0; i < got->coordsystems.size() && i < w["coordsystems"].size(); ++i) CHECK_EQ(got->coordsystems[i].path, w["coordsystems"][i].get<std::string>());
    CHECK(got->events_sidecar.has_value());
  }
  // a headshape file is attached to its recording; the odd file is not in any list of files
  for (const auto* r : recs)
    if (r->entities.get("sub") == std::string("02")) { CHECK_EQ(r->other_files.size(), std::size_t(1)); CHECK(r->companions.size() == 2); }
  // ---- the inheritance principle ----
  {
    const std::string t = "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-02_eeg.edf";
    const auto files = bids::applicable_files(src, t, "eeg", ".json");
    CHECK_EQ(files.size(), std::size_t(4));
    CHECK_EQ(files.front(), std::string("task-rest_eeg.json"));   // the root first
    CHECK_EQ(files.back(), std::string("sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-02_eeg.json"));
    CHECK(bids::applicable_files(src, "sub-01/ses-01/eeg/sub-01_ses-01_task-oddball_eeg.vhdr", "eeg", ".json").size() == 1);   // task-rest files do not apply to another task
    CHECK(bids::applicable_files(src, "sub-02/eeg/sub-02_task-rest_eeg.vhdr", "eeg", ".json").size() == 2);                  // the subject-01 file does not apply to subject 02
    CHECK(bids::applicable_files(src, "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-02_eeg.edf", "electrodes", ".tsv").empty());   // `space` is not in the recording: ignored only on request
    CHECK_EQ(bids::applicable_files(src, "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_run-02_eeg.edf", "electrodes", ".tsv", {"space"}).size(), std::size_t(1));
    const json m = bids::merged_sidecar(src, t, "eeg");
    CHECK_EQ(m["CustomKey"].get<int>(), 2);                       // the subject file overrides the root's
    CHECK_EQ(m["Manufacturer"].get<std::string>(), std::string("n/a"));   // the recording's own file overrides the root's
    CHECK(m["Nested"] == json::parse(R"({"a": 9})"));             // an object is replaced whole, not merged
  }
  // ---- a tree in memory: split recordings, CTF folders, options, derivatives, refusals ----
  {
    bids::MemoryTree t;
    t.add("dataset_description.json", R"({"Name":"mem","BIDSVersion":"1.9.0"})");
    t.add("sub-A/ses-1/meg/sub-A_ses-1_task-x_split-02_meg.fif", "");
    t.add("sub-A/ses-1/meg/sub-A_ses-1_task-x_split-01_meg.fif", "");
    t.add("sub-A/ses-1/meg/sub-A_ses-1_task-x_split-10_meg.fif", "");
    t.add("sub-A/ses-1/meg/sub-A_ses-1_task-x_channels.tsv", "name\ttype\nMEG1\tMEGMAG\n");
    t.add("sub-A/ses-1/meg/sub-A_ses-1_task-y_meg.ds/placeholder", "");
    t.add("sub-A/ses-1/meg/sub-A_ses-1_task-x_meg.json", R"({"SamplingFrequency":1000})");
    t.add("sub-A/ses-1/meg/sub-A_ses-1_coordsystem.json", R"({"MEGCoordinateSystem":"CTF"})");
    t.add("sub-B/anat/sub-B_T1w.nii.gz", "");
    t.add("sub-B/anat/sub-B_run-1_T1w.nii.gz", "");
    const bids::Dataset d = bids::read_dataset(t);
    CHECK_EQ(d.subjects.size(), std::size_t(2));
    const bids::Session& s = d.subjects[0].sessions[0];
    CHECK_EQ(s.recordings.size(), std::size_t(2));
    CHECK_EQ(s.recordings[0].data_files.size(), std::size_t(3));
    CHECK(s.recordings[0].data_files[0].find("split-01") != std::string::npos);
    CHECK(s.recordings[0].data_files[1].find("split-02") != std::string::npos);
    CHECK(s.recordings[0].data_files[2].find("split-10") != std::string::npos);   // numeric order: 10 after 02
    CHECK(!s.recordings[0].entities.has("split"));
    CHECK_EQ(s.recordings[0].sidecar["SamplingFrequency"].get<int>(), 1000);
    CHECK(s.recordings[0].channels.has_value());
    CHECK_EQ(s.recordings[1].data_files[0], std::string("sub-A/ses-1/meg/sub-A_ses-1_task-y_meg.ds"));   // a CTF folder is one data file
    CHECK(!s.recordings[1].channels.has_value());
    CHECK_EQ(s.recordings[0].coordsystems.size(), std::size_t(1));
    CHECK_EQ(d.subjects[1].sessions[0].files.size(), std::size_t(2));
    CHECK_EQ(d.subjects[1].sessions[0].files[0].datatype, std::string("anat"));
    bids::ReadOptions only;
    only.subjects = {"B"};
    CHECK_EQ(bids::read_dataset(t, only).subjects.size(), std::size_t(1));
  }
  // ---- refusals ----
  auto refusal = [&](const std::function<void()>& fn, const char* code_, const char* needle) {
    try {
      fn();
      std::printf("  expected a refusal containing '%s'\n", needle);
      CHECK(false);
    } catch (const nxr::io::Error& e) {
      CHECK(e.code() == code_);
      const bool found = std::string(e.what()).find(needle) != std::string::npos;
      if (!found) std::printf("  message '%s' lacks '%s'\n", e.what(), needle);
      CHECK(found);
    }
  };
  refusal([&] { bids::MemoryTree t; t.add("sub-01/eeg/sub-01_eeg.edf", ""); bids::read_dataset(t); }, code::format, "no dataset_description.json");
  refusal([&] { bids::MemoryTree t; t.add("dataset_description.json", R"({"BIDSVersion":"1.9.0"})"); bids::read_dataset(t); }, code::format, "no Name");
  refusal([&] { bids::MemoryTree t; t.add("dataset_description.json", R"({"Name":"x"})"); bids::read_dataset(t); }, code::format, "no BIDSVersion");
  refusal([&] { bids::MemoryTree t; t.add("dataset_description.json", R"([1])"); bids::read_dataset(t); }, code::format, "not a JSON object");
  refusal([&] { bids::MemoryTree t; t.add("dataset_description.json", R"({"Name":"x","BIDSVersion":"1","Authors":3})"); bids::read_dataset(t); }, code::format, "not a list");
  refusal([&] { bids::DirectorySource s("/no/such/folder"); }, code::io, "not a folder");
  refusal([&] { bids::parse_name("sub-01_task_eeg.edf"); }, code::format, "not a key-value entity");
  refusal([&] { bids::parse_name("sub-01_sub-02_eeg.edf"); }, code::format, "repeated");
  refusal([&] {
    bids::MemoryTree t;
    t.add("dataset_description.json", R"({"Name":"x","BIDSVersion":"1.9.0"})");
    t.add("sub-01/eeg/notes_misc.txt", "");
    bids::ReadOptions o;
    o.strict = true;
    bids::read_dataset(t, o);
  }, code::format, "notes_misc.txt");
  return nxrtest::finish("formats_bids");
}
