// The BIDS importer (nxr/convert/bids.h) on the readers' synthetic BIDS tree: a catalogue of subject packages, the participant variables, the recordings imported through
// `import_recording` (copied and in place), and the tables mapped (channel status and types, events, electrode positions); refusals. Nothing here is a real dataset.
#include <unistd.h>

#include <cmath>
#include <filesystem>

#include "check.h"
#include "nxr/convert/bids.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/participants.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/task/task.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace cv = nxr::convert;
namespace tsv = nxr::formats::tsv;

template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; } return ""; }

static tsv::Table table(std::vector<std::string> columns, std::vector<std::vector<std::string>> rows) { return tsv::Table{std::move(columns), std::move(rows)}; }

int main() {
  const std::string tree = std::string(NXR_IO_DATA_DIR) + "/../../formats/test/data/bids";
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_import_bids_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);

  // the tables mapped onto a header (pure)
  {
    nxr::formats::RecordingHeader base;
    base.sampling_rate = 100;
    base.n_samples = 10;
    for (const char* n : {"Fz", "Cz", "EOG1"}) { nxr::formats::Channel c; c.name = n; c.kind = "eeg"; base.channels.push_back(c); }
    nxr::formats::Annotation own, bad;
    own.label = "marker"; own.onset = 0.1;
    bad.label = "BAD_noise"; bad.onset = 0.2; bad.duration = 0.1;
    base.annotations = {own, bad};
    const tsv::Table ch = table({"name", "type", "status"}, {{"Fz", "EEG", "good"}, {"Cz", "EEG", "bad"}, {"EOG1", "EOG", "n/a"}});
    const tsv::Table ev = table({"onset", "duration", "trial_type", "value"}, {{"0.5", "n/a", "stim", "2"}, {"1.5", "0.25", "n/a", "7"}, {"2.5", "0", "n/a", "n/a"}});
    const tsv::Table el = table({"name", "x", "y", "z"}, {{"Fz", "10", "20", "30"}, {"Cz", "n/a", "1", "2"}});
    const auto h = cv::bids_header(base, &ch, &ev, &el, "mm");
    CHECK(h.channels[1].bad && !h.channels[0].bad);
    CHECK_EQ(h.bad_channels.size(), static_cast<std::size_t>(1));
    CHECK_EQ(h.channels[2].kind, std::string("eog"));
    CHECK(h.channels[0].location.has_value() && !h.channels[1].location.has_value());   // a position with a missing coordinate is left out
    CHECK_EQ(h.position_unit, std::string("mm"));
    CHECK((*h.channels[0].location)[1] == 20.0);
    // the events replace the file's markers; the file's bad segment stays
    CHECK_EQ(h.annotations.size(), static_cast<std::size_t>(4));
    CHECK_EQ(h.annotations[0].label, std::string("BAD_noise"));
    CHECK_EQ(h.annotations[1].label, std::string("stim"));
    CHECK_EQ(h.annotations[2].label, std::string("7"));                                  // no trial_type: the value
    CHECK_EQ(h.annotations[3].label, std::string("event"));
    CHECK(h.annotations[1].duration == 0.0);
    // positions in centimetres land in the unit the file's header states
    base.position_unit = "m";
    const auto h2 = cv::bids_header(base, nullptr, nullptr, &el, "cm");
    CHECK(std::fabs((*h2.channels[0].location)[0] - 0.1) < 1e-12);
    CHECK_EQ(h2.position_unit, std::string("m"));
    CHECK_EQ(code_of([&] { cv::bids_header(base, nullptr, nullptr, &el, "inch"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { cv::bids_header(base, &ev, nullptr, nullptr, ""); }), std::string(code::invalid_argument));   // a events table given as channels: no `name` column
  }

  // the tree, copied and in place
  for (const bool in_place : {false, true}) {
    const fs::path ds = root / (in_place ? "linked" : "copied");
    cv::BidsImport opt;
    opt.in_place = in_place;
    const Json out = cv::import_bids(tree, ds, opt);
    CHECK_EQ(out["subjects"].size(), static_cast<std::size_t>(2));
    CHECK(out["recordings"].size() >= 4);
    CHECK_EQ(out["licence"].get<std::string>(), std::string("CC0"));
    for (const auto& p : out["packages"]) CHECK(p["outcome"].get<std::string>() == "registered");
    // the catalogue holds the subjects, the participants the variables
    const Catalogue cat = Catalogue::open(ds);
    CHECK(cat.find("sub-01").has_value() && cat.find("sub-02").has_value());
    const auto vars = participant_variables(ds);
    bool age = false, sex = false;
    for (const auto& v : vars) { age = age || v.name == "age"; sex = sex || v.name == "sex"; }
    CHECK(age && sex);
    CHECK_EQ(participant_values(ds).size(), static_cast<std::size_t>(2));
    // the filter works on them
    SubjectSelection sel;
    sel.where = "age > 34";
    const SelectedSubjects chosen = select_subjects(ds, sel);
    CHECK_EQ(chosen.members.size(), static_cast<std::size_t>(1));
    CHECK_EQ(chosen.members[0].name, std::string("sub-01"));
    // sub-02's recording: events, positions and coordinate system as the tables say; every item has its origin
    SubjectPackage p = SubjectPackage::open(ds / "sub-02.nxr.zarr");
    CHECK(p.verify().ok);
    CHECK(items_without_origin(p.db()).empty());
    const task::Node rec = task::Subject::of(p).require({.cls = task::Cls::field, .name = "task-rest"}, "sub-02's recording");
    CHECK_EQ(rec.producer["bids"]["coordinate_system"]["EEGCoordinateSystem"].get<std::string>(), std::string("CapTrak"));
    CHECK_EQ(rec.producer["bids"]["subject"].get<std::string>(), std::string("sub-02"));
    CHECK_EQ(rec.producer["origin"]["licence"].get<std::string>(), std::string("CC0"));
    CHECK(rec.producer.contains("link") == in_place);
    {   // the electrodes are stored as the recording's channels manifold: points with a positions array, in the coordinate system the table names (what a head model finds)
      const auto sid = p.db().one("SELECT id FROM subject LIMIT 1");
      CHECK(sid.has_value());
      const auto chm = manifold_with_geometry_named(p.db(), text(*sid, "id"), "task-rest_channels");
      CHECK(chm.has_value());
      CHECK_EQ(manifold_coordinate_system_name(p.db(), text(*chm, "id")), std::string("bids-captrak"));
      CHECK(integer(*chm, "n_vertices") >= 6);
    }
    for (const auto& r : out["recordings"]) if (r["subject"] == "sub-02") CHECK_EQ(r["events"].get<std::int64_t>(), static_cast<std::int64_t>(2));
  }

  // refusals: a dataset folder that exists, an unknown subject
  CHECK_EQ(code_of([&] { cv::import_bids(tree, root / "copied", {}); }), std::string(code::already_exists));
  {
    cv::BidsImport opt;
    opt.subjects = {"nobody"};
    CHECK_EQ(code_of([&] { cv::import_bids(tree, root / "none", opt); }), std::string(code::invalid_argument));
  }
  CHECK(!code_of([&] { cv::import_bids(root / "not-a-folder", root / "x", {}); }).empty());
  fs::remove_all(root);
  return nxrtest::finish("import_bids");
}
