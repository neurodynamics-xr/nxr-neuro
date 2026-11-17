// BrainVision against goldens made with MNE-Python (test/golden/gen_brainvision.py): values, markers, coordinates, both data orientations, refusals.
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/brainvision.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

int main() {
  std::ifstream gf(kGold + "brainvision_golden.json");
  const json g = json::parse(gf);
  for (const auto& [name, c] : g["cases"].items()) {
    auto r = brainvision::open_file(kData + c["file"].get<std::string>());
    const RecordingHeader& h = r->header();
    CHECK_EQ(h.format, std::string("brainvision"));
    CHECK_EQ(h.sampling_rate, c["sfreq"].get<double>());
    CHECK_EQ(h.n_samples, c["n_times"].get<std::uint64_t>());
    CHECK_EQ(h.channels.size(), c["mne_channels"].size());
    for (std::size_t i = 0; i < h.channels.size() && i < c["mne_channels"].size(); ++i) CHECK_EQ(h.channels[i].name, c["mne_channels"][i].get<std::string>());
    CHECK(h.gaps.empty());
    // markers (MNE leaves the New Segment markers out of its annotations; here they are annotations too)
    std::vector<Annotation> marks;
    for (const Annotation& a : h.annotations) if (a.label.compare(0, 11, "New Segment") != 0) marks.push_back(a);
    CHECK_EQ(marks.size(), c["annotations"].size());
    for (std::size_t i = 0; i < marks.size() && i < c["annotations"].size(); ++i) {
      CHECK_EQ(marks[i].label, c["annotations"][i]["label"].get<std::string>());
      CHECK(std::fabs(marks[i].onset - c["annotations"][i]["onset"].get<double>()) < 1e-9);
      CHECK(std::fabs(marks[i].duration - c["annotations"][i]["duration"].get<double>()) < 1e-9);
    }
    // MNE places the coordinates on a head sphere of 95 mm radius for a radius of 1
    if (c.contains("positions")) {
      for (const Channel& ch : h.channels) {
        CHECK(ch.location.has_value());
        if (!ch.location) continue;
        const auto& w = c["positions"][ch.name];
        for (int k = 0; k < 3; ++k) CHECK(std::fabs((*ch.location)[k] * 0.095 - w[k].get<double>()) < 1e-9);
      }
    } else {
      for (const Channel& ch : h.channels) CHECK(!ch.location.has_value());
    }
    rg::check_against(*r, rg::load_f32(kGold + c["data"].get<std::string>()), name.c_str());
  }
  {
    auto r = brainvision::open_file(kData + "bv_mux_i16.vhdr");
    const RecordingHeader& h = r->header();
    CHECK_EQ(h.channels[0].file_unit, std::string("\xC2\xB5V"));
    CHECK_EQ(h.channels[0].unit, std::string("V"));
    CHECK_EQ(h.channels[0].scale_to_si, 1e-6);
    CHECK_EQ(h.channels[0].gain, 0.1);
    CHECK_EQ(h.start_time, std::string("2026-01-02T15:04:05.000000"));
    CHECK_EQ(h.info.at("segment_starts"), std::string("0"));
    CHECK_EQ(h.info.at("data_orientation"), std::string("MULTIPLEXED"));
    CHECK(h.annotations.back().channels == "Fp2");
  }
  {
    auto r = brainvision::open_file(kData + "bv_mux_i32.vhdr");
    CHECK_EQ(r->header().channels[0].file_unit, std::string("mV"));
    CHECK_EQ(r->header().channels[0].scale_to_si, 1e-3);
  }
  // refusals
  CHECK(rg::throws_code([&] { brainvision::open_file(kData + "bv_ascii.vhdr"); }, code::refused));
  CHECK(rg::throws_code([&] { brainvision::open_file(kData + "bv_uint16.vhdr"); }, code::refused));
  CHECK(rg::throws_code([&] { brainvision::open_file(kData + "bv_truncated.vhdr"); }, code::format));
  CHECK(rg::throws_code([&] { brainvision::open_file(kData + "missing.vhdr"); }, code::io));
  CHECK(rg::throws_code([&] { brainvision::open_file(kData + "bv_mux_i16.vmrk"); }, code::format));   // a marker file is not a header
  return nxrtest::finish("formats_brainvision");
}
