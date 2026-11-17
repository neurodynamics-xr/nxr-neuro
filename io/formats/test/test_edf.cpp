// EDF, EDF+, BDF, BDF+ against goldens made with MNE-Python (test/golden/gen_edf.py): values, annotations, gaps, rate selection, refusals.
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/edf.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

int main() {
  std::ifstream gf(kGold + "edf_golden.json");
  const json g = json::parse(gf);
  for (const auto& [name, c] : g["cases"].items()) {
    if (!c.contains("data")) continue;
    auto r = edf::open_file(kData + c["file"].get<std::string>());
    const RecordingHeader& h = r->header();
    CHECK_EQ(h.sampling_rate, c["sfreq"].get<double>());
    CHECK_EQ(h.channels.size(), c["mne_channels"].size());
    for (std::size_t i = 0; i < h.channels.size() && i < c["mne_channels"].size(); ++i) CHECK_EQ(h.channels[i].name, c["mne_channels"][i].get<std::string>());
    if (c.contains("n_times")) CHECK_EQ(h.n_samples, c["n_times"].get<std::uint64_t>());
    if (c.contains("n_samples")) CHECK_EQ(h.n_samples, c["n_samples"].get<std::uint64_t>());
    CHECK_EQ(h.format, c["format"].get<std::string>());
    CHECK(h.precision == Precision::float32);
    // annotations
    CHECK_EQ(h.annotations.size(), c["annotations"].size());
    for (std::size_t i = 0; i < h.annotations.size() && i < c["annotations"].size(); ++i) {
      const auto& a = c["annotations"][i];
      CHECK_EQ(h.annotations[i].label, a["label"].get<std::string>());
      CHECK(std::fabs(h.annotations[i].onset - a["onset"].get<double>()) < 1e-6);
      CHECK(std::fabs(h.annotations[i].duration - a["duration"].get<double>()) < 1e-6);
    }
    // gaps
    if (c.contains("gaps")) {
      CHECK_EQ(h.gaps.size(), c["gaps"].size());
      for (std::size_t i = 0; i < h.gaps.size() && i < c["gaps"].size(); ++i) {
        CHECK_EQ(h.gaps[i].first, c["gaps"][i]["first"].get<std::uint64_t>());
        CHECK_EQ(h.gaps[i].count, c["gaps"][i]["count"].get<std::uint64_t>());
      }
    } else {
      CHECK(h.gaps.empty());
    }
    rg::check_against(*r, rg::load_f32(kGold + c["data"].get<std::string>()), name.c_str());
  }

  // units: the file's microvolts and millivolts come out in volts, the file's unit is kept
  {
    auto r = edf::open_file(kData + "edf_plain.edf");
    const RecordingHeader& h = r->header();
    CHECK_EQ(h.channels[0].file_unit, std::string("uV"));
    CHECK_EQ(h.channels[0].unit, std::string("V"));
    CHECK_EQ(h.channels[0].scale_to_si, 1e-6);
    CHECK_EQ(h.channels[1].file_unit, std::string("mV"));
    CHECK_EQ(h.channels[1].scale_to_si, 1e-3);
    CHECK_EQ(h.channels[0].transducer, std::string("AgAgCl electrode"));
    CHECK_EQ(h.channels[0].prefilter, std::string("HP:0.1Hz LP:75Hz"));
    CHECK_EQ(h.start_time, std::string("2003-02-01T04:05:06"));
    CHECK_EQ(h.info.at("patient"), std::string("X X X X"));   // parsed; what to keep is the importer's call
  }
  {
    auto r = edf::open_file(kData + "bdf_plain.bdf");
    CHECK_EQ(r->header().channels[3].kind, std::string("stim"));
    CHECK_EQ(r->header().channels[3].file_unit, std::string("Boolean"));
    CHECK_EQ(r->header().format, std::string("bdf"));
  }
  // several rates: refused naming them, then read by rate
  {
    const std::string f = kData + "edf_mixed_rates.edf";
    CHECK(rg::throws_code([&] { edf::open_file(f); }, code::refused));
    try { edf::open_file(f); } catch (const nxr::io::Error& e) { CHECK(std::string(e.what()).find("100") != std::string::npos && std::string(e.what()).find("25") != std::string::npos); }
    edf::OpenOptions o;
    o.sampling_rate = 100.0;
    rg::check_against(*edf::open_file(f, o), rg::load_f32(kGold + "edf_mixed_fast.f32"), "mixed fast");
    o.sampling_rate = 25.0;
    auto slow = edf::open_file(f, o);
    CHECK_EQ(slow->header().channels.size(), 1u);
    rg::check_against(*slow, rg::load_f32(kGold + "edf_mixed_slow.f32"), "mixed slow");
    o.sampling_rate = 50.0;
    CHECK(rg::throws_code([&] { edf::open_file(f, o); }, code::refused));
  }
  // gaps read as zeros
  {
    auto r = edf::open_file(kData + "edf_plus_d.edf");
    std::vector<float> v(40);
    r->read(60, 40, 0, 1, v.data());
    bool zeros = true;
    for (float x : v) zeros = zeros && x == 0.0f;
    CHECK(zeros);
  }
  // refusals
  CHECK(rg::throws_code([&] { edf::open_file(kData + "edf_truncated.edf"); }, code::format));
  CHECK(rg::throws_code([&] { edf::open_file(kData + "edf_bad_version.edf"); }, code::format));
  CHECK(rg::throws_code([&] { edf::open_file(kData + "missing.edf"); }, code::io));
  CHECK(rg::throws_code([&] { edf::open(std::make_shared<MemorySource>(std::string("short"))); }, code::format));
  return nxrtest::finish("formats_edf");
}
