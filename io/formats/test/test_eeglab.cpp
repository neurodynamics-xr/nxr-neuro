// EEGLAB against goldens made with MNE-Python (test/golden/gen_eeglab.py): the .fdt form, data inside the .set (single and double), the
// top-level-variables form, events, channel locations, and the refusals (compressed variable, epoched, truncated, missing .fdt).
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/eeglab.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

int main() {
  std::ifstream gf(kGold + "eeglab_golden.json");
  const json g = json::parse(gf);
  for (const auto& [name, c] : g["cases"].items()) {
    auto r = eeglab::open_file(kData + c["file"].get<std::string>());
    const RecordingHeader& h = r->header();
    CHECK_EQ(h.format, std::string("eeglab"));
    CHECK_EQ(h.sampling_rate, c["sfreq"].get<double>());
    CHECK_EQ(h.n_samples, c["n_times"].get<std::uint64_t>());
    CHECK_EQ(h.channels.size(), c["mne_channels"].size());
    for (std::size_t i = 0; i < h.channels.size() && i < c["mne_channels"].size(); ++i) CHECK_EQ(h.channels[i].name, c["mne_channels"][i].get<std::string>());
    CHECK_EQ(h.annotations.size(), c["annotations"].size());
    for (std::size_t i = 0; i < h.annotations.size() && i < c["annotations"].size(); ++i) {
      CHECK_EQ(h.annotations[i].label, c["annotations"][i]["label"].get<std::string>());
      CHECK(std::fabs(h.annotations[i].onset - c["annotations"][i]["onset"].get<double>()) < 1e-9);
      CHECK(std::fabs(h.annotations[i].duration - c["annotations"][i]["duration"].get<double>()) < 1e-9);
    }
    CHECK(h.gaps.empty());
    const std::vector<float> want = rg::load_f32(kGold + c["data"].get<std::string>());
    if (h.precision == Precision::float32) {
      rg::check_against(*r, want, name.c_str());
    } else {
      // double data: the double read gives the file's values (times 1e-6); the float read refuses
      const std::size_t nc = h.channels.size();
      const std::uint64_t n = h.n_samples;
      std::vector<double> whole(nc * n);
      r->read(0, n, 0, nc, whole.data());
      bool ok = true;
      for (std::size_t i = 0; ok && i < whole.size(); ++i) ok = rg::close(static_cast<float>(whole[i]), want[i]);
      CHECK(ok);
      std::vector<double> part(2 * 100);
      r->read(300, 100, 1, 3, part.data());
      ok = true;
      for (std::size_t c2 = 0; ok && c2 < 2; ++c2)
        for (std::size_t i = 0; ok && i < 100; ++i) ok = part[c2 * 100 + i] == whole[(c2 + 1) * n + 300 + i];
      CHECK(ok);
      std::vector<float> f(10);
      CHECK(rg::throws_code([&] { r->read(0, 10, 0, 1, f.data()); }, code::refused));
    }
  }
  {
    auto r = eeglab::open_file(kData + "eeglab_fdt.set");
    const RecordingHeader& h = r->header();
    CHECK(h.precision == Precision::float32);
    CHECK_EQ(h.channels[0].file_unit, std::string("uV"));
    CHECK_EQ(h.channels[0].scale_to_si, 1e-6);
    CHECK_EQ(h.channels[0].kind, std::string("eeg"));
    CHECK(h.channels[0].location.has_value());
    if (h.channels[0].location) { CHECK_EQ((*h.channels[0].location)[0], 80.1); CHECK_EQ((*h.channels[3].location)[2], 70.0); }
    CHECK_EQ(h.info.at("data_file"), std::string("eeglab_fdt.fdt"));
    CHECK_EQ(h.info.at("comments"), std::string("made up"));
    CHECK_EQ(h.info.at("icaweights"), std::string("4x4"));   // recorded by size, not applied
    CHECK_EQ(h.annotations[3].label, std::string("boundary"));
    CHECK(std::fabs(h.annotations[3].duration - 50.0 / 250.0) < 1e-12);
  }
  {
    auto r = eeglab::open_file(kData + "eeglab_inline_double.set");
    CHECK(r->header().precision == Precision::float64);
  }
  // refusals
  CHECK(rg::throws_code([&] { eeglab::open_file(kData + "eeglab_compressed.set"); }, code::refused));
  try { eeglab::open_file(kData + "eeglab_compressed.set"); } catch (const nxr::io::Error& e) { CHECK(std::string(e.what()).find(".fdt") != std::string::npos); }
  CHECK(rg::throws_code([&] { eeglab::open_file(kData + "eeglab_epoched.set"); }, code::refused));
  CHECK(rg::throws_code([&] { eeglab::open_file(kData + "eeglab_fdt_short.set"); }, code::format));
  CHECK(rg::throws_code([&] { eeglab::open_file(kData + "eeglab_nofdt.set"); }, code::io));
  CHECK(rg::throws_code([&] { eeglab::open_file(kData + "eeglab_not_mat.set"); }, code::format));
  CHECK(rg::throws_code([&] { eeglab::open_file(kData + "missing.set"); }, code::io));
  return nxrtest::finish("formats_eeglab");
}
