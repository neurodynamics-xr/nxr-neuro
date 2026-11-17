// KIT/Yokogawa files against goldens made by MNE-Python reading synthetic files (test/golden/gen_kit.py): the directory and system block field by field,
// the channels and their conversion factors, the data by block (int16 and int32), the digitization, and the refusals.
#include <cmath>
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/errors.h"
#include "nxr/formats/kit.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/kit/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

static bool near(double a, double b, double rel = 1e-9, double abs_ = 1e-12) { return std::fabs(a - b) <= rel * std::fabs(b) + abs_; }

static void check_case(const std::string& file, const std::string& key, const json& g) {
  std::printf("  case %s\n", key.c_str());
  auto r = kit::open_file(kData + file);
  const RecordingHeader& h = r->header();
  const kit::Info& I = r->info();
  CHECK_EQ(h.n_samples, g["n_times"].get<std::uint64_t>());
  CHECK(near(h.sampling_rate, g["sfreq"].get<double>()));
  CHECK_EQ(I.nchan, g["nchan"].get<int>());
  CHECK(near(I.highpass, g["highpass"].get<double>()));
  CHECK(near(I.lowpass, g["lowpass"].get<double>()));
  CHECK(near(I.notch, g["notch"].get<double>()));
  CHECK_EQ(I.acq_type, g["acq_type"].get<int>());
  CHECK_EQ(I.system_id, g["kit_system_id"].get<int>());
  CHECK_EQ(I.directories.size(), g["dirs"].size());
  for (std::size_t i = 0; i < I.directories.size() && i < g["dirs"].size(); ++i) {
    CHECK_EQ(I.directories[i].offset, g["dirs"][i][0].get<std::uint32_t>());
    CHECK_EQ(I.directories[i].size, g["dirs"][i][1].get<int>());
    CHECK_EQ(I.directories[i].count, g["dirs"][i][3].get<int>());
  }
  CHECK_EQ(h.start_time, g["meas_date"].get<std::string>().substr(0, 19));
  // channels
  CHECK_EQ(h.channels.size(), g["ch_names"].size());
  for (std::size_t i = 0; i < h.channels.size() && i < g["ch_names"].size(); ++i) {
    const Channel& c = h.channels[i];
    CHECK_EQ(c.name, g["ch_names"][i].get<std::string>());
    const int kind = g["kinds"][i].get<int>();
    CHECK_EQ(c.kind, std::string(kind == 1 ? "meg" : kind == 301 ? "meg_ref" : kind == 2 ? "eeg" : kind == 402 ? "ecg" : "misc"));
    CHECK_EQ(c.unit, std::string(g["units"][i].get<int>() == 112 ? "T" : "V"));
    const int coil = g["coil_types"][i].get<int>();
    CHECK_EQ(c.coil_type, std::string(coil == 6001 ? "KIT_GRAD" : coil == 6002 ? "KIT_REF_MAG" : coil == 1 ? "EEG" : ""));
    CHECK(near(c.gain, g["conv_factor"][i].get<double>(), 1e-12, 0));
    const json& w = g["channels"][i];
    CHECK_EQ(I.channels[i].type, w["type"].get<int>());
    if (!w["loc"].is_null()) for (int a = 0; a < 5; ++a) CHECK(near(I.channels[i].loc[static_cast<std::size_t>(a)], w["loc"][a].get<double>(), 1e-12, 1e-15));
  }
  // digitization
  if (g["elp"].is_null()) {
    CHECK(!I.digitization.has_value());
  } else {
    CHECK(I.digitization.has_value());
    if (I.digitization) {
      const auto& d = *I.digitization;
      CHECK_EQ(d.elp.size(), g["elp"].size());
      for (std::size_t i = 0; i < d.elp.size() && i < g["elp"].size(); ++i)
        for (int a = 0; a < 3; ++a) CHECK(near(d.elp[i][static_cast<std::size_t>(a)], g["elp"][i][a].get<double>(), 1e-12, 1e-15));
      CHECK_EQ(d.head_shape.size(), g["hsp"].size());
      for (std::size_t i = 0; i < d.head_shape.size() && i < g["hsp"].size(); ++i)
        for (int a = 0; a < 3; ++a) CHECK(near(d.head_shape[i][static_cast<std::size_t>(a)], g["hsp"][i][a].get<double>(), 1e-12, 1e-15));
      std::vector<std::array<double, 3>> done;   // MNE keeps the coils measured in the device frame
      for (std::size_t i = 0; i < d.marker_coils.size(); ++i) if (d.marker_done[i]) done.push_back(d.marker_coils[i]);
      CHECK_EQ(done.size(), g["mrk"].size());
      for (std::size_t i = 0; i < done.size() && i < g["mrk"].size(); ++i)
        for (int a = 0; a < 3; ++a) CHECK(near(done[i][static_cast<std::size_t>(a)], g["mrk"][i][a].get<double>(), 1e-12, 1e-15));
      for (auto it = g["eeg_dig"].begin(); it != g["eeg_dig"].end(); ++it) {
        bool found = false;
        for (const auto& n : d.named)
          if (n.first == it.key()) { found = true; for (int a = 0; a < 3; ++a) CHECK(near(n.second[static_cast<std::size_t>(a)], it.value()[a].get<double>(), 1e-12, 1e-15)); }
        CHECK(found);
      }
    }
  }
  rg::check_against(*r, rg::load_f32(kGold + "kit_" + key + ".f32"), key.c_str(), 2e-6, 1e-20);
}

int main() {
  std::ifstream gf(kGold + "kit_golden.json");
  const json G = json::parse(gf);
  check_case("kit_cont.con", "cont", G["cases"]["cont"]);
  check_case("kit_nodig_i32.con", "nodig_i32", G["cases"]["nodig_i32"]);
  auto refusal = [&](const char* f, const char* code_, const char* needle) {
    try {
      kit::open_file(kData + f);
      std::printf("  %s was not refused\n", f);
      CHECK(false);
    } catch (const nxr::io::Error& e) {
      CHECK(e.code() == code_);
      const bool found = std::string(e.what()).find(needle) != std::string::npos;
      if (!found) std::printf("  %s: message '%s' lacks '%s'\n", f, e.what(), needle);
      CHECK(found);
    }
  };
  refusal("kit_old.con", code::refused, "not officially supported");
  refusal("kit_epoched.con", code::refused, "epoched");
  refusal("kit_evoked.con", code::refused, "evoked");
  refusal("kit_truncated.con", code::format, "truncated");
  refusal("kit_not.con", code::format, "not a KIT file");
  refusal("missing.con", code::io, "");
  return nxrtest::finish("formats_kit");
}
