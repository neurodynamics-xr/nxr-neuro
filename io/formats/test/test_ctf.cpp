// CTF data sets against goldens made by MNE-Python reading synthetic data sets (test/golden/gen_ctf.py): the .res4 header field by field, the channels,
// the data by block (several trials, a continuation file, the system clock), the head coils and their transforms, markers, bad segments and channels,
// the electrode and digitizer files; and the refusals.
#include <cmath>
#include <fstream>

#include "nlohmann/json.hpp"
#include "nxr/formats/ctf.h"
#include "nxr/formats/errors.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/ctf/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

static bool near(double a, double b, double rel = 1e-9, double abs_ = 1e-12) { return std::fabs(a - b) <= rel * std::fabs(b) + abs_; }

static void check_matrix(const std::optional<ctf::Matrix4>& got, const json& want, const char* what) {
  if (want.is_null()) { CHECK(!got); return; }
  CHECK(got.has_value());
  if (!got) return;
  bool ok = true;
  for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) ok = ok && near((*got)[static_cast<std::size_t>(r * 4 + c)], want[r][c].get<double>(), 1e-6, 1e-8);
  if (!ok) std::printf("  transform %s differs\n", what);
  CHECK(ok);
}

static void check_case(const std::string& file, const std::string& key, const json& g, ctf::OpenOptions opt = {}) {
  std::printf("  case %s\n", key.c_str());
  ctf::DirectoryFolder folder(kData + file);
  auto r = ctf::open(folder, opt);
  const RecordingHeader& h = r->header();
  const ctf::Res4& res = r->res4();
  const json& g4 = g["res4"];
  CHECK_EQ(h.n_samples, g["n_times"].get<std::uint64_t>());
  CHECK(near(h.sampling_rate, g["sfreq"].get<double>()));
  CHECK_EQ(h.channels.size(), g["ch_names"].size());
  // channels: names, kinds, units, calibration, coil types and the compensation grade
  for (std::size_t i = 0; i < h.channels.size() && i < g["ch_names"].size(); ++i) {
    const Channel& c = h.channels[i];
    CHECK_EQ(c.name, g["ch_names"][i].get<std::string>());
    const int kind = g["kinds"][i].get<int>();
    const std::string want_kind = kind == 1 ? "meg" : kind == 301 ? "meg_ref" : kind == 2 ? "eeg" : kind == 3 ? "stim" : "misc";
    CHECK_EQ(c.kind, want_kind);
    CHECK_EQ(c.unit, std::string(g["units"][i].get<int>() == 112 ? "T" : "V"));
    CHECK(near(c.gain, g["cals"][i].get<double>(), 1e-12, 0));
    const int coil = g["coil_types"][i].get<int>();
    const std::string want_coil = coil == 5001 ? "CTF_GRAD" : coil == 5002 ? "CTF_REF_MAG" : coil == 5003 ? "CTF_REF_GRAD" : coil == 5004 ? "CTF_OFFDIAG_REF_GRAD" : coil == 1 ? "EEG" : "";
    CHECK_EQ(c.coil_type, want_coil);
    CHECK_EQ(c.bad, std::find(g["bads"].begin(), g["bads"].end(), g["ch_names"][i]) != g["bads"].end());
  }
  CHECK_EQ(h.bad_channels.size(), g["bads"].size());
  CHECK_EQ(r->compensation_grade(), g["grades"][0].get<int>() ? g["grades"][0].get<int>() : 0);
  // the .res4 header, field by field
  CHECK_EQ(res.head, g4["head"].get<std::string>());
  CHECK_EQ(res.appname, g4["appname"].get<std::string>());
  CHECK_EQ(res.origin, g4["origin"].get<std::string>());
  CHECK_EQ(res.desc, g4["desc"].get<std::string>());
  CHECK_EQ(res.nave, g4["nave"].get<int>());
  CHECK_EQ(res.data_time, g4["data_time"].get<std::string>());
  CHECK_EQ(res.data_date, g4["data_date"].get<std::string>());
  CHECK_EQ(res.nsamp, g4["nsamp"].get<int>());
  CHECK_EQ(res.nchan, g4["nchan"].get<int>());
  CHECK(near(res.sfreq, g4["sfreq"].get<double>()));
  CHECK(near(res.epoch_time, g4["epoch_time"].get<double>()));
  CHECK_EQ(res.no_trials, g4["no_trials"].get<int>());
  CHECK_EQ(res.pre_trig_pts, g4["pre_trig_pts"].get<int>());
  CHECK_EQ(res.no_trials_done, g4["no_trials_done"].get<int>());
  CHECK_EQ(res.save_trials, g4["save_trials"].get<int>());
  CHECK_EQ(static_cast<int>(res.primary_trigger), static_cast<int>(g4["primary_trigger"].get<std::string>()[0]));
  for (std::size_t i = 0; i < 8; ++i) CHECK_EQ(static_cast<int>(res.secondary_trigger[i]), static_cast<int>(g4["secondary_trigger"][i].get<std::string>()[0]));
  CHECK_EQ(res.trigger_mode, g4["trigger_mode"].get<int>());
  CHECK_EQ(res.accept_reject, g4["accept_reject"].get<int>());
  CHECK_EQ(res.zero_head, g4["zero_head"].get<int>());
  CHECK_EQ(res.artifact_mode, g4["artifact_mode"].get<int>());
  CHECK_EQ(res.nf_run_name, g4["nf_run_name"].get<std::string>());
  CHECK_EQ(res.nf_run_title, g4["nf_run_title"].get<std::string>());
  CHECK_EQ(res.nf_instruments, g4["nf_instruments"].get<std::string>());
  CHECK_EQ(res.nf_collect_descriptor, g4["nf_collect_descriptor"].get<std::string>());
  CHECK_EQ(res.nf_subject_id, g4["nf_subject_id"].get<std::string>());
  CHECK_EQ(res.nf_operator, g["experimenter"].get<std::string>());
  CHECK_EQ(res.nf_sensor_file_name.size(), std::size_t(60));
  CHECK_EQ(res.rdlen, g4["rdlen"].get<int>());
  CHECK_EQ(res.run_desc, g4["run_desc"].get<std::string>());
  CHECK_EQ(res.filters.size(), g4["filters"].size());
  for (std::size_t i = 0; i < res.filters.size() && i < g4["filters"].size(); ++i) {
    CHECK(near(res.filters[i].freq, g4["filters"][i]["freq"].get<double>()));
    CHECK_EQ(res.filters[i].type, g4["filters"][i]["type"].get<int>());
    CHECK_EQ(res.filters[i].klass, g4["filters"][i]["class"].get<int>());
    CHECK_EQ(res.filters[i].pars.size(), g4["filters"][i]["pars"].size());
  }
  CHECK_EQ(res.channels.size(), g4["chs"].size());
  for (std::size_t i = 0; i < res.channels.size() && i < g4["chs"].size(); ++i) {
    const ctf::ChannelRecord& c = res.channels[i];
    const json& w = g4["chs"][i];
    CHECK_EQ(c.name, w["ch_name"].get<std::string>());
    CHECK_EQ(c.sensor_type_index, w["sensor_type_index"].get<int>());
    CHECK_EQ(c.original_run_no, w["original_run_no"].get<int>());
    CHECK_EQ(c.coil_type, w["coil_type"].get<int>());
    CHECK(near(c.proper_gain, w["proper_gain"].get<double>()));
    CHECK(near(c.qgain, w["qgain"].get<double>()));
    CHECK(near(c.io_gain, w["io_gain"].get<double>()));
    CHECK_EQ(c.num_coils, w["num_coils"].get<int>());
    CHECK_EQ(c.grad_order_no, w["grad_order_no"].get<int>());
    for (std::size_t k = 0; k < 8; ++k) {
      for (int a = 0; a < 3; ++a) {
        CHECK(near(c.coil[k].pos[static_cast<std::size_t>(a)], w["coil"][k][0][a].get<double>(), 1e-9, 1e-12));
        CHECK(near(c.coil[k].norm[static_cast<std::size_t>(a)], w["coil"][k][2][a].get<double>(), 1e-9, 1e-12));
        CHECK(near(c.head_coil[k].pos[static_cast<std::size_t>(a)], w["head_coil"][k][0][a].get<double>(), 1e-9, 1e-12));
      }
      CHECK_EQ(c.coil[k].turns, w["coil"][k][4].get<int>());
      CHECK(near(c.coil[k].area, w["coil"][k][7].get<double>(), 1e-9, 1e-15));
    }
  }
  CHECK_EQ(res.compensation.size(), g4["comp"].size());
  for (std::size_t i = 0; i < res.compensation.size() && i < g4["comp"].size(); ++i) {
    CHECK_EQ(res.compensation[i].sensor_name, g4["comp"][i]["sensor_name"].get<std::string>());
    CHECK_EQ(res.compensation[i].coeff_type, g4["comp"][i]["coeff_type"].get<int>());
    CHECK_EQ(res.compensation[i].sensors.size(), g4["comp"][i]["sensors"].size());
    for (std::size_t k = 0; k < res.compensation[i].coeffs.size() && k < g4["comp"][i]["coeffs"].size(); ++k)
      CHECK(near(res.compensation[i].coeffs[k], g4["comp"][i]["coeffs"][k].get<double>()));
  }
  // time and identity
  CHECK(!g["meas_date"].is_null());
  CHECK_EQ(h.start_time, g["meas_date"].get<std::string>().substr(0, 19));
  // head coils: the measured ones, in file order
  std::vector<const ctf::CoilPoint*> valid;
  for (const auto& p : r->coils()) if (p.valid) valid.push_back(&p);
  CHECK_EQ(valid.size(), g["coils"].is_null() ? std::size_t(0) : g["coils"].size());
  for (std::size_t i = 0; i < valid.size() && !g["coils"].is_null() && i < g["coils"].size(); ++i) {
    CHECK_EQ(valid[i]->coord_frame, g["coils"][i]["coord_frame"].get<int>());
    CHECK_EQ(valid[i]->kind, g["coils"][i]["kind"].get<int>());
    for (int a = 0; a < 3; ++a) CHECK(near(valid[i]->r[static_cast<std::size_t>(a)], g["coils"][i]["r"][a].get<double>(), 1e-9, 1e-12));
  }
  if (g["trans"].is_null()) {
    CHECK(!r->transforms().ctf_head_to_head.has_value());
  } else {
    const json& t = g["trans"];
    check_matrix(r->transforms().ctf_head_to_head, t["t_ctf_head_head"], "ctf head -> head");
    check_matrix(r->transforms().ctf_dev_to_ctf_head, t["t_ctf_dev_ctf_head"], "ctf dev -> ctf head");
    check_matrix(r->transforms().ctf_dev_to_head, t["t_ctf_dev_head"], "ctf dev -> head");
    check_matrix(r->transforms().dev_head, t["t_dev_head"], "dev -> head");
    check_matrix(std::optional<ctf::Matrix4>(r->transforms().ctf_dev_to_dev), t["t_ctf_dev_dev"], "ctf dev -> dev");
  }
  // electrodes (.eeg) and digitizer points (.pos, mapped to the head by the transform, as MNE does)
  if (g["eeg"].is_null()) {
    CHECK(r->electrodes().empty());
  } else {
    CHECK_EQ(r->electrodes().size(), g["eeg"]["labels"].size());
    for (std::size_t i = 0; i < r->electrodes().size() && i < g["eeg"]["labels"].size(); ++i) {
      CHECK_EQ(r->electrodes()[i].label, g["eeg"]["labels"][i].get<std::string>());
      for (int a = 0; a < 3; ++a) CHECK(near(r->electrodes()[i].r[static_cast<std::size_t>(a)], g["eeg"]["rr"][i][a].get<double>(), 1e-9, 1e-12));
    }
  }
  {
    std::vector<std::array<double, 3>> mapped;
    for (const auto& p : r->polhemus())
      if (p.r[0] * p.r[0] + p.r[1] * p.r[1] + p.r[2] * p.r[2] > 1e-4) mapped.push_back(ctf::apply(*r->transforms().ctf_head_to_head, p.r));
    CHECK_EQ(mapped.size(), g["pos"].size());
    for (std::size_t i = 0; i < mapped.size() && i < g["pos"].size(); ++i)
      for (int a = 0; a < 3; ++a) CHECK(near(mapped[i][static_cast<std::size_t>(a)], g["pos"][i]["r"][a].get<double>(), 1e-6, 1e-9));
  }
  // annotations: bad segments and markers, compared in onset order
  std::vector<std::tuple<double, double, std::string>> got, want;
  for (const auto& a : h.annotations) got.emplace_back(a.onset, a.duration, a.label);
  for (const auto& a : g["annotations"]) want.emplace_back(a[0].get<double>(), a[1].get<double>(), a[2].get<std::string>());
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  CHECK_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
    CHECK(near(std::get<0>(got[i]), std::get<0>(want[i]), 1e-9, 1e-9));
    CHECK(near(std::get<1>(got[i]), std::get<1>(want[i]), 1e-9, 1e-9));
    CHECK_EQ(std::get<2>(got[i]), std::get<2>(want[i]));
  }
  // the data, by block
  const std::string f32 = kGold + "ctf_" + key + ".f32";
  rg::check_against(*r, rg::load_f32(f32), key.c_str(), 2e-6, 1e-20);
}

int main() {
  std::ifstream gf(kGold + "ctf_golden.json");
  const json G = json::parse(gf);
  check_case("ctf_cont.ds", "cont", G["cases"]["cont"]);
  check_case("ctf_clock.ds", "clock", G["cases"]["clock"]);
  ctf::OpenOptions ignore;
  ignore.system_clock = ctf::OpenOptions::SystemClock::ignore;
  check_case("ctf_clock.ds", "clock_ignored", G["cases"]["clock_ignored"], ignore);
  {
    // the structure of a recording over several files: a read straddling the continuation file equals the two halves read apart
    ctf::DirectoryFolder folder(kData + "ctf_cont.ds");
    auto r = ctf::open(folder);
    std::vector<float> whole(r->header().channels.size() * 100), a(r->header().channels.size() * 40), b(r->header().channels.size() * 60);
    r->read(330, 100, 0, r->header().channels.size(), whole.data());
    r->read(330, 40, 0, r->header().channels.size(), a.data());
    r->read(370, 60, 0, r->header().channels.size(), b.data());
    bool ok = true;
    for (std::size_t c = 0; c < r->header().channels.size(); ++c) {
      for (std::size_t i = 0; i < 40; ++i) ok = ok && whole[c * 100 + i] == a[c * 40 + i];
      for (std::size_t i = 0; i < 60; ++i) ok = ok && whole[c * 100 + 40 + i] == b[c * 60 + i];
    }
    CHECK(ok);
    CHECK_EQ(r->files().size() > 8, true);
    CHECK(rg::throws_code([&] { std::vector<float> x(10); r->read(r->header().n_samples, 1, 0, 1, x.data()); }, code::out_of_range));
  }
  auto refusal = [&](const char* ds, const char* code_, const char* needle) {
    try {
      ctf::open_directory(kData + ds);
      std::printf("  %s was not refused\n", ds);
      CHECK(false);
    } catch (const nxr::io::Error& e) {
      CHECK(e.code() == code_);
      const bool found = std::string(e.what()).find(needle) != std::string::npos;
      if (!found) std::printf("  %s: message '%s' lacks '%s'\n", ds, e.what(), needle);
      CHECK(found);
    }
  };
  refusal("ctf_epoched.ds", code::refused, "epoched");
  refusal("ctf_averaged.ds", code::refused, "averaged");
  refusal("ctf_badtrial.ds", code::format, "even multiple of the trial size");
  refusal("ctf_notctf.ds", code::format, "MEG4");
  refusal("ctf_nodata.ds", code::io, "no data");
  refusal("ctf_badgrade.ds", code::format, "inconsistent compensation grade");
  refusal("ctf_nores4.ds", code::io, "ctf_nores4.res4 not found");
  refusal("missing.ds", code::io, "not a folder");
  CHECK(rg::throws_code([&] { ctf::DirectoryFolder f("/tmp"); }, code::refused));
  return nxrtest::finish("formats_ctf");
}
