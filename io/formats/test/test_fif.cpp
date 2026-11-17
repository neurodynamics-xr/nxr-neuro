// FIF raw recordings against goldens made with MNE-Python (test/golden/gen_fif.py): the whole header, the data by block, split files, skips,
// every data type, and the refusals.
#include <cmath>
#include <fstream>
#include <map>
#include <set>

#include "nlohmann/json.hpp"
#include "nxr/formats/endian.h"
#include "nxr/formats/fif.h"
#include "recording_golden.h"

using namespace nxr::formats;
using nlohmann::json;

static const std::string kData = std::string(NXR_FORMATS_TEST_DIR) + "/data/";
static const std::string kGold = std::string(NXR_FORMATS_TEST_DIR) + "/golden/";

static std::vector<double> load_f64(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<double> v(b.size() / 8);
  std::memcpy(v.data(), b.data(), v.size() * 8);
  return v;
}

static bool near(double a, double b, double rel = 1e-12, double abs_ = 1e-30) { return std::fabs(a - b) <= rel * std::fabs(b) + abs_; }

static void check_transform(const std::optional<fif::CoordTrans>& got, const json& want, const char* what) {
  if (want.is_null()) { CHECK(!got); return; }
  CHECK(got.has_value());
  if (!got) return;
  CHECK_EQ(got->from, want["from"].get<std::int32_t>());
  CHECK_EQ(got->to, want["to"].get<std::int32_t>());
  bool ok = true;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) ok = ok && near(got->trans[r * 4 + c], want["trans"][r][c].get<double>(), 1e-6, 1e-9);
  if (!ok) std::printf("  transform %s differs\n", what);
  CHECK(ok);
}

// The measurement info against MNE's, field by field.
static void check_info(const fif::RawReader& r, const json& c, const std::string& name) {
  const fif::Info& info = r.info();
  const json& g = c["info"];
  const RecordingHeader& h = r.header();
  std::printf("  case %s\n", name.c_str());
  CHECK_EQ(info.nchan, g["nchan"].get<std::int32_t>());
  CHECK_EQ(info.sfreq, g["sfreq"].get<double>());
  CHECK(near(info.highpass.value_or(0.0), g["highpass"].get<double>()));
  CHECK(near(info.lowpass.value_or(info.sfreq / 2.0), g["lowpass"].get<double>()));
  CHECK_EQ(info.line_freq.has_value(), !g["line_freq"].is_null());
  if (info.line_freq && !g["line_freq"].is_null()) CHECK(near(*info.line_freq, g["line_freq"].get<double>()));
  CHECK_EQ(info.meas_date.has_value(), !g["meas_date_stamp"].is_null());
  if (info.meas_date && !g["meas_date_stamp"].is_null()) {
    CHECK_EQ((*info.meas_date)[0], g["meas_date_stamp"][0].get<std::int32_t>());
    CHECK_EQ((*info.meas_date)[1], g["meas_date_stamp"][1].get<std::int32_t>());
  }
  auto opt_str = [&](const std::optional<std::string>& got, const char* key) {
    CHECK_EQ(got.has_value(), !g[key].is_null());
    if (got && !g[key].is_null()) CHECK_EQ(*got, g[key].get<std::string>());
  };
  opt_str(info.experimenter, "experimenter");
  opt_str(info.description, "description");
  opt_str(info.proj_name, "proj_name");
  opt_str(info.utc_offset, "utc_offset");
  CHECK_EQ(info.proj_id.has_value(), !g["proj_id"].is_null());
  CHECK_EQ(info.gantry_angle.has_value(), !g["gantry_angle"].is_null());
  if (info.gantry_angle && !g["gantry_angle"].is_null()) CHECK_EQ(*info.gantry_angle, g["gantry_angle"].get<std::int32_t>());
  CHECK_EQ(info.custom_ref_applied, g["custom_ref_applied"].get<std::int32_t>());
  CHECK_EQ(info.file_id.secs, g["file_id"]["secs"].get<std::int32_t>());
  CHECK_EQ(info.file_id.usecs, g["file_id"]["usecs"].get<std::int32_t>());

  // channels
  CHECK_EQ(info.channels.size(), g["chs"].size());
  for (std::size_t i = 0; i < info.channels.size() && i < g["chs"].size(); ++i) {
    const auto& a = info.channels[i];
    const auto& b = g["chs"][i];
    CHECK_EQ(a.scanno, b["scanno"].get<std::int32_t>());
    CHECK_EQ(a.logno, b["logno"].get<std::int32_t>());
    CHECK_EQ(a.kind, b["kind"].get<std::int32_t>());
    CHECK_EQ(a.range, b["range"].get<double>());
    CHECK_EQ(a.cal, b["cal"].get<double>());
    CHECK_EQ(a.coil_type, b["coil_type"].get<std::int32_t>());
    CHECK_EQ(a.unit, b["unit"].get<std::int32_t>());
    CHECK_EQ(a.unit_mul, b["unit_mul"].get<std::int32_t>());
    CHECK_EQ(a.name, b["ch_name"].get<std::string>());
    CHECK_EQ(a.coord_frame, b["coord_frame"].get<std::int32_t>());
    bool loc_ok = true;
    for (int k = 0; k < 12; ++k) {
      const double w = b["loc"][k].is_null() ? NAN : b["loc"][k].get<double>();
      loc_ok = loc_ok && ((std::isnan(w) && std::isnan(a.loc[k])) || a.loc[k] == w);
    }
    CHECK(loc_ok);
    CHECK_EQ(h.channels[i].name, b["ch_name"].get<std::string>());
  }
  // bads: the file's list restricted to the channels, as MNE cleans it
  CHECK_EQ(h.bad_channels.size(), g["bads"].size());
  for (std::size_t i = 0; i < h.bad_channels.size() && i < g["bads"].size(); ++i) CHECK_EQ(h.bad_channels[i], g["bads"][i].get<std::string>());
  std::size_t flagged = 0;
  for (const Channel& ch : h.channels) flagged += ch.bad ? 1 : 0;
  CHECK_EQ(flagged, g["bads"].size());

  // transforms
  check_transform(info.dev_head_t, g["dev_head_t"], "dev_head_t");
  check_transform(info.ctf_head_t, g["ctf_head_t"], "ctf_head_t");

  // digitisation
  CHECK_EQ(info.has_dig, !g["dig"].is_null());
  if (info.has_dig && !g["dig"].is_null()) {
    CHECK_EQ(info.dig.size(), g["dig"].size());
    for (std::size_t i = 0; i < info.dig.size() && i < g["dig"].size(); ++i) {
      const auto& d = g["dig"][i];
      CHECK_EQ(info.dig[i].kind, d["kind"].get<std::int32_t>());
      CHECK_EQ(info.dig[i].ident, d["ident"].get<std::int32_t>());
      CHECK_EQ(info.dig[i].coord_frame, d["coord_frame"].get<std::int32_t>());
      for (int k = 0; k < 3; ++k) CHECK_EQ(info.dig[i].r[k], d["r"][k].get<double>());
    }
  }

  // projectors
  CHECK_EQ(info.projectors.size(), g["projs"].size());
  for (std::size_t i = 0; i < info.projectors.size() && i < g["projs"].size(); ++i) {
    const auto& p = info.projectors[i];
    const auto& w = g["projs"][i];
    CHECK_EQ(p.desc, w["desc"].get<std::string>());
    CHECK_EQ(p.kind, w["kind"].get<std::int32_t>());
    CHECK_EQ(p.active, w["active"].get<bool>());
    CHECK_EQ(p.nvec, w["nvec"].get<std::int32_t>());
    CHECK_EQ(p.channels.size(), w["col_names"].size());
    for (std::size_t k = 0; k < p.channels.size() && k < w["col_names"].size(); ++k) CHECK_EQ(p.channels[k], w["col_names"][k].get<std::string>());
    CHECK_EQ(p.vectors.size(), static_cast<std::size_t>(p.nvec) * p.channels.size());
    for (std::size_t k = 0; k < p.vectors.size() && k < p.channels.size(); ++k) CHECK(near(p.vectors[k], w["data"][0][k].get<double>()));
    CHECK_EQ(p.explained_var.has_value(), !w["explained_var"].is_null());
    if (p.explained_var && !w["explained_var"].is_null()) CHECK(near(*p.explained_var, w["explained_var"].get<double>()));
  }

  // compensation
  CHECK_EQ(info.compensations.size(), g["comps"].size());
  for (std::size_t i = 0; i < info.compensations.size() && i < g["comps"].size(); ++i) {
    const auto& cp = info.compensations[i];
    const auto& w = g["comps"][i];
    CHECK_EQ(cp.ctfkind, w["ctfkind"].get<std::int32_t>());
    CHECK_EQ(cp.grade, w["kind"].get<std::int32_t>());
    CHECK_EQ(cp.calibrated, w["save_calibrated"].get<bool>());
    CHECK_EQ(cp.row_names.size(), w["row_names"].size());
    CHECK_EQ(cp.col_names.size(), w["col_names"].size());
    CHECK_EQ(cp.data.size(), static_cast<std::size_t>(cp.nrow) * cp.ncol);
    for (std::size_t rr = 0; rr < w["data"].size(); ++rr)
      for (std::size_t cc = 0; cc < w["data"][rr].size(); ++cc) CHECK(near(cp.data[rr * cp.ncol + cc], w["data"][rr][cc].get<double>()));
  }

  // subject, device, helium
  CHECK_EQ(info.subject_info.has_value(), !g["subject_info"].is_null());
  if (info.subject_info && !g["subject_info"].is_null()) {
    const auto& s = *info.subject_info;
    const auto& w = g["subject_info"];
    CHECK_EQ(*s.id, w["id"].get<std::int32_t>());
    CHECK_EQ(*s.his_id, w["his_id"].get<std::string>());
    CHECK_EQ(*s.last_name, w["last_name"].get<std::string>());
    CHECK_EQ(*s.first_name, w["first_name"].get<std::string>());
    CHECK_EQ(*s.middle_name, w["middle_name"].get<std::string>());
    CHECK_EQ(*s.birthday, w["birthday"].get<std::string>());
    CHECK_EQ(*s.sex, w["sex"].get<std::int32_t>());
    CHECK_EQ(*s.hand, w["hand"].get<std::int32_t>());
    CHECK_EQ(*s.weight, w["weight"].get<double>());
    CHECK_EQ(*s.height, w["height"].get<double>());
    CHECK_EQ(h.info.at("subject_info.id"), std::string("7"));
  }
  CHECK_EQ(info.device_info.has_value(), !g["device_info"].is_null());
  if (info.device_info && !g["device_info"].is_null()) {
    CHECK_EQ(*info.device_info->type, g["device_info"]["type"].get<std::string>());
    CHECK_EQ(*info.device_info->model, g["device_info"]["model"].get<std::string>());
    CHECK_EQ(*info.device_info->serial, g["device_info"]["serial"].get<std::string>());
    CHECK_EQ(*info.device_info->site, g["device_info"]["site"].get<std::string>());
  }
  CHECK_EQ(info.helium_info.has_value(), !g["helium_info"].is_null());
  if (info.helium_info && !g["helium_info"].is_null()) {
    const auto& w = g["helium_info"];
    CHECK_EQ(*info.helium_info->he_level_raw, w["he_level_raw"].get<double>());
    CHECK_EQ(*info.helium_info->helium_level, w["helium_level"].get<double>());
    CHECK_EQ(*info.helium_info->orig_file_guid, w["orig_file_guid"].get<std::string>());
    CHECK(info.helium_info->meas_date.has_value());
  }

  // time axis, annotations, gaps
  CHECK_EQ(info.first_samp, c["first_samp"].get<std::int32_t>());
  CHECK_EQ(h.n_samples, c["n_times"].get<std::uint64_t>());
  CHECK_EQ(r.parts().size(), c["n_parts"].get<std::size_t>());
  CHECK_EQ(h.annotations.size(), c["annotations"].size());
  for (std::size_t i = 0; i < h.annotations.size() && i < c["annotations"].size(); ++i) {
    const auto& a = c["annotations"][i];
    CHECK_EQ(h.annotations[i].label, a["label"].get<std::string>());
    CHECK(near(h.annotations[i].onset, a["onset"].get<double>(), 1e-9, 1e-6));
    CHECK(near(h.annotations[i].duration, a["duration"].get<double>(), 1e-9, 1e-6));
  }
  CHECK_EQ(h.gaps.size(), c["gaps"].size());
  for (std::size_t i = 0; i < h.gaps.size() && i < c["gaps"].size(); ++i) {
    CHECK_EQ(h.gaps[i].first, c["gaps"][i]["first"].get<std::uint64_t>());
    CHECK_EQ(h.gaps[i].count, c["gaps"][i]["count"].get<std::uint64_t>());
  }
}

// Reads the whole recording in doubles in several block sizes and compares with the golden (channel after channel).
static void check_double_data(const fif::RawReader& r, const std::vector<double>& want, const char* what) {
  const RecordingHeader& h = r.header();
  const std::size_t nc = h.channels.size();
  const std::uint64_t n = h.n_samples;
  CHECK_EQ(want.size(), nc * n);
  for (std::uint64_t block : {std::uint64_t(1), std::uint64_t(37), std::uint64_t(256), n}) {
    std::vector<double> got(nc * n, -777.0), tmp;
    for (std::uint64_t s = 0; s < n; s += block) {
      const std::uint64_t m = std::min<std::uint64_t>(block, n - s);
      tmp.resize(nc * m);
      r.read(s, m, 0, nc, tmp.data());
      for (std::size_t c = 0; c < nc; ++c)
        for (std::uint64_t i = 0; i < m; ++i) got[c * n + s + i] = tmp[c * m + i];
    }
    bool ok = true;
    for (std::size_t i = 0; ok && i < got.size(); ++i) ok = near(got[i], want[i], 1e-14, 1e-30);
    if (!ok) std::printf("  %s: values differ from the reference (block %llu)\n", what, static_cast<unsigned long long>(block));
    CHECK(ok);
  }
}

int main() {
  std::ifstream gf(kGold + "fif_golden.json");
  const json g = json::parse(gf);

  for (const auto& [name, c] : g["cases"].items()) {
    auto r = fif::open_file(kData + c["file"].get<std::string>());
    check_info(*r, c, name);
    CHECK_EQ(r->header().format, std::string("fif"));
    const std::string data = c["data"].get<std::string>();
    if (data.size() > 4 && data.substr(data.size() - 4) == ".c64") continue;
    if (data.size() > 5 && data.substr(data.size() - 5) == ".c128") {
      // complex: pairs of doubles (real, imaginary) channel after channel; the real reads refuse
      CHECK(r->is_complex());
      const std::vector<double> want = load_f64(kGold + data);
      const std::size_t nc = r->header().channels.size();
      const std::uint64_t n = r->header().n_samples;
      CHECK_EQ(want.size(), nc * n * 2);
      std::vector<std::complex<double>> got(nc * n);
      for (std::uint64_t s = 0; s < n; s += 57) {
        const std::uint64_t m = std::min<std::uint64_t>(57, n - s);
        std::vector<std::complex<double>> tmp(nc * m);
        r->read_complex(s, m, 0, nc, tmp.data());
        for (std::size_t cc = 0; cc < nc; ++cc)
          for (std::uint64_t i = 0; i < m; ++i) got[cc * n + s + i] = tmp[cc * m + i];
      }
      bool ok = true;
      for (std::size_t i = 0; ok && i < got.size(); ++i) ok = near(got[i].real(), want[2 * i], 1e-14, 1e-30) && near(got[i].imag(), want[2 * i + 1], 1e-14, 1e-30);
      CHECK(ok);
      std::vector<double> junk(4);
      CHECK(rg::throws_code([&] { r->read(0, 1, 0, 1, junk.data()); }, code::refused));
      continue;
    }
    if (data.size() > 4 && data.substr(data.size() - 4) == ".f64") {
      CHECK(r->header().precision == Precision::float64);
      check_double_data(*r, load_f64(kGold + data), name.c_str());
      std::vector<float> junk(4);
      CHECK(rg::throws_code([&] { r->read(0, 1, 0, 1, junk.data()); }, code::refused));   // never narrowed silently
      continue;
    }
    CHECK(r->header().precision == Precision::float32);
    // an active projector is not applied by this reader (MNE applies it to the samples it reads): the header lists it
    if (c.contains("data_has_projector") && c["data_has_projector"].get<bool>()) {
      CHECK(r->info().projectors.size() == 1 && r->info().projectors[0].active);
      continue;
    }
    rg::check_against(*r, rg::load_f32(kGold + data), name.c_str(), 2e-6, 1e-30);
  }

  // the golden's header details that differ by case
  {
    auto r = fif::open_file(kData + "fif_basic_raw.fif");
    const RecordingHeader& h = r->header();
    CHECK_EQ(h.channels[0].kind, std::string("meg"));
    CHECK_EQ(h.channels[0].unit, std::string("T"));
    CHECK_EQ(h.channels[2].unit, std::string("T/m"));
    CHECK_EQ(h.channels[4].kind, std::string("eeg"));
    CHECK_EQ(h.channels[4].unit, std::string("V"));
    CHECK_EQ(h.channels[7].kind, std::string("eog"));
    CHECK_EQ(h.channels[9].kind, std::string("stim"));
    CHECK_EQ(h.channels[0].coil_type, std::string("VV_MAG_T3"));
    CHECK_EQ(h.channels[0].gain, 2.0e-3f * 1.0f);   // cal x range, as 32-bit floats
    CHECK(h.channels[0].location.has_value());
    CHECK(h.channels[0].orientation.has_value());
    CHECK_EQ(h.position_unit, std::string("m"));
    CHECK_EQ(h.start_time, std::string("2026-01-02T03:04:08.178900Z"));   // 03:04:05.6789 plus the first sample (250 at 100 Hz = 2.5 s)
    CHECK_EQ(r->info().first_samp, 250);
    CHECK(r->info().subject_info.has_value());
    CHECK_EQ(*r->info().subject_info->birthday, std::string("1980-02-29"));
  }
  // the grade in effect, and the compensation matrix, on the CTF-style file
  {
    auto r = fif::open_file(kData + "fif_comp_raw.fif");
    CHECK_EQ(r->info().compensation_grade, 3);
    CHECK_EQ(r->info().compensations[0].grade, 3);
    CHECK_EQ(r->header().channels[2].kind, std::string("meg_ref"));
    CHECK_EQ(r->header().channels[0].coil_type, std::string("CTF_GRAD"));
  }
  // split file: three parts as one axis, each part's range; a read across the joints equals one made part by part
  {
    auto r = fif::open_file(kData + "fif_split_raw.fif");
    CHECK_EQ(r->parts().size(), 3u);
    CHECK_EQ(r->parts()[0].offset, std::uint64_t(0));
    CHECK_EQ(r->parts()[1].name, std::string("fif_split_raw-1.fif"));
    CHECK_EQ(r->parts()[1].offset, r->parts()[0].n_samples);
    CHECK_EQ(r->parts()[2].offset, r->parts()[0].n_samples + r->parts()[1].n_samples);
    CHECK_EQ(r->parts()[0].n_samples + r->parts()[1].n_samples + r->parts()[2].n_samples, r->header().n_samples);
    // the same file through a memory resolver
    auto mem = [](const std::string& name) -> std::shared_ptr<const ByteSource> {
      std::ifstream f(kData + name, std::ios::binary);
      if (!f) return nullptr;
      std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      return std::make_shared<MemorySource>(s);
    };
    auto r2 = fif::open(mem("fif_split_raw.fif"), mem);
    CHECK_EQ(r2->header().n_samples, r->header().n_samples);
    // the first part alone, with no way to find the others: refused, naming the missing part
    CHECK(rg::throws_code([&] { fif::open(mem("fif_split_raw.fif")); }, code::io));
    try { fif::open(mem("fif_split_raw.fif")); } catch (const nxr::io::Error& e) { CHECK(std::string(e.what()).find("fif_split_raw-1.fif") != std::string::npos); }
  }
  // skips: the gap reads as zeros
  {
    auto r = fif::open_file(kData + "fif_skip_raw.fif");
    CHECK_EQ(r->header().gaps.size(), 1u);
    std::vector<float> v(300);
    r->read(150, 100, 0, 3, v.data());
    bool zeros = true;
    for (std::size_t i = 50; i < 100; ++i) zeros = zeros && v[i] == 0.0f && v[100 + i] == 0.0f && v[200 + i] == 0.0f;
    CHECK(zeros);
  }
  // long names (extended channel info) and the bads named by them
  {
    auto r = fif::open_file(kData + "fif_longnames_raw.fif");
    CHECK_EQ(r->header().channels[0].name, std::string("a_rather_long_channel_name_0001"));
    CHECK(r->header().channels[0].bad);
  }
  // the tag layer
  {
    fif::File f(std::make_shared<FileSource>(kData + "fif_basic_raw.fif"));
    CHECK(!fif::find_blocks(f.tree(), fif::block::kMeas).empty());
    CHECK(!fif::find_blocks(f.tree(), fif::block::kRawData).empty());
    CHECK(!fif::find_blocks(f.tree(), fif::block::kMneAnnotations).empty());
    CHECK_EQ(f.directory()[0].kind, fif::tag::kFileId);
    const fif::Block& mi = *fif::find_blocks(f.tree(), fif::block::kMeasInfo)[0];
    const fif::TagHeader* sf = fif::find_tag(mi, fif::tag::kSfreq);
    CHECK(sf != nullptr);
    if (sf) CHECK_EQ(f.value(*sf).reals.at(0), 100.0);
  }
  // MaxShield (a raw block of the IAS kind) is refused unless allowed: patch the block kind in a copy
  {
    std::ifstream f(kData + "fif_short_raw.fif", std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int patched = 0;
    for (std::size_t i = 0; i + 20 <= bytes.size(); ++i) {   // a block-start tag (kind 104, int, size 4) holding 102 (raw data)
      const unsigned char* p = reinterpret_cast<const unsigned char*>(bytes.data()) + i;
      if (load_i32(p, false) == 104 && load_u32(p + 4, false) == 3 && load_i32(p + 8, false) == 4 && load_i32(p + 16, false) == 102) {
        store_u32(&bytes[i + 16], 119, false);
        ++patched;
      }
    }
    CHECK_EQ(patched, 1);
    auto src = std::make_shared<MemorySource>(bytes);
    CHECK(rg::throws_code([&] { fif::open(src); }, code::refused));
    fif::OpenOptions o;
    o.allow_maxshield = true;
    auto r = fif::open(src, nullptr, o);
    CHECK_EQ(r->header().n_samples, std::uint64_t(300));
  }
  // refusals, each with its reason
  auto refusal = [&](const char* file, const char* code_, const char* needle) {
    try {
      fif::open_file(kData + file);
      std::printf("  %s was not refused\n", file);
      CHECK(false);
    } catch (const nxr::io::Error& e) {
      CHECK(e.code() == code_);
      const bool found = std::string(e.what()).find(needle) != std::string::npos;
      if (!found) std::printf("  %s: message '%s' lacks '%s'\n", file, e.what(), needle);
      CHECK(found);
    }
  };
  refusal("fif_refuse_epo.fif", code::refused, "epochs");
  refusal("fif_refuse_ave.fif", code::refused, "evoked");
  refusal("fif_refuse_cov.fif", code::refused, "not a raw recording");
  refusal("fif_refuse_notfif.fif", code::format, "not a FIF file");
  refusal("fif_refuse_truncated_raw.fif", code::format, "truncated");
  refusal("fif_orphan_raw.fif", code::io, "fif_orphan_raw-1.fif");
  refusal("missing.fif", code::io, "");
  CHECK(rg::throws_code([&] { fif::open(std::make_shared<MemorySource>(std::string("short"))); }, code::format));
  return nxrtest::finish("formats_fif");
}
