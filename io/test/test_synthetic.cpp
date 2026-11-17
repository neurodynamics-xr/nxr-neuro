// The synthetic recording (nxr/convert/synthetic.h): the same options give the same samples, a part reads as the whole does, the sine and the noise are what the options say,
// the options are refused when they make no recording, and the recording is stored as an imported one is, with its origin.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>

#include "check.h"
#include "nxr/convert/synthetic.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/task/task.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace cv = nxr::convert;

template <class F> static std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } catch (const std::exception&) { return "other"; } return ""; }

static std::vector<float> all_of(const cv::SyntheticRecording& o) {
  const auto r = cv::synthetic_reader(o);
  const auto& h = r->header();
  std::vector<float> x(h.channels.size() * h.n_samples);
  r->read(0, h.n_samples, 0, h.channels.size(), x.data());
  return x;
}

int main() {
  cv::SyntheticRecording o;
  o.n_channels = 3;
  o.sfreq = 200;
  o.duration = 20;
  o.seed = 7;
  o.frequencies = {10.0};
  const auto a = all_of(o), b = all_of(o);
  CHECK_EQ(a.size(), static_cast<std::size_t>(3 * 4000));
  CHECK(a == b);                                                   // the same options, the same samples
  cv::SyntheticRecording o2 = o;
  o2.seed = 8;
  CHECK(all_of(o2) != a);

  // a part reads as the whole does (channels 1-2, samples 1000-1499)
  {
    const auto r = cv::synthetic_reader(o);
    std::vector<float> part(2 * 500);
    r->read(1000, 500, 1, 3, part.data());
    for (int c = 0; c < 2; ++c) for (int j = 0; j < 500; ++j) CHECK(part[c * 500 + j] == a[(c + 1) * 4000 + 1000 + j]);
  }

  // the sine: its amplitude recovered at 10 Hz (a whole number of cycles in the record), and the noise alone has the standard deviation asked for
  {
    cv::SyntheticRecording s = o;
    s.noise = 0;
    const auto x = all_of(s);
    double sc = 0, ss = 0;
    for (int i = 0; i < 4000; ++i) { sc += x[i] * std::cos(2 * M_PI * 10 * i / 200.0); ss += x[i] * std::sin(2 * M_PI * 10 * i / 200.0); }
    const double amp = 2.0 * std::hypot(sc, ss) / 4000.0;
    CHECK(std::fabs(amp - s.amplitude) < 1e-3 * s.amplitude);
    cv::SyntheticRecording n = o;
    n.amplitude = 0;
    n.noise = 3e-6;
    const auto y = all_of(n);
    double m = 0, v = 0;
    for (float f : y) m += f;
    m /= static_cast<double>(y.size());
    for (float f : y) v += (f - m) * (f - m);
    v = std::sqrt(v / static_cast<double>(y.size()));
    CHECK(std::fabs(v - 3e-6) < 0.05 * 3e-6);
    CHECK(std::fabs(m) < 0.05 * 3e-6);
  }

  // refusals
  for (int what = 0; what < 6; ++what) {
    cv::SyntheticRecording bad = o;
    if (what == 0) bad.n_channels = 0;
    if (what == 1) bad.sfreq = 0;
    if (what == 2) bad.duration = -1;
    if (what == 3) bad.frequencies = {100.0};     // at the Nyquist frequency
    if (what == 4) bad.noise = -1;
    if (what == 5) { bad.sfreq = 1; bad.duration = 0.1; }
    CHECK_EQ(code_of([&] { cv::synthetic_reader(bad); }), std::string(code::invalid_argument));
  }

  // stored as an imported recording is
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_synthetic_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  {
    CreatePackageOptions po;
    po.home_dataset = "demo";
    DatasetWriter w = create_package(root / "p.nxr.zarr", "S", po);
    SubjectWriter sub = SubjectWriter::open(w, "S");
    o.name = "rest";
    const Json out = cv::create_synthetic(w, sub, o);
    CHECK_EQ(out["n_channels"].get<std::int64_t>(), static_cast<std::int64_t>(3));
    CHECK_EQ(out["n_samples"].get<std::int64_t>(), static_cast<std::int64_t>(4000));
    CHECK(out["origin"]["redistributable"].get<bool>());
  }
  SubjectPackage p = SubjectPackage::open(root / "p.nxr.zarr");
  CHECK(p.verify().ok);
  CHECK(items_without_origin(p.db()).empty());
  const task::Node rec = task::Subject::of(p).require({.cls = task::Cls::field, .name = "rest"}, "the recording");
  CHECK_EQ(rec.producer["synthetic"]["seed"].get<std::uint64_t>(), static_cast<std::uint64_t>(7));
  CHECK_EQ(rec.producer["origin"]["licence"].get<std::string>(), std::string("CC0-1.0"));
  const ArrayReader ar = ArrayReader::open(p.store(), *rec.path);
  const std::vector<std::uint8_t> raw = ar.read({0, 0}, {3, 4000});
  CHECK(std::memcmp(raw.data(), a.data(), raw.size()) == 0);          // the stored samples are the generator's
  fs::remove_all(root);
  return nxrtest::finish("synthetic");
}
