// The BST-BIN binary of a Brainstorm raw link (BstReader): a header, then epochs of [n_channels x epochsize] float32, channel after channel, each channel's block
// `epochsize` samples apart in EVERY epoch, the last one included (Brainstorm's in_fread_bst / out_fwrite_bst): the file ends after the last channel's declared samples,
// or holds the last epoch whole. A last epoch re-laid with its channels their declared length apart (`cut`) is read too. A file CUT SHORT keeps the samples every channel
// has in full (read in Brainstorm's layout), one under half is refused, and a size that fits no layout is refused. Synthetic (issue #276's file among them).
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

#include "check.h"
#include "nxr/bst/mat.h"
#include "nxr/bst/session.h"
#include "nxr/io.h"

using namespace nxr::bst;
namespace fs = std::filesystem;

namespace {
Value num(double x) {
  Value v;
  v.cls = Class::Double;
  v.dims = {1, 1};
  v.re.resize(sizeof(double));
  std::memcpy(v.re.data(), &x, sizeof(double));
  return v;
}
Value sfile(std::int64_t nchan, std::int64_t epoch, std::int64_t hdr, std::int64_t nsamples) {
  Value h;
  h.cls = Class::Struct;
  h.dims = {1, 1};
  h.fields = {"nchannels", "epochsize", "hdrsize", "nsamples"};
  h.elems = {num(static_cast<double>(nchan)), num(static_cast<double>(epoch)), num(static_cast<double>(hdr)), num(static_cast<double>(nsamples))};
  Value s;
  s.cls = Class::Struct;
  s.dims = {1, 1};
  s.fields = {"header"};
  s.elems = {h};
  return s;
}
float truth(std::int64_t c, std::int64_t t) { return static_cast<float>(c * 100000 + t); }
constexpr float kPad = -7.0f;   // what the padding of a last epoch holds (no reader may return it)
enum class Layout { stride, whole, cut };
// the file as Brainstorm writes it (`stride`: every block `epoch` apart, the last channel ending at its declared samples), with the last epoch whole (`whole`) or cut to its
// declared samples (`cut`), then cut after `keep_floats` floats of data (-1: as laid out) or extended by `extra` floats
void write_bin(const fs::path& p, std::int64_t nchan, std::int64_t epoch, std::int64_t hdr, std::int64_t nsamples, Layout layout, std::int64_t keep_floats = -1, std::int64_t extra = 0) {
  std::vector<float> data;
  for (std::int64_t e = 0; e * epoch < nsamples; ++e) {
    const std::int64_t len = std::min(epoch, nsamples - e * epoch);
    for (std::int64_t c = 0; c < nchan; ++c) {
      for (std::int64_t t = 0; t < len; ++t) data.push_back(truth(c, e * epoch + t));
      const bool pad = layout == Layout::whole || (layout == Layout::stride && c + 1 < nchan);
      if (pad) for (std::int64_t t = len; t < epoch; ++t) data.push_back(kPad);
    }
  }
  if (keep_floats >= 0) data.resize(static_cast<std::size_t>(keep_floats));
  for (std::int64_t k = 0; k < extra; ++k) data.push_back(kPad);
  std::ofstream f(p, std::ios::binary);
  f << std::string(static_cast<std::size_t>(hdr), 'h');
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 4));
}
bool exact(BstReader& r, std::int64_t nchan, std::int64_t n) {
  std::vector<float> out;
  r.read(0, n, 0, nchan, out);
  for (std::int64_t c = 0; c < nchan; ++c) for (std::int64_t t = 0; t < n; ++t) if (out[c * n + t] != truth(c, t)) return false;
  return true;
}
std::string refusal(const std::function<void()>& f) {
  try { f(); } catch (const nxr::io::Error& e) { return e.what(); }
  return "";
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path dir = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_bst_binary_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{dir};
  {   // issue #276: 2 channels, epochsize 4, nsamples 6; after the header 1 2 3 4 | 5 6 7 8 | 9 10 x x | 11 12, as Brainstorm lays it out: channel 1 reads 5 6 7 8 11 12
    const std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, kPad, kPad, 11, 12};
    {
      std::ofstream f(dir / "issue276.bin", std::ios::binary);
      f << std::string(8, 'h');
      f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * 4));
    }
    BstReader r(sfile(2, 4, 8, 6), dir / "issue276.bin");
    CHECK_EQ(r.layout(), std::string("epoch stride"));
    CHECK(!r.truncated());
    std::vector<float> out;
    r.read(0, 6, 0, 2, out);
    CHECK(out == std::vector<float>({1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 11, 12}));
    r.read(4, 2, 1, 2, out);   // channel 1 alone, in the last epoch
    CHECK(out == std::vector<float>({11, 12}));
  }
  const std::int64_t nc = 5, ep = 40, hdr = 16;
  {   // Brainstorm's layout, not a whole number of epochs (130 = 3 x 40 + 10): every sample exact, whole and across the last epoch's edge; a channel subset too
    write_bin(dir / "stride.bin", nc, ep, hdr, 130, Layout::stride);
    BstReader r(sfile(nc, ep, hdr, 130), dir / "stride.bin");
    CHECK_EQ(r.layout(), std::string("epoch stride"));
    CHECK_EQ(r.n_samples(), std::int64_t(130));
    CHECK(!r.truncated());
    CHECK(exact(r, nc, 130));
    std::vector<float> out;
    r.read(115, 15, 2, 4, out);
    CHECK_EQ(out.size(), std::size_t(30));
    CHECK(out[0] == truth(2, 115) && out[14] == truth(2, 129) && out[15] == truth(3, 115) && out[29] == truth(3, 129));
  }
  {   // the last epoch written whole (a whole number of epochs in the file): the padding is never read
    write_bin(dir / "whole.bin", nc, ep, hdr, 130, Layout::whole);
    BstReader r(sfile(nc, ep, hdr, 130), dir / "whole.bin");
    CHECK_EQ(r.layout(), std::string("whole epochs"));
    CHECK(!r.truncated());
    CHECK(exact(r, nc, 130));
  }
  {   // the declared samples fill whole epochs: the three layouts are one
    write_bin(dir / "full.bin", nc, ep, hdr, 120, Layout::stride);
    BstReader r(sfile(nc, ep, hdr, 120), dir / "full.bin");
    CHECK(!r.truncated());
    CHECK(exact(r, nc, 120));
  }
  {   // the last epoch re-laid with its channels their declared length apart (what an earlier relay writes): still read exactly
    write_bin(dir / "cut.bin", nc, ep, hdr, 130, Layout::cut);
    BstReader r(sfile(nc, ep, hdr, 130), dir / "cut.bin");
    CHECK_EQ(r.layout(), std::string("cut"));
    CHECK(exact(r, nc, 130));
  }
  {   // cut short in Brainstorm's layout: the last channel of the last epoch lacks its final 4 samples -> 126 samples are whole, and it is reported
    const std::int64_t floats = 3 * nc * ep + (nc - 1) * ep + 10;
    write_bin(dir / "short.bin", nc, ep, hdr, 130, Layout::stride, floats - 4);
    BstReader r(sfile(nc, ep, hdr, 130), dir / "short.bin");
    CHECK(r.truncated());
    CHECK_EQ(r.declared_samples(), std::int64_t(130));
    CHECK_EQ(r.n_samples(), std::int64_t(126));
    CHECK(exact(r, nc, 126));
  }
  {   // the cohort's case: one second (half an epoch of one channel) missing from full epochs
    write_bin(dir / "c.bin", nc, ep, hdr, 120, Layout::stride, 3 * nc * ep - ep / 2);
    BstReader r(sfile(nc, ep, hdr, 120), dir / "c.bin");
    CHECK_EQ(r.n_samples(), std::int64_t(2 * ep + ep / 2));
    CHECK(exact(r, nc, 100));
  }
  {   // under half the declared samples: refused, naming the file
    write_bin(dir / "d.bin", nc, ep, hdr, 130, Layout::stride, 1 * nc * ep);
    const std::string what = refusal([&] { BstReader r(sfile(nc, ep, hdr, 130), dir / "d.bin"); });
    CHECK(what.find("d.bin") != std::string::npos && what.find("refused") != std::string::npos);
  }
  {   // a size that fits no layout (longer than Brainstorm's, shorter than the whole last epoch; longer than the whole last epoch): refused, naming the file and the sizes
    write_bin(dir / "e.bin", nc, ep, hdr, 130, Layout::stride, -1, 3);
    const std::string what = refusal([&] { BstReader r(sfile(nc, ep, hdr, 130), dir / "e.bin"); });
    CHECK(what.find("e.bin") != std::string::npos && what.find("fits neither layout") != std::string::npos);
    CHECK(what.find(std::to_string(hdr + 4 * (3 * nc * ep + (nc - 1) * ep + 10))) != std::string::npos);
    write_bin(dir / "f.bin", nc, ep, hdr, 130, Layout::whole, -1, 1);
    CHECK(refusal([&] { BstReader r(sfile(nc, ep, hdr, 130), dir / "f.bin"); }).find("fits neither layout") != std::string::npos);
    write_bin(dir / "g.bin", nc, ep, hdr, 120, Layout::stride, -1, 1);   // whole epochs with a float too many
    CHECK(refusal([&] { BstReader r(sfile(nc, ep, hdr, 120), dir / "g.bin"); }).find("fits neither layout") != std::string::npos);
  }
  return nxrtest::finish("bst_binary");
}
