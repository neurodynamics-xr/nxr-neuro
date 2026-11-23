// The BST-BIN binary of a Brainstorm raw link (BstReader): a header, then epochs of [n_channels x epoch length] float32, channel after channel. The LAST epoch is partial
// (nsamples - e*epochsize) and its channels are that length apart; a file CUT SHORT keeps the samples every channel has in full, and one under half is refused. Synthetic.
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

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
// the file as Brainstorm writes it, cut after `keep_floats` floats of data (-1: whole)
void write_bin(const fs::path& p, std::int64_t nchan, std::int64_t epoch, std::int64_t hdr, std::int64_t nsamples, std::int64_t keep_floats) {
  std::vector<float> data;
  for (std::int64_t e = 0; e * epoch < nsamples; ++e) {
    const std::int64_t len = std::min(epoch, nsamples - e * epoch);
    for (std::int64_t c = 0; c < nchan; ++c) for (std::int64_t t = 0; t < len; ++t) data.push_back(truth(c, e * epoch + t));
  }
  if (keep_floats >= 0) data.resize(static_cast<std::size_t>(keep_floats));
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
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path dir = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_bst_binary_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{dir};
  const std::int64_t nc = 5, ep = 40, hdr = 16;
  {   // (a) a partial last epoch (130 = 3 x 40 + 10) reads every sample exactly, whole and across the epoch edge; a channel subset too
    write_bin(dir / "a.bin", nc, ep, hdr, 130, -1);
    BstReader r(sfile(nc, ep, hdr, 130), dir / "a.bin");
    CHECK_EQ(r.n_samples(), std::int64_t(130));
    CHECK(!r.truncated());
    CHECK(exact(r, nc, 130));
    std::vector<float> out;
    r.read(115, 15, 2, 4, out);
    CHECK_EQ(out.size(), std::size_t(30));
    CHECK(out[0] == truth(2, 115) && out[14] == truth(2, 129) && out[15] == truth(3, 115) && out[29] == truth(3, 129));
  }
  {   // (b) cut short: the last channel of the last epoch lacks its final 4 samples -> 126 samples are whole, and it is reported
    const std::int64_t full_floats = 3 * nc * ep + nc * 10;
    write_bin(dir / "b.bin", nc, ep, hdr, 130, full_floats - 4);
    BstReader r(sfile(nc, ep, hdr, 130), dir / "b.bin");
    CHECK(r.truncated());
    CHECK_EQ(r.declared_samples(), std::int64_t(130));
    CHECK_EQ(r.n_samples(), std::int64_t(126));
    CHECK(exact(r, nc, 126));
  }
  {   // the cohort's case: one second (half an epoch of one channel) missing from full epochs
    write_bin(dir / "c.bin", nc, ep, hdr, 120, 3 * nc * ep - ep / 2);
    BstReader r(sfile(nc, ep, hdr, 120), dir / "c.bin");
    CHECK_EQ(r.n_samples(), std::int64_t(2 * ep + ep / 2));
    CHECK(exact(r, nc, 100));
  }
  {   // (c) under half the declared samples: refused, naming the file
    write_bin(dir / "d.bin", nc, ep, hdr, 130, 1 * nc * ep);
    std::string what;
    try { BstReader r(sfile(nc, ep, hdr, 130), dir / "d.bin"); } catch (const nxr::io::Error& e) { what = e.what(); }
    CHECK(what.find("d.bin") != std::string::npos && what.find("refused") != std::string::npos);
  }
  return nxrtest::finish("bst_binary");
}
