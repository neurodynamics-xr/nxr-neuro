// ZIP archives (nxr/formats/zip.h) on archives built here, byte by byte: stored and deflate entries read back in whole and in small blocks, the CRC-32 and size checks, ZIP64
// sizes, an archive comment, and the refusals (not a ZIP, truncated, encrypted, an unsupported method). Nothing here is a real file.
#include <zlib.h>

#include <cstring>

#include "check.h"
#include "nxr/formats/deflate.h"
#include "nxr/formats/errors.h"
#include "nxr/formats/zip.h"

using namespace nxr::formats;
using Bytes = std::vector<unsigned char>;

static void p16(Bytes& v, unsigned x) { v.push_back(static_cast<unsigned char>(x)); v.push_back(static_cast<unsigned char>(x >> 8)); }
static void p32(Bytes& v, std::uint32_t x) { p16(v, x & 0xFFFF); p16(v, x >> 16); }
static void p64(Bytes& v, std::uint64_t x) { p32(v, static_cast<std::uint32_t>(x)); p32(v, static_cast<std::uint32_t>(x >> 32)); }

static std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code(); } catch (const std::exception&) { return "other"; }
  return "";
}

static std::shared_ptr<const ByteSource> mem(const Bytes& b) { return std::make_shared<MemorySource>(b); }

struct Item { std::string name; Bytes data; int method = 0; unsigned flags = 0; bool zip64 = false; };

// an archive written by hand: local headers, central directory, end record (with a comment); `deflate` entries use raw deflate (zlib's stream without its wrapper)
static Bytes archive(const std::vector<Item>& items, bool zip64_end = false) {
  Bytes out, central;
  for (const auto& it : items) {
    Bytes stored = it.data;
    if (it.method == 8) {
      const auto z = deflate_bytes(it.data.data(), it.data.size(), DeflateWrapper::zlib);
      stored.assign(z.begin() + 2, z.end() - 4);
    }
    const std::uint32_t crc = static_cast<std::uint32_t>(crc32(crc32(0L, Z_NULL, 0), it.data.data(), static_cast<uInt>(it.data.size())));
    const std::uint32_t at = static_cast<std::uint32_t>(out.size());
    p32(out, 0x04034b50); p16(out, 20); p16(out, it.flags); p16(out, it.method); p16(out, 0); p16(out, 0x21);
    p32(out, crc); p32(out, static_cast<std::uint32_t>(stored.size())); p32(out, static_cast<std::uint32_t>(it.data.size()));
    p16(out, static_cast<unsigned>(it.name.size())); p16(out, 0);
    out.insert(out.end(), it.name.begin(), it.name.end());
    out.insert(out.end(), stored.begin(), stored.end());
    p32(central, 0x02014b50); p16(central, 45); p16(central, 45); p16(central, it.flags); p16(central, it.method); p16(central, 0); p16(central, 0x21);
    p32(central, crc);
    Bytes extra;
    if (it.zip64) { p16(extra, 1); p16(extra, 16); p64(extra, it.data.size()); p64(extra, stored.size()); }
    p32(central, it.zip64 ? 0xFFFFFFFFu : static_cast<std::uint32_t>(stored.size()));
    p32(central, it.zip64 ? 0xFFFFFFFFu : static_cast<std::uint32_t>(it.data.size()));
    p16(central, static_cast<unsigned>(it.name.size())); p16(central, static_cast<unsigned>(extra.size())); p16(central, 0); p16(central, 0); p16(central, 0); p32(central, 0);
    p32(central, at);
    central.insert(central.end(), it.name.begin(), it.name.end());
    central.insert(central.end(), extra.begin(), extra.end());
  }
  const std::uint32_t cd_at = static_cast<std::uint32_t>(out.size());
  out.insert(out.end(), central.begin(), central.end());
  const std::string comment = "an archive comment";
  if (zip64_end) {
    const std::uint64_t rec_at = out.size();
    p32(out, 0x06064b50); p64(out, 44); p16(out, 45); p16(out, 45); p32(out, 0); p32(out, 0); p64(out, items.size()); p64(out, items.size()); p64(out, central.size()); p64(out, cd_at);
    p32(out, 0x07064b50); p32(out, 0); p64(out, rec_at); p32(out, 1);
    p32(out, 0x06054b50); p16(out, 0xFFFF); p16(out, 0xFFFF); p16(out, 0xFFFF); p16(out, 0xFFFF); p32(out, 0xFFFFFFFFu); p32(out, 0xFFFFFFFFu); p16(out, 0);
  } else {
    p32(out, 0x06054b50); p16(out, 0); p16(out, 0); p16(out, static_cast<unsigned>(items.size())); p16(out, static_cast<unsigned>(items.size()));
    p32(out, static_cast<std::uint32_t>(central.size())); p32(out, cd_at); p16(out, static_cast<unsigned>(comment.size()));
    out.insert(out.end(), comment.begin(), comment.end());
  }
  return out;
}

static Bytes pattern(std::size_t n) {
  Bytes b(n);
  for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<unsigned char>((i * 7 + (i >> 8)) & 0xFF);
  return b;
}

int main() {
  const Bytes small = pattern(1000), big = pattern(300000);
  std::vector<Item> items = {{"a.bin", small, 0}, {"dir/", {}, 0}, {"dir/b.bin", big, 8}, {"c.bin", big, 0}, {"empty", {}, 0}};
  {
    const zip::Reader z = zip::Reader::open(mem(archive(items)));
    CHECK_EQ(z.entries().size(), static_cast<std::size_t>(5));
    CHECK(z.find("dir/") && z.find("dir/")->is_directory());
    CHECK(!z.find("nope"));
    for (const auto& it : items) {
      const zip::Entry* e = z.find(it.name);
      CHECK(e != nullptr);
      if (!e) continue;
      CHECK(z.read(*e) == it.data);
      Bytes via;
      std::size_t most = 0;
      z.stream(*e, [&](const unsigned char* d, std::size_t n) { via.insert(via.end(), d, d + n); most = std::max(most, n); }, 4096);   // in blocks, never whole
      CHECK(via == it.data);
      CHECK(most <= 4096);
    }
    CHECK_EQ(z.find("dir/b.bin")->method, static_cast<std::uint16_t>(8));
    CHECK(z.find("dir/b.bin")->compressed_size < big.size());
    CHECK_EQ(code_of([&] { z.read(*z.find("c.bin"), 1000); }), std::string(code::format));       // beyond the limit
  }
  // ZIP64: sizes in the extra field, the end record behind a ZIP64 record and locator
  {
    std::vector<Item> z64 = {{"x", small, 0, 0, true}, {"y", big, 8, 0, true}};
    const zip::Reader z = zip::Reader::open(mem(archive(z64, true)));
    CHECK_EQ(z.entries().size(), static_cast<std::size_t>(2));
    CHECK(z.read(*z.find("x")) == small && z.read(*z.find("y")) == big);
  }
  // the bytes are checked as they are read: a flipped byte of a stored entry fails its CRC
  {
    Bytes a = archive({{"s", small, 0}});
    a[30 + 1 + 500] ^= 0x55;
    const zip::Reader z = zip::Reader::open(mem(a));
    CHECK_EQ(code_of([&] { z.read(*z.find("s")); }), std::string(code::format));
    Bytes d = archive({{"s", big, 8}});
    d[30 + 1 + 100] ^= 0xFF;
    const zip::Reader zd = zip::Reader::open(mem(d));
    CHECK_EQ(code_of([&] { zd.read(*zd.find("s")); }), std::string(code::format));
  }
  // refusals
  CHECK_EQ(code_of([&] { zip::Reader::open(mem(Bytes(100, 'x'))); }), std::string(code::format));
  CHECK_EQ(code_of([&] { zip::Reader::open(mem(Bytes(5, 0))); }), std::string(code::format));
  {
    Bytes a = archive({{"s", small, 0}});
    a.resize(a.size() - 30);                                                                      // the end record cut off
    CHECK_EQ(code_of([&] { zip::Reader::open(mem(a)); }), std::string(code::format));
  }
  CHECK_EQ(code_of([&] { zip::Reader::open(mem(archive({{"s", small, 0, 1}}))); }), std::string(code::refused));      // encrypted
  CHECK_EQ(code_of([&] { zip::Reader::open(mem(archive({{"s", small, 12}}))); }), std::string(code::refused));        // bzip2
  // the writer: what it writes reads back (stored), in the order given
  {
    MemorySink sink;
    zip::write(sink, {{"one", small}, {"two/three", big}});
    const zip::Reader z = zip::Reader::open(mem(sink.data()));
    CHECK_EQ(z.entries().size(), static_cast<std::size_t>(2));
    CHECK(z.entries()[1].name == "two/three" && z.entries()[1].method == 0);
    CHECK(z.read(z.entries()[0]) == small && z.read(z.entries()[1]) == big);
  }
  return nxrtest::finish("formats_zip");
}
