// ZIP archives: see nxr/formats/zip.h.
#include "nxr/formats/zip.h"

#include <zlib.h>

#include <algorithm>
#include <cstring>

namespace nxr::formats::zip {

namespace {
constexpr std::uint32_t kLocal = 0x04034b50, kCentral = 0x02014b50, kEnd = 0x06054b50, kEnd64 = 0x06064b50, kLocator64 = 0x07064b50;

std::uint16_t u16(const unsigned char* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }
std::uint32_t u32(const unsigned char* p) { return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) | (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24); }
std::uint64_t u64(const unsigned char* p) { return static_cast<std::uint64_t>(u32(p)) | (static_cast<std::uint64_t>(u32(p + 4)) << 32); }
void put16(std::vector<unsigned char>& v, std::uint16_t x) { v.push_back(static_cast<unsigned char>(x)); v.push_back(static_cast<unsigned char>(x >> 8)); }
void put32(std::vector<unsigned char>& v, std::uint32_t x) { put16(v, static_cast<std::uint16_t>(x)); put16(v, static_cast<std::uint16_t>(x >> 16)); }
}  // namespace

Reader Reader::open(std::shared_ptr<const ByteSource> source) {
  Reader r;
  r.src_ = std::move(source);
  const ByteSource& s = *r.src_;
  const std::uint64_t size = s.size();
  if (size < 22) throw format_error("zip: too small to be a ZIP archive");
  // the end-of-central-directory record: the last 22 bytes, or found backwards through the archive comment
  const std::uint64_t window = std::min<std::uint64_t>(size, 22 + 65535);
  const std::vector<unsigned char> tail = s.bytes(size - window, static_cast<std::size_t>(window));
  std::int64_t at = -1;
  for (std::int64_t i = static_cast<std::int64_t>(tail.size()) - 22; i >= 0; --i)
    if (u32(&tail[static_cast<std::size_t>(i)]) == kEnd) { at = i; break; }
  if (at < 0) throw format_error("zip: no end-of-central-directory record (not a ZIP archive)");
  const unsigned char* e = &tail[static_cast<std::size_t>(at)];
  std::uint64_t n_entries = u16(e + 10), cd_size = u32(e + 12), cd_offset = u32(e + 16);
  const std::uint16_t disk = u16(e + 4), cd_disk = u16(e + 6);
  const std::uint64_t end_pos = size - window + static_cast<std::uint64_t>(at);
  if (n_entries == 0xFFFF || cd_size == 0xFFFFFFFFu || cd_offset == 0xFFFFFFFFu) {   // ZIP64: the locator is the 20 bytes before the end record
    if (end_pos < 20) throw format_error("zip: a ZIP64 archive without its locator");
    const auto loc = s.bytes(end_pos - 20, 20);
    if (u32(loc.data()) != kLocator64) throw format_error("zip: a ZIP64 archive without its locator");
    const std::uint64_t rec_at = u64(loc.data() + 8);
    if (rec_at > size || size - rec_at < 56) throw format_error("zip: the ZIP64 end record is outside the file");
    const auto rec = s.bytes(rec_at, 56);
    if (u32(rec.data()) != kEnd64) throw format_error("zip: a bad ZIP64 end record");
    if (u32(rec.data() + 16) != 0 || u32(rec.data() + 20) != 0) throw refused("zip: a multi-disk archive");
    n_entries = u64(rec.data() + 32);
    cd_size = u64(rec.data() + 40);
    cd_offset = u64(rec.data() + 48);
  } else if (disk != 0 || cd_disk != 0) {
    throw refused("zip: a multi-disk archive");
  }
  if (cd_offset > size || cd_size > size - cd_offset) throw format_error("zip: the central directory is outside the file");
  if (n_entries > cd_size / 46) throw format_error("zip: more entries than the central directory can hold");
  const std::vector<unsigned char> cd = s.bytes(cd_offset, static_cast<std::size_t>(cd_size));
  std::size_t p = 0;
  for (std::uint64_t k = 0; k < n_entries; ++k) {
    if (cd.size() - p < 46 || u32(&cd[p]) != kCentral) throw format_error("zip: a bad central directory entry");
    const unsigned char* h = &cd[p];
    Entry en;
    const std::uint16_t flags = u16(h + 8);
    en.method = u16(h + 10);
    en.crc32 = u32(h + 16);
    en.compressed_size = u32(h + 20);
    en.size = u32(h + 24);
    const std::size_t nlen = u16(h + 28), xlen = u16(h + 30), clen = u16(h + 32);
    en.local_offset = u32(h + 42);
    if (cd.size() - p < 46 + nlen + xlen + clen) throw format_error("zip: a truncated central directory entry");
    en.name.assign(reinterpret_cast<const char*>(h + 46), nlen);
    // ZIP64 extended information: the fields whose 32-bit value is all ones, in this order
    const unsigned char* x = h + 46 + nlen;
    for (std::size_t q = 0; q + 4 <= xlen;) {
      const std::uint16_t id = u16(x + q), len = u16(x + q + 2);
      if (q + 4 + len > xlen) break;
      if (id == 0x0001) {
        std::size_t o = q + 4;
        auto take = [&](std::uint64_t& field, bool wanted) {
          if (!wanted) return;
          if (o + 8 > q + 4 + len) throw format_error("zip: a short ZIP64 extra field");
          field = u64(x + o);
          o += 8;
        };
        take(en.size, u32(h + 24) == 0xFFFFFFFFu);
        take(en.compressed_size, u32(h + 20) == 0xFFFFFFFFu);
        take(en.local_offset, u32(h + 42) == 0xFFFFFFFFu);
      }
      q += 4 + len;
    }
    if (flags & 1) throw refused("zip: '" + en.name + "' is encrypted");
    if (en.method != 0 && en.method != 8) throw refused("zip: '" + en.name + "' uses compression method " + std::to_string(en.method) + " (only stored and deflate are read)");
    r.entries_.push_back(std::move(en));
    p += 46 + nlen + xlen + clen;
  }
  return r;
}

const Entry* Reader::find(const std::string& name) const {
  for (const auto& e : entries_) if (e.name == name) return &e;
  return nullptr;
}

void Reader::stream(const Entry& e, const std::function<void(const unsigned char*, std::size_t)>& fn, std::size_t block) const {
  if (block == 0) block = std::size_t(1) << 20;
  const ByteSource& s = *src_;
  if (e.local_offset > s.size() || s.size() - e.local_offset < 30) throw format_error("zip: '" + e.name + "': its local header is outside the file");
  const auto lh = s.bytes(e.local_offset, 30);
  if (u32(lh.data()) != kLocal) throw format_error("zip: '" + e.name + "': a bad local header");
  const std::uint64_t data_at = e.local_offset + 30 + u16(lh.data() + 26) + u16(lh.data() + 28);
  if (data_at > s.size() || e.compressed_size > s.size() - data_at) throw format_error("zip: '" + e.name + "': its data run past the end of the file");
  uLong crc = crc32(0L, Z_NULL, 0);
  std::uint64_t produced = 0;
  auto emit = [&](const unsigned char* d, std::size_t n) {
    if (n == 0) return;
    crc = crc32(crc, d, static_cast<uInt>(n));
    produced += n;
    fn(d, n);
  };
  if (e.method == 0) {
    if (e.compressed_size != e.size) throw format_error("zip: '" + e.name + "': a stored entry whose sizes differ");
    std::vector<unsigned char> buf(static_cast<std::size_t>(std::min<std::uint64_t>(block, std::max<std::uint64_t>(e.size, 1))));
    for (std::uint64_t done = 0; done < e.size;) {
      const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(buf.size(), e.size - done));
      s.read(data_at + done, n, buf.data());
      emit(buf.data(), n);
      done += n;
    }
  } else {
    z_stream zs;
    std::memset(&zs, 0, sizeof zs);
    if (inflateInit2(&zs, -15) != Z_OK) throw format_error("zip: cannot start inflating");
    std::vector<unsigned char> in(static_cast<std::size_t>(std::min<std::uint64_t>(block, std::max<std::uint64_t>(e.compressed_size, 1)))), out(block);
    std::uint64_t fed = 0;
    int rc = Z_OK;
    try {
      while (rc != Z_STREAM_END) {
        if (zs.avail_in == 0) {
          if (fed == e.compressed_size) throw format_error("zip: '" + e.name + "': the deflate stream ends early");
          const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(in.size(), e.compressed_size - fed));
          s.read(data_at + fed, n, in.data());
          fed += n;
          zs.next_in = in.data();
          zs.avail_in = static_cast<uInt>(n);
        }
        zs.next_out = out.data();
        zs.avail_out = static_cast<uInt>(out.size());
        rc = inflate(&zs, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) throw format_error("zip: '" + e.name + "': a corrupt deflate stream");
        emit(out.data(), out.size() - zs.avail_out);
        if (produced > e.size) throw format_error("zip: '" + e.name + "': inflates beyond its recorded size");
      }
    } catch (...) {
      inflateEnd(&zs);
      throw;
    }
    inflateEnd(&zs);
  }
  if (produced != e.size) throw format_error("zip: '" + e.name + "': its size is " + std::to_string(produced) + " bytes, the directory says " + std::to_string(e.size));
  if (static_cast<std::uint32_t>(crc) != e.crc32) throw format_error("zip: '" + e.name + "': its CRC-32 differs from the recorded one");
}

std::vector<unsigned char> Reader::read(const Entry& e, std::uint64_t max_size) const {
  if (e.size > max_size) throw format_error("zip: '" + e.name + "' is " + std::to_string(e.size) + " bytes, beyond the limit " + std::to_string(max_size));
  std::vector<unsigned char> out;
  out.reserve(static_cast<std::size_t>(e.size));
  stream(e, [&](const unsigned char* d, std::size_t n) { out.insert(out.end(), d, d + n); });
  return out;
}

void write(ByteSink& sink, const std::vector<NewEntry>& entries) {
  constexpr std::uint64_t kLimit = 0xFFFFFFFFull;
  if (entries.size() >= 0xFFFF) throw refused("zip: too many entries for a ZIP archive without ZIP64");
  std::vector<unsigned char> central;
  std::uint64_t at = 0;
  for (const auto& en : entries) {
    if (en.data.size() >= kLimit || en.name.size() >= 0xFFFF) throw refused("zip: '" + en.name + "' is too large for a ZIP archive without ZIP64");
    const std::uint32_t crc = static_cast<std::uint32_t>(crc32(crc32(0L, Z_NULL, 0), en.data.data(), static_cast<uInt>(en.data.size())));
    std::vector<unsigned char> lh;
    put32(lh, kLocal);
    put16(lh, 10);            // version needed
    put16(lh, 0);             // flags
    put16(lh, 0);             // stored
    put16(lh, 0);             // time
    put16(lh, 0x21);          // date: 1980-01-01
    put32(lh, crc);
    put32(lh, static_cast<std::uint32_t>(en.data.size()));
    put32(lh, static_cast<std::uint32_t>(en.data.size()));
    put16(lh, static_cast<std::uint16_t>(en.name.size()));
    put16(lh, 0);
    lh.insert(lh.end(), en.name.begin(), en.name.end());
    put32(central, kCentral);
    put16(central, 10);       // version made by
    put16(central, 10);
    put16(central, 0);
    put16(central, 0);
    put16(central, 0);
    put16(central, 0x21);
    put32(central, crc);
    put32(central, static_cast<std::uint32_t>(en.data.size()));
    put32(central, static_cast<std::uint32_t>(en.data.size()));
    put16(central, static_cast<std::uint16_t>(en.name.size()));
    put16(central, 0);        // extra
    put16(central, 0);        // comment
    put16(central, 0);        // disk
    put16(central, 0);        // internal attributes
    put32(central, 0);        // external attributes
    if (at >= kLimit) throw refused("zip: the archive is too large without ZIP64");
    put32(central, static_cast<std::uint32_t>(at));
    central.insert(central.end(), en.name.begin(), en.name.end());
    sink.write(lh);
    sink.write(en.data);
    at += lh.size() + en.data.size();
  }
  std::vector<unsigned char> end;
  put32(end, kEnd);
  put16(end, 0);
  put16(end, 0);
  put16(end, static_cast<std::uint16_t>(entries.size()));
  put16(end, static_cast<std::uint16_t>(entries.size()));
  put32(end, static_cast<std::uint32_t>(central.size()));
  put32(end, static_cast<std::uint32_t>(at));
  put16(end, 0);
  sink.write(central);
  sink.write(end);
}

}  // namespace nxr::formats::zip
