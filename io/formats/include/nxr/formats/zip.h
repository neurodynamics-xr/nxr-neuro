// ZIP archives (PKWARE APPNOTE 6.3): the container of TRX tractography files. Pure: a ByteSource in, entries and their bytes out; a ByteSink for the writer.
//
// Reads what TRX needs and nothing more: stored (0) and deflate (8) entries, ZIP64 sizes and offsets and the ZIP64 end record, the central directory (the local headers are
// only used to find where an entry's data begin). Encrypted entries, other compression methods and multi-disk archives are refused. Every entry's CRC-32 is checked as its
// bytes are read. An entry is read in blocks (`stream`), so a large array is never held whole unless the caller keeps it. Deflate is zlib's raw mode (zlib is already linked
// for NIfTI and GIFTI): no zip library is needed. The writer stores entries uncompressed (what TRX asks for: its arrays are memory-mappable) and is for tests and exporters.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/formats/byte_source.h"
#include "nxr/formats/errors.h"

namespace nxr::formats::zip {

struct Entry {
  std::string name;                  // as stored, `/` separated; a folder's name ends with `/`
  std::uint16_t method = 0;          // 0 stored, 8 deflate
  std::uint32_t crc32 = 0;
  std::uint64_t compressed_size = 0, size = 0;
  std::uint64_t local_offset = 0;    // of the entry's local header
  bool is_directory() const { return !name.empty() && name.back() == '/'; }
};

class Reader {
 public:
  // Reads the central directory. Throws code::format when it is not a ZIP archive or is truncated, code::refused for what is listed above.
  static Reader open(std::shared_ptr<const ByteSource> source);
  const std::vector<Entry>& entries() const { return entries_; }
  const Entry* find(const std::string& name) const;   // null when absent
  // The entry's bytes, in blocks of at most `block`, in order; `fn(data, n)`. Throws code::format on a CRC or size mismatch or a corrupt stream.
  void stream(const Entry& e, const std::function<void(const unsigned char*, std::size_t)>& fn, std::size_t block = std::size_t(1) << 20) const;
  // The whole entry; refused (code::format) beyond `max_size` bytes (a bomb guard).
  std::vector<unsigned char> read(const Entry& e, std::uint64_t max_size = std::uint64_t(1) << 34) const;

 private:
  std::shared_ptr<const ByteSource> src_;
  std::vector<Entry> entries_;
};

struct NewEntry {
  std::string name;
  std::vector<unsigned char> data;
};
// A ZIP archive of the entries, stored uncompressed, in order. Refused when an entry or the archive is 4 GiB or more (no ZIP64 on writing).
void write(ByteSink& sink, const std::vector<NewEntry>& entries);

}  // namespace nxr::formats::zip
