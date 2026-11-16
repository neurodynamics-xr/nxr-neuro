// nxr-io — read a Zarr v3 array exactly as nxr writes it: one chunk or a regular grid, optionally `sharding_indexed`
// (an inner chunk is read through the shard's index, only those a region touches); codecs bytes · blosc · zstd · gzip ·
// crc32c; a missing chunk is the fill value. Values come back as raw little-endian bytes in C order.
#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

#include "nxr/io/store.h"
#include "nxr/io/types.h"

namespace nxr::io {

class ArrayReader {
 public:
  // Open the array whose `zarr.json` is at `<key>/zarr.json` in `store` (the store must outlive the reader). Throws ZarrIOError / ZarrFormatError.
  static ArrayReader open(const Store& store, const std::string& key);
  // The same, the reader keeping the store alive (a store made for this one array).
  static ArrayReader open(std::shared_ptr<const Store> store, const std::string& key);
  ArrayReader(ArrayReader&&) noexcept;
  ArrayReader& operator=(ArrayReader&&) noexcept;
  ~ArrayReader();

  const std::vector<std::int64_t>& shape() const;
  DType dtype() const;
  std::size_t itemsize() const;

  // The whole array.
  std::vector<std::uint8_t> read() const;
  // The region [start, stop) along every axis (clipped to the shape is an error: stop <= shape).
  std::vector<std::uint8_t> read(const std::vector<std::int64_t>& start,
                                 const std::vector<std::int64_t>& stop) const;

 private:
  struct Impl;
  explicit ArrayReader(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace nxr::io
