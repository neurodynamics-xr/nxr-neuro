#include "nxr/io/codec.h"

#include <stdexcept>
#include <string>

#include <zstd.h>

namespace nxr::io {

std::vector<uint8_t> zstd_compress(const uint8_t* data, std::size_t nbytes, int level) {
  const std::size_t bound = ZSTD_compressBound(nbytes);
  std::vector<uint8_t> out(bound);
  const std::size_t n = ZSTD_compress(out.data(), bound, data, nbytes, level);
  if (ZSTD_isError(n)) {
    throw std::runtime_error(std::string("[nxr/io] zstd compress failed: ") + ZSTD_getErrorName(n));
  }
  out.resize(n);
  return out;
}

std::vector<uint8_t> zstd_decompress(const uint8_t* data, std::size_t nbytes,
                                     std::size_t expected_size) {
  std::size_t size = expected_size;
  if (size == 0) {
    const unsigned long long cs = ZSTD_getFrameContentSize(data, nbytes);
    if (cs == ZSTD_CONTENTSIZE_ERROR) {
      throw std::runtime_error("[nxr/io] zstd: input is not a valid zstd frame");
    }
    if (cs == ZSTD_CONTENTSIZE_UNKNOWN) {  // streamed frame: decode until the input ends
      std::vector<uint8_t> out;
      std::vector<uint8_t> tmp(ZSTD_DStreamOutSize());
      ZSTD_DCtx* ctx = ZSTD_createDCtx();
      ZSTD_inBuffer in{data, nbytes, 0};
      while (in.pos < in.size) {
        ZSTD_outBuffer ob{tmp.data(), tmp.size(), 0};
        const std::size_t r = ZSTD_decompressStream(ctx, &ob, &in);
        if (ZSTD_isError(r)) {
          ZSTD_freeDCtx(ctx);
          throw std::runtime_error(std::string("[nxr/io] zstd decompress failed: ") + ZSTD_getErrorName(r));
        }
        out.insert(out.end(), tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(ob.pos));
      }
      ZSTD_freeDCtx(ctx);
      return out;
    }
    size = static_cast<std::size_t>(cs);
  }
  std::vector<uint8_t> out(size);
  const std::size_t n = ZSTD_decompress(out.data(), size, data, nbytes);
  if (ZSTD_isError(n)) {
    throw std::runtime_error(std::string("[nxr/io] zstd decompress failed: ") + ZSTD_getErrorName(n));
  }
  out.resize(n);
  return out;
}

}  // namespace nxr::io
