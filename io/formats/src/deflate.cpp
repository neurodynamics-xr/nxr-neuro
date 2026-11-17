#include "nxr/formats/deflate.h"

#include <zlib.h>

#include <cstring>

#include "nxr/formats/errors.h"

namespace nxr::formats {

std::vector<unsigned char> inflate_bytes(const unsigned char* data, std::size_t n, DeflateWrapper w, std::size_t size_hint, std::size_t max_size) {
  if (w == DeflateWrapper::detect) w = (n >= 2 && data[0] == 0x1f && data[1] == 0x8b) ? DeflateWrapper::gzip : DeflateWrapper::zlib;
  z_stream z;
  std::memset(&z, 0, sizeof z);
  if (inflateInit2(&z, w == DeflateWrapper::gzip ? 16 + MAX_WBITS : MAX_WBITS) != Z_OK) throw format_error("zlib: cannot start inflating");
  std::vector<unsigned char> out(size_hint ? size_hint : std::max<std::size_t>(n * 4, 4096));
  z.next_in = const_cast<unsigned char*>(data);
  z.avail_in = static_cast<uInt>(n);
  std::size_t have = 0;
  int rc = Z_OK;
  while (rc != Z_STREAM_END) {
    if (have == out.size()) {
      if (out.size() >= max_size) { inflateEnd(&z); throw format_error("compressed data inflate beyond the allowed size"); }
      out.resize(std::min(max_size, out.size() * 2));
    }
    z.next_out = out.data() + have;
    z.avail_out = static_cast<uInt>(std::min<std::size_t>(out.size() - have, 1u << 30));
    const uInt before = z.avail_out;
    rc = inflate(&z, Z_NO_FLUSH);
    have += before - z.avail_out;
    if (rc == Z_BUF_ERROR && z.avail_in == 0) { inflateEnd(&z); throw format_error("compressed data are truncated"); }
    if (rc != Z_OK && rc != Z_STREAM_END && rc != Z_BUF_ERROR) { inflateEnd(&z); throw format_error(std::string("compressed data are corrupt (zlib: ") + (z.msg ? z.msg : "error") + ")"); }
  }
  inflateEnd(&z);
  out.resize(have);
  return out;
}

std::vector<unsigned char> deflate_bytes(const unsigned char* data, std::size_t n, DeflateWrapper w, int level) {
  z_stream z;
  std::memset(&z, 0, sizeof z);
  const int bits = (w == DeflateWrapper::gzip) ? 16 + MAX_WBITS : MAX_WBITS;
  if (deflateInit2(&z, level, Z_DEFLATED, bits, 8, Z_DEFAULT_STRATEGY) != Z_OK) throw format_error("zlib: cannot start deflating");
  std::vector<unsigned char> out(deflateBound(&z, static_cast<uLong>(n)) + 64);
  z.next_in = const_cast<unsigned char*>(data);
  z.avail_in = static_cast<uInt>(n);
  z.next_out = out.data();
  z.avail_out = static_cast<uInt>(out.size());
  const int rc = deflate(&z, Z_FINISH);
  if (rc != Z_STREAM_END) { deflateEnd(&z); throw format_error("zlib: deflate failed"); }
  out.resize(z.total_out);
  deflateEnd(&z);
  return out;
}

}  // namespace nxr::formats
