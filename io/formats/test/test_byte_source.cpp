#include <cstdio>
#include <functional>
#include <string>

#include "check.h"
#include "nxr/formats/byte_source.h"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"

using namespace nxr::formats;

static bool throws_code(const std::function<void()>& f, const char* code) {
  try { f(); } catch (const nxr::io::Error& e) { return e.code() == code; }
  return false;
}

int main() {
  // memory source
  MemorySource m(std::string("abcdefgh"));
  CHECK_EQ(m.size(), 8u);
  CHECK_EQ(m.string(2, 3), std::string("cde"));
  CHECK(throws_code([&] { m.bytes(6, 3); }, code::format));
  CHECK(m.bytes(8, 0).empty());

  // file sink then file source: the bytes come back
  const std::string path = std::string(NXR_FORMATS_WORK_DIR) + "/formats_byte_source.tmp";
  {
    FileSink s(path);
    s.write(std::string("hello "));
    s.write(std::string("world"));
    CHECK_EQ(s.position(), 11u);
    s.close();
  }
  {
    FileSource f(path);
    CHECK_EQ(f.size(), 11u);
    CHECK_EQ(f.string(0, 11), std::string("hello world"));
    CHECK_EQ(f.string(6, 5), std::string("world"));
    CHECK(throws_code([&] { f.bytes(9, 5); }, code::format));
  }
  std::remove(path.c_str());
  CHECK(throws_code([&] { FileSource f("/nonexistent/folder/file"); }, code::io));

  MemorySink k;
  k.write("ab", 2);
  CHECK_EQ(k.str(), std::string("ab"));

  // endian
  unsigned char b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  CHECK_EQ(load_u16(b, true), 0x0201u);
  CHECK_EQ(load_u16(b, false), 0x0102u);
  CHECK_EQ(load_u32(b, true), 0x04030201u);
  CHECK_EQ(load_u64(b, false), 0x0102030405060708ull);
  unsigned char neg[3] = {0xFF, 0xFF, 0xFF};
  CHECK_EQ(load_i24(neg, true), -1);
  unsigned char out[8];
  store_f64(out, -2.5, false);
  CHECK_EQ(load_f64(out, false), -2.5);
  store_f32(out, 1.25f, true);
  CHECK_EQ(load_f32(out, true), 1.25f);
  store_u32(out, 0xAABBCCDDu, false);
  CHECK_EQ(out[0], 0xAAu);
  return nxrtest::finish("formats_byte_source");
}
