#include "nxr/formats/byte_source.h"

#include <cstring>

#include "nxr/formats/errors.h"

namespace nxr::formats {

std::vector<unsigned char> ByteSource::bytes(std::uint64_t offset, std::size_t n) const {
  std::vector<unsigned char> v(n);
  if (n) read(offset, n, v.data());
  return v;
}

std::string ByteSource::string(std::uint64_t offset, std::size_t n) const {
  std::string s(n, '\0');
  if (n) read(offset, n, &s[0]);
  return s;
}

FileSource::FileSource(const std::string& path) : path_(path) {
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_) throw io_error("cannot open " + path + " for reading");
  if (fseeko(f_, 0, SEEK_END) != 0) { std::fclose(f_); throw io_error("cannot seek in " + path); }
  const off_t end = ftello(f_);
  if (end < 0) { std::fclose(f_); throw io_error("cannot size " + path); }
  size_ = static_cast<std::uint64_t>(end);
}

FileSource::~FileSource() { if (f_) std::fclose(f_); }

void FileSource::read(std::uint64_t offset, std::size_t n, void* out) const {
  if (offset > size_ || n > size_ - offset)
    throw format_error(path_ + ": read of " + std::to_string(n) + " bytes at " + std::to_string(offset) + " runs past the end (" + std::to_string(size_) + " bytes): the file is truncated");
  if (n == 0) return;
  if (fseeko(f_, static_cast<off_t>(offset), SEEK_SET) != 0) throw io_error(path_ + ": seek failed");
  if (std::fread(out, 1, n, f_) != n) throw io_error(path_ + ": read failed");
}

void MemorySource::read(std::uint64_t offset, std::size_t n, void* out) const {
  if (offset > data_.size() || n > data_.size() - offset)
    throw format_error("read of " + std::to_string(n) + " bytes at " + std::to_string(offset) + " runs past the end (" + std::to_string(data_.size()) + " bytes): the data are truncated");
  if (n) std::memcpy(out, data_.data() + offset, n);
}

FileSink::FileSink(const std::string& path) : path_(path) {
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) throw io_error("cannot open " + path + " for writing");
}

FileSink::~FileSink() { if (f_) std::fclose(f_); }

void FileSink::write(const void* data, std::size_t n) {
  if (!f_) throw io_error(path_ + ": write after close");
  if (n && std::fwrite(data, 1, n, f_) != n) throw io_error(path_ + ": write failed");
  pos_ += n;
}

void FileSink::close() {
  if (!f_) return;
  const int r = std::fclose(f_);
  f_ = nullptr;
  if (r != 0) throw io_error(path_ + ": close failed");
}

void MemorySink::write(const void* data, std::size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  data_.insert(data_.end(), p, p + n);
}

}  // namespace nxr::formats
