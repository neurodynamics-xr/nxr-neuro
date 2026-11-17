// Where a pure format's bytes come from and go to. A reader asks a ByteSource for `n` bytes at an offset and never holds more than it asked
// for; a file source reads from disk, a memory source from a buffer (unit tests build a file in memory; a browser client hands one over).
// A writer gives its bytes to a ByteSink. No store, no database: a pure format sees bytes only.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace nxr::formats {

class ByteSource {
 public:
  virtual ~ByteSource() = default;
  virtual std::uint64_t size() const = 0;
  // Exactly `n` bytes at `offset` into `out`; throws (format_error) when the range runs past the end, (io) when the read fails.
  virtual void read(std::uint64_t offset, std::size_t n, void* out) const = 0;

  std::vector<unsigned char> bytes(std::uint64_t offset, std::size_t n) const;
  std::string string(std::uint64_t offset, std::size_t n) const;   // as read, no trimming
};

// A file on disk, opened for reading. Not thread-safe (one seek position): one reader per thread.
class FileSource final : public ByteSource {
 public:
  explicit FileSource(const std::string& path);   // io error when it cannot be opened
  ~FileSource() override;
  FileSource(const FileSource&) = delete;
  FileSource& operator=(const FileSource&) = delete;
  std::uint64_t size() const override { return size_; }
  void read(std::uint64_t offset, std::size_t n, void* out) const override;
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  std::FILE* f_ = nullptr;
  std::uint64_t size_ = 0;
};

class MemorySource final : public ByteSource {
 public:
  explicit MemorySource(std::vector<unsigned char> data) : data_(std::move(data)) {}
  explicit MemorySource(const std::string& data) : data_(data.begin(), data.end()) {}
  std::uint64_t size() const override { return data_.size(); }
  void read(std::uint64_t offset, std::size_t n, void* out) const override;
  const std::vector<unsigned char>& data() const { return data_; }

 private:
  std::vector<unsigned char> data_;
};

class ByteSink {
 public:
  virtual ~ByteSink() = default;
  virtual void write(const void* data, std::size_t n) = 0;
  virtual std::uint64_t position() const = 0;   // bytes written so far
  void write(const std::string& s) { write(s.data(), s.size()); }
  void write(const std::vector<unsigned char>& v) { write(v.data(), v.size()); }
};

class FileSink final : public ByteSink {
 public:
  explicit FileSink(const std::string& path);   // creates or truncates; io error when it cannot
  ~FileSink() override;
  FileSink(const FileSink&) = delete;
  FileSink& operator=(const FileSink&) = delete;
  using ByteSink::write;
  void write(const void* data, std::size_t n) override;
  std::uint64_t position() const override { return pos_; }
  void close();   // flushes; the destructor does it too but cannot report a failure

 private:
  std::string path_;
  std::FILE* f_ = nullptr;
  std::uint64_t pos_ = 0;
};

class MemorySink final : public ByteSink {
 public:
  using ByteSink::write;
  void write(const void* data, std::size_t n) override;
  std::uint64_t position() const override { return data_.size(); }
  const std::vector<unsigned char>& data() const { return data_; }
  std::string str() const { return std::string(data_.begin(), data_.end()); }

 private:
  std::vector<unsigned char> data_;
};

}  // namespace nxr::formats
