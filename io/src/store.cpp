#include "nxr/io/store.h"

#include <algorithm>
#include <fstream>

#include "nxr/io/database.h"
#include "nxr/io/errors.h"

namespace fs = std::filesystem;

namespace nxr::io {

std::string join_key(const std::string& prefix, const std::string& key) {
  if (prefix.empty()) return key;
  if (key.empty()) return prefix;
  return prefix + "/" + key;
}

void check_store_key(const std::string& key) {
  if (!key.empty() && key.front() == '/') throw Error(code::invalid_argument, "store key \"" + key + "\" is absolute");
  // a segment `..` would leave the root
  std::size_t b = 0;
  while (b <= key.size()) {
    std::size_t e = key.find('/', b);
    if (e == std::string::npos) e = key.size();
    if (key.compare(b, e - b, "..") == 0) throw Error(code::invalid_argument, "store key \"" + key + "\" leaves the store");
    b = e + 1;
  }
}

std::optional<std::string> Store::get_text(const std::string& key) const {
  auto b = get(key);
  if (!b) return std::nullopt;
  return std::string(b->begin(), b->end());
}

// ── PrefixStore ─────────────────────────────────────────────────────────────────────────────────────

PrefixStore::PrefixStore(Store& base, std::string prefix) : base_(&base), prefix_(std::move(prefix)) {
  while (!prefix_.empty() && prefix_.back() == '/') prefix_.pop_back();
}
std::string PrefixStore::at(const std::string& key) const { return join_key(prefix_, key); }
std::optional<Bytes> PrefixStore::get(const std::string& key) const { return base_->get(at(key)); }
std::optional<Bytes> PrefixStore::get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const { return base_->get_range(at(key), offset, length); }
std::optional<std::uint64_t> PrefixStore::size(const std::string& key) const { return base_->size(at(key)); }
bool PrefixStore::exists(const std::string& key) const { return base_->exists(at(key)); }
bool PrefixStore::exists_prefix(const std::string& prefix) const { return base_->exists_prefix(at(prefix)); }
Listing PrefixStore::list(const std::string& prefix) const { return base_->list(at(prefix)); }
bool PrefixStore::is_link(const std::string& prefix) const { return base_->is_link(at(prefix)); }
void PrefixStore::check_inside(const std::string& key) const { base_->check_inside(at(key)); }
void PrefixStore::set(const std::string& key, const std::uint8_t* data, std::size_t n) { base_->set(at(key), data, n); }
std::unique_ptr<StoreWriter> PrefixStore::open_writer(const std::string& key) { return base_->open_writer(at(key)); }
void PrefixStore::erase(const std::string& key) { base_->erase(at(key)); }
void PrefixStore::erase_prefix(const std::string& prefix) { base_->erase_prefix(at(prefix)); }
void PrefixStore::rename_prefix(const std::string& from, const std::string& to) { base_->rename_prefix(at(from), at(to)); }

// ── FolderStore ─────────────────────────────────────────────────────────────────────────────────────

namespace {

class FileWriter : public StoreWriter {
 public:
  explicit FileWriter(const fs::path& p) : path_(p) {
    fs::create_directories(p.parent_path());
    f_.open(p, std::ios::binary | std::ios::trunc);
    if (!f_) throw Error(code::io, p.string() + ": cannot be written");
  }
  void append(const std::uint8_t* data, std::size_t n) override {
    if (n) f_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
  }
  void close() override {
    f_.close();
    if (!f_) throw Error(code::io, path_.string() + ": cannot be written");
  }

 private:
  fs::path path_;
  std::ofstream f_;
};

}  // namespace

FolderStore::FolderStore(fs::path root) : root_(std::move(root)) {}

fs::path FolderStore::path_of(const std::string& key) const {
  check_store_key(key);
  return key.empty() ? root_ : root_ / key;
}

std::optional<Bytes> FolderStore::get(const std::string& key) const {
  const fs::path p = path_of(key);
  if (!fs::is_regular_file(p)) return std::nullopt;
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw ZarrIOError("[nxr/io] cannot read " + p.string());
  const auto n = static_cast<std::size_t>(f.tellg());
  Bytes b(n);
  f.seekg(0);
  if (n) f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(n));
  if (!f) throw ZarrIOError("[nxr/io] short read");
  return b;
}

std::optional<Bytes> FolderStore::get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const {
  const fs::path p = path_of(key);
  if (!fs::is_regular_file(p)) return std::nullopt;
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw ZarrIOError("[nxr/io] cannot read " + p.string());
  const auto total = static_cast<std::uint64_t>(f.tellg());
  if (offset + length > total) throw ZarrFormatError("[nxr/io] a chunk reaches past the end of its file");
  Bytes b(length);
  f.seekg(static_cast<std::streamoff>(offset));
  if (length) f.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(length));
  if (!f) throw ZarrIOError("[nxr/io] short read");
  return b;
}

std::optional<std::uint64_t> FolderStore::size(const std::string& key) const {
  const fs::path p = path_of(key);
  std::error_code ec;
  if (!fs::is_regular_file(p, ec)) return std::nullopt;
  const auto n = fs::file_size(p, ec);
  if (ec) throw ZarrIOError("[nxr/io] cannot read " + p.string());
  return static_cast<std::uint64_t>(n);
}

bool FolderStore::exists(const std::string& key) const { return fs::is_regular_file(path_of(key)); }

bool FolderStore::exists_prefix(const std::string& prefix) const { return fs::exists(fs::symlink_status(path_of(prefix))); }

Listing FolderStore::list(const std::string& prefix) const {
  Listing out;
  const fs::path dir = path_of(prefix);
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return out;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_directory()) out.prefixes.push_back(e.path().filename().string());
    else if (e.is_regular_file()) out.keys.push_back(e.path().filename().string());
  }
  std::sort(out.keys.begin(), out.keys.end());
  std::sort(out.prefixes.begin(), out.prefixes.end());
  return out;
}

bool FolderStore::is_link(const std::string& prefix) const {
  std::error_code ec;
  return fs::is_symlink(fs::symlink_status(path_of(prefix), ec));
}

// A write acts only inside the store: the place the key resolves to (through every link on the way, and a link at the key itself, whose target a write
// would change) is under the root. Checked before a folder is made or a byte written.
void FolderStore::check_inside(const std::string& key) const {
  const fs::path p = path_of(key);
  std::error_code ec;
  const fs::path root = fs::weakly_canonical(root_, ec);
  const fs::path at = fs::weakly_canonical(p, ec);
  auto r = root.begin();
  auto q = at.begin();
  for (; r != root.end(); ++r, ++q)
    if (q == at.end() || *q != *r) throw Error(code::invalid_argument, p.string() + ": resolves outside the store (a link on its way) — refused");
}

void FolderStore::set(const std::string& key, const std::uint8_t* data, std::size_t n) {
  check_inside(key);
  const fs::path p = path_of(key);
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) throw Error(code::io, p.string() + ": cannot be written");
  if (n) f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
}

std::unique_ptr<StoreWriter> FolderStore::open_writer(const std::string& key) {
  check_inside(key);
  return std::make_unique<FileWriter>(path_of(key));
}

// Defence in depth for every delete: the folder holding what is removed resolves (through any link on the way) to a place inside the store's root.
// The removed entry itself may be a link (the link goes, never its target).
void FolderStore::require_inside(const fs::path& p) const {
  std::error_code ec;
  const fs::path root = fs::weakly_canonical(root_, ec);
  const fs::path parent = fs::weakly_canonical(p.parent_path(), ec);
  auto r = root.begin();
  auto q = parent.begin();
  for (; r != root.end(); ++r, ++q)
    if (q == parent.end() || *q != *r) throw Error(code::invalid_argument, p.string() + ": resolves outside the store — refused");
}

void FolderStore::erase(const std::string& key) {
  const fs::path p = path_of(key);
  require_inside(p);
  std::error_code ec;
  fs::remove(p, ec);
}

void FolderStore::erase_prefix(const std::string& prefix) {
  if (prefix.empty()) throw Error(code::invalid_argument, "the store's root is not erased");
  const fs::path p = path_of(prefix);
  require_inside(p);
  remove_folder(p);   // entries that vanish while it goes are fine; tried twice
}

void FolderStore::rename_prefix(const std::string& from, const std::string& to) {
  const fs::path a = path_of(from), b = path_of(to);
  require_inside(a);   // what moves, and where it lands, are inside the store (a moved link goes as a link)
  check_inside(to);
  if (fs::exists(fs::symlink_status(b))) throw Error(code::already_exists, b.string() + ": exists");
  fs::create_directories(b.parent_path());
  fs::rename(a, b);
}

}  // namespace nxr::io
