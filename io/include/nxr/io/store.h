// nxr-io — the STORE: where an array's and a node's bytes live, as zarr KEYS ("a/b/zarr.json", "a/b/c/0/1": '/'-separated, relative to the store's
// root, no leading slash). The interface is the Studio's zarrita `Readable` (get, ranged get) plus what a writer needs (set, a streamed set, erase,
// list, rename). Everything that reads or writes an array, a node's zarr.json or removes a node goes through it, so where the keys live is one
// class's business: today `FolderStore` (a key is the file `root / key`, exactly the folder tree nxr has always written).
//
//   FolderStore store(root);
//   store.set("a/zarr.json", bytes);   auto b = store.get_range("a/c/0", 16, 64);   store.erase_prefix("a");
//
// A key names a VALUE; a prefix names every key under it (`a` holds `a/zarr.json` and `a/c/0`). Folders are not keys: `set` makes whatever is needed
// above a key, and erasing the last key under a prefix leaves nothing (a FolderStore leaves its empty folders, as the folder tree always has).
#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nxr::io {

using Bytes = std::vector<std::uint8_t>;

// The immediate children of a prefix (zarrita's list dir): `keys` the values directly under it, `prefixes` the names that hold more keys. Names only,
// sorted, no '/'.
struct Listing {
  std::vector<std::string> keys, prefixes;
};

// A value written piece by piece, never held whole (a shard is streamed to its key chunk after chunk). The value exists once `open_writer` returned;
// `close` finishes it (and throws when the bytes could not be written).
class StoreWriter {
 public:
  virtual ~StoreWriter() = default;
  virtual void append(const std::uint8_t* data, std::size_t n) = 0;
  virtual void close() = 0;
};

class Store {
 public:
  virtual ~Store() = default;

  // The value at `key`; nullopt when there is none.
  virtual std::optional<Bytes> get(const std::string& key) const = 0;
  // `length` bytes from `offset` of the value at `key`; nullopt when there is no such key; throws ZarrIOError when the range reaches past its end.
  virtual std::optional<Bytes> get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const = 0;
  // The value's size in bytes; nullopt when there is none.
  virtual std::optional<std::uint64_t> size(const std::string& key) const = 0;
  // Whether a VALUE is stored at `key` (not a prefix).
  virtual bool exists(const std::string& key) const = 0;
  // Whether `key` or anything under it is stored.
  virtual bool exists_prefix(const std::string& prefix) const = 0;
  virtual Listing list(const std::string& prefix) const = 0;
  // Whether `prefix` stands for another place (a symbolic link in a folder). A store without such keys says no.
  virtual bool is_link(const std::string& prefix) const { (void)prefix; return false; }

  // Throws (invalid_argument) when a write or a move at `key` would act outside the store: a link on the way (or at the key) leading out of the root.
  // The writes below call it themselves; a store that routes a write elsewhere first (small keys into a database) calls it before anything is written.
  virtual void check_inside(const std::string& key) const { (void)key; }

  // Store `data` at `key`, replacing what is there; the keys above it come into being.
  virtual void set(const std::string& key, const std::uint8_t* data, std::size_t n) = 0;
  void set(const std::string& key, const Bytes& b) { set(key, b.data(), b.size()); }
  void set(const std::string& key, const std::string& s) { set(key, reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }
  virtual std::unique_ptr<StoreWriter> open_writer(const std::string& key) = 0;
  // Remove the value at `key` (nothing when there is none).
  virtual void erase(const std::string& key) = 0;
  // Remove `prefix` and every key under it (nothing when there is none). Throws when it cannot be removed whole.
  virtual void erase_prefix(const std::string& prefix) = 0;
  // Move every key under `from` to the same place under `to` (`from` itself, when it is a value). Refused when `to` exists.
  virtual void rename_prefix(const std::string& from, const std::string& to) = 0;

  // The value as text; nullopt when there is none.
  std::optional<std::string> get_text(const std::string& key) const;
};

// A store's keys under `prefix`, seen as a store of their own: `get("zarr.json")` reads `<prefix>/zarr.json`. (A subject's keys in its dataset's store.)
class PrefixStore : public Store {
 public:
  PrefixStore(Store& base, std::string prefix);
  const std::string& prefix() const { return prefix_; }
  Store& base() const { return *base_; }
  std::optional<Bytes> get(const std::string& key) const override;
  std::optional<Bytes> get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const override;
  std::optional<std::uint64_t> size(const std::string& key) const override;
  bool exists(const std::string& key) const override;
  bool exists_prefix(const std::string& prefix) const override;
  Listing list(const std::string& prefix) const override;
  bool is_link(const std::string& prefix) const override;
  void check_inside(const std::string& key) const override;
  using Store::set;
  void set(const std::string& key, const std::uint8_t* data, std::size_t n) override;
  std::unique_ptr<StoreWriter> open_writer(const std::string& key) override;
  void erase(const std::string& key) override;
  void erase_prefix(const std::string& prefix) override;
  void rename_prefix(const std::string& from, const std::string& to) override;

 private:
  std::string at(const std::string& key) const;
  Store* base_;
  std::string prefix_;
};

// The folder tree as a store: key `a/b` is the file `root / a / b`. Keys may not climb out of the root (`..`, a leading '/').
class FolderStore : public Store {
 public:
  explicit FolderStore(std::filesystem::path root);
  const std::filesystem::path& root() const { return root_; }
  // The file a key is (for messages and for the code that still hands a folder to a tool outside nxr-io).
  std::filesystem::path path_of(const std::string& key) const;
  std::optional<Bytes> get(const std::string& key) const override;
  std::optional<Bytes> get_range(const std::string& key, std::uint64_t offset, std::uint64_t length) const override;
  std::optional<std::uint64_t> size(const std::string& key) const override;
  bool exists(const std::string& key) const override;
  bool exists_prefix(const std::string& prefix) const override;
  Listing list(const std::string& prefix) const override;
  bool is_link(const std::string& prefix) const override;
  void check_inside(const std::string& key) const override;
  using Store::set;
  void set(const std::string& key, const std::uint8_t* data, std::size_t n) override;
  std::unique_ptr<StoreWriter> open_writer(const std::string& key) override;
  void erase(const std::string& key) override;
  void erase_prefix(const std::string& prefix) override;
  void rename_prefix(const std::string& from, const std::string& to) override;

 private:
  void require_inside(const std::filesystem::path& p) const;   // a delete's resolved place stays inside root_
  std::filesystem::path root_;
};

// A key must be relative and stay inside its store: no leading '/', no `..` segment. Throws invalid_argument.
void check_store_key(const std::string& key);

// `a` + `b` as a key ("" is the root: join_key("", "b") is "b").
std::string join_key(const std::string& prefix, const std::string& key);

}  // namespace nxr::io
