// Test helper: the array whose folder is `p`, read through a store made for it (the library takes a store and a key; a test that has a folder path uses this).
// A subject of a format 52 dataset keeps its small values in `<subject>.nxr.zarr/small_files.sqlite`: when a folder above `p` holds one, the array is read through
// that subject's SmallFileStore (read only), the key relative to the subject's folder; otherwise through the folder tree, as before.
#pragma once
#include <filesystem>
#include <cstring>
#include <memory>
#include <vector>

#include "nxr/io/array_reader.h"
#include "nxr/io/small_file_store.h"
#include "nxr/io/store.h"

namespace nxrtest {
namespace detail {
struct SubjectStore {
  nxr::io::FolderStore files;
  nxr::io::SmallFileStore small;
  explicit SubjectStore(const std::filesystem::path& subject)
      : files(subject), small(files, subject / nxr::io::kSmallFilesFile, nxr::io::SmallFileOptions{nxr::io::kSmallFileThreshold, true}) {}
};
}  // namespace detail

inline nxr::io::ArrayReader open_array(const std::filesystem::path& folder) {
  namespace fs = std::filesystem;
  const fs::path p = folder.lexically_normal();
  for (fs::path a = p.has_filename() ? p.parent_path() : p.parent_path().parent_path(); !a.empty() && a != a.parent_path(); a = a.parent_path()) {
    std::error_code ec;
    if (!fs::is_regular_file(a / nxr::io::kSmallFilesFile, ec)) continue;
    auto subject = std::make_shared<detail::SubjectStore>(a);
    const std::string key = fs::relative(p.has_filename() ? p : p.parent_path(), a).generic_string();
    return nxr::io::ArrayReader::open(std::shared_ptr<nxr::io::Store>(subject, &subject->small), key);
  }
  const fs::path parent = p.has_filename() ? p.parent_path() : p.parent_path().parent_path();
  const std::string name = p.has_filename() ? p.filename().string() : p.parent_path().filename().string();
  return nxr::io::ArrayReader::open(std::make_shared<nxr::io::FolderStore>(parent), name);
}

// A measurement a pipeline run stored (its `stored` entry: `array`, `component`, `components`) in the layout of one measure alone: [rows x members x (components + 1)], its own components then the
// count. The run packs the measurements over one Selection into one array (store_measurements), each at its `component`; this reads its part.
template <class J>
inline std::vector<float> read_packed(const std::filesystem::path& subject_folder, const J& stored, std::vector<std::int64_t>* shape = nullptr) {
  nxr::io::ArrayReader r = open_array(subject_folder / stored["array"].template get<std::string>());
  const auto sh = r.shape();
  const auto raw = r.read();
  std::vector<float> all(raw.size() / sizeof(float));
  std::memcpy(all.data(), raw.data(), all.size() * sizeof(float));
  const std::int64_t CC = sh[2], off = stored.value("component", std::int64_t(0)), C = stored.value("components", CC - 1);
  std::vector<float> out(static_cast<std::size_t>(sh[0] * sh[1] * (C + 1)));
  for (std::int64_t i = 0; i < sh[0] * sh[1]; ++i) {
    for (std::int64_t c = 0; c < C; ++c) out[static_cast<std::size_t>(i * (C + 1) + c)] = all[static_cast<std::size_t>(i * CC + off + c)];
    out[static_cast<std::size_t>(i * (C + 1) + C)] = all[static_cast<std::size_t>(i * CC + CC - 1)];
  }
  if (shape) *shape = {sh[0], sh[1], C + 1};
  return out;
}
}  // namespace nxrtest
