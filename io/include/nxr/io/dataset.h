// nxr-io — an nxr dataset's database, read-only: `<dataset folder>/dataset.sqlite` at schema 50.
#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/errors.h"

namespace nxr::io {

constexpr int kSchemaVersion = 50;  // the version this build reads (nxr.io.database.schema_version)

class Dataset {
 public:
  // Open read-only. Throws DatasetError with nxr's wording when the file is missing, is not a dataset database, or is at
  // another `user_version` (49 -> `nxr migrate`; any other -> re-convert).
  static Dataset open(const std::filesystem::path& folder);
  Dataset(Dataset&&) noexcept;
  Dataset& operator=(Dataset&&) noexcept;
  ~Dataset();

  const std::filesystem::path& folder() const { return folder_; }
  // A parameterised query (`?` placeholders); rows in order.
  std::vector<Row> query(const std::string& sql, const std::vector<Value>& params = {}) const;

 private:
  Dataset() = default;
  std::filesystem::path folder_;
  void* db_ = nullptr;  // sqlite3*
};

}  // namespace nxr::io
