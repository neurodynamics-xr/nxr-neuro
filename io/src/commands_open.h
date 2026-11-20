// The commands' shared openers (io/src only; nxr/io/task/commands.h): how every command opens the dataset it names, and its notes.
#pragma once
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/catalogue.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/task.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/writer.h"

namespace nxr::io::task::commands_detail {

namespace fs = std::filesystem;


inline void note(const Progress& progress, const std::string& text) {
  if (progress) progress(Json{{"stage", "note"}, {"note", text}});
}

inline TemplateSearch search_of(const std::vector<std::string>& templates) {
  TemplateSearch s;
  for (const auto& p : templates) s.command_paths.push_back(p);
  return s;
}

// THE DATASET A COMMAND WORKS ON: a catalogue's subject package (found through the catalogue), or a one-database dataset (store formats 50-52), as every command opens it.
inline DatasetWriter open_dataset(const std::string& folder, const std::optional<std::string>& subject, const OpenOptions& o, const std::vector<std::string>& templates,
                           const Progress& progress) {
  if (is_catalogue_dataset(folder)) {
    if (!subject) throw Refused(folder + " is a catalogue of subject packages: name the subject to work on with --subject", true);
    return open_subject_package(folder, *subject, o, search_of(templates));
  }
  if (fs::is_regular_file(fs::path(folder) / "dataset.sqlite"))
    note(progress, folder + " is a one-database dataset; `nxr-neuro split dataset` makes it a catalogue of subject packages (the default form).");
  return open_writer(folder, o);
}

// The commands that make or change a one-database dataset itself refuse a catalogue, naming what to use.
inline DatasetWriter open_one_database(const std::string& folder, const std::string& what, const OpenOptions& o = {}) {
  if (is_catalogue_dataset(folder))
    throw Refused(what + " works on a one-database dataset; " + folder + " is a catalogue of subject packages (`list subjects`, `show subject` and the `compute` verbs open its packages)", true);
  return open_writer(folder, o);
}

// ---- ARRAYS A COMMAND TAKES (ArrayInput): a .npy file, or an array in memory; the same checks and the same stored precision either way ----

namespace detail_array {
template <class S, class T>
std::vector<T> convert_from(const std::vector<std::uint8_t>& bytes) {
  std::vector<T> out(bytes.size() / sizeof(S));
  for (std::size_t i = 0; i < out.size(); ++i) {
    S x;
    std::memcpy(&x, bytes.data() + i * sizeof(S), sizeof(S));
    out[i] = static_cast<T>(x);
  }
  return out;
}
inline std::size_t itemsize_of(const std::string& dtype) {
  if (dtype == "float64" || dtype == "int64" || dtype == "uint64") return 8;
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "int16" || dtype == "uint16") return 2;
  if (dtype == "int8" || dtype == "uint8" || dtype == "bool") return 1;
  throw Refused("an array of dtype " + dtype + " is not read (float32, float64, int8 to int64, uint8 to uint64, bool)");
}
}  // namespace detail_array

/** the shape of an array input (a .npy file's header, or the in-memory array's, checked against its bytes) */
inline std::vector<std::int64_t> array_shape(const ArrayInput& a) {
  if (!a.in_memory) return NpyFile::open(a.path).shape();
  std::int64_t n = 1;
  for (const auto d : a.shape) {
    if (d < 0) throw Refused(a.label() + ": a negative size in its shape");
    n *= d;
  }
  if (static_cast<std::size_t>(n) * detail_array::itemsize_of(a.dtype) != a.bytes.size())
    throw Refused(a.label() + ": " + std::to_string(a.bytes.size()) + " bytes are not " + std::to_string(n) + " values of " + a.dtype);
  return a.shape;
}
/** the store's name of an array input's dtype (float32, float64, int32, uint8, bool, ...) */
inline std::string array_dtype(const ArrayInput& a) { return a.in_memory ? a.dtype : npy_dtype_name(NpyFile::open(a.path).descr()); }
/** the whole array converted to T, in C order */
template <class T>
std::vector<T> array_all(const ArrayInput& a) {
  if (!a.in_memory) return NpyFile::open(a.path).all<T>();
  (void)array_shape(a);
  const std::string& d = a.dtype;
  if (d == "float32") return detail_array::convert_from<float, T>(a.bytes);
  if (d == "float64") return detail_array::convert_from<double, T>(a.bytes);
  if (d == "int8") return detail_array::convert_from<std::int8_t, T>(a.bytes);
  if (d == "int16") return detail_array::convert_from<std::int16_t, T>(a.bytes);
  if (d == "int32") return detail_array::convert_from<std::int32_t, T>(a.bytes);
  if (d == "int64") return detail_array::convert_from<std::int64_t, T>(a.bytes);
  if (d == "uint8" || d == "bool") return detail_array::convert_from<std::uint8_t, T>(a.bytes);
  if (d == "uint16") return detail_array::convert_from<std::uint16_t, T>(a.bytes);
  if (d == "uint32") return detail_array::convert_from<std::uint32_t, T>(a.bytes);
  if (d == "uint64") return detail_array::convert_from<std::uint64_t, T>(a.bytes);
  throw Refused(a.label() + ": an array of dtype " + d + " is not read");
}

inline OpenOptions reader() {
  OpenOptions o;
  o.lock = false;
  return o;
}

}  // namespace nxr::io::task::commands_detail

namespace nxr::io::task {
/** the count of each class of node a dataset holds (summed over its subjects; `show dataset`): {manifolds, fields, selections, operators, statistics, dictionaries, spaces, acquisitions, jobs}; a package of a
 *  catalogue that cannot be opened is named in `unavailable` */
Json class_counts(const std::string& dataset, const Progress& progress, Json* unavailable);
}  // namespace nxr::io::task
