// nxr-io — one exception type with a stable code.
//
//   try { ... } catch (const nxr::io::Error& e) { e.code(); e.what(); }
//
// `code()` is stable (bindings and the CLI key on it); `what()` is nxr's own wording. `python_name()` is the name of the
// exception nxr's Python raises for the same failure, which the command line prints in front of the message (as nxr's does).
#pragma once
#include <stdexcept>
#include <string>

namespace nxr::io {

// The codes (a closed set; a new one is a new line here and in `python_name`).
namespace code {
constexpr const char* invalid_argument = "invalid_argument";   // ValueError
constexpr const char* already_exists = "already_exists";       // FileExistsError
constexpr const char* not_found = "not_found";                 // FileNotFoundError
constexpr const char* no_such_row = "no_such_row";             // KeyError
constexpr const char* writer_lock_held = "writer_lock_held";   // WriterLockHeld
constexpr const char* not_writable = "not_writable";           // DatasetNotWritable
constexpr const char* schema_version = "schema_version";       // RuntimeError: a database at another user_version
constexpr const char* not_a_database = "not_a_database";       // RuntimeError
constexpr const char* sqlite = "sqlite";                       // sqlite3.Error
constexpr const char* io = "io";                               // OSError
constexpr const char* unsupported = "unsupported";             // NotImplementedError
constexpr const char* zarr_format = "zarr_format";             // a Zarr store that is malformed or not supported
constexpr const char* zarr_io = "zarr_io";
constexpr const char* id_clash = "id_clash";                   // a row id that two packages both own (a subject package, section 2.1 of the subject-centred spec)
constexpr const char* hash_mismatch = "hash_mismatch";         // a package whose bytes are not the content hash it records
constexpr const char* ambiguous = "ambiguous";               // a record `find` that several rows match (the record API names the rows)
constexpr const char* cancelled = "cancelled";                 // a job stopped on request (the chunk loop): nothing of it is stored
}  // namespace code

class Error : public std::runtime_error {
 public:
  Error(std::string code, const std::string& message) : std::runtime_error(message), code_(std::move(code)) {}
  const std::string& code() const noexcept { return code_; }
  // The name of the Python exception nxr raises for this code.
  std::string python_name() const;

 private:
  std::string code_;
};

// --- the engine's older classes, kept: each is an Error with its code ---
struct ZarrError : Error {
  explicit ZarrError(const std::string& m, const char* c = code::zarr_format) : Error(c, m) {}
};
// Filesystem / I/O failures.
struct ZarrIOError : ZarrError {
  explicit ZarrIOError(const std::string& m) : ZarrError(m, code::zarr_io) {}
};
// Malformed metadata, unknown dtype, shape mismatch, bad chunk.
struct ZarrFormatError : ZarrError {
  explicit ZarrFormatError(const std::string& m) : ZarrError(m, code::zarr_format) {}
};
// A store does not conform to its declared schema (validation layer).
struct ZarrSchemaError : ZarrError {
  explicit ZarrSchemaError(const std::string& m) : ZarrError(m, code::zarr_format) {}
};

// A dataset database that cannot be opened, or is at another schema version.
struct DatasetError : Error {
  explicit DatasetError(const std::string& m, const char* c = code::not_a_database) : Error(c, m) {}
};

inline std::string Error::python_name() const {
  const std::string& c = code_;
  if (c == code::invalid_argument) return "ValueError";
  if (c == code::already_exists) return "FileExistsError";
  if (c == code::not_found) return "FileNotFoundError";
  if (c == code::no_such_row) return "KeyError";
  if (c == code::writer_lock_held) return "WriterLockHeld";
  if (c == code::not_writable) return "DatasetNotWritable";
  if (c == code::unsupported) return "NotImplementedError";
  if (c == code::sqlite) return "sqlite3.Error";
  if (c == code::io) return "OSError";
  return "RuntimeError";
}

}  // namespace nxr::io
