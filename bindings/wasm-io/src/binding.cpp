// nxr-neuro in WebAssembly: the module behind `nxr.<verb>.<noun>(options)` (js/index.mjs; docs/specs/2026-11-19-io-service.md, step 5).
//
// A face: it CONVERTS. Every command of the command line's table is the one library function bindings/common/commands.h calls (the same call the Python face makes); this file
// hands it the options (JSON text), the arrays in memory (typed arrays' bytes) and the progress callback, and hands back the result (JSON text) and the arrays (bytes, dtype,
// shape). No SQL, no logic, no repository: a dataset is a path.
//
//   command(handle, verb, noun, optionsJson, arrays, progress)   { result, arrays: [{ name, dtype, shape, bytes }] }  or  { error, code }
//   run(handle, argsJson, progress)                              { result }   `run PIPELINE` on a subject or a group (args: see bindings/common/commands.h `run`)
//   pipelines()                                                  { result }   what `run` takes (list pipelines)
//   openHandle(dataset, mode) · closeHandle(handle)              a dataset opened once (nxr/io/task/handle.h): 'read' keeps its databases open read-only between calls,
//                                                                'write' holds the dataset's writer lock until closed; a handle is a number (0: none)
//   syntheticDataset(datastore, name, subject, recording, channels, sfreq, seconds)   a small synthetic dataset for tests: its folder
//   nxrVersion()
// `arrays` (in): [{ option, name, dtype, shape, bytes }]: `option` the option the array is the value of (`values`; `measure.NAME` for a measure of `add measurements`), `name` what the
// producer records for it (`values.npy`). A failure is `{ error, code }` (code: nxr::io::code), which js/index.mjs throws as an NxrError.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "commands.h"   // bindings/common
#include "nxr/core/errors.h"
#include "nxr/io/errors.h"
#include "nxr/io/task/handle.h"
#include "nxr/pipelines/face.h"

namespace {

using emscripten::val;
using nxr::io::Json;
namespace task = nxr::io::task;
namespace face = nxr::pipelines::face;

val failure(const std::string& message, const std::string& code) {
  val o = val::object();
  o.set("error", message);
  o.set("code", code);
  return o;
}
template <class F>
val guarded(F&& f) {
  try {
    return f();
  } catch (const task::Refused& e) {   // a command's options refused as given (the command line's usage error)
    return failure(e.what(), "invalid_argument");
  } catch (const nxr::io::Error& e) {
    return failure(e.what(), e.code());
  } catch (const nxr::core::Error& e) {
    return failure(e.what(), std::string(nxr::core::errorCodeName(e.code())));
  } catch (const std::exception& e) {
    return failure(e.what(), "internal");
  }
}

// the handles open in this module, by number
std::map<int, std::unique_ptr<task::Handle>>& handles() {
  static std::map<int, std::unique_ptr<task::Handle>> h;
  return h;
}
int next_handle = 1;
task::Handle* handle_of(int id) {
  if (id == 0) return nullptr;
  const auto it = handles().find(id);
  if (it == handles().end()) throw task::Refused("the handle is closed: open it again", true);
  return it->second.get();
}

task::Progress progress_of(const val& callback) {
  if (callback.isUndefined() || callback.isNull()) return {};
  return [callback](const Json& j) mutable { callback(j.dump()); };   // called from the library: the line as JSON text
}

val bytes_val(const std::uint8_t* p, std::size_t n) { return val::global("Uint8Array").new_(emscripten::typed_memory_view(n, p)); }   // a copy

nxr::bindings::Arrays arrays_of(const val& list) {
  nxr::bindings::Arrays out;
  if (list.isUndefined() || list.isNull()) return out;
  const unsigned n = list["length"].as<unsigned>();
  for (unsigned i = 0; i < n; ++i) {
    const val e = list[i];
    std::vector<std::int64_t> shape;
    for (const double d : emscripten::vecFromJSArray<double>(e["shape"])) shape.push_back(static_cast<std::int64_t>(d));
    const val bytes = e["bytes"];
    const unsigned len = bytes["length"].as<unsigned>();
    std::vector<std::uint8_t> data(len);
    if (len) val(emscripten::typed_memory_view(len, data.data())).call<void>("set", bytes);
    out[e["option"].as<std::string>()] = task::ArrayInput::memory(e["name"].as<std::string>(), e["dtype"].as<std::string>(), std::move(shape), std::move(data));
  }
  return out;
}

val result_val(const task::CommandResult& r) {
  val o = val::object();
  o.set("result", r.result.dump());
  val arrays = val::array();
  for (const auto& a : r.arrays) {
    val x = val::object();
    x.set("name", a.name);
    x.set("dtype", a.dtype);
    val shape = val::array();
    for (const auto d : a.shape) shape.call<void>("push", static_cast<double>(d));
    x.set("shape", shape);
    x.set("bytes", bytes_val(a.bytes.data(), a.bytes.size()));
    arrays.call<void>("push", x);
  }
  o.set("arrays", arrays);
  return o;
}

val command(int handle, const std::string& verb, const std::string& noun, const std::string& options, const val& arrays, const val& progress) {
  return guarded([&] {
    const task::CommandResult r = nxr::bindings::command(verb, noun, Json::parse(options), progress_of(progress), face::stamp(), arrays_of(arrays), handle_of(handle));
    return result_val(r);
  });
}

val run(int handle, const std::string& args, const val& progress) {
  return guarded([&] {
    val o = val::object();
    o.set("result", nxr::bindings::run(Json::parse(args), progress_of(progress), handle_of(handle)).dump());
    return o;
  });
}

val pipelines() {
  return guarded([&] {
    val o = val::object();
    o.set("result", face::pipelines().dump());
    return o;
  });
}

val open_handle(const std::string& dataset, const std::string& mode) {
  return guarded([&] {
    if (mode != "read" && mode != "write") throw task::Refused("mode is 'read' or 'write' (got '" + mode + "')", true);
    const int id = next_handle++;
    handles()[id] = std::make_unique<task::Handle>(dataset, mode == "write" ? task::Handle::Mode::write : task::Handle::Mode::read, face::stamp());
    val o = val::object();
    o.set("handle", id);
    return o;
  });
}

val close_handle(int id) {
  return guarded([&] {
    const auto it = handles().find(id);
    if (it != handles().end()) {
      it->second->close();
      handles().erase(it);
    }
    return val::object();
  });
}

val synthetic_dataset(const std::string& datastore, const std::string& name, const std::string& subject, const std::string& recording, int channels, double sfreq, double seconds) {
  return guarded([&] {
    val o = val::object();
    o.set("result", face::synthetic_dataset(datastore, name, subject, recording, channels, sfreq, seconds).string());
    return o;
  });
}

std::string version() { return NXR_NEURO_VERSION; }

}  // namespace

EMSCRIPTEN_BINDINGS(nxr_neuro) {
  emscripten::function("command", &command);
  emscripten::function("run", &run);
  emscripten::function("pipelines", &pipelines);
  emscripten::function("openHandle", &open_handle);
  emscripten::function("closeHandle", &close_handle);
  emscripten::function("syntheticDataset", &synthetic_dataset);
  emscripten::function("nxrVersion", &version);
}
