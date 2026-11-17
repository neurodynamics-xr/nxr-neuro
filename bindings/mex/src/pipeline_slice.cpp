// The pipeline level of the MATLAB face (`run.*` commands of the one `nxr_compute` mexFunction): `nxr.run`, `nxr.list.*` and `nxr.read`. A binding only converts: the library's calls are
// pipelines/face.h (the ones `nxr-neuro run`, `list` and `read` make); structured answers go back as JSON text (decoded by the MATLAB functions), arrays as MATLAB arrays.
// Progress lines call back a MATLAB function handle with the JSON text of each line.
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "first_slice.h"
#include "first_slice_internal.h"
#include "nxr/io/errors.h"
#include "nxr/pipelines/face.h"

namespace nxr::mex {

namespace {

using namespace nxr::mex::slice;
namespace face = nxr::pipelines::face;
using Json = face::Json;

void need(int nrhs, int n, const char* cmd) {
  if (nrhs != n + 1) throw std::invalid_argument(std::string("nxr_compute('") + cmd + "') takes " + std::to_string(n) + " arguments (use the nxr package functions)");
}

// "no_such_row" -> "nxr:noSuchRow"
std::string identifier(const std::string& code) {
  std::string id = "nxr:";
  bool up = false;
  for (const char c : code) {
    if (c == '_') { up = true; continue; }
    id.push_back(up ? static_cast<char>(std::toupper(static_cast<unsigned char>(c))) : c);
    up = false;
  }
  return id;
}

std::optional<double> opt_scalar(const mxArray* a, const char* what) {
  if (is_unset(a)) return std::nullopt;
  return get_scalar(a, what);
}

// a C-ordered array of `dtype` (little-endian bytes) as a MATLAB array of the matching class with the same dimensions (a vector is a column)
mxArray* give_stored(const face::StoredArray& a) {
  mxClassID cls;
  std::size_t item;
  const std::string& d = a.dtype;
  if (d == "float32") { cls = mxSINGLE_CLASS; item = 4; }
  else if (d == "float64") { cls = mxDOUBLE_CLASS; item = 8; }
  else if (d == "int8") { cls = mxINT8_CLASS; item = 1; }
  else if (d == "uint8") { cls = mxUINT8_CLASS; item = 1; }
  else if (d == "int16") { cls = mxINT16_CLASS; item = 2; }
  else if (d == "uint16") { cls = mxUINT16_CLASS; item = 2; }
  else if (d == "int32") { cls = mxINT32_CLASS; item = 4; }
  else if (d == "uint32") { cls = mxUINT32_CLASS; item = 4; }
  else if (d == "int64") { cls = mxINT64_CLASS; item = 8; }
  else if (d == "uint64") { cls = mxUINT64_CLASS; item = 8; }
  else throw std::invalid_argument("'" + a.name + "' has the stored type " + d + ", which nxr.read does not hand over");
  std::vector<mwSize> dims;
  for (const auto s : a.shape) dims.push_back(static_cast<mwSize>(s));
  if (dims.empty()) dims = {1, 1};
  if (dims.size() == 1) dims.push_back(1);
  mxArray* out = mxCreateNumericArray(dims.size(), dims.data(), cls, mxREAL);
  const std::size_t n = a.bytes.size() / item;
  auto* dst = static_cast<std::uint8_t*>(mxGetData(out));
  // element i of the C-ordered bytes goes to the column-major place of the same multi-index
  std::vector<std::size_t> cstride(dims.size(), 1), mstride(dims.size(), 1);
  for (std::size_t k = dims.size() - 1; k-- > 0;) cstride[k] = cstride[k + 1] * dims[k + 1];
  for (std::size_t k = 1; k < dims.size(); ++k) mstride[k] = mstride[k - 1] * dims[k - 1];
  std::vector<std::size_t> idx(dims.size(), 0);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t src = 0, to = 0;
    for (std::size_t k = 0; k < dims.size(); ++k) { src += idx[k] * cstride[k]; to += idx[k] * mstride[k]; }
    std::memcpy(dst + to * item, a.bytes.data() + src * item, item);
    for (std::size_t k = dims.size(); k-- > 0;) { if (++idx[k] < dims[k]) break; idx[k] = 0; }
  }
  return out;
}

void runPipeline(int, mxArray** plhs, int nrhs, const mxArray** prhs) {   // run.run(dataset, subject, pipeline, parametersJson, from, to, all, replace, progress)
  need(nrhs, 9, "run.run");
  face::RunRequest rq;
  rq.pipeline = get_string(prhs[3], "pipeline");
  rq.parameters = Json::parse(get_string(prhs[4], "parameters"));
  rq.from = opt_scalar(prhs[5], "from");
  rq.to = opt_scalar(prhs[6], "to");
  rq.all = get_bool(prhs[7], "all");
  rq.replace = get_bool(prhs[8], "replace");
  const mxArray* handle = prhs[9];
  face::Progress say;
  if (!is_unset(handle)) say = [handle](const Json& j) {
    mxArray* args[2] = {const_cast<mxArray*>(handle), mxCreateString(j.dump().c_str())};
    mxArray* trap = mexCallMATLABWithTrap(0, nullptr, 2, args, "feval");
    mxDestroyArray(args[1]);
    if (trap) throw std::runtime_error("the progress function failed");
  };
  plhs[0] = give_string(face::run(get_string(prhs[1], "dataset"), get_string(prhs[2], "subject"), rq, say).dump());
}

void readStored(int, mxArray** plhs, int nrhs, const mxArray** prhs) {   // run.read(dataset, subject, name, measure, of) -> [metaJson, arrays struct]
  need(nrhs, 5, "run.read");
  const face::Stored s = face::read(get_string(prhs[1], "dataset"), get_string(prhs[2], "subject"), get_string(prhs[3], "name"),
                                    is_unset(prhs[4]) ? std::nullopt : std::optional<std::string>(get_string(prhs[4], "measure")),
                                    is_unset(prhs[5]) ? std::nullopt : std::optional<std::string>(get_string(prhs[5], "of")));
  std::vector<std::pair<std::string, mxArray*>> fields;
  for (const auto& a : s.arrays) fields.emplace_back(field_name(a.name), give_stored(a));
  plhs[0] = give_string(s.meta.dump());
  plhs[1] = make_struct(fields);
}

}  // namespace

bool dispatchPipeline(const std::string& cmd, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (cmd.rfind("run.", 0) != 0) return false;
  try {
    if (cmd == "run.pipelines") { need(nrhs, 0, "run.pipelines"); plhs[0] = give_string(face::pipelines().dump()); }
    else if (cmd == "run.datasets") { need(nrhs, 1, "run.datasets"); plhs[0] = give_string(face::datasets(get_string(prhs[1], "datastore")).dump()); }
    else if (cmd == "run.subjects") {   // (dataset, names as JSON text, where)
      need(nrhs, 3, "run.subjects");
      std::vector<std::string> names;
      for (const auto& n : Json::parse(get_string(prhs[2], "names"))) names.push_back(n.get<std::string>());
      plhs[0] = give_string(face::subjects(get_string(prhs[1], "dataset"), names, is_unset(prhs[3]) ? std::nullopt : std::optional<std::string>(get_string(prhs[3], "where"))).dump());
    }
    else if (cmd == "run.results") { need(nrhs, 3, "run.results"); plhs[0] = give_string(face::results(get_string(prhs[1], "dataset"), get_string(prhs[2], "subject"), get_string(prhs[3], "kind")).dump()); }
    else if (cmd == "run.run") runPipeline(nlhs, plhs, nrhs, prhs);
    else if (cmd == "run.read") readStored(nlhs, plhs, nrhs, prhs);
    else if (cmd == "run.synthetic_dataset") {   // (datastore, name, subject, recording, channels, sfreq, seconds): for tests and demos
      need(nrhs, 7, "run.synthetic_dataset");
      plhs[0] = give_string(face::synthetic_dataset(get_string(prhs[1], "datastore"), get_string(prhs[2], "name"), get_string(prhs[3], "subject"), get_string(prhs[4], "recording"),
                                                    static_cast<int>(get_scalar(prhs[5], "channels")), get_scalar(prhs[6], "sfreq"), get_scalar(prhs[7], "seconds")).string());
    }
    else throw SliceFailure{"nxr:unknownCommand", "Unknown command: \"" + cmd + "\"."};
  } catch (const nxr::io::Error& e) {
    throw SliceFailure{identifier(e.code()), e.what()};
  } catch (const std::invalid_argument& e) {
    throw SliceFailure{"nxr:invalidArgument", e.what()};
  }
  return true;
}

}  // namespace nxr::mex
