// Writes in pieces (pieces.h; nxr/io/task/commands.h `PieceOptions`; docs/specs/2026-11-20-data-layer-queries.md, 3.7): `add field | statistic | measurements` as begin, append, commit
// (a measurements write holds ONE measure: its values [members], [members x N] or [members x N x C], as the one-shot command takes them).
//
// The pieces wait in `<dataset>/.writes/<write id>/` (write.json, values.npy), outside the database: nothing is in the subject until the commit, which is the one-shot command's own unit
// (add_field | add_statistic called on the staged file), so a commit stores exactly what the one-shot command stores. Rolling back is removing the folder.
#include "pieces.h"

#include <signal.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <type_traits>

#include "commands_open.h"
#include "nxr/io/database.h"
#include "nxr/io/errors.h"
#include "nxr/io/npy.h"
#include "nxr/io/task/handle.h"

namespace nxr::io::task::pieces {

using namespace commands_detail;
namespace fs = std::filesystem;

namespace {
using Box = std::vector<std::array<std::int64_t, 2>>;

fs::path root_of(const std::string& dataset) { return fs::path(dataset) / ".writes"; }

std::string new_id() {
  static std::mt19937_64 rng{std::random_device{}()};
  static const char* hex = "0123456789abcdef";
  std::string id = "w-";
  for (int i = 0; i < 12; ++i) id.push_back(hex[rng() & 15]);
  return id;
}
bool is_id(const std::string& id) {
  if (id.size() < 3 || id.size() > 40) return false;
  for (const char c : id) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) return false;
  return true;
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
  std::string t;
  for (const auto d : shape) t += (t.empty() ? "" : " x ") + std::to_string(d);
  return "[" + t + "]";
}
std::string box_text(const Box& b) {
  std::string t;
  for (const auto& r : b) t += (t.empty() ? "" : ", ") + std::to_string(r[0]) + ":" + std::to_string(r[1]);
  return t;
}
Json box_json(const Box& b) {
  Json j = Json::array();
  for (const auto& r : b) j.push_back(Json::array({r[0], r[1]}));
  return j;
}
Box box_of(const Json& j) {
  Box b;
  for (const auto& r : j) b.push_back({r[0].get<std::int64_t>(), r[1].get<std::int64_t>()});
  return b;
}
bool overlap(const Box& a, const Box& b) {
  for (std::size_t i = 0; i < a.size(); ++i) if (!(a[i][0] < b[i][1] && b[i][0] < a[i][1])) return false;
  return true;
}
std::int64_t volume(const Box& b) {
  std::int64_t v = 1;
  for (const auto& r : b) v *= r[1] - r[0];
  return v;
}

// a write's folder is held while it is appended to or committed: one at a time (flock, released when the process ends)
class Held {
 public:
  explicit Held(const fs::path& dir) : fd_(::open((dir / "lock").c_str(), O_RDWR | O_CREAT, 0644)) {
    if (fd_ < 0) throw Error(code::not_writable, dir.string() + ": cannot be locked");
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      ::close(fd_);
      fd_ = -1;
      throw Refused("write " + dir.filename().string() + " is in use by another call (one append or commit at a time)", true);
    }
  }
  Held(const Held&) = delete;
  Held& operator=(const Held&) = delete;
  ~Held() { if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); } }

 private:
  int fd_;
};

Json load(const fs::path& dir) {
  std::ifstream f(dir / "write.json");
  if (!f) return Json();
  Json j = Json::parse(f, nullptr, false);
  return j.is_discarded() ? Json() : j;
}
void save(const fs::path& dir, const Json& j) {
  const fs::path tmp = dir / "write.json.tmp";
  {
    std::ofstream o(tmp, std::ios::trunc);
    o << j.dump();
    if (!o) throw Error(code::not_writable, tmp.string() + ": a short write");
  }
  fs::rename(tmp, dir / "write.json");
}

std::int64_t file_age_seconds(const fs::path& p) {
  std::error_code ec;
  const auto t = fs::last_write_time(p, ec);
  if (ec) return 0;
  return std::chrono::duration_cast<std::chrono::seconds>(fs::file_time_type::clock::now() - t).count();
}

bool process_gone(long long pid) {
#ifdef __EMSCRIPTEN__
  (void)pid;
  return false;   // the host's process, not this module's: never judged
#else
  return pid > 0 && ::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
#endif
}

// ---- the request of the one-shot command, kept with the write ----
Json json_of(const AddFieldRequest& r) {
  Json j = Json{{"dataset", r.dataset}, {"subject", r.subject}, {"name", r.name}, {"manifold", r.manifold}, {"value_type", r.value_type}, {"kind", r.kind}, {"replace", r.replace},
                {"contribution", r.contribution}, {"created_by", r.created_by}};
  for (const auto& [k, v] : std::vector<std::pair<const char*, const std::optional<std::string>*>>{{"time", &r.time}, {"unit", &r.unit}, {"cell", &r.cell}, {"derived_from", &r.derived_from},
                                                                                               {"description", &r.description}, {"by_operator", &r.by_operator}, {"producer", &r.producer}, {"by", &r.by}})
    if (*v) j[k] = **v;
  return j;
}
Json json_of(const AddStatisticRequest& r) {
  Json j = Json{{"dataset", r.dataset}, {"subject", r.subject}, {"name", r.name}, {"measure", r.measure}, {"method", r.method}, {"value_type", r.value_type}, {"axes", r.axes}, {"replace", r.replace},
                {"created_by", r.created_by}};
  for (const auto& [k, v] : std::vector<std::pair<const char*, const std::optional<std::string>*>>{{"of_field", &r.of_field}, {"partition", &r.partition}, {"unit", &r.unit}, {"description", &r.description}, {"by", &r.by},
                                                                                               {"producer", &r.producer}, {"declare", &r.declare}})
    if (*v) j[k] = **v;
  return j;
}
Json json_of(const AddMeasurementsRequest& r) {
  Json j = Json{{"dataset", r.dataset}, {"subject", r.subject}, {"selection", r.selection}, {"measure", r.measure}, {"declare", r.declare}, {"replace", r.replace}, {"created_by", r.created_by}};
  for (const auto& [k, v] : std::vector<std::pair<const char*, const std::optional<std::string>*>>{{"field", &r.field}, {"unit", &r.unit}, {"producer", &r.producer}, {"by", &r.by}, {"rows_on", &r.rows_on}})
    if (*v) j[k] = **v;
  return j;
}
std::optional<std::string> opt(const Json& j, const char* k) { return j.contains(k) ? std::optional<std::string>(j[k].get<std::string>()) : std::nullopt; }
std::vector<std::string> list(const Json& j, const char* k) { return j.contains(k) ? j[k].get<std::vector<std::string>>() : std::vector<std::string>(); }

AddFieldRequest field_of(const Json& j) {
  AddFieldRequest r;
  r.dataset = j["dataset"]; r.subject = j["subject"]; r.name = j["name"]; r.manifold = j["manifold"]; r.value_type = j["value_type"]; r.kind = j["kind"]; r.replace = j["replace"];
  r.contribution = list(j, "contribution"); r.created_by = j["created_by"];
  r.time = opt(j, "time"); r.unit = opt(j, "unit"); r.cell = opt(j, "cell"); r.derived_from = opt(j, "derived_from"); r.description = opt(j, "description"); r.by_operator = opt(j, "by_operator");
  r.producer = opt(j, "producer");
  r.by = opt(j, "by");
  return r;
}
AddStatisticRequest statistic_of(const Json& j) {
  AddStatisticRequest r;
  r.dataset = j["dataset"]; r.subject = j["subject"]; r.name = j["name"]; r.measure = j["measure"]; r.method = j["method"]; r.value_type = j["value_type"]; r.replace = j["replace"];
  r.axes = list(j, "axes"); r.created_by = j["created_by"];
  r.of_field = opt(j, "of_field"); r.partition = opt(j, "partition"); r.unit = opt(j, "unit"); r.description = opt(j, "description");
  r.producer = opt(j, "producer"); r.declare = opt(j, "declare");
  r.by = opt(j, "by");
  return r;
}
AddMeasurementsRequest measurements_of(const Json& j) {
  AddMeasurementsRequest r;
  r.dataset = j["dataset"]; r.subject = j["subject"]; r.selection = j["selection"]; r.replace = j["replace"]; r.created_by = j["created_by"];
  r.measure = list(j, "measure"); r.declare = list(j, "declare");
  r.field = opt(j, "field"); r.unit = opt(j, "unit"); r.producer = opt(j, "producer"); r.by = opt(j, "by"); r.rows_on = opt(j, "rows_on");
  return r;
}

// ---- the regions ----
// the parts of the shape no region covers, named (at most six), found on the grid the regions' edges cut the shape into
std::string missing_text(const std::vector<std::int64_t>& shape, const std::vector<Box>& regions) {
  const std::size_t rank = shape.size();
  std::vector<std::vector<std::int64_t>> cuts(rank);
  double cells = 1;
  for (std::size_t i = 0; i < rank; ++i) {
    std::set<std::int64_t> c = {0, shape[i]};
    for (const auto& b : regions) { c.insert(b[i][0]); c.insert(b[i][1]); }
    cuts[i].assign(c.begin(), c.end());
    cells *= static_cast<double>(cuts[i].size() - 1);
  }
  std::int64_t total = 1, covered = 0;
  for (const auto d : shape) total *= d;
  for (const auto& b : regions) covered += volume(b);
  std::string out = std::to_string(total - covered) + " of " + std::to_string(total) + " values";
  if (cells > 2e6) return out;
  std::vector<std::size_t> at(rank, 0);
  std::vector<std::string> named;
  std::size_t more = 0;
  while (true) {
    Box cell;
    for (std::size_t i = 0; i < rank; ++i) cell.push_back({cuts[i][at[i]], cuts[i][at[i] + 1]});
    bool hit = false;
    for (const auto& b : regions) if (overlap(b, cell)) { hit = true; break; }
    if (!hit) { if (named.size() < 6) named.push_back("[" + box_text(cell) + "]"); else ++more; }
    std::size_t k = rank;
    while (k > 0) {
      --k;
      if (++at[k] + 1 < cuts[k].size()) break;
      at[k] = 0;
      if (k == 0) { k = rank + 1; break; }
    }
    if (k == rank + 1) break;
  }
  std::string list;
  for (const auto& n : named) list += (list.empty() ? "" : ", ") + n;
  return out + ": " + list + (more ? " and " + std::to_string(more) + " more" : "");
}

// the region's values converted to the staged dtype, as bytes
template <class T>
std::vector<std::uint8_t> bytes_as(const ArrayInput& in) {
  const std::vector<T> v = array_all<T>(in);
  std::vector<std::uint8_t> b(v.size() * sizeof(T));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}
std::size_t itemsize(const std::string& dtype) { return dtype == "float64" || dtype == "int64" ? 8 : dtype == "float32" || dtype == "int32" ? 4 : 1; }
std::vector<std::uint8_t> region_bytes(const ArrayInput& in, const std::string& dtype) {
  if (dtype == "float32") return bytes_as<float>(in);
  if (dtype == "float64") return bytes_as<double>(in);
  if (dtype == "int32") return bytes_as<std::int32_t>(in);
  if (dtype == "int64") return bytes_as<std::int64_t>(in);
  return bytes_as<std::uint8_t>(in);
}

// the region's values written into the staged file, in C order, in as few runs as the region allows: the trailing axes the region covers in full are
// merged into one run with the first axis it cuts (a block of whole rows of a [N x 2] complex field is ONE run, not N runs of two numbers)
void write_region(const fs::path& file, std::uint64_t offset, std::size_t item, const std::vector<std::int64_t>& shape, const Box& box, const std::vector<std::uint8_t>& bytes) {
  const std::size_t rank = shape.size();
  std::vector<std::uint64_t> stride(rank, 1);
  for (std::size_t i = rank - 1; i > 0; --i) stride[i - 1] = stride[i] * static_cast<std::uint64_t>(shape[i]);
  std::size_t inner = rank - 1;   // the innermost axis the region cuts; every axis after it is covered in full
  while (inner > 0 && box[inner][0] == 0 && box[inner][1] == shape[inner]) --inner;
  const std::uint64_t run = static_cast<std::uint64_t>(box[inner][1] - box[inner][0]) * stride[inner] * item;
  std::FILE* f = std::fopen(file.c_str(), "r+b");
  if (!f) throw Error(code::not_writable, file.string() + ": cannot be opened");
  std::vector<std::int64_t> at(rank, 0);
  for (std::size_t i = 0; i <= inner; ++i) at[i] = box[i][0];
  std::uint64_t from = 0;
  bool ok = true;
  while (ok) {
    std::uint64_t cell = 0;
    for (std::size_t i = 0; i <= inner; ++i) cell += static_cast<std::uint64_t>(at[i]) * stride[i];
    if (::fseeko(f, static_cast<off_t>(offset + cell * item), SEEK_SET) != 0 || std::fwrite(bytes.data() + from, 1, run, f) != run) ok = false;
    from += run;
    std::size_t k = inner;   // the next run: the axes before `inner` advance
    bool done = true;
    while (k > 0) {
      --k;
      if (++at[k] < box[k][1]) { done = false; break; }
      at[k] = box[k][0];
    }
    if (done) break;
  }
  if (std::fclose(f) != 0) ok = false;
  if (!ok) throw Error(code::not_writable, file.string() + ": a short write");
}

// ---- finding a write ----
std::string owner_token(const Json& m) { return m.contains("owner") && m["owner"].contains("handle") && m["owner"]["handle"].is_string() ? m["owner"]["handle"].get<std::string>() : std::string(); }

struct Found {
  fs::path dir;
  Json manifest;
  std::string id;
};
Found find_write(const std::string& dataset, const std::optional<std::string>& write, const std::string& cls) {
  const std::optional<std::string> token = current_handle_token(dataset);
  std::string id;
  if (write) {
    id = *write;
  } else {
    if (!token) throw Refused("name the write: --write ID (the id `begin` gave)");
    std::vector<std::string> mine;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(root_of(dataset), ec)) {
      const Json m = load(e.path());
      if (m.is_object() && owner_token(m) == *token) mine.push_back(e.path().filename().string());
    }
    std::sort(mine.begin(), mine.end());
    if (mine.empty()) throw Error(code::no_such_row, "this handle holds no unfinished write: begin one (add " + cls + " with begin)");
    if (mine.size() > 1) {
      std::string all;
      for (const auto& m : mine) all += (all.empty() ? "" : ", ") + m;
      throw Refused("this handle holds several unfinished writes (" + all + "): name one with write", true);
    }
    id = mine.front();
  }
  if (!is_id(id)) throw Error(code::no_such_row, "'" + id + "' is not a write id");
  Found f{root_of(dataset) / id, Json(), id};
  f.manifest = load(f.dir);
  if (!f.manifest.is_object()) throw Error(code::no_such_row, "no unfinished write '" + id + "' in " + dataset + " (it was committed, rolled back, or is another dataset's)");
  const std::string owner = owner_token(f.manifest);
  if (!owner.empty() && (!token || *token != owner))
    throw Refused("write " + id + " belongs to another handle (the one that began it): use that handle, or `recover dataset` to roll it back", true);
  if (f.manifest["class"] != cls) throw Refused("write " + id + " is a write of a " + f.manifest["class"].get<std::string>() + ", not a " + cls, true);
  return f;
}

// ---- the three steps, for a field, a statistic or one measure's measurements ----
// what the three request kinds share, read the same way: the values given, and the node's name (a measurements write is named by its one measure)
bool has_values(const AddFieldRequest& r) { return r.values.in_memory || !r.values.path.empty(); }
bool has_values(const AddStatisticRequest& r) { return r.values.in_memory || !r.values.path.empty(); }
bool has_values(const AddMeasurementsRequest& r) {
  for (const auto& m : r.measure) if (m.find('=') != std::string::npos) return true;
  return !r.measure_arrays.empty();
}
const ArrayInput& values_of_request(const AddFieldRequest& r) { return r.values; }
const ArrayInput& values_of_request(const AddStatisticRequest& r) { return r.values; }
const ArrayInput& values_of_request(const AddMeasurementsRequest& r) {
  static thread_local ArrayInput from_file;
  if (r.measure_arrays.size() == 1 && r.measure.empty()) return r.measure_arrays.front().second;
  if (r.measure.size() == 1 && r.measure_arrays.empty()) {
    const std::string& m = r.measure.front();
    const auto eq = m.find('=');
    if (eq == std::string::npos) throw Refused("append takes the region's values as --values FILE.npy (or --measure NAME=FILE.npy)");
    from_file = ArrayInput(m.substr(eq + 1));
    return from_file;
  }
  throw Refused("append writes the values of ONE measure's region: --values FILE.npy");
}
std::string name_of(const AddFieldRequest& r) { return r.name; }
std::string name_of(const AddStatisticRequest& r) { return r.name; }
std::string name_of(const AddMeasurementsRequest& r) { return r.measure.empty() ? std::string() : r.measure.front(); }

template <class Req>
CommandResult run(const Req& rq, const std::string& cls, const Progress& progress) {
  const PieceOptions& p = rq.piece;
  if ((p.begin ? 1 : 0) + (p.append ? 1 : 0) + (p.commit ? 1 : 0) != 1) throw Refused("a write in pieces is one step at a time: begin, append or commit");
  const std::string dataset = rq.dataset;
  CommandResult out;

  if (p.begin) {
    if (p.write || p.region) throw Refused("begin gives the write id: it takes no write and no region");
    if (has_values(rq)) throw Refused("begin takes the shape and dtype of the values, not the values (append writes them)");
    if (!p.shape || p.shape->empty()) throw Refused("begin needs the full shape of the values: --shape N,N,...");
    if (!p.dtype) throw Refused("begin needs the dtype the values are stored in: --dtype float32|float64|int32|int64|uint8|bool");
    static const std::set<std::string> dtypes = {"float32", "float64", "int32", "int64", "uint8", "bool"};
    if (!dtypes.count(*p.dtype)) throw Refused("--dtype " + *p.dtype + " (float32, float64, int32, int64, uint8 or bool)");
    for (const auto d : *p.shape) if (d < 1) throw Refused("--shape " + shape_text(*p.shape) + ": every axis has at least one element");
    if constexpr (std::is_same_v<Req, AddFieldRequest>) plan_field(rq, *p.shape, progress);
    else if constexpr (std::is_same_v<Req, AddStatisticRequest>) plan_statistic(rq, *p.shape, progress);
    else plan_measurements(rq, *p.shape, progress);
    discard_orphans(dataset);
    const std::string id = new_id();
    const fs::path dir = root_of(dataset) / id;
    fs::create_directories(dir);
    try {
      const std::uint64_t offset = create_npy(dir / "values.npy", *p.dtype, *p.shape);
      const std::optional<std::string> token = current_handle_token(dataset);
      Json m = Json{{"id", id}, {"class", cls}, {"request", json_of(rq)}, {"shape", *p.shape}, {"dtype", *p.dtype}, {"offset", offset}, {"regions", Json::array()},
                    {"owner", Json{{"handle", token ? Json(*token) : Json()}, {"pid", static_cast<long long>(::getpid())}}}, {"created_utc", now_utc()}};
      save(dir, m);
    } catch (...) {
      std::error_code ec;
      fs::remove_all(dir, ec);
      throw;
    }
    out.result = Json{{"write", id}, {"class", cls}, {"name", name_of(rq)}, {"shape", *p.shape}, {"dtype", *p.dtype}, {"step", "begin"}};
    return out;
  }

  Found w = find_write(dataset, p.write, cls);
  Held held(w.dir);
  w.manifest = load(w.dir);   // read again under the hold: another call may have written a region meanwhile
  const Json& req = w.manifest["request"];
  if (!rq.subject.empty() && rq.subject != req["subject"].get<std::string>()) throw Refused("write " + w.id + " is of subject " + req["subject"].get<std::string>() + ", not " + rq.subject, true);
  if constexpr (std::is_same_v<Req, AddMeasurementsRequest>) {
    if (!rq.selection.empty() && rq.selection != req["selection"].get<std::string>()) throw Refused("write " + w.id + " is of the selection " + req["selection"].get<std::string>() + ", not " + rq.selection, true);
  } else if (!rq.name.empty() && rq.name != req["name"].get<std::string>()) {
    throw Refused("write " + w.id + " is of the " + cls + " " + req["name"].get<std::string>() + ", not " + rq.name, true);
  }
  const std::vector<std::int64_t> shape = w.manifest["shape"].get<std::vector<std::int64_t>>();
  const std::string dtype = w.manifest["dtype"];

  if (p.append) {
    if (p.shape || p.dtype) throw Refused("append takes the region and its values; the shape and dtype are begin's");
    if (!p.region) throw Refused("append needs the region: --region START:STOP per axis, comma separated (one entry for each of the " + std::to_string(shape.size()) + " axes)");
    if (!has_values(rq)) throw Refused("append needs the values of the region (--values)");
    Box box(p.region->begin(), p.region->end());
    if (box.size() != shape.size()) throw Refused("--region has " + std::to_string(box.size()) + " axes; the values of write " + w.id + " are " + shape_text(shape) + " (" + std::to_string(shape.size()) + " axes)");
    std::vector<std::int64_t> want;
    for (std::size_t i = 0; i < box.size(); ++i) {
      if (box[i][0] < 0 || box[i][1] > shape[i] || box[i][0] >= box[i][1])
        throw Refused("--region " + box_text(box) + " is not inside the shape " + shape_text(shape) + ": axis " + std::to_string(i) + " runs [0, " + std::to_string(shape[i]) + ") and a region has start < stop");
      want.push_back(box[i][1] - box[i][0]);
    }
    std::vector<Box> regions;
    for (const auto& r : w.manifest["regions"]) regions.push_back(box_of(r));
    for (const auto& r : regions) {
      if (r == box) throw Refused("region [" + box_text(box) + "] of write " + w.id + " was already written (a region is written once)", true);
      if (overlap(r, box)) throw Refused("region [" + box_text(box) + "] overlaps region [" + box_text(r) + "] already written in write " + w.id + " (regions do not overlap)", true);
    }
    const ArrayInput& values = values_of_request(rq);
    const std::vector<std::int64_t> got = array_shape(values);
    if (got != want) throw Refused("--values " + values.label() + " is " + shape_text(got) + "; region [" + box_text(box) + "] is " + shape_text(want));
    const std::string have = array_dtype(values);
    if (have != dtype) throw Refused("--values " + values.label() + " is " + have + "; write " + w.id + " stores " + dtype + " (the dtype begin named)");
    write_region(w.dir / "values.npy", w.manifest["offset"].get<std::uint64_t>(), itemsize(dtype), shape, box, region_bytes(values, dtype));
    w.manifest["regions"].push_back(box_json(box));
    save(w.dir, w.manifest);
    regions.push_back(box);
    std::int64_t covered = 0, total = 1;
    for (const auto& r : regions) covered += volume(r);
    for (const auto d : shape) total *= d;
    out.result = Json{{"write", w.id}, {"step", "append"}, {"region", box_json(box)}, {"values", volume(box)}, {"regions", regions.size()}, {"written", covered}, {"of", total}};
    return out;
  }

  // commit
  if (p.shape || p.dtype || p.region) throw Refused("commit takes only the write");
  if (has_values(rq)) throw Refused("commit takes no values: append wrote them");
  std::vector<Box> regions;
  for (const auto& r : w.manifest["regions"]) regions.push_back(box_of(r));
  std::int64_t covered = 0, total = 1;
  for (const auto& r : regions) covered += volume(r);
  for (const auto d : shape) total *= d;
  if (covered != total) throw Refused("write " + w.id + " is not complete: " + missing_text(shape, regions) + " missing; append them, then commit", true);
  CommandResult r;
  if constexpr (std::is_same_v<Req, AddFieldRequest>) {
    AddFieldRequest q = field_of(req);
    q.dataset = dataset;
    q.values = ArrayInput((w.dir / "values.npy").string());
    if (!rq.created_by.empty()) q.created_by = rq.created_by;
    if (rq.by) q.by = rq.by;
    r = add_field(q, progress);
  } else if constexpr (std::is_same_v<Req, AddStatisticRequest>) {
    AddStatisticRequest q = statistic_of(req);
    q.dataset = dataset;
    q.values = ArrayInput((w.dir / "values.npy").string());
    if (!rq.created_by.empty()) q.created_by = rq.created_by;
    if (rq.by) q.by = rq.by;
    r = add_statistic(q, progress);
  } else {
    AddMeasurementsRequest q = measurements_of(req);
    q.dataset = dataset;
    const std::string measure = q.measure.front();
    q.measure.clear();
    q.measure_arrays = {{measure, ArrayInput((w.dir / "values.npy").string())}};
    if (!rq.created_by.empty()) q.created_by = rq.created_by;
    if (rq.by) q.by = rq.by;
    r = add_measurements(q, progress);
  }
  r.result["write"] = w.id;
  r.result["step"] = "commit";
  r.result["regions"] = regions.size();
  std::error_code ec;
  fs::remove_all(w.dir, ec);   // stored: the pieces are no longer needed (a leftover folder is a write `recover dataset` rolls back)
  return r;
}

}  // namespace

CommandResult field(const AddFieldRequest& request, const Progress& progress) { return run(request, "field", progress); }
CommandResult statistic(const AddStatisticRequest& request, const Progress& progress) { return run(request, "statistic", progress); }
CommandResult measurements(const AddMeasurementsRequest& request, const Progress& progress) { return run(request, "measurements", progress); }

int discard_all(const std::string& dataset) {
  int n = 0;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(root_of(dataset), ec)) {
    if (!e.is_directory()) continue;
    fs::remove_all(e.path(), ec);
    ++n;
  }
  fs::remove(root_of(dataset), ec);
  return n;
}

int discard_owned(const std::string& dataset, const std::string& token) {
  int n = 0;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(root_of(dataset), ec)) {
    const Json m = load(e.path());
    if (m.is_object() && owner_token(m) == token) { fs::remove_all(e.path(), ec); ++n; }
  }
  fs::remove(root_of(dataset), ec);   // only when empty
  return n;
}

int discard_orphans(const std::string& dataset) {
  int n = 0;
  std::error_code ec;
  if (!fs::exists(root_of(dataset), ec)) return 0;
  for (const auto& e : fs::directory_iterator(root_of(dataset), ec)) {
    if (!e.is_directory()) continue;
    const Json m = load(e.path());
    bool orphan = false;
    if (!m.is_object()) orphan = file_age_seconds(e.path()) > 120;   // a begin that never finished
    else if (!owner_token(m).empty() && m.contains("owner") && m["owner"].contains("pid") && process_gone(m["owner"]["pid"].get<long long>())) orphan = true;   // its handle's process is gone
    if (orphan) { fs::remove_all(e.path(), ec); ++n; }
  }
  fs::remove(root_of(dataset), ec);
  return n;
}

int count(const std::string& dataset) {
  int n = 0;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(root_of(dataset), ec)) if (e.is_directory()) ++n;
  return n;
}

}  // namespace nxr::io::task::pieces
