// SubjectWriter and the entities — nxr.io.writer.SubjectWriter, coordinate_system, product_counts.
#include <algorithm>
#include <cmath>
#include <regex>
#include <set>
#include <unordered_set>

#include "nxr/io/layout.h"
#include "nxr/io/writer.h"
#include "nxr/io/statistic.h"
#include "nxr/io/measures.h"

namespace fs = std::filesystem;

namespace nxr::io {

namespace {
Error value_error(const std::string& m) { return Error(code::invalid_argument, m); }
Error file_exists(const std::string& m) { return Error(code::already_exists, m); }
Error key_error(const std::string& m) { return Error(code::no_such_row, m); }
bool in_status_tables(const std::string& t) {
  return std::any_of(std::begin(kStatusTables), std::end(kStatusTables), [&](const char* x) { return t == x; });
}
void merge(Row& into, const Row& from) { for (const auto& [k, v] : from) into[k] = v; }   // a later dict wins, as `{**a, **b}`
Value vint(std::int64_t x) { return x; }
Value vopt(const std::optional<std::string>& s) { return V(s); }
std::string id_or_new(const std::optional<std::string>& id) { return id ? *id : uuid7(); }
}  // namespace

// ══ BIDS ═════════════════════════════════════════════════════════════════════════════════════════════════════════
std::map<std::string, std::optional<std::string>> bids_entities(const std::string& label) {
  std::map<std::string, std::optional<std::string>> out;
  for (const char* e : {"ses", "task", "run", "acq", "proc", "split"}) {
    const std::regex re(std::string("(?:^|[^A-Za-z0-9])") + e + "-([A-Za-z0-9]+)");
    std::smatch m;
    if (std::regex_search(label, m, re)) out[e] = m[1].str();
    else out[e] = std::nullopt;
  }
  return out;
}

// ══ COORDINATE SYSTEMS (spacesFor, D25–D28) ══════════════════════════════════════════════════════════════════════
namespace {
struct Sys {
  std::string key, name;
  std::optional<std::string> unit;
  bool reference = false;
  std::string transition = "identity";
  std::optional<Json> params;
  std::optional<std::string> description;
};
struct SpaceSpec {
  std::string space, geometry, description;
  std::vector<Sys> systems;
};

constexpr double kTwoPi = 6.283185307179586;  // 2 * math.pi

// (space name, geometry, description, the systems it is created with — reference first) for a key: `spacesFor` of `database.ts`
SpaceSpec space_spec(const std::string& key) {
  auto starts = [&](const char* p) { return key.rfind(p, 0) == 0; };
  if (key == "scs" || key == "ncs" || key == "mni" || key == "freesurfer-tkr" || starts("voxel:")) {
    SpaceSpec s{"head", "volume", "R³ around the subject's head", {}};
    s.systems.push_back({"scs", "scs", std::string("m"), true, "identity", std::nullopt, std::string("Brainstorm's subject coordinate system: origin between the ears, x toward the nasion")});
    if (key == "ncs") s.systems.push_back({"ncs", "ncs", std::string("mm"), false, "affine", std::nullopt, std::string("Brainstorm's normalised coordinate system (AC · PC · IH)")});
    if (key == "freesurfer-tkr") s.systems.push_back({key, key, std::string("m"), false, "affine", std::nullopt, std::string("FreeSurfer's surface RAS (tkr) in metres, as its own surfaces are stored")});
    if (key == "mni") s.systems.push_back({"mni", "mni", std::string("mm"), false, "affine", std::nullopt, std::string("MNI template space")});
    if (starts("voxel:")) s.systems.push_back({key, key, std::string("voxel"), false, "affine", std::nullopt, key.substr(6) + "'s voxel indices; the transition to scs is its voxel→SCS affine"});
    return s;
  }
  if (key == "freesurfer-sphere")
    return {"sphere", "volume", "the R³ FreeSurfer's registration sphere sits in (its positions are stored as three numbers each)",
            {{key, "freesurfer", std::nullopt, true, "identity", std::nullopt, std::string("radius 100")}}};
  if (key == "frequency" || key == "octave") {
    SpaceSpec s{"frequency", "line", "the frequency line, R¹ — what a time → frequency operator maps the time line onto", {}};
    s.systems.push_back({"frequency", "hz", std::string("Hz"), true, "identity", std::nullopt, std::string("cycles per second")});
    if (key == "octave") s.systems.push_back({"octave", "octave", std::string("log2 Hz"), false, "log", Json{{"base", 2}}, std::string("log2 of the frequency in hertz")});
    return s;
  }
  if (key == "clock")
    return {"time", "line", "the time line, R¹", {{key, "clock", std::string("s"), true, "identity", std::nullopt, std::string("the recording's clock")}}};
  if (starts("eigenvalue:")) {
    const std::string op = key.substr(11);
    std::string lo = op;
    std::transform(lo.begin(), lo.end(), lo.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool dirac = lo.find("dirac") != std::string::npos;
    SpaceSpec s{"spectrum:" + op, "line", "the spectral line of the " + op + " operator — its eigenvalues, in " + (dirac ? "1/m" : "1/m²"), {}};
    s.systems.push_back({key, "eigenvalue", std::string(dirac ? "1/m" : "1/m²"), true, "identity", std::nullopt, std::string("λ")});
    s.systems.push_back({"wavelength:" + op, "wavelength", std::string("m"), false, "power",
                         dirac ? Json{{"scale", kTwoPi}, {"exponent", -1}, {"abs", true}} : Json{{"scale", kTwoPi}, {"exponent", -0.5}},
                         std::string(dirac ? "2π / |λ| — the spatial wavelength of a mode" : "2π / √λ — the spatial wavelength of a mode")});
    return s;
  }
  return {key, "volume", "the " + key + " space", {{key, key, std::nullopt, true, "identity", std::nullopt, std::nullopt}}};
}

int space_rank(const std::string& g) {
  if (g == "line" || g == "circle") return 1;
  if (g == "plane" || g == "sphere" || g == "torus") return 2;
  return 3;
}
}  // namespace

std::string coordinate_system(Database& db, const std::string& subject_id, const std::string& key) {
  const SpaceSpec sp_spec = space_spec(key);
  const Sys* want = nullptr;
  for (const auto& s : sp_spec.systems) if (s.key == key) { want = &s; break; }
  if (!want) throw value_error("coordinate_system: " + key);
  const auto sp = db.one("SELECT id FROM space WHERE subject_id = ? AND name = ?", {subject_id, sp_spec.space});
  std::string sid;
  db.tx([&] {
    if (!sp) {
      sid = uuid7();
      db.insert("space", {{"id", sid}, {"subject_id", subject_id}, {"name", sp_spec.space}, {"geometry", sp_spec.geometry}, {"rank", V(space_rank(sp_spec.geometry))},
                          {"parameters_json", std::monostate{}}, {"description", sp_spec.description}});
    } else {
      sid = text(*sp, "id");
    }
    std::vector<const Sys*> todo;
    if (!sp) for (const auto& s : sp_spec.systems) todo.push_back(&s);
    else { todo.push_back(&sp_spec.systems[0]); todo.push_back(want); }
    for (const Sys* s : todo) {
      if (!db.one("SELECT id FROM coordinate_system WHERE space_id = ? AND name = ?", {sid, s->name})) {
        db.insert("coordinate_system", {{"id", uuid7()}, {"space_id", sid}, {"name", s->name}, {"unit", vopt(s->unit)}, {"is_reference", V(s->reference ? 1 : 0)},
                                        {"transition", s->reference ? std::string("identity") : s->transition},
                                        {"transition_json", s->params ? Value(jdump(*s->params)) : Value(std::monostate{})}, {"description", vopt(s->description)}});
      }
    }
  });
  return text(*db.one("SELECT id FROM coordinate_system WHERE space_id = ? AND name = ?", {sid, want->name}), "id");
}

// ══ THE f-VECTOR ═════════════════════════════════════════════════════════════════════════════════════════════════
FVector lattice_counts(const std::vector<std::int64_t>& ns) {
  const std::size_t n = ns.size();
  auto prodex = [&](std::initializer_list<std::size_t> skip) {
    std::int64_t p = 1;
    for (std::size_t k = 0; k < n; ++k) if (std::find(skip.begin(), skip.end(), k) == skip.end()) p *= ns[k];
    return p;
  };
  std::int64_t edges = 0, faces = 0;
  for (std::size_t i = 0; i < n; ++i) edges += (ns[i] - 1) * prodex({i});
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j) faces += (ns[i] - 1) * (ns[j] - 1) * prodex({i, j});
  FVector f;
  f.n_edges = edges;
  if (n >= 2) f.n_faces = faces;
  if (n >= 3) { std::int64_t v = 1; for (auto x : ns) v *= x - 1; f.n_volumes = v; }
  return f;
}

FVector product_counts(const std::vector<Factor>& factors) {
  using O = std::optional<std::int64_t>;
  std::vector<O> poly{1};
  int rank = 0;
  for (const auto& f : factors) {
    const O g[4] = {f.n_vertices, f.rank >= 1 ? f.n_edges : O(0), f.rank >= 2 ? f.n_faces : O(0), f.rank >= 3 ? f.n_volumes : O(0)};
    O out[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < poly.size(); ++i) {
      for (int j = 0; j < 4; ++j) {
        if (i + j >= 4) continue;
        const O a = poly[i], b = g[j], cur = out[i + j];
        // (cur if (a == 0 or b == 0) else None) if (cur is None or a is None or b is None) else cur + a * b
        if (!cur || !a || !b) out[i + j] = (a == O(0) || b == O(0)) ? cur : O();
        else out[i + j] = *cur + *a * *b;
      }
    }
    poly.assign(out, out + 4);
    rank += f.rank;
  }
  auto at = [&](int k) -> std::optional<std::int64_t> { return k > rank ? std::nullopt : poly[static_cast<std::size_t>(k)]; };
  FVector r;
  r.n_vertices = poly[0];
  r.n_edges = at(1);
  r.n_faces = at(2);
  r.n_volumes = at(3);
  r.rank = rank;
  return r;
}

// ══ A SUBJECT — the composition ══════════════════════════════════════════════════════════════════════════════════
namespace {
using Color = std::array<int, 3>;
const std::map<std::string, std::map<std::string, Color>>& family_colors() {
  static const std::map<std::string, std::map<std::string, Color>> m = {
      {"hemisphere", {{"L", {76, 120, 168}}, {"R", {228, 87, 86}}, {"Z", {157, 157, 157}}, {"-", {107, 107, 107}}}},
      {"lobe", {{"F", {242, 142, 43}}, {"C", {89, 161, 79}}, {"P", {76, 120, 168}}, {"O", {176, 122, 161}}, {"T", {237, 201, 72}}, {"-", {107, 107, 107}}}},
      {"type", {{"MEG", {143, 211, 244}}, {"MEG REF", {157, 117, 93}}, {"MEG MAG", {143, 211, 244}}, {"MEG GRAD", {118, 183, 178}}, {"EEG", {118, 183, 178}},
                {"ECG", {228, 87, 86}}, {"EOG", {255, 157, 167}}, {"VEOG", {255, 157, 167}}, {"HEOG", {255, 190, 120}}, {"EMG", {186, 176, 172}},
                {"SYSCLOCK", {107, 107, 107}}, {"STIM", {157, 157, 157}}, {"MISC", {157, 157, 157}}}},
      {"quality", {{"good", {90, 170, 110}}, {"bad", {228, 87, 86}}}},
  };
  return m;
}

int n_components_of(const std::string& value_type) {
  if (value_type == "scalar" || value_type == "integer") return 1;
  if (value_type == "complex" || value_type == "vector2") return 2;
  if (value_type == "vector3") return 3;
  if (value_type == "matrix3") return 9;
  if (value_type == "quaternion") return 4;
  if (value_type == "hermitian3") return 18;                // a complex 3×3 Hermitian matrix a cell: [.., 3, 3, 2], real and imaginary parts
  throw Error(code::no_such_row, value_type);
}

// The members of a partition: its distinct codes, a NEGATIVE code excluded (-1 is in no member: the partition arithmetic's rule, nxr::partition)
std::int64_t distinct_count(const ArrayView& a) {
  const std::size_t n = static_cast<std::size_t>(a.size());
  auto count = [&](auto* p) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(p)>>;
    std::set<T> s;
    for (std::size_t i = 0; i < n; ++i) if (!(p[i] < T(0))) s.insert(p[i]);
    return static_cast<std::int64_t>(s.size());
  };
  switch (a.dtype) {
    case DType::Bool: case DType::UInt8: return count(static_cast<const std::uint8_t*>(a.data));
    case DType::Int8: return count(static_cast<const std::int8_t*>(a.data));
    case DType::Int16: return count(static_cast<const std::int16_t*>(a.data));
    case DType::UInt16: return count(static_cast<const std::uint16_t*>(a.data));
    case DType::Int32: return count(static_cast<const std::int32_t*>(a.data));
    case DType::UInt32: return count(static_cast<const std::uint32_t*>(a.data));
    case DType::Int64: return count(static_cast<const std::int64_t*>(a.data));
    case DType::UInt64: return count(static_cast<const std::uint64_t*>(a.data));
    case DType::Float32: return count(static_cast<const float*>(a.data));
    case DType::Float64: return count(static_cast<const double*>(a.data));
    default: throw Error(code::unsupported, "selection elements: this dtype has no distinct count");
  }
}
}  // namespace

SubjectWriter::SubjectWriter(DatasetWriter& ds, Row row, bool immediate, bool owns_subject, std::optional<std::string> created_by)
    : ds_(&ds), db_(&ds.db()), root_(ds.root()), row_(std::move(row)), immediate_(immediate), owns_subject_(owns_subject), created_by_(std::move(created_by)) {
  id_ = text(row_, "id");
  name_ = text(row_, "name");
  rel_ = text(ds.row(), "path") + "/" + text(row_, "path");
  store_ = ds.is_package() ? root_ : root_ / rel_;   // a package writer's root is the package folder itself
  kv_ = std::make_unique<PrefixStore>(ds.node_store(), rel_);
}

SubjectWriter SubjectWriter::create(DatasetWriter& ds, const std::string& name, const SubjectOptions& opt) {
  const std::string folder = opt.folder.value_or(name + ".nxr.zarr");
  if (ds.db().one("SELECT 1 FROM subject WHERE dataset_id = ? AND (path = ? OR name = ?)", {ds.id(), folder, name}))
    throw file_exists(name + ": already a subject of " + text(ds.row(), "name") + " — delete it first (a subject is converted whole)");
  ds.node_store().erase_prefix(join_key(text(ds.row(), "path"), folder));  // debris of a crash
  std::optional<std::string> template_id = opt.template_id;
  if (!template_id && opt.default_template) {
    const auto t = opt.kind == "subject" ? ds.template_() : std::nullopt;
    if (t) template_id = text(*t, "id");
  }
  const std::string now = now_utc(false);
  const std::string sid = uuid7();
  Row row = {{"id", sid}, {"dataset_id", ds.id()}, {"name", name}, {"path", folder}, {"kind", opt.kind}, {"template_id", V(template_id)},
             {"source_format", V(opt.source_format)}, {"source_path", V(opt.source_path)}, {"created_utc", now}, {"created_by", V(opt.created_by)},
             {"modified_utc", now}, {"modified_by", V(opt.created_by)}};
  if (opt.immediate) {
    create_node(ds.db(), ds.node_store(), "subject", sid, [&] { ds.db().insert("subject", row); });
    return SubjectWriter(ds, *ds.db().read("subject", sid), true, true, opt.created_by);
  }
  ds.db().tx([&] {
    Row r = row;
    r["status"] = "pending";
    ds.db().insert("subject", r);
  });
  SubjectWriter sub(ds, *ds.db().read("subject", sid), false, true, opt.created_by);
  ensure_groups(ds.node_store(), sub.rel_);
  sub.created_.push_back({"subject", sid});
  return sub;
}

SubjectWriter SubjectWriter::open(DatasetWriter& ds, const std::string& name, const OpenSubjectOptions& opt) {
  const auto row = ds.db().one("SELECT * FROM subject WHERE dataset_id = ? AND (name = ? OR path = ? OR id = ?) AND status = 'complete'", {ds.id(), name, name, name});
  if (!row) throw key_error(name + ": no subject of " + text(ds.row(), "name"));
  return SubjectWriter(ds, *row, opt.immediate, false, opt.created_by);
}

Drained SubjectWriter::complete() {
  db_->tx([&] {
    for (const auto& [table, id] : created_) if (in_status_tables(table)) set_status(*db_, table, id, "complete");
  });
  Drained out = drain(*db_, ds_->node_store());
  created_.clear();
  if (owns_subject_ && !ds_->is_package()) write_catalog(root_);
  return out;
}

void SubjectWriter::abort() {
  std::vector<std::string> dicts, acqs;
  for (const auto& [t, id] : created_) { if (t == "dictionary") dicts.push_back(id); if (t == "acquisition") acqs.push_back(id); }
  if (owns_subject_) {
    remove_node(*db_, ds_->node_store(), "subject", id_);
  } else {
    for (auto it = created_.rbegin(); it != created_.rend(); ++it)
      if (in_status_tables(it->first)) remove_node(*db_, ds_->node_store(), it->first, it->second);
  }
  drop_unused(dicts, acqs);
  created_.clear();
  if (owns_subject_ && !ds_->is_package()) write_catalog(root_);
}

void SubjectWriter::compose(const std::function<void()>& body) {
  try {
    body();
  } catch (...) {
    try { abort(); } catch (...) {}
    throw;
  }
  complete();
}

void SubjectWriter::drop_unused(const std::vector<std::string>& dictionary_ids, const std::vector<std::string>& acquisition_ids) {
  db_->tx([&] {
    for (const auto& did : dictionary_ids)
      if (!db_->one("SELECT 1 FROM selection WHERE dictionary_id = ? LIMIT 1", {did})) db_->execute("DELETE FROM dictionary WHERE id = ?", {did});
    for (const auto& aid : acquisition_ids) {  // an acquisition this composition made, which no node it kept names
      bool used = false;
      for (const char* t : {"manifold", "field", "selection", "operator"})
        if (db_->one(std::string("SELECT 1 FROM ") + t + " WHERE acquisition_id = ? LIMIT 1", {aid})) { used = true; break; }
      if (!used) db_->execute("DELETE FROM acquisition WHERE id = ?", {aid});
    }
  });
}

std::size_t SubjectWriter::rollback_to(std::size_t m) {
  std::vector<std::pair<std::string, std::string>> undone(created_.begin() + static_cast<std::ptrdiff_t>(m), created_.end());
  for (auto it = undone.rbegin(); it != undone.rend(); ++it)
    if (in_status_tables(it->first) && !(it->first == "subject" && it->second == id_)) remove_node(*db_, ds_->node_store(), it->first, it->second);
  std::vector<std::string> dicts, acqs;
  for (const auto& [t, id] : undone) { if (t == "dictionary") dicts.push_back(id); if (t == "acquisition") acqs.push_back(id); }
  drop_unused(dicts, acqs);
  created_.resize(m);
  return undone.size();
}

std::string SubjectWriter::node(const std::string& table, const std::string& id, const std::function<void()>& rows, const Populate& populate) {
  if (immediate_) {
    create_node(*db_, ds_->node_store(), table, id, rows, populate);
    return id;
  }
  db_->tx([&] {
    rows();
    set_status(*db_, table, id, "pending");
  });
  created_.push_back({table, id});
  if (populate) {
    const auto at = location_of(*db_, table, id);
    try {
      populate(ds_->node_store(), at ? *at : std::string());
    } catch (...) {
      remove_node(*db_, ds_->node_store(), table, id);
      created_.erase(std::find(created_.begin(), created_.end(), std::make_pair(table, id)));
      throw;
    }
  }
  return id;
}

std::optional<Row> SubjectWriter::find(const std::string& table, const std::string& path) const {
  if (table == "statistic" && !has_table(*db_, table)) return std::nullopt;   // a dataset made before the statistic table has none
  return db_->one("SELECT * FROM " + table + " WHERE subject_id = ? AND path = ?", {id_, path});
}
std::optional<Row> SubjectWriter::by_name(const std::string& table, const std::string& name) const {
  if (table == "statistic" && !has_table(*db_, table)) return std::nullopt;
  return db_->one("SELECT * FROM " + table + " WHERE subject_id = ? AND name = ?", {id_, name});
}
void SubjectWriter::remove(const std::string& table, const std::string& id) { remove_node(*db_, ds_->node_store(), table, id); }

ArrayMeta SubjectWriter::write_array(const std::string& path, const ArrayView& data, const ArrayOptions& opt, const std::optional<ArrayMeta>& meta_in,
                                     const std::string& table, const std::string& id) {
  const ArrayMeta meta = meta_in ? *meta_in : array_meta(data.dtype, data.shape, opt);
  ensure_groups(*kv_, path, false);
  const auto text_ = !table.empty() && !id.empty() ? attributes_text(*db_, table, id) : std::nullopt;
  layout_array(*kv_, path, meta, text_);
  populate_array(*kv_, path, data, opt.dense);
  return meta;
}

void SubjectWriter::write_sparse(const std::string& path, const SparseArrays& parts) {
  for (const auto& [part, arr] : parts.parts) write_array(path + "/" + part, arr);
}

std::string SubjectWriter::cs(const std::string& key) { return coordinate_system(*db_, id_, key); }

std::optional<std::string> SubjectWriter::acquisition(const std::optional<std::string>& label) {
  if (!label) return std::nullopt;
  if (const auto row = db_->one("SELECT id FROM acquisition WHERE subject_id = ? AND label = ?", {id_, *label})) return text(*row, "id");
  const std::string aid = uuid7();
  Row r = {{"id", aid}, {"subject_id", id_}, {"label", *label}};
  for (const auto& [k, v] : bids_entities(*label)) r[k] = V(v);
  db_->insert("acquisition", r);
  created_.push_back({"acquisition", aid});  // an abort removes it again when no row it kept names it
  return aid;
}

Cols SubjectWriter::acquired(Cols cols, const std::optional<std::string>& label) {
  if (label) {
    const auto aid = acquisition(label);
    if (!cols.count("acquisition_id")) cols.set("acquisition_id", *aid);
  }
  return cols;
}

Row SubjectWriter::prov(const std::optional<std::string>& created_utc) const {
  return {{"created_utc", V(created_utc)}, {"created_by", V(created_by_)}, {"modified_utc", V(created_utc)}, {"modified_by", V(created_by_)}};
}

std::string SubjectWriter::topology(const TopologySpec& spec) {
  const std::string tid = uuid7();
  db_->insert("topology", {{"id", tid}, {"subject_id", id_}, {"kind", spec.kind}, {"rank", V(spec.rank)}, {"n_components", V(spec.n_components)}, {"winding", V(spec.winding)}});
  for (const auto& c : spec.cells) db_->insert("topology_cell", {{"topology_id", tid}, {"rank", V(c.rank)}, {"path", c.path}});
  return tid;
}

std::string SubjectWriter::geometry(const GeometrySpec& spec) {
  const std::optional<std::string> csid = spec.cs_key ? std::optional<std::string>(cs(*spec.cs_key)) : std::nullopt;
  const std::string gid = uuid7();
  db_->insert("geometry", {{"id", gid}, {"subject_id", id_}, {"form", spec.form}, {"coordinate_system_id", V(csid)}, {"positions_path", V(spec.positions_path)},
                           {"direction_json", spec.direction ? Value(jdump(*spec.direction)) : Value(std::monostate{})}});
  int k = 0;
  for (const auto& d : spec.dimensions) {
    Row r = {{"geometry_id", gid}, {"ordinal", V(k++)}, {"n_vertices", vint(d.n_vertices)}};
    if (d.name) r["name"] = *d.name;
    if (d.spacing) r["spacing"] = *d.spacing;
    if (d.origin) r["origin"] = *d.origin;
    if (d.unit) r["unit"] = *d.unit;
    if (d.manifold_id) r["manifold_id"] = *d.manifold_id;
    db_->insert("geometry_dimension", r);
  }
  return gid;
}

std::string SubjectWriter::manifold(ManifoldSpec spec) {
  const std::string mid = id_or_new(spec.id);
  Cols cols = acquired(spec.cols, spec.acquisition);
  auto rows = [&] {
    const std::string tid = !spec.topology_id.empty() ? spec.topology_id : topology(spec.topology);
    const std::optional<std::string> gid = spec.geometry ? std::optional<std::string>(geometry(*spec.geometry)) : std::nullopt;
    Row r = {{"id", mid}, {"subject_id", id_}, {"name", spec.name}, {"path", V(spec.path)}, {"type", spec.type}, {"n_vertices", vint(spec.n_vertices)},
             {"topology_id", tid}, {"geometry_id", V(gid)}};
    merge(r, prov(spec.created_utc));
    merge(r, cols);
    db_->insert("manifold", r);
  };
  return node("manifold", mid, rows, spec.populate);
}

std::optional<std::string> SubjectWriter::find_product(const std::vector<std::string>& factors) const {
  for (const auto& r : db_->all("SELECT m.id FROM manifold m WHERE m.subject_id = ? AND m.type = 'product' AND "
                                "(SELECT count(*) FROM geometry_dimension d WHERE d.geometry_id = m.geometry_id) = ?",
                                {id_, vint(static_cast<std::int64_t>(factors.size()))})) {
    std::vector<std::string> dims;
    for (const auto& d : db_->all("SELECT d.manifold_id FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? ORDER BY d.ordinal",
                                  {text(r, "id")}))
      dims.push_back(text(d, "manifold_id"));
    if (dims == factors) return text(r, "id");
  }
  return std::nullopt;
}

std::string SubjectWriter::product(const std::vector<std::string>& factors, const std::optional<std::string>& acquisition_id_in, bool inherit_acquisition,
                                   const std::optional<std::string>& name) {
  if (const auto it = products_.find(factors); it != products_.end()) return it->second;
  if (const auto have = find_product(factors)) { products_[factors] = *have; return *have; }
  std::vector<Row> rows;
  std::vector<Factor> fs_;
  std::int64_t ncomp = 1;
  bool reduces = true;
  for (const auto& f : factors) {
    const auto r = db_->one("SELECT m.*, t.kind AS tkind, t.rank AS trank, t.n_components AS tcomp FROM manifold m JOIN topology t ON t.id = m.topology_id WHERE m.id = ?", {f});
    if (!r) throw key_error("manifold " + f + ": no such row");
    rows.push_back(*r);
    Factor fa;
    auto opt = [&](const char* c) -> std::optional<std::int64_t> { return is_null(*r, c) ? std::nullopt : std::optional<std::int64_t>(integer(*r, c)); };
    fa.n_vertices = opt("n_vertices"); fa.n_edges = opt("n_edges"); fa.n_faces = opt("n_faces"); fa.n_volumes = opt("n_volumes");
    fa.rank = static_cast<int>(integer(*r, "trank"));
    fs_.push_back(fa);
    const std::string k = text(*r, "tkind");
    reduces = reduces && (k == "none" || k == "lattice" || k == "path");
    ncomp *= k == "none" ? integer(*r, "n_vertices") : integer(*r, "tcomp");
  }
  const FVector counts = product_counts(fs_);
  const std::string kind = counts.rank == 0 ? "none" : (reduces ? "lattice" : "product");
  std::optional<std::string> acq = acquisition_id_in;
  if (!acq && inherit_acquisition) for (const auto& r : rows) if (!is_null(r, "acquisition_id")) { acq = text(r, "acquisition_id"); break; }
  std::string pname;
  if (name) pname = *name;
  else for (std::size_t i = 0; i < rows.size(); ++i) pname += (i ? " × " : "") + text(rows[i], "name");
  ManifoldSpec m;
  m.name = pname;
  m.type = "product";
  m.n_vertices = counts.n_vertices.value_or(0);
  m.topology = {kind, counts.rank, static_cast<int>(ncomp), std::nullopt, {}};
  GeometrySpec g;
  g.form = "product";
  for (const auto& r : rows) { DimensionSpec d; d.name = text(r, "name"); d.n_vertices = integer(r, "n_vertices"); d.manifold_id = text(r, "id"); g.dimensions.push_back(d); }
  m.geometry = g;
  auto setn = [&](const char* c, const std::optional<std::int64_t>& v) { if (v) m.cols.set(c, *v); else m.cols.null(c); };
  setn("n_edges", counts.n_edges); setn("n_faces", counts.n_faces); setn("n_volumes", counts.n_volumes);
  if (acq) m.cols.set("acquisition_id", *acq); else m.cols.null("acquisition_id");
  const std::string pid = manifold(std::move(m));
  products_[factors] = pid;
  return pid;
}

Populate SubjectWriter::array_populate(const std::string& table, const std::string& id, const std::optional<ArrayMeta>& meta, const std::optional<ArrayView>& data,
                                       const Populate& populate, bool dense) {
  if (!meta) return populate;
  // the views are not owned: the caller keeps them alive through the call, which populates before it returns
  return [this, table, id, meta = *meta, data, populate, dense](Store& store, const std::string& at_) {
    ensure_groups(store, at_, false);
    layout_array(store, at_, meta, attributes_text(*db_, table, id));
    if (data) populate_array(store, at_, *data, dense);
    if (populate) populate(store, at_);
  };
}

std::string SubjectWriter::field(FieldSpec spec) {
  const std::string fid = id_or_new(spec.id);
  Cols cols = acquired(spec.cols, spec.acquisition);
  if (spec.sparse) {
    if (spec.data || spec.meta || spec.populate) throw value_error("field '" + spec.name + "': sparse, or data / meta / populate — not both");
    std::set<std::string> have;
    for (const auto& p : spec.sparse->parts) have.insert(p.first);
    std::vector<std::string> missing;
    for (const char* need : {"data", "indices", "indptr"}) if (!have.count(need)) missing.push_back(need);
    if (!missing.empty()) {
      std::string m;
      for (const auto& x : missing) m += (m.empty() ? "'" : ", '") + x + "'";
      throw value_error("field '" + spec.name + "': a sparse field needs indptr · indices · data (missing [" + m + "])");
    }
    const ArrayView* d = nullptr;
    for (const auto& p : spec.sparse->parts) if (p.first == "data") d = &p.second;
    cols.set("layout", "csc").set("nnz", d->size()).set("data_type", dtype_to_string(d->dtype));
  }
  const std::string value_type = spec.value_type;
  const int n = n_components_of(value_type);
  if (value_type == "hermitian3") allow_value_type(*db_, value_type);   // an older database: its CHECK relaxed in place, once
  std::optional<ArrayMeta> meta = spec.meta;
  if (spec.data && !meta) meta = array_meta(spec.data->dtype, spec.data->shape, spec.array);
  auto rows = [&] {
    Row r = {{"id", fid}, {"subject_id", id_}, {"name", spec.name}, {"path", V(spec.path)}, {"kind", spec.kind}, {"manifold_id", spec.manifold_id},
             {"value_type", value_type}, {"n_components", V(n)}};
    if (meta) merge(r, meta->row());
    else r["data_type"] = cols.count("data_type") ? cols["data_type"] : Value(std::string("float32"));
    merge(r, prov(spec.created_utc));
    Cols rest = cols;
    if (!meta) rest.erase("data_type");
    merge(r, rest);
    db_->insert("field", r);
    int k = 0;
    for (const auto& c : spec.contributions)
      db_->insert("field_contribution", {{"field_id", fid}, {"ordinal", V(k++)}, {"weight", c.weight}, {"subject_id", c.subject_id}, {"source_field_id", V(c.source_field_id)}});
  };
  if (spec.sparse) {
    // a field's parts go in the fixed order indptr · indices · data (an operator's, in the order given)
    SparseArrays sp;
    for (const char* part : {"indptr", "indices", "data"})
      for (const auto& p : spec.sparse->parts) if (p.first == part) sp.parts.push_back(p);
    const auto pop = spec.path ? Populate([this, spec_path = *spec.path, sp](Store&, const std::string&) { write_sparse(spec_path, sp); }) : Populate();
    return node("field", fid, rows, pop);
  }
  return node("field", fid, rows, spec.path ? array_populate("field", fid, meta, spec.data, spec.populate, spec.array.dense) : Populate());
}

// A STATISTIC (statistic.h): its row and its values in one node; the table added in place to a dataset made before it
std::string SubjectWriter::statistic(StatisticSpec spec) {
  ensure_statistics(*db_);
  const std::string who = "statistic '" + spec.name + "'";
  if (spec.name.empty()) throw value_error("a statistic needs a name");
  auto names = [](const char* list) { std::string n; for (const auto& e : statistic_registry()[list]) n += (n.empty() ? "" : ", ") + e["name"].get<std::string>(); return n; };
  // the statistics registry's measures, or a measurement's (a summary keeps the measure it summarises: amplitude, mean, …)
  if (!statistic_measure_known(spec.measure) && !measure_spec(spec.measure))
    throw value_error(who + ": measure '" + spec.measure + "' is in neither the statistics registry (" + names("measures") + ") nor the measure registry");
  if (!statistic_method_known(spec.method)) throw value_error(who + ": method '" + spec.method + "' is not in the statistics registry (" + names("methods") + ")");
  if (!spec.of_field_id && !spec.of_statistic_id) throw value_error(who + ": a statistic is OF a field or of a statistic (of_field_id · of_statistic_id)");
  if (!spec.axes.is_array() || spec.axes.empty()) throw value_error(who + ": axes is a list of {name, size, …}, one per dimension of the values");
  if (spec.data.dtype != DType::Float32 && spec.data.dtype != DType::Float64) throw value_error(who + ": the values are float32 or float64");
  // each value's shape (as a field's), trailing the axes
  std::vector<std::int64_t> elem;
  if (spec.value_type == "complex" || spec.value_type == "vector2") elem = {2};
  else if (spec.value_type == "vector3") elem = {3};
  else if (spec.value_type == "quaternion") elem = {4};
  else if (spec.value_type == "matrix3") elem = {3, 3};
  else if (spec.value_type == "hermitian3") elem = {3, 3, 2};
  else if (spec.value_type != "scalar" && spec.value_type != "integer") throw value_error(who + ": value_type '" + spec.value_type + "' (scalar, integer, complex, vector2, vector3, matrix3, quaternion, hermitian3)");
  if (spec.data.shape.size() != spec.axes.size() + elem.size())
    throw value_error(who + ": " + std::to_string(spec.axes.size()) + " axes and a " + spec.value_type + " value make " + std::to_string(spec.axes.size() + elem.size()) + " dimensions; the values have " + std::to_string(spec.data.shape.size()));
  for (std::size_t j = 0; j < elem.size(); ++j)
    if (spec.data.shape[spec.axes.size() + j] != elem[j]) throw value_error(who + ": a " + spec.value_type + " value is [" + [&] { std::string t; for (auto e : elem) t += (t.empty() ? "" : " x ") + std::to_string(e); return t; }() + "]");
  for (std::size_t i = 0; i < spec.axes.size(); ++i) {
    const Json& a = spec.axes[i];
    if (!a.is_object() || !a.contains("name") || !a["name"].is_string() || !a.contains("size") || !a["size"].is_number_integer())
      throw value_error(who + ": axis " + std::to_string(i) + " needs a name and a size");
    if (a["size"].get<std::int64_t>() != spec.data.shape[i]) throw value_error(who + ": axis '" + a["name"].get<std::string>() + "' has size " + a["size"].dump() + ", the values " + std::to_string(spec.data.shape[i]));
    int refs = 0;
    for (const char* k : {"manifold_id", "selection_id", "edges", "values_path", "values"}) refs += a.contains(k) ? 1 : 0;
    const bool components = !spec.components.is_null() && i + 1 == spec.axes.size();
    if (refs != 1 && !components) throw value_error(who + ": axis '" + a["name"].get<std::string>() + "' names what it runs along: one of manifold_id, selection_id, edges, values_path, values");
  }
  if (!spec.components.is_null() && (!spec.components.is_array() || static_cast<std::int64_t>(spec.components.size()) != spec.axes.back()["size"].get<std::int64_t>()))
    throw value_error(who + ": components name the last axis's " + spec.axes.back()["size"].dump() + " entries");
  std::string path = spec.path.value_or("");
  if (path.empty()) {
    const auto parent = spec.of_field_id ? db_->read("field", *spec.of_field_id) : std::nullopt;
    path = (parent && !is_null(*parent, "path") ? text(*parent, "path") + "_statistics/" : std::string("statistics/")) + safe_name(spec.name);
  }
  const std::string sid = id_or_new(spec.id);
  Cols cols = acquired(Cols{}, spec.acquisition);
  const ArrayMeta meta = array_meta(spec.data.dtype, spec.data.shape, spec.array);
  auto js = [](const Json& j) { return j.is_null() ? Value() : Value(j.dump()); };
  auto rows = [&] {
    Row r = {{"id", sid}, {"subject_id", id_}, {"name", spec.name}, {"path", path}, {"measure", spec.measure}, {"method", spec.method},
             {"value_type", spec.value_type}, {"n_components", V(static_cast<std::int64_t>(n_components_of(spec.value_type)))},
             {"of_field_id", vopt(spec.of_field_id)}, {"of_statistic_id", vopt(spec.of_statistic_id)}, {"partition_selection_id", vopt(spec.partition_selection_id)},
             {"span_json", js(spec.span)}, {"axes_json", spec.axes.dump()}, {"components_json", js(spec.components)}, {"scaling_json", js(spec.scaling)},
             {"options_json", js(spec.options)}, {"producer_json", js(spec.producer)}, {"unit", vopt(spec.unit)}, {"description", vopt(spec.description)},
             {"label", vopt(spec.label)}};
    if (spec.n_members) r["n_members"] = *spec.n_members;
    if (spec.members_skipped) r["members_skipped"] = *spec.members_skipped;
    if (spec.dof) r["dof"] = *spec.dof;
    merge(r, meta.row());
    merge(r, prov(spec.created_utc));
    merge(r, cols);
    db_->insert("statistic", r);
  };
  return node("statistic", sid, rows, array_populate("statistic", sid, meta, spec.data, Populate(), spec.array.dense));
}

std::string SubjectWriter::operator_(OperatorSpec spec) {
  const std::string oid = id_or_new(spec.id);
  Cols cols = acquired(spec.cols, spec.acquisition);
  std::optional<ArrayMeta> meta = spec.meta;
  if (spec.data && !meta) meta = array_meta(spec.data->dtype, spec.data->shape, spec.array);
  for (const char* end : {"from_value_type", "to_value_type"})
    if (const auto it = cols.find(end); it != cols.end() && std::holds_alternative<std::string>(it->second) && std::get<std::string>(it->second) == "hermitian3") allow_value_type(*db_, "hermitian3");
  auto rows = [&] {
    Row r = {{"id", oid}, {"subject_id", id_}, {"name", spec.name}, {"path", V(spec.path)}, {"kind", spec.kind}, {"from_manifold_id", spec.from_manifold_id},
             {"layout", spec.layout}, {"n_rows", vint(spec.n_rows)}, {"n_cols", vint(spec.n_cols)}};
    if (meta) merge(r, meta->row());
    merge(r, prov(spec.created_utc));
    merge(r, cols);
    db_->insert("operator", r);
  };
  Populate pop;
  if (spec.path) {
    if (spec.sparse) {
      auto sp = *spec.sparse;
      pop = [this, p = *spec.path, sp](Store&, const std::string&) { write_sparse(p, sp); };
    } else {
      pop = array_populate("operator", oid, meta, spec.data, spec.populate, false);
    }
  }
  return node("operator", oid, rows, pop);
}

std::string SubjectWriter::selection(SelectionSpec spec) {
  const std::string sid = id_or_new(spec.id);
  Cols cols = acquired(spec.cols, spec.acquisition);
  std::optional<ArrayMeta> meta = spec.meta;
  if (spec.data && !meta) meta = array_meta(spec.data->dtype, spec.data->shape);
  std::optional<ArrayView> el = spec.data;
  if (spec.elements) el = spec.elements;
  if (spec.no_elements) el = std::nullopt;
  auto rows = [&] {
    Row r = {{"id", sid}, {"subject_id", id_}, {"name", spec.name}, {"path", V(spec.path)}, {"type", spec.type}, {"manifold_id", V(spec.manifold_id)}};
    if (spec.data && spec.path) r["array_path"] = *spec.path;
    merge(r, prov(spec.created_utc));
    merge(r, cols);
    db_->insert("selection", r);
    if (el) {
      // THE CODES ARE THE ARRAY'S: no per-element rows — the row carries the counts only, a subset its picked elements, a partition its
      // elements and its distinct codes (D48)
      if (spec.picked) db_->execute("UPDATE selection SET n_elements = ? WHERE id = ?", {vint(el->size()), sid});
      else db_->execute("UPDATE selection SET n_elements = ?, n_members = ? WHERE id = ?", {vint(el->size()), vint(distinct_count(*el)), sid});
    }
    if (spec.spans) {
      int k = 0;
      for (const auto& s : *spec.spans)
        db_->execute("INSERT INTO selection_span (selection_id, ordinal, start, stop, code) VALUES (?, ?, ?, ?, ?)",
                     {sid, V(k++), s.start, s.stop, s.code ? Value(*s.code) : Value(std::monostate{})});
      db_->execute("UPDATE selection SET n_members = ? WHERE id = ?", {vint(static_cast<std::int64_t>(spec.spans->size())), sid});
    }
    for (const auto& e : spec.extents) db_->insert("selection_extent", {{"selection_id", sid}, {"axis", vint(e.axis)}, {"start", e.start}, {"stop", e.stop}});
    for (const auto& m : spec.measurements) {
      Row mr = {{"selection_id", sid}};
      merge(mr, m);
      db_->insert("selection_measurement", mr);
    }
  };
  const bool has_array = spec.path && (meta || spec.populate);
  return node("selection", sid, rows, has_array ? array_populate("selection", sid, meta, spec.data, spec.populate, spec.dense) : Populate());
}

void SubjectWriter::measurement(const std::string& selection_id, const Cols& row) {
  auto go = [&] {
    Row mr = {{"selection_id", selection_id}};
    merge(mr, row);
    db_->insert("selection_measurement", mr);
  };
  if (immediate_) update_node(*db_, ds_->node_store(), go);
  else db_->tx(go);
}

DictionaryResult SubjectWriter::dictionary(const DictionarySpec& spec) {
  Cols scope_cols;
  std::string where;
  std::vector<Value> args;
  if (spec.scope == "app") where = "scope = 'app'";
  else if (spec.scope == "dataset") { where = "scope = 'dataset' AND dataset_id = ?"; args = {ds_->id()}; scope_cols.set("dataset_id", ds_->id()); }
  else if (spec.scope == "subject") { where = "scope = 'subject' AND subject_id = ?"; args = {id_}; scope_cols.set("subject_id", id_); }
  else throw key_error(spec.scope);
  std::vector<DictionaryEntry> entries;
  if (spec.entries) {
    entries = *spec.entries;
  } else {
    const auto& fams = family_colors();
    const std::map<std::string, Color>* fam = nullptr;
    if (spec.scope == "app") if (auto it = fams.find(spec.name); it != fams.end()) fam = &it->second;
    for (std::size_t k = 0; k < spec.names.size(); ++k) {
      DictionaryEntry e;
      e.code = static_cast<std::int64_t>(k);
      e.name = spec.names[k];
      e.ordinal = static_cast<std::int64_t>(k);
      std::optional<Color> c = k < spec.colors.size() ? spec.colors[k] : std::nullopt;
      if (!c && fam) if (auto it = fam->find(spec.names[k]); it != fam->end()) c = it->second;
      e.color = c;
      entries.push_back(e);
    }
  }
  DictionaryResult out;
  db_->tx([&] {
    std::vector<Value> a = args;
    a.push_back(spec.name);
    const auto have = db_->one("SELECT * FROM dictionary WHERE " + where + " AND name = ?", a);
    std::string did;
    if (!have) {
      did = uuid7();
      Row r = {{"id", did}, {"name", spec.name}, {"scope", spec.scope}, {"description", V(spec.description)}};
      merge(r, scope_cols);
      merge(r, prov(std::nullopt));
      db_->insert("dictionary", r);
      if (!immediate_) created_.push_back({"dictionary", did});
    } else {
      did = text(*have, "id");
    }
    std::map<std::string, std::int64_t> known;
    std::set<std::int64_t> taken;
    for (const auto& r : db_->all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", {did})) { known[text(r, "name")] = integer(r, "code"); taken.insert(integer(r, "code")); }
    std::int64_t nxt = taken.empty() ? 0 : *taken.rbegin() + 1;
    for (const auto& e : entries) {
      if (const auto it = known.find(e.name); it != known.end()) { out.codes.push_back(it->second); continue; }
      std::int64_t code = e.code;
      if (taken.count(code)) { code = nxt; ++nxt; }
      nxt = std::max(nxt, code + 1);
      Row row = {{"dictionary_id", did}, {"code", code}, {"name", e.name}};
      if (e.ordinal) row["ordinal"] = (have ? code : *e.ordinal);
      if (e.color) row["color_json"] = jdump(Json::array({(*e.color)[0], (*e.color)[1], (*e.color)[2]}));
      else if (!spec.entries) row["color_json"] = std::monostate{};
      merge(row, e.cols);
      db_->insert("dictionary_entry", row);
      known[e.name] = code;
      taken.insert(code);
      out.codes.push_back(code);
    }
    out.id = did;
  });
  return out;
}

}  // namespace nxr::io
