// nxr::bst — nxr.convert.sensors.export_sensors with the full sensor geometry (register D166).
#include "nxr/bst/sensors.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "nxr/bst/entities.h"
#include "nxr/io/layout.h"

namespace nxr::bst {

namespace code = nxr::io::code;
using nxr::io::Cols;
using nxr::io::Json;
using nxr::io::SubjectWriter;
using nxr::io::view;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::string strip(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const auto a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  return s.substr(a, s.find_last_not_of(ws) - a + 1);
}
std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

// `str(x)` of a char value (the channel's Name, Type); a number is printed as Python prints it only for the integers a name could be
std::string text_of(const Value* v) {
  if (!v) return "";
  if (v->is_char()) return v->text.empty() ? "" : v->text[0];
  return "";
}

// A per-channel array ([3, k] or [3]) as k rows of 3; none when empty (or not a multiple of 3): `_cols`
std::optional<std::vector<std::array<double, 3>>> cols3(const Value* v) {
  if (!v || !v->is_numeric()) return std::nullopt;
  const std::vector<double> a = v->doubles();     // column-major: column j is a[3j..3j+2]
  if (a.empty() || a.size() % 3) return std::nullopt;
  std::vector<std::array<double, 3>> out(a.size() / 3);
  for (std::size_t j = 0; j < out.size(); ++j) out[j] = {a[3 * j], a[3 * j + 1], a[3 * j + 2]};
  return out;
}

// a list of strings of the given length (a cell of chars, a char array's rows, one char), stripped; empty when the length differs: `_strings`
std::vector<std::string> strings(const Value* v, std::size_t n) {
  std::vector<std::string> out;
  if (!v) return out;
  if (v->is_char()) {
    if (v->text.size() <= 1) out.push_back(strip(v->text.empty() ? "" : v->text[0]));
    else for (const auto& t : v->text) out.push_back(strip(t));
  } else if (v->is_cell()) {
    for (const auto& e : v->elems) out.push_back(e.is_char() ? strip(e.text.empty() ? "" : e.text[0]) : "");
  }
  if (out.size() != n) out.clear();
  return out;
}

// TransfMeg / TransfEeg: a cell of 4x4 matrices (or one), each as 16 numbers row after row
std::vector<std::vector<double>> matrices(const Value* v) {
  std::vector<std::vector<double>> out;
  if (!v) return out;
  auto one = [&](const Value& m) { if (m.is_numeric() && m.numel() == 16 && m.dims.size() == 2) out.push_back(m.row_major()); };
  if (v->is_cell()) for (const auto& e : v->elems) one(e);
  else one(*v);
  return out;
}

const Value* find(const std::map<std::string, Value>& m, const std::string& k) {
  const auto it = m.find(k);
  return it == m.end() ? nullptr : &it->second;
}

Json transforms(const std::map<std::string, Value>& mat) {
  Json t = Json::object();
  for (const auto& [key, lab, name] : {std::tuple<const char*, const char*, const char*>{"TransfMeg", "TransfMegLabels", "meg"}, {"TransfEeg", "TransfEegLabels", "eeg"}}) {
    const auto m = matrices(find(mat, key));
    if (m.empty()) continue;
    Json mats = Json::array();
    for (const auto& x : m) mats.push_back(x);
    t[name] = Json{{"matrices", mats}, {"labels", strings(find(mat, lab), m.size())}, {"layout", "4x4 row-major, metres"}};
  }
  if (const Value* scs = find(mat, "SCS"); scs && scs->is_struct() && scs->numel() >= 1) {
    Json fid = Json::object();
    for (const char* k : {"NAS", "LPA", "RPA", "R", "T"}) {
      const Value* f = scs->field(k, 0);
      if (f && f->is_numeric() && f->numel() && f->dims.size() == 2) fid[k] = f->row_major();
    }
    if (!fid.empty()) t["scs"] = fid;
  }
  return t;
}

// order-stable unique levels and a code per value
std::pair<std::vector<std::int64_t>, std::vector<std::string>> family(const std::vector<std::string>& values) {
  std::vector<std::string> levels;
  std::vector<std::int64_t> codes;
  for (const auto& v : values) {
    auto it = std::find(levels.begin(), levels.end(), v);
    if (it == levels.end()) { levels.push_back(v); it = levels.end() - 1; }
    codes.push_back(it - levels.begin());
  }
  return {codes, levels};
}

std::vector<float> f32(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

}  // namespace

std::vector<std::string> channel_units_for_types(const std::vector<std::string>& types, const std::map<std::string, std::string>& override_) {
  std::map<std::string, std::string> table = {{"MEG", "T"}, {"MEG REF", "T"}, {"MEG MAG", "T"}, {"MEG GRAD", "T"}, {"EEG", "V"}, {"SEEG", "V"},
                                              {"ECOG", "V"}, {"EOG", "V"}, {"ECG", "V"}, {"EMG", "V"}};
  for (const auto& [k, v] : override_) table[k] = v;
  std::vector<std::string> out;
  for (const auto& t : types) {
    const auto it = table.find(upper(strip(t)));
    out.push_back(it == table.end() ? "" : it->second);
  }
  return out;
}

SensorInfo export_sensors(SubjectWriter& sub, const std::map<std::string, Value>& channel_mat, const std::vector<std::int64_t>& channel_flag, const SensorOptions& opt) {
  const std::string& name = opt.name;
  const Value* chv = find(channel_mat, "Channel");
  if (!chv || !chv->is_struct()) throw nxr::io::Error(code::invalid_argument, "Channel: not a struct array");
  const std::size_t n = chv->numel();
  std::vector<std::string> names, types;
  for (std::size_t i = 0; i < n; ++i) {
    if (!chv->field("Name", i)) throw nxr::io::Error(code::invalid_argument, "'Name'");
    names.push_back(text_of(chv->field("Name", i)));
    types.push_back(text_of(chv->field("Type", i)));
  }
  const std::vector<std::string> units = channel_units_for_types(types, opt.units);

  std::vector<double> loc(n * 3, kNaN), contacts(n * 2 * 3, kNaN), orient(n * 3, kNaN);
  std::int64_t bipolar = 0;
  std::vector<double> coil_pos, coil_ori, coil_w;
  std::vector<std::int64_t> coil_row;
  std::size_t max_cols = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Value* lv = chv->field("Loc", i);
    const auto o = cols3(chv->field("Orient", i));
    if (o) for (int c = 0; c < 3; ++c) orient[3 * i + c] = (*o)[0][c];
    if (lv && lv->is_numeric() && lv->numel()) {
      const std::vector<double> l = lv->doubles();
      if (l.size() % 3) throw nxr::io::Error(code::invalid_argument, "cannot reshape array of size " + std::to_string(l.size()) + " into shape (3,newaxis)");
      const std::size_t k = l.size() / 3;
      auto col = [&](std::size_t j, int c) { return l[3 * j + c]; };
      const bool is_bipolar = k == 2 && upper(strip(types[i])).rfind("MEG", 0) != 0;
      if (!is_bipolar) {
        const Value* wv = chv->field("Weight", i);
        std::vector<double> w(k, 1.0);
        if (wv && wv->is_numeric() && wv->numel() == k) w = wv->doubles();
        for (std::size_t j = 0; j < k; ++j) {
          for (int c = 0; c < 3; ++c) {
            coil_pos.push_back(col(j, c));
            coil_ori.push_back(o && j < o->size() ? (*o)[j][c] : kNaN);
          }
          coil_w.push_back(w[j]);
          coil_row.push_back(static_cast<std::int64_t>(i));
        }
        max_cols = std::max(max_cols, k);
      }
      if (is_bipolar) {
        for (int j = 0; j < 2; ++j) for (int c = 0; c < 3; ++c) contacts[(i * 2 + j) * 3 + c] = col(j, c);
        ++bipolar;
        for (int c = 0; c < 3; ++c) loc[3 * i + c] = (col(0, c) + col(1, c)) / 2.0;     // the midpoint
      } else {
        for (int j = 0; j < 2; ++j) for (int c = 0; c < 3; ++c) contacts[(i * 2 + j) * 3 + c] = col(0, c);
        for (int c = 0; c < 3; ++c) loc[3 * i + c] = col(0, c);                        // first coil position
      }
    }
  }
  std::vector<std::string> groups;
  for (std::size_t i = 0; i < n; ++i) {
    const Value* g = chv->field("Group", i);
    groups.push_back(g && g->is_char() ? strip(text_of(g)) : "");
  }
  std::vector<std::string> modalities;
  for (const auto& t : types) if (std::find(modalities.begin(), modalities.end(), t) == modalities.end()) modalities.push_back(t);
  const std::string path = nxr::io::folder_for("points") + "/" + name;
  const std::int64_t N = static_cast<std::int64_t>(n);

  auto manifold_points = [&](const std::string& mid, const std::string& nm, const std::string& p, const std::vector<float>& pos, std::int64_t count, const Json& producer,
                             const std::string& acq, bool placed = true, const std::string* cs = nullptr) {
    nxr::io::ManifoldSpec m;
    m.id = mid;
    m.name = nm;
    m.path = p;
    m.type = "points";
    if (!placed) m.path.reset();
    m.n_vertices = count;
    m.acquisition = acq;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    if (placed) {
      nxr::io::GeometrySpec g;
      g.form = "stored";
      g.cs_key = cs ? *cs : opt.cs_key;
      g.positions_path = p;
      m.geometry = g;
      m.populate = [&sub, p, &pos, count, mid](nxr::io::Store&, const std::string&) { sub.write_array(p, view(pos, {count, 3}), {}, std::nullopt, "manifold", mid); };
    }
    m.cols.json("producer_json", producer);
    sub.manifold(std::move(m));
  };

  // THE MANIFOLD — a 0-manifold: isolated elements, its node IS its positions
  const std::string mid = nxr::io::uuid7();
  const std::vector<float> pos = f32(loc);
  Json producer = Json::object();
  producer["modalities"] = modalities;
  producer["channel_units_source"] = "derived-from-type";
  if (bipolar) {
    producer["positions"] = "bipolar channels at the midpoint of Loc's two contacts (both: " + name + "_contact1/2)";
    producer["bipolar"] = bipolar;
  }
  if (Json tf = transforms(channel_mat); !tf.empty()) producer["transforms"] = tf;
  if (opt.source_sha1) producer["source_sha1"] = *opt.source_sha1;
  if (opt.producer) for (auto it = opt.producer->begin(); it != opt.producer->end(); ++it) producer[it.key()] = it.value();
  manifold_points(mid, name, path, pos, N, producer, opt.acquisition, opt.placed);

  auto vec3_field = [&](const std::string& fname, const std::string& fpath, const std::string& kind, const std::string& manifold, const std::vector<float>& data,
                        std::int64_t rows, const std::string& description, const std::optional<std::string>& unit) {
    nxr::io::FieldSpec f;
    f.name = fname;
    f.path = fpath;
    f.kind = kind;
    f.manifold_id = manifold;
    f.value_type = "vector3";
    f.data = view(data, {rows, 3});
    f.acquisition = opt.acquisition;
    if (unit) f.cols.set("unit", *unit);
    f.cols.set("description", description);
    sub.field(std::move(f));
  };

  if (bipolar) {
    for (int k = 0; k < 2; ++k) {
      std::vector<float> c(n * 3);
      for (std::size_t i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) c[3 * i + d] = static_cast<float>(contacts[(i * 2 + k) * 3 + d]);
      vec3_field(name + "_contact" + std::to_string(k + 1), path + "_contact" + std::to_string(k + 1), "contact position", mid, c, N,
                 "contact " + std::to_string(k + 1) + " of each bipolar channel (Loc column " + std::to_string(k + 1) + "), scs metres", "m");
    }
  }

  // THE ORIENTATIONS — Orient(:,1) per channel
  if (std::any_of(orient.begin(), orient.end(), [](double x) { return std::isfinite(x); }))
    vec3_field(name + "_orientation", path + "_orientation", "sensor orientation", mid, f32(orient), N,
               "the direction each channel measures along (Orient column 1, unit vector, scs); NaN where it has none", std::nullopt);

  // THE COILS — every Loc column of every non-bipolar channel, and the weights that integrate them into a channel
  if (!coil_w.empty() && max_cols > 1) {
    const std::int64_t P = static_cast<std::int64_t>(coil_w.size());
    const std::string cid = nxr::io::uuid7();
    const std::string cpath = path + "_coils";
    const std::vector<float> cpos = f32(coil_pos);
    Json cprod = Json{{"description", "every Loc column of every non-bipolar channel, in channel order (MEG coils and integration points)"}, {"of", name}};
    manifold_points(cid, name + "_coils", cpath, cpos, P, cprod, opt.acquisition);
    vec3_field(name + "_coils_orientation", cpath + "_orientation", "sensor orientation", cid, f32(coil_ori), P,
               "Orient of each coil / integration point, scs; NaN where none", std::nullopt);
    std::vector<std::int32_t> indptr(n + 1, 0), indices(coil_w.size());
    for (auto r : coil_row) ++indptr[static_cast<std::size_t>(r) + 1];
    for (std::size_t i = 0; i < n; ++i) indptr[i + 1] += indptr[i];
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<std::int32_t>(i);
    const std::vector<float> w = f32(coil_w);
    nxr::io::OperatorSpec op;
    op.name = name + "_coil_weights";
    op.path = op.name;
    op.kind = "coil weights";
    op.from_manifold_id = cid;
    op.layout = "csr";
    op.n_rows = N;
    op.n_cols = P;
    op.acquisition = opt.acquisition;
    op.cols.set("to_manifold_id", mid).set("nnz", P)
        .set("description", "channel m = sum_k w_k * coil k \xe2\x80\x94 the forward model's integration over a channel's coils (Brainstorm's Weight; 1 where a channel has none)");
    op.sparse = nxr::io::SparseArrays{{{"indptr", view(indptr, {N + 1})}, {"indices", view(indices, {P})}, {"data", view(w, {P})}}};
    sub.operator_(std::move(op));
  }

  // THE HEAD POINTS — the digitised head, with their labels and types as Selections
  if (const Value* hp = find(channel_mat, "HeadPoints"); hp && hp->is_struct() && hp->numel() >= 1) {
    if (const auto hloc = cols3(hp->field("Loc", 0))) {
      const std::int64_t H = static_cast<std::int64_t>(hloc->size());
      const std::string hid = nxr::io::uuid7();
      const std::string hpath = path + "_headpoints";
      std::vector<float> hpos;
      for (const auto& p : *hloc) for (int c = 0; c < 3; ++c) hpos.push_back(static_cast<float>(p[c]));
      const std::string hcs = opt.headpoints_cs_key.value_or(opt.cs_key);
      manifold_points(hid, name + "_headpoints", hpath, hpos, H, Json{{"description", "the digitised head points (HeadPoints.Loc), in " + hcs}, {"of", name}}, opt.acquisition, true, &hcs);
      std::vector<LabelMember> fams;
      for (const auto& [fam, key] : {std::pair<const char*, const char*>{"label", "Label"}, {"type", "Type"}}) {
        const auto vals = strings(hp->field(key, 0), hloc->size());
        if (vals.empty()) continue;
        auto [codes, levels] = family(vals);
        fams.push_back({fam, codes, levels, {}});
      }
      if (!fams.empty()) {
        std::string d = "the head points' ";
        for (std::size_t i = 0; i < fams.size(); ++i) d += (i ? " and " : "") + fams[i].family;
        label_set(sub, hpath + "_labels", fams, hid, d, opt.acquisition,
                  [&](const std::string& fam) { return std::make_pair(name + "_headpoints_" + fam, std::string("subject")); });
      }
    }
  }

  // THE NAMES — one level per channel, the subject's own dictionary
  {
    std::vector<std::int64_t> codes(n);
    for (std::size_t i = 0; i < n; ++i) codes[i] = static_cast<std::int64_t>(i);
    LabellingOptions lo;
    lo.manifold_id = mid;
    lo.dictionary = name + "_names";
    lo.scope = "subject";
    lo.acquisition = opt.acquisition;
    lo.description = "the channel names \xe2\x80\x94 one level per channel";
    labelling(sub, path + "_names", codes, names, lo);
  }

  // THE FAMILIES — one set, each family an app-wide dictionary
  std::vector<LabelMember> fams;
  {
    auto [c, l] = family(types);
    fams.push_back({"type", c, l, {}});
  }
  {
    auto [c, l] = family(units);
    fams.push_back({"unit", c, l, {}});
  }
  for (int group : {1, 2}) {
    std::vector<std::string> parsed;
    for (const auto& nm : names) {
      const bool ok = nm.size() >= 3 && nm[0] == 'M' && std::string("LRZ").find(nm[1]) != std::string::npos && std::string("CFOPT").find(nm[2]) != std::string::npos;
      parsed.push_back(ok ? std::string(1, nm[group]) : "-");
    }
    auto [c, l] = family(parsed);
    if (l.size() > 1) fams.push_back({group == 1 ? "hemisphere" : "lobe", c, l, {}});
  }
  if (std::any_of(groups.begin(), groups.end(), [](const std::string& g) { return !g.empty(); })) {
    std::vector<std::string> g2;
    for (const auto& g : groups) g2.push_back(g.empty() ? "-" : g);
    auto [c, l] = family(g2);
    fams.push_back({"group", c, l, {}});
  }
  if (channel_flag.size() == n && n > 0) {
    FlagLevels fl = flag_levels(channel_flag);
    fams.push_back({"quality", fl.codes, fl.levels, fl.colors});
  }
  std::string fd = "the montage families \xe2\x80\x94 ";
  for (std::size_t i = 0; i < fams.size(); ++i) fd += (i ? ", " : "") + fams[i].family;
  label_set(sub, path + "_labels", fams, mid, fd, opt.acquisition, [&](const std::string& fam) {
    return fam == "group" ? std::make_pair(name + "_group", std::string("subject")) : std::make_pair(fam, std::string("app"));
  });

  SensorInfo info;
  info.name = name;
  info.id = mid;
  info.n_channels = N;
  info.bipolar = bipolar;
  info.modalities = modalities;
  info.names = names;
  return info;
}

}  // namespace nxr::bst
