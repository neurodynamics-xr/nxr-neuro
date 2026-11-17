#include "nxr/pipelines/frames.h"

#include <cmath>
#include <cstring>
#include <filesystem>

#include "nxr/chart.h"
#include "nxr/pipelines/task.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;
using nxr::frames::Mat3;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }

std::vector<float> read_f32(const nxr::io::Store& store, const std::string& key, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(store, key);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  const std::size_t n = raw.size() / r.itemsize();
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (r.dtype() == DType::Float32) { float v; std::memcpy(&v, raw.data() + i * 4, 4); out[i] = v; }
    else if (r.dtype() == DType::Float64) { double v; std::memcpy(&v, raw.data() + i * 8, 8); out[i] = static_cast<float>(v); }
    else if (r.dtype() == DType::Int32) { std::int32_t v; std::memcpy(&v, raw.data() + i * 4, 4); out[i] = static_cast<float>(v); }
    else throw Error(code::unsupported, key + ": a frames array of type " + dtype_to_string(r.dtype()) + " is not read here");
  }
  return out;
}
std::optional<task::Node> field_named(SubjectWriter& sub, const std::string& name) {
  return task::Subject::of(sub).find({.cls = task::Cls::field, .name = name, .stored = true});
}
std::optional<std::vector<std::int32_t>> indices_named(SubjectWriter& sub, const std::string& name) {
  const auto r = task::Subject::of(sub).find({.cls = task::Cls::selection, .name = name, .stored = true});
  if (!r) return std::nullopt;
  ArrayReader a = ArrayReader::open(sub.store(), *r->array_path);
  const std::vector<std::uint8_t> raw = a.read();
  std::vector<std::int32_t> out(raw.size() / 4);
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * 4);
  return out;
}
std::optional<SurfaceInput> sphere_of(SubjectWriter& sub, const std::string& surface) {
  try { return read_surface(sub, surface + "_sphere", true); } catch (const Error& e) { if (e.code() == code::no_such_row) return std::nullopt; throw; }
}
nxr::frames::Axes axes_of(const std::string& convention) {
  nxr::frames::Axes a;
  if (convention == "scs") return a;
  if (convention == "ras") { a.anterior = {0, 1, 0}; a.superior = {0, 0, 1}; a.left = {-1, 0, 0}; return a; }
  invalid("frames: convention is scs or ras");
}
Json vec_json(const Eigen::Vector3d& v) { return Json::array({v.x(), v.y(), v.z()}); }
}  // namespace

StoredFrames read_stored_frames(SubjectWriter& sub, const std::string& surface, std::int64_t N) {
  StoredFrames s;
  if (const auto f = field_named(sub, surface + "_canonical_frames")) {
    std::vector<std::int64_t> shape;
    const std::vector<float> d = read_f32(sub.store(), *f->path, &shape);
    if (static_cast<std::int64_t>(d.size()) != N * 9) invalid(surface + "_canonical_frames: " + std::to_string(d.size()) + " values for " + std::to_string(N) + " vertices (3x3 each)");
    s.canonical = true;
    s.e1.resize(N, 3); s.e2.resize(N, 3); s.n.resize(N, 3);
    for (std::int64_t i = 0; i < N; ++i) for (int k = 0; k < 3; ++k) { s.e1(i, k) = d[static_cast<std::size_t>(i * 9 + k)]; s.e2(i, k) = d[static_cast<std::size_t>(i * 9 + 3 + k)]; s.n(i, k) = d[static_cast<std::size_t>(i * 9 + 6 + k)]; }
  }
  const auto ang = field_named(sub, surface + "_anatomical_angle"), sec = field_named(sub, surface + "_anatomical_second"), han = field_named(sub, surface + "_anatomical_handed");
  const auto rel = indices_named(sub, surface + "_anatomical_reliable");
  if (ang && sec && han && rel) {
    const std::vector<float> a = read_f32(sub.store(), *ang->path), b = read_f32(sub.store(), *sec->path), c = read_f32(sub.store(), *han->path);
    std::vector<char> isrel(static_cast<std::size_t>(N), 0);
    for (std::int32_t i : *rel) if (i >= 0 && i < N) isrel[static_cast<std::size_t>(i)] = 1;
    if (static_cast<std::int64_t>(a.size()) != N) invalid(surface + "_anatomical_angle: " + std::to_string(a.size()) + " values for " + std::to_string(N) + " vertices");
    nxr::frames::Anatomical& an = s.anatomy;
    an.angle.resize(N); an.projection = Eigen::VectorXd::Constant(N, std::numeric_limits<double>::quiet_NaN());
    an.second.resize(static_cast<std::size_t>(N)); an.handed.resize(static_cast<std::size_t>(N)); an.reliable.resize(static_cast<std::size_t>(N));
    for (std::int64_t i = 0; i < N; ++i) {
      const std::size_t k = static_cast<std::size_t>(i);
      an.angle[i] = a[k]; an.second[k] = static_cast<std::int8_t>(b[k]); an.handed[k] = static_cast<std::int8_t>(c[k]); an.reliable[k] = isrel[k] && std::isfinite(a[k]);
    }
    const Json& pj = ang->producer;
    an.definition = pj.is_object() ? pj.value("definition", std::string()) : std::string();
    an.reference = pj.is_object() ? pj.value("reference", std::string("canonical frames")) : std::string("canonical frames");
    s.anatomical = true;
    s.anatomical_id = ang->id;
    s.anatomical_modified = ang->modified_utc.value_or("");
  }
  return s;
}

FramesComputeResult frames_compute(DatasetWriter& ds, SubjectWriter& sub, const FramesComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (!(opt.min_projection > 0 && opt.min_projection < 1)) invalid("frames: min_projection is in (0, 1)");
  if (opt.side != "auto" && opt.side != "left" && opt.side != "right") invalid("frames: side is auto, left or right");
  const SurfaceInput S = read_surface(sub, opt.surface, true);   // float64 as stored (Rheome and nxr compute the frames in double)
  const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
  const std::string sname = S.name;
  const Mat3 normals = nxr::frames::vertexNormals(S.vertices.data(), nV, S.faces.data(), nF);
  const auto sphere = sphere_of(sub, sname);
  FramesComputeResult res;
  StoredFrames have = read_stored_frames(sub, sname, nV);
  Json record = Json::object();
  std::vector<float> gauge_angle;
  Mat3 e1, e2;
  Json canonical_info = Json::object();
  nxr::frames::Canonical canon;
  bool stored_canon = false;

  // the canonical frames: read, or computed from the registered sphere
  if (have.canonical) {
    e1 = have.e1; e2 = have.e2;
    canonical_info = Json{{"source", "stored"}};
  } else if (sphere) {
    if (sphere->n_vertices() != nV) invalid(sname + "_sphere has " + std::to_string(sphere->n_vertices()) + " vertices, the surface " + std::to_string(nV));
    std::vector<double> nn(static_cast<std::size_t>(3 * nV));
    for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) nn[static_cast<std::size_t>(3 * i + k)] = normals(i, k);
    canon = nxr::frames::canonicalFrames(S.vertices.data(), nV, S.faces.data(), nF, nn.data(), sphere->vertices.data(), Eigen::Vector3d(0, 0, 1), opt.max_condition);
    e1 = canon.e1; e2 = canon.e2;
    stored_canon = true;
    res.canonical_computed = true;
    int ns = 0;
    for (char c : canon.singular) ns += c != 0;
    canonical_info = Json{{"source", "computed from " + sname + "_sphere"}, {"singular", ns}, {"max_condition", opt.max_condition}, {"poles", canon.poles}};
    say(Json{{"stage", "canonical"}, {"surface", sname}, {"singular", ns}});
  }
  // the signpost gauge (the reference when there is no canonical frame, and the gauge angle of north)
  std::optional<nxr::chart::ChartKit> kit;
  Mat3 ref1, ref2;
  std::string reference = "canonical frames";
  if (e1.rows() == nV) { ref1 = e1; ref2 = e2; }
  else {
    kit.emplace(S.vertices.data(), nV, S.faces.data(), nF);
    const nxr::chart::Basis b = kit->basis();
    ref1.resize(nV, 3); ref2.resize(nV, 3);
    for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) { ref1(i, k) = b.e1[static_cast<std::size_t>(3 * i + k)]; ref2(i, k) = b.e2[static_cast<std::size_t>(3 * i + k)]; }
    reference = "signpost gauge";
  }
  if (stored_canon) {
    nxr::chart::ChartKit k2(S.vertices.data(), nV, S.faces.data(), nF);
    const Eigen::VectorXd g = nxr::frames::gaugeAngle(k2, e1);
    gauge_angle.resize(static_cast<std::size_t>(nV));
    for (int i = 0; i < nV; ++i) gauge_angle[static_cast<std::size_t>(i)] = static_cast<float>(g[i]);
  }

  // the anatomical frame: carried from the template, or from the surface's own axes
  nxr::frames::AnatomicalOptions ao;
  ao.axes = axes_of(opt.convention);
  ao.minProjection = opt.min_projection;
  ao.side = opt.side == "left" ? nxr::frames::Side::Left : opt.side == "right" ? nxr::frames::Side::Right : nxr::frames::Side::Auto;
  nxr::frames::Anatomical A;
  Json axes = Json{{"anterior", vec_json(ao.axes.anterior)}, {"superior", vec_json(ao.axes.superior)}, {"left", vec_json(ao.axes.left)}, {"convention", opt.convention}};
  if (opt.templ) {
    if (!sphere) invalid(sname + ": no registered sphere (" + sname + "_sphere) to carry a template's frame through");
    if (!have.canonical && !stored_canon) invalid(sname + ": no canonical frames");
    SubjectWriter tsub = SubjectWriter::open(ds, *opt.templ);
    const SurfaceInput T = read_surface(tsub, opt.template_surface, true);
    const auto tsphere = sphere_of(tsub, T.name);
    if (!tsphere) invalid(*opt.templ + ": no registered sphere (" + T.name + "_sphere)");
    const StoredFrames th = read_stored_frames(tsub, T.name, T.n_vertices());
    if (!th.anatomical) invalid(*opt.templ + ": no anatomical frame on " + T.name + " (run frames compute on the template first)");
    if (th.anatomy.reference != "canonical frames") invalid(*opt.templ + ": its anatomical frame is not measured from canonical frames");
    A = nxr::frames::carryAnatomical(th.anatomy, tsphere->vertices.data(), static_cast<int>(T.n_vertices()), nxr::frames::pieces(T.faces.data(), static_cast<int>(T.n_faces()), static_cast<int>(T.n_vertices())),
                                     sphere->vertices.data(), nV, nxr::frames::pieces(S.faces.data(), nF, nV), e1);
    A.definition = "template:" + *opt.templ;
    A.reference = "canonical frames";
    axes = Json{{"template", *opt.templ}, {"template_surface", T.name}, {"template_definition", th.anatomy.definition}};
  } else {
    A = nxr::frames::anatomicalFrame(S.vertices.data(), nV, S.faces.data(), nF, normals, ref1, ref2, ao);
    A.definition = opt.convention;
    A.reference = reference;
  }
  int nrel = 0;
  for (char c : A.reliable) nrel += c != 0;
  say(Json{{"stage", "anatomical"}, {"surface", sname}, {"definition", A.definition}, {"reference", A.reference}, {"reliable", nrel}, {"vertices", nV}});

  // store
  bool replaced = false;
  std::string moved;
  const std::string mid = S.id;
  std::vector<std::pair<const char*, std::string>> nodes;
  for (const char* w : {"angle", "second", "handed"}) nodes.push_back({"field", task::Subject::of(sub).node_path(mid, sname + "_anatomical_" + w)});
  nodes.push_back({"selection", sname + "_anatomical_reliable"});
  moved = internal::clear_stored_nodes(ds, sub, nodes, opt.replace, sname + " anatomical frame", replaced);
  Json pj = Json{{"by", "nxr-neuro frames compute"}, {"producer", internal::producer_string("frames compute")}, {"definition", A.definition}, {"reference", A.reference}, {"min_projection", opt.min_projection},
                 {"side", opt.side}, {"axes", axes}, {"canonical", canonical_info}, {"storage", "SCS (x anterior, y left, z up); the angle is from north (e1) towards west (e2) to anterior; the second axis is handed * (n x anterior)"},
                 {"warnings", Json::array({"the anterior direction is undefined where the surface faces forward or back (the poles): `reliable` is 0 there",
                                           "the second axis is superior where superior lies in the surface (lateral, medial cortex) and medial where medial does (dorsal, ventral), per `second`"})}};
  sub.compose([&] {
    auto put = [&](const std::string& name, const std::string& kind, const std::string& vt, const std::string& unit, const std::string& desc, ArrayView data, const Json& j) {
      FieldSpec f;
      f.name = name;
      f.path = task::Subject::of(sub).node_path(mid, name);
      f.kind = kind;
      f.manifold_id = mid;
      f.value_type = vt;
      f.data = data;
      f.cols.set("function", "frames").set("description", desc);
      if (!unit.empty()) f.cols.set("unit", unit);
      f.cols.json("producer_json", j);
      sub.field(std::move(f));
    };
    if (stored_canon) {
      std::vector<float> fr(static_cast<std::size_t>(nV) * 9);
      for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) { fr[static_cast<std::size_t>(i * 9 + k)] = static_cast<float>(e1(i, k)); fr[static_cast<std::size_t>(i * 9 + 3 + k)] = static_cast<float>(e2(i, k)); fr[static_cast<std::size_t>(i * 9 + 6 + k)] = static_cast<float>(normals(i, k)); }
      Json cj = Json{{"by", "nxr-neuro frames compute"}, {"producer", internal::producer_string("frames compute")}, {"method", "Rheome sphereframe: the registered sphere's meridian towards +z pushed forward by a 2x2 Jacobian per vertex (nxr hemisphere_frames, domain sphere)"},
                     {"domain", "sphere"}, {"north_word", "north"}, {"e2_word", "west (n x north; nxr coords.py says east)"}, {"detail", canonical_info}};
      put(sname + "_canonical_frames", "canonical frames", "matrix3", "", "per vertex rows e1 (north), e2 = n x e1 (west), n (outward); NaN at singular vertices", view(fr, {static_cast<std::int64_t>(nV), 3, 3}), cj);
      std::vector<std::int32_t> sing;
      for (int i = 0; i < nV; ++i) if (canon.singular[static_cast<std::size_t>(i)]) sing.push_back(i);
      SelectionSpec ss;
      ss.name = sname + "_frame_singularities"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = mid; ss.data = view(sing, {static_cast<std::int64_t>(sing.size())}); ss.picked = true;
      ss.cols.set("description", "the vertices with no canonical frame: the poles, and ill-conditioned push-forwards");
      sub.selection(std::move(ss));
      put(sname + "_canonical_gauge_angle", "canonical frames", "scalar", "rad", "the angle of north (e1) in each vertex's signpost basis (gaugeHeading); a heading from north = heading in the gauge minus this", view(gauge_angle, {static_cast<std::int64_t>(nV)}), cj);
    }
    std::vector<float> ang(static_cast<std::size_t>(nV));
    std::vector<std::int32_t> sec(static_cast<std::size_t>(nV)), han(static_cast<std::size_t>(nV)), rel;
    for (int i = 0; i < nV; ++i) {
      const std::size_t k = static_cast<std::size_t>(i);
      ang[k] = static_cast<float>(A.angle[i]); sec[k] = A.second[k]; han[k] = A.handed[k]; if (A.reliable[k]) rel.push_back(i);
    }
    const std::int64_t n = nV;
    put(sname + "_anatomical_angle", "anatomical frames", "scalar", "rad", "the angle from north (towards west) to ANTERIOR at each vertex; anatomical components are a fixed rotation of the canonical frame by it", view(ang, {n}), pj);
    put(sname + "_anatomical_second", "anatomical frames", "integer", "", "which way the second axis points: 0 superior, 1 medial", view(sec, {n}), pj);
    put(sname + "_anatomical_handed", "anatomical frames", "integer", "", "the sign of n x anterior the second axis takes (+1 / -1)", view(han, {n}), pj);
    SelectionSpec rs;
    rs.name = sname + "_anatomical_reliable"; rs.path = rs.name; rs.type = "indices"; rs.manifold_id = mid; rs.data = view(rel, {static_cast<std::int64_t>(rel.size())}); rs.picked = true;
    rs.cols.set("description", "the vertices where the anatomical frame is reliable: the anterior axis projects on the tangent plane with at least min_projection (not the frontal and occipital poles)");
    sub.selection(std::move(rs));
  });
  res.info = Json{{"surface", sname}, {"vertices", nV}, {"canonical", canonical_info}, {"definition", A.definition}, {"reference", A.reference}, {"reliable", nrel}, {"min_projection", opt.min_projection},
                  {"fields", Json::array({sname + "_anatomical_angle", sname + "_anatomical_second", sname + "_anatomical_handed"})}, {"selections", Json::array({sname + "_anatomical_reliable"})}, {"replaced", replaced},
                  {"moved_aside", moved.empty() ? Json() : Json(moved)}};
  return res;
}

}  // namespace nxr::pipelines
