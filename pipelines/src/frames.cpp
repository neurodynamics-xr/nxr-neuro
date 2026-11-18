#include "nxr/pipelines/frames.h"

#include <cmath>
#include <cstring>
#include <filesystem>

#include "nxr/geometry/charts.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/pipelines/task.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;
using nxr::geometry::Mat3;

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
nxr::geometry::Axes axes_of(const std::string& convention) {
  nxr::geometry::Axes a;
  if (convention == "scs") return a;
  if (convention == "ras") { a.anterior = {0, 1, 0}; a.superior = {0, 0, 1}; a.left = {-1, 0, 0}; return a; }
  invalid("frames: convention is scs or ras");
}
Json vec_json(const Eigen::Vector3d& v) { return Json::array({v.x(), v.y(), v.z()}); }
}  // namespace

Json gauge_block(nxr::chart::ChartKit& kit, int n_vertices, const std::string& surface_id) {
  std::vector<const nxr::geometry::IntrinsicSurface*> seen;
  std::string fingerprint;
  Json rec;
  for (int v = 0; v < n_vertices; ++v) {
    nxr::geometry::IntrinsicSurface& s = kit.surfaceOf(v);
    if (std::find(seen.begin(), seen.end(), &s) != seen.end()) continue;
    seen.push_back(&s);
    const nxr::geometry::IntrinsicRecord& r = s.record();
    fingerprint += (fingerprint.empty() ? "" : "+") + r.fingerprint;
    if (rec.is_null()) rec = Json{{"gauge", r.gauge}, {"mollify", r.mollify}, {"epsilon", r.epsilon}, {"flipEps", r.flipEps}, {"mass", r.mass}, {"boundary_vertices", 0}};
    rec["epsilon"] = std::max(rec["epsilon"].get<double>(), r.epsilon);
    rec["boundary_vertices"] = rec["boundary_vertices"].get<int>() + r.boundaryVertices;
  }
  if (rec.is_null()) return Json();
  const std::string gauge = rec["gauge"].get<std::string>();
  rec.erase("gauge");
  rec["fingerprint"] = fingerprint;
  return Json{{"gauge", gauge}, {"gauge_surface", surface_id}, {"gauge_record", rec}};
}

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
    nxr::geometry::Anatomical& an = s.anatomy;
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

FramesMade make_frames(DatasetWriter* dsp, SubjectWriter& sub, const FramesComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (!(opt.min_projection > 0 && opt.min_projection < 1)) invalid("frames: min_projection is in (0, 1)");
  if (opt.side != "auto" && opt.side != "left" && opt.side != "right") invalid("frames: side is auto, left or right");
  const SurfaceInput S = read_surface(sub, opt.surface, true);   // float64 as stored (Rheome and nxr compute the frames in double)
  const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
  const std::string sname = S.name;
  const Mat3 normals = nxr::geometry::vertexNormals(S.vertices.data(), nV, S.faces.data(), nF);
  const auto sphere = sphere_of(sub, sname);
  FramesMade res;
  StoredFrames have = read_stored_frames(sub, sname, nV);
  Json record = Json::object();
  std::vector<float> gauge_angle;
  Json angle_gauge;                                                     // the gauge record of the gauge angle (G6)
  Mat3 e1, e2;
  Json canonical_info = Json::object();
  nxr::geometry::Canonical canon;
  bool stored_canon = false;

  // the canonical frames: read, or computed from the registered sphere
  if (have.canonical) {
    e1 = have.e1; e2 = have.e2;
    canonical_info = Json{{"source", "stored"}};
  } else if (sphere) {
    if (sphere->n_vertices() != nV) invalid(sname + "_sphere has " + std::to_string(sphere->n_vertices()) + " vertices, the surface " + std::to_string(nV));
    std::vector<double> nn(static_cast<std::size_t>(3 * nV));
    for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) nn[static_cast<std::size_t>(3 * i + k)] = normals(i, k);
    canon = nxr::geometry::canonicalFrames(S.vertices.data(), nV, S.faces.data(), nF, nn.data(), sphere->vertices.data(), Eigen::Vector3d(0, 0, 1), opt.max_condition);
    e1 = canon.e1; e2 = canon.e2;
    stored_canon = true;
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
    const Eigen::VectorXd g = nxr::geometry::gaugeAngle(k2, e1);
    angle_gauge = gauge_block(k2, nV, S.id);
    gauge_angle.resize(static_cast<std::size_t>(nV));
    for (int i = 0; i < nV; ++i) gauge_angle[static_cast<std::size_t>(i)] = static_cast<float>(g[i]);
  }

  // the anatomical frame: carried from the template, or from the surface's own axes
  nxr::geometry::AnatomicalOptions ao;
  ao.axes = axes_of(opt.convention);
  ao.minProjection = opt.min_projection;
  ao.side = opt.side == "left" ? nxr::geometry::Side::Left : opt.side == "right" ? nxr::geometry::Side::Right : nxr::geometry::Side::Auto;
  nxr::geometry::Anatomical A;
  Json axes = Json{{"anterior", vec_json(ao.axes.anterior)}, {"superior", vec_json(ao.axes.superior)}, {"left", vec_json(ao.axes.left)}, {"convention", opt.convention}};
  if (opt.templ) {
    if (!sphere) invalid(sname + ": no registered sphere (" + sname + "_sphere) to carry a template's frame through");
    if (!have.canonical && !stored_canon) invalid(sname + ": no canonical frames");
    if (!dsp) invalid("frames: a template's frame is carried with the dataset open");
    SubjectWriter tsub = SubjectWriter::open(*dsp, *opt.templ);
    const SurfaceInput T = read_surface(tsub, opt.template_surface, true);
    const auto tsphere = sphere_of(tsub, T.name);
    if (!tsphere) invalid(*opt.templ + ": no registered sphere (" + T.name + "_sphere)");
    const StoredFrames th = read_stored_frames(tsub, T.name, T.n_vertices());
    if (!th.anatomical) invalid(*opt.templ + ": no anatomical frame on " + T.name + " (run frames compute on the template first)");
    if (th.anatomy.reference != "canonical frames") invalid(*opt.templ + ": its anatomical frame is not measured from canonical frames");
    A = nxr::geometry::carryAnatomical(th.anatomy, tsphere->vertices.data(), static_cast<int>(T.n_vertices()), nxr::geometry::pieces(T.faces.data(), static_cast<int>(T.n_faces()), static_cast<int>(T.n_vertices())),
                                     sphere->vertices.data(), nV, nxr::geometry::pieces(S.faces.data(), nF, nV), e1);
    A.definition = "template:" + *opt.templ;
    A.reference = "canonical frames";
    axes = Json{{"template", *opt.templ}, {"template_surface", T.name}, {"template_definition", th.anatomy.definition}};
  } else {
    A = nxr::geometry::anatomicalFrame(S.vertices.data(), nV, S.faces.data(), nF, normals, ref1, ref2, ao);
    A.definition = opt.convention;
    A.reference = reference;
  }
  int nrel = 0;
  for (char c : A.reliable) nrel += c != 0;
  say(Json{{"stage", "anatomical"}, {"surface", sname}, {"definition", A.definition}, {"reference", A.reference}, {"reliable", nrel}, {"vertices", nV}});

  // WHAT WAS MADE (stored by the definition `frames`): the canonical frames when computed here, their singular vertices and the gauge angle; the anatomical frame
  int nrel2 = 0;
  for (char c : A.reliable) nrel2 += c != 0;
  (void)nrel2;
  res.surface_id = S.id; res.surface = sname; res.n = nV;
  res.producer = Json{{"definition", A.definition}, {"reference", A.reference}, {"min_projection", opt.min_projection}, {"side", opt.side}, {"axes", axes}, {"canonical", canonical_info},
                      {"storage", "SCS (x anterior, y left, z up); the angle is from north (e1) towards west (e2) to anterior; the second axis superior or medial per `second`"},
                      {"warnings", Json::array({"the anterior direction is undefined where the surface faces forward or back (the poles): `reliable` is 0 there",
                                                "the second axis is superior where superior lies in the surface (lateral, medial cortex) and medial where medial does (dorsal, ventral), per `second`"})}};
  if (stored_canon) {
    res.canonical.resize(static_cast<std::size_t>(nV) * 9);
    for (int i = 0; i < nV; ++i) {
      const Eigen::Vector3d a(e1(i, 0), e1(i, 1), e1(i, 2)), c(e2(i, 0), e2(i, 1), e2(i, 2)), nn = a.cross(c);   // n = e1 x e2 (outward)
      for (int k = 0; k < 3; ++k) {
        res.canonical[static_cast<std::size_t>(i * 9 + k)] = e1(i, k);
        res.canonical[static_cast<std::size_t>(i * 9 + 3 + k)] = e2(i, k);
        res.canonical[static_cast<std::size_t>(i * 9 + 6 + k)] = nn[k];
      }
    }
    for (int i = 0; i < nV; ++i) if (canon.singular[static_cast<std::size_t>(i)]) res.singular.push_back(i);
    res.gauge_angle.assign(gauge_angle.begin(), gauge_angle.end());
    res.gauge = angle_gauge;
    res.canonical_producer = Json{{"method", "Rheome sphereframe: the registered sphere's meridian towards +z pushed forward to the surface"}, {"domain", "sphere"}, {"north_word", "north"},
                                  {"e2_word", "west (n x north)"}, {"detail", canonical_info}};
  }
  res.angle.resize(static_cast<std::size_t>(nV)); res.second.resize(static_cast<std::size_t>(nV)); res.handed.resize(static_cast<std::size_t>(nV));
  for (int i = 0; i < nV; ++i) {
    const std::size_t k = static_cast<std::size_t>(i);
    res.angle[k] = A.angle[i]; res.second[k] = A.second[k]; res.handed[k] = A.handed[k];
    if (A.reliable[k]) res.reliable.push_back(i);
  }
  return res;
}

}  // namespace nxr::pipelines
