// nxr-pipelines — the registered NODE of the standard electrode montage. A pure function of arrays from compute/ (nxr/meeg/montage.h) with its ports, parameters and traits; it reads no file, subject
// or database, and has no time axis, so it runs once, before the chunk loop (kind `setup`).
//
//   eeg_montage_1005   scalp_vertices, scalp_faces, landmarks → positions   the 10-05 positions (the 10-10 and 10-20 are subsets of its names) on a closed scalp surface by the published rule
//
// ARRAY CONVENTIONS (an `Array` is [rows x cols] doubles with a `meta`): scalp_vertices [V x 3] in the scalp's unit; scalp_faces [F x 3] 0-based vertex indices stored as doubles; landmarks [4 x 3],
// the rows nasion, inion, LPA, RPA. positions [345 x 3] in the same unit and frame, row k named meta.labels[k]; meta.systems lists the 10-10 and 10-20 names. Precision follows the input: meta
// {"precision": "float32" | "float64"} on scalp_vertices sets the precision the positions are given back in (the arcs are computed in double either way); the three inputs must agree.
//
// What it cannot resolve: see the method's section in docs/reference.md (the system's positions on a head model, not a cap's placement; arcs along a triangle mesh).
#include <cmath>
#include <set>
#include <stdexcept>

#include "nxr/meeg/montage.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
bool is_f32(const Array& a) { return a.meta.is_object() && a.meta.value("precision", std::string("float64")) == "float32"; }

Value array_value(const Json& facts) {
  Value v;
  v.cls = "array";
  v.facts = facts;
  return v;
}

const std::vector<std::string> kWarnings = {
    "these are the standard system's positions on this head model, not where a cap sits on a person: a cap's placement, the hair and the head's shape under it are not resolved",
    "arcs are measured along a triangle mesh, which shortens them by a term that goes as the square of the edge length; the landmarks are the caller's (the inion above all) and each is moved to its nearest point of the scalp"};

}  // namespace

namespace {
const std::vector<std::string> kLandmarkNames = {"NAS", "LPA", "RPA"};

Array column_of(const std::vector<double>& x, const std::vector<std::string>& entries, const std::string& description) {
  Array a;
  a.rows = static_cast<std::int64_t>(x.size()); a.cols = 1;
  a.data = x;
  a.meta["describe"] = Json{{"entries", entries}, {"description", description}};
  return a;
}
StepSpec setup_step(const std::string& name, std::vector<std::string> ins, std::vector<std::string> outs, Json params, const std::string& description, const std::string& why, bool reads_arrays) {
  StepSpec s;
  s.name = name; s.kind = "setup";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, why);
  s.inputs = std::move(ins); s.outputs = std::move(outs);
  s.parameters = std::move(params);
  s.description = description;
  s.reads_arrays = reads_arrays;
  s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  return s;
}
}  // namespace

void add_montage_steps(std::vector<StepSpec>& r) {
  // ── geometry.surface_parts: a stored surface as its two arrays ──
  {
    StepSpec s = setup_step("geometry.surface_parts", {"surface"}, {"vertices", "faces"}, Json::object(),
                            "A surface as two arrays: vertices [V x 3] in the surface's unit (its precision and coordinate system in meta) and faces [F x 3] (0-based indices as doubles).", "indexing only", true);
    s.matches = "the surface's stored vertices and faces";
    s.plan = [](const Values& ins, const Json&, const PlanContext&) {
      need(ins.count("surface") != 0, "surface_parts: the input `surface` is missing");
      Values out;
      for (const char* o : {"vertices", "faces"}) out[o] = array_value(Json{{"node", "geometry.surface_parts"}, {"port", o}});
      return out;
    };
    s.run = [](const std::any&, const Arrays& ins, ExecContext& ctx) {
      const Array& S = ins.at("surface");
      Array V, F;
      V.rows = S.rows; V.cols = 3; V.data = S.data;
      V.meta = Json{{"precision", S.meta.value("precision", std::string("float64"))}, {"unit", S.meta.value("unit", std::string())}, {"name", S.meta.value("name", std::string())},
                    {"cs", ctx.bound.coordinate_system_key(S.meta.value("id", std::string()))}};
      const auto faces = S.meta.value("faces", std::vector<std::int32_t>());
      need(faces.size() % 3 == 0 && !faces.empty(), "surface_parts: the surface has no triangles");
      F.rows = static_cast<std::int64_t>(faces.size() / 3); F.cols = 3;
      for (const auto f : faces) F.data.push_back(f);
      Arrays o;
      o["vertices"] = std::move(V); o["faces"] = std::move(F);
      return o;
    };
    r.push_back(std::move(s));
  }
  // ── meeg.inion_from_landmarks ──
  {
    StepSpec s = setup_step("meeg.inion_from_landmarks", {"vertices", "faces", "landmarks"}, {"landmarks_with_inion"}, Json{{"inion", nullptr}},
                            "The four landmarks the 10-05 rule needs: landmarks [3 x 3] (rows NAS, LPA, RPA, named in meta.names) on a scalp (vertices, faces) with the inion added -> landmarks_with_inion [4 x 3] (nasion, inion, LPA, RPA). "
                            "The inion is by default `nxr::meeg::inionFromLandmarks` (the farthest crossing of the line from the nasion through the midpoint of the preauricular points with the scalp); the parameter `inion` "
                            "([x, y, z] in the scalp's unit) gives it instead. The convention is in meta.inion. The landmarks and the scalp must be in one coordinate system.", "no time axis: runs once on arrays", true);
    s.matches = "nxr::meeg::inionFromLandmarks (a stated convention, not the bump the inion is)";
    s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
      for (const char* k : {"vertices", "faces", "landmarks"}) need(ins.count(k) != 0, std::string("inion_from_landmarks: the input `") + k + "` is missing");
      need(p["inion"].is_null() || (p["inion"].is_array() && p["inion"].size() == 3), "inion_from_landmarks: inion is [x, y, z] in the scalp's unit, or null for the convention");
      Values out;
      out["landmarks_with_inion"] = array_value(Json{{"node", "meeg.inion_from_landmarks"}, {"port", "landmarks_with_inion"}});
      return out;
    };
    s.run = [](const std::any& st, const Arrays& ins, ExecContext& ctx) {
      const Json& p = std::any_cast<const Json&>(st);
      const Array &V = ins.at("vertices"), &F = ins.at("faces"), &L = ins.at("landmarks");
      need(L.cols == 3 && L.rows == 3, "inion_from_landmarks: the landmarks hold " + std::to_string(L.rows) + " points, not the nasion, LPA and RPA");
      const auto names = L.meta.value("names", std::vector<std::string>());
      need(names == kLandmarkNames, "inion_from_landmarks: the landmarks name their points " + (names.empty() ? std::string("not at all") : names[0] + ",...") + ", not NAS, LPA, RPA in that order");
      const std::string scs = V.meta.value("cs", std::string()), lcs = ctx.bound.coordinate_system_key(L.meta.value("id", std::string()));
      need(!scs.empty() && !lcs.empty(), "inion_from_landmarks: the scalp or the landmarks have no coordinate system");
      need(scs == lcs, "inion_from_landmarks: the scalp is in '" + scs + "' and the landmarks in '" + lcs + "': place them in one system first");
      std::vector<std::int32_t> faces(F.data.size());
      for (std::size_t i = 0; i < faces.size(); ++i) faces[i] = static_cast<std::int32_t>(std::llround(F.data[i]));
      nxr::meeg::Point nas, lpa, rpa, inion;
      for (int c = 0; c < 3; ++c) { nas[c] = L.data[static_cast<std::size_t>(c)]; lpa[c] = L.data[static_cast<std::size_t>(3 + c)]; rpa[c] = L.data[static_cast<std::size_t>(6 + c)]; }
      Json info;
      if (p["inion"].is_array()) {
        for (int c = 0; c < 3; ++c) inion[c] = p["inion"][static_cast<std::size_t>(c)].get<double>();
        info = Json{{"convention", "given by the caller"}};
      } else {
        try { inion = nxr::meeg::inionFromLandmarks(V.data, faces, nas, lpa, rpa); } catch (const std::invalid_argument& e) { refuse(std::string("inion_from_landmarks: ") + e.what()); }
        info = Json{{"convention", "nxr::meeg::inionFromLandmarks: the farthest point where the line from the nasion through the midpoint of the preauricular points crosses the scalp"}};
      }
      info["point"] = Json::array({inion[0], inion[1], inion[2]});
      Array out;
      out.rows = 4; out.cols = 3;
      for (const auto* q : {&nas, &inion, &lpa, &rpa}) for (int c = 0; c < 3; ++c) out.data.push_back((*q)[c]);
      out.meta = Json{{"precision", V.meta.value("precision", std::string("float64"))}, {"inion", info}};
      Arrays o;
      o["landmarks_with_inion"] = std::move(out);
      return o;
    };
    r.push_back(std::move(s));
  }
  // ── meeg.montage_labels ──
  {
    StepSpec s = setup_step("meeg.montage_labels", {"positions"}, {"names", "system"}, Json::object(),
                            "The labellings of a montage: positions [n x 3] (meta.labels, meta.systems) -> names [n] (one level per point, the 10-05 name) and system [n] (the narrowest standard system a point belongs to: level 0 `10-20`, 1 `10-10`, "
                            "2 `10-05`; the 10-10 system is levels 0 and 1 together, the 10-05 all).", "each point is named on its own", false);
    s.matches = "Oostenveld and Praamstra (2001); Jurcak, Tsuzuki and Dan (2007) for the 10-10 names";
    s.plan = [](const Values& ins, const Json&, const PlanContext&) {
      need(ins.count("positions") != 0, "montage_labels: the input `positions` is missing");
      Values out;
      for (const char* o : {"names", "system"}) out[o] = array_value(Json{{"node", "meeg.montage_labels"}, {"port", o}});
      return out;
    };
    s.run = [](const std::any&, const Arrays& ins, ExecContext&) {
      const Array& P = ins.at("positions");
      const auto labels = P.meta.value("labels", std::vector<std::string>());
      need(static_cast<std::int64_t>(labels.size()) == P.rows && P.rows > 0, "montage_labels: the positions carry no name for each point (meta.labels)");
      const std::set<std::string> s1020(nxr::meeg::labels1020().begin(), nxr::meeg::labels1020().end()), s1010(nxr::meeg::labels1010().begin(), nxr::meeg::labels1010().end());
      std::vector<double> codes, sys;
      for (std::size_t i = 0; i < labels.size(); ++i) { codes.push_back(static_cast<double>(i)); sys.push_back(s1020.count(labels[i]) ? 0 : s1010.count(labels[i]) ? 1 : 2); }
      Arrays o;
      o["names"] = column_of(codes, labels, "the 10-05 names, one level per point; the old 10-20 names (T3, T4, T5, T6) are aliases listed in the montage's producer");
      o["system"] = column_of(sys, {"10-20", "10-10", "10-05"}, "the narrowest standard system each point belongs to: 10-20, then 10-10 (the 10-10 system is both), then 10-05 (the whole montage)");
      return o;
    };
    r.push_back(std::move(s));
  }

  StepSpec s;
  s.name = "meeg.eeg_montage_1005";
  s.kind = "setup";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::follows_input, "no time axis: runs once on arrays; the arcs are computed in double and the positions given back in the scalp's precision");
  s.inputs = {"scalp_vertices", "scalp_faces", "landmarks"};
  s.outputs = {"positions"};
  s.parameters = Json{{"tolerance", 1e-7}, {"max_iterations", 100}};
  s.matches = "Oostenveld R, Praamstra P (2001) Clin Neurophysiol 112:713-719, the 10-05 rule; held equal to MNE-Python's spherical_1005 (eeg_positions) on a sphere (nxr::meeg::tenFive)";
  s.description =
      "The 10-05 electrode positions on a closed scalp surface by the published rule: scalp_vertices [V x 3] and scalp_faces [F x 3] (0-based indices as doubles) and landmarks [4 x 3] (rows nasion, inion, LPA, RPA), "
      "all in the scalp's unit → positions [345 x 3], row k named meta.labels[k]; the 10-10 (77 names) and 10-20 (21) are subsets of the names (meta.systems). Parameters: tolerance (when Cz stops moving, a fraction "
      "of the nasion - inion distance, default 1e-7), max_iterations (a count, default 100; the node refuses when Cz has not settled).";
  s.plan = [](const Values& ins, const Json&, const PlanContext&) {
    for (const char* p : {"scalp_vertices", "scalp_faces", "landmarks"}) need(ins.count(p) != 0, std::string("eeg_montage_1005: the input `") + p + "` is missing");
    Values out;
    out["positions"] = array_value(Json{{"node", "meeg.eeg_montage_1005"}, {"port", "positions"}});
    return out;
  };
  s.prepare = [](const Values&, const Json& p, ExecContext&) {
    need(p["tolerance"].is_number() && p["tolerance"].get<double>() > 0, "eeg_montage_1005: tolerance is a fraction of the nasion - inion distance, a number > 0");
    need(p["max_iterations"].is_number_integer() && p["max_iterations"].get<int>() >= 1, "eeg_montage_1005: max_iterations is a count, a whole number >= 1");
    nxr::meeg::Options o;
    o.tolerance = p["tolerance"].get<double>();
    o.max_iterations = p["max_iterations"].get<int>();
    return std::any(o);
  };
  s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const auto opt = std::any_cast<nxr::meeg::Options>(st);
    const Array &V = ins.at("scalp_vertices"), &F = ins.at("scalp_faces"), &L = ins.at("landmarks");
    need(V.cols == 3 && F.cols == 3 && L.cols == 3 && L.rows == 4, "eeg_montage_1005: scalp_vertices [V x 3], scalp_faces [F x 3] and landmarks [4 x 3] (nasion, inion, LPA, RPA)");
    need(is_f32(V) == is_f32(L), "eeg_montage_1005: the scalp and the landmarks are in different precision: convert one explicitly");
    std::vector<std::int32_t> faces(F.data.size());
    for (std::size_t i = 0; i < faces.size(); ++i) faces[i] = static_cast<std::int32_t>(std::llround(F.data[i]));
    nxr::meeg::Landmarks lm;
    nxr::meeg::Point* pts[4] = {&lm.nasion, &lm.inion, &lm.lpa, &lm.rpa};
    for (int k = 0; k < 4; ++k) for (int c = 0; c < 3; ++c) (*pts[k])[c] = L.data[static_cast<std::size_t>(3 * k + c)];
    nxr::meeg::Placement p;
    try { p = nxr::meeg::tenFive(V.data, faces, lm, opt); } catch (const std::invalid_argument& e) { refuse(e.what()); }
    Array out;
    out.rows = static_cast<std::int64_t>(p.labels.size());
    out.cols = 3;
    out.data = p.positions;
    const bool f32 = is_f32(V);
    if (f32) for (auto& x : out.data) x = static_cast<double>(static_cast<float>(x));
    Json shift = Json::array();
    for (const double d : p.landmark_shift) shift.push_back(d);
    out.meta = Json{{"precision", f32 ? "float32" : "float64"},
                    {"labels", p.labels},
                    {"systems", Json{{"10-10", nxr::meeg::labels1010()}, {"10-20", nxr::meeg::labels1020()}}},
                    {"old_names", nxr::meeg::oldNames()},
                    {"cz_iterations", p.iterations},
                    {"cz_residual", p.cz_residual},
                    {"landmark_shift", shift},
                    {"warnings", kWarnings}};
    Json inion = L.meta.is_object() && L.meta.contains("inion") ? L.meta["inion"] : Json();
    const Json counts = Json{{"10-05", static_cast<std::int64_t>(p.labels.size())}, {"10-10", static_cast<std::int64_t>(nxr::meeg::labels1010().size())}, {"10-20", static_cast<std::int64_t>(nxr::meeg::labels1020().size())}};
    out.meta["describe"] = Json{{"cs", V.meta.is_object() ? V.meta.value("cs", std::string()) : std::string()}, {"unit", V.meta.is_object() ? V.meta.value("unit", std::string("m")) : std::string("m")},
        {"description", "the standard 10-05 electrode positions on the scalp, one point per name (10-10 and 10-20 are subsets of the names)"},
        {"producer", Json{{"method", "Oostenveld and Praamstra (2001), the 10-05 rule: nxr::meeg::tenFive"}, {"inion", inion}, {"landmark_shift", shift}, {"cz_iterations", p.iterations}, {"cz_residual", p.cz_residual},
                          {"tolerance", opt.tolerance}, {"aliases", nxr::meeg::oldNames()}, {"systems", counts}, {"warnings", kWarnings}}}};
    Arrays o;
    o["positions"] = std::move(out);
    return o;
  };
  r.push_back(std::move(s));
}

}  // namespace nxr::pipelines::graph
