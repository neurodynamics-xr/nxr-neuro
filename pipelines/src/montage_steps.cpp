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

void add_montage_steps(std::vector<StepSpec>& r) {
  StepSpec s;
  s.name = "eeg_montage_1005";
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
    out["positions"] = array_value(Json{{"node", "eeg_montage_1005"}, {"port", "positions"}});
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
    Arrays o;
    o["positions"] = std::move(out);
    return o;
  };
  r.push_back(std::move(s));
}

}  // namespace nxr::pipelines::graph
