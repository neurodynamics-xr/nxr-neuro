#include "nxr/io/subject_api.h"
#include "nxr/pipelines/montage_create.h"

#include <cmath>
#include <map>
#include <set>

#include "nxr/bst/entities.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/writer.h"
#include "nxr/meeg/montage.h"
#include "nxr/pipelines/input.h"

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
}  // namespace

MontageCreateResult montage_create(const MontageCreateOptions& opt, const std::function<void(const Json&)>& progress) {
  DatasetWriter w = open_package_writer(opt.package);
  std::string subject = opt.subject;
  if (subject.empty()) {
    const auto rows = nxr::io::complete_subjects(w);
    if (rows.size() != 1) invalid("montage create: the package holds " + std::to_string(rows.size()) + " subjects: name one (--subject)");
    subject = text(rows[0], "name");
  }
  SubjectWriter sub = SubjectWriter::open(w, subject);
  // the scalp and the landmarks
  const SurfaceInput scalp = read_surface(sub, opt.scalp);
  const auto lm_found = nxr::io::find_manifold(sub, opt.landmarks, std::string("points"));
  if (!lm_found) invalid("montage create: " + subject + " has no points manifold '" + opt.landmarks + "' (the nasion, LPA and RPA of the template)");
  const auto lm_row = nxr::io::manifold_with_geometry(sub, text(*lm_found, "id"));
  if (!lm_row) invalid("montage create: '" + opt.landmarks + "' has no geometry");
  if (integer(*lm_row, "n_vertices") != 3) invalid("montage create: '" + opt.landmarks + "' holds " + std::to_string(integer(*lm_row, "n_vertices")) + " points, not the nasion, LPA and RPA");
  if (nxr::io::find_manifold(sub, opt.name)) invalid("montage create: " + opt.name + " exists already");
  // the same coordinate system: the scalp's and the landmarks' (a landmark in another system is not the scalp's landmark)
  const std::string scalp_cs = nxr::io::manifold_coordinate_system_name(sub, scalp.id), lm_cs = nxr::io::manifold_coordinate_system_name(sub, text(*lm_row, "id"));
  if (scalp_cs.empty() || lm_cs.empty()) invalid("montage create: the scalp or the landmarks have no coordinate system");
  if (scalp_cs != lm_cs) invalid("montage create: the scalp is in '" + scalp_cs + "' and the landmarks in '" + lm_cs + "': place them in one system first");
  const std::string cs_key = scalp_cs;
  // landmark names: the landmarks' names labelling's lookup, in order NAS, LPA, RPA (by code)
  const std::vector<double> lp = [&] {
    ArrayReader r = ArrayReader::open(sub.store(), text(*lm_row, "positions_path"));
    const std::vector<std::uint8_t> raw = r.read();
    std::vector<double> out;
    for (std::size_t i = 0; i < raw.size() / r.itemsize(); ++i) {
      if (r.dtype() == DType::Float32) { float v; std::memcpy(&v, raw.data() + 4 * i, 4); out.push_back(v); }
      else if (r.dtype() == DType::Float64) { double v; std::memcpy(&v, raw.data() + 8 * i, 8); out.push_back(v); }
      else throw Error(code::unsupported, opt.landmarks + ": positions of type " + dtype_to_string(r.dtype()) + " are not read here");
    }
    return out;
  }();
  if (lp.size() != 9) invalid("montage create: '" + opt.landmarks + "' positions are not [3 x 3]");
  std::vector<std::string> names;
  if (const auto lab = nxr::io::selection::labelling_row(sub, opt.landmarks + "_names", text(*lm_row, "id")); lab && !opt_text(*lab, "dictionary_id").value_or("").empty())
    for (const auto& [code_, name] : nxr::io::selection::lookup_names(sub, text(*lab, "dictionary_id"))) names.push_back(name);
  if (names != std::vector<std::string>({"NAS", "LPA", "RPA"})) invalid("montage create: '" + opt.landmarks + "' names its points " + (names.empty() ? std::string("not at all") : names[0] + ",...") + ", not NAS, LPA, RPA in that order");

  nxr::meeg::Landmarks L;
  for (int c = 0; c < 3; ++c) { L.nasion[static_cast<std::size_t>(c)] = lp[static_cast<std::size_t>(c)]; L.lpa[static_cast<std::size_t>(c)] = lp[static_cast<std::size_t>(3 + c)]; L.rpa[static_cast<std::size_t>(c)] = lp[static_cast<std::size_t>(6 + c)]; }
  Json inion_info;
  try {
    if (opt.inion) {
      L.inion = *opt.inion;
      inion_info = Json{{"convention", "given by the caller"}, {"point", Json::array({L.inion[0], L.inion[1], L.inion[2]})}};
    } else {
      L.inion = nxr::meeg::inionFromLandmarks(scalp.vertices, scalp.faces, L.nasion, L.lpa, L.rpa);
      inion_info = Json{{"convention", "nxr::meeg::inionFromLandmarks: the farthest point where the line from the nasion through the midpoint of the preauricular points crosses the scalp"},
                        {"point", Json::array({L.inion[0], L.inion[1], L.inion[2]})}};
    }
  } catch (const std::invalid_argument& e) { invalid(std::string("montage create: the inion: ") + e.what()); }
  nxr::meeg::Options mo;
  mo.tolerance = opt.tolerance;
  mo.max_iterations = opt.max_iterations;
  nxr::meeg::Placement P;
  try { P = nxr::meeg::tenFive(scalp.vertices, scalp.faces, L, mo); } catch (const std::invalid_argument& e) { invalid(std::string("montage create: ") + e.what()); }
  const std::int64_t n = static_cast<std::int64_t>(P.labels.size());
  if (progress) progress(Json{{"stage", "placed"}, {"points", n}, {"cz_iterations", P.iterations}});

  // the system partition: the narrowest system of each point
  const std::set<std::string> s1020(nxr::meeg::labels1020().begin(), nxr::meeg::labels1020().end()), s1010(nxr::meeg::labels1010().begin(), nxr::meeg::labels1010().end());
  std::vector<std::int64_t> sys(static_cast<std::size_t>(n));
  Json counts = Json{{"10-20", 0}, {"10-10", 0}, {"10-05", 0}};
  for (std::int64_t i = 0; i < n; ++i) {
    const std::string& l = P.labels[static_cast<std::size_t>(i)];
    sys[static_cast<std::size_t>(i)] = s1020.count(l) ? 0 : s1010.count(l) ? 1 : 2;
  }
  for (const auto c : sys) counts[c == 0 ? "10-20" : c == 1 ? "10-10" : "10-05"] = counts[c == 0 ? "10-20" : c == 1 ? "10-10" : "10-05"].get<int>() + 1;

  Json aliases = Json::object();
  for (const auto& [old, now] : nxr::meeg::oldNames()) aliases[old] = now;
  Json shift = Json::array();
  for (const double d : P.landmark_shift) shift.push_back(d);
  const Json warnings = Json::array({"these are the standard system's positions on this head model, not where a cap sits on a person",
                                     "the inion is a stated convention (" + inion_info["convention"].get<std::string>() + "), not the bump the inion is"});
  const Json producer = Json{{"method", "Oostenveld and Praamstra (2001), the 10-05 rule: nxr::meeg::tenFive"}, {"scalp", opt.scalp}, {"landmarks", opt.landmarks}, {"inion", inion_info},
                             {"landmark_shift", shift}, {"cz_iterations", P.iterations}, {"cz_residual", P.cz_residual}, {"tolerance", opt.tolerance}, {"aliases", aliases},
                             {"systems", Json{{"10-05", n}, {"10-10", static_cast<std::int64_t>(s1010.size())}, {"10-20", static_cast<std::int64_t>(s1020.size())}}},
                             {"unit", "m"}, {"warnings", warnings}, {"computed_by", opt.created_by}};

  const std::string path = folder_for("points") + "/" + opt.name, mid = uuid7();
  std::vector<float> pos(P.positions.begin(), P.positions.end());
  ManifoldSpec m;
  m.id = mid;
  m.name = opt.name;
  m.path = path;
  m.type = "points";
  m.n_vertices = n;
  m.topology = {"none", 0, 1, std::nullopt, {}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = cs_key;
  g.positions_path = path;
  m.geometry = g;
  m.populate = [&sub, path, pos, mid, n](Store&, const std::string&) { sub.write_array(path, view(pos, {n, 3}), {}, std::nullopt, "manifold", mid); };
  m.cols.json("producer_json", producer);
  sub.manifold(std::move(m));
  nxr::bst::LabellingOptions lo;
  lo.manifold_id = mid;
  lo.dictionary = opt.name + "_names";
  lo.scope = "subject";
  lo.description = "the 10-05 names, one level per point; the old 10-20 names (T3, T4, T5, T6) are aliases listed in the montage's producer";
  std::vector<std::int64_t> codes(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) codes[static_cast<std::size_t>(i)] = i;
  nxr::bst::labelling(sub, path + "_names", codes, P.labels, lo);
  nxr::bst::LabellingOptions so;
  so.manifold_id = mid;
  so.dictionary = opt.name + "_system";
  so.scope = "subject";
  so.description = "the narrowest standard system each point belongs to: 10-20, then 10-10 (the 10-10 system is both), then 10-05 (the whole montage)";
  nxr::bst::labelling(sub, path + "_system", sys, {"10-20", "10-10", "10-05"}, so);
  // the origin: derived from the scalp's and the landmarks' records
  OriginRecord o;
  const auto os = origin_of(sub, "manifold", scalp.id), ol = origin_of(sub, "manifold", text(*lm_row, "id"));
  o.source = "computed by nxr-neuro (the 10-05 rule) on the scalp '" + opt.scalp + "' and the landmarks '" + opt.landmarks + "' of " + subject;
  if (os && ol) {
    o.licence = os->licence == ol->licence ? os->licence : os->licence + "; " + ol->licence;
    o.attribution = os->attribution == ol->attribution ? os->attribution : os->attribution + "; " + ol->attribution;
    o.redistributable = os->redistributable && ol->redistributable;
  } else {
    o.licence = "unknown";
    o.attribution = "unknown";
  }
  set_origin(w, "manifold", mid, o);
  sub.complete();
  Json info = Json{{"name", opt.name}, {"points", n}, {"systems", counts}, {"inion", inion_info}, {"landmark_shift", shift}, {"cz_iterations", P.iterations}, {"cz_residual", P.cz_residual}, {"cs_key", cs_key}, {"warnings", warnings}, {"origin", Json{{"licence", o.licence}, {"redistributable", o.redistributable}}}};
  return MontageCreateResult{info};
}

}  // namespace nxr::pipelines
