#include "nxr/pipelines/registration.h"

#include <algorithm>
#include <set>

#include "nxr/pipelines/input.h"
#include "nxr/sphere_registration.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;
namespace reg = nxr::registration;

namespace {

[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }

// The released template `ref` names, with its hash: by name, edition and version on the search path; one hash only, else refused naming each.
TemplateRef with_hash(TemplateRef ref, const TemplateSearch& search) {
  std::set<std::string> hashes;
  for (const auto& l : list_templates(search))
    if (l.name == ref.name && l.version == ref.version && l.edition == ref.edition && l.kind == "template") hashes.insert(l.content_hash);
  if (hashes.empty()) {
    (void)resolve_template(ref, search);   // refuses, listing every location searched
    invalid(ref.name + "@" + ref.version + ": no such template found");
  }
  if (hashes.size() > 1) {
    std::string h;
    for (const auto& x : hashes) h += (h.empty() ? "" : ", ") + x;
    invalid(ref.name + "@" + ref.version + ": the search path holds different packages of this name and version (" + h + "); name one folder with --templates");
  }
  ref.content_hash = *hashes.begin();
  return ref;
}


std::string join_unique(const std::vector<std::string>& parts) {
  std::string out;
  std::set<std::string> seen;
  for (const auto& p : parts) if (!p.empty() && seen.insert(p).second) out += (out.empty() ? "" : "; ") + p;
  return out;
}

reg::Csr<double> operator_between(const SurfaceInput& a, const std::vector<reg::Piece>& pa, const SurfaceInput& b, const std::vector<reg::Piece>& pb, const reg::Options& o, reg::Report& rep) {
  try {
    return reg::resample_operator<double>(a.vertices, a.faces, pa, b.vertices, pb, o, &rep);
  } catch (const std::invalid_argument& e) {
    invalid(a.name + " to " + b.name + ": " + e.what());
  }
}

}  // namespace

RegistrationCreateResult registration_create(const RegistrationCreateOptions& opt, const std::function<void(const Json&)>& progress) {
  if (opt.from.empty() || opt.to.empty()) invalid("registration create: name both templates (--from NAME@VERSION --to NAME@VERSION)");
  if (opt.from_sphere.empty() || opt.to_sphere.empty()) invalid("registration create: name the registered sphere of each template (--from-sphere, --to-sphere): a registration is made through them");
  if (opt.name.empty()) invalid("registration create: name the registration (--name)");
  reg::Options ro;
  try { ro.method = reg::parse_method(opt.method); } catch (const std::invalid_argument& e) { invalid(e.what()); }
  ro.neighbours = opt.neighbours;
  if (ro.neighbours < 1) invalid("registration create: --neighbours is at least 1");

  TemplateSearch search;
  search.command_paths = opt.templates;
  const TemplateRef from = with_hash(parse_template_ref(opt.from), search), to = with_hash(parse_template_ref(opt.to), search);
  if (from.name == to.name && from.version == to.version && from.edition == to.edition) invalid("a registration links two different templates");
  const ResolvedTemplate rf = resolve_template(from, search), rt = resolve_template(to, search);
  const SubjectPackage pf = SubjectPackage::open(rf.folder), pt = SubjectPackage::open(rt.folder);
  const SurfaceInput sf = read_surface(pf, opt.from_sphere), st = read_surface(pt, opt.to_sphere);
  if (progress) progress(Json{{"stage", "spheres"}, {"from", opt.from_sphere}, {"from_vertices", sf.n_vertices()}, {"to", opt.to_sphere}, {"to_vertices", st.n_vertices()}});

  std::vector<reg::Piece> pcf, pct;
  try {
    pcf = reg::contiguous_pieces(sf.faces, sf.n_vertices());
    pct = reg::contiguous_pieces(st.faces, st.n_vertices());
  } catch (const std::invalid_argument& e) {
    invalid(std::string("the registered spheres' pieces: ") + e.what());
  }
  if (pcf.size() != pct.size())
    invalid(from.name + "." + opt.from_sphere + " has " + std::to_string(pcf.size()) + " pieces and " + to.name + "." + opt.to_sphere + " has " + std::to_string(pct.size()) + ": pieces (hemispheres) are matched one to one");

  // the origin of every item of the package: derived from the two spheres' own records unless the caller gives one
  const auto of = origin_of(pf.db(), "manifold", sf.id), ot = origin_of(pt.db(), "manifold", st.id);
  OriginRecord origin;
  origin.source = opt.source.value_or("computed by nxr-neuro from the registered spheres of " + from.name + "@" + from.version + " and " + to.name + "@" + to.version +
                                      (of ? "; sources: " + of->source : "") + (ot ? "; " + ot->source : ""));
  origin.licence = opt.licence.value_or(join_unique({of ? of->licence : "", ot ? ot->licence : ""}));
  origin.attribution = opt.attribution.value_or(join_unique({of ? of->attribution : "", ot ? ot->attribution : ""}));
  origin.redistributable = opt.redistributable.value_or(of && ot && of->redistributable && ot->redistributable);
  if (origin.licence.empty()) invalid("registration create: the licence of the registration is neither derived (the spheres carry no origin record) nor given (--licence)");

  RegistrationSpec spec;
  spec.name = opt.name;
  spec.from = from;
  spec.to = to;
  spec.from_manifold = opt.from_sphere;
  spec.to_manifold = opt.to_sphere;
  spec.search = search;
  spec.method = std::string(ro.method == reg::Method::Shepard ? "modified Shepard weights (Franke and Nielson 1980) over the " + std::to_string(ro.neighbours) + " nearest vertices on the registered spheres"
                            : ro.method == reg::Method::Nearest ? "nearest vertex on the registered spheres"
                                                                : "barycentric coordinates in the source triangle on the registered spheres");
  spec.warnings = {"the correspondence is the registered spheres', not anatomy's",
                   "every value carried is a mixture of source values: a field carried there and back is not the field",
                   "a source vertex may take part in no destination vertex's weights when the destination is the coarser surface"};
  spec.origin = origin;
  spec.package.home_dataset = opt.home_dataset;
  spec.package.built_by = opt.created_by;
  spec.package.subject.created_by = opt.created_by;

  Json operators = Json::array();
  auto write = [&](SubjectWriter& sub, const SurfaceInput& a, const std::vector<reg::Piece>& pa, const std::string& a_name, const std::string& a_id, const SurfaceInput& b,
                   const std::vector<reg::Piece>& pb, const std::string& b_name, const std::string& b_id) {
    reg::Report rep;
    const reg::Csr<double> w = operator_between(a, pa, b, pb, ro, rep);
    const std::string name = a_name + "_to_" + b_name;
    OperatorSpec o;
    o.name = name;
    o.path = name;
    o.kind = "resampling";
    o.layout = "csr";
    o.from_manifold_id = a_id;
    o.n_rows = w.rows;
    o.n_cols = w.cols;
    o.sparse = SparseArrays{{{"data", view(w.data, {w.nnz()})}, {"indices", view(w.indices, {w.nnz()})}, {"indptr", view(w.indptr, {w.rows + 1})}}};
    o.cols.set("to_manifold_id", b_id).set("nnz", w.nnz()).set("from_value_type", "scalar").set("to_value_type", "scalar");
    o.cols.set("description", "carries a scalar field on " + a_name + " to " + b_name + " through the registered spheres: a [" + std::to_string(w.rows) + " x " + std::to_string(w.cols) + "] matrix, one row per destination vertex");
    const Json producer = Json{{"method", reg::method_name(ro.method)}, {"neighbours", ro.method == reg::Method::Shepard ? Json(ro.neighbours) : Json(nullptr)},
                               {"pieces", pa.size()}, {"empty_columns", rep.empty_columns}, {"max_chord_unit_sphere", rep.max_chord},
                               {"fallback_to_nearest", rep.fallback_to_nearest}, {"warnings", spec.warnings}};
    o.cols.json("producer_json", producer);
    sub.operator_(o);
    operators.push_back(Json{{"name", name}, {"from", a_name}, {"to", b_name}, {"nnz", w.nnz()}, {"empty_columns", rep.empty_columns}, {"max_chord", rep.max_chord}});
    if (progress) progress(Json{{"stage", "operator"}, {"name", name}, {"nnz", w.nnz()}, {"empty_columns", rep.empty_columns}});
  };
  RegistrationCreateResult out;
  out.package = create_registration_package(opt.package, spec, [&](SubjectWriter& sub, const RegistrationIds& ids) {
    const std::string fname = from.name, tname = to.name;
    write(sub, sf, pcf, fname, ids.from_manifold_id, st, pct, tname, ids.to_manifold_id);
    if (opt.both_directions) write(sub, st, pct, tname, ids.to_manifold_id, sf, pcf, fname, ids.from_manifold_id);
  });
  out.info = Json{{"name", opt.name}, {"method", reg::method_name(ro.method)}, {"neighbours", ro.neighbours}, {"from", from.to_json()}, {"to", to.to_json()},
                  {"operators", operators}, {"warnings", spec.warnings}, {"redistributable", origin.redistributable}, {"licence", origin.licence}};
  return out;
}

}  // namespace nxr::pipelines
