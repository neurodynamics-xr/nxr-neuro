// nxr::select levels for Python (spec docs/specs/2026-11-26-levels-of-one-manifold.md): a level of one manifold as a navigable view (`Level`, `ProductLevel`), made from the
// fine complex the existing faces give (operators.assembleDECOperators) and a grid's codes (select.subjectTree). A face: arguments and results converted, the library
// called, no maths.
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <tuple>

#include "arrays.h"
#include "nxr/select/dyadic.h"
#include "nxr/select/levels.h"

namespace nxrpy {

namespace sel = nxr::select;
using Sparse = Eigen::SparseMatrix<double>;
using Ints = Eigen::Matrix<std::int32_t, Eigen::Dynamic, 1>;
using IntRows = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

sel::FineComplex fine_in(const Sparse& d0, const std::optional<Sparse>& d1, const Eigen::VectorXd& star0, const Eigen::VectorXd& star1) {
  sel::FineComplex f;
  f.d0 = d0; if (d1) f.d1 = *d1; f.star0 = star0; f.star1 = star1;
  return f;
}
std::vector<std::int32_t> ints_in(const Ints& a) { return {a.data(), a.data() + a.size()}; }
nb::object address_out(const sel::LevelAddress& a) {
  if (a.level == sel::kNoLevel) return nb::none();
  return nb::make_tuple(a.level, a.code);
}
const char* ladder_name(sel::Level::Ladder l) { return l == sel::Level::Ladder::vertices ? "vertices" : l == sel::Level::Ladder::dyadic ? "dyadic" : "time"; }

sel::Level vertex_level(const Sparse& d0, const std::optional<Sparse>& d1, const Eigen::VectorXd& star0, const Eigen::VectorXd& star1, const std::optional<Ints>& codes, int depth,
                        std::int64_t slots) {
  const sel::FineComplex f = fine_in(d0, d1, star0, star1);
  nb::gil_scoped_release release;
  return sel::vertexLevel(f, codes ? ints_in(*codes) : std::vector<std::int32_t>{}, depth, slots);
}
sel::Level dyadic_level(const Sparse& d0, const std::optional<Sparse>& d1, const Eigen::VectorXd& star0, const Eigen::VectorXd& star1, const Ints& codes, int depth,
                        std::int64_t slots, int level) {
  const sel::FineComplex f = fine_in(d0, d1, star0, star1);
  const auto c = ints_in(codes);
  nb::gil_scoped_release release;
  return sel::dyadicLevel(f, c, depth, slots, level);
}
sel::Level level_complex(const Sparse& d0, const std::optional<Sparse>& d1, const Eigen::VectorXd& star0, const Eigen::VectorXd& star1, const Ints& cell_of, std::int64_t cells) {
  const sel::FineComplex f = fine_in(d0, d1, star0, star1);
  const auto c = ints_in(cell_of);
  nb::gil_scoped_release release;
  return sel::levelComplex(f, c, cells);
}
sel::ProductLevel product_level(const sel::Level& space, const sel::Level& time, const std::optional<nb::dict>& coupling) {
  sel::Coupling c;
  if (coupling) {
    const std::string kind = coupling->contains("kind") ? nb::cast<std::string>((*coupling)["kind"]) : std::string("separable");
    if (kind == "dispersion") c.kind = sel::Coupling::Kind::dispersion;
    else if (kind != "separable") throw std::invalid_argument("coupling kind '" + kind + "' is not one of separable, dispersion");
    if (coupling->contains("speed")) c.speed = nb::cast<double>((*coupling)["speed"]);
    if (coupling->contains("sense")) c.sense = nb::cast<int>((*coupling)["sense"]);
  }
  return sel::productLevel(space, time, c);
}
nb::dict subject_tree(const RowMajor& vertices, const IntRows& faces, int depth) {
  sel::Tree t;
  {
    nb::gil_scoped_release release;
    t = sel::subjectTree(vertices.data(), static_cast<int>(vertices.rows()), faces.data(), static_cast<int>(faces.rows()), depth);
  }
  nb::dict out;
  out["codes"] = Ints(Eigen::Map<const Ints>(t.codes.data(), static_cast<Eigen::Index>(t.codes.size())));
  out["depth"] = t.depth; out["finest"] = t.finest; out["slots"] = t.slots;
  out["diameter"] = Eigen::VectorXd(Eigen::Map<const Eigen::VectorXd>(t.diameter.data(), static_cast<Eigen::Index>(t.diameter.size())));
  return out;
}

}  // namespace

void bind_levels(nb::module_& m) {
  using nb::arg;
  m.attr("select_kVertexLevel") = sel::kVertexLevel;
  nb::class_<sel::Level>(m, "Level")
      .def_prop_ro("ladder", [](const sel::Level& L) { return ladder_name(L.ladder); })
      .def_ro("level", &sel::Level::level)
      .def_ro("depth", &sel::Level::depth)
      .def_ro("slots", &sel::Level::slots)
      .def_ro("samples", &sel::Level::samples)
      .def_ro("rate", &sel::Level::rate)
      .def_ro("cells", &sel::Level::cells)
      .def_ro("cellOf", &sel::Level::cellOf)
      .def_ro("edges", &sel::Level::edges)
      .def_ro("d0", &sel::Level::d0)
      .def_ro("star0", &sel::Level::star0)
      .def_ro("star1", &sel::Level::star1)
      .def_ro("crossing", &sel::Level::crossing)
      .def_ro("faces", &sel::Level::faces)
      .def_ro("junctions", &sel::Level::junctions)
      .def_ro("d1", &sel::Level::d1)
      .def_ro("valid", &sel::Level::valid)
      .def_ro("why", &sel::Level::why)
      .def_ro("negativeConductances", &sel::Level::negativeConductances)
      .def_ro("euler", &sel::Level::euler)
      .def_ro("fineEuler", &sel::Level::fineEuler)
      .def("neighbours", [](const sel::Level& L, std::int64_t c) { return sel::neighbours(L, c); }, arg("cell"))
      .def("incidentEdges", [](const sel::Level& L, std::int64_t c) { return sel::incidentEdges(L, c); }, arg("cell"))
      .def("outgoing", [](const sel::Level& L, std::int64_t c) { return sel::outgoing(L, c); }, arg("cell"))
      .def("twin", [](const sel::Level& L, std::int64_t h) { return sel::twin(L, h); }, arg("halfedge"))
      .def("next", [](const sel::Level& L, std::int64_t h) { return sel::next(L, h); }, arg("halfedge"))
      .def("tail", [](const sel::Level& L, std::int64_t h) { return sel::tail(L, h).code; }, arg("halfedge"))
      .def("tip", [](const sel::Level& L, std::int64_t h) { return sel::tip(L, h).code; }, arg("halfedge"))
      .def("parent", [](const sel::Level& L, std::int64_t c) { return address_out(sel::parent(L, c)); }, arg("cell"))
      .def("children", [](const sel::Level& L, std::int64_t c) { const auto ch = sel::children(L, c); return nb::make_tuple(ch.level, ch.codes); }, arg("cell"))
      .def("area", [](const sel::Level& L, std::int64_t c) { return sel::area(L, c); }, arg("cell"))
      .def("cell", [](const sel::Level& L, std::int64_t c) { if (c < 0 || c >= L.cells) throw std::invalid_argument("cell outside the level"); return nb::make_tuple(L.level, c); }, arg("code"))
      .def("crossingEdge", [](const sel::Level& L, std::int64_t e) { return sel::crossingEdge(L, e); }, arg("edge"))
      .def("star", [](const sel::Level& L, int rank) -> Eigen::VectorXd { if (rank == 0) return L.star0; if (rank == 1) return L.star1; throw std::invalid_argument("star: rank 0 or 1 (a level's faces carry no ⋆2)"); }, arg("rank"))
      .def("laplacian", [](const sel::Level& L) { return sel::laplacian(L); })
      .def("cellMap", [](const sel::Level& L, const sel::Level& to) { return sel::cellMap(L, to); }, arg("to"))
      .def("reductionOperator", [](const sel::Level& L, const sel::Level& to, const std::string& rule) { return sel::reductionOperator(L, to, rule); }, arg("to"), arg("rule"))
      .def("expansionOperator", [](const sel::Level& L, const sel::Level& fine, const std::string& rule) { return sel::expansionOperator(L, fine, rule); }, arg("fine"), arg("rule"))
      .def("reduce", [](const sel::Level& L, const sel::Level& to, const std::string& rule, const Eigen::MatrixXd& values) { return sel::reduce(L, to, rule, values); }, arg("to"), arg("rule"), arg("values"));
  nb::class_<sel::ProductLevel>(m, "ProductLevel")
      .def_ro("space", &sel::ProductLevel::space)
      .def_ro("time", &sel::ProductLevel::time)
      .def_prop_ro("cells", [](const sel::ProductLevel& p) { return p.cells(); })
      .def_prop_ro("coupling", [](const sel::ProductLevel& p) { return p.coupling.kind == sel::Coupling::Kind::separable ? "separable" : "dispersion"; })
      .def("jointCode", [](const sel::ProductLevel& p, std::int64_t s, std::int64_t t) { return sel::jointCode(p, s, t); }, arg("space"), arg("time"))
      .def("jointAddress", [](const sel::ProductLevel& p, std::int64_t c) { const auto a = sel::jointAddress(p, c); return nb::make_tuple(nb::make_tuple(a.space.level, a.space.code), nb::make_tuple(a.time.level, a.time.code)); }, arg("code"))
      .def("neighbours", [](const sel::ProductLevel& p, std::int64_t c) { return sel::productNeighbours(p, c); }, arg("code"))
      .def("operators", [](const sel::ProductLevel& p) { auto o = sel::productOperators(p); nb::dict d; d["d0"] = std::move(o.d0); d["star0"] = std::move(o.star0); d["star1"] = std::move(o.star1); return d; });
  m.def("select_levelComplex", &level_complex, arg("d0"), arg("d1").none(), arg("star0"), arg("star1"), arg("cell_of"), arg("cells"));
  m.def("select_vertexLevel", &vertex_level, arg("d0"), arg("d1").none(), arg("star0"), arg("star1"), arg("codes").none(), arg("depth"), arg("slots"));
  m.def("select_dyadicLevel", &dyadic_level, arg("d0"), arg("d1").none(), arg("star0"), arg("star1"), arg("codes"), arg("depth"), arg("slots"), arg("level"));
  m.def("select_lineLevel", &sel::lineLevel, arg("samples"), arg("rate"), arg("k"));
  m.def("select_productLevel", &product_level, arg("space"), arg("time"), arg("coupling").none());
  m.def("select_subjectTree", &subject_tree, arg("vertices"), arg("faces"), arg("depth"));
}

}  // namespace nxrpy
