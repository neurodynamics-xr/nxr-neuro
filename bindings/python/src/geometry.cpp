// nxr::geometry for Python: the surface as data. `geometry_surfaceArrays` makes it once (the intrinsic Delaunay triangulation is the expensive step) and
// returns the library's SurfaceArrays as a dict of NumPy arrays; every surface function takes that dict back and rebuilds SurfaceArrays from it. Nothing is
// kept between calls: no handle, no cache. A face: arguments and results converted, the library called, no maths.
//
//   surface = {"positions": float64 [nV, 3], "faces": int32 [nF, 3] (0-based),
//              "triangulation": {"options": {...}, "record": {...}, "mass", "inputEdgeLength", "edgeLength", "signpost", "vertexAngleSum",
//                                "halfedgeVectorInVertex": float64 [n],
//                                "halfedgeNext", "halfedgeVertex", "halfedgeFace", "vertexHalfedge", "faceHalfedge": int64 [n] (geometry-central's ids),
//                                "boundaryLoops": int}}
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "arrays.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/surface_arrays.h"

namespace nxrpy {

namespace geom = nxr::geometry;

namespace {

const char* mass_name(geom::MassKind k) {
  switch (k) {
    case geom::MassKind::InputMixedVoronoi: return "input-mixed-voronoi";
    case geom::MassKind::Barycentric: return "barycentric";
    case geom::MassKind::Galerkin: return "galerkin";
  }
  return "input-mixed-voronoi";
}

geom::MassKind mass_kind(const std::string& s) {
  if (s == "input-mixed-voronoi") return geom::MassKind::InputMixedVoronoi;
  if (s == "barycentric") return geom::MassKind::Barycentric;
  if (s == "galerkin") return geom::MassKind::Galerkin;
  throw std::invalid_argument("mass '" + s + "' is not one of input-mixed-voronoi, barycentric, galerkin");
}

template <class T>
nb::object vec_out(std::vector<T> v) {
  const size_t n = v.size();
  return own_array<T>(std::move(v), {n});
}

template <class T>
nb::object rows_out(std::vector<T> v, size_t cols) {
  const size_t n = v.size() / cols;
  return own_array<T>(std::move(v), {n, cols});
}

nb::object field(const nb::dict& d, const char* name) {
  if (!d.contains(name)) throw std::invalid_argument(std::string("the surface has no field '") + name + "': make it with geometry.surfaceArrays(vertices, faces)");
  return d[name];
}

template <class T>
std::vector<T> vec_in(const nb::dict& d, const char* name) {
  const auto a = nb::cast<Array1<T>>(field(d, name));
  return std::vector<T>(a.data(), a.data() + a.shape(0));
}

nb::dict surface_out(geom::SurfaceArrays&& s) {
  const geom::IntrinsicTriangulation& t = s.triangulation;
  nb::dict options, record, tri, out;
  options["intrinsic"] = t.options.intrinsic;
  options["mollify"] = t.options.mollify;
  options["flipEps"] = t.options.flipEps;
  options["mass"] = mass_name(t.options.mass);
  options["surface"] = t.options.surface;
  record["mesh"] = t.record.mesh;
  record["mollify"] = t.record.mollify;
  record["epsilon"] = t.record.epsilon;
  record["flipEps"] = t.record.flipEps;
  record["flips"] = t.record.flips;
  record["delaunay"] = t.record.delaunay;
  record["mass"] = t.record.mass;
  record["boundaryVertices"] = t.record.boundaryVertices;
  record["geometryCentral"] = t.record.geometryCentral;
  record["fingerprint"] = t.record.fingerprint;
  record["gauge"] = t.record.gauge;
  record["gaugeId"] = t.record.gaugeId;
  tri["options"] = options;
  tri["record"] = record;
  tri["mass"] = vec_out(t.mass);
  tri["inputEdgeLength"] = vec_out(t.inputEdgeLength);
  tri["halfedgeNext"] = vec_out(t.halfedgeNext);
  tri["halfedgeVertex"] = vec_out(t.halfedgeVertex);
  tri["halfedgeFace"] = vec_out(t.halfedgeFace);
  tri["vertexHalfedge"] = vec_out(t.vertexHalfedge);
  tri["faceHalfedge"] = vec_out(t.faceHalfedge);
  tri["boundaryLoops"] = t.boundaryLoops;
  tri["edgeLength"] = vec_out(t.edgeLength);
  tri["signpost"] = vec_out(t.signpost);
  tri["vertexAngleSum"] = vec_out(t.vertexAngleSum);
  tri["halfedgeVectorInVertex"] = vec_out(t.halfedgeVectorInVertex);
  out["positions"] = rows_out(std::vector<double>(s.positions.data(), s.positions.data() + s.positions.size()), 3);
  out["faces"] = rows_out(std::vector<int32_t>(s.faces.data(), s.faces.data() + s.faces.size()), 3);
  out["triangulation"] = tri;
  return out;
}

/** The surface dict back to the library's SurfaceArrays (the inverse of surface_out). */
geom::SurfaceArrays surface_in(const nb::dict& d) {
  geom::SurfaceArrays s;
  const auto p = nb::cast<Array2<double>>(field(d, "positions"));
  const auto f = nb::cast<Array2<int32_t>>(field(d, "faces"));
  if (p.shape(1) != 3 || f.shape(1) != 3) throw std::invalid_argument("the surface's positions and faces are [n, 3]");
  s.positions = Eigen::Map<const geom::RowMat3>(p.data(), static_cast<Eigen::Index>(p.shape(0)), 3);
  s.faces = Eigen::Map<const geom::FaceMat>(f.data(), static_cast<Eigen::Index>(f.shape(0)), 3);
  const nb::dict tri = nb::cast<nb::dict>(field(d, "triangulation"));
  const nb::dict options = nb::cast<nb::dict>(field(tri, "options"));
  const nb::dict record = nb::cast<nb::dict>(field(tri, "record"));
  geom::IntrinsicTriangulation& t = s.triangulation;
  t.options.intrinsic = nb::cast<bool>(field(options, "intrinsic"));
  t.options.mollify = nb::cast<double>(field(options, "mollify"));
  t.options.flipEps = nb::cast<double>(field(options, "flipEps"));
  t.options.mass = mass_kind(nb::cast<std::string>(field(options, "mass")));
  t.options.surface = nb::cast<std::string>(field(options, "surface"));
  t.record.mesh = nb::cast<std::string>(field(record, "mesh"));
  t.record.mollify = nb::cast<double>(field(record, "mollify"));
  t.record.epsilon = nb::cast<double>(field(record, "epsilon"));
  t.record.flipEps = nb::cast<double>(field(record, "flipEps"));
  t.record.flips = nb::cast<int>(field(record, "flips"));
  t.record.delaunay = nb::cast<bool>(field(record, "delaunay"));
  t.record.mass = nb::cast<std::string>(field(record, "mass"));
  t.record.boundaryVertices = nb::cast<int>(field(record, "boundaryVertices"));
  t.record.geometryCentral = nb::cast<std::string>(field(record, "geometryCentral"));
  t.record.fingerprint = nb::cast<std::string>(field(record, "fingerprint"));
  t.record.gauge = nb::cast<std::string>(field(record, "gauge"));
  t.record.gaugeId = nb::cast<std::string>(field(record, "gaugeId"));
  t.mass = vec_in<double>(tri, "mass");
  t.inputEdgeLength = vec_in<double>(tri, "inputEdgeLength");
  t.halfedgeNext = vec_in<int64_t>(tri, "halfedgeNext");
  t.halfedgeVertex = vec_in<int64_t>(tri, "halfedgeVertex");
  t.halfedgeFace = vec_in<int64_t>(tri, "halfedgeFace");
  t.vertexHalfedge = vec_in<int64_t>(tri, "vertexHalfedge");
  t.faceHalfedge = vec_in<int64_t>(tri, "faceHalfedge");
  t.boundaryLoops = nb::cast<int64_t>(field(tri, "boundaryLoops"));
  t.edgeLength = vec_in<double>(tri, "edgeLength");
  t.signpost = vec_in<double>(tri, "signpost");
  t.vertexAngleSum = vec_in<double>(tri, "vertexAngleSum");
  t.halfedgeVectorInVertex = vec_in<double>(tri, "halfedgeVectorInVertex");
  return s;
}

nb::dict surface_arrays(const Array2<double>& vertices, const Array2<int32_t>& faces, bool intrinsic_delaunay, double mollify, double flip_eps,
                        const std::string& surface) {
  if (vertices.shape(1) != 3 || faces.shape(1) != 3) throw std::invalid_argument("vertices and faces are [n, 3]");
  geom::DelaunayOptions delaunay;
  delaunay.mollify = mollify;
  delaunay.flipEps = flip_eps;
  geom::IntrinsicOptions options = geom::intrinsicOptions(intrinsic_delaunay, delaunay);
  options.surface = surface;
  geom::SurfaceArrays s;
  {
    nb::gil_scoped_release release;
    s = geom::surfaceArrays(vertices.data(), static_cast<int>(vertices.shape(0)), faces.data(), static_cast<int>(faces.shape(0)), options);
  }
  return surface_out(std::move(s));
}

nb::dict complex_of(const nb::dict& surface) {
  const geom::SurfaceArrays s = surface_in(surface);
  geom::IntrinsicSurface::Complex c;
  {
    nb::gil_scoped_release release;
    geom::IntrinsicSurface rebuilt(s);
    c = rebuilt.complex();
  }
  nb::dict out;
  out["faces"] = rows_out(std::move(c.faces), 3);
  out["edges"] = rows_out(std::move(c.edges), 2);
  out["faceEdges"] = rows_out(std::move(c.faceEdges), 3);
  out["length"] = vec_out(std::move(c.length));
  out["angle"] = rows_out(std::move(c.angle), 3);
  out["angleSum"] = vec_out(std::move(c.angleSum));
  out["reference"] = vec_out(std::move(c.reference));
  return out;
}

}  // namespace

void bind_geometry(nb::module_& m) {
  using nb::arg;
  m.def("geometry_surfaceArrays", &surface_arrays, arg("vertices"), arg("faces"), arg("intrinsic_delaunay"), arg("mollify"), arg("flip_eps"), arg("surface"));
  m.def("geometry_complex", &complex_of, arg("surface"));
}

}  // namespace nxrpy
