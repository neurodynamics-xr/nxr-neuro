// The surface across the boundary: the library's SurfaceArrays <-> a dict of NumPy arrays (made by geometry.surfaceArrays, passed back to every surface
// function; nothing is kept between calls). Also the construction's record as a dict. Shared by the geometry, operator and form bindings.
#pragma once

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

inline const char* mass_name(geom::MassKind k) {
  switch (k) {
    case geom::MassKind::InputMixedVoronoi: return "input-mixed-voronoi";
    case geom::MassKind::Barycentric: return "barycentric";
    case geom::MassKind::Galerkin: return "galerkin";
  }
  return "input-mixed-voronoi";
}

inline geom::MassKind mass_kind(const std::string& s) {
  if (s == "input-mixed-voronoi") return geom::MassKind::InputMixedVoronoi;
  if (s == "barycentric") return geom::MassKind::Barycentric;
  if (s == "galerkin") return geom::MassKind::Galerkin;
  throw std::invalid_argument("mass '" + s + "' is not one of input-mixed-voronoi, barycentric, galerkin");
}

template <class T>
inline nb::object vec_out(std::vector<T> v) {
  const size_t n = v.size();
  return own_array<T>(std::move(v), {n});
}

template <class T>
inline nb::object rows_out(std::vector<T> v, size_t cols) {
  const size_t n = v.size() / cols;
  return own_array<T>(std::move(v), {n, cols});
}

inline nb::object field(const nb::dict& d, const char* name) {
  if (!d.contains(name)) throw std::invalid_argument(std::string("the surface has no field '") + name + "': make it with geometry.surfaceArrays(vertices, faces)");
  return d[name];
}

template <class T>
inline std::vector<T> vec_in(const nb::dict& d, const char* name) {
  const auto a = nb::cast<Array1<T>>(field(d, name));
  return std::vector<T>(a.data(), a.data() + a.shape(0));
}

inline nb::dict record_out(const geom::IntrinsicRecord& r) {
  nb::dict record;
  record["mesh"] = r.mesh;
  record["mollify"] = r.mollify;
  record["epsilon"] = r.epsilon;
  record["flipEps"] = r.flipEps;
  record["flips"] = r.flips;
  record["delaunay"] = r.delaunay;
  record["mass"] = r.mass;
  record["boundaryVertices"] = r.boundaryVertices;
  record["geometryCentral"] = r.geometryCentral;
  record["fingerprint"] = r.fingerprint;
  record["gauge"] = r.gauge;
  record["gaugeId"] = r.gaugeId;
  return record;
}

inline nb::dict surface_out(geom::SurfaceArrays&& s) {
  const geom::IntrinsicTriangulation& t = s.triangulation;
  nb::dict options, tri, out;
  options["intrinsic"] = t.options.intrinsic;
  options["mollify"] = t.options.mollify;
  options["flipEps"] = t.options.flipEps;
  options["mass"] = mass_name(t.options.mass);
  options["surface"] = t.options.surface;
  tri["options"] = options;
  tri["record"] = record_out(t.record);
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
inline geom::SurfaceArrays surface_in(const nb::dict& d) {
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

}  // namespace nxrpy
