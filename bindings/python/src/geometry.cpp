// nxr::geometry for Python: the surface as data. `geometry_surfaceArrays` makes it once (the intrinsic Delaunay triangulation is the expensive step) and
// returns the library's SurfaceArrays as a dict of NumPy arrays; every surface function takes that dict back and rebuilds SurfaceArrays from it. Nothing is
// kept between calls: no handle, no cache. A face: arguments and results converted, the library called, no maths.
//
//   surface = {"positions": float64 [nV, 3], "faces": int32 [nF, 3] (0-based),
//              "triangulation": {"options": {...}, "record": {...}, "mass", "inputEdgeLength", "edgeLength", "signpost", "vertexAngleSum",
//                                "halfedgeVectorInVertex": float64 [n],
//                                "halfedgeNext", "halfedgeVertex", "halfedgeFace", "vertexHalfedge", "faceHalfedge": int64 [n] (geometry-central's ids),
//                                "boundaryLoops": int}}
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/mesh.h"
#include "surface.h"

namespace nxrpy {

namespace {

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

nb::dict mesh_topology(const nb::dict& surface) {
  const geom::SurfaceArrays s = surface_in(surface);
  geom::MeshTopology t;
  {
    nb::gil_scoped_release release;
    t = geom::meshTopology(s);
  }
  auto longs = [](std::vector<long> v) { std::vector<int64_t> o(v.begin(), v.end()); return vec_out(std::move(o)); };
  auto flags = [](const std::vector<char>& v) { std::vector<uint8_t> o(v.begin(), v.end()); return vec_out(std::move(o)); };
  nb::dict out;
  out["nV"] = t.nV; out["nE"] = t.nE; out["nF"] = t.nF; out["nH"] = t.nH; out["nC"] = t.nC;
  out["vertexHalfedge"] = longs(t.vertexHalfedge); out["edgeHalfedge"] = longs(t.edgeHalfedge);
  out["faceHalfedge"] = longs(t.faceHalfedge); out["cornerHalfedge"] = longs(t.cornerHalfedge);
  out["heTwin"] = longs(t.heTwin); out["heNext"] = longs(t.heNext); out["heVertex"] = longs(t.heVertex);
  out["heEdge"] = longs(t.heEdge); out["heFace"] = longs(t.heFace); out["heCorner"] = longs(t.heCorner);
  out["heOrientation"] = flags(t.heOrientation); out["heIsInterior"] = flags(t.heIsInterior);
  return out;
}

int euler_characteristic(const nb::dict& surface) { return geom::eulerCharacteristic(surface_in(surface)); }

template <class Frames>
nb::dict frames_out(const Frames& f) {
  nb::dict out;
  out["e1"] = f.e1; out["e2"] = f.e2; out["normals"] = f.normals;
  return out;
}

nb::dict vertex_frames(const nb::dict& surface) {
  const geom::SurfaceArrays s = surface_in(surface);
  geom::VertexFrames f;
  {
    nb::gil_scoped_release release;
    f = geom::vertexFrames(s);
  }
  return frames_out(f);
}

nb::dict face_frames(const nb::dict& surface) {
  const geom::SurfaceArrays s = surface_in(surface);
  geom::FaceFrames f;
  {
    nb::gil_scoped_release release;
    f = geom::frames(s);
  }
  return frames_out(f);
}

geom::HeatMesh heat_mesh(const std::string& m) {
  if (m == "intrinsic") return geom::HeatMesh::Intrinsic;
  if (m == "input") return geom::HeatMesh::Input;
  throw std::invalid_argument("mesh '" + m + "' is not one of intrinsic, input");
}

Eigen::MatrixXd heat(const nb::dict& surface, const std::vector<std::vector<int>>& source_sets, double t_coef, const std::string& mesh) {
  const geom::SurfaceArrays s = surface_in(surface);
  const geom::HeatMesh hm = heat_mesh(mesh);
  nb::gil_scoped_release release;
  return geom::heat(s, source_sets, t_coef, hm);
}

}  // namespace

void bind_geometry(nb::module_& m) {
  using nb::arg;
  m.def("geometry_surfaceArrays", &surface_arrays, arg("vertices"), arg("faces"), arg("intrinsic_delaunay"), arg("mollify"), arg("flip_eps"), arg("surface"));
  m.def("geometry_complex", &complex_of, arg("surface"));
  m.def("geometry_meshTopology", &mesh_topology, arg("surface"));
  m.def("geometry_eulerCharacteristic", &euler_characteristic, arg("surface"));
  m.def("geometry_vertexFrames", &vertex_frames, arg("surface"));
  m.def("geometry_frames", &face_frames, arg("surface"));
  m.def("geometry_heat", &heat, arg("surface"), arg("source_sets"), arg("t_coef"), arg("mesh"));
}

}  // namespace nxrpy
