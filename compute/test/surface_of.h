// Tests: the surface arrays of a mesh with the former construction's arguments (the intrinsic Delaunay option and its two named options).
#pragma once
#include "nxr/geometry/surface_arrays.h"

inline nxr::geometry::SurfaceArrays surfaceOf(const double* V, int nV, const int32_t* F, int nF, bool intrinsicDelaunay = true,
                                              const nxr::geometry::DelaunayOptions& d = {}) {
    return nxr::geometry::surfaceArrays(V, nV, F, nF, nxr::geometry::intrinsicOptions(intrinsicDelaunay, d));
}

#include <map>
#include <memory>

// Tests only: geometry-central's objects of a surface, kept alive for the test's lifetime (a test reads `embedded()`, `inputMesh()`, `geometry()` of
// the same surface many times). Keyed by the arrays' address and checked against their content, so a surface made later at the same address is rebuilt.
inline nxr::geometry::IntrinsicSurface& gcOf(const nxr::geometry::SurfaceArrays& s) {
    struct Entry { nxr::geometry::RowMat3 P; nxr::geometry::FaceMat F; std::unique_ptr<nxr::geometry::IntrinsicSurface> g; };
    static std::map<const nxr::geometry::SurfaceArrays*, Entry> held;
    Entry& e = held[&s];
    if (!e.g || e.P.rows() != s.positions.rows() || e.F.rows() != s.faces.rows() || e.P != s.positions || e.F != s.faces) {
        e.g.reset(new nxr::geometry::IntrinsicSurface(s));
        e.P = s.positions; e.F = s.faces;
    }
    return *e.g;
}
