// Rigid-body and landmark geometry shared by the readers and the importers: 4 x 4 row-major transforms (translation in the last column), the
// head frame the landmarks define (Neuromag's and Brainstorm's), and the least-squares rigid fit of matched points.
#pragma once
#include <array>
#include <vector>

namespace nxr::formats::geom {

using Matrix4 = std::array<double, 16>;
using Point = std::array<double, 3>;

Point apply(const Matrix4& m, const Point& p);                 // rotation, scale and translation
Point rotate(const Matrix4& m, const Point& v);                // rotation and scale only: for directions
Matrix4 multiply(const Matrix4& a, const Matrix4& b);          // a * b: b first, then a
Matrix4 invert(const Matrix4& m);                              // throws a format error when singular
// Neuromag's head frame from its landmarks: x toward the right ear, y toward the nasion through the nasion's projection on the ear line, z up
Matrix4 neuromag_head_from_landmarks(const Point& nas, const Point& lpa, const Point& rpa);
// Brainstorm's subject coordinate system (SCS) from its landmarks: origin midway between the ears, x toward the nasion, y toward the left ear
// (orthogonalised against x), z up
Matrix4 scs_from_landmarks(const Point& nas, const Point& lpa, const Point& rpa);
// the rigid transform taking `from` onto `to` in the least-squares sense (Horn); at least three matched points
Matrix4 fit_rigid(const std::vector<Point>& from, const std::vector<Point>& to);

}  // namespace nxr::formats::geom
