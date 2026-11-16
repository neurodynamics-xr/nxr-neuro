#pragma once
// nxr::transport — THE GEOMETRY TRANSPORT: carrying a vector along the cortex as a current NORMAL to it would be carried (a rotation R_uv per edge, SO(3)).
// It ENCODES THE PRIOR that currents are normal to the cortex: a named option for that model, NOT the
// default and NOT the direction route. Direction analysis makes no orientation assumption: the joint data-block magnetic Laplacian
// (magnetic.h, blockMagneticLaplacian) takes the relation between two currents from the data. With a measured lag θ, e^{iθ} R_uv is the transport of
// magnetic.h's transportLaplacian (the normal-current model).
//
//   THE TRANSPORT (a named option; the rotations are world-coordinate, R_uv J_v is v's vector carried to u):
//     levi_civita   the minimal rotation taking v's normal to u's (about n_v × n_u): the surface's own parallel transport along the edge, from the normals
//                   alone (no gauge); tangent vectors keep their angle to the edge. It has holonomy where the surface is curved (Gaussian curvature).
//     frames        F_u F_vᵀ with F = [e1 | e2 | n] the vertices' frames: compute's flat full-frame transport (manifold::differential::frameTransport,
//                   P = F_uᵀ F_v in frame coordinates) in world coordinates; path-independent (no holonomy); its tangential part depends on how each
//                   vertex's e1 was chosen (geometry::vertexFrames takes it from a reference edge).
//   compare(T, R, Tuu, Tvv)   per edge the transported cross-spectrum tr(T_uv R_uvᵀ) = Σ_a ⟨J_u^a conj((R_uv J_v)^a)⟩ (T_uv = ⟨J_u J_vᴴ⟩, a 3 x 3 block:
//                   algebra::blockCongruence), its lag θ = arg and its strength |·| / √(tr T_uu tr T_vv) (a coherency, 0 … 1).
//
// AN ASSUMPTION, NOT A DEFAULT: the geometry transport ENCODES the prior that currents are normal to the cortex (it carries a vector as
// a normal current would be carried). Nothing requires an estimate to obey it; direction analysis makes no orientation assumption (the magnetic
// Laplacian spec, "direction analysis makes no orientation assumption": the joint data-block route). On MEG a free minimum-norm estimate had ~1/3 of its
// power along the normal, and transport rotated its nearly equal world vectors apart. And `compare` reduces a 3 x 3 block to its
// trace (like components only): on MEG most of the lag content sat in the CROSS-COMPONENT terms, which the trace discards. Read lag content from the
// full block (‖Im T‖ / ‖T‖) before trusting a trace.
//
// PURE MATHS: arrays in, arrays out.
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::transport {

using Edges = Eigen::Matrix<int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
enum class Transport { LeviCivita, Frames };
/** the option by name ("levi_civita", "frames"); std::invalid_argument naming the valid ones otherwise */
Transport transportOf(const std::string& name);

/** the minimal rotations R_uv (n_v → n_u) for edges (u, v); `normals` [n x 3] unit. Antiparallel normals: a half-turn about an axis perpendicular to both. */
std::vector<Eigen::Matrix3d> leviCivita(const Eigen::MatrixXd& normals, const Edges& edges);
/** R_uv = F_u F_vᵀ, F_v = [e1_v | e2_v | n_v] (each [n x 3], orthonormal a vertex) */
std::vector<Eigen::Matrix3d> frames(const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2, const Eigen::MatrixXd& normals, const Edges& edges);

struct Compared {
  std::vector<std::complex<double>> S;   // tr(T_uv R_uvᵀ) per edge
  Eigen::VectorXd lag;                   // arg S, rad
  Eigen::VectorXd strength;              // |S| / √(tr T_uu tr T_vv)
};
/** T [E] the edges' 3 x 3 blocks ⟨J_u J_vᴴ⟩, R [E] their rotations, Tuu / Tvv [E] the ends' powers (tr of their own blocks) */
Compared compare(const std::vector<Eigen::MatrixXcd>& T, const std::vector<Eigen::Matrix3d>& R, const Eigen::VectorXd& Tuu, const Eigen::VectorXd& Tvv);

}  // namespace nxr::transport
