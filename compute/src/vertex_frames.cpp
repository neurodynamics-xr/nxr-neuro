#include "nxr/compute.h"

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <iostream>

namespace nxr::manifold::geometry {

using namespace geometrycentral;
using namespace geometrycentral::surface;

// See VertexFrames in compute.h for the gauge convention: the signposts' gauge of the operator mesh (the Studio's `tangentBasis`), so it matches the
// tangent space the connection Laplacian uses.
VertexFrames vertexFrames(Manifold& m) {
    const delaunay::Mesh& om = m.operatorMesh();
    const Eigen::MatrixXd& P = m.vertexPositions();
    const Eigen::MatrixXi& Fi = m.faces();
    const int nV = m.nV();
    Eigen::MatrixXd Nrm = Eigen::MatrixXd::Zero(nV, 3);
    for (int f = 0; f < Fi.rows(); ++f) {
        const Eigen::Vector3d a = P.row(Fi(f, 0)), b = P.row(Fi(f, 1)), c = P.row(Fi(f, 2));
        const Eigen::Vector3d fn = (b - a).cross(c - a);
        for (int k = 0; k < 3; ++k) Nrm.row(Fi(f, k)) += fn.transpose();
    }
    VertexFrames out;
    out.e1.resize(nV, 3); out.e2.resize(nV, 3); out.normals.resize(nV, 3);
    for (int v = 0; v < nV; ++v) {
        Eigen::Vector3d n = Nrm.row(v);
        double l = n.norm(); if (l == 0) l = 1;
        n /= l;
        const int r = om.ref[static_cast<size_t>(v)];
        Eigen::Vector3d d = Eigen::Vector3d(P.row(r)) - Eigen::Vector3d(P.row(v));
        d -= d.dot(n) * n;
        double k = d.norm(); if (k == 0) k = 1;
        const Eigen::Vector3d e1 = d / k;
        out.e1.row(v) = e1.transpose();
        out.e2.row(v) = n.cross(e1).transpose();
        out.normals.row(v) = n.transpose();
    }
    return out;
}

// geometry-central's vertexTangentBasis (basisX, basisY = n × basisX) plus its vertex normal.
VertexFrames vertexFramesEmbedded(Manifold& m) {
    auto& mesh = m.mesh();
    auto& geom = m.geometry();

    geom.requireVertexTangentBasis();
    geom.requireVertexNormals();

    int nV = m.nV();
    VertexFrames out;
    out.e1.resize(nV, 3);
    out.e2.resize(nV, 3);
    out.normals.resize(nV, 3);

    for (Vertex v : mesh.vertices()) {
        int vi = static_cast<int>(v.getIndex());
        Vector3 b0 = geom.vertexTangentBasis[v][0];
        Vector3 b1 = geom.vertexTangentBasis[v][1];
        Vector3 N  = geom.vertexNormals[v];
        out.e1     (vi, 0) = b0.x; out.e1     (vi, 1) = b0.y; out.e1     (vi, 2) = b0.z;
        out.e2     (vi, 0) = b1.x; out.e2     (vi, 1) = b1.y; out.e2     (vi, 2) = b1.z;
        out.normals(vi, 0) = N.x;  out.normals(vi, 1) = N.y;  out.normals(vi, 2) = N.z;
    }

    std::cout << "[vertex_frames] " << nV << " vertex frames computed" << std::endl;
    return out;
}

} // namespace nxr::manifold::geometry
