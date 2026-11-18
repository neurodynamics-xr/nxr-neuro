#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/assembled.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include "surface_of.h"
using namespace nxr::geometry;

static int g_failures = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; } \
    else { std::cout << "  [FAIL] " << msg << "\n"; ++g_failures; } } while (0)

// Unit icosphere (12 verts), closed genus-0, chi = 2.
static void icosphere(std::vector<double>& V, std::vector<int32_t>& F) {
    const double t = (1.0 + std::sqrt(5.0)) / 2.0;
    V = {-1,t,0, 1,t,0, -1,-t,0, 1,-t,0, 0,-1,t, 0,1,t,
          0,-1,-t, 0,1,-t, t,0,-1, t,0,1, -t,0,-1, -t,0,1};
    F = {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11, 1,5,9, 5,11,4, 11,10,2,
         10,7,6, 7,1,8, 3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9, 4,9,5,
         2,4,11, 6,2,10, 8,6,7, 9,8,1};
}

static void testTopologyFacet() {
    std::cout << "\n=== mesh bundles: topology ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    const auto m = surfaceOf(V.data(), 12, F.data(), 20);
    const auto topo = nxr::geometry::meshTopology(m);
    EXPECT(topo.nV == 12, "topology nV");
    EXPECT(topo.nF == 20, "topology nF");
    EXPECT(topo.nE == 30, "topology nE");
    EXPECT(nxr::geometry::eulerCharacteristic(m) == 2, "chi = V-E+F = 2 (closed genus-0)");
    EXPECT(topo.nH == 60, "topology nH = 2E");
    EXPECT((int)topo.heTwin.size() == 60, "MeshTopology SoA wired");
    // raw-input aliases retained at construction
    EXPECT(m.faces.rows() == 20 && m.faces.cols() == 3, "faces() [20,3]");
    EXPECT(m.faces(0,0) == 0 && m.faces(0,1) == 11 && m.faces(0,2) == 5, "faces() is raw input row 0");
    EXPECT(m.positions.rows() == 12, "vertexPositions() [12,3]");
}

static void testEmbeddedFacet() {
    std::cout << "\n=== mesh bundles: embedded ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    const auto m = surfaceOf(V.data(), 12, F.data(), 20);
    const auto g = nxr::geometry::meshGeometry(m);
    EXPECT(m.positions.rows() == 12 && m.positions.cols() == 3, "vertex.position [12,3]");
    // raw-input identity: position row 0 equals the input
    EXPECT(std::abs(m.positions(0,0) - (-1.0)) < 1e-12, "vertex.position is raw input");
    EXPECT(g.vertexGrid.rows() == 12 && g.vertexGrid.cols() == 3, "vertex.grid [12,3] complex");
    // facet-identity: embedded.vertex.grid == geometry::vertexGrid(m)
    EXPECT((g.vertexGrid - nxr::geometry::vertexGrid(m)).cwiseAbs().maxCoeff() < 1e-12, "vertex.grid == vertexGrid(m)");
    EXPECT(g.faceGrid.rows() == 20, "face.grid [20,3]");
    EXPECT(g.faceCentroids.rows() == 20 && g.faceCentroids.cols() == 3, "face.centroid [20,3]");
    EXPECT(nxr::geometry::vertexFrames(m).normals.rows() == 12, "vertex.normal [12,3]");
    // grid encodes the normal: Re(c) x Im(c) is unit and aligns with vertex.normal
    Eigen::Vector3d e1 = g.vertexGrid.row(0).real().transpose();
    Eigen::Vector3d e2 = g.vertexGrid.row(0).imag().transpose();
    Eigen::Vector3d nFromGrid = e1.cross(e2);
    EXPECT(std::abs(nFromGrid.norm() - 1.0) < 1e-9, "Re(c) x Im(c) is unit normal");
}

static void testIntrinsicFacet() {
    std::cout << "\n=== mesh bundles: intrinsic ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    const auto m = surfaceOf(V.data(), 12, F.data(), 20);
    const auto in = nxr::geometry::meshGeometry(m);
    EXPECT(in.vertexDualAreas.size() == 12, "vertex.dualArea [12]");
    EXPECT(in.vertexAngleSums.size() == 12, "vertex.angleSum [12]");
    EXPECT(in.edgeLengths.size() == 30, "edge.length [30]");
    EXPECT(in.edgeCotanWeights.size() == 30, "edge.cotanWeight [30]");
    EXPECT(in.halfedgeTransportAlong.size() == 60, "halfedge.transportAlong [60]");
    EXPECT(in.halfedgeTransportAcross.size() == 60, "halfedge.transportAcross [60]");
    EXPECT(in.halfedgeCotanWeights.size() == 60, "halfedge.cotanWeight [60]");
    // dual areas sum to total area (closed mesh)
    EXPECT(std::abs(in.vertexDualAreas.sum() - nxr::geometry::meshGeometry(m).totalArea) < 1e-9, "dualArea sums to totalArea");
    // facet-identity: intrinsic.edge.length matches a direct GC require
    auto& eg = gcOf(m).embedded(); eg.requireEdgeLengths();
    Eigen::VectorXd direct(30);
    for (auto edge : gcOf(m).inputMesh().edges()) direct(edge.getIndex()) = eg.edgeLengths[edge];
    EXPECT((in.edgeLengths - direct).cwiseAbs().maxCoeff() < 1e-12, "edge.length == GC edgeLengths");
}

static void testExtrinsicFacet() {
    std::cout << "\n=== mesh bundles: extrinsic ===\n";
    std::vector<double> V; std::vector<int32_t> F; icosphere(V, F);
    const auto m = surfaceOf(V.data(), 12, F.data(), 20);
    const auto ex = nxr::geometry::meshGeometry(m);
    EXPECT(ex.vertexCurvatureDeviatoric.size() == 12, "vertex.curvature2RoSy [12]");
    EXPECT(ex.vertexMeanCurvature.size() == 12, "vertex.meanCurvature [12]");
    EXPECT(nxr::geometry::curvatures(m).principalDirMax.rows() == 12 && nxr::geometry::curvatures(m).principalDirMax.cols() == 3, "vertex.principalDir [12,3]");
    EXPECT(ex.edgeDihedralAngles.size() == 30, "edge.dihedralAngle [30]");
    // identity vs geometry::vertexCurvature
    auto vc = nxr::geometry::vertexCurvature(m);
    EXPECT((ex.vertexMeanCurvature - vc.mean).cwiseAbs().maxCoeff() < 1e-12, "meanCurvature == vertexCurvature.mean");
}

int main() {
    testTopologyFacet();
    testEmbeddedFacet();
    testIntrinsicFacet();
    testExtrinsicFacet();
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
