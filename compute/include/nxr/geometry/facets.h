#pragma once
#include "nxr/geometry/mesh.h"

namespace nxr::geometry::facet {

// Thin view over the halfedge combinatorics. Holds a Manifold& only.
class TopologyFacet {
public:
    explicit TopologyFacet(Manifold& m) : m_(m) {}
    int nV() const; int nE() const; int nF() const; int nH() const; int nC() const;
    int eulerCharacteristic() const;                  // nV - nE + nF
    const geometry::MeshTopology& all() const;        // cached SoA
private:
    Manifold& m_;
};

class EmbeddedFacet {
public:
    explicit EmbeddedFacet(Manifold& m) : m_(m) {}
    struct VertexView { Manifold& m;
        Eigen::MatrixXd  position() const;   // [nV,3] raw input
        Eigen::MatrixXd  normal()   const;   // [nV,3]
        Eigen::MatrixXcd grid()     const;   // [nV,3] c = e1 + i e2
    };
    struct FaceView { Manifold& m;
        Eigen::MatrixXd  normal()   const;   // [nF,3]
        Eigen::MatrixXcd grid()     const;   // [nF,3]
        Eigen::MatrixXd  centroid() const;   // [nF,3]
        Eigen::VectorXd  area()     const;   // [nF]
    };
    VertexView vertex() const { return VertexView{m_}; }
    FaceView   face()   const { return FaceView{m_}; }
private:
    Manifold& m_;
};

// NOTE (intrinsic-Delaunay scope): the intrinsic facet DATA below slices from
// the embedded-sourced light geometry bundle, so under intrinsicDelaunay=true it
// is NOT swapped to the Delaunay metric — it stays consistent with the input
// embedding. The intrinsic OPERATORS (operators().laplacian().cotan(), mass) DO
// source from the operator mesh (the one Delaunay construction) and are certified-PSD under it. So
// edge().cotanWeight() here can differ from the weights backing the cotan
// operator under the intrinsic mesh option. Routing this data through the operator mesh is
// deferred to the fully-intrinsic Phase 3.
class IntrinsicFacet {
public:
    explicit IntrinsicFacet(Manifold& m) : m_(m) {}
    struct VertexView { Manifold& m;
        Eigen::VectorXd dualArea() const;     // [nV]
        Eigen::VectorXd angleSum() const;     // [nV]
    };
    struct EdgeView { Manifold& m;
        Eigen::VectorXd length()      const;  // [nE]
        Eigen::VectorXd cotanWeight() const;  // [nE]
    };
    struct HalfedgeView { Manifold& m;
        Eigen::VectorXd  cotanWeight()     const;  // [nH]
        Eigen::VectorXcd transportAlong()  const;  // [nH]
        Eigen::VectorXcd transportAcross() const;  // [nH]
    };
    VertexView   vertex()   const { return VertexView{m_}; }
    EdgeView     edge()     const { return EdgeView{m_}; }
    HalfedgeView halfedge() const { return HalfedgeView{m_}; }
private:
    Manifold& m_;
};

class ExtrinsicFacet {
public:
    explicit ExtrinsicFacet(Manifold& m) : m_(m) {}
    struct VertexView { Manifold& m;
        Eigen::VectorXcd curvature2RoSy() const;  // [nV] deviatoric q
        Eigen::VectorXd  meanCurvature()  const;  // [nV] H
        Eigen::MatrixXd  principalDir()   const;  // [nV,3] max principal dir
    };
    struct EdgeView { Manifold& m;
        Eigen::VectorXd dihedralAngle() const;    // [nE]
    };
    VertexView vertex() const { return VertexView{m_}; }
    EdgeView   edge()   const { return EdgeView{m_}; }
private:
    Manifold& m_;
};

} // namespace nxr::geometry::facet
