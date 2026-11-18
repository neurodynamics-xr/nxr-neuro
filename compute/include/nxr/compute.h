#pragma once
// TEMPORARY FORWARDER (toolbox refactor): deleted at the end of batch B5, see tools/refactor.py forwarders.
// compute.h was split into nxr/geometry/*.h and nxr/operators/*.h (and synthesis/fields.h, signal/surface_calculus.h).
// The layout convention for matrix returns (row-major [V x 3] / [F x 3]; [V x K] eigenvectors row-major flat; [T x V] frame-major) stays as stated in the git history of this file.
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/geometry/facets.h"
#include "nxr/operators/cache.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/operators/hodge_decomposition.h"
#include "nxr/operators/facets.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"

namespace nxr::manifold {
using namespace nxr::geometry;
namespace geometry = nxr::geometry;
namespace ops = nxr::operators;
namespace query = nxr::geometry::query;
namespace measure = nxr::geometry::measure;
namespace transport = nxr::geometry::transport;
namespace connection = nxr::geometry::connection;
namespace parametrization = nxr::geometry::parametrization;
namespace differential = nxr::operators::differential;
}
namespace nxr::manifold::solve { using namespace nxr::operators; using namespace nxr::geometry; }
namespace nxr::field {
namespace generate = nxr::synthesis;
namespace interp = nxr::signal;
namespace extract = nxr::signal;
namespace op = nxr::operators;
}

namespace nxr::manifold {
    // Type using-declarations only — function using-decls conflict with
    // out-of-line definitions in src/*.cpp (the .cpp opens a sub-namespace
    // block like `namespace nxr::manifold::connection { trivial(...) { … } }`
    // and the using-decl `using connection::trivial` is treated as
    // declaring `nxr::manifold::trivial`, which clashes with the
    // canonical sub-namespace definition). Types don't have this issue.
    //
    // Consumers must use the sub-namespace paths for free functions
    // (e.g. `nxr::manifold::ops::d0(m)`, `nxr::manifold::solve::eigen(...)`).
    using ops::MassMatrixVariant;
    using ops::ManifoldOperators;
    using ops::DECOperators;
    using ops::CholeskyCache;
    using ops::laplacian::connection::ConnectionDomain;
    using ops::laplacian::connection::ConnectionLaplacianFormat;
    using ops::laplacian::connection::ConnectionLaplacianOptions;
    using ops::laplacian::connection::ConnectionLaplacian;
    using solve::EigenResult;
    using solve::HeatGeodesicSolver;
    using solve::SignedHeatSolver;
    using solve::SignedHeatLevelSet;
    using solve::HodgeResult;
    using transport::VectorHeatSolver;
    using transport::LogMapResult;
    using transport::LogMapStrategy;
    using connection::DirectionFieldResult;
    using connection::SmoothVertexFieldResult;
    using parametrization::stripes::StripePatternResult;
    using geometry::CurvatureResult;
    using geometry::NormalType;
    using geometry::FaceFrames;
    using geometry::VertexFrames;
    using geometry::MeshTopology;
} // namespace nxr::manifold
