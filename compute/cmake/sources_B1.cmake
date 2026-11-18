# Sources of nxr_compute owned by refactor batch B1 (geometry + operators), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/operators/mesh_operators.cpp
  src/geometry/normalize_delaunay.cpp
  src/operators/eigensolver.cpp
  src/geometry/nearest.cpp
  src/operators/magnetic.cpp
  src/geometry/transport.cpp
  src/operators/normalize.cpp
  src/operators/cholesky_cache.cpp
  src/operators/poisson_solver.cpp
  src/geometry/geodesic.cpp
  src/geometry/chart.cpp
  src/geometry/frames.cpp
  src/operators/hodge.cpp
  src/synthesis/field_generators.cpp
  src/geometry/curvatures.cpp
  src/geometry/normals.cpp
  src/operators/vector_field.cpp
  src/signal/isolines.cpp
  src/geometry/direction_field.cpp
  src/geometry/face_frames.cpp
  src/geometry/vertex_frames.cpp
  src/geometry/geodesic_path.cpp
  src/geometry/parametrization.cpp
  src/geometry/vector_heat.cpp
  src/geometry/signed_heat.cpp
  src/geometry/smooth_field.cpp
  src/geometry/stripe_patterns.cpp
  src/operators/connection_laplacian.cpp
  src/operators/covariant_laplacian.cpp
  src/geometry/query_measure.cpp
  src/geometry/geometry_grid.cpp
  src/geometry/intrinsic_surface.cpp
  src/geometry/gauge.cpp
  src/geometry/topology.cpp
  src/geometry/geometry_bundle.cpp
  src/operators/graph_laplacian.cpp
  src/geometry/facets.cpp
  src/operators/covariant_differential.cpp
  src/operators/extrinsic_weitzenbock.cpp
  src/operators/dirac_operator.cpp
  src/operators/face_gradient.cpp
  src/operators/hodge_weights.cpp
  src/operators/sphere_registration.cpp
  src/operators/eigenproblem.cpp
  src/operators/hodge_dirac.cpp
  src/operators/spin_dirac.cpp
  src/operators/operator_registry.cpp
  src/operators/field_registry.cpp
  src/geometry/glue.cpp
)

# Treat an unhandled enumerator in a no-default switch as an error for the
# operator-registry TU (so a new OperatorId can't ship without metadata).
# GCC/Clang: -Werror=switch. MSVC: C4062 is the no-default analogue; promote
# it to an error with /we4062. The flags are mutually exclusive: cl.exe parses
# "/Wswitch" as "/W<arg>" and fails with D8021.
if(MSVC)
    set_source_files_properties(src/operators/operator_registry.cpp PROPERTIES
        COMPILE_OPTIONS "/we4062")
else()
    set_source_files_properties(src/operators/operator_registry.cpp PROPERTIES
        COMPILE_OPTIONS "-Wswitch;-Werror=switch")
endif()
# glue.cpp is a port of the Studio's glue.ts (no FMA in JavaScript): contraction off, so the bridge's positions and the stitch's ties are the Studio's
set_source_files_properties(src/geometry/glue.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# the pinned geometry-central commit (NXR_GC_COMMIT, read in the main file) goes into intrinsic_surface.cpp
set_property(SOURCE src/geometry/intrinsic_surface.cpp APPEND PROPERTY COMPILE_DEFINITIONS NXR_GEOMETRY_CENTRAL_COMMIT="${NXR_GC_COMMIT}")
