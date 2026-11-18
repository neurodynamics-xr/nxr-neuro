# Sources of nxr_compute owned by refactor batch B1 (geometry + operators), included by compute/CMakeLists.txt.
# Paths are relative to compute/. Per-file flags sit with their files. Temporary: consolidated into compute/cmake/<toolbox>.cmake in B5.

target_sources(nxr_compute PRIVATE
  src/mesh_operators.cpp
  src/normalize_delaunay.cpp
  src/eigensolver.cpp
  src/nearest.cpp
  src/magnetic.cpp
  src/transport.cpp
  src/normalize.cpp
  src/cholesky_cache.cpp
  src/poisson_solver.cpp
  src/geodesic.cpp
  src/chart.cpp
  src/frames.cpp
  src/hodge.cpp
  src/field_generators.cpp
  src/curvatures.cpp
  src/normals.cpp
  src/vector_field.cpp
  src/isolines.cpp
  src/direction_field.cpp
  src/face_frames.cpp
  src/vertex_frames.cpp
  src/geodesic_path.cpp
  src/parametrization.cpp
  src/vector_heat.cpp
  src/signed_heat.cpp
  src/smooth_field.cpp
  src/stripe_patterns.cpp
  src/connection_laplacian.cpp
  src/covariant_laplacian.cpp
  src/query_measure.cpp
  src/geometry_grid.cpp
  src/intrinsic_surface.cpp
  src/gauge.cpp
  src/topology.cpp
  src/geometry_bundle.cpp
  src/graph_laplacian.cpp
  src/facets.cpp
  src/covariant_differential.cpp
  src/extrinsic_weitzenbock.cpp
  src/dirac_operator.cpp
  src/face_gradient.cpp
  src/hodge_weights.cpp
  src/sphere_registration.cpp
  src/eigenproblem.cpp
  src/hodge_dirac.cpp
  src/spin_dirac.cpp
  src/operator_registry.cpp
  src/field_registry.cpp
  src/glue.cpp
)

# Treat an unhandled enumerator in a no-default switch as an error for the
# operator-registry TU (so a new OperatorId can't ship without metadata).
# GCC/Clang: -Werror=switch. MSVC: C4062 is the no-default analogue; promote
# it to an error with /we4062. The flags are mutually exclusive: cl.exe parses
# "/Wswitch" as "/W<arg>" and fails with D8021.
if(MSVC)
    set_source_files_properties(src/operator_registry.cpp PROPERTIES
        COMPILE_OPTIONS "/we4062")
else()
    set_source_files_properties(src/operator_registry.cpp PROPERTIES
        COMPILE_OPTIONS "-Wswitch;-Werror=switch")
endif()
# glue.cpp is a port of the Studio's glue.ts (no FMA in JavaScript): contraction off, so the bridge's positions and the stitch's ties are the Studio's
set_source_files_properties(src/glue.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
# the pinned geometry-central commit (NXR_GC_COMMIT, read in the main file) goes into intrinsic_surface.cpp
set_property(SOURCE src/intrinsic_surface.cpp APPEND PROPERTY COMPILE_DEFINITIONS NXR_GEOMETRY_CENTRAL_COMMIT="${NXR_GC_COMMIT}")
