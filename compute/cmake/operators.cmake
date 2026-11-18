# nxr_compute: the operators toolbox (compute/include/nxr/operators/, compute/src/operators/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/operators/assembled.cpp
  src/operators/factorization.cpp
  src/operators/connection_laplacian.cpp
  src/operators/covariant_differential.cpp
  src/operators/covariant_laplacian.cpp
  src/operators/dirac_operator.cpp
  src/operators/eigenproblem.cpp
  src/operators/eigensolver.cpp
  src/operators/extrinsic_weitzenbock.cpp
  src/operators/face_gradient.cpp
  src/operators/field_registry.cpp
  src/operators/graph_laplacian.cpp
  src/operators/hodge_dirac.cpp
  src/operators/hodge_weights.cpp
  src/operators/magnetic.cpp
  src/operators/mesh_operators.cpp
  src/operators/normalize.cpp
  src/operators/operator_registry.cpp
  src/operators/poisson_solver.cpp
  src/operators/sphere_registration.cpp
  src/operators/spin_dirac.cpp
  src/operators/trivial_connection.cpp
  src/operators/trivial_gauge.cpp
  src/operators/vector_field.cpp
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

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_eigen test/test_eigen.cpp)
  target_link_libraries(test_eigen PRIVATE nxr_compute)
  add_test(NAME test_eigen COMMAND test_eigen)
  add_executable(test_eigen_consistency test/test_eigen_consistency.cpp)
  target_link_libraries(test_eigen_consistency PRIVATE nxr_compute)
  add_test(NAME test_eigen_consistency COMMAND test_eigen_consistency)
  add_executable(test_factorization test/test_factorization.cpp)
  target_link_libraries(test_factorization PRIVATE nxr_compute)
  add_test(NAME test_factorization COMMAND test_factorization)
  add_executable(test_mass_variants test/test_mass_variants.cpp)
  target_link_libraries(test_mass_variants PRIVATE nxr_compute)
  add_test(NAME test_mass_variants COMMAND test_mass_variants)
  add_executable(test_operator_registry test/test_operator_registry.cpp)
  target_link_libraries(test_operator_registry PRIVATE nxr_compute)
  add_test(NAME test_operator_registry COMMAND test_operator_registry)
  add_executable(test_field_registry test/test_field_registry.cpp)
  target_link_libraries(test_field_registry PRIVATE nxr_compute)
  add_test(NAME test_field_registry COMMAND test_field_registry)
  add_executable(test_connection_laplacian test/test_connection_laplacian.cpp)
  target_link_libraries(test_connection_laplacian PRIVATE nxr_compute)
  add_test(NAME test_connection_laplacian COMMAND test_connection_laplacian)
  add_executable(test_operators_facet test/test_operators_facet.cpp)
  target_link_libraries(test_operators_facet PRIVATE nxr_compute)
  add_test(NAME test_operators_facet COMMAND test_operators_facet)
  add_executable(test_covariant_differential test/test_covariant_differential.cpp)
  target_link_libraries(test_covariant_differential PRIVATE nxr_compute)
  add_test(NAME test_covariant_differential COMMAND test_covariant_differential)
  add_executable(test_dirac_operator test/test_dirac_operator.cpp)
  target_link_libraries(test_dirac_operator PRIVATE nxr_compute)
  add_test(NAME test_dirac_operator COMMAND test_dirac_operator)
  add_executable(test_dirac_face_operator test/test_dirac_face_operator.cpp)
  target_link_libraries(test_dirac_face_operator PRIVATE nxr_compute)
  add_test(NAME test_dirac_face_operator COMMAND test_dirac_face_operator)
  add_executable(test_operator_eigensolve test/test_operator_eigensolve.cpp)
  target_link_libraries(test_operator_eigensolve PRIVATE nxr_compute)
  add_test(NAME test_operator_eigensolve COMMAND test_operator_eigensolve)
  add_executable(test_hodge_dirac test/test_hodge_dirac.cpp)
  target_link_libraries(test_hodge_dirac PRIVATE nxr_compute)
  add_test(NAME test_hodge_dirac COMMAND test_hodge_dirac)
  add_executable(test_spin_dirac test/test_spin_dirac.cpp)
  target_link_libraries(test_spin_dirac PRIVATE nxr_compute)
  add_test(NAME test_spin_dirac COMMAND test_spin_dirac)
  add_executable(test_graph_eigen test/test_graph_eigen.cpp)
  target_link_libraries(test_graph_eigen PRIVATE nxr_compute)
  add_test(NAME test_graph_eigen COMMAND test_graph_eigen)
  add_executable(test_face_gradient test/test_face_gradient.cpp)
  target_link_libraries(test_face_gradient PRIVATE nxr_compute)
  add_test(NAME test_face_gradient COMMAND test_face_gradient)
  add_executable(test_hodge_weights test/test_hodge_weights.cpp)
  target_link_libraries(test_hodge_weights PRIVATE nxr_compute)
  target_compile_definitions(test_hodge_weights PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_hodge_weights COMMAND test_hodge_weights)
  add_executable(test_eigen_complete test/test_eigen_complete.cpp)
  target_link_libraries(test_eigen_complete PRIVATE nxr_compute)
  add_test(NAME test_eigen_complete COMMAND test_eigen_complete)
  add_executable(test_magnetic test/test_magnetic.cpp)
  target_link_libraries(test_magnetic PRIVATE nxr_compute)
  target_compile_definitions(test_magnetic PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_magnetic COMMAND test_magnetic)
  add_executable(test_sphere_registration test/test_sphere_registration.cpp)
  target_link_libraries(test_sphere_registration PRIVATE nxr_compute)
  add_test(NAME test_sphere_registration COMMAND test_sphere_registration)
  add_executable(test_vector_covariant_operators test/test_vector_covariant_operators.cpp)
  target_link_libraries(test_vector_covariant_operators PRIVATE nxr_compute)
  add_test(NAME test_vector_covariant_operators COMMAND test_vector_covariant_operators)
endif()
