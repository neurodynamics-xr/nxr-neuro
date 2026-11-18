# nxr_compute: the geometry toolbox (compute/include/nxr/geometry/, compute/src/geometry/): its sources, their per-file flags and its tests. Included by compute/CMakeLists.txt.
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/geometry/chart.cpp
  src/geometry/curvatures.cpp
  src/geometry/face_areas.cpp
  src/geometry/face_frames.cpp
  src/geometry/facets.cpp
  src/geometry/frames.cpp
  src/geometry/manifold.cpp
  src/geometry/geodesic.cpp
  src/geometry/geodesic_path.cpp
  src/geometry/geometry_bundle.cpp
  src/geometry/geometry_grid.cpp
  src/geometry/glue.cpp
  src/geometry/intrinsic_surface.cpp
  src/geometry/nearest.cpp
  src/geometry/normalize_delaunay.cpp
  src/geometry/normals.cpp
  src/geometry/parametrization.cpp
  src/geometry/query_measure.cpp
  src/geometry/signed_heat.cpp
  src/geometry/smooth_field.cpp
  src/geometry/stripe_patterns.cpp
  src/geometry/topology.cpp
  src/geometry/transport.cpp
  src/geometry/vector_heat.cpp
  src/geometry/vertex_frames.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
  add_executable(test_graph_agnostic test/test_graph_agnostic.cpp)
  target_link_libraries(test_graph_agnostic PRIVATE nxr_compute)
  add_test(NAME test_graph_agnostic COMMAND test_graph_agnostic)
  add_executable(test_geometry_central_extras test/test_geometry_central_extras.cpp)
  target_link_libraries(test_geometry_central_extras PRIVATE nxr_compute)
  add_test(NAME test_geometry_central_extras COMMAND test_geometry_central_extras)
  add_executable(test_geometry_bundle test/test_geometry_bundle.cpp)
  target_link_libraries(test_geometry_bundle PRIVATE nxr_compute)
  add_test(NAME test_geometry_bundle COMMAND test_geometry_bundle)
  add_executable(test_passthrough_accessors test/test_passthrough_accessors.cpp)
  target_link_libraries(test_passthrough_accessors PRIVATE nxr_compute)
  add_test(NAME test_passthrough_accessors COMMAND test_passthrough_accessors)
  add_executable(test_query_measure test/test_query_measure.cpp)
  target_link_libraries(test_query_measure PRIVATE nxr_compute)
  add_test(NAME test_query_measure COMMAND test_query_measure)
  add_executable(test_normalize test/test_normalize.cpp)
  target_link_libraries(test_normalize PRIVATE nxr_compute)
  add_test(NAME test_normalize COMMAND test_normalize)
  add_executable(test_intrinsic_surface test/test_intrinsic_surface.cpp)
  target_link_libraries(test_intrinsic_surface PRIVATE nxr_compute)
  target_compile_definitions(test_intrinsic_surface PRIVATE NXR_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
  add_test(NAME test_intrinsic_surface COMMAND test_intrinsic_surface)
  add_executable(test_surface_arrays test/test_surface_arrays.cpp)
  target_link_libraries(test_surface_arrays PRIVATE nxr_compute)
  add_test(NAME test_surface_arrays COMMAND test_surface_arrays)
  add_executable(test_intrinsic_delaunay test/test_intrinsic_delaunay.cpp)
  target_link_libraries(test_intrinsic_delaunay PRIVATE nxr_compute)
  add_test(NAME test_intrinsic_delaunay COMMAND test_intrinsic_delaunay)
  add_executable(test_facets test/test_facets.cpp)
  target_link_libraries(test_facets PRIVATE nxr_compute)
  add_test(NAME test_facets COMMAND test_facets)
  add_executable(test_gauge_workflow test/test_gauge_workflow.cpp)
  target_link_libraries(test_gauge_workflow PRIVATE nxr_compute)
  add_test(NAME test_gauge_workflow COMMAND test_gauge_workflow)
  add_executable(test_cf_parity test/test_cf_parity.cpp)
  target_link_libraries(test_cf_parity PRIVATE nxr_compute)
  add_test(NAME test_cf_parity COMMAND test_cf_parity)
  add_executable(test_chart test/test_chart.cpp)
  target_link_libraries(test_chart PRIVATE nxr_compute)
  target_compile_definitions(test_chart PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_chart COMMAND test_chart)
  add_executable(test_transport test/test_transport.cpp)
  target_link_libraries(test_transport PRIVATE nxr_compute)
  add_test(NAME test_transport COMMAND test_transport)
  add_executable(test_glue test/test_glue.cpp)
  target_link_libraries(test_glue PRIVATE nxr_compute)
  target_compile_definitions(test_glue PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_glue COMMAND test_glue)
  add_executable(test_frames test/test_frames.cpp)
  target_link_libraries(test_frames PRIVATE nxr_compute)
  target_compile_definitions(test_frames PRIVATE NXR_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
  add_test(NAME test_frames COMMAND test_frames)
endif()
