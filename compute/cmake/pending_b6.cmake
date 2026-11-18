# nxr_compute: the top-level sources that stay until the behaviour batch deletes them (pyramid.h and rotary.h are in the plan's removed list; integrals.h and region.h are gone: operators/weak_forms.h and measurement::reduce).
# Paths are relative to compute/.

target_sources(nxr_compute PRIVATE
  src/pyramid.cpp
)

if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
endif()
