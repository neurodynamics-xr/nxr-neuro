# nxr_compute: the top-level sources that stay until the behaviour batch deletes them (pyramid.h and rotary.h are gone; integrals.h and region.h are gone: operators/weak_forms.h and measurement::reduce).
# Paths are relative to compute/.


if(NOT EMSCRIPTEN AND NXR_COMPUTE_BUILD_TESTS)
endif()
