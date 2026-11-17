#pragma once
// The first slice of the MATLAB face (the same functions as the Python wheel's first slice): the pure functions (`compute.*`) and the readers of the
// field's file formats (`io.*`), as commands of the one `nxr_compute` mexFunction. The MATLAB package (+nxr/+compute, +nxr/+io) is what users call.
//
// A binding only converts and calls (ARCHITECTURE.md, bindings): arrays in, the library's call, arrays out. The 1-based MATLAB indices
// (faces, bins, samples, channels) are converted at this boundary and nowhere else.
#include <string>

#include "mex.h"

namespace nxr::mex {

// A failure of a first-slice command, with its MATLAB identifier: nxr:formatError, nxr:unsupported, nxr:invalidArgument, nxr:io (the library's error
// codes). Thrown (not raised here): mexErrMsgIdAndTxt throws a C++ exception on current MATLAB, which a catch-all inside the dispatcher would swallow,
// so the mexFunction raises it from a catch block of its own. Not a std::exception, so no other handler takes it.
struct SliceFailure {
  std::string id, text;
};

// Runs the command when it is one of the first slice's (its name starts with "compute." or "io."); returns false when it is not. Throws SliceFailure for the
// library's invalid-argument and io errors; the library's other errors (nxr::core::Error) pass through.
bool dispatchFirstSlice(const std::string& cmd, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);

// The pipeline level's commands (names starting with "run.": `nxr.run`, `nxr.list`, `nxr.read`; pipeline_slice.cpp, built when the pipelines are); false when it is not one of them.
bool dispatchPipeline(const std::string& cmd, int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);

}  // namespace nxr::mex
