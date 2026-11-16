// nxr-pipelines — what a subject holds, by kind: its recordings, inverse kernels and stored results. `nxr-neuro list recordings|kernels|results` prints it.
//
//   recordings  the fields of kind "recording"
//   kernels     the operators of kind "inverse kernel"
//   results     every other complete field and statistic (what the computations stored)
//
// Rows are read from the dataset's database: nothing is loaded, nothing computed. The command line only parses and prints.
#pragma once
#include <string>

#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::pipelines {

/** [{name, kind, table}] of the subject's complete nodes of `what` ("recordings", "kernels" or "results"), by name; any other `what` is an Error (invalid_argument) */
nxr::io::Json list_subject_nodes(nxr::io::DatasetWriter& dataset, nxr::io::SubjectWriter& subject, const std::string& what);

}  // namespace nxr::pipelines
