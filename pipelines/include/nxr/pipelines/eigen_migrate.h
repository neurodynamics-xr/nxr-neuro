// nxr-pipelines — `nxr-neuro migrate eigensolve`: a stored Hodge–Dirac eigensolve of the earlier form (three operators _eigenvectors, _forms1, _forms2; a kind field; stored d0, d1 and stars)
// converted in place to the four classes (eigen.h): no solve, idempotent, an unknown form refused. See eigen_migrate.cpp.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

/** the subject's eigensolve named `stem` (`hodgeDirac`, `hodgeDirac__<surface>`), or every stored one (null): each converted in ONE unit, or reported already in the new form.
 *  Returns {subject, stems: [{stem, status, ...}]}. Throws Error(invalid_argument) for a form it does not know (nothing of that stem changed). Memory: one stem's eigenvectors, (V + E + F) x K float32. */
nxr::io::Json migrate_eigensolve(nxr::io::SubjectWriter& sub, const std::optional<std::string>& stem, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
