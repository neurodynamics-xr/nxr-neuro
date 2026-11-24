// nxr-pipelines — `nxr-neuro migrate eigensolve`: a stored Hodge–Dirac eigensolve of the earlier form (three operators _eigenvectors, _forms1, _forms2; a kind field; stored d0, d1 and stars)
// converted in place to the four classes (eigen.h); a solve already in the four classes, made by an earlier version, brought to the current form (a coexact mode's vertex rows held its stream
// function: set to 0, its 0-form part). No solve, idempotent, an unknown form refused. See eigen_migrate.cpp.
#pragma once
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

/** the subject's eigensolve named `stem` (`hodgeDirac`, `hodgeDirac__<surface>`), or every stored one (null): each converted in ONE unit, or reported already in the new form.
 *  Returns {subject, stems: [{stem, status, ...}]}. Throws Error(invalid_argument) for a form it does not know (nothing of that stem changed). Memory: one stem's eigenvectors, (V + E + F) x K float32. */
nxr::io::Json migrate_eigensolve(nxr::io::SubjectWriter& sub, const std::optional<std::string>& stem, const std::function<void(const nxr::io::Json&)>& progress = {});

/** `nxr-neuro migrate glue`: the subject's glued cortex named `glued` (every one: null) of the earlier form (the field `<glued>_map` and the indices `<glued>_bridge`) converted in place to the
 *  hemispheres labelling of the glued cortex (the original's labelling `hemispheres`, its entries `left` and `right` checked against the original surface, carried over, `bridge` added) and the
 *  original's kept vertices; the map is checked against the glue's rule before it is moved aside. Idempotent; an unknown form refused (nothing of it changed). */
nxr::io::Json migrate_glue(nxr::io::SubjectWriter& sub, const std::optional<std::string>& glued, const std::string& hemispheres, const std::string& left, const std::string& right,
                           const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
