// Generator of psd_linear_map_golden.txt: the OLD nxr::meeg::sourcePower (compiled against the code of commit 5c67a87, before it became the psd of a linear map) on compute/test/linear_map_inputs.h; hashes of the
// raw doubles. Not built by CMake; to remake it, check out 5c67a87.
#include "../golden_hash.h"
#include "../linear_map_inputs.h"
#include "nxr/meeg/sources.h"
int main(int argc, char** argv) {
    goldenhash::Writer w(argc > 1 ? argv[1] : "psd_linear_map_golden.txt");
    const auto S = lmi::spectra();
    using nxr::meeg::Orientation;
    w.put("free.norm", nxr::meeg::sourcePower(lmi::kernel(3), S, 3, Orientation::Norm));
    w.put("free.components", nxr::meeg::sourcePower(lmi::kernel(3), S, 3, Orientation::Components));
    w.put("fixed.norm", nxr::meeg::sourcePower(lmi::kernel(1), S, 1, Orientation::Norm));
    w.put("fixed.components", nxr::meeg::sourcePower(lmi::kernel(1), S, 1, Orientation::Components));
}
