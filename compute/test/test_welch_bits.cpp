// test_welch_bits: the Welch classes as the composition of the segment family's stft and estimate give BITWISE what the old classes gave (golden/welch_bits_golden.txt, made by golden/gen_welch_golden_bits.cpp from
// the code of commit 23f04a3, before the composition): psd, complex psd, cross-spectrum with and without quadrature, the linear map, the functions psd / xpsd, with a bad span, three option sets; fed whole
// and in chunks of 333 and 1024 samples (a segment crosses chunk edges: the result does not depend on the cut). Synthetic only.
#include "welch_inputs.h"
#include <iostream>
int main() {
    goldenhash::Reader r(std::string(NXR_TEST_GOLDEN_DIR) + "/welch_bits_golden.txt");
    if (!r.good()) { std::cout << "  [FAIL] the golden does not open\n"; return 1; }
    int failures = 0, checked = 0;
    for (const int chunk : {0, 333, 1024}) {
        wi::runs([&](const std::string& n, const auto& v) {
            ++checked; std::string why;
            const bool ok = [&] { if constexpr (std::is_arithmetic_v<std::decay_t<decltype(v)>>) return r.same(n, double(v), &why); else return r.same(n, v, &why); }();
            if (!ok) { ++failures; std::cout << "  [FAIL] chunk " << chunk << ": " << why << "\n"; }
        }, chunk);
    }
    std::cout << checked << " arrays checked, " << failures << " differ\n" << (failures ? "FAILED" : "all passed") << "\n";
    return failures ? 1 : 0;
}
