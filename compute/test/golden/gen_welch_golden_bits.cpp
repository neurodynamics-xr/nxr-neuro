// Generator of welch_bits_golden.txt: the OLD Welch, ComplexWelch, CrossWelch and LinearWelch (compiled against the code of commit 23f04a3, before the stft + estimate composition) on the synthetic
// recording of compute/test/welch_inputs.h, fed whole; hashes of the raw doubles. Not built by CMake; to remake it, check out 23f04a3.
#include "../welch_inputs.h"
int main(int argc, char** argv) {
    goldenhash::Writer w(argc > 1 ? argv[1] : "welch_bits_golden.txt");
    wi::runs([&](const std::string& n, const auto& v) { w.put(n, v); }, 0);
}
