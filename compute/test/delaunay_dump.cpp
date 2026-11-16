// Parity helper: runs THE one Delaunay construction (nxr/delaunay.h) on a mesh and prints the result as JSON (full precision) for
// test/parity/test_delaunay_parity.py, which compares it to the Studio's intrinsicDelaunay.ts and nxr's `prepare`.
//
//   delaunay_dump <mesh.bin> <mollify> <flipEps> [embedded]
//
// mesh.bin: int32 nV, int32 nF, float64 positions [3 nV], int32 faces [3 nF] (little-endian). Synthetic meshes only.
#include "nxr/delaunay.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: delaunay_dump mesh.bin mollify flipEps [embedded]\n"; return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    int32_t nV = 0, nF = 0;
    in.read(reinterpret_cast<char*>(&nV), 4);
    in.read(reinterpret_cast<char*>(&nF), 4);
    std::vector<double> P(3 * static_cast<size_t>(nV));
    std::vector<int32_t> F(3 * static_cast<size_t>(nF));
    in.read(reinterpret_cast<char*>(P.data()), static_cast<std::streamsize>(P.size() * 8));
    in.read(reinterpret_cast<char*>(F.data()), static_cast<std::streamsize>(F.size() * 4));
    if (!in) { std::cerr << "delaunay_dump: short read\n"; return 2; }
    nxr::manifold::delaunay::DelaunayOptions o;
    o.mollify = std::atof(argv[2]);
    o.flipEps = std::atof(argv[3]);
    const bool intrinsic = !(argc > 4 && std::string(argv[4]) == "embedded");
    const nxr::manifold::delaunay::Mesh m = nxr::manifold::delaunay::prepare(P.data(), nV, F.data(), nF, o, intrinsic);
    auto arr = [](const char* name, const auto& v, bool last = false) {
        std::printf("\"%s\":[", name);
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) std::printf(",");
            if constexpr (std::is_floating_point_v<typename std::decay_t<decltype(v)>::value_type>) std::printf("%.17g", static_cast<double>(v[i]));
            else std::printf("%lld", static_cast<long long>(v[i]));
        }
        std::printf("]%s", last ? "" : ",");
    };
    std::printf("{");
    arr("F", m.F); arr("twin", m.twin); arr("length", m.length); arr("mass", m.mass);
    arr("angle", m.angle); arr("angleSum", m.angleSum); arr("ref", m.ref);
    std::printf("\"record\":{\"mollify\":%.17g,\"epsilon\":%.17g,\"flipEps\":%.17g,\"flips\":%d},", m.record.mollify, m.record.epsilon, m.record.flipEps, m.record.flips);
    int ties = 0;   // edges with a cotan weight of exactly ~0 in the result: cocircular quads, where several Delaunay triangulations are valid
    for (int e = 0; e < m.nE(); ++e) if (std::fabs(nxr::manifold::delaunay::edgeWeight(m, e)) < 1e-9) ++ties;
    std::printf("\"ties\":%d,\"nonDelaunay\":%d}\n", ties, nxr::manifold::delaunay::nonDelaunayEdges(m, o.flipEps));
    return 0;
}
