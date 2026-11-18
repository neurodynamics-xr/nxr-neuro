// nxr::selection::bisection — see nxr/selection/dyadic.h. A line port of the Cortical Flow Studio's engine/ladder.ts; compiled with -ffp-contract=off
// (JavaScript has no fused multiply-add).
#include "nxr/select/dyadic.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

namespace nxr::select::bisection {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

std::vector<int> pieces(const int32_t* F, int nF, int nV) {
    std::vector<int> par(static_cast<size_t>(nV));
    std::iota(par.begin(), par.end(), 0);
    auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    for (int f = 0; f < nF; ++f) { par[find(F[3 * f])] = find(F[3 * f + 1]); par[find(F[3 * f + 1])] = find(F[3 * f + 2]); }
    std::vector<int> id(static_cast<size_t>(nV), -1), out(static_cast<size_t>(nV));
    int k = 0;
    for (int i = 0; i < nV; ++i) { const int r = find(i); if (id[r] < 0) id[r] = k++; out[i] = id[r]; }
    return out;
}

double weylLambda(double modes, double area) { return (4 * M_PI * modes) / area; }
double weylModes(double lambda, double area) { return (area * lambda) / (4 * M_PI); }
double wavelengthOf(double lambda) { return lambda > 0 ? (2 * M_PI) / std::sqrt(lambda) : kInf; }

double jsHypot(double a, double b, double c) {
    const double x[3] = {std::fabs(a), std::fabs(b), std::fabs(c)};
    double mx = 0;
    for (double v : x) { if (std::isnan(v)) return kNaN; mx = std::max(mx, v); }
    if (mx == 0) return 0.0;
    if (std::isinf(mx)) return kInf;
    double sum = 0, comp = 0;
    for (double v : x) {
        const double n = v / mx, summand = n * n - comp, pre = sum + summand;
        comp = (pre - sum) - summand;
        sum = pre;
    }
    return std::sqrt(sum) * mx;
}

Mesh componentMesh(const double* positions, const int32_t* faces, int nF, int from, int n) {
    Mesh m;
    m.n = n;
    m.positions.assign(positions + 3 * static_cast<size_t>(from), positions + 3 * (static_cast<size_t>(from) + static_cast<size_t>(n)));
    for (int f = 0; f < nF; ++f) {
        const int a = faces[3 * f] - from, b = faces[3 * f + 1] - from, c = faces[3 * f + 2] - from;
        if (a >= 0 && b >= 0 && c >= 0 && a < n && b < n && c < n) { m.faces.push_back(a); m.faces.push_back(b); m.faces.push_back(c); }
    }
    return m;
}

std::vector<double> vertexAreas(const Mesh& mesh) {
    const double* p = mesh.positions.data();
    std::vector<double> mass(static_cast<size_t>(mesh.n), 0.0);
    for (size_t f = 0; f + 2 < mesh.faces.size(); f += 3) {
        const size_t a = static_cast<size_t>(mesh.faces[f]), b = static_cast<size_t>(mesh.faces[f + 1]), c = static_cast<size_t>(mesh.faces[f + 2]);
        const double ux = p[3 * b] - p[3 * a], uy = p[3 * b + 1] - p[3 * a + 1], uz = p[3 * b + 2] - p[3 * a + 2];
        const double vx = p[3 * c] - p[3 * a], vy = p[3 * c + 1] - p[3 * a + 1], vz = p[3 * c + 2] - p[3 * a + 2];
        const double area = 0.5 * jsHypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
        mass[a] += area / 3; mass[b] += area / 3; mass[c] += area / 3;
    }
    return mass;
}

namespace {
struct Graph { std::vector<int32_t> to, start; std::vector<double> len; };

Graph meshGraph(const Mesh& mesh) {
    const double* p = mesh.positions.data();
    const size_t n = static_cast<size_t>(mesh.n);
    std::vector<std::vector<int32_t>> sets(n);
    auto add = [&](int32_t a, int32_t b) {
        auto& sa = sets[static_cast<size_t>(a)];
        if (std::find(sa.begin(), sa.end(), b) == sa.end()) { sa.push_back(b); sets[static_cast<size_t>(b)].push_back(a); }
    };
    for (size_t f = 0; f + 2 < mesh.faces.size(); f += 3) { add(mesh.faces[f], mesh.faces[f + 1]); add(mesh.faces[f + 1], mesh.faces[f + 2]); add(mesh.faces[f + 2], mesh.faces[f]); }
    Graph g;
    g.start.assign(n + 1, 0);
    for (size_t v = 0; v < n; ++v) g.start[v + 1] = g.start[v] + static_cast<int32_t>(sets[v].size());
    g.to.resize(static_cast<size_t>(g.start[n])); g.len.resize(static_cast<size_t>(g.start[n]));
    for (size_t v = 0; v < n; ++v) {
        size_t i = static_cast<size_t>(g.start[v]);
        for (int32_t w : sets[v]) { const size_t u = static_cast<size_t>(w); g.to[i] = w; g.len[i] = jsHypot(p[3 * v] - p[3 * u], p[3 * v + 1] - p[3 * u + 1], p[3 * v + 2] - p[3 * u + 2]); ++i; }
    }
    return g;
}

// V8's Array.prototype.sort with a comparator that may return NaN (read as 0): below 8 elements a plain binary insertion; 8..63 one run (CountAndMakeRun) extended by binary insertion
// (array-sort.tq); longer arrays fall back to a stable merge sort (V8's TimSort, identical for a consistent comparator). As nxr `geom.ladder.v8_sort`.
std::vector<int32_t> v8Sort(std::vector<int32_t> a, const std::function<double(int32_t, int32_t)>& compare) {
    auto cmp = [&](int32_t x, int32_t y) { const double c = compare(x, y); return std::isnan(c) ? 0.0 : c; };
    const size_t n = a.size();
    if (n < 2) return a;
    if (n >= 64) { std::stable_sort(a.begin(), a.end(), [&](int32_t x, int32_t y) { return cmp(x, y) < 0; }); return a; }
    size_t run = 1;
    if (n >= 8) {
        run = 2;
        const bool desc = cmp(a[1], a[0]) < 0;
        int32_t prev = a[1];
        for (size_t i = 2; i < n; ++i) {
            const double o = cmp(a[i], prev);
            if ((desc && o >= 0) || (!desc && o < 0)) break;
            prev = a[i];
            ++run;
        }
        if (desc) std::reverse(a.begin(), a.begin() + static_cast<long>(run));
    }
    for (size_t start = run; start < n; ++start) {
        const int32_t pivot = a[start];
        size_t left = 0, right = start;
        while (left < right) {
            const size_t mid = left + ((right - left) >> 1);
            if (cmp(pivot, a[mid]) < 0) right = mid; else left = mid + 1;
        }
        for (size_t k = start; k > left; --k) a[k] = a[k - 1];
        a[left] = pivot;
    }
    return a;
}
}  // namespace

std::vector<int32_t> onePiece(const std::vector<int32_t>& lab, const std::vector<double>& mass, const std::vector<int32_t>& to, const std::vector<int32_t>& start, int parity) {
    const size_t n = lab.size();
    std::vector<int32_t> comp(n, -1), tileOf, stack;
    for (size_t s = 0; s < n; ++s) {
        if (comp[s] != -1) continue;
        const int32_t c = static_cast<int32_t>(tileOf.size());
        tileOf.push_back(lab[s]); comp[s] = c; stack.push_back(static_cast<int32_t>(s));
        while (!stack.empty()) {
            const size_t u = static_cast<size_t>(stack.back()); stack.pop_back();
            for (int32_t i = start[u]; i < start[u + 1]; ++i) { const size_t v = static_cast<size_t>(to[static_cast<size_t>(i)]); if (comp[v] == -1 && lab[v] == lab[u]) { comp[v] = c; stack.push_back(static_cast<int32_t>(v)); } }
        }
    }
    std::vector<double> cm(tileOf.size(), 0.0);
    for (size_t v = 0; v < n; ++v) cm[static_cast<size_t>(comp[v])] += mass[v];     // in vertex order, as numpy's bincount
    int32_t maxTile = 0;
    for (int32_t t : tileOf) maxTile = std::max(maxTile, t);
    std::vector<int32_t> best(static_cast<size_t>(maxTile) + 1, -1);
    for (size_t c = 0; c < tileOf.size(); ++c) { const size_t t = static_cast<size_t>(tileOf[c]); if (best[t] == -1 || cm[c] > cm[static_cast<size_t>(best[t])]) best[t] = static_cast<int32_t>(c); }
    std::vector<int32_t> out = lab;
    for (size_t v = 0; v < n; ++v) if ((lab[v] & 1) == parity && best[static_cast<size_t>(lab[v])] != comp[v]) out[v] = lab[v] ^ 1;
    return out;
}

SurfaceLadder surfaceLadder(const Mesh& mesh, const SurfaceLadderParams& params) {
    const int n = mesh.n;
    if (!(n > 0) || mesh.faces.size() < 3) throw std::invalid_argument("surfaceLadder: a mesh of " + std::to_string(n) + " vertices and " + std::to_string(mesh.faces.size() / 3) + " faces");
    const size_t N = static_cast<size_t>(n);
    SurfaceLadder out;
    out.nVertices = n;
    out.mass = vertexAreas(mesh);
    const std::vector<double>& mass = out.mass;
    double area = 0;
    for (double m : mass) area += m;
    out.area = area;
    const Graph g = meshGraph(mesh);
    const int cap = 1 << static_cast<int>(std::floor(std::log2(static_cast<double>(n))));
    const int maxTiles = std::min(params.maxTiles > 0 ? params.maxTiles : cap, cap);

    std::vector<double> dist(N);
    std::vector<char> inTile(N, 0);
    // Dijkstra from `sources`, restricted to the vertices `inTile` marks; `dist` is the scratch
    auto sweep = [&](const std::vector<int32_t>& sources) {
        std::fill(dist.begin(), dist.end(), kInf);
        using Item = std::pair<double, int32_t>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> h;
        for (int32_t s : sources) { dist[static_cast<size_t>(s)] = 0; h.push({0.0, s}); }
        while (!h.empty()) {
            const auto [d, u] = h.top(); h.pop();
            if (d > dist[static_cast<size_t>(u)]) continue;
            for (int32_t i = g.start[static_cast<size_t>(u)]; i < g.start[static_cast<size_t>(u) + 1]; ++i) {
                const size_t v = static_cast<size_t>(g.to[static_cast<size_t>(i)]);
                if (!inTile[v]) continue;
                const double nd = d + g.len[static_cast<size_t>(i)];
                if (nd < dist[v]) { dist[v] = nd; h.push({nd, static_cast<int32_t>(v)}); }
            }
        }
    };
    // the farthest pair inside a tile (two sweeps) and the distances from one end
    auto diameterOf = [&](const std::vector<int32_t>& tile, std::vector<double>* from) {
        for (int32_t v : tile) inTile[static_cast<size_t>(v)] = 1;
        int32_t a = tile[0];
        sweep({a});
        for (int32_t v : tile) if (dist[static_cast<size_t>(v)] < kInf && dist[static_cast<size_t>(v)] > dist[static_cast<size_t>(a)]) a = v;
        sweep({a});
        int32_t b = a;
        for (int32_t v : tile) if (dist[static_cast<size_t>(v)] < kInf && dist[static_cast<size_t>(v)] > dist[static_cast<size_t>(b)]) b = v;
        const double far = dist[static_cast<size_t>(b)];
        if (from) *from = dist;
        for (int32_t v : tile) inTile[static_cast<size_t>(v)] = 0;
        return std::isfinite(far) ? far : 0.0;
    };

    out.labels.push_back(std::vector<int32_t>(N, 0));
    for (int tiles = 1; tiles * 2 <= maxTiles; tiles *= 2) {
        const std::vector<int32_t>& prev = out.labels.back();
        std::vector<int32_t> next(N, 0);
        std::vector<std::vector<int32_t>> groups(static_cast<size_t>(tiles));
        for (int v = 0; v < n; ++v) groups[static_cast<size_t>(prev[static_cast<size_t>(v)])].push_back(v);
        for (int i = 0; i < tiles; ++i) {
            const std::vector<int32_t>& tile = groups[static_cast<size_t>(i)];
            if (tile.size() < 2) { for (int32_t v : tile) next[static_cast<size_t>(v)] = 2 * i; continue; }
            std::vector<double> da;
            diameterOf(tile, &da);
            // the far end, then the distance from it: the ordering is by (d_a - d_b) across the diameter
            int32_t b = tile[0];
            for (int32_t v : tile) if (da[static_cast<size_t>(v)] < kInf && da[static_cast<size_t>(v)] > da[static_cast<size_t>(b)]) b = v;
            for (int32_t v : tile) inTile[static_cast<size_t>(v)] = 1;
            sweep({b});
            const std::vector<double> db = dist;
            for (int32_t v : tile) inTile[static_cast<size_t>(v)] = 0;
            auto key = [&](int32_t x) { return da[static_cast<size_t>(x)] - db[static_cast<size_t>(x)]; };
            std::vector<int32_t> order = tile;
            bool finite = true;
            for (int32_t v : tile) if (!std::isfinite(key(v))) { finite = false; break; }
            if (finite) std::stable_sort(order.begin(), order.end(), [&](int32_t x, int32_t y) { return key(x) < key(y); });
            else order = v8Sort(order, [&](int32_t x, int32_t y) { return key(x) - key(y); });
            double total = 0;
            for (int32_t v : tile) total += mass[static_cast<size_t>(v)];
            double acc = 0;
            size_t cut = order.size() - 1;
            for (size_t j = 0; j < order.size(); ++j) { acc += mass[static_cast<size_t>(order[j])]; if (acc >= total / 2) { cut = j + 1; break; } }
            cut = std::min(std::max<size_t>(cut, 1), order.size() - 1);      // both halves keep a vertex
            for (size_t j = 0; j < order.size(); ++j) next[static_cast<size_t>(order[j])] = 2 * i + (j < cut ? 0 : 1);
        }
        next = onePiece(onePiece(next, mass, g.to, g.start, 0), mass, g.to, g.start, 1);      // every tile ONE PIECE
        // THE DELTA LEVEL: stop before a level that would leave a tile EMPTY
        std::vector<int> counts(static_cast<size_t>(tiles) * 2, 0);
        for (size_t v = 0; v < N; ++v) ++counts[static_cast<size_t>(next[v])];
        if (std::any_of(counts.begin(), counts.end(), [](int c) { return c == 0; })) break;
        out.labels.push_back(std::move(next));
    }

    const int finest = static_cast<int>(out.labels.size()) - 1;
    for (int level = 0; level <= finest; ++level) {
        const std::vector<int32_t>& lab = out.labels[static_cast<size_t>(level)];
        const int tiles = 1 << level;
        std::vector<std::vector<int32_t>> groups(static_cast<size_t>(tiles));
        for (int v = 0; v < n; ++v) groups[static_cast<size_t>(lab[static_cast<size_t>(v)])].push_back(v);
        std::vector<double> areas(static_cast<size_t>(tiles), 0.0);
        std::vector<int32_t> nv(static_cast<size_t>(tiles), 0);
        for (int t = 0; t < tiles; ++t) {
            for (int32_t v : groups[static_cast<size_t>(t)]) areas[static_cast<size_t>(t)] += mass[static_cast<size_t>(v)];
            nv[static_cast<size_t>(t)] = static_cast<int32_t>(groups[static_cast<size_t>(t)].size());
        }
        out.tileAreas.push_back(areas); out.tileVertices.push_back(nv);
        SurfaceLevel L;
        L.level = level; L.tiles = tiles;
        const double mean = area / tiles;
        double ss = 0;
        for (double a : areas) ss += (a - mean) * (a - mean);
        L.areaPerTile = {*std::min_element(areas.begin(), areas.end()), mean, *std::max_element(areas.begin(), areas.end()), std::sqrt(ss / tiles) / mean};
        L.verticesPerTile = {static_cast<double>(*std::min_element(nv.begin(), nv.end())), static_cast<double>(n) / tiles, static_cast<double>(*std::max_element(nv.begin(), nv.end())), 0};
        L.diameterMeasured = tiles <= params.exactDiameterUpTo;
        if (L.diameterMeasured) {
            double s = 0;
            for (const auto& t : groups) s += t.size() > 1 ? diameterOf(t, nullptr) : 0.0;
            L.diameter = s / tiles;
        } else {
            L.diameter = 2 * std::sqrt(mean / M_PI);
        }
        L.lambdaMax = weylLambda(tiles, area);
        L.lambdaMin = level == 0 ? 0 : weylLambda(tiles / 2.0, area);
        L.wavelength = wavelengthOf(L.lambdaMax);
        L.diameterInWavelengths = level > 0 ? L.diameter / wavelengthOf(weylLambda(std::max(1.0, tiles / 4.0), area)) : 0;
        L.tau = 1 / L.lambdaMax;
        L.modes = tiles;
        if (!params.eigenvalues.empty() && static_cast<size_t>(tiles) <= params.eigenvalues.size()) L.lambdaStored = params.eigenvalues[static_cast<size_t>(tiles) - 1];
        L.finestTiles = 1 << (finest - level);
        L.resolutionSpace = L.diameter; L.resolutionLambda = L.lambdaMax - L.lambdaMin;
        out.levels.push_back(L);
    }
    return out;
}

// ── the tree ─────────────────────────────────────────────────────────────────────────────

std::vector<int32_t> Tree::at(int level) const {
    if (level < 0 || level > depth) throw std::invalid_argument("tree: level " + std::to_string(level) + " outside 0.." + std::to_string(depth));
    std::vector<int32_t> out(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) out[i] = codes[i] >= 0 ? (codes[i] >> (depth - level)) : -1;
    return out;
}

std::vector<std::pair<int, int>> pieceRuns(const int32_t* faces, int nF, int nV) {
    const std::vector<int> piece = pieces(faces, nF, nV);
    std::vector<std::pair<int, int>> runs;
    if (piece.empty()) { runs.push_back({0, nV}); return runs; }
    const int np = *std::max_element(piece.begin(), piece.end()) + 1;
    std::vector<int> first(static_cast<size_t>(np), -1), last(static_cast<size_t>(np), -1), cnt(static_cast<size_t>(np), 0);
    for (int v = 0; v < nV; ++v) { const size_t p = static_cast<size_t>(piece[static_cast<size_t>(v)]); if (first[p] < 0) first[p] = v; last[p] = v; ++cnt[p]; }
    for (int p = 0; p < np; ++p) {
        const size_t q = static_cast<size_t>(p);
        if (last[q] - first[q] + 1 != cnt[q]) throw std::invalid_argument("tree: connected piece " + std::to_string(p) + " is not one contiguous block of vertices");
        runs.push_back({first[q], cnt[q]});
    }
    return runs;
}

Tree subjectTree(const double* positions, int nV, const int32_t* faces, int nF, int depth) {
    Tree t;
    t.runs = pieceRuns(faces, nF, nV);
    std::vector<SurfaceLadder> ladders;
    for (const auto& [s, len] : t.runs) {
        ladders.push_back(surfaceLadder(componentMesh(positions, faces, nF, s, len)));
        t.pieceArea.push_back(ladders.back().area);
    }
    t.finest = static_cast<int>(ladders[0].labels.size()) - 1;
    for (const auto& l : ladders) t.finest = std::min(t.finest, static_cast<int>(l.labels.size()) - 1);
    const int d = depth < 0 ? t.finest : depth;
    if (d < 0 || d > t.finest) throw std::invalid_argument("tree: depth " + std::to_string(d) + " outside 0.." + std::to_string(t.finest) + " (the finest level every piece reaches)");
    t.depth = d;
    t.slots = std::max<int>(2, static_cast<int>(t.runs.size()));
    t.diameter.assign(static_cast<size_t>(d) + 1, 0.0);
    for (int l = 0; l <= d; ++l) {                                      // the measured tile diameter of level l: the mean over the pieces
        for (const auto& lad : ladders) t.diameter[static_cast<size_t>(l)] += lad.levels[static_cast<size_t>(l)].diameter;
        t.diameter[static_cast<size_t>(l)] /= static_cast<double>(ladders.size());
    }
    t.codes.assign(static_cast<size_t>(nV), -1);
    for (size_t h = 0; h < t.runs.size(); ++h)
        for (int i = 0; i < t.runs[h].second; ++i) t.codes[static_cast<size_t>(t.runs[h].first + i)] = static_cast<int32_t>(h * (static_cast<size_t>(1) << d)) + ladders[h].labels[static_cast<size_t>(d)][static_cast<size_t>(i)];
    return t;
}

Tree groupTreeFor(const double* tplSphere, const int32_t* tplCodes, const std::vector<std::pair<int, int>>& tplRuns, int finest,
                  const double* subSphere, int nV, const std::vector<std::pair<int, int>>& subRuns, int depth, double* matchMedian, double* matchMax) {
    if (depth < 0 || depth > finest) throw std::invalid_argument("groupTreeFor: depth " + std::to_string(depth) + " is outside 0.." + std::to_string(finest));
    if (tplRuns.size() != subRuns.size()) throw std::invalid_argument("groupTreeFor: the subject has " + std::to_string(subRuns.size()) + " pieces, the template " + std::to_string(tplRuns.size()));
    Tree t;
    t.depth = depth; t.finest = finest; t.slots = 2;
    t.codes.assign(static_cast<std::size_t>(nV), -1);
    t.runs = subRuns;
    std::vector<double> dist(static_cast<std::size_t>(nV), std::nan(""));
    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    for (std::size_t h = 0; h < subRuns.size(); ++h) {
        const auto [ts, tn] = tplRuns[h];
        const auto [ss, sn] = subRuns[h];
        if (tn <= 0 || sn < 0 || ss < 0 || ss + sn > nV) throw std::invalid_argument("groupTreeFor: a piece lies outside its sphere");
        RowMat T = Eigen::Map<const RowMat>(tplSphere + 3 * static_cast<std::size_t>(ts), tn, 3);
        for (int i = 0; i < tn; ++i) T.row(i).normalize();
        // the nearest on the unit sphere is the largest dot product; blocks keep the product small
        const int block = 512;
        for (int b0 = 0; b0 < sn; b0 += block) {
            const int nb = std::min(block, sn - b0);
            RowMat Q = Eigen::Map<const RowMat>(subSphere + 3 * (static_cast<std::size_t>(ss) + b0), nb, 3);
            for (int i = 0; i < nb; ++i) Q.row(i).normalize();
            const Eigen::MatrixXd D = Q * T.transpose();
            for (int i = 0; i < nb; ++i) {
                Eigen::Index j;
                const double best = D.row(i).maxCoeff(&j);
                t.codes[static_cast<std::size_t>(ss + b0 + i)] = tplCodes[ts + j] >> (finest - depth);
                dist[static_cast<std::size_t>(ss + b0 + i)] = (Q.row(i) - T.row(j)).norm();
                (void)best;
            }
        }
    }
    if (matchMedian || matchMax) {
        std::vector<double> d;
        for (double x : dist) if (std::isfinite(x)) d.push_back(x);
        if (matchMax) *matchMax = d.empty() ? std::nan("") : *std::max_element(d.begin(), d.end());
        if (matchMedian) {
            if (d.empty()) *matchMedian = std::nan("");
            else { std::sort(d.begin(), d.end()); const std::size_t n = d.size(); *matchMedian = n % 2 ? d[n / 2] : 0.5 * (d[n / 2 - 1] + d[n / 2]); }
        }
    }
    t.pieceArea.assign(subRuns.size(), 0.0);
    return t;
}

}  // namespace nxr::select::bisection
