// glabs.cu - GPU continuation search for skew-symmetric low-autocorrelation binary sequences.
// One CUDA block = one tabu walk (iterated local search with per-block elites); threads evaluate
// all single (skew-pair) flips in parallel in O(L) each. Lengths are grown two at a time
// (one element at each end, which preserves skew-symmetry) from a seed sequence.
//
// usage: ./glabs SEED01 Lmax secs_per_len [blocks=160] [steps=4000] [pert=0.04] [rng=1] [Lstep=2] [Pfix=0]
//   SEED01: odd-length skew-symmetric 0/1 string ('1' = +1). If Lmax == len(SEED), just optimises it.
// output (stdout): "L E F seq" once per length (best found), stderr: progress.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

#define LMAX 2048
#define MMAX (LMAX / 2 + 1)
#define NT 256

__device__ __forceinline__ uint32_t xr(uint32_t &x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }

// elite: blocks x MMAX free-variable values (+1/-1 as int8); eliteE: energies
__global__ void walk(int L, int steps, int pk, int8_t *elite, long long *eliteE, uint32_t seed, int P) {
    __shared__ int8_t sb[3 * LMAX];
    __shared__ int C[LMAX];
    __shared__ int tabu[MMAX];
    __shared__ unsigned long long bestkey;
    __shared__ long long e, be;
    __shared__ int8_t bests[MMAX];
    int8_t *s = sb + LMAX;
    const int M = (L + 1) / 2, c = M - 1, b = blockIdx.x, t = threadIdx.x;
    uint32_t r = seed * 2654435761u ^ (b * 40503u + t * 9973u + 1u); xr(r); xr(r);
    for (int i = t; i < 3 * LMAX; i += NT) sb[i] = 0;
    __syncthreads();
    for (int i = t; i < M; i += NT) { s[i] = elite[(size_t)b * MMAX + i]; tabu[i] = 0; }
    __syncthreads();
    if (t == 0) for (int j = 0; j < pk; j++) { int v = P + xr(r) % (M - P); s[v] = -s[v]; }   // perturb free vars
    __syncthreads();
    for (int j = 1 + t; j <= c; j += NT) s[c + j] = (j & 1) ? -s[c - j] : s[c - j];
    __syncthreads();
    for (int k = t; k < L; k += NT) { int a = 0; for (int i = 0; i + k < L; i++) a += s[i] * s[i + k]; C[k] = a; }
    __syncthreads();
    if (t == 0) { long long a = 0; for (int k = 1; k < L; k++) a += (long long)C[k] * C[k]; e = a; be = eliteE[b]; }
    __syncthreads();
    const int tmin = M / 10 + 1, tmax = M / 5 + 2;
    for (int it = 1; it <= steps; it++) {
        if (t == 0) bestkey = ~0ull;
        __syncthreads();
        unsigned long long mykey = ~0ull;
        for (int v = P + t; v < M; v += NT) {
            int p = v, q = 2 * c - v, sp = s[p], sq = s[q]; int acc = 0;
            if (p == q) {
                for (int k = 2; k < L; k += 2) { int d = sp * (s[p + k] + s[p - k]); acc += d * (d - C[k]); }
            } else {
                int dist = q - p;
                for (int k = 2; k < L; k += 2) {
                    int d = sp * (s[p + k] + s[p - k]) + sq * (s[q + k] + s[q - k]);
                    if (k == dist) d -= 2 * sp * sq;
                    acc += d * (d - C[k]);
                }
            }
            long long de = 4LL * acc;
            bool ok = tabu[v] < it || e + de < be;          // tabu with aspiration
            if (ok) {
                unsigned long long key = ((unsigned long long)(de + (1LL << 40)) << 22) | ((xr(r) & 0x3FF) << 12) | v;
                if (key < mykey) mykey = key;
            }
        }
        if (mykey != ~0ull) atomicMin(&bestkey, mykey);
        __syncthreads();
        if (bestkey == ~0ull) break;
        int v = bestkey & 0xFFF; long long de = (long long)(bestkey >> 22) - (1LL << 40);
        int p = v, q = 2 * c - v;
        for (int k = 2 + 2 * t; k < L; k += 2 * NT) {
            int d = s[p] * (s[p + k] + s[p - k]);
            if (p != q) { d += s[q] * (s[q + k] + s[q - k]); if (k == q - p) d -= 2 * s[p] * s[q]; }
            C[k] -= 2 * d;
        }
        __syncthreads();
        if (t == 0) {
            s[p] = -s[p]; if (p != q) s[q] = -s[q];
            tabu[v] = it + tmin + xr(r) % (tmax - tmin + 1);
            e += de;
        }
        __syncthreads();
        if (e < be) {
            for (int i = t; i < M; i += NT) bests[i] = s[i];
            __syncthreads();
            if (t == 0) be = e;
            __syncthreads();
        }
    }
    __syncthreads();
    if (be < eliteE[b]) {   // be only drops below eliteE if this walk improved it
        for (int i = t; i < M; i += NT) elite[(size_t)b * MMAX + i] = bests[i];
        if (t == 0) eliteE[b] = be;
    }
}

static long long energy(const std::vector<int> &s) {
    long long e = 0; int L = s.size();
    for (int k = 1; k < L; k++) { long long a = 0; for (int i = 0; i + k < L; i++) a += s[i] * s[i + k]; e += a * a; }
    return e;
}
static std::vector<int> skew_complete(std::vector<int> half) {  // half = s[0..c]
    int c = half.size() - 1; std::vector<int> s(2 * c + 1);
    for (int i = 0; i <= c; i++) s[i] = half[i];
    for (int j = 1; j <= c; j++) s[c + j] = (j & 1) ? -s[c - j] : s[c - j];
    return s;
}
static std::vector<int> extend(const std::vector<int> &s) {  // L -> L+2, preserving skew-symmetry
    int c = s.size() / 2; std::vector<int> best; long long be = -1;
    for (int x = -1; x <= 1; x += 2) {
        std::vector<int> half(c + 2); half[0] = x; for (int i = 0; i <= c; i++) half[i + 1] = s[i];
        auto t = skew_complete(half); long long e = energy(t);
        if (be < 0 || e < be) { be = e; best = t; }
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s SEED01 Lmax secs [blocks] [steps] [pert] [rng] [Lstep]\n", argv[0]); return 1; }
    std::string seed = argv[1]; int Lmax = atoi(argv[2]); double secs = atof(argv[3]);
    int B = argc > 4 ? atoi(argv[4]) : 160, steps = argc > 5 ? atoi(argv[5]) : 4000;
    double pf = argc > 6 ? atof(argv[6]) : 0.04; uint32_t rng = argc > 7 ? atoi(argv[7]) : 1;
    int Lstep = argc > 8 ? atoi(argv[8]) : 2;
    int Pfix = argc > 9 ? atoi(argv[9]) : 0;   // number of frozen leading positions (run-structured head)
    std::vector<int> cur(seed.size()); for (size_t i = 0; i < seed.size(); i++) cur[i] = seed[i] == '1' ? 1 : -1;
    { std::vector<int> h(cur.begin(), cur.begin() + (cur.size() + 1) / 2); auto sk = skew_complete(h);
      if (sk != cur) { fprintf(stderr, "seed not skew-symmetric: using its skew projection\n"); cur = sk; } }
    int8_t *d_el; long long *d_E;
    cudaMalloc(&d_el, (size_t)B * MMAX); cudaMalloc(&d_E, B * sizeof(long long));
    std::vector<int8_t> h_el((size_t)B * MMAX); std::vector<long long> h_E(B);
    bool first = true;
    while (true) {
        if (!first) for (int g = 0; g < Lstep / 2; g++) cur = extend(cur);
        first = false;
        int L = cur.size(), M = (L + 1) / 2; if (L > Lmax || L > LMAX - 2) break;
        long long E0 = energy(cur);
        for (int b = 0; b < B; b++) { for (int i = 0; i < M; i++) h_el[(size_t)b * MMAX + i] = cur[i]; h_E[b] = E0; }
        cudaMemcpy(d_el, h_el.data(), h_el.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_E, h_E.data(), B * sizeof(long long), cudaMemcpyHostToDevice);
        auto t0 = std::chrono::steady_clock::now(); long long gbest = E0; int round = 0;
        int pk = (int)(pf * M); if (pk < 1) pk = 1;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
            walk<<<B, NT>>>(L, steps, pk, d_el, d_E, rng * 7919u + L * 131u + round, Pfix);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(err)); return 1; }
            round++;
            cudaMemcpy(h_E.data(), d_E, B * sizeof(long long), cudaMemcpyDeviceToHost);
            int bi = 0; for (int b = 1; b < B; b++) if (h_E[b] < h_E[bi]) bi = b;
            if (h_E[bi] < gbest) gbest = h_E[bi];
            if (round % 8 == 0) {   // spread: worst quarter of blocks restart from the global best
                cudaMemcpy(h_el.data(), d_el, h_el.size(), cudaMemcpyDeviceToHost);
                std::vector<int> idx(B); for (int b = 0; b < B; b++) idx[b] = b;
                std::sort(idx.begin(), idx.end(), [&](int a, int z) { return h_E[a] > h_E[z]; });
                for (int j = 0; j < B / 4; j++) { int w = idx[j]; memcpy(&h_el[(size_t)w * MMAX], &h_el[(size_t)bi * MMAX], MMAX); h_E[w] = h_E[bi]; }
                cudaMemcpy(d_el, h_el.data(), h_el.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_E, h_E.data(), B * sizeof(long long), cudaMemcpyHostToDevice);
            }
        }
        cudaMemcpy(h_el.data(), d_el, h_el.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_E.data(), d_E, B * sizeof(long long), cudaMemcpyDeviceToHost);
        int bi = 0; for (int b = 1; b < B; b++) if (h_E[b] < h_E[bi]) bi = b;
        std::vector<int> half(M); for (int i = 0; i < M; i++) half[i] = h_el[(size_t)bi * MMAX + i];
        std::vector<int> best = skew_complete(half); long long eb = energy(best);
        if (eb != h_E[bi]) { fprintf(stderr, "energy mismatch %lld vs %lld\n", eb, h_E[bi]); return 2; }
        if (eb < E0) cur = best;
        long long ef = energy(cur);
        printf("%d %lld %.4f ", L, ef, (double)L * L / (2.0 * ef));
        for (int v : cur) putchar(v > 0 ? '1' : '0');
        printf("\n"); fflush(stdout);
        fprintf(stderr, "L=%d E0=%lld -> %lld F=%.4f rounds=%d\n", L, E0, ef, (double)L * L / (2.0 * ef), round);
    }
    return 0;
}
