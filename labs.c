// labs.c - self-avoiding-walk search for skew-symmetric low-autocorrelation binary sequences.
// usage: ./labs L seconds seed [walk_mult]     (L odd)
// prints "L E F seq" (seq as 0/1, 1 = +1) on every improvement to stdout.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static uint64_t rng;
static inline uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int L, M, c;          // length, free vars, center index
static int *s;               // padded: s[-L..2L) valid, zeros outside [0,L)
static int64_t *C;           // C[k], k = 0..L-1
static uint64_t *zob, *tab;  // zobrist keys per var; visited table
static uint32_t *gen, curgen; static uint64_t tmask;

static int64_t energy(void) {
    int64_t e = 0;
    for (int k = 1; k < L; k++) { int64_t ck = 0; for (int i = 0; i + k < L; i++) ck += s[i] * s[i + k]; C[k] = ck; e += ck * ck; }
    return e;
}

// delta energy of flipping var v (positions p and q = 2c - p)
static inline int64_t delta(int v) {
    int p = v, q = 2 * c - v; int64_t de = 0;
    int sp = s[p], sq = s[q];
    if (p == q) {
        for (int k = 2; k < L; k += 2) { int64_t d = sp * (s[p + k] + s[p - k]); de += 4 * d * (d - C[k]); }
    } else {
        int dist = q - p;
        for (int k = 2; k < L; k += 2) {
            int64_t d = sp * (s[p + k] + s[p - k]) + sq * (s[q + k] + s[q - k]);
            if (k == dist) d -= 2 * sp * sq;
            de += 4 * d * (d - C[k]);
        }
    }
    return de;
}

static void flip(int v) {
    int p = v, q = 2 * c - v;
    if (p == q) {
        for (int k = 2; k < L; k += 2) C[k] -= 2 * s[p] * (s[p + k] + s[p - k]);
        s[p] = -s[p];
    } else {
        int dist = q - p;
        for (int k = 2; k < L; k += 2) {
            int64_t d = s[p] * (s[p + k] + s[p - k]) + s[q] * (s[q + k] + s[q - k]);
            if (k == dist) d -= 2 * s[p] * s[q];
            C[k] -= 2 * d;
        }
        s[p] = -s[p]; s[q] = -s[q];
    }
}

static int seen(uint64_t h) {  // returns 1 if h visited this walk
    uint64_t i = h & tmask;
    while (gen[i] == curgen) { if (tab[i] == h) return 1; i = (i + 1) & tmask; }
    return 0;
}
static void mark(uint64_t h) {
    uint64_t i = h & tmask;
    while (gen[i] == curgen) { if (tab[i] == h) return; i = (i + 1) & tmask; }
    gen[i] = curgen; tab[i] = h;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s L seconds seed [walk_mult]\n", argv[0]); return 1; }
    L = atoi(argv[1]); double secs = atof(argv[2]); rng = strtoull(argv[3], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    double wm = argc > 4 ? atof(argv[4]) : 8.0;
    double pf = argc > 5 ? atof(argv[5]) : 0.0;   // perturbation = pf * M flips from walk-best; 0 = random restarts
    int head[4096], P = 0;   // optional frozen prefix: runs "28,13,12,8,6" -> +1 x28, -1 x13, ...
    if (argc > 6 && argv[6][0] != '-') { int sg = 1; for (char *t = strtok(argv[6], ","); t; t = strtok(0, ","), sg = -sg) for (int r = atoi(t); r--;) head[P++] = sg; }
    if (L < 5 || !(L & 1)) { fprintf(stderr, "L must be odd >= 5\n"); return 1; }
    M = (L + 1) / 2; c = M - 1;
    int *buf = calloc(4 * L, sizeof(int)); s = buf + L;
    C = calloc(L, sizeof(int64_t)); zob = malloc(M * sizeof(uint64_t));
    for (int i = 0; i < M; i++) zob[i] = rnd();
    long W = (long)(wm * M); int tb = 1; while ((1L << tb) < 4 * W) tb++;
    tab = malloc(sizeof(uint64_t) << tb); gen = calloc(1L << tb, sizeof(uint32_t)); tmask = (1ULL << tb) - 1;
    int64_t best = INT64_MAX; double t0 = now(); long steps = 0, walks = 0;
    int *elite = malloc(L * sizeof(int)); int pk = (int)(pf * M); int64_t ebest = INT64_MAX;
    const char *seed = argc > 7 ? argv[7] : 0;   // optional skew-symmetric seed sequence (0/1 string of length L)
    if (seed) { if ((int)strlen(seed) != L) { fprintf(stderr, "seed length != L\n"); return 1; }
        for (int i = 0; i < L; i++) elite[i] = seed[i] == '1' ? 1 : -1; memcpy(s, elite, L * sizeof(int)); ebest = best = energy(); }
    while (now() - t0 < secs) {
        // start: random first walk, then perturbed elite (iterated local search)
        if (!(seed && walks == 0) && (walks == 0 || pk <= 0)) { for (int i = 0; i <= c; i++) s[i] = (rnd() & 1) ? 1 : -1; for (int i = 0; i < P; i++) s[i] = head[i]; }
        else { memcpy(s, elite, L * sizeof(int)); if (walks) for (int r = 0; r < pk; r++) { int v = P + rnd() % (M - P); s[v] = -s[v]; } }
        for (int j = 1; j <= c; j++) s[c + j] = (j & 1 ? -1 : 1) * s[c - j];
        int64_t e = energy(); uint64_t h = 0;
        for (int i = 0; i <= c; i++) if (s[i] > 0) h ^= zob[i];
        curgen++; walks++; mark(h);
        for (long w = 0; w < W; w++) {
            int bv = -1; int64_t bd = INT64_MAX; int ties = 0;
            for (int v = P; v < M; v++) {
                int64_t d = delta(v);
                if (d > bd) continue;
                if (seen(h ^ zob[v])) continue;
                if (d < bd) { bd = d; bv = v; ties = 1; }
                else if (rnd() % ++ties == 0) bv = v;   // reservoir tie-break
            }
            if (bv < 0) break;
            flip(bv); e += bd; h ^= zob[bv]; mark(h); steps++;
            if (e < ebest) { ebest = e; memcpy(elite, s, L * sizeof(int)); }
            if (e < best) {
                best = e;
                printf("%d %lld %.4f ", L, (long long)e, (double)L * L / (2.0 * e));
                for (int i = 0; i < L; i++) putchar(s[i] > 0 ? '1' : '0');
                printf(" t=%.1f\n", now() - t0); fflush(stdout);
            }
        }
    }
    fprintf(stderr, "L=%d best=%lld F=%.4f steps=%ld walks=%ld steps/s=%.0f\n", L, (long long)best,
            (double)L * L / (2.0 * best), steps, walks, steps / (now() - t0));
    return 0;
}
