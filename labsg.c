// labsg.c - general (unconstrained) LABS search: self-avoiding walks + iterated local search,
// started from a given seed sequence (e.g. a Legendre construction).
// usage: ./labsg SEQ01 seconds rngseed [walk_mult=8] [perturb_frac=0.02]
// prints "L E F seq t=..." on every improvement over the seed.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static uint64_t rng;
static inline uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int L; static int32_t *s, *C;
static uint64_t *zob, *tab; static uint32_t *gen, curgen; static uint64_t tmask;

static int64_t energy(void) {
    int64_t e = 0;
    for (int k = 1; k < L; k++) { int32_t ck = 0; for (int i = 0; i + k < L; i++) ck += s[i] * s[i + k]; C[k] = ck; e += (int64_t)ck * ck; }
    return e;
}
static inline int32_t delta(int p) {   // s padded with zeros, so no bounds checks
    int32_t de = 0, sp = s[p];
    for (int k = 1; k < L; k++) { int32_t d = sp * (s[p + k] + s[p - k]); de += d * (d - C[k]); }
    return 4 * de;
}
static void flip(int p) {
    for (int k = 1; k < L; k++) C[k] -= 2 * s[p] * (s[p + k] + s[p - k]);
    s[p] = -s[p];
}
static int seen(uint64_t h) { uint64_t i = h & tmask; while (gen[i] == curgen) { if (tab[i] == h) return 1; i = (i + 1) & tmask; } return 0; }
static void mark(uint64_t h) { uint64_t i = h & tmask; while (gen[i] == curgen) { if (tab[i] == h) return; i = (i + 1) & tmask; } gen[i] = curgen; tab[i] = h; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s SEQ01 seconds rngseed [walk_mult] [perturb_frac]\n", argv[0]); return 1; }
    const char *seed = argv[1]; L = strlen(seed); double secs = atof(argv[2]);
    rng = strtoull(argv[3], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    double wm = argc > 4 ? atof(argv[4]) : 8.0, pf = argc > 5 ? atof(argv[5]) : 0.02;
    int32_t *buf = calloc(3 * L, sizeof(int32_t)); s = buf + L; C = calloc(L, sizeof(int32_t));
    int32_t *elite = malloc(L * sizeof(int32_t));
    zob = malloc(L * sizeof(uint64_t)); for (int i = 0; i < L; i++) zob[i] = rnd();
    long W = (long)(wm * L); int tb = 1; while ((1L << tb) < 4 * W) tb++;
    tab = malloc(sizeof(uint64_t) << tb); gen = calloc(1L << tb, sizeof(uint32_t)); tmask = (1ULL << tb) - 1;
    for (int i = 0; i < L; i++) elite[i] = seed[i] == '1' ? 1 : -1;
    memcpy(s, elite, L * sizeof(int32_t));
    int64_t best = energy(); int pk = (int)(pf * L); if (pk < 1) pk = 1;
    fprintf(stderr, "L=%d seedE=%lld F=%.4f\n", L, (long long)best, (double)L * L / (2.0 * best));
    double t0 = now(); long steps = 0, walks = 0;
    while (now() - t0 < secs) {
        memcpy(s, elite, L * sizeof(int32_t));
        if (walks) for (int r = 0; r < pk; r++) { int v = rnd() % L; s[v] = -s[v]; }
        int64_t e = energy(); uint64_t h = 0;
        for (int i = 0; i < L; i++) if (s[i] > 0) h ^= zob[i];
        curgen++; walks++; mark(h);
        for (long w = 0; w < W; w++) {
            int bv = -1; int32_t bd = INT32_MAX; int ties = 0;
            for (int v = 0; v < L; v++) {
                int32_t d = delta(v);
                if (d > bd || seen(h ^ zob[v])) continue;
                if (d < bd) { bd = d; bv = v; ties = 1; } else if (rnd() % ++ties == 0) bv = v;
            }
            if (bv < 0) break;
            flip(bv); e += bd; h ^= zob[bv]; mark(h); steps++;
            if (e < best) {
                best = e; memcpy(elite, s, L * sizeof(int32_t));
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
