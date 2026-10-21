"""Gap filler: grow/shrink every skew-symmetric record by best insertion/deletion (ops.py), then make
even lengths by trimming or appending one end element. Keeps the best sequence per length.
usage: python3 fill.py out.tsv steps seeds.tsv [seeds.tsv ...]"""
import sys
import numpy as np
from ops import step, energies
from stretch import is_skew

out, nsteps, srcs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
best = {}


def keep(x, e):
    L = len(x)
    if L not in best or e < best[L][0]: best[L] = (int(e), "".join("1" if v > 0 else "0" for v in x))


seeds = []
for f in srcs:
    for l in open(f):
        if l.strip() and not l.startswith("#"):
            b = l.split()[3]
            if len(b) % 2 and is_skew(b): seeds.append(np.array([1 if ch == "1" else -1 for ch in b]))
for s in seeds:
    for mode in ("grow", "shrink"):
        x = s
        for _ in range(nsteps):
            e, x = step(x, mode)[0]; keep(x, e)
# even lengths from every odd candidate: drop or add one end element
for L in sorted(k for k in best if k % 2):
    x = np.array([1 if ch == "1" else -1 for ch in best[L][1]])
    cands = [x[1:], x[:-1]] + [np.concatenate([[v], x]) for v in (1, -1)] + [np.concatenate([x, [v]]) for v in (1, -1)]
    for n in (L - 1, L + 1):
        X = np.array([c for c in cands if len(c) == n]); e = energies(X); i = int(e.argmin()); keep(X[i], e[i])
with open(out, "w") as f:
    for L in sorted(best):
        e, b = best[L]; f.write(f"{L} {e} {L * L / (2 * e):.4f} {b}\n")
print(len(best), "lengths written")
