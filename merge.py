"""Merge candidate files: best verified sequence per length, then derive neighbouring lengths by
dropping/adding one end element (repeated until no length improves).
usage: python3 merge.py out.tsv cand1.tsv [cand2.tsv ...]"""
import sys
import numpy as np
from ops import energies
from verify import energy

best = {}
def keep(b, e):
    L = len(b)
    if L not in best or e < best[L][0]: best[L] = (int(e), b); return True
    return False

for f in sys.argv[2:]:
    for l in open(f):
        if l.strip() and not l.startswith("#"):
            b = l.split()[3]; keep(b, energy(b))

lo, hi = min(best), max(best)   # propagate only inside the input range
changed = True
while changed:
    changed = False
    for L in sorted(best):
        x = np.array([1 if ch == "1" else -1 for ch in best[L][1]])
        cands = [x[1:], x[:-1]] + [np.concatenate([[v], x]) for v in (1, -1)] + [np.concatenate([x, [v]]) for v in (1, -1)]
        for n in (L - 1, L + 1):
            if not lo <= n <= hi: continue
            X = np.array([c for c in cands if len(c) == n]); e = energies(X); i = int(e.argmin())
            changed |= keep("".join("1" if v > 0 else "0" for v in X[i]), e[i])

with open(sys.argv[1], "w") as f:
    for L in sorted(best):
        e, b = best[L]; assert energy(b) == e
        f.write(f"{L} {e} {L * L / (2 * e):.4f} {b}\n")
print(len(best), "lengths")
