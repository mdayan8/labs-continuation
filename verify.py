"""Independently recompute energy and merit factor of binary sequences.
usage: python3 verify.py records.tsv   (columns: L E F ... seq ...; seq = the 0/1 column of length L; '#' skipped)
Pure stdlib, no trust in the solver: E = sum_{k>=1} C_k^2, F = L^2 / (2E)."""
import sys


def energy(seq):
    s = [1 if ch == "1" else -1 for ch in seq]
    n = len(s)
    return sum(sum(s[i] * s[i + k] for i in range(n - k)) ** 2 for k in range(1, n))


def check(path):
    bad = 0
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split(); L, E = f[0], f[1]
        seq = next(t for t in f if len(t) == int(L) and set(t) <= {"0", "1"})
        e = energy(seq)
        ok = len(seq) == int(L) and e == int(E) and set(seq) <= {"0", "1"}
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} L={L} E={e} F={len(seq)**2 / (2 * e):.4f}")
    return bad


if __name__ == "__main__":
    assert energy("1111100110101") == 6  # Barker 13, the known optimum E=6
    sys.exit(1 if check(sys.argv[1]) else 0)
