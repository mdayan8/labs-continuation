"""Length-changing operators on skew-symmetric sequences that keep skew-symmetry:
insert (L -> L+2) or delete (L -> L-2) one free variable at the best position.
usage: python3 ops.py SEQ01 {grow|shrink}   -> prints best new sequence and its F"""
import sys
import numpy as np


def complete(half):  # half = s[0..c] as +-1 array -> full skew-symmetric sequence
    c = len(half) - 1; j = np.arange(1, c + 1)
    tail = np.where(j % 2 == 1, -1, 1) * half[c - j]
    return np.concatenate([half, tail])


def energies(X):
    n = X.shape[1]; m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(X, m, axis=1)
    C = np.rint(np.fft.irfft(f * np.conj(f), m, axis=1)[:, 1:n]).astype(np.int64)
    return (C * C).sum(1)


def candidates(s, mode):
    half = s[: len(s) // 2 + 1]
    out = []
    if mode == "grow":
        for i in range(len(half) + 1):
            for x in (1, -1):
                out.append(complete(np.insert(half, i, x)))
    else:
        for i in range(len(half) - 1):  # keep at least the centre
            out.append(complete(np.delete(half, i)))
    return np.array(out)


def step(s, mode, keep=1):
    X = candidates(s, mode); e = energies(X); idx = np.argsort(e)[:keep]
    return [(int(e[i]), X[i]) for i in idx]


if __name__ == "__main__":
    s = np.array([1 if ch == "1" else -1 for ch in sys.argv[1]])
    e, t = step(s, sys.argv[2])[0]
    L = len(t); print(L, e, f"{L * L / (2 * e):.4f}", "".join("1" if v > 0 else "0" for v in t))
