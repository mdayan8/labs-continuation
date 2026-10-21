"""Best rotated + periodically extended/truncated Legendre sequence for each length L
(the strongest known algebraic construction, asymptotic F = 6.342...).
usage: python3 construct.py Lmin Lmax > seeds.tsv     -> L E F seq p r"""
import sys
import numpy as np


def primes(n):
    s = np.ones(n + 1, bool); s[:2] = False
    for i in range(2, int(n ** .5) + 1):
        if s[i]: s[i * i::i] = False
    return np.nonzero(s)[0]


def energies(X):  # rows are +-1 sequences; E = sum_{k>=1} C_k^2 via FFT
    n = X.shape[1]; m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(X, m, axis=1)
    C = np.rint(np.fft.irfft(f * np.conj(f), m, axis=1)[:, 1:n]).astype(np.int64)
    return (C * C).sum(1)


def legendre(p):
    chi = -np.ones(p, np.int64); chi[(np.arange(1, p) ** 2) % p] = 1; chi[0] = 1
    return chi


def best(L, P):
    out = (None, 1 << 62)
    for p in P[(P >= L / 1.25) & (P <= L * 1.1)]:
        chi = legendre(int(p))
        idx = (np.arange(p)[:, None] + np.arange(L)[None, :]) % p  # all rotations, extended to L
        X = chi[idx]
        e = energies(X); r = int(e.argmin())
        if e[r] < out[1]: out = ((int(p), r, X[r]), int(e[r]))
    (p, r, x), e = out
    return e, x, p, r


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    P = primes(2 * b)
    assert energies(np.array([[1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]]))[0] == 6
    for L in range(a, b + 1):
        e, x, p, r = best(L, P)
        print(L, e, f"{L * L / (2 * e):.4f}", "".join("1" if v > 0 else "0" for v in x), p, r, flush=True)
