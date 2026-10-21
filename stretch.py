"""Stretch a skew-symmetric record sequence to a new odd length by scaling its run lengths.
usage: python3 stretch.py known.tsv Lsrc Ldst  -> prints the stretched skew-symmetric sequence"""
import sys
from verify import energy


def is_skew(b):
    n = len(b); c = n // 2
    return n % 2 == 1 and all((b[c + j] == b[c - j]) == (j % 2 == 0) for j in range(1, c + 1))


def stretch(b, n2):
    c, c2 = len(b) // 2, n2 // 2
    half = b[:c + 1]
    runs = []
    for ch in half:
        if runs and runs[-1][0] == ch: runs[-1][1] += 1
        else: runs.append([ch, 1])
    lam, acc, out = (c2 + 1) / (c + 1), 0.0, []
    for ch, r in runs:  # error-diffused rounding keeps the total exact
        acc += r * lam; k = max(1, round(acc) - len(out)); out += [ch] * k
    out = (out + [out[-1]] * (c2 + 1))[:c2 + 1]
    s = out + ["0"] * c2
    for j in range(1, c2 + 1):
        s[c2 + j] = s[c2 - j] if j % 2 == 0 else "10"[int(s[c2 - j])]
    return "".join(s)


if __name__ == "__main__":
    rec = {int(l.split()[0]): l.split()[3] for l in open(sys.argv[1])}
    b = rec[int(sys.argv[2])]; n2 = int(sys.argv[3])
    assert is_skew(b)
    s = stretch(b, n2); assert is_skew(s) and len(s) == n2
    e = energy(s); print(n2, e, f"{n2 * n2 / (2 * e):.4f}", s)
