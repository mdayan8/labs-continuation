"""Continuation search: grow a skew-symmetric sequence two elements at a time (one at each end, which
preserves skew-symmetry) and re-optimise with parallel self-avoiding walks at every length.
usage: python3 chain.py START_TSV L0 Lmax seconds_per_length procs out.tsv"""
import subprocess, sys
from verify import energy


def extend(b):  # odd length n -> n+2, skew-symmetric if b is; pick the better end bit
    n2 = len(b) + 2; c2 = n2 // 2
    best = None
    for x in "01":
        half = x + b[:len(b) // 2 + 1]
        s = list(half) + ["0"] * c2
        for j in range(1, c2 + 1):
            s[c2 + j] = s[c2 - j] if j % 2 == 0 else "10"[int(s[c2 - j])]
        s = "".join(s); e = energy(s)
        if best is None or e < best[0]: best = (e, s)
    return best[1]


def run(seq, secs, procs, tag):
    L = len(seq)
    ps = [subprocess.Popen(["./labs", str(L), str(secs), str(tag * 100 + i), "8", "0.05", "-", seq],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for i in range(procs)]
    best = (energy(seq), seq)
    for p in ps:
        out = p.communicate()[0].strip().split("\n")
        for line in out:
            if line:
                f = line.split()
                if int(f[1]) < best[0]: best = (int(f[1]), f[3])
    return best


if __name__ == "__main__":
    src, L0, Lmax, secs, procs, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    seq = [l.split()[3] for l in open(src) if l.split()[0] == str(L0)][0]
    while len(seq) < Lmax:
        seq = extend(seq)
        e, seq = run(seq, secs, procs, len(seq))
        assert energy(seq) == e
        L = len(seq)
        with open(out, "a") as f: f.write(f"{L} {e} {L * L / (2 * e):.4f} {seq}\n")
        print(L, e, f"{L * L / (2 * e):.4f}", flush=True)
