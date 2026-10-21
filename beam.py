"""Beam search over length-changing operators on skew-symmetric sequences (GPU, torch).
Negative result: with K=128 it matched the greedy chain (fill.py) at every length we tried.
usage: python beam.py seeds_hex.txt K steps out.tsv      (seeds_hex lines: "L:hex")"""
import sys, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"


def complete(half):                      # (N, c+1) int8 -> (N, 2c+1) skew-symmetric
    c = half.shape[1] - 1
    j = torch.arange(1, c + 1, device=dev)
    sign = torch.where(j % 2 == 1, -1, 1).to(torch.int8)
    return torch.cat([half, half[:, c - j] * sign], 1)


def energies(X):                         # (N, L) +-1 -> int64 energies via FFT autocorrelation
    N, L = X.shape; m = 1 << (2 * L - 1).bit_length(); out = []
    for i in range(0, N, 4096):
        f = torch.fft.rfft(X[i:i + 4096].float(), m)
        C = torch.fft.irfft(f * f.conj(), m)[:, 1:L].round().long()
        out.append((C * C).sum(1))
    return torch.cat(out)


def expand(H, mode):                     # all single insertions (grow) or deletions (shrink)
    K, n = H.shape; cands = []
    if mode == "grow":
        for i in range(n + 1):
            for x in (1, -1):
                col = torch.full((K, 1), x, dtype=torch.int8, device=dev)
                cands.append(torch.cat([H[:, :i], col, H[:, i:]], 1))
    else:
        for i in range(n - 1):
            cands.append(torch.cat([H[:, :i], H[:, i + 1:]], 1))
    return torch.unique(torch.cat(cands, 0), dim=0)


def run(seed, K, steps, mode, out):
    L = len(seed)
    H = torch.tensor([[1 if ch == "1" else -1 for ch in seed[:L // 2 + 1]]], dtype=torch.int8, device=dev)
    for _ in range(steps):
        Hc = expand(H, mode); X = complete(Hc); e = energies(X)
        idx = torch.topk(e, min(K, len(e)), largest=False).indices
        H = Hc[idx]; best = X[idx[0]]; L = X.shape[1]; eb = int(e[idx[0]])
        out.write(f"{L} {eb} {L * L / (2 * eb):.4f} " + "".join("1" if v > 0 else "0" for v in best.tolist()) + "\n")
        out.flush()
        if L < 7: break


if __name__ == "__main__":
    seeds = []
    for l in open(sys.argv[1]):
        L, h = l.strip().split(":"); L = int(L); seeds.append(bin(int(h, 16))[2:].zfill(L)[-L:])
    K, steps = int(sys.argv[2]), int(sys.argv[3])
    with open(sys.argv[4], "w") as out:
        for i, sd in enumerate(seeds):
            for mode in ("grow", "shrink"):
                run(sd, K, steps, mode, out)
            print(i, len(sd), flush=True)
