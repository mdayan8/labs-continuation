"""Figure: our merit factors vs best previously published, per length.
usage: python3 plot.py records.tsv bestknown.tsv out.png"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ours = {}
for l in open(sys.argv[1]):
    if l.strip() and not l.startswith("#"):
        f = l.split(); L, F = int(f[0]), float(f[2]); ours[L] = max(F, ours.get(L, 0))
known = {int(l.split("\t")[0]): (float(l.split("\t")[1]), l.split("\t")[2].strip()) for l in open(sys.argv[2]) if not l.startswith("#")}

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
Ls = sorted(known)
pub = [L for L in Ls if not known[L][1].startswith("Legendre")]
leg = [L for L in Ls if known[L][1].startswith("Legendre")]
ax.plot(leg, [known[L][0] for L in leg], ".", color="#9aa0a6", ms=3, label="best previous: Legendre construction")
ax.plot(pub, [known[L][0] for L in pub], "s", color="#1a73e8", ms=3, label="best previous: published search (GPU grids, 2022-2026)")
o = sorted(ours)
new = [L for L in o if ours[L] > known.get(L, (0,))[0] + 1e-4]
copied = {L for L in o if L in pub and abs(ours[L] - known[L][0]) < 1e-4}   # published records we merely reproduce
rest = [L for L in o if L not in new and L not in copied]
ax.plot(rest, [ours[L] for L in rest], "o", color="#f4b6b0", ms=2.5, label="this work: continuation chain, not a record")
ax.plot(new, [ours[L] for L in new], "o", color="#d93025", ms=4, label=f"this work: new best-known ({len(new)} lengths)")
ax.axhline(7.0, color="k", lw=0.6, ls="--")
ax.set_xlabel("sequence length L"); ax.set_ylabel("merit factor F = L² / 2E")
ax.set_xlim(445, 640); ax.set_ylim(4.8, 8.3); ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(sys.argv[3])
