"""Compare our verified sequences with the best previously known merit factor per length.
usage: python3 compare.py records.tsv bestknown.tsv   (prints L, ours, previous, gain, source)"""
import sys
from verify import energy

known = {int(l.split("\t")[0]): (float(l.split("\t")[1]), l.split("\t")[2].strip())
         for l in open(sys.argv[2]) if not l.startswith("#")}
ours = {}
for l in open(sys.argv[1]):
    if not l.strip() or l.startswith("#"): continue
    f = l.split(); seq = f[3]; L = len(seq); e = energy(seq)   # never trust the file's E/F columns
    if L not in ours or e < ours[L][0]: ours[L] = (e, seq)
wins = 0
for L in sorted(ours):
    F = L * L / (2 * ours[L][0]); pF, src = known.get(L, (0.0, "none"))
    gain = F - pF; wins += gain > 1e-4
    print(f"{L}\t{F:.4f}\t{pF:.4f}\t{gain:+.4f}\t{src}")
print(f"# {wins} of {len(ours)} lengths beat the best previously known value", file=sys.stderr)
