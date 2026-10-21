"""Extract and independently verify the record sequences in Table 3 of arXiv:2607.09688
(pdftotext -layout output). usage: python3 parse_published.py paper.txt > published.tsv"""
import re, sys
from verify import energy

lines = open(sys.argv[1]).read().split("\n")
start = next(i for i, l in enumerate(lines) if "Table 3: Improved binary" in l)
recs, cur = [], None
for l in lines[start:start + 220]:
    m = re.match(r"\s*(\d{3})\s+(\d\.\d{4})\s+(\d\.\d{4})\s+([0-9a-f]+)\s*$", l)
    if m: cur = [int(m[1]), m[3], m[4]]; recs.append(cur); continue
    m = re.match(r"\s+([0-9a-f]{2,})\s*$", l)
    if not l.strip(): cur = None          # an entry ends at a blank line (page numbers follow)
    elif m and cur: cur[2] += m[1]
for L, F, h in recs:
    b = bin(int(h, 16))[2:].zfill(L)[-L:]
    e = energy(b)
    print(L, e, f"{L * L / (2 * e):.4f}", b, "claimed", F, "OK" if f"{L * L / (2 * e):.4f}" == F else "MISMATCH")
