#!/usr/bin/env python3
"""Replay an assembled-sequence deliverable through a construction you did not choose.

Problem class this closes: you design pieces, lay them out in YOUR order, and
self-check the pieces as YOU laid them out. The consumer instead ASSEMBLES the
pieces in a canonical order and locates a boundary by SEARCHING inside that
assembly. When your fragment shares an edge with its surroundings (a repeated
substring, a common prefix/suffix), the located boundary can sit at a different
offset than you intended, and every derived length shifts with it — so the
pieces pass your check and fail the consumer's.

This tool takes the delivered pieces and does the OTHER construction:

  * assemble them in the order a splice-style consumer would (reverse-complement
    the reverse piece, prepend it, append the forward piece),
  * locate the inserted region by SEARCHING inside the assembly (first match),
  * derive each flank's length from the located boundary,
  * check every derived length against the stated bound and report the MARGIN,
  * and evaluate every candidate offset when the search can match more than once
    (a repeated substring), reporting the WORST-case derived numbers.

It reads ONLY the delivered file and the input sequence, never your in-memory
design values, so a disagreement it reports is a real disagreement.

Usage:
  python3 assembled_deliverable_replay.py --input original.fasta \
      --deliverable pieces.fasta --insert-fasta insert.fasta [--bound 15:45]

Or, when the inserted region must be derived from the input/output pair:
  python3 assembled_deliverable_replay.py --input original.fasta \
      --output modified.fasta --deliverable pieces.fasta --bound 15:45

Exit 0 = every derived quantity is inside its bound; 1 = at least one is outside
(or the boundary could not be located); 2 = nothing measurable.
No target identifier and no target-specific constant appears in this file: the
bound, the file paths and the sequences are all supplied at run time.
"""

from __future__ import annotations

import argparse
import sys

COMP = str.maketrans("ACGTURYKMBDHVNacgturykmbdhvn", "TGCAAYRMKVHDBNtgcaayrmkvhdbn")


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def read_fasta(path: str) -> list[tuple[str, str]]:
    entries, name, buf = [], None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    entries.append((name, "".join(buf)))
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.upper())
    if name is not None:
        entries.append((name, "".join(buf)))
    return entries


def load_seq(path: str) -> str:
    """Accept either FASTA or a bare sequence file."""
    ents = read_fasta(path)
    if ents:
        return "".join(seq for _, seq in ents)
    with open(path, encoding="utf-8", errors="replace") as fh:
        return "".join(l.strip().upper() for l in fh if not l.startswith(">"))


def find_all(hay: str, needle: str) -> list[int]:
    out, i = [], hay.find(needle)
    while i != -1:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out


def derive_insert(inp: str, out: str) -> str | None:
    """Inserted region of `out` relative to `inp` = the maximal differing span.

    Ambiguous when the insert shares an edge with the flank, which is exactly why
    the caller may pass the insert explicitly instead.
    """
    p = 0
    while p < min(len(inp), len(out)) and inp[p] == out[p]:
        p += 1
    s = 0
    while s < min(len(inp), len(out)) - p and inp[-1 - s] == out[-1 - s]:
        s += 1
    return out[p:len(out) - s] or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="original sequence (FASTA or bare)")
    ap.add_argument("--output", help="modified sequence, used to derive the inserted region")
    ap.add_argument("--insert-fasta", help="explicit inserted region")
    ap.add_argument("--deliverable", required=True, help="the delivered pieces file")
    ap.add_argument("--bound", default="15:45", help="MIN:MAX permitted derived length")
    ap.add_argument("--min-hits", type=int, default=1)
    args = ap.parse_args(argv)

    lo_s, _, hi_s = args.bound.partition(":")
    lo, hi = int(lo_s), int(hi_s)

    inp = load_seq(args.input)
    if args.insert_fasta:
        insert = load_seq(args.insert_fasta)
    elif args.output:
        insert = derive_insert(inp, load_seq(args.output))
    else:
        print("SKIP: need --insert-fasta or --output to know the inserted region")
        print("VERDICT: UNMEASURED")
        return 2
    if not insert:
        print("SKIP: could not derive a non-empty inserted region")
        print("VERDICT: UNMEASURED")
        return 2

    pieces = read_fasta(args.deliverable)
    if len(pieces) < 2:
        print(f"FAIL: deliverable has {len(pieces)} sequence(s); two flanks are required")
        print("VERDICT: FAIL")
        return 1
    fwd = next((s for n, s in pieces if "f" in n.lower()), pieces[0][1])
    rev = next((s for n, s in pieces if "r" in n.lower()), pieces[1][1])

    # The construction a splice consumer performs: rc(reverse piece) + forward piece.
    assembly = rc(rev) + fwd
    hits = find_all(assembly, insert)
    print(f"insert: {len(insert)} nt; assembly: {len(assembly)} nt; "
          f"offsets where the inserted region matches: {hits or 'NONE'}")

    if not hits:
        print("FAIL: the delivered pieces do not assemble into a sequence containing the "
              "inserted region — a consumer searching this assembly finds nothing")
        print("VERDICT: FAIL")
        return 1
    if len(hits) < args.min_hits:
        print(f"FAIL: only {len(hits)} match(es); expected at least {args.min_hits}")
        print("VERDICT: FAIL")
        return 1

    worst_exit = 0
    for off in hits:
        rev_side = off                       # everything before the insert is the rc'd reverse flank
        fwd_side = len(assembly) - off - len(insert)
        ok_r = lo <= rev_side <= hi
        ok_f = lo <= fwd_side <= hi
        print(f"  offset {off}: reverse-flank length={rev_side} (margin "
              f"{min(rev_side - lo, hi - rev_side)}) {'OK' if ok_r else 'OUT OF BOUND'}")
        print(f"  offset {off}: forward-flank length={fwd_side} (margin "
              f"{min(fwd_side - lo, hi - fwd_side)}) {'OK' if ok_f else 'OUT OF BOUND'}")
        if not (ok_r and ok_f):
            worst_exit = 1

    if len(hits) > 1:
        print("note: the inserted region matches at more than one offset — a consumer taking "
              "the FIRST match may derive different lengths than you intended; the worst case "
              "above is the one that decides.")

    print("VERDICT: PASS" if worst_exit == 0 else "VERDICT: FAIL")
    return worst_exit


if __name__ == "__main__":
    sys.exit(main())
