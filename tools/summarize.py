"""Summarise eval JSONs: success count, outcome string by init state, failed states, steps.
With two files, also prints the paired comparison (fixed / broken) and McNemar exact p."""
import json
import sys
from math import comb


def load(path):
    return {(r["task_id"], r["init_state"]): r for r in json.load(open(path))}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


runs = [load(p) for p in sys.argv[1:]]
for path, rows in zip(sys.argv[1:], runs):
    keys = sorted(rows)
    succ = [rows[k]["success"] for k in keys]
    ok_steps = [rows[k]["steps"] for k in keys if rows[k]["success"]]
    print(f"{path}: {sum(succ)}/{len(succ)}")
    print("  ", "".join("1" if s else "." for s in succ))
    print("   failed:", [k[1] if len({t for t, _ in keys}) == 1 else k for k in keys if not rows[k]["success"]])
    if ok_steps:
        print(f"   steps when solved: mean {sum(ok_steps) / len(ok_steps):.0f}, max {max(ok_steps)}")
if len(runs) == 2:
    a, b = runs
    common = sorted(set(a) & set(b))
    fixed = sum(not a[k]["success"] and b[k]["success"] for k in common)
    broken = sum(a[k]["success"] and not b[k]["success"] for k in common)
    print(f"paired on {len(common)}: fixed {fixed}, broken {broken}, McNemar exact p = {mcnemar(fixed, broken):.4f}")
