"""Recovery metrics of arXiv 2611.33102 from eval_ckpt.py outputs.

pass@k is the unbiased estimator from n samples with c correct: 1 - C(n-c, k) / C(n, k), averaged over questions.
eps_traj = mean over evaluated steps and k in K of |F_k(student_t) - F_k(online_t)|; eps_end uses the last step only.
usage: metrics.py <runs/replay/RUN> [--json out.json]
"""
import sys, os, json, glob
from math import comb
run = sys.argv[1]; K = [1, 2, 4, 8, 16, 32]
def load(group):
    out = {}
    for f in sorted(glob.glob(f"{run}/evals/{group}/step_*.json")):
        d = json.load(open(f)); n = d["n"]; ks = [k for k in K if k <= n]
        out[int(os.path.basename(f)[5:9])] = {k: sum(1 - comb(n - r["correct"], k) / comb(n, k) for r in d["rows"]) / len(d["rows"]) for k in ks}
    return out
on = load("online"); res = {"online": on}; T = max(on)
def err(a, b): return sum(abs(a[k] - b[k]) for k in a) / len(a)
print(f"{'model':<16}" + "".join(f"{'p@1 t=' + str(s):>11}" for s in sorted(on)) + f"{'eps_traj':>10}{'eps_end':>9}")
print(f"{'online':<16}" + "".join(f"{on[s][1]:>11.3f}" for s in sorted(on)))
rows = {"base (pi_0)": {s: on[0] for s in on if s > 0}}
for g in sorted(os.listdir(f"{run}/evals")):
    if g != "online" and load(g): rows[g] = load(g); res[g] = rows[g]
summary = {}
for g, d in rows.items():
    steps = [s for s in sorted(d) if s in on and s > 0]
    traj = sum(err(d[s], on[s]) for s in steps) / len(steps); end = err(d[T], on[T]) if T in d else float("nan")
    summary[g] = dict(eps_traj=traj, eps_end=end, n_steps=len(steps))
    print(f"{g:<16}" + "".join((f"{d[s][1]:>11.3f}" if s in d and g != 'base (pi_0)' else f"{'':>11}") if s else f"{'':>11}" for s in sorted(on)) + f"{traj:>10.3f}{end:>9.3f}")
if "--json" in sys.argv: json.dump(dict(curves=res, summary=summary), open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
