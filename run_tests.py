#!/usr/bin/env python3
"""
run_tests.py -- benchmark + regression harness for the container corridor sim.

What it does
------------
1. SWEEP: for each seed, run baseline (COORD=0) and COORD=1, both arch1, 5000
   steps, and record (steps, stop_reason, total tasks, per-robot tasks).
2. DETERMINISM SELF-CHECK: re-run a couple of seeds and assert identical results
   (this sim is designed to be fully deterministic given seed + env).
3. GOLDEN-BASELINE REGRESSION (optional): compare this run against a saved
   snapshot (results/golden.json) and flag per-seed improvements/regressions.
4. AGGREGATE THRESHOLDS: survival rate and mean tasks must clear configurable
   floors, else the run is marked FAIL.

Outputs
-------
- A terminal table + analysis text.
- results/last_run.csv  (machine-readable archive of this run)
- results/last_run.json (full structured results)
- With --update-golden: writes results/golden.json from this run.

Because container_sim reads COORD at import time, each configuration is run in a
FRESH SUBPROCESS (clean interpreter) -- this is the robust way to flip COORD and
guarantees no state leaks between runs.

Usage
-----
    python3 run_tests.py                      # default 12-seed sweep
    python3 run_tests.py --seeds 1 2 3        # custom seeds
    python3 run_tests.py --max-steps 5000     # change horizon
    python3 run_tests.py --update-golden      # save this run as the golden baseline
    python3 run_tests.py --no-determinism     # skip the determinism self-check
    python3 run_tests.py --min-survival 0.6   # aggregate survival floor (COORD)
    python3 run_tests.py --min-mean-tasks 25  # aggregate mean-tasks floor (COORD)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
GOLDEN_PATH = os.path.join(RESULTS_DIR, "golden.json")

DEFAULT_SEEDS = [7, 12, 23, 31, 42, 88, 100, 256, 512, 777, 999, 1234]

# Small runner script executed in a subprocess so COORD (read at import time) is
# always fresh. It prints one JSON line so the parent can parse it unambiguously.
_RUNNER = r"""
import os, json, sys
sys.path.insert(0, %r)
import container_sim as cs
res = cs.run_sim()
out = {
    "seed": res["seed"],
    "steps": res["steps"],
    "stop_reason": res["stop_reason"],
    "tasks_done": res["tasks_done"],
    "total": sum(res["tasks_done"]),
    "n_bottlenecks": len(res["annotation"]["bottlenecks"]),
}
print("RESULT_JSON " + json.dumps(out))
"""


def run_one(seed, coord, max_steps, steer="arch1"):
    """Run a single configuration in a fresh subprocess; return a result dict."""
    env = dict(os.environ)
    env["SEED"] = str(seed)
    env["MAX_STEPS"] = str(max_steps)
    env["STEER_MODE"] = steer
    env["COORD"] = "1" if coord else "0"
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER % HERE],
        env=env, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "run failed (seed=%s coord=%s):\n%s" % (seed, coord, proc.stderr[-2000:])
        )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
    if line is None:
        raise RuntimeError("no result line (seed=%s coord=%s):\n%s" % (seed, coord, proc.stdout[-2000:]))
    return json.loads(line[len("RESULT_JSON "):])


REASON_TAG = {"max_steps": "survived", "all_locked": "DEADLOCK", "collision": "COLLISION"}


def sweep(seeds, max_steps):
    """Run baseline + COORD for every seed. Returns list of per-seed dicts."""
    rows = []
    for s in seeds:
        t0 = time.time()
        base = run_one(s, coord=False, max_steps=max_steps)
        cdr = run_one(s, coord=True, max_steps=max_steps)
        dt = time.time() - t0
        rows.append({
            "seed": s,
            "n_bottlenecks": cdr["n_bottlenecks"],
            "base_steps": base["steps"], "base_reason": base["stop_reason"],
            "base_total": base["total"], "base_tasks": base["tasks_done"],
            "coord_steps": cdr["steps"], "coord_reason": cdr["stop_reason"],
            "coord_total": cdr["total"], "coord_tasks": cdr["tasks_done"],
            "secs": round(dt, 1),
        })
        print("  seed %-5d done in %4.1fs  | baseline %s/%d tasks  | COORD %s/%d tasks"
              % (s, dt, REASON_TAG[base["stop_reason"]], base["total"],
                 REASON_TAG[cdr["stop_reason"]], cdr["total"]))
    return rows


def determinism_check(seeds, max_steps):
    """Re-run a couple of configs and confirm identical output (steps + tasks)."""
    problems = []
    for s in seeds:
        for coord in (False, True):
            a = run_one(s, coord, max_steps)
            b = run_one(s, coord, max_steps)
            key_a = (a["steps"], a["stop_reason"], tuple(a["tasks_done"]))
            key_b = (b["steps"], b["stop_reason"], tuple(b["tasks_done"]))
            tag = "COORD" if coord else "baseline"
            if key_a != key_b:
                problems.append("seed %s %s: %s != %s" % (s, tag, key_a, key_b))
    return problems


def print_table(rows):
    print()
    print("=" * 86)
    print("SWEEP RESULTS  (baseline = ORCA only, COORD = + coordination layer; arch1)")
    print("=" * 86)
    hdr = "%-6s %-4s | %-9s %-8s %-6s | %-9s %-8s %-6s | %s"
    print(hdr % ("seed", "bn", "base_stop", "b_steps", "b_tsk",
                 "coord_stop", "c_steps", "c_tsk", "verdict"))
    print("-" * 86)
    for r in rows:
        verdict = seed_verdict(r)
        print(hdr % (
            r["seed"], r["n_bottlenecks"],
            REASON_TAG[r["base_reason"]], r["base_steps"], r["base_total"],
            REASON_TAG[r["coord_reason"]], r["coord_steps"], r["coord_total"],
            verdict,
        ))
    print("-" * 86)


def seed_verdict(r):
    """One-word read on whether COORD helped vs baseline on this seed."""
    surv = lambda reason: reason == "max_steps"
    bs, cs_ = surv(r["base_reason"]), surv(r["coord_reason"])
    if cs_ and not bs:
        return "COORD wins (survives)"
    if bs and not cs_:
        return "COORD worse (fails)"
    if cs_ and bs:
        d = r["coord_total"] - r["base_total"]
        return "both survive (%+d tasks)" % d
    return "both fail"


def aggregate(rows, min_survival, min_mean_tasks):
    n = len(rows)
    coord_surv = sum(1 for r in rows if r["coord_reason"] == "max_steps")
    base_surv = sum(1 for r in rows if r["base_reason"] == "max_steps")
    coord_mean = sum(r["coord_total"] for r in rows) / n if n else 0.0
    base_mean = sum(r["base_total"] for r in rows) / n if n else 0.0
    surv_rate = coord_surv / n if n else 0.0

    print()
    print("AGGREGATE")
    print("  COORD survival : %d/%d (%.0f%%)   baseline survival: %d/%d (%.0f%%)"
          % (coord_surv, n, 100 * surv_rate, base_surv, n, 100 * base_surv / n if n else 0))
    print("  COORD mean tasks: %.1f            baseline mean tasks: %.1f"
          % (coord_mean, base_mean))

    fails = []
    if surv_rate < min_survival:
        fails.append("survival rate %.0f%% < floor %.0f%%" % (100 * surv_rate, 100 * min_survival))
    if coord_mean < min_mean_tasks:
        fails.append("mean tasks %.1f < floor %.1f" % (coord_mean, min_mean_tasks))
    return fails, {
        "coord_survival": coord_surv, "base_survival": base_surv, "n": n,
        "coord_mean_tasks": round(coord_mean, 2), "base_mean_tasks": round(base_mean, 2),
        "coord_survival_rate": round(surv_rate, 3),
    }


def compare_golden(rows):
    """Compare against results/golden.json if present; print per-seed deltas."""
    if not os.path.exists(GOLDEN_PATH):
        print()
        print("GOLDEN BASELINE: none saved yet (run with --update-golden to create one).")
        return []
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    gmap = {r["seed"]: r for r in golden["rows"]}
    print()
    print("REGRESSION vs golden (%s):" % golden.get("timestamp", "?"))
    regressions = []
    improvements = []
    for r in rows:
        g = gmap.get(r["seed"])
        if g is None:
            print("  seed %-5d : new seed (not in golden)" % r["seed"])
            continue
        # focus on COORD: did survival or task count change?
        g_surv = g["coord_reason"] == "max_steps"
        r_surv = r["coord_reason"] == "max_steps"
        dt = r["coord_total"] - g["coord_total"]
        if g_surv and not r_surv:
            msg = "seed %-5d : REGRESSION  COORD %s -> %s" % (
                r["seed"], REASON_TAG[g["coord_reason"]], REASON_TAG[r["coord_reason"]])
            print("  " + msg); regressions.append(msg)
        elif not g_surv and r_surv:
            msg = "seed %-5d : IMPROVED    COORD %s -> %s" % (
                r["seed"], REASON_TAG[g["coord_reason"]], REASON_TAG[r["coord_reason"]])
            print("  " + msg); improvements.append(msg)
        elif dt <= -3:
            msg = "seed %-5d : task drop   COORD %d -> %d (%+d)" % (
                r["seed"], g["coord_total"], r["coord_total"], dt)
            print("  " + msg); regressions.append(msg)
        elif dt >= 3:
            print("  seed %-5d : task gain   COORD %d -> %d (%+d)"
                  % (r["seed"], g["coord_total"], r["coord_total"], dt))
        else:
            print("  seed %-5d : stable      COORD %s/%d (%+d tasks)"
                  % (r["seed"], REASON_TAG[r["coord_reason"]], r["coord_total"], dt))
    return regressions


def save_outputs(rows, agg, update_golden):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    # CSV
    csv_path = os.path.join(RESULTS_DIR, "last_run.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "n_bottlenecks",
                    "base_steps", "base_reason", "base_total",
                    "coord_steps", "coord_reason", "coord_total", "secs"])
        for r in rows:
            w.writerow([r["seed"], r["n_bottlenecks"],
                        r["base_steps"], r["base_reason"], r["base_total"],
                        r["coord_steps"], r["coord_reason"], r["coord_total"], r["secs"]])
    # JSON
    payload = {"timestamp": ts, "rows": rows, "aggregate": agg}
    with open(os.path.join(RESULTS_DIR, "last_run.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print("Saved results/last_run.csv and results/last_run.json")
    if update_golden:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print("Saved results/golden.json  (new golden baseline)")


def main():
    ap = argparse.ArgumentParser(description="Container sim benchmark + regression harness")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--update-golden", action="store_true",
                    help="save this run as results/golden.json")
    ap.add_argument("--no-determinism", action="store_true",
                    help="skip the determinism self-check")
    ap.add_argument("--determinism-seeds", type=int, nargs="+", default=None,
                    help="which seeds to use for the determinism check (default: first 2 of --seeds)")
    ap.add_argument("--min-survival", type=float, default=0.5,
                    help="aggregate COORD survival-rate floor (0..1)")
    ap.add_argument("--min-mean-tasks", type=float, default=20.0,
                    help="aggregate COORD mean-tasks floor")
    args = ap.parse_args()

    print("Container sim test harness")
    print("  seeds      : %s" % args.seeds)
    print("  max_steps  : %d   steer: arch1   (baseline vs COORD, each in a fresh subprocess)"
          % args.max_steps)
    print()
    print("Running sweep (~24s/seed) ...")
    rows = sweep(args.seeds, args.max_steps)

    print_table(rows)
    fails, agg = aggregate(rows, args.min_survival, args.min_mean_tasks)
    regressions = compare_golden(rows)

    det_problems = []
    if not args.no_determinism:
        det_seeds = args.determinism_seeds or args.seeds[:2]
        print()
        print("DETERMINISM SELF-CHECK on seeds %s ..." % det_seeds)
        det_problems = determinism_check(det_seeds, args.max_steps)
        if det_problems:
            for p in det_problems:
                print("  NON-DETERMINISTIC: " + p)
        else:
            print("  OK: identical results on re-run.")

    save_outputs(rows, agg, args.update_golden)

    # ---- overall verdict --------------------------------------------------
    print()
    print("=" * 86)
    problems = []
    if det_problems:
        problems.append("%d determinism failure(s)" % len(det_problems))
    if regressions:
        problems.append("%d regression(s) vs golden" % len(regressions))
    if fails:
        problems += fails
    if problems:
        print("OVERALL: FAIL")
        for p in problems:
            print("  - " + p)
        print("=" * 86)
        sys.exit(1)
    print("OVERALL: PASS")
    print("=" * 86)


if __name__ == "__main__":
    main()
