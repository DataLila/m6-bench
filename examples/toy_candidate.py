#!/usr/bin/env python3
"""A toy candidate that speaks PROTOCOL.md: no CFD, synthetic forces converging geometrically to fixed values (the
seed changes the rate a little, as a perturbed start changes a real solver's path), with a CPU busy-wait per
iteration so that energy is measurable. For testing the harness and as a template for adapters.

    python3 examples/toy_candidate.py --level 3 --mach 0.84 --alpha 3.06 --seed 1 --perturb 1e-6 --max-seconds 20
"""
import argparse
import json
import random
import signal
import sys
import time

TARGET = dict(cl=0.2671, cd=0.01712, cm=-0.1874)


def emit(**rec):
    print("@bench " + json.dumps(rec), flush=True)


def busy(seconds):
    t = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < t:
        x += 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--mach", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", type=float, default=1e-6)
    ap.add_argument("--max-seconds", type=float, default=60)
    ap.add_argument("--iter-s", type=float, default=0.05, help="CPU time per iteration")
    ap.add_argument("--rate", type=float, default=0.6, help="error factor per iteration")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    rng = random.Random(a.seed)
    rate = a.rate * (1 + (0.1 * (rng.random() - 0.5) if a.seed else 0.0))
    emit(type="hello", name="toy", version="0", precision="float64", method="synthetic geometric convergence")
    busy(0.3)                                                 # "setup"
    emit(type="setup_done")
    t0 = time.perf_counter()
    err, it = 0.5, 0
    for lev in range(a.level + 2, a.level, -1):               # two coarse levels first, as a cascade would
        for k in range(3):
            busy(a.iter_s / 4)
            emit(type="iter", level=lev, iter=k + 1, **{c: v * (1 + err) for c, v in TARGET.items()})
    while time.perf_counter() - t0 < a.max_seconds:
        it += 1
        busy(a.iter_s)
        err *= rate
        noise = 1e-12 * (rng.random() - 0.5)
        emit(type="iter", level=a.level, iter=it, res=err, **{c: v * (1 + err + noise) for c, v in TARGET.items()})
    emit(type="done", reason="budget")


if __name__ == "__main__":
    main()
