"""Scoring for the benchmark harness (PROTOCOL.md §4-5): reference forces, band entry, energy integration, summary.
Pure functions on recorded data (Python standard library only)."""
from __future__ import annotations

import statistics

COEFS = ("cl", "cd", "cm")
TICK_S = 0.5                       # SMC SystemLoad accumulator tick (sensors.system_power)

# CFL3D on the same TMR grid, M 0.84 alpha 3.06 (L3 from the challenge sheet; L2 from src/accuracy.py)
CFL3D = {3: dict(cl=0.266033, cd=0.0173521, cm=-0.186608), 2: dict(cl=0.26641, cd=0.016917)}


def fine(records: list[dict], level: int) -> list[dict]:
    """The scored records: `iter` records of the finest level, with all three coefficients."""
    return [r for r in records if r.get("type") == "iter" and r.get("level") == level
            and all(isinstance(r.get(c), (int, float)) for c in COEFS)]


def reference(recs: list[dict], window: int = 10, frac: float = 0.25) -> dict | None:
    """C_ref = mean over the last `frac` of the run's finest-level time (at least `window` records); steadiness =
    largest relative deviation from it over that window; drift = largest relative change per minute (least-squares
    slope over the window). A time window, not a record count: a stalled solver taking tiny steps looks steady over
    ten records while its forces still creep (first L3 session: 8e-9 over 10 records, 2e-5 drift)."""
    if len(recs) < 2:
        return None
    t0, t1 = recs[0].get("t", 0.0), recs[-1].get("t", 0.0)
    w = [r for r in recs if r.get("t", 0.0) >= t1 - frac * (t1 - t0)]
    if len(w) < window:
        w = recs[-window:]
    ref = {c: statistics.fmean(r[c] for r in w) for c in COEFS}
    spread = max(abs(r[c] - ref[c]) / abs(ref[c]) for r in w for c in COEFS if ref[c] != 0)
    drift = 0.0
    ts = [r.get("t", 0.0) for r in w]
    tm = statistics.fmean(ts)
    den = sum((t - tm) ** 2 for t in ts)
    if den > 0:
        for c in COEFS:
            slope = sum((t - tm) * r[c] for t, r in zip(ts, w)) / den
            drift = max(drift, abs(slope) * 60.0 / abs(ref[c]))
    return dict(ref=ref, steadiness=spread, drift_per_min=drift, records=len(recs), window=len(w),
                window_s=ts[-1] - ts[0], **{f"final_{k}": recs[-1][k] for k in ("cdp", "cdv") if k in recs[-1]})


def outside(r: dict, ref: dict, tol: float) -> bool:
    return any(abs(r[c] - ref[c]) > tol * abs(ref[c]) for c in COEFS)


def band_entry(recs: list[dict], ref: dict, tol: float) -> int | None:
    """Index of the first record from which every later record is inside the band; None if the last one is out."""
    last_out = -1
    for i, r in enumerate(recs):
        if outside(r, ref, tol):
            last_out = i
    return None if last_out == len(recs) - 1 else last_out + 1


def integrate(samples: list[tuple[float, dict]], key: str, ta: float, tb: float) -> float | None:
    """Trapezoidal integral of samples[.][1][key] over [ta, tb] (samples: (t, dict), t ascending); the value is
    held constant before the first and after the last sample. None without any sample of `key`."""
    pts = [(t, s[key]) for t, s in samples if isinstance(s.get(key), (int, float))]
    if not pts or tb <= ta:
        return None if not pts else 0.0

    def at(t):
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        lo, hi = 0, len(pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if pts[mid][0] <= t else (lo, mid)
        (t0, v0), (t1, v1) = pts[lo], pts[hi]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    knots = [ta] + [t for t, _ in pts if ta < t < tb] + [tb]
    return sum(0.5 * (at(a) + at(b)) * (b - a) for a, b in zip(knots, knots[1:]))


def mean(samples, key, ta, tb):
    e = integrate(samples, key, ta, tb)
    return None if e is None or tb <= ta else e / (tb - ta)


def value_at(samples: list[tuple[float, dict]], key: str, t: float):
    """Linear interpolation of a cumulative counter at t (held at the ends)."""
    pts = [(ts, s[key]) for ts, s in samples if isinstance(s.get(key), (int, float))]
    if not pts:
        return None
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
    return pts[-1][1]


def system_blocks(samples: list[tuple[float, dict]]) -> list[tuple[float, float, float]]:
    """(t0, t1, mean W) per published batch of the SMC SystemLoad accumulator: each change of load_count between
    consecutive samples is one batch of `dc` ticks ending at the sample time."""
    pts = [(t, s["load_acc"], s["load_count"]) for t, s in samples
           if isinstance(s.get("load_acc"), int) and isinstance(s.get("load_count"), int)]
    out = []
    for (ta, aa, ca), (tb, ab, cb) in zip(pts, pts[1:]):
        if cb > ca:
            out.append((tb - (cb - ca) * TICK_S, tb, (ab - aa) / (cb - ca) / 1000.0))
    return out


def block_energy(blocks, ta: float, tb: float) -> tuple[float | None, float]:
    """Energy of the piecewise-constant block power over [ta, tb] and the fraction of [ta, tb] the blocks cover."""
    e, cov = 0.0, 0.0
    for b0, b1, w in blocks:
        o = max(0.0, min(b1, tb) - max(b0, ta))
        e += w * o
        cov += o
    if cov == 0:
        return None, 0.0
    return e, cov / (tb - ta) if tb > ta else 0.0


def score_run(run: dict, ref: dict, bands: list[float], idle: dict) -> dict:
    """Time, energy and energy x time to each band for one run (run: as written by bench.py; times from launch).

    Energies (J):
      pkg       CPU + GPU + ANE from powermetrics (250 ms; sudo), integrated from launch to band entry;
      system    pkg + (system - pkg) mean over the run x time, the rest of the machine (display, SSD, ...) from the
                SMC accumulator; without powermetrics, the SMC batches alone (~30 s resolution: coarse);
      cpu_proc  the candidate process tree's own CPU energy (libproc, no GPU);
    each also above idle (idle: the baseline measured before the run)."""
    recs = fine(run["records"], run["level"])
    pm, proc = run.get("powermetrics", []), run.get("proc", [])
    blocks = system_blocks(run.get("system", []))
    t_end = run.get("t_end", recs[-1]["t"] if recs else 0.0)
    pkg_run = mean(pm, "combined_w", 0.0, t_end)
    sys_run, cov_run = block_energy(blocks, 0.0, t_end)
    rest_w = None
    if pkg_run is not None and sys_run is not None and cov_run > 0.5:
        rest_w = max(0.0, sys_run / (cov_run * t_end) - pkg_run)
    elif pkg_run is not None and idle.get("system_w") is not None and idle.get("combined_w") is not None:
        rest_w = max(0.0, idle["system_w"] - idle["combined_w"])     # session idle split as the fallback
    out = {}
    for tol in bands:
        i = band_entry(recs, ref, tol) if recs else None
        if i is None:
            out[str(tol)] = dict(reached=False)
            continue
        t = recs[i]["t"]
        pkg = integrate(pm, "combined_w", 0.0, t)
        if pkg is not None and rest_w is not None:
            system, how = pkg + rest_w * t, "pkg+rest"
        else:                                                 # mean power of the covering batches x time
            system, cov = block_energy(blocks, 0.0, t)
            system = system / cov if system is not None and cov > 0 else None
            how = f"smc batches ({cov:.0%} covered)"
        cpu = value_at(proc, "cpu_energy_j", t)
        d = dict(reached=True, t=t, t_setup=run.get("t_setup"), t_loop=None if run.get("t_setup") is None else
                 t - run["t_setup"], iters=recs[i].get("iter"), index=i, system_j=system, system_how=how,
                 pkg_j=pkg, gpu_j=integrate(pm, "gpu_w", 0.0, t), cpu_j=integrate(pm, "cpu_w", 0.0, t),
                 cpu_proc_j=cpu)
        for k, ik in (("system_j", "system_w"), ("pkg_j", "combined_w")):
            if d[k] is not None and idle.get(ik) is not None:
                d[k.replace("_j", "_above_idle_j")] = d[k] - idle[ik] * t
        if system is not None:
            d["edp_js"] = system * t
        out[str(tol)] = d
    return out


def summarize(scored: list[dict], bands: list[float]) -> list[dict]:
    """Median / min / max over seeds per (candidate, scenario, band). scored: dicts with candidate, scenario, seed,
    bands (score_run output)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for s in scored:
        groups.setdefault((s["candidate"], s["scenario"]), []).append(s)
    rows = []
    for (cand, scen), runs in sorted(groups.items()):
        for tol in bands:
            got = [r["bands"][str(tol)] for r in runs if r["bands"].get(str(tol), {}).get("reached")]
            row = dict(candidate=cand, scenario=scen, band=tol, runs=len(runs), reached=len(got))
            for k in ("t", "t_loop", "iters", "system_j", "system_above_idle_j", "pkg_j", "gpu_j", "edp_js",
                      "cpu_proc_j"):
                v = [g[k] for g in got if isinstance(g.get(k), (int, float))]
                if v:
                    row[k] = dict(median=statistics.median(v), min=min(v), max=max(v))
            rows.append(row)
    return rows
