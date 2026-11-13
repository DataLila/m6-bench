#!/usr/bin/env python3
"""CFD efficiency benchmark harness: runs candidate solvers (PROTOCOL.md) one at a time and measures time and energy
to the force bands, plus every machine metric it can read. Python 3.9+ standard library only.

    python3 harness/bench.py probe                       # sensors, sudo, machine
    python3 harness/bench.py sudoers                     # print a sudoers rule for passwordless powermetrics
    python3 harness/bench.py run CONFIG.json OUTDIR      # references (once), then all runs, then score + report
    python3 harness/bench.py score OUTDIR                # re-score and re-write the report from the recorded runs

CONFIG.json (paths relative to it):
    {"candidates": ["candidates/mach_m5.json"],
     "scenarios": [{"name": "L3-M0.84", "level": 3, "mach": 0.84, "alpha": 3.06}],
     "seeds": [1, 2, 3], "bands": [1e-3, 1e-4, 1e-5], "max_seconds": 600, ...}    (all keys: DEFAULTS below)
A candidate manifest:
    {"name": "mach-m5", "workdir": "..", "command": ["uv", "run", "python", "-m", "src.bench_candidate"],
     "extra_args": [], "env": {}, "env_by_level": {"2": {"MACH_MEM_BUDGET_GB": "11.5"}},
     "memory_gb": {"3": 4, "2": 12.5}, "version_cmd": ["git", "rev-parse", "--short", "HEAD"]}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as SC  # noqa: E402
import sensors as S  # noqa: E402

DEFAULTS = dict(
    seeds=[1, 2, 3],                 # seed 0 is the reference run's (plain free stream)
    perturb=1e-6,
    bands=[1e-3, 1e-4, 1e-5],
    max_seconds=600,                 # solve budget passed to the candidate (--max-seconds)
    setup_grace_s=300,               # extra wall time allowed for setup before the harness kills a run
    sample_s=1.0,                    # proc / memory / GPU / SMC sampling interval
    powermetrics_ms=250,
    session_idle_s=65,               # idle baseline at session start (>= 2 SMC batches of ~30 s)
    idle_s=10,                       # idle baseline before each run (powermetrics)
    cooldown=dict(min_s=60, max_s=300, settle_frac=1.25, settle_add_w=0.5),
    hold=dict(records=5, frac=0.25),
    reference=dict(max_seconds=1800, window=10, frac=0.25, steady_rel=1e-8),
    shuffle_seed=1,
)


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path: Path) -> dict:
    cfg = _merge(DEFAULTS, json.loads(path.read_text()))
    cands = []
    for c in cfg["candidates"]:                              # a manifest path, or {"manifest": path, overrides...}
        over = dict(c) if isinstance(c, dict) else {}
        p = (path.parent / over.pop("manifest", c if isinstance(c, str) else "")).resolve()
        m = json.loads(p.read_text())
        m["workdir"] = str((p.parent / m.get("workdir", ".")).resolve())
        m.update(over)
        cands.append(m)
    cfg["candidates"] = cands
    return cfg


def log(msg):
    print(f"[harness {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -------------------------------------------------------------------------------------------- one run ---

class StopRule:
    """Band run: stop once the tightest band has held for `records` records and `frac` x the entry time.
    Reference run: stop once the forces over the last `frac` of the run's time agree to `steady_rel`
    (scoring.reference)."""

    def __init__(self, level, ref=None, tol=None, hold=None, window=None, steady=None, frac=0.25):
        self.level, self.ref, self.tol, self.hold, self.window, self.steady = level, ref, tol, hold, window, steady
        self.frac = frac
        self.recs, self.last_out = [], -1

    def __call__(self, rec, t) -> bool:
        if rec.get("type") != "iter" or rec.get("level") != self.level or not all(
                isinstance(rec.get(c), (int, float)) for c in SC.COEFS):
            return False
        self.recs.append(rec)
        if self.ref is not None:
            if SC.outside(rec, self.ref, self.tol):
                self.last_out = len(self.recs) - 1
                return False
            n_in = len(self.recs) - 1 - self.last_out
            t_entry = self.recs[self.last_out + 1]["t"]
            return n_in >= self.hold["records"] and t - t_entry >= self.hold["frac"] * t_entry
        if self.window and len(self.recs) >= self.window:
            r = SC.reference(self.recs, self.window, self.frac)
            return r is not None and r["steadiness"] <= self.steady
        return False


def _kill(p: subprocess.Popen, why: list, reason: str):
    if p.poll() is not None:
        return
    why.append(reason)
    try:
        os.killpg(p.pid, signal.SIGTERM)
        for _ in range(100):
            if p.poll() is not None:
                return
            time.sleep(0.1)
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_one(cfg, cand, scen, seed, kind, outdir: Path, pm: S.PowerMetrics | None, stop: StopRule,
            max_seconds: float) -> dict:
    L = scen["level"]
    d = outdir / "runs" / f"{cand['name']}__{scen['name']}__{kind}{seed}"
    d.mkdir(parents=True, exist_ok=True)
    cmd = list(cand["command"]) + ["--level", str(L), "--mach", str(scen["mach"]), "--alpha", str(scen["alpha"]),
                                   "--seed", str(seed), "--perturb", str(cfg["perturb"]),
                                   "--max-seconds", str(max_seconds)] + list(cand.get("extra_args", []))
    env = dict(os.environ, **cand.get("env", {}), **cand.get("env_by_level", {}).get(str(L), {}))
    mem_limit = cand.get("memory_gb", {}).get(str(L))
    info = dict(candidate=cand["name"], scenario=scen["name"], level=L, mach=scen["mach"], alpha=scen["alpha"],
                seed=seed, kind=kind, cmd=cmd, top_before=S.top_processes(), power_source=S.power_source(),
                battery=S.battery_percent(), mem_before=S.memory_stats())
    out_f, err_f = open(d / "stdout.log", "w"), open(d / "stderr.log", "w")
    tm0, tw0 = time.monotonic(), time.time()
    p = subprocess.Popen(cmd, cwd=cand["workdir"], env=env, stdout=subprocess.PIPE, stderr=err_f, text=True,
                         bufsize=1, start_new_session=True)
    tree = S.ProcessTree(p.pid)
    samplers = dict(proc=S.Sampler(tree.sample, cfg["sample_s"]), memory=S.Sampler(S.memory_stats, cfg["sample_s"]),
                    system=S.Sampler(S.system_power, cfg["sample_s"]), gpu=S.Sampler(S.gpu_stats, cfg["sample_s"]))
    for s in samplers.values():
        s.start()
    why: list[str] = []
    wall = max_seconds + cfg["setup_grace_s"]

    def watchdog():
        while p.poll() is None:
            t = time.monotonic() - tm0
            if t > wall:
                _kill(p, why, f"wall time {wall:.0f} s")
            ps = samplers["proc"].samples
            if mem_limit and ps and ps[-1][1].get("footprint_gb", 0) > 1.1 * mem_limit:
                _kill(p, why, f"memory {ps[-1][1]['footprint_gb']:.1f} GB > 1.1 x {mem_limit} GB")
            ms = samplers["memory"].samples
            if ms and ms[-1][1].get("pressure_level") == 4:
                _kill(p, why, "macOS memory pressure critical")
            time.sleep(0.5)

    threading.Thread(target=watchdog, daemon=True).start()
    records, t_setup, hello, done = [], None, None, None
    for line in p.stdout:
        t = time.monotonic() - tm0
        out_f.write(line)
        if not line.startswith("@bench "):
            continue
        try:
            rec = json.loads(line[7:])
        except json.JSONDecodeError:
            continue
        rec["t"] = t
        records.append(rec)
        typ = rec.get("type")
        if typ == "setup_done" and t_setup is None:
            t_setup = t
        elif typ == "hello":
            hello = rec
        elif typ == "done":
            done = rec
        elif stop(rec, t):
            _kill(p, why, "band held" if stop.ref is not None else "reference steady")
    rc = p.wait()
    t_end = time.monotonic() - tm0
    slept = max(0.0, (time.time() - tw0) - t_end)            # the monotonic clock stops while the Mac sleeps
    for s in samplers.values():
        s.stop()
    out_f.close()
    err_f.close()
    rel = lambda ss: [(t - tm0, v) for t, v in ss]           # noqa: E731
    run = dict(info, rc=rc, stop=why[0] if why else ("done" if done else f"exit {rc}"), t_setup=t_setup, t_end=t_end,
               slept_s=slept, battery_after=S.battery_percent(),
               hello=hello, done=done, records=records, mem_after=S.memory_stats(),
               **{k: rel(s.samples) for k, s in samplers.items()},
               powermetrics=rel(pm.window(tm0 - 2, tm0 + t_end + 2)) if pm else [])
    (d / "run.json").write_text(json.dumps(run))
    n = len(SC.fine(records, L))
    log(f"{cand['name']} {scen['name']} {kind}{seed}: {run['stop']}, {n} L{L} records, setup "
        f"{t_setup if t_setup is None else round(t_setup, 1)} s, {t_end:.1f} s")
    return run


# ------------------------------------------------------------------------------------------ the session ---

SLEEP_TOL_S = 2.0                    # a run or baseline during which the Mac slept longer than this is discarded


def idle_baseline(pm, seconds, tries: int = 3) -> dict:
    """Mean idle power over `seconds` (repeated if the Mac slept meanwhile: the reading is then meaningless)."""
    for _ in range(tries):
        sys_s = S.Sampler(S.system_power, 1.0)
        sys_s.start()
        t0, w0 = time.monotonic(), time.time()
        time.sleep(seconds)
        t1 = time.monotonic()
        sys_s.stop()
        if (time.time() - w0) - (t1 - t0) <= SLEEP_TOL_S:
            break
        log("the Mac slept during the idle baseline: repeating it")
    out = {}
    if pm:
        w = pm.window(t0, t1)
        for k in ("combined_w", "cpu_w", "gpu_w"):
            v = SC.mean(w, k, t0, t1)
            if v is not None:
                out[k] = v
    blocks = SC.system_blocks(sys_s.samples)
    inside = [b for b in blocks if b[0] >= t0 - 1.0]          # batches entirely within the idle window
    use = inside or blocks
    if use:
        out["system_w"] = sum(w * (b1 - b0) for b0, b1, w in use) / sum(b1 - b0 for b0, b1, w in use)
        out["system_batches"] = len(use)
    return out


def cooldown(cfg, pm, session_idle):
    c = cfg["cooldown"]
    time.sleep(c["min_s"])
    if not pm or "combined_w" not in session_idle:
        return
    target = session_idle["combined_w"] * c["settle_frac"] + c["settle_add_w"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < c["max_s"] - c["min_s"]:
        now = time.monotonic()
        w = SC.mean(pm.window(now - 5, now), "combined_w", now - 5, now)
        th = [s.get("thermal") for _, s in pm.window(now - 5, now)]
        if w is not None and w <= target and all(x in (None, "Nominal") for x in th):
            return
        time.sleep(2)
    log(f"cool-down: package power still above {target:.1f} W after {c['max_s']} s, starting anyway")


def version(cand):
    vc = cand.get("version_cmd")
    return S.sh(vc, cwd=cand["workdir"]).strip() if vc else None


def cmd_run(a):
    cfg = load_config(a.config)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    # keep the Mac awake while the harness runs (idle sleep on battery corrupts baselines; a closed lid still sleeps,
    # which every run detects and discards, see SLEEP_TOL_S)
    caf = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    src, batt = S.power_source(), S.battery_percent()
    if src != "AC Power" and batt is not None and batt < 40:
        log(f"warning: on battery at {batt}%: a long session may run out or enter low power mode; plug in")
    pm = None
    if not a.no_sudo and S.sudo_prompt():
        pm = S.PowerMetrics(cfg["powermetrics_ms"])
        if not pm.start():
            log(f"powermetrics unavailable: {pm.error}")
            pm = None
    if pm is None:
        log("no powermetrics: energy from the SMC accumulator only (~30 s resolution) and per-process CPU energy")
    rng = random.Random(cfg["shuffle_seed"])
    order = [(ci, si, seed) for ci in range(len(cfg["candidates"])) for si in range(len(cfg["scenarios"]))
             for seed in cfg["seeds"]]
    rng.shuffle(order)
    log(f"session idle baseline ({cfg['session_idle_s']} s) ...")
    idle = idle_baseline(pm, cfg["session_idle_s"])
    session = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), machine=S.machine_info(), powermetrics=pm is not None,
                   idle=idle, config={k: v for k, v in cfg.items() if k != "candidates"},
                   candidates=[dict(c, version=version(c)) for c in cfg["candidates"]],
                   order=[(cfg["candidates"][c]["name"], cfg["scenarios"][s]["name"], seed) for c, s, seed in order])
    (out / "session.json").write_text(json.dumps(session, indent=1))
    log(f"idle: {idle}")
    refs_p = out / "refs.json"
    refs = json.loads(refs_p.read_text()) if refs_p.exists() else {}
    rc = cfg["reference"]
    for cand in cfg["candidates"]:
        for scen in cfg["scenarios"]:
            key = f"{cand['name']}|{scen['name']}"
            if key in refs and not a.refresh_refs:
                continue
            for _ in range(3):                                # repeated if the Mac slept during it
                cooldown(cfg, pm, idle)
                run = run_one(cfg, cand, scen, 0, "ref", out, pm,
                              StopRule(scen["level"], window=rc["window"], steady=rc["steady_rel"], frac=rc["frac"]),
                              scen.get("reference_max_seconds", rc["max_seconds"]))
                if run["slept_s"] <= SLEEP_TOL_S:
                    break
                log(f"the Mac slept {run['slept_s']:.0f} s during the reference run: repeating it")
            r = SC.reference(SC.fine(run["records"], scen["level"]), rc["window"], rc["frac"])
            refs[key] = r
            refs_p.write_text(json.dumps(refs, indent=1))
            log(f"reference {key}: {r}")
    for ci, si, seed in order:
        cand, scen = cfg["candidates"][ci], cfg["scenarios"][si]
        r = refs.get(f"{cand['name']}|{scen['name']}")
        if r is None:
            log(f"no reference for {cand['name']} {scen['name']}: skipped")
            continue
        for _ in range(2):                                    # repeated once if the Mac slept during it
            cooldown(cfg, pm, idle)
            run_idle = idle_baseline(pm, cfg["idle_s"]) if pm else {}
            stop = StopRule(scen["level"], ref=r["ref"], tol=min(cfg["bands"]), hold=cfg["hold"])
            run = run_one(cfg, cand, scen, seed, "s", out, pm, stop, scen.get("max_seconds", cfg["max_seconds"]))
            if run["slept_s"] <= SLEEP_TOL_S:
                break
            log(f"the Mac slept {run['slept_s']:.0f} s during the run: repeating it")
        run["idle"] = dict(idle, **run_idle)
        p = out / "runs" / f"{cand['name']}__{scen['name']}__s{seed}" / "run.json"
        p.write_text(json.dumps(run))
    if pm:
        pm.stop()
    caf.terminate()
    cmd_score(argparse.Namespace(out=out))


# ---------------------------------------------------------------------------------------- score / report ---

def _resources(run) -> dict:
    proc, mem, gpu = [s for _, s in run["proc"]], [s for _, s in run["memory"]], [s for _, s in run["gpu"]]
    pm = [s for _, s in run.get("powermetrics", [])]
    last = proc[-1] if proc else {}
    m0, m1 = run.get("mem_before", {}), run.get("mem_after", {})
    delta = {k: (m1.get(k) or 0) - (m0.get(k) or 0) for k in ("compressions", "decompressions", "swapins", "swapouts",
                                                                "pageins", "pageouts")}
    gu = [g["gpu_util"] for g in gpu if isinstance(g.get("gpu_util"), (int, float))]
    return dict(peak_footprint_gb=max((s.get("footprint_gb", 0) for s in proc), default=None),
                lifetime_max_footprint_gb=last.get("max_footprint_gb"), cpu_user_s=last.get("user_s"),
                cpu_system_s=last.get("system_s"), cpu_proc_energy_j=last.get("cpu_energy_j"),
                instructions=last.get("instructions"), disk_read_gb=last.get("disk_read_gb"),
                disk_write_gb=last.get("disk_write_gb"), proc_pageins=last.get("pageins"),
                swap_used_gb_max=max((s.get("swap_used_gb") or 0 for s in mem), default=None),
                pressure_max=max((s.get("pressure_level") or 0 for s in mem), default=None),
                free_pct_min=min((s.get("free_pct") or 100 for s in mem), default=None),
                compressor_gb_max=max((s.get("compressor_gb") or 0 for s in mem), default=None),
                gpu_util_mean=sum(gu) / len(gu) if gu else None,
                gpu_active_mean=(lambda v: sum(v) / len(v) if v else None)(
                    [s["gpu_active"] for s in pm if "gpu_active" in s]),
                thermal=sorted({s["thermal"] for s in pm if "thermal" in s}), **{f"vm_{k}": v for k, v in delta.items()})


def cmd_score(a):
    out = a.out
    session = json.loads((out / "session.json").read_text())
    refs = json.loads((out / "refs.json").read_text()) if (out / "refs.json").exists() else {}
    rc = _merge(DEFAULTS["reference"], session["config"].get("reference", {}))
    for p in sorted((out / "runs").glob("*__ref0/run.json")):  # references re-derived from the recorded runs
        run = json.loads(p.read_text())
        refs[f"{run['candidate']}|{run['scenario']}"] = SC.reference(SC.fine(run["records"], run["level"]),
                                                                    rc["window"], rc["frac"])
    (out / "refs.json").write_text(json.dumps(refs, indent=1))
    bands = session["config"]["bands"]
    scored = []
    for p in sorted((out / "runs").glob("*__s*/run.json")):
        run = json.loads(p.read_text())
        r = refs.get(f"{run['candidate']}|{run['scenario']}")
        if r is None or run.get("kind") != "s":
            continue
        if run.get("slept_s", 0.0) > SLEEP_TOL_S:                 # the machine slept: not a valid measurement
            log(f"{p.parent.name}: the Mac slept {run['slept_s']:.0f} s during this run, not scored")
            continue
        idle = run.get("idle", session.get("idle", {}))
        scored.append(dict(candidate=run["candidate"], scenario=run["scenario"], seed=run["seed"], stop=run["stop"],
                           t_setup=run["t_setup"], t_end=run["t_end"], power_source=run.get("power_source"),
                           bands=SC.score_run(run, r["ref"], bands, idle), resources=_resources(run)))
    summary = dict(session=session, refs=refs, runs=scored, table=SC.summarize(scored, bands))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "report.md").write_text(report(summary))
    log(f"wrote {out / 'summary.json'} and {out / 'report.md'}")


def _f(x, fmt="{:.1f}"):
    return "–" if x is None else fmt.format(x)


def _rng(d, fmt="{:.1f}"):
    if not d:
        return "–"
    return f"{fmt.format(d['median'])} ({fmt.format(d['min'])}–{fmt.format(d['max'])})"


def report(summary) -> str:
    s = summary["session"]
    m = s["machine"]
    lines = [f"# Benchmark report ({s['started']})", "",
             f"Machine: {m['model']}, {m['chip']}, {m['memory_gb']} GB, {m['os'].get('productName')} "
             f"{m['os'].get('productVersion')}; power source {m['power_source']}, low power mode "
             f"{m['low_power_mode']}; powermetrics {'yes' if s['powermetrics'] else 'no (energy: SMC batches)'}; "
             f"session idle {', '.join(f'{k} {v:.2f}' for k, v in s['idle'].items())} W.", "",
             "Candidates: " + "; ".join(f"{c['name']} ({c.get('version')}) `{' '.join(c['command'] + c.get('extra_args', []))}`"
                                        for c in s["candidates"]), "",
             "## Reference forces (seed 0, long run)", "",
             "| candidate / scenario | C_L | C_D | C_M | steadiness | drift / min | records | vs CFL3D (C_L, C_D, C_M) |",
             "|---|---|---|---|---|---|---|---|"]
    scen = {x["name"]: x for x in s["config"]["scenarios"]}
    for key, r in summary["refs"].items():
        if r is None:
            continue
        c, sn = key.split("|")
        sc = scen.get(sn, {})
        cf = SC.CFL3D.get(sc.get("level")) if sc.get("mach") == 0.84 and sc.get("alpha") == 3.06 else None
        vs = "–" if not cf else ", ".join(f"{100 * (r['ref'][k] - cf[k]) / abs(cf[k]):+.2f} %" for k in cf)
        lines.append(f"| {c} / {sn} | {r['ref']['cl']:.6f} | {r['ref']['cd']:.7f} | {r['ref']['cm']:.6f} | "
                     f"{r['steadiness']:.1e} | {r.get('drift_per_min', float('nan')):.1e} | {r['records']} | {vs} |")
    lines += ["", "## Time and energy to each band (median over seeds, min–max in brackets)", "",
              "| scenario | candidate | band | reached | time s | solve loop s | iterations | system J | "
              "system J above idle | package J | process CPU J | energy x time J·s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in summary["table"]:
        ref = summary["refs"].get(f"{row['candidate']}|{row['scenario']}") or {}
        mark = " ⚠" if row["band"] < 10 * ref.get("steadiness", 0.0) else ""   # PROTOCOL §4: not verifiable
        lines.append(f"| {row['scenario']} | {row['candidate']} | {row['band']:g}{mark} | {row['reached']}/{row['runs']} | "
                     f"{_rng(row.get('t'))} | {_rng(row.get('t_loop'))} | {_rng(row.get('iters'), '{:.0f}')} | "
                     f"{_rng(row.get('system_j'), '{:.0f}')} | {_rng(row.get('system_above_idle_j'), '{:.0f}')} | "
                     f"{_rng(row.get('pkg_j'), '{:.0f}')} | {_rng(row.get('cpu_proc_j'), '{:.1f}')} | "
                     f"{_rng(row.get('edp_js'), '{:.0f}')} |")
    lines += ["", "## Resources per run", "",
              "| run | stop | setup s | end s | peak footprint GB | CPU s | disk R/W GB | swap-outs | "
              "decompressions | pressure max | GPU util % | thermal |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["runs"]:
        x = r["resources"]
        cpu = None if x.get("cpu_user_s") is None else x["cpu_user_s"] + (x.get("cpu_system_s") or 0)
        lines.append(f"| {r['candidate']} {r['scenario']} s{r['seed']} | {r['stop']} | {_f(r['t_setup'])} | "
                     f"{_f(r['t_end'])} | {_f(x.get('peak_footprint_gb'), '{:.2f}')} | {_f(cpu)} | "
                     f"{_f(x.get('disk_read_gb'), '{:.2f}')} / {_f(x.get('disk_write_gb'), '{:.2f}')} | "
                     f"{x.get('vm_swapouts')} | {x.get('vm_decompressions')} | {x.get('pressure_max')} | "
                     f"{_f(x.get('gpu_util_mean'), '{:.0f}')} | {','.join(x.get('thermal') or []) or '–'} |")
    lines += ["", "⚠: band tighter than 10 x the reference steadiness (PROTOCOL.md §4): not verifiable for this candidate.",
              "", "Energy: package = CPU + GPU + ANE (powermetrics, 250 ms); system = package + the rest of the machine "
              "(SMC SystemLoad accumulator, mean over the run); above idle subtracts the idle baseline measured before "
              "the run. Times from process launch (harness clock); the solve loop starts at `setup_done`.", ""]
    return "\n".join(lines)


def cmd_sudoers(_):
    user = os.environ.get("USER", "youruser")
    print("# Lets the harness run powermetrics without a password (nothing else). Install with:\n"
          "#   sudo visudo -f /etc/sudoers.d/cfd-bench-powermetrics\n"
          "# and paste the line below; remove the file to undo.\n"
          f"{user} ALL=(root) NOPASSWD: /usr/bin/powermetrics")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    sub.add_parser("sudoers")
    r = sub.add_parser("run")
    r.add_argument("config", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--no-sudo", action="store_true", help="skip powermetrics (no CPU / GPU power)")
    r.add_argument("--refresh-refs", action="store_true", help="re-run the reference runs")
    sc = sub.add_parser("score")
    sc.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "probe":
        print(json.dumps(S.probe(), indent=1))
    elif a.cmd == "sudoers":
        cmd_sudoers(a)
    elif a.cmd == "run":
        cmd_run(a)
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()
