# m6-bench: an efficiency benchmark for ONERA M6 RANS solvers

Runs CFD solvers ("candidates") one at a time on one Mac and measures **time and energy to force-convergence bands**
(C_L, C_D, C_M within 1e-3, 1e-4, 1e-5 of each solver's own converged answer), plus everything else the machine
reports: memory pressure, swap, compression, disk, GPU utilisation, thermal state. The case is the NASA TMR ONERA M6
wing (SA-neg, TMR grid family L3 and L2). The point is efficiency on the same machine, not who brings the bigger
computer.

The interface a solver implements is in [PROTOCOL.md](PROTOCOL.md): five command-line options and a few `@bench` JSON
lines on stdout. The harness is Python 3.9+ standard library only (macOS's `/usr/bin/python3` is enough).

## Running a session

    git clone https://github.com/DataLila/m6-bench && cd m6-bench
    git checkout <the agreed tag>                     # both sides measure with the same harness

1. **Probe the machine** and allow `powermetrics` without a password (the only thing that needs sudo; delete the
   file to undo):

       python3 bench.py probe
       python3 bench.py sudoers                     # prints one sudoers line
       sudo visudo -f /etc/sudoers.d/cfd-bench-powermetrics

2. **Fetch the pinned candidate builds** (downloaded, checked against the SHA-256 in their manifest, unpacked into
   `artifacts/`):

       python3 bench.py fetch

3. **Prepare each candidate once** (untimed). For Mach-M5 (below), point `--from` at your TMR grid files:

       python3 bench.py prepare candidates/mach_m5.json --from /path/to/your/grids --levels 3 2

4. **Add your own solver**: an adapter that speaks [PROTOCOL.md](PROTOCOL.md) and a manifest (below). Check it with
   the toy session first (`examples/toy_candidate.py` is a complete minimal adapter):

       python3 bench.py run configs/toy.json results-toy

5. **Run the session**: close everything else, keep the lid open and the Mac on mains with a full battery (or on
   battery), then

       python3 bench.py run configs/challenge.json results --add-candidate /path/to/your_manifest.json

   About 2 hours per candidate (three L3 conditions and L2, three seeds each, reference runs, cool-downs).

6. **Send back** `results/report.md` and `results/summary.json` (and `results/runs/` if you are willing: every record
   and sensor sample, for checking).

## Keeping sessions comparable

- **The measuring code is pinned.** `run` records the harness commit and the hashes of `bench.py`, `sensors.py`,
  `scoring.py` and `PROTOCOL.md` in `session.json` and the report, and refuses to start when they differ from the
  commit (`--dev` overrides, and the report then says so). Changes go through pull requests and agreed tags.
- **Candidates are pinned builds.** A manifest with an `artifact` names a URL and a SHA-256; `fetch` refuses anything
  else and `run` refuses a build that was not fetched or does not match. Tuning on either side changes nothing until
  a new build is published and its manifest updated in a pull request. A candidate run from a local checkout instead
  (`workdir`) is recorded by its `version_cmd`.
- **Configs and manifests are recorded verbatim** in `session.json`, together with the machine, the power source and
  the shuffled run order.

## Candidates

### Mach-M5 (`candidates/mach_m5.json`)

A frozen binary (Apple Silicon, macOS; MLX / Metal; float32): FAS on L6, then pseudo-transient Newton-Krylov on L5 to
the finest level (GMRES with tangent-linear Jacobian products, a linear multigrid preconditioner with block ILU(0)
relaxation). Its Metal kernels are compiled by the driver on your machine at first use (~0.4 s per run).

`prepare --from DIR` needs the finest TMR level you will run (`wing_strct.2.ufmt`, or `.1`; unformatted PLOT3D as
written by the TMR grid generator): each coarser TMR level is every other node of the next finer one, so it derives
L3–L6 itself. It writes into `work/mach-m5/` here, never into your directory: links to your files, the derived
levels, wall distances (CPU: ~2.5 min for L3, roughly 30 min for L2 on an M5 Air) and FP32 metrics caches (~1.6 GB
in all), then runs a 10 s smoke test. A rerun skips what exists. For L3 only, use `--levels 3` and a config without
the L2 scenario. Memory: the L2 run peaks at ~11 GB.

### Adding yours

A manifest for a solver in a local checkout:

    {"name": "metal-cfd", "workdir": "/path/to/solver", "command": ["./bench_adapter"],
     "extra_args": [], "env": {}, "env_by_level": {}, "memory_gb": {"3": 8, "2": 40},
     "version_cmd": ["git", "rev-parse", "--short", "HEAD"]}

or, to let the other side run it too, a pinned build (a `.tar.gz`, e.g. a GitHub release asset):

    {"name": "metal-cfd", "artifact": {"name": "metal-cfd-1a2b3c4", "url": "https://.../metal-cfd-1a2b3c4.tar.gz",
                                       "sha256": "..."},
     "command": ["metal-cfd/bench_adapter"], "prepare": ["metal-cfd/prepare"], "env": {"DATA": "{bench}/work/metal-cfd"},
     "memory_gb": {"3": 8, "2": 40}, "version_cmd": ["cat", "metal-cfd/VERSION"]}

Paths in a manifest are relative to it; a pinned build's working directory is `artifacts/<name>/`, and `{bench}` in
`env` values is this directory. Pass it with `--add-candidate`, or add it to `candidates/` by pull request.

## Power measurement and sudo

| Channel | Source | Resolution | Needs |
|---|---|---|---|
| package (CPU + GPU + ANE), GPU residency and MHz, thermal pressure | `powermetrics` | 250 ms | sudo |
| whole system | SMC SystemLoad accumulator (`ioreg` PowerTelemetryData) | ~30 s batches | nothing |
| candidate's CPU energy, instructions, cycles | libproc `proc_pid_rusage` | 1 s | nothing |

Energy to a band = package energy integrated at 250 ms + (system − package) mean over the run × time. Without sudo the
system batches alone are used (fine for L2-length runs, coarse for runs of a few seconds). Without the sudoers rule,
`run` asks for your password once per session (in a terminal) and keeps the ticket alive; `--no-sudo` skips
powermetrics.

**Machine state matters:** close other applications, keep the display dim or off, and either run on battery (the
whole-system channel is then exact battery drain) or on mains with the battery full (charging power would count). The
report records the power source, battery level, low-power mode, the busiest processes before each run and the
memory state before and after. The harness holds a `caffeinate` assertion (no idle sleep) for the whole session;
a closed lid still sleeps the Mac: every run and idle baseline compares wall and monotonic time, and one during which
the Mac slept is repeated (references, baselines) or discarded from scoring. Keep the lid open and, for long sessions,
the Mac plugged in.

## What `run` does

A 65 s idle baseline (no load), one long **reference run** per candidate and scenario (seed 1000; C_ref, its
steadiness and drift), then every (candidate, scenario, seed) run in a shuffled order, each after a cool-down (until
package power is back near idle with thermal pressure nominal) and a 10 s idle baseline. A run ends when the forces
have stayed inside the tightest band long enough (PROTOCOL.md §5), at the time budget, or when the watchdog kills it
(memory above the manifest's limit, macOS memory pressure critical). Results: `OUT/report.md` (tables),
`OUT/summary.json` (all scores), `OUT/session.json` (provenance), `OUT/runs/*/run.json` (every record and sample),
`OUT/runs/*/stdout.log`, `stderr.log`. `python3 bench.py score OUT` re-scores from the recorded runs.

## Config keys

`candidates` (manifest paths, or {"manifest": path, "name": ..., "extra_args": [...]} variants), `scenarios` (name,
level, mach, alpha; optional max_seconds, reference_max_seconds), `seeds`, `perturb`, `bands`, `max_seconds`,
`setup_grace_s`, `sample_s`, `powermetrics_ms`, `session_idle_s`, `idle_s`, `cooldown` {min_s, max_s, settle_frac,
settle_add_w}, `hold` {records, frac}, `reference` {max_seconds, window, frac, steady_rel, seed}, `shuffle_seed`;
defaults in `bench.py`.

## Files

| File | |
|---|---|
| `PROTOCOL.md` | the candidate interface and the scoring rules |
| `bench.py` | fetch, prepare, runner, scorer, report |
| `sensors.py` | machine and process sensors (`python3 sensors.py` probes them) |
| `scoring.py` | band entry, reference forces, energy integration |
| `examples/toy_candidate.py` | a synthetic candidate (tests, template) |
| `candidates/`, `configs/` | manifests and session configs |
| `tests/test_harness.py` | scoring, energy, stop rule, powermetrics parsing, fetch, a toy session end to end |

Tests: `python3 -m pytest tests` (CI runs them on macOS for every pull request).
