# CFD efficiency benchmark harness

Runs CFD solvers ("candidates") one at a time on one machine and measures **time and energy to force-convergence
bands**, plus everything else the machine reports (memory pressure, swap, compression, disk, GPU utilisation, thermal
state). The interface a solver must implement is in [PROTOCOL.md](PROTOCOL.md): five command-line options and a few
`@bench` JSON lines on stdout. Python 3.9+ standard library only (macOS's `/usr/bin/python3` is enough).

## Quick start

    python3 harness/bench.py probe                                   # what this machine exposes
    python3 harness/bench.py run harness/configs/toy.json /tmp/toy   # 30 s self-test with the toy candidate
    python3 harness/bench.py run harness/configs/challenge.json OUT  # the real thing
    python3 harness/bench.py score OUT                               # re-score / re-write OUT/report.md

`run` does, in order: a 65 s idle baseline (no load), one long **reference run** per candidate and scenario
(seed 0; C_ref and its steadiness), then every (candidate, scenario, seed) run in a shuffled order, each after a
cool-down (≥ 60 s and until package power is back near idle with thermal pressure nominal) and a 10 s idle baseline.
A run ends when the forces have stayed inside the tightest band long enough (PROTOCOL.md §5), at the time budget, or
when the watchdog kills it (memory above the manifest's limit, macOS memory pressure critical). Results:
`OUT/report.md` (tables), `OUT/summary.json` (all scores), `OUT/runs/*/run.json` (every record and sample),
`OUT/runs/*/stdout.log`, `stderr.log`.

## Power measurement and sudo

| Channel | Source | Resolution | Needs |
|---|---|---|---|
| package (CPU + GPU + ANE), GPU residency and MHz, thermal pressure | `powermetrics` | 250 ms | sudo |
| whole system | SMC SystemLoad accumulator (`ioreg` PowerTelemetryData) | ~30 s batches | nothing |
| candidate's CPU energy, instructions, cycles | libproc `proc_pid_rusage` | 1 s | nothing |

Energy to a band = package energy integrated at 250 ms + (system − package) mean over the run × time. Without sudo the
system batches alone are used (fine for L2-length runs, coarse for runs of a few seconds).

**Convenient sudo:** `run` asks for your password once per session (in a terminal) and keeps the ticket alive. To
never be asked, allow `powermetrics` alone without a password:

    python3 harness/bench.py sudoers          # prints the one-line rule
    sudo visudo -f /etc/sudoers.d/cfd-bench-powermetrics    # paste it; delete the file to undo

`--no-sudo` skips powermetrics.

**Machine state matters:** close other applications, keep the display dim or off, and either run on battery (the
whole-system channel is then exact battery drain) or on mains with the battery full (charging power would count). The
report records the power source, battery level, low-power mode, the busiest processes before each run and the
memory state before and after.

## Adding a candidate

Write a manifest (see `candidates/*.json`):

    {"name": "metal-cfd", "workdir": "/path/to/solver", "command": ["./bench_adapter"],
     "extra_args": [], "env": {}, "env_by_level": {}, "memory_gb": {"3": 8, "2": 40},
     "version_cmd": ["git", "rev-parse", "--short", "HEAD"]}

and list it in the config's `candidates`. `examples/toy_candidate.py` is a complete minimal adapter.

## Config keys

`candidates` (manifest paths, or {"manifest": path, "name": ..., "extra_args": [...]} variants), `scenarios` (name, level, mach, alpha; optional max_seconds, reference_max_seconds), `seeds`, `perturb`, `bands`, `max_seconds`, `setup_grace_s`,
`sample_s`, `powermetrics_ms`, `session_idle_s`, `idle_s`, `cooldown` {min_s, max_s, settle_frac, settle_add_w},
`hold` {records, frac}, `reference` {max_seconds, window, steady_rel}, `shuffle_seed`; defaults in `bench.py`.

## Files

| File | |
|---|---|
| `PROTOCOL.md` | the candidate interface and the scoring rules |
| `bench.py` | runner, scorer, report |
| `sensors.py` | machine and process sensors (`python3 harness/sensors.py` probes them) |
| `scoring.py` | band entry, reference forces, energy integration |
| `examples/toy_candidate.py` | a synthetic candidate (tests, template) |
| `candidates/`, `configs/` | manifests and session configs |

Tests: `tests/test_harness.py` (scoring, energy, stop rule, powermetrics parsing, a toy session end to end).
