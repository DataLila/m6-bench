# CFD efficiency benchmark: candidate protocol (v0.1, draft)

The harness (`harness/bench.py`) runs every candidate solver on the same machine, one run at a time, with a cool-down
and an idle power baseline before each run, and measures time **and energy** to the force bands. A candidate is any
solver that accepts the command-line options below and prints the `@bench` lines below. Nothing else is required: the
harness never links against, imports or inspects a candidate.

## 1. The case

NASA TMR ONERA M6 (https://turbmodels.larc.nasa.gov), SA-neg RANS, Re = 14.6 million per root chord, TMR grid family
(L3 = 1,081,344 cells, L2 = 8,650,752 cells). Conditions (scenarios): M 0.84 α 3.06° (the TMR case, CFL3D reference
forces exist), M 0.83 α 2.5°, M 0.70 α 1.0°. Forces: C_L, C_D, C_M with the TMR reference values
(S_ref = 1.15315084119231, c_ref = 0.801672958512342, moment about the TMR reference point; state yours in `hello`).

## 2. Command line (what the harness calls)

    <candidate command> --level L --mach M --alpha A --seed S --perturb EPS --max-seconds T [extra args]

| Option | Meaning |
|---|---|
| `--level L` | finest grid level (3 or 2). Coarser levels may be used on the way (FMG, cascades). |
| `--mach M`, `--alpha A` | free-stream Mach number and angle of attack in degrees |
| `--seed S` | integer. `0`: the plain free-stream start. `S > 0`: perturb the starting state, see below |
| `--perturb EPS` | perturbation size (default 1e-6) |
| `--max-seconds T` | wall-time budget for the solve loop: stop on your own when it is spent |

The candidate starts from free stream (a coarse-grid start is allowed and counts in the time), then **keeps
iterating** — it must not stop on its own convergence test before `T`, except when it cannot make further progress
(say so in `done`). The harness stops it (SIGTERM to the process group, SIGKILL 10 s later) once the forces have
stayed inside the tightest band for a while (§5), so a candidate needs no stop rule of its own.

**Seeds.** With `S > 0` every conserved variable (and the turbulence variable) of every cell of the *first* state the
candidate creates is multiplied by `1 + EPS * u`, `u` uniform in [-1, 1], from a generator seeded with `S`. The
converged answer does not change; the *path* to it does. Iterative solvers can be sensitive to tiny differences
(one candidate's L2 time moved by 40 % between two runs that differed only in floating-point rounding), so the
harness reports the spread over seeds, not one lucky run.

## 3. Output (what the harness reads)

Lines on **stdout** that start with `@bench ` followed by one JSON object, flushed immediately. All other output
(stdout or stderr) is kept in the run's log files and ignored. The harness timestamps each line when it arrives
(its own monotonic clock, from process launch): that is the clock that is scored. A candidate may add its own `t`
(seconds) for cross-checking.

| `type` | When | Fields (required **bold**) |
|---|---|---|
| `hello` | first, optional | `name`, `version`, `precision` (`float32` / `float64` / mixed), `method` (free text: FAS levels and cycle, smoother, Krylov settings), `moment_ref` |
| `setup_done` | **once**, when grid reading and preprocessing (including any shader / kernel compilation done ahead of the loop) are finished and the solve loop starts | — |
| `iter` | **every** iteration, cycle or Newton step | **`level`**, **`iter`**, **`cl`**, **`cd`**, **`cm`**; optional `res` (density residual, any norm), `cdp`, `cdv`, `work` (work units, any definition: say which in `hello`) |
| `done` | optional, last | `reason` (`budget`, `no_progress`, …), optional final forces `cl`, `cd`, `cm`, `cdp`, `cdv` |

Example:

    @bench {"type": "hello", "name": "mach-m5", "version": "a879296", "precision": "float32", "method": "FAS L6, Newton-Krylov L5..L3"}
    @bench {"type": "setup_done"}
    @bench {"type": "iter", "level": 5, "iter": 1, "cl": 0.201, "cd": 0.0301, "cm": -0.142, "res": 12.1}
    ...
    @bench {"type": "iter", "level": 3, "iter": 14, "cl": 0.267121, "cd": 0.0171203, "cm": -0.18736}

Only `iter` records of the finest level (`level == L`) are scored; coarser ones are kept for the plots.

## 4. Reference forces

For each candidate and scenario the harness first makes one **reference run** (seed 1000, a perturbed start like every
other run, long budget; the plain start can keep exact ties that perturbed starts break and land on a different slow
path: one candidate's seed 0 converged 2e-5 away from every perturbed seed, which agreed among themselves to 1e-6).
C_ref is the mean over the last quarter of that run's time, reported with its steadiness (largest relative deviation
over that window) and drift per minute. A band tighter than about 10× that steadiness is reported as *not verifiable* for that candidate.
C_ref is each candidate's own converged answer (the bands measure iterative convergence); accuracy against CFL3D is a
separate table (§6).

## 5. Scoring

A **band** τ (1e-3, 1e-4, 1e-5) is reached at the first finest-level record from which on **every** later record has
|C − C_ref| ≤ τ |C_ref| for each of C_L, C_D and C_M. For each run and band the harness reports:

- **time** to the band (harness clock, from launch), split into setup (launch → `setup_done`) and solve loop;
- **energy** to the band (J = W·s, integrated from launch): whole-system power (macOS `ioreg` power telemetry: input
  power on mains, battery discharge on battery), and, when `sudo powermetrics` is available, CPU / GPU / ANE package
  power; each as **total** and **above idle** (the idle baseline measured just before the run);
- **energy × time** (J·s), so that running slowly at low power does not win by itself;
- iterations to the band (`iter` of the entry record).

The harness stops a band run once every coefficient has been inside the tightest band for at least 5 records and
for at least a quarter of the time it took to get there (the "stays inside" condition is verified up to that point).

Per candidate and scenario the report gives the median, minimum and maximum over seeds, and how many seeds reached
each band.

## 6. Everything else the harness records

Per run, sampled about once a second: whole-system power, CPU / GPU / ANE power, GPU active residency and frequency,
thermal pressure (with `sudo`); GPU utilisation (`ioreg` IOAccelerator); the process tree's physical footprint,
CPU time, instructions and cycles, per-process CPU energy, disk bytes read / written and page-ins; system memory
pressure level, free memory, compressor occupancy, compressions / decompressions, swap used, swap-ins / swap-outs.
Once per session: machine model, chip, memory, OS version, power source, low-power mode, candidate versions, the run
order (shuffled with a recorded seed). Converged forces against CFL3D (M 0.84 α 3.06 only) are listed per candidate.

## 7. Fairness rules

- One run at a time; nothing else running (the harness records the top processes before each run).
- Same machine for every candidate in a comparison; runs interleaved in a shuffled order.
- A candidate may use any algorithm, precision and hardware unit (CPU, GPU, ANE) on that machine.
- Candidates are built and configured by their authors; the harness only passes the options above (and any
  `extra args` fixed in the candidate's manifest for the whole session).
