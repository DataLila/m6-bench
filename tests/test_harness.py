"""The benchmark harness (standard library only): scoring rules, energy integration, the stop rule,
powermetrics parsing, and one end-to-end session with the toy candidate (CPU only, ~15 s)."""
import json
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402
import scoring as SC  # noqa: E402
import sensors as S  # noqa: E402

REF = dict(cl=0.25, cd=0.02, cm=-0.2)


def rec(t, err, level=3, it=0, **kw):
    return dict(type="iter", level=level, iter=it, t=t, **{c: v * (1 + err) for c, v in REF.items()}, **kw)


def test_band_entry_requires_staying_inside():
    errs = [1e-2, 5e-4, 2e-3, 5e-4, 5e-5, 5e-6, 4e-6]          # enters 1e-3 at index 1, leaves at 2, back at 3
    recs = [rec(i, e, it=i) for i, e in enumerate(errs)]
    assert SC.band_entry(recs, REF, 1e-3) == 3
    assert SC.band_entry(recs, REF, 1e-4) == 4
    assert SC.band_entry(recs, REF, 1e-5) == 5
    assert SC.band_entry(recs + [rec(7, 2e-5)], REF, 1e-5) is None   # last record outside: not reached
    one_coef = dict(rec(8, 0.0), cm=REF["cm"] * (1 + 2e-3))         # every coefficient counts
    assert SC.outside(one_coef, REF, 1e-3)


def test_reference_and_fine_records():
    recs = [rec(i, 1e-9 * (-1) ** i, it=i) for i in range(12)] + [rec(13, 0.5, level=4)]
    r = SC.reference(SC.fine(recs, 3), window=10)
    assert r["records"] == 12 and r["steadiness"] == pytest.approx(1e-9, rel=1e-3)
    assert r["ref"]["cl"] == pytest.approx(REF["cl"], rel=1e-12)
    assert len(SC.fine(recs, 4)) == 1


def test_integrate_and_counters():
    s = [(0.0, dict(w=10.0)), (1.0, dict(w=20.0)), (3.0, dict(w=20.0))]
    assert SC.integrate(s, "w", 0.0, 3.0) == pytest.approx(15 + 40)
    assert SC.integrate(s, "w", 0.5, 2.0) == pytest.approx(0.5 * (15 + 20) * 0.5 + 20 * 1.0)
    assert SC.integrate(s, "w", -1.0, 0.0) == pytest.approx(10.0)            # held before the first sample
    assert SC.integrate(s, "missing", 0, 1) is None
    c = [(0.0, dict(e=0.0)), (2.0, dict(e=4.0))]
    assert SC.value_at(c, "e", 1.0) == pytest.approx(2.0)


def test_smc_batches_give_mean_power():
    # 60 ticks of 0.5 s published at t = 30 (mean 8 W) and t = 60 (mean 12 W); samples every second
    samples = []
    for t in range(0, 70):
        if t < 30:
            acc, cnt = 0, 0
        elif t < 60:
            acc, cnt = 60 * 8000, 60
        else:
            acc, cnt = 60 * 8000 + 60 * 12000, 120
        samples.append((float(t), dict(load_acc=acc, load_count=cnt)))
    blocks = SC.system_blocks(samples)
    assert [(b0, b1, round(w, 3)) for b0, b1, w in blocks] == [(0.0, 30.0, 8.0), (30.0, 60.0, 12.0)]
    e, cov = SC.block_energy(blocks, 15.0, 45.0)
    assert e == pytest.approx(15 * 8 + 15 * 12) and cov == pytest.approx(1.0)


def test_score_run_energy_and_edp():
    recs = [dict(type="setup_done", t=1.0)] + [rec(1.0 + i, 10 ** -(i + 1), it=i + 1) for i in range(6)]
    pm = [(t / 4, dict(combined_w=10.0, gpu_w=8.0, cpu_w=2.0)) for t in range(0, 40)]
    run = dict(level=3, records=recs, t_setup=1.0, t_end=8.0, powermetrics=pm,
               system=[(0.0, dict(load_acc=0, load_count=0)), (9.0, dict(load_acc=18 * 12000, load_count=18))],
               proc=[(0.0, dict(cpu_energy_j=0.0)), (8.0, dict(cpu_energy_j=16.0))])
    out = SC.score_run(run, REF, [1e-3, 1e-5, 1e-8], dict(combined_w=2.0, system_w=4.0))
    b = out["0.001"]                                           # errors 1e-1, 1e-2, 1e-3 (= tol: inside), ...
    assert b["reached"] and b["iters"] == 3 and b["t"] == pytest.approx(3.0)
    assert b["pkg_j"] == pytest.approx(30.0)                   # 10 W x 3 s
    assert b["system_j"] == pytest.approx(30.0 + 2.0 * 3.0)    # rest = 12 W system - 10 W package
    assert b["system_above_idle_j"] == pytest.approx(36.0 - 4.0 * 3.0)
    assert b["edp_js"] == pytest.approx(36.0 * 3.0)
    assert b["t_loop"] == pytest.approx(2.0) and b["cpu_proc_j"] == pytest.approx(6.0)
    assert not out["1e-08"]["reached"]


def test_stop_rule_holds_the_tightest_band():
    st = bench.StopRule(3, ref=REF, tol=1e-5, hold=dict(records=3, frac=0.25))
    ts = [(1.0, 1e-3), (2.0, 1e-6), (2.1, 1e-6), (2.2, 1e-6), (2.3, 1e-6), (2.6, 1e-6)]
    fired = [st(rec(t, e), t) for t, e in ts]
    assert fired == [False, False, False, False, False, True]  # 3 records in, and 0.6 s >= 0.25 x 2.0 s
    assert not st(dict(type="iter", level=4, cl=1, cd=1, cm=1), 3.0)   # coarse records are ignored


def test_powermetrics_sample_parsing():
    d = dict(elapsed_ns=250_000_000, thermal_pressure="Nominal",
             processor=dict(cpu_power=1500.0, gpu_power=9000.0, ane_power=0.0, combined_power=10500.0),
             gpu=dict(freq_hz=1.2e9, idle_ratio=0.1))
    s = S.parse_powermetrics(plistlib.loads(plistlib.dumps(d)))
    assert s == dict(cpu_w=1.5, gpu_w=9.0, ane_w=0.0, combined_w=10.5, gpu_active=pytest.approx(0.9), gpu_mhz=1200.0,
                     thermal="Nominal")


def test_libproc_counters_of_this_process():
    r = S.proc_rusage(__import__("os").getpid())
    assert r is not None and r["footprint_gb"] > 0 and r["max_footprint_gb"] >= r["footprint_gb"]
    assert r["instructions"] > 0 and r["user_s"] > 0


def test_toy_session_end_to_end(tmp_path):
    cfg = json.loads((ROOT / "configs/toy.json").read_text())
    cfg["candidates"] = [str(ROOT / "candidates/toy.json")]
    cfg["seeds"] = [1, 2]
    p = tmp_path / "toy.json"
    p.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "bench.py"), "run", str(p), str(out), "--no-sudo", "--dev"], check=True,
                   capture_output=True, timeout=120)
    summ = json.loads((out / "summary.json").read_text())
    assert summ["refs"]["toy|L3-M0.84"]["steadiness"] < 1e-8
    rows = {r["band"]: r for r in summ["table"]}
    assert all(rows[b]["reached"] == 2 for b in (1e-3, 1e-4, 1e-5))
    assert rows[1e-3]["t"]["median"] < rows[1e-5]["t"]["median"]
    assert all(r["stop"] == "band held" for r in summ["runs"])
    report = (out / "report.md").read_text()
    assert report.startswith("# Benchmark report") and "Harness: " in report
    assert json.loads((out / "session.json").read_text())["harness"]["files"].keys() == set(bench.CODE)


def _artifact(tmp_path, monkeypatch):
    import hashlib
    import tarfile
    monkeypatch.setattr(bench, "ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "run.sh").write_text("echo hi\n")
    tgz = tmp_path / "pkg.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(tmp_path / "pkg", arcname="pkg")
    sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
    return dict(name="t", artifact=dict(name="pkg-1", url=tgz.as_uri(), sha256=sha))


def test_fetch_verifies_and_unpacks_a_pinned_build(tmp_path, monkeypatch):
    m = _artifact(tmp_path, monkeypatch)
    assert not bench.artifact_ok(m)
    bench.fetch(m)
    assert bench.artifact_ok(m) and (tmp_path / "artifacts/pkg-1/pkg/run.sh").exists()
    m2 = dict(m, artifact=dict(m["artifact"], name="pkg-2", sha256="0" * 64))
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        bench.fetch(m2)
    assert not (tmp_path / "artifacts/pkg-2").exists() and not bench.artifact_ok(m2)


def test_manifest_workdir_env_and_duplicate_names(tmp_path, monkeypatch):
    m = _artifact(tmp_path, monkeypatch)
    p = tmp_path / "m.json"
    p.write_text(json.dumps(dict(m, command=["x"], env={"D": "{bench}/work"})))
    lm = bench.load_manifest(p)
    assert lm["workdir"] == str(tmp_path / "artifacts/pkg-1")
    assert bench.cand_env(lm)["D"] == str(bench.BENCH / "work")
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"candidates": ["m.json"], "scenarios": []}))
    with pytest.raises(SystemExit, match="must differ"):
        bench.load_config(cfg, [p])


def test_provenance_names_the_measuring_code():
    h = bench.harness_provenance()
    assert set(h["files"]) == set(bench.CODE) and all(len(v) == 16 for v in h["files"].values())
