"""Run experiments on Kaggle from this machine, via the official Kaggle CLI.

The CLI authenticates with ~/.kaggle/kaggle.json; this script never reads it.
Code reaches Kaggle as a private dataset built from the committed tree (plus a
COMMIT stamp the ledger records), so no GitHub token is involved.

    python scripts/kaggle_run.py push-code                 # upload HEAD as the code dataset
    python scripts/kaggle_run.py push-banks                # upload results/shadows for reuse
    python scripts/kaggle_run.py run --mode check          # ~10 min end-to-end check on a T4
    python scripts/kaggle_run.py run --mode full --seeds 42 43 44
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py logs                      # tail of the run's log
    python scripts/kaggle_run.py pull                      # download and merge results/

`run` pushes the code first if HEAD has not been uploaded yet, and the notebook
refuses to run if the dataset it sees is not that commit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "kaggle"
KAGGLE = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")


def settings() -> dict:
    s = yaml.safe_load((ROOT / "configs" / "kaggle.yaml").read_text(encoding="utf-8"))
    # another Kaggle account runs the same committed code without editing a tracked file
    if os.environ.get("POF_KAGGLE_USER"):
        s["user"] = os.environ["POF_KAGGLE_USER"]
    return s


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # UTF-8 mode: run logs contain progress-bar characters a Windows console code page lacks
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if check and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args[:2])} failed:\n{r.stdout}{r.stderr}")
    return r


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def head_commit() -> str:
    # results/ is output (pulling a run appends to the tracked ledger); only code counts
    if git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results"):
        sys.exit("commit your changes first: Kaggle runs only committed code")
    return git("rev-parse", "--short", "HEAD")


# ------------------------------------------------------------------ code dataset

def served_commit(ds_id: str) -> str | None:
    """The COMMIT stamp Kaggle currently serves for the code dataset.

    Kaggle unzips uploads and can report a new version "ready" while still
    serving the previous one, so the only reliable check is to fetch the stamp.
    """
    probe = BUILD / "probe"
    shutil.rmtree(probe, ignore_errors=True)
    r = kaggle("datasets", "download", ds_id, "-f", "Proof-of-Forgetting/COMMIT", "-p", str(probe),
               check=False)
    files = [f for f in probe.rglob("*") if f.is_file()] if r.returncode == 0 else []
    if not files:
        return None
    data = files[0].read_bytes()
    if data[:2] == b"PK":  # the CLI sometimes wraps single files in a zip
        with zipfile.ZipFile(files[0]) as z:
            data = z.read(z.namelist()[0])
    return data.decode("utf-8").strip()


def push_code() -> str:
    s, sha = settings(), head_commit()
    ds_id = f"{s['user']}/{s['dataset']}"
    out = BUILD / "dataset"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    archive = out / f"pof-code-{sha}.zip"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=Proof-of-Forgetting/",
                    "-o", str(archive), "HEAD"], cwd=ROOT, check=True)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("Proof-of-Forgetting/COMMIT", sha + "\n")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": s["dataset"], "id": ds_id, "licenses": [{"name": "other"}]}), encoding="utf-8")

    exists = kaggle("datasets", "status", ds_id, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(out), "-m", f"code {sha}")
    else:
        kaggle("datasets", "create", "-p", str(out))  # private unless --public is passed
    print(f"uploaded code {sha} to dataset {ds_id} (private); waiting until Kaggle serves it")
    for _ in range(90):
        if served_commit(ds_id) == sha:
            break
        time.sleep(10)
    else:
        sys.exit(f"Kaggle still serves {served_commit(ds_id)!r} instead of {sha} after 15 min")
    (BUILD / "pushed_commit").write_text(sha, encoding="utf-8")
    print("dataset ready")
    return sha


def push_banks() -> None:
    """Upload the local shadow banks (results/shadows) as a private dataset.

    Runs copy complete banks from any attached input and skip retraining them, so
    an experiment can be rerun against a bank that an earlier session trained.
    """
    s = settings()
    ds_id = f"{s['user']}/{s['banks_dataset']}"
    src = ROOT / "results" / "shadows"
    banks = [b for b in src.iterdir() if (b / "manifest.json").exists()] if src.exists() else []
    if not banks:
        sys.exit("no shadow banks under results/shadows")
    out = BUILD / "banks"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with zipfile.ZipFile(out / "banks.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for b in banks:
            for f in b.iterdir():
                z.write(f, f"results/shadows/{b.name}/{f.name}")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": s["banks_dataset"], "id": ds_id, "licenses": [{"name": "other"}]}), encoding="utf-8")
    exists = kaggle("datasets", "status", ds_id, check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(out), "-m", f"{len(banks)} banks")
    else:
        kaggle("datasets", "create", "-p", str(out))
    want = {f"results/shadows/{b.name}/manifest.json" for b in banks}
    for _ in range(90):
        listed = kaggle("datasets", "files", ds_id, "--page-size", "200", check=False).stdout
        if all(w in listed for w in want):
            break
        time.sleep(10)
    else:
        sys.exit("Kaggle does not list the uploaded banks after 15 min")
    (BUILD / "banks_pushed").write_text("\n".join(sorted(b.name for b in banks)), encoding="utf-8")
    print(f"uploaded {len(banks)} bank(s) to {ds_id} (private): " + ", ".join(b.name for b in banks))


# ------------------------------------------------------------------ notebook

def _cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


PARTS = ("clean", "canary", "power")


def build_notebook(mode: str, seeds: list[int], extra: list[str], commit: str,
                   parts: tuple[str, ...] = PARTS) -> dict:
    """The run notebook. Parameters are baked into its first code cell."""
    params = (f"MODE = {mode!r}\nSEEDS = {seeds!r}\nEXTRA = {extra!r}\n"
              f"EXPECTED_COMMIT = {commit!r}\n"
              "CANARY_FRACTIONS = [0.1, 0.25]  # tried in order by the canary pilot\n"
              f"PARTS = {list(parts)!r}  # full mode: clean (bank, E5, E0), power (E4), canary (E2, E3)\n")
    cells = [
        _cell("markdown", "# Proof of Forgetting run\nGenerated by `scripts/kaggle_run.py`; "
                          "edit that script, not this notebook."),
        _cell("code", params),
        _cell("code", '''
import glob, os, pathlib, shutil, subprocess, sys, time, zipfile
WORK = "/kaggle/working/Proof-of-Forgetting"
if not os.path.exists(WORK):
    extracted = glob.glob("/kaggle/input/**/Proof-of-Forgetting/COMMIT", recursive=True)
    archives = glob.glob("/kaggle/input/**/pof-code*.zip", recursive=True)
    if extracted:
        shutil.copytree(pathlib.Path(extracted[0]).parent, WORK)
    elif archives:
        zipfile.ZipFile(archives[0]).extractall("/kaggle/working")
    else:
        raise RuntimeError("code dataset not attached to this notebook")
os.chdir(WORK)
# attached CIFAR-10 instead of the slow download (md5-checked by torchvision)
batches = glob.glob("/kaggle/input/**/cifar-10-batches-py/data_batch_1", recursive=True)
if batches:
    os.environ["POF_DATA_MIRROR"] = str(pathlib.Path(batches[0]).parent.parent)
    print("data mirror:", os.environ["POF_DATA_MIRROR"])
COMMIT = pathlib.Path("COMMIT").read_text().strip()
print("running commit:", COMMIT)
assert COMMIT == EXPECTED_COMMIT, (f"dataset holds {COMMIT}, expected {EXPECTED_COMMIT}: "
                                   "the new dataset version was not live yet; run again")
'''),
        _cell("code", '''
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__)
assert torch.cuda.is_available(), "no GPU in this session"
NUM_GPUS = torch.cuda.device_count()
print(f"{NUM_GPUS} GPU(s):", [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "pytest"], check=True)
'''),
        _cell("code", '''
def sh(*cmd):
    """Run a command, stream its output, and stop the notebook if it fails."""
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise RuntimeError(f"command failed with exit code {r.returncode}")

sh(sys.executable, "-m", "pytest", "-q", "--runslow")
'''),
        _cell("code", '''
if MODE == "check":
    t0 = time.time()
    quick = ["fl.rounds=1", "unlearn.finetune_rounds=1", "unlearn.ga.recovery_rounds=1",
             "output.dir=results/check", "output.ledger=results/check/ledger.jsonl", *EXTRA]
    sh(sys.executable, "scripts/run_e0.py", "--config", "configs/e0_sanity.yaml", "--set", *quick)
    sh(sys.executable, "scripts/run_e2.py", "--config", "configs/e2_canary.yaml", "--set",
       *quick, "e2.epochs=1")
    sh(sys.executable, "scripts/run_e0.py", "--config", "configs/e3_dishonest.yaml", "--set", *quick)
    sh(sys.executable, "scripts/run_e4.py", "--config", "configs/e4_power.yaml", "--set", *quick,
       "e4.sizes=[25,100]", "e4.reps=20", "e4.n_perm=100")
    # the pilot's code path; one round cannot plant a backdoor, so "not planted" is expected
    r = subprocess.run([sys.executable, "scripts/check_canary.py", "--config",
                        "configs/e2_canary.yaml", "--set", *quick], capture_output=True, text=True)
    print("canary pilot (exit", r.returncode, "):", r.stdout.strip().splitlines()[-1]
          if r.stdout.strip() else r.stderr[-2000:])
    assert r.returncode in (0, 3), "canary pilot crashed"
    print(f"check run took {(time.time() - t0) / 60:.1f} min")
'''),
        _cell("code", '''
def parallel(script, config, sets, label):
    """Run `script --shard g/N` once per GPU, each pinned to its own GPU, and wait."""
    procs = []
    for g in range(NUM_GPUS):
        log = open(f"/kaggle/working/{label}_gpu{g}.log", "w")
        cmd = [sys.executable, script, "--config", config, "--shard", f"{g}/{NUM_GPUS}",
               "--set", *sets, f"fl.device=cuda:{g}"]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log.name))
    while any(p.poll() is None for p, _ in procs):
        time.sleep(120)
        for g, (_, name) in enumerate(procs):
            lines = open(name).read().strip().splitlines()
            print(f"[{label} gpu {g}] {lines[-1] if lines else '(starting)'}", flush=True)
    for p, name in procs:
        if p.returncode != 0:
            print(open(name).read()[-3000:])
            raise RuntimeError(f"{script} failed, see {name}")


def train_bank_parallel(seed, config, extra=None):
    extra = EXTRA if extra is None else extra
    parallel("scripts/run_shadows.py", config, [f"seed={seed}", *extra],
             f"shadows_{pathlib.Path(config).stem}_seed{seed}")


if MODE == "tune":
    # training variants on both GPUs, then E0 (no shadows) to check that every
    # unlearning method keeps a usable model
    for seed in SEEDS:
        parallel("scripts/tune.py", "configs/tune.yaml", [f"seed={seed}", *EXTRA], f"tune_seed{seed}")
        sh(sys.executable, "scripts/run_e0.py", "--config", "configs/e0_sanity.yaml",
           "--set", f"seed={seed}", *EXTRA)
        sh(sys.executable, "scripts/summarize.py", "--experiment", "e0_sanity", "--metrics",
           "test_acc", "v1_matched_auc", "v4_w_rel", "v5_p")

if MODE == "full":
    sys.path.insert(0, WORK)
    from pof.config import load_config, parse_overrides
    from pof.shadows import bank_dir
    for src in glob.glob("/kaggle/input/**/results/shadows/*/manifest.json", recursive=True):
        src_bank, n = pathlib.Path(src).parent, 0
        dst = pathlib.Path("results/shadows") / src_bank.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in src_bank.iterdir():
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
                n += 1
        print(f"resumed {n} files from {src_bank}")
    bank_of = lambda config, seed: bank_dir(load_config(config, parse_overrides([f"seed={seed}", *EXTRA])))

    def run_clean(seed):
        # shadow bank -> E5 (attack strength) -> E0 (all methods, all verifiers)
        train_bank_parallel(seed, "configs/e5_lira.yaml")
        sh(sys.executable, "scripts/run_e5.py", "--config", "configs/e5_lira.yaml",
           "--set", f"seed={seed}", *EXTRA)
        sh(sys.executable, "scripts/run_e0.py", "--config", "configs/e0_sanity.yaml",
           "--shadows", bank_of("configs/e5_lira.yaml", seed), "--set", f"seed={seed}", *EXTRA)

    def run_power(seed):
        # verifiability limit: V5 power against the client's data size
        sh(sys.executable, "scripts/run_e4.py", "--config", "configs/e4_power.yaml",
           "--set", f"seed={seed}", *EXTRA)

    def run_canary(seed):
        # a pilot picks a canary fraction that actually plants a backdoor (a
        # precondition for E2/E3, not a result); then one shared bank, E2 (canary
        # vs influence) and E3 (dishonest server)
        frac = None
        for f in CANARY_FRACTIONS:
            r = subprocess.run([sys.executable, "scripts/check_canary.py", "--config",
                                "configs/e2_canary.yaml", "--set", f"seed={seed}", *EXTRA,
                                f"data.canary.fraction={f}"], capture_output=True, text=True)
            print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:],
                  flush=True)
            if r.returncode == 0:
                frac = f
                break
        if frac is None:
            print(f"!!! seed {seed}: no canary fraction in {CANARY_FRACTIONS} planted a backdoor; "
                  "skipping E2 and E3 for this seed", flush=True)
        else:
            cx = [*EXTRA, f"data.canary.fraction={frac}"]
            canary_bank = bank_dir(load_config("configs/e2_canary.yaml",
                                               parse_overrides([f"seed={seed}", *cx])))
            train_bank_parallel(seed, "configs/e2_canary.yaml", cx)
            sh(sys.executable, "scripts/run_e2.py", "--config", "configs/e2_canary.yaml",
               "--shadows", canary_bank, "--set", f"seed={seed}", *cx)
            sh(sys.executable, "scripts/run_e0.py", "--config", "configs/e3_dishonest.yaml",
               "--shadows", canary_bank, "--set", f"seed={seed}", *cx)

    for seed in SEEDS:
        for part, fn in (("clean", run_clean), ("power", run_power), ("canary", run_canary)):
            if part in PARTS:
                t0 = time.time()
                fn(seed)
                print(f"seed {seed}: {part} done in {(time.time() - t0) / 60:.0f} min", flush=True)
    sh(sys.executable, "scripts/summarize.py", "--experiment", "e5_lira", "--metrics",
       "auc", "tpr@0.01", "tpr@0.001", "tpr@0.1", "tpr@0.05")
    sh(sys.executable, "scripts/summarize.py", "--experiment", "e0_sanity", "--metrics",
       "test_acc", "v1_pooled_auc", "v1_matched_auc", "v3_z", "v3_p", "v4_w_rel", "v5_p")
    sh(sys.executable, "scripts/summarize.py", "--experiment", "e3_dishonest", "--metrics",
       "test_acc", "v2_bsr", "v2_p", "v3_p", "v5_p", "nc_target_correct")
'''),
        _cell("code", '''
out = pathlib.Path("/kaggle/working/pof_results.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in pathlib.Path("results").rglob("*"):
        if p.is_file() and p.suffix != ".pt" and "smoke" not in p.parts:
            z.write(p, p.as_posix())
    for log in pathlib.Path("/kaggle/working").glob("*_gpu*.log"):
        z.write(log, f"kaggle_logs/{log.name}")
shutil.rmtree(WORK)  # keep the Output tab to the results zip and logs
print(out, f"{out.stat().st_size / 1e6:.1f} MB")
'''),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"}, "language_info": {"name": "python"}}}


def run(mode: str, seeds: list[int], extra: list[str], parts: tuple[str, ...] = PARTS) -> None:
    s = settings()
    pushed = BUILD / "pushed_commit"
    sha = head_commit()
    if not pushed.exists() or pushed.read_text(encoding="utf-8").strip() != sha:
        push_code()
    kdir = BUILD / "kernel"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    (kdir / "run.ipynb").write_text(json.dumps(build_notebook(mode, seeds, extra, sha, parts), indent=1),
                                    encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{s['user']}/{s['kernel']}", "title": s["kernel_title"], "code_file": "run.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": s["accelerator"],
        "dataset_sources": [f"{s['user']}/{s['dataset']}", *s.get("data_sources", []),
                            *([f"{s['user']}/{s['banks_dataset']}"]
                              if (BUILD / "banks_pushed").exists() else [])],
    }, indent=1), encoding="utf-8")
    r = kaggle("kernels", "push", "-p", str(kdir), "--accelerator", s["accelerator"])
    print(r.stdout.strip())
    print(f"started {mode} run of {sha} on {s['accelerator']}; "
          "check with: python scripts/kaggle_run.py status")


# ------------------------------------------------------------------ status / results

def kernel_id() -> str:
    s = settings()
    return f"{s['user']}/{s['kernel']}"


def status() -> None:
    print(kaggle("kernels", "status", kernel_id()).stdout.strip())


def logs() -> None:
    r = kaggle("kernels", "logs", kernel_id(), check=False)
    text = r.stdout
    try:  # the log is a JSON list of {stream_name, time, data}
        text = "".join(e.get("data", "") for e in json.loads(text))
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    print(text[-6000:] or r.stderr)


def merge_results(src: Path, dst: Path) -> dict:
    """Copy result files that are not present yet, and append ledger lines not already in
    the destination ledger. Returns counts."""
    counts = {"files": 0, "ledger_lines": 0}
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix == ".pt":  # checkpoints stay on Kaggle
            continue
        rel = p.relative_to(src)
        target = dst / rel
        if rel.name == "ledger.jsonl":
            target.parent.mkdir(parents=True, exist_ok=True)
            have = set(target.read_text(encoding="utf-8").splitlines()) if target.exists() else set()
            new = [line for line in p.read_text(encoding="utf-8").splitlines()
                   if line.strip() and line not in have]
            with open(target, "a", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in new)
            counts["ledger_lines"] += len(new)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            counts["files"] += 1
    return counts


def pull() -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dl = BUILD / "output" / stamp
    dl.mkdir(parents=True)
    kaggle("kernels", "output", kernel_id(), "-p", str(dl))
    archive = next(dl.rglob("pof_results.zip"), None)
    if archive is not None:
        with tempfile.TemporaryDirectory() as tmp:
            zipfile.ZipFile(archive).extractall(tmp)
            tmp = Path(tmp)
            if (tmp / "kaggle_logs").exists():
                merge_results(tmp / "kaggle_logs", ROOT / "results" / "kaggle_logs")
            counts = merge_results(tmp / "results", ROOT / "results")
    else:
        # the run failed before packaging; Kaggle still keeps the working directory
        raw = next(dl.rglob("Proof-of-Forgetting/results"), None)
        if raw is None:
            sys.exit(f"no results in the run output (see {dl})")
        print(f"no pof_results.zip (the run did not finish); recovering from {raw}")
        logs = ROOT / "results" / "kaggle_logs"
        for log in dl.glob("*_gpu*.log"):
            logs.mkdir(parents=True, exist_ok=True)
            if not (logs / log.name).exists():
                shutil.copy2(log, logs / log.name)
        counts = merge_results(raw, ROOT / "results")
    print(f"merged {counts['files']} files and {counts['ledger_lines']} ledger lines into results/")
    log = next(dl.rglob("*.log"), None)
    if log:
        print(f"full run log: {log}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push-code")
    sub.add_parser("push-banks")
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["check", "tune", "full"], required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[42])
    r.add_argument("--extra", nargs="*", default=[], help="config overrides, e.g. data.alpha=0.1")
    r.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS),
                   help="full mode: which experiments this session runs (each seed needs all three)")
    sub.add_parser("status")
    sub.add_parser("logs")
    sub.add_parser("pull")
    args = ap.parse_args(argv)
    if args.cmd == "push-code":
        push_code()
    elif args.cmd == "push-banks":
        push_banks()
    elif args.cmd == "run":
        run(args.mode, args.seeds, args.extra, tuple(args.parts))
    elif args.cmd == "status":
        status()
    elif args.cmd == "logs":
        logs()
    else:
        pull()


if __name__ == "__main__":
    main()
