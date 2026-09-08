#!/usr/bin/env python3
"""
Spool-directory driver for the selective-tracking pipeline.

Use this instead of the ROS node. This watches a directory on a mounted volume,
runs `demo/inference_w_worker.py` over each video that appears, and publishes a
result directory per job.

    /spool/in       drop videos here (+ optional <stem>.json sidecar)
    /spool/work     claimed, in flight
    /spool/out      published results, one directory per job
    /spool/failed   inputs that errored, beside <stem>.error.txt

Because a spool directory is shared
with writers and readers that this process does not coordinate with:

  * **Claiming** is `os.rename(in/x, work/x)`.  Rename is atomic within a
    filesystem and fails for the loser, so two workers on one volume cannot take
    the same job -- no lock files, no leases.
  * **Publishing** builds the result in `out/.tmp-*` and renames it into place.
    A consumer polling `out/` therefore sees a job directory only once it is
    complete, never half-written.

Everything is environment-configured; see the ENV block in Dockerfile.spool.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import shlex
import traceback
from datetime import datetime, timezone
from pathlib import Path

SPOOL_DIR = Path(os.environ.get("SPOOL_DIR", "/spool"))
IN_DIR = SPOOL_DIR / "in"
WORK_DIR = SPOOL_DIR / "work"
OUT_DIR = SPOOL_DIR / "out"
FAILED_DIR = SPOOL_DIR / "failed"

SCRATCH_DIR = Path(os.environ.get("SPOOL_SCRATCH_DIR", "/var/tmp/spool-work"))

POLL_SECONDS = float(os.environ.get("SPOOL_POLL_SECONDS", "2"))
SETTLE_SECONDS = float(os.environ.get("SPOOL_SETTLE_SECONDS", "5"))

PIPELINE = Path(os.environ.get("SPOOL_PIPELINE", "/app/demo/inference_w_worker.py"))
# Derive the project root from the pipeline path unless explicitly configured.
APP_DIR = Path(os.environ.get("SPOOL_APP_DIR", str(PIPELINE.parent.parent)))
WEIGHTS = os.environ.get("GDINO_WEIGHTS", "/weights/groundingdino_swinb_cogcoor.pth")
CONFIG = os.environ.get(
    "GDINO_CONFIG", "/app/groundingdino/config/GroundingDINO_SwinB_cfg.py"
)
PROMPT = os.environ.get("GDINO_PROMPT", "red car.")
EXTRA_ARGS = shlex.split(os.environ.get("GDINO_EXTRA_ARGS", ""))

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Requeuing orphans is only safe with a single worker.
REQUEUE_ORPHANS = _env_flag("SPOOL_REQUEUE_ORPHANS", False)
RUN_ONCE = _env_flag("SPOOL_ONCE", False)


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[spool {stamp}] {msg}", flush=True)


class Stop(Exception):
    """Raised when shutdown interrupts a running job."""


_stop_requested = False


def _on_signal(signum, _frame):
    global _stop_requested
    _stop_requested = True
    log(
        f"signal {signal.Signals(signum).name} received; aborting current job "
        "and requeuing it"
    )


# Spool Operations


def ensure_dirs() -> None:
    for d in (IN_DIR, WORK_DIR, OUT_DIR, FAILED_DIR, SCRATCH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_settled(path: Path) -> bool:
    """Return True when the file has not changed for SETTLE_SECONDS."""
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age >= SETTLE_SECONDS


def candidates() -> list[Path]:
    if not IN_DIR.is_dir():
        return []
    found = [
        p
        for p in IN_DIR.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in VIDEO_SUFFIXES
    ]
    # Process the oldest inputs first.
    return sorted(found, key=lambda p: p.stat().st_mtime)


def claim(src: Path) -> Path | None:
    """Atomically move src into work; return None if the claim fails."""
    dest = WORK_DIR / src.name
    try:
        os.rename(src, dest)
    except OSError:
        return None
    return dest


def claim_sidecar(video_in_spool: Path, claimed: Path) -> Path | None:
    """Claim the optional JSON sidecar for a video."""
    sidecar = video_in_spool.with_suffix(".json")
    if not sidecar.is_file():
        return None
    dest = WORK_DIR / sidecar.name
    try:
        os.rename(sidecar, dest)
    except OSError:
        return None
    return dest


def read_sidecar(sidecar: Path | None) -> dict:
    """Read and validate per-job overrides from a JSON sidecar."""
    if sidecar is None:
        return {}
    try:
        data = json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sidecar.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{sidecar.name} must contain a JSON object")
    if "prompt" in data and not isinstance(data["prompt"], str):
        raise ValueError(f"{sidecar.name}: 'prompt' must be a string")
    if "args" in data and not (
        isinstance(data["args"], list) and all(isinstance(a, str) for a in data["args"])
    ):
        raise ValueError(f"{sidecar.name}: 'args' must be a list of strings")
    return data


def unique_job_dir(stem: str) -> Path:
    """Return an unused output directory name for the job."""
    candidate = OUT_DIR / stem
    n = 2
    while candidate.exists():
        candidate = OUT_DIR / f"{stem}-{n}"
        n += 1
    return candidate


# Job Execution


def run_job(video: Path, sidecar: Path | None) -> Path:
    """Run the pipeline over `video`; return the published job directory."""
    stem = video.stem
    overrides = read_sidecar(sidecar)
    prompt = overrides.get("prompt", PROMPT)
    extra = list(overrides.get("args", EXTRA_ARGS))

    staging = OUT_DIR / f".tmp-{stem}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    # Keep extracted frames on local scratch instead of the spool volume.
    scratch = SCRATCH_DIR / f"{stem}-{os.getpid()}"
    shutil.rmtree(scratch, ignore_errors=True)

    cmd = [
        sys.executable,
        str(PIPELINE),
        "--video", str(video),
        "--output", str(staging / "annotated.mp4"),
        "--config", CONFIG,
        "--weights", WEIGHTS,
        "--text-prompt", prompt,
        "--workdir", str(scratch),
        # Keep the work directory until MOT results are copied below.
        "--keep-frames",
        *extra,
    ]

    log(f"running {stem}: prompt={prompt!r} extra={extra}")
    started = time.time()
    with (staging / "run.log").open("w") as logfile:
        logfile.write(f"$ {shlex.join(cmd)}\n\n")
        logfile.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(APP_DIR), stdout=logfile, stderr=subprocess.STDOUT
        )
        # Poll rather than block, so a termination signal can interrupt a running inference job.
        while True:
            try:
                returncode = proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if _stop_requested:
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    shutil.rmtree(scratch, ignore_errors=True)
                    shutil.rmtree(staging, ignore_errors=True)
                    raise Stop()
    elapsed = time.time() - started

    if returncode != 0:
        tail = (staging / "run.log").read_text().splitlines()[-40:]
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"pipeline exited {returncode} after {elapsed:.1f}s\n" + "\n".join(tail)
        )

    # Copy the MOT results before removing scratch data.
    mot_src = scratch / "mot" / "0000.txt"
    if mot_src.is_file():
        shutil.copyfile(mot_src, staging / "tracks.txt")
    else:
        log(f"warning: {stem} produced no MOT file at {mot_src}")
    shutil.rmtree(scratch, ignore_errors=True)

    # Move the input and sidecar into the published job directory.
    os.rename(video, staging / video.name)
    if sidecar is not None:
        os.rename(sidecar, staging / sidecar.name)

    (staging / "result.json").write_text(
        json.dumps(
            {
                "job": stem,
                "status": "ok",
                "prompt": prompt,
                "extra_args": extra,
                "input": video.name,
                "weights": WEIGHTS,
                "config": CONFIG,
                "elapsed_seconds": round(elapsed, 2),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": sorted(p.name for p in staging.iterdir()),
            },
            indent=2,
        )
        + "\n"
    )

    published = unique_job_dir(stem)
    os.rename(staging, published)
    log(f"published {published} in {elapsed:.1f}s")
    return published


def fail_job(video: Path, sidecar: Path | None, error: str) -> None:
    """Move a failed input to failed/ and record the error."""
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    dest = FAILED_DIR / video.name
    n = 2
    while dest.exists():
        dest = FAILED_DIR / f"{video.stem}-{n}{video.suffix}"
        n += 1
    try:
        os.rename(video, dest)
        if sidecar is not None:
            os.rename(sidecar, dest.with_suffix(".json"))
    except OSError as exc:
        log(f"could not park failed input {video.name}: {exc}")
    dest.with_suffix(dest.suffix + ".error.txt").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n\n{error}\n"
    )
    log(f"FAILED {video.name} -> {dest}")


def requeue(video: Path, sidecar: Path | None) -> None:
    """Return an unfinished job to in/."""
    try:
        os.rename(video, IN_DIR / video.name)
        if sidecar is not None:
            os.rename(sidecar, IN_DIR / sidecar.name)
        log(f"requeued {video.name}")
    except OSError as exc:
        log(f"could not requeue {video.name}: {exc}")


def requeue_orphans() -> None:
    """Return leftover work items to in/ when orphan requeue is enabled."""
    if not WORK_DIR.is_dir():
        return
    leftovers = [p for p in WORK_DIR.iterdir() if p.is_file()]
    if not leftovers:
        return
    if not REQUEUE_ORPHANS:
        log(
            f"{len(leftovers)} file(s) in {WORK_DIR} left by an earlier run; "
            "not requeuing (set SPOOL_REQUEUE_ORPHANS=1 if this is the only "
            "worker on this volume)"
        )
        return
    for p in leftovers:
        try:
            os.rename(p, IN_DIR / p.name)
            log(f"requeued orphan {p.name}")
        except OSError as exc:
            log(f"could not requeue orphan {p.name}: {exc}")


# Main Loop


def main() -> int:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    ensure_dirs()
    log(f"watching {IN_DIR}")
    log(f"prompt default {PROMPT!r}, extra args {EXTRA_ARGS}")
    requeue_orphans()

    idle_announced = False
    while True:
        if _stop_requested:
            log("stopping")
            return 0

        ready = [p for p in candidates() if is_settled(p)]
        if not ready:
            if RUN_ONCE:
                log("spool drained; SPOOL_ONCE is set, exiting")
                return 0
            if not idle_announced:
                log("idle")
                idle_announced = True
            time.sleep(POLL_SECONDS)
            continue

        idle_announced = False
        for src in ready:
            if _stop_requested:
                break
            claimed = claim(src)
            if claimed is None:
                continue  # another worker took it
            sidecar = claim_sidecar(src, claimed)
            try:
                run_job(claimed, sidecar)
            except Stop:
                # Requeue an interrupted job instead of marking it failed.
                requeue(claimed, sidecar)
                log("stopped")
                return 0
            except Exception:
                fail_job(claimed, sidecar, traceback.format_exc())


if __name__ == "__main__":
    sys.exit(main())
