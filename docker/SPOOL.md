# Spool worker — selective tracking without ROS

`Dockerfile.spool` runs the same pipeline as the ROS image
(`GroundingDINO → ByteTrack → scene-graph mission filter → colour re-ID`) but
takes its work from a directory on a mounted volume instead of a camera topic.
Anything that can write a file can drive it.

## Build and run

```bash
docker build -f docker/Dockerfile.spool -t dains_perception-spool:latest .

docker run --rm --gpus all \
  -v /path/to/spool:/spool \
  -v /path/to/weights:/weights:ro \
  -e GDINO_PROMPT="red car." \
  dains_perception-spool:latest
```

No staging script is needed. The ROS image requires `docker/build_ros2.sh` to
vendor a pinned `perception_msgs` into the build context first; there are no
message definitions here, so a plain `docker build` is the whole story.

Or with compose: `docker compose -f docker/docker-compose.spool.yml up --build`.

## The spool contract

```
/spool/in       drop videos here (+ optional <stem>.json sidecar)
/spool/work     claimed, in flight
/spool/out      published results, one directory per job
/spool/failed   inputs that errored, beside a <name>.error.txt
```

All four are created at startup if absent. Recognised inputs are `.mp4`,
`.avi`, `.mov`, `.mkv`, `.m4v`, `.webm`; dotfiles are ignored.

A published job directory contains:

| File | What |
|---|---|
| `annotated.mp4` | Rendered video with track boxes and IDs |
| `tracks.txt` | MOT-format results — `frame,id,x,y,w,h,score,-1,-1,-1` |
| `result.json` | Prompt, arguments, timing, artifact list |
| `run.log` | Full stdout/stderr of the pipeline |
| `<original>.mp4` | The input, moved in so the directory is self-describing |

### Writing into the spool safely

**Write to a temp name and rename into `in/`.** A rename is atomic within a
filesystem, so the worker can never observe a half-copied file:

```bash
cp big.mp4 /path/to/spool/in/.staging-big.mp4
mv /path/to/spool/in/.staging-big.mp4 /path/to/spool/in/big.mp4
```

For writers that cannot do this, `SPOOL_SETTLE_SECONDS` (default 5) is the
fallback. A file is not claimed until its mtime has stopped moving for that long.
Raise it if a slow copy still gets picked up early.

### Reading results safely

Poll `/spool/out` for directories. Results are assembled under `out/.tmp-*` and
renamed into place, so a directory not starting with `.` is always complete — a
consumer never sees a partial job.

### Per-job overrides

A `<stem>.json` beside the video overrides the defaults for that job only:

```json
{
  "prompt": "red car behind the bus",
  "args": ["--box-threshold", "0.4", "--fp16"]
}
```

`args` is passed through verbatim to `demo/inference_w_worker.py` — see
`--help` there for the full set. Note it *replaces* `GDINO_EXTRA_ARGS` rather
than adding to it, so repeat `--fp16` if you want it. A malformed sidecar fails
the job rather than being ignored. A prompt that silently reverted to the
default would produce a plausible-looking answer to the wrong question.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SPOOL_DIR` | `/spool` | Root of the spool |
| `GDINO_WEIGHTS` | `/weights/groundingdino_swinb_cogcoor.pth` | Detector checkpoint |
| `GDINO_CONFIG` | `/app/groundingdino/config/GroundingDINO_SwinB_cfg.py` | Model config |
| `GDINO_PROMPT` | `red car.` | Default prompt |
| `GDINO_EXTRA_ARGS` | `--fp16` | Default extra pipeline flags |
| `SPOOL_POLL_SECONDS` | `2` | Idle poll interval |
| `SPOOL_SETTLE_SECONDS` | `5` | Quiet period before a file is claimed |
| `SPOOL_SCRATCH_DIR` | `/var/tmp/spool-work` | Frame extraction scratch |
| `SPOOL_ONCE` | unset | Drain the spool and exit, instead of watching |
| `SPOOL_REQUEUE_ORPHANS` | unset (off) | On startup, return `work/` leftovers to `in/` |

## Operational notes

**Weights are not in the image.** Mount them read-only. Without them every job
fails; the entrypoint warns at startup rather than waiting for the first job.

**GPU.** CUDA comes from conda-forge as a dependency of pytorch, so there is no
toolkit inside the image and only the host driver is needed — run with
`--gpus all`. The entrypoint reports what `torch.cuda.is_available()` says. On
CPU the pipeline is correct but very slow, and `--fp16` must be dropped
(`grid_sample` has no half-precision CPU kernel).

**One model load per job.** Each job starts a new `demo/inference_w_worker.py`
process, so the SwinB checkpoint and BERT encoder are loaded for every job --
tens of seconds before frame one. The cost is per-process, not per-prompt:
`Worker.__init__` calls `load_model` (`eval/worker_simple.py:160`), while
`text_prompt` is a plain attribute re-read on each `process_sequence`, so one
long-lived `Worker` can serve jobs with different prompts by reassigning it --
`groundingdino_node.py` already does this. For a spool of many short clips that
startup dominates; batch them into longer videos, or drive the pipeline
in-process rather than as a subprocess per job.

The `bert-base-uncased` files are cached in `/opt/hf-cache`, so they do not need
to be downloaded at runtime.

**Multiple workers** can safely claim jobs through atomic rename. Do not enable
`SPOOL_REQUEUE_ORPHANS` when multiple workers share a spool; a restarting worker
cannot distinguish orphaned work from another worker's active job, and would
yank it back to `in/` to be processed twice. Leave it off when scaling out and
reconcile `work/` by hand.

**Shutdown.** On `SIGTERM`, the worker stops the current job and returns its
input to `in/`. Compose uses a 60-second grace period; use `--stop-timeout 60`
with `docker run`. A forced kill may leave the input in `work/`. 
See `SPOOL_REQUEUE_ORPHANS` variable.

**File ownership.** The container runs as root by default, so results are
root-owned on the host. `docker run --user "$(id -u):$(id -g)"` works — the
environment lives at `/app/.pixi` with world-readable permissions, and the only
writes are to `/spool` and `SPOOL_SCRATCH_DIR`.
