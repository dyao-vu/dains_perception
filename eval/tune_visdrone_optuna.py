import optuna
import subprocess
import os
import re
import shutil
from datetime import datetime
import csv

EVAL_SCRIPT = "eval/eval_visdrone.py"
DATA_ROOT = "/isis/home/hasana3/vlmtest/GroundingDINO/dataset/visdrone_mot_format"
WEIGHTS = "/isis/home/hasana3/vlmtest/GroundingDINO/output/visdrone_swinb_finetune_no_backbone_training/checkpoint_best_regular.pth"

def run_eval(box_thresh, text_thresh, track_thresh, match_thresh, track_buffer,
             lambda_weight, text_sim_weight, *,
             tracker="bytetrack", devices="0,1", jobs=2,
             text_prompt="car. pedestrian.", fp16=True, show_output=False):

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    outdir = f"tmp_eval_out_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "python3", EVAL_SCRIPT,
        "--data_root", DATA_ROOT,
        "--split", "val",
        "--box_threshold", str(box_thresh),
        "--text_threshold", str(text_thresh),
        "--track_thresh", str(track_thresh),
        "--match_thresh", str(match_thresh),
        "--track_buffer", str(track_buffer),
        "--tracker", tracker,
        "--text_prompt", text_prompt,
        "--devices", devices,
        "--jobs", str(jobs),
        "--detector", "dino",
        "--outdir", outdir,
        "--weights", WEIGHTS,
    ]
    if fp16:
        cmd.append("--fp16")

    # Pass CLIP-specific knobs via tracker_kv ONLY (eval_new.py expects this)
    if tracker == "clip":
        cmd += ["--tracker_kv", f"lambda_weight={lambda_weight}"]
        cmd += ["--tracker_kv", f"text_sim_thresh={text_sim_weight}"]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    # Parse the machine-readable line printed by eval_new.py
    m_mota = re.search(r"OPTUNA:MOTA=([\-0-9.]+)", output)
    m_idf1 = re.search(r"IDF1=([\-0-9.]+)", output)

    mota = -1.0
    idf1 = float(m_idf1.group(1)) if m_idf1 else float("nan")

    if m_mota:
        mota = float(m_mota.group(1))
        print(f"[Trial] MOTA={mota:.3f} | IDF1={idf1:.3f} | "
              f"box={box_thresh:.3f}, text={text_thresh:.3f}, "
              f"track={track_thresh:.3f}, match={match_thresh:.3f}, buf={track_buffer}")
    else:
        print("[Trial] ⚠️  Failed to extract MOTA")
        if show_output:
            print(output)

    # Clean up output directory after trial
    try:
        shutil.rmtree(outdir)
    except Exception as e:
        print(f"Warning: Failed to delete temp dir {outdir} ({e})")

    return mota

def objective(trial):
    box_thresh   = trial.suggest_float("box_threshold", 0.05, 0.9)
    text_thresh  = trial.suggest_float("text_threshold", 0.05, 0.9)
    track_thresh = trial.suggest_float("track_thresh", 0.1, 0.9)
    match_thresh = trial.suggest_float("match_thresh", 0.1, 0.9)
    track_buffer = trial.suggest_int("track_buffer", 30, 300, step=10)
    lambda_weight   = trial.suggest_float("lambda_weight", 0.0, 0.5)
    text_sim_weight = trial.suggest_float("text_sim_weight", 0.0, 0.5)

    mota = run_eval(
        box_thresh, text_thresh, track_thresh, match_thresh, track_buffer,
        lambda_weight, text_sim_weight,
        tracker="bytetrack", devices="0,1", jobs=2, text_prompt="car. pedestrian.", fp16=True,
        show_output=True
    )
    if mota < 0:
        return 1.0  # penalize failed parse/run
    return 1.0 - mota  # maximize MOTA

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    print("\nBest hyperparameters:", study.best_params)
    print("Best MOTA score:", round(1.0 - study.best_value, 6))

    # ---- SAVE ALL TRIAL RESULTS TO CSV ----
    param_names = ["box_threshold", "text_threshold", "track_thresh", "match_thresh",
                   "track_buffer", "lambda_weight", "text_sim_weight"]
    all_trials = []
    with open("optuna_visdrone.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "MOTA"] + param_names)
        for trial in study.trials:
            mota = round(1.0 - trial.value, 6)
            row = [trial.number, mota] + [
                round(trial.params.get(p), 6) if isinstance(trial.params.get(p), float)
                else trial.params.get(p)
                for p in param_names
            ]
            writer.writerow(row)
            all_trials.append(row)

    print("Saved all trial results to optuna_trials_log.csv.")

    # ---- PRINT TOP 10 RESULTS ----
    all_trials.sort(key=lambda x: -x[1])  # sort descending by MOTA
    print("\nTop 10 trials by MOTA:")
    print("{:>5} | {:>7} | {:>10} | {:>10} | {:>12} | {:>12} | {:>13} | {:>15} | {:>16}".format(
        "Trial", "MOTA", "box_threshold", "text_threshold", "track_thresh",
        "match_thresh", "track_buffer", "lambda_weight", "text_sim_weight"))
    print("-" * 120)
    for row in all_trials[:10]:
        print("{:>5} | {:>7} | {:>10} | {:>10} | {:>12} | {:>12} | {:>13} | {:>15} | {:>16}".format(*row))
