"""Run Rough scratch training, then HighPlatform optimizer-preserving continuation.

Launch through scripts/skill.sh highplatform python so the local Isaac Lab and
RSL-RL packages are selected. Each stage must finish and save its final model
before the next stage is allowed to start. No author checkpoint is loaded.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--rough-iterations", type=int, default=5000)
    parser.add_argument("--highplatform-iterations", type=int, default=5000)
    parser.add_argument("--evaluate-final", action="store_true", help="Evaluate the final policy on fixed platforms.")
    args = parser.parse_args()
    job = args.job_dir.resolve()
    job.mkdir(parents=True, exist_ok=True)
    status_path = job / "status.json"
    if status_path.exists():
        raise RuntimeError("Job already exists; use a new directory to avoid duplicate training.")
    if min(args.num_envs, args.rough_iterations, args.highplatform_iterations) <= 0:
        raise ValueError("Environment and iteration counts must be positive")
    state = {"pid": os.getpid(), "started_at": datetime.now().isoformat(), "stages": [], "status": "starting"}
    child = None

    def save():
        state["updated_at"] = datetime.now().isoformat()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_path)

    def stop(signum, frame):
        # Stopping the supervisor also stops its active trainer; never advance stages.
        state["status"] = "stopped"
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        save()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    checkpoint = None
    try:
        # The native runner resumes at the stored iteration label (not label + 1).
        # Each stage still executes exactly its requested number of updates.
        for name, task, iterations, final_iteration in (
            ("rough", "OmniBot-Go2W-Rough-MLP-v0", args.rough_iterations, args.rough_iterations - 1),
            (
                "highplatform",
                "OmniBot-Go2W-HighPlatform-MLP-v0",
                args.highplatform_iterations,
                args.rough_iterations + args.highplatform_iterations - 2,
            ),
        ):
            experiment = f"barrierloco_{name}_two_stage_{job.name}"
            run_root = ROOT / "locolab/logs/rsl_rl" / experiment
            if run_root.exists():
                raise RuntimeError(f"Refusing to reuse experiment directory: {run_root}")
            command = [
                sys.executable,
                str(ROOT / "locolab/scripts/locomotion/train.py"),
                "--task",
                task,
                "--headless",
                "--num_envs",
                str(args.num_envs),
                "--max_iterations",
                str(iterations),
                "--seed",
                "42",
                "--device",
                "cuda:0",
                "--logger",
                "tensorboard",
                "--validate-finite",
                "--experiment_name",
                experiment,
                "--run_name",
                f"{name}_{args.num_envs}_seed42_{iterations}iter",
            ]
            if checkpoint is None:
                command += [
                    "--from-scratch",
                    "--highplatform-pretrain",
                    # Match the native HighPlatform dynamics without mutating shared Rough defaults.
                ]
            else:
                command += ["--resume", "--checkpoint", str(checkpoint)]
            stage = {
                "name": name,
                "command": command,
                "checkpoint_in": str(checkpoint) if checkpoint else None,
                "started_at": datetime.now().isoformat(),
                "log": str(job / f"{name}.log"),
            }
            state["stages"].append(stage)
            state["status"] = f"training_{name}"
            with Path(stage["log"]).open("wb") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                stage["pid"] = child.pid
                save()
                stage["exit_code"] = child.wait()
            if stage["exit_code"] != 0:
                raise RuntimeError(f"{name} exited with code {stage['exit_code']}; next stage cancelled")
            candidates = list(run_root.glob(f"*/model_{final_iteration}.pt"))
            if len(candidates) != 1:
                raise RuntimeError(f"Missing or ambiguous final {name} checkpoint")
            checkpoint = candidates[0]
            # Verify a complete checkpoint, not merely a filename left by a failed write.
            import torch

            model = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if model["iter"] != final_iteration or not model["optimizer_state_dict"]["state"]:
                raise RuntimeError("Final checkpoint has unexpected iteration or missing optimizer state")
            del model
            stage.update(
                checkpoint_out=str(checkpoint),
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                finished_at=datetime.now().isoformat(),
            )
            # Stage completion updates raw job status only; formal docs require validated task success.
            save()
        if args.evaluate_final:
            output = job / "final_evaluation"
            command = [
                sys.executable,
                str(ROOT / "locolab/scripts/locomotion/evaluate_highplatform.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--headless",
                "--device",
                "cuda:0",
            ]
            state["status"] = "evaluating"
            state["evaluation"] = {"command": command, "output": str(output)}
            with (job / "evaluation.log").open("wb") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state["evaluation"]["pid"] = child.pid
                save()
                state["evaluation"]["exit_code"] = child.wait()
            if state["evaluation"]["exit_code"] != 0:
                raise RuntimeError("Final evaluation failed; inspect evaluation.log")
        # Completion denotes process completion, not successful platform traversal.
        state["status"] = "completed"
        save()
    except Exception as error:
        state.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
