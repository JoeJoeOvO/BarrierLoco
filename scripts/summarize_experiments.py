#!/usr/bin/env python3
"""Summarize RL training experiments into Markdown reports."""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize RL experiments")
    parser.add_argument(
        "--logs-dir",
        default=os.path.expanduser("~/OmniBot/locolab/logs/rsl_rl/"),
        help="Root log directory",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/OmniBot/research/experiments/"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Only process matching run (format: group/timestamp_runname, supports substring match)",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Only generate summary.md, skip per-run detail reports",
    )
    return parser.parse_args()


def discover_experiments(logs_dir, run_filter=None):
    """Discover all experiment directories."""
    experiments = []
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}", file=sys.stderr)
        return experiments

    for group_dir in sorted(logs_path.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            full_key = f"{group_name}/{run_dir.name}"
            if run_filter and run_filter not in full_key:
                continue
            experiments.append((group_name, run_dir.name, run_dir))
    return experiments


# Regex to parse timestamp_runname directory names
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_(.+))?$")


def parse_experiment_meta(group, dirname, run_dir):
    """Extract metadata from an experiment directory."""
    m = _TS_RE.match(dirname)
    start_time = m.group(1).replace("_", " ", 1) if m else dirname
    run_name = m.group(2) if m and m.group(2) else ""

    has_export = (run_dir / "exported" / "policy.onnx").exists()

    # Find latest checkpoint
    max_ckpt = -1
    for f in run_dir.glob("model_*.pt"):
        try:
            n = int(f.stem.split("_")[1])
            max_ckpt = max(max_ckpt, n)
        except (IndexError, ValueError):
            pass

    return {
        "group": group,
        "dirname": dirname,
        "run_name": run_name,
        "start_time": start_time,
        "has_export": has_export,
        "max_ckpt": max_ckpt if max_ckpt >= 0 else None,
        "run_dir": run_dir,
    }


def parse_tensorboard(run_dir):
    """Parse TensorBoard events for key metrics."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[WARN] tensorboard not installed, skipping metrics", file=sys.stderr)
        return {}

    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {}

    print(f"  Loading TensorBoard: {run_dir.name}...", file=sys.stderr)
    ea = EventAccumulator(str(event_files[0]))
    try:
        ea.Reload()
    except Exception as e:
        print(f"  [WARN] Failed to load events: {e}", file=sys.stderr)
        return {}

    tags_of_interest = {
        "Train/mean_reward": "mean_reward",
        "Curriculum/terrain_levels": "terrain_levels",
        "Curriculum/lin_vel_cmd_levels": "lin_vel_cmd_levels",
        "Metrics/base_velocity/error_vel_xy": "error_vel_xy",
        "Episode_Termination/time_out": "time_out",
        "Episode_Termination/base_contact": "base_contact",
    }

    metrics = {}
    available_tags = ea.Tags().get("scalars", [])
    for tag, key in tags_of_interest.items():
        if tag not in available_tags:
            continue
        try:
            events = ea.Scalars(tag)
            if not events:
                continue
            values = [e.value for e in events]
            last_100 = values[-100:] if len(values) >= 100 else values
            final_avg = sum(last_100) / len(last_100)
            peak = max(values)
            metrics[key] = {"final": final_avg, "peak": peak}
        except Exception as e:
            print(f"  [WARN] Failed to read {tag}: {e}", file=sys.stderr)

    return metrics


def parse_rewards_from_yaml(run_dir):
    """Extract reward term names and weights from env.yaml."""
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.exists():
        return {}

    try:
        text = env_yaml.read_text()
    except Exception:
        return {}

    # Try yaml.safe_load first (will likely fail due to !!python tags)
    try:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict) and "rewards" in data:
            rewards = {}
            for name, cfg in data["rewards"].items():
                if isinstance(cfg, dict) and "weight" in cfg:
                    rewards[name] = cfg["weight"]
            if rewards:
                return rewards
    except Exception:
        pass

    # Fallback: regex-based extraction from the rewards section
    rewards = {}
    # Find the rewards: section and extract term_name + weight pairs
    in_rewards = False
    current_term = None
    for line in text.splitlines():
        stripped = line.rstrip()
        # Detect start of rewards section (top-level key, no leading spaces)
        if re.match(r"^rewards:\s*$", stripped):
            in_rewards = True
            continue
        if in_rewards:
            # Another top-level key ends the rewards section
            if re.match(r"^[a-z]", stripped) and not stripped.startswith(" "):
                break
            # Reward term name (2-space indent)
            m = re.match(r"^  ([a-z_]\w*):\s*$", stripped)
            if m:
                current_term = m.group(1)
                continue
            # Weight line under a term
            m = re.match(r"^\s+weight:\s+(.+)$", stripped)
            if m and current_term:
                try:
                    rewards[current_term] = float(m.group(1))
                except ValueError:
                    pass
                current_term = None

    return rewards


def parse_existing_notes(summary_path):
    """Parse existing summary.md to preserve Notes column content."""
    notes = {}  # key: "group/dirname" -> notes text
    if not summary_path.exists():
        return notes

    try:
        text = summary_path.read_text()
    except Exception:
        return notes

    current_group = None
    for line in text.splitlines():
        # Group header
        m = re.match(r"^## (.+)$", line)
        if m:
            current_group = m.group(1).strip()
            continue
        # Table row
        if current_group and line.startswith("|") and not line.startswith("|-"):
            cells = [c.strip() for c in line.split("|")]
            # cells[0] is empty (before first |), cells[-1] is empty (after last |)
            cells = [c for c in cells if c != ""]
            # Header row check
            if len(cells) >= 7 and cells[0] == "Run":
                continue
            if len(cells) >= 7:
                run_name = cells[0]
                note = cells[6] if len(cells) > 6 else ""
                # Reconstruct dirname from run_name and start time
                # We store notes by group + run_name for matching
                notes_key = f"{current_group}/{run_name}"
                if note:
                    notes[notes_key] = note

    return notes


def fmt(val, precision=2):
    """Format a numeric value or return N/A."""
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}"


def generate_detail_report(meta, metrics, rewards, output_dir):
    """Generate a per-experiment detail report."""
    group_dir = Path(output_dir) / meta["group"]
    group_dir.mkdir(parents=True, exist_ok=True)
    report_path = group_dir / f"{meta['dirname']}.md"

    title = meta["run_name"] or meta["start_time"]
    ckpt_str = f"model_{meta['max_ckpt']}.pt ({meta['max_ckpt']} iterations)" if meta["max_ckpt"] is not None else "N/A"
    exported = "✅" if meta["has_export"] else "❌"

    lines = [
        f"# {title}",
        "",
        f"- **Group**: {meta['group']}",
        f"- **Start**: {meta['start_time']}",
        f"- **Checkpoint**: {ckpt_str}",
        f"- **Exported**: {exported}",
        "",
        "## Key Metrics",
        "| Metric | Final (last 100 avg) | Peak |",
        "|--------|---------------------|------|",
    ]

    metric_rows = [
        ("Mean Reward", "mean_reward"),
        ("Terrain Level", "terrain_levels"),
        ("Lin Vel Cmd Level", "lin_vel_cmd_levels"),
        ("Vel Tracking Error", "error_vel_xy"),
        ("Timeout Rate", "time_out"),
        ("Base Contact Rate", "base_contact"),
    ]

    for label, key in metric_rows:
        if key in metrics:
            m = metrics[key]
            lines.append(f"| {label} | {fmt(m['final'])} | {fmt(m['peak'])} |")
        else:
            lines.append(f"| {label} | N/A | N/A |")

    lines.extend(["", "## Reward Configuration"])
    if rewards:
        lines.extend(["| Term | Scale |", "|------|-------|"])
        for term, scale in rewards.items():
            lines.append(f"| {term} | {scale} |")
    else:
        lines.append("_Could not parse reward configuration._")

    lines.extend(["", "## Notes", "", ""])

    report_path.write_text("\n".join(lines))
    return report_path


def generate_summary(all_data, output_dir, existing_notes):
    """Generate the summary.md overview table."""
    output_path = Path(output_dir) / "summary.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Experiment Summary", f"_Updated: {now}_", ""]

    # Group experiments
    groups = {}
    for meta, metrics, _ in all_data:
        groups.setdefault(meta["group"], []).append((meta, metrics))

    for group_name in sorted(groups.keys()):
        lines.append(f"## {group_name}")
        lines.append("")
        lines.append("| Run | Start | Iters | Final Reward | Terrain | Exported | Notes |")
        lines.append("|-----|-------|-------|-------------|---------|----------|-------|")

        # Sort by start_time descending (most recent first)
        runs = sorted(groups[group_name], key=lambda x: x[0]["start_time"], reverse=True)
        for meta, metrics in runs:
            run_label = meta["run_name"] or meta["dirname"]
            start = meta["start_time"][:16]  # trim seconds
            iters = str(meta["max_ckpt"]) if meta["max_ckpt"] is not None else "N/A"

            reward_final = "N/A"
            if "mean_reward" in metrics:
                reward_final = fmt(metrics["mean_reward"]["final"], 1)

            terrain_final = "N/A"
            if "terrain_levels" in metrics:
                terrain_final = fmt(metrics["terrain_levels"]["final"], 1)

            exported = "✅" if meta["has_export"] else "❌"

            # Preserve existing notes
            notes_key = f"{group_name}/{run_label}"
            note = existing_notes.get(notes_key, "")

            lines.append(
                f"| {run_label} | {start} | {iters} | {reward_final} | {terrain_final} | {exported} | {note} |"
            )

        lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path


def main():
    args = parse_args()

    experiments = discover_experiments(args.logs_dir, args.run)
    if not experiments:
        print("No experiments found.", file=sys.stderr)
        return

    print(f"Found {len(experiments)} experiment(s)", file=sys.stderr)

    # Parse existing notes before overwriting
    summary_path = Path(args.output_dir) / "summary.md"
    existing_notes = parse_existing_notes(summary_path)

    all_data = []
    for group, dirname, run_dir in experiments:
        print(f"Processing {group}/{dirname}...", file=sys.stderr)
        meta = parse_experiment_meta(group, dirname, run_dir)
        metrics = parse_tensorboard(run_dir)
        rewards = parse_rewards_from_yaml(run_dir)
        all_data.append((meta, metrics, rewards))

        if not args.no_detail:
            path = generate_detail_report(meta, metrics, rewards, args.output_dir)
            print(f"  -> {path}", file=sys.stderr)

    summary = generate_summary(all_data, args.output_dir, existing_notes)
    print(f"Summary -> {summary}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
