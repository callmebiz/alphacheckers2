"""
Examine an AlphaCheckers MLflow run DB.

Usage:
    python scripts/examine_run.py [--db PATH] [--s3 BUCKET/KEY] [--experiment NAME]

Defaults to downloading mlflow-exp-full.db from S3 if --db is not provided.
"""
import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict


S3_DEFAULT = "alphacheckers-biz/mlflow-exp-full.db"


# ── helpers ──────────────────────────────────────────────────────────────────

def download_db(s3_path: str, dest: str) -> str:
    bucket, key = s3_path.split("/", 1)
    uri = f"s3://{bucket}/{key}"
    print(f"Downloading {uri} → {dest}")
    subprocess.run(["aws", "s3", "cp", uri, dest], check=True)
    return dest


def get_connection(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


# ── query helpers ─────────────────────────────────────────────────────────────

def list_experiments(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT experiment_id, name FROM experiments ORDER BY experiment_id"
    )]


def get_runs(con: sqlite3.Connection, experiment_id: str) -> list[dict]:
    return [dict(r) for r in con.execute(
        """SELECT run_uuid, status, start_time, end_time
           FROM runs WHERE experiment_id = ? ORDER BY start_time""",
        (experiment_id,)
    )]


def _metrics_pk(con: sqlite3.Connection) -> str:
    """MLflow schema uses run_uuid in older versions, run_id in newer ones."""
    cols = {c[1] for c in con.execute("PRAGMA table_info(metrics)").fetchall()}
    return "run_uuid" if "run_uuid" in cols else "run_id"


def _params_pk(con: sqlite3.Connection) -> str:
    cols = {c[1] for c in con.execute("PRAGMA table_info(params)").fetchall()}
    return "run_uuid" if "run_uuid" in cols else "run_id"


def get_metrics(con: sqlite3.Connection, run_id: str) -> dict[str, list[tuple[int, float]]]:
    """Returns {metric_key: [(step, value), ...]} sorted by step."""
    pk = _metrics_pk(con)
    rows = con.execute(
        f"SELECT key, step, value FROM metrics WHERE {pk} = ? ORDER BY key, step",
        (run_id,)
    ).fetchall()
    result: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        result[r["key"]].append((r["step"], r["value"]))
    return dict(result)


def get_params(con: sqlite3.Connection, run_id: str) -> dict[str, str]:
    pk = _params_pk(con)
    rows = con.execute(
        f"SELECT key, value FROM params WHERE {pk} = ?", (run_id,)
    ).fetchall()
    return {r["key"]: r["value"] for r in rows}


# ── display ───────────────────────────────────────────────────────────────────

def fmt_val(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def print_section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def print_metric_table(metrics: dict[str, list[tuple[int, float]]], prefix: str) -> None:
    keys = sorted(k for k in metrics if k.startswith(prefix))
    if not keys:
        print("  (none)")
        return
    for key in keys:
        vals = metrics[key]
        steps = [s for s, _ in vals]
        values = [v for _, v in vals]
        latest = values[-1]
        trend = ""
        if len(values) >= 2:
            delta = values[-1] - values[0]
            trend = f"  Δ={delta:+.4f}" if abs(delta) >= 0.0001 else "  (flat)"
        step_range = f"steps {steps[0]}–{steps[-1]}" if len(steps) > 1 else f"step {steps[0]}"
        print(f"  {key:<40s}  {fmt_val(latest):>10s}   [{step_range}, n={len(vals)}]{trend}")


def print_all_iters(metrics: dict[str, list[tuple[int, float]]], keys: list[str]) -> None:
    """Print a table of selected metrics across all iterations."""
    if not keys:
        return
    # gather all steps
    all_steps: set[int] = set()
    for k in keys:
        if k in metrics:
            all_steps.update(s for s, _ in metrics[k])
    if not all_steps:
        return

    # build lookup: {key: {step: val}}
    lookup: dict[str, dict[int, float]] = {}
    for k in keys:
        if k in metrics:
            lookup[k] = {s: v for s, v in metrics[k]}

    steps = sorted(all_steps)
    col_w = 10
    short_keys = [k.split("/")[-1][:col_w] for k in keys]

    header = f"  {'iter':>5}  " + "  ".join(f"{sk:>{col_w}}" for sk in short_keys)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for step in steps:
        row = f"  {step:>5}  "
        for k in keys:
            v = lookup.get(k, {}).get(step)
            row += f"  {fmt_val(v):>{col_w}}" if v is not None else f"  {'—':>{col_w}}"
        print(row)


def examine(db_path: str, experiment_name: str | None = None) -> None:
    con = get_connection(db_path)

    experiments = list_experiments(con)
    print_section("Experiments in DB")
    for e in experiments:
        print(f"  [{e['experiment_id']}] {e['name']}")

    # pick target experiment
    if experiment_name:
        target = next((e for e in experiments if e["name"] == experiment_name), None)
        if target is None:
            print(f"\nExperiment '{experiment_name}' not found.")
            sys.exit(1)
    else:
        # pick first non-default experiment or first
        target = next((e for e in experiments if e["name"] != "Default"), experiments[0])

    exp_id = target["experiment_id"]
    exp_name = target["name"]
    print(f"\nAnalysing experiment: [{exp_id}] {exp_name}")

    runs = get_runs(con, exp_id)
    if not runs:
        print("  No runs found.")
        return

    print(f"  {len(runs)} run(s) found")

    # merge all runs (for multi-segment spot runs each segment is a separate run)
    all_metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)
    all_params: dict[str, str] = {}
    for run in runs:
        rid = run["run_uuid"]
        m = get_metrics(con, rid)
        for k, vals in m.items():
            all_metrics[k].extend(vals)
        if not all_params:
            all_params = get_params(con, rid)

    # sort each metric by step
    for k in all_metrics:
        all_metrics[k].sort(key=lambda x: x[0])

    n_iters = max((s for vals in all_metrics.values() for s, _ in vals), default=0)
    print(f"  Iterations logged: {n_iters}")

    # ── params ────────────────────────────────────────────────────────────────
    if all_params:
        print_section("Config / Params (sample)")
        important = ["num_simulations", "num_iterations", "num_self_play_games",
                     "batch_size", "replay_buffer_size", "lr_milestones",
                     "dirichlet_alpha", "num_resblocks", "num_hidden"]
        for k in important:
            if k in all_params:
                print(f"  {k:<30s}  {all_params[k]}")

    # ── per-section summaries ─────────────────────────────────────────────────
    print_section("Loss metrics (latest values)")
    print_metric_table(all_metrics, "loss/")

    print_section("Self-play metrics (latest values)")
    print_metric_table(all_metrics, "selfplay/")

    print_section("Analysis metrics (latest values)")
    print_metric_table(all_metrics, "analysis/")

    print_section("System metrics (latest values)")
    print_metric_table(all_metrics, "system/")

    if any(k.startswith("eval/") for k in all_metrics):
        print_section("Eval metrics (latest values)")
        print_metric_table(all_metrics, "eval/")

    # ── iteration-by-iteration table ──────────────────────────────────────────
    print_section("Iteration table — key metrics per step")
    tracked = [
        "loss/policy", "loss/value", "loss/total",
        "selfplay/draw_rate", "selfplay/avg_game_length",
        "selfplay/top1_prob_mean",
        "system/iter_time_s", "system/selfplay_time_s",
        "system/train_time_s", "system/eta_hours",
        "analysis/value_mae",
    ]
    present = [k for k in tracked if k in all_metrics]
    print_all_iters(all_metrics, present)

    con.close()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Examine an AlphaCheckers MLflow DB")
    parser.add_argument("--db", help="Local path to mlflow .db file")
    parser.add_argument("--s3", default=S3_DEFAULT, help="S3 path (bucket/key) to download from")
    parser.add_argument("--experiment", help="Experiment name to analyse (default: first non-Default)")
    args = parser.parse_args()

    db_path = args.db
    if db_path is None:
        # check for already-downloaded copy next to the script or at project root
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "mlflow_exp-full.db"),
            os.path.join(os.path.dirname(__file__), "..", "mlflow-exp-full.db"),
        ]
        existing = next((p for p in candidates if os.path.exists(p)), None)
        if existing:
            db_path = os.path.abspath(existing)
            print(f"Using cached DB: {db_path}")
        else:
            tmp = tempfile.mktemp(suffix=".db")
            db_path = download_db(args.s3, tmp)

    examine(db_path, args.experiment)


if __name__ == "__main__":
    main()
