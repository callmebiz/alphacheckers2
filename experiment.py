#!/usr/bin/env python3
"""
experiment.py - AlphaCheckers experiment manager

  python experiment.py -ls                   list all experiments
  python experiment.py --kill NAME           kill NAME (confirms first)
  python experiment.py --kill NAME --yes     kill without prompt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── defaults (match ec2_manager.py) ───────────────────────────────────────────
REGION    = "us-east-1"
S3_BUCKET = "alphacheckers-biz"
EC2_TAG   = "AlphaCheckers-Experiment"
REPO_ROOT = Path(__file__).parent


# ── AWS helpers ────────────────────────────────────────────────────────────────

def _aws_json(*args):
    r = subprocess.run(
        ["aws", "--region", REGION, *args, "--output", "json"],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _aws_text(*args):
    r = subprocess.run(
        ["aws", "--region", REGION, *args, "--output", "text"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def _s3_experiments():
    """List experiment names that have a runs/ prefix in S3."""
    data = _aws_json(
        "s3api", "list-objects-v2",
        "--bucket", S3_BUCKET,
        "--prefix", "runs/",
        "--delimiter", "/",
        "--query", "CommonPrefixes[].Prefix",
    )
    if not data:
        return []
    return [p.replace("runs/", "").rstrip("/") for p in data]


def _s3_mlflow_dbs():
    """Return set of experiment names that have an mlflow DB in S3."""
    data = _aws_json(
        "s3api", "list-objects-v2",
        "--bucket", S3_BUCKET,
        "--prefix", "mlflow-",
        "--query", "Contents[].Key",
    )
    if not data:
        return set()
    names = set()
    for key in data:
        # mlflow-{experiment}.db
        if key.endswith(".db"):
            names.add(key[len("mlflow-"):-len(".db")])
    return names


def _s3_checkpoint_iter(experiment):
    """Download and parse checkpoint_latest.json; return iteration or None."""
    r = subprocess.run(
        ["aws", "--region", REGION, "s3", "cp",
         f"s3://{S3_BUCKET}/runs/{experiment}/checkpoints/checkpoint_latest.json",
         "-"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout).get("iteration")
    except (json.JSONDecodeError, AttributeError):
        return None


def _ec2_instances_by_experiment():
    """Return dict of experiment_name → list of (instance_id, state)."""
    data = _aws_json(
        "ec2", "describe-instances",
        "--filters", f"Name=tag-key,Values={EC2_TAG}",
        "--query",
        "Reservations[].Instances[].[InstanceId,State.Name,"
        f"Tags[?Key==`{EC2_TAG}`].Value|[0]]",
    )
    result = {}
    for item in (data or []):
        iid, state, exp = item[0], item[1], item[2]
        if state in ("terminated",):
            continue
        result.setdefault(exp, []).append((iid, state))
    return result


def _local_experiments():
    """Set of experiment names with a local runs/ directory."""
    runs = REPO_ROOT / "runs"
    if not runs.is_dir():
        return set()
    return {d.name for d in runs.iterdir() if d.is_dir()}


def _local_mlflow_dbs():
    """Map experiment_name → local mlflow DB path (if it exists)."""
    result = {}
    for p in REPO_ROOT.glob("mlflow*.db"):
        # mlflow_{exp}.db  or  mlflow-{exp}.db  or  mlflow.db
        stem = p.stem  # e.g. "mlflow_exp-continuous"
        if stem == "mlflow":
            continue
        if stem.startswith("mlflow_"):
            result[stem[len("mlflow_"):]] = p
        elif stem.startswith("mlflow-"):
            result[stem[len("mlflow-"):]] = p
    return result


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list():
    print("Fetching experiment info...\n")

    s3_exps   = set(_s3_experiments())
    s3_mldbs  = _s3_mlflow_dbs()
    ec2_map   = _ec2_instances_by_experiment()
    local_exp = _local_experiments()
    local_dbs = _local_mlflow_dbs()

    all_names = sorted(s3_exps | set(ec2_map) | local_exp | set(local_dbs))
    if not all_names:
        print("No experiments found.")
        return

    # header
    print(f"{'EXPERIMENT':<24} {'EC2':<22} {'ITER':>5}  {'S3':^3}  LOCAL")
    print("-" * 72)

    for name in all_names:
        instances = ec2_map.get(name, [])
        if instances:
            ec2_col = ", ".join(f"{iid}({st})" for iid, st in instances)
        else:
            ec2_col = "-"

        if name in s3_exps:
            itr = _s3_checkpoint_iter(name)
            s3_col  = "yes"
            iter_col = str(itr) if itr is not None else "-"
        else:
            s3_col   = "no"
            iter_col = "-"

        local_parts = []
        if name in local_exp:
            local_parts.append("runs/")
        if name in local_dbs:
            local_parts.append(local_dbs[name].name)
        local_col = " ".join(local_parts) if local_parts else "-"

        print(f"{name:<24} {ec2_col:<22} {iter_col:>5}  {s3_col:^3}  {local_col}")

    print()


# ── kill ──────────────────────────────────────────────────────────────────────

def cmd_kill(name: str, yes: bool):
    print(f"Gathering resources for experiment '{name}'...\n")

    ec2_map  = _ec2_instances_by_experiment()
    instances = ec2_map.get(name, [])
    s3_exps  = set(_s3_experiments())
    s3_mldb  = f"s3://{S3_BUCKET}/mlflow-{name}.db"
    s3_runs  = f"s3://{S3_BUCKET}/runs/{name}/"
    local_dbs = _local_mlflow_dbs()

    local_runs_dir = REPO_ROOT / "runs" / name
    local_log      = REPO_ROOT / f"ec2_manager_{name}.log"

    print("Will delete:")
    if instances:
        for iid, st in instances:
            print(f"  EC2  {iid}  ({st})")
    else:
        print("  EC2  (no running instance)")

    if name in s3_exps:
        print(f"  S3   {s3_runs}  (recursive)")
    if name in {db for db in _s3_mlflow_dbs()}:
        print(f"  S3   {s3_mldb}")
    if local_runs_dir.is_dir():
        print(f"  local  {local_runs_dir}")
    if name in local_dbs:
        print(f"  local  {local_dbs[name]}")
    if local_log.exists():
        print(f"  local  {local_log}")

    print()

    if not yes:
        ans = input(f"Confirm kill '{name}'? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    # ── EC2 ────────────────────────────────────────────────────────────────
    if instances:
        ids = [iid for iid, _ in instances]
        print(f"Terminating {ids}...")
        r = subprocess.run(
            ["aws", "--region", REGION, "ec2", "terminate-instances",
             "--instance-ids", *ids],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print("  terminated.")
        else:
            print(f"  ERROR: {r.stderr.strip()}")

    # ── S3 ─────────────────────────────────────────────────────────────────
    if name in s3_exps:
        print(f"Deleting {s3_runs}...")
        r = subprocess.run(
            ["aws", "--region", REGION, "s3", "rm", s3_runs, "--recursive"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print("  done.")
        else:
            print(f"  ERROR: {r.stderr.strip()}")

    # check again for mlflow DB specifically
    r = subprocess.run(
        ["aws", "--region", REGION, "s3", "ls", s3_mldb],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print(f"Deleting {s3_mldb}...")
        subprocess.run(
            ["aws", "--region", REGION, "s3", "rm", s3_mldb],
            capture_output=True, text=True,
        )
        print("  done.")

    # ── local ──────────────────────────────────────────────────────────────
    if local_runs_dir.is_dir():
        print(f"Removing {local_runs_dir}...")
        shutil.rmtree(local_runs_dir)
        print("  done.")

    if name in local_dbs:
        p = local_dbs[name]
        print(f"Removing {p}...")
        p.unlink()
        print("  done.")

    if local_log.exists():
        print(f"Removing {local_log}...")
        local_log.unlink()
        print("  done.")

    print(f"\nExperiment '{name}' wiped.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global S3_BUCKET, REGION

    ap = argparse.ArgumentParser(
        description="AlphaCheckers experiment manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-ls", "--list", action="store_true", help="List all experiments")
    ap.add_argument("--kill", metavar="NAME", help="Kill and delete experiment NAME")
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    ap.add_argument("--s3-bucket", default=S3_BUCKET, help="S3 bucket name")
    ap.add_argument("--region", default=REGION, help="AWS region")
    args = ap.parse_args()

    S3_BUCKET = args.s3_bucket
    REGION    = args.region

    if args.list:
        cmd_list()
    elif args.kill:
        cmd_kill(args.kill, args.yes)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
