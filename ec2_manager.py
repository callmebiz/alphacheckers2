#!/usr/bin/env python3
"""
AlphaCheckers EC2 Training Manager
====================================
Launches a spot instance, configures it, starts training, and automatically
relaunches on spot interruption  - resuming from the S3 checkpoint each time.

Usage:
    pip install paramiko
    python ec2_manager.py

MLflow is available at http://localhost:5000 while an instance is running.
Logs are written to ec2_manager.log alongside the console output.
Press Ctrl+C to stop the manager (does not terminate the running instance).
"""

import argparse
import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("Missing dependency  - run: pip install paramiko")
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────────

REGION         = "us-east-1"
INSTANCE_TYPE  = "c5.4xlarge"   # override with --instance-type
KEY_NAME       = "alphaCheckers"
KEY_PATH       = Path.home() / ".ssh" / "alphaCheckers.pem"
SECURITY_GROUP = "sg-01735e08ceeac8ae0"
IAM_PROFILE    = "AlphaCheckersEC2"
S3_BUCKET      = "alphacheckers-biz"
GITHUB_REPO    = "https://github.com/callmebiz/alphacheckers2"
TRAIN_CONFIG   = "medium"
WORKERS        = 12
EXPERIMENT     = "baseline"
MLFLOW_PORT        = 5000   # local port — auto-bumped if occupied; can differ per manager
MLFLOW_REMOTE_PORT = 5000   # port MLflow listens on inside the instance — always 5000
NUM_ITERATIONS = 300   # must match the config preset's num_iterations
NO_GATED       = False  # True → pass --no-gated to train.py (continuous mode)

# AZs tried in order  - reorder if one is consistently cheaper/available
AZS = ["us-east-1b", "us-east-1a", "us-east-1c", "us-east-1d", "us-east-1f"]

# How long to wait for S3 emergency upload after spot termination
TERMINATION_BUFFER_SECS = 30

# How often to check if the training screen is still alive
HEALTH_CHECK_INTERVAL   = 300   # seconds


# ── Logging ────────────────────────────────────────────────────────────────────
# Configured in main() after args are parsed so the log file is named per-experiment.

log  = logging.getLogger(__name__).info
warn = logging.getLogger(__name__).warning


def _setup_logging(experiment: str) -> None:
    """Configure root logger once, after we know the experiment name."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for handler in [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"ec2_manager_{experiment}.log"),
    ]:
        handler.setFormatter(fmt)
        root.addHandler(handler)


# ── AWS helpers ────────────────────────────────────────────────────────────────

def _aws(*args: str) -> str:
    """Run an AWS CLI command and return stdout. Raises on non-zero exit."""
    r = subprocess.run(["aws", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()


def _write_json_files() -> None:
    """Write spot.json and bdm.json without BOM (PowerShell Set-Content adds one)."""
    Path("spot.json").write_text('{"MarketType":"spot"}', encoding="utf-8")
    Path("bdm.json").write_text(
        '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]',
        encoding="utf-8",
    )


def _drain_shutting_down(timeout: int = 120) -> None:
    """
    Wait for any exp-tagged instances in shutting-down state to fully terminate.

    With a 32-vCPU spot quota and c6i.8xlarge consuming all 32, a new launch
    while the old instance is still shutting down will fail with
    MaxSpotInstanceCountExceeded.  Draining first ensures the quota is free.
    """
    deadline = time.time() + timeout
    logged = False
    while time.time() < deadline:
        try:
            ids = _aws(
                "ec2", "describe-instances",
                "--region", REGION,
                "--filters",
                f"Name=tag:AlphaCheckers-Experiment,Values={EXPERIMENT}",
                "Name=instance-state-name,Values=shutting-down",
                "--query", "Reservations[].Instances[].InstanceId",
                "--output", "text",
            )
        except RuntimeError:
            return  # can't check; proceed and let the launch fail if needed
        if not ids or ids.strip() in ("", "None"):
            return
        if not logged:
            log("Previous instance still shutting down — waiting for spot quota to free...")
            logged = True
        time.sleep(10)
    log("Timed out waiting for previous instance to terminate — proceeding anyway")


def launch_instance() -> str:
    """Try each AZ in order until a spot instance is successfully launched."""
    _write_json_files()
    _drain_shutting_down()  # ensure old instance freed the vCPU quota before launching
    ami = _aws(
        "ssm", "get-parameter",
        "--name", "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64",
        "--region", REGION,
        "--query", "Parameter.Value",
        "--output", "text",
    )
    for az in AZS:
        try:
            subnet = _aws(
                "ec2", "describe-subnets",
                "--region", REGION,
                "--filters", f"Name=availabilityZone,Values={az}",
                "--query", "Subnets[0].SubnetId",
                "--output", "text",
            )
            instance_id = _aws(
                "ec2", "run-instances",
                "--region", REGION,
                "--image-id", ami,
                "--instance-type", INSTANCE_TYPE,
                "--key-name", KEY_NAME,
                "--subnet-id", subnet,
                "--security-group-ids", SECURITY_GROUP,
                "--iam-instance-profile", f"Name={IAM_PROFILE}",
                "--instance-market-options", "file://spot.json",
                "--block-device-mappings", "file://bdm.json",
                "--tag-specifications",
                f"ResourceType=instance,Tags=[{{Key=Name,Value={EXPERIMENT}}},{{Key=AlphaCheckers-Experiment,Value={EXPERIMENT}}}]",
                "--count", "1",
                "--query", "Instances[0].InstanceId",
                "--output", "text",
            )
            log(f"Launched {instance_id} in {az}")
            return instance_id
        except RuntimeError as e:
            if "InsufficientInstanceCapacity" in str(e):
                log(f"No spot capacity in {az}  - trying next AZ")
            else:
                raise
    raise RuntimeError("No spot capacity available in any AZ")


def get_instance_state(instance_id: str) -> str:
    return _aws(
        "ec2", "describe-instances",
        "--instance-ids", instance_id,
        "--region", REGION,
        "--query", "Reservations[0].Instances[0].State.Name",
        "--output", "text",
    )


def terminate_instance(instance_id: str) -> None:
    try:
        _aws("ec2", "terminate-instances", "--instance-ids", instance_id, "--region", REGION)
        log(f"Terminated {instance_id}")
    except RuntimeError as e:
        warn(f"Could not terminate {instance_id}: {e}")


def wait_for_running(instance_id: str) -> str:
    """Block until the instance is running, return its public IP."""
    log("Waiting for instance to reach running state...")
    while True:
        state = get_instance_state(instance_id)
        if state == "running":
            ip = _aws(
                "ec2", "describe-instances",
                "--instance-ids", instance_id,
                "--region", REGION,
                "--query", "Reservations[0].Instances[0].PublicIpAddress",
                "--output", "text",
            )
            log(f"Instance running  - IP: {ip}")
            return ip
        if state in ("shutting-down", "terminated"):
            raise RuntimeError(f"Instance {instance_id} went {state} before reaching running")
        time.sleep(5)


def s3_checkpoint_info() -> dict | None:
    """Return the checkpoint_latest.json contents from S3, or None if absent.

    Keyed by EXPERIMENT (not TRAIN_CONFIG) so concurrent experiments with the
    same config preset don't read each other's checkpoints.
    """
    r = subprocess.run(
        ["aws", "s3", "cp",
         f"s3://{S3_BUCKET}/runs/{EXPERIMENT}/checkpoints/checkpoint_latest.json",
         "-"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def find_running_instance() -> tuple[str, str] | None:
    """
    Look for a running EC2 instance tagged AlphaCheckers-Experiment=EXPERIMENT.
    Returns (instance_id, public_ip) if found, else None.
    """
    try:
        raw = _aws(
            "ec2", "describe-instances",
            "--region", REGION,
            "--filters",
            f"Name=tag:AlphaCheckers-Experiment,Values={EXPERIMENT}",
            "Name=instance-state-name,Values=running",
            "--query", "Reservations[0].Instances[0].[InstanceId,PublicIpAddress]",
            "--output", "json",
        )
        data = json.loads(raw)
        if not data or data[0] is None:
            return None
        instance_id, ip = data[0], data[1]
        if instance_id and ip and instance_id != "None" and ip != "None":
            return instance_id, ip
        return None
    except (RuntimeError, json.JSONDecodeError, TypeError, IndexError):
        return None


# ── SSH helpers ────────────────────────────────────────────────────────────────

def ssh_connect(ip: str, retries: int = 24, interval: int = 15) -> paramiko.SSHClient:
    """Connect to the instance, retrying until SSH is available (~3 min budget)."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, retries + 1):
        try:
            client.connect(
                ip,
                username="ec2-user",
                key_filename=str(KEY_PATH),
                timeout=10,
                banner_timeout=60,
            )
            log("SSH connected")
            return client
        except Exception as e:
            log(f"SSH attempt {attempt}/{retries}  - {e}")
            time.sleep(interval)
    raise RuntimeError(f"Could not SSH into {ip} after {retries} attempts")


def ssh_run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> int:
    """
    Run a command on the remote host via a login shell, streaming output locally.
    Uses a heredoc so the command content never needs shell-escaping.
    Returns the exit code.
    """
    full_cmd = f"bash -l << 'HEREDOC'\n{cmd}\nHEREDOC\n"
    chan = client.get_transport().open_session()
    chan.get_pty()
    chan.settimeout(timeout)
    chan.exec_command(full_cmd)
    while True:
        if chan.recv_ready():
            print(chan.recv(4096).decode(errors="replace"), end="", flush=True)
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        time.sleep(0.05)
    return chan.recv_exit_status()


def ssh_run_bg(client: paramiko.SSHClient, cmd: str) -> None:
    """Run a command in a detached screen  - returns as soon as screen starts."""
    full_cmd = f"bash -l << 'HEREDOC'\n{cmd}\nHEREDOC\n"
    _, stdout, _ = client.exec_command(full_cmd)
    stdout.channel.recv_exit_status()


def training_screen_alive(client: paramiko.SSHClient) -> bool:
    """Return True if the training screen session is still running."""
    _, stdout, _ = client.exec_command("screen -ls 2>/dev/null | grep training")
    output = stdout.read().decode()
    return "training" in output


# ── MLflow tunnel ──────────────────────────────────────────────────────────────

_tunnel: subprocess.Popen | None = None


def find_free_port(start: int, max_tries: int = 20) -> int:
    """Return the first free local TCP port starting from `start`."""
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + max_tries - 1}")


def open_tunnel(ip: str) -> None:
    global _tunnel, MLFLOW_PORT
    close_tunnel()
    MLFLOW_PORT = find_free_port(MLFLOW_PORT)
    _tunnel = subprocess.Popen(
        [
            "ssh", "-i", str(KEY_PATH),
            "-L", f"{MLFLOW_PORT}:localhost:{MLFLOW_REMOTE_PORT}",
            "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            f"ec2-user@{ip}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log(f"MLflow tunnel open -> http://localhost:{MLFLOW_PORT}")
    log(f"  ssh -i {KEY_PATH} -L {MLFLOW_PORT}:localhost:{MLFLOW_REMOTE_PORT} -N ec2-user@{ip}")


def close_tunnel() -> None:
    global _tunnel
    if _tunnel and _tunnel.poll() is None:
        _tunnel.terminate()
        _tunnel = None


# ── Instance setup ─────────────────────────────────────────────────────────────

SETUP_CMD = (
    "wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && "
    "bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3 && "
    "$HOME/miniconda3/bin/conda init bash && "
    "source ~/.bashrc && "
    "sudo dnf install -y git screen && "
    f"git clone {GITHUB_REPO} alphacheckers2 && "
    "cd alphacheckers2 && "
    "$HOME/miniconda3/bin/pip install -r requirements.txt && "
    "chmod +x train.sh"
)

def _write_scripts(
    client: paramiko.SSHClient,
    resume_path: str | None,
    mlflow_run_id: str = "",
) -> None:
    """
    Write start scripts directly to the instance via SFTP.
    Avoids all shell quoting issues — the script content is plain text.
    """
    resume       = f"--resume {resume_path}" if resume_path else ""
    mlflow_flag  = f"--mlflow-run-id {mlflow_run_id}" if mlflow_run_id else ""
    gated_flag   = "--no-gated" if NO_GATED else ""

    training_script = (
        "#!/bin/bash\n"
        "set -o pipefail\n"
        "source ~/.bashrc\n"
        "cd ~/alphacheckers2\n"
        # --name sets config.name so local dirs and S3 paths are namespaced by
        # experiment rather than by the preset name. This is what lets two managers
        # using the same preset (e.g. "medium") coexist without clobbering each other.
        f"./train.sh --config {TRAIN_CONFIG} --workers {WORKERS} --iters {NUM_ITERATIONS}"
        f" --name {EXPERIMENT} --experiment {EXPERIMENT}"
        f" {resume}"
        f" {gated_flag}"
        f" --s3-bucket {S3_BUCKET}"
        f" {mlflow_flag}"
        " 2>&1 | tee training.log\n"
        "TRAIN_RC=${PIPESTATUS[0]}\n"
        "if [ $TRAIN_RC -eq 0 ]; then sudo shutdown -h now; fi\n"
        "exit $TRAIN_RC\n"
    )

    mlflow_script = (
        "#!/bin/bash\n"
        "source ~/.bashrc\n"
        "cd ~/alphacheckers2\n"
        "mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0\n"
    )

    sftp = client.open_sftp()
    for path, content in (
        ("/tmp/run_training.sh", training_script),
        ("/tmp/run_mlflow.sh",   mlflow_script),
    ):
        with sftp.open(path, "w") as f:
            f.write(content)
        sftp.chmod(path, 0o755)
    sftp.close()


def configure_instance(client: paramiko.SSHClient, resume_path: str | None, mlflow_run_id: str = "") -> None:
    log("Installing environment (3-5 min)...")
    code = ssh_run(client, SETUP_CMD, timeout=600)
    if code != 0:
        raise RuntimeError("Environment setup failed - check ec2_manager.log")

    if resume_path:
        log(f"Downloading checkpoint from S3: {resume_path}")
        remote_dir = f"~/alphacheckers2/runs/{EXPERIMENT}/checkpoints"
        dl = (
            f"mkdir -p {remote_dir} && "
            f"aws s3 cp s3://{S3_BUCKET}/runs/{EXPERIMENT}/checkpoints/checkpoint_latest.pt "
            f"{remote_dir}/checkpoint_latest.pt && "
            f"aws s3 cp s3://{S3_BUCKET}/runs/{EXPERIMENT}/checkpoints/checkpoint_latest.json "
            f"{remote_dir}/checkpoint_latest.json"
        )
        code = ssh_run(client, dl, timeout=300)
        if code != 0:
            warn("Checkpoint download failed - starting fresh")
            resume_path = None

    log("Downloading MLflow database from S3...")
    ssh_run(
        client,
        f"aws s3 cp s3://{S3_BUCKET}/mlflow-{EXPERIMENT}.db ~/alphacheckers2/mlflow.db 2>/dev/null"
        " || echo 'No mlflow.db in S3 yet - starting fresh'",
        timeout=60,
    )

    log("Writing start scripts to instance...")
    _write_scripts(client, resume_path, mlflow_run_id)

    # Start training first so it creates and migrates mlflow.db before the
    # MLflow UI server connects. If MLflow UI starts at the same time, both
    # race to run Alembic migrations on the same SQLite file and one gets
    # "database is locked", crashing the training screen immediately.
    log("Starting training screen...")
    ssh_run_bg(client, "screen -S training -dm /tmp/run_training.sh")

    # Give the trainer enough time to create and initialise mlflow.db (~10s
    # to import, create experiment, and release the schema write lock) before
    # the MLflow UI server connects as a reader.
    log("Waiting for MLflow DB to initialise before starting UI...")
    time.sleep(20)

    log("Starting MLflow screen...")
    ssh_run_bg(client, "screen -S mlflow -dm /tmp/run_mlflow.sh")

    # Verify training screen is still alive (would have exited by now if it crashed)
    time.sleep(3)
    _, chk, _ = client.exec_command("screen -ls 2>/dev/null | grep training")
    if "training" not in chk.read().decode():
        _, log_out, _ = client.exec_command(
            "tail -50 ~/alphacheckers2/training.log 2>/dev/null"
            " || echo '(training.log not found - screen may have failed before starting)'"
        )
        tail = log_out.read().decode().strip()
        raise RuntimeError(
            f"Training screen exited immediately. training.log tail:\n{tail}"
        )
    log("Training confirmed running in screen.")


# ── Post-training download ─────────────────────────────────────────────────────

def download_results() -> None:
    """
    Download the best model and MLflow DB for this experiment from S3 to local disk.

    Best model lands in two places:
      runs/{EXPERIMENT}/checkpoints/checkpoint_best.pt  — picked up by the UI server
      models/{arch}_{EXPERIMENT}_best.pt                — descriptive copy for easy ID

    MLflow DB lands at mlflow_{EXPERIMENT}.db.
    """
    import shutil

    log("Downloading training results from S3...")

    # ── best model ─────────────────────────────────────────────────────────
    ckpt_dir = Path(f"runs/{EXPERIMENT}/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    local_pt = ckpt_dir / "checkpoint_best.pt"

    r = subprocess.run(
        ["aws", "s3", "cp",
         f"s3://{S3_BUCKET}/runs/{EXPERIMENT}/checkpoints/checkpoint_best.pt",
         str(local_pt)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        log(f"Best model -> {local_pt}")
        # Sidecar JSON
        subprocess.run(
            ["aws", "s3", "cp",
             f"s3://{S3_BUCKET}/runs/{EXPERIMENT}/checkpoints/checkpoint_best.json",
             str(ckpt_dir / "checkpoint_best.json")],
            capture_output=True, text=True,
        )
        # Descriptive copy: res{N}h{H}_{experiment}_best.pt
        try:
            import torch
            data = torch.load(str(local_pt), map_location="cpu", weights_only=False)
            m = data.get("config", {}).get("model", {})
            arch = f"res{m.get('num_resblocks', '?')}h{m.get('num_hidden', '?')}"
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            friendly = models_dir / f"{arch}_{EXPERIMENT}_best.pt"
            shutil.copy2(local_pt, friendly)
            log(f"Model alias  -> {friendly}  [{arch}]")
        except Exception as e:
            warn(f"Could not create descriptive copy: {e}")
    else:
        warn(f"Best model download failed: {r.stderr.strip()}")

    # ── MLflow DB ──────────────────────────────────────────────────────────
    db_local = Path(f"mlflow_{EXPERIMENT}.db")
    r2 = subprocess.run(
        ["aws", "s3", "cp",
         f"s3://{S3_BUCKET}/mlflow-{EXPERIMENT}.db",
         str(db_local)],
        capture_output=True, text=True,
    )
    if r2.returncode == 0:
        log(f"MLflow DB    -> {db_local}")
        log(f"  Inspect: python scripts/dump_metrics.py --experiment {EXPERIMENT}")
    else:
        warn(f"MLflow DB download failed: {r2.stderr.strip()}")


# ── Monitor ────────────────────────────────────────────────────────────────────

def monitor(instance_id: str, client: paramiko.SSHClient) -> str:
    """
    Poll instance state and periodically check training health.
    Returns the final instance state when it transitions out of 'running'.
    """
    last_health_check = time.time()
    log("Monitoring (Ctrl+C to stop manager without terminating instance)...")

    while True:
        time.sleep(30)

        state = get_instance_state(instance_id)
        if state != "running":
            log(f"Instance -> {state}")
            return state

        # Periodic SSH health check
        if time.time() - last_health_check >= HEALTH_CHECK_INTERVAL:
            last_health_check = time.time()
            try:
                alive = training_screen_alive(client)
                if not alive:
                    warn("Training screen not found  - instance may have finished or crashed")
            except Exception:
                pass  # SSH blip; don't act on it


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaCheckers EC2 Training Manager")
    parser.add_argument("--experiment",     default=None, help=f"Experiment name — used as S3 key prefix and instance tag (default: {EXPERIMENT})")
    parser.add_argument("--config",         default=None, help=f"Training config preset (default: {TRAIN_CONFIG})")
    parser.add_argument("--workers",        type=int, default=None, help=f"Self-play worker count (default: {WORKERS})")
    parser.add_argument("--num-iters",      type=int, default=None, dest="num_iters",
                        help=f"Total training iterations (default: {NUM_ITERATIONS})")
    parser.add_argument("--instance-type",  default=None, dest="instance_type",
                        help=f"EC2 instance type (default: {INSTANCE_TYPE}). "
                             "Options by vCPU: c5.4xlarge=16, c5.9xlarge=36, c6i.8xlarge=32, "
                             "c5.18xlarge=72, c6a.16xlarge=64, c6a.24xlarge=96")
    parser.add_argument("--mlflow-port",    type=int, default=None, dest="mlflow_port",
                        help=f"Local port for the MLflow SSH tunnel (default: {MLFLOW_PORT}). Use different ports when running multiple managers.")
    parser.add_argument("--no-gated",       action="store_true", dest="no_gated",
                        help="Pass --no-gated to train.py: continuous mode where non-promotion never resets the model.")
    return parser.parse_args()


# ── Main loop ──────────────────────────────────────────────────────────────────

def _get_or_launch_instance(
    resume_path: str | None,
    mlflow_run_id: str,
) -> tuple[str, str, "paramiko.SSHClient"]:
    """
    If a running instance tagged for this experiment already exists, SSH into it
    and return (instance_id, ip, client) without installing anything.
    Otherwise launch a fresh spot instance, configure it fully, and return the
    same tuple.
    """
    existing = find_running_instance()
    if existing:
        instance_id, ip = existing
        log(f"Found running instance {instance_id} ({ip}) — attaching")
        client = ssh_connect(ip)
        return instance_id, ip, client

    # No running instance — launch and configure a new one
    try:
        instance_id = launch_instance()
    except RuntimeError as e:
        raise RuntimeError(f"Launch failed: {e}")

    ip = wait_for_running(instance_id)
    client = ssh_connect(ip)
    configure_instance(client, resume_path, mlflow_run_id)
    return instance_id, ip, client


def main() -> None:
    global EXPERIMENT, TRAIN_CONFIG, WORKERS, NUM_ITERATIONS, INSTANCE_TYPE, MLFLOW_PORT, NO_GATED

    args = parse_args()
    if args.experiment is not None:
        EXPERIMENT = args.experiment
    if args.config is not None:
        TRAIN_CONFIG = args.config
    if args.workers is not None:
        WORKERS = args.workers
    if args.num_iters is not None:
        NUM_ITERATIONS = args.num_iters
    if args.instance_type is not None:
        INSTANCE_TYPE = args.instance_type
    if args.mlflow_port is not None:
        MLFLOW_PORT = args.mlflow_port
    if args.no_gated:
        NO_GATED = True

    _setup_logging(EXPERIMENT)

    log("=== AlphaCheckers EC2 Training Manager ===")
    log(f"Experiment: {EXPERIMENT} | Config: {TRAIN_CONFIG} | Instance: {INSTANCE_TYPE} | Workers: {WORKERS} | Target: {NUM_ITERATIONS} iters | MLflow port: {MLFLOW_PORT}")

    run_number = 0

    while True:
        run_number += 1
        log("-" * 50)
        log(f"Run #{run_number}")

        # ── Check S3 for existing checkpoint ──────────────────────────────────
        ckpt_info = s3_checkpoint_info()
        if ckpt_info:
            s3_iter       = ckpt_info.get("iteration", -1)
            s3_promos     = ckpt_info.get("promotions", 0)
            s3_time       = ckpt_info.get("saved_at", "?")
            mlflow_run_id = ckpt_info.get("mlflow_run_id", "")
            log(f"S3 checkpoint: iter {s3_iter}/{NUM_ITERATIONS - 1}  promotions {s3_promos}  saved {s3_time}")
            if mlflow_run_id:
                log(f"Resuming MLflow run {mlflow_run_id}")
            if s3_iter >= NUM_ITERATIONS - 1:
                log("Training is already complete  - nothing to do.")
                download_results()
                return
            resume_path = f"runs/{EXPERIMENT}/checkpoints/checkpoint_latest.pt"
        else:
            log("No S3 checkpoint found  - starting fresh")
            resume_path   = None
            mlflow_run_id = ""

        # ── Get or launch instance ────────────────────────────────────────────
        try:
            instance_id, ip, client = _get_or_launch_instance(resume_path, mlflow_run_id)
        except RuntimeError as e:
            wait = 60
            warn(f"{e}  - retrying in {wait}s")
            time.sleep(wait)
            continue

        # ── MLflow tunnel ─────────────────────────────────────────────────────
        open_tunnel(ip)
        log(f"Open http://localhost:{MLFLOW_PORT} to monitor training")

        # ── Monitor until instance exits ──────────────────────────────────────
        try:
            monitor(instance_id, client)
        except KeyboardInterrupt:
            log("Manager stopped by user (instance still running)")
            close_tunnel()
            return
        finally:
            try:
                client.close()
            except Exception:
                pass

        close_tunnel()

        # ── Post-termination: wait for emergency S3 upload ────────────────────
        log(f"Waiting {TERMINATION_BUFFER_SECS}s for SIGTERM S3 upload to complete...")
        time.sleep(TERMINATION_BUFFER_SECS)

        # ── Check if training is done ─────────────────────────────────────────
        ckpt_info = s3_checkpoint_info()
        if ckpt_info:
            s3_iter = ckpt_info.get("iteration", -1)
            log(f"S3 checkpoint after termination: iter {s3_iter}/{NUM_ITERATIONS - 1}")
            if s3_iter >= NUM_ITERATIONS - 1:
                log("Training complete!")
                download_results()
                return
            log(f"Interrupted at iter {s3_iter}  - relaunching...")
        else:
            log("No S3 checkpoint (no promotion yet)  - relaunching fresh...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped.")
        close_tunnel()
