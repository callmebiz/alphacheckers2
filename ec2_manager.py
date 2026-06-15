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

import json
import logging
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
INSTANCE_TYPE  = "c5.4xlarge"
KEY_NAME       = "alphaCheckers"
KEY_PATH       = Path.home() / ".ssh" / "alphaCheckers.pem"
SECURITY_GROUP = "sg-01735e08ceeac8ae0"
IAM_PROFILE    = "AlphaCheckersEC2"
S3_BUCKET      = "alphacheckers-biz"
GITHUB_REPO    = "https://github.com/callmebiz/alphacheckers2"
TRAIN_CONFIG   = "medium"
WORKERS        = 12
EXPERIMENT     = "baseline"
MLFLOW_PORT    = 5000
NUM_ITERATIONS = 150   # must match the config preset's num_iterations

# AZs tried in order  - reorder if one is consistently cheaper/available
AZS = ["us-east-1b", "us-east-1a", "us-east-1c", "us-east-1d", "us-east-1f"]

# How long to wait for S3 emergency upload after spot termination
TERMINATION_BUFFER_SECS = 120

# How often to check if the training screen is still alive
HEALTH_CHECK_INTERVAL   = 300   # seconds


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ec2_manager.log"),
    ],
)
log = logging.getLogger(__name__).info
warn = logging.getLogger(__name__).warning


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


def launch_instance() -> str:
    """Try each AZ in order until a spot instance is successfully launched."""
    _write_json_files()
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
    """Return the checkpoint_latest.json contents from S3, or None if absent."""
    r = subprocess.run(
        ["aws", "s3", "cp",
         f"s3://{S3_BUCKET}/runs/{TRAIN_CONFIG}/checkpoints/checkpoint_latest.json",
         "-"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
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


def open_tunnel(ip: str) -> None:
    global _tunnel
    close_tunnel()
    _tunnel = subprocess.Popen(
        [
            "ssh", "-i", str(KEY_PATH),
            "-L", f"{MLFLOW_PORT}:localhost:{MLFLOW_PORT}",
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

MLFLOW_SCREEN = (
    "screen -S mlflow -dm bash -c "
    "'source ~/.bashrc && cd ~/alphacheckers2 && "
    "mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0'"
)


def training_screen(resume_path: str | None) -> str:
    resume = f"--resume {resume_path}" if resume_path else ""
    return (
        "screen -S training -dm bash -c "
        f"'source ~/.bashrc && cd ~/alphacheckers2 && "
        f"./train.sh --config {TRAIN_CONFIG} --workers {WORKERS} "
        f"{resume} --experiment {EXPERIMENT} "
        f"--s3-bucket {S3_BUCKET} && sudo shutdown -h now'"
    )


def configure_instance(client: paramiko.SSHClient, resume_path: str | None) -> None:
    log("Installing environment (3-5 min)...")
    code = ssh_run(client, SETUP_CMD, timeout=600)
    if code != 0:
        raise RuntimeError("Environment setup failed - check ec2_manager.log")

    if resume_path:
        log(f"Downloading checkpoint from S3: {resume_path}")
        remote_dir = f"~/alphacheckers2/runs/{TRAIN_CONFIG}/checkpoints"
        dl = (
            f"mkdir -p {remote_dir} && "
            f"aws s3 cp s3://{S3_BUCKET}/runs/{TRAIN_CONFIG}/checkpoints/checkpoint_latest.pt "
            f"{remote_dir}/checkpoint_latest.pt && "
            f"aws s3 cp s3://{S3_BUCKET}/runs/{TRAIN_CONFIG}/checkpoints/checkpoint_latest.json "
            f"{remote_dir}/checkpoint_latest.json"
        )
        code = ssh_run(client, dl, timeout=300)
        if code != 0:
            warn("Checkpoint download failed - starting fresh")
            resume_path = None

    log("Downloading MLflow database from S3...")
    ssh_run(
        client,
        f"aws s3 cp s3://{S3_BUCKET}/mlflow.db ~/alphacheckers2/mlflow.db"
        " || echo 'No mlflow.db in S3 yet - starting fresh'",
        timeout=60,
    )

    log("Starting MLflow screen...")
    ssh_run_bg(client, MLFLOW_SCREEN)
    time.sleep(2)

    log("Starting training screen...")
    ssh_run_bg(client, training_screen(resume_path))
    log("Training started. Detach from terminal safely any time.")


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


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    log("=== AlphaCheckers EC2 Training Manager ===")
    log(f"Config: {TRAIN_CONFIG} | Workers: {WORKERS} | Target: {NUM_ITERATIONS} iters")

    run_number = 0

    while True:
        run_number += 1
        log("-" * 50)
        log(f"Run #{run_number}")

        # ── Check S3 for existing checkpoint ──────────────────────────────────
        ckpt_info = s3_checkpoint_info()
        if ckpt_info:
            s3_iter = ckpt_info.get("iteration", -1)
            s3_elo  = ckpt_info.get("elo", 0.0)
            s3_time = ckpt_info.get("saved_at", "?")
            log(f"S3 checkpoint: iter {s3_iter}/{NUM_ITERATIONS - 1}  ELO {s3_elo:.0f}  saved {s3_time}")
            if s3_iter >= NUM_ITERATIONS - 1:
                log("Training is already complete  - nothing to do.")
                return
            resume_path = f"runs/{TRAIN_CONFIG}/checkpoints/checkpoint_latest.pt"
        else:
            log("No S3 checkpoint found  - starting fresh")
            resume_path = None

        # ── Launch ────────────────────────────────────────────────────────────
        try:
            instance_id = launch_instance()
        except RuntimeError as e:
            wait = 60
            warn(f"Launch failed: {e}  - retrying in {wait}s")
            time.sleep(wait)
            continue

        # ── Wait for boot ─────────────────────────────────────────────────────
        try:
            ip = wait_for_running(instance_id)
        except RuntimeError as e:
            warn(f"Instance failed before running: {e}")
            continue

        # ── SSH + setup ───────────────────────────────────────────────────────
        try:
            client = ssh_connect(ip)
            configure_instance(client, resume_path)
        except Exception as e:
            warn(f"Setup failed: {e}  - terminating instance")
            terminate_instance(instance_id)
            close_tunnel()
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
