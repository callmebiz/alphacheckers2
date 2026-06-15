# Cloud Training Guide

Training AlphaCheckers on cloud compute. The MCTS self-play loop is CPU-bound
(tree search runs on CPU regardless of GPU), so the biggest speedup comes from
parallelising games across many CPU cores rather than from GPU acceleration.

## Provider Choice

| Option | Cost (medium config) | Session limit | Setup |
|---|---|---|---|
| AWS EC2 Spot (c5.4xlarge) | ~$0.35 | None | Medium |
| Vast.ai CPU (16+ cores) | ~$0.50–0.75 | None | Low |
| Kaggle (T4 GPU, free) | Free | 9 hours | Low |

**Recommended: AWS EC2 Spot** — cheapest per run, no session limits, spot
interruptions are safe because training checkpoints every iteration.

## AWS One-Time Setup

### 1. SSH key (Windows)

Download your `.pem` file from the EC2 key pair page. Move it into your SSH
folder and set permissions:

```powershell
Move-Item ~/Downloads/alphaCheckers.pem $HOME\alphacheckers_temp.pem
New-Item -ItemType Directory -Path "$HOME\.ssh" -Force
Move-Item "$HOME\alphacheckers_temp.pem" "$HOME\.ssh\alphaCheckers.pem"
icacls "$HOME\.ssh\alphaCheckers.pem" /inheritance:r /grant:r "${env:USERNAME}:(R)"
```

> **Note:** Do not use `mv key.pem ~/.ssh/` if `~/.ssh/` does not already exist —
> PowerShell will rename the file to `.ssh` rather than moving it into a folder.

### 2. Launch an EC2 instance

| Setting | Value |
|---|---|
| AMI | Amazon Linux 2023 |
| Instance type | `c5.4xlarge` (16 vCPU, 32 GB RAM) |
| Key pair | Your downloaded `.pem` |
| SSH source | My IP only |
| Storage | 20 GiB gp3 |
| Purchasing option | Spot instances (Advanced details) |

## Per-Run Workflow

### Connect

```powershell
ssh -i ~/.ssh/alphaCheckers.pem ec2-user@<PUBLIC_IP>
```

Accept the fingerprint prompt on first connection (`yes`).

### Environment setup (first run only)

```bash
# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Git + repo
sudo dnf install -y git screen
git clone https://github.com/callmebiz/alphacheckers2
cd alphacheckers2
pip install -r requirements.txt
```

### Start training

Always run inside `screen` so training survives SSH disconnects:

```bash
screen -S training
cd alphacheckers2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py --config medium --workers 12 --experiment baseline
```

**screen commands:**

| Action | Command |
|---|---|
| Detach (leave running) | `Ctrl+A` then `D` |
| List sessions | `screen -ls` |
| Reattach | `screen -r training` |
| Kill session | type `exit` inside screen |

### Worker count and thread pinning

PyTorch/NumPy use internal BLAS thread pools. Without pinning, each worker
spawns its own thread pool — 4 workers × 4 threads each = 16 threads fighting
over 16 cores, leaving none for the main training loop.

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` pins each worker to exactly 1 thread,
so you can safely run 1 worker per core.

**Choosing worker count** for c5.4xlarge (16 vCPUs):

```
16 vCPUs  −  4 reserved for main process  =  12 game workers
```

The 4 reserved cores handle: neural net backprop, evaluation, MLflow logging,
and process orchestration. Fewer than 4 reserved starves the training step.

```bash
# Quick formula to compute for any instance
python -c "import os; print(os.cpu_count() - 4)"
```

### Monitor training

**tqdm progress bars** are visible when attached to the screen session.

**MLflow UI** — live metrics accessible from your local browser via SSH tunnel:

On EC2 (second screen session):
```bash
screen -S mlflow
cd alphacheckers2
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0
# Ctrl+A D to detach
```

On your local machine (keep this terminal open):
```powershell
ssh -i ~/.ssh/alphaCheckers.pem -L 5000:localhost:5000 ec2-user@<PUBLIC_IP> -N
```

Open `http://localhost:5000` in your browser.

### Resume after interruption

Spot instances give 2 minutes notice before reclaiming. At most one iteration
of work is lost. Resume on the same or a new instance:

```bash
cd alphacheckers2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python train.py --config medium --workers 12 --resume --experiment baseline
```

`--resume` with no path auto-finds the latest checkpoint.

### Download results

From your local PowerShell when training is complete:

```powershell
# Best model checkpoint
scp -i ~/.ssh/alphaCheckers.pem \
    ec2-user@<PUBLIC_IP>:~/alphacheckers2/runs/medium/checkpoints/best.pt \
    ./best.pt

# MLflow database (to view run history locally)
scp -i ~/.ssh/alphaCheckers.pem \
    ec2-user@<PUBLIC_IP>:~/alphacheckers2/mlflow.db \
    ./mlflow.db
```

View run history locally:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Clean up

**Stop** (pauses compute billing, ~$0.13/month disk fee continues):
EC2 Console → Instances → Instance State → Stop

**Terminate** (deletes everything, zero ongoing cost):
EC2 Console → Instances → Instance State → Terminate

Always terminate after downloading your checkpoint unless you plan to resume
on the same instance soon.

## Cost Reference

| c5.4xlarge | Rate |
|---|---|
| Spot price | ~$0.07/hr |
| On-demand | $0.68/hr |
| EBS (20 GB) while stopped | ~$0.05/month |

Medium config (~5 hrs with 12 workers): **~$0.35 total**.

Set a billing alert: AWS Console → Billing → Budgets → Create Budget → $5
threshold. Emails you before any unexpected spend accumulates.

## Training Config Reference

```bash
# Standard medium run on c5.4xlarge (12 = 16 vCPUs - 4 reserved)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python train.py --config medium --workers 12 --experiment baseline

# Name a specific hypothesis
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python train.py --config medium --workers 12 \
    --experiment sims-ablation --run-name 200sims

# Resume latest checkpoint
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python train.py --config medium --workers 12 --resume --experiment baseline

# Resume specific checkpoint
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python train.py --config medium --workers 12 \
    --resume runs/medium/checkpoints/checkpoint_42.pt
```

| Flag | Purpose |
|---|---|
| `--config` | Preset: `debug`, `dev`, `medium`, `full` |
| `--workers N` | Parallel self-play processes; use `cpu_count - 4` on EC2 |
| `--experiment` | MLflow experiment name (groups related runs) |
| `--run-name` | Prefix for auto-generated run name |
| `--resume` | Resume from latest checkpoint |
