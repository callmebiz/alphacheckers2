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

### 1. S3 bucket for training output

Spot instances **terminate** (not stop) when shut down — the EBS disk is deleted.
S3 is the only way to preserve data across runs. Create a bucket once:

```bash
aws s3 mb s3://your-alphacheckers-bucket --region us-east-1
```

Pick any globally unique name (e.g. `alphacheckers-biz`). Then when you launch
your EC2 instance, attach an IAM role with S3 write access:

EC2 Launch Wizard → **Advanced details → IAM instance profile → Create new IAM role**:
- Trusted entity: EC2
- Permissions policy: `AmazonS3FullAccess` (or a scoped policy for just your bucket)

With `--s3-bucket` set, `train.sh` uploads `checkpoint_best.pt` and `mlflow.db`
to S3 automatically before the instance shuts down.

### 2. SSH key (Windows)

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
chmod +x train.sh
```

### Start training

Always run inside `screen` so training survives SSH disconnects:

```bash
screen -S training
cd alphacheckers2
./train.sh --config medium --workers 12 --experiment baseline --s3-bucket your-alphacheckers-bucket && sudo shutdown -h now
```

`train.sh` sets `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` automatically. When
`--s3-bucket` is provided, it uploads `checkpoint_best.pt` and `mlflow.db` to
S3 after training finishes — before `shutdown` runs. The instance then
terminates and all data is safe in S3.

> **Never run `sudo shutdown` without `--s3-bucket`** — spot instances delete
> their disk on termination.

At startup you will see a disk warning if free space is below 5 GB:

```
WARNING: only 3.2 GB free — checkpoints may fail if disk fills up.
```

The trainer keeps only the 3 most recent numbered checkpoints and always
preserves `checkpoint_best.pt`, so disk usage stays bounded.

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
so you can safely run 1 worker per core. `train.sh` handles this automatically.

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

**MLflow system metrics** — each iteration logs two additional metrics:
- `system/iter_time_s` — wall-clock seconds for the full iteration
- `system/disk_free_gb` — remaining disk space at checkpoint time

These appear under the `system/` group in the MLflow runs table.

### Resume after interruption

Spot instances give 2 minutes notice before reclaiming. At most one iteration
of work is lost. Resume on the same or a new instance:

```bash
cd alphacheckers2
./train.sh --config medium --workers 12 --resume --experiment baseline
```

`--resume` with no path auto-finds the latest checkpoint. On startup you will
see a resume banner confirming what was restored:

```
Resumed iter 25 | ELO 2020 | buffer 88,000 | disk free 9.3 GB
```

### Download results from S3

After training completes (instance already terminated), pull the data locally:

```powershell
# MLflow database
aws s3 cp s3://your-alphacheckers-bucket/mlflow.db ./mlflow.db

# Best model checkpoint
New-Item -ItemType Directory -Force runs/medium/checkpoints
aws s3 cp s3://your-alphacheckers-bucket/runs/medium/checkpoints/checkpoint_best.pt ./runs/medium/checkpoints/checkpoint_best.pt
```

View run history:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Play against the trained model locally

Start the local server and open the UI:

```powershell
conda activate alphacheckers2
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`, switch Opponent to AI, select the downloaded
checkpoint from the panel, and start a game.

### Resume from S3 on a new instance

If training was interrupted and you want to resume on a fresh instance:

```bash
# On the new instance, after cloning + pip install:
aws s3 cp s3://your-alphacheckers-bucket/runs/medium/checkpoints/checkpoint_best.pt \
    runs/medium/checkpoints/checkpoint_best.pt
./train.sh --config medium --workers 12 --resume --experiment baseline \
    --s3-bucket your-alphacheckers-bucket && sudo shutdown -h now
```

### Clean up

Spot instances terminate automatically on shutdown — no manual cleanup needed.
S3 storage costs ~$0.023/GB/month; a checkpoint_best.pt is ~1 GB → ~$0.02/month.

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
# Standard run — upload to S3 and shut down when done
./train.sh --config medium --workers 12 --experiment baseline \
    --s3-bucket your-alphacheckers-bucket && sudo shutdown -h now

# Without S3 (stay on instance to manually download data)
./train.sh --config medium --workers 12 --experiment baseline

# Name a specific hypothesis
./train.sh --config medium --workers 12 --experiment sims-ablation --run-name 200sims \
    --s3-bucket your-alphacheckers-bucket && sudo shutdown -h now

# Quick ablation — override sims/iters without editing config
./train.sh --config medium --workers 12 --sims 400 --iters 50 --experiment sims-ablation \
    --s3-bucket your-alphacheckers-bucket && sudo shutdown -h now

# Resume from S3 checkpoint on a new instance
./train.sh --config medium --workers 12 --resume --experiment baseline \
    --s3-bucket your-alphacheckers-bucket && sudo shutdown -h now
```

| Flag | Purpose |
|---|---|
| `--config` | Preset: `debug`, `dev`, `medium`, `full` |
| `--workers N` | Parallel self-play processes; use `cpu_count - 4` on EC2 |
| `--experiment` | MLflow experiment name (groups related runs) |
| `--run-name` | Prefix for auto-generated run name |
| `--resume` | Resume from latest checkpoint (or pass a path) |
| `--sims N` | Override MCTS simulations per move for this run |
| `--iters N` | Override number of training iterations for this run |
| `--s3-bucket NAME` | Upload checkpoint_best.pt + mlflow.db to S3 after training |
