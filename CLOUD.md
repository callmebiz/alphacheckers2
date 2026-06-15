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
interruptions are handled gracefully (see below).

> **Why S3?** Spot instances **terminate** (not stop) on shutdown — the EBS
> disk is deleted. S3 is the only way to preserve data across runs.
> `train.sh` uploads `checkpoint_best.pt` to S3 after **every promotion** and
> uploads both `checkpoint_best.pt` + `mlflow.db` at end of training.
> A SIGTERM trap in `train.sh` also triggers an emergency upload if the spot
> instance is interrupted mid-run.

---

## Shell syntax note

AWS CLI commands work the same in PowerShell, cmd, and bash. The differences
are only in variable assignment and JSON file creation:

| Task | PowerShell | cmd.exe | bash (Linux) |
|---|---|---|---|
| Set variable | `$VAR = cmd` | *(use PowerShell)* | `VAR=$(cmd)` |
| Create JSON file | `[System.IO.File]::WriteAllText(...)` | *(use PowerShell)* | `echo '...' > file.json` |
| SSH | `ssh -i ~/.ssh/key.pem user@IP` | same | same |

**Always use PowerShell on Windows** (not cmd) for the launch commands.
If `aws` is not found in PowerShell, open a fresh terminal window or run:
```powershell
$env:PATH += ";C:\Program Files\Amazon\AWSCLIV2"
```

---

## One-Time Setup (CLI)

Everything below only needs to be done once per AWS account.

### 1. Install and configure AWS CLI

```powershell
winget install Amazon.AWSCLI
```

Close and reopen the terminal, then configure with your IAM access key
(IAM → Users → your user → Security credentials → Create access key →
select "CLI" use case):

```powershell
aws configure
# Access Key ID:     <paste>
# Secret Access Key: <paste>
# Default region:    us-east-1
# Output format:     json
```

### 2. Import your SSH key

```powershell
ssh-keygen -y -f $HOME\.ssh\alphaCheckers.pem > $HOME\.ssh\alphaCheckers.pub
aws ec2 import-key-pair --key-name alphaCheckers --public-key-material fileb://$HOME/.ssh/alphaCheckers.pub --region us-east-1
```

### 3. Create the S3 bucket

```powershell
aws s3 mb s3://alphacheckers-biz --region us-east-1
```

### 4. Create the IAM role (lets EC2 write to S3)

```powershell
[System.IO.File]::WriteAllText("$PWD\trust.json", '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}')
aws iam create-role --role-name AlphaCheckersEC2 --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name AlphaCheckersEC2 --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam create-instance-profile --instance-profile-name AlphaCheckersEC2
aws iam add-role-to-instance-profile --instance-profile-name AlphaCheckersEC2 --role-name AlphaCheckersEC2
```

### 5. Create the security group

```powershell
$VPC = aws ec2 describe-subnets --region us-east-1 --query "Subnets[0].VpcId" --output text
aws ec2 create-security-group --group-name AlphaCheckersSSH --description "SSH for AlphaCheckers" --vpc-id $VPC --region us-east-1 --query GroupId --output text
```

Add SSH access from your current IP:

```powershell
curl checkip.amazonaws.com
aws ec2 authorize-security-group-ingress --group-id sg-01735e08ceeac8ae0 --protocol tcp --port 22 --cidr YOUR_IP/32 --region us-east-1
```

> **If SSH times out on a future run:** your IP likely changed. Get it again
> with `curl checkip.amazonaws.com` and re-run the authorize command with the
> new IP. `InvalidPermission.Duplicate` means your IP hasn't changed — skip it.

---

## Per-Run: Launch an Instance

### 1. Check your IP hasn't changed

```powershell
curl checkip.amazonaws.com
```

If it changed, authorize the new IP (old rule stays, both will work):
```powershell
aws ec2 authorize-security-group-ingress --group-id sg-01735e08ceeac8ae0 --protocol tcp --port 22 --cidr NEW_IP/32 --region us-east-1
```

### 2. Launch — one command (PowerShell)

This creates the required JSON files, looks up the latest AMI and a subnet,
and launches the instance in one shot:

```powershell
[System.IO.File]::WriteAllText("$PWD\spot.json", '{"MarketType":"spot"}'); [System.IO.File]::WriteAllText("$PWD\bdm.json", '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]'); $AMI = aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --region us-east-1 --query Parameter.Value --output text; $SUBNET = aws ec2 describe-subnets --region us-east-1 --filters "Name=availabilityZone,Values=us-east-1b" --query "Subnets[0].SubnetId" --output text; aws ec2 run-instances --region us-east-1 --image-id $AMI --instance-type c5.4xlarge --key-name alphaCheckers --subnet-id $SUBNET --security-group-ids sg-01735e08ceeac8ae0 --iam-instance-profile Name=AlphaCheckersEC2 --instance-market-options file://spot.json --block-device-mappings file://bdm.json --count 1 --query "Instances[0].InstanceId" --output text
```

> **`InsufficientInstanceCapacity`?** Change `us-east-1b` to `us-east-1a`,
> `us-east-1c`, `us-east-1d`, etc. until one works.

### 3. SSH in

Wait ~30 seconds for the instance to boot:

```powershell
# PowerShell / cmd / bash — all the same
aws ec2 describe-instances --instance-ids i-REPLACE --region us-east-1 --query "Reservations[0].Instances[0].PublicIpAddress" --output text
ssh -i ~/.ssh/alphaCheckers.pem ec2-user@PUBLIC_IP
```

On Linux/Mac:
```bash
ssh -i ~/.ssh/alphaCheckers.pem ec2-user@PUBLIC_IP
```

---

## Per-Run: Train on the Instance

### Environment setup (fresh instance — paste as one block)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init bash && source ~/.bashrc && sudo dnf install -y git screen && git clone https://github.com/callmebiz/alphacheckers2 && cd alphacheckers2 && pip install -r requirements.txt && chmod +x train.sh
```

### Start training

Always inside `screen` so training survives SSH disconnects:

```bash
screen -S training
./train.sh --config medium --workers 12 --experiment baseline --s3-bucket alphacheckers-biz && sudo shutdown -h now
```

Detach with `Ctrl+A D` — safe to close the SSH window.

> **Never run `sudo shutdown` without `--s3-bucket`** — spot instances delete
> their disk on termination and the training data will be lost.

**screen commands:**

| Action | Command |
|---|---|
| Detach (leave running) | `Ctrl+A` then `D` |
| List sessions | `screen -ls` |
| Reattach | `screen -r training` |

### Monitor training (second SSH window)

SSH in again from a second terminal:

```bash
# bash / PowerShell / cmd — same command
ssh -i ~/.ssh/alphaCheckers.pem ec2-user@PUBLIC_IP
```

Start MLflow in a screen session on the instance:

```bash
screen -S mlflow
cd alphacheckers2
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0
# Ctrl+A D to detach
```

Open an SSH tunnel on your **local** machine (keep this terminal open):

```powershell
# PowerShell or cmd
ssh -i ~/.ssh/alphaCheckers.pem -L 5000:localhost:5000 ec2-user@PUBLIC_IP -N
```

```bash
# bash (Linux/Mac)
ssh -i ~/.ssh/alphaCheckers.pem -L 5000:localhost:5000 ec2-user@PUBLIC_IP -N
```

Open `http://localhost:5000`. Each iteration logs:
- `loss/policy`, `loss/value` — network training losses
- `eval/elo`, `eval/win_rate` — tournament results
- `system/iter_time_s`, `system/disk_free_gb` — health metrics

### Worker count

c5.4xlarge has 16 vCPUs. Reserve 4 for the main process:

```bash
python -c "import os; print(os.cpu_count() - 4)"  # → 12
```

### Resume after spot interruption

`train.sh` uploads `checkpoint_best.pt` to S3 after every promotion, so at
most one iteration's worth of games is lost on interruption. On a new instance,
after env setup:

```bash
mkdir -p runs/medium/checkpoints
aws s3 cp s3://alphacheckers-biz/runs/medium/checkpoints/checkpoint_best.pt runs/medium/checkpoints/checkpoint_best.pt
screen -S training
./train.sh --config medium --workers 12 --resume --experiment baseline --s3-bucket alphacheckers-biz && sudo shutdown -h now
```

If the bucket is empty (no promotion had occurred yet), skip the `aws s3 cp`
and start fresh without `--resume`.

---

## After Training: Download Locally

```powershell
# PowerShell
aws s3 cp s3://alphacheckers-biz/mlflow.db ./mlflow.db
New-Item -ItemType Directory -Force runs/medium/checkpoints
aws s3 cp s3://alphacheckers-biz/runs/medium/checkpoints/checkpoint_best.pt ./runs/medium/checkpoints/checkpoint_best.pt
```

```bash
# bash (Linux/Mac)
aws s3 cp s3://alphacheckers-biz/mlflow.db ./mlflow.db
mkdir -p runs/medium/checkpoints
aws s3 cp s3://alphacheckers-biz/runs/medium/checkpoints/checkpoint_best.pt ./runs/medium/checkpoints/checkpoint_best.pt
```

View training history:

```powershell
conda activate alphacheckers2
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Play against the model:

```powershell
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`, select AI opponent, pick the checkpoint.

---

## Cost Reference

| Resource | Rate |
|---|---|
| c5.4xlarge spot | ~$0.07/hr |
| c5.4xlarge on-demand | $0.68/hr |
| S3 storage | ~$0.023/GB/month |

Medium config (~5 hrs, 12 workers): **~$0.35 per run**.
S3 cost for checkpoint_best.pt + mlflow.db: **~$0.02/month**.

Set a billing alert: AWS Console → Billing → Budgets → Create Budget → $5.

**Known resource IDs (us-east-1):**
- Security group: `sg-01735e08ceeac8ae0`
- S3 bucket: `alphacheckers-biz`
- IAM role / instance profile: `AlphaCheckersEC2`
- Key pair: `alphaCheckers`

---

## Training Config Reference

```bash
# Standard run — upload to S3 and shut down when done
./train.sh --config medium --workers 12 --experiment baseline \
    --s3-bucket alphacheckers-biz && sudo shutdown -h now

# Quick ablation — override sims/iters without editing config
./train.sh --config medium --workers 12 --sims 400 --iters 50 \
    --experiment sims-ablation --s3-bucket alphacheckers-biz && sudo shutdown -h now

# Resume from latest checkpoint
./train.sh --config medium --workers 12 --resume --experiment baseline \
    --s3-bucket alphacheckers-biz && sudo shutdown -h now

# Without S3 (stay on instance to download manually)
./train.sh --config medium --workers 12 --experiment baseline
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
| `--s3-bucket NAME` | Upload `checkpoint_best.pt` to S3 after each promotion and `mlflow.db` at end of training; SIGTERM trap does an emergency upload on spot interruption |
