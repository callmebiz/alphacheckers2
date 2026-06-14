# AlphaCheckers 2

A from-scratch AlphaZero implementation for checkers with a browser-based UI
and a full self-play training pipeline.

---

## Running everything

All commands require the conda env active — your prompt should show `(alphacheckers2)`.
If it shows `(base)`, run `conda activate alphacheckers2` first.

Open **three terminals**, all with the env active:

---

**Terminal 1 — Game / UI server**
```powershell
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```
→ **http://localhost:8000** — Play tab, Training dashboard, Replay viewer

---

**Terminal 2 — Training**
```powershell
python train.py --config debug          # smoke-test: CPU, tiny model, 3 iterations
python train.py --config dev            # standard run
python train.py --config dev --resume   # continue from last checkpoint
```
The Play/Training tabs at http://localhost:8000 update live while this runs.

---

**Terminal 3 — MLflow experiment tracker** *(start after training has begun)*
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
→ **http://localhost:5000** — loss curves, ELO history, hyperparams, checkpoint artifacts

---

## Setup (first time)

**1. Create the conda environment**
```powershell
conda create -n alphacheckers2 python=3.12
conda activate alphacheckers2
pip install -r requirements.txt
```

**2. GPU training** (RTX 3070 or cloud — optional, CPU works fine for debug/dev):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**3. Run the tests to verify everything works**
```powershell
python -m pytest tests/ -v
# Expected: 94 passed
```

---

## Training configs

| Config | Iterations | Self-play games | MCTS sims | Use for |
|--------|-----------|-----------------|-----------|---------|
| `debug` | 3 | 4 | 10 | Verify code works (minutes on CPU) |
| `dev` | 30 | 20 | 100 | Iterating on the model (hours on GPU) |
| `full` | 100 | 100 | 200 | Serious training run (overnight on GPU) |

Switching devices:
```powershell
python train.py --config full --device cuda   # force GPU
python train.py --config full --device cpu    # force CPU
```

---

## How it works

```
Board state
    │
    ▼
Encoder ──────── perspective-aware tensor (always from current player's POV)
    │
    ▼
AlphaNet ──────── ResNet → policy head (move probs) + value head (who's winning)
    │
    ▼
MCTS ──────────── tree search guided by the network; outputs improved move probs
    │
    ▼
Self-play ──────── model plays itself → (state, policy, outcome) training examples
    │
    ▼
Training ──────── policy loss (cross-entropy) + value loss (MSE); AMP on CUDA
    │
    ▼
Tournament ─────── challenger vs best model; promote if win rate ≥ threshold
    │
    ▼
MLflow ─────────── logs every metric, loss, ELO, and checkpoint artifact per run
```

---

## Components

| File | What it does |
|---|---|
| [core/game.py](core/game.py) | American checkers rules — move generation, forced captures, multi-jump, kings |
| [core/encoder.py](core/encoder.py) | Board → float32 tensor; perspective-flipped for player 2 |
| [core/model.py](core/model.py) | ResNet dual-head: policy logits + value scalar |
| [core/mcts.py](core/mcts.py) | MCTS with UCB, Dirichlet noise, temperature sampling |
| [training/config.py](training/config.py) | Typed dataclasses + debug / dev / full presets |
| [training/replay_buffer.py](training/replay_buffer.py) | Circular experience buffer |
| [training/self_play.py](training/self_play.py) | Game generation with temperature schedule; saves replay JSONs |
| [training/evaluator.py](training/evaluator.py) | Model tournament, Wilson CI on win rate, promotion gate |
| [training/analysis.py](training/analysis.py) | Opening entropy, policy entropy, value calibration MAE |
| [training/elo.py](training/elo.py) | ELO ratings persisted to JSON across runs |
| [training/checkpoints.py](training/checkpoints.py) | Save / resume (always CPU-mapped for portability) |
| [training/tracking.py](training/tracking.py) | MLflow wrapper — params, metrics, artifacts |
| [training/trainer.py](training/trainer.py) | Main AlphaZero loop — self-play → train → evaluate → log |
| [train.py](train.py) | CLI entry point (`--config`, `--resume`, `--device`, `--seed`) |
| [server/main.py](server/main.py) | FastAPI: game WebSocket + training API + replay API |
| [ui/dist/index.html](ui/dist/index.html) | Browser UI — Play, Training dashboard, Replay viewer |

---

## Project status

| Phase | Status |
|---|---|
| Game engine + tests | ✅ Done |
| Neural network (encoder + ResNet) | ✅ Done |
| MCTS | ✅ Done |
| Training loop (self-play, tournament, ELO, MLflow) | ✅ Done |
| AI opponent in the UI | ⬜ Planned |

---

## Repository layout

```
alphacheckers2/
├── core/
│   ├── game.py
│   ├── encoder.py
│   ├── model.py
│   └── mcts.py
├── training/
│   ├── config.py
│   ├── replay_buffer.py
│   ├── self_play.py
│   ├── evaluator.py
│   ├── analysis.py
│   ├── elo.py
│   ├── checkpoints.py
│   ├── tracking.py
│   └── trainer.py
├── server/
│   └── main.py
├── ui/dist/
│   └── index.html
├── tests/
│   ├── test_game.py       # 35 tests
│   ├── test_encoder.py    # 15 tests
│   ├── test_model.py      # 8 tests
│   ├── test_mcts.py       # 14 tests
│   └── test_training.py   # 22 tests  →  94 total
├── train.py
├── requirements.txt
├── PLAN.md
└── README.md
```
