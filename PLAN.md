# AlphaCheckers 2 — Project Plan

Sophisticated AlphaZero-style checkers engine with a playable web UI.

---

## Phase 1 — Game Engine + UI ✅

- [x] Core game engine (`core/game.py`) — full American checkers rules
  - Forced captures, multi-jump continuation, king promotion
  - Draw by repetition / no-progress limit
  - Pre-computed move table, `list_moves()` for UI consumption
- [x] 35-test suite (`tests/test_game.py`) — all passing
- [x] FastAPI + WebSocket server (`server/main.py`)
- [x] Web UI (`ui/dist/index.html`) — human vs human
  - Click-to-select and drag-and-drop pieces
  - Pulsing gold ring on moveable pieces
  - Static board size, no layout shifts
  - Win / draw detection and display

---

## Phase 2 — Neural Network ✅

- [x] State encoder (`core/encoder.py`)
  - Perspective-aware: always encode from current player's POV
  - Planes: player men, player kings, opponent men, opponent kings (× history timesteps)
  - Extra planes: repetition count, no-progress counter, player identity
- [x] ResNet model (`core/model.py`)
  - Configurable depth (num_resblocks) and width (num_hidden)
  - Policy head → action logits (size = action_size)
  - Value head → scalar in [-1, 1] via Tanh
- [x] Unit tests for encoder and model shapes (15 tests)

---

## Phase 3 — MCTS ✅

- [x] MCTS (`core/mcts.py`)
  - Node class with UCB scoring, Q-value tracking, expansion
  - SELECT → EVALUATE → EXPAND → BACKPROPAGATE cycle
  - Dirichlet noise at root for exploration during self-play
  - Temperature-controlled action sampling
  - Correct value negation for multi-jump chains (player-comparison, not blind negate)
- [x] MCTS tests — UCB properties, backpropagation correctness, legal-only probs, immediate win detection (22 tests)

---

## Phase 4 — Training + Monitoring ✅

- [x] Config system (`training/config.py`) — typed dataclasses, `debug` / `dev` / `full` presets
- [x] Replay buffer (`training/replay_buffer.py`) — circular buffer with state_dict save/restore
- [x] Self-play worker (`training/self_play.py`)
  - Sequential (Windows-compatible); temperature schedule (explore early, exploit late)
  - Records (encoded_state, mcts_policy, outcome) tuples
  - Saves board-state-per-move JSON replays for UI playback
- [x] Model tournament (`training/evaluator.py`)
  - Challenger must win ≥ promotion_threshold to replace best model
  - Wilson score 95% CI on win rate; alternating colour assignment
- [x] Game quality analysis (`training/analysis.py`)
  - Game length stats, draw rate, opening entropy, policy entropy, value calibration MAE
- [x] Checkpoint save/resume (`training/checkpoints.py`)
  - Always saves to CPU for cross-device portability; auto-finds latest via glob
- [x] MLflow tracking (`training/tracking.py`)
  - Full params, per-iteration metrics, config artifact, checkpoint artifacts
  - Context manager interface; supports remote tracking URI
- [x] Training loop (`training/trainer.py`)
  - Policy loss (cross-entropy vs MCTS) + value loss (MSE vs outcome)
  - Mixed precision (AMP) on CUDA; gradient clipping
  - Graceful shutdown on SIGINT/SIGTERM with checkpoint write
  - `status.json` written after every iteration for UI polling
- [x] CLI entry point (`train.py`) — `--config`, `--resume`, `--device`, `--seed`
- [x] Training test suite (`tests/test_training.py`) — 22 tests, all passing
- [x] Server API routes (`server/main.py`)
  - `GET /api/status` — latest training status
  - `GET /api/checkpoints` — list saved checkpoints
  - `GET /api/replays`, `GET /api/replays/{id}` — replay browser
  - `WS /ws/training` — live push (1-second heartbeat, change-only)
- [x] UI tabs — Play | Training | Replay
  - Training dashboard: live metrics, iteration progress bar, MLflow link, start instructions
  - Replay viewer: pick any saved game, step forward/backward through moves

---

## Phase 5 — AI in UI

- [ ] AI move endpoint in server — accepts difficulty (MCTS simulations count)
- [ ] UI mode selector: Human vs Human / Human vs AI / AI vs AI
- [ ] "AI thinking" indicator while MCTS runs
- [ ] Difficulty slider (maps to num_searches: e.g. 50 / 200 / 800)
- [ ] Show AI's top move confidence as a subtle highlight

---

## Infrastructure Notes

- **Conda env:** `alphacheckers2` (Python 3.12)
- **GPU:** RTX 3070 locally; may move training to cloud compute
- **Portability:** `requirements.txt` covers all deps; torch GPU variant installed separately
- **Run tests:** `python -m pytest tests/ -v`
- **Run UI:** `python -m uvicorn server.main:app --host 0.0.0.0 --port 8000`
