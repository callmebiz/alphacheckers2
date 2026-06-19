# AlphaCheckers Training Metrics

Reference for every metric logged to MLflow during training.
View them with: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

---

## Loss metrics (`loss/`)

| Metric | What it measures |
|---|---|
| `loss/policy` | Cross-entropy between the network's policy output and the MCTS visit-count distribution. Measures how well the network predicts which moves MCTS finds good. |
| `loss/value` | Mean-squared error between the network's value head and the actual game outcome (+1 win / 0 draw / −1 loss). Measures position evaluation accuracy. |
| `loss/total` | Sum of policy + value loss. The quantity being minimised each training step. |

**Expected trends:** All three should decrease over training, though not monotonically.
Early training: `loss/value` often drops quickly (the network learns who is winning); `loss/policy` takes longer (learning *how* to win is harder).

---

## Training metadata (`train/`)

| Metric | What it measures |
|---|---|
| `train/lr` | Current learning rate after the scheduler step. |
| `train/log10_lr` | Log₁₀ of the learning rate. Useful because `lr = 1e-3` shows as −3, `lr = 1e-4` as −4 — changes are visible in the MLflow chart whereas a raw LR of 0.001 and 0.0001 look nearly identical on a linear scale. |
| `train/buffer_size` | Number of (state, policy, outcome) examples currently in the replay buffer. Rises until it hits `replay_buffer_size` capacity, then stays flat. Training is skipped until `min_buffer_size` is reached. |

---

## Self-play outcome metrics (`selfplay/`)

These are computed from the batch of self-play games played at the start of each iteration.

| Metric | What it measures |
|---|---|
| `selfplay/avg_game_length` | Mean number of half-moves (plies) per self-play game. Early training: long games as neither player can win decisively. Later: games shorten as the model learns to convert advantages. |
| `selfplay/p1_win_rate` | Fraction of self-play games won by P1 (the first mover). Should hover near 0.5 if the game is well-balanced. Persistent drift signals first-mover advantage. |
| `selfplay/draw_rate` | Fraction of self-play games that ended in a draw. Typically rises early (the model doesn't know how to win) and falls later. |

---

## Per-move search statistics (`selfplay/`)

These are computed from the raw MCTS visit-count distributions (at temperature=1) for every individual move across all self-play games this iteration. They describe *how decisive* the model is during search.

### Entropy background

After MCTS runs N simulations, each legal action has a visit count. The raw visit proportions form a probability distribution. Shannon entropy of that distribution (in bits) measures how spread out it is:

- **High entropy** (e.g. 3.0 bits): many actions have similar visit counts — the model is genuinely uncertain about what to do, perhaps in a complex midgame where multiple plans are plausible.
- **Low entropy** (e.g. 0.3 bits): one action gets almost all visits — the model is confident, perhaps because there is a clear capture or the position is nearly decided.

Maximum possible entropy = log₂(n_legal_moves). For a position with 8 legal moves, max entropy ≈ 3.0 bits.

| Metric | What it measures |
|---|---|
| `selfplay/move_entropy_mean` | Mean entropy across all moves this iteration. The primary "model decisiveness" signal. **Expected trend:** starts high (random/uncertain model), decreases over training as the model learns which moves are good. A trained model will have notably lower mean entropy than an untrained one. |
| `selfplay/move_entropy_min` | Minimum entropy across all *non-forced* moves this iteration. Forced captures (positions with exactly one legal move) are excluded because their entropy is always 0 by definition, not because the model is confident. What remains reflects the most decisive genuine choice the model made. |
| `selfplay/move_entropy_std` | Standard deviation of per-move entropies. High std means some positions are very clear to the model (low entropy) while others are very uncertain (high entropy) — this is normal and healthy. Very low std means the model is uniformly uncertain about everything. |
| `selfplay/top1_prob_mean` | Mean of max(visit_probs) per move. Complementary to entropy: high top1_prob = the model strongly prefers one action. At the start of training, with random weights, top1_prob is near 1/n_legal. As training progresses this rises. |

**Example reading:** `move_entropy_mean=2.1`, `move_entropy_std=0.9`, `top1_prob_mean=0.55` means: on average the model sees about 4 roughly-plausible actions (`2^2.1 ≈ 4.3`), but with significant variation — some positions it's very sure about, others not. The top action gets 55% of visits on average.

---

## Analysis metrics (`analysis/`)

| Metric | What it measures |
|---|---|
| `analysis/policy_entropy` | Entropy of the *training-adjusted* MCTS policy (after temperature scaling), normalised to [0, 1], averaged across all self-play moves. Unlike `selfplay/move_entropy_mean`, this uses the temperature-adjusted distribution (which may be one-hot for late-game moves) and is normalised. Less granular than the selfplay/* entropy metrics. |
| `analysis/opening_entropy` | Entropy of the distribution of first moves chosen across all self-play games this iteration, normalised to [0, 1]. **Low (near 0):** the model always opens with the same move — a sign of premature convergence or insufficient exploration. **High (near 1):** diverse openings, good exploration. |
| `analysis/value_mae` | Mean absolute error between the value head's predictions and actual game outcomes, measured on a held-out sample from the replay buffer after training. Unlike `loss/value` (MSE during training), MAE is in the same units as the targets (−1 to +1) and easier to interpret. `MAE = 0.5` means predictions are off by 0.5 on average. Early training: typically 0.5–0.8. A well-trained model: 0.1–0.3. |

---

## Evaluation / tournament metrics (`eval/`)

These are logged only on iterations where a tournament is run.

### Challenger win rate

| Metric | What it measures |
|---|---|
| `eval/win_rate` | The *challenger model's* win rate against the current best model. The challenger alternates sides (P1 in half the games, P2 in the other half), so this is side-adjusted. A challenger must exceed `promotion_threshold` (default 0.55) to be promoted. |
| `eval/win_rate_ci_lo` | Lower bound of the 95% Wilson confidence interval on the win rate. With few games (e.g. 8), the interval is wide; with 40+ games it tightens substantially. |
| `eval/win_rate_ci_hi` | Upper bound of the confidence interval. Example: win_rate=0.75 with 8 games → CI ≈ [0.41, 0.93]. The true win rate could plausibly be anywhere in that range — treat small-sample win rates with caution. |
| `eval/wins` | Raw win count for the challenger. |
| `eval/draws` | Raw draw count. |
| `eval/losses` | Raw loss count. |

### Side-based outcomes

These are **not** about which model won — they measure which *side* won (P1 = first mover, P2 = second mover), regardless of which model that was. They help detect first-mover advantage.

| Metric | What it measures |
|---|---|
| `eval/p1_win_rate` | Fraction of tournament games where P1 (the player who moves first) won. Should be near 0.5 if the game is balanced. |
| `eval/p2_win_rate` | Same for P2 (second mover). |
| `eval/draw_rate` | Fraction of draws in the tournament. |

### Game quality

| Metric | What it measures |
|---|---|
| `eval/game_length_mean` | Average moves per tournament game. |
| `eval/game_length_std` | Standard deviation of game lengths — high means some games ended quickly (decisive wins), others dragged on. |
| `eval/avg_pieces_remaining` | Average total pieces on the board at game end. Low = games ended by captures; high = games ended by draw/stalemate. |

### Model status

| Metric | What it measures |
|---|---|
| `eval/promoted` | 1 if the challenger was promoted this iteration, 0 if not. |

---

## Reading a training run at a glance

A **healthy early run** (iteration 1–10):
- `loss/policy` high (3–5), falling slowly
- `loss/value` high (0.8–1.0), falling faster
- `selfplay/move_entropy_mean` high (2.5+)
- `analysis/value_mae` high (0.5–0.8)
- `eval/win_rate` near 0.5 (models are roughly equal)

A **well-progressed run** (iteration 30+):
- `loss/policy` lower (1–2)
- `loss/value` lower (0.3–0.5)
- `selfplay/move_entropy_mean` lower (1.0–1.5)
- `selfplay/top1_prob_mean` higher (0.65+)
- `analysis/value_mae` lower (0.2–0.4)
- `eval/win_rate` consistently above promotion threshold
