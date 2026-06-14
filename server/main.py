"""
FastAPI Server
==============
Serves the web UI and exposes all real-time and REST endpoints the frontend
needs to play games, watch training progress, and replay past games.

Endpoints
---------
WebSocket  /ws/game         — real-time human-vs-human (or human-vs-AI) game
WebSocket  /ws/training     — live training status push (1-second heartbeat)
GET        /api/status      — latest training status.json as JSON
GET        /api/checkpoints — list of saved checkpoint files for the UI selector
GET        /api/replays     — list of saved game replay files
GET        /api/replays/{id} — full move-by-move data for one replay

Why WebSocket for training status?
  REST polling every N seconds creates unnecessary load and adds latency.
  A WebSocket push lets the UI update the moment a new iteration finishes
  without hammering the server with repeated GET requests.

Why separate /ws/game and /ws/training?
  They have completely different message shapes and lifecycles. Mixing them
  would complicate both the server logic and the frontend message handling.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys

# Allow "from core.game import ..." when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.game import Checkers
from core.encoder import StateEncoder
from core.model import AlphaNet
from core.mcts import MCTS

app = FastAPI()

# ── Path constants ────────────────────────────────────────────────────────────

_ROOT    = os.path.join(os.path.dirname(__file__), "..")
UI_DIST  = os.path.join(_ROOT, "ui", "dist")

# Training outputs live under runs/{name}/ — look for whichever run exists
_RUNS_DIR = os.path.join(_ROOT, "runs")


def _status_path() -> str | None:
    """Return the first status.json found across all run directories."""
    pattern = os.path.join(_RUNS_DIR, "**", "status.json")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def _checkpoint_dir() -> str | None:
    pattern = os.path.join(_RUNS_DIR, "**", "checkpoints")
    matches = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
    return matches[0] if matches else None


def _replay_dirs() -> list[tuple[str, str]]:
    """Return all (run_name, replay_dir) pairs across every run directory."""
    pattern = os.path.join(_RUNS_DIR, "**", "replays")
    result = []
    for path in glob.glob(pattern, recursive=True):
        if os.path.isdir(path):
            parts = os.path.normpath(path).split(os.sep)
            run_name = parts[-2] if len(parts) >= 2 else "unknown"
            result.append((run_name, path))
    return sorted(result)


# ── Game engine + AI helpers ──────────────────────────────────────────────────

game_engine   = Checkers()
_game_encoder = StateEncoder(game_engine)
_ai_model_cache: dict[str, AlphaNet] = {}


def _load_ai_mcts(checkpoint_id: str, num_simulations: int = 200) -> MCTS:
    """Load (or return cached) model and return a fresh MCTS instance for it."""
    if checkpoint_id not in _ai_model_cache:
        ckpt_dir = _checkpoint_dir()
        if not ckpt_dir:
            raise ValueError("No checkpoint directory found")
        path = os.path.join(ckpt_dir, f"{checkpoint_id}.pt")
        if not os.path.exists(path):
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")
        data  = torch.load(path, map_location="cpu", weights_only=False)
        cfg   = data["config"]
        model = AlphaNet(
            num_channels=_game_encoder.num_channels,
            action_size=game_engine.action_size,
            num_resblocks=cfg["model"]["num_resblocks"],
            num_hidden=cfg["model"]["num_hidden"],
        )
        model.load_state_dict(data["model_state"])
        model.eval()
        _ai_model_cache[checkpoint_id] = model
    return MCTS(
        game_engine, _game_encoder, _ai_model_cache[checkpoint_id],
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_eps=0.0,
        dirichlet_alpha=0.3,
        device=torch.device("cpu"),
    )


def _state_payload(state: dict, current_player: int, message: str = "") -> dict:
    value, terminated = game_engine.get_value_and_terminated(state, current_player)
    moves = [] if terminated else game_engine.list_moves(state, current_player)
    return {
        "board":          game_engine.board_to_list(state),
        "current_player": current_player,
        "valid_moves":    moves,
        "terminated":     terminated,
        "value":          value,
        "no_progress":    state["no_progress"],
        "jump_again":     list(state["jump_again"]) if state["jump_again"] else None,
        "message":        message,
    }


# ── WebSocket: game ──────────────────────────────────────────────────────────

@app.websocket("/ws/game")
async def game_ws(ws: WebSocket):
    """
    Real-time checkers game session supporting Human vs Human and Human vs AI.

    Message types the client can send
    ----------------------------------
    {"type": "reset"}
    {"type": "move", "action": N}
    {"type": "ai_config",
     "checkpoint": "best"|"checkpoint_N"|"",   # "" → human vs human
     "human_player": 1|-1,
     "simulations": 200}

    Extra fields the server adds to state payloads in AI mode
    ---------------------------------------------------------
    "thinking": true   — AI is computing its move (board already shows your move)
    """
    await ws.accept()

    state          = game_engine.get_initial_state()
    current_player = 1
    ai_player:  int | None  = None   # which player index the AI controls
    ai_mcts:    MCTS | None = None

    def _win_msg() -> str:
        val, _ = game_engine.get_value_and_terminated(state, current_player)
        if val == 1.0:
            return f"Player {current_player} wins!"
        if val == -1.0:
            return f"Player {game_engine.get_opponent(current_player)} wins!"
        return "Draw!"

    async def do_ai_turn() -> None:
        """Drive AI moves until it's the human's turn or the game ends."""
        nonlocal state, current_player
        if ai_mcts is None or ai_player is None:
            return
        loop = asyncio.get_running_loop()
        while current_player == ai_player:
            _, terminated = game_engine.get_value_and_terminated(state, current_player)
            if terminated:
                break
            # Signal to the client that the AI is computing
            payload = _state_payload(state, current_player)
            payload["thinking"] = True
            await ws.send_text(json.dumps(payload))
            # Run MCTS in a thread so the event loop stays responsive
            probs  = await loop.run_in_executor(None, ai_mcts.search, state, current_player, 0)
            action = int(np.argmax(probs))
            state  = game_engine.get_next_state(state, action, current_player)
            if state["jump_again"] is None:
                current_player = game_engine.get_opponent(current_player)
            _, terminated = game_engine.get_value_and_terminated(state, current_player)
            msg = _win_msg() if terminated else ""
            await ws.send_text(json.dumps(_state_payload(state, current_player, msg)))

    await ws.send_text(json.dumps(_state_payload(state, current_player, "Game started")))

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "reset":
                state          = game_engine.get_initial_state()
                current_player = 1
                mode_msg = (f"vs AI — you are Player {game_engine.get_opponent(ai_player)}"
                            if ai_player else "Game reset")
                await ws.send_text(json.dumps(_state_payload(state, current_player, mode_msg)))
                try:
                    await do_ai_turn()
                except Exception as exc:
                    await ws.send_text(json.dumps({"error": f"AI error: {exc}"}))

            elif msg.get("type") == "ai_config":
                checkpoint_id = msg.get("checkpoint", "")
                human_player  = int(msg.get("human_player", 1))
                num_sims      = max(1, int(msg.get("simulations", 200)))

                if checkpoint_id:
                    loop = asyncio.get_running_loop()
                    try:
                        new_mcts = await loop.run_in_executor(
                            None, _load_ai_mcts, checkpoint_id, num_sims
                        )
                    except Exception as exc:
                        await ws.send_text(json.dumps({"error": f"Failed to load model: {exc}"}))
                        continue
                    ai_mcts   = new_mcts
                    ai_player = game_engine.get_opponent(human_player)
                    mode_msg  = f"vs AI — you are Player {human_player}"
                else:
                    ai_mcts   = None
                    ai_player = None
                    mode_msg  = "Human vs Human"

                state          = game_engine.get_initial_state()
                current_player = 1
                await ws.send_text(json.dumps(_state_payload(state, current_player, mode_msg)))
                try:
                    await do_ai_turn()
                except Exception as exc:
                    await ws.send_text(json.dumps({"error": f"AI error: {exc}"}))

            elif msg.get("type") == "move":
                action = int(msg["action"])
                valid  = game_engine.get_valid_moves(state, current_player)
                if not valid[action]:
                    await ws.send_text(json.dumps({"error": "Invalid move"}))
                    continue

                state = game_engine.get_next_state(state, action, current_player)
                if state["jump_again"] is None:
                    current_player = game_engine.get_opponent(current_player)

                _, terminated = game_engine.get_value_and_terminated(state, current_player)
                if terminated:
                    await ws.send_text(
                        json.dumps(_state_payload(state, current_player, _win_msg()))
                    )
                else:
                    await ws.send_text(json.dumps(_state_payload(state, current_player)))
                    try:
                        await do_ai_turn()
                    except Exception as exc:
                        await ws.send_text(json.dumps({"error": f"AI error: {exc}"}))

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


# ── WebSocket: training status ────────────────────────────────────────────────

@app.websocket("/ws/training")
async def training_ws(ws: WebSocket):
    """
    Live training status stream.

    The server reads status.json once per second and pushes it to the client
    whenever the content changes. If training is not running, sends a single
    {"is_training": false} and keeps the connection alive for future updates.

    This way the dashboard reacts instantly when an iteration finishes without
    the client needing to poll.
    """
    await ws.accept()
    last_payload = None

    try:
        while True:
            path = _status_path()
            if path and os.path.exists(path):
                try:
                    with open(path) as f:
                        payload = json.load(f)
                except (json.JSONDecodeError, OSError):
                    payload = {"is_training": False, "error": "status file unreadable"}
            else:
                payload = {"is_training": False}

            # Only push when state changes to avoid drowning the client in dups
            payload_str = json.dumps(payload)
            if payload_str != last_payload:
                await ws.send_text(payload_str)
                last_payload = payload_str

            await asyncio.sleep(1.0)

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


# ── REST: training status ────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """
    Return the latest training status as JSON.

    Used by the UI on first load so it doesn't have to wait for the WebSocket
    to push its first update.
    """
    path = _status_path()
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                return JSONResponse(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return JSONResponse({"is_training": False})


# ── REST: checkpoints ─────────────────────────────────────────────────────────

@app.get("/api/checkpoints")
async def list_checkpoints():
    """
    Return metadata for all saved checkpoints so the UI can populate a
    model-selection dropdown (e.g. for "Play vs AI" difficulty).

    Each item: {"id": "checkpoint_5", "iteration": 5, "path": "..."}
    """
    ckpt_dir = _checkpoint_dir()
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return JSONResponse([])

    entries = []
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".pt"):
            continue
        stem = fname[:-3]  # strip .pt
        # Parse iteration number from "checkpoint_N" or "best"
        if stem.startswith("checkpoint_"):
            try:
                iteration = int(stem.split("_")[-1])
            except ValueError:
                iteration = -1
        elif stem == "best":
            iteration = -1
        else:
            continue
        entries.append({
            "id":        stem,
            "iteration": iteration,
            "filename":  fname,
            "path":      os.path.join(ckpt_dir, fname),
        })

    # Sort: "best" first, then by iteration descending
    entries.sort(key=lambda e: (-1 if e["id"] == "best" else -e["iteration"]))
    return JSONResponse(entries)


# ── REST: replays ─────────────────────────────────────────────────────────────

@app.get("/api/replays")
async def list_replays():
    """
    Return a list of all saved game replay files across all run directories.

    Entries are sorted newest first (by modification time). Each item includes
    a "run" field with the run name so the UI can label games by which training
    run produced them.
    """
    all_dirs = _replay_dirs()
    if not all_dirs:
        return JSONResponse([])

    entries = []
    for run_name, replay_dir in all_dirs:
        for fname in sorted(os.listdir(replay_dir)):
            if not fname.endswith(".json"):
                continue
            stem = fname[:-5]
            fpath = os.path.join(replay_dir, fname)
            iteration = -1
            for part in stem.split("_"):
                if part.startswith("iter"):
                    try:
                        iteration = int(part[4:])
                    except ValueError:
                        pass
            outcome = num_moves = winner = mlflow_run = None
            resigned = False
            game_type = "selfplay"
            challenger_is_p1 = challenger_won = None
            mtime = 0.0
            try:
                mtime = os.path.getmtime(fpath)
                with open(fpath) as f:
                    meta = json.load(f)
                outcome          = meta.get("outcome")
                num_moves        = meta.get("num_moves")
                winner           = meta.get("winner")
                resigned         = bool(meta.get("resigned", False))
                game_type        = meta.get("type", "selfplay")
                challenger_is_p1 = meta.get("challenger_is_p1")
                challenger_won   = meta.get("challenger_won")
                # mlflow_run_name was added in a later version; fall back to
                # the config directory name for older replays.
                mlflow_run = meta.get("mlflow_run_name") or run_name
            except (OSError, json.JSONDecodeError):
                mlflow_run = run_name
            entries.append({
                "id":               stem,
                "run":              run_name,
                "session":          mlflow_run,
                "iteration":        iteration,
                "filename":         fname,
                "outcome":          outcome,
                "num_moves":        num_moves,
                "winner":           winner,
                "resigned":         resigned,
                "game_type":        game_type,
                "challenger_is_p1": challenger_is_p1,
                "challenger_won":   challenger_won,
                "_mtime":           mtime,
            })

    # Newest files first
    entries.sort(key=lambda e: -e["_mtime"])
    for e in entries:
        del e["_mtime"]
    return JSONResponse(entries)


@app.get("/api/replays/{replay_id}")
async def get_replay(replay_id: str):
    """
    Return the full move-by-move data for a single replay file.

    Searches all run directories for the replay with matching stem name.
    """
    for _, replay_dir in _replay_dirs():
        path = os.path.join(replay_dir, f"{replay_id}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return JSONResponse(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"error": "Replay not found"}, status_code=404)


# ── REST: action map ─────────────────────────────────────────────────────────

@app.get("/api/action-map")
async def get_action_map():
    """
    Return the full action-index → [from_r, from_c, to_r, to_c] mapping.

    The UI uses this to convert MCTS policy vectors into per-square probabilities
    for the action-probability heatmap in the replay viewer.
    """
    return JSONResponse({str(k): list(v) for k, v in game_engine._action_to_move.items()})


# ── Static file serving ───────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    index = os.path.join(UI_DIST, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "UI not built yet — open ui/dist/index.html directly."}


if os.path.isdir(UI_DIST):
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="static")
