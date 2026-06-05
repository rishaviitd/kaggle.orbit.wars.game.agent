from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from modules.analysis_utils import opening_report, planetary_report

try:
    from kaggle_environments import make  # type: ignore[reportMissingImports]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "kaggle_environments is required to play locally. Use "
        "`/Users/pwm-co-56335/miniconda3/bin/python play_against_agent.py` "
        "or install it with `python3 -m pip install kaggle-environments`."
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "self-plays"
Agent = Callable[[dict[str, Any], Any], list[list[Any]]]
RELOADED_ENV = "ORBIT_WARS_PLAY_RELOADED"
WATCHED_CODE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "modules" / "analysis_utils.py",
)


def code_revision() -> str:
    parts = []
    for path in WATCHED_CODE_PATHS:
        try:
            stat = path.stat()
        except FileNotFoundError:
            parts.append(f"{path.name}:missing")
            continue
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def start_code_reloader(interval_seconds: float = 0.8) -> None:
    initial_revision = code_revision()

    def watch() -> None:
        while True:
            time.sleep(interval_seconds)
            if code_revision() == initial_revision:
                continue
            time.sleep(0.35)
            print("Code change detected; restarting play UI server...", flush=True)
            os.environ[RELOADED_ENV] = "1"
            os.execv(sys.executable, [sys.executable, *sys.argv])

    threading.Thread(target=watch, name="play-ui-reloader", daemon=True).start()


def load_agent(path: Path, module_name: str) -> Agent:
    module = load_agent_module(path, module_name)
    return module.agent


def load_agent_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def step_agent_state(state: Any) -> dict[str, Any]:
    return {
        "reward": getattr(state, "reward", None),
        "status": str(getattr(state, "status", "")),
        "action": to_plain(getattr(state, "action", None)),
    }


def parse_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return int(value)


class HumanOrbitGame:
    def __init__(
        self,
        agent_key: str,
        human_player: int,
        seed: int | None,
        episode_steps: int,
        output_dir: Path,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_key = agent_key
        self.human_player = human_player
        self.seed = seed
        self.episode_steps = episode_steps
        self.env: Any | None = None
        self.trainer: Any | None = None
        self.obs: Any | None = None
        self.done = False
        self.reward: Any = 0
        self.info: Any = {}
        self.agent_module: Any | None = None
        self.last_agent_debug: dict[str, Any] | None = None
        self.last_turn_actions: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.opening_trace: list[dict[str, Any]] = []
        self.reset(agent_key=agent_key, human_player=human_player, seed=seed, episode_steps=episode_steps)

    def agent_path(self, agent_key: str) -> Path:
        if agent_key == "v0":
            return ROOT / "agents" / "v0.py"
        path = Path(agent_key)
        return path if path.is_absolute() else ROOT / path

    def reset(
        self,
        agent_key: str | None = None,
        human_player: int | None = None,
        seed: Any | None = None,
        episode_steps: int | None = None,
    ) -> dict[str, Any]:
        if agent_key is not None:
            self.agent_key = agent_key
        if human_player is not None:
            self.human_player = int(human_player)
        if seed is not None:
            self.seed = parse_seed(seed)
        if episode_steps is not None:
            self.episode_steps = int(episode_steps)

        self.agent_module = load_agent_module(self.agent_path(self.agent_key), f"human_play_{self.agent_key.replace('/', '_')}")
        agent = self.agent_module.agent
        agents: list[Any] = [None, agent] if self.human_player == 0 else [agent, None]
        config: dict[str, Any] = {"episodeSteps": self.episode_steps}
        if self.seed is not None:
            config["seed"] = self.seed
        self.env = make("orbit_wars", debug=True, configuration=config)
        self.env.reset(len(agents))
        self.trainer = None
        self.obs = self.player_obs(self.human_player)
        self.done = False
        self.reward = 0
        self.info = {}
        self.last_agent_debug = self.compute_agent_debug()
        self.last_turn_actions = None
        self.opening_trace = []
        self.history = []
        self.record_history()
        return self.payload()

    def player_obs(self, player_id: int) -> dict[str, Any]:
        if self.env is None or not getattr(self.env, "state", None):
            return {}
        obs = to_plain(getattr(self.env.state[int(player_id)], "observation", {}))
        if isinstance(obs, dict) and "step" not in obs and self.env.steps:
            shared_obs = self.env.steps[-1][0].get("observation", {})
            if "step" in shared_obs:
                obs["step"] = shared_obs["step"]
        return obs

    def agent_obs(self) -> dict[str, Any]:
        return self.player_obs(1 - self.human_player)

    def compute_agent_debug(self) -> dict[str, Any] | None:
        if self.agent_module is None or not hasattr(self.agent_module, "opening_debug"):
            return None
        try:
            return to_plain(self.agent_module.opening_debug(self.agent_obs()))
        except Exception as exc:
            return {"error": str(exc)}

    def agent_action(self) -> list[list[Any]]:
        if self.agent_module is None:
            return []
        try:
            action = self.agent_module.agent(self.agent_obs(), getattr(self.env, "configuration", None))
            return to_plain(action or [])
        except Exception:
            return []

    def opening_trace_entry(
        self,
        agent_obs: dict[str, Any],
        agent_debug: dict[str, Any] | None,
        human_moves: list[list[Any]],
        agent_moves: list[list[Any]],
    ) -> dict[str, Any]:
        debug = to_plain(agent_debug or {})
        comparisons = debug.get("comparisons", []) if isinstance(debug, dict) else []
        selected = debug.get("selected", []) if isinstance(debug, dict) else []
        selected_by_pair = {
            (int(item.get("source_id")), int(item.get("target_id"))): item
            for item in selected
            if isinstance(item, dict) and item.get("source_id") is not None and item.get("target_id") is not None
        }
        launched_pairs = {
            (int(item.get("source_id")), int(item.get("target_id")))
            for item in selected
            if isinstance(item, dict)
            and item.get("source_id") is not None
            and item.get("target_id") is not None
            and str(item.get("kind", "launch")) == "launch"
        }
        rows = []
        for row in comparisons:
            if not isinstance(row, dict):
                continue
            source_id = int(row.get("source_id"))
            target_id = int(row.get("target_id"))
            selected_row = selected_by_pair.get((source_id, target_id))
            rows.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "owner": int(row.get("owner", -1)),
                    "production": int(row.get("production", 0)),
                    "ships_needed": int(row.get("ships_needed", 0)),
                    "wait": int(row.get("wait_turns", 0)),
                    "travel": int(row.get("travel_turns", 0)),
                    "total_time": float(row.get("total_time", 0.0)),
                    "net": float(row.get("net_value", 0.0)),
                    "cheap": float(row.get("cheap_production", 0.0)),
                    "selected": selected_row is not None,
                    "selected_kind": None if selected_row is None else str(selected_row.get("kind", "launch")),
                    "launched": (source_id, target_id) in launched_pairs,
                }
            )
        rows.sort(key=lambda item: (item["source"], -item["net"], item["total_time"], item["target"]))
        sources = sorted({row["source"] for row in rows})
        targets = sorted({row["target"] for row in rows})
        by_source = {
            f"p{source_id}": [
                row
                for row in rows
                if row["source"] == source_id
            ]
            for source_id in sources
        }
        readable = [
            f"turn {len(self.opening_trace) + 1} step {agent_obs.get('step', 0) if isinstance(agent_obs, dict) else '?'} "
            f"agent_player={1 - self.human_player} agent_moves={to_plain(agent_moves)}"
        ]
        for item in selected:
            if not isinstance(item, dict):
                continue
            readable.append(
                "SELECT "
                f"{item.get('kind', 'launch')} "
                f"p{item.get('source_id')}->p{item.get('target_id')} "
                f"ships={item.get('ships')} "
                f"eta_step={item.get('available_step')} "
                f"score={item.get('assignment_score', item.get('net_value', ''))}"
            )
        for source_id in sources:
            top_rows = by_source[f"p{source_id}"][:8]
            row_text = "; ".join(
                f"p{row['source']}->p{row['target']} n={row['net']:.0f} w={row['wait']}+{row['travel']}t ships={row['ships_needed']}"
                + (" SELECTED" if row["selected"] else "")
                for row in top_rows
            )
            readable.append(f"p{source_id}: {row_text}")
        return {
            "turn_index": len(self.opening_trace) + 1,
            "step": int(agent_obs.get("step", 0)) if isinstance(agent_obs, dict) else None,
            "agent_player": 1 - self.human_player,
            "human_player": self.human_player,
            "agent": self.agent_key,
            "human_moves": to_plain(human_moves),
            "agent_moves": to_plain(agent_moves),
            "selected": selected,
            "matrix": {
                "sources": sources,
                "targets": targets,
                "rows": rows,
                "by_source": by_source,
            },
            "future_sources_by_id": debug.get("future_sources_by_id", {}) if isinstance(debug, dict) else {},
            "blocked_quadrant": debug.get("blocked_quadrant") if isinstance(debug, dict) else None,
            "eliminated": debug.get("eliminated", []) if isinstance(debug, dict) else [],
            "debug_error": debug.get("error") if isinstance(debug, dict) else None,
            "readable": readable,
        }

    def record_history(self) -> None:
        if self.env is None:
            return
        self.history.append(
            {
                "env": self.env.clone(),
                "obs": to_plain(self.obs or {}),
                "done": self.done,
                "reward": self.reward,
                "info": to_plain(self.info),
                "opening_trace": to_plain(self.opening_trace),
                "last_turn_actions": to_plain(self.last_turn_actions),
            }
        )

    def step(self, moves: list[list[Any]]) -> dict[str, Any]:
        if self.done:
            return self.payload()
        if self.env is None:
            raise RuntimeError("Game is not initialized")
        agent_obs = self.agent_obs()
        self.last_agent_debug = self.compute_agent_debug()
        agent_moves = self.agent_action()
        self.last_turn_actions = {
            "step": int(agent_obs.get("step", 0)) if isinstance(agent_obs, dict) else None,
            "human_player": self.human_player,
            "agent_player": 1 - self.human_player,
            "human_moves": to_plain(moves),
            "agent_moves": to_plain(agent_moves),
        }
        self.opening_trace.append(
            self.opening_trace_entry(
                agent_obs=agent_obs,
                agent_debug=self.last_agent_debug,
                human_moves=moves,
                agent_moves=agent_moves,
            )
        )
        previous_state = self.env.state[self.human_player]
        previous_reward = getattr(previous_state, "reward", None)
        actions: list[Any] = [[], []]
        actions[self.human_player] = moves
        actions[1 - self.human_player] = agent_moves
        states = self.env.step(actions)
        human_state = states[self.human_player]
        self.obs = self.player_obs(self.human_player)
        current_reward = getattr(human_state, "reward", None)
        if current_reward is not None and previous_reward is not None:
            self.reward = current_reward - previous_reward
        else:
            self.reward = current_reward
        self.done = bool(getattr(self.env, "done", False)) or getattr(human_state, "status", "") != "ACTIVE"
        self.info = to_plain(getattr(human_state, "info", {}))
        if self.done:
            self.save_episode()
        if not self.done:
            current_debug = self.compute_agent_debug()
            if current_debug is not None:
                self.last_agent_debug = current_debug
        self.record_history()
        return self.payload()

    def back(self) -> dict[str, Any]:
        if len(self.history) <= 1:
            return self.payload()
        self.history.pop()
        snapshot = self.history[-1]
        self.env = snapshot["env"].clone()
        self.trainer = None
        self.obs = to_plain(snapshot["obs"])
        self.done = bool(snapshot["done"])
        self.reward = snapshot["reward"]
        self.info = to_plain(snapshot["info"])
        self.opening_trace = to_plain(snapshot.get("opening_trace", []))
        self.last_turn_actions = to_plain(snapshot.get("last_turn_actions"))
        self.last_agent_debug = self.compute_agent_debug()
        return self.payload()

    def save_episode(self) -> str | None:
        if self.env is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_label = "random" if self.seed is None else str(self.seed)
        obs = to_plain(self.obs or {})
        current_step = int(obs.get("step", 0)) if isinstance(obs, dict) else 0
        path = self.output_dir / f"human_vs_{self.agent_key}_seed-{seed_label}_step-{current_step}_{stamp}.json"
        episode = self.env.toJSON()
        episode["local_human_play"] = {
            "agent": self.agent_key,
            "human_player": self.human_player,
            "seed": self.seed,
            "episode_steps": self.episode_steps,
            "current_step": current_step,
            "done": self.done,
            "saved_at": stamp,
            "opening_trace": to_plain(self.opening_trace),
            "last_turn_actions": to_plain(self.last_turn_actions),
        }
        path.write_text(json.dumps(episode, indent=2, default=str), encoding="utf-8")
        return str(path)

    def payload(self) -> dict[str, Any]:
        obs = to_plain(self.obs or {})
        try:
            human_analysis = {
                "opening": to_plain(opening_report(obs)),
                "planetary": to_plain(planetary_report(obs)),
            }
        except Exception as exc:
            human_analysis = {
                "opening": {"error": str(exc), "rows": []},
                "planetary": {"error": str(exc), "rows": []},
            }
        final_states: list[dict[str, Any]] = []
        if self.env is not None and self.env.steps:
            final_states = [step_agent_state(state) for state in self.env.steps[-1]]
        return {
            "obs": obs,
            "human_player": self.human_player,
            "agent_player": 1 - self.human_player,
            "agent": self.agent_key,
            "seed": self.seed,
            "seed_text": None if self.seed is None else str(self.seed),
            "episode_steps": self.episode_steps,
            "done": self.done,
            "reward": self.reward,
            "info": to_plain(self.info),
            "states": final_states,
            "agent_debug": to_plain(self.last_agent_debug),
            "human_analysis": human_analysis,
            "last_turn_actions": to_plain(self.last_turn_actions),
            "can_go_back": len(self.history) > 1,
            "code_revision": code_revision(),
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Orbit Wars Human Play</title>
  <style>
    :root {
      --ink: #eef4ff;
      --muted: #90a0b8;
      --panel: rgba(10, 16, 28, 0.86);
      --panel-2: rgba(20, 31, 50, 0.92);
      --line: rgba(135, 164, 210, 0.32);
      --accent: #ffb545;
      --blue: #3d8bdb;
      --orange: #df7a25;
      --green: #45d2a3;
      --danger: #ff6b70;
      --space: #050811;
    }
    * { box-sizing: border-box; }
    html, body {
      height: 100%;
      overflow: hidden;
    }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font: 15px/1.45 Avenir Next, Optima, Trebuchet MS, sans-serif;
      display: flex;
      align-items: center;
      justify-content: flex-start;
      padding: 8px 12px;
      background:
        radial-gradient(circle at 50% 44%, rgba(255, 183, 68, 0.22), transparent 28rem),
        radial-gradient(circle at 8% 14%, rgba(61, 139, 219, 0.25), transparent 22rem),
        linear-gradient(145deg, #02040a 0%, #091322 48%, #04070f 100%);
    }
    .app {
      display: grid;
      grid-template-columns: auto minmax(360px, 1fr);
      gap: 16px;
      align-items: center;
      width: 100%;
      height: min(900px, calc(100vh - 16px));
      margin: 0;
    }
    .board-wrap {
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 16px 46px rgba(0,0,0,0.36);
      border-radius: 14px;
      overflow: hidden;
      backdrop-filter: blur(14px);
      position: relative;
      justify-self: start;
      width: min(calc(100vh - 16px), calc(100vw - 400px), 900px);
      aspect-ratio: 1 / 1;
    }
    h1 {
      margin: 0;
      font: 800 22px/1.1 Didot, Georgia, serif;
      letter-spacing: 0.3px;
    }
    .subtle { color: var(--muted); }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      aspect-ratio: 1 / 1;
      background: #000;
      cursor: crosshair;
    }
    .play-strip {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 4px 0 6px;
      padding: 3px;
      border: 1px solid rgba(150, 165, 210, 0.18);
      border-radius: 7px;
      background: rgba(15, 16, 29, 0.45);
    }
    .fleet-menu {
      display: none;
      flex: 0 0 auto;
      min-width: 0;
      width: auto;
      margin: 0;
      border: 0;
      border-radius: 0;
      padding: 0;
      background: transparent;
      box-shadow: none;
    }
    .fleet-menu-grid {
      display: grid;
      grid-template-columns: 52px 62px 28px 28px;
      gap: 5px;
      align-items: center;
    }
    .fleet-menu input {
      min-width: 0;
      height: 19px;
      padding: 2px 7px;
      border-radius: 5px;
      font-size: 12px;
      background: rgba(4, 15, 30, 0.92);
      border-color: rgba(83, 116, 163, 0.72);
    }
    .fleet-menu button {
      width: auto;
      margin: 0;
      min-width: 28px;
      min-height: 19px;
      padding: 2px 4px;
      border-radius: 5px;
      font-size: 12px;
      line-height: 1;
    }
    .fleet-menu .ghost {
      color: var(--ink);
      background: #30304a;
      border: 1px solid rgba(150, 165, 210, 0.42);
    }
    .turn-actions {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: auto auto;
      gap: 5px;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
    }
    .hud {
      display: flex;
      flex-direction: column;
      position: static;
      z-index: 3;
      width: 100%;
      min-width: 0;
      align-self: stretch;
      padding: 10px;
      border: 1px solid rgba(150, 165, 210, 0.28);
      border-radius: 12px;
      background: rgba(18, 18, 31, 0.88);
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.36);
      backdrop-filter: blur(10px);
      overflow: auto;
    }
    .hud-status {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      margin-bottom: 8px;
      white-space: nowrap;
    }
    .hud-title {
      font-size: 12px;
      font-weight: 900;
      margin: 0;
      color: #eef4ff;
    }
    .score-row {
      display: flex;
      align-items: center;
      gap: 5px;
      flex: 0 1 auto;
      min-width: 0;
      margin: 0;
      padding: 3px 7px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.045);
      font-size: 12px;
      font-weight: 700;
    }
    .score-row span:not(.dot) {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .score-row b {
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .hud-status .dot {
      width: 7px;
      height: 7px;
    }
    .hud-seed {
      display: grid;
      grid-template-columns: 84px auto;
      gap: 6px;
      margin-left: auto;
      align-items: center;
    }
    .hud-seed input {
      min-width: 0;
      height: 24px;
      padding: 3px 7px;
      border-radius: 6px;
      font-size: 12px;
    }
    .toast-stack {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 20;
      display: grid;
      gap: 8px;
      width: min(260px, calc(100vw - 36px));
      pointer-events: none;
    }
    .toast {
      padding: 9px 11px;
      border: 1px solid rgba(255, 181, 69, 0.42);
      border-radius: 8px;
      background: rgba(18, 18, 31, 0.94);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.34);
      color: #f5d089;
      font-size: 13px;
      font-weight: 850;
      line-height: 1.25;
      transform: translateY(6px);
      opacity: 0;
      animation: toast-in 150ms ease-out forwards, toast-out 220ms ease-in forwards 9800ms;
    }
    .toast span {
      color: #d9e2f3;
      font-weight: 800;
    }
    @keyframes toast-in {
      to { transform: translateY(0); opacity: 1; }
    }
    @keyframes toast-out {
      to { transform: translateY(6px); opacity: 0; }
    }
    .hud button {
      width: 100%;
      margin: 5px 0;
      color: var(--ink);
      background: #30304a;
      border: 1px solid rgba(150, 165, 210, 0.34);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 14px;
    }
    .hud .hud-status button {
      width: auto;
      min-width: 46px;
      margin: 0;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.1;
      white-space: nowrap;
      flex: 0 0 auto;
    }
    .hud .turn-actions button {
      width: auto;
      min-width: 50px;
      margin: 0;
      min-height: 19px;
      padding: 2px 7px;
      border-radius: 5px;
      font-size: 12px;
      line-height: 1.1;
      white-space: nowrap;
    }
    .hud button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
      filter: none;
    }
    .hud details {
      margin-top: 10px;
      color: var(--muted);
    }
    .hud details.options-panel {
      margin-top: auto;
      padding-top: 8px;
    }
    .hud summary {
      cursor: pointer;
      font-weight: 800;
      color: var(--muted);
    }
    .hud .settings {
      display: grid;
      grid-template-columns: minmax(180px, 320px) auto auto;
      gap: 6px;
      margin-top: 8px;
      align-items: center;
    }
    .hud .settings select {
      min-width: 0;
      height: 28px;
      padding: 4px 9px;
      border-radius: 7px;
      font-size: 12px;
    }
    .hud .settings button {
      width: auto;
      margin: 0;
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 7px;
      font-size: 12px;
      white-space: nowrap;
    }
    .hidden-controls {
      display: none;
    }
    .card {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 18px;
      padding: 14px;
    }
    .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .split { display: flex; justify-content: space-between; gap: 12px; }
    label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    select, input {
      width: 100%;
      color: var(--ink);
      background: #07111f;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 11px;
      outline: none;
    }
    input[type="number"] { font-variant-numeric: tabular-nums; }
    button {
      border: 0;
      color: #15110a;
      background: var(--accent);
      border-radius: 14px;
      padding: 11px 13px;
      font-weight: 800;
      cursor: pointer;
      transition: transform 120ms ease, filter 120ms ease;
    }
    button:hover { transform: translateY(-1px); filter: brightness(1.06); }
    button.secondary { color: var(--ink); background: #18283f; border: 1px solid var(--line); }
    button.danger { color: #2b0508; background: var(--danger); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .metric {
      min-width: 86px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 9px 10px;
      background: rgba(255,255,255,0.035);
    }
    .metric b { display: block; font-size: 18px; }
    .move {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 9px 0;
      border-bottom: 1px solid rgba(135,164,210,0.2);
    }
    .move:last-child { border-bottom: 0; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      color: var(--muted);
      background: rgba(255,255,255,0.04);
      font-size: 12px;
    }
    .dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
    .hint {
      color: var(--muted);
      font-size: 13px;
      margin-top: 7px;
    }
    .debug-panel {
      margin: 8px 0 10px;
      min-height: 0;
      border: 1px solid rgba(150, 165, 210, 0.18);
      border-radius: 8px;
      background: rgba(15, 16, 29, 0.45);
      padding: 7px;
    }
    .insights-tabs {
      display: flex;
      align-items: center;
      gap: 3px;
      margin-bottom: 5px;
      overflow: visible;
      flex-wrap: wrap;
    }
    .insights-tab {
      width: auto;
      margin: 0;
      min-height: 18px;
      padding: 2px 6px;
      border-radius: 999px;
      font-size: 10px;
      line-height: 1.05;
      flex: 0 0 auto;
    }
    .hud .insights-tab {
      width: auto;
      min-width: 0;
      margin: 0;
      padding: 2px 6px;
      border-radius: 999px;
      font-size: 10px;
      line-height: 1.05;
      white-space: nowrap;
    }
    .insights-tab.active {
      color: #111711;
      background: #47ff9a;
      border-color: #95ffc8;
    }
    .tab-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 13px;
      height: 13px;
      margin-left: 4px;
      padding: 0 3px;
      border-radius: 999px;
      background: #ffb545;
      color: #17130d;
      font-size: 8px;
      font-weight: 900;
      line-height: 1;
    }
    .insights-body {
      min-height: 0;
    }
    .priority-strip {
      display: flex;
      align-items: center;
      gap: 7px;
      flex-wrap: wrap;
      margin-bottom: 6px;
      padding: 5px 7px;
      border: 1px solid rgba(255, 181, 69, 0.35);
      border-radius: 7px;
      background: rgba(255, 181, 69, 0.08);
      color: #ffd48a;
      font-size: 11px;
      font-weight: 900;
    }
    .priority-strip span {
      color: var(--text);
    }
    .opening-report-meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }
    .opening-report-wrap {
      max-height: 45vh;
      overflow: auto;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 7px;
    }
    .opening-report {
      width: 100%;
      min-width: 760px;
      border-collapse: collapse;
      color: var(--muted);
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }
    .opening-report th,
    .opening-report td {
      padding: 5px 6px;
      border: 1px solid rgba(255,255,255,0.07);
      text-align: right;
      white-space: nowrap;
    }
    .opening-report th {
      position: sticky;
      top: 0;
      z-index: 1;
      color: #ffd48a;
      background: rgba(35, 28, 29, 0.98);
      font-weight: 900;
    }
    .opening-report th:first-child,
    .opening-report td:first-child {
      position: sticky;
      left: 0;
      z-index: 2;
      text-align: left;
      background: rgba(22, 23, 38, 0.98);
      color: #eef4ff;
      font-weight: 900;
    }
    .opening-report th:first-child {
      z-index: 3;
      background: rgba(35, 28, 29, 0.98);
    }
    .opening-report tr.action-now td {
      color: #eef4ff;
      background: rgba(71, 255, 154, 0.075);
      box-shadow: inset 0 1px 0 rgba(71, 255, 154, 0.22), inset 0 -1px 0 rgba(71, 255, 154, 0.22);
    }
    .opening-report tr.action-now td:first-child {
      background: rgba(36, 68, 62, 0.98);
    }
    .opening-report tr.primary-guide td {
      color: #ffffff;
      background: rgba(255, 181, 69, 0.14);
      box-shadow: inset 0 0 0 1px rgba(255, 181, 69, 0.36);
    }
    .opening-report tr.primary-guide td:first-child {
      background: rgba(73, 54, 34, 0.98);
    }
    .opening-report tr.best-route td {
      color: #ffffff;
      background: rgba(71, 255, 154, 0.08);
      box-shadow: inset 0 0 0 1px rgba(71, 255, 154, 0.26);
    }
    .opening-report tr.best-route td:first-child {
      background: rgba(28, 59, 49, 0.98);
    }
    .opening-report tr.source-unsafe td {
      color: #ffd3d5;
      background: rgba(255, 70, 85, 0.10);
      box-shadow: inset 0 0 0 1px rgba(255, 70, 85, 0.24);
    }
    .opening-report tr.source-unsafe td:first-child {
      background: rgba(68, 28, 36, 0.98);
    }
    .opening-report tr.outlier-row td {
      color: #fff2f2;
      background: rgba(255, 70, 85, 0.16);
      box-shadow: inset 0 0 0 1px rgba(255, 70, 85, 0.42);
    }
    .opening-report tr.outlier-row td:first-child {
      background: rgba(82, 31, 38, 0.98);
    }
    .opening-report tr.outlier-section td {
      color: #ff9ba3;
      background: rgba(58, 25, 31, 0.96);
    }
    .opening-report tr.role-row td {
      color: #f1e8ff;
      background: rgba(163, 92, 255, 0.12);
      box-shadow: inset 0 0 0 1px rgba(163, 92, 255, 0.28);
    }
    .opening-report tr.role-row td:first-child {
      background: rgba(50, 32, 78, 0.98);
    }
    .opening-report .now-badge {
      display: inline-block;
      min-width: 28px;
      padding: 1px 5px;
      border-radius: 999px;
      background: rgba(71, 255, 154, 0.18);
      color: #8fffc0;
      font-weight: 900;
      text-align: center;
    }
    .opening-report tr.row-selected td {
      background: rgba(71, 168, 255, 0.13);
      color: #ffffff;
      box-shadow: inset 0 0 0 1px rgba(71, 168, 255, 0.42);
    }
    .opening-report tr.row-selected td:first-child {
      background: rgba(25, 50, 80, 0.98);
    }
    .opening-report tbody tr {
      cursor: pointer;
    }
    .opening-report tr.section-row td {
      position: static;
      text-align: left;
      color: #ffd48a;
      background: rgba(35, 28, 29, 0.92);
      font-weight: 900;
      letter-spacing: 0.02em;
      cursor: default;
    }
    .opening-report .good-value {
      color: #d8e5ff;
    }
    .opening-report .bad-value {
      color: rgba(255, 107, 112, 0.82);
    }
    .debug-matrix-wrap {
      max-width: 100%;
      max-height: 45vh;
      overflow: auto;
      margin-top: 0;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 7px;
    }
    .debug-matrix {
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      color: var(--muted);
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }
    .debug-matrix th,
    .debug-matrix td {
      min-width: 34px;
      padding: 5px 5px;
      border: 1px solid rgba(255,255,255,0.07);
      text-align: right;
      white-space: nowrap;
    }
    .debug-matrix th {
      color: #ffd48a;
      background: rgba(255, 181, 69, 0.07);
      font-weight: 800;
    }
    .debug-matrix th:first-child {
      position: sticky;
      left: 0;
      z-index: 1;
      text-align: left;
      background: rgba(35, 28, 29, 0.98);
    }
    .debug-matrix .source-head {
      color: #eef4ff;
      text-align: left;
      background: rgba(35, 28, 29, 0.98);
      font-weight: 800;
    }
    .debug-matrix .empty-cell {
      color: rgba(144, 160, 184, 0.32);
    }
    .debug-matrix .positive-cell {
      color: #d8e5ff;
    }
    .debug-matrix .negative-cell {
      color: rgba(255, 107, 112, 0.82);
    }
    .debug-matrix .selected-cell {
      color: #111711;
      background: #47ff9a;
      border-color: #95ffc8;
      font-weight: 900;
      box-shadow: inset 0 0 0 2px rgba(0,0,0,0.22);
    }
    .debug-panel .debug-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .debug-actions {
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }
    .debug-actions button {
      width: auto;
      min-height: 32px;
      margin: 0;
      padding: 7px 10px;
      border-radius: 7px;
      font-size: 12px;
    }
    .matrix-modal {
      position: fixed;
      inset: 0;
      z-index: 40;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(3, 7, 15, 0.72);
      backdrop-filter: blur(8px);
    }
    .matrix-modal.open {
      display: flex;
    }
    .matrix-dialog {
      width: min(1280px, 96vw);
      max-height: 90vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      border: 1px solid rgba(255, 181, 69, 0.34);
      border-radius: 10px;
      background: rgba(24, 21, 31, 0.98);
      box-shadow: 0 26px 70px rgba(0, 0, 0, 0.56);
      overflow: hidden;
    }
    .matrix-modal-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.09);
    }
    .matrix-modal-header h2 {
      margin: 0;
      color: #ffd48a;
      font-size: 16px;
      line-height: 1.2;
    }
    .matrix-modal-subtitle {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .matrix-close {
      width: auto;
      min-height: 32px;
      margin: 0;
      padding: 7px 11px;
      border-radius: 7px;
      font-size: 12px;
    }
    .matrix-modal-body {
      min-height: 0;
      overflow: auto;
      padding: 12px 14px 14px;
    }
    .matrix-modal-body .debug-matrix-wrap {
      max-height: calc(90vh - 105px);
      margin-top: 0;
    }
    .debug-toggle {
      display: flex;
      align-items: center;
      gap: 5px;
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      cursor: pointer;
      user-select: none;
    }
    .turn-actions + .debug-toggle {
      margin-left: auto;
    }
    .debug-toggle input {
      width: 14px;
      height: 14px;
      margin: 0;
      accent-color: var(--accent);
    }
    .log {
      max-height: 145px;
      overflow: auto;
      color: var(--muted);
      font: 12px/1.45 Menlo, Consolas, monospace;
      white-space: pre-wrap;
    }
    @media (max-width: 900px) {
      html, body {
        height: auto;
        overflow: auto;
      }
      body {
        align-items: flex-start;
        padding: 8px 0;
      }
      .app {
        grid-template-columns: 1fr;
        width: calc(100vw - 16px);
        height: auto;
      }
      .board-wrap {
        justify-self: center;
        width: min(100%, calc(100vh - 16px));
      }
      .hud {
        max-height: none;
      }
      .debug-panel {
        position: static;
        max-height: none;
        margin: 10px 0;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="board-wrap">
      <canvas id="board" width="900" height="900"></canvas>
      <div class="hidden-controls">
        <span id="subtitle"></span>
        <div id="selectionText">Click one of your planets, then click target/aim point.</div>
        <input id="shipsInput" type="number" min="1" value="1" />
        <input id="angleInput" type="number" step="0.001" value="0" />
        <button id="addMoveBtn">Add Move</button>
        <button id="clearMoveBtn">Clear</button>
        <button id="holdBtn">Hold Step</button>
        <div id="movesList"></div>
        <div id="log"></div>
      </div>
    </section>
    <aside class="hud">
      <div class="hud-status">
        <div class="hud-title">Step <span id="stepMetric">0</span>/<span id="episodeMetric">500</span></div>
        <div class="score-row"><span class="dot" id="humanDot"></span><span>You</span><b id="humanProd">0</b></div>
        <div class="score-row"><span class="dot" id="agentDot"></span><span id="agentName">Agent</span><b id="agentProd">0</b></div>
        <button id="resetBtn" type="button">Reset</button>
        <button id="newGameBtn" type="button">New</button>
        <div class="hud-seed">
          <input id="seedInput" type="text" inputmode="numeric" value="20260507" aria-label="Kaggle seed" placeholder="Seed" />
          <button id="replaySeedBtn" type="button">Load</button>
        </div>
      </div>
      <div class="play-strip">
        <div class="fleet-menu" id="fleetMenu">
          <div class="fleet-menu-grid">
            <input id="menuShipsInput" type="number" min="1" value="1" aria-label="Ships" />
            <input id="menuAngleInput" type="number" step="0.1" value="0" aria-label="Angle degrees" />
            <button id="menuSendBtn" type="button" title="Send" aria-label="Send">></button>
            <button class="ghost" id="menuCancelBtn" type="button" title="Cancel" aria-label="Cancel">x</button>
          </div>
        </div>
        <div class="turn-actions">
          <button id="submitBtn">Step</button>
          <button id="backBtn">Back</button>
        </div>
        <label class="debug-toggle">
          <input id="debugToggle" type="checkbox" />
          <span>Debug</span>
        </label>
        <label class="debug-toggle">
          <input id="insightsToggle" type="checkbox" />
          <span>Insights</span>
        </label>
      </div>
      <div class="debug-panel" id="openingDebugPanel">
        <div class="insights-tabs">
          <button class="insights-tab active" type="button">Matrix</button>
          <button class="insights-tab" type="button" disabled>Filters</button>
          <button class="insights-tab" type="button" disabled>Selected</button>
          <button class="insights-tab" type="button" disabled>Values</button>
        </div>
        <div id="openingDebugBody" class="insights-body"></div>
      </div>
      <div class="debug-panel" id="humanInsightsPanel">
        <div class="insights-tabs">
          <button class="insights-tab active" data-human-tab="opening" type="button">Capture</button>
          <button class="insights-tab" data-human-tab="multi" type="button">Multi</button>
          <button class="insights-tab" data-human-tab="saves" type="button">Saves</button>
          <button class="insights-tab" data-human-tab="recaptures" type="button">Recaptures</button>
          <button class="insights-tab" data-human-tab="reinforce" type="button">Reinforce</button>
          <button class="insights-tab" data-human-tab="ignore" type="button">Ignore</button>
          <button class="insights-tab" data-human-tab="roles" type="button">Roles</button>
          <button class="insights-tab" type="button" disabled>Targets</button>
          <button class="insights-tab" data-human-tab="routes" type="button">Routes</button>
          <button class="insights-tab" type="button" disabled>My Matrix</button>
        </div>
        <div id="humanInsightsBody" class="insights-body debug-note">Human-facing insights will appear here.</div>
      </div>
      <details class="options-panel">
        <summary>Options</summary>
        <div class="settings">
          <select id="agentSelect">
            <option value="v0">v0</option>
          </select>
          <button id="saveBtn">Save Episode</button>
          <button id="clearQueueBtn">Clear Queue</button>
        </div>
      </details>
    </aside>
  </main>
  <div id="toastStack" class="toast-stack" aria-live="polite"></div>
  <div class="matrix-modal" id="matrixModal" aria-hidden="true">
    <div class="matrix-dialog" role="dialog" aria-modal="true" aria-labelledby="matrixModalTitle">
      <div class="matrix-modal-header">
        <div>
          <h2 id="matrixModalTitle">Opening matrix</h2>
          <div id="matrixModalSubtitle" class="matrix-modal-subtitle">Net values for source-target comparisons.</div>
        </div>
        <button id="closeMatrixBtn" class="matrix-close" type="button">Minimize</button>
      </div>
      <div id="matrixModalBody" class="matrix-modal-body">
        <div class="debug-note">Turn on Debug to inspect the agent matrix.</div>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('board');
    const ctx = canvas.getContext('2d');
    const colors = ['#3d8bdb', '#df7a25', '#45d2a3', '#f2ce55', '#777f8d'];
    const WORLD_PAD = 5;
    const BOARD_SIZE = 100;
    const CENTER = 50;
    const SUN_RADIUS = 10;
    const ROTATION_RADIUS_LIMIT = 50;
    const MAX_SPEED = 6;
    let state = null;
    let selectedSource = null;
    let aim = null;
    let aimTargetId = null;
    let queuedMoves = [];
    let isDraggingAim = false;
    let aimLocked = false;
    let debugVisible = false;
    let insightsVisible = false;
    let activeHumanInsightsTab = 'opening';
    let selectedInsightPlanetId = null;
    let selectedInsightSourceId = null;
    let selectedInsightSourceIds = [];
    let matrixModalOpen = false;
    let serverRevision = null;
    let revisionWatcherStarted = false;

    function log(line) {
      const el = document.getElementById('log');
      const time = new Date().toLocaleTimeString();
      el.textContent = `[${time}] ${line}\n` + el.textContent;
    }
    function showToast(html) {
      const stack = document.getElementById('toastStack');
      if (!stack) return;
      const toast = document.createElement('div');
      toast.className = 'toast';
      toast.innerHTML = html;
      stack.prepend(toast);
      while (stack.children.length > 4) stack.lastElementChild?.remove();
      setTimeout(() => toast.remove(), 10250);
    }

    async function api(path, body = null) {
      const res = await fetch(path, {
        method: body ? 'POST' : 'GET',
        headers: body ? {'Content-Type': 'application/json'} : {},
        cache: 'no-store',
        body: body ? JSON.stringify(body) : null,
      });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    function obs() { return state?.obs || {}; }
    function humanPlayer() { return state?.human_player ?? 0; }
    function agentPlayer() { return state?.agent_player ?? 1; }
    function scale() { return canvas.width / (100 + WORLD_PAD * 2); }
    function screenScale() { return canvas.getBoundingClientRect().width / (100 + WORLD_PAD * 2); }
    function sx(x) { return (x + WORLD_PAD) * scale(); }
    function sy(y) { return (y + WORLD_PAD) * scale(); }
    function boardPoint(evt) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: ((evt.clientX - rect.left) / rect.width) * (100 + WORLD_PAD * 2) - WORLD_PAD,
        y: ((evt.clientY - rect.top) / rect.height) * (100 + WORLD_PAD * 2) - WORLD_PAD,
      };
    }
    function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
    function planetObj(p) {
      return {id:p[0], owner:p[1], x:p[2], y:p[3], r:p[4], ships:p[5], prod:p[6]};
    }
    function fleetSpeed(ships) {
      ships = Math.max(1, Math.floor(Number(ships || 1)));
      if (ships <= 1) return 1.0;
      const ratio = Math.log(ships) / Math.log(1000);
      return Math.min(1.0 + (MAX_SPEED - 1.0) * Math.pow(ratio, 1.5), MAX_SPEED);
    }
    function pointToSegmentDistance(point, start, end) {
      const lengthSq = Math.pow(start.x - end.x, 2) + Math.pow(start.y - end.y, 2);
      if (lengthSq === 0) return dist(point, start);
      const t = Math.max(0, Math.min(1, ((point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y)) / lengthSq));
      return dist(point, {x: start.x + t * (end.x - start.x), y: start.y + t * (end.y - start.y)});
    }
    function pointToSegmentProgress(point, start, end) {
      const lengthSq = Math.pow(start.x - end.x, 2) + Math.pow(start.y - end.y, 2);
      if (lengthSq === 0) return 0;
      return Math.max(0, Math.min(1, ((point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y)) / lengthSq));
    }
    function cross2d(a, b) {
      return a.x * b.y - a.y * b.x;
    }
    function segmentIntersectionProgress(startA, endA, startB, endB) {
      const r = {x: endA.x - startA.x, y: endA.y - startA.y};
      const s = {x: endB.x - startB.x, y: endB.y - startB.y};
      const denominator = cross2d(r, s);
      if (Math.abs(denominator) < 1e-12) return null;
      const qp = {x: startB.x - startA.x, y: startB.y - startA.y};
      const t = cross2d(qp, s) / denominator;
      const u = cross2d(qp, r) / denominator;
      if (t >= -1e-9 && t <= 1 + 1e-9 && u >= -1e-9 && u <= 1 + 1e-9) {
        return Math.max(0, Math.min(1, t));
      }
      return null;
    }
    function segmentToSegmentDistance(startA, endA, startB, endB) {
      if (segmentIntersectionProgress(startA, endA, startB, endB) !== null) return 0;
      return Math.min(
        pointToSegmentDistance(startA, startB, endB),
        pointToSegmentDistance(endA, startB, endB),
        pointToSegmentDistance(startB, startA, endA),
        pointToSegmentDistance(endB, startA, endA),
      );
    }
    function segmentCollisionProgress(fleetStart, fleetEnd, targetStart, targetEnd) {
      const intersectionProgress = segmentIntersectionProgress(fleetStart, fleetEnd, targetStart, targetEnd);
      if (intersectionProgress !== null) return intersectionProgress;
      return Math.min(
        pointToSegmentProgress(targetStart, fleetStart, fleetEnd),
        pointToSegmentProgress(targetEnd, fleetStart, fleetEnd),
      );
    }
    function initialPlanetById() {
      const rows = obs().initial_planets || obs().planets || [];
      return new Map(rows.map(row => [row[0], planetObj(row)]));
    }
    function isOrbitingPlanet(planet, initialById = initialPlanetById()) {
      if ((obs().comet_planet_ids || []).includes(planet.id)) return false;
      const initial = initialById.get(planet.id);
      if (!initial) return false;
      return Math.hypot(initial.x - CENTER, initial.y - CENTER) + initial.r < ROTATION_RADIUS_LIMIT;
    }
    function planetPositionAfterTurns(planet, turns, initialById = initialPlanetById()) {
      if ((obs().comet_planet_ids || []).includes(planet.id)) return {x: planet.x, y: planet.y};
      const initial = initialById.get(planet.id);
      const angularVelocity = Number(obs().angular_velocity || 0);
      if (!initial || !isOrbitingPlanet(planet, initialById)) return {x: planet.x, y: planet.y};
      const radius = Math.hypot(initial.x - CENTER, initial.y - CENTER);
      const initialAngle = Math.atan2(initial.y - CENTER, initial.x - CENTER);
      const envStep = Math.max(1, Number(obs().step || 0)) + Math.max(0, turns) - 1;
      return {
        x: CENTER + radius * Math.cos(initialAngle + angularVelocity * envStep),
        y: CENTER + radius * Math.sin(initialAngle + angularVelocity * envStep),
      };
    }
    function planetMotionVector(planet, initialById = initialPlanetById()) {
      const next = planetPositionAfterTurns(planet, 1, initialById);
      const dx = next.x - planet.x;
      const dy = next.y - planet.y;
      const length = Math.hypot(dx, dy);
      if (length < 0.001) return null;
      return {x: dx / length, y: dy / length, length};
    }
    function nearestPlanetToAimPoint(source) {
      if (!aim) return null;
      const planets = (obs().planets || []).map(planetObj).filter(p => p.id !== source.id);
      const initialById = initialPlanetById();

      // If the crosshair sits ahead of a moving planet, assume the user is
      // asking for that planet's future hit point, even in crowded overlaps.
      let movingAheadBest = null;
      for (const p of planets) {
        if (!isOrbitingPlanet(p, initialById)) continue;
        const motion = planetMotionVector(p, initialById);
        if (!motion) continue;
        const relX = aim.x - p.x;
        const relY = aim.y - p.y;
        const ahead = relX * motion.x + relY * motion.y;
        if (ahead < -0.8 || ahead > Math.max(16, p.r * 8)) continue;
        const perp = Math.abs(relX * motion.y - relY * motion.x);
        if (perp > p.r + 5.5) continue;
        const edgeMiss = Math.max(0, perp - p.r);
        const score = edgeMiss + ahead * 0.018 - p.r * 0.08;
        if (!movingAheadBest || score < movingAheadBest.score) {
          movingAheadBest = {planet: p, score, ahead, edgeMiss};
        }
      }
      if (movingAheadBest && movingAheadBest.score <= 5.2) return movingAheadBest.planet;

      let currentBest = null;
      for (const p of planets) {
        const edgeDistance = dist(aim, p) - p.r;
        const score = Math.max(0, edgeDistance) - p.r * 0.04;
        if (!currentBest || score < currentBest.score) currentBest = {planet: p, score};
      }
      if (currentBest && currentBest.score <= 5.2) return currentBest.planet;

      let best = null;
      for (const p of planets) {
        const maxTurn = isOrbitingPlanet(p, initialById) ? 70 : 0;
        for (let turn = 1; turn <= maxTurn; turn += 1) {
          const pos = planetPositionAfterTurns(p, turn, initialById);
          const edgeDistance = dist(aim, pos) - p.r;
          const score = Math.max(0, edgeDistance) + turn * 0.012;
          if (!best || score < best.score) best = {planet: p, score, turn};
        }
      }
      return best && best.score <= 3.2 ? best.planet : null;
    }
    function focusedAimTarget(source, angle) {
      if (aimLocked && aimTargetId !== null) {
        const lockedTarget = (obs().planets || [])
          .map(planetObj)
          .find(p => p.id === aimTargetId && p.id !== source.id);
        if (lockedTarget) return lockedTarget;
      }
      return nearestPlanetToAimPoint(source);
    }
    function fleetSegmentAtTurn(source, angle, ships, turn) {
      const dir = {x: Math.cos(angle), y: Math.sin(angle)};
      const speed = fleetSpeed(ships);
      const start = {x: source.x + dir.x * (source.r + 0.1), y: source.y + dir.y * (source.r + 0.1)};
      return {
        oldPos: {x: start.x + dir.x * speed * Math.max(0, turn - 1), y: start.y + dir.y * speed * Math.max(0, turn - 1)},
        newPos: {x: start.x + dir.x * speed * turn, y: start.y + dir.y * speed * turn},
      };
    }
    function targetSegmentAtTurn(target, turn, initialById = initialPlanetById()) {
      return {
        oldPos: planetPositionAfterTurns(target, Math.max(0, turn - 1), initialById),
        newPos: planetPositionAfterTurns(target, turn, initialById),
      };
    }
    function shotHitsTargetAtTurn(source, target, angle, ships, turn, initialById = initialPlanetById()) {
      const fleet = fleetSegmentAtTurn(source, angle, ships, turn);
      const targetMove = targetSegmentAtTurn(target, turn, initialById);
      const timing = targetAimTiming(source, target, angle, ships, turn, initialById);
      if (timing.hit) return true;
      return segmentToSegmentDistance(fleet.oldPos, fleet.newPos, targetMove.oldPos, targetMove.newPos) < target.r;
    }
    function targetAimTiming(source, target, angle, ships, turn, initialById = initialPlanetById()) {
      const dir = {x: Math.cos(angle), y: Math.sin(angle)};
      const speed = fleetSpeed(ships);
      const start = {x: source.x + dir.x * (source.r + 0.1), y: source.y + dir.y * (source.r + 0.1)};
      const targetPos = planetPositionAfterTurns(target, turn, initialById);
      const rel = {x: targetPos.x - start.x, y: targetPos.y - start.y};
      const along = rel.x * dir.x + rel.y * dir.y;
      if (along < 0) return {hit: false, perp: Infinity, timeError: Infinity};
      const closest = {x: start.x + dir.x * along, y: start.y + dir.y * along};
      const perp = dist(targetPos, closest);
      const fleetTurn = along / Math.max(0.001, speed);
      const timeError = Math.abs(fleetTurn - turn);
      const timeTolerance = Math.max(1.5, target.r / Math.max(1, speed) + 0.75);
      return {
        hit: perp <= target.r + 0.75 && timeError <= timeTolerance,
        perp,
        timeError,
      };
    }
    function predictShot(source, angle, ships, focusTarget) {
      const dir = {x: Math.cos(angle), y: Math.sin(angle)};
      const speed = fleetSpeed(ships);
      const start = {x: source.x + dir.x * (source.r + 0.1), y: source.y + dir.y * (source.r + 0.1)};
      const initialById = initialPlanetById();
      const planets = (obs().planets || []).map(planetObj);
      let pos = start;
      let bestFocusTurn = null;

      for (let turn = 1; turn <= 80; turn++) {
        const oldPos = pos;
        const newPos = {x: start.x + dir.x * speed * turn, y: start.y + dir.y * speed * turn};
        const focusHitThisTurn = focusTarget && shotHitsTargetAtTurn(source, focusTarget, angle, ships, turn, initialById);
        if (focusHitThisTurn && bestFocusTurn === null) bestFocusTurn = turn;

        let bestHit = null;
        for (const planet of planets) {
          if (planet.id === source.id && turn <= 1) continue;
          const targetMove = targetSegmentAtTurn(planet, turn, initialById);
          if (segmentToSegmentDistance(oldPos, newPos, targetMove.oldPos, targetMove.newPos) < planet.r) {
            const progress = segmentCollisionProgress(oldPos, newPos, targetMove.oldPos, targetMove.newPos);
            if (!bestHit || progress < bestHit.progress) {
              bestHit = {planet, progress};
            }
          }
        }
        if (bestHit) {
          return {
            hitType: 'planet',
            hitPlanetId: bestHit.planet.id,
            hitTurn: turn,
            willHitFocus: Boolean(focusTarget && bestHit.planet.id === focusTarget.id),
            bestFocusTurn,
          };
        }

        if (focusHitThisTurn) {
          return {
            hitType: 'planet',
            hitPlanetId: focusTarget.id,
            hitTurn: turn,
            willHitFocus: true,
            bestFocusTurn,
          };
        }

        if (pointToSegmentDistance({x: CENTER, y: CENTER}, oldPos, newPos) < SUN_RADIUS) {
          return {hitType: 'sun', hitPlanetId: null, hitTurn: turn, willHitFocus: false, bestFocusTurn};
        }
        if (newPos.x < 0 || newPos.x > BOARD_SIZE || newPos.y < 0 || newPos.y > BOARD_SIZE) {
          return {hitType: 'edge', hitPlanetId: null, hitTurn: turn, willHitFocus: false, bestFocusTurn};
        }

        pos = newPos;
      }
      return {hitType: 'none', hitPlanetId: null, hitTurn: null, willHitFocus: false, bestFocusTurn};
    }
    function findLockedTargetIntercept(source, target, ships) {
      const initialById = initialPlanetById();
      const clampedShips = Math.max(1, Math.floor(Number(ships || 1)));
      for (let turn = 1; turn <= 80; turn += 1) {
        const targetPos = planetPositionAfterTurns(target, turn, initialById);
        const angle = Math.atan2(targetPos.y - source.y, targetPos.x - source.x);
        if (!shotHitsTargetAtTurn(source, target, angle, clampedShips, turn, initialById)) continue;
        const prediction = predictShot(source, angle, clampedShips, target);
        if (prediction?.willHitFocus) {
          const hitTurn = prediction.hitTurn ?? turn;
          return {
            angle,
            turn: hitTurn,
            point: planetPositionAfterTurns(target, hitTurn, initialById),
          };
        }
      }
      return null;
    }
    function retargetLockedAimForShips(ships) {
      if (!selectedSource || !aimLocked || aimTargetId === null) return false;
      const target = (obs().planets || [])
        .map(planetObj)
        .find(p => p.id === aimTargetId && p.id !== selectedSource.id);
      if (!target) return false;
      const intercept = findLockedTargetIntercept(selectedSource, target, ships);
      if (!intercept) return false;
      aim = intercept.point;
      aimTargetId = target.id;
      aimLocked = true;
      setBothAngleInputs(intercept.angle);
      return true;
    }
    function predictActionMoveFromObs(snapshotObs, move, playerId) {
      if (!snapshotObs || !Array.isArray(move) || move.length !== 3) {
        return {ok: false, reason: 'bad move'};
      }
      const sourceId = Number(move[0]);
      const angle = Number(move[1]);
      const ships = Math.floor(Number(move[2]));
      const planets = (snapshotObs.planets || []).map(planetObj);
      const source = planets.find(p => p.id === sourceId);
      if (!source) return {ok: false, sourceId, angle, ships, reason: 'source missing'};
      if (source.owner !== Number(playerId)) return {ok: false, sourceId, angle, ships, reason: `source owner p${source.owner}`};
      if (!Number.isFinite(angle) || ships <= 0) return {ok: false, sourceId, angle, ships, reason: 'bad angle/ships'};

      const initialRows = snapshotObs.initial_planets || snapshotObs.planets || [];
      const initialById = new Map(initialRows.map(row => [row[0], planetObj(row)]));
      const cometIds = new Set(snapshotObs.comet_planet_ids || []);
      const angularVelocity = Number(snapshotObs.angular_velocity || 0);
      const step = Number(snapshotObs.step || 0);
      const positionAfterTurns = (planet, turns) => {
        if (cometIds.has(planet.id)) return {x: planet.x, y: planet.y};
        const initial = initialById.get(planet.id);
        if (!initial) return {x: planet.x, y: planet.y};
        const orbitalRadius = Math.hypot(initial.x - CENTER, initial.y - CENTER);
        if (orbitalRadius + initial.r >= ROTATION_RADIUS_LIMIT) return {x: planet.x, y: planet.y};
        const initialAngle = Math.atan2(initial.y - CENTER, initial.x - CENTER);
        const envStep = Math.max(1, step) + Math.max(0, turns) - 1;
        return {
          x: CENTER + orbitalRadius * Math.cos(initialAngle + angularVelocity * envStep),
          y: CENTER + orbitalRadius * Math.sin(initialAngle + angularVelocity * envStep),
        };
      };

      const dir = {x: Math.cos(angle), y: Math.sin(angle)};
      const speed = fleetSpeed(ships);
      const start = {x: source.x + dir.x * (source.r + 0.1), y: source.y + dir.y * (source.r + 0.1)};
      let oldPos = start;
      for (let turn = 1; turn <= 120; turn += 1) {
        const newPos = {x: start.x + dir.x * speed * turn, y: start.y + dir.y * speed * turn};
        let bestHit = null;
        for (const planet of planets) {
          if (planet.id === source.id && turn <= 1) continue;
          const planetOld = positionAfterTurns(planet, Math.max(0, turn - 1));
          const planetNew = positionAfterTurns(planet, turn);
          const hit = segmentToSegmentDistance(oldPos, newPos, planetOld, planetNew) < planet.r;
          if (hit) {
            const progress = segmentCollisionProgress(oldPos, newPos, planetOld, planetNew);
            if (!bestHit || progress < bestHit.progress) {
              bestHit = {planet, progress};
            }
          }
        }
        if (bestHit) {
          return {
            ok: true,
            sourceId,
            sourceOwner: Number(playerId),
            angle,
            ships,
            hitType: 'planet',
            targetId: bestHit.planet.id,
            targetOwner: bestHit.planet.owner,
            hitTurn: turn,
            etaStep: step + turn,
          };
        }
        if (pointToSegmentDistance({x: CENTER, y: CENTER}, oldPos, newPos) < SUN_RADIUS) {
          return {ok: true, sourceId, sourceOwner: Number(playerId), angle, ships, hitType: 'sun', targetId: null, hitTurn: turn, etaStep: step + turn};
        }
        if (newPos.x < 0 || newPos.x > BOARD_SIZE || newPos.y < 0 || newPos.y > BOARD_SIZE) {
          return {ok: true, sourceId, sourceOwner: Number(playerId), angle, ships, hitType: 'edge', targetId: null, hitTurn: turn, etaStep: step + turn};
        }
        oldPos = newPos;
      }
      return {ok: true, sourceId, sourceOwner: Number(playerId), angle, ships, hitType: 'none', targetId: null, hitTurn: null, etaStep: null};
    }
    function logActionBreadcrumbs(snapshotObs, moves, playerId, label) {
      if (!Array.isArray(moves) || !moves.length) return;
      moves.forEach((move, index) => {
        const prediction = predictActionMoveFromObs(snapshotObs, move, playerId);
        if (!prediction.ok) {
          log(`${label} launch ${index + 1}: p${move?.[0]} ships=${move?.[2]} could not infer target (${prediction.reason}).`);
          return;
        }
        const angleText = Number(prediction.angle).toFixed(3);
        if (prediction.hitType === 'planet') {
          showToast(`${label}: <span>p${prediction.sourceId} -> p${prediction.targetId}</span> in ${prediction.hitTurn} steps`);
          log(`${label} launch ${index + 1}: p${prediction.sourceId}->p${prediction.targetId} ships=${prediction.ships} eta=${prediction.etaStep} (${prediction.hitTurn}t) angle=${angleText}.`);
        } else if (prediction.hitType === 'sun' || prediction.hitType === 'edge') {
          showToast(`${label}: <span>p${prediction.sourceId} -> ${prediction.hitType}</span> in ${prediction.hitTurn} steps`);
          log(`${label} launch ${index + 1}: p${prediction.sourceId}->${prediction.hitType} ships=${prediction.ships} eta=${prediction.etaStep} (${prediction.hitTurn}t) angle=${angleText}.`);
        } else {
          showToast(`${label}: <span>p${prediction.sourceId} -> ?</span> no hit`);
          log(`${label} launch ${index + 1}: p${prediction.sourceId}->? ships=${prediction.ships} no hit within 120t angle=${angleText}.`);
        }
      });
    }
    function queuedShipsBySource() {
      const totals = new Map();
      queuedMoves.forEach(move => {
        totals.set(move[0], (totals.get(move[0]) || 0) + move[2]);
      });
      return totals;
    }
    function planetAt(pt) {
      const candidates = (obs().planets || [])
        .map(planetObj)
        .map(p => ({planet: p, d: dist(pt, p), hit: Math.max(p.r + 3.5, 5.2)}))
        .filter(item => item.d <= item.hit)
        .sort((a, b) => a.d - b.d);
      return candidates[0]?.planet || null;
    }
    function setBothShipInputs(value, retarget = true) {
      const ships = Math.max(1, Math.floor(Number(value || 1)));
      document.getElementById('shipsInput').value = ships;
      document.getElementById('menuShipsInput').value = ships;
      if (retarget) retargetLockedAimForShips(ships);
    }
    function setBothAngleInputs(rad) {
      const angle = Number(rad || 0);
      document.getElementById('angleInput').value = angle.toFixed(6);
      document.getElementById('menuAngleInput').value = (angle * 180 / Math.PI).toFixed(1);
    }
    function setAimFromAngle(angle) {
      if (!selectedSource) return;
      const currentDistance = aim ? Math.max(8, dist(selectedSource, aim)) : 24;
      aim = {
        x: selectedSource.x + Math.cos(angle) * currentDistance,
        y: selectedSource.y + Math.sin(angle) * currentDistance,
      };
      aimLocked = true;
    }
    function setAngleFromDegrees(deg) {
      const angle = Number(deg || 0) * Math.PI / 180;
      document.getElementById('angleInput').value = angle.toFixed(6);
      document.getElementById('menuAngleInput').value = Number(deg || 0).toFixed(1);
      setAimFromAngle(angle);
      render();
    }
    function showFleetMenu() {
      const menu = document.getElementById('fleetMenu');
      if (!selectedSource) {
        menu.style.display = 'none';
        return;
      }
      menu.style.display = 'block';
    }
    function closeFleetMenu() {
      selectedSource = null;
      aim = null;
      aimTargetId = null;
      isDraggingAim = false;
      aimLocked = false;
      document.getElementById('fleetMenu').style.display = 'none';
      document.getElementById('selectionText').textContent = 'Click one of your planets, then drag or move the mouse to aim.';
      render();
    }
    function addCurrentMove() {
      if (!selectedSource) {
        log('Pick one of your planets first.');
        return;
      }
      if (!aim) {
        log('Aim on the board first.');
        return;
      }
      const ships = Math.max(1, Math.floor(Number(document.getElementById('menuShipsInput').value || document.getElementById('shipsInput').value || 0)));
      const angle = Number(document.getElementById('angleInput').value || 0);
      queuedMoves.push([selectedSource.id, angle, ships]);
      log(`Queued p${selectedSource.id}, angle=${angle.toFixed(3)}, ships=${ships}`);
      selectedSource = null;
      aim = null;
      aimTargetId = null;
      isDraggingAim = false;
      aimLocked = false;
      document.getElementById('fleetMenu').style.display = 'none';
      updateStats();
      render();
    }
    function drawCircle(x, y, r, color, fill = false, width = 2) {
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), r * scale(), 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      fill ? ctx.fill() : ctx.stroke();
    }
    function drawSelectionRing(planet, color, label = '') {
      if (!planet) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.95;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(sx(planet.x), sy(planet.y), (planet.r + 2.6) * scale(), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      if (label) {
        drawText(label, planet.x, planet.y - planet.r - 4.2, color, 5.4);
      }
      ctx.restore();
    }
    function drawOutlierRing(planet) {
      if (!planet) return;
      ctx.save();
      ctx.strokeStyle = '#ff4655';
      ctx.lineWidth = 2.6;
      ctx.globalAlpha = 0.98;
      ctx.setLineDash([2.2, 3.4]);
      ctx.beginPath();
      ctx.arc(sx(planet.x), sy(planet.y), (planet.r + 2.25) * scale(), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = '#ff4655';
      ctx.beginPath();
      ctx.arc(sx(planet.x), sy(planet.y), (planet.r + 2.25) * scale(), 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();
    }
    function drawRoleRing(planet, label = '') {
      if (!planet) return;
      ctx.save();
      ctx.strokeStyle = '#b46cff';
      ctx.fillStyle = '#b46cff';
      ctx.lineWidth = 2.7;
      ctx.globalAlpha = 0.96;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.arc(sx(planet.x), sy(planet.y), (planet.r + 2.35) * scale(), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.12;
      ctx.beginPath();
      ctx.arc(sx(planet.x), sy(planet.y), (planet.r + 2.35) * scale(), 0, 2 * Math.PI);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (label) drawText(label, planet.x, planet.y - planet.r - 4.2, '#c995ff', 5.6);
      ctx.restore();
    }
    function drawText(text, x, y, color, size = 13, align = 'center') {
      ctx.font = `700 ${Math.max(8, size * scale() / 5.2)}px Avenir Next, sans-serif`;
      ctx.fillStyle = color;
      ctx.textAlign = align;
      ctx.textBaseline = 'middle';
      ctx.fillText(text, sx(x), sy(y));
    }
    function openingDebug() {
      return state?.agent_debug || null;
    }
    function eliminatedByPlanetId() {
      if (!debugVisible) return new Map();
      const byId = new Map();
      (openingDebug()?.eliminated || []).forEach(item => {
        const id = Number(item.planet_id);
        if (!Number.isFinite(id)) return;
        if (!byId.has(id)) byId.set(id, new Set());
        byId.get(id).add(item.reason || 'eliminated');
      });
      return byId;
    }
    function selectedDebugTargets() {
      if (!debugVisible) return new Set();
      return new Set((openingDebug()?.selected || []).map(item => Number(item.target_id)));
    }
    function drawCross(planet, reasons) {
      const radius = Math.max(planet.r * 0.72, 1.9);
      ctx.save();
      ctx.strokeStyle = '#ff4f62';
      ctx.lineWidth = 1.55;
      ctx.globalAlpha = 0.86;
      ctx.beginPath();
      ctx.moveTo(sx(planet.x - radius), sy(planet.y - radius));
      ctx.lineTo(sx(planet.x + radius), sy(planet.y + radius));
      ctx.moveTo(sx(planet.x + radius), sy(planet.y - radius));
      ctx.lineTo(sx(planet.x - radius), sy(planet.y + radius));
      ctx.stroke();
      ctx.restore();
      if (reasons?.size) {
        const label = [...reasons][0].replace('opponent quadrant', 'opp quad');
        if (label !== 'outside top 37%') {
          drawText(label, planet.x, planet.y + planet.r + 2.7, '#ff8d98', 3.2);
        }
      }
    }
    function drawArrow(from, angle, ships, color, alpha = 0.95) {
      const len = Math.min(16, 7 + Math.log(Math.max(1, ships)) * 3);
      const x1 = from.x + Math.cos(angle) * len;
      const y1 = from.y + Math.sin(angle) * len;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sx(from.x), sy(from.y));
      ctx.lineTo(sx(x1), sy(y1));
      ctx.stroke();
      ctx.translate(sx(x1), sy(y1));
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-8, -5);
      ctx.lineTo(-8, 5);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    function drawQueuedOrder(from, angle, ships, color) {
      const dirX = Math.cos(angle);
      const dirY = Math.sin(angle);
      const perpX = -dirY;
      const perpY = dirX;
      const startDistance = from.r + 1.1;
      const len = Math.min(18, 8 + Math.log(Math.max(1, ships)) * 3);
      const start = {x: from.x + dirX * startDistance, y: from.y + dirY * startDistance};
      const end = {x: from.x + dirX * (startDistance + len), y: from.y + dirY * (startDistance + len)};

      ctx.save();
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2.1;
      ctx.setLineDash([2.8, 3.8]);
      ctx.beginPath();
      ctx.moveTo(sx(start.x), sy(start.y));
      ctx.lineTo(sx(end.x), sy(end.y));
      ctx.stroke();
      ctx.setLineDash([]);

      const fleetX = from.x + dirX * (startDistance + len * 0.68);
      const fleetY = from.y + dirY * (startDistance + len * 0.68);
      ctx.translate(sx(fleetX), sy(fleetY));
      ctx.rotate(angle);
      const sz = 1.25 * scale();
      ctx.beginPath();
      ctx.moveTo(sz, 0);
      ctx.lineTo(-sz, -sz * 0.72);
      ctx.lineTo(-sz * 0.25, 0);
      ctx.lineTo(-sz, sz * 0.72);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      drawText(String(ships), from.x + dirX * (startDistance + 3.2) + perpX * 2.1, from.y + dirY * (startDistance + 3.2) + perpY * 2.1, color, 5.7);
    }
    function drawInsightGuide(row) {
      if (!row || row.route_ok === false) return;
      const planets = (obs().planets || []).map(planetObj);
      const source = planets.find(p => p.id === Number(row.source_id));
      const target = planets.find(p => p.id === Number(row.target_id));
      if (!source || !target) return;
      const inferredAngle = Math.atan2(target.y - source.y, target.x - source.x);
      const angle = Number.isFinite(Number(row.angle)) ? Number(row.angle) : inferredAngle;
      const roughDistance = Math.max(source.r + target.r + 4, dist(source, target) - source.r - target.r);
      const start = {
        x: source.x + Math.cos(angle) * (source.r + 0.85),
        y: source.y + Math.sin(angle) * (source.r + 0.85),
      };
      const end = {
        x: start.x + Math.cos(angle) * roughDistance,
        y: start.y + Math.sin(angle) * roughDistance,
      };
      const color = ['reinforce', 'save', 'recapture'].includes(row.recommendation) ? '#ffd06a' : '#47ff9a';

      ctx.save();
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.lineCap = 'round';
      ctx.setLineDash([7, 6]);
      ctx.beginPath();
      ctx.moveTo(sx(start.x), sy(start.y));
      ctx.lineTo(sx(end.x), sy(end.y));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.88;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.arc(sx(source.x), sy(source.y), (source.r + 1.7) * scale(), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(sx(target.x), sy(target.y), (target.r + 1.7) * scale(), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    }
    function drawSniperAim(source, target, angle, locked) {
      const dirX = Math.cos(angle);
      const dirY = Math.sin(angle);
      const start = {x: source.x - dirX * 140, y: source.y - dirY * 140};
      const end = {x: source.x + dirX * 140, y: source.y + dirY * 140};
      const color = locked ? '#47a8ff' : 'rgba(71, 168, 255, 0.62)';
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = locked ? 2.2 : 1.45;
      ctx.globalAlpha = locked ? 0.9 : 0.58;
      ctx.beginPath();
      ctx.moveTo(sx(start.x), sy(start.y));
      ctx.lineTo(sx(end.x), sy(end.y));
      ctx.stroke();

      // Dots along the aim ray make the trajectory feel like Kaggle's sniper guide.
      for (let t = 7; t <= 130; t += 5.8) {
        const x = source.x + dirX * t;
        const y = source.y + dirY * t;
        if (x < -WORLD_PAD || x > 100 + WORLD_PAD || y < -WORLD_PAD || y > 100 + WORLD_PAD) continue;
        ctx.beginPath();
        ctx.arc(sx(x), sy(y), Math.max(1.0, 0.26 * scale()), 0, 2 * Math.PI);
        ctx.fill();
      }

      const r = locked ? 1.95 : 1.55;
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2.1;
      ctx.beginPath();
      ctx.arc(sx(target.x), sy(target.y), r * scale(), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sx(target.x - r * 1.45), sy(target.y));
      ctx.lineTo(sx(target.x + r * 1.45), sy(target.y));
      ctx.moveTo(sx(target.x), sy(target.y - r * 1.45));
      ctx.lineTo(sx(target.x), sy(target.y + r * 1.45));
      ctx.stroke();
      ctx.restore();
    }
    function drawTargetHitPreview(source, target, prediction, angle, ships) {
      const initialById = initialPlanetById();
      const willHit = Boolean(prediction?.willHitFocus);
      const isMoving = isOrbitingPlanet(target, initialById);
      const blockerTurn = prediction && !prediction.willHitFocus && prediction.hitTurn !== null ? prediction.hitTurn : null;

      ctx.save();
      ctx.globalAlpha = 0.82;
      ctx.lineWidth = 1.55;

      if (!isMoving) {
        const color = willHit ? '#28e279' : '#ff4655';
        drawCircle(target.x, target.y, target.r + 1.45, color, false, 2.5);
        ctx.restore();
        return;
      }

      const hitTurn = prediction?.hitTurn ?? prediction?.bestFocusTurn;
      for (let turn = 0; turn <= 70; turn += 1) {
        const pos = planetPositionAfterTurns(target, turn, initialById);
        const blocked = blockerTurn !== null && turn > blockerTurn;
        const sectionHit = !blocked && turn > 0 && targetAimTiming(source, target, angle, ships, turn, initialById).hit;
        const nearHit = sectionHit || (willHit && hitTurn !== null && Math.abs(turn - hitTurn) <= 2);
        const color = nearHit ? '#28e279' : '#ff4655';
        ctx.strokeStyle = color;
        ctx.fillStyle = nearHit ? 'rgba(40, 226, 121, 0.12)' : 'rgba(255, 70, 85, 0.12)';
        ctx.globalAlpha = nearHit ? 0.95 : 0.52;
        ctx.lineWidth = nearHit ? 2.15 : 1.25;
        ctx.beginPath();
        ctx.arc(sx(pos.x), sy(pos.y), target.r * scale(), 0, 2 * Math.PI);
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
      ctx.lineWidth = 2.8;
      ctx.strokeStyle = willHit ? '#28e279' : '#ff4655';
      const current = planetPositionAfterTurns(target, 0, initialById);
      ctx.beginPath();
      ctx.arc(sx(current.x), sy(current.y), (target.r + 1.15) * scale(), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
    }
    function render() {
      const o = obs();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const sun = ctx.createRadialGradient(sx(50), sy(50), 5*scale(), sx(50), sy(50), 26*scale());
      sun.addColorStop(0, 'rgba(255,197,60,0.68)');
      sun.addColorStop(0.45, 'rgba(255,158,33,0.14)');
      sun.addColorStop(1, 'rgba(255,100,0,0)');
      ctx.fillStyle = sun;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      drawCircle(50, 50, 10, '#ffc247', true);
      drawCircle(50, 50, 10, '#ffdc6a', false, 2);

      const cometIds = new Set(o.comet_planet_ids || []);
      const queuedBySource = queuedShipsBySource();
      const eliminated = eliminatedByPlanetId();
      const selectedTargets = selectedDebugTargets();
      const outlierIds = planetaryOutlierIds();
      const roleRows = roleRowsByPlanetId();
      (o.planets || []).forEach(raw => {
        const p = planetObj(raw);
        const color = p.owner === -1 ? colors[4] : colors[p.owner];
        const pendingShips = p.owner === humanPlayer() ? (queuedBySource.get(p.id) || 0) : 0;
        drawCircle(p.x, p.y, p.r, color, true);
        if (p.owner === humanPlayer()) drawCircle(p.x, p.y, p.r + 0.55, '#ffffff', false, 2);
        if (p.owner === agentPlayer()) drawCircle(p.x, p.y, p.r + 0.55, 'rgba(0,0,0,0.65)', false, 2);
        if (cometIds.has(p.id)) drawCircle(p.x, p.y, p.r + 0.9, '#dfeaff', false, 2);
        if (selectedTargets.has(p.id)) drawCircle(p.x, p.y, p.r + 2.0, '#47ff9a', false, 2.8);
        if (eliminated.has(p.id)) drawCross(p, eliminated.get(p.id));
        if (pendingShips > 0) {
          drawText(String(Math.max(0, Math.floor(p.ships) - pendingShips)), p.x - 0.35, p.y, '#fff', 10.5);
          drawText(`-${pendingShips}`, p.x + p.r * 0.72, p.y - p.r * 0.55, '#ffd06a', 5.6, 'left');
        } else {
          drawText(String(Math.floor(p.ships)), p.x, p.y, '#fff', 10.5);
        }
        drawText(`+${p.prod}`, p.x, p.y - p.r - 1.35, p.owner === -1 ? 'rgba(170,179,194,0.75)' : color, 6.8);
        drawText(String(p.id), p.x, p.y + p.r + 1.35, 'rgba(174,184,200,0.72)', 6.2);
      });
      if (outlierIds.size) {
        (o.planets || []).forEach(raw => {
          const p = planetObj(raw);
          if (outlierIds.has(p.id)) drawOutlierRing(p);
        });
      }
      if (roleRows.size) {
        (o.planets || []).forEach(raw => {
          const p = planetObj(raw);
          const role = roleRows.get(p.id);
          if (role) drawRoleRing(p, role.role || '');
        });
      }

      const selectedPlanets = (o.planets || []).map(planetObj);
      const clickedTarget = selectedInsightPlanetId === null || selectedInsightPlanetId === undefined
        ? null
        : selectedPlanets.find(p => p.id === Number(selectedInsightPlanetId));
      const clickedSources = selectedInsightSourceIds
        .map(sourceId => selectedPlanets.find(p => p.id === Number(sourceId)))
        .filter(Boolean);
      clickedSources.forEach((clickedSource, index) => {
        if (clickedSource && clickedSource.id !== clickedTarget?.id) {
          drawSelectionRing(clickedSource, 'rgba(255, 208, 106, 0.82)', clickedSources.length > 1 ? `src${index + 1}` : 'src');
        }
      });
      if (clickedTarget) {
        const selectedColor = isIgnoreInsightsTab() ? '#ff4655' : (isRolesInsightsTab() ? '#b46cff' : '#47a8ff');
        drawSelectionRing(clickedTarget, selectedColor, `p${clickedTarget.id}`);
      }

      drawInsightGuide(currentInsightGuideRow());

      (o.fleets || []).forEach(f => {
        const owner = f[1], x = f[2], y = f[3], angle = f[4], ships = f[6];
        const color = colors[owner];
        ctx.save();
        ctx.translate(sx(x), sy(y));
        ctx.rotate(angle);
        const sz = (0.7 + 2.7 * Math.log(Math.max(2, ships)) / Math.log(1000)) * scale();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(sz, 0);
        ctx.lineTo(-sz, -sz * 0.7);
        ctx.lineTo(-sz * 0.25, 0);
        ctx.lineTo(-sz, sz * 0.7);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
        drawText(String(ships), x, y + (y >= 50 ? -2.6 : 2.6), color, 6.8);
      });

      if (selectedSource && aim) {
        const angle = Number(document.getElementById('angleInput').value || 0);
        const ships = Number(document.getElementById('menuShipsInput').value || document.getElementById('shipsInput').value || 1);
        const focusTarget = focusedAimTarget(selectedSource, angle);
        const prediction = focusTarget ? predictShot(selectedSource, angle, ships, focusTarget) : null;
        if (focusTarget) drawTargetHitPreview(selectedSource, focusTarget, prediction, angle, ships);
        drawSniperAim(selectedSource, aim, angle, aimLocked);
      }
      if (selectedSource) {
        drawCircle(selectedSource.x, selectedSource.y, selectedSource.r + 1.5, aimLocked ? '#47a8ff' : '#ffeb9b', false, 3);
      }
      queuedMoves.forEach(move => {
        const src = (o.planets || []).map(planetObj).find(p => p.id === move[0]);
        if (src) drawQueuedOrder(src, move[1], move[2], colors[src.owner] || '#ffd06a');
      });
    }
    function updateStats() {
      const o = obs();
      const planets = (o.planets || []).map(planetObj);
      const shipTotal = player => {
        const planetShips = planets
          .filter(p => p.owner === player)
          .reduce((s, p) => s + Math.floor(Number(p.ships || 0)), 0);
        const fleetShips = (o.fleets || [])
          .filter(f => Number(f[1]) === player)
          .reduce((s, f) => s + Math.floor(Number(f[6] || 0)), 0);
        return planetShips + fleetShips;
      };
      document.getElementById('subtitle').textContent = `Step ${o.step ?? 0} | you are player ${humanPlayer()} vs ${state?.agent}`;
      document.getElementById('stepMetric').textContent = o.step ?? 0;
      document.getElementById('episodeMetric').textContent = state?.episode_steps ?? 500;
      document.getElementById('humanProd').textContent = shipTotal(humanPlayer());
      document.getElementById('agentProd').textContent = shipTotal(agentPlayer());
      document.getElementById('agentName').textContent =
        state?.agent === 'v0' || String(state?.agent || '').endsWith('/agents/v0.py') ? 'V0' :
        'Improved';
      document.getElementById('humanDot').style.background = colors[humanPlayer()];
      document.getElementById('agentDot').style.background = colors[agentPlayer()];
      document.getElementById('backBtn').disabled = !state?.can_go_back;
      updateOpeningDebugPanel();
      updateHumanInsightsPanel();
      const list = document.getElementById('movesList');
      list.innerHTML = queuedMoves.length ? '' : '<div class="hint">No moves queued.</div>';
      queuedMoves.forEach((m, idx) => {
        const div = document.createElement('div');
        div.className = 'move';
        div.innerHTML = `<span>p${m[0]} angle=${m[1].toFixed(3)} ships=${m[2]}</span>`;
        const btn = document.createElement('button');
        btn.className = 'secondary';
        btn.textContent = 'remove';
        btn.onclick = () => { queuedMoves.splice(idx, 1); updateStats(); render(); };
        div.appendChild(btn);
        list.appendChild(div);
      });
      if (state?.done) log(`Game done. Human reward=${state.reward}. Episode auto-saved.`);
    }
    function openingMatrixModel(debug) {
      const selectedByKey = new Map((debug.selected || []).map(item => [`${item.source_id}:${item.target_id}`, item]));
      const rowsByKey = new Map();
      const sourceSet = new Set();
      (debug.comparisons || []).forEach(row => {
        const sourceId = Number(row.source_id);
        const targetId = Number(row.target_id);
        if (!Number.isFinite(sourceId) || !Number.isFinite(targetId)) return;
        sourceSet.add(sourceId);
        rowsByKey.set(`${sourceId}:${targetId}`, row);
      });
      const planets = (obs().planets || []).map(planetObj);
      const sourceIds = planets
        .filter(planet => planet.owner === agentPlayer())
        .map(planet => planet.id)
        .sort((a, b) => a - b);
      if (!sourceIds.length) {
        [...sourceSet].sort((a, b) => a - b).forEach(sourceId => sourceIds.push(sourceId));
      }
      const targetIds = planets.map(planet => planet.id).sort((a, b) => a - b);
      const eliminatedReasonByTarget = new Map();
      (debug.eliminated || []).forEach(item => {
        const planetId = Number(item.planet_id);
        if (!Number.isFinite(planetId) || eliminatedReasonByTarget.has(planetId)) return;
        eliminatedReasonByTarget.set(planetId, item.reason || 'filtered');
      });
      const eliminatedCount = new Set((debug.eliminated || []).map(item => Number(item.planet_id))).size;
      return {selectedByKey, rowsByKey, sourceIds, targetIds, eliminatedReasonByTarget, eliminatedCount};
    }
    function buildOpeningMatrix(debug) {
      const model = openingMatrixModel(debug);
      const wrap = document.createElement('div');
      wrap.className = 'debug-matrix-wrap';
      wrap.title = 'Double-click to open full-screen matrix';
      wrap.ondblclick = openOpeningMatrixModal;
      if (!model.sourceIds.length || !model.targetIds.length) {
        wrap.innerHTML = '<div class="debug-note">No matrix rows for this state.</div>';
        return wrap;
      }
      const table = document.createElement('table');
      table.className = 'debug-matrix';
      const thead = document.createElement('thead');
      const headRow = document.createElement('tr');
      const corner = document.createElement('th');
      corner.textContent = 'S\\T';
      headRow.appendChild(corner);
      model.targetIds.forEach(targetId => {
        const th = document.createElement('th');
        th.textContent = `p${targetId}`;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      model.sourceIds.forEach(sourceId => {
        const tr = document.createElement('tr');
        const sourceHead = document.createElement('th');
        sourceHead.className = 'source-head';
        sourceHead.textContent = `p${sourceId}`;
        tr.appendChild(sourceHead);
        model.targetIds.forEach(targetId => {
          const key = `${sourceId}:${targetId}`;
          const row = model.rowsByKey.get(key);
          const td = document.createElement('td');
          const selected = model.selectedByKey.get(key);
          if (!row) {
            td.className = 'empty-cell';
            td.textContent = '·';
            td.title = `p${sourceId}->p${targetId}: ${model.eliminatedReasonByTarget.get(targetId) || 'no comparison row'}`;
          } else {
            const net = Number(row.net_value);
            td.textContent = Number.isFinite(net) ? net.toFixed(0) : '';
            td.className = `${net >= 0 ? 'positive-cell' : 'negative-cell'}${selected ? ' selected-cell' : ''}`;
            td.title = `p${sourceId}->p${targetId} net=${Number(row.net_value).toFixed(1)} wait=${row.wait_turns} travel=${row.travel_turns} ships=${row.ships_needed}${selected ? ` selected ${selected.kind || 'launch'}` : ''}`;
          }
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      return wrap;
    }
    function selectedSummary(debug) {
      const selected = debug.selected || [];
      if (!selected.length) return 'Selected: none this step';
      return 'Selected: ' + selected
        .map(item => `p${item.source_id}->p${item.target_id}${item.kind && item.kind !== 'launch' ? ` ${item.kind}` : ''}`)
        .join(', ');
    }
    function updateOpeningMatrixModal() {
      const modal = document.getElementById('matrixModal');
      const body = document.getElementById('matrixModalBody');
      const subtitle = document.getElementById('matrixModalSubtitle');
      modal.classList.toggle('open', matrixModalOpen);
      modal.setAttribute('aria-hidden', matrixModalOpen ? 'false' : 'true');
      if (!matrixModalOpen) return;
      const debug = openingDebug();
      body.innerHTML = '';
      body.className = 'matrix-modal-body';
      if (!debugVisible || !debug) {
        subtitle.textContent = 'Turn on Debug with an opening agent selected.';
        body.innerHTML = '<div class="debug-note">Turn on Debug to inspect the agent matrix.</div>';
        return;
      }
      if (debug.error) {
        subtitle.textContent = 'Debug error';
        body.textContent = `debug error: ${debug.error}`;
        body.className = 'matrix-modal-body debug-note';
        return;
      }
      const model = openingMatrixModel(debug);
      subtitle.textContent = `Step ${obs().step ?? 0} | crossed planets: ${model.eliminatedCount} | blocked quadrant: ${(debug.blocked_quadrant || []).join(',')} | ${selectedSummary(debug)}`;
      body.className = 'matrix-modal-body';
      body.appendChild(buildOpeningMatrix(debug));
    }
    function openOpeningMatrixModal() {
      matrixModalOpen = true;
      updateOpeningMatrixModal();
    }
    function closeOpeningMatrixModal() {
      matrixModalOpen = false;
      updateOpeningMatrixModal();
    }
    function fmtNumber(value, digits = 0) {
      const num = Number(value);
      if (!Number.isFinite(num)) return '';
      return num.toFixed(digits);
    }
    function isInsightActionNow(row) {
      return Boolean(
        row
        && row.source_id !== null
        && row.source_id !== undefined
        && row.target_id !== null
        && row.target_id !== undefined
        && row.route_ok !== false
        && row.source_wait_insufficient !== true
        && row.source_survival_blocked !== true
        && Number(row.wait_turns || 0) <= 0
      );
    }
    function insightRowKey(row) {
      if (!row) return '';
      return `${row.recommendation || ''}:${row.source_id}:${row.target_id}:${row.ships_needed}:${row.wait_turns}`;
    }
    function isIgnoreInsightsTab() {
      return activeHumanInsightsTab === 'ignore' || activeHumanInsightsTab === 'planetary';
    }
    function isRolesInsightsTab() {
      return activeHumanInsightsTab === 'roles';
    }
    function currentInsightGuideRow() {
      if (!insightsVisible) return null;
      if (isIgnoreInsightsTab() || isRolesInsightsTab()) return null;
      if (activeHumanInsightsTab === 'multi') return null;
      const report = state?.human_analysis?.opening;
      if (!report || report.error) return null;
      const reinforcementRows = report.priority_rows || report.reinforcement_rows || [];
      const reinforcementNow = reinforcementRows.filter(isInsightActionNow);
      if (reinforcementNow.length) return reinforcementNow[0];
      const rows = activeHumanInsightsTab === 'reinforce'
        ? (report.reinforcement_rows || [])
        : activeHumanInsightsTab === 'saves'
          ? (report.save_rows || [])
          : activeHumanInsightsTab === 'recaptures'
            ? (report.recapture_rows || [])
            : (report.rows || []);
      return rows.filter(isInsightActionNow)[0] || null;
    }
    function planetaryOutlierIds() {
      if (!insightsVisible || !isIgnoreInsightsTab()) return new Set();
      const report = state?.human_analysis?.planetary;
      if (!report || report.error) return new Set();
      return new Set((report.outlier_rows || []).map(row => Number(row.planet_id)));
    }
    function roleRowsByPlanetId() {
      if (!insightsVisible || !isRolesInsightsTab()) return new Map();
      const report = state?.human_analysis?.opening?.role_report || {};
      const rows = report.rows || state?.human_analysis?.opening?.role_rows || [];
      return new Map(rows.map(row => [Number(row.planet_id), row]));
    }
    function selectInsightPlanet(targetId, sourceId = null, sourceIds = null) {
      selectedInsightPlanetId = targetId === null || targetId === undefined ? null : Number(targetId);
      selectedInsightSourceId = sourceId === null || sourceId === undefined ? null : Number(sourceId);
      selectedInsightSourceIds = Array.isArray(sourceIds)
        ? sourceIds.map(Number).filter(Number.isFinite)
        : selectedInsightSourceId === null || selectedInsightSourceId === undefined
          ? []
          : [Number(selectedInsightSourceId)];
      updateSelectedInsightRows();
      render();
    }
    function isSelectedInsightRow(row) {
      if (!row) return false;
      if (selectedInsightPlanetId === null || selectedInsightPlanetId === undefined) return false;
      const targetId = row.target_id ?? row.planet_id;
      const rowSourceId = row.source_id === null || row.source_id === undefined ? null : Number(row.source_id);
      return Number(targetId) === Number(selectedInsightPlanetId)
        && rowSourceId === selectedInsightSourceId;
    }
    function updateSelectedInsightRows() {
      document.querySelectorAll('#humanInsightsBody .opening-report tbody tr').forEach(row => {
        const targetId = row.dataset.targetId === '' ? null : Number(row.dataset.targetId);
        const sourceId = row.dataset.sourceId === '' ? null : Number(row.dataset.sourceId);
        row.classList.toggle(
          'row-selected',
          targetId === selectedInsightPlanetId && sourceId === selectedInsightSourceId
        );
      });
    }
    function decorateInsightRow(tr, row) {
      const canActNow = isInsightActionNow(row);
      const guide = currentInsightGuideRow();
      const primary = canActNow && insightRowKey(row) === insightRowKey(guide);
      tr.classList.toggle('action-now', primary);
      tr.classList.toggle('primary-guide', primary);
      tr.classList.toggle('best-route', Boolean(row.is_best));
      tr.classList.toggle('source-unsafe', Boolean(row.source_survival_blocked));
      tr.classList.toggle('row-selected', isSelectedInsightRow(row));
      tr.dataset.sourceId = row.source_id ?? '';
      tr.dataset.targetId = row.target_id ?? '';
      tr.addEventListener('click', () => selectInsightPlanet(row.target_id, row.source_id));
      tr.title = primary
        ? 'Best clear move available now.'
        : row.source_wait_insufficient === true
          ? `Insufficient source: p${row.source_id} has ${row.source_available_at_launch ?? 0} available at launch, but this move needs ${row.ships_needed ?? '?'}.`
          : row.source_survival_blocked === true
          ? `Unsafe source: p${row.source_id} can safely spare ${row.source_max_safe_ships ?? 0}, but this move needs ${row.ships_needed ?? '?'}.`
          : row.route_ok === false
          ? 'Not highlighted because this source-target route is blocked.'
          : `Wait ${row.wait_turns ?? '?'} step(s) before acting.`;
      return {actionNow: primary, canActNow, primary};
    }
    function buildPriorityStrip(report) {
      const rows = report?.priority_rows || report?.reinforcement_rows || [];
      if (!rows.length) return null;
      const row = rows[0];
      const strip = document.createElement('div');
      strip.className = 'priority-strip';
      const source = row.source_id === null || row.source_id === undefined ? '-' : `p${row.source_id}`;
      const by = row.reinforce_by_turn === null || row.reinforce_by_turn === undefined ? '?' : row.reinforce_by_turn;
      const outcome = row.rescue_outcome || (row.rescue_timely === true || row.reinforces_before_loss === true ? 'saves' : 'recaptures');
      const label = row.recommendation === 'reinforce'
        ? (outcome === 'saves' ? 'Priority reinforce' : 'Recover reinforce')
        : (outcome === 'saves' ? 'Save capture' : 'Recapture capture');
      strip.innerHTML = `${label} <span>p${row.target_id} from ${source}</span> wait <span>${row.wait_turns}</span> send <span>${row.ships_needed}</span> arrive <span>${row.arrival_turns}</span> danger <span>${by}</span> <span>${outcome}</span>`;
      return strip;
    }
    function buildOpeningReport(report) {
      const root = document.createElement('div');
      if (!report || report.error) {
        root.className = 'debug-note';
        root.textContent = report?.error ? `opening report error: ${report.error}` : 'No opening report available.';
        return root;
      }

      const meta = document.createElement('div');
      meta.className = 'opening-report-meta';
      meta.innerHTML = [
        `Home ${report.home_quadrant || '?'}`,
        `Frontiers ${(report.frontier_quadrants || []).join(', ') || '?'}`,
        `Enemy ${report.enemy_quadrant || '?'}`,
        report.enemy_high_prod_centroid ? `Enemy core (${report.enemy_high_prod_centroid.x}, ${report.enemy_high_prod_centroid.y})` : 'Enemy core ?',
        `Unlock ${report.outlier_unlock_progress?.progress_percent ?? 0}%`,
        `Unlocked ${(report.unlocked_outlier_target_ids || []).length}`,
        `Saves ${(report.save_rows || []).length}`,
        `Recaptures ${(report.recapture_rows || []).length}`,
        `Reinforce ${(report.reinforcement_rows || []).length}`,
        `Multi ${report.multi_source_opportunity_count || 0}`,
        `Unsafe sources ${report.unsafe_source_route_count || 0}`,
        `Insufficient ${(report.insufficient_source_route_count || 0)}`,
        `Excluded ${(report.excluded_outlier_target_ids || []).length}`,
        `Hidden handled ${(report.handled_targets || []).length}`,
      ].map(text => `<span>${text}</span>`).join('');
      root.appendChild(meta);

      const priority = buildPriorityStrip(report);
      if (priority) root.appendChild(priority);

      const wrap = document.createElement('div');
      wrap.className = 'opening-report-wrap';
      const table = document.createElement('table');
      table.className = 'opening-report';
      table.innerHTML = `
        <thead>
          <tr>
            <th>Target</th>
            <th>Src</th>
            <th>Q</th>
            <th>Role</th>
            <th>Motion</th>
            <th>Status</th>
            <th>Prod</th>
            <th>Ships</th>
            <th>Src Safe</th>
            <th>Wait</th>
            <th>Travel</th>
            <th>Own Prod</th>
            <th>Surv +</th>
            <th>Base</th>
            <th>Low Pen</th>
            <th>High +</th>
            <th>Core +</th>
            <th>Final</th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement('tbody');
      (report.rows || []).forEach(row => {
        const finalValue = Number(row.strategic_net);
        const tr = document.createElement('tr');
        const rowState = decorateInsightRow(tr, row);
        tr.innerHTML = `
          <td>p${row.target_id}</td>
          <td>${row.source_id === null || row.source_id === undefined ? '-' : `p${row.source_id}`}</td>
          <td>${row.quadrant || ''}</td>
          <td>${row.role || ''}</td>
          <td>${row.motion || ''}</td>
          <td>${row.source_wait_insufficient === true ? 'insufficient source' : row.source_survival_blocked === true ? 'unsafe source' : row.claim_status || (row.route_ok === false ? row.route_status || 'blocked shot' : '')}</td>
          <td>${row.production}</td>
          <td>${row.ships_needed}</td>
          <td>${row.source_max_safe_ships ?? ''}</td>
          <td>${rowState.actionNow ? '<span class="now-badge">now</span>' : row.wait_turns}</td>
          <td>${row.travel_turns}</td>
          <td>${row.owned_production ?? ''}</td>
          <td>${row.survival_extra_ships ?? 0}</td>
          <td>${fmtNumber(row.tactical_net)}</td>
          <td>${fmtNumber(row.low_prod_penalty)}</td>
          <td>${fmtNumber(row.high_prod_bonus)}</td>
          <td>${fmtNumber(row.enemy_centroid_bonus)}</td>
          <td class="${finalValue >= 0 ? 'good-value' : 'bad-value'}">${fmtNumber(row.strategic_net)}</td>
        `;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      root.appendChild(wrap);
      return root;
    }
    function buildMultiSourceReport(report) {
      const root = document.createElement('div');
      if (!report || report.error) {
        root.className = 'debug-note';
        root.textContent = report?.error ? `multi-source report error: ${report.error}` : 'No multi-source report available.';
        return root;
      }

      const rows = report.multi_source_rows || [];
      const meta = document.createElement('div');
      meta.className = 'opening-report-meta';
      meta.innerHTML = [
        `Opportunities ${rows.length}`,
        `Targets ${report.multi_source_target_count || 0}`,
        report.multi_source_target_mode || `ignored high-prod outliers`,
        `2 sources only`,
        `Launch now`,
        `route + source safety checked`,
      ].map(text => `<span>${text}</span>`).join('');
      root.appendChild(meta);

      if (!rows.length) {
        const empty = document.createElement('div');
        empty.className = 'debug-note';
        empty.textContent = 'No safe 2-source capture opportunity right now.';
        root.appendChild(empty);
        return root;
      }

      const wrap = document.createElement('div');
      wrap.className = 'opening-report-wrap';
      const table = document.createElement('table');
      table.className = 'opening-report';
      table.innerHTML = `
        <thead>
          <tr>
            <th>Target</th>
            <th>Q</th>
            <th>Prod</th>
            <th>Ships</th>
            <th>O/H</th>
            <th>Req</th>
            <th>In F/E</th>
            <th>Src A</th>
            <th>Send A</th>
            <th>Safe A</th>
            <th>Wt A</th>
            <th>Travel A</th>
            <th>Src B</th>
            <th>Send B</th>
            <th>Safe B</th>
            <th>Wt B</th>
            <th>Travel B</th>
            <th>Total</th>
            <th>Owned</th>
            <th>Final</th>
            <th>Score</th>
            <th>Net</th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement('tbody');
      rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.classList.toggle('row-selected', isSelectedInsightRow({planet_id: row.target_id}));
        tr.dataset.sourceId = '';
        tr.dataset.targetId = row.target_id ?? '';
        tr.addEventListener('click', () => selectInsightPlanet(row.target_id, null, row.source_ids || [row.source_a_id, row.source_b_id]));
        tr.title = `p${row.source_a_id}->p${row.target_id}: ${row.ships_a} ships, p${row.source_b_id}->p${row.target_id}: ${row.ships_b} ships`;
        const netValue = Number(row.tactical_net);
        const scoreValue = Number(row.multi_score);
        tr.innerHTML = `
          <td>p${row.target_id}</td>
          <td>${row.quadrant || ''}</td>
          <td>${row.production}</td>
          <td>${row.target_ships}</td>
          <td>${fmtNumber(row.target_overhead, 1)}</td>
          <td>${row.required_ships ?? row.total_ships}</td>
          <td>${row.incoming_friendly ?? 0}/${row.incoming_enemy ?? 0}</td>
          <td>p${row.source_a_id}</td>
          <td>${row.ships_a}</td>
          <td>${row.source_a_safe}</td>
          <td>${fmtNumber(Number(row.split_weight_a || 0) * 100, 0)}%</td>
          <td>${row.travel_a}</td>
          <td>p${row.source_b_id}</td>
          <td>${row.ships_b}</td>
          <td>${row.source_b_safe}</td>
          <td>${fmtNumber(Number(row.split_weight_b || 0) * 100, 0)}%</td>
          <td>${row.travel_b}</td>
          <td>${row.total_ships}</td>
          <td>${row.owned_production}</td>
          <td>${row.final_owner}:${row.final_ships}</td>
          <td class="${scoreValue >= 0 ? 'good-value' : 'bad-value'}">${fmtNumber(row.multi_score)}</td>
          <td class="${netValue >= 0 ? 'good-value' : 'bad-value'}">${fmtNumber(row.tactical_net)}</td>
        `;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      root.appendChild(wrap);
      return root;
    }
    function buildReinforceReport(report, rowList = null, emptyText = 'No reinforcement needed right now.') {
      const root = document.createElement('div');
      if (!report || report.error) {
        root.className = 'debug-note';
        root.textContent = report?.error ? `reinforce report error: ${report.error}` : 'No reinforce report available.';
        return root;
      }

      const rows = rowList || report.reinforcement_rows || [];
      const priority = buildPriorityStrip({priority_rows: rows});
      if (priority) root.appendChild(priority);
      if (!rows.length) {
        const empty = document.createElement('div');
        empty.className = 'debug-note';
        empty.textContent = emptyText;
        root.appendChild(empty);
        return root;
      }

      const wrap = document.createElement('div');
      wrap.className = 'opening-report-wrap';
      const table = document.createElement('table');
      table.className = 'opening-report';
      table.innerHTML = `
        <thead>
          <tr>
            <th>Target</th>
            <th>Src</th>
            <th>Q</th>
            <th>Prod</th>
            <th>Ships</th>
            <th>Src Safe</th>
            <th>Wait</th>
            <th>Travel</th>
            <th>Arrive</th>
            <th>Danger</th>
            <th>Outcome</th>
            <th>In F/E</th>
            <th>Surv +</th>
            <th>Final</th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement('tbody');
      rows.forEach(row => {
        const tr = document.createElement('tr');
        const rowState = decorateInsightRow(tr, row);
        const finalOwner = row.final_owner === null || row.final_owner === undefined ? '' : row.final_owner;
        const finalShips = row.final_ships === null || row.final_ships === undefined ? '' : row.final_ships;
        tr.innerHTML = `
          <td>p${row.target_id}</td>
          <td>${row.source_id === null || row.source_id === undefined ? '-' : `p${row.source_id}`}</td>
          <td>${row.quadrant || ''}</td>
          <td>${row.production}</td>
          <td>${row.ships_needed}</td>
          <td>${row.source_max_safe_ships ?? ''}</td>
          <td>${rowState.actionNow ? '<span class="now-badge">now</span>' : row.wait_turns}</td>
          <td>${row.travel_turns}</td>
          <td>${row.arrival_turns}</td>
          <td>${row.reinforce_by_turn ?? ''}</td>
          <td>${row.rescue_outcome || (row.rescue_timely === true || row.reinforces_before_loss ? 'saves' : 'recaptures')}</td>
          <td>${row.incoming_friendly ?? 0}/${row.incoming_enemy ?? 0}</td>
          <td>${row.survival_extra_ships ?? 0}</td>
          <td>${finalOwner}:${finalShips}</td>
        `;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      root.appendChild(wrap);
      return root;
    }
    function buildRouteCandidatesTable(report) {
      const rows = report.route_rows || [];
      if (!rows.length) {
        const empty = document.createElement('div');
        empty.className = 'debug-note';
        empty.textContent = 'No route candidates available right now.';
        return empty;
      }

      const root = document.createElement('div');
      const meta = document.createElement('div');
      meta.className = 'opening-report-meta';
      const targetCount = new Set(rows.map(row => row.target_id)).size;
      const blockedCount = rows.filter(row => row.route_ok === false).length;
      const unsafeSourceCount = rows.filter(row => row.source_survival_blocked === true).length;
      const insufficientSourceCount = rows.filter(row => row.source_wait_insufficient === true).length;
      meta.innerHTML = [
        `Route candidates`,
        `Targets ${targetCount}`,
        `Routes ${rows.length}`,
        `Blocked ${blockedCount}`,
        `Unsafe source ${unsafeSourceCount}`,
        `Insufficient ${insufficientSourceCount}`,
        `Best rows are green`,
      ].map(text => `<span>${text}</span>`).join('');
      root.appendChild(meta);

      const wrap = document.createElement('div');
      wrap.className = 'opening-report-wrap';
      const table = document.createElement('table');
      table.className = 'opening-report';
      table.innerHTML = `
        <thead>
          <tr>
            <th>Target</th>
            <th>Src</th>
            <th>Best</th>
            <th>Q</th>
            <th>Role</th>
            <th>Status</th>
            <th>Prod</th>
            <th>Ships</th>
            <th>Src Safe</th>
            <th>Wait</th>
            <th>Travel</th>
            <th>Arrive</th>
            <th>Src End</th>
            <th>Own Prod</th>
            <th>Base</th>
            <th>Dist</th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement('tbody');
      rows.forEach((row, index) => {
        const previous = rows[index - 1];
        if (!previous || previous.target_id !== row.target_id) {
          const sectionRow = document.createElement('tr');
          sectionRow.className = 'section-row';
          sectionRow.innerHTML = `<td colspan="16">Target p${row.target_id}</td>`;
          tbody.appendChild(sectionRow);
        }
        const tr = document.createElement('tr');
        decorateInsightRow(tr, row);
        const baseValue = Number(row.tactical_net);
        const sourceEndOwner = row.source_final_owner === null || row.source_final_owner === undefined ? '' : row.source_final_owner;
        const sourceEndShips = row.source_final_ships === null || row.source_final_ships === undefined ? '' : row.source_final_ships;
        tr.innerHTML = `
          <td>p${row.target_id}</td>
          <td>p${row.source_id}</td>
          <td>${row.is_best ? 'yes' : ''}</td>
          <td>${row.quadrant || ''}</td>
          <td>${row.role || ''}</td>
          <td>${row.source_wait_insufficient === true ? 'insufficient source' : row.source_survival_blocked === true ? 'unsafe source' : row.route_ok === false ? row.route_status || 'blocked' : 'clear'}</td>
          <td>${row.production}</td>
          <td>${row.ships_needed}</td>
          <td>${row.source_max_safe_ships ?? ''}</td>
          <td>${row.wait_turns}</td>
          <td>${row.travel_turns}</td>
          <td>${row.arrival_turns}</td>
          <td>${sourceEndOwner === '' ? '' : `${sourceEndOwner}:${sourceEndShips}`}</td>
          <td>${row.owned_production ?? ''}</td>
          <td class="${baseValue >= 0 ? 'good-value' : 'bad-value'}">${fmtNumber(row.tactical_net)}</td>
          <td>${fmtNumber(row.path_distance, 1)}</td>
        `;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      root.appendChild(wrap);
      return root;
    }
    function buildRoutesReport(report) {
      if (!report || report.error) {
        const root = document.createElement('div');
        root.className = 'debug-note';
        root.textContent = report?.error ? `routes report error: ${report.error}` : 'No routes report available.';
        return root;
      }
      return buildRouteCandidatesTable(report);
    }
    function buildPlanetaryReport(report) {
      const root = document.createElement('div');
      if (!report || report.error) {
        root.className = 'debug-note';
        root.textContent = report?.error ? `planetary report error: ${report.error}` : 'No planetary report available.';
        return root;
      }

      const meta = document.createElement('div');
      meta.className = 'opening-report-meta';
      const counts = report.cohort_counts || {};
      const stats = report.overhead_stats || {};
      meta.innerHTML = [
        `Low ${counts.low ?? 0} (+1)`,
        `Medium ${counts.medium ?? 0} (+2/+3)`,
        `High ${counts.high ?? 0} (+4/+5)`,
        `Outliers ${(report.outlier_rows || []).length}`,
        `Normal ${(report.normal_rows || []).length}`,
        `Low keep ${stats.low_production_keep_cutoff ?? '-'}`,
        `O/H cut ${stats.threshold ?? '-'}`,
      ].map(text => `<span>${text}</span>`).join('');
      root.appendChild(meta);

      const wrap = document.createElement('div');
      wrap.className = 'opening-report-wrap';
      const table = document.createElement('table');
      table.className = 'opening-report';
      table.innerHTML = `
        <thead>
          <tr>
            <th>Planet</th>
            <th>Q</th>
            <th>Prod</th>
            <th>Cohort</th>
            <th>Ships</th>
            <th>O/H</th>
            <th>Cut</th>
            <th>Group</th>
            <th>Reason</th>
            <th>Family</th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement('tbody');
      const allRows = [
        ...(report.outlier_rows || []),
        ...(report.normal_rows || []),
      ];
      allRows.forEach((row, index) => {
        const previous = allRows[index - 1];
        if (!previous || previous.section !== row.section) {
          const sectionRow = document.createElement('tr');
          sectionRow.className = 'section-row';
          if (row.section === 'outlier') sectionRow.classList.add('outlier-section');
          sectionRow.innerHTML = `<td colspan="10">${row.section === 'outlier' ? 'Outliers to avoid/delay' : 'Normal planets'}</td>`;
          tbody.appendChild(sectionRow);
        }
        const tr = document.createElement('tr');
        tr.classList.toggle('row-selected', isSelectedInsightRow(row));
        tr.dataset.sourceId = '';
        tr.dataset.targetId = row.planet_id ?? '';
        tr.addEventListener('click', () => selectInsightPlanet(row.planet_id, null));
        if (row.section === 'outlier') {
          tr.classList.add('outlier-row');
        }
        tr.innerHTML = `
          <td>p${row.planet_id}</td>
          <td>${row.quadrant || ''}</td>
          <td>${row.production}</td>
          <td>${row.cohort || ''}</td>
          <td>${row.ships}</td>
          <td>${fmtNumber(row.overhead, 1)}</td>
          <td>${fmtNumber(row.threshold, 1)}</td>
          <td>${row.section || ''}</td>
          <td>${row.reason || ''}</td>
          <td>${row.family_label || ''}</td>
        `;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      root.appendChild(wrap);
      return root;
    }
    function buildRolesReport(openingReport) {
      const root = document.createElement('div');
      if (!openingReport || openingReport.error) {
        root.className = 'debug-note';
        root.textContent = openingReport?.error ? `roles report error: ${openingReport.error}` : 'No roles report available.';
        return root;
      }
      const report = openingReport.role_report || {rows: openingReport.role_rows || []};
      const rows = report.rows || [];
      const meta = document.createElement('div');
      meta.className = 'opening-report-meta';
      meta.innerHTML = [
        `Home ${report.home_quadrant || openingReport.home_quadrant || '?'}`,
        `Enemy ${report.enemy_quadrant || openingReport.enemy_quadrant || '?'}`,
        `Roles ${rows.length}`,
        `Score -5 O/H -3 distance +2 prod`,
        `C conductor`,
        `SF supplier frontier`,
        `AF attack frontier`,
      ].map(text => `<span>${text}</span>`).join('');
      root.appendChild(meta);

      if (!rows.length) {
        const empty = document.createElement('div');
        empty.className = 'debug-note';
        empty.textContent = 'No role planets found right now.';
        root.appendChild(empty);
        return root;
      }

      const wrap = document.createElement('div');
      wrap.className = 'opening-report-wrap';
      const table = document.createElement('table');
      table.className = 'opening-report';
      table.innerHTML = `
        <thead>
          <tr>
            <th>Planet</th>
            <th>Role</th>
            <th>Q</th>
            <th>Owner</th>
            <th>Prod</th>
            <th>Ships</th>
            <th>O/H</th>
            <th>Score</th>
            <th>Motion</th>
            <th>Home Dist</th>
            <th>Enemy Dist</th>
            <th>Detail</th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement('tbody');
      rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.classList.add('role-row');
        tr.classList.toggle('row-selected', isSelectedInsightRow(row));
        tr.dataset.sourceId = '';
        tr.dataset.targetId = row.planet_id ?? '';
        tr.addEventListener('click', () => selectInsightPlanet(row.planet_id, null));
        tr.title = `${row.role_detail || row.role || 'role'} in ${row.quadrant || '?'}`;
        tr.innerHTML = `
          <td>p${row.planet_id}</td>
          <td>${row.role || ''}</td>
          <td>${row.quadrant || ''}</td>
          <td>${row.owner}</td>
          <td>${row.production}</td>
          <td>${row.ships}</td>
          <td>${fmtNumber(row.overhead, 1)}</td>
          <td>${fmtNumber(row.score, 1)}</td>
          <td>${row.motion || ''}</td>
          <td>${fmtNumber(row.home_distance, 1)}</td>
          <td>${fmtNumber(row.enemy_distance, 1)}</td>
          <td>${row.role_detail || ''}</td>
        `;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      root.appendChild(wrap);
      return root;
    }
    function updateOpeningDebugPanel() {
      const panel = document.getElementById('openingDebugPanel');
      const body = document.getElementById('openingDebugBody');
      panel.style.display = debugVisible ? 'block' : 'none';
      body.innerHTML = '';
      if (!debugVisible) {
        closeOpeningMatrixModal();
        return;
      }
      const debug = openingDebug();
      if (!debug) {
        body.className = 'insights-body debug-note';
        body.textContent = 'Select opening agent to inspect the matrix.';
        updateOpeningMatrixModal();
        return;
      }
      if (debug.error) {
        body.className = 'insights-body debug-note';
        body.textContent = `debug error: ${debug.error}`;
        updateOpeningMatrixModal();
        return;
      }
      body.className = 'insights-body';
      body.appendChild(buildOpeningMatrix(debug));
      updateOpeningMatrixModal();
    }
    function updateHumanInsightsPanel() {
      const panel = document.getElementById('humanInsightsPanel');
      const body = document.getElementById('humanInsightsBody');
      panel.style.display = insightsVisible ? 'block' : 'none';
      if (!insightsVisible) return;
      const openingReport = state?.human_analysis?.opening;
      const saveCount = (openingReport?.save_rows || []).length;
      const recaptureCount = (openingReport?.recapture_rows || []).length;
      const reinforceCount = (openingReport?.reinforcement_rows || []).length;
      const multiCount = (openingReport?.multi_source_rows || []).length;
      const multiTab = panel.querySelector('[data-human-tab="multi"]');
      if (multiTab) {
        multiTab.innerHTML = `Multi${multiCount ? `<span class="tab-badge">${multiCount}</span>` : ''}`;
      }
      const saveTab = panel.querySelector('[data-human-tab="saves"]');
      if (saveTab) {
        saveTab.innerHTML = `Saves${saveCount ? `<span class="tab-badge">${saveCount}</span>` : ''}`;
      }
      const recaptureTab = panel.querySelector('[data-human-tab="recaptures"]');
      if (recaptureTab) {
        recaptureTab.innerHTML = `Recaptures${recaptureCount ? `<span class="tab-badge">${recaptureCount}</span>` : ''}`;
      }
      const reinforceTab = panel.querySelector('[data-human-tab="reinforce"]');
      if (reinforceTab) {
        reinforceTab.innerHTML = `Reinforce${reinforceCount ? `<span class="tab-badge">${reinforceCount}</span>` : ''}`;
      }
      panel.querySelectorAll('[data-human-tab]').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.humanTab === activeHumanInsightsTab);
      });
      body.innerHTML = '';
      body.className = 'insights-body';
      const planetaryReport = state?.human_analysis?.planetary;
      let content = null;
      if (activeHumanInsightsTab === 'reinforce') {
        content = buildReinforceReport(openingReport, openingReport?.reinforcement_rows || [], 'No owned planets need reinforcement right now.');
      } else if (activeHumanInsightsTab === 'multi') {
        content = buildMultiSourceReport(openingReport);
      } else if (activeHumanInsightsTab === 'saves') {
        content = buildReinforceReport(openingReport, openingReport?.save_rows || [], 'No committed captures need save support right now.');
      } else if (activeHumanInsightsTab === 'recaptures') {
        content = buildReinforceReport(openingReport, openingReport?.recapture_rows || [], 'No committed captures need recapture support right now.');
      } else if (activeHumanInsightsTab === 'routes') {
        content = buildRoutesReport(openingReport);
      } else if (isRolesInsightsTab()) {
        content = buildRolesReport(openingReport);
      } else if (isIgnoreInsightsTab()) {
        content = buildPlanetaryReport(planetaryReport);
      } else {
        content = buildOpeningReport(openingReport);
      }
      body.appendChild(content);
    }
    function selectSource(p) {
      selectedSource = p;
      aim = null;
      aimTargetId = null;
      aimLocked = false;
      setBothShipInputs(Math.max(1, Math.floor(p.ships)));
      setBothAngleInputs(0);
      document.getElementById('selectionText').textContent = `Source p${p.id}, ships=${Math.floor(p.ships)}. Move to preview, then click/release to lock aim.`;
      showFleetMenu();
      render();
    }
    function selectAim(pt, target = null, locked = false) {
      if (!selectedSource) return;
      aim = pt;
      aimTargetId = locked && target && target.id !== selectedSource.id ? target.id : null;
      aimLocked = Boolean(locked);
      const angle = Math.atan2(pt.y - selectedSource.y, pt.x - selectedSource.x);
      setBothAngleInputs(angle);
      let ships = Math.floor(selectedSource.ships);
      const currentShips = Math.floor(Number(document.getElementById('menuShipsInput').value || document.getElementById('shipsInput').value || ships));
      ships = Math.min(ships, Math.max(1, currentShips));
      setBothShipInputs(Math.max(1, ships), locked);
      const label = target ? `target p${target.id}` : `aim (${pt.x.toFixed(1)}, ${pt.y.toFixed(1)})`;
      const prefix = aimLocked ? 'Aim locked' : 'Preview';
      document.getElementById('selectionText').textContent = `${prefix}: p${selectedSource.id} -> ${label}. Adjust ships if needed, then Send.`;
      showFleetMenu();
      render();
    }
    canvas.addEventListener('mousedown', evt => {
      if (state?.done) return;
      const pt = boardPoint(evt);
      const p = planetAt(pt);
      if (!selectedSource && p && p.owner === humanPlayer()) {
        evt.preventDefault();
        isDraggingAim = true;
        selectSource(p);
      } else if (selectedSource) {
        evt.preventDefault();
        isDraggingAim = true;
        aimLocked = false;
        if (!(p && p.id === selectedSource.id)) {
          selectAim(pt, p && p.id !== selectedSource.id ? p : null, false);
        }
      }
    });
    canvas.addEventListener('mousemove', evt => {
      if (!selectedSource || state?.done || aimLocked) return;
      const pt = boardPoint(evt);
      const p = planetAt(pt);
      selectAim(pt, p && p.id !== selectedSource.id ? p : null, false);
    });
    canvas.addEventListener('mouseup', evt => {
      if (!selectedSource || state?.done) return;
      const pt = boardPoint(evt);
      const p = planetAt(pt);
      if (!(p && p.id === selectedSource.id)) {
        selectAim(pt, p && p.id !== selectedSource.id ? p : null, true);
      }
      isDraggingAim = false;
    });
    canvas.addEventListener('click', evt => {
      if (state?.done) return;
      const pt = boardPoint(evt);
      const p = planetAt(pt);
      if (!selectedSource && p && p.owner === humanPlayer()) {
        selectSource(p);
      } else if (selectedSource && !(p && p.id === selectedSource.id)) {
        selectAim(pt, p && p.id !== selectedSource.id ? p : null, true);
      }
    });
    canvas.addEventListener('contextmenu', evt => {
      evt.preventDefault();
      closeFleetMenu();
    });
    document.getElementById('addMoveBtn').onclick = () => {
      addCurrentMove();
    };
    document.getElementById('clearMoveBtn').onclick = () => {
      closeFleetMenu();
    };
    document.getElementById('menuSendBtn').onclick = () => addCurrentMove();
    document.getElementById('menuCancelBtn').onclick = () => closeFleetMenu();
    document.getElementById('menuShipsInput').addEventListener('input', evt => {
      setBothShipInputs(evt.target.value);
      render();
    });
    document.getElementById('shipsInput').addEventListener('input', evt => {
      setBothShipInputs(evt.target.value);
      render();
    });
    document.getElementById('menuAngleInput').addEventListener('input', evt => setAngleFromDegrees(evt.target.value));
    document.getElementById('angleInput').addEventListener('input', evt => {
      document.getElementById('menuAngleInput').value = (Number(evt.target.value || 0) * 180 / Math.PI).toFixed(1);
      render();
    });
    document.getElementById('clearQueueBtn').onclick = () => { queuedMoves = []; updateStats(); render(); };
    document.getElementById('debugToggle').addEventListener('change', evt => {
      debugVisible = Boolean(evt.target.checked);
      if (debugVisible) {
        insightsVisible = false;
        document.getElementById('insightsToggle').checked = false;
      }
      localStorage.setItem('orbitWarsDebugVisible', debugVisible ? '1' : '0');
      localStorage.setItem('orbitWarsInsightsVisible', insightsVisible ? '1' : '0');
      updateStats();
      render();
    });
    document.getElementById('insightsToggle').addEventListener('change', evt => {
      insightsVisible = Boolean(evt.target.checked);
      if (insightsVisible) {
        debugVisible = false;
        document.getElementById('debugToggle').checked = false;
      }
      if (!insightsVisible) {
        selectedInsightPlanetId = null;
        selectedInsightSourceId = null;
        selectedInsightSourceIds = [];
      }
      localStorage.setItem('orbitWarsDebugVisible', debugVisible ? '1' : '0');
      localStorage.setItem('orbitWarsInsightsVisible', insightsVisible ? '1' : '0');
      updateStats();
      render();
    });
    document.querySelectorAll('[data-human-tab]').forEach(tab => {
      tab.addEventListener('click', () => {
        activeHumanInsightsTab = tab.dataset.humanTab || 'opening';
        selectedInsightPlanetId = null;
        selectedInsightSourceId = null;
        selectedInsightSourceIds = [];
        updateHumanInsightsPanel();
        render();
      });
    });
    document.getElementById('closeMatrixBtn').onclick = closeOpeningMatrixModal;
    document.getElementById('matrixModal').addEventListener('click', evt => {
      if (evt.target === evt.currentTarget) closeOpeningMatrixModal();
    });
    function currentSeedText(fallback = '20260507') {
      return state?.seed_text || (state?.seed !== null && state?.seed !== undefined ? String(state.seed) : fallback);
    }
    function incrementSeedText(seedText) {
      const trimmed = String(seedText || '0').trim();
      try {
        return String(BigInt(trimmed || '0') + 1n);
      } catch (_err) {
        return String(Number(trimmed || 0) + 1);
      }
    }
    async function submitTurn() {
      try {
        const beforeObs = JSON.parse(JSON.stringify(obs()));
        const beforeHuman = humanPlayer();
        const beforeAgent = agentPlayer();
        const moves = queuedMoves;
        queuedMoves = [];
        state = await api('/api/step', {moves});
        selectedSource = null; aim = null; aimTargetId = null; isDraggingAim = false; aimLocked = false;
        selectedInsightPlanetId = null; selectedInsightSourceId = null;
        selectedInsightSourceIds = [];
        document.getElementById('fleetMenu').style.display = 'none';
        logActionBreadcrumbs(beforeObs, moves, beforeHuman, 'You');
        log(`Submitted ${moves.length} move(s). Now step ${state.obs.step}.`);
        updateStats(); render();
      } catch (err) { log(`ERROR: ${err.message}`); }
    }
    document.getElementById('submitBtn').onclick = submitTurn;
    async function stepBack() {
      try {
        state = await api('/api/back', {});
        queuedMoves = [];
        selectedSource = null; aim = null; aimTargetId = null; isDraggingAim = false; aimLocked = false;
        selectedInsightPlanetId = null; selectedInsightSourceId = null;
        selectedInsightSourceIds = [];
        document.getElementById('fleetMenu').style.display = 'none';
        log(`Back to step ${state.obs.step}.`);
        updateStats(); render();
      } catch (err) { log(`ERROR: ${err.message}`); }
    }
    document.getElementById('backBtn').onclick = stepBack;
    document.getElementById('holdBtn').onclick = async () => {
      const beforeObs = JSON.parse(JSON.stringify(obs()));
      queuedMoves = [];
      state = await api('/api/step', {moves: []});
      selectedSource = null; aim = null; aimTargetId = null; isDraggingAim = false; aimLocked = false;
      selectedInsightPlanetId = null; selectedInsightSourceId = null;
      selectedInsightSourceIds = [];
      document.getElementById('fleetMenu').style.display = 'none';
      log(`Held. Now step ${state.obs.step}.`);
      updateStats(); render();
    };
    async function resetGame() {
      const seedText = document.getElementById('seedInput').value.trim();
      const body = {
        agent: document.getElementById('agentSelect').value,
        human_player: Number(state?.human_player ?? 0),
        seed: seedText,
        episode_steps: Number(state?.episode_steps ?? 500),
      };
      state = await api('/api/reset', body);
      queuedMoves = []; selectedSource = null; aim = null; aimTargetId = null; isDraggingAim = false; aimLocked = false;
      selectedInsightPlanetId = null; selectedInsightSourceId = null;
      selectedInsightSourceIds = [];
      document.getElementById('seedInput').value = currentSeedText(body.seed);
      document.getElementById('fleetMenu').style.display = 'none';
      log(`New game vs ${body.agent}, human player ${body.human_player}, seed ${currentSeedText(body.seed)}.`);
      updateStats(); render();
    }
    document.getElementById('resetBtn').onclick = resetGame;
    document.getElementById('agentSelect').addEventListener('change', async () => {
      await resetGame();
    });
    document.getElementById('replaySeedBtn').onclick = async () => {
      const seedInput = document.getElementById('seedInput');
      seedInput.value = seedInput.value.trim() || currentSeedText(seedInput.value);
      await resetGame();
    };
    document.getElementById('newGameBtn').onclick = async () => {
      const seedInput = document.getElementById('seedInput');
      seedInput.value = incrementSeedText(seedInput.value || currentSeedText());
      await resetGame();
    };
    document.getElementById('saveBtn').onclick = async () => {
      const result = await api('/api/save', {});
      log(`Saved episode: ${result.path}`);
    };
    function startRevisionWatcher() {
      if (revisionWatcherStarted) return;
      revisionWatcherStarted = true;
      setInterval(async () => {
        try {
          const info = await api('/api/revision');
          if (!serverRevision) {
            serverRevision = info.revision || null;
            return;
          }
          if (info.revision && info.revision !== serverRevision) {
            window.location.reload();
          }
        } catch (_err) {
          // The server may be between exec/restart; the next successful poll will reload.
        }
      }, 1000);
    }
    async function boot() {
      state = await api('/api/state');
      serverRevision = state.code_revision || null;
      debugVisible = localStorage.getItem('orbitWarsDebugVisible') === '1';
      insightsVisible = localStorage.getItem('orbitWarsInsightsVisible') === '1';
      if (debugVisible && insightsVisible) insightsVisible = false;
      document.getElementById('debugToggle').checked = debugVisible;
      document.getElementById('insightsToggle').checked = insightsVisible;
      document.getElementById('agentSelect').value = state.agent;
      document.getElementById('seedInput').value = currentSeedText();
      updateStats();
      render();
      startRevisionWatcher();
      log('Ready. Click or drag from your planet, Send, then Step.');
    }
    window.addEventListener('keydown', evt => {
      if (evt.key === 'Escape') {
        if (matrixModalOpen) {
          closeOpeningMatrixModal();
          return;
        }
        closeFleetMenu();
      } else if (evt.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName || '')) {
        evt.preventDefault();
        submitTurn();
      }
    });
    window.addEventListener('resize', () => showFleetMenu());
    boot().catch(err => log(`BOOT ERROR: ${err.message}`));
  </script>
</body>
</html>
"""


class PlayHandler(BaseHTTPRequestHandler):
    game: HumanOrbitGame

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            self.send_json(self.game.payload())
            return
        if parsed.path == "/api/revision":
            self.send_json({"revision": code_revision()})
            return
        self.send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self.read_json()
            if parsed.path == "/api/reset":
                self.send_json(
                    self.game.reset(
                        agent_key=body.get("agent"),
                        human_player=body.get("human_player"),
                        seed=body.get("seed"),
                        episode_steps=body.get("episode_steps"),
                    )
                )
                return
            if parsed.path == "/api/step":
                self.send_json(self.game.step(body.get("moves", [])))
                return
            if parsed.path == "/api/back":
                self.send_json(self.game.back())
                return
            if parsed.path == "/api/save":
                self.send_json({"path": self.game.save_episode()})
                return
            self.send_json({"error": "not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class ReloadableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Orbit Wars manually against a local agent in a browser UI.")
    parser.add_argument(
        "--agent",
        default="v0",
        help="v0 or path to an agent .py file",
    )
    parser.add_argument("--seat", choices=["first", "second"], default="first")
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--no-reload", action="store_true", help="Disable automatic server restart on code changes.")
    parser.add_argument("--smoke-test", action="store_true", help="Initialize once and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_player = 0 if args.seat == "first" else 1
    game = HumanOrbitGame(
        agent_key=args.agent,
        human_player=human_player,
        seed=args.seed,
        episode_steps=args.episode_steps,
        output_dir=Path(args.output_dir),
    )
    if args.smoke_test:
        payload = game.payload()
        print(
            "ok "
            f"agent={payload['agent']} human_player={payload['human_player']} "
            f"step={payload['obs'].get('step')} seed={payload['seed']}"
        )
        return

    PlayHandler.game = game
    server = ReloadableThreadingHTTPServer((args.host, args.port), PlayHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Play UI: {url}")
    print("Use Ctrl+C to stop the server.")
    if not args.no_reload:
        start_code_reloader()
    if not args.no_open and os.environ.get(RELOADED_ENV) != "1":
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
