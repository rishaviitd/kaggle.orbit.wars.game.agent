from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

TOTAL_STEPS = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Config — v15: v14 + active end-game (full-commit + wave ramp + regroup fade)
#
# v14 was too passive in late game — same 6 waves, same regroup, never
# full-commits. High-ELO opponents exploit this by being more aggressive
# when it matters.
#
# v15 adds three time-aware mechanisms (all continuous, zero hard cuts):
#   1. Full-commit tier (size_mult=1.0): when ROI is high enough, send
#      everything. v14 only tried 50%/75% drain, missing sure wins.
#   2. Wave ramp: waves gradually increase as game progresses.
#      v14 was stuck at 6 (2P) / 7 (4P) all game — in late game you need
#      more simultaneous pressure. Quadratic ramp keeps early game stable.
#   3. Regroup fade: regroup time decreases as game runs out, and
#      regroup disables entirely when there's not enough time left.
#      Late-game regroup is wasteful — ships sitting idle while enemies
#      grab the last planets.
#
# All three use the same time progress = step / TOTAL_STEPS, composed
# with the existing ROI time decay and dynamic ROI. No phase boundaries.
#
# v14 features preserved unchanged:
#   - β auto-adapts aggression via reinforcement risk
#   - Dynamic ROI (lead/trail economic position)
#   - ROI time decay (replaces terminal phase)
#   - Floor-sized fleets + multi-tier drain
#   - Comet handling + FFA anti-snowball
#   - Quadratic enemy-pressure decay
#   - Permanent neutral/prod bonuses
#   - Warm-up pass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProducerLiteConfig:
    """v15 — zero-phase config with time-aware dynamic ROI + active end-game.

    No phase boundaries. Aggression controlled by five automatic mechanisms:
      1. Reinforcement risk (beta): inflates capture floor near enemies
      2. Dynamic ROI: adjusts threshold based on relative economic position
      3. Time decay: ROI gradually decreases as game progresses
      4. Wave ramp: waves gradually increase as game progresses
      5. Regroup fade: regroup time shrinks and eventually disables
    """

    # ── planning window ───────────────────────────────────────────────────
    horizon: int = 18

    # ── shortlists ────────────────────────────────────────────────────────
    max_sources_per_lane: int = 10
    max_offensive_targets: int = 14
    max_defensive_targets: int = 5

    # ── scoring / greedy ──────────────────────────────────────────────────
    max_waves_per_turn: int = 6           # base waves (early game)
    roi_threshold: float = 1.40           # β handles quality, this sets base bar
    min_ships_to_launch: float = 4.0      # only big fleets

    # ── ETA-aware reinforcement risk ──────────────────────────────────────
    reinforce_size_beta: float = 2.5
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0

    # ── regroup ───────────────────────────────────────────────────────────
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 9
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 5e-4

    # ── FFA bonuses ───────────────────────────────────────────────────────
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0

    # ── floor-sized fleets ────────────────────────────────────────────────
    enable_floor_sized_fleets: bool = True
    floor_pad_ships: float = 3.0
    floor_pad_frac: float = 0.12

    # ── comet handling ────────────────────────────────────────────────────
    comet_min_hold: float = 4.0
    comet_evac_steps: int = 6

    # ── multi-tier drain fractions (v15: added 1.0 full-commit) ───────────
    # 0.5 = half drain (scout/probe), 0.75 = main force, 1.0 = full commit
    # Full-commit option allows the agent to send everything when ROI is
    # clearly positive — v14 missed these high-confidence attacks.
    size_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0)

    # ── permanent scoring bonuses ─────────────────────────────────────────
    neutral_bonus: float = 0.15
    prod_bonus: float = 0.05

    # ── dynamic ROI ───────────────────────────────────────────────────────
    enable_dynamic_roi: bool = True
    roi_lead_modifier: float = 1.15
    roi_trail_modifier: float = 0.92
    roi_lead_threshold: float = 0.55
    roi_trail_threshold: float = 0.40

    # ── ROI time decay ────────────────────────────────────────────────────
    roi_time_aggression: float = 0.25

    # ── wave ramp (v15 new) ───────────────────────────────────────────────
    # Waves gradually increase as game progresses, giving more simultaneous
    # pressure in late game without a hard terminal switch.
    # Formula: extra_waves = round(waves_time_bonus × progress²)
    # Quadratic so early game stays near base, late game ramps fast.
    #
    # 2P with bonus=3: Step 0→6 waves, Step 300→7, Step 400→8, Step 480→9
    # 4P with bonus=3: Step 0→7 waves, Step 200→8, Step 300→9, Step 400→9-10
    waves_time_bonus: int = 3

    # ── regroup fade (v15 new) ────────────────────────────────────────────
    # As game progresses, reduce regroup wait time. Late-game regroup is
    # wasteful — ships sitting idle while enemies grab planets.
    # regroup_time_decay: effective_regroup_time = max_regroup_time × (1 - progress × decay)
    #   0.0 = no decay (v14 behavior), 0.5 = half time at game end
    # regroup_disable_remaining: disable regroup when remaining steps < this
    #   (ships can't arrive in time if regroup waits)
    regroup_time_decay: float = 0.40         # at step 500: regroup_time × 0.60
    regroup_disable_remaining: int = 30       # disable regroup after step 470


# ═══════════════════════════════════════════════════════════════════════════════
# 4P FFA preset
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=12,
    max_sources_per_lane=7,
    max_offensive_targets=9,
    max_defensive_targets=3,
    roi_threshold=1.30,
    min_ships_to_launch=3.0,
    max_waves_per_turn=7,
    # ── regroup: faster cycle ─────────────────────────────────────────────
    max_regroup_time=5.0,
    regroup_pressure_delta_min=0.20,
    max_regroup_targets_per_source=8,
    # ── reinforcement risk: lower for 4P ──────────────────────────────────
    reinforce_size_beta=1.3,
    # ── FFA: anti-snowball ────────────────────────────────────────────────
    ffa_leader_attack_bonus=0.20,
    ffa_target_prod_bonus=0.25,
    # ── permanent scoring bonuses ─────────────────────────────────────────
    neutral_bonus=0.20,
    prod_bonus=0.08,
    # ── dynamic ROI ───────────────────────────────────────────────────────
    roi_lead_modifier=1.15,
    roi_trail_modifier=0.85,
    # ── time decay: faster ramp for shorter 4P games ──────────────────────
    roi_time_aggression=0.30,
    # ── wave ramp: same bonus, starts from higher base (7) ────────────────
    waves_time_bonus=3,                     # 7→~10 by game end
    # ── regroup fade: faster decay + earlier disable (4P ends sooner) ─────
    regroup_time_decay=0.50,                # at game end: regroup_time × 0.50
    regroup_disable_remaining=25,            # disable after step 475
)


# ═══════════════════════════════════════════════════════════════════════════════
# Movement config
# ═══════════════════════════════════════════════════════════════════════════════

def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Enemy pressure (quadratic decay — from Tavacation)
# ═══════════════════════════════════════════════════════════════════════════════

def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Quadratic-decay enemy-mass proxy per planet — [P]."""
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    linear = (1.0 - d0 / reach_dist).clamp(min=0.0)
    decay = linear * linear  # quadratic — sharper frontline gradient
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Strength & time-aware dynamic ROI
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_strength(obs, prod, player_count, dtype, device):
    """Per-player strength = production + 0.025 * ships."""
    owner = obs.owner_abs.to(torch.long)
    owner_valid = (owner >= 0) & (owner < int(player_count)) & obs.alive
    owner_idx = owner.clamp(min=0, max=max(int(player_count) - 1, 0))
    prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
    ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
    prod_by_owner.scatter_add_(
        0, owner_idx,
        torch.where(owner_valid, prod.to(dtype), torch.zeros_like(prod.to(dtype))),
    )
    ships_by_owner.scatter_add_(
        0, owner_idx,
        torch.where(owner_valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype))),
    )
    return prod_by_owner + 0.025 * ships_by_owner


def _dynamic_roi(base_roi, obs, prod, player_count, pid, config, dtype, device, step=0):
    """Adjust ROI smoothly based on game time and relative strength.

    Leaders retain the configured caution modifier. When behind, lower the
    threshold quadratically with the deficit so small gaps produce only small
    changes while larger deficits permit proportionally more aggression.
    """
    if not config.enable_dynamic_roi or int(player_count) < 2:
        step_frac = min(1.0, float(step) / TOTAL_STEPS)
        time_factor = 1.0 - step_frac * float(config.roi_time_aggression)
        return base_roi * time_factor

    step_frac = min(1.0, float(step) / TOTAL_STEPS)
    time_factor = 1.0 - step_frac * float(config.roi_time_aggression)

    strength = _compute_strength(obs, prod, int(player_count), dtype, device)
    max_strength = strength.max()
    my_strength = strength[pid]
    ratio = float(my_strength) / max(float(max_strength), 1e-6)
    timed_roi = float(base_roi) * time_factor
    if ratio >= 1.0:
        return timed_roi * float(config.roi_lead_modifier)

    deficit = 1.0 - ratio
    roi_drop = 0.25 * deficit * deficit
    return max(1.10, timed_roi - roi_drop)


# ═══════════════════════════════════════════════════════════════════════════════
# Time-aware wave count & regroup config (v15 new)
# ═══════════════════════════════════════════════════════════════════════════════

def _effective_waves(config: ProducerLiteConfig, step: int) -> int:
    """Compute effective wave count with quadratic time ramp.

    extra_waves = round(bonus × progress²)
    Quadratic keeps early game near base, ramps hard in late game.

    2P (base=6, bonus=3):
      Step   0: 6 + round(3 × 0.00²) = 6
      Step 200: 6 + round(3 × 0.16²) = 6
      Step 300: 6 + round(3 × 0.36²) = 6 + 0 = 6  (0.39 → 0)
      Step 350: 6 + round(3 × 0.49²) = 6 + 1 = 7  (0.72 → 1)
      Step 400: 6 + round(3 × 0.64²) = 6 + 1 = 7  (1.23 → 1)
      Step 440: 6 + round(3 × 0.77²) = 6 + 2 = 8  (1.78 → 2)
      Step 470: 6 + round(3 × 0.88²) = 6 + 2 = 8  (2.32 → 2)
      Step 490: 6 + round(3 × 0.96²) = 6 + 3 = 9  (2.77 → 3)

    4P (base=7, bonus=3):
      Step   0: 7 + 0 = 7
      Step 200: 7 + round(3 × 0.16²) = 7
      Step 300: 7 + round(3 × 0.36²) = 7 + 0 = 7  (0.39)
      Step 350: 7 + round(3 × 0.49²) = 7 + 1 = 8  (0.72)
      Step 400: 7 + round(3 × 0.64²) = 7 + 1 = 8  (1.23)
      Step 440: 7 + round(3 × 0.77²) = 7 + 2 = 9  (1.78)
      Step 470: 7 + round(3 × 0.88²) = 7 + 2 = 9  (2.32)
      Step 490: 7 + round(3 × 0.96²) = 7 + 3 = 10 (2.77)
    """
    progress = min(1.0, float(step) / TOTAL_STEPS)
    extra = round(float(config.waves_time_bonus) * progress * progress)
    return int(config.max_waves_per_turn) + extra


def _apply_regroup_fade(config: ProducerLiteConfig, step: int) -> ProducerLiteConfig:
    """Apply time-aware regroup fade — shrink regroup time, disable if too late.

    Two mechanisms:
      1. Decay: effective_regroup_time = max_regroup_time × (1 - progress × decay)
         As game progresses, regroup waits less — ships shouldn't sit idle.
      2. Disable: when remaining steps < regroup_disable_remaining, turn off
         regroup entirely — ships can't arrive in time if we wait for regroup.

    Both are continuous/soft — no hard phase boundary.
    """
    remaining = TOTAL_STEPS - step
    if remaining < int(config.regroup_disable_remaining):
        return dataclasses.replace(config, enable_regroup=False)

    progress = min(1.0, float(step) / TOTAL_STEPS)
    decay_factor = 1.0 - progress * float(config.regroup_time_decay)
    new_regroup_time = max(1.0, float(config.max_regroup_time) * decay_factor)

    return dataclasses.replace(config, max_regroup_time=new_regroup_time)


# ═══════════════════════════════════════════════════════════════════════════════
# Core wave planner
# ═══════════════════════════════════════════════════════════════════════════════

def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    step: int = 0,
    comet_remaining: Tensor | None = None,
):
    """Multi-option attack planner with reinforcement risk + regroup + permanent bonuses."""
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))

    # ── time-aware wave count (v15) ───────────────────────────────────────
    W = max(1, _effective_waves(config, step))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)

    # ── enemy_mass: computed once, reused for reinforcement risk AND regroup ──
    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
        if beta > 0.0 or bool(config.enable_regroup) else None
    )

    # ── reinforcement risk → inflate capture floor ─────────────────────────
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange, eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )
    K = int(floor.shape[-1])

    # ── aiming helpers ────────────────────────────────────────────────────

    def _aim_for(sizes_st: Tensor):
        active = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes_st.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1)
        aim = intercept_angle(
            movement,
            source_idx.unsqueeze(1),
            target_idx.unsqueeze(0),
            sizes_st,
            active=active,
        )
        eta = aim["eta"]
        viable = aim["viable"] & (eta <= eta_cap.view(1, T))
        return aim["angle"], eta, viable

    def _floor_at(eta: Tensor) -> Tensor:
        if K > 0:
            k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            return floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        return torch.ones(S, T, dtype=dtype, device=device)

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    base_ok = src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
    drain_int = drain.view(S, 1).expand(S, T).floor()

    # ── build candidate options ────────────────────────────────────────────

    options: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    # Option A: full safe-drain wave — always present
    angle_a, eta_a, viable_a = _aim_for(drain_int)
    floor_a = _floor_at(eta_a)
    valid_a = viable_a & (drain_int >= floor_a) & (drain_int >= 1.0) & base_ok
    options.append((drain_int, angle_a, eta_a, valid_a))

    # Option B: floor-matched fleet — precise capture, saves ships
    if bool(config.enable_floor_sized_fleets):
        pad = float(config.floor_pad_ships)
        frac = 1.0 + float(config.floor_pad_frac)
        size_b = torch.minimum(drain_int, (floor_a * frac + pad).ceil())
        _, eta_b0, _ = _aim_for(size_b)
        floor_b0 = _floor_at(eta_b0)
        size_b = torch.minimum(
            drain_int, torch.maximum(size_b, (floor_b0 * frac + pad).ceil()),
        ).floor()
        angle_b, eta_b, viable_b = _aim_for(size_b)
        floor_b = _floor_at(eta_b)
        valid_b = (
            viable_b & (size_b >= floor_b) & (size_b >= 1.0) & base_ok
            & (size_b < drain_int)
            & ~target_is_mine.view(1, T)
        )
        options.append((size_b, angle_b, eta_b, valid_b))

    # Option C/D/…: fractional drain tiers (0.5, 0.75, 1.0)
    # v15: 1.0 is now included — full-commit when ROI is clearly positive
    for mult in config.size_multipliers:
        sizes_m = (drain_int * float(mult)).floor().clamp(min=1.0)
        angle_m, eta_m, viable_m = _aim_for(sizes_m)
        floor_m = _floor_at(eta_m)
        valid_m = viable_m & (sizes_m >= floor_m) & (sizes_m >= 1.0) & base_ok
        options.append((sizes_m, angle_m, eta_m, valid_m))

    # ── concatenate all options ────────────────────────────────────────────

    L = 1
    short_range = torch.arange(T, device=device)
    p_src, p_send, p_ang, p_eta, p_val, p_short = [], [], [], [], [], []
    for sizes_o, angle_o, eta_o, valid_o in options:
        p_src.append(source_idx.view(S, 1).expand(S, T).reshape(-1, L))
        p_send.append(torch.where(valid_o, sizes_o, torch.zeros_like(sizes_o)).reshape(-1, L))
        p_ang.append(angle_o.reshape(-1, L))
        p_eta.append(torch.where(valid_o, eta_o, torch.ones_like(eta_o)).reshape(-1, L))
        p_val.append(valid_o.reshape(-1))
        p_short.append(short_range.view(1, T).expand(S, T).reshape(-1))
    cand_src = torch.cat(p_src, dim=0)
    cand_send = torch.cat(p_send, dim=0)
    cand_angle = torch.cat(p_ang, dim=0)
    cand_eta = torch.cat(p_eta, dim=0)
    cand_valid = torch.cat(p_val, dim=0)
    cand_tgt_short = torch.cat(p_short, dim=0)
    cand_tgt_slot = target_idx[cand_tgt_short]
    C = int(cand_valid.shape[0])
    cand_is_def = target_is_mine[cand_tgt_short]
    cand_active = cand_valid.view(C, L)

    # ── comet attack filter ────────────────────────────────────────────────

    if comet_remaining is not None and bool(torch.isfinite(comet_remaining).any()):
        rem_c = comet_remaining[target_idx.clamp(0, P - 1)][cand_tgt_short]
        is_comet = torch.isfinite(rem_c)
        too_late = (cand_eta.reshape(-1) + float(config.comet_min_hold)) > rem_c
        cand_valid = cand_valid & ~(is_comet & too_late)
        cand_active = cand_valid.view(C, L)

    # ── score candidates ───────────────────────────────────────────────────

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )

    # ════════════════════════════════════════════════════════════════════════
    # Permanent scoring bonuses
    # ════════════════════════════════════════════════════════════════════════
    target_is_neutral = obs.is_neutral[target_idx.clamp(0, P - 1)]
    target_prod = prod[target_idx.clamp(0, P - 1)].to(dtype)

    # ── Neutral target bonus ──────────────────────────────────────────────
    if float(config.neutral_bonus) > 0.0:
        score = score + torch.where(
            target_is_neutral[cand_tgt_short] & cand_valid & ~cand_is_def,
            torch.full_like(score, float(config.neutral_bonus)),
            torch.zeros_like(score),
        )

    # ── Production bonus ──────────────────────────────────────────────────
    if float(config.prod_bonus) > 0.0:
        score = score + torch.where(
            cand_valid & ~cand_is_def,
            float(config.prod_bonus) * target_prod[cand_tgt_short],
            torch.zeros_like(score),
        )

    # ── FFA bonuses ───────────────────────────────────────────────────────
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0
        or float(config.ffa_target_prod_bonus) > 0.0
    ):
        strength = _compute_strength(obs, prod, int(player_count), dtype, device)
        my_strength = strength[pid].detach()
        target_owner = obs.owner_abs.to(torch.long)[target_idx.clamp(0, P - 1)].clamp(
            min=0, max=max(int(player_count) - 1, 0),
        )
        target_owned_enemy = (
            target_exists
            & obs.is_enemy[target_idx.clamp(0, P - 1)]
            & (obs.owner_abs[target_idx.clamp(0, P - 1)] >= 0)
        )
        owner_strength = strength[target_owner]
        leader_delta = (owner_strength - my_strength).clamp(min=0.0)
        target_bonus_short = torch.where(
            target_owned_enemy,
            float(config.ffa_leader_attack_bonus) * leader_delta
            + float(config.ffa_target_prod_bonus) * target_prod,
            torch.zeros_like(owner_strength),
        )
        score = score + target_bonus_short[cand_tgt_short]

    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # ── greedy select ──────────────────────────────────────────────────────

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


# ═══════════════════════════════════════════════════════════════════════════════
# run_turn — single-turn orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def run_turn(
    obs_tensors: dict,
    *,
    config: ProducerLiteConfig,
    player_count: int,
    memory,
    comet_info: dict | None = None,
) -> dict:
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    # Parse comet remaining steps
    comet_remaining = None
    if comet_info:
        try:
            ids = obs_tensors["planets"][..., 0].reshape(-1).long()
            if int(ids.numel()) == P:
                rem = torch.full((P,), float("inf"), dtype=obs.ships.dtype, device=device)
                for cid, r in comet_info.items():
                    hit = (ids == int(cid)).nonzero(as_tuple=True)[0]
                    if int(hit.numel()) > 0:
                        rem[int(hit[0])] = float(r)
                comet_remaining = rem
        except Exception:
            comet_remaining = None

    # ── time-aware dynamic ROI ────────────────────────────────────────────
    step = int(obs_tensors["step"].reshape(-1)[0].item())
    prod = movement.planet_prod
    effective_roi = _dynamic_roi(
        float(config.roi_threshold), obs, prod, player_count,
        int(obs.player_id), config, obs.ships.dtype, device, step=step,
    )
    config = dataclasses.replace(config, roi_threshold=float(effective_roi))

    # ── time-aware regroup fade (v15) ─────────────────────────────────────
    config = _apply_regroup_fade(config, step)

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
        step=step,
        comet_remaining=comet_remaining,
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# Config selector — zero-phase
# ═══════════════════════════════════════════════════════════════════════════════

def _config_for(player_count: int) -> ProducerLiteConfig:
    """Select config by player count. No phase logic — same config all game."""
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime + Memory
# ═══════════════════════════════════════════════════════════════════════════════

class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict, comet_info: dict | None = None):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
            comet_info=comet_info,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ═══════════════════════════════════════════════════════════════════════════════
# Comet utilities
# ═══════════════════════════════════════════════════════════════════════════════

_SUN_X, _SUN_Y, _SUN_R = 50.0, 50.0, 10.0


def _oget(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _parse_comet_remaining(obs) -> dict:
    out: dict = {}
    try:
        groups = _oget(obs, "comets", None) or []
        for g in groups:
            if isinstance(g, dict):
                pids = g.get("planet_ids") or []
                paths = g.get("paths") or []
                idx = int(g.get("path_index", 0) or 0)
            else:
                pids = getattr(g, "planet_ids", None) or []
                paths = getattr(g, "paths", None) or []
                idx = int(getattr(g, "path_index", 0) or 0)
            for i, cid in enumerate(pids):
                path = paths[i] if i < len(paths) else (paths[0] if len(paths) else None)
                if path is None:
                    continue
                out[int(cid)] = max(0, int(len(path)) - 1 - idx)
    except Exception:
        return {}
    return out


def _segment_clears_sun(x0, y0, x1, y1, margin: float = 1.5) -> bool:
    dx, dy = x1 - x0, y1 - y0
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return math.hypot(x0 - _SUN_X, y0 - _SUN_Y) > _SUN_R + margin
    t = max(0.0, min(1.0, ((_SUN_X - x0) * dx + (_SUN_Y - y0) * dy) / l2))
    cx, cy = x0 + t * dx, y0 + t * dy
    return math.hypot(cx - _SUN_X, cy - _SUN_Y) > _SUN_R + margin


def _is_static_planet(p) -> bool:
    return math.hypot(float(p[2]) - 50.0, float(p[3]) - 50.0) + float(p[4]) >= 49.999


def _comet_evac_moves(obs, player_id: int, moves, remaining: dict, evac_steps: int):
    """Evacuate ships from comets about to leave the board."""
    try:
        base = [list(m) for m in (moves or [])]
        if not remaining:
            return base
        planets = _oget(obs, "planets", None) or []
        comet_ids = set(int(c) for c in (_oget(obs, "comet_planet_ids", None) or []))
        by_id = {int(p[0]): p for p in planets}
        committed: dict = {}
        for m in base:
            committed[int(m[0])] = committed.get(int(m[0]), 0) + int(m[2])

        own_static = [p for p in planets
                      if int(p[1]) == int(player_id) and int(p[0]) not in comet_ids
                      and _is_static_planet(p)]
        own_orbit = [p for p in planets
                     if int(p[1]) == int(player_id) and int(p[0]) not in comet_ids
                     and not _is_static_planet(p)]
        others = [p for p in planets
                  if int(p[0]) not in comet_ids and int(p[1]) != int(player_id)]

        for cid, rem in remaining.items():
            if int(rem) > int(evac_steps):
                continue
            p = by_id.get(int(cid))
            if p is None or int(p[1]) != int(player_id):
                continue
            avail = int(p[5]) - committed.get(int(cid), 0)
            if avail < 1:
                continue
            px, py = float(p[2]), float(p[3])
            best = None
            pools = (
                sorted(own_static, key=lambda q: (q[2] - px) ** 2 + (q[3] - py) ** 2),
                sorted(own_orbit, key=lambda q: (q[2] - px) ** 2 + (q[3] - py) ** 2),
                sorted(others, key=lambda q: (float(q[5]) >= avail,
                                              (q[2] - px) ** 2 + (q[3] - py) ** 2)),
            )
            for pool in pools:
                for q in pool:
                    if int(q[0]) == int(cid):
                        continue
                    if _segment_clears_sun(px, py, float(q[2]), float(q[3])):
                        best = q
                        break
                if best is not None:
                    break
            if best is None:
                continue
            ang = math.atan2(float(best[3]) - py, float(best[2]) - px)
            base.append([int(cid), float(ang), int(avail)])
            committed[int(cid)] = committed.get(int(cid), 0) + int(avail)
        return base
    except Exception:
        return moves


# ═══════════════════════════════════════════════════════════════════════════════
# Warm-up pass
# ═══════════════════════════════════════════════════════════════════════════════

def _warmup():
    """Run a dummy turn at import time to trigger JIT compilation."""
    try:
        dummy_obs = {
            "planets": [
                [0, 0, 25.0, 50.0, 3.0, 50.0, 5.0],
                [1, -1, 75.0, 50.0, 3.0, 30.0, 3.0],
                [2, 1, 50.0, 25.0, 3.0, 40.0, 4.0],
            ],
            "initial_planets": [
                [0, 0, 25.0, 50.0, 3.0, 50.0, 5.0],
                [1, -1, 75.0, 50.0, 3.0, 30.0, 3.0],
                [2, 1, 50.0, 25.0, 3.0, 40.0, 4.0],
            ],
            "fleets": [],
            "step": 0,
            "player": 0,
            "angular_velocity": 0.03,
            "episode_steps": 500,
            "remainingOverageTime": 2.0,
            "next_fleet_id": 0,
            "comets": [],
            "comet_planet_ids": [],
        }
        with torch.no_grad():
            _ = agent(dummy_obs)
        _RUNTIME.reset()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = _oget(obs, "player", 0)
    player_id = int(player if player is not None else 0)
    comet_info = _parse_comet_remaining(obs)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors, comet_info=comet_info)
    moves = sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    cfg = _config_for(_RUNTIME.memory.cached_player_count or 2)
    return _comet_evac_moves(obs, player_id, moves, comet_info, cfg.comet_evac_steps)

# Trigger warm-up at import time
_warmup()
