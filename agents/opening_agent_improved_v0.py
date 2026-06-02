"""Improved cheap-production opening agent.

Each owned source uses the baseline target rule, then all source-target rows
are resolved together as one assignment matrix:

1. Consider every non-owned, non-comet planet not already claimed by friendly fleets.
2. Score each source-target row by net fleet contribution over a rolling horizon.
3. Pick the best one-source/one-target matrix assignment.
4. Allow negative rows so the agent can choose the least-bad move instead of freezing.
5. Launch only assigned rows that are ready now; delayed assigned rows wait.
"""



# strategy summary 

# Opening Expansion Core: It scores every possible source-to-target move using capture cost, travel time, target production, and expected production gained inside a rolling horizon.

# Rolling Horizon: It scores with `current_step + 30`, so the value window moves forward with the game instead of ending at a fixed early step.

# All Target Scan: It considers all non-owned valid planets, not only the old top 37% cheap-production candidates.

# Opponent Quadrant Delayed: It removes opponent-quadrant planets from ranking through step 44, then reintroduces them from step 45 onward.

# Matrix Assignment: It builds a full source-target matrix and selects a non-conflicting assignment, so one source gets at most one target and one target gets at most one source.

# Negative Scores Allowed: It does not drop negative-net rows automatically, because sometimes all remaining options are negative but the least bad move is still better than doing nothing.

# Real Future Source Filter: If an already in-flight friendly fleet is expected to capture a planet, that planet is excluded as a target to avoid duplicate attacks.

# Incoming Fleet Awareness: For each target, it accounts for visible incoming fleets before arrival and after capture.

# Enemy Planet Growth: For enemy-owned targets, it includes enemy production during our travel time, so capture cost grows with distance/time.

# Survival-Aware Capture: It increases required ships if we need extra ships to survive known enemy fleets after we capture the target.

# Source Safety Check: It avoids sending so many ships from a source that the source would fall to known incoming enemy fleets.

# Moving Planet Intercept: It computes travel/angle against moving planets, including orbit motion and path blockers, instead of assuming static targets.

# Sun/Planet Blocker Check: It rejects shots whose path would collide with the sun or another planet before the intended target.

# Performance Caches: It caches travel, target projection, and source safety calculations inside each decision call to keep runtime under the Kaggle limit.

# Debug Support: It exports comparison rows, selected moves, future-source labels, and matrix values for the local UI/save traces.


from __future__ import annotations

import math
import time
from collections import namedtuple
from typing import Any

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
MAX_COLLISION_TURN = 120
FUTURE_SOURCE_LOOKAHEAD = 90
OPENING_SCORE_TURNS = 30
OPPONENT_QUADRANT_ATTACK_STEP = 45
AGENT_TIME_BUDGET_SECONDS = 0.85
LOW_OVERAGE_TIME_BUDGET_SECONDS = 0.30


def get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000)
    return min(1.0 + (max_speed - 1.0) * (ratio**1.5), max_speed)


def planet_motion_types(planets: list[Planet], initial_by_id: dict[int, Any], comet_ids: set[int]) -> dict[int, str]:
    motion_types: dict[int, str] = {}
    for planet in planets:
        if planet.id in comet_ids:
            motion_types[planet.id] = "comet"
            continue
        initial = initial_by_id.get(planet.id)
        if initial is None:
            motion_types[planet.id] = "unknown"
            continue
        orbital_radius = math.hypot(initial[2] - CENTER, initial[3] - CENTER)
        motion_types[planet.id] = "orbiting" if orbital_radius + initial[4] < ROTATION_RADIUS_LIMIT - 1e-9 else "static"
    return motion_types


def planet_position_after_moves(
    planet: Planet,
    moves_done: float,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
) -> tuple[float, float]:
    if moves_done <= 0 or planet.id in comet_ids:
        return planet.x, planet.y
    initial = initial_by_id.get(planet.id)
    if initial is None:
        return planet.x, planet.y
    orbital_radius = math.hypot(initial[2] - CENTER, initial[3] - CENTER)
    if orbital_radius + initial[4] >= ROTATION_RADIUS_LIMIT:
        return planet.x, planet.y
    initial_angle = math.atan2(initial[3] - CENTER, initial[2] - CENTER)
    env_step = max(1, int(current_step)) + moves_done - 1
    future_angle = initial_angle + angular_velocity * env_step
    return CENTER + orbital_radius * math.cos(future_angle), CENTER + orbital_radius * math.sin(future_angle)


def circle_collision_time(
    start_x: float,
    start_y: float,
    dir_x: float,
    dir_y: float,
    speed: float,
    center_x: float,
    center_y: float,
    radius: float,
    max_time: float,
) -> float | None:
    rel_x = start_x - center_x
    rel_y = start_y - center_y
    velocity_x = dir_x * speed
    velocity_y = dir_y * speed
    a = velocity_x * velocity_x + velocity_y * velocity_y
    b = 2.0 * (rel_x * velocity_x + rel_y * velocity_y)
    c = rel_x * rel_x + rel_y * rel_y - radius * radius
    if c <= 0:
        return 0.0
    if a == 0:
        return None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return None
    sqrt_discriminant = math.sqrt(discriminant)
    first = (-b - sqrt_discriminant) / (2.0 * a)
    second = (-b + sqrt_discriminant) / (2.0 * a)
    for collision_time in (first, second):
        if 0.0 <= collision_time <= max_time:
            return collision_time
    return None


def board_exit_time(start_x: float, start_y: float, dir_x: float, dir_y: float, speed: float) -> float | None:
    exit_times = []
    velocity_x = dir_x * speed
    velocity_y = dir_y * speed
    if velocity_x > 0:
        exit_times.append((BOARD_SIZE - start_x) / velocity_x)
    elif velocity_x < 0:
        exit_times.append((0.0 - start_x) / velocity_x)
    if velocity_y > 0:
        exit_times.append((BOARD_SIZE - start_y) / velocity_y)
    elif velocity_y < 0:
        exit_times.append((0.0 - start_y) / velocity_y)
    positive_times = [time_value for time_value in exit_times if time_value >= 0.0]
    return min(positive_times) if positive_times else None


def segment_orbit_clearance(
    start: tuple[float, float],
    end: tuple[float, float],
    orbit_radius: float,
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_sq = segment_x * segment_x + segment_y * segment_y
    if length_sq == 0.0:
        distance = math.hypot(start[0] - CENTER, start[1] - CENTER)
        return abs(distance - orbit_radius)

    projection = -((start[0] - CENTER) * segment_x + (start[1] - CENTER) * segment_y) / length_sq
    projection = min(max(projection, 0.0), 1.0)
    closest_x = start[0] + projection * segment_x
    closest_y = start[1] + projection * segment_y
    min_radius = math.hypot(closest_x - CENTER, closest_y - CENTER)
    max_radius = max(
        math.hypot(start[0] - CENTER, start[1] - CENTER),
        math.hypot(end[0] - CENTER, end[1] - CENTER),
    )
    if min_radius <= orbit_radius <= max_radius:
        return 0.0
    return min(abs(orbit_radius - min_radius), abs(orbit_radius - max_radius))


def moving_blocker_margin(
    source: Planet,
    blocker: Planet,
    angle: float,
    ships: int,
    hit_time: float,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
) -> float:
    speed = fleet_speed(ships)
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    start_x = source.x + dir_x * (source.radius + 0.1)
    start_y = source.y + dir_y * (source.radius + 0.1)
    fleet_x = start_x + dir_x * speed * hit_time
    fleet_y = start_y + dir_y * speed * hit_time
    blocker_x, blocker_y = planet_position_after_moves(
        blocker,
        hit_time,
        initial_by_id,
        angular_velocity,
        current_step,
        comet_ids,
    )
    return math.hypot(fleet_x - blocker_x, fleet_y - blocker_y) - blocker.radius


def moving_blocker_margin_derivative(
    source: Planet,
    blocker: Planet,
    angle: float,
    ships: int,
    hit_time: float,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
) -> float:
    speed = fleet_speed(ships)
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    start_x = source.x + dir_x * (source.radius + 0.1)
    start_y = source.y + dir_y * (source.radius + 0.1)
    fleet_x = start_x + dir_x * speed * hit_time
    fleet_y = start_y + dir_y * speed * hit_time
    blocker_x, blocker_y = planet_position_after_moves(
        blocker,
        hit_time,
        initial_by_id,
        angular_velocity,
        current_step,
        comet_ids,
    )
    dx = fleet_x - blocker_x
    dy = fleet_y - blocker_y
    dist = math.hypot(dx, dy)
    if dist == 0:
        return -speed
    initial = initial_by_id.get(blocker.id)
    if initial is None or blocker.id in comet_ids:
        blocker_vx = blocker_vy = 0.0
    else:
        orbital_radius = math.hypot(initial[2] - CENTER, initial[3] - CENTER)
        if orbital_radius + initial[4] >= ROTATION_RADIUS_LIMIT:
            blocker_vx = blocker_vy = 0.0
        else:
            initial_angle = math.atan2(initial[3] - CENTER, initial[2] - CENTER)
            env_time = max(1, int(current_step)) + hit_time - 1
            future_angle = initial_angle + angular_velocity * env_time
            blocker_vx = -orbital_radius * angular_velocity * math.sin(future_angle)
            blocker_vy = orbital_radius * angular_velocity * math.cos(future_angle)
    relative_vx = dir_x * speed - blocker_vx
    relative_vy = dir_y * speed - blocker_vy
    return (dx * relative_vx + dy * relative_vy) / dist


def moving_blocker_collision_time(
    source: Planet,
    blocker: Planet,
    angle: float,
    ships: int,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
    max_time: float,
) -> float | None:
    speed = fleet_speed(ships)
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    start_x = source.x + dir_x * (source.radius + 0.1)
    start_y = source.y + dir_y * (source.radius + 0.1)
    initial = initial_by_id.get(blocker.id)
    blocker_orbits = False
    orbital_radius = 0.0
    initial_angle = 0.0
    if initial is not None and blocker.id not in comet_ids:
        orbital_radius = math.hypot(initial[2] - CENTER, initial[3] - CENTER)
        if orbital_radius + initial[4] < ROTATION_RADIUS_LIMIT:
            blocker_orbits = True
            initial_angle = math.atan2(initial[3] - CENTER, initial[2] - CENTER)

    def blocker_position(hit_time: float) -> tuple[float, float]:
        if hit_time <= 0.0 or not blocker_orbits:
            return blocker.x, blocker.y
        env_time = max(1, int(current_step)) + hit_time - 1
        future_angle = initial_angle + angular_velocity * env_time
        return (
            CENTER + orbital_radius * math.cos(future_angle),
            CENTER + orbital_radius * math.sin(future_angle),
        )

    def margin_at(hit_time: float) -> float:
        fleet_x = start_x + dir_x * speed * hit_time
        fleet_y = start_y + dir_y * speed * hit_time
        blocker_x, blocker_y = blocker_position(hit_time)
        return math.hypot(fleet_x - blocker_x, fleet_y - blocker_y) - blocker.radius

    def derivative_at(hit_time: float) -> float:
        fleet_x = start_x + dir_x * speed * hit_time
        fleet_y = start_y + dir_y * speed * hit_time
        blocker_x, blocker_y = blocker_position(hit_time)
        dx = fleet_x - blocker_x
        dy = fleet_y - blocker_y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return -speed
        if blocker_orbits:
            env_time = max(1, int(current_step)) + hit_time - 1
            future_angle = initial_angle + angular_velocity * env_time
            blocker_vx = -orbital_radius * angular_velocity * math.sin(future_angle)
            blocker_vy = orbital_radius * angular_velocity * math.cos(future_angle)
        else:
            blocker_vx = blocker_vy = 0.0
        relative_vx = dir_x * speed - blocker_vx
        relative_vy = dir_y * speed - blocker_vy
        return (dx * relative_vx + dy * relative_vy) / dist

    projection_time = ((blocker.x - start_x) * dir_x + (blocker.y - start_y) * dir_y) / speed
    seed_times = []
    for offset in (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0):
        seed = min(max(projection_time + offset, 0.0), max_time)
        if seed not in seed_times:
            seed_times.append(seed)
    best_time = None
    for seed in seed_times:
        hit_time = seed
        for _ in range(10):
            margin = margin_at(hit_time)
            if abs(margin) <= 1e-4:
                break
            derivative = derivative_at(hit_time)
            if abs(derivative) < 1e-6:
                break
            next_time = hit_time - margin / derivative
            if not math.isfinite(next_time):
                break
            next_time = min(max(next_time, 0.0), max_time)
            if abs(next_time - hit_time) <= 1e-4:
                hit_time = next_time
                break
            hit_time = next_time
        if not (0.0 <= hit_time <= max_time):
            continue
        if margin_at(hit_time) <= 0.05:
            if best_time is None or hit_time < best_time:
                best_time = hit_time
    return best_time


def moving_intercept_margin(
    source: Planet,
    target: Planet,
    ships: int,
    hit_time: float,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
) -> float:
    speed = fleet_speed(ships)
    target_x, target_y = planet_position_after_moves(target, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
    distance_to_target_edge = math.hypot(target_x - source.x, target_y - source.y) - source.radius - 0.1 - target.radius
    return distance_to_target_edge - speed * hit_time


def moving_intercept_margin_derivative(
    source: Planet,
    target: Planet,
    ships: int,
    hit_time: float,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
) -> float:
    speed = fleet_speed(ships)
    target_x, target_y = planet_position_after_moves(target, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
    dx = target_x - source.x
    dy = target_y - source.y
    distance = math.hypot(dx, dy)
    if distance == 0:
        return -speed
    initial = initial_by_id.get(target.id)
    if initial is None or target.id in comet_ids:
        return -speed
    orbital_radius = math.hypot(initial[2] - CENTER, initial[3] - CENTER)
    if orbital_radius + initial[4] >= ROTATION_RADIUS_LIMIT:
        return -speed
    initial_angle = math.atan2(initial[3] - CENTER, initial[2] - CENTER)
    env_time = max(1, int(current_step)) + hit_time - 1
    future_angle = initial_angle + angular_velocity * env_time
    target_vx = -orbital_radius * angular_velocity * math.sin(future_angle)
    target_vy = orbital_radius * angular_velocity * math.cos(future_angle)
    distance_derivative = (dx * target_vx + dy * target_vy) / distance
    return distance_derivative - speed


def find_moving_intercept_newton(
    source: Planet,
    target: Planet,
    ships: int,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
    max_turn: int = 90,
) -> tuple[float, int, float] | None:
    speed = fleet_speed(ships)
    initial_distance = max(0.0, math.hypot(target.x - source.x, target.y - source.y) - source.radius - 0.1 - target.radius)
    estimated_time = max(0.25, initial_distance / speed)
    seed_times = []
    for offset in (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0, 18.0):
        seed = min(max(estimated_time + offset, 0.25), float(max_turn))
        if seed not in seed_times:
            seed_times.append(seed)
    best = None
    for seed in seed_times:
        hit_time = seed
        for _ in range(12):
            margin = moving_intercept_margin(source, target, ships, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
            if abs(margin) <= 1e-4:
                break
            derivative = moving_intercept_margin_derivative(source, target, ships, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
            if abs(derivative) < 1e-6:
                break
            next_time = hit_time - margin / derivative
            if not math.isfinite(next_time):
                break
            clamped_next_time = min(max(next_time, 0.25), float(max_turn))
            if clamped_next_time != next_time:
                clamped_next_time = (hit_time + clamped_next_time) / 2.0
            if abs(clamped_next_time - hit_time) <= 1e-4:
                hit_time = clamped_next_time
                break
            hit_time = clamped_next_time
        if not (0.0 < hit_time <= max_turn):
            continue
        final_margin = moving_intercept_margin(source, target, ships, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
        if final_margin > 0.05:
            continue
        target_x, target_y = planet_position_after_moves(target, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
        angle = math.atan2(target_y - source.y, target_x - source.x)
        hit_turn = max(1, int(math.ceil(hit_time)))
        candidate = (angle, hit_turn, hit_time)
        if best is None or hit_time < best[2]:
            best = candidate
    return best


def target_collision_time(
    source: Planet,
    target: Planet,
    angle: float,
    ships: int,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
    max_turn: int,
) -> float:
    if target.id in comet_ids:
        return float(max_turn)
    speed = fleet_speed(ships)
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    start_x = source.x + dir_x * (source.radius + 0.1)
    start_y = source.y + dir_y * (source.radius + 0.1)
    initial = initial_by_id.get(target.id)
    if initial is not None:
        orbital_radius = math.hypot(initial[2] - CENTER, initial[3] - CENTER)
        if orbital_radius + initial[4] < ROTATION_RADIUS_LIMIT:
            intercept = find_moving_intercept_newton(source, target, ships, initial_by_id, angular_velocity, current_step, comet_ids, max_turn=max_turn)
            if intercept is not None:
                return intercept[2]
    return max(0.0, math.hypot(target.x - start_x, target.y - start_y) - target.radius) / speed


def find_attack_blocker(
    source: Planet,
    target: Planet,
    angle: float,
    ships: int,
    planets: list[Planet],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
    max_turn: int = 90,
    target_hit_time: float | None = None,
    motion_types: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    speed = fleet_speed(ships)
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    start_x = source.x + dir_x * (source.radius + 0.1)
    start_y = source.y + dir_y * (source.radius + 0.1)
    if target_hit_time is None:
        target_hit_time = target_collision_time(source, target, angle, ships, initial_by_id, angular_velocity, current_step, comet_ids, max_turn)
    target_hit_time = min(float(target_hit_time), float(max_turn))
    best = None
    exit_time = board_exit_time(start_x, start_y, dir_x, dir_y, speed)
    if exit_time is not None and exit_time < target_hit_time:
        best = (exit_time, {"kind": "board", "planet_id": None, "turn": math.ceil(exit_time)})
    sun_time = circle_collision_time(start_x, start_y, dir_x, dir_y, speed, CENTER, CENTER, SUN_RADIUS, target_hit_time)
    if sun_time is not None and (best is None or sun_time < best[0]):
        best = (sun_time, {"kind": "sun", "planet_id": None, "turn": math.ceil(sun_time)})
    if motion_types is None:
        motion_types = planet_motion_types(planets, initial_by_id, comet_ids)
    path_end = (start_x + dir_x * speed * target_hit_time, start_y + dir_y * speed * target_hit_time)
    for planet in planets:
        if planet.id in (source.id, target.id) or planet.id in comet_ids:
            continue
        motion_type = motion_types.get(planet.id, "unknown")
        if motion_type == "orbiting":
            initial = initial_by_id.get(planet.id)
            if initial is not None:
                orbital_radius = math.hypot(initial[2] - CENTER, initial[3] - CENTER)
                if segment_orbit_clearance((start_x, start_y), path_end, orbital_radius) > planet.radius + 1e-6:
                    continue
            blocker_time = moving_blocker_collision_time(source, planet, angle, ships, initial_by_id, angular_velocity, current_step, comet_ids, target_hit_time)
        else:
            blocker_time = circle_collision_time(start_x, start_y, dir_x, dir_y, speed, planet.x, planet.y, planet.radius, target_hit_time)
        if blocker_time is not None and (best is None or blocker_time < best[0]):
            best = (blocker_time, {"kind": "planet", "planet_id": planet.id, "turn": math.ceil(blocker_time)})
    return best[1] if best is not None else None


def find_valid_attack_angle(
    source: Planet,
    target: Planet,
    ships: int,
    planets: list[Planet],
    initial_by_id: dict[int, Any],
    motion_types: dict[int, str],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
) -> float | None:
    if motion_types.get(target.id) == "orbiting":
        intercept = find_moving_intercept_newton(source, target, ships, initial_by_id, angular_velocity, current_step, comet_ids)
        if intercept is None:
            return None
        angle, hit_turn, hit_time = intercept
        blocker = find_attack_blocker(
            source,
            target,
            angle,
            ships,
            planets,
            initial_by_id,
            angular_velocity,
            current_step,
            comet_ids,
            max_turn=hit_turn + 2,
            target_hit_time=hit_time,
            motion_types=motion_types,
        )
        return angle if blocker is None else None
    angle = math.atan2(target.y - source.y, target.x - source.x)
    blocker = find_attack_blocker(source, target, angle, ships, planets, initial_by_id, angular_velocity, current_step, comet_ids, motion_types=motion_types)
    return angle if blocker is None else None


def _base_fleet_floor(target: Planet) -> int:
    return int(target.ships) + 1


def _target_fleet_floor(target: Planet, player_id: int, available_step: int, current_step: int) -> int:
    if target.owner in (-1, player_id):
        return _base_fleet_floor(target)
    growth_turns = max(0, int(available_step) - int(current_step))
    return int(target.ships) + int(target.production) * growth_turns + 1


def _base_cheap_production_score(target: Planet) -> float:
    return (5.0 * float(target.production)) / max(1.0, float(_base_fleet_floor(target)))


def _wait_turns_to_leave_one(source: Planet, ships_to_send: int) -> int:
    required_source_ships = ships_to_send + 1
    if int(source.ships) >= required_source_ships:
        return 0
    production = max(1, int(source.production))
    return int(math.ceil((required_source_ships - int(source.ships)) / production))


def _opening_horizon(step: int) -> int:
    return int(step) + OPENING_SCORE_TURNS


def _net_opening_value(target: Planet, ships_to_send: int, capture_step: int, horizon_step: int) -> float:
    producing_turns = max(0, int(horizon_step) - int(capture_step))
    return float(target.production) * producing_turns - float(ships_to_send)


def _quadrant(planet: Planet) -> tuple[int, int]:
    return (0 if float(planet.x) < CENTER else 1, 0 if float(planet.y) < CENTER else 1)


def _opponent_quadrant(home_quadrant: tuple[int, int]) -> tuple[int, int]:
    home_qx, home_qy = home_quadrant
    return (1 - home_qx, 1 - home_qy)


def _home_quadrant(
    player_id: int,
    planets: list[Planet],
    initial_planets: list[Planet],
) -> tuple[int, int] | None:
    for planet in initial_planets:
        if int(planet.owner) == int(player_id):
            return _quadrant(planet)
    for planet in planets:
        if int(planet.owner) == int(player_id):
            return _quadrant(planet)
    return None


def _target_filter_reason(
    planet: Planet,
    player_id: int,
    comet_ids: set[int],
    future_sources_by_id: dict[int, int],
    blocked_quadrant: tuple[int, int] | None,
) -> str | None:
    if planet.owner == player_id:
        return "owned"
    if planet.id in comet_ids:
        return "comet"
    if planet.id in future_sources_by_id:
        return f"future source@{future_sources_by_id[int(planet.id)]}"
    if blocked_quadrant is not None and _quadrant(planet) == blocked_quadrant:
        return f"opponent quadrant before step {OPPONENT_QUADRANT_ATTACK_STEP}"
    return None


def _first_capture_candidates(
    planets: list[Planet],
    player_id: int,
    comet_ids: set[int],
    future_sources_by_id: dict[int, int],
    blocked_quadrant: tuple[int, int] | None,
) -> tuple[list[Planet], list[dict[str, Any]]]:
    eliminated = []
    for planet in planets:
        reason = _target_filter_reason(planet, player_id, comet_ids, future_sources_by_id, blocked_quadrant)
        if reason is not None:
            eliminated.append({"planet_id": int(planet.id), "reason": reason})

    targets = [
        planet
        for planet in planets
        if _target_filter_reason(planet, player_id, comet_ids, future_sources_by_id, blocked_quadrant) is None
    ]
    targets.sort(key=_base_cheap_production_score, reverse=True)
    return targets, eliminated


def _travel_turns(
    source: Planet,
    target: Planet,
    ships_to_send: int,
    planets: list[Planet],
    initial_by_id: dict[int, Any],
    motion_types: dict[int, str],
    angular_velocity: float,
    launch_step: int,
    comet_ids: set[int],
) -> tuple[float, float | None]:
    angle = find_valid_attack_angle(
        source,
        target,
        ships_to_send,
        planets,
        initial_by_id,
        motion_types,
        angular_velocity,
        launch_step,
        comet_ids,
    )
    if angle is None:
        return float("inf"), None

    hit_time = target_collision_time(
        source,
        target,
        angle,
        ships_to_send,
        initial_by_id,
        angular_velocity,
        launch_step,
        comet_ids,
        MAX_COLLISION_TURN,
    )
    return max(1.0, float(math.ceil(hit_time))), float(angle)


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    length_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if length_sq == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1]))
            / length_sq,
        ),
    )
    projection = (start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1]))
    return math.hypot(point[0] - projection[0], point[1] - projection[1])


def _point_to_segment_progress(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    length_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if length_sq == 0.0:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * (end[0] - start[0]) + (point[1] - start[1]) * (end[1] - start[1]))
            / length_sq,
        ),
    )


def _cross_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _segment_intersection_progress(
    start_a: tuple[float, float],
    end_a: tuple[float, float],
    start_b: tuple[float, float],
    end_b: tuple[float, float],
) -> float | None:
    r = (end_a[0] - start_a[0], end_a[1] - start_a[1])
    s = (end_b[0] - start_b[0], end_b[1] - start_b[1])
    denominator = _cross_2d(r, s)
    if abs(denominator) < 1e-12:
        return None
    qp = (start_b[0] - start_a[0], start_b[1] - start_a[1])
    t = _cross_2d(qp, s) / denominator
    u = _cross_2d(qp, r) / denominator
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return max(0.0, min(1.0, t))
    return None


def _segment_to_segment_distance(
    start_a: tuple[float, float],
    end_a: tuple[float, float],
    start_b: tuple[float, float],
    end_b: tuple[float, float],
) -> float:
    if _segment_intersection_progress(start_a, end_a, start_b, end_b) is not None:
        return 0.0
    return min(
        _point_to_segment_distance(start_a, start_b, end_b),
        _point_to_segment_distance(end_a, start_b, end_b),
        _point_to_segment_distance(start_b, start_a, end_a),
        _point_to_segment_distance(end_b, start_a, end_a),
    )


def _segment_collision_progress(
    fleet_start: tuple[float, float],
    fleet_end: tuple[float, float],
    target_start: tuple[float, float],
    target_end: tuple[float, float],
) -> float:
    intersection_progress = _segment_intersection_progress(fleet_start, fleet_end, target_start, target_end)
    if intersection_progress is not None:
        return intersection_progress
    return min(
        _point_to_segment_progress(target_start, fleet_start, fleet_end),
        _point_to_segment_progress(target_end, fleet_start, fleet_end),
    )


def _fleet_future_hit(
    fleet: Fleet,
    target_planets: list[Planet],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
    lookahead: int = FUTURE_SOURCE_LOOKAHEAD,
    deadline: float | None = None,
) -> tuple[int, int] | None:
    speed = fleet_speed(fleet.ships, MAX_SPEED)
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed
    old_x = float(fleet.x)
    old_y = float(fleet.y)
    for turn in range(1, int(lookahead) + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        new_x = old_x + dx
        new_y = old_y + dy
        if not (0.0 <= new_x <= 100.0 and 0.0 <= new_y <= 100.0):
            break

        best_hit: tuple[float, int] | None = None
        for planet in target_planets:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            target_old = planet_position_after_moves(
                planet,
                max(0, turn - 1),
                initial_by_id,
                angular_velocity,
                step,
                comet_ids,
            )
            target_new = planet_position_after_moves(
                planet,
                turn,
                initial_by_id,
                angular_velocity,
                step,
                comet_ids,
            )
            hit = _segment_to_segment_distance((old_x, old_y), (new_x, new_y), target_old, target_new) < planet.radius
            if hit:
                progress = _segment_collision_progress((old_x, old_y), (new_x, new_y), target_old, target_new)
                if best_hit is None or progress < best_hit[0]:
                    best_hit = (progress, int(planet.id))

        if best_hit is not None:
            return best_hit[1], step + turn

        old_x = new_x
        old_y = new_y

    return None


def _incoming_events_by_target(
    fleets: list[Fleet],
    target_planets: list[Planet],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
    lookahead: int = MAX_COLLISION_TURN,
    deadline: float | None = None,
) -> dict[int, dict[int, dict[int, int]]]:
    events_by_target: dict[int, dict[int, dict[int, int]]] = {}
    if not fleets or not target_planets:
        return events_by_target

    for fleet in fleets:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        future_hit = _fleet_future_hit(
            fleet,
            target_planets,
            initial_by_id,
            angular_velocity,
            step,
            comet_ids,
            lookahead=lookahead,
            deadline=deadline,
        )
        if future_hit is None:
            continue
        target_id, hit_step = future_hit
        hit_turn = int(hit_step) - int(step)
        if hit_turn <= 0 or hit_turn > lookahead:
            continue
        target_events = events_by_target.setdefault(int(target_id), {})
        turn_events = target_events.setdefault(hit_turn, {})
        owner = int(fleet.owner)
        turn_events[owner] = turn_events.get(owner, 0) + int(fleet.ships)

    return events_by_target


def _resolve_fleet_group_against_planet(
    owner: int,
    ships: int,
    fleet_ships_by_owner: dict[int, int],
) -> tuple[int, int]:
    if not fleet_ships_by_owner:
        return int(owner), int(ships)

    sorted_players = sorted(fleet_ships_by_owner.items(), key=lambda item: item[1], reverse=True)
    top_player, top_ships = sorted_players[0]
    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        survivor_ships = int(top_ships) - int(second_ships)
        survivor_owner = int(top_player) if survivor_ships > 0 else -1
    else:
        survivor_owner = int(top_player)
        survivor_ships = int(top_ships)

    owner = int(owner)
    ships = int(ships)
    if survivor_ships > 0:
        if owner == survivor_owner:
            ships += survivor_ships
        else:
            ships -= survivor_ships
            if ships < 0:
                owner = survivor_owner
                ships = abs(ships)
    return owner, ships


def _project_target_before_arrival(
    target: Planet,
    arrival_turns: int,
    incoming_events_by_target: dict[int, dict[int, dict[int, int]]],
) -> tuple[int, int, dict[int, int]]:
    owner = int(target.owner)
    ships = int(target.ships)
    arrival_turns = max(0, int(arrival_turns))
    target_events = incoming_events_by_target.get(int(target.id), {})

    current_turn = 0
    for turn in sorted(turn for turn in target_events if 1 <= int(turn) < arrival_turns):
        turn = int(turn)
        if owner != -1:
            ships += int(target.production) * (turn - current_turn)
        turn_events = target_events.get(turn, {})
        if turn_events:
            owner, ships = _resolve_fleet_group_against_planet(owner, ships, turn_events)
        current_turn = turn

    if owner != -1:
        ships += int(target.production) * (arrival_turns - current_turn)
    if arrival_turns > 0:
        return owner, ships, dict(target_events.get(arrival_turns, {}))

    return owner, ships, {}


def _minimum_ships_to_own_after_arrival(
    target_owner: int,
    target_ships: int,
    same_turn_fleets: dict[int, int],
    player_id: int,
) -> int:
    if int(target_owner) == int(player_id):
        return 0

    friendly_same_turn = int(same_turn_fleets.get(int(player_id), 0))
    strongest_other = 0
    for owner, ships in same_turn_fleets.items():
        if int(owner) != int(player_id):
            strongest_other = max(strongest_other, int(ships))

    required_friendly_total = strongest_other + int(target_ships) + 1
    return max(1, required_friendly_total - friendly_same_turn)


def _projected_target_fleet_floor(
    target: Planet,
    player_id: int,
    available_step: int,
    current_step: int,
    incoming_events_by_target: dict[int, dict[int, dict[int, int]]],
) -> tuple[int, dict[str, Any]]:
    arrival_turns = max(0, int(available_step) - int(current_step))
    projected_owner, projected_ships, same_turn_fleets = _project_target_before_arrival(
        target,
        arrival_turns,
        incoming_events_by_target,
    )
    ships_needed = _minimum_ships_to_own_after_arrival(
        projected_owner,
        projected_ships,
        same_turn_fleets,
        player_id,
    )
    strongest_other = 0
    friendly_same_turn = int(same_turn_fleets.get(int(player_id), 0))
    for owner, ships in same_turn_fleets.items():
        if int(owner) != int(player_id):
            strongest_other = max(strongest_other, int(ships))
    details = {
        "arrival_turns": int(arrival_turns),
        "projected_owner": int(projected_owner),
        "projected_ships": int(projected_ships),
        "same_turn_friendly": int(friendly_same_turn),
        "same_turn_enemy_max": int(strongest_other),
        "same_turn_fleets": {str(owner): int(ships) for owner, ships in same_turn_fleets.items()},
    }
    return int(ships_needed), details


def _simulate_target_with_planned_fleet(
    target: Planet,
    player_id: int,
    planned_ships: int,
    arrival_turns: int,
    horizon_turns: int,
    incoming_events_by_target: dict[int, dict[int, dict[int, int]]],
) -> dict[str, Any]:
    owner = int(target.owner)
    ships = int(target.ships)
    arrival_turns = max(1, int(arrival_turns))
    horizon_turns = max(arrival_turns, int(horizon_turns))
    target_events = incoming_events_by_target.get(int(target.id), {})
    owned_production = 0
    arrived_securely = False
    survived_known_incoming = True
    lost_turn: int | None = None
    arrival_owner = owner
    arrival_ships = ships

    relevant_turns = {
        int(turn)
        for turn in target_events
        if 1 <= int(turn) <= horizon_turns
    }
    relevant_turns.add(arrival_turns)
    current_turn = 0
    for turn in sorted(relevant_turns):
        if turn <= current_turn:
            continue
        production_turns = turn - current_turn
        if owner == int(player_id) and ships > 0:
            ships += int(target.production) * production_turns
            owned_production += int(target.production) * production_turns
        elif owner != -1:
            ships += int(target.production) * production_turns
        turn_events = dict(target_events.get(turn, {}))
        if turn == arrival_turns:
            turn_events[int(player_id)] = turn_events.get(int(player_id), 0) + int(planned_ships)

        if turn_events:
            owner, ships = _resolve_fleet_group_against_planet(owner, ships, turn_events)

        if turn == arrival_turns:
            arrival_owner = int(owner)
            arrival_ships = int(ships)
            arrived_securely = owner == int(player_id) and ships > 0
            if not arrived_securely:
                survived_known_incoming = False
                lost_turn = turn
        elif turn > arrival_turns and arrived_securely and (owner != int(player_id) or ships <= 0):
            survived_known_incoming = False
            lost_turn = turn
            arrived_securely = False
        current_turn = turn

    if current_turn < horizon_turns:
        production_turns = horizon_turns - current_turn
        if owner == int(player_id) and ships > 0:
            ships += int(target.production) * production_turns
            owned_production += int(target.production) * production_turns
        elif owner != -1:
            ships += int(target.production) * production_turns

    return {
        "arrival_owner": int(arrival_owner),
        "arrival_ships": int(arrival_ships),
        "final_owner": int(owner),
        "final_ships": int(ships),
        "owned_production": int(owned_production),
        "survived_known_incoming": bool(survived_known_incoming),
        "lost_turn": lost_turn,
    }


def _survival_aware_target_fleet_floor(
    target: Planet,
    player_id: int,
    available_step: int,
    current_step: int,
    horizon_step: int,
    incoming_events_by_target: dict[int, dict[int, dict[int, int]]],
) -> tuple[int, dict[str, Any]]:
    capture_floor, details = _projected_target_fleet_floor(
        target,
        player_id,
        available_step,
        current_step,
        incoming_events_by_target,
    )
    details["capture_ships_needed"] = int(capture_floor)
    if capture_floor <= 0:
        details.update(
            {
                "survival_ships_needed": int(capture_floor),
                "survival_extra_ships": 0,
                "owned_production": 0,
                "survived_known_incoming": True,
                "lost_turn": None,
            }
        )
        return int(capture_floor), details

    arrival_turns = max(1, int(available_step) - int(current_step))
    horizon_turns = max(arrival_turns, int(horizon_step) - int(current_step))
    target_events = incoming_events_by_target.get(int(target.id), {})
    other_incoming = 0
    for turn, ships_by_owner in target_events.items():
        if int(turn) < arrival_turns or int(turn) > horizon_turns:
            continue
        for owner, ships in ships_by_owner.items():
            if int(owner) != int(player_id):
                other_incoming += int(ships)

    low = max(1, int(capture_floor))
    high = max(
        low,
        int(target.ships) + int(target.production) * horizon_turns + other_incoming + 2,
    )

    def survives(planned_ships: int) -> tuple[bool, dict[str, Any]]:
        simulation = _simulate_target_with_planned_fleet(
            target,
            player_id,
            planned_ships,
            arrival_turns,
            horizon_turns,
            incoming_events_by_target,
        )
        return bool(simulation["survived_known_incoming"]), simulation

    survived, simulation = survives(high)
    while not survived and high < 1_000_000:
        high *= 2
        survived, simulation = survives(high)

    if survived:
        while low < high:
            mid = (low + high) // 2
            mid_survived, _mid_simulation = survives(mid)
            if mid_survived:
                high = mid
            else:
                low = mid + 1
        survived, simulation = survives(low)
    else:
        low = high

    details.update(simulation)
    details["survival_ships_needed"] = int(low)
    details["survival_extra_ships"] = max(0, int(low) - int(capture_floor))
    return int(low), details


def _simulate_source_after_launch(
    source: Planet,
    player_id: int,
    planned_ships: int,
    launch_turns: int,
    horizon_turns: int,
    incoming_events_by_target: dict[int, dict[int, dict[int, int]]],
) -> dict[str, Any]:
    owner = int(source.owner)
    ships = int(source.ships)
    launch_turns = max(0, int(launch_turns))
    horizon_turns = max(launch_turns, int(horizon_turns))
    source_events = incoming_events_by_target.get(int(source.id), {})
    launched = False
    survived = owner == int(player_id) and ships > 0
    lost_turn: int | None = None

    if launch_turns == 0:
        ships -= int(planned_ships)
        launched = True
        if owner != int(player_id) or ships <= 0:
            survived = False
            lost_turn = 0

    relevant_turns = {
        int(turn)
        for turn in source_events
        if 1 <= int(turn) <= horizon_turns
    }
    if launch_turns > 0:
        relevant_turns.add(launch_turns)

    current_turn = 0
    for turn in sorted(relevant_turns):
        if turn <= current_turn:
            continue
        production_turns = turn - current_turn
        if owner == int(player_id) and ships > 0:
            ships += int(source.production) * production_turns
        elif owner != -1:
            ships += int(source.production) * production_turns

        if turn == launch_turns and not launched:
            if owner != int(player_id) or ships <= int(planned_ships):
                survived = False
                lost_turn = turn
                launched = True
            else:
                ships -= int(planned_ships)
                launched = True

        turn_events = source_events.get(turn, {})
        if turn_events:
            owner, ships = _resolve_fleet_group_against_planet(owner, ships, turn_events)

        if launched and (owner != int(player_id) or ships <= 0):
            survived = False
            if lost_turn is None:
                lost_turn = turn
            break
        current_turn = turn

    if survived and current_turn < horizon_turns:
        production_turns = horizon_turns - current_turn
        if owner == int(player_id) and ships > 0:
            ships += int(source.production) * production_turns
        elif owner != -1:
            ships += int(source.production) * production_turns

    return {
        "source_final_owner": int(owner),
        "source_final_ships": int(ships),
        "source_survives_known_incoming": bool(survived),
        "source_lost_turn": lost_turn,
    }


def _max_safe_source_ships_to_send(
    source: Planet,
    player_id: int,
    launch_step: int,
    current_step: int,
    horizon_step: int,
    incoming_events_by_target: dict[int, dict[int, dict[int, int]]],
) -> tuple[int, dict[str, Any]]:
    launch_turns = max(0, int(launch_step) - int(current_step))
    horizon_turns = max(launch_turns, int(horizon_step) - int(current_step))
    max_available = max(0, int(source.ships) + int(source.production) * launch_turns - 1)

    def survives(planned_ships: int) -> tuple[bool, dict[str, Any]]:
        simulation = _simulate_source_after_launch(
            source,
            player_id,
            planned_ships,
            launch_turns,
            horizon_turns,
            incoming_events_by_target,
        )
        return bool(simulation["source_survives_known_incoming"]), simulation

    zero_survives, zero_details = survives(0)
    if not zero_survives or max_available <= 0:
        zero_details["source_max_safe_ships"] = 0
        zero_details["source_launch_turns"] = int(launch_turns)
        return 0, zero_details

    low = 0
    high = max_available
    best_details = zero_details
    while low < high:
        mid = (low + high + 1) // 2
        mid_survives, mid_details = survives(mid)
        if mid_survives:
            low = mid
            best_details = mid_details
        else:
            high = mid - 1

    survived, best_details = survives(low)
    if not survived:
        low = 0
    best_details["source_max_safe_ships"] = int(low)
    best_details["source_launch_turns"] = int(launch_turns)
    return int(low), best_details


def _future_sources_by_id(
    fleets: list[Fleet],
    target_planets: list[Planet],
    player_id: int,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
    incoming_events_by_target: dict[int, dict[int, dict[int, int]]] | None = None,
    deadline: float | None = None,
) -> dict[int, int]:
    if not fleets:
        return {}

    if incoming_events_by_target is not None:
        future_sources: dict[int, int] = {}
        for target in target_planets:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            owner = int(target.owner)
            ships = int(target.ships)
            target_events = incoming_events_by_target.get(int(target.id), {})
            for turn in range(1, FUTURE_SOURCE_LOOKAHEAD + 1):
                if deadline is not None and time.perf_counter() >= deadline:
                    return future_sources
                if owner != -1:
                    ships += int(target.production)
                turn_events = target_events.get(turn, {})
                if turn_events:
                    owner, ships = _resolve_fleet_group_against_planet(owner, ships, turn_events)
                if owner == int(player_id):
                    future_sources[int(target.id)] = int(step) + turn
                    break
        return future_sources

    target_by_id = {int(planet.id): planet for planet in target_planets}
    incoming_by_target: dict[int, list[tuple[int, int]]] = {}
    for fleet in fleets:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if int(fleet.owner) != player_id:
            continue
        future_hit = _fleet_future_hit(
            fleet,
            target_planets,
            initial_by_id,
            angular_velocity,
            step,
            comet_ids,
            deadline=deadline,
        )
        if future_hit is None:
            continue
        target_id, available_step = future_hit
        incoming_by_target.setdefault(target_id, []).append((available_step, int(fleet.ships)))

    future_sources: dict[int, int] = {}
    for target_id, incoming in incoming_by_target.items():
        if deadline is not None and time.perf_counter() >= deadline:
            break
        target = target_by_id.get(target_id)
        if target is None:
            continue
        cumulative_ships = 0
        for available_step, ships in sorted(incoming):
            cumulative_ships += ships
            if cumulative_ships >= _target_fleet_floor(target, player_id, available_step, step):
                future_sources[target_id] = available_step
                break

    return future_sources


def _matrix_weight(row: dict[str, Any]) -> float:
    return (
        float(row["net_value"]) * 1_000_000_000_000.0
        - float(row["total_time"]) * 1_000_000.0
        - float(row["wait_turns"]) * 10_000.0
        - float(row["travel_turns"]) * 100.0
        + float(row["cheap_production"])
    )


def _best_matrix_assignment(rows: list[dict[str, Any]], source_order: list[int]) -> list[dict[str, Any]]:
    if not rows:
        return []

    best_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["source_id"]), int(row["target_id"]))
        existing = best_by_pair.get(key)
        if existing is None or _matrix_weight(row) > _matrix_weight(existing):
            best_by_pair[key] = row

    source_ids = [
        source_id
        for source_id in source_order
        if any(row_source == source_id for row_source, _target_id in best_by_pair)
    ]
    target_ids = sorted({target_id for _source_id, target_id in best_by_pair})
    if not source_ids or not target_ids:
        return []

    real_weights = [_matrix_weight(row) for row in best_by_pair.values()]
    max_weight = max(real_weights)
    min_weight = min(real_weights)
    skip_weight = min_weight - 1.0e12
    missing_weight = -1.0e30
    columns: list[int | None] = target_ids + [None] * len(source_ids)

    costs: list[list[float]] = []
    for source_id in source_ids:
        cost_row = []
        for target_id in columns:
            if target_id is None:
                weight = skip_weight
            else:
                row = best_by_pair.get((source_id, target_id))
                weight = _matrix_weight(row) if row is not None else missing_weight
            cost_row.append(max_weight - weight)
        costs.append(cost_row)

    column_by_source = _hungarian_min_cost(costs)
    selected = []
    for row_index, column_index in enumerate(column_by_source):
        if column_index < 0 or column_index >= len(columns):
            continue
        target_id = columns[column_index]
        if target_id is None:
            continue
        row = best_by_pair.get((source_ids[row_index], target_id))
        if row is not None:
            selected.append(row)
    return selected


def _hungarian_min_cost(costs: list[list[float]]) -> list[int]:
    row_count = len(costs)
    column_count = len(costs[0]) if costs else 0
    if row_count == 0 or column_count == 0:
        return []
    if row_count > column_count:
        raise ValueError("matrix assignment requires rows <= columns")

    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        p[0] = row
        current_column = 0
        minv = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = p[current_column]
            delta = float("inf")
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current_cost = costs[current_row - 1][column - 1] - u[current_row] - v[column]
                if current_cost < minv[column]:
                    minv[column] = current_cost
                    way[column] = current_column
                if minv[column] < delta:
                    delta = minv[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            current_column = next_column
            if p[current_column] == 0:
                break

        while True:
            previous_column = way[current_column]
            p[current_column] = p[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    return assignment


def _choose_opening(obs: Any, include_debug: bool = False) -> tuple[list[list[Any]], dict[str, Any]]:
    try:
        remaining_overage_time = float(get(obs, "remainingOverageTime", 60.0) or 60.0)
    except (TypeError, ValueError):
        remaining_overage_time = 60.0
    time_budget = LOW_OVERAGE_TIME_BUDGET_SECONDS if remaining_overage_time < 1.0 else AGENT_TIME_BUDGET_SECONDS
    deadline = time.perf_counter() + time_budget

    player_id = int(get(obs, "player", 0))
    step = int(get(obs, "step", 0))
    raw_planets = get(obs, "planets", [])
    raw_initial = get(obs, "initial_planets", raw_planets)
    angular_velocity = float(get(obs, "angular_velocity", 0.0))
    comet_ids = set(get(obs, "comet_planet_ids", []))

    planets = [Planet(*row) for row in raw_planets]
    fleets = [Fleet(*row) for row in get(obs, "fleets", [])]
    my_planets = [planet for planet in planets if planet.owner == player_id]
    debug: dict[str, Any] = {
        "player": player_id,
        "step": step,
        "blocked_quadrant": None,
        "future_sources_by_id": {},
        "eliminated": [],
        "comparisons": [],
        "selected": [],
        "time_budget_seconds": time_budget,
        "time_budget_exhausted": False,
    }
    if not my_planets:
        return [], debug

    initial_by_id = {row[0]: row for row in raw_initial}
    initial_planets = [Planet(*row) for row in raw_initial]
    home_quadrant = _home_quadrant(player_id, planets, initial_planets)
    blocked_quadrant = None
    if home_quadrant is not None and step < OPPONENT_QUADRANT_ATTACK_STEP:
        blocked_quadrant = _opponent_quadrant(home_quadrant)
        debug["blocked_quadrant"] = list(blocked_quadrant)
    debug["opponent_quadrant_attack_step"] = OPPONENT_QUADRANT_ATTACK_STEP

    motion_types = planet_motion_types(planets, initial_by_id, comet_ids)
    all_target_planets = [planet for planet in planets if planet.id not in comet_ids]
    incoming_events_by_target = _incoming_events_by_target(
        fleets,
        all_target_planets,
        initial_by_id,
        angular_velocity,
        step,
        comet_ids,
        lookahead=MAX_COLLISION_TURN,
        deadline=deadline,
    )
    target_planets = [planet for planet in planets if planet.owner != player_id and planet.id not in comet_ids]
    future_sources_by_id = _future_sources_by_id(
        fleets,
        target_planets,
        player_id,
        initial_by_id,
        angular_velocity,
        step,
        comet_ids,
        incoming_events_by_target,
        deadline,
    )
    debug["future_sources_by_id"] = {str(key): value for key, value in future_sources_by_id.items()}

    sources = sorted(my_planets, key=lambda planet: (planet.ships, planet.production), reverse=True)
    candidates, eliminated = _first_capture_candidates(
        planets,
        player_id,
        comet_ids,
        future_sources_by_id,
        blocked_quadrant,
    )
    if not candidates:
        if include_debug:
            for source in sources:
                debug["eliminated"].extend(
                    {"source_id": int(source.id), **item}
                    for item in eliminated
                )
        return [], debug

    travel_cache: dict[tuple[int, int, int, int], tuple[float, float | None]] = {}
    target_floor_cache: dict[tuple[int, int, int], tuple[int, dict[str, Any]]] = {}
    source_safe_cache: dict[tuple[int, int, int], tuple[int, dict[str, Any]]] = {}

    def source_at_launch_step(source: Planet, reference_step: int, launch_step: int) -> Planet:
        moves_done = max(0, int(launch_step) - int(reference_step))
        source_x, source_y = planet_position_after_moves(
            source,
            moves_done,
            initial_by_id,
            angular_velocity,
            int(reference_step),
            comet_ids,
        )
        return Planet(
            int(source.id),
            int(source.owner),
            float(source_x),
            float(source_y),
            float(source.radius),
            int(source.ships),
            int(source.production),
        )

    def cached_travel_turns(
        source: Planet,
        target: Planet,
        ships_to_send: int,
        launch_step: int,
    ) -> tuple[float, float | None]:
        cache_key = (int(source.id), int(target.id), int(ships_to_send), int(launch_step))
        cached = travel_cache.get(cache_key)
        if cached is not None:
            return cached
        result = _travel_turns(
            source,
            target,
            ships_to_send,
            planets,
            initial_by_id,
            motion_types,
            angular_velocity,
            launch_step,
            comet_ids,
        )
        travel_cache[cache_key] = result
        return result

    def cached_target_fleet_floor(
        target: Planet,
        arrival_step: int,
        horizon_step: int,
    ) -> tuple[int, dict[str, Any]]:
        cache_key = (int(target.id), int(arrival_step), int(horizon_step))
        cached = target_floor_cache.get(cache_key)
        if cached is not None:
            ships_needed, details = cached
            return ships_needed, dict(details)
        result = _survival_aware_target_fleet_floor(
            target,
            player_id,
            arrival_step,
            step,
            horizon_step,
            incoming_events_by_target,
        )
        target_floor_cache[cache_key] = (result[0], dict(result[1]))
        return result[0], dict(result[1])

    def cached_source_safe_ships(
        source: Planet,
        launch_step: int,
        horizon_step: int,
    ) -> tuple[int, dict[str, Any]]:
        cache_key = (int(source.id), int(launch_step), int(horizon_step))
        cached = source_safe_cache.get(cache_key)
        if cached is not None:
            safe_ships, details = cached
            return safe_ships, dict(details)
        result = _max_safe_source_ships_to_send(
            source,
            player_id,
            launch_step,
            step,
            horizon_step,
            incoming_events_by_target,
        )
        source_safe_cache[cache_key] = (result[0], dict(result[1]))
        return result[0], dict(result[1])

    matrix_rows: list[dict[str, Any]] = []
    for source in sources:
        if time.perf_counter() >= deadline:
            debug["time_budget_exhausted"] = True
            break
        if include_debug:
            debug["eliminated"].extend(
                {"source_id": int(source.id), **item}
                for item in eliminated
            )

        source_comparisons: list[dict[str, Any]] = []
        for target in candidates:
            if time.perf_counter() >= deadline:
                debug["time_budget_exhausted"] = True
                break
            ships_to_send = _base_fleet_floor(target)
            wait_turns = 0
            travel_turns = float("inf")
            angle = None
            projection_details: dict[str, Any] = {}
            horizon_step = _opening_horizon(step)
            timed_out = False
            for _ in range(8):
                if time.perf_counter() >= deadline:
                    debug["time_budget_exhausted"] = True
                    timed_out = True
                    break
                wait_turns = _wait_turns_to_leave_one(source, ships_to_send)
                launch_step = step + wait_turns
                launch_source = source_at_launch_step(source, step, launch_step)
                travel_turns, angle = cached_travel_turns(
                    launch_source,
                    target,
                    ships_to_send,
                    launch_step,
                )
                if not math.isfinite(travel_turns):
                    break
                arrival_step = launch_step + int(travel_turns)
                next_ships_to_send, projection_details = cached_target_fleet_floor(
                    target,
                    arrival_step,
                    horizon_step,
                )
                if next_ships_to_send <= 0:
                    ships_to_send = 0
                    break
                if next_ships_to_send == ships_to_send:
                    break
                ships_to_send = next_ships_to_send
            if timed_out:
                break
            if ships_to_send <= 0 or not math.isfinite(travel_turns):
                continue

            wait_turns = _wait_turns_to_leave_one(source, ships_to_send)
            launch_step = step + wait_turns
            if time.perf_counter() >= deadline:
                debug["time_budget_exhausted"] = True
                break
            source_max_safe_ships, source_projection_details = cached_source_safe_ships(
                source,
                launch_step,
                horizon_step,
            )
            cheap_production = (5.0 * float(target.production)) / max(1.0, float(ships_to_send))
            total_time = float(wait_turns) + travel_turns
            capture_step = step + int(total_time)
            owned_production = int(
                projection_details.get(
                    "owned_production",
                    max(0, int(horizon_step) - int(capture_step)) * int(target.production),
                )
            )
            net_value = float(owned_production) - float(ships_to_send)
            comparison = {
                "source_id": int(source.id),
                "target_id": int(target.id),
                "source_available_step": int(step),
                "owner": int(target.owner),
                "production": int(target.production),
                "ships_needed": int(ships_to_send),
                "wait_turns": int(wait_turns),
                "travel_turns": int(travel_turns),
                "total_time": float(total_time),
                "cheap_production": float(cheap_production),
                "net_value": float(net_value),
                "projected_owner": int(projection_details.get("projected_owner", target.owner)),
                "projected_ships": int(projection_details.get("projected_ships", target.ships)),
                "same_turn_friendly": int(projection_details.get("same_turn_friendly", 0)),
                "same_turn_enemy_max": int(projection_details.get("same_turn_enemy_max", 0)),
                "capture_ships_needed": int(projection_details.get("capture_ships_needed", ships_to_send)),
                "survival_extra_ships": int(projection_details.get("survival_extra_ships", 0)),
                "owned_production": int(owned_production),
                "survived_known_incoming": bool(projection_details.get("survived_known_incoming", True)),
                "lost_turn": projection_details.get("lost_turn"),
                "final_owner": int(projection_details.get("final_owner", target.owner)),
                "final_ships": int(projection_details.get("final_ships", target.ships)),
                "source_max_safe_ships": int(source_max_safe_ships),
                "source_survival_blocked": bool(ships_to_send > source_max_safe_ships),
                "source_survives_known_incoming": bool(source_projection_details.get("source_survives_known_incoming", True)),
                "source_lost_turn": source_projection_details.get("source_lost_turn"),
                "source_final_owner": int(source_projection_details.get("source_final_owner", source.owner)),
                "source_final_ships": int(source_projection_details.get("source_final_ships", source.ships)),
            }
            source_comparisons.append(comparison)
            if ships_to_send > source_max_safe_ships:
                continue
            matrix_row = dict(comparison)
            matrix_row["angle"] = float(angle)
            matrix_rows.append(matrix_row)

        if include_debug:
            debug["comparisons"].extend(source_comparisons)

    moves: list[list[Any]] = []
    source_by_id = {int(source.id): source for source in sources}
    selected_rows = _best_matrix_assignment(
        matrix_rows,
        [int(source.id) for source in sources],
    )
    for row in selected_rows:
        wait_turns = int(row["wait_turns"])
        travel_turns = int(row["travel_turns"])
        ships_to_send = int(row["ships_needed"])
        if wait_turns == 0:
            source = source_by_id.get(int(row["source_id"]))
            if source is not None and int(source.ships) >= ships_to_send + 1:
                moves.append([int(row["source_id"]), float(row["angle"]), ships_to_send])
        selected = {
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "ships": ships_to_send,
            "available_step": step + wait_turns + travel_turns,
            "kind": "launch" if wait_turns == 0 else "wait",
            "assignment_score": float(row["net_value"]),
            "source_available_step": int(row.get("source_available_step", step)),
        }
        if include_debug:
            debug["selected"].append(selected)

    return moves, debug


def choose_opening_moves(obs: Any) -> list[list[Any]]:
    moves, _debug = _choose_opening(obs)
    return moves


def opening_debug(obs: Any) -> dict[str, Any]:
    _moves, debug = _choose_opening(obs, include_debug=True)
    return debug


def agent(obs: Any, config: Any = None) -> list[list[Any]]:
    try:
        return choose_opening_moves(obs)
    except Exception:
        return []
