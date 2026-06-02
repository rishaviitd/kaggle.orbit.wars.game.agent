"""Cheap-production opening agent.

Each owned source repeats the same target rule:

1. Ignore the opponent quadrant.
2. Keep non-owned planets only, excluding planets already claimed by friendly fleets.
3. Rank by cheap production: ``(5 * production) / ships_needed``.
4. Keep the top 37%.
5. Choose the candidate with best net fleet contribution by the opening horizon.
6. Launch only when the source can send the needed fleet and leave 1 ship.
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Any

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

BOARD_SIZE = 100.0
CENTER = 50.0
BOARD_CENTER = CENTER
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
FIRST_CAPTURE_TOP_FRACTION = 0.37
MAX_COLLISION_TURN = 120
FUTURE_SOURCE_LOOKAHEAD = 90
OPENING_SCORE_TURNS = 30
OPENING_SCORE_CAP_STEP = 50


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
            margin = moving_blocker_margin(source, blocker, angle, ships, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
            if abs(margin) <= 1e-4:
                break
            derivative = moving_blocker_margin_derivative(source, blocker, angle, ships, hit_time, initial_by_id, angular_velocity, current_step, comet_ids)
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
        if moving_blocker_margin(source, blocker, angle, ships, hit_time, initial_by_id, angular_velocity, current_step, comet_ids) <= 0.05:
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
    for planet in planets:
        if planet.id in (source.id, target.id) or planet.id in comet_ids:
            continue
        motion_type = motion_types.get(planet.id, "unknown")
        if motion_type == "orbiting":
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


def _quadrant(planet: Planet) -> tuple[int, int]:
    return (0 if planet.x < BOARD_CENTER else 1, 0 if planet.y < BOARD_CENTER else 1)


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
    return min(OPENING_SCORE_CAP_STEP, int(step) + OPENING_SCORE_TURNS)


def _net_opening_value(target: Planet, ships_to_send: int, capture_step: int, horizon_step: int) -> float:
    producing_turns = max(0, int(horizon_step) - int(capture_step))
    return float(target.production) * producing_turns - float(ships_to_send)


def _opponent_quadrant(home_quadrant: tuple[int, int]) -> tuple[int, int]:
    home_qx, home_qy = home_quadrant
    return (1 - home_qx, 1 - home_qy)


def _home_quadrant(
    player_id: int,
    planets: list[Planet],
    initial_planets: list[Planet],
) -> tuple[int, int] | None:
    for planet in initial_planets:
        if planet.owner == player_id:
            return _quadrant(planet)
    for planet in planets:
        if planet.owner == player_id:
            return _quadrant(planet)
    return None


def _target_filter_reason(
    planet: Planet,
    player_id: int,
    comet_ids: set[int],
    future_sources_by_id: dict[int, int],
    blocked_quadrant: tuple[int, int],
) -> str | None:
    if planet.owner == player_id:
        return "owned"
    if planet.id in comet_ids:
        return "comet"
    if planet.id in future_sources_by_id:
        return f"future source@{future_sources_by_id[int(planet.id)]}"
    if _quadrant(planet) == blocked_quadrant:
        return "opponent quadrant"
    return None


def _first_capture_candidates(
    planets: list[Planet],
    player_id: int,
    comet_ids: set[int],
    future_sources_by_id: dict[int, int],
    blocked_quadrant: tuple[int, int],
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
    if not targets:
        return [], eliminated

    targets.sort(key=_base_cheap_production_score, reverse=True)
    keep_count = max(1, int(math.ceil(len(targets) * FIRST_CAPTURE_TOP_FRACTION)))
    cutoff_score = _base_cheap_production_score(targets[keep_count - 1])
    kept_ids = {
        int(planet.id)
        for planet in targets
        if _base_cheap_production_score(planet) >= cutoff_score - 1e-12
    }
    eliminated.extend(
        {"planet_id": int(planet.id), "reason": "outside top 37%"}
        for planet in targets[keep_count:]
        if int(planet.id) not in kept_ids
    )
    return [planet for planet in targets if int(planet.id) in kept_ids], eliminated


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


def _fleet_future_hit(
    fleet: Fleet,
    target_planets: list[Planet],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> tuple[int, int] | None:
    speed = fleet_speed(fleet.ships, MAX_SPEED)
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed
    best: tuple[int, int] | None = None

    old_x = float(fleet.x)
    old_y = float(fleet.y)
    for turn in range(1, FUTURE_SOURCE_LOOKAHEAD + 1):
        new_x = old_x + dx
        new_y = old_y + dy
        if not (0.0 <= new_x <= 100.0 and 0.0 <= new_y <= 100.0):
            break

        for planet in target_planets:
            target_x, target_y = planet_position_after_moves(
                planet,
                turn,
                initial_by_id,
                angular_velocity,
                step,
                comet_ids,
            )
            if _point_to_segment_distance((target_x, target_y), (old_x, old_y), (new_x, new_y)) < planet.radius:
                candidate = (turn, int(planet.id))
                if best is None or candidate < best:
                    best = candidate

        if best is not None and best[0] == turn:
            return best[1], step + best[0]

        old_x = new_x
        old_y = new_y

    return None


def _future_sources_by_id(
    fleets: list[Fleet],
    target_planets: list[Planet],
    player_id: int,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> dict[int, int]:
    if not fleets:
        return {}

    target_by_id = {int(planet.id): planet for planet in target_planets}
    incoming_by_target: dict[int, list[tuple[int, int]]] = {}
    for fleet in fleets:
        if int(fleet.owner) != player_id:
            continue
        future_hit = _fleet_future_hit(
            fleet,
            target_planets,
            initial_by_id,
            angular_velocity,
            step,
            comet_ids,
        )
        if future_hit is None:
            continue
        target_id, available_step = future_hit
        incoming_by_target.setdefault(target_id, []).append((available_step, int(fleet.ships)))

    future_sources: dict[int, int] = {}
    for target_id, incoming in incoming_by_target.items():
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


def _choose_opening(obs: Any, include_debug: bool = False) -> tuple[list[list[Any]], dict[str, Any]]:
    player_id = int(get(obs, "player", 0))
    step = int(get(obs, "step", 0))
    raw_planets = get(obs, "planets", [])
    raw_initial = get(obs, "initial_planets", raw_planets)
    angular_velocity = float(get(obs, "angular_velocity", 0.0))
    comet_ids = set(get(obs, "comet_planet_ids", []))

    planets = [Planet(*row) for row in raw_planets]
    initial_planets = [Planet(*row) for row in raw_initial]
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
    }
    if not my_planets:
        return [], debug

    home_quadrant = _home_quadrant(player_id, planets, initial_planets)
    if home_quadrant is None:
        return [], debug
    blocked_quadrant = _opponent_quadrant(home_quadrant)
    debug["blocked_quadrant"] = list(blocked_quadrant)

    initial_by_id = {row[0]: row for row in raw_initial}
    motion_types = planet_motion_types(planets, initial_by_id, comet_ids)
    target_planets = [planet for planet in planets if planet.owner != player_id and planet.id not in comet_ids]
    future_sources_by_id = _future_sources_by_id(
        fleets,
        target_planets,
        player_id,
        initial_by_id,
        angular_velocity,
        step,
        comet_ids,
    )
    debug["future_sources_by_id"] = {str(key): value for key, value in future_sources_by_id.items()}

    moves: list[list[Any]] = []
    sources = sorted(my_planets, key=lambda planet: (planet.ships, planet.production), reverse=True)
    for source in sources:
        candidates, eliminated = _first_capture_candidates(
            planets,
            player_id,
            comet_ids,
            future_sources_by_id,
            blocked_quadrant,
        )
        if include_debug:
            debug["eliminated"].extend(
                {"source_id": int(source.id), **item}
                for item in eliminated
            )
        if not candidates:
            continue

        source_comparisons: list[dict[str, Any]] = []
        best: tuple[float, float, int, float, float, Planet, float | None, int] | None = None
        for target in candidates:
            ships_to_send = _base_fleet_floor(target)
            wait_turns = 0
            travel_turns = float("inf")
            angle = None
            for _ in range(4):
                wait_turns = _wait_turns_to_leave_one(source, ships_to_send)
                travel_turns, angle = _travel_turns(
                    source,
                    target,
                    ships_to_send,
                    planets,
                    initial_by_id,
                    motion_types,
                    angular_velocity,
                    step + wait_turns,
                    comet_ids,
                )
                if not math.isfinite(travel_turns):
                    break
                arrival_step = step + wait_turns + int(travel_turns)
                next_ships_to_send = _target_fleet_floor(target, player_id, arrival_step, step)
                if next_ships_to_send == ships_to_send:
                    break
                ships_to_send = next_ships_to_send
            if not math.isfinite(travel_turns):
                continue

            cheap_production = (5.0 * float(target.production)) / max(1.0, float(ships_to_send))
            total_time = float(wait_turns) + travel_turns
            capture_step = step + int(total_time)
            net_value = _net_opening_value(target, ships_to_send, capture_step, _opening_horizon(step))
            comparison = {
                "source_id": int(source.id),
                "target_id": int(target.id),
                "owner": int(target.owner),
                "production": int(target.production),
                "ships_needed": int(ships_to_send),
                "wait_turns": int(wait_turns),
                "travel_turns": int(travel_turns),
                "total_time": float(total_time),
                "cheap_production": float(cheap_production),
                "net_value": float(net_value),
            }
            source_comparisons.append(comparison)
            candidate = (net_value, -total_time, -wait_turns, -travel_turns, cheap_production, target, angle, ships_to_send)
            if best is None or candidate[:5] > best[:5]:
                best = candidate

        if include_debug:
            debug["comparisons"].extend(source_comparisons)
        if best is None:
            continue

        _net_value, _negative_total_time, negative_wait_turns, negative_travel_turns, _cheap_production, target, angle, ships_to_send = best
        wait_turns = -negative_wait_turns
        _travel_turns_value = -negative_travel_turns
        if wait_turns > 0:
            continue

        if angle is None or int(source.ships) < ships_to_send + 1:
            continue

        moves.append([int(source.id), float(angle), int(ships_to_send)])
        future_sources_by_id[int(target.id)] = step + int(_travel_turns_value)
        selected = {
            "source_id": int(source.id),
            "target_id": int(target.id),
            "ships": int(ships_to_send),
            "available_step": step + int(_travel_turns_value),
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
