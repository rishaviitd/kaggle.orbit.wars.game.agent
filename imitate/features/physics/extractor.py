from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from imitate.features.physics.numba_kernels import (
    MOTION_COMET,
    MOTION_ORBITING,
    MOTION_STATIC,
    MOTION_UNKNOWN,
    is_numba_available,
    predict_fleet_hits_kernel,
)

BOARD_SIZE = 100.0
BOARD_CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_FLEET_SPEED = 6.0
DEFAULT_COLLISION_LOOKAHEAD = 120
COLLISION_FILTER_EPSILON = 1e-7
USE_NUMBA_COLLISION_KERNEL = is_numba_available()

OPPONENT_SLOTS = (1, 2, 3)
PLAYER_HOME_QUADRANTS_4P = {
    0: (1, 1),
    1: (0, 1),
    2: (1, 0),
    3: (0, 0),
}
PLAYER_HOME_QUADRANTS_2P = {
    0: (1, 1),
    1: (0, 0),
}

PLANET_PHYSICS_FEATURES = (
    "incoming_friendly_fleet_count",
    "incoming_friendly_ships_total",
    "incoming_enemy_fleet_count",
    "incoming_enemy_ships_total",
    *(f"incoming_enemy_fleet_count_opponent_{slot}" for slot in OPPONENT_SLOTS),
    *(f"incoming_enemy_ships_opponent_{slot}" for slot in OPPONENT_SLOTS),
    "incoming_enemy_player_count",
    "incoming_friendly_first_arrival_turn",
    "incoming_enemy_first_arrival_turn",
    "incoming_friendly_last_arrival_turn",
    "incoming_enemy_last_arrival_turn",
    "incoming_net_ship_balance",
)

FLEET_PHYSICS_FEATURES = (
    "fleet_has_predicted_hit",
    "fleet_predicted_collision_turns",
    "fleet_predicted_collision_distance",
    "fleet_target_owner_is_friendly",
    "fleet_target_owner_is_neutral",
    "fleet_target_owner_is_hostile",
    "fleet_target_ship_count",
    "fleet_target_production",
    "fleet_target_is_source",
)

PLANET_PAIR_PHYSICS_FEATURES = (
    "planet_pair_target_incoming_friendly_ships",
    "planet_pair_target_incoming_enemy_ships",
)

FLEET_PLANET_PHYSICS_FEATURES = (
    "fleet_planet_is_predicted_destination",
    "fleet_planet_collision_eta",
    "fleet_planet_arrives_before_friendly",
    "fleet_planet_arrives_before_enemy",
    "fleet_planet_same_turn_friendly_ships",
    "fleet_planet_same_turn_enemy_ships",
)


class PlanetState(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


class FleetState(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    angle: float
    source_planet_id: int
    ships: int


class CollisionPrediction(NamedTuple):
    fleet_index: int
    fleet_id: int
    owner: int
    ships: int
    target_index: int
    target_id: int
    turn: int
    distance: float


CollisionCacheEntry = tuple[tuple[int, int, int, float], int, int, float]
CollisionCache = dict[Any, Any]
_CACHE_BOARD_SIGNATURE_KEY = "__board_signature__"


def _fleet_speed(ships: int, max_speed: float = MAX_FLEET_SPEED) -> float:
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    return min(1.0 + (max_speed - 1.0) * (ratio**1.5), max_speed)


def _fleet_cache_fingerprint(fleet: FleetState) -> tuple[int, int, int, float]:
    return (
        int(fleet.owner),
        int(fleet.source_planet_id),
        int(fleet.ships),
        round(float(fleet.angle), 12),
    )


def _prediction_from_cache(
    fleet_index: int,
    fleet: FleetState,
    *,
    step: int,
    lookahead: int,
    planet_index_by_id: dict[int, int],
    static_target_by_id: dict[int, PlanetState],
    collision_cache: CollisionCache,
    cache_stats: dict[str, int],
) -> CollisionPrediction | None:
    cached = collision_cache.get(int(fleet.id))
    if cached is None:
        return None

    fingerprint, target_id, absolute_hit_step, _absolute_collision_time = cached
    if fingerprint != _fleet_cache_fingerprint(fleet):
        del collision_cache[int(fleet.id)]
        cache_stats["stale"] += 1
        return None

    target = static_target_by_id.get(int(target_id))
    if (
        int(absolute_hit_step) <= int(step)
        or int(target_id) not in planet_index_by_id
        or target is None
    ):
        del collision_cache[int(fleet.id)]
        cache_stats["stale"] += 1
        return None

    remaining_turns = int(absolute_hit_step) - int(step)
    if remaining_turns > int(lookahead):
        cache_stats["beyond_lookahead"] += 1
        return None

    collision_distance = _static_target_collision_distance(
        fleet,
        target,
        remaining_turns,
    )
    if collision_distance is None:
        del collision_cache[int(fleet.id)]
        cache_stats["stale"] += 1
        return None

    cache_stats["reused"] += 1
    return CollisionPrediction(
        fleet_index=fleet_index,
        fleet_id=fleet.id,
        owner=fleet.owner,
        ships=fleet.ships,
        target_index=planet_index_by_id[int(target_id)],
        target_id=int(target_id),
        turn=remaining_turns,
        distance=collision_distance,
    )


def _static_target_collision_distance(
    fleet: FleetState,
    target: PlanetState,
    hit_turn: int,
) -> float | None:
    speed = _fleet_speed(fleet.ships, MAX_FLEET_SPEED)
    velocity_x = math.cos(fleet.angle) * speed
    velocity_y = math.sin(fleet.angle) * speed
    fleet_start = (
        fleet.x + velocity_x * (int(hit_turn) - 1),
        fleet.y + velocity_y * (int(hit_turn) - 1),
    )
    fleet_end = (
        fleet_start[0] + velocity_x,
        fleet_start[1] + velocity_y,
    )
    target_point = (target.x, target.y)
    if _point_to_segment_distance(target_point, fleet_start, fleet_end) >= target.radius:
        return None

    progress = _point_to_segment_progress(target_point, fleet_start, fleet_end)
    return float(speed * (float(hit_turn) - 1.0 + progress))


def _store_prediction_in_cache(
    prediction: CollisionPrediction,
    fleet: FleetState,
    *,
    step: int,
    static_target_ids: set[int],
    collision_cache: CollisionCache,
    cache_stats: dict[str, int],
) -> None:
    if int(prediction.target_id) not in static_target_ids:
        cache_stats["dynamic_target_skipped"] += 1
        return

    speed = _fleet_speed(fleet.ships, MAX_FLEET_SPEED)
    if speed <= 0.0:
        return

    absolute_collision_time = float(step) + float(prediction.distance) / speed
    collision_cache[int(fleet.id)] = (
        _fleet_cache_fingerprint(fleet),
        int(prediction.target_id),
        int(step) + int(prediction.turn),
        absolute_collision_time,
    )
    cache_stats["stored"] += 1


def _rotate_left(quadrant: tuple[int, int]) -> tuple[int, int]:
    x, y = quadrant
    return (1 - y, x)


def _rotate_right(quadrant: tuple[int, int]) -> tuple[int, int]:
    x, y = quadrant
    return (y, 1 - x)


def _opposite(quadrant: tuple[int, int]) -> tuple[int, int]:
    return (1 - quadrant[0], 1 - quadrant[1])


def _opponent_player_ids(
    player_id: int,
    player_count: int,
) -> tuple[int | None, int | None, int | None]:
    if player_count == 2:
        opponent = 1 - player_id if player_id in PLAYER_HOME_QUADRANTS_2P else None
        return None, opponent, None

    home = PLAYER_HOME_QUADRANTS_4P.get(player_id)
    if home is None:
        return None, None, None
    player_by_home = {
        quadrant: owner
        for owner, quadrant in PLAYER_HOME_QUADRANTS_4P.items()
    }
    return (
        player_by_home.get(_rotate_left(home)),
        player_by_home.get(_opposite(home)),
        player_by_home.get(_rotate_right(home)),
    )


def _planet_motion_types(
    planets: list[PlanetState],
    initial_by_id: dict[int, np.ndarray],
    comet_ids: set[int],
) -> dict[int, str]:
    motion_types: dict[int, str] = {}
    for planet in planets:
        if planet.id in comet_ids:
            motion_types[planet.id] = "comet"
            continue

        initial = initial_by_id.get(planet.id)
        if initial is None:
            motion_types[planet.id] = "unknown"
            continue

        orbital_radius = math.hypot(
            float(initial[2]) - BOARD_CENTER,
            float(initial[3]) - BOARD_CENTER,
        )
        if orbital_radius + float(initial[4]) < ROTATION_RADIUS_LIMIT - 1e-9:
            motion_types[planet.id] = "orbiting"
        else:
            motion_types[planet.id] = "static"
    return motion_types


def _comet_paths_by_planet(
    comets: list[dict[str, Any]],
) -> dict[int, tuple[int, list[Any]]]:
    paths_by_planet: dict[int, tuple[int, list[Any]]] = {}
    for comet in comets:
        path_index = int(comet.get("path_index", 0) or 0)
        planet_ids = comet.get("planet_ids") or []
        paths = comet.get("paths") or []
        for planet_id, path in zip(planet_ids, paths, strict=False):
            if path is None:
                continue
            paths_by_planet[int(planet_id)] = (path_index, path)
    return paths_by_planet


def _comet_position_after_moves(
    planet_id: int,
    moves_done: int,
    comet_paths: dict[int, tuple[int, list[Any]]],
) -> tuple[float, float] | None:
    comet_path = comet_paths.get(int(planet_id))
    if comet_path is None:
        return None

    path_index, path = comet_path
    future_index = int(path_index) + max(0, int(moves_done))
    if future_index < 0 or future_index >= len(path):
        return None

    point = path[future_index]
    if point is None or len(point) < 2:
        return None
    return float(point[0]), float(point[1])


def _planet_position_after_moves(
    planet: PlanetState,
    moves_done: int,
    initial_by_id: dict[int, np.ndarray],
    angular_velocity: float,
    current_step: int,
    comet_ids: set[int],
    comet_paths: dict[int, tuple[int, list[Any]]],
) -> tuple[float, float] | None:
    moves_done = int(moves_done)
    if moves_done <= 0:
        return planet.x, planet.y

    if planet.id in comet_ids:
        return _comet_position_after_moves(planet.id, moves_done, comet_paths)

    initial = initial_by_id.get(planet.id)
    if initial is None:
        return planet.x, planet.y

    orbital_radius = math.hypot(
        float(initial[2]) - BOARD_CENTER,
        float(initial[3]) - BOARD_CENTER,
    )
    if orbital_radius + float(initial[4]) >= ROTATION_RADIUS_LIMIT:
        return planet.x, planet.y

    initial_angle = math.atan2(
        float(initial[3]) - BOARD_CENTER,
        float(initial[2]) - BOARD_CENTER,
    )
    env_step = max(1, int(current_step)) + moves_done - 1
    future_angle = initial_angle + float(angular_velocity) * env_step
    return (
        BOARD_CENTER + orbital_radius * math.cos(future_angle),
        BOARD_CENTER + orbital_radius * math.sin(future_angle),
    )


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    length_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if length_sq == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])

    progress = (
        (point[0] - start[0]) * (end[0] - start[0])
        + (point[1] - start[1]) * (end[1] - start[1])
    ) / length_sq
    progress = max(0.0, min(1.0, progress))
    projection = (
        start[0] + progress * (end[0] - start[0]),
        start[1] + progress * (end[1] - start[1]),
    )
    return math.hypot(point[0] - projection[0], point[1] - projection[1])


def _point_to_segment_progress(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    length_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if length_sq == 0.0:
        return 0.0
    progress = (
        (point[0] - start[0]) * (end[0] - start[0])
        + (point[1] - start[1]) * (end[1] - start[1])
    ) / length_sq
    return max(0.0, min(1.0, progress))


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
    progress_a = _cross_2d(qp, s) / denominator
    progress_b = _cross_2d(qp, r) / denominator
    if -1e-9 <= progress_a <= 1.0 + 1e-9 and -1e-9 <= progress_b <= 1.0 + 1e-9:
        return max(0.0, min(1.0, progress_a))
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
    intersection_progress = _segment_intersection_progress(
        fleet_start,
        fleet_end,
        target_start,
        target_end,
    )
    if intersection_progress is not None:
        return intersection_progress
    return min(
        _point_to_segment_progress(target_start, fleet_start, fleet_end),
        _point_to_segment_progress(target_end, fleet_start, fleet_end),
    )


def _board_exit_time(
    start_x: float,
    start_y: float,
    direction_x: float,
    direction_y: float,
    speed: float,
) -> float | None:
    exit_times = []
    velocity_x = direction_x * speed
    velocity_y = direction_y * speed

    if velocity_x > 0.0:
        exit_times.append((BOARD_SIZE - start_x) / velocity_x)
    elif velocity_x < 0.0:
        exit_times.append((0.0 - start_x) / velocity_x)

    if velocity_y > 0.0:
        exit_times.append((BOARD_SIZE - start_y) / velocity_y)
    elif velocity_y < 0.0:
        exit_times.append((0.0 - start_y) / velocity_y)

    positive_times = [time_value for time_value in exit_times if time_value >= 0.0]
    return min(positive_times) if positive_times else None


def _ray_circle_interval(
    start_x: float,
    start_y: float,
    velocity_x: float,
    velocity_y: float,
    center_x: float,
    center_y: float,
    radius: float,
    max_time: float,
) -> tuple[float, float] | None:
    relative_x = start_x - center_x
    relative_y = start_y - center_y
    a = velocity_x * velocity_x + velocity_y * velocity_y
    if a <= 0.0:
        return None

    b = 2.0 * (relative_x * velocity_x + relative_y * velocity_y)
    c = relative_x * relative_x + relative_y * relative_y - radius * radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < -COLLISION_FILTER_EPSILON:
        return None

    sqrt_discriminant = math.sqrt(max(0.0, discriminant))
    first = (-b - sqrt_discriminant) / (2.0 * a)
    second = (-b + sqrt_discriminant) / (2.0 * a)
    interval_start = max(0.0, min(first, second))
    interval_end = min(float(max_time), max(first, second))
    if interval_end + COLLISION_FILTER_EPSILON < interval_start:
        return None
    return interval_start, interval_end


def _candidate_turn_bounds(
    interval_start: float,
    interval_end: float,
    horizon: int,
    padding: int = 1,
) -> tuple[int, int] | None:
    first_turn = max(1, int(math.floor(interval_start)) - int(padding))
    last_turn = min(int(horizon), int(math.ceil(interval_end)) + int(padding))
    if last_turn < first_turn:
        return None
    return first_turn, last_turn


def _fleet_search_horizon(
    fleet: FleetState,
    velocity_x: float,
    velocity_y: float,
    lookahead: int,
) -> int:
    speed = math.hypot(velocity_x, velocity_y)
    if speed <= 0.0:
        return 0

    direction_x = velocity_x / speed
    direction_y = velocity_y / speed
    exit_time = _board_exit_time(
        fleet.x,
        fleet.y,
        direction_x,
        direction_y,
        speed,
    )
    if exit_time is None:
        horizon = int(lookahead)
    else:
        horizon = min(
            int(lookahead),
            max(0, int(math.floor(float(exit_time) + COLLISION_FILTER_EPSILON))),
        )

    def endpoint_is_inside(turn: int) -> bool:
        x = fleet.x + velocity_x * int(turn)
        y = fleet.y + velocity_y * int(turn)
        return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE

    while horizon > 0 and not endpoint_is_inside(horizon):
        horizon -= 1
    while horizon < int(lookahead) and endpoint_is_inside(horizon + 1):
        horizon += 1
    return horizon


def _build_collision_filter_context(
    planets: list[PlanetState],
    initial_by_id: dict[int, np.ndarray],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
    comet_paths: dict[int, tuple[int, list[Any]]],
) -> dict[str, Any]:
    motion_types = _planet_motion_types(planets, initial_by_id, comet_ids)
    orbit_metadata: dict[int, tuple[float, float]] = {}
    for planet in planets:
        if motion_types.get(planet.id) != "orbiting":
            continue
        initial = initial_by_id.get(planet.id)
        if initial is None:
            continue
        orbital_radius = math.hypot(
            float(initial[2]) - BOARD_CENTER,
            float(initial[3]) - BOARD_CENTER,
        )
        orbit_metadata[planet.id] = (orbital_radius, planet.radius)

    return {
        "initial_by_id": initial_by_id,
        "angular_velocity": float(angular_velocity),
        "step": int(step),
        "comet_ids": comet_ids,
        "comet_paths": comet_paths,
        "motion_types": motion_types,
        "orbit_metadata": orbit_metadata,
        "planet_segment_cache": {},
    }


def _planet_segment_for_turn(
    planet: PlanetState,
    turn: int,
    filter_context: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    cache_key = (planet.id, int(turn))
    segment_cache = filter_context["planet_segment_cache"]
    cached = segment_cache.get(cache_key)
    if cached is not None:
        return cached

    old_position = _planet_position_after_moves(
        planet,
        max(0, int(turn) - 1),
        filter_context["initial_by_id"],
        filter_context["angular_velocity"],
        filter_context["step"],
        filter_context["comet_ids"],
        filter_context["comet_paths"],
    )
    new_position = _planet_position_after_moves(
        planet,
        int(turn),
        filter_context["initial_by_id"],
        filter_context["angular_velocity"],
        filter_context["step"],
        filter_context["comet_ids"],
        filter_context["comet_paths"],
    )
    if old_position is None or new_position is None:
        return None

    segment = (old_position, new_position)
    segment_cache[cache_key] = segment
    return segment


def _schedule_planet_turns(
    schedule: dict[int, list[PlanetState]],
    planet: PlanetState,
    first_turn: int,
    last_turn: int,
) -> None:
    for turn in range(int(first_turn), int(last_turn) + 1):
        schedule.setdefault(turn, []).append(planet)


def _schedule_candidate_turns(
    schedule: dict[int, list[PlanetState]],
    planets: list[PlanetState],
    filter_context: dict[str, Any],
    fleet: FleetState,
    velocity_x: float,
    velocity_y: float,
    horizon: int,
) -> None:
    start_x = fleet.x
    start_y = fleet.y
    motion_types = filter_context["motion_types"]
    orbit_metadata = filter_context["orbit_metadata"]
    angular_velocity = filter_context["angular_velocity"]

    for planet in planets:
        motion_type = motion_types.get(planet.id, "unknown")
        if motion_type == "static":
            interval = _ray_circle_interval(
                start_x,
                start_y,
                velocity_x,
                velocity_y,
                planet.x,
                planet.y,
                planet.radius + COLLISION_FILTER_EPSILON,
                horizon,
            )
            if interval is None:
                continue
            turn_bounds = _candidate_turn_bounds(*interval, horizon, padding=1)
            if turn_bounds is not None:
                _schedule_planet_turns(schedule, planet, *turn_bounds)
            continue

        if motion_type == "orbiting" and planet.id in orbit_metadata:
            orbital_radius, planet_radius = orbit_metadata[planet.id]
            angular_step = min(math.pi, abs(float(angular_velocity)))
            outer_radius = orbital_radius + planet_radius + COLLISION_FILTER_EPSILON
            inner_radius = max(
                0.0,
                orbital_radius * math.cos(angular_step / 2.0)
                - planet_radius
                - COLLISION_FILTER_EPSILON,
            )
            outer_interval = _ray_circle_interval(
                start_x,
                start_y,
                velocity_x,
                velocity_y,
                BOARD_CENTER,
                BOARD_CENTER,
                outer_radius,
                horizon,
            )
            if outer_interval is None:
                continue

            inner_interval = None
            if inner_radius > COLLISION_FILTER_EPSILON:
                inner_interval = _ray_circle_interval(
                    start_x,
                    start_y,
                    velocity_x,
                    velocity_y,
                    BOARD_CENTER,
                    BOARD_CENTER,
                    inner_radius,
                    horizon,
                )
            turn_bounds = _candidate_turn_bounds(*outer_interval, horizon, padding=1)
            if turn_bounds is None:
                continue

            first_turn, last_turn = turn_bounds
            for turn in range(first_turn, last_turn + 1):
                if (
                    inner_interval is not None
                    and float(turn - 1) >= inner_interval[0] + COLLISION_FILTER_EPSILON
                    and float(turn) <= inner_interval[1] - COLLISION_FILTER_EPSILON
                ):
                    continue
                schedule.setdefault(turn, []).append(planet)
            continue

        _schedule_planet_turns(schedule, planet, 1, horizon)


def _fleet_future_hit(
    fleet_index: int,
    fleet: FleetState,
    planets: list[PlanetState],
    planet_index_by_id: dict[int, int],
    filter_context: dict[str, Any],
    lookahead: int,
) -> CollisionPrediction | None:
    speed = _fleet_speed(fleet.ships, MAX_FLEET_SPEED)
    velocity_x = math.cos(fleet.angle) * speed
    velocity_y = math.sin(fleet.angle) * speed
    horizon = _fleet_search_horizon(fleet, velocity_x, velocity_y, lookahead)
    if horizon <= 0:
        return None

    schedule: dict[int, list[PlanetState]] = {}
    _schedule_candidate_turns(
        schedule,
        planets,
        filter_context,
        fleet,
        velocity_x,
        velocity_y,
        horizon,
    )
    if not schedule:
        return None

    fleet_positions = [(fleet.x, fleet.y)]
    for _turn in range(1, int(horizon) + 1):
        old_x, old_y = fleet_positions[-1]
        fleet_positions.append((old_x + velocity_x, old_y + velocity_y))

    for turn in sorted(schedule):
        fleet_start = fleet_positions[turn - 1]
        fleet_end = fleet_positions[turn]
        best_hit: tuple[float, PlanetState] | None = None
        for planet in schedule[turn]:
            target_segment = _planet_segment_for_turn(
                planet,
                turn,
                filter_context,
            )
            if target_segment is None:
                continue
            target_start, target_end = target_segment
            hit = (
                _segment_to_segment_distance(
                    fleet_start,
                    fleet_end,
                    target_start,
                    target_end,
                )
                < planet.radius
            )
            if not hit:
                continue

            progress = _segment_collision_progress(
                fleet_start,
                fleet_end,
                target_start,
                target_end,
            )
            if best_hit is None or progress < best_hit[0]:
                best_hit = (progress, planet)

        if best_hit is not None:
            progress, target = best_hit
            collision_distance = speed * (float(turn) - 1.0 + float(progress))
            return CollisionPrediction(
                fleet_index=fleet_index,
                fleet_id=fleet.id,
                owner=fleet.owner,
                ships=fleet.ships,
                target_index=planet_index_by_id[target.id],
                target_id=target.id,
                turn=int(turn),
                distance=float(collision_distance),
            )
    return None


def _build_planets(direct: dict[str, Any]) -> list[PlanetState]:
    context = direct["context"]
    metadata = direct["metadata"]
    return [
        PlanetState(
            id=int(planet_id),
            owner=int(owner),
            x=float(position[0]),
            y=float(position[1]),
            radius=float(radius),
            ships=int(ships),
            production=int(production),
        )
        for planet_id, owner, position, radius, ships, production in zip(
            metadata["planet_ids"],
            context["planet_owners"],
            context["planet_positions"],
            context["planet_radii"],
            context["planet_ships"],
            context["planet_production"],
            strict=True,
        )
    ]


def _build_fleets(direct: dict[str, Any]) -> list[FleetState]:
    context = direct["context"]
    metadata = direct["metadata"]
    fleet_values = np.asarray(direct["values"]["fleet"])
    return [
        FleetState(
            id=int(fleet_id),
            owner=int(owner),
            x=float(values[0]),
            y=float(values[1]),
            angle=float(angle),
            source_planet_id=int(source_id),
            ships=int(ships),
        )
        for fleet_id, owner, values, angle, source_id, ships in zip(
            metadata["fleet_ids"],
            context["fleet_owners"],
            fleet_values,
            context["fleet_angles"],
            metadata["fleet_source_planet_ids"],
            context["fleet_ships"],
            strict=True,
        )
    ]


def _comet_position_tables(
    planets: list[PlanetState],
    comet_paths: dict[int, tuple[int, list[Any]]],
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray]:
    planet_count = len(planets)
    table_width = max(1, int(lookahead) + 1)
    comet_x = np.full((planet_count, table_width), np.nan, dtype=np.float64)
    comet_y = np.full((planet_count, table_width), np.nan, dtype=np.float64)

    for planet_index, planet in enumerate(planets):
        comet_path = comet_paths.get(int(planet.id))
        if comet_path is None:
            continue
        path_index, path = comet_path
        for moves_done in range(1, table_width):
            future_index = int(path_index) + moves_done
            if future_index < 0 or future_index >= len(path):
                continue
            point = path[future_index]
            if point is None or len(point) < 2:
                continue
            comet_x[planet_index, moves_done] = float(point[0])
            comet_y[planet_index, moves_done] = float(point[1])

    return comet_x, comet_y


def _numeric_collision_inputs(
    planets: list[PlanetState],
    fleets: list[FleetState],
    filter_context: dict[str, Any],
    *,
    lookahead: int,
) -> dict[str, np.ndarray]:
    motion_types_by_id = filter_context["motion_types"]
    initial_by_id = filter_context["initial_by_id"]
    comet_paths = filter_context["comet_paths"]

    planet_count = len(planets)
    planet_ids = np.empty(planet_count, dtype=np.int32)
    planet_x = np.empty(planet_count, dtype=np.float64)
    planet_y = np.empty(planet_count, dtype=np.float64)
    planet_radius = np.empty(planet_count, dtype=np.float64)
    motion_types = np.empty(planet_count, dtype=np.int8)
    orbit_radius = np.zeros(planet_count, dtype=np.float64)
    orbit_initial_angle = np.zeros(planet_count, dtype=np.float64)

    for planet_index, planet in enumerate(planets):
        planet_ids[planet_index] = int(planet.id)
        planet_x[planet_index] = float(planet.x)
        planet_y[planet_index] = float(planet.y)
        planet_radius[planet_index] = float(planet.radius)

        motion_type = motion_types_by_id.get(int(planet.id), "unknown")
        if motion_type == "static":
            motion_types[planet_index] = MOTION_STATIC
        elif motion_type == "orbiting":
            motion_types[planet_index] = MOTION_ORBITING
            initial = initial_by_id.get(int(planet.id))
            if initial is not None:
                initial_x = float(initial[2])
                initial_y = float(initial[3])
                orbit_radius[planet_index] = math.hypot(
                    initial_x - BOARD_CENTER,
                    initial_y - BOARD_CENTER,
                )
                orbit_initial_angle[planet_index] = math.atan2(
                    initial_y - BOARD_CENTER,
                    initial_x - BOARD_CENTER,
                )
        elif motion_type == "comet":
            motion_types[planet_index] = MOTION_COMET
        else:
            motion_types[planet_index] = MOTION_UNKNOWN

    fleet_count = len(fleets)
    fleet_x = np.empty(fleet_count, dtype=np.float64)
    fleet_y = np.empty(fleet_count, dtype=np.float64)
    fleet_angles = np.empty(fleet_count, dtype=np.float64)
    fleet_ships = np.empty(fleet_count, dtype=np.float64)
    for fleet_index, fleet in enumerate(fleets):
        fleet_x[fleet_index] = float(fleet.x)
        fleet_y[fleet_index] = float(fleet.y)
        fleet_angles[fleet_index] = float(fleet.angle)
        fleet_ships[fleet_index] = float(fleet.ships)

    comet_x, comet_y = _comet_position_tables(
        planets,
        comet_paths,
        lookahead,
    )

    return {
        "planet_ids": planet_ids,
        "planet_x": planet_x,
        "planet_y": planet_y,
        "planet_radius": planet_radius,
        "motion_types": motion_types,
        "orbit_radius": orbit_radius,
        "orbit_initial_angle": orbit_initial_angle,
        "comet_x": comet_x,
        "comet_y": comet_y,
        "fleet_x": fleet_x,
        "fleet_y": fleet_y,
        "fleet_angles": fleet_angles,
        "fleet_ships": fleet_ships,
    }


def _predict_fleet_hits(
    direct: dict[str, Any],
    *,
    lookahead: int,
    collision_cache: CollisionCache | None = None,
) -> tuple[
    list[CollisionPrediction],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    planets = _build_planets(direct)
    fleets = _build_fleets(direct)
    target_ids = np.full(len(fleets), -1, dtype=np.int32)
    target_indices = np.full(len(fleets), -1, dtype=np.int32)
    hit_turns = np.zeros(len(fleets), dtype=np.int32)
    cache_stats = {
        "reused": 0,
        "computed": 0,
        "stored": 0,
        "stale": 0,
        "beyond_lookahead": 0,
        "board_resets": 0,
        "dynamic_target_skipped": 0,
    }
    if not planets or not fleets:
        return [], target_ids, target_indices, hit_turns, cache_stats

    context = direct["context"]
    metadata = direct["metadata"]
    step = int(metadata["step"])
    initial_planets = np.asarray(context["initial_planets"])
    initial_by_id = {
        int(row[0]): row
        for row in initial_planets
    }
    comet_ids = {
        int(planet_id)
        for planet_id in np.asarray(context["comet_planet_ids"], dtype=np.int32)
    }
    comet_paths = _comet_paths_by_planet(context["comets"])
    filter_context = _build_collision_filter_context(
        planets,
        initial_by_id,
        float(direct["values"]["global"][2]),
        step,
        comet_ids,
        comet_paths,
    )
    planet_index_by_id = {planet.id: index for index, planet in enumerate(planets)}
    static_target_ids = {
        planet_id
        for planet_id, motion_type in filter_context["motion_types"].items()
        if motion_type == "static"
    }
    static_target_by_id = {
        planet.id: planet
        for planet in planets
        if planet.id in static_target_ids
    }
    if collision_cache is not None:
        board_signature = (
            tuple(int(planet_id) for planet_id in direct["metadata"]["planet_ids"]),
            tuple(sorted(int(planet_id) for planet_id in comet_ids)),
        )
        if collision_cache.get(_CACHE_BOARD_SIGNATURE_KEY) != board_signature:
            collision_cache.clear()
            collision_cache[_CACHE_BOARD_SIGNATURE_KEY] = board_signature
            cache_stats["board_resets"] += 1

    prediction_by_fleet: list[CollisionPrediction | None] = [None] * len(fleets)
    needs_compute = np.ones(len(fleets), dtype=np.bool_)
    for fleet_index, fleet in enumerate(fleets):
        if collision_cache is not None:
            prediction = _prediction_from_cache(
                fleet_index,
                fleet,
                step=step,
                lookahead=lookahead,
                planet_index_by_id=planet_index_by_id,
                static_target_by_id=static_target_by_id,
                collision_cache=collision_cache,
                cache_stats=cache_stats,
            )
            if prediction is not None:
                prediction_by_fleet[fleet_index] = prediction
                needs_compute[fleet_index] = False
                target_ids[fleet_index] = prediction.target_id
                target_indices[fleet_index] = prediction.target_index
                hit_turns[fleet_index] = prediction.turn

    if np.any(needs_compute) and USE_NUMBA_COLLISION_KERNEL:
        cache_stats["computed"] += int(np.count_nonzero(needs_compute))
        numeric_inputs = _numeric_collision_inputs(
            planets,
            fleets,
            filter_context,
            lookahead=lookahead,
        )
        (
            computed_target_indices,
            computed_hit_turns,
            computed_hit_distances,
        ) = predict_fleet_hits_kernel(
            numeric_inputs["fleet_x"],
            numeric_inputs["fleet_y"],
            numeric_inputs["fleet_angles"],
            numeric_inputs["fleet_ships"],
            numeric_inputs["planet_x"],
            numeric_inputs["planet_y"],
            numeric_inputs["planet_radius"],
            numeric_inputs["motion_types"],
            numeric_inputs["orbit_radius"],
            numeric_inputs["orbit_initial_angle"],
            numeric_inputs["comet_x"],
            numeric_inputs["comet_y"],
            float(direct["values"]["global"][2]),
            step,
            int(lookahead),
            needs_compute,
        )

        for fleet_index, target_index in enumerate(computed_target_indices):
            if not needs_compute[fleet_index] or int(target_index) < 0:
                continue
            fleet = fleets[fleet_index]
            prediction = CollisionPrediction(
                fleet_index=fleet_index,
                fleet_id=fleet.id,
                owner=fleet.owner,
                ships=fleet.ships,
                target_index=int(target_index),
                target_id=int(numeric_inputs["planet_ids"][int(target_index)]),
                turn=int(computed_hit_turns[fleet_index]),
                distance=float(computed_hit_distances[fleet_index]),
            )
            prediction_by_fleet[fleet_index] = prediction
            target_ids[fleet_index] = prediction.target_id
            target_indices[fleet_index] = prediction.target_index
            hit_turns[fleet_index] = prediction.turn
            if collision_cache is not None:
                _store_prediction_in_cache(
                    prediction,
                    fleet,
                    step=step,
                    static_target_ids=static_target_ids,
                    collision_cache=collision_cache,
                    cache_stats=cache_stats,
                )
    else:
        for fleet_index, fleet in enumerate(fleets):
            if not needs_compute[fleet_index]:
                continue

            cache_stats["computed"] += 1
            prediction = _fleet_future_hit(
                fleet_index,
                fleet,
                planets,
                planet_index_by_id,
                filter_context,
                lookahead,
            )
            if prediction is None:
                continue
            prediction_by_fleet[fleet_index] = prediction
            target_ids[fleet_index] = prediction.target_id
            target_indices[fleet_index] = prediction.target_index
            hit_turns[fleet_index] = prediction.turn
            if collision_cache is not None:
                _store_prediction_in_cache(
                    prediction,
                    fleet,
                    step=step,
                    static_target_ids=static_target_ids,
                    collision_cache=collision_cache,
                    cache_stats=cache_stats,
                )

    predictions = [
        prediction
        for prediction in prediction_by_fleet
        if prediction is not None
    ]
    return predictions, target_ids, target_indices, hit_turns, cache_stats


def _planet_physics_values(
    direct: dict[str, Any],
    predictions: list[CollisionPrediction],
) -> tuple[np.ndarray, np.ndarray]:
    planet_count = direct["metadata"]["planet_ids"].shape[0]
    feature_count = len(PLANET_PHYSICS_FEATURES)
    values = np.zeros((planet_count, feature_count), dtype=np.float32)
    masks = np.ones((planet_count, feature_count), dtype=np.bool_)
    if planet_count == 0:
        return values, masks

    player_id = int(direct["metadata"]["player_id"])
    player_count = int(direct["metadata"]["player_count"])
    slot_ids = _opponent_player_ids(player_id, player_count)
    slot_by_owner = {
        int(owner_id): slot_index
        for slot_index, owner_id in enumerate(slot_ids)
        if owner_id is not None
    }
    enemy_owner_sets: list[set[int]] = [set() for _ in range(planet_count)]
    friendly_first = np.full(planet_count, np.inf, dtype=np.float32)
    enemy_first = np.full(planet_count, np.inf, dtype=np.float32)
    friendly_last = np.zeros(planet_count, dtype=np.float32)
    enemy_last = np.zeros(planet_count, dtype=np.float32)

    for prediction in predictions:
        target_index = prediction.target_index
        owner = int(prediction.owner)
        ships = float(prediction.ships)
        turn = float(prediction.turn)
        if owner == player_id:
            values[target_index, 0] += 1.0
            values[target_index, 1] += ships
            friendly_first[target_index] = min(friendly_first[target_index], turn)
            friendly_last[target_index] = max(friendly_last[target_index], turn)
        else:
            values[target_index, 2] += 1.0
            values[target_index, 3] += ships
            enemy_owner_sets[target_index].add(owner)
            enemy_first[target_index] = min(enemy_first[target_index], turn)
            enemy_last[target_index] = max(enemy_last[target_index], turn)
            slot_index = slot_by_owner.get(owner)
            if slot_index is not None:
                values[target_index, 4 + slot_index] += 1.0
                values[target_index, 7 + slot_index] += ships

    values[:, 10] = np.asarray(
        [len(enemy_owners) for enemy_owners in enemy_owner_sets],
        dtype=np.float32,
    )
    friendly_has = np.isfinite(friendly_first)
    enemy_has = np.isfinite(enemy_first)
    values[:, 11] = np.where(friendly_has, friendly_first, 0.0)
    values[:, 12] = np.where(enemy_has, enemy_first, 0.0)
    values[:, 13] = friendly_last
    values[:, 14] = enemy_last
    values[:, 15] = values[:, 1] - values[:, 3]

    masks[:, 11] = friendly_has
    masks[:, 12] = enemy_has
    masks[:, 13] = friendly_last > 0.0
    masks[:, 14] = enemy_last > 0.0
    return values, masks


def _fleet_physics_values(
    direct: dict[str, Any],
    predictions: list[CollisionPrediction],
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fleet_count = direct["metadata"]["fleet_ids"].shape[0]
    feature_count = len(FLEET_PHYSICS_FEATURES)
    values = np.zeros((fleet_count, feature_count), dtype=np.float32)
    masks = np.zeros((fleet_count, feature_count), dtype=np.bool_)
    if fleet_count == 0:
        return values, masks

    masks[:, 0] = True
    prediction_by_fleet = {
        prediction.fleet_index: prediction
        for prediction in predictions
    }
    planet_owners = direct["context"]["planet_owners"]
    planet_ships = direct["context"]["planet_ships"]
    planet_production = direct["context"]["planet_production"]
    fleet_source_ids = direct["metadata"]["fleet_source_planet_ids"]

    for fleet_index, prediction in prediction_by_fleet.items():
        target_index = int(target_indices[fleet_index])
        if target_index < 0:
            continue
        target_owner = int(planet_owners[target_index])
        values[fleet_index, 0] = 1.0
        values[fleet_index, 1] = float(prediction.turn)
        values[fleet_index, 2] = float(prediction.distance)
        values[fleet_index, 3] = float(target_owner == prediction.owner)
        values[fleet_index, 4] = float(target_owner == -1)
        values[fleet_index, 5] = float(
            target_owner != -1 and target_owner != prediction.owner
        )
        values[fleet_index, 6] = float(planet_ships[target_index])
        values[fleet_index, 7] = float(planet_production[target_index])
        values[fleet_index, 8] = float(
            prediction.target_id == int(fleet_source_ids[fleet_index])
        )
        masks[fleet_index, :] = True

    return values, masks


def _planet_pair_physics_values(
    planet_count: int,
    planet_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(
        (planet_count, planet_count, len(PLANET_PAIR_PHYSICS_FEATURES)),
        dtype=np.float32,
    )
    masks = np.ones(values.shape, dtype=np.bool_)
    if planet_count == 0:
        return values, masks

    values[:, :, 0] = planet_values[None, :, 1]
    values[:, :, 1] = planet_values[None, :, 3]
    return values, masks


def _fleet_planet_physics_values(
    direct: dict[str, Any],
    predictions: list[CollisionPrediction],
    target_indices: np.ndarray,
    hit_turns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fleet_count = direct["metadata"]["fleet_ids"].shape[0]
    planet_count = direct["metadata"]["planet_ids"].shape[0]
    values = np.zeros(
        (fleet_count, planet_count, len(FLEET_PLANET_PHYSICS_FEATURES)),
        dtype=np.float32,
    )
    masks = np.zeros(values.shape, dtype=np.bool_)
    if fleet_count == 0 or planet_count == 0:
        return values, masks

    masks[:, :, 0] = True
    events_by_target: dict[int, list[CollisionPrediction]] = {}
    for prediction in predictions:
        events_by_target.setdefault(prediction.target_index, []).append(prediction)

    for prediction in predictions:
        fleet_index = prediction.fleet_index
        target_index = int(target_indices[fleet_index])
        if target_index < 0:
            continue

        target_events = events_by_target.get(target_index, [])
        friendly_turns = [
            event.turn
            for event in target_events
            if event.owner == prediction.owner and event.fleet_index != fleet_index
        ]
        enemy_turns = [
            event.turn
            for event in target_events
            if event.owner != prediction.owner
        ]
        same_turn_friendly_ships = sum(
            event.ships
            for event in target_events
            if (
                event.owner == prediction.owner
                and event.turn == prediction.turn
                and event.fleet_index != fleet_index
            )
        )
        same_turn_enemy_ships = sum(
            event.ships
            for event in target_events
            if event.owner != prediction.owner and event.turn == prediction.turn
        )

        values[fleet_index, target_index, 0] = 1.0
        values[fleet_index, target_index, 1] = float(hit_turns[fleet_index])
        values[fleet_index, target_index, 2] = float(
            not friendly_turns or prediction.turn < min(friendly_turns)
        )
        values[fleet_index, target_index, 3] = float(
            not enemy_turns or prediction.turn < min(enemy_turns)
        )
        values[fleet_index, target_index, 4] = float(same_turn_friendly_ships)
        values[fleet_index, target_index, 5] = float(same_turn_enemy_ships)
        masks[fleet_index, target_index, :] = True

    return values, masks


def extract_physics_features(
    direct: dict[str, Any],
    formula: dict[str, Any] | None = None,
    *,
    lookahead: int = DEFAULT_COLLISION_LOOKAHEAD,
    collision_cache: CollisionCache | None = None,
) -> dict[str, Any]:
    """Predict active-fleet collisions and derive physics feature tensors."""
    del formula
    predictions, target_ids, target_indices, hit_turns, cache_stats = _predict_fleet_hits(
        direct,
        lookahead=int(lookahead),
        collision_cache=collision_cache,
    )
    planet_values, planet_masks = _planet_physics_values(direct, predictions)
    fleet_values, fleet_masks = _fleet_physics_values(
        direct,
        predictions,
        target_indices,
    )
    planet_pair_values, planet_pair_masks = _planet_pair_physics_values(
        direct["metadata"]["planet_ids"].shape[0],
        planet_values,
    )
    fleet_planet_values, fleet_planet_masks = _fleet_planet_physics_values(
        direct,
        predictions,
        target_indices,
        hit_turns,
    )

    return {
        "feature_names": {
            "planet": PLANET_PHYSICS_FEATURES,
            "fleet": FLEET_PHYSICS_FEATURES,
            "planet_pair": PLANET_PAIR_PHYSICS_FEATURES,
            "fleet_planet": FLEET_PLANET_PHYSICS_FEATURES,
        },
        "values": {
            "planet": planet_values,
            "fleet": fleet_values,
            "planet_pair": planet_pair_values,
            "fleet_planet": fleet_planet_values,
        },
        "masks": {
            "planet": planet_masks,
            "fleet": fleet_masks,
            "planet_pair": planet_pair_masks,
            "fleet_planet": fleet_planet_masks,
        },
        "metadata": {
            "fleet_predicted_target_ids": target_ids,
            "fleet_predicted_target_indices": target_indices,
            "collision_cache_stats": cache_stats,
        },
    }
