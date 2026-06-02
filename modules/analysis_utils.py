from __future__ import annotations

import math
from typing import Any

from modules.utils import (
    Fleet,
    Planet,
    find_valid_attack_angle,
    incoming_events_by_target,
    max_safe_source_ships_to_send,
    planet_motion_types,
    planet_position_after_moves,
    resolve_fleet_group_against_planet,
    simulate_target_with_planned_fleet,
    target_collision_time,
)

CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_COLLISION_TURN = 120
MAX_SPEED = 6.0
OPENING_SCORE_TURNS = 30
OPENING_STRATEGIC_STEP_LIMIT = 60
HIGH_PRODUCTION_MIN = 3
LOW_PRODUCTION_PENALTIES = {1: 50.0, 2: 20.0}
HIGH_PRODUCTION_FRONTIER_BONUS = {3: 10.0, 4: 20.0, 5: 30.0}
ENEMY_CENTROID_RADIUS = 75.0
ENEMY_CENTROID_WEIGHT = 0.75
OVERHEAD_OUTLIER_FLOORS = {
    "medium": 14.0,
    "high": 12.0,
}


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * min(1.0, max(0.0, float(q)))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


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


def planet_from_row(row: Any) -> dict[str, Any]:
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "radius": float(row[4]),
        "ships": int(row[5]),
        "production": int(row[6]),
    }


def planet_to_utils_planet(planet: dict[str, Any]) -> Planet:
    return Planet(
        int(planet["id"]),
        int(planet["owner"]),
        float(planet["x"]),
        float(planet["y"]),
        float(planet["radius"]),
        int(planet["ships"]),
        int(planet["production"]),
    )


def is_orbiting_planet(planet: dict[str, Any], initial_by_id: dict[int, Any], comet_ids: set[int]) -> bool:
    if int(planet["id"]) in comet_ids:
        return False
    initial = initial_by_id.get(int(planet["id"]))
    if initial is None:
        return False
    orbital_radius = math.hypot(float(initial[2]) - CENTER, float(initial[3]) - CENTER)
    return orbital_radius + float(initial[4]) < ROTATION_RADIUS_LIMIT


def quadrant(planet: dict[str, Any]) -> tuple[int, int]:
    return (0 if float(planet["x"]) < CENTER else 1, 0 if float(planet["y"]) < CENTER else 1)


def quadrant_label(q: tuple[int, int] | None) -> str:
    if q is None:
        return "unknown"
    return f"Q{q[0]}{q[1]}"


def opposite_quadrant(q: tuple[int, int]) -> tuple[int, int]:
    return (1 - q[0], 1 - q[1])


def adjacent_quadrants(q: tuple[int, int]) -> set[tuple[int, int]]:
    return {(1 - q[0], q[1]), (q[0], 1 - q[1])}


def quadrant_center(q: tuple[int, int]) -> tuple[float, float]:
    return (25.0 + 50.0 * q[0], 25.0 + 50.0 * q[1])


def centroid(planets: list[dict[str, Any]]) -> tuple[float, float] | None:
    if not planets:
        return None
    return (
        sum(float(planet["x"]) for planet in planets) / len(planets),
        sum(float(planet["y"]) for planet in planets) / len(planets),
    )


def distance(a: dict[str, Any], b: dict[str, Any] | tuple[float, float]) -> float:
    bx, by = (b if isinstance(b, tuple) else (float(b["x"]), float(b["y"])))
    return math.hypot(float(a["x"]) - bx, float(a["y"]) - by)


def production_cohort(production: int) -> str:
    production = int(production)
    if production <= 1:
        return "low"
    if production <= 3:
        return "medium"
    return "high"


def family_position(planet: dict[str, Any], initial_by_id: dict[int, Any]) -> tuple[float, float]:
    initial = initial_by_id.get(int(planet["id"]))
    if initial is None:
        x = float(planet["x"])
        y = float(planet["y"])
    else:
        x = float(initial[2])
        y = float(initial[3])
    return (min(x, 100.0 - x), min(y, 100.0 - y))


def symmetry_family_key(planet: dict[str, Any], initial_by_id: dict[int, Any]) -> str:
    x, y = family_position(planet, initial_by_id)
    return f"+{int(planet['production'])}:{round(x, 2)}:{round(y, 2)}"


def home_quadrant(player_id: int, planets: list[dict[str, Any]], initial_planets: list[dict[str, Any]]) -> tuple[int, int] | None:
    for planet in initial_planets:
        if int(planet["owner"]) == int(player_id):
            return quadrant(planet)
    for planet in planets:
        if int(planet["owner"]) == int(player_id):
            return quadrant(planet)
    return None


def frontier_role_score(planet: dict[str, Any], reference_center: tuple[float, float]) -> float:
    overhead = float(planet["ships"]) / max(1.0, float(planet["production"]))
    reference_distance = distance(planet, reference_center)
    return -5.0 * overhead - 3.0 * reference_distance + 2.0 * float(planet["production"])


def add_frontier_role_entry(
    role_by_planet: dict[int, dict[str, Any]],
    planet: dict[str, Any],
    role_name: str,
    role_short: str,
    role_quadrant: tuple[int, int],
    reference_name: str,
    reference_center: tuple[float, float],
    home_center: tuple[float, float],
    enemy_center: tuple[float, float],
    initial_by_id: dict[int, Any],
    comet_ids: set[int],
) -> None:
    planet_id = int(planet["id"])
    overhead = float(planet["ships"]) / max(1.0, float(planet["production"]))
    role_score = frontier_role_score(planet, reference_center)
    entry = role_by_planet.setdefault(
        planet_id,
        {
            "planet_id": planet_id,
            "quadrant": quadrant_label(role_quadrant),
            "owner": int(planet["owner"]),
            "production": int(planet["production"]),
            "ships": int(planet["ships"]),
            "motion": "orbiting" if is_orbiting_planet(planet, initial_by_id, comet_ids) else "static",
            "x": round(float(planet["x"]), 2),
            "y": round(float(planet["y"]), 2),
            "roles": [],
            "role_labels": [],
            "reference": [],
            "role_scores": [],
            "home_distance": round(distance(planet, home_center), 2),
            "enemy_distance": round(distance(planet, enemy_center), 2),
            "overhead": round(overhead, 2),
            "score": round(role_score, 2),
        },
    )
    if role_name not in entry["roles"]:
        entry["roles"].append(role_name)
        entry["role_labels"].append(role_short)
        entry["reference"].append(reference_name)
        entry["role_scores"].append(round(role_score, 2))
        entry["score"] = round(max(float(entry["score"]), role_score), 2)


def frontier_role_report(
    planets: list[dict[str, Any]],
    initial_by_id: dict[int, Any],
    comet_ids: set[int],
    home_q: tuple[int, int] | None,
    player_id: int,
) -> dict[str, Any]:
    if home_q is None:
        return {
            "home_quadrant": "unknown",
            "enemy_quadrant": "unknown",
            "quadrant_centers": {},
            "rows": [],
            "role_planet_ids": [],
        }

    enemy_q = opposite_quadrant(home_q)
    home_center = quadrant_center(home_q)
    enemy_center = quadrant_center(enemy_q)
    role_by_planet: dict[int, dict[str, Any]] = {}

    for frontier_q in sorted(adjacent_quadrants(home_q)):
        # Orbiting planets are excluded from frontier roles. They can be valuable
        # captures, but their moving geometry makes them unstable anchors for the
        # supplier/attack frontier logistics layer.
        candidates = [
            planet
            for planet in planets
            if int(planet["id"]) not in comet_ids
            and quadrant(planet) == frontier_q
            and not is_orbiting_planet(planet, initial_by_id, comet_ids)
        ]
        if not candidates:
            continue

        supplier = max(
            candidates,
            key=lambda planet: (
                frontier_role_score(planet, home_center),
                -distance(planet, home_center),
                int(planet["production"]),
                -int(planet["id"]),
            ),
        )
        attack = max(
            candidates,
            key=lambda planet: (
                frontier_role_score(planet, enemy_center),
                -distance(planet, enemy_center),
                int(planet["production"]),
                -int(planet["id"]),
            ),
        )

        conductor_candidates = [
            planet
            for planet in planets
            if int(planet["id"]) not in comet_ids
            and quadrant(planet) == home_q
            and not is_orbiting_planet(planet, initial_by_id, comet_ids)
        ]
        if conductor_candidates:
            supplier_point = (float(supplier["x"]), float(supplier["y"]))
            conductor = max(
                conductor_candidates,
                key=lambda planet: (
                    frontier_role_score(planet, supplier_point),
                    -distance(planet, supplier_point),
                    int(planet["production"]),
                    -int(planet["id"]),
                ),
            )
            add_frontier_role_entry(
                role_by_planet,
                conductor,
                f"conductor to {quadrant_label(frontier_q)}",
                "C",
                home_q,
                f"SF p{int(supplier['id'])}",
                supplier_point,
                home_center,
                enemy_center,
                initial_by_id,
                comet_ids,
            )

        for role_name, role_short, reference_name, reference_center, planet in (
            ("supplier frontier", "SF", "home", home_center, supplier),
            ("attack frontier", "AF", "enemy", enemy_center, attack),
        ):
            add_frontier_role_entry(
                role_by_planet,
                planet,
                role_name,
                role_short,
                frontier_q,
                reference_name,
                reference_center,
                home_center,
                enemy_center,
                initial_by_id,
                comet_ids,
            )

    rows = sorted(
        role_by_planet.values(),
        key=lambda row: (
            0 if "C" in row["role_labels"] else 1,
            row["quadrant"],
            0 if "SF" in row["role_labels"] else 1,
            0 if "AF" in row["role_labels"] else 1,
            -int(row["production"]),
            -float(row["score"]),
            int(row["planet_id"]),
        ),
    )
    for row in rows:
        row["role"] = "/".join(row["role_labels"])
        row["role_detail"] = " + ".join(row["roles"])
        row["reference"] = "/".join(row["reference"])

    centers = {quadrant_label(q): {"x": quadrant_center(q)[0], "y": quadrant_center(q)[1]} for q in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    return {
        "home_quadrant": quadrant_label(home_q),
        "enemy_quadrant": quadrant_label(enemy_q),
        "quadrant_centers": centers,
        "rows": rows,
        "role_planet_ids": sorted(int(row["planet_id"]) for row in rows),
    }


def closest_source(target: dict[str, Any], sources: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    if not sources:
        return None, float("inf")
    best_source = None
    best_distance = float("inf")
    for source in sources:
        path_distance = max(0.0, distance(source, target) - float(source["radius"]) - float(target["radius"]))
        if path_distance < best_distance:
            best_source = source
            best_distance = path_distance
    if best_source is None:
        return None, float("inf")
    return best_source, best_distance


def source_target_path_distance(source: dict[str, Any], target: dict[str, Any]) -> float:
    return max(0.0, distance(source, target) - float(source["radius"]) - float(target["radius"]))


def wait_turns_to_leave_one(source: dict[str, Any], ships_to_send: int) -> int:
    source_ships = int(source["ships"])
    production = max(1, int(source["production"]))
    required_source_ships = int(ships_to_send) + 1
    if source_ships >= required_source_ships:
        return 0
    return int(math.ceil((required_source_ships - source_ships) / production))


def source_safety_after_launch(
    source: dict[str, Any],
    player_id: int,
    ships_to_send: int,
    wait_turns: int,
    horizon_turns: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    step: int,
) -> dict[str, Any]:
    launch_turns = max(0, int(wait_turns))
    launch_step = int(step) + launch_turns
    horizon_step = int(step) + max(launch_turns, int(horizon_turns))
    source_available_at_launch = max(
        0,
        int(source["ships"]) + max(1, int(source["production"])) * launch_turns - 1,
    )
    safe_ships, details = max_safe_source_ships_to_send(
        planet_to_utils_planet(source),
        int(player_id),
        launch_step,
        int(step),
        horizon_step,
        incoming_events,
    )
    wait_insufficient = int(ships_to_send) > int(source_available_at_launch)
    return {
        "source_available_at_launch": int(source_available_at_launch),
        "source_max_safe_ships": int(safe_ships),
        "source_wait_insufficient": bool(wait_insufficient),
        "source_survival_blocked": bool(
            not wait_insufficient and int(ships_to_send) > int(safe_ships)
        ),
        "source_survives_known_incoming": bool(details.get("source_survives_known_incoming", True)),
        "source_lost_turn": details.get("source_lost_turn"),
        "source_final_owner": int(details.get("source_final_owner", source["owner"])),
        "source_final_ships": int(details.get("source_final_ships", source["ships"])),
    }


def source_at_launch(
    source: dict[str, Any],
    wait_turns: int,
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> Planet:
    source_planet = planet_to_utils_planet(source)
    launch_x, launch_y = planet_position_after_moves(
        source_planet,
        max(0, int(wait_turns)),
        initial_by_id,
        angular_velocity,
        int(step),
        comet_ids,
    )
    return Planet(
        int(source_planet.id),
        int(source_planet.owner),
        float(launch_x),
        float(launch_y),
        float(source_planet.radius),
        int(source_planet.ships),
        int(source_planet.production),
    )


def route_to_target(
    source: dict[str, Any],
    target: dict[str, Any],
    ships_to_send: int,
    wait_turns: int,
    planets: list[dict[str, Any]],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> dict[str, Any]:
    utils_planets = [planet_to_utils_planet(planet) for planet in planets]
    launch_step = int(step) + int(wait_turns)
    launch_source = source_at_launch(source, wait_turns, initial_by_id, angular_velocity, step, comet_ids)
    target_planet = planet_to_utils_planet(target)
    motion_types = planet_motion_types(utils_planets, initial_by_id, comet_ids)
    angle = find_valid_attack_angle(
        launch_source,
        target_planet,
        int(ships_to_send),
        utils_planets,
        initial_by_id,
        motion_types,
        angular_velocity,
        launch_step,
        comet_ids,
    )
    if angle is None:
        return {
            "route_ok": False,
            "route_status": "blocked shot",
            "angle": None,
            "travel_turns": 999,
        }

    hit_time = target_collision_time(
        launch_source,
        target_planet,
        float(angle),
        int(ships_to_send),
        initial_by_id,
        angular_velocity,
        launch_step,
        comet_ids,
        MAX_COLLISION_TURN,
    )
    if not math.isfinite(hit_time):
        return {
            "route_ok": False,
            "route_status": "no intercept",
            "angle": None,
            "travel_turns": 999,
        }
    return {
        "route_ok": True,
        "route_status": "clear",
        "angle": float(angle),
        "travel_turns": max(1, int(math.ceil(hit_time))),
    }


def evaluate_source_target(
    source: dict[str, Any],
    target: dict[str, Any],
    player_id: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    planets: list[dict[str, Any]],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> dict[str, Any]:
    path_distance = source_target_path_distance(source, target)
    ships_needed = int(target["ships"]) + 1
    wait_turns = 0
    travel_turns = 999
    floor_details: dict[str, Any] = {}
    route_details: dict[str, Any] = {"route_ok": False, "route_status": "not evaluated", "angle": None}
    for _ in range(8):
        wait_turns = wait_turns_to_leave_one(source, ships_needed)
        route_details = route_to_target(
            source,
            target,
            ships_needed,
            wait_turns,
            planets,
            initial_by_id,
            angular_velocity,
            step,
            comet_ids,
        )
        if not route_details["route_ok"]:
            break
        travel_turns = int(route_details["travel_turns"])
        arrival_turns = wait_turns + travel_turns
        next_ships_needed, floor_details = minimum_planned_ships_to_own_and_survive(
            target,
            player_id,
            arrival_turns,
            OPENING_SCORE_TURNS,
            incoming_events,
        )
        next_ships_needed = max(0, int(next_ships_needed))
        if next_ships_needed == ships_needed:
            break
        ships_needed = next_ships_needed

    route_ok = bool(route_details.get("route_ok"))
    producing_turns = max(0, OPENING_SCORE_TURNS - wait_turns - travel_turns) if route_ok else 0
    owned_production = int(
        floor_details.get(
            "owned_production",
            float(target["production"]) * producing_turns,
        )
    ) if route_ok else 0
    tactical_net = float(owned_production) - float(ships_needed)
    source_safety = (
        source_safety_after_launch(
            source,
            player_id,
            ships_needed,
            wait_turns,
            OPENING_SCORE_TURNS,
            incoming_events,
            step,
        )
        if route_ok
        else {
            "source_available_at_launch": 0,
            "source_max_safe_ships": 0,
            "source_wait_insufficient": False,
            "source_survival_blocked": False,
            "source_survives_known_incoming": True,
            "source_lost_turn": None,
            "source_final_owner": int(source["owner"]),
            "source_final_ships": int(source["ships"]),
        }
    )
    return {
        "source": source,
        "path_distance": float(path_distance),
        "ships_needed": int(ships_needed),
        "wait_turns": int(wait_turns),
        "travel_turns": int(travel_turns),
        "producing_turns": int(producing_turns),
        "owned_production": int(owned_production),
        "tactical_net": float(tactical_net),
        "floor_details": floor_details,
        "route_ok": route_ok,
        "route_status": str(route_details.get("route_status", "")),
        "angle": route_details.get("angle"),
        **source_safety,
    }


def best_source_target_evaluation(
    target: dict[str, Any],
    sources: list[dict[str, Any]],
    player_id: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    planets: list[dict[str, Any]],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> dict[str, Any] | None:
    evaluations = source_target_evaluations(
        target,
        sources,
        player_id,
        incoming_events,
        planets,
        initial_by_id,
        angular_velocity,
        step,
        comet_ids,
    )
    evaluations = [
        evaluation
        for evaluation in evaluations
        if int(evaluation["ships_needed"]) > 0
        and evaluation["route_ok"]
        and not bool(evaluation.get("source_survival_blocked", False))
        and not bool(evaluation.get("source_wait_insufficient", False))
    ]
    if not evaluations:
        return None
    return max(evaluations, key=source_target_evaluation_sort_key)


def source_target_evaluations(
    target: dict[str, Any],
    sources: list[dict[str, Any]],
    player_id: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    planets: list[dict[str, Any]],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> list[dict[str, Any]]:
    return [
        evaluate_source_target(
            source,
            target,
            player_id,
            incoming_events,
            planets,
            initial_by_id,
            angular_velocity,
            step,
            comet_ids,
        )
        for source in sources
    ]


def source_target_evaluation_sort_key(item: dict[str, Any]) -> tuple[float, int, float, int, int]:
    return (
        float(item["tactical_net"]),
        -int(item["wait_turns"]) - int(item["travel_turns"]),
        -float(item["path_distance"]),
        int(item["source"]["production"]),
        int(item["source"]["ships"]),
    )


def reinforcement_wait_candidates(
    target: dict[str, Any],
    incoming_events: dict[int, dict[int, dict[int, int]]],
    travel_turns: int,
    minimum_wait: int,
    target_status: dict[str, Any],
) -> list[int]:
    target_events = incoming_events.get(int(target["id"]), {})
    event_turns = [int(turn) for turn in target_events]
    if target_status.get("friendly_eta_turns") is not None:
        event_turns.append(int(target_status["friendly_eta_turns"]))
    if target_status.get("lost_turn") is not None:
        event_turns.append(int(target_status["lost_turn"]))

    candidates = {int(minimum_wait)}
    for event_turn in event_turns:
        for offset in (-1, 0, 1):
            candidates.add(int(event_turn) - int(travel_turns) + offset)
    return sorted(wait for wait in candidates if wait >= int(minimum_wait))


def evaluate_reinforcement_source(
    source: dict[str, Any],
    target: dict[str, Any],
    player_id: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    target_status: dict[str, Any],
    planets: list[dict[str, Any]],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> dict[str, Any] | None:
    path_distance = source_target_path_distance(source, target)
    lost_turn = target_status.get("lost_turn")
    horizon_turns = max(OPENING_SCORE_TURNS, int(lost_turn or 0), 1)
    source_capacity = int(source["ships"]) + max(1, int(source["production"])) * horizon_turns - 1
    max_ships = max(0, min(source_capacity, 1000))
    if max_ships <= 0:
        return None

    best_late_candidate = None
    for planned_ships in range(1, max_ships + 1):
        speed = fleet_speed(planned_ships)
        travel_turns = int(math.ceil(path_distance / speed))
        minimum_wait = wait_turns_to_leave_one(source, planned_ships)
        wait_candidates = reinforcement_wait_candidates(
            target,
            incoming_events,
            travel_turns,
            minimum_wait,
            target_status,
        )
        for wait_turns in wait_candidates:
            route_details = route_to_target(
                source,
                target,
                planned_ships,
                wait_turns,
                planets,
                initial_by_id,
                angular_velocity,
                step,
                comet_ids,
            )
            if not route_details["route_ok"]:
                continue
            travel_turns = int(route_details["travel_turns"])
            arrival_turns = max(1, int(wait_turns) + int(travel_turns))
            source_safety = source_safety_after_launch(
                source,
                player_id,
                planned_ships,
                wait_turns,
                max(horizon_turns, arrival_turns),
                incoming_events,
                step,
            )
            if source_safety["source_survival_blocked"] or source_safety["source_wait_insufficient"]:
                continue
            simulation = simulate_target_with_planned_fleet(
                planet_to_utils_planet(target),
                player_id,
                planned_ships,
                arrival_turns,
                max(horizon_turns, arrival_turns),
                incoming_events,
            )
            survives = (
                bool(simulation["survived_known_incoming"])
                and int(simulation["final_owner"]) == int(player_id)
            )
            if not survives:
                continue

            loss_turn = target_status.get("lost_turn")
            first_friendly_turn = target_status.get("first_friendly_arrival_turns")
            rescue_deadline = loss_turn if loss_turn is not None else first_friendly_turn
            timely = rescue_deadline is None or arrival_turns <= int(rescue_deadline)
            baseline_simulation = simulate_target_with_planned_fleet(
                planet_to_utils_planet(target),
                player_id,
                0,
                arrival_turns,
                max(horizon_turns, arrival_turns),
                incoming_events,
            )
            baseline_already_owns = int(baseline_simulation["final_owner"]) == int(player_id)
            prevents_known_loss = bool(target_status.get("lost_after_first_owned")) and timely
            if baseline_already_owns and not prevents_known_loss:
                continue

            floor_ships, floor_details = minimum_planned_ships_to_own_and_survive(
                target,
                player_id,
                arrival_turns,
                max(horizon_turns, arrival_turns),
                incoming_events,
            )
            details = dict(simulation)
            details["capture_ships_needed"] = int(floor_ships)
            details["survival_extra_ships"] = max(0, int(planned_ships) - int(details["capture_ships_needed"]))
            details["rescue_deadline_turn"] = rescue_deadline
            details["rescue_timely"] = bool(timely)
            details["rescue_outcome"] = "saves" if timely else "recaptures"
            details["baseline_final_owner"] = int(baseline_simulation["final_owner"])
            details["baseline_final_ships"] = int(baseline_simulation["final_ships"])
            details["baseline_already_owns"] = bool(baseline_already_owns)
            producing_turns = (
                int(simulation["owned_production"]) // max(1, int(target["production"]))
                if int(target["production"]) > 0
                else 0
            )
            candidate = {
                "source": source,
                "path_distance": float(path_distance),
                "ships_needed": int(planned_ships),
                "wait_turns": int(wait_turns),
                "travel_turns": int(travel_turns),
                "producing_turns": int(producing_turns),
                "owned_production": int(simulation["owned_production"]),
                "tactical_net": float(simulation["owned_production"]) - float(planned_ships),
                "floor_details": details,
                "route_ok": True,
                "route_status": str(route_details.get("route_status", "clear")),
                "angle": route_details.get("angle"),
                **source_safety,
            }
            if timely:
                return candidate
            if best_late_candidate is None:
                best_late_candidate = candidate
    return best_late_candidate


def best_reinforcement_evaluation(
    target: dict[str, Any],
    sources: list[dict[str, Any]],
    player_id: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    target_status: dict[str, Any],
    planets: list[dict[str, Any]],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
) -> dict[str, Any] | None:
    evaluations = [
        evaluate_reinforcement_source(
            source,
            target,
            player_id,
            incoming_events,
            target_status,
            planets,
            initial_by_id,
            angular_velocity,
            step,
            comet_ids,
        )
        for source in sources
    ]
    evaluations = [evaluation for evaluation in evaluations if evaluation is not None]
    if not evaluations:
        return None

    lost_turn = target_status.get("lost_turn")

    def reinforcement_key(item: dict[str, Any]) -> tuple[Any, ...]:
        arrival_turns = int(item["wait_turns"]) + int(item["travel_turns"])
        timely = lost_turn is None or arrival_turns <= int(lost_turn)
        return (
            1 if timely else 0,
            -int(item["ships_needed"]),
            -arrival_turns,
            -int(item["wait_turns"]),
            -int(item["travel_turns"]),
            float(item["tactical_net"]),
            int(item["source"]["production"]),
            int(item["source"]["ships"]),
        )

    return max(evaluations, key=reinforcement_key)


def known_fleet_target_status(
    target: dict[str, Any],
    player_id: int,
    current_step: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    horizon_turns: int = OPENING_SCORE_TURNS,
) -> dict[str, Any]:
    owner = int(target["owner"])
    ships = int(target["ships"])
    events = incoming_events.get(int(target["id"]), {})
    friendly_arrival_turns = sorted(
        int(turn)
        for turn, by_owner in events.items()
        if int(player_id) in {int(owner_id) for owner_id in by_owner}
    )
    current_turn = 0
    first_owned_turn = 0 if owner == int(player_id) else None
    contested = False
    lost_after_first_owned = False
    lost_turn = None

    for turn in sorted(int(turn) for turn in events if 1 <= int(turn) <= int(horizon_turns)):
        if owner != -1:
            ships += int(target["production"]) * (turn - current_turn)
        turn_events = {int(owner_id): int(count) for owner_id, count in events.get(turn, {}).items()}
        if int(player_id) in turn_events and any(owner_id != int(player_id) for owner_id in turn_events):
            contested = True
        owner, ships = resolve_fleet_group_against_planet(owner, ships, turn_events)
        if first_owned_turn is None and owner == int(player_id):
            first_owned_turn = turn
        elif first_owned_turn is not None and owner != int(player_id):
            lost_after_first_owned = True
            if lost_turn is None:
                lost_turn = turn
        current_turn = turn

    if owner != -1 and current_turn < int(horizon_turns):
        ships += int(target["production"]) * (int(horizon_turns) - current_turn)

    handled_by_known_fleets = first_owned_turn is not None and int(owner) == int(player_id) and not lost_after_first_owned
    return {
        "projected_owner": int(owner),
        "projected_ships": int(ships),
        "friendly_eta": None if first_owned_turn is None else int(current_step) + int(first_owned_turn),
        "friendly_eta_turns": first_owned_turn,
        "first_friendly_arrival": None
        if not friendly_arrival_turns
        else int(current_step) + int(friendly_arrival_turns[0]),
        "first_friendly_arrival_turns": None if not friendly_arrival_turns else int(friendly_arrival_turns[0]),
        "contested_incoming": bool(contested),
        "lost_after_first_owned": bool(lost_after_first_owned),
        "lost_turn": lost_turn,
        "handled_by_known_fleets": bool(handled_by_known_fleets),
        "incoming_friendly": sum(int(by_owner.get(int(player_id), 0)) for by_owner in events.values()),
        "incoming_enemy": sum(
            int(ships)
            for by_owner in events.values()
            for owner_id, ships in by_owner.items()
            if int(owner_id) != int(player_id)
        ),
    }


def minimum_planned_ships_to_own_and_survive(
    target: dict[str, Any],
    player_id: int,
    arrival_turns: int,
    horizon_turns: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
) -> tuple[int, dict[str, Any]]:
    target_planet = planet_to_utils_planet(target)
    arrival_turns = max(1, int(arrival_turns))
    horizon_turns = max(arrival_turns, int(horizon_turns))
    target_events = incoming_events.get(int(target["id"]), {})
    other_incoming = sum(
        int(ships)
        for turn, ships_by_owner in target_events.items()
        if arrival_turns <= int(turn) <= horizon_turns
        for owner, ships in ships_by_owner.items()
        if int(owner) != int(player_id)
    )

    def survives(planned_ships: int) -> tuple[bool, dict[str, Any]]:
        simulation = simulate_target_with_planned_fleet(
            target_planet,
            player_id,
            int(planned_ships),
            arrival_turns,
            horizon_turns,
            incoming_events,
        )
        survived = bool(simulation["survived_known_incoming"]) and int(simulation["final_owner"]) == int(player_id)
        return survived, simulation

    zero_survives, zero_simulation = survives(0)
    if zero_survives:
        zero_simulation["capture_ships_needed"] = 0
        zero_simulation["survival_extra_ships"] = 0
        return 0, zero_simulation

    high = max(1, int(target["ships"]) + int(target["production"]) * horizon_turns + other_incoming + 2)
    high_survives, high_simulation = survives(high)
    while not high_survives and high < 1_000_000:
        high *= 2
        high_survives, high_simulation = survives(high)

    if not high_survives:
        high_simulation["capture_ships_needed"] = high
        high_simulation["survival_extra_ships"] = 0
        return high, high_simulation

    low = 1
    best_simulation = high_simulation
    while low < high:
        mid = (low + high) // 2
        mid_survives, mid_simulation = survives(mid)
        if mid_survives:
            high = mid
            best_simulation = mid_simulation
        else:
            low = mid + 1

    survived, best_simulation = survives(low)
    if not survived:
        low = high
    basic_capture = int(target["ships"]) + 1
    best_simulation["capture_ships_needed"] = int(basic_capture)
    best_simulation["survival_extra_ships"] = max(0, int(low) - int(basic_capture))
    return int(low), best_simulation


def simulate_target_with_planned_fleets(
    target: dict[str, Any],
    player_id: int,
    planned_fleets: list[tuple[int, int]],
    horizon_turns: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
) -> dict[str, Any]:
    owner = int(target["owner"])
    ships = int(target["ships"])
    production = int(target["production"])
    events: dict[int, dict[int, int]] = {
        int(turn): {int(owner_id): int(count) for owner_id, count in by_owner.items()}
        for turn, by_owner in incoming_events.get(int(target["id"]), {}).items()
    }
    for planned_ships, arrival_turn in planned_fleets:
        turn = max(1, int(arrival_turn))
        events.setdefault(turn, {})
        events[turn][int(player_id)] = events[turn].get(int(player_id), 0) + int(planned_ships)

    current_turn = 0
    first_owned_turn = 0 if owner == int(player_id) else None
    lost_after_first_owned = False
    lost_turn = None
    owned_production = 0
    horizon_turns = max(1, int(horizon_turns))

    for turn in sorted(int(turn) for turn in events if 1 <= int(turn) <= horizon_turns):
        elapsed = turn - current_turn
        if owner == int(player_id):
            owned_production += production * elapsed
        if owner != -1:
            ships += production * elapsed
        owner, ships = resolve_fleet_group_against_planet(owner, ships, events.get(turn, {}))
        if first_owned_turn is None and owner == int(player_id):
            first_owned_turn = turn
        elif first_owned_turn is not None and owner != int(player_id):
            lost_after_first_owned = True
            if lost_turn is None:
                lost_turn = turn
        current_turn = turn

    elapsed = horizon_turns - current_turn
    if elapsed > 0:
        if owner == int(player_id):
            owned_production += production * elapsed
        if owner != -1:
            ships += production * elapsed

    return {
        "final_owner": int(owner),
        "final_ships": int(ships),
        "first_owned_turn": first_owned_turn,
        "lost_after_first_owned": bool(lost_after_first_owned),
        "lost_turn": lost_turn,
        "owned_production": int(owned_production),
        "survived_known_incoming": bool(first_owned_turn is not None and owner == int(player_id) and not lost_after_first_owned),
    }


def weighted_two_source_split_candidates(
    total_ships: int,
    source_a: dict[str, Any],
    source_b: dict[str, Any],
    cap_a: int,
    cap_b: int,
    distance_a: float,
    distance_b: float,
) -> list[dict[str, Any]]:
    total_ships = int(total_ships)
    if total_ships < 2:
        return []

    min_a = max(1, total_ships - int(cap_b))
    max_a = min(int(cap_a), total_ships - 1)
    if min_a > max_a:
        return []

    current_a = max(1.0, float(source_a["ships"]))
    current_b = max(1.0, float(source_b["ships"]))
    fleet_share_a = current_a / (current_a + current_b)

    distance_a = max(0.0, float(distance_a))
    distance_b = max(0.0, float(distance_b))
    distance_total = distance_a + distance_b
    distance_share_a = 0.5 if distance_total <= 0.0 else distance_a / distance_total

    cap_a = max(0, int(cap_a))
    cap_b = max(0, int(cap_b))
    safety_total = cap_a + cap_b
    safety_share_a = 0.5 if safety_total <= 0 else cap_a / safety_total

    # Split intent:
    # 40% current fleet mass, 30% distance, 30% safe spare capacity.
    # Distant sources sending more helps their fleet speed catch up, but the split
    # is still clamped by safety so logistics never drains a vulnerable planet.
    weight_a = 0.4 * fleet_share_a + 0.3 * distance_share_a + 0.3 * safety_share_a
    ideal_a = float(total_ships) * weight_a
    center_a = int(round(ideal_a))
    candidate_a_values = {min(max(center_a + offset, min_a), max_a) for offset in range(-3, 4)}
    candidate_a_values.add(min_a)
    candidate_a_values.add(max_a)

    candidates = []
    for ships_a in sorted(candidate_a_values, key=lambda value: (abs(float(value) - ideal_a), value)):
        ships_b = total_ships - int(ships_a)
        if not (1 <= ships_b <= cap_b):
            continue
        candidates.append(
            {
                "ships_a": int(ships_a),
                "ships_b": int(ships_b),
                "weight_a": round(weight_a, 4),
                "weight_b": round(1.0 - weight_a, 4),
                "ideal_a": round(ideal_a, 2),
                "ideal_b": round(float(total_ships) - ideal_a, 2),
                "split_deviation": round(abs(float(ships_a) - ideal_a), 3),
            }
        )
    return candidates


def multi_source_triplet_score(
    target: dict[str, Any],
    total_sent: int,
    safe_capacity: int,
    arrival_turns: int,
    travel_a: int,
    travel_b: int,
    source_a_buffer: int,
    source_b_buffer: int,
    final_ships: int,
    frontier_quadrants: set[tuple[int, int]],
) -> dict[str, Any]:
    production = int(target["production"])
    prod_norm = clamp01((float(production) - 3.0) / 2.0)
    frontier_norm = 1.0 if quadrant(target) in frontier_quadrants else 0.0
    surplus_norm = clamp01(float(final_ships) / max(1.0, float(production) * 3.0))
    spend_norm = clamp01(float(total_sent) / max(1.0, float(safe_capacity)))
    arrival_norm = clamp01(float(arrival_turns) / 45.0)
    gap_norm = clamp01(abs(float(travel_a) - float(travel_b)) / 15.0)
    post_buffer = min(int(source_a_buffer), int(source_b_buffer))
    risk_norm = clamp01(1.0 - float(post_buffer) / 20.0)
    score = (
        14.0 * prod_norm
        + 12.0 * frontier_norm
        + 10.0 * surplus_norm
        - 26.0 * spend_norm
        - 22.0 * arrival_norm
        - 8.0 * gap_norm
        - 12.0 * risk_norm
    )
    return {
        "multi_score": round(score, 2),
        "prod_norm": round(prod_norm, 3),
        "frontier_norm": round(frontier_norm, 3),
        "surplus_norm": round(surplus_norm, 3),
        "spend_norm": round(spend_norm, 3),
        "arrival_norm": round(arrival_norm, 3),
        "gap_norm": round(gap_norm, 3),
        "risk_norm": round(risk_norm, 3),
        "source_post_buffer": int(post_buffer),
    }


def multi_source_capture_report(
    planets: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    player_id: int,
    incoming_events: dict[int, dict[int, dict[int, int]]],
    initial_by_id: dict[int, Any],
    angular_velocity: float,
    step: int,
    comet_ids: set[int],
    high_priority_target_ids: set[int] | None = None,
    frontier_quadrants: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    high_priority_target_ids = None if high_priority_target_ids is None else {int(item) for item in high_priority_target_ids}
    frontier_quadrants = set() if frontier_quadrants is None else set(frontier_quadrants)
    target_planets = [
        planet
        for planet in planets
        if int(planet["id"]) not in comet_ids
        and int(planet["owner"]) != int(player_id)
        and int(planet["production"]) >= 4
        and (high_priority_target_ids is None or int(planet["id"]) in high_priority_target_ids)
    ]

    source_caps: dict[int, int] = {}
    for source in sources:
        safety = source_safety_after_launch(
            source,
            player_id,
            0,
            0,
            OPENING_SCORE_TURNS,
            incoming_events,
            step,
        )
        source_caps[int(source["id"])] = max(
            0,
            min(
                int(safety["source_available_at_launch"]),
                int(safety["source_max_safe_ships"]),
            ),
        )

    for target in target_planets:
        best_row = None
        target_status = known_fleet_target_status(
            target,
            player_id,
            step,
            incoming_events,
            horizon_turns=MAX_COLLISION_TURN,
        )
        if bool(target_status["handled_by_known_fleets"]):
            continue
        basic_needed = int(target["ships"]) + 1
        target_events = incoming_events.get(int(target["id"]), {})
        enemy_incoming = sum(
            int(count)
            for by_owner in target_events.values()
            for owner_id, count in by_owner.items()
            if int(owner_id) != int(player_id)
        )
        friendly_incoming = int(target_status["incoming_friendly"])
        lower_total = max(2, basic_needed - friendly_incoming)
        max_needed = max(basic_needed, int(target["ships"]) + int(target["production"]) * OPENING_SCORE_TURNS + enemy_incoming + 2)
        usable_sources = [source for source in sources if source_caps.get(int(source["id"]), 0) > 0]
        for index, source_a in enumerate(usable_sources):
            cap_a = source_caps[int(source_a["id"])]
            for source_b in usable_sources[index + 1:]:
                cap_b = source_caps[int(source_b["id"])]
                if cap_a + cap_b < lower_total:
                    continue

                upper_total = min(cap_a + cap_b, max_needed)
                distance_a = source_target_path_distance(source_a, target)
                distance_b = source_target_path_distance(source_b, target)
                for total_ships in range(lower_total, upper_total + 1):
                    pair_found = False
                    for split in weighted_two_source_split_candidates(
                        total_ships,
                        source_a,
                        source_b,
                        cap_a,
                        cap_b,
                        distance_a,
                        distance_b,
                    ):
                        ships_a = int(split["ships_a"])
                        ships_b = int(split["ships_b"])
                        route_a = route_to_target(
                            source_a,
                            target,
                            ships_a,
                            0,
                            planets,
                            initial_by_id,
                            angular_velocity,
                            step,
                            comet_ids,
                        )
                        if not route_a["route_ok"]:
                            continue
                        route_b = route_to_target(
                            source_b,
                            target,
                            ships_b,
                            0,
                            planets,
                            initial_by_id,
                            angular_velocity,
                            step,
                            comet_ids,
                        )
                        if not route_b["route_ok"]:
                            continue
                        travel_a = int(route_a["travel_turns"])
                        travel_b = int(route_b["travel_turns"])
                        horizon = max(OPENING_SCORE_TURNS, travel_a, travel_b)
                        safety_a = source_safety_after_launch(
                            source_a,
                            player_id,
                            ships_a,
                            0,
                            horizon,
                            incoming_events,
                            step,
                        )
                        safety_b = source_safety_after_launch(
                            source_b,
                            player_id,
                            ships_b,
                            0,
                            horizon,
                            incoming_events,
                            step,
                        )
                        if (
                            safety_a["source_wait_insufficient"]
                            or safety_a["source_survival_blocked"]
                            or safety_b["source_wait_insufficient"]
                            or safety_b["source_survival_blocked"]
                        ):
                            continue
                        simulation = simulate_target_with_planned_fleets(
                            target,
                            player_id,
                            [(ships_a, travel_a), (ships_b, travel_b)],
                            horizon,
                            incoming_events,
                        )
                        if int(simulation["final_owner"]) != int(player_id) or not bool(simulation["survived_known_incoming"]):
                            continue
                        total_sent = int(ships_a) + int(ships_b)
                        tactical_net = float(simulation["owned_production"]) - float(total_sent)
                        source_a_buffer = min(
                            int(safety_a["source_available_at_launch"]) - ships_a,
                            int(safety_a["source_max_safe_ships"]) - ships_a,
                        )
                        source_b_buffer = min(
                            int(safety_b["source_available_at_launch"]) - ships_b,
                            int(safety_b["source_max_safe_ships"]) - ships_b,
                        )
                        score_details = multi_source_triplet_score(
                            target,
                            total_sent,
                            int(safety_a["source_max_safe_ships"]) + int(safety_b["source_max_safe_ships"]),
                            max(travel_a, travel_b),
                            travel_a,
                            travel_b,
                            source_a_buffer,
                            source_b_buffer,
                            int(simulation["final_ships"]),
                            frontier_quadrants,
                        )
                        candidate = {
                            "target_id": int(target["id"]),
                            "quadrant": quadrant_label(quadrant(target)),
                            "production": int(target["production"]),
                            "target_ships": int(target["ships"]),
                            "target_overhead": round(float(target["ships"]) / max(1.0, float(target["production"])), 2),
                            "required_ships": int(total_sent),
                            "incoming_friendly": int(friendly_incoming),
                            "incoming_enemy": int(enemy_incoming),
                            "source_a_id": int(source_a["id"]),
                            "source_b_id": int(source_b["id"]),
                            "source_ids": [int(source_a["id"]), int(source_b["id"])],
                            "ships_a": int(ships_a),
                            "ships_b": int(ships_b),
                            "total_ships": int(total_sent),
                            "source_a_safe": int(safety_a["source_max_safe_ships"]),
                            "source_b_safe": int(safety_b["source_max_safe_ships"]),
                            "source_a_available": int(safety_a["source_available_at_launch"]),
                            "source_b_available": int(safety_b["source_available_at_launch"]),
                            "split_weight_a": float(split["weight_a"]),
                            "split_weight_b": float(split["weight_b"]),
                            "split_ideal_a": float(split["ideal_a"]),
                            "split_ideal_b": float(split["ideal_b"]),
                            "split_deviation": float(split["split_deviation"]),
                            "angle_a": round(float(route_a["angle"]), 6),
                            "angle_b": round(float(route_b["angle"]), 6),
                            "travel_a": travel_a,
                            "travel_b": travel_b,
                            "arrival_turns": max(travel_a, travel_b),
                            "first_owned_turn": simulation["first_owned_turn"],
                            "final_owner": int(simulation["final_owner"]),
                            "final_ships": int(simulation["final_ships"]),
                            "owned_production": int(simulation["owned_production"]),
                            "tactical_net": round(tactical_net, 2),
                            "target_pool": "ignored high-prod outlier",
                            "source_a_post_buffer": int(source_a_buffer),
                            "source_b_post_buffer": int(source_b_buffer),
                            **score_details,
                        }
                        key = (
                            float(candidate["multi_score"]),
                            int(candidate["production"]),
                            -int(candidate["arrival_turns"]),
                            -int(candidate["total_ships"]),
                            -float(candidate["split_deviation"]),
                            -int(candidate["target_id"]),
                        )
                        best_key = None if best_row is None else (
                            float(best_row["multi_score"]),
                            int(best_row["production"]),
                            -int(best_row["arrival_turns"]),
                            -int(best_row["total_ships"]),
                            -float(best_row["split_deviation"]),
                            -int(best_row["target_id"]),
                        )
                        if best_key is None or key > best_key:
                            best_row = candidate
                        pair_found = True
                    if pair_found:
                        break
        if best_row is not None:
            rows.append(best_row)

    rows.sort(
        key=lambda row: (
            -float(row["multi_score"]),
            int(row["arrival_turns"]),
            int(row["total_ships"]),
            -int(row["production"]),
            int(row["target_id"]),
        )
    )
    return {
        "rows": rows,
        "opportunity_count": len(rows),
        "target_count": len(target_planets),
        "target_mode": "ignored high-prod outliers",
    }


def ignored_planet_unlock_stage(progress: float) -> str:
    if progress >= 0.80:
        return "all"
    if progress >= 0.70:
        return "prod3"
    if progress >= 0.60:
        return "prod4_5"
    return "locked"


def ignored_planet_is_unlocked(production: int, progress: float) -> bool:
    production = int(production)
    if progress >= 0.80:
        return True
    if progress >= 0.70:
        return production >= 3
    if progress >= 0.60:
        return production >= 4
    return False


def ignored_planet_unlock_label(production: int, progress: float) -> str:
    if not ignored_planet_is_unlocked(production, progress):
        return ""
    if progress >= 0.80:
        return "deferred cleanup unlocked"
    if progress >= 0.70:
        return "deferred +3 unlocked"
    return "deferred +4/+5 unlocked"


def expansion_progress_report(
    planets: list[dict[str, Any]],
    player_id: int,
    enemy_q: tuple[int, int] | None,
    comet_ids: set[int],
    locked_outlier_ids: set[int],
    incoming_events: dict[int, dict[int, dict[int, int]]],
    step: int,
) -> dict[str, Any]:
    # Progress is measured over non-enemy-quadrant planets that were not marked
    # as ignored. Safely committed captures count as controlled so the unlock does
    # not lag behind moves already in flight.
    candidate_planets = [
        planet
        for planet in planets
        if int(planet["id"]) not in comet_ids
        and int(planet["id"]) not in locked_outlier_ids
        and int(planet["production"]) > 0
        and (enemy_q is None or quadrant(planet) != enemy_q)
    ]
    controlled_ids = []
    committed_ids = []
    for planet in candidate_planets:
        if int(planet["owner"]) == int(player_id):
            controlled_ids.append(int(planet["id"]))
            continue
        status = known_fleet_target_status(
            planet,
            player_id,
            step,
            incoming_events,
            horizon_turns=MAX_COLLISION_TURN,
        )
        if bool(status["handled_by_known_fleets"]):
            committed_ids.append(int(planet["id"]))

    total = len(candidate_planets)
    controlled_count = len(controlled_ids) + len(committed_ids)
    progress = 1.0 if total == 0 else controlled_count / total
    return {
        "progress": round(progress, 4),
        "progress_percent": round(progress * 100.0, 1),
        "controlled": controlled_count,
        "total": total,
        "owned_ids": sorted(controlled_ids),
        "committed_ids": sorted(committed_ids),
        "stage": ignored_planet_unlock_stage(progress),
    }


def planetary_report(obs: dict[str, Any]) -> dict[str, Any]:
    player_id = int(get(obs, "player", 0))
    step = int(get(obs, "step", 0) or 0)
    comet_ids = {int(item) for item in get(obs, "comet_planet_ids", [])}
    planets = [planet_from_row(row) for row in get(obs, "planets", [])]
    raw_initial = get(obs, "initial_planets", get(obs, "planets", []))
    initial_by_id = {int(row[0]): row for row in raw_initial}

    all_non_comets = [
        planet
        for planet in planets
        if int(planet["id"]) not in comet_ids
        and int(planet["production"]) > 0
    ]
    target_planets = [
        planet
        for planet in all_non_comets
        if int(planet["owner"]) != player_id
    ]

    families: dict[str, list[dict[str, Any]]] = {}
    for planet in all_non_comets:
        families.setdefault(symmetry_family_key(planet, initial_by_id), []).append(planet)

    overheads = [
        float(planet["ships"]) / max(1.0, float(planet["production"]))
        for planet in target_planets
    ]
    non_low_overheads = [
        float(planet["ships"]) / max(1.0, float(planet["production"]))
        for planet in target_planets
        if int(planet["production"]) > 1
    ]
    global_q25 = quantile(overheads, 0.25)
    global_median = median(overheads)
    global_q75 = quantile(overheads, 0.75)
    global_iqr = max(0.0, global_q75 - global_q25)
    global_overhead_threshold = max(global_q75 + 0.75 * global_iqr, global_median * 1.6)
    if non_low_overheads:
        non_low_q25 = quantile(non_low_overheads, 0.25)
        low_production_keep_cutoff = math.expm1(
            (math.log1p(non_low_q25) + math.log1p(global_median)) / 2.0
        )
    else:
        non_low_q25 = global_q25
        low_production_keep_cutoff = global_q25

    overheads_by_cohort: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
    for planet in target_planets:
        cohort = production_cohort(int(planet["production"]))
        overheads_by_cohort[cohort].append(float(planet["ships"]) / max(1.0, float(planet["production"])))

    cohort_stats: dict[str, dict[str, Any]] = {}
    for cohort, cohort_overheads in overheads_by_cohort.items():
        cohort_q25 = quantile(cohort_overheads, 0.25)
        cohort_median = median(cohort_overheads)
        cohort_q75 = quantile(cohort_overheads, 0.75)
        threshold = max(cohort_q75, cohort_median + 0.5 * max(0.0, cohort_q75 - cohort_q25))
        cohort_stats[cohort] = {
            "count": len(cohort_overheads),
            "median": round(cohort_median, 2),
            "q25": round(cohort_q25, 2),
            "q75": round(cohort_q75, 2),
            "high_overhead_threshold": round(threshold, 2),
            "strategy_floor": OVERHEAD_OUTLIER_FLOORS.get(cohort),
        }

    production_counts: dict[str, int] = {}
    cohort_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    for planet in target_planets:
        production = int(planet["production"])
        production_counts[str(production)] = production_counts.get(str(production), 0) + 1
        cohort_counts[production_cohort(production)] += 1

    rows = []
    for planet in target_planets:
        production = int(planet["production"])
        cohort = production_cohort(production)
        overhead = float(planet["ships"]) / max(1.0, float(production))
        stats = cohort_stats[cohort]
        cohort_threshold = float(stats["high_overhead_threshold"])
        strategy_floor = OVERHEAD_OUTLIER_FLOORS.get(cohort)
        effective_threshold = (
            low_production_keep_cutoff
            if production == 1
            else (
                global_overhead_threshold
                if strategy_floor is None
                else min(global_overhead_threshold, max(cohort_threshold, strategy_floor))
            )
        )
        is_low_production_outlier = production == 1 and overhead > low_production_keep_cutoff
        is_global_overhead_outlier = (
            len(overheads) > 1
            and overhead >= global_overhead_threshold
            and overhead > global_median + 1e-9
        )
        is_strategy_overhead_outlier = (
            strategy_floor is not None
            and len(overheads_by_cohort[cohort]) > 1
            and overhead >= strategy_floor
            and overhead >= cohort_threshold
        )
        is_overhead_outlier = is_global_overhead_outlier or is_strategy_overhead_outlier
        if is_low_production_outlier:
            section = "outlier"
            category = "low production outlier"
            reason = "low production"
        elif is_overhead_outlier and cohort == "high":
            section = "outlier"
            category = "high production overhead outlier"
            reason = "high prod high overhead"
        elif is_overhead_outlier:
            section = "outlier"
            category = "overhead outlier"
            reason = f"{cohort} high overhead"
        elif production == 1:
            section = "normal"
            category = "normal"
            reason = "cheap low-prod exception"
        else:
            section = "normal"
            category = "normal"
            reason = f"{cohort} production"

        key = symmetry_family_key(planet, initial_by_id)
        family_members = sorted(
            [
                {
                    "id": int(member["id"]),
                    "quadrant": quadrant_label(quadrant(member)),
                    "owner": int(member["owner"]),
                    "ships": int(member["ships"]),
                }
                for member in families.get(key, [])
            ],
            key=lambda item: (item["quadrant"], item["id"]),
        )
        rows.append(
            {
                "planet_id": int(planet["id"]),
                "quadrant": quadrant_label(quadrant(planet)),
                "owner": int(planet["owner"]),
                "production": production,
                "cohort": cohort,
                "ships": int(planet["ships"]),
                "overhead": round(overhead, 2),
                "threshold": round(effective_threshold, 2),
                "global_threshold": round(global_overhead_threshold, 2),
                "cohort_threshold": round(cohort_threshold, 2),
                "strategy_floor": None if strategy_floor is None else round(strategy_floor, 2),
                "section": section,
                "category": category,
                "reason": reason,
                "motion": "orbiting" if is_orbiting_planet(planet, initial_by_id, comet_ids) else "static",
                "family_key": key,
                "family_members": family_members,
                "family_label": " ".join(f"p{item['id']}({item['quadrant']})" for item in family_members),
            }
        )

    section_order = {"outlier": 0, "normal": 1}
    category_order = {
        "low production outlier": 0,
        "high production overhead outlier": 1,
        "overhead outlier": 2,
        "normal": 3,
    }
    rows.sort(
        key=lambda row: (
            section_order.get(str(row["section"]), 9),
            category_order.get(str(row["category"]), 9),
            -float(row["overhead"]),
            -int(row["production"]),
            int(row["planet_id"]),
        )
    )
    return {
        "step": step,
        "player": player_id,
        "cohort_definition": {
            "low": "1",
            "medium": "2-3",
            "high": "4-5",
        },
        "production_counts": production_counts,
        "cohort_counts": cohort_counts,
        "overhead_stats": {
            "median": round(global_median, 2),
            "q25": round(global_q25, 2),
            "q75": round(global_q75, 2),
            "threshold": round(global_overhead_threshold, 2),
            "non_low_q25": round(non_low_q25, 2),
            "low_production_keep_cutoff": round(low_production_keep_cutoff, 2),
        },
        "cohort_stats": cohort_stats,
        "rows": rows,
        "outlier_rows": [row for row in rows if row["section"] == "outlier"],
        "normal_rows": [row for row in rows if row["section"] == "normal"],
    }


def opening_report(obs: dict[str, Any]) -> dict[str, Any]:
    player_id = int(get(obs, "player", 0))
    step = int(get(obs, "step", 0) or 0)
    comet_ids = {int(item) for item in get(obs, "comet_planet_ids", [])}
    planets = [planet_from_row(row) for row in get(obs, "planets", [])]
    initial_planets = [planet_from_row(row) for row in get(obs, "initial_planets", get(obs, "planets", []))]
    raw_initial = get(obs, "initial_planets", get(obs, "planets", []))
    initial_by_id = {int(row[0]): row for row in raw_initial}
    sources = [planet for planet in planets if int(planet["owner"]) == player_id]
    home_q = home_quadrant(player_id, planets, initial_planets)
    enemy_q = opposite_quadrant(home_q) if home_q is not None else None
    frontier_qs = adjacent_quadrants(home_q) if home_q is not None else set()
    role_report = frontier_role_report(planets, initial_by_id, comet_ids, home_q, player_id)

    raw_planets = get(obs, "planets", [])
    raw_fleets = get(obs, "fleets", [])
    angular_velocity = float(get(obs, "angular_velocity", 0.0) or 0.0)
    utils_planets = [Planet(*row) for row in raw_planets]
    utils_fleets = [Fleet(*row) for row in raw_fleets]
    incoming_events = incoming_events_by_target(
        utils_fleets,
        [planet for planet in utils_planets if int(planet.id) not in comet_ids],
        initial_by_id,
        angular_velocity,
        step,
        comet_ids,
        lookahead=MAX_COLLISION_TURN,
    )
    # Early-game shortcut: keep Planetary outliers out of Capture analysis so the
    # table focuses on useful home/frontier expansion. This is a staged deferral,
    # not a permanent ban: once enough non-enemy, non-ignored planets are owned or
    # safely committed, high-production ignored planets re-enter first.
    planetary = planetary_report(obs)
    outlier_rows_by_id = {
        int(row["planet_id"]): row
        for row in planetary.get("outlier_rows", [])
    }
    planetary_outlier_ids = set(outlier_rows_by_id)
    base_expansion_progress = expansion_progress_report(
        planets,
        player_id,
        enemy_q,
        comet_ids,
        planetary_outlier_ids,
        incoming_events,
        step,
    )
    expansion_progress = float(base_expansion_progress["progress"])
    unlocked_outlier_ids = {
        planet_id
        for planet_id, row in outlier_rows_by_id.items()
        if ignored_planet_is_unlocked(int(row["production"]), expansion_progress)
    }
    locked_outlier_ids = planetary_outlier_ids - unlocked_outlier_ids

    high_prod_by_quadrant: dict[str, list[dict[str, Any]]] = {}
    for planet in planets:
        if (
            planet["id"] in comet_ids
            or planet["id"] in locked_outlier_ids
            or int(planet["production"]) < HIGH_PRODUCTION_MIN
        ):
            continue
        label = quadrant_label(quadrant(planet))
        high_prod_by_quadrant.setdefault(label, []).append(
            {
                "id": int(planet["id"]),
                "production": int(planet["production"]),
                "ships": int(planet["ships"]),
                "owner": int(planet["owner"]),
                "motion": "orbiting" if is_orbiting_planet(planet, initial_by_id, comet_ids) else "static",
                "x": round(float(planet["x"]), 2),
                "y": round(float(planet["y"]), 2),
            }
        )

    enemy_high_prod = [
        planet
        for planet in planets
        if enemy_q is not None
        and quadrant(planet) == enemy_q
        and int(planet["production"]) >= HIGH_PRODUCTION_MIN
        and not is_orbiting_planet(planet, initial_by_id, comet_ids)
        and planet["id"] not in comet_ids
        and planet["id"] not in locked_outlier_ids
    ]
    enemy_high_prod_centroid = centroid(enemy_high_prod)
    high_prod_frontier_remaining = any(
        int(planet["owner"]) != player_id
        and quadrant(planet) in frontier_qs
        and int(planet["production"]) >= HIGH_PRODUCTION_MIN
        and not is_orbiting_planet(planet, initial_by_id, comet_ids)
        and planet["id"] not in comet_ids
        and planet["id"] not in locked_outlier_ids
        for planet in planets
    )

    rows = []
    save_rows = []
    recapture_rows = []
    reinforcement_rows = []
    route_rows = []
    handled_targets = []

    def rescue_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            0 if row["rescue_outcome"] == "saves" else 1,
            int(row["ships_needed"]),
            int(row["arrival_turns"]),
            -int(row["production"]),
            int(row["target_id"]),
        )

    for target in planets:
        if target["id"] in comet_ids or int(target["owner"]) != player_id:
            continue
        target_status = known_fleet_target_status(target, player_id, step, incoming_events)
        if int(target_status["incoming_enemy"]) <= 0 or not bool(target_status["lost_after_first_owned"]):
            continue
        defense_sources = [source for source in sources if int(source["id"]) != int(target["id"])]
        evaluation = best_reinforcement_evaluation(
            target,
            defense_sources,
            player_id,
            incoming_events,
            target_status,
            planets,
            initial_by_id,
            angular_velocity,
            step,
            comet_ids,
        )
        if evaluation is None:
            continue
        source = evaluation["source"]
        floor_details = dict(evaluation["floor_details"])
        wait_turns = int(evaluation["wait_turns"])
        travel_turns = int(evaluation["travel_turns"])
        arrival_turns = wait_turns + travel_turns
        reinforce_by_turn = target_status["lost_turn"]
        target_q = quadrant(target)
        target_orbits = is_orbiting_planet(target, initial_by_id, comet_ids)
        rescue_outcome = str(
            floor_details.get(
                "rescue_outcome",
                "saves"
                if reinforce_by_turn is None or arrival_turns <= int(reinforce_by_turn)
                else "recaptures",
            )
        )
        reinforcement_rows.append(
            {
                "target_id": int(target["id"]),
                "source_id": int(source["id"]),
                "quadrant": quadrant_label(target_q),
                "role": "owned",
                "motion": "orbiting" if target_orbits else "static",
                "recommendation": "reinforce",
                "route_ok": bool(evaluation.get("route_ok", True)),
                "route_status": str(evaluation.get("route_status", "clear")),
                "angle": None if evaluation.get("angle") is None else round(float(evaluation["angle"]), 6),
                "source_available_at_launch": int(evaluation.get("source_available_at_launch", 0)),
                "source_max_safe_ships": int(evaluation.get("source_max_safe_ships", 0)),
                "source_wait_insufficient": bool(evaluation.get("source_wait_insufficient", False)),
                "source_survival_blocked": bool(evaluation.get("source_survival_blocked", False)),
                "source_survives_known_incoming": bool(evaluation.get("source_survives_known_incoming", True)),
                "source_lost_turn": evaluation.get("source_lost_turn"),
                "source_final_owner": int(evaluation.get("source_final_owner", source["owner"])),
                "source_final_ships": int(evaluation.get("source_final_ships", source["ships"])),
                "production": int(target["production"]),
                "ships_needed": int(evaluation["ships_needed"]),
                "wait_turns": wait_turns,
                "travel_turns": travel_turns,
                "arrival_turns": arrival_turns,
                "arrival_step": int(step) + arrival_turns,
                "owned_production": int(evaluation["owned_production"]),
                "tactical_net": round(float(evaluation["tactical_net"]), 2),
                "projected_owner": int(target_status["projected_owner"]),
                "projected_ships": int(target_status["projected_ships"]),
                "capture_ships_needed": int(floor_details.get("capture_ships_needed", evaluation["ships_needed"])),
                "survival_extra_ships": int(floor_details.get("survival_extra_ships", 0)),
                "survived_known_incoming": bool(floor_details.get("survived_known_incoming", True)),
                "lost_turn": floor_details.get("lost_turn"),
                "reinforce_by_turn": reinforce_by_turn,
                "reinforce_by_step": None if reinforce_by_turn is None else int(step) + int(reinforce_by_turn),
                "reinforces_before_loss": bool(rescue_outcome == "saves"),
                "rescue_timely": bool(rescue_outcome == "saves"),
                "rescue_deadline_turn": floor_details.get("rescue_deadline_turn", reinforce_by_turn),
                "rescue_outcome": rescue_outcome,
                "arrival_owner": floor_details.get("arrival_owner"),
                "arrival_ships": floor_details.get("arrival_ships"),
                "final_owner": floor_details.get("final_owner"),
                "final_ships": floor_details.get("final_ships"),
                "friendly_eta": target_status["friendly_eta"],
                "friendly_eta_turns": target_status["friendly_eta_turns"],
                "first_friendly_arrival": target_status["first_friendly_arrival"],
                "first_friendly_arrival_turns": target_status["first_friendly_arrival_turns"],
                "contested_incoming": bool(target_status["contested_incoming"]),
                "lost_after_first_owned": bool(target_status["lost_after_first_owned"]),
                "incoming_friendly": int(target_status["incoming_friendly"]),
                "incoming_enemy": int(target_status["incoming_enemy"]),
                "committed_needs_reinforcement": False,
                "claim_status": "owned under attack",
            }
        )

    for target in planets:
        if target["id"] in comet_ids or int(target["owner"]) == player_id:
            continue

        target_status = known_fleet_target_status(target, player_id, step, incoming_events)
        committed_target_status = known_fleet_target_status(
            target,
            player_id,
            step,
            incoming_events,
            horizon_turns=MAX_COLLISION_TURN,
        )
        if committed_target_status["handled_by_known_fleets"]:
            handled_targets.append(
                {
                    "target_id": int(target["id"]),
                    "friendly_eta": committed_target_status["friendly_eta"],
                    "friendly_eta_turns": committed_target_status["friendly_eta_turns"],
                    "first_friendly_arrival": committed_target_status["first_friendly_arrival"],
                    "first_friendly_arrival_turns": committed_target_status["first_friendly_arrival_turns"],
                    "projected_ships": int(committed_target_status["projected_ships"]),
                    "contested_incoming": bool(committed_target_status["contested_incoming"]),
                    "handled_horizon_turns": MAX_COLLISION_TURN,
                }
            )
            continue

        if int(committed_target_status["incoming_friendly"]) > 0:
            target_status = committed_target_status

        enemy_pressure_on_committed_capture = (
            int(target_status["incoming_enemy"]) > 0
            or bool(target_status["contested_incoming"])
            or bool(target_status["lost_after_first_owned"])
        )
        committed_needs_reinforcement = (
            int(target_status["incoming_friendly"]) > 0
            and enemy_pressure_on_committed_capture
            and not bool(target_status["handled_by_known_fleets"])
        )
        needs_reinforcement = (
            (
                target_status["friendly_eta_turns"] is not None
                and bool(target_status["lost_after_first_owned"])
            )
            or committed_needs_reinforcement
        )
        # Safety exception: even if a target is a Planetary outlier, keep it in
        # Reinforce once we already have a fleet committed and the capture may be
        # lost. Otherwise the UI can hide the exact rescue move needed to avoid
        # wasting an earlier launch.
        if target["id"] in locked_outlier_ids and not needs_reinforcement:
            continue
        target_q = quadrant(target)
        target_orbits = is_orbiting_planet(target, initial_by_id, comet_ids)
        in_frontier = target_q in frontier_qs and not target_orbits
        target_role = "orbiting" if target_orbits else ("frontier" if in_frontier else ("enemy" if enemy_q is not None and target_q == enemy_q else "home"))
        if not needs_reinforcement:
            candidate_evaluations = source_target_evaluations(
                target,
                sources,
                player_id,
                incoming_events,
                planets,
                initial_by_id,
                angular_velocity,
                step,
                comet_ids,
            )
            valid_candidates = [
                candidate
                for candidate in candidate_evaluations
                if int(candidate["ships_needed"]) > 0
                and candidate["route_ok"]
                and not bool(candidate.get("source_survival_blocked", False))
                and not bool(candidate.get("source_wait_insufficient", False))
            ]
            best_candidate = max(valid_candidates, key=source_target_evaluation_sort_key) if valid_candidates else None
            for candidate in sorted(
                candidate_evaluations,
                key=lambda item: (
                    not bool(item["route_ok"]),
                    bool(item.get("source_survival_blocked", False)),
                    -float(item["tactical_net"]),
                    int(item["wait_turns"]) + int(item["travel_turns"]),
                    int(item["source"]["id"]),
                ),
            ):
                if int(candidate["ships_needed"]) <= 0:
                    continue
                source = candidate["source"]
                is_best = bool(
                    best_candidate is not None
                    and int(source["id"]) == int(best_candidate["source"]["id"])
                    and bool(candidate["route_ok"])
                    and not bool(candidate.get("source_survival_blocked", False))
                    and not bool(candidate.get("source_wait_insufficient", False))
                )
                route_rows.append(
                    {
                        "target_id": int(target["id"]),
                        "source_id": int(source["id"]),
                        "quadrant": quadrant_label(target_q),
                        "role": target_role,
                        "motion": "orbiting" if target_orbits else "static",
                        "is_best": is_best,
                        "route_ok": bool(candidate["route_ok"]),
                        "route_status": str(candidate.get("route_status", "")),
                        "angle": None if candidate.get("angle") is None else round(float(candidate["angle"]), 6),
                        "production": int(target["production"]),
                        "ships_needed": int(candidate["ships_needed"]),
                        "wait_turns": int(candidate["wait_turns"]),
                        "travel_turns": int(candidate["travel_turns"]),
                        "arrival_turns": int(candidate["wait_turns"]) + int(candidate["travel_turns"]),
                        "owned_production": int(candidate["owned_production"]),
                        "tactical_net": round(float(candidate["tactical_net"]), 2),
                        "path_distance": round(float(candidate["path_distance"]), 2),
                        "source_available_at_launch": int(candidate.get("source_available_at_launch", 0)),
                        "source_max_safe_ships": int(candidate.get("source_max_safe_ships", 0)),
                        "source_wait_insufficient": bool(candidate.get("source_wait_insufficient", False)),
                        "source_survival_blocked": bool(candidate.get("source_survival_blocked", False)),
                        "source_survives_known_incoming": bool(candidate.get("source_survives_known_incoming", True)),
                        "source_lost_turn": candidate.get("source_lost_turn"),
                        "source_final_owner": int(candidate.get("source_final_owner", source["owner"])),
                        "source_final_ships": int(candidate.get("source_final_ships", source["ships"])),
                    }
                )
            unsafe_candidates = [
                candidate
                for candidate in candidate_evaluations
                if int(candidate["ships_needed"]) > 0
                and candidate["route_ok"]
                and bool(candidate.get("source_survival_blocked", False))
                and not bool(candidate.get("source_wait_insufficient", False))
            ]
            if unsafe_candidates:
                unsafe_candidate = max(unsafe_candidates, key=source_target_evaluation_sort_key)
                unsafe_source = unsafe_candidate["source"]
                unsafe_floor_details = dict(unsafe_candidate["floor_details"])
                unsafe_low_prod_penalty = 0.0
                if step < OPENING_STRATEGIC_STEP_LIMIT and high_prod_frontier_remaining:
                    unsafe_low_prod_penalty = LOW_PRODUCTION_PENALTIES.get(int(target["production"]), 0.0)
                unsafe_high_prod_bonus = (
                    HIGH_PRODUCTION_FRONTIER_BONUS.get(int(target["production"]), 0.0)
                    if in_frontier
                    else 0.0
                )
                unsafe_enemy_centroid_distance = None
                unsafe_enemy_centroid_bonus = 0.0
                if (
                    in_frontier
                    and int(target["production"]) >= HIGH_PRODUCTION_MIN
                    and enemy_high_prod_centroid is not None
                ):
                    unsafe_enemy_centroid_distance = distance(target, enemy_high_prod_centroid)
                    unsafe_enemy_centroid_bonus = (
                        max(0.0, ENEMY_CENTROID_RADIUS - unsafe_enemy_centroid_distance)
                        * ENEMY_CENTROID_WEIGHT
                    )
                unsafe_tactical_net = float(unsafe_candidate["tactical_net"])
                unsafe_strategic_net = (
                    unsafe_tactical_net
                    - unsafe_low_prod_penalty
                    + unsafe_high_prod_bonus
                    + unsafe_enemy_centroid_bonus
                )
                rows.append(
                    {
                        "target_id": int(target["id"]),
                        "source_id": int(unsafe_source["id"]),
                        "quadrant": quadrant_label(target_q),
                        "role": target_role,
                        "motion": "orbiting" if target_orbits else "static",
                        "recommendation": "opening",
                        "capture_candidate_kind": "unsafe_source",
                        "route_ok": True,
                        "route_status": "unsafe source",
                        "angle": None
                        if unsafe_candidate.get("angle") is None
                        else round(float(unsafe_candidate["angle"]), 6),
                        "source_available_at_launch": int(unsafe_candidate.get("source_available_at_launch", 0)),
                        "source_max_safe_ships": int(unsafe_candidate.get("source_max_safe_ships", 0)),
                        "source_wait_insufficient": False,
                        "source_survival_blocked": True,
                        "source_survives_known_incoming": bool(
                            unsafe_candidate.get("source_survives_known_incoming", True)
                        ),
                        "source_lost_turn": unsafe_candidate.get("source_lost_turn"),
                        "source_final_owner": int(
                            unsafe_candidate.get("source_final_owner", unsafe_source["owner"])
                        ),
                        "source_final_ships": int(
                            unsafe_candidate.get("source_final_ships", unsafe_source["ships"])
                        ),
                        "production": int(target["production"]),
                        "ships_needed": int(unsafe_candidate["ships_needed"]),
                        "wait_turns": int(unsafe_candidate["wait_turns"]),
                        "travel_turns": int(unsafe_candidate["travel_turns"]),
                        "arrival_turns": int(unsafe_candidate["wait_turns"]) + int(unsafe_candidate["travel_turns"]),
                        "arrival_step": int(step)
                        + int(unsafe_candidate["wait_turns"])
                        + int(unsafe_candidate["travel_turns"]),
                        "producing_turns": int(unsafe_candidate["producing_turns"]),
                        "owned_production": int(unsafe_candidate["owned_production"]),
                        "tactical_net": round(unsafe_tactical_net, 2),
                        "low_prod_penalty": round(unsafe_low_prod_penalty, 2),
                        "high_prod_bonus": round(unsafe_high_prod_bonus, 2),
                        "enemy_centroid_distance": None
                        if unsafe_enemy_centroid_distance is None
                        else round(unsafe_enemy_centroid_distance, 2),
                        "enemy_centroid_bonus": round(unsafe_enemy_centroid_bonus, 2),
                        "strategic_net": round(unsafe_strategic_net, 2),
                        "projected_owner": int(target_status["projected_owner"]),
                        "projected_ships": int(target_status["projected_ships"]),
                        "capture_ships_needed": int(
                            unsafe_floor_details.get("capture_ships_needed", unsafe_candidate["ships_needed"])
                        ),
                        "survival_extra_ships": int(unsafe_floor_details.get("survival_extra_ships", 0)),
                        "survived_known_incoming": bool(
                            unsafe_floor_details.get("survived_known_incoming", True)
                        ),
                        "lost_turn": unsafe_floor_details.get("lost_turn"),
                        "reinforce_by_turn": None,
                        "reinforce_by_step": None,
                        "reinforces_before_loss": False,
                        "rescue_timely": False,
                        "rescue_deadline_turn": None,
                        "rescue_outcome": "",
                        "arrival_owner": unsafe_floor_details.get("arrival_owner"),
                        "arrival_ships": unsafe_floor_details.get("arrival_ships"),
                        "final_owner": unsafe_floor_details.get("final_owner"),
                        "final_ships": unsafe_floor_details.get("final_ships"),
                        "friendly_eta": target_status["friendly_eta"],
                        "friendly_eta_turns": target_status["friendly_eta_turns"],
                        "first_friendly_arrival": target_status["first_friendly_arrival"],
                        "first_friendly_arrival_turns": target_status["first_friendly_arrival_turns"],
                        "contested_incoming": bool(target_status["contested_incoming"]),
                        "lost_after_first_owned": bool(target_status["lost_after_first_owned"]),
                        "incoming_friendly": int(target_status["incoming_friendly"]),
                        "incoming_enemy": int(target_status["incoming_enemy"]),
                        "committed_needs_reinforcement": False,
                        "claim_status": "unsafe source",
                    }
                )
        evaluation = (
            best_reinforcement_evaluation(
                target,
                sources,
                player_id,
                incoming_events,
                target_status,
                planets,
                initial_by_id,
                angular_velocity,
                step,
                comet_ids,
            )
            if needs_reinforcement
            else best_source_target_evaluation(
                target,
                sources,
                player_id,
                incoming_events,
                planets,
                initial_by_id,
                angular_velocity,
                step,
                comet_ids,
            )
        )
        if evaluation is None:
            continue
        source = None if evaluation is None else evaluation["source"]
        ships_needed = 0 if evaluation is None else int(evaluation["ships_needed"])
        wait_turns = 0 if evaluation is None else int(evaluation["wait_turns"])
        travel_turns = 999 if evaluation is None else int(evaluation["travel_turns"])
        arrival_turns = wait_turns + travel_turns
        producing_turns = 0 if evaluation is None else int(evaluation["producing_turns"])
        owned_production = 0 if evaluation is None else int(evaluation["owned_production"])
        tactical_net = float("-inf") if evaluation is None else float(evaluation["tactical_net"])
        floor_details = {} if evaluation is None else dict(evaluation["floor_details"])
        route_ok = False if evaluation is None else bool(evaluation.get("route_ok", True))
        route_status = "no clear route" if evaluation is None else str(evaluation.get("route_status", "clear"))
        angle = None if evaluation is None else evaluation.get("angle")

        low_prod_penalty = 0.0
        if step < OPENING_STRATEGIC_STEP_LIMIT and high_prod_frontier_remaining:
            low_prod_penalty = LOW_PRODUCTION_PENALTIES.get(int(target["production"]), 0.0)

        high_prod_bonus = 0.0
        if in_frontier:
            high_prod_bonus = HIGH_PRODUCTION_FRONTIER_BONUS.get(int(target["production"]), 0.0)

        enemy_centroid_distance = None
        enemy_centroid_bonus = 0.0
        if in_frontier and int(target["production"]) >= HIGH_PRODUCTION_MIN and enemy_high_prod_centroid is not None:
            enemy_centroid_distance = distance(target, enemy_high_prod_centroid)
            enemy_centroid_bonus = max(0.0, ENEMY_CENTROID_RADIUS - enemy_centroid_distance) * ENEMY_CENTROID_WEIGHT

        strategic_net = tactical_net - low_prod_penalty + high_prod_bonus + enemy_centroid_bonus
        if committed_needs_reinforcement and target_status["friendly_eta_turns"] is None:
            claim_status = "committed needs extra"
        elif committed_needs_reinforcement:
            claim_status = "needs reinforce"
        elif target_status["friendly_eta_turns"] is not None:
            claim_status = "needs reinforce" if target_status["lost_after_first_owned"] else "friendly eta contested"
        elif target_status["contested_incoming"]:
            claim_status = "contested"
        elif int(target["id"]) in unlocked_outlier_ids:
            claim_status = ignored_planet_unlock_label(int(target["production"]), expansion_progress)
        else:
            claim_status = ""
        reinforce_by_turn = (
            target_status["lost_turn"]
            if target_status["lost_turn"] is not None
            else target_status["first_friendly_arrival_turns"]
        )
        reinforces_before_loss = (
            needs_reinforcement
            and reinforce_by_turn is not None
            and arrival_turns <= int(reinforce_by_turn)
        )
        row = (
            {
                "target_id": int(target["id"]),
                "source_id": None if source is None else int(source["id"]),
                "quadrant": quadrant_label(target_q),
                "role": target_role,
                "motion": "orbiting" if target_orbits else "static",
                "recommendation": "reinforce" if needs_reinforcement else "opening",
                "route_ok": bool(route_ok),
                "route_status": route_status,
                "angle": None if angle is None else round(float(angle), 6),
                "source_available_at_launch": int(evaluation.get("source_available_at_launch", 0)),
                "source_max_safe_ships": int(evaluation.get("source_max_safe_ships", 0)),
                "source_wait_insufficient": bool(evaluation.get("source_wait_insufficient", False)),
                "source_survival_blocked": bool(evaluation.get("source_survival_blocked", False)),
                "source_survives_known_incoming": bool(evaluation.get("source_survives_known_incoming", True)),
                "source_lost_turn": evaluation.get("source_lost_turn"),
                "source_final_owner": int(evaluation.get("source_final_owner", source["owner"])),
                "source_final_ships": int(evaluation.get("source_final_ships", source["ships"])),
                "production": int(target["production"]),
                "ships_needed": int(ships_needed),
                "wait_turns": int(wait_turns),
                "travel_turns": int(travel_turns),
                "arrival_turns": int(arrival_turns),
                "arrival_step": int(step) + int(arrival_turns),
                "producing_turns": int(producing_turns),
                "owned_production": int(owned_production),
                "tactical_net": round(tactical_net, 2),
                "low_prod_penalty": round(low_prod_penalty, 2),
                "high_prod_bonus": round(high_prod_bonus, 2),
                "enemy_centroid_distance": None if enemy_centroid_distance is None else round(enemy_centroid_distance, 2),
                "enemy_centroid_bonus": round(enemy_centroid_bonus, 2),
                "strategic_net": round(strategic_net, 2),
                "projected_owner": int(target_status["projected_owner"]),
                "projected_ships": int(target_status["projected_ships"]),
                "capture_ships_needed": int(floor_details.get("capture_ships_needed", ships_needed)),
                "survival_extra_ships": int(floor_details.get("survival_extra_ships", 0)),
                "survived_known_incoming": bool(floor_details.get("survived_known_incoming", True)),
                "lost_turn": floor_details.get("lost_turn"),
                "reinforce_by_turn": reinforce_by_turn,
                "reinforce_by_step": None if reinforce_by_turn is None else int(step) + int(reinforce_by_turn),
                "reinforces_before_loss": bool(reinforces_before_loss),
                "rescue_timely": bool(floor_details.get("rescue_timely", reinforces_before_loss)),
                "rescue_deadline_turn": floor_details.get("rescue_deadline_turn", reinforce_by_turn),
                "rescue_outcome": floor_details.get(
                    "rescue_outcome",
                    "saves" if reinforces_before_loss else ("recaptures" if needs_reinforcement else ""),
                ),
                "arrival_owner": floor_details.get("arrival_owner"),
                "arrival_ships": floor_details.get("arrival_ships"),
                "final_owner": floor_details.get("final_owner"),
                "final_ships": floor_details.get("final_ships"),
                "friendly_eta": target_status["friendly_eta"],
                "friendly_eta_turns": target_status["friendly_eta_turns"],
                "first_friendly_arrival": target_status["first_friendly_arrival"],
                "first_friendly_arrival_turns": target_status["first_friendly_arrival_turns"],
                "contested_incoming": bool(target_status["contested_incoming"]),
                "lost_after_first_owned": bool(target_status["lost_after_first_owned"]),
                "incoming_friendly": int(target_status["incoming_friendly"]),
                "incoming_enemy": int(target_status["incoming_enemy"]),
                "committed_needs_reinforcement": bool(committed_needs_reinforcement),
                "outlier_unlocked": int(target["id"]) in unlocked_outlier_ids,
                "outlier_unlock_stage": base_expansion_progress["stage"]
                if int(target["id"]) in unlocked_outlier_ids
                else "",
                "claim_status": claim_status,
            }
        )
        if needs_reinforcement:
            if row["rescue_outcome"] == "saves":
                row["recommendation"] = "save"
                save_rows.append(row)
            else:
                row["recommendation"] = "recapture"
                recapture_rows.append(row)
        else:
            rows.append(row)

    rows.sort(key=lambda row: (-float(row["strategic_net"]), -int(row["production"]), int(row["travel_turns"]), int(row["target_id"])))
    save_rows.sort(key=rescue_sort_key)
    recapture_rows.sort(key=rescue_sort_key)
    reinforcement_rows.sort(key=rescue_sort_key)
    multi_source_target_ids = {
        planet_id
        for planet_id, outlier_row in outlier_rows_by_id.items()
        if int(outlier_row["production"]) >= 4
    }
    multi_source = multi_source_capture_report(
        planets,
        sources,
        player_id,
        incoming_events,
        initial_by_id,
        angular_velocity,
        step,
        comet_ids,
        high_priority_target_ids=multi_source_target_ids,
        frontier_quadrants=frontier_qs,
    )
    # Naming contract for the UI:
    # - save/recapture rows support captures we already committed fleets toward.
    # - reinforcement rows defend planets we already own that are under attack.
    priority_rows = reinforcement_rows + save_rows + recapture_rows
    return {
        "step": step,
        "player": player_id,
        "home_quadrant": quadrant_label(home_q),
        "enemy_quadrant": quadrant_label(enemy_q),
        "frontier_quadrants": [quadrant_label(q) for q in sorted(frontier_qs)],
        "enemy_high_prod_centroid": None
        if enemy_high_prod_centroid is None
        else {"x": round(enemy_high_prod_centroid[0], 2), "y": round(enemy_high_prod_centroid[1], 2)},
        "high_prod_frontier_remaining": bool(high_prod_frontier_remaining),
        "high_prod_by_quadrant": high_prod_by_quadrant,
        "role_report": role_report,
        "role_rows": role_report["rows"],
        "outlier_unlock_progress": base_expansion_progress,
        "ignored_outlier_target_ids": sorted(planetary_outlier_ids),
        "excluded_outlier_target_ids": sorted(locked_outlier_ids),
        "unlocked_outlier_target_ids": sorted(unlocked_outlier_ids),
        "handled_targets": handled_targets,
        "unsafe_source_route_count": sum(1 for row in route_rows if bool(row.get("source_survival_blocked", False))),
        "insufficient_source_route_count": sum(1 for row in route_rows if bool(row.get("source_wait_insufficient", False))),
        "priority_rows": priority_rows,
        "save_rows": save_rows,
        "recapture_rows": recapture_rows,
        "reinforcement_rows": reinforcement_rows,
        "multi_source_rows": multi_source["rows"],
        "multi_source_opportunity_count": multi_source["opportunity_count"],
        "multi_source_target_count": multi_source["target_count"],
        "multi_source_target_mode": multi_source["target_mode"],
        "route_rows": route_rows,
        "rows": rows,
    }
