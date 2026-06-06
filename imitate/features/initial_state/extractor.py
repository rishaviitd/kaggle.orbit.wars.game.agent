from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

BOARD_CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
ROLE_GAME_CACHE_SIZE = 4096

PLAYER_HOME_QUADRANTS = {
    2: {
        0: (1, 1),
        1: (0, 0),
    },
    4: {
        0: (1, 1),
        1: (0, 1),
        2: (1, 0),
        3: (0, 0),
    },
}

_ROLE_GAME_CACHE: dict[
    tuple[str, str, int, int],
    tuple[frozenset[int], frozenset[int], frozenset[int]],
] = {}

PLANET_INITIAL_STATE_FEATURES = (
    "is_supplier_frontier",
    "is_attack_frontier",
    "is_conductor",
)


def _quadrant(x: float, y: float) -> tuple[int, int]:
    return (
        0 if x < BOARD_CENTER else 1,
        0 if y < BOARD_CENTER else 1,
    )


def _quadrant_center(quadrant: tuple[int, int]) -> tuple[float, float]:
    return (
        25.0 + 50.0 * quadrant[0],
        25.0 + 50.0 * quadrant[1],
    )


def _distance(
    planet: tuple[float, ...],
    point: tuple[float, float],
) -> float:
    return float(np.hypot(planet[2] - point[0], planet[3] - point[1]))


def _is_orbiting(planet: tuple[float, ...]) -> bool:
    orbital_radius = float(
        np.hypot(planet[2] - BOARD_CENTER, planet[3] - BOARD_CENTER)
    )
    return orbital_radius + planet[4] < ROTATION_RADIUS_LIMIT


def _role_score(
    planet: tuple[float, ...],
    reference_point: tuple[float, float],
) -> float:
    overhead = planet[5] / max(1.0, planet[6])
    return (
        -5.0 * overhead
        - 3.0 * _distance(planet, reference_point)
        + 2.0 * planet[6]
    )


def _best_role_planet(
    candidates: list[tuple[float, ...]],
    reference_point: tuple[float, float],
) -> tuple[float, ...]:
    return max(
        candidates,
        key=lambda planet: (
            _role_score(planet, reference_point),
            -_distance(planet, reference_point),
            planet[6],
            -planet[0],
        ),
    )


@lru_cache(maxsize=2048)
def _role_planet_ids(
    player_id: int,
    player_count: int,
    initial_planets: tuple[tuple[float, ...], ...],
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    home_quadrant = PLAYER_HOME_QUADRANTS.get(player_count, {}).get(player_id)
    if home_quadrant is None:
        return frozenset(), frozenset(), frozenset()

    enemy_quadrant = (1 - home_quadrant[0], 1 - home_quadrant[1])
    home_center = _quadrant_center(home_quadrant)
    enemy_center = _quadrant_center(enemy_quadrant)
    adjacent_quadrants = sorted(
        {
            (1 - home_quadrant[0], home_quadrant[1]),
            (home_quadrant[0], 1 - home_quadrant[1]),
        }
    )
    static_planets = [
        planet
        for planet in initial_planets
        if not _is_orbiting(planet)
    ]
    conductor_candidates = [
        planet
        for planet in static_planets
        if _quadrant(planet[2], planet[3]) == home_quadrant
    ]

    suppliers: set[int] = set()
    attackers: set[int] = set()
    conductors: set[int] = set()
    for frontier_quadrant in adjacent_quadrants:
        candidates = [
            planet
            for planet in static_planets
            if _quadrant(planet[2], planet[3]) == frontier_quadrant
        ]
        if not candidates:
            continue

        supplier = _best_role_planet(candidates, home_center)
        attacker = _best_role_planet(candidates, enemy_center)
        suppliers.add(int(supplier[0]))
        attackers.add(int(attacker[0]))

        if conductor_candidates:
            supplier_point = (supplier[2], supplier[3])
            conductor = _best_role_planet(
                conductor_candidates,
                supplier_point,
            )
            conductors.add(int(conductor[0]))

    return (
        frozenset(suppliers),
        frozenset(attackers),
        frozenset(conductors),
    )


def extract_initial_state_features(
    direct: dict[str, Any],
) -> dict[str, Any]:
    """Assign fixed SF, AF, and conductor roles from the initial board."""
    metadata = direct["metadata"]
    context = direct["context"]
    cache_key = (
        str(metadata["date"]),
        str(metadata["game_id"]),
        int(metadata["player_id"]),
        int(metadata["player_count"]),
    )
    roles = _ROLE_GAME_CACHE.get(cache_key)
    if roles is None:
        comet_ids = set(int(value) for value in context["comet_planet_ids"])
        initial_planets = tuple(
            tuple(float(value) for value in planet)
            for planet in context["initial_planets"]
            if int(planet[0]) not in comet_ids
        )
        roles = _role_planet_ids(
            int(metadata["player_id"]),
            int(metadata["player_count"]),
            initial_planets,
        )
        if len(_ROLE_GAME_CACHE) >= ROLE_GAME_CACHE_SIZE:
            _ROLE_GAME_CACHE.pop(next(iter(_ROLE_GAME_CACHE)))
        _ROLE_GAME_CACHE[cache_key] = roles

    supplier_ids, attacker_ids, conductor_ids = roles
    planet_ids = metadata["planet_ids"]
    values = np.column_stack(
        (
            np.isin(planet_ids, tuple(supplier_ids)),
            np.isin(planet_ids, tuple(attacker_ids)),
            np.isin(planet_ids, tuple(conductor_ids)),
        )
    ).astype(np.float32)

    return {
        "feature_names": {
            "planet": PLANET_INITIAL_STATE_FEATURES,
        },
        "values": {
            "planet": values,
        },
    }
