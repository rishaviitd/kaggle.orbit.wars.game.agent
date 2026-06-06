from __future__ import annotations

from typing import Any

import numpy as np

OWNERSHIP_GROUPS = ("owned", "enemy", "neutral")
PRODUCTION_LEVELS = (1, 2, 3, 4, 5)
OPPONENT_SLOTS = (1, 2, 3)

GLOBAL_AGGREGATE_FEATURES = (
    "total_planet_count",
    "static_planet_count",
    "orbiting_planet_count",
    "comet_planet_count",
    "owned_planet_count",
    "enemy_planet_count",
    "neutral_planet_count",
    *(
        f"{group}_planets_prod_{production}_count"
        for group in OWNERSHIP_GROUPS
        for production in PRODUCTION_LEVELS
    ),
    "owned_production_total",
    "enemy_production_total",
    "neutral_production_total",
    "owned_stationed_ships_total",
    "enemy_stationed_ships_total",
    "neutral_stationed_ships_total",
    "owned_stationed_ships_mean",
    "enemy_stationed_ships_mean",
    "neutral_stationed_ships_mean",
    "owned_stationed_ships_max",
    "enemy_stationed_ships_max",
    "neutral_stationed_ships_max",
    "friendly_active_fleet_count",
    "enemy_active_fleet_count",
    "friendly_active_fleet_ships_total",
    "enemy_active_fleet_ships_total",
    "friendly_active_fleet_ships_max",
    "enemy_active_fleet_ships_max",
    *(f"opponent_{slot}_planet_count" for slot in OPPONENT_SLOTS),
    *(f"opponent_{slot}_production_total" for slot in OPPONENT_SLOTS),
    *(f"opponent_{slot}_stationed_ships_total" for slot in OPPONENT_SLOTS),
    *(f"opponent_{slot}_active_fleet_count" for slot in OPPONENT_SLOTS),
    *(f"opponent_{slot}_active_fleet_ships_total" for slot in OPPONENT_SLOTS),
)

PLANET_AGGREGATE_FEATURES = (
    "outgoing_friendly_fleet_count",
    "outgoing_friendly_ships_total",
    "outgoing_enemy_fleet_count",
    "outgoing_enemy_ships_total",
)

COMET_AGGREGATE_FEATURES = ("comet_path_length",)

# Orbit Wars assigns player homes to these fixed spawn quadrants.
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


def _group_stats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    selected = values[mask]
    if selected.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.sum(selected, dtype=np.float64)),
        float(np.mean(selected, dtype=np.float64)),
        float(np.max(selected)),
    )


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
    """Return raw player IDs for left, opposite, and right slots."""
    if player_count == 2:
        opponent = 1 - player_id if player_id in PLAYER_HOME_QUADRANTS_2P else None
        return None, opponent, None

    home_by_player = PLAYER_HOME_QUADRANTS_4P
    home = home_by_player.get(player_id)
    if home is None:
        return None, None, None
    player_by_home = {quadrant: owner for owner, quadrant in home_by_player.items()}
    return (
        player_by_home.get(_rotate_left(home)),
        player_by_home.get(_opposite(home)),
        player_by_home.get(_rotate_right(home)),
    )


def _production_counts(
    production: np.ndarray,
    mask: np.ndarray,
) -> list[float]:
    return [
        float(np.count_nonzero(mask & (production == level)))
        for level in PRODUCTION_LEVELS
    ]


def _opponent_aggregate_values(
    opponent_ids: tuple[int | None, int | None, int | None],
    planet_owners: np.ndarray,
    planet_production: np.ndarray,
    planet_ships: np.ndarray,
    fleet_owners: np.ndarray,
    fleet_ships: np.ndarray,
) -> list[float]:
    planet_counts: list[float] = []
    production_totals: list[float] = []
    stationed_ship_totals: list[float] = []
    fleet_counts: list[float] = []
    fleet_ship_totals: list[float] = []

    for opponent_id in opponent_ids:
        if opponent_id is None:
            planet_mask = np.zeros(planet_owners.shape, dtype=np.bool_)
            fleet_mask = np.zeros(fleet_owners.shape, dtype=np.bool_)
        else:
            planet_mask = planet_owners == opponent_id
            fleet_mask = fleet_owners == opponent_id
        planet_counts.append(float(np.count_nonzero(planet_mask)))
        production_totals.append(
            float(np.sum(planet_production[planet_mask], dtype=np.float64))
        )
        stationed_ship_totals.append(
            float(np.sum(planet_ships[planet_mask], dtype=np.float64))
        )
        fleet_counts.append(float(np.count_nonzero(fleet_mask)))
        fleet_ship_totals.append(
            float(np.sum(fleet_ships[fleet_mask], dtype=np.float64))
        )

    return [
        *planet_counts,
        *production_totals,
        *stationed_ship_totals,
        *fleet_counts,
        *fleet_ship_totals,
    ]


def _outgoing_planet_values(
    planet_ids: np.ndarray,
    fleet_source_ids: np.ndarray,
    fleet_owners: np.ndarray,
    fleet_ships: np.ndarray,
    player_id: int,
) -> np.ndarray:
    output = np.zeros((planet_ids.shape[0], len(PLANET_AGGREGATE_FEATURES)), dtype=np.float32)
    if planet_ids.size == 0 or fleet_source_ids.size == 0:
        return output

    planet_index = {int(planet_id): index for index, planet_id in enumerate(planet_ids)}
    valid = np.asarray(
        [int(source_id) in planet_index for source_id in fleet_source_ids],
        dtype=np.bool_,
    )
    if not np.any(valid):
        return output

    source_indices = np.asarray(
        [planet_index[int(source_id)] for source_id in fleet_source_ids[valid]],
        dtype=np.int32,
    )
    owners = fleet_owners[valid]
    ships = fleet_ships[valid]
    friendly = owners == player_id
    enemy = ~friendly

    np.add.at(output[:, 0], source_indices[friendly], 1.0)
    np.add.at(output[:, 1], source_indices[friendly], ships[friendly])
    np.add.at(output[:, 2], source_indices[enemy], 1.0)
    np.add.at(output[:, 3], source_indices[enemy], ships[enemy])
    return output


def extract_aggregate_features(
    direct: dict[str, Any],
    formula: dict[str, Any],
) -> dict[str, Any]:
    """Reduce planet and fleet entities into aggregate feature tensors."""
    context = direct["context"]
    metadata = direct["metadata"]
    player_id = int(metadata["player_id"])
    player_count = int(metadata["player_count"])

    planet_ids = metadata["planet_ids"]
    planet_owners = context["planet_owners"]
    planet_production = context["planet_production"]
    planet_ships = context["planet_ships"]
    fleet_owners = context["fleet_owners"]
    fleet_ships = context["fleet_ships"]
    fleet_source_ids = metadata["fleet_source_planet_ids"]

    planet_formula = np.asarray(formula["values"]["planet"])
    is_static = planet_formula[:, 3].astype(np.bool_)
    is_orbiting = planet_formula[:, 4].astype(np.bool_)
    is_comet = planet_formula[:, 5].astype(np.bool_)

    owned = planet_owners == player_id
    neutral = planet_owners == -1
    enemy = ~owned & ~neutral
    ownership_masks = (owned, enemy, neutral)

    production_counts = [
        value
        for mask in ownership_masks
        for value in _production_counts(planet_production, mask)
    ]

    production_totals = [
        float(np.sum(planet_production[mask], dtype=np.float64))
        for mask in ownership_masks
    ]
    stationed_stats = [
        _group_stats(planet_ships, mask)
        for mask in ownership_masks
    ]
    stationed_totals = [stats[0] for stats in stationed_stats]
    stationed_means = [stats[1] for stats in stationed_stats]
    stationed_maxes = [stats[2] for stats in stationed_stats]

    friendly_fleets = fleet_owners == player_id
    enemy_fleets = ~friendly_fleets
    friendly_fleet_stats = _group_stats(fleet_ships, friendly_fleets)
    enemy_fleet_stats = _group_stats(fleet_ships, enemy_fleets)

    opponent_values = _opponent_aggregate_values(
        _opponent_player_ids(player_id, player_count),
        planet_owners,
        planet_production,
        planet_ships,
        fleet_owners,
        fleet_ships,
    )

    global_values = np.asarray(
        [
            float(planet_ids.shape[0]),
            float(np.count_nonzero(is_static)),
            float(np.count_nonzero(is_orbiting)),
            float(np.count_nonzero(is_comet)),
            float(np.count_nonzero(owned)),
            float(np.count_nonzero(enemy)),
            float(np.count_nonzero(neutral)),
            *production_counts,
            *production_totals,
            *stationed_totals,
            *stationed_means,
            *stationed_maxes,
            float(np.count_nonzero(friendly_fleets)),
            float(np.count_nonzero(enemy_fleets)),
            friendly_fleet_stats[0],
            enemy_fleet_stats[0],
            friendly_fleet_stats[2],
            enemy_fleet_stats[2],
            *opponent_values,
        ],
        dtype=np.float32,
    )
    if global_values.shape[0] != len(GLOBAL_AGGREGATE_FEATURES):
        raise RuntimeError(
            "Aggregate feature width mismatch: "
            f"{global_values.shape[0]} values for {len(GLOBAL_AGGREGATE_FEATURES)} names."
        )

    planet_values = _outgoing_planet_values(
        planet_ids,
        fleet_source_ids,
        fleet_owners,
        fleet_ships,
        player_id,
    )
    comet_values = context["comet_path_lengths"].astype(np.float32).reshape(-1, 1)

    return {
        "feature_names": {
            "global": GLOBAL_AGGREGATE_FEATURES,
            "planet": PLANET_AGGREGATE_FEATURES,
            "comet": COMET_AGGREGATE_FEATURES,
        },
        "values": {
            "global": global_values,
            "planet": planet_values,
            "comet": comet_values,
        },
    }
