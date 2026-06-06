from __future__ import annotations

from typing import Any

import numpy as np

BOARD_SIZE = 100.0
BOARD_CENTER = 50.0
CANONICAL_HOME_CENTER = (75.0, 75.0)
CANONICAL_OPPOSITE_CENTER = (25.0, 25.0)

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

PLANET_PLAYER_RELATIVE_FEATURES = (
    "player_relative_x",
    "player_relative_y",
    "owner_opponent_slot",
    "initial_owner_is_ours",
    "initial_owner_is_enemy",
    "initial_owner_is_neutral",
    "quadrant_is_home",
    "quadrant_is_opposite",
    "distance_from_home_center",
    "distance_from_opposite_center",
)

FLEET_PLAYER_RELATIVE_FEATURES = (
    "fleet_player_relative_x",
    "fleet_player_relative_y",
    "fleet_owner_opponent_slot",
    "fleet_quadrant_is_home",
    "fleet_quadrant_is_opposite",
)


def _rotate_left(quadrant: tuple[int, int]) -> tuple[int, int]:
    x, y = quadrant
    return (1 - y, x)


def _rotate_right(quadrant: tuple[int, int]) -> tuple[int, int]:
    x, y = quadrant
    return (y, 1 - x)


def _opposite(quadrant: tuple[int, int]) -> tuple[int, int]:
    return (1 - quadrant[0], 1 - quadrant[1])


def _home_quadrant(player_id: int, player_count: int) -> tuple[int, int] | None:
    return PLAYER_HOME_QUADRANTS.get(int(player_count), {}).get(int(player_id))


def _opponent_player_ids(
    player_id: int,
    player_count: int,
) -> tuple[int | None, int | None, int | None]:
    if player_count == 2:
        opponent = 1 - player_id if player_id in PLAYER_HOME_QUADRANTS[2] else None
        return None, opponent, None

    home_by_player = PLAYER_HOME_QUADRANTS.get(4, {})
    home = home_by_player.get(player_id)
    if home is None:
        return None, None, None
    player_by_home = {quadrant: owner for owner, quadrant in home_by_player.items()}
    return (
        player_by_home.get(_rotate_left(home)),
        player_by_home.get(_opposite(home)),
        player_by_home.get(_rotate_right(home)),
    )


def _player_relative_positions(
    positions: np.ndarray,
    home_quadrant: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    relative = positions.astype(np.float32, copy=True)
    valid = np.full(positions.shape[0], home_quadrant is not None, dtype=np.bool_)
    if positions.shape[0] == 0 or home_quadrant is None:
        return relative, valid

    if home_quadrant[0] == 0:
        relative[:, 0] = BOARD_SIZE - relative[:, 0]
    if home_quadrant[1] == 0:
        relative[:, 1] = BOARD_SIZE - relative[:, 1]
    return relative, valid


def _opponent_slot_values(
    owners: np.ndarray,
    opponent_ids: tuple[int | None, int | None, int | None],
) -> tuple[np.ndarray, np.ndarray]:
    slots = np.zeros(owners.shape[0], dtype=np.float32)
    for index, opponent_id in enumerate(opponent_ids, start=1):
        if opponent_id is None:
            continue
        slots[owners == int(opponent_id)] = float(index)
    return slots, slots > 0.0


def _initial_owner_flags(
    planet_ids: np.ndarray,
    initial_planets: np.ndarray,
    player_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((planet_ids.shape[0], 3), dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.bool_)
    if planet_ids.size == 0 or initial_planets.shape[0] == 0:
        return values, masks

    initial_ids = initial_planets[:, 0].astype(np.int32, copy=False)
    matches = planet_ids[:, None] == initial_ids[None, :]
    present = np.any(matches, axis=1)
    if not np.any(present):
        return values, masks

    initial_indices = np.argmax(matches, axis=1)
    initial_owners = np.full(planet_ids.shape[0], -2, dtype=np.int32)
    initial_owners[present] = initial_planets[initial_indices[present], 1].astype(
        np.int32,
        copy=False,
    )
    values[:, 0] = present & (initial_owners == int(player_id))
    values[:, 1] = present & (initial_owners != int(player_id)) & (initial_owners != -1)
    values[:, 2] = present & (initial_owners == -1)
    masks[present, :] = True
    return values, masks


def _quadrant_flags(relative_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    home = (
        (relative_positions[:, 0] >= BOARD_CENTER)
        & (relative_positions[:, 1] >= BOARD_CENTER)
    )
    opposite = (
        (relative_positions[:, 0] < BOARD_CENTER)
        & (relative_positions[:, 1] < BOARD_CENTER)
    )
    return home.astype(np.float32), opposite.astype(np.float32)


def _center_distances(
    relative_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    home_center = np.asarray(CANONICAL_HOME_CENTER, dtype=np.float32)
    opposite_center = np.asarray(CANONICAL_OPPOSITE_CENTER, dtype=np.float32)
    return (
        np.linalg.norm(relative_positions - home_center, axis=1),
        np.linalg.norm(relative_positions - opposite_center, axis=1),
    )


def _planet_features(
    direct: dict[str, Any],
    home_quadrant: tuple[int, int] | None,
    opponent_ids: tuple[int | None, int | None, int | None],
) -> tuple[np.ndarray, np.ndarray]:
    metadata = direct["metadata"]
    context = direct["context"]
    planet_ids = metadata["planet_ids"]
    player_id = int(metadata["player_id"])
    relative_positions, position_valid = _player_relative_positions(
        context["planet_positions"].astype(np.float32, copy=False),
        home_quadrant,
    )
    owner_slots, owner_slot_valid = _opponent_slot_values(
        context["planet_owners"],
        opponent_ids,
    )
    initial_flags, initial_masks = _initial_owner_flags(
        planet_ids,
        context["initial_planets"],
        player_id,
    )
    home_quadrant_flags, opposite_quadrant_flags = _quadrant_flags(relative_positions)
    home_distance, opposite_distance = _center_distances(relative_positions)

    values = np.column_stack(
        (
            relative_positions[:, 0],
            relative_positions[:, 1],
            owner_slots,
            initial_flags[:, 0],
            initial_flags[:, 1],
            initial_flags[:, 2],
            home_quadrant_flags,
            opposite_quadrant_flags,
            home_distance,
            opposite_distance,
        )
    ).astype(np.float32, copy=False)
    masks = np.column_stack(
        (
            position_valid,
            position_valid,
            owner_slot_valid,
            initial_masks[:, 0],
            initial_masks[:, 1],
            initial_masks[:, 2],
            position_valid,
            position_valid,
            position_valid,
            position_valid,
        )
    )
    return values, masks


def _fleet_features(
    direct: dict[str, Any],
    home_quadrant: tuple[int, int] | None,
    opponent_ids: tuple[int | None, int | None, int | None],
) -> tuple[np.ndarray, np.ndarray]:
    fleet_values = np.asarray(direct["values"]["fleet"])
    fleet_positions = fleet_values[:, :2]
    relative_positions, position_valid = _player_relative_positions(
        fleet_positions.astype(np.float32, copy=False),
        home_quadrant,
    )
    owner_slots, owner_slot_valid = _opponent_slot_values(
        direct["context"]["fleet_owners"],
        opponent_ids,
    )
    home_quadrant_flags, opposite_quadrant_flags = _quadrant_flags(relative_positions)

    values = np.column_stack(
        (
            relative_positions[:, 0],
            relative_positions[:, 1],
            owner_slots,
            home_quadrant_flags,
            opposite_quadrant_flags,
        )
    ).astype(np.float32, copy=False)
    masks = np.column_stack(
        (
            position_valid,
            position_valid,
            owner_slot_valid,
            position_valid,
            position_valid,
        )
    )
    return values, masks


def extract_player_relative_features(
    direct: dict[str, Any],
) -> dict[str, Any]:
    """Compute player-canonical coordinates, quadrants, and opponent slots."""
    metadata = direct["metadata"]
    player_id = int(metadata["player_id"])
    player_count = int(metadata["player_count"])
    home_quadrant = _home_quadrant(player_id, player_count)
    opponent_ids = _opponent_player_ids(player_id, player_count)

    planet_values, planet_masks = _planet_features(
        direct,
        home_quadrant,
        opponent_ids,
    )
    fleet_values, fleet_masks = _fleet_features(
        direct,
        home_quadrant,
        opponent_ids,
    )

    return {
        "feature_names": {
            "planet": PLANET_PLAYER_RELATIVE_FEATURES,
            "fleet": FLEET_PLAYER_RELATIVE_FEATURES,
        },
        "values": {
            "planet": planet_values,
            "fleet": fleet_values,
        },
        "masks": {
            "planet": planet_masks,
            "fleet": fleet_masks,
        },
    }
