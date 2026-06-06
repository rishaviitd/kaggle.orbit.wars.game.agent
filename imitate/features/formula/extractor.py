from __future__ import annotations

from typing import Any

import numpy as np

BOARD_CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_FLEET_SPEED = 6.0

GLOBAL_FORMULA_FEATURES = (
    "remaining_steps",
    "game_progress",
    "total_ships_controlled",
    "total_enemy_ships_controlled",
)

PLANET_FORMULA_FEATURES = (
    "owner_is_ours",
    "owner_is_neutral",
    "owner_is_enemy",
    "is_static",
    "is_orbiting",
    "is_comet",
)

FLEET_FORMULA_FEATURES = (
    "fleet_speed",
    "fleet_angle_sin",
    "fleet_angle_cos",
    "fleet_owner_is_ours",
    "fleet_owner_is_enemy",
    "fleet_source_is_ours",
    "fleet_source_is_neutral",
    "fleet_source_is_enemy",
)

PLANET_PAIR_FORMULA_FEATURES = (
    "planet_pair_same_owner",
    "planet_pair_both_owned",
    "planet_pair_source_owned_target_neutral",
    "planet_pair_source_owned_target_enemy",
    "planet_pair_same_quadrant",
    "planet_pair_production_difference",
    "planet_pair_ship_difference",
    "planet_pair_ship_ratio",
)

FLEET_PLANET_FORMULA_FEATURES = (
    "fleet_planet_owner_matches",
    "fleet_planet_is_source",
    "fleet_planet_is_friendly_destination",
    "fleet_planet_is_hostile_destination",
)

COMET_FORMULA_FEATURES = ("comet_path_progress",)


def _fleet_speed(ships: np.ndarray) -> np.ndarray:
    safe_ships = np.maximum(ships, 1.0)
    ratio = np.log(safe_ships) / np.log(1000.0)
    scaled = 1.0 + (MAX_FLEET_SPEED - 1.0) * np.power(ratio, 1.5)
    return np.where(ships <= 1.0, 1.0, np.minimum(scaled, MAX_FLEET_SPEED))


def _planet_motion_flags(
    planet_ids: np.ndarray,
    initial_planets: np.ndarray,
    comet_planet_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    is_comet = np.any(
        planet_ids[:, None] == comet_planet_ids[None, :],
        axis=1,
    ) if comet_planet_ids.size else np.zeros(planet_ids.shape, dtype=np.bool_)
    if initial_planets.shape[0] == 0:
        unavailable = np.zeros(planet_ids.shape, dtype=np.bool_)
        return unavailable, unavailable, is_comet

    initial_ids = initial_planets[:, 0].astype(np.int32)
    matches = planet_ids[:, None] == initial_ids[None, :]
    initial_present = np.any(matches, axis=1)
    initial_index = np.argmax(matches, axis=1)
    initial_x = initial_planets[initial_index, 2]
    initial_y = initial_planets[initial_index, 3]
    initial_radius = initial_planets[initial_index, 4]
    orbital_radius = np.hypot(
        initial_x - BOARD_CENTER,
        initial_y - BOARD_CENTER,
    )
    is_orbiting = (
        initial_present
        & ~is_comet
        & (orbital_radius + initial_radius < ROTATION_RADIUS_LIMIT - 1e-6)
    )
    is_static = initial_present & ~is_comet & ~is_orbiting
    return is_static, is_orbiting, is_comet


def extract_formula_features(
    direct: dict[str, Any],
    *,
    episode_steps: int = 500,
) -> dict[str, Any]:
    """Derive formula-only features from direct snapshot values."""
    context = direct["context"]
    metadata = direct["metadata"]
    player_id = int(metadata["player_id"])

    planet_ids = metadata["planet_ids"]
    planet_owners = context["planet_owners"]
    planet_positions = context["planet_positions"]
    planet_ships = context["planet_ships"]
    planet_production = context["planet_production"]

    fleet_owners = context["fleet_owners"]
    fleet_angles = context["fleet_angles"]
    fleet_ships = context["fleet_ships"]
    fleet_source_ids = metadata["fleet_source_planet_ids"]
    fleet_source_owners = context["fleet_source_owners"]
    fleet_source_present = context["fleet_source_present"]

    current_step = float(metadata["step"])
    remaining_steps = max(float(episode_steps) - current_step, 0.0)
    game_progress = np.clip(current_step / max(1.0, float(episode_steps)), 0.0, 1.0)
    friendly_stationed = np.sum(np.where(planet_owners == player_id, planet_ships, 0.0))
    enemy_stationed = np.sum(
        np.where((planet_owners != player_id) & (planet_owners != -1), planet_ships, 0.0)
    )
    friendly_fleets = np.sum(np.where(fleet_owners == player_id, fleet_ships, 0.0))
    enemy_fleets = np.sum(
        np.where((fleet_owners != player_id) & (fleet_owners != -1), fleet_ships, 0.0)
    )
    global_values = np.asarray(
        (
            remaining_steps,
            game_progress,
            friendly_stationed + friendly_fleets,
            enemy_stationed + enemy_fleets,
        ),
        dtype=np.float32,
    )

    is_static, is_orbiting, is_comet = _planet_motion_flags(
        planet_ids,
        context["initial_planets"],
        context["comet_planet_ids"],
    )
    planet_values = np.stack(
        (
            planet_owners == player_id,
            planet_owners == -1,
            (planet_owners != player_id) & (planet_owners != -1),
            is_static,
            is_orbiting,
            is_comet,
        ),
        axis=1,
    ).astype(np.float32)

    fleet_values = np.stack(
        (
            _fleet_speed(fleet_ships),
            np.sin(fleet_angles),
            np.cos(fleet_angles),
            fleet_owners == player_id,
            fleet_owners != player_id,
            fleet_source_present & (fleet_source_owners == player_id),
            fleet_source_present & (fleet_source_owners == -1),
            fleet_source_present
            & (fleet_source_owners != player_id)
            & (fleet_source_owners != -1),
        ),
        axis=1,
    ).astype(np.float32)

    source_owner = planet_owners[:, None]
    target_owner = planet_owners[None, :]
    source_ships = planet_ships[:, None]
    target_ships = planet_ships[None, :]
    source_production = planet_production[:, None]
    target_production = planet_production[None, :]
    source_quadrant_x = planet_positions[:, 0, None] >= BOARD_CENTER
    source_quadrant_y = planet_positions[:, 1, None] >= BOARD_CENTER
    target_quadrant_x = planet_positions[None, :, 0] >= BOARD_CENTER
    target_quadrant_y = planet_positions[None, :, 1] >= BOARD_CENTER
    planet_pair_values = np.stack(
        (
            source_owner == target_owner,
            (source_owner == player_id) & (target_owner == player_id),
            (source_owner == player_id) & (target_owner == -1),
            (source_owner == player_id)
            & (target_owner != player_id)
            & (target_owner != -1),
            (source_quadrant_x == target_quadrant_x)
            & (source_quadrant_y == target_quadrant_y),
            np.broadcast_to(
                target_production - source_production,
                (planet_ids.shape[0], planet_ids.shape[0]),
            ),
            np.broadcast_to(
                source_ships - target_ships,
                (planet_ids.shape[0], planet_ids.shape[0]),
            ),
            np.broadcast_to(
                source_ships / np.maximum(target_ships, 1.0),
                (planet_ids.shape[0], planet_ids.shape[0]),
            ),
        ),
        axis=2,
    ).astype(np.float32)

    fleet_owner_matrix = fleet_owners[:, None]
    planet_owner_matrix = planet_owners[None, :]
    fleet_planet_values = np.stack(
        (
            fleet_owner_matrix == planet_owner_matrix,
            fleet_source_ids[:, None] == planet_ids[None, :],
            fleet_owner_matrix == planet_owner_matrix,
            fleet_owner_matrix != planet_owner_matrix,
        ),
        axis=2,
    ).astype(np.float32)

    comet_path_indices = np.asarray(direct["values"]["comet"])[:, 0]
    comet_path_lengths = context["comet_path_lengths"].astype(np.float32)
    comet_values = (
        comet_path_indices / np.maximum(comet_path_lengths, 1.0)
    ).reshape(-1, 1).astype(np.float32)

    return {
        "feature_names": {
            "global": GLOBAL_FORMULA_FEATURES,
            "planet": PLANET_FORMULA_FEATURES,
            "fleet": FLEET_FORMULA_FEATURES,
            "planet_pair": PLANET_PAIR_FORMULA_FEATURES,
            "fleet_planet": FLEET_PLANET_FORMULA_FEATURES,
            "comet": COMET_FORMULA_FEATURES,
        },
        "values": {
            "global": global_values,
            "planet": planet_values,
            "fleet": fleet_values,
            "planet_pair": planet_pair_values,
            "fleet_planet": fleet_planet_values,
            "comet": comet_values,
        },
    }
