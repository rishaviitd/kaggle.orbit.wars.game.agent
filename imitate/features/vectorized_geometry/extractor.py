from __future__ import annotations

from typing import Any

import numpy as np

BOARD_CENTER = 50.0

PLANET_GEOMETRY_FEATURES = (
    "orbital_radius",
    "orbital_angle_sin",
    "orbital_angle_cos",
    "nearest_owned_planet_distance",
    "nearest_enemy_planet_distance",
    "nearest_neutral_planet_distance",
    "distance_from_board_center",
)

FLEET_GEOMETRY_FEATURES = (
    "fleet_velocity_x",
    "fleet_velocity_y",
    "fleet_distance_to_source",
    "fleet_distance_from_board_center",
    "fleet_distance_to_owned_centroid",
    "fleet_distance_to_enemy_centroid",
    "fleet_nearest_planet_distance",
    "fleet_nearest_owned_planet_distance",
    "fleet_nearest_enemy_planet_distance",
    "fleet_nearest_neutral_planet_distance",
)

PLANET_PAIR_GEOMETRY_FEATURES = (
    "planet_pair_delta_x",
    "planet_pair_delta_y",
    "planet_pair_distance",
    "planet_pair_surface_distance",
    "planet_pair_direction_sin",
    "planet_pair_direction_cos",
    "planet_pair_relative_velocity_x",
    "planet_pair_relative_velocity_y",
    "planet_pair_relative_speed",
)

FLEET_PLANET_GEOMETRY_FEATURES = (
    "fleet_planet_delta_x",
    "fleet_planet_delta_y",
    "fleet_planet_distance",
    "fleet_planet_surface_distance",
    "fleet_planet_direction_sin",
    "fleet_planet_direction_cos",
    "fleet_planet_heading_alignment",
    "fleet_planet_cross_track_offset",
    "fleet_planet_relative_velocity_x",
    "fleet_planet_relative_velocity_y",
    "fleet_planet_relative_speed",
    "fleet_planet_static_eta",
)

COMET_GEOMETRY_FEATURES = (
    "comet_velocity_x",
    "comet_velocity_y",
    "comet_speed_current",
)


def _nearest_distances(
    distances: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    exclude_diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = distances.shape[0]
    output = np.zeros(row_count, dtype=np.float32)
    valid = np.zeros(row_count, dtype=np.bool_)
    if row_count == 0 or distances.shape[1] == 0 or not np.any(candidate_mask):
        return output, valid

    candidates = np.broadcast_to(candidate_mask[None, :], distances.shape).copy()
    if exclude_diagonal:
        np.fill_diagonal(candidates, False)
    valid = np.any(candidates, axis=1)
    output[valid] = np.min(
        np.where(candidates[valid], distances[valid], np.inf),
        axis=1,
    )
    return output, valid


def _centroid_distances(
    positions: np.ndarray,
    target_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros(positions.shape[0], dtype=np.float32)
    valid = np.full(positions.shape[0], target_positions.shape[0] > 0, dtype=np.bool_)
    if positions.shape[0] == 0 or target_positions.shape[0] == 0:
        return output, valid

    centroid = np.mean(target_positions, axis=0, dtype=np.float64)
    output[:] = np.linalg.norm(positions - centroid, axis=1)
    return output, valid


def _comet_planet_velocities(
    planet_ids: np.ndarray,
    comets: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planet_velocity = np.zeros((planet_ids.shape[0], 2), dtype=np.float32)
    group_velocity = np.zeros((len(comets), 2), dtype=np.float32)
    group_valid = np.zeros(len(comets), dtype=np.bool_)
    planet_index = {int(planet_id): index for index, planet_id in enumerate(planet_ids)}

    for group_index, comet in enumerate(comets):
        path_index = int(comet.get("path_index", 0) or 0)
        paths = comet.get("paths") or []
        comet_planet_ids = comet.get("planet_ids") or []
        canonical_velocity: np.ndarray | None = None

        for comet_planet_id, path in zip(comet_planet_ids, paths, strict=False):
            if path_index < 0 or path_index + 1 >= len(path):
                continue
            current = np.asarray(path[path_index], dtype=np.float32)
            following = np.asarray(path[path_index + 1], dtype=np.float32)
            velocity = following - current
            index = planet_index.get(int(comet_planet_id))
            if index is not None:
                planet_velocity[index] = velocity
            if canonical_velocity is None:
                canonical_velocity = velocity

        # Comet groups contain symmetric paths. The first path is their stable,
        # canonical orientation; per-planet relations use each path's own velocity.
        if canonical_velocity is not None:
            group_velocity[group_index] = canonical_velocity
            group_valid[group_index] = True

    return planet_velocity, group_velocity, group_valid


def _planet_velocities(
    direct: dict[str, Any],
    formula: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = direct["context"]
    positions = context["planet_positions"].astype(np.float32, copy=False)
    formula_planets = np.asarray(formula["values"]["planet"])
    is_orbiting = formula_planets[:, 4].astype(np.bool_)
    angular_velocity = float(direct["values"]["global"][2])

    velocities = np.zeros_like(positions, dtype=np.float32)
    if np.any(is_orbiting):
        centered = positions[is_orbiting] - BOARD_CENTER
        cosine = np.float32(np.cos(angular_velocity))
        sine = np.float32(np.sin(angular_velocity))
        next_centered = np.column_stack(
            (
                centered[:, 0] * cosine - centered[:, 1] * sine,
                centered[:, 0] * sine + centered[:, 1] * cosine,
            )
        )
        velocities[is_orbiting] = next_centered - centered

    comet_velocity, group_velocity, group_valid = _comet_planet_velocities(
        direct["metadata"]["planet_ids"],
        context["comets"],
    )
    is_comet = formula_planets[:, 5].astype(np.bool_)
    velocities[is_comet] = comet_velocity[is_comet]
    return velocities, group_velocity, group_valid


def _planet_geometry(
    direct: dict[str, Any],
    planet_pair_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    context = direct["context"]
    metadata = direct["metadata"]
    positions = context["planet_positions"].astype(np.float32, copy=False)
    owners = context["planet_owners"]
    player_id = int(metadata["player_id"])

    centered = positions - BOARD_CENTER
    orbital_radius = np.linalg.norm(centered, axis=1)
    noncentral = orbital_radius > 0.0
    angle_sin = np.divide(
        centered[:, 1],
        orbital_radius,
        out=np.zeros_like(orbital_radius),
        where=noncentral,
    )
    angle_cos = np.divide(
        centered[:, 0],
        orbital_radius,
        out=np.zeros_like(orbital_radius),
        where=noncentral,
    )

    owned = owners == player_id
    neutral = owners == -1
    enemy = ~owned & ~neutral
    nearest_owned, nearest_owned_valid = _nearest_distances(
        planet_pair_distance,
        owned,
        exclude_diagonal=True,
    )
    nearest_enemy, nearest_enemy_valid = _nearest_distances(
        planet_pair_distance,
        enemy,
        exclude_diagonal=True,
    )
    nearest_neutral, nearest_neutral_valid = _nearest_distances(
        planet_pair_distance,
        neutral,
        exclude_diagonal=True,
    )

    values = np.column_stack(
        (
            orbital_radius,
            angle_sin,
            angle_cos,
            nearest_owned,
            nearest_enemy,
            nearest_neutral,
            orbital_radius,
        )
    ).astype(np.float32, copy=False)
    masks = np.column_stack(
        (
            np.ones(positions.shape[0], dtype=np.bool_),
            noncentral,
            noncentral,
            nearest_owned_valid,
            nearest_enemy_valid,
            nearest_neutral_valid,
            np.ones(positions.shape[0], dtype=np.bool_),
        )
    )
    return values, masks


def _planet_pair_geometry(
    direct: dict[str, Any],
    planet_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = direct["context"]["planet_positions"].astype(np.float32, copy=False)
    radii = direct["context"]["planet_radii"].astype(np.float32, copy=False)

    delta_x = positions[None, :, 0] - positions[:, None, 0]
    delta_y = positions[None, :, 1] - positions[:, None, 1]
    distance = np.hypot(delta_x, delta_y)
    surface_distance = np.maximum(
        distance - radii[:, None] - radii[None, :],
        0.0,
    )
    nonzero = distance > 0.0
    direction_sin = np.divide(
        delta_y,
        distance,
        out=np.zeros_like(distance),
        where=nonzero,
    )
    direction_cos = np.divide(
        delta_x,
        distance,
        out=np.zeros_like(distance),
        where=nonzero,
    )
    relative_velocity_x = (
        planet_velocity[None, :, 0] - planet_velocity[:, None, 0]
    )
    relative_velocity_y = (
        planet_velocity[None, :, 1] - planet_velocity[:, None, 1]
    )
    relative_speed = np.hypot(relative_velocity_x, relative_velocity_y)

    values = np.stack(
        (
            delta_x,
            delta_y,
            distance,
            surface_distance,
            direction_sin,
            direction_cos,
            relative_velocity_x,
            relative_velocity_y,
            relative_speed,
        ),
        axis=2,
    ).astype(np.float32, copy=False)
    masks = np.ones(values.shape, dtype=np.bool_)
    masks[:, :, 4] = nonzero
    masks[:, :, 5] = nonzero
    return values, masks, distance


def _fleet_planet_geometry(
    direct: dict[str, Any],
    formula: dict[str, Any],
    planet_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fleet_positions = np.asarray(direct["values"]["fleet"])[:, :2]
    planet_positions = direct["context"]["planet_positions"].astype(np.float32, copy=False)
    planet_radii = direct["context"]["planet_radii"].astype(np.float32, copy=False)
    fleet_formula = np.asarray(formula["values"]["fleet"])

    fleet_speed = fleet_formula[:, 0]
    heading_sin = fleet_formula[:, 1]
    heading_cos = fleet_formula[:, 2]
    fleet_velocity = np.column_stack(
        (fleet_speed * heading_cos, fleet_speed * heading_sin)
    ).astype(np.float32, copy=False)

    delta_x = planet_positions[None, :, 0] - fleet_positions[:, None, 0]
    delta_y = planet_positions[None, :, 1] - fleet_positions[:, None, 1]
    distance = np.hypot(delta_x, delta_y)
    surface_distance = np.maximum(distance - planet_radii[None, :], 0.0)
    nonzero = distance > 0.0
    direction_sin = np.divide(
        delta_y,
        distance,
        out=np.zeros_like(distance),
        where=nonzero,
    )
    direction_cos = np.divide(
        delta_x,
        distance,
        out=np.zeros_like(distance),
        where=nonzero,
    )
    alignment = (
        heading_cos[:, None] * direction_cos
        + heading_sin[:, None] * direction_sin
    )
    cross_track = np.abs(
        heading_cos[:, None] * delta_y
        - heading_sin[:, None] * delta_x
    )
    relative_velocity_x = (
        planet_velocity[None, :, 0] - fleet_velocity[:, None, 0]
    )
    relative_velocity_y = (
        planet_velocity[None, :, 1] - fleet_velocity[:, None, 1]
    )
    relative_speed = np.hypot(relative_velocity_x, relative_velocity_y)
    static_eta = np.divide(
        distance,
        fleet_speed[:, None],
        out=np.zeros_like(distance),
        where=fleet_speed[:, None] > 0.0,
    )

    values = np.stack(
        (
            delta_x,
            delta_y,
            distance,
            surface_distance,
            direction_sin,
            direction_cos,
            alignment,
            cross_track,
            relative_velocity_x,
            relative_velocity_y,
            relative_speed,
            static_eta,
        ),
        axis=2,
    ).astype(np.float32, copy=False)
    masks = np.ones(values.shape, dtype=np.bool_)
    masks[:, :, 4] = nonzero
    masks[:, :, 5] = nonzero
    masks[:, :, 6] = nonzero
    return values, masks, distance, fleet_velocity


def _fleet_geometry(
    direct: dict[str, Any],
    fleet_velocity: np.ndarray,
    fleet_planet_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    metadata = direct["metadata"]
    context = direct["context"]
    fleet_positions = np.asarray(direct["values"]["fleet"])[:, :2]
    planet_positions = context["planet_positions"].astype(np.float32, copy=False)
    planet_ids = metadata["planet_ids"]
    fleet_source_ids = metadata["fleet_source_planet_ids"]
    owners = context["planet_owners"]
    player_id = int(metadata["player_id"])

    source_matches = fleet_source_ids[:, None] == planet_ids[None, :]
    source_valid = np.any(source_matches, axis=1)
    source_distance = np.zeros(fleet_positions.shape[0], dtype=np.float32)
    if planet_ids.shape[0] > 0 and fleet_positions.shape[0] > 0:
        source_indices = np.argmax(source_matches, axis=1)
        source_distance[source_valid] = np.linalg.norm(
            fleet_positions[source_valid] - planet_positions[source_indices[source_valid]],
            axis=1,
        )

    board_distance = np.linalg.norm(fleet_positions - BOARD_CENTER, axis=1)
    owned = owners == player_id
    neutral = owners == -1
    enemy = ~owned & ~neutral
    owned_centroid, owned_centroid_valid = _centroid_distances(
        fleet_positions,
        planet_positions[owned],
    )
    enemy_centroid, enemy_centroid_valid = _centroid_distances(
        fleet_positions,
        planet_positions[enemy],
    )
    nearest_planet, nearest_planet_valid = _nearest_distances(
        fleet_planet_distance,
        np.ones(planet_ids.shape[0], dtype=np.bool_),
    )
    nearest_owned, nearest_owned_valid = _nearest_distances(
        fleet_planet_distance,
        owned,
    )
    nearest_enemy, nearest_enemy_valid = _nearest_distances(
        fleet_planet_distance,
        enemy,
    )
    nearest_neutral, nearest_neutral_valid = _nearest_distances(
        fleet_planet_distance,
        neutral,
    )

    values = np.column_stack(
        (
            fleet_velocity[:, 0],
            fleet_velocity[:, 1],
            source_distance,
            board_distance,
            owned_centroid,
            enemy_centroid,
            nearest_planet,
            nearest_owned,
            nearest_enemy,
            nearest_neutral,
        )
    ).astype(np.float32, copy=False)
    masks = np.column_stack(
        (
            np.ones(fleet_positions.shape[0], dtype=np.bool_),
            np.ones(fleet_positions.shape[0], dtype=np.bool_),
            source_valid,
            np.ones(fleet_positions.shape[0], dtype=np.bool_),
            owned_centroid_valid,
            enemy_centroid_valid,
            nearest_planet_valid,
            nearest_owned_valid,
            nearest_enemy_valid,
            nearest_neutral_valid,
        )
    )
    return values, masks


def extract_vectorized_geometry_features(
    direct: dict[str, Any],
    formula: dict[str, Any],
) -> dict[str, Any]:
    """Compute current-snapshot spatial features with NumPy broadcasting."""
    planet_velocity, comet_velocity, comet_valid = _planet_velocities(direct, formula)
    planet_pair_values, planet_pair_masks, planet_pair_distance = (
        _planet_pair_geometry(direct, planet_velocity)
    )
    fleet_planet_values, fleet_planet_masks, fleet_planet_distance, fleet_velocity = (
        _fleet_planet_geometry(direct, formula, planet_velocity)
    )
    planet_values, planet_masks = _planet_geometry(direct, planet_pair_distance)
    fleet_values, fleet_masks = _fleet_geometry(
        direct,
        fleet_velocity,
        fleet_planet_distance,
    )
    comet_values = np.column_stack(
        (
            comet_velocity[:, 0],
            comet_velocity[:, 1],
            np.linalg.norm(comet_velocity, axis=1),
        )
    ).astype(np.float32, copy=False)
    comet_masks = np.broadcast_to(comet_valid[:, None], comet_values.shape).copy()

    return {
        "feature_names": {
            "planet": PLANET_GEOMETRY_FEATURES,
            "fleet": FLEET_GEOMETRY_FEATURES,
            "planet_pair": PLANET_PAIR_GEOMETRY_FEATURES,
            "fleet_planet": FLEET_PLANET_GEOMETRY_FEATURES,
            "comet": COMET_GEOMETRY_FEATURES,
        },
        "values": {
            "planet": planet_values,
            "fleet": fleet_values,
            "planet_pair": planet_pair_values,
            "fleet_planet": fleet_planet_values,
            "comet": comet_values,
        },
        "masks": {
            "planet": planet_masks,
            "fleet": fleet_masks,
            "planet_pair": planet_pair_masks,
            "fleet_planet": fleet_planet_masks,
            "comet": comet_masks,
        },
    }
