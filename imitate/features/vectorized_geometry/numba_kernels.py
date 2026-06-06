from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only when numba is absent.
    njit = None

BOARD_CENTER = 50.0


def is_numba_available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True)
    def geometry_kernel(
        planet_positions: np.ndarray,
        planet_radii: np.ndarray,
        planet_owners: np.ndarray,
        planet_ids: np.ndarray,
        player_id: int,
        planet_velocity: np.ndarray,
        fleet_positions: np.ndarray,
        fleet_source_ids: np.ndarray,
        fleet_speed: np.ndarray,
        heading_sin: np.ndarray,
        heading_cos: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        planet_count = planet_positions.shape[0]
        fleet_count = fleet_positions.shape[0]

        planet_values = np.zeros((planet_count, 7), dtype=np.float32)
        planet_masks = np.zeros((planet_count, 7), dtype=np.bool_)
        fleet_values = np.zeros((fleet_count, 10), dtype=np.float32)
        fleet_masks = np.zeros((fleet_count, 10), dtype=np.bool_)
        planet_pair_values = np.zeros((planet_count, planet_count, 9), dtype=np.float32)
        planet_pair_masks = np.ones((planet_count, planet_count, 9), dtype=np.bool_)
        fleet_planet_values = np.zeros((fleet_count, planet_count, 12), dtype=np.float32)
        fleet_planet_masks = np.ones((fleet_count, planet_count, 12), dtype=np.bool_)

        owned_centroid_x = 0.0
        owned_centroid_y = 0.0
        owned_count = 0
        enemy_centroid_x = 0.0
        enemy_centroid_y = 0.0
        enemy_count = 0
        for planet_index in range(planet_count):
            owner = planet_owners[planet_index]
            if owner == player_id:
                owned_centroid_x += planet_positions[planet_index, 0]
                owned_centroid_y += planet_positions[planet_index, 1]
                owned_count += 1
            elif owner != -1:
                enemy_centroid_x += planet_positions[planet_index, 0]
                enemy_centroid_y += planet_positions[planet_index, 1]
                enemy_count += 1

        if owned_count > 0:
            owned_centroid_x /= owned_count
            owned_centroid_y /= owned_count
        if enemy_count > 0:
            enemy_centroid_x /= enemy_count
            enemy_centroid_y /= enemy_count

        nearest_owned = np.full(planet_count, math.inf, dtype=np.float64)
        nearest_enemy = np.full(planet_count, math.inf, dtype=np.float64)
        nearest_neutral = np.full(planet_count, math.inf, dtype=np.float64)
        nearest_owned_valid = np.zeros(planet_count, dtype=np.bool_)
        nearest_enemy_valid = np.zeros(planet_count, dtype=np.bool_)
        nearest_neutral_valid = np.zeros(planet_count, dtype=np.bool_)

        for source_index in range(planet_count):
            source_x = planet_positions[source_index, 0]
            source_y = planet_positions[source_index, 1]
            source_radius = planet_radii[source_index]
            centered_x = source_x - BOARD_CENTER
            centered_y = source_y - BOARD_CENTER
            orbital_radius = math.hypot(centered_x, centered_y)
            planet_values[source_index, 0] = orbital_radius
            planet_values[source_index, 6] = orbital_radius
            planet_masks[source_index, 0] = True
            planet_masks[source_index, 6] = True
            if orbital_radius > 0.0:
                planet_values[source_index, 1] = centered_y / orbital_radius
                planet_values[source_index, 2] = centered_x / orbital_radius
                planet_masks[source_index, 1] = True
                planet_masks[source_index, 2] = True

            for target_index in range(planet_count):
                target_x = planet_positions[target_index, 0]
                target_y = planet_positions[target_index, 1]
                delta_x = target_x - source_x
                delta_y = target_y - source_y
                distance = math.hypot(delta_x, delta_y)
                surface_distance = max(
                    distance - source_radius - planet_radii[target_index],
                    0.0,
                )

                planet_pair_values[source_index, target_index, 0] = delta_x
                planet_pair_values[source_index, target_index, 1] = delta_y
                planet_pair_values[source_index, target_index, 2] = distance
                planet_pair_values[source_index, target_index, 3] = surface_distance
                if distance > 0.0:
                    planet_pair_values[source_index, target_index, 4] = delta_y / distance
                    planet_pair_values[source_index, target_index, 5] = delta_x / distance
                else:
                    planet_pair_masks[source_index, target_index, 4] = False
                    planet_pair_masks[source_index, target_index, 5] = False

                relative_velocity_x = (
                    planet_velocity[target_index, 0] - planet_velocity[source_index, 0]
                )
                relative_velocity_y = (
                    planet_velocity[target_index, 1] - planet_velocity[source_index, 1]
                )
                planet_pair_values[source_index, target_index, 6] = relative_velocity_x
                planet_pair_values[source_index, target_index, 7] = relative_velocity_y
                planet_pair_values[source_index, target_index, 8] = math.hypot(
                    relative_velocity_x,
                    relative_velocity_y,
                )

                if source_index == target_index:
                    continue
                target_owner = planet_owners[target_index]
                if target_owner == player_id:
                    nearest_owned_valid[source_index] = True
                    if distance < nearest_owned[source_index]:
                        nearest_owned[source_index] = distance
                elif target_owner == -1:
                    nearest_neutral_valid[source_index] = True
                    if distance < nearest_neutral[source_index]:
                        nearest_neutral[source_index] = distance
                else:
                    nearest_enemy_valid[source_index] = True
                    if distance < nearest_enemy[source_index]:
                        nearest_enemy[source_index] = distance

        for planet_index in range(planet_count):
            if nearest_owned_valid[planet_index]:
                planet_values[planet_index, 3] = nearest_owned[planet_index]
                planet_masks[planet_index, 3] = True
            if nearest_enemy_valid[planet_index]:
                planet_values[planet_index, 4] = nearest_enemy[planet_index]
                planet_masks[planet_index, 4] = True
            if nearest_neutral_valid[planet_index]:
                planet_values[planet_index, 5] = nearest_neutral[planet_index]
                planet_masks[planet_index, 5] = True

        for fleet_index in range(fleet_count):
            fleet_x = fleet_positions[fleet_index, 0]
            fleet_y = fleet_positions[fleet_index, 1]
            speed = fleet_speed[fleet_index]
            velocity_x = speed * heading_cos[fleet_index]
            velocity_y = speed * heading_sin[fleet_index]
            fleet_values[fleet_index, 0] = velocity_x
            fleet_values[fleet_index, 1] = velocity_y
            fleet_values[fleet_index, 3] = math.hypot(
                fleet_x - BOARD_CENTER,
                fleet_y - BOARD_CENTER,
            )
            fleet_masks[fleet_index, 0] = True
            fleet_masks[fleet_index, 1] = True
            fleet_masks[fleet_index, 3] = True

            source_valid = False
            source_distance = 0.0
            nearest_planet = math.inf
            nearest_owned_fleet = math.inf
            nearest_enemy_fleet = math.inf
            nearest_neutral_fleet = math.inf
            nearest_planet_valid = False
            nearest_owned_fleet_valid = False
            nearest_enemy_fleet_valid = False
            nearest_neutral_fleet_valid = False

            if owned_count > 0:
                fleet_values[fleet_index, 4] = math.hypot(
                    fleet_x - owned_centroid_x,
                    fleet_y - owned_centroid_y,
                )
                fleet_masks[fleet_index, 4] = True
            if enemy_count > 0:
                fleet_values[fleet_index, 5] = math.hypot(
                    fleet_x - enemy_centroid_x,
                    fleet_y - enemy_centroid_y,
                )
                fleet_masks[fleet_index, 5] = True

            for planet_index in range(planet_count):
                planet_x = planet_positions[planet_index, 0]
                planet_y = planet_positions[planet_index, 1]
                delta_x = planet_x - fleet_x
                delta_y = planet_y - fleet_y
                distance = math.hypot(delta_x, delta_y)
                surface_distance = max(distance - planet_radii[planet_index], 0.0)
                direction_sin = 0.0
                direction_cos = 0.0
                if distance > 0.0:
                    direction_sin = delta_y / distance
                    direction_cos = delta_x / distance
                else:
                    fleet_planet_masks[fleet_index, planet_index, 4] = False
                    fleet_planet_masks[fleet_index, planet_index, 5] = False
                    fleet_planet_masks[fleet_index, planet_index, 6] = False

                alignment = (
                    heading_cos[fleet_index] * direction_cos
                    + heading_sin[fleet_index] * direction_sin
                )
                cross_track = abs(
                    heading_cos[fleet_index] * delta_y
                    - heading_sin[fleet_index] * delta_x
                )
                relative_velocity_x = planet_velocity[planet_index, 0] - velocity_x
                relative_velocity_y = planet_velocity[planet_index, 1] - velocity_y
                static_eta = 0.0
                if speed > 0.0:
                    static_eta = distance / speed

                fleet_planet_values[fleet_index, planet_index, 0] = delta_x
                fleet_planet_values[fleet_index, planet_index, 1] = delta_y
                fleet_planet_values[fleet_index, planet_index, 2] = distance
                fleet_planet_values[fleet_index, planet_index, 3] = surface_distance
                fleet_planet_values[fleet_index, planet_index, 4] = direction_sin
                fleet_planet_values[fleet_index, planet_index, 5] = direction_cos
                fleet_planet_values[fleet_index, planet_index, 6] = alignment
                fleet_planet_values[fleet_index, planet_index, 7] = cross_track
                fleet_planet_values[fleet_index, planet_index, 8] = relative_velocity_x
                fleet_planet_values[fleet_index, planet_index, 9] = relative_velocity_y
                fleet_planet_values[fleet_index, planet_index, 10] = math.hypot(
                    relative_velocity_x,
                    relative_velocity_y,
                )
                fleet_planet_values[fleet_index, planet_index, 11] = static_eta

                nearest_planet_valid = True
                if distance < nearest_planet:
                    nearest_planet = distance

                owner = planet_owners[planet_index]
                if owner == player_id:
                    nearest_owned_fleet_valid = True
                    if distance < nearest_owned_fleet:
                        nearest_owned_fleet = distance
                elif owner == -1:
                    nearest_neutral_fleet_valid = True
                    if distance < nearest_neutral_fleet:
                        nearest_neutral_fleet = distance
                else:
                    nearest_enemy_fleet_valid = True
                    if distance < nearest_enemy_fleet:
                        nearest_enemy_fleet = distance

                if fleet_source_ids[fleet_index] == planet_ids[planet_index]:
                    source_valid = True
                    source_distance = distance

            if source_valid:
                fleet_values[fleet_index, 2] = source_distance
                fleet_masks[fleet_index, 2] = True
            if nearest_planet_valid:
                fleet_values[fleet_index, 6] = nearest_planet
                fleet_masks[fleet_index, 6] = True
            if nearest_owned_fleet_valid:
                fleet_values[fleet_index, 7] = nearest_owned_fleet
                fleet_masks[fleet_index, 7] = True
            if nearest_enemy_fleet_valid:
                fleet_values[fleet_index, 8] = nearest_enemy_fleet
                fleet_masks[fleet_index, 8] = True
            if nearest_neutral_fleet_valid:
                fleet_values[fleet_index, 9] = nearest_neutral_fleet
                fleet_masks[fleet_index, 9] = True

        return (
            planet_values,
            planet_masks,
            fleet_values,
            fleet_masks,
            planet_pair_values,
            planet_pair_masks,
            fleet_planet_values,
            fleet_planet_masks,
        )


    def warm_geometry_kernels() -> bool:
        geometry_kernel(
            np.asarray([[50.0, 50.0]], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            0,
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            np.asarray([[40.0, 50.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )
        return True

else:
    geometry_kernel = None

    def warm_geometry_kernels() -> bool:
        return False
