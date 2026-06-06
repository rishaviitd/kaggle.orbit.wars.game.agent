from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only when numba is absent.
    njit = None

BOARD_CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_FLEET_SPEED = 6.0


def is_numba_available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True)
    def _fleet_speed_scalar(ships: float) -> float:
        ships_value = max(ships, 1.0)
        if ships <= 1.0:
            return 1.0
        ratio = math.log(ships_value) / math.log(1000.0)
        scaled = 1.0 + (MAX_FLEET_SPEED - 1.0) * (ratio**1.5)
        return min(scaled, MAX_FLEET_SPEED)


    @njit(cache=True)
    def formula_kernel(
        current_step: int,
        episode_steps: int,
        player_id: int,
        planet_ids: np.ndarray,
        planet_owners: np.ndarray,
        planet_positions: np.ndarray,
        planet_ships: np.ndarray,
        planet_production: np.ndarray,
        initial_planets: np.ndarray,
        comet_planet_ids: np.ndarray,
        fleet_owners: np.ndarray,
        fleet_angles: np.ndarray,
        fleet_ships: np.ndarray,
        fleet_source_ids: np.ndarray,
        fleet_source_owners: np.ndarray,
        fleet_source_present: np.ndarray,
        comet_path_indices: np.ndarray,
        comet_path_lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        planet_count = planet_ids.shape[0]
        fleet_count = fleet_owners.shape[0]
        comet_count = comet_path_indices.shape[0]

        global_values = np.zeros(4, dtype=np.float32)
        planet_values = np.zeros((planet_count, 6), dtype=np.float32)
        fleet_values = np.zeros((fleet_count, 8), dtype=np.float32)
        planet_pair_values = np.zeros((planet_count, planet_count, 8), dtype=np.float32)
        fleet_planet_values = np.zeros((fleet_count, planet_count, 4), dtype=np.float32)
        comet_values = np.zeros((comet_count, 1), dtype=np.float32)

        remaining_steps = max(float(episode_steps) - float(current_step), 0.0)
        game_progress = min(
            max(float(current_step) / max(1.0, float(episode_steps)), 0.0),
            1.0,
        )
        friendly_stationed = 0.0
        enemy_stationed = 0.0
        for planet_index in range(planet_count):
            owner = planet_owners[planet_index]
            ships = planet_ships[planet_index]
            if owner == player_id:
                friendly_stationed += ships
            elif owner != -1:
                enemy_stationed += ships

        friendly_fleets = 0.0
        enemy_fleets = 0.0
        for fleet_index in range(fleet_count):
            owner = fleet_owners[fleet_index]
            ships = fleet_ships[fleet_index]
            if owner == player_id:
                friendly_fleets += ships
            elif owner != -1:
                enemy_fleets += ships

        global_values[0] = remaining_steps
        global_values[1] = game_progress
        global_values[2] = friendly_stationed + friendly_fleets
        global_values[3] = enemy_stationed + enemy_fleets

        for planet_index in range(planet_count):
            planet_id = planet_ids[planet_index]
            owner = planet_owners[planet_index]
            is_comet = False
            for comet_index in range(comet_planet_ids.shape[0]):
                if planet_id == comet_planet_ids[comet_index]:
                    is_comet = True
                    break

            initial_present = False
            initial_x = 0.0
            initial_y = 0.0
            initial_radius = 0.0
            for initial_index in range(initial_planets.shape[0]):
                if planet_id == int(initial_planets[initial_index, 0]):
                    initial_present = True
                    initial_x = initial_planets[initial_index, 2]
                    initial_y = initial_planets[initial_index, 3]
                    initial_radius = initial_planets[initial_index, 4]
                    break

            is_orbiting = False
            is_static = False
            if initial_present and not is_comet:
                orbital_radius = math.hypot(
                    initial_x - BOARD_CENTER,
                    initial_y - BOARD_CENTER,
                )
                is_orbiting = (
                    orbital_radius + initial_radius
                    < ROTATION_RADIUS_LIMIT - 1e-6
                )
                is_static = not is_orbiting

            planet_values[planet_index, 0] = owner == player_id
            planet_values[planet_index, 1] = owner == -1
            planet_values[planet_index, 2] = owner != player_id and owner != -1
            planet_values[planet_index, 3] = is_static
            planet_values[planet_index, 4] = is_orbiting
            planet_values[planet_index, 5] = is_comet

        for fleet_index in range(fleet_count):
            owner = fleet_owners[fleet_index]
            source_present = fleet_source_present[fleet_index]
            source_owner = fleet_source_owners[fleet_index]
            fleet_values[fleet_index, 0] = _fleet_speed_scalar(
                fleet_ships[fleet_index]
            )
            fleet_values[fleet_index, 1] = math.sin(fleet_angles[fleet_index])
            fleet_values[fleet_index, 2] = math.cos(fleet_angles[fleet_index])
            fleet_values[fleet_index, 3] = owner == player_id
            fleet_values[fleet_index, 4] = owner != player_id
            fleet_values[fleet_index, 5] = (
                source_present and source_owner == player_id
            )
            fleet_values[fleet_index, 6] = source_present and source_owner == -1
            fleet_values[fleet_index, 7] = (
                source_present
                and source_owner != player_id
                and source_owner != -1
            )

        for source_index in range(planet_count):
            source_owner = planet_owners[source_index]
            source_x = planet_positions[source_index, 0]
            source_y = planet_positions[source_index, 1]
            source_ships = planet_ships[source_index]
            source_production = planet_production[source_index]
            source_quadrant_x = source_x >= BOARD_CENTER
            source_quadrant_y = source_y >= BOARD_CENTER
            for target_index in range(planet_count):
                target_owner = planet_owners[target_index]
                target_x = planet_positions[target_index, 0]
                target_y = planet_positions[target_index, 1]
                target_ships = planet_ships[target_index]
                target_production = planet_production[target_index]
                target_quadrant_x = target_x >= BOARD_CENTER
                target_quadrant_y = target_y >= BOARD_CENTER

                planet_pair_values[source_index, target_index, 0] = (
                    source_owner == target_owner
                )
                planet_pair_values[source_index, target_index, 1] = (
                    source_owner == player_id and target_owner == player_id
                )
                planet_pair_values[source_index, target_index, 2] = (
                    source_owner == player_id and target_owner == -1
                )
                planet_pair_values[source_index, target_index, 3] = (
                    source_owner == player_id
                    and target_owner != player_id
                    and target_owner != -1
                )
                planet_pair_values[source_index, target_index, 4] = (
                    source_quadrant_x == target_quadrant_x
                    and source_quadrant_y == target_quadrant_y
                )
                planet_pair_values[source_index, target_index, 5] = (
                    target_production - source_production
                )
                planet_pair_values[source_index, target_index, 6] = (
                    source_ships - target_ships
                )
                planet_pair_values[source_index, target_index, 7] = (
                    source_ships / max(target_ships, 1.0)
                )

        for fleet_index in range(fleet_count):
            fleet_owner = fleet_owners[fleet_index]
            source_id = fleet_source_ids[fleet_index]
            for planet_index in range(planet_count):
                planet_owner = planet_owners[planet_index]
                fleet_planet_values[fleet_index, planet_index, 0] = (
                    fleet_owner == planet_owner
                )
                fleet_planet_values[fleet_index, planet_index, 1] = (
                    source_id == planet_ids[planet_index]
                )
                fleet_planet_values[fleet_index, planet_index, 2] = (
                    fleet_owner == planet_owner
                )
                fleet_planet_values[fleet_index, planet_index, 3] = (
                    fleet_owner != planet_owner
                )

        for comet_index in range(comet_count):
            comet_values[comet_index, 0] = comet_path_indices[comet_index] / max(
                comet_path_lengths[comet_index],
                1.0,
            )

        return (
            global_values,
            planet_values,
            fleet_values,
            planet_pair_values,
            fleet_planet_values,
            comet_values,
        )


    def warm_formula_kernels() -> bool:
        formula_kernel(
            0,
            500,
            0,
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([[50.0, 50.0]], dtype=np.float32),
            np.asarray([10.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([[0.0, 0.0, 50.0, 50.0, 1.0, 10.0, 1.0]], dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([10.0], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([True], dtype=np.bool_),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        return True

else:
    formula_kernel = None

    def warm_formula_kernels() -> bool:
        return False
