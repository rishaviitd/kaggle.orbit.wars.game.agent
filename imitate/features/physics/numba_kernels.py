from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only when numba is absent.
    njit = None

BOARD_SIZE = 100.0
BOARD_CENTER = 50.0
MAX_FLEET_SPEED = 6.0
ROTATION_RADIUS_LIMIT = 50.0
COLLISION_FILTER_EPSILON = 1e-7

MOTION_STATIC = 0
MOTION_ORBITING = 1
MOTION_COMET = 2
MOTION_UNKNOWN = 3


def is_numba_available() -> bool:
    return njit is not None


if njit is not None:

    @njit(cache=True)
    def _fleet_speed_numba(ships: float) -> float:
        ships_int = max(1, int(ships))
        if ships_int <= 1:
            return 1.0
        ratio = math.log(float(ships_int)) / math.log(1000.0)
        speed = 1.0 + (MAX_FLEET_SPEED - 1.0) * (ratio**1.5)
        if speed > MAX_FLEET_SPEED:
            return MAX_FLEET_SPEED
        return speed


    @njit(cache=True)
    def _board_exit_time_numba(
        start_x: float,
        start_y: float,
        direction_x: float,
        direction_y: float,
        speed: float,
    ) -> float:
        velocity_x = direction_x * speed
        velocity_y = direction_y * speed
        best = math.inf

        if velocity_x > 0.0:
            candidate = (BOARD_SIZE - start_x) / velocity_x
            if candidate >= 0.0 and candidate < best:
                best = candidate
        elif velocity_x < 0.0:
            candidate = (0.0 - start_x) / velocity_x
            if candidate >= 0.0 and candidate < best:
                best = candidate

        if velocity_y > 0.0:
            candidate = (BOARD_SIZE - start_y) / velocity_y
            if candidate >= 0.0 and candidate < best:
                best = candidate
        elif velocity_y < 0.0:
            candidate = (0.0 - start_y) / velocity_y
            if candidate >= 0.0 and candidate < best:
                best = candidate

        return best


    @njit(cache=True)
    def _endpoint_is_inside_numba(
        start_x: float,
        start_y: float,
        velocity_x: float,
        velocity_y: float,
        turn: int,
    ) -> bool:
        x = start_x + velocity_x * turn
        y = start_y + velocity_y * turn
        return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE


    @njit(cache=True)
    def _fleet_search_horizon_numba(
        start_x: float,
        start_y: float,
        velocity_x: float,
        velocity_y: float,
        lookahead: int,
    ) -> int:
        speed = math.hypot(velocity_x, velocity_y)
        if speed <= 0.0:
            return 0

        exit_time = _board_exit_time_numba(
            start_x,
            start_y,
            velocity_x / speed,
            velocity_y / speed,
            speed,
        )
        if math.isinf(exit_time):
            horizon = int(lookahead)
        else:
            horizon = min(
                int(lookahead),
                max(0, int(math.floor(exit_time + COLLISION_FILTER_EPSILON))),
            )

        while horizon > 0 and not _endpoint_is_inside_numba(
            start_x,
            start_y,
            velocity_x,
            velocity_y,
            horizon,
        ):
            horizon -= 1
        while horizon < int(lookahead) and _endpoint_is_inside_numba(
            start_x,
            start_y,
            velocity_x,
            velocity_y,
            horizon + 1,
        ):
            horizon += 1
        return horizon


    @njit(cache=True)
    def _ray_circle_interval_numba(
        start_x: float,
        start_y: float,
        velocity_x: float,
        velocity_y: float,
        center_x: float,
        center_y: float,
        radius: float,
        max_time: float,
    ) -> tuple[bool, float, float]:
        relative_x = start_x - center_x
        relative_y = start_y - center_y
        a = velocity_x * velocity_x + velocity_y * velocity_y
        if a <= 0.0:
            return False, 0.0, 0.0

        b = 2.0 * (relative_x * velocity_x + relative_y * velocity_y)
        c = relative_x * relative_x + relative_y * relative_y - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < -COLLISION_FILTER_EPSILON:
            return False, 0.0, 0.0

        sqrt_discriminant = math.sqrt(max(0.0, discriminant))
        first = (-b - sqrt_discriminant) / (2.0 * a)
        second = (-b + sqrt_discriminant) / (2.0 * a)
        interval_start = max(0.0, min(first, second))
        interval_end = min(float(max_time), max(first, second))
        if interval_end + COLLISION_FILTER_EPSILON < interval_start:
            return False, 0.0, 0.0
        return True, interval_start, interval_end


    @njit(cache=True)
    def _candidate_turn_bounds_numba(
        interval_start: float,
        interval_end: float,
        horizon: int,
        padding: int,
    ) -> tuple[bool, int, int]:
        first_turn = max(1, int(math.floor(interval_start)) - int(padding))
        last_turn = min(int(horizon), int(math.ceil(interval_end)) + int(padding))
        if last_turn < first_turn:
            return False, 0, -1
        return True, first_turn, last_turn


    @njit(cache=True)
    def _cross_2d_numba(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
        return a_x * b_y - a_y * b_x


    @njit(cache=True)
    def _segment_intersection_progress_numba(
        start_a_x: float,
        start_a_y: float,
        end_a_x: float,
        end_a_y: float,
        start_b_x: float,
        start_b_y: float,
        end_b_x: float,
        end_b_y: float,
    ) -> tuple[bool, float]:
        r_x = end_a_x - start_a_x
        r_y = end_a_y - start_a_y
        s_x = end_b_x - start_b_x
        s_y = end_b_y - start_b_y
        denominator = _cross_2d_numba(r_x, r_y, s_x, s_y)
        if abs(denominator) < 1e-12:
            return False, 0.0

        qp_x = start_b_x - start_a_x
        qp_y = start_b_y - start_a_y
        progress_a = _cross_2d_numba(qp_x, qp_y, s_x, s_y) / denominator
        progress_b = _cross_2d_numba(qp_x, qp_y, r_x, r_y) / denominator
        if (
            -1e-9 <= progress_a <= 1.0 + 1e-9
            and -1e-9 <= progress_b <= 1.0 + 1e-9
        ):
            return True, max(0.0, min(1.0, progress_a))
        return False, 0.0


    @njit(cache=True)
    def _point_to_segment_distance_numba(
        point_x: float,
        point_y: float,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> float:
        length_sq = (start_x - end_x) ** 2 + (start_y - end_y) ** 2
        if length_sq == 0.0:
            return math.hypot(point_x - start_x, point_y - start_y)

        progress = (
            (point_x - start_x) * (end_x - start_x)
            + (point_y - start_y) * (end_y - start_y)
        ) / length_sq
        progress = max(0.0, min(1.0, progress))
        projection_x = start_x + progress * (end_x - start_x)
        projection_y = start_y + progress * (end_y - start_y)
        return math.hypot(point_x - projection_x, point_y - projection_y)


    @njit(cache=True)
    def _point_to_segment_progress_numba(
        point_x: float,
        point_y: float,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> float:
        length_sq = (start_x - end_x) ** 2 + (start_y - end_y) ** 2
        if length_sq == 0.0:
            return 0.0
        progress = (
            (point_x - start_x) * (end_x - start_x)
            + (point_y - start_y) * (end_y - start_y)
        ) / length_sq
        return max(0.0, min(1.0, progress))


    @njit(cache=True)
    def _segment_to_segment_distance_numba(
        start_a_x: float,
        start_a_y: float,
        end_a_x: float,
        end_a_y: float,
        start_b_x: float,
        start_b_y: float,
        end_b_x: float,
        end_b_y: float,
    ) -> float:
        intersects, _progress = _segment_intersection_progress_numba(
            start_a_x,
            start_a_y,
            end_a_x,
            end_a_y,
            start_b_x,
            start_b_y,
            end_b_x,
            end_b_y,
        )
        if intersects:
            return 0.0

        return min(
            _point_to_segment_distance_numba(
                start_a_x,
                start_a_y,
                start_b_x,
                start_b_y,
                end_b_x,
                end_b_y,
            ),
            _point_to_segment_distance_numba(
                end_a_x,
                end_a_y,
                start_b_x,
                start_b_y,
                end_b_x,
                end_b_y,
            ),
            _point_to_segment_distance_numba(
                start_b_x,
                start_b_y,
                start_a_x,
                start_a_y,
                end_a_x,
                end_a_y,
            ),
            _point_to_segment_distance_numba(
                end_b_x,
                end_b_y,
                start_a_x,
                start_a_y,
                end_a_x,
                end_a_y,
            ),
        )


    @njit(cache=True)
    def _segment_collision_progress_numba(
        fleet_start_x: float,
        fleet_start_y: float,
        fleet_end_x: float,
        fleet_end_y: float,
        target_start_x: float,
        target_start_y: float,
        target_end_x: float,
        target_end_y: float,
    ) -> float:
        intersects, intersection_progress = _segment_intersection_progress_numba(
            fleet_start_x,
            fleet_start_y,
            fleet_end_x,
            fleet_end_y,
            target_start_x,
            target_start_y,
            target_end_x,
            target_end_y,
        )
        if intersects:
            return intersection_progress
        return min(
            _point_to_segment_progress_numba(
                target_start_x,
                target_start_y,
                fleet_start_x,
                fleet_start_y,
                fleet_end_x,
                fleet_end_y,
            ),
            _point_to_segment_progress_numba(
                target_end_x,
                target_end_y,
                fleet_start_x,
                fleet_start_y,
                fleet_end_x,
                fleet_end_y,
            ),
        )


    @njit(cache=True)
    def _planet_position_after_moves_numba(
        planet_index: int,
        moves_done: int,
        planet_x: np.ndarray,
        planet_y: np.ndarray,
        motion_types: np.ndarray,
        orbit_radius: np.ndarray,
        orbit_initial_angle: np.ndarray,
        angular_velocity: float,
        current_step: int,
        comet_x: np.ndarray,
        comet_y: np.ndarray,
    ) -> tuple[bool, float, float]:
        if moves_done <= 0:
            return True, planet_x[planet_index], planet_y[planet_index]

        motion_type = motion_types[planet_index]
        if motion_type == MOTION_COMET:
            if moves_done >= comet_x.shape[1]:
                return False, 0.0, 0.0
            x = comet_x[planet_index, moves_done]
            y = comet_y[planet_index, moves_done]
            if math.isnan(x) or math.isnan(y):
                return False, 0.0, 0.0
            return True, x, y

        if motion_type == MOTION_ORBITING:
            env_step = max(1, int(current_step)) + moves_done - 1
            future_angle = orbit_initial_angle[planet_index] + angular_velocity * env_step
            return (
                True,
                BOARD_CENTER + orbit_radius[planet_index] * math.cos(future_angle),
                BOARD_CENTER + orbit_radius[planet_index] * math.sin(future_angle),
            )

        return True, planet_x[planet_index], planet_y[planet_index]


    @njit(cache=True)
    def predict_fleet_hits_kernel(
        fleet_x: np.ndarray,
        fleet_y: np.ndarray,
        fleet_angles: np.ndarray,
        fleet_ships: np.ndarray,
        planet_x: np.ndarray,
        planet_y: np.ndarray,
        planet_radius: np.ndarray,
        motion_types: np.ndarray,
        orbit_radius: np.ndarray,
        orbit_initial_angle: np.ndarray,
        comet_x: np.ndarray,
        comet_y: np.ndarray,
        angular_velocity: float,
        current_step: int,
        lookahead: int,
        compute_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fleet_count = fleet_x.shape[0]
        planet_count = planet_x.shape[0]
        target_indices = np.full(fleet_count, -1, dtype=np.int32)
        hit_turns = np.zeros(fleet_count, dtype=np.int32)
        hit_distances = np.zeros(fleet_count, dtype=np.float64)

        first_turns = np.empty(planet_count, dtype=np.int32)
        last_turns = np.empty(planet_count, dtype=np.int32)
        has_inner = np.empty(planet_count, dtype=np.bool_)
        inner_starts = np.empty(planet_count, dtype=np.float64)
        inner_ends = np.empty(planet_count, dtype=np.float64)

        for fleet_index in range(fleet_count):
            if not compute_mask[fleet_index]:
                continue

            speed = _fleet_speed_numba(fleet_ships[fleet_index])
            velocity_x = math.cos(fleet_angles[fleet_index]) * speed
            velocity_y = math.sin(fleet_angles[fleet_index]) * speed
            horizon = _fleet_search_horizon_numba(
                fleet_x[fleet_index],
                fleet_y[fleet_index],
                velocity_x,
                velocity_y,
                lookahead,
            )
            if horizon <= 0:
                continue

            for planet_index in range(planet_count):
                first_turns[planet_index] = 1
                last_turns[planet_index] = 0
                has_inner[planet_index] = False
                inner_starts[planet_index] = 0.0
                inner_ends[planet_index] = 0.0

                motion_type = motion_types[planet_index]
                if motion_type == MOTION_STATIC:
                    has_interval, interval_start, interval_end = _ray_circle_interval_numba(
                        fleet_x[fleet_index],
                        fleet_y[fleet_index],
                        velocity_x,
                        velocity_y,
                        planet_x[planet_index],
                        planet_y[planet_index],
                        planet_radius[planet_index] + COLLISION_FILTER_EPSILON,
                        horizon,
                    )
                    if not has_interval:
                        continue

                    has_bounds, first_turn, last_turn = _candidate_turn_bounds_numba(
                        interval_start,
                        interval_end,
                        horizon,
                        1,
                    )
                    if has_bounds:
                        first_turns[planet_index] = first_turn
                        last_turns[planet_index] = last_turn
                    continue

                if motion_type == MOTION_ORBITING:
                    angular_step = min(math.pi, abs(angular_velocity))
                    outer_radius = (
                        orbit_radius[planet_index]
                        + planet_radius[planet_index]
                        + COLLISION_FILTER_EPSILON
                    )
                    inner_radius = max(
                        0.0,
                        orbit_radius[planet_index] * math.cos(angular_step / 2.0)
                        - planet_radius[planet_index]
                        - COLLISION_FILTER_EPSILON,
                    )
                    has_outer, outer_start, outer_end = _ray_circle_interval_numba(
                        fleet_x[fleet_index],
                        fleet_y[fleet_index],
                        velocity_x,
                        velocity_y,
                        BOARD_CENTER,
                        BOARD_CENTER,
                        outer_radius,
                        horizon,
                    )
                    if not has_outer:
                        continue

                    if inner_radius > COLLISION_FILTER_EPSILON:
                        has_inner_interval, inner_start, inner_end = (
                            _ray_circle_interval_numba(
                                fleet_x[fleet_index],
                                fleet_y[fleet_index],
                                velocity_x,
                                velocity_y,
                                BOARD_CENTER,
                                BOARD_CENTER,
                                inner_radius,
                                horizon,
                            )
                        )
                        if has_inner_interval:
                            has_inner[planet_index] = True
                            inner_starts[planet_index] = inner_start
                            inner_ends[planet_index] = inner_end

                    has_bounds, first_turn, last_turn = _candidate_turn_bounds_numba(
                        outer_start,
                        outer_end,
                        horizon,
                        1,
                    )
                    if has_bounds:
                        first_turns[planet_index] = first_turn
                        last_turns[planet_index] = last_turn
                    continue

                first_turns[planet_index] = 1
                last_turns[planet_index] = horizon

            for turn in range(1, horizon + 1):
                fleet_start_x = fleet_x[fleet_index] + velocity_x * (turn - 1)
                fleet_start_y = fleet_y[fleet_index] + velocity_y * (turn - 1)
                fleet_end_x = fleet_start_x + velocity_x
                fleet_end_y = fleet_start_y + velocity_y
                best_progress = math.inf
                best_planet_index = -1

                for planet_index in range(planet_count):
                    if turn < first_turns[planet_index] or turn > last_turns[planet_index]:
                        continue
                    if (
                        has_inner[planet_index]
                        and float(turn - 1)
                        >= inner_starts[planet_index] + COLLISION_FILTER_EPSILON
                        and float(turn)
                        <= inner_ends[planet_index] - COLLISION_FILTER_EPSILON
                    ):
                        continue

                    old_ok, target_start_x, target_start_y = (
                        _planet_position_after_moves_numba(
                            planet_index,
                            max(0, turn - 1),
                            planet_x,
                            planet_y,
                            motion_types,
                            orbit_radius,
                            orbit_initial_angle,
                            angular_velocity,
                            current_step,
                            comet_x,
                            comet_y,
                        )
                    )
                    if not old_ok:
                        continue

                    new_ok, target_end_x, target_end_y = (
                        _planet_position_after_moves_numba(
                            planet_index,
                            turn,
                            planet_x,
                            planet_y,
                            motion_types,
                            orbit_radius,
                            orbit_initial_angle,
                            angular_velocity,
                            current_step,
                            comet_x,
                            comet_y,
                        )
                    )
                    if not new_ok:
                        continue

                    distance = _segment_to_segment_distance_numba(
                        fleet_start_x,
                        fleet_start_y,
                        fleet_end_x,
                        fleet_end_y,
                        target_start_x,
                        target_start_y,
                        target_end_x,
                        target_end_y,
                    )
                    if distance >= planet_radius[planet_index]:
                        continue

                    progress = _segment_collision_progress_numba(
                        fleet_start_x,
                        fleet_start_y,
                        fleet_end_x,
                        fleet_end_y,
                        target_start_x,
                        target_start_y,
                        target_end_x,
                        target_end_y,
                    )
                    if progress < best_progress:
                        best_progress = progress
                        best_planet_index = planet_index

                if best_planet_index >= 0:
                    target_indices[fleet_index] = best_planet_index
                    hit_turns[fleet_index] = turn
                    hit_distances[fleet_index] = speed * (
                        float(turn) - 1.0 + best_progress
                    )
                    break

        return target_indices, hit_turns, hit_distances

    def warm_numba_kernels() -> bool:
        """Compile the collision kernel once in the current process."""
        predict_fleet_hits_kernel(
            np.asarray([10.0], dtype=np.float64),
            np.asarray([10.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([10.0], dtype=np.float64),
            np.asarray([20.0], dtype=np.float64),
            np.asarray([10.0], dtype=np.float64),
            np.asarray([2.0], dtype=np.float64),
            np.asarray([MOTION_STATIC], dtype=np.int8),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.full((1, 2), np.nan, dtype=np.float64),
            np.full((1, 2), np.nan, dtype=np.float64),
            0.0,
            1,
            1,
            np.asarray([True], dtype=np.bool_),
        )
        return True

else:
    predict_fleet_hits_kernel = None

    def warm_numba_kernels() -> bool:
        return False
