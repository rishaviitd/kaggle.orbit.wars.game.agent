from __future__ import annotations

from typing import Any

import numpy as np

GLOBAL_DIRECT_FEATURES = (
    "current_step",
    "player_count",
    "angular_velocity",
)

PLANET_DIRECT_FEATURES = (
    "planet_x",
    "planet_y",
    "planet_radius",
    "planet_production",
    "planet_stationed_ships",
    "initial_stationed_ships",
)

FLEET_DIRECT_FEATURES = (
    "fleet_x",
    "fleet_y",
    "fleet_ship_count",
    "fleet_source_production",
    "fleet_source_current_ships",
)

COMET_DIRECT_FEATURES = ("comet_path_index",)

PLANET_WIDTH = 7
FLEET_WIDTH = 7


def _entity_matrix(rows: Any, width: int) -> np.ndarray:
    if rows is None or len(rows) == 0:
        return np.empty((0, width), dtype=np.float32)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError(f"Expected entity rows with width {width}, got {matrix.shape}.")
    return matrix


def _comet_path_length(comet: dict[str, Any]) -> int:
    paths = comet.get("paths") or []
    lengths = [len(path) for path in paths if path is not None]
    return max(lengths, default=0)


def extract_direct_features(row: dict[str, Any]) -> dict[str, Any]:
    """Extract typed direct features and linking metadata from one Parquet row."""
    planets = _entity_matrix(row.get("planets"), PLANET_WIDTH)
    initial_planets = _entity_matrix(row.get("initial_planets"), PLANET_WIDTH)
    fleets = _entity_matrix(row.get("fleets"), FLEET_WIDTH)

    planet_ids = planets[:, 0].astype(np.int32, copy=False)
    planet_owners = planets[:, 1].astype(np.int32, copy=False)
    initial_by_id = {
        int(initial[0]): initial
        for initial in initial_planets
    }
    current_by_id = {
        int(planet[0]): planet
        for planet in planets
    }

    initial_ships = np.asarray(
        [
            initial_by_id.get(int(planet_id), planet)[5]
            for planet_id, planet in zip(planet_ids, planets, strict=True)
        ],
        dtype=np.float32,
    )
    planet_values = np.column_stack(
        (
            planets[:, 2],
            planets[:, 3],
            planets[:, 4],
            planets[:, 6],
            planets[:, 5],
            initial_ships,
        )
    ).astype(np.float32, copy=False)

    fleet_ids = fleets[:, 0].astype(np.int32, copy=False)
    fleet_owners = fleets[:, 1].astype(np.int32, copy=False)
    fleet_source_ids = fleets[:, 5].astype(np.int32, copy=False)
    source_rows = [current_by_id.get(int(source_id)) for source_id in fleet_source_ids]
    source_present = np.asarray([source is not None for source in source_rows], dtype=np.bool_)
    source_owners = np.asarray(
        [int(source[1]) if source is not None else -2 for source in source_rows],
        dtype=np.int32,
    )
    source_production = np.asarray(
        [float(source[6]) if source is not None else 0.0 for source in source_rows],
        dtype=np.float32,
    )
    source_ships = np.asarray(
        [float(source[5]) if source is not None else 0.0 for source in source_rows],
        dtype=np.float32,
    )
    fleet_values = np.column_stack(
        (
            fleets[:, 2],
            fleets[:, 3],
            fleets[:, 6],
            source_production,
            source_ships,
        )
    ).astype(np.float32, copy=False)

    comets = row.get("comets") or []
    comet_group_ids = np.arange(len(comets), dtype=np.int32)
    comet_path_indices = np.asarray(
        [int(comet.get("path_index", 0) or 0) for comet in comets],
        dtype=np.float32,
    )
    comet_path_lengths = np.asarray(
        [_comet_path_length(comet) for comet in comets],
        dtype=np.int32,
    )
    comet_values = comet_path_indices.reshape(-1, 1)

    global_values = np.asarray(
        [
            float(row.get("step", 0) or 0),
            float(row.get("player_count", 0) or 0),
            float(row.get("angular_velocity", 0.0) or 0.0),
        ],
        dtype=np.float32,
    )

    return {
        "feature_names": {
            "global": GLOBAL_DIRECT_FEATURES,
            "planet": PLANET_DIRECT_FEATURES,
            "fleet": FLEET_DIRECT_FEATURES,
            "comet": COMET_DIRECT_FEATURES,
        },
        "values": {
            "global": global_values,
            "planet": planet_values,
            "fleet": fleet_values,
            "comet": comet_values,
        },
        "metadata": {
            "game_id": str(row.get("game_id", "")),
            "date": str(row.get("date", "")),
            "step": int(row.get("step", 0) or 0),
            "player_count": int(row.get("player_count", 0) or 0),
            "player_id": int(row.get("player", row.get("expert_player_id", 0)) or 0),
            "planet_ids": planet_ids,
            "fleet_ids": fleet_ids,
            "fleet_source_planet_ids": fleet_source_ids,
            "comet_group_ids": comet_group_ids,
            "action": row.get("action") or [],
        },
        "context": {
            "planet_owners": planet_owners,
            "planet_positions": planets[:, 2:4],
            "planet_radii": planets[:, 4],
            "planet_ships": planets[:, 5],
            "planet_production": planets[:, 6],
            "initial_planets": initial_planets,
            "comet_planet_ids": np.asarray(
                row.get("comet_planet_ids") or [],
                dtype=np.int32,
            ),
            "fleet_owners": fleet_owners,
            "fleet_angles": fleets[:, 4],
            "fleet_ships": fleets[:, 6],
            "fleet_source_owners": source_owners,
            "fleet_source_present": source_present,
            "comet_path_lengths": comet_path_lengths,
            "comets": comets,
        },
    }
