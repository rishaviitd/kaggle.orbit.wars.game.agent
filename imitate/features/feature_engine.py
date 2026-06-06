from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from imitate.features.aggregates import extract_aggregate_features
from imitate.features.direct import extract_direct_features
from imitate.features.formula import extract_formula_features
from imitate.features.formula.extractor import USE_NUMBA_FORMULA_KERNEL
from imitate.features.formula.numba_kernels import warm_formula_kernels
from imitate.features.initial_state import extract_initial_state_features
from imitate.features.physics import (
    DEFAULT_COLLISION_LOOKAHEAD,
    extract_physics_features,
)
from imitate.features.physics.extractor import USE_NUMBA_COLLISION_KERNEL
from imitate.features.physics.numba_kernels import warm_numba_kernels
from imitate.features.player_relative import extract_player_relative_features
from imitate.features.vectorized_geometry import (
    extract_vectorized_geometry_features,
)
from imitate.features.vectorized_geometry.extractor import USE_NUMBA_GEOMETRY_KERNEL
from imitate.features.vectorized_geometry.numba_kernels import warm_geometry_kernels

DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = "auto"
DEFAULT_WORKER_FRACTION = 0.80
DEFAULT_NUMBA_WORKERS = 4
DEFAULT_MP_START_METHOD = "auto"


def extract_features(
    row: dict[str, Any],
    *,
    episode_steps: int = 500,
    physics_lookahead: int = DEFAULT_COLLISION_LOOKAHEAD,
    physics_collision_cache: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Extract the implemented feature stages for one snapshot."""
    direct = extract_direct_features(row)
    formula = extract_formula_features(direct, episode_steps=episode_steps)
    initial_state = extract_initial_state_features(direct)
    player_relative = extract_player_relative_features(direct)
    aggregates = extract_aggregate_features(direct, formula)
    vectorized_geometry = extract_vectorized_geometry_features(direct, formula)
    physics = extract_physics_features(
        direct,
        formula,
        lookahead=physics_lookahead,
        collision_cache=physics_collision_cache,
    )
    return {
        "metadata": direct["metadata"],
        "direct": {
            "feature_names": direct["feature_names"],
            "values": direct["values"],
        },
        "formula": formula,
        "initial_state": initial_state,
        "player_relative": player_relative,
        "aggregates": aggregates,
        "vectorized_geometry": vectorized_geometry,
        "physics": physics,
    }


def _extract_feature_chunk(
    task: tuple[list[dict[str, Any]], int, int],
) -> list[dict[str, Any]]:
    rows, episode_steps, physics_lookahead = task
    physics_collision_cache: dict[int, Any] = {}
    return [
        extract_features(
            row,
            episode_steps=episode_steps,
            physics_lookahead=physics_lookahead,
            physics_collision_cache=physics_collision_cache,
        )
        for row in rows
    ]


def _auto_worker_count(worker_fraction: float = DEFAULT_WORKER_FRACTION) -> int:
    cpu_count = os.cpu_count() or 1
    fraction = min(max(float(worker_fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return 1
    worker_count = max(1, min(cpu_count, int(round(cpu_count * fraction))))
    if USE_NUMBA_COLLISION_KERNEL:
        return max(1, min(worker_count, DEFAULT_NUMBA_WORKERS))
    return worker_count


def _resolve_worker_count(
    workers: int | str,
    *,
    worker_fraction: float = DEFAULT_WORKER_FRACTION,
) -> int:
    if isinstance(workers, str):
        normalized = workers.strip().lower()
        if normalized == "auto":
            return _auto_worker_count(worker_fraction)
        if normalized in {"all", "max"}:
            return max(1, os.cpu_count() or 1)
        try:
            return max(1, int(normalized))
        except ValueError as error:
            raise ValueError(
                "--workers must be an integer, 'auto', 'all', or 'max'."
            ) from error
    return max(1, int(workers))


def _resolve_mp_start_method(start_method: str) -> str:
    available_methods = set(mp.get_all_start_methods())
    normalized = start_method.strip().lower()
    if normalized == "auto":
        if USE_NUMBA_COLLISION_KERNEL and "fork" in available_methods:
            return "fork"
        return mp.get_start_method()
    if normalized not in available_methods:
        available = ", ".join(sorted(available_methods))
        raise ValueError(
            f"--mp-start-method must be 'auto' or one of: {available}."
        )
    return normalized


def _iter_game_row_chunks(
    parquet: pq.ParquetFile,
    *,
    batch_size: int,
    columns: tuple[str, ...],
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    pending_rows: list[dict[str, Any]] = []
    pending_game_id: str | None = None
    read_rows = 0

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in record_batch.to_pylist():
            if limit is not None and read_rows >= int(limit):
                break

            game_id = str(row.get("game_id", ""))
            if pending_rows and game_id != pending_game_id:
                yield {
                    "game_id": pending_game_id,
                    "rows": pending_rows,
                }
                pending_rows = []

            pending_rows.append(row)
            pending_game_id = game_id
            read_rows += 1

        if limit is not None and read_rows >= int(limit):
            break

    if pending_rows:
        yield {
            "game_id": pending_game_id,
            "rows": pending_rows,
        }


def iter_parquet_features(
    parquet_path: str | Path,
    *,
    episode_steps: int = 500,
    physics_lookahead: int = DEFAULT_COLLISION_LOOKAHEAD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int | str = DEFAULT_WORKERS,
    worker_fraction: float = DEFAULT_WORKER_FRACTION,
    mp_start_method: str = DEFAULT_MP_START_METHOD,
    limit: int | None = None,
    log_every: int = 1000,
) -> Iterator[dict[str, Any]]:
    """Stream minified Parquet snapshots without loading the full file into RAM."""
    path = Path(parquet_path)
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    target_rows = min(total_rows, limit) if limit is not None else total_rows
    worker_count = min(
        _resolve_worker_count(workers, worker_fraction=worker_fraction),
        max(1, int(target_rows)),
    )
    resolved_start_method = _resolve_mp_start_method(mp_start_method)
    if worker_count > 1 and resolved_start_method == "fork":
        warm_formula_kernels()
        warm_numba_kernels()
        warm_geometry_kernels()
    processed = 0
    started = time.perf_counter()
    collision_backend = (
        "NumPy + Numba formula/physics/geometry kernels"
        if (
            USE_NUMBA_FORMULA_KERNEL
            and USE_NUMBA_COLLISION_KERNEL
            and USE_NUMBA_GEOMETRY_KERNEL
        )
        else "NumPy + Numba physics/geometry kernels"
        if USE_NUMBA_COLLISION_KERNEL and USE_NUMBA_GEOMETRY_KERNEL
        else (
            "NumPy + Numba collision kernel"
            if USE_NUMBA_COLLISION_KERNEL
            else (
                "NumPy + Numba geometry kernel"
                if USE_NUMBA_GEOMETRY_KERNEL
                else "NumPy extraction"
            )
        )
    )

    print(f"[features] Input: {path}")
    print(f"[features] Rows: {target_rows}/{total_rows}")
    print(
        "[features] Stages: direct -> formula -> initial roles -> player-relative -> "
        "aggregates -> vectorized geometry -> physics"
    )
    print(
        f"[features] Backend: {collision_backend}; "
        f"workers={worker_count}; mp_start={resolved_start_method}; "
        "streaming Parquet batches"
    )

    columns = (
        "game_id",
        "date",
        "step",
        "player_count",
        "expert_player_id",
        "player",
        "planets",
        "initial_planets",
        "fleets",
        "angular_velocity",
        "comet_planet_ids",
        "comets",
        "action",
    )
    pool = None
    if worker_count > 1:
        pool = mp.get_context(resolved_start_method).Pool(processes=worker_count)
    completed = False
    try:
        game_chunks = _iter_game_row_chunks(
            parquet,
            batch_size=batch_size,
            columns=columns,
            limit=limit,
        )
        tasks = (
            (
                game_chunk["rows"],
                int(episode_steps),
                int(physics_lookahead),
            )
            for game_chunk in game_chunks
        )

        if pool is None:
            feature_chunks = map(_extract_feature_chunk, tasks)
        else:
            feature_chunks = pool.imap(_extract_feature_chunk, tasks, chunksize=1)

        for feature_chunk in feature_chunks:
            for features in feature_chunk:
                yield features
                processed += 1
                if processed == 1 or processed % max(1, log_every) == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"[features] Processed {processed}/{target_rows} snapshots "
                        f"({processed / elapsed:.1f} rows/s)"
                    )
            if limit is not None and processed >= limit:
                break
        completed = True
    finally:
        if pool is not None:
            if completed:
                pool.close()
            else:
                pool.terminate()
            pool.join()

    elapsed = max(time.perf_counter() - started, 1e-9)
    print(
        f"[features] Complete: {processed} snapshots in {elapsed:.2f}s "
        f"({processed / elapsed:.1f} rows/s)"
    )


def _shape_summary(features: dict[str, Any]) -> str:
    sections = []
    for stage in (
        "direct",
        "formula",
        "initial_state",
        "player_relative",
        "aggregates",
        "vectorized_geometry",
        "physics",
    ):
        shapes = ", ".join(
            f"{name}={tuple(value.shape)}"
            for name, value in features[stage]["values"].items()
        )
        sections.append(f"{stage}[{shapes}]")
    return " | ".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract implemented entity-transformer features from minified Parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/batch/batch-1.parquet"),
        help="Minified replay Parquet file.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--workers",
        default=DEFAULT_WORKERS,
        help=(
            "Worker processes for per-snapshot extraction. Use an integer, "
            "'auto' for about 80%% of CPU cores, or 'all'/'max'."
        ),
    )
    parser.add_argument(
        "--worker-fraction",
        type=float,
        default=DEFAULT_WORKER_FRACTION,
        help="CPU fraction used when --workers=auto.",
    )
    parser.add_argument(
        "--mp-start-method",
        default=DEFAULT_MP_START_METHOD,
        help=(
            "Multiprocessing start method. Use 'auto' for the optimized default, "
            "or choose from this Python install's supported methods."
        ),
    )
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--physics-lookahead",
        type=int,
        default=DEFAULT_COLLISION_LOOKAHEAD,
    )
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if not 0.0 < args.worker_fraction <= 1.0:
        parser.error("--worker-fraction must be greater than 0 and at most 1.")
    try:
        _resolve_worker_count(args.workers, worker_fraction=args.worker_fraction)
    except ValueError as error:
        parser.error(str(error))
    try:
        _resolve_mp_start_method(args.mp_start_method)
    except ValueError as error:
        parser.error(str(error))

    last_features = None
    for last_features in iter_parquet_features(
        args.input,
        episode_steps=args.episode_steps,
        physics_lookahead=args.physics_lookahead,
        batch_size=args.batch_size,
        workers=args.workers,
        worker_fraction=args.worker_fraction,
        mp_start_method=args.mp_start_method,
        limit=args.limit,
        log_every=args.log_every,
    ):
        pass

    if last_features is not None:
        print(f"[features] Last snapshot shapes: {_shape_summary(last_features)}")


if __name__ == "__main__":
    main()
