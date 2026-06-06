from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from imitate.features.aggregates import extract_aggregate_features
from imitate.features.direct import extract_direct_features
from imitate.features.formula import extract_formula_features
from imitate.features.initial_state import extract_initial_state_features
from imitate.features.vectorized_geometry import (
    extract_vectorized_geometry_features,
)


def extract_features(
    row: dict[str, Any],
    *,
    episode_steps: int = 500,
) -> dict[str, Any]:
    """Extract the implemented feature stages for one snapshot."""
    direct = extract_direct_features(row)
    formula = extract_formula_features(direct, episode_steps=episode_steps)
    initial_state = extract_initial_state_features(direct)
    aggregates = extract_aggregate_features(direct, formula)
    vectorized_geometry = extract_vectorized_geometry_features(direct, formula)
    return {
        "metadata": direct["metadata"],
        "direct": {
            "feature_names": direct["feature_names"],
            "values": direct["values"],
        },
        "formula": formula,
        "initial_state": initial_state,
        "aggregates": aggregates,
        "vectorized_geometry": vectorized_geometry,
    }


def iter_parquet_features(
    parquet_path: str | Path,
    *,
    episode_steps: int = 500,
    batch_size: int = 1024,
    limit: int | None = None,
    log_every: int = 1000,
) -> Iterator[dict[str, Any]]:
    """Stream minified Parquet snapshots without loading the full file into RAM."""
    path = Path(parquet_path)
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    target_rows = min(total_rows, limit) if limit is not None else total_rows
    processed = 0
    started = time.perf_counter()

    print(f"[features] Input: {path}")
    print(f"[features] Rows: {target_rows}/{total_rows}")
    print(
        "[features] Stages: direct -> formula -> initial roles -> "
        "aggregates -> vectorized geometry"
    )
    print("[features] Backend: NumPy extraction; JAX transfer after padded batching")

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
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in record_batch.to_pylist():
            if limit is not None and processed >= limit:
                break
            yield extract_features(row, episode_steps=episode_steps)
            processed += 1
            if processed == 1 or processed % max(1, log_every) == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[features] Processed {processed}/{target_rows} snapshots "
                    f"({processed / elapsed:.1f} rows/s)"
                )
        if limit is not None and processed >= limit:
            break

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
        "aggregates",
        "vectorized_geometry",
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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    last_features = None
    for last_features in iter_parquet_features(
        args.input,
        episode_steps=args.episode_steps,
        batch_size=args.batch_size,
        limit=args.limit,
        log_every=args.log_every,
    ):
        pass

    if last_features is not None:
        print(f"[features] Last snapshot shapes: {_shape_summary(last_features)}")


if __name__ == "__main__":
    main()
