from __future__ import annotations

# ============================================================
# generate_graphs.py — Self-contained Kaggle PyG extraction script
# Built by clean_builder.py from modules/utils.py + modules/analysis_utils.py
# ============================================================
import math
import os
import argparse
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np

try:
    import torch_geometric
except ImportError:
    print("torch_geometric not found, installing...")
    os.system("pip install torch-geometric")

import torch
from torch_geometric.data import Data

# ============================================================
# Constants
# ============================================================
SUN_RADIUS = 10.0

BOARD_SIZE = 100.0

STEP_NORMALIZER = 500.0

PACKAGED_TARGET_DATE = None

NODE_FEATURE_NAMES = [
    "is_mine",
    "is_neutral",
    "is_enemy_1",
    "is_enemy_2",
    "is_enemy_3",
    "ships_norm",
    "production_norm",
    "step_norm",
    "x_norm",
    "y_norm",
    "radius_norm",
    "is_comet",
    "is_2p",
    "is_4p",
    "angular_velocity",
]

def planet_from_row(row: Any) -> dict[str, Any]:
    return {
        "id": int(row[0]),
        "owner": int(row[1]),
        "x": float(row[2]),
        "y": float(row[3]),
        "radius": float(row[4]),
        "ships": int(row[5]),
        "production": int(row[6]),
    }

# ============================================================
# STEP 4: Node Feature Extraction
# ============================================================
def extract_node_features_step_4(row, expert_player_id, planet_dicts=None):
    """
    Extract a small set of raw and normalized planet features.
    """
    planets_array = row['planets']
    comet_ids = set(row['comet_planet_ids']) if 'comet_planet_ids' in row else set()

    if planet_dicts is None:
        planet_dicts = [planet_from_row(p) for p in planets_array]
    player_count = int(row.get("player_count", 0) or 0)
    step_norm = float(row.get("step", 0) or 0) / STEP_NORMALIZER
    angular_velocity = float(row.get("angular_velocity", 0.0) or 0.0)
    is_2p = 1.0 if player_count == 2 else 0.0
    is_4p = 1.0 if player_count == 4 else 0.0

    node_features = []
    all_player_ids = {0, 1, 2, 3}
    enemy_ids = sorted(list(all_player_ids - {expert_player_id}))
    if len(enemy_ids) < 3:
        enemy_ids += [-99] * (3 - len(enemy_ids))
    e1, e2, e3 = enemy_ids[0], enemy_ids[1], enemy_ids[2]
    
    for p_dict in planet_dicts:
        owner = p_dict["owner"]
        
        is_mine = 1.0 if owner == expert_player_id else 0.0
        is_neutral = 1.0 if owner == -1 else 0.0
        is_e1 = 1.0 if owner == e1 else 0.0
        is_e2 = 1.0 if owner == e2 else 0.0
        is_e3 = 1.0 if owner == e3 else 0.0
        norm_ships = p_dict["ships"] / 100.0
        norm_prod = p_dict["production"] / 5.0
        is_comet = 1.0 if int(p_dict["id"]) in comet_ids else 0.0
        
        features = [
            is_mine, is_neutral, is_e1, is_e2, is_e3, 
            norm_ships, norm_prod,
            step_norm,
            float(p_dict["x"]) / BOARD_SIZE,
            float(p_dict["y"]) / BOARD_SIZE,
            float(p_dict["radius"]) / SUN_RADIUS,
            is_comet,
            is_2p,
            is_4p,
            angular_velocity,
        ]
        node_features.append(features)
        
    return np.array(node_features, dtype=np.float32)

# ============================================================
# STEP 6: Edge Feature Extraction
# ============================================================
def extract_edge_features_step_6(row, expert_player_id, planet_dicts=None):
    """
    Create a fully connected graph with only relative geometry.
    """
    planets_array = row['planets']
    
    if planet_dicts is None:
        planet_dicts = [planet_from_row(p) for p in planets_array]
    num_planets = len(planet_dicts)
    
    edge_index_t, edge_pairs = get_full_directed_edge_template(num_planets)
    edge_features = np.zeros((len(edge_pairs), 3), dtype=np.float32)
    
    for edge_idx, (i, j) in enumerate(edge_pairs):
        source = planet_dicts[i]
        target = planet_dicts[j]
        dx = (float(target["x"]) - float(source["x"])) / BOARD_SIZE
        dy = (float(target["y"]) - float(source["y"])) / BOARD_SIZE
        edge_features[edge_idx] = (dx, dy, math.hypot(dx, dy))
    
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    return edge_index_t, edge_attr

def edge_attr_from_node_features(x, edge_index):
    x_col = NODE_FEATURE_NAMES.index("x_norm")
    y_col = NODE_FEATURE_NAMES.index("y_norm")
    source_indices = edge_index[0]
    target_indices = edge_index[1]
    dx = x[target_indices, x_col] - x[source_indices, x_col]
    dy = x[target_indices, y_col] - x[source_indices, y_col]
    return torch.stack([dx, dy, torch.sqrt(dx * dx + dy * dy)], dim=1)

# ============================================================
# STEP 7: Label Extraction
# ============================================================
def extract_labels_step_7(
    row,
    expert_player_id,
    planet_dicts,
):
    """
    Extract direct per-source labels from [source_id, angle, ships] actions.

    Multiple launches from one source are merged using ship-weighted direction
    and total sent fraction. This is intentionally simple for the baseline.
    """
    num_nodes = len(planet_dicts)
    y_source_fire = torch.zeros(num_nodes, 1, dtype=torch.float32)
    y_angle_sin = torch.zeros(num_nodes, 1, dtype=torch.float32)
    y_angle_cos = torch.zeros(num_nodes, 1, dtype=torch.float32)
    y_ship_fraction = torch.zeros(num_nodes, 1, dtype=torch.float32)
    angle_weight = torch.zeros(num_nodes, 1, dtype=torch.float32)

    actions = row.get("action")
    if actions is None:
        actions = []
    planet_id_to_idx = {int(p["id"]): idx for idx, p in enumerate(planet_dicts)}
    total_actions = 0

    for move in actions:
        if move is None or len(move) < 3:
            continue
        total_actions += 1
        source_id = int(move[0])
        angle = float(move[1])
        ships = float(move[2])
        source_idx = planet_id_to_idx.get(source_id)
        if source_idx is None or ships <= 0:
            continue

        source = planet_dicts[source_idx]
        source_ships = max(1.0, float(source["ships"]))
        weight = max(1.0, ships)
        y_source_fire[source_idx] = 1.0
        y_angle_sin[source_idx] += math.sin(angle) * weight
        y_angle_cos[source_idx] += math.cos(angle) * weight
        angle_weight[source_idx] += weight
        y_ship_fraction[source_idx] += ships / source_ships

    active = angle_weight[:, 0] > 0
    if active.any():
        y_angle_sin[active] /= angle_weight[active]
        y_angle_cos[active] /= angle_weight[active]
        magnitude = torch.sqrt(y_angle_sin[active] ** 2 + y_angle_cos[active] ** 2).clamp_min(1e-8)
        y_angle_sin[active] /= magnitude
        y_angle_cos[active] /= magnitude
    y_ship_fraction.clamp_(0.0, 1.0)

    return (
        y_source_fire,
        y_angle_sin,
        y_angle_cos,
        y_ship_fraction,
        total_actions,
        int(y_source_fire.sum().item()),
    )

# ============================================================
# MAIN: Generate PyG Data Object
# ============================================================
def generate_pyg_data(row_dict):
    """
    Main function: Takes a single row dict and returns a PyG Data object.
    """
    expert_player_id = int(row_dict.get('expert_player_id', 0))
    planet_dicts = [planet_from_row(p) for p in row_dict['planets']]
    
    node_features = extract_node_features_step_4(
        row_dict,
        expert_player_id,
        planet_dicts=planet_dicts,
    )
    edge_index, edge_attr = extract_edge_features_step_6(
        row_dict,
        expert_player_id,
        planet_dicts=planet_dicts,
    )
    (
        y_source_fire,
        y_angle_sin,
        y_angle_cos,
        y_ship_fraction,
        num_actions,
        num_action_sources,
    ) = extract_labels_step_7(
        row_dict,
        expert_player_id,
        planet_dicts,
    )
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_source_fire=y_source_fire,
        y_angle_sin=y_angle_sin,
        y_angle_cos=y_angle_cos,
        y_ship_fraction=y_ship_fraction,
        num_actions=torch.tensor(num_actions, dtype=torch.int16),
        num_action_sources=torch.tensor(num_action_sources, dtype=torch.int16),
    )
    
    return data

# ============================================================
# KAGGLE PIPELINE: Process batches
# ============================================================
import glob
import time
import gc

_EDGE_INDEX_CACHE = {}
_EDGE_PAIR_CACHE = {}

def get_full_directed_edge_template(num_planets):
    cached = _EDGE_INDEX_CACHE.get(num_planets)
    if cached is not None:
        return cached, _EDGE_PAIR_CACHE[num_planets]

    edge_pairs = [(i, j) for i in range(num_planets) for j in range(num_planets) if i != j]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    _EDGE_INDEX_CACHE[num_planets] = edge_index
    _EDGE_PAIR_CACHE[num_planets] = edge_pairs
    return edge_index, edge_pairs

def process_row_task(row_dict):
    try:
        return generate_pyg_data(row_dict)
    except Exception as e:
        return None

def pack_graphs(graphs):
    """
    Store simple baseline graphs grouped by node count.
    Action labels are sparse because most source planets do not launch.
    """
    groups = {}
    for original_idx, graph in enumerate(graphs):
        num_nodes = int(graph.x.size(0))
        group = groups.setdefault(num_nodes, {
            "indices": [],
            "edge_index": graph.edge_index,
            "x": [],
            "action_label_indices": [],
            "action_label_values": [],
            "num_actions": [],
            "num_action_sources": [],
        })

        group["indices"].append(original_idx)
        group["x"].append(graph.x)
        action_nodes = torch.nonzero(graph.y_source_fire[:, 0] > 0, as_tuple=False)[:, 0]
        if action_nodes.numel() > 0:
            graph_col = torch.full(
                (action_nodes.size(0), 1),
                len(group["indices"]) - 1,
                dtype=torch.long,
            )
            group["action_label_indices"].append(
                torch.cat([graph_col, action_nodes[:, None]], dim=1)
            )
            group["action_label_values"].append(
                torch.cat(
                    [
                        graph.y_angle_sin[action_nodes],
                        graph.y_angle_cos[action_nodes],
                        graph.y_ship_fraction[action_nodes],
                    ],
                    dim=1,
                )
            )
        group["num_actions"].append(graph.num_actions)
        group["num_action_sources"].append(graph.num_action_sources)

    packed_groups = {}
    for num_nodes, group in groups.items():
        key = str(num_nodes)
        if group["action_label_indices"]:
            action_label_indices = torch.cat(group["action_label_indices"], dim=0).to(torch.int32)
            action_label_values = torch.cat(group["action_label_values"], dim=0)
        else:
            action_label_indices = torch.empty((0, 2), dtype=torch.int32)
            action_label_values = torch.empty((0, 3), dtype=torch.float32)

        packed_groups[key] = {
            "indices": torch.tensor(group["indices"], dtype=torch.long),
            "edge_index": group["edge_index"],
            "x": torch.stack(group["x"], dim=0),
            "action_label_indices": action_label_indices,
            "action_label_values": action_label_values,
            "num_actions": torch.stack(group["num_actions"], dim=0),
            "num_action_sources": torch.stack(group["num_action_sources"], dim=0),
        }

    return {
        "format": "packed_pyg_graphs_v5_simple",
        "num_graphs": len(graphs),
        "node_feature_names": NODE_FEATURE_NAMES,
        "label_names": [
            "y_source_fire",
            "y_angle_sin",
            "y_angle_cos",
            "y_ship_fraction",
        ],
        "action_label_value_layout": [
            "angle_sin",
            "angle_cos",
            "ship_fraction",
        ],
        "action_alignment": "observation_at_step_t_with_action_stored_at_step_t_plus_1",
        "edge_attr_layout": {
            "names": ["dx_norm", "dy_norm", "distance_norm"],
            "total_cols": 3,
            "storage": "derived_from_node_xy_at_load_time",
        },
        "notes": "No fleet trajectory simulation or physics-derived target labels.",
        "groups": packed_groups,
    }

def unpack_packed_graphs(payload):
    """Convert a simple packed chunk back into PyG Data objects."""
    if not isinstance(payload, dict) or payload.get("format") != "packed_pyg_graphs_v5_simple":
        raise ValueError("Unsupported graph payload format")

    graphs = [None] * int(payload["num_graphs"])
    for group in payload["groups"].values():
        action_indices = group["action_label_indices"].long()
        for local_idx, original_idx in enumerate(group["indices"].tolist()):
            num_nodes = group["x"].size(1)
            y_source_fire = torch.zeros((num_nodes, 1), dtype=group["x"].dtype)
            y_angle_sin = torch.zeros((num_nodes, 1), dtype=group["x"].dtype)
            y_angle_cos = torch.zeros((num_nodes, 1), dtype=group["x"].dtype)
            y_ship_fraction = torch.zeros((num_nodes, 1), dtype=group["x"].dtype)
            action_mask = action_indices[:, 0] == local_idx
            action_nodes = action_indices[action_mask, 1]
            action_values = group["action_label_values"][action_mask]
            if action_nodes.numel() > 0:
                y_source_fire[action_nodes] = 1.0
                y_angle_sin[action_nodes] = action_values[:, 0:1]
                y_angle_cos[action_nodes] = action_values[:, 1:2]
                y_ship_fraction[action_nodes] = action_values[:, 2:3]
            graphs[original_idx] = Data(
                x=group["x"][local_idx],
                edge_index=group["edge_index"],
                edge_attr=edge_attr_from_node_features(
                    group["x"][local_idx],
                    group["edge_index"],
                ),
                y_source_fire=y_source_fire,
                y_angle_sin=y_angle_sin,
                y_angle_cos=y_angle_cos,
                y_ship_fraction=y_ship_fraction,
                num_actions=group["num_actions"][local_idx],
                num_action_sources=group["num_action_sources"][local_idx],
            )
    return graphs

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def process_chunk_rows(chunk_rows, pool, pool_chunksize):
    if pool is None:
        return [process_row_task(row) for row in chunk_rows]
    return pool.map(process_row_task, chunk_rows, chunksize=pool_chunksize)

def align_actions_to_observations(df):
    required = {"game_id", "step", "action"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Cannot align replay actions; missing parquet columns: "
            + ", ".join(sorted(missing))
        )

    aligned = df.sort_values(["game_id", "step"], kind="stable").reset_index(drop=True)
    groups = aligned.groupby("game_id", sort=False)
    next_actions = groups["action"].shift(-1)
    next_steps = groups["step"].shift(-1)
    valid = next_steps.eq(aligned["step"] + 1)
    aligned = aligned.loc[valid].copy()
    aligned["action"] = next_actions.loc[valid].to_numpy()
    return aligned

def find_batch_files(input_dir):
    patterns = ("batch_*.parquet", "batch-*.parquet")
    files = []
    for pattern in patterns:
        files.extend(glob.glob(f"{input_dir}/**/{pattern}", recursive=True))
    return sorted(set(files))

def validate_target_date(value):
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc
    return value

def resolve_date_input_dir(input_dir, target_date):
    direct = os.path.join(input_dir, target_date)
    if os.path.isdir(direct):
        return direct

    matches = sorted(
        path for path in glob.glob(f"/kaggle/input/**/{target_date}", recursive=True)
        if os.path.isdir(path) and find_batch_files(path)
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return direct
    raise RuntimeError(f"Multiple Kaggle inputs contain date {target_date}: {matches}")

def process_kaggle_batches(
    input_dir="/kaggle/input/orbit-datewise-replay-minify-fullmeta-latest10",
    output_dir="/kaggle/working/pyg_chunks",
    local_mode=False,
    target_date=None,
):
    # Override for local testing if running locally
    if local_mode:
        input_dir = "data"
        output_dir = "data/pyg_chunks"
    
    print(f"=== KAGGLE PIPELINE ===")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    
    if target_date:
        input_dir = resolve_date_input_dir(input_dir, target_date)
        output_dir = os.path.join(output_dir, target_date)
        print(f"Target date: {target_date}")

    batch_files = find_batch_files(input_dir)
    
    if not batch_files:
        print("No batch files found! Please check the input path.")
        return
    
    print(f"Found {len(batch_files)} parquet batches to process.")
    os.makedirs(output_dir, exist_ok=True)
    
    total_batches = len(batch_files)
    import multiprocessing
    num_cores = min(env_int("PYG_NUM_CORES", 1), max(1, multiprocessing.cpu_count()))
    chunk_size = env_int("PYG_CHUNK_SIZE", 5000)
    pool_chunksize = env_int("PYG_POOL_CHUNKSIZE", 128)
    start_batch_index = max(1, env_int("PYG_START_BATCH_INDEX", 1))
    end_batch_index = env_int("PYG_END_BATCH_INDEX", total_batches)
    if end_batch_index < start_batch_index:
        print(f"No batches selected: start={start_batch_index}, end={end_batch_index}.")
        return
    batch_files = batch_files[start_batch_index - 1:end_batch_index]
    selected_total_batches = len(batch_files)
    print(f"Using {num_cores} CPU cores for parallel extraction.")
    print(f"Chunk size: {chunk_size}; pool chunksize: {pool_chunksize}.")
    print(f"Selected parquet batch files: {start_batch_index}-{end_batch_index} of {total_batches} ({selected_total_batches} files).")
    print("Output format: packed_pyg_graphs_v5_simple.")
    global_start = time.time()
    pipeline_actions = 0
    pipeline_action_sources = 0
    pipeline_failed_rows = 0

    pool = None
    if num_cores > 1:
        pool = multiprocessing.Pool(processes=num_cores, maxtasksperchild=10000)
    
    try:
        for idx, file in enumerate(batch_files):
            absolute_batch_index = start_batch_index + idx
            rel_path = os.path.relpath(file, input_dir)
            rel_parent = os.path.dirname(rel_path)
            file_name = os.path.basename(file).replace('.parquet', '')
            print(f"\n[{absolute_batch_index}/{total_batches}] Processing {rel_path}...")
            
            df = pd.read_parquet(file)
            raw_rows = len(df)
            df = align_actions_to_observations(df)
            print(f"  -> Aligned {len(df)} trainable states from {raw_rows} replay rows.")
            
            total_rows = len(df)
            for chunk_idx in range(0, total_rows, chunk_size):
                chunk_df = df.iloc[chunk_idx:chunk_idx+chunk_size]
                chunk_rows = chunk_df.to_dict('records')
                
                chunk_start = time.time()
                graphs = process_chunk_rows(chunk_rows, pool, pool_chunksize)
                valid_graphs = [g for g in graphs if g is not None]
                pipeline_failed_rows += len(graphs) - len(valid_graphs)
                chunk_actions = sum(int(g.num_actions) for g in valid_graphs)
                chunk_action_sources = sum(int(g.num_action_sources) for g in valid_graphs)
                pipeline_actions += chunk_actions
                pipeline_action_sources += chunk_action_sources
                
                chunk_id = chunk_idx // chunk_size
                chunk_output_dir = os.path.join(output_dir, rel_parent)
                os.makedirs(chunk_output_dir, exist_ok=True)
                out_file = os.path.join(chunk_output_dir, f"{file_name}_chunk_{chunk_id}.pt")
                tmp_file = f"{out_file}.tmp"
                payload = pack_graphs(valid_graphs)
                try:
                    torch.save(payload, tmp_file)
                    os.replace(tmp_file, out_file)
                except Exception:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                    raise
                
                elapsed = time.time() - chunk_start
                rps = len(chunk_rows) / elapsed
                print(f"  -> Saved {len(valid_graphs)} graphs to {out_file} (packed, {rps:.2f} rows/s)")
                if chunk_actions:
                    print(
                        f"     Actions represented: {chunk_actions} launches "
                        f"across {chunk_action_sources} source labels."
                    )

                del graphs, valid_graphs, payload, chunk_rows, chunk_df
                gc.collect()
                
            print(f"\u2705 Completed {rel_path} ({absolute_batch_index}/{total_batches})")
            del df
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\n\u2705 All batches processed successfully in {(time.time()-global_start)/60:.2f} minutes!")
    if pipeline_actions:
        print(
            f"Actions represented: {pipeline_actions} launches across "
            f"{pipeline_action_sources} source labels."
        )
    print(f"Rows skipped after extraction errors: {pipeline_failed_rows}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=validate_target_date,
        default=os.environ.get("PYG_TARGET_DATE") or PACKAGED_TARGET_DATE,
        help="Process one date in YYYY-MM-DD format.",
    )
    args = parser.parse_args()
    if not args.date:
        parser.error("--date is required (or set PYG_TARGET_DATE)")

    # Automatically detect if running on Kaggle
    is_kaggle = os.path.exists("/kaggle")
    process_kaggle_batches(local_mode=not is_kaggle, target_date=args.date)
