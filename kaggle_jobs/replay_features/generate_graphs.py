from __future__ import annotations

# ============================================================
# generate_graphs.py — Self-contained Kaggle PyG extraction script
# Built by clean_builder.py from modules/utils.py + modules/analysis_utils.py
# ============================================================
import math
import random
import os
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

# ---- Dummy classes (replace kaggle_environments imports) ----
class Planet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production

class Fleet:
    def __init__(self, id, owner, ships, x, y, angle, target_id):
        self.id = id
        self.owner = owner
        self.ships = ships
        self.x = x
        self.y = y
        self.angle = angle
        self.target_id = target_id

# ============================================================
# Constants from utils.py
# ============================================================
CENTER = 50.0

ROTATION_RADIUS_LIMIT = 50.0

MAX_SPEED = 6.0

SUN_RADIUS = 10.0

BOARD_SIZE = 100.0

MAX_COLLISION_TURN = 120

FUTURE_SOURCE_LOOKAHEAD = 90

OPENING_SCORE_TURNS = 30

OPPONENT_QUADRANT_ATTACK_STEP = 45

# ============================================================
# Physics Engine (from utils.py)
# ============================================================
def get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)

def fleet_speed(ships, max_speed=MAX_SPEED):
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000)
    return min(1.0 + (max_speed - 1.0) * (ratio**1.5), max_speed)

# ---- Public aliases ----

# ============================================================
# Constants from analysis_utils.py
# ============================================================
CENTER = 50.0

ROTATION_RADIUS_LIMIT = 50.0

MAX_COLLISION_TURN = 120

MAX_SPEED = 6.0

OPENING_SCORE_TURNS = 30

OPENING_STRATEGIC_STEP_LIMIT = 60

HIGH_PRODUCTION_MIN = 3

LOW_PRODUCTION_PENALTIES = {1: 50.0, 2: 20.0}

HIGH_PRODUCTION_FRONTIER_BONUS = {3: 10.0, 4: 20.0, 5: 30.0}

ENEMY_CENTROID_RADIUS = 75.0

ENEMY_CENTROID_WEIGHT = 0.75

OVERHEAD_OUTLIER_FLOORS = {
    "medium": 14.0,
    "high": 12.0,
}

# ============================================================
# Analysis Functions (from analysis_utils.py)
# ============================================================

def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000)
    return min(1.0 + (max_speed - 1.0) * (ratio**1.5), max_speed)

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

def is_orbiting_planet(planet: dict[str, Any], initial_by_id: dict[int, Any], comet_ids: set[int]) -> bool:
    if int(planet["id"]) in comet_ids:
        return False
    initial = initial_by_id.get(int(planet["id"]))
    if initial is None:
        return False
    orbital_radius = math.hypot(float(initial[2]) - CENTER, float(initial[3]) - CENTER)
    return orbital_radius + float(initial[4]) < ROTATION_RADIUS_LIMIT

def quadrant(planet: dict[str, Any]) -> tuple[int, int]:
    return (0 if float(planet["x"]) < CENTER else 1, 0 if float(planet["y"]) < CENTER else 1)

def opposite_quadrant(q: tuple[int, int]) -> tuple[int, int]:
    return (1 - q[0], 1 - q[1])

def quadrant_center(q: tuple[int, int]) -> tuple[float, float]:
    return (25.0 + 50.0 * q[0], 25.0 + 50.0 * q[1])

def distance(a: dict[str, Any], b: dict[str, Any] | tuple[float, float]) -> float:
    bx, by = (b if isinstance(b, tuple) else (float(b["x"]), float(b["y"])))
    return math.hypot(float(a["x"]) - bx, float(a["y"]) - by)

def home_quadrant(player_id: int, planets: list[dict[str, Any]], initial_planets: list[dict[str, Any]]) -> tuple[int, int] | None:
    for planet in initial_planets:
        if int(planet["owner"]) == int(player_id):
            return quadrant(planet)
    for planet in planets:
        if int(planet["owner"]) == int(player_id):
            return quadrant(planet)
    return None

def frontier_role_score(planet: dict[str, Any], reference_center: tuple[float, float]) -> float:
    overhead = float(planet["ships"]) / max(1.0, float(planet["production"]))
    reference_distance = distance(planet, reference_center)
    return -5.0 * overhead - 3.0 * reference_distance + 2.0 * float(planet["production"])

def source_target_path_distance(source: dict[str, Any], target: dict[str, Any]) -> float:
    return max(0.0, distance(source, target) - float(source["radius"]) - float(target["radius"]))

# ============================================================
# STEP 4: Node Feature Extraction
# ============================================================
def extract_node_features_step_4(row, expert_player_id, planet_dicts=None, initial_planet_dicts=None):
    """
    Step 4: Extract Strategic Features + Physics + Core Features
    """
    planets_array = row['planets']
    initial_planets_array = row['initial_planets']
    comet_ids = set(row['comet_planet_ids']) if 'comet_planet_ids' in row else set()

    # Build dictionaries for utils
    initial_by_id = {int(p[0]): p for p in initial_planets_array}
    
    if planet_dicts is None:
        planet_dicts = [planet_from_row(p) for p in planets_array]
    if initial_planet_dicts is None:
        initial_planet_dicts = [planet_from_row(p) for p in initial_planets_array]
            
    # Global incoming physics simulation removed to speed up extraction.
    # The GNN will learn threats from the temporal fleet edge buckets instead.

    # Calculate Quadrant centers for Frontier Role Scoring
    home_q = home_quadrant(expert_player_id, planet_dicts, initial_planet_dicts)
    if home_q is not None:
        home_center = quadrant_center(home_q)
        enemy_center = quadrant_center(opposite_quadrant(home_q))
    else:
        home_center = (50.0, 50.0)
        enemy_center = (50.0, 50.0)

    node_features = []
    all_player_ids = {0, 1, 2, 3}
    enemy_ids = sorted(list(all_player_ids - {expert_player_id}))
    if len(enemy_ids) < 3:
        enemy_ids += [-99] * (3 - len(enemy_ids))
    e1, e2, e3 = enemy_ids[0], enemy_ids[1], enemy_ids[2]
    
    for p_dict in planet_dicts:
        owner = p_dict["owner"]
        
        # Core
        is_mine = 1.0 if owner == expert_player_id else 0.0
        is_neutral = 1.0 if owner == -1 else 0.0
        is_e1 = 1.0 if owner == e1 else 0.0
        is_e2 = 1.0 if owner == e2 else 0.0
        is_e3 = 1.0 if owner == e3 else 0.0
        norm_ships = p_dict["ships"] / 100.0
        norm_prod = p_dict["production"] / 5.0
        
        # Step 3 Physics
        orbiting = 1.0 if is_orbiting_planet(p_dict, initial_by_id, comet_ids) else 0.0

        # Score how valuable this planet is as a frontline or backline
        f_score_home = frontier_role_score(p_dict, home_center) / 100.0
        f_score_enemy = frontier_role_score(p_dict, enemy_center) / 100.0
        
        features = [
            is_mine, is_neutral, is_e1, is_e2, is_e3, 
            norm_ships, norm_prod,
            orbiting, f_score_home, f_score_enemy
        ]
        node_features.append(features)
        
    return np.array(node_features, dtype=np.float32)

# ============================================================
# STEP 6: Edge Feature Extraction
# ============================================================
def extract_edge_features_step_6(row, expert_player_id, planet_dicts=None):
    """
    Step 6: Extract Base Edge Features + Temporal Fleet Buckets
    Creates a fully connected directed graph (excluding self-loops).
    Returns edge_index [2, E] and edge_features [E, 18].
    """
    planets_array = row['planets']
    fleets_array = row['fleets']
    
    if planet_dicts is None:
        planet_dicts = [planet_from_row(p) for p in planets_array]
    num_planets = len(planet_dicts)
    
    all_player_ids = {0, 1, 2, 3}
    enemy_ids = sorted(list(all_player_ids - {expert_player_id}))
    if len(enemy_ids) < 3:
        enemy_ids += [-99] * (3 - len(enemy_ids))
    e1, e2, e3 = enemy_ids[0], enemy_ids[1], enemy_ids[2]
    
    # Pre-group fleets by (source_id, target_id)
    edge_fleets = {}
    for f in fleets_array:
        f_owner = int(f[1])
        f_ships = float(f[2])
        f_x = float(f[3])
        f_y = float(f[4])
        s_id = int(f[5])
        t_id = int(f[6])
        
        edge_key = (s_id, t_id)
        if edge_key not in edge_fleets:
            edge_fleets[edge_key] = []
        edge_fleets[edge_key].append((f_owner, f_ships, f_x, f_y))
    
    edge_index_t, edge_pairs = get_full_directed_edge_template(num_planets)
    edge_features = np.zeros((len(edge_pairs), 42), dtype=np.float64)
    
    baseline_speed = fleet_speed(1, MAX_SPEED)
    
    for edge_idx, (i, j) in enumerate(edge_pairs):
        source = planet_dicts[i]
        target = planet_dicts[j]
        s_id = source['id']
        t_id = target['id']
        
        # 1. Spatial Features
        path_distance = source_target_path_distance(source, target)
        travel_turns = math.ceil(path_distance / baseline_speed)
        
        edge_features[edge_idx, 0] = path_distance / 100.0
        edge_features[edge_idx, 1] = travel_turns / 100.0
        
        for owner, fleet_ships, fleet_x, fleet_y in edge_fleets.get((s_id, t_id), ()):
            # Calculate ETA
            dist_to_target = math.hypot(target['x'] - fleet_x, target['y'] - fleet_y) - target['radius']
            dist_to_target = max(0.0, dist_to_target)
            speed = fleet_speed(fleet_ships, MAX_SPEED)
            eta = math.ceil(dist_to_target / speed)
            
            bucket_idx = min(9, int(eta // 5))
            
            norm_ships = fleet_ships / 100.0
            if owner == expert_player_id:
                edge_features[edge_idx, 2 + bucket_idx] += norm_ships
            elif owner == e1:
                edge_features[edge_idx, 12 + bucket_idx] += norm_ships
            elif owner == e2:
                edge_features[edge_idx, 22 + bucket_idx] += norm_ships
            elif owner == e3:
                edge_features[edge_idx, 32 + bucket_idx] += norm_ships
    
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    return edge_index_t, edge_attr

# ============================================================
# STEP 7: Label Extraction
# ============================================================
def extract_labels_step_7(row, expert_player_id):
    """
    Step 7: Extract y_intent and y_fraction labels.
    """
    planets_array = row['planets']
    num_planets = len(planets_array)
    
    action = row.get('action', None)
    
    y_intent = torch.zeros(num_planets, 1, dtype=torch.float32)
    y_fraction = torch.zeros(num_planets, 1, dtype=torch.float32)
    
    if action is None or not isinstance(action, dict):
        return y_intent, y_fraction
        
    target_id = action.get('target_id', None)
    ships_percent = action.get('ships_percent', 0.0)
    
    if target_id is None:
        return y_intent, y_fraction
        
    planet_id_to_idx = {int(p[0]): idx for idx, p in enumerate(planets_array)}
    
    if target_id in planet_id_to_idx:
        idx = planet_id_to_idx[target_id]
        y_intent[idx] = 1.0
        y_fraction[idx] = float(ships_percent) / 100.0
        
    return y_intent, y_fraction

# ============================================================
# MAIN: Generate PyG Data Object
# ============================================================
def generate_pyg_data(row_dict):
    """
    Main function: Takes a single row dict and returns a PyG Data object.
    """
    expert_player_id = int(row_dict.get('expert_player_id', 0))
    planet_dicts = [planet_from_row(p) for p in row_dict['planets']]
    initial_planet_dicts = [planet_from_row(p) for p in row_dict['initial_planets']]
    
    node_features = extract_node_features_step_4(
        row_dict,
        expert_player_id,
        planet_dicts=planet_dicts,
        initial_planet_dicts=initial_planet_dicts,
    )
    edge_index, edge_attr = extract_edge_features_step_6(
        row_dict,
        expert_player_id,
        planet_dicts=planet_dicts,
    )
    y_intent, y_fraction = extract_labels_step_7(row_dict, expert_player_id)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_intent=y_intent,
        y_fraction=y_fraction,
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
    Store graph tensors compactly while preserving the same training data.
    Graphs are grouped by node count so each tensor stack has fixed shapes.
    Edge features are split into dense spatial columns and sparse fleet buckets.
    """
    groups = {}
    for original_idx, graph in enumerate(graphs):
        num_nodes = int(graph.x.size(0))
        group = groups.setdefault(num_nodes, {
            "indices": [],
            "edge_index": graph.edge_index,
            "x": [],
            "edge_static": [],
            "edge_bucket_indices": [],
            "edge_bucket_values": [],
            "y_intent": [],
            "y_fraction": [],
        })
        edge_buckets = graph.edge_attr[:, 2:]
        bucket_indices = torch.nonzero(edge_buckets, as_tuple=False)
        if bucket_indices.numel() > 0:
            graph_col = torch.full((bucket_indices.size(0), 1), len(group["indices"]), dtype=torch.long)
            group["edge_bucket_indices"].append(torch.cat([graph_col, bucket_indices], dim=1))
            group["edge_bucket_values"].append(edge_buckets[bucket_indices[:, 0], bucket_indices[:, 1]])

        group["indices"].append(original_idx)
        group["x"].append(graph.x)
        group["edge_static"].append(graph.edge_attr[:, :2])
        group["y_intent"].append(graph.y_intent)
        group["y_fraction"].append(graph.y_fraction)

    packed_groups = {}
    for num_nodes, group in groups.items():
        key = str(num_nodes)
        if group["edge_bucket_indices"]:
            edge_bucket_indices = torch.cat(group["edge_bucket_indices"], dim=0).to(torch.int32)
            edge_bucket_values = torch.cat(group["edge_bucket_values"], dim=0)
        else:
            edge_bucket_indices = torch.empty((0, 3), dtype=torch.int32)
            edge_bucket_values = torch.empty((0,), dtype=torch.float32)

        packed_groups[key] = {
            "indices": torch.tensor(group["indices"], dtype=torch.long),
            "edge_index": group["edge_index"],
            "x": torch.stack(group["x"], dim=0),
            "edge_static": torch.stack(group["edge_static"], dim=0),
            "edge_bucket_indices": edge_bucket_indices,
            "edge_bucket_values": edge_bucket_values,
            "y_intent": torch.stack(group["y_intent"], dim=0),
            "y_fraction": torch.stack(group["y_fraction"], dim=0),
        }

    return {
        "format": "packed_pyg_graphs_v2",
        "num_graphs": len(graphs),
        "edge_attr_layout": {
            "static_cols": 2,
            "bucket_cols": 40,
            "total_cols": 42,
        },
        "groups": packed_groups,
    }

def unpack_packed_graphs(payload):
    """
    Convert a packed chunk back into a list of PyG Data objects.
    This is useful for future training code that wants the old format.
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict) or payload.get("format") not in {"packed_pyg_graphs_v1", "packed_pyg_graphs_v2"}:
        raise ValueError("Unsupported graph payload format")

    graphs = [None] * int(payload["num_graphs"])
    for group in payload["groups"].values():
        indices = group["indices"].tolist()
        if "edge_attr" in group:
            edge_attr = group["edge_attr"]
        else:
            edge_attr = torch.zeros(
                (
                    group["x"].size(0),
                    group["edge_static"].size(1),
                    payload["edge_attr_layout"]["total_cols"],
                ),
                dtype=group["edge_static"].dtype,
            )
            edge_attr[:, :, :2] = group["edge_static"]
            bucket_indices = group["edge_bucket_indices"].long()
            if bucket_indices.numel() > 0:
                edge_attr[
                    bucket_indices[:, 0],
                    bucket_indices[:, 1],
                    bucket_indices[:, 2] + 2,
                ] = group["edge_bucket_values"]

        for local_idx, original_idx in enumerate(indices):
            graphs[original_idx] = Data(
                x=group["x"][local_idx],
                edge_index=group["edge_index"],
                edge_attr=edge_attr[local_idx],
                y_intent=group["y_intent"][local_idx],
                y_fraction=group["y_fraction"][local_idx],
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

def find_batch_files(input_dir):
    patterns = ("batch_*.parquet", "batch-*.parquet")
    files = []
    for pattern in patterns:
        files.extend(glob.glob(f"{input_dir}/**/{pattern}", recursive=True))
    return sorted(set(files))

def process_kaggle_batches(input_dir="/kaggle/input/notebooks/atomstack001/orbit-parquet-extractor-p1", output_dir="/kaggle/working/pyg_chunks", local_mode=False):
    # Override for local testing if running locally
    if local_mode:
        input_dir = "data"
        output_dir = "data/pyg_chunks"
    
    print(f"=== KAGGLE PIPELINE ===")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    
    batch_files = find_batch_files(input_dir)
    
    if not batch_files:
        print("No batch files found! Please check the input path.")
        return
    
    print(f"Found {len(batch_files)} batches to process.")
    os.makedirs(output_dir, exist_ok=True)
    
    total_batches = len(batch_files)
    import multiprocessing
    num_cores = min(env_int("PYG_NUM_CORES", 4), max(1, multiprocessing.cpu_count()))
    chunk_size = env_int("PYG_CHUNK_SIZE", 5000)
    pool_chunksize = env_int("PYG_POOL_CHUNKSIZE", 128)
    print(f"Using {num_cores} CPU cores for parallel extraction.")
    print(f"Chunk size: {chunk_size}; pool chunksize: {pool_chunksize}.")
    print("Output format: packed_pyg_graphs_v2.")
    global_start = time.time()

    pool = None
    if num_cores > 1:
        pool = multiprocessing.Pool(processes=num_cores, maxtasksperchild=10000)
    
    try:
        for idx, file in enumerate(batch_files):
            rel_path = os.path.relpath(file, input_dir)
            rel_parent = os.path.dirname(rel_path)
            file_name = os.path.basename(file).replace('.parquet', '')
            print(f"\n[{idx + 1}/{total_batches}] Processing {rel_path}...")
            
            df = pd.read_parquet(file)
            
            total_rows = len(df)
            for chunk_idx in range(0, total_rows, chunk_size):
                chunk_df = df.iloc[chunk_idx:chunk_idx+chunk_size]
                chunk_rows = chunk_df.to_dict('records')
                
                chunk_start = time.time()
                graphs = process_chunk_rows(chunk_rows, pool, pool_chunksize)
                valid_graphs = [g for g in graphs if g is not None]
                
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

                del graphs, valid_graphs, payload, chunk_rows, chunk_df
                gc.collect()
                
            print(f"\u2705 Completed {rel_path} ({idx + 1}/{total_batches})")
            del df
            gc.collect()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\n\u2705 All batches processed successfully in {(time.time()-global_start)/60:.2f} minutes!")

if __name__ == "__main__":
    # Automatically detect if running on Kaggle
    is_kaggle = os.path.exists("/kaggle")
    process_kaggle_batches(local_mode=not is_kaggle)
