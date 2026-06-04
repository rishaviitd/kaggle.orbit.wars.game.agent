import os
import re
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

try:
    import orjson
    import pandas as pd
    import pyarrow
except ImportError:
    os.system("pip install orjson pandas pyarrow")
    import orjson
    import pandas as pd

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def date_key_from_path(json_path):
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", str(json_path))
    return match.group(1) if match else "unknown-date"

def cleanup_memory():
    gc.collect()
    if os.name == "posix":
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

def process_single_replay(json_path):
    """Reads a raw JSON, minifies it, and returns a flat list of dicts (one per turn)."""
    date_key = date_key_from_path(json_path)
    try:
        with open(json_path, 'rb') as f:
            replay = orjson.loads(f.read())
            
        final_rewards = replay.get('rewards', [])
        if not final_rewards:
            return date_key, []
            
        winner_idx = max(range(len(final_rewards)), key=lambda i: final_rewards[i] if final_rewards[i] is not None else -1)
        info = replay.get('info', {})
        team_names = info.get('TeamNames') or [
            agent.get('Name') for agent in info.get('Agents', [])
        ]
        statuses = replay.get('statuses', [])
        seed = info.get('seed')
        flat_steps = []
        game_id = Path(json_path).stem
        
        for step_idx, step_data in enumerate(replay['steps']):
            winner_data = step_data[winner_idx]
            if winner_data is None or 'observation' not in winner_data:
                continue
                
            obs = winner_data['observation']
            action = winner_data.get('action', [])
            expert_player_id = obs.get('player')
            
            row = {
                'game_id': game_id,
                'date': date_key,
                'step': step_idx,
                'player_count': len(final_rewards),
                'winner_idx': winner_idx,
                'team_names': team_names,
                'final_rewards': final_rewards,
                'statuses': statuses,
                'seed': seed,
                'expert_player_id': expert_player_id,
                'player': expert_player_id,
                'planets': obs.get('planets', []),
                'initial_planets': obs.get('initial_planets', []),
                'fleets': obs.get('fleets', []),
                'angular_velocity': obs.get('angular_velocity', 0.0),
                'comet_planet_ids': obs.get('comet_planet_ids', []),
                'comets': obs.get('comets', []),
                'next_fleet_id': obs.get('next_fleet_id'),
                'remaining_overage_time': obs.get('remainingOverageTime'),
                'action': action,
            }

            flat_steps.append(row)
            
        del replay
        cleanup_memory()
        return date_key, flat_steps
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        cleanup_memory()
        return date_key, []

def save_parquet_batch(output_dir, date_key, batch_index, rows, total_processed, total_files, final=False):
    if not rows:
        return
    date_dir = output_dir / date_key
    date_dir.mkdir(parents=True, exist_ok=True)
    out_path = date_dir / f"batch-{batch_index}.parquet"
    tmp_path = date_dir / f".{out_path.name}.tmp"
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(tmp_path, engine='pyarrow', compression='zstd')
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    label = "Saved final" if final else "Saved"
    print(f"[{total_processed}/{total_files}] {label} {date_key}/{out_path.name} with {len(df)} turns.")
    del df
    cleanup_memory()

if __name__ == '__main__':
    KAGGLE_MODE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
    repo_root = Path(__file__).resolve().parents[2]
    
    if KAGGLE_MODE:
        print("Running in KAGGLE_MODE")
        input_dir = Path('/kaggle/input')
        output_dir = Path(os.environ.get("MINIFY_OUTPUT_DIR", "/kaggle/working"))
        raw_files = list(input_dir.rglob("*.json"))
    else:
        print("Running in LOCAL_MODE")
        input_dir = Path(os.environ.get("MINIFY_INPUT_DIR", repo_root / "data" / "raw"))
        output_dir = Path(os.environ.get("MINIFY_OUTPUT_DIR", repo_root / "data"))
        raw_files = list(input_dir.glob("*.json"))

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = [f for f in raw_files if "minified" not in f.name and "manifest" not in f.name]
    raw_files = sorted(raw_files, key=lambda p: (date_key_from_path(p), str(p)))
    print(f"Found {len(raw_files)} JSON files to process.")
    
    TEST_LIMIT = env_int("MINIFY_TEST_LIMIT", 0) or None
    if TEST_LIMIT:
        raw_files = raw_files[:TEST_LIMIT]
        print(f"Applying TEST_LIMIT of {TEST_LIMIT} files.")
    
    max_workers = min(env_int("MINIFY_NUM_WORKERS", 4), os.cpu_count() or 4)
    batch_size = env_int("MINIFY_GAMES_PER_PARQUET", 1000)
    executor_chunksize = env_int("MINIFY_EXECUTOR_CHUNKSIZE", 10)
    
    current_batch_data = []
    current_date = None
    current_date_games = 0
    batch_index_by_date = {}
    total_processed = 0
    
    print(f"Output directory: {output_dir}")
    print(f"Starting Parquet batch processing with {max_workers} workers...")
    print(f"Games per parquet: {batch_size}")
    print(f"Executor chunksize: {executor_chunksize}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to stream results instead of storing 31,000 Future objects in RAM
        for date_key, steps_data in executor.map(process_single_replay, raw_files, chunksize=executor_chunksize):
            if current_date is None:
                current_date = date_key

            if date_key != current_date:
                if current_batch_data:
                    save_parquet_batch(
                        output_dir,
                        current_date,
                        batch_index_by_date.get(current_date, 1),
                        current_batch_data,
                        total_processed,
                        len(raw_files),
                        final=True,
                    )
                    batch_index_by_date[current_date] = batch_index_by_date.get(current_date, 1) + 1
                    current_batch_data.clear()
                    cleanup_memory()

                current_date = date_key
                current_date_games = 0

            if steps_data:
                current_batch_data.extend(steps_data)
            del steps_data
                
            total_processed += 1
            current_date_games += 1
            
            # If we've processed a date-local batch size of games, export to Parquet.
            if current_date_games % batch_size == 0:
                if current_batch_data:
                    batch_index = batch_index_by_date.get(current_date, 1)
                    save_parquet_batch(
                        output_dir,
                        current_date,
                        batch_index,
                        current_batch_data,
                        total_processed,
                        len(raw_files),
                    )
                    
                    # Force Memory Cleanup
                    current_batch_data.clear()
                    cleanup_memory()
                    
                    batch_index_by_date[current_date] = batch_index + 1
                
        # Save any remaining data in the final batch
        if current_batch_data:
            save_parquet_batch(
                output_dir,
                current_date or "unknown-date",
                batch_index_by_date.get(current_date or "unknown-date", 1),
                current_batch_data,
                total_processed,
                len(raw_files),
                final=True,
            )
            current_batch_data.clear()
            cleanup_memory()
            
    print("Processing complete!")
