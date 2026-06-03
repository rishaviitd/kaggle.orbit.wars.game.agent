import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import orjson
    import pandas as pd
    import pyarrow
except ImportError:
    os.system("pip install orjson pandas pyarrow")
    import orjson
    import pandas as pd

def process_single_replay(json_path):
    """Reads a raw JSON, minifies it, and returns a flat list of dicts (one per turn)."""
    try:
        with open(json_path, 'rb') as f:
            replay = orjson.loads(f.read())
            
        final_rewards = replay.get('rewards', [])
        if not final_rewards:
            return []
            
        winner_idx = max(range(len(final_rewards)), key=lambda i: final_rewards[i] if final_rewards[i] is not None else -1)
        
        flat_steps = []
        game_id = Path(json_path).stem
        
        for step_idx, step_data in enumerate(replay['steps']):
            winner_data = step_data[winner_idx]
            if winner_data is None or 'observation' not in winner_data:
                continue
                
            obs = winner_data['observation']
            action = winner_data.get('action', [])
            
            flat_steps.append({
                'game_id': game_id,
                'step': step_idx,
                'player': obs.get('player'),
                'planets': obs.get('planets', []),
                'initial_planets': obs.get('initial_planets', []),
                'fleets': obs.get('fleets', []),
                'angular_velocity': obs.get('angular_velocity', 0.0),
                'comet_planet_ids': obs.get('comet_planet_ids', []),
                'comets': obs.get('comets', []),
                'action': action 
            })
            
        return flat_steps
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return []

if __name__ == '__main__':
    KAGGLE_MODE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
    
    if KAGGLE_MODE:
        print("Running in KAGGLE_MODE")
        input_dir = Path('/kaggle/input')
        output_dir = Path('/kaggle/working')
        raw_files = list(input_dir.rglob("*.json"))
    else:
        print("Running in LOCAL_MODE")
        input_dir = Path(os.path.join(os.path.dirname(__file__), '../data/raw'))
        output_dir = Path(os.path.join(os.path.dirname(__file__), '../data/processed'))
        raw_files = list(input_dir.glob("*.json"))

    raw_files = [f for f in raw_files if "minified" not in f.name and "manifest" not in f.name]
    print(f"Found {len(raw_files)} JSON files to process.")
    
    TEST_LIMIT = None 
    if TEST_LIMIT:
        raw_files = raw_files[:TEST_LIMIT]
        print(f"Applying TEST_LIMIT of {TEST_LIMIT} files.")
    
    import gc
    max_workers = os.cpu_count() or 4
    batch_size = 500 # Reduced batch size to prevent Pandas OOM during conversion
    
    current_batch_data = []
    batch_index = 1
    total_processed = 0
    
    print(f"Starting Parquet batch processing with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to stream results instead of storing 31,000 Future objects in RAM
        for steps_data in executor.map(process_single_replay, raw_files, chunksize=10):
            if steps_data:
                current_batch_data.extend(steps_data)
                
            total_processed += 1
            
            # If we've processed a batch size of games, export to Parquet
            if total_processed % batch_size == 0:
                if current_batch_data:
                    out_path = output_dir / f"batch_{batch_index}.parquet"
                    df = pd.DataFrame(current_batch_data)
                    df.to_parquet(out_path, engine='pyarrow', compression='zstd')
                    print(f"[{total_processed}/{len(raw_files)}] Saved {out_path.name} with {len(df)} turns.")
                    
                    # Force Memory Cleanup
                    del df
                    current_batch_data.clear()
                    gc.collect()
                    
                    batch_index += 1
                
        # Save any remaining data in the final batch
        if current_batch_data:
            out_path = output_dir / f"batch_{batch_index}.parquet"
            df = pd.DataFrame(current_batch_data)
            df.to_parquet(out_path, engine='pyarrow', compression='zstd')
            print(f"[{total_processed}/{len(raw_files)}] Saved final {out_path.name} with {len(df)} turns.")
            del df
            current_batch_data.clear()
            gc.collect()
            
    print("Processing complete!")
