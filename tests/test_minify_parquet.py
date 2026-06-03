import json
import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path so we can import our minifier
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/data_prep'))
from minify_replay import minify_replay

def test_pipeline():
    raw_dir = Path(os.path.join(os.path.dirname(__file__), '../data/raw'))
    processed_dir = Path(os.path.join(os.path.dirname(__file__), '../data/processed'))
    
    # Get all raw replays that aren't already minified
    json_files = [f for f in raw_dir.glob("*.json") if "minified" not in f.name]
    
    if len(json_files) == 0:
        print("No raw JSON files found!")
        return
        
    print(f"Starting test on {len(json_files)} replay files...\n")
    
    success_count = 0
    total_raw_size = 0
    total_parquet_size = 0
    
    for file in json_files:
        minified_file = raw_dir / f"{file.stem}_minified.json"
        parquet_file = processed_dir / f"{file.stem}.parquet"
        
        try:
            # 1. Test Minifier
            res = minify_replay(file, minified_file)
            if not res:
                print(f"Skipped {file.name} (No rewards array found)")
                continue
                
            # 2. Test Pandas Loading
            with open(minified_file, 'r') as f:
                d = json.load(f)
                
            df = pd.DataFrame(d['steps'])
            
            # 3. Test Parquet ZSTD export
            df.to_parquet(parquet_file, engine='pyarrow', compression='zstd')
            
            # 4. Test Parquet Read (Data Integrity)
            df_test = pd.read_parquet(parquet_file, engine='pyarrow')
            assert len(df_test) == len(df), "Row count mismatch after Parquet serialization!"
            assert 'planets' in df_test.columns, "Missing columns in Parquet!"
            assert 'action' in df_test.columns, "Missing columns in Parquet!"
            
            # Track Sizes
            raw_size = os.path.getsize(file)
            pq_size = os.path.getsize(parquet_file)
            total_raw_size += raw_size
            total_parquet_size += pq_size
            
            print(f"[PASS] {file.name}: {raw_size/1024:.0f}KB -> {pq_size/1024:.0f}KB ({(1 - pq_size/raw_size)*100:.1f}%) | {len(df)} turns")
            
            # Clean up the intermediate minified json to save space
            if os.path.exists(minified_file):
                os.remove(minified_file)
                
            success_count += 1
            
        except Exception as e:
            print(f"[FAIL] {file.name}: {str(e)}")
            
    print(f"\n--- TEST SUMMARY ---")
    print(f"Files Processed successfully: {success_count} / {len(json_files)}")
    print(f"Total Raw Size: {total_raw_size / (1024*1024):.2f} MB")
    print(f"Total Parquet Size: {total_parquet_size / (1024*1024):.2f} MB")
    if total_raw_size > 0:
        print(f"Overall Compression: {(1 - total_parquet_size/total_raw_size)*100:.2f}%")

if __name__ == '__main__':
    test_pipeline()
