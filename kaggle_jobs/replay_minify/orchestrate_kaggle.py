import os
import json
import shutil
import time
import subprocess
import pandas as pd
from pathlib import Path

def orchestrate():
    root_dir = Path(__file__).parent.parent.parent
    manifest_path = root_dir / "data" / "raw" / "manifest.csv"
    runner_script = root_dir / "kaggle_jobs" / "replay_minify" / "minify_replay.py"
    
    if not manifest_path.exists():
        print("Manifest not found! Cannot orchestrate.")
        return
        
    df = pd.read_csv(manifest_path)
    slugs = ["kaggle/" + slug for slug in df['daily_dataset_slug'].dropna()]
    
    print(f"Total datasets to process: {len(slugs)}")
    
    num_chunks = 10
    chunk_size = len(slugs) // num_chunks
    
    # Split slugs into 10 chunks cleanly
    chunks = [slugs[i:i + chunk_size] for i in range(0, len(slugs), chunk_size)]
    if len(chunks) > num_chunks:
        chunks[-2].extend(chunks[-1])
        chunks = chunks[:-1]
        
    print(f"Splitting datasets perfectly into {num_chunks} chunks.")
    
    # ==========================================
    # TOGGLE THIS TO 1 OR 2
    WAVE = 2
    # ==========================================
    
    if WAVE == 1:
        wave_chunks = chunks[0:5] # Chunks 1 to 5
        start_part_num = 1
    elif WAVE == 2:
        wave_chunks = chunks[5:10] # Chunks 6 to 10
        start_part_num = 6
    else:
        print("Invalid WAVE. Must be 1 or 2.")
        return
        
    print(f"--- RUNNING WAVE {WAVE} (Parts {start_part_num} to {start_part_num + len(wave_chunks) - 1}) ---")
    
    for idx, chunk in enumerate(wave_chunks):
        part_num = start_part_num + idx
        
        # Fresh distinct naming! No more v2 or v3 confusion.
        job_id = f"atomstack001/orbit-parquet-extractor-p{part_num}"
        tmp_dir = root_dir / f"kaggle_runner_tmp_{part_num}"
        
        # Setup temp directory
        tmp_dir.mkdir(exist_ok=True)
        shutil.copy(runner_script, tmp_dir / "minify_replay.py")
        
        # Generate metadata
        metadata = {
          "id": job_id,
          "title": f"Orbit Parquet Extractor - P{part_num}",
          "code_file": "minify_replay.py",
          "language": "python",
          "kernel_type": "script",
          "is_private": "true",
          "enable_gpu": "false",
          "enable_tpu": "false",
          "enable_internet": "true",
          "dataset_sources": chunk,
          "competition_sources": [],
          "kernel_sources": [],
          "model_sources": []
        }
        
        with open(tmp_dir / "kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Pushing Job {part_num}/10 [{job_id}] with {len(chunk)} datasets attached...")
        
        try:
            # Run the kaggle push command
            subprocess.run(
                ["uv", "run", "kaggle", "kernels", "push", "-p", str(tmp_dir)],
                cwd=root_dir,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error pushing job {part_num}: {e}")
            break
            
        # Clean up temp directory
        shutil.rmtree(tmp_dir)
        
        if idx < len(wave_chunks) - 1:
            print("Sleeping for 15 seconds to respect Kaggle API rate limits...")
            time.sleep(15)
            
    print(f"\nWave {WAVE} jobs successfully orchestrated!")
    
if __name__ == '__main__':
    orchestrate()
