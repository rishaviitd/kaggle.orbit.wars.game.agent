# Kaggle Remote Execution Setup

This directory is strictly configured for executing massive data processing jobs on Kaggle's cloud infrastructure from your local machine.

Because the Kaggle `orbit-wars-episodes` dataset is over 1.2 Terabytes, it is impossible to download and process it locally. Instead, we use this setup to push our Python scripts to Kaggle, run them where the data natively lives, and download *only* the tiny, processed result files back to our Mac.

---

## The Workflow

### Step 1: Authentication
Before you can interact with Kaggle from the command line, you must authenticate.
1. Make sure the CLI is installed: `uv pip install kaggle`
2. Run the web-based authentication flow in your terminal:
   ```bash
   uv run kaggle auth login
   ```
   *This opens a browser where you log in securely. Your credentials are automatically cached locally.*

### Step 2: The Metadata File
The magic happens inside `kernel-metadata.json`. This file tells Kaggle exactly what to do with the code you push.
- **`id`**: Must be in the format `your-username/kernel-slug` (e.g., `atomstack001/orbit-wars-data-minifier`).
- **`dataset_sources`**: This is critical. By listing a Kaggle dataset here (e.g., `kaggle/orbit-wars-episodes-2026-04-16`), Kaggle automatically mounts that massive dataset into the `/kaggle/input/` directory of your running server for free.

### Step 3: Failsafe Imports
Kaggle's base Python environment sometimes lacks specialized, high-performance libraries like `orjson`. To prevent your script from crashing in the cloud, always inject a failsafe pip install at the very top of your execution script:
```python
import os
try:
    import orjson
except ImportError:
    os.system("pip install orjson")
    import orjson
```

### Step 4: Pushing to the Cloud
Once your script and `kernel-metadata.json` are ready in this folder, you trigger the cloud job by running:
```bash
uv run kaggle kernels push -p kaggle_runner
```
Kaggle will package everything in this folder, upload it, and spin up a cloud server. You can check the status at any time with:
```bash
uv run kaggle kernels status atomstack001/orbit-wars-data-minifier
```

### Step 5: Retrieving the Data
Our scripts are designed to write their output to `/kaggle/working/`. When the Kaggle job finishes, you don't need to use the website to get your data. You simply pull the outputs straight back to your local machine:
```bash
uv run kaggle kernels output atomstack001/orbit-wars-data-minifier -p data/processed/
```

This workflow gives us infinite scale while keeping our local development environment clean and fast!
