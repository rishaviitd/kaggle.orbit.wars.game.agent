from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


KERNEL_SOURCE = "atomstack001/orbit-datewise-replay-minify-fullmeta-latest10"


def valid_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; use YYYY-MM-DD") from exc
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push one Kaggle graph-generation kernel per requested date."
    )
    parser.add_argument("dates", nargs="+", type=valid_date)
    args = parser.parse_args()

    job_dir = Path(__file__).resolve().parent
    generator = job_dir / "generate_graphs.py"
    generator_source = generator.read_text(encoding="utf-8")
    date_marker = "PACKAGED_TARGET_DATE = None"
    if generator_source.count(date_marker) != 1:
        raise SystemExit(f"Expected exactly one {date_marker!r} marker in {generator}.")
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit("kaggle command not found; run this utility with `uv run python`.")

    for date in args.dates:
        slug = f"orbit-simple-gnn-features-{date}"
        metadata = {
            "id": f"atomstack001/{slug}",
            "title": f"Orbit Simple GNN Features {date}",
            "code_file": "generate_graphs.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [KERNEL_SOURCE],
            "model_sources": [],
        }

        with tempfile.TemporaryDirectory(prefix=f"{slug}-") as tmp:
            package = Path(tmp)
            packaged_source = generator_source.replace(
                date_marker,
                f'PACKAGED_TARGET_DATE = "{date}"',
            )
            (package / generator.name).write_text(packaged_source, encoding="utf-8")
            (package / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"\nPushing {metadata['id']}...")
            subprocess.run(
                [kaggle, "kernels", "push", "-p", str(package)],
                check=True,
            )


if __name__ == "__main__":
    main()
