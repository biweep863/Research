"""Downloads the BEANS-Watkins parquet shards needed for the gpu_full run.

The dataset ships 5 files: train-00000-of-00002, train-00001-of-00002,
valid, test, and train_low. We download the first four and deliberately
skip train_low, which is very likely a subset of train and would let the
same clip appear in both a training split and a held-out split if mixed
in (see doc/01_dataset.md and src/data/watkins.py::EXCLUDED_SHARDS).
"""
from __future__ import annotations

from pathlib import Path

REPO_ID = "DBD-research-group/beans_watkins"
FILES = [
    "data/train-00000-of-00002-*.parquet",
    "data/train-00001-of-00002-*.parquet",
    "data/valid-00000-of-00001-*.parquet",
    "data/test-00000-of-00001-*.parquet",
]


def main():
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet") and "train_low" not in f]
    if not parquet_files:
        raise RuntimeError(f"No non-train_low parquet files found in {REPO_ID}")

    print(f"[fetch_watkins] downloading {len(parquet_files)} shards (train_low excluded):")
    for f in parquet_files:
        print(f"  {f}")
        path = hf_hub_download(REPO_ID, f, repo_type="dataset", local_dir=str(out_dir))
        # flatten into data/raw/ regardless of the repo's internal folder layout
        dest = out_dir / Path(f).name
        if Path(path) != dest:
            dest.write_bytes(Path(path).read_bytes())
    print(f"[fetch_watkins] done -> {out_dir}")


if __name__ == "__main__":
    main()
