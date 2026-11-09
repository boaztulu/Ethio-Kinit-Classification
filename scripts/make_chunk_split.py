"""Create results/split_chunk_level.json from cache/index.json."""
from pathlib import Path
from src.data.splits import chunk_level_split, save_split

if __name__ == "__main__":
    sp = chunk_level_split("cache/index.json", seed=1234)
    save_split(sp, Path("results/split_chunk_level.json"))
    print(f"chunk-split sizes: train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}")
