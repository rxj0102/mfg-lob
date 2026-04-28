"""Generate sample synthetic LOB data for testing and calibration.

Usage:
    python scripts/generate_sample_data.py [--n-snapshots 100] [--output data/sample.npy]
"""

from __future__ import annotations
import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic_lob import SyntheticLOBConfig, generate_lob_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic LOB data")
    parser.add_argument("--n-snapshots", type=int, default=100)
    parser.add_argument("--depth-shape", default="power_law",
                        choices=["power_law", "exponential", "gaussian"])
    parser.add_argument("--depth-param", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/sample_lob.npy")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = SyntheticLOBConfig(
        depth_shape=args.depth_shape,
        depth_param=args.depth_param,
    )

    snapshots = [generate_lob_snapshot(cfg, rng=rng) for _ in range(args.n_snapshots)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), snapshots, allow_pickle=True)
    print(f"Saved {args.n_snapshots} LOB snapshots to {out_path}")


if __name__ == "__main__":
    main()
