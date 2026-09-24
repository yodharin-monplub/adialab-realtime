"""Sequential matched inference benchmarks after compatibility/warm-up checks."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.model23.deployment.validate import benchmark


if __name__ == '__main__':
    # Run separately, never concurrently: both see the same configured worker
    # count and deterministic 24-series sample. Model22 may continue in parallel.
    benchmark(24, 2, baseline=True)
    benchmark(24, 2, baseline=False)
