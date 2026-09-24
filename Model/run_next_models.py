#!/usr/bin/env python3
"""Prepare features or run the next fixed CPU model recipes."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model11.pipeline import prepare, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--models", type=int, nargs="+", choices=(11, 12, 13), default=[11, 12, 13])
    args = parser.parse_args()
    if args.command == "prepare":
        x, layout = prepare()
        print(f"Ready: {x.shape}, {layout.series_count} development series")
    else:
        for number in args.models:
            run(number)


if __name__ == "__main__":
    main()
