#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from src.pipeline import rank_candidates, PASS2_POOL_SIZE

def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Bug Hunters - Redrob Hackathon Ranker v6.0\n"
                    f"Two-pass: 100K stream → fast filter (top {PASS2_POOL_SIZE}) → deep score → CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl, .jsonl.gz, or sample_candidates.json")
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g. bug_hunters.csv)")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)

if __name__ == "__main__":
    main()
