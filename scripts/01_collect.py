#!/usr/bin/env python3
"""
01_collect.py
─────────────
Scrapes GitHub for repos across all (domain, language) cells.

Usage:
    GITHUB_TOKEN=ghp_... python scripts/01_collect.py --output data/raw/
    python scripts/01_collect.py --output data/raw/ --max-per-cell 100
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collectors.github_collector import GitHubCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect GitHub repo structures")
    parser.add_argument("--output", default="data/raw/", help="Output directory")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument(
        "--max-per-cell",
        type=int,
        default=None,
        help="Override max repos per (domain, language) cell",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    collection_cfg = config.get("collection", {})
    max_per_cell = args.max_per_cell or collection_cfg.get("max_repos_per_cell", 200)

    output_dir = Path(args.output)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max repos per cell: {max_per_cell}")

    collector = GitHubCollector(token=token, config=collection_cfg)
    collector.collect_all(output_dir=output_dir, max_per_cell=max_per_cell)

    # Summary
    files = list(output_dir.glob("*.jsonl"))
    total_records = sum(sum(1 for _ in open(f)) for f in files)
    logger.info(f"\nCollection complete.")
    logger.info(f"  Files: {len(files)}")
    logger.info(f"  Total records: {total_records}")


if __name__ == "__main__":
    main()
