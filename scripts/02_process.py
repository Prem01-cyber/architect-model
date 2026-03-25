#!/usr/bin/env python3
"""
02_process.py
─────────────
Normalises raw collected records and builds (input, output) training pairs.

Usage:
    python scripts/02_process.py --input data/raw/ --output data/processed/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.processors.normalizer import StructureNormalizer
from data.processors.pair_builder import PairBuilder
from data.collectors.github_collector import DOMAIN_LANGUAGE_MATRIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/")
    parser.add_argument("--output", default="data/processed/")
    parser.add_argument("--min-stars", type=int, default=50)
    parser.add_argument("--min-files", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = StructureNormalizer()
    builder = PairBuilder(domain_language_matrix=DOMAIN_LANGUAGE_MATRIX)

    raw_files = list(input_dir.glob("*.jsonl"))
    logger.info(f"Processing {len(raw_files)} raw files from {input_dir}")

    all_pairs = []
    stats = {"read": 0, "normalized": 0, "paired": 0, "skipped": 0}

    for raw_file in raw_files:
        with open(raw_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["read"] += 1

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped"] += 1
                    continue

                # Quality gate
                if raw.get("stars", 0) < args.min_stars:
                    stats["skipped"] += 1
                    continue

                normalized = normalizer.normalize(raw)
                if not normalized:
                    stats["skipped"] += 1
                    continue

                if normalized.num_files < args.min_files:
                    stats["skipped"] += 1
                    continue

                stats["normalized"] += 1

                pair = builder.build(normalized)
                if not pair:
                    stats["skipped"] += 1
                    continue

                stats["paired"] += 1
                all_pairs.append(pair.to_dict())

    logger.info(f"Stats: {stats}")
    logger.info(f"Total training pairs: {len(all_pairs)}")

    # Write all pairs to a single JSONL file
    output_file = output_dir / "pairs.jsonl"
    with open(output_file, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Written to {output_file}")

    # Print distribution across domains
    from collections import Counter

    domain_counts = Counter(p["metadata"]["domain"] for p in all_pairs)
    logger.info("\nDomain distribution:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain:30s}: {count}")


if __name__ == "__main__":
    main()
