#!/usr/bin/env python3
"""
03_augment.py
─────────────
1. Rewrites each goal N times for phrasing variety
2. Generates synthetic examples for sparse (domain, language) cells
3. Produces final train.jsonl and val.jsonl

Usage:
    python scripts/03_augment.py --input data/processed/ --output data/augmented/
    OPENAI_API_KEY=sk-... python scripts/03_augment.py --input data/processed/ \
        --output data/augmented/ --use-openai
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentors.goal_augmentor import GoalAugmentor
from data.collectors.github_collector import DOMAIN_LANGUAGE_MATRIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Minimum examples per (domain, language) cell before considering it "covered"
MIN_CELL_COVERAGE = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/")
    parser.add_argument("--output", default="data/augmented/")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    aug_cfg = config.get("augmentation", {})
    n_rewrites = aug_cfg.get("rewrites_per_example", 5)

    # Set up OpenAI client if requested
    openai_client = None
    if args.use_openai:
        try:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set — falling back to rule-based augmentation"
                )
            else:
                openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client ready")
        except ImportError:
            logger.warning("openai package not installed")

    augmentor = GoalAugmentor(
        openai_client=openai_client,
        rewrites_per_example=n_rewrites,
    )

    # ── Load processed pairs ──────────────────────────────────────────────────
    input_file = Path(args.input) / "pairs.jsonl"
    all_pairs = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                all_pairs.append(json.loads(line))

    logger.info(f"Loaded {len(all_pairs)} base pairs")

    # ── Augment each pair ─────────────────────────────────────────────────────
    augmented: list[dict] = []
    for pair in all_pairs:
        variants = augmentor.augment(pair)
        augmented.extend(variants)

    logger.info(f"After augmentation: {len(augmented)} pairs")

    # ── Fill sparse cells ─────────────────────────────────────────────────────
    cell_counts: dict[tuple, int] = defaultdict(int)
    for p in augmented:
        domain = p["metadata"]["domain"]
        lang = p["output"]["file_structure"]["language"]
        cell_counts[(domain, lang)] += 1

    sparse_cells = [
        (domain, lang)
        for domain, langs in DOMAIN_LANGUAGE_MATRIX.items()
        for lang in langs
        if cell_counts[(domain, lang)] < MIN_CELL_COVERAGE
    ]

    if sparse_cells and openai_client:
        logger.info(
            f"Filling {len(sparse_cells)} sparse cells via synthetic generation"
        )
        for domain, lang in sparse_cells:
            current = cell_counts[(domain, lang)]
            needed = MIN_CELL_COVERAGE - current
            logger.info(f"  {domain}/{lang}: {current} → generating {needed} more")
            synthetic = augmentor.generate_synthetic(domain, lang, n=needed)
            augmented.extend(synthetic)
    elif sparse_cells:
        logger.warning(
            f"{len(sparse_cells)} sparse cells found but OpenAI not available for synthetic fill. "
            f"Pass --use-openai to fill them."
        )

    logger.info(f"Total pairs after sparse fill: {len(augmented)}")

    # ── Train/val split ───────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(augmented)

    n_val = max(100, int(len(augmented) * 0.05))
    val_pairs = augmented[:n_val]
    train_pairs = augmented[n_val:]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(pairs: list[dict], path: Path) -> None:
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

    write_jsonl(train_pairs, output_dir / "train.jsonl")
    write_jsonl(val_pairs, output_dir / "val.jsonl")

    logger.info(f"\nFinal split:")
    logger.info(f"  Train: {len(train_pairs)} pairs → {output_dir}/train.jsonl")
    logger.info(f"  Val:   {len(val_pairs)} pairs → {output_dir}/val.jsonl")

    # Coverage report
    final_counts: dict[tuple, int] = defaultdict(int)
    for p in train_pairs:
        domain = p["metadata"]["domain"]
        lang = p["output"]["file_structure"]["language"]
        final_counts[(domain, lang)] += 1

    logger.info("\nCoverage per (domain, language) cell:")
    for (domain, lang), count in sorted(final_counts.items()):
        bar = "█" * min(count // 5, 30)
        logger.info(f"  {domain:25s} / {lang:12s}: {count:4d} {bar}")


if __name__ == "__main__":
    main()
