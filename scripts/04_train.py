#!/usr/bin/env python3
"""
04_train.py
───────────
Fine-tunes the base model with LoRA (QLoRA 4-bit).

Usage:
    python scripts/04_train.py --config configs/train_config.yaml
    python scripts/04_train.py --config configs/train_config.yaml --resume-from outputs/checkpoints/checkpoint-400
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import ArchitectTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument(
        "--resume-from", default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--base-model", default=None, help="Override base model in config"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Override output dir in config"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.base_model:
        config["model"]["base_model"] = args.base_model
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.resume_from:
        config["training"]["resume_from_checkpoint"] = args.resume_from

    logger.info("Training configuration:")
    logger.info(f"  Base model:  {config['model']['base_model']}")
    logger.info(f"  LoRA rank:   {config['lora']['r']}")
    logger.info(f"  Epochs:      {config['training']['num_train_epochs']}")
    logger.info(
        f"  Batch size:  {config['training']['per_device_train_batch_size']} "
        f"× {config['training']['gradient_accumulation_steps']} grad accum "
        f"= {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']} effective"
    )
    logger.info(f"  LR:          {config['training']['learning_rate']}")
    logger.info(f"  Train data:  {config['data']['train_path']}")
    logger.info(f"  Output:      {config['training']['output_dir']}")

    trainer = ArchitectTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
