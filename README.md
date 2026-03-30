# Architect Model — Training Pipeline

Fine-tunes a 7B code model to act as a software architect:
given any project goal, it recommends the best language with
reasoning and generates a canonical file structure.

## Project layout

```
architect-model/
├── configs/
│   └── train_config.yaml        # all hyperparameters
├── data/
│   ├── collectors/
│   │   └── github_collector.py  # scrapes GitHub by topic+language
│   ├── processors/
│   │   ├── normalizer.py        # cleans raw repo structures
│   │   └── pair_builder.py      # builds (goal, output) training pairs
│   └── augmentors/
│       └── goal_augmentor.py    # rewrites goals 5x for variety
├── training/
│   ├── dataset.py               # PyTorch Dataset + collator
│   ├── trainer.py               # LoRA fine-tune loop
│   └── callbacks.py             # checkpoint + eval callbacks
├── inference/
│   ├── model.py                 # inference wrapper with constrained decoding
│   └── schema.py                # pydantic output schemas
├── evaluation/
│   └── evealuator.py             # structure validity + consistency metrics
└── scripts/
    ├── 01_collect.py            # run data collection
    ├── 02_process.py            # run normalization + pair building
    ├── 03_augment.py            # run goal augmentation
    ├── 04_train.py              # run fine-tuning
    └── 05_evaluate.py           # run evaluation
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Collect data from GitHub (needs GITHUB_TOKEN env var)
python scripts/01_collect.py --output data/raw/

# 2. Process into training pairs
python scripts/02_process.py --input data/raw/ --output data/processed/

# 3. Augment goals for variety
python scripts/03_augment.py --input data/processed/ --output data/augmented/

# 4. Fine-tune
python scripts/04_train.py --config configs/train_config.yaml

# 5. Evaluate
python scripts/05_evaluate.py --model outputs/final/
```

## Requirements

- Python 3.10+
- CUDA GPU (1x A100 40GB minimum for 7B LoRA, or use 2x A6000)
- GITHUB_TOKEN for data collection
- OPENAI_API_KEY for synthetic reasoning augmentation (optional)
