"""
dataset.py
──────────
PyTorch Dataset that reads JSONL training pairs and tokenises them
into the chat format expected by the base model.

The assistant turn is the only part that contributes to the loss —
system and user turns are masked out with -100.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def build_chat_prompt(messages: list[dict], tokenizer: PreTrainedTokenizer) -> str:
    """
    Applies the model's chat template to a list of messages.
    Falls back to a generic template if the tokenizer doesn't have one.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Generic fallback
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
    return "".join(parts)


class ArchitectDataset(Dataset):

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        split: str = "train",
        val_split: float = 0.05,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        all_examples = self._load(Path(data_path))

        # Deterministic train/val split
        import random

        rng = random.Random(seed)
        rng.shuffle(all_examples)

        n_val = max(1, int(len(all_examples) * val_split))
        if split == "val":
            self.examples = all_examples[:n_val]
        else:
            self.examples = all_examples[n_val:]

        logger.info(f"Loaded {len(self.examples)} examples for split='{split}'")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        pair = self.examples[idx]
        messages = self._pair_to_messages(pair)
        return self._tokenize(messages)

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> list[dict]:
        examples = []
        # Accept a single file or a directory of .jsonl files
        files = list(path.glob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        logger.info(f"Read {len(examples)} raw examples from {path}")
        return examples

    def _pair_to_messages(self, pair: dict) -> list[dict]:
        """Convert stored pair dict back to chat message list."""
        system_msg = (
            "You are an expert software architect. "
            "Given a project goal, you:\n"
            "1. Recommend the best programming language with clear technical reasoning, "
            "listing a primary choice and up to two alternatives.\n"
            "2. Generate a canonical file structure for a basic implementation "
            "following real-world conventions.\n\n"
            "Always respond with a single JSON object. No text outside the JSON."
        )

        inp = pair["input"]
        scale_hint = {
            "basic": "This is a basic/educational implementation.",
            "intermediate": "This is a functional intermediate implementation.",
            "production": "This is a production-scale implementation.",
        }.get(inp.get("scale", "basic"), "")

        constraints = inp.get("constraints", [])
        constraint_str = ""
        if constraints:
            constraint_str = "\nConstraints: " + ", ".join(constraints)

        user_msg = f"Goal: {inp['goal']}\n{scale_hint}{constraint_str}".strip()
        assistant_msg = json.dumps(pair["output"], indent=2)

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]

    def _tokenize(self, messages: list[dict]) -> dict:
        full_text = build_chat_prompt(messages, self.tokenizer)

        # Tokenise the full conversation
        full_ids = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Build labels: mask out everything except the assistant turn
        # We do this by finding where the assistant response starts
        assistant_text = json.dumps(
            messages[-1]["content"]
            if isinstance(messages[-1]["content"], dict)
            else messages[-1]["content"]
        )

        # Encode just the prefix (system + user) to find the split point
        prefix_messages = messages[:-1]
        prefix_text = build_chat_prompt(prefix_messages, self.tokenizer)
        # Add the assistant start token so we mask it correctly
        if hasattr(self.tokenizer, "apply_chat_template"):
            prefix_with_start = self.tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,  # adds the assistant prefix token
            )
        else:
            prefix_with_start = prefix_text + "<|assistant|>\n"

        prefix_ids = self.tokenizer(
            prefix_with_start,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        prefix_len = len(prefix_ids)

        # Labels: -100 for prefix (no loss), real ids for assistant response
        labels = full_ids.clone()
        labels[:prefix_len] = -100

        # Pad/truncate to max_seq_length
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        seq_len = len(full_ids)

        if seq_len < self.max_seq_length:
            pad_len = self.max_seq_length - seq_len
            full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
            attention_mask = torch.cat(
                [
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ]
            )
        else:
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollator:
    """Simple collator — Dataset already pads to max_seq_length."""

    def __call__(self, features: list[dict]) -> dict:
        return {key: torch.stack([f[key] for f in features]) for key in features[0]}
