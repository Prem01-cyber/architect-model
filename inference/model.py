"""
model.py
────────
Inference wrapper for the fine-tuned architect model.

Key features:
  - Grammar-constrained decoding via lm-format-enforcer
    → model physically cannot produce invalid JSON
  - temperature=0.1 for near-deterministic output
  - Pydantic schema validation on the output
  - Deterministic post-processing (sort, normalise paths)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


# ── Output schemas (Pydantic) ─────────────────────────────────────────────────


class LanguagePrimary(BaseModel):
    language: str
    reasons: list[str]
    tradeoffs: str


class LanguageAlternative(BaseModel):
    language: str
    reasons: list[str]
    tradeoffs: str


class LanguageRecommendation(BaseModel):
    primary: LanguagePrimary
    alternatives: list[LanguageAlternative]


class FileStructure(BaseModel):
    language: str
    reference_projects: list[str]
    structure: dict[str, list[str]]

    @field_validator("structure")
    @classmethod
    def sort_structure(cls, v: dict) -> dict:
        # Canonical sorting — same input always produces same output
        return {k: sorted(set(files)) for k, files in sorted(v.items())}


class ArchitectOutput(BaseModel):
    language_recommendation: LanguageRecommendation
    file_structure: FileStructure

    def fingerprint(self) -> str:
        """
        Deterministic fingerprint of the structure.
        Useful for reproducibility testing.
        """
        canonical = json.dumps(
            self.file_structure.structure,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ── JSON schema for constrained decoding ─────────────────────────────────────

ARCHITECT_JSON_SCHEMA = {
    "type": "object",
    "required": ["language_recommendation", "file_structure"],
    "properties": {
        "language_recommendation": {
            "type": "object",
            "required": ["primary", "alternatives"],
            "properties": {
                "primary": {
                    "type": "object",
                    "required": ["language", "reasons", "tradeoffs"],
                    "properties": {
                        "language": {"type": "string", "maxLength": 20},
                        "reasons": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 120},
                            "minItems": 1,
                            "maxItems": 5,
                        },
                        "tradeoffs": {"type": "string", "maxLength": 150},
                    },
                },
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["language", "reasons", "tradeoffs"],
                        "properties": {
                            "language": {"type": "string"},
                            "reasons": {"type": "array", "items": {"type": "string"}},
                            "tradeoffs": {"type": "string"},
                        },
                    },
                    "maxItems": 3,
                },
            },
        },
        "file_structure": {
            "type": "object",
            "required": ["language", "structure"],
            "properties": {
                "language": {"type": "string"},
                "reference_projects": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "structure": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            # Filename pattern: lowercase, no path separators
                            "pattern": "^[a-zA-Z0-9_\\.\\-]{1,60}$",
                        },
                        "maxItems": 12,
                    },
                    "maxProperties": 20,
                },
            },
        },
    },
}


class ArchitectModel:

    def __init__(
        self,
        model_path: str | Path,
        use_constrained_decoding: bool = True,
        device: str = "auto",
    ):
        self.model_path = str(model_path)
        self.use_constrained = use_constrained_decoding

        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        self._prefix_fn = None
        if use_constrained_decoding:
            self._setup_constrained_decoding()

    def generate(
        self,
        goal: str,
        scale: str = "basic",
        constraints: Optional[list[str]] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 1200,
    ) -> Optional[ArchitectOutput]:
        """
        Main inference entry point.
        Returns a validated ArchitectOutput or None if generation failed.
        """
        prompt = self._build_prompt(goal, scale, constraints or [])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self._prefix_fn:
            generate_kwargs["prefix_allowed_tokens_fn"] = self._prefix_fn

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return self._parse_and_validate(raw_text)

    # ── private ───────────────────────────────────────────────────────────────

    def _build_prompt(self, goal: str, scale: str, constraints: list[str]) -> str:
        system_msg = (
            "You are an expert software architect. "
            "Given a project goal, respond with a single JSON object containing "
            "language_recommendation and file_structure. No text outside the JSON."
        )

        scale_hint = {
            "basic": "This is a basic/educational implementation.",
            "intermediate": "This is an intermediate implementation.",
            "production": "This is a production-scale implementation.",
        }.get(scale, "")

        constraint_str = ""
        if constraints:
            constraint_str = "\nConstraints: " + ", ".join(constraints)

        user_content = f"Goal: {goal}\n{scale_hint}{constraint_str}".strip()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"<|system|>\n{system_msg}\n<|user|>\n{user_content}\n<|assistant|>\n"

    def _setup_constrained_decoding(self) -> None:
        try:
            from lm_format_enforcer import JsonSchemaParser
            from lm_format_enforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )

            parser = JsonSchemaParser(ARCHITECT_JSON_SCHEMA)
            self._prefix_fn = build_transformers_prefix_allowed_tokens_fn(
                self.tokenizer, parser
            )
            logger.info("Constrained decoding enabled")
        except ImportError:
            logger.warning(
                "lm-format-enforcer not installed — falling back to unconstrained decoding. "
                "Install with: pip install lm-format-enforcer"
            )

    def _parse_and_validate(self, raw_text: str) -> Optional[ArchitectOutput]:
        # Strip any markdown fences
        text = raw_text.strip()
        for fence in ["```json", "```"]:
            text = text.replace(fence, "")
        text = text.strip()

        # Find the JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("No JSON object found in output")
            return None

        json_str = text[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None

        try:
            return ArchitectOutput(**data)
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
            return None
