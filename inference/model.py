"""
model.py
────────
Inference wrapper for the fine-tuned architect model.

Key features:
  - temperature=0.1 for near-deterministic output
  - Pydantic schema validation on the output
  - Deterministic post-processing (sort, normalise paths)
  - Automatic retry (up to 3 attempts) on malformed output
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


class ArchitectModel:

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        quantize: Optional[str] = "4bit",  # "4bit", "8bit", or None
    ):
        self.model_path = str(model_path)

        bnb_config = None
        if quantize == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit NF4 quantization")
        elif quantize == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")

        # Detect whether model_path is a LoRA adapter or a full/merged model
        adapter_cfg_path = Path(self.model_path) / "adapter_config.json"
        is_adapter = adapter_cfg_path.exists()

        if is_adapter:
            with open(adapter_cfg_path) as f:
                base_model_id = json.load(f)["base_model_name_or_path"]
            logger.info(f"Detected LoRA adapter — base model: {base_model_id}")
        else:
            base_model_id = self.model_path

        logger.info(f"Loading {'base ' if is_adapter else ''}model from {base_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map=device,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

        if is_adapter:
            logger.info(f"Loading LoRA adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(base, self.model_path)
        else:
            self.model = base

        self.model.eval()

    def generate(
        self,
        goal: str,
        scale: str = "basic",
        constraints: Optional[list[str]] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 1200,
        max_retries: int = 3,
    ) -> Optional[ArchitectOutput]:
        """
        Main inference entry point.
        Returns a validated ArchitectOutput or None if all attempts failed.
        Retries up to max_retries times with slightly raised temperature on failure.
        """
        prompt = self._build_prompt(goal, scale, constraints or [])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        for attempt in range(max_retries):
            # Nudge temperature up on retries so we don't get the same bad output
            effective_temp = min(temperature + attempt * 0.1, 1.0)

            generate_kwargs: dict = {
                "max_new_tokens": max_new_tokens,
                "temperature": effective_temp,
                "do_sample": effective_temp > 0,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **generate_kwargs)

            new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            result = self._parse_and_validate(raw_text)
            if result is not None:
                if attempt > 0:
                    logger.info(f"Succeeded on attempt {attempt + 1}")
                return result

            logger.warning(f"Attempt {attempt + 1}/{max_retries} produced invalid output — retrying")

        logger.error("All generation attempts failed")
        return None

    # ── private ───────────────────────────────────────────────────────────────

    def _build_prompt(self, goal: str, scale: str, constraints: list[str]) -> str:
        system_msg = (
            "You are an expert software architect. "
            "Given a project goal, you:\n"
            "1. Recommend the best programming language with clear technical reasoning, "
            "listing a primary choice and up to two alternatives.\n"
            "2. Generate a canonical file structure for a basic implementation "
            "following real-world conventions.\n\n"
            "Always respond with a single JSON object matching this schema exactly. "
            "No explanation outside the JSON.\n\n"
            "Schema:\n"
            "{\n"
            '  "language_recommendation": {\n'
            '    "primary": { "language": str, "reasons": [str], "tradeoffs": str },\n'
            '    "alternatives": [{ "language": str, "reasons": [str], "tradeoffs": str }]\n'
            "  },\n"
            '  "file_structure": {\n'
            '    "language": str,\n'
            '    "reference_projects": [str],\n'
            '    "structure": { "dir/path": ["filename.ext"] }\n'
            "  }\n"
            "}"
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
