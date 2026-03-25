"""
goal_augmentor.py
─────────────────
Rewrites each training example's goal N times using GPT-4o-mini.
This ensures the model handles "build an OS", "create an operating system",
"write a unix kernel from scratch" etc. as the same intent.

Also generates synthetic examples for (domain, language) cells
that had sparse GitHub coverage.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)


REWRITE_SYSTEM_PROMPT = """
You rewrite software project goals into different phrasings.
Keep the same intent and domain — only change the wording.
Output a JSON array of strings, each a different phrasing.
No explanation, no markdown, just the JSON array.
"""

REWRITE_USER_TEMPLATE = """
Original goal: "{goal}"
Domain: {domain}
Scale: {scale}

Generate {n} different phrasings of this goal.
Vary: formal/informal tone, verb choice (build/create/implement/write/develop),
level of detail (brief/descriptive), and framing (from scratch/basic/minimal).
Keep each phrasing under 20 words.
"""

SYNTHETIC_SYSTEM_PROMPT = """
You are a software architect generating training data.
Given a domain and language, generate a realistic project goal
and a canonical file structure a developer would create.

Output JSON matching this schema exactly:
{
  "goal": "a concise one-sentence project description",
  "structure": {
    "dir/path": ["filename.ext", ...]
  },
  "reference_projects": ["real-project-name"]
}

Rules:
- Use conventions from real, well-known projects in that language
- Maximum 8 files per directory
- Maximum 6 top-level directories
- Filenames must be lowercase_snake_case with correct extension
- Sort filenames alphabetically within each directory
"""


class GoalAugmentor:

    def __init__(self, openai_client=None, rewrites_per_example: int = 5):
        self.client = openai_client
        self.n_rewrites = rewrites_per_example

    def augment(self, pair_dict: dict) -> list[dict]:
        """
        Takes a training pair dict, returns the original + N augmented variants.
        If no OpenAI client, falls back to rule-based augmentation.
        """
        goal = pair_dict["input"]["goal"]
        domain = pair_dict["metadata"]["domain"]
        scale = pair_dict["input"]["scale"]

        if self.client:
            rewrites = self._openai_rewrite(goal, domain, scale)
        else:
            rewrites = self._rule_based_rewrite(goal)

        variants = [pair_dict]  # original always first

        for rewrite in rewrites[: self.n_rewrites]:
            variant = json.loads(json.dumps(pair_dict))  # deep copy
            variant["input"]["goal"] = rewrite
            variant["metadata"]["augmented"] = True
            variants.append(variant)

        return variants

    def generate_synthetic(self, domain: str, language: str, n: int = 10) -> list[dict]:
        """
        Generates n synthetic (goal, structure) pairs for a given cell.
        Used to fill sparse cells in the training matrix.
        """
        if not self.client:
            logger.warning("OpenAI client required for synthetic generation")
            return []

        results = []
        for _ in range(n):
            pair = self._generate_synthetic_pair(domain, language)
            if pair:
                results.append(pair)
            time.sleep(0.5)  # rate limit

        return results

    # ── private ───────────────────────────────────────────────────────────────

    def _openai_rewrite(self, goal: str, domain: str, scale: str) -> list[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": REWRITE_USER_TEMPLATE.format(
                            goal=goal,
                            domain=domain,
                            scale=scale,
                            n=self.n_rewrites,
                        ),
                    },
                ],
                temperature=0.8,
                max_tokens=400,
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            text = text.replace("```json", "").replace("```", "").strip()
            rewrites = json.loads(text)
            if isinstance(rewrites, list):
                return [r for r in rewrites if isinstance(r, str)]
        except Exception as e:
            logger.debug(f"OpenAI rewrite failed: {e}")

        return self._rule_based_rewrite(goal)

    def _rule_based_rewrite(self, goal: str) -> list[str]:
        """
        Deterministic rewriting rules as fallback.
        Covers the most common phrasing variations.
        """
        templates = [
            "build {core}",
            "create {core} from scratch",
            "implement {core}",
            "develop {core}",
            "write {core} in {lang_hint}",
            "design and build {core}",
            "build a basic {core}",
            "build a minimal {core}",
        ]

        # Strip common prefixes to extract the core noun phrase
        core = goal
        for prefix in [
            "build ",
            "create ",
            "implement ",
            "develop ",
            "write ",
            "make ",
        ]:
            if goal.lower().startswith(prefix):
                core = goal[len(prefix) :]
                break

        lang_hints = [
            "a systems language",
            "a compiled language",
            "the appropriate language",
        ]

        results = []
        for tmpl in templates[: self.n_rewrites]:
            rewrite = tmpl.format(
                core=core,
                lang_hint=random.choice(lang_hints),
            )
            if rewrite != goal:
                results.append(rewrite)

        return results

    def _generate_synthetic_pair(self, domain: str, language: str) -> Optional[dict]:
        prompt = (
            f"Domain: {domain}\n"
            f"Language: {language}\n\n"
            f"Generate a realistic project for this domain in this language. "
            f"Use conventions from well-known real projects."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYNTHETIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text)

            # Validate required fields
            if not all(k in data for k in ["goal", "structure"]):
                return None

            return {
                "input": {"goal": data["goal"], "scale": "basic", "constraints": []},
                "output": {
                    "language_recommendation": {
                        "primary": {
                            "language": language,
                            "reasons": [],
                            "tradeoffs": "",
                        },
                        "alternatives": [],
                    },
                    "file_structure": {
                        "language": language,
                        "reference_projects": data.get("reference_projects", []),
                        "structure": data["structure"],
                    },
                },
                "metadata": {
                    "domain": domain,
                    "source_repo": "synthetic",
                    "stars": 0,
                    "num_files": sum(len(v) for v in data["structure"].values()),
                    "augmented": True,
                },
            }
        except Exception as e:
            logger.debug(f"Synthetic generation failed for {domain}/{language}: {e}")
            return None
