"""
normalizer.py
─────────────
Cleans raw collected repo structures:
  - Removes noise files/dirs
  - Enforces depth limits
  - Canonicalises filenames (synonym map)
  - Extracts goal text from README + description
  - Labels project scale (basic / intermediate / production)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Canonical filename synonyms ───────────────────────────────────────────────
# Maps common variants → canonical name the model should learn

CANONICAL_FILENAMES: dict[str, str] = {
    # C/C++ entry points
    "init.c": "main.c",
    "start.c": "main.c",
    "entry.c": "main.c",
    "init.cpp": "main.cpp",
    "start.cpp": "main.cpp",
    # Schedulers
    "sched.c": "scheduler.c",
    "proc.c": "process.c",
    "task.c": "process.c",
    # Memory
    "mem.c": "memory.c",
    "mem.h": "memory.h",
    "mem.cpp": "memory.cpp",
    "alloc.c": "allocator.c",
    # Filesystem
    "fs.c": "filesystem.c",
    "vfs.c": "virtual_fs.c",
    # Networking
    "net.c": "network.c",
    "sock.c": "socket.c",
    # Rust
    "lib.rs": "lib.rs",  # keep as-is, canonical in Rust
    "mod.rs": "mod.rs",
    # Python
    "__init__.py": "__init__.py",
    "app.py": "app.py",
    "main.py": "main.py",
    "server.py": "server.py",
    # Go
    "main.go": "main.go",
    "server.go": "server.go",
}

# ── Scale heuristics ──────────────────────────────────────────────────────────

SCALE_THRESHOLDS = {
    "basic": (0, 30),  # total files
    "intermediate": (30, 200),
    "production": (200, 9999),
}

# Dirs that suggest production scale regardless of file count
PRODUCTION_SIGNALS = {
    "kubernetes",
    "k8s",
    "helm",
    "ci",
    ".github",
    "deploy",
    "terraform",
    "ansible",
    "docker",
}


@dataclass
class NormalizedStructure:
    domain: str
    language: str
    stars: int
    goal: str  # extracted natural language goal
    scale: str  # basic | intermediate | production
    structure: dict[str, list[str]]  # cleaned dir → [filenames]
    source_repo: str
    num_files: int


class StructureNormalizer:

    def normalize(self, raw: dict) -> Optional[NormalizedStructure]:
        """
        Takes a raw RepoRecord dict, returns a NormalizedStructure or None
        if the record doesn't pass quality gates.
        """
        structure = raw.get("structure", {})
        if not structure:
            return None

        cleaned = self._clean_structure(structure)
        if not cleaned:
            return None

        goal = self._extract_goal(raw)
        if not goal:
            return None

        num_files = sum(len(v) for v in cleaned.values())
        scale = self._classify_scale(cleaned, num_files)

        return NormalizedStructure(
            domain=raw["domain"],
            language=raw["language"],
            stars=raw.get("stars", 0),
            goal=goal,
            scale=scale,
            structure=cleaned,
            source_repo=raw.get("repo_full_name", ""),
            num_files=num_files,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _clean_structure(self, structure: dict[str, list[str]]) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}

        for dir_path, files in structure.items():
            # Skip dirs that are pure noise or generated output
            if self._is_noise_dir(dir_path):
                continue

            clean_files = []
            for f in files:
                if self._is_noise_file(f):
                    continue
                canonical = CANONICAL_FILENAMES.get(f, f)
                clean_files.append(canonical)

            if clean_files:
                cleaned[dir_path] = sorted(set(clean_files))

        return {k: v for k, v in sorted(cleaned.items())}

    def _is_noise_dir(self, dir_path: str) -> bool:
        noise = {
            "node_modules",
            "__pycache__",
            ".git",
            "dist",
            "build",
            ".next",
            "vendor",
            "third_party",
            "external",
            ".cache",
            "coverage",
            ".nyc_output",
            "target/debug",
            "target/release",
            ".pytest_cache",
            ".mypy_cache",
            "site-packages",
            "eggs",
            ".eggs",
            "htmlcov",
            "docs/_build",
        }
        parts = set(dir_path.split("/"))
        return bool(parts & noise)

    def _is_noise_file(self, filename: str) -> bool:
        noise_exts = {
            ".lock",
            ".min.js",
            ".min.css",
            ".map",
            ".pyc",
            ".o",
            ".a",
            ".so",
            ".dylib",
            ".dll",
            ".exe",
            ".class",
            ".jar",
            ".war",
            ".DS_Store",
        }
        noise_names = {
            "package-lock.json",
            "yarn.lock",
            "Cargo.lock",
            "poetry.lock",
            "Pipfile.lock",
            "go.sum",
            ".gitkeep",
            ".gitmodules",
        }
        if filename in noise_names:
            return True
        _, _, ext = filename.rpartition(".")
        return f".{ext}" in noise_exts if ext else False

    def _extract_goal(self, raw: dict) -> Optional[str]:
        """
        Try to extract a clean 1-sentence project goal.
        Priority: description > first sentence of README > repo name.
        """
        # 1. GitHub description (usually the cleanest)
        desc = (raw.get("description") or "").strip()
        if desc and len(desc) > 15 and len(desc) < 200:
            return self._clean_text(desc)

        # 2. First meaningful sentence from README
        readme = (raw.get("readme_excerpt") or "").strip()
        if readme:
            sentences = re.split(r"(?<=[.!?])\s+", readme)
            for s in sentences[:5]:
                s = self._clean_text(s)
                if 20 < len(s) < 150 and not self._is_badge_text(s):
                    return s

        # 3. Derive from repo name as last resort
        name = raw.get("repo_full_name", "").split("/")[-1]
        if name:
            readable = name.replace("-", " ").replace("_", " ").lower()
            return f"build a {readable}"

        return None

    def _clean_text(self, text: str) -> str:
        # Remove markdown, badges, URLs
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Lowercase first char for consistency
        if text:
            text = text[0].lower() + text[1:]
        return text

    def _is_badge_text(self, text: str) -> bool:
        badge_signals = [
            "build passing",
            "license",
            "npm version",
            "coverage",
            "downloads",
        ]
        return any(s in text.lower() for s in badge_signals)

    def _classify_scale(self, structure: dict[str, list[str]], num_files: int) -> str:
        # Check for production signals first
        all_dirs = set(structure.keys())
        for d in all_dirs:
            for part in d.split("/"):
                if part.lower() in PRODUCTION_SIGNALS:
                    return "production"

        for scale, (lo, hi) in SCALE_THRESHOLDS.items():
            if lo <= num_files < hi:
                return scale

        return "production"
