"""
github_collector.py
───────────────────
Collects GitHub repositories for each (domain, language) cell.
Outputs one JSONL file per cell into the output directory.

Usage:
    python -m data.collectors.github_collector --output data/raw/
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from github import Github, GithubException, RateLimitExceededException
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Domain → GitHub topic mappings ───────────────────────────────────────────

DOMAIN_TOPICS: dict[str, list[str]] = {
    "operating_system": [
        "operating-system",
        "kernel",
        "bootloader",
        "os-kernel",
        "rtos",
    ],
    "compiler": [
        "compiler",
        "interpreter",
        "programming-language",
        "bytecode-compiler",
    ],
    "database": [
        "database",
        "storage-engine",
        "key-value-store",
        "sql-database",
        "nosql",
    ],
    "web_framework": [
        "web-framework",
        "http-server",
        "rest-api",
        "graphql-server",
        "microframework",
    ],
    "ml_framework": [
        "deep-learning",
        "neural-network",
        "machine-learning",
        "pytorch",
        "tensorflow",
    ],
    "game_engine": [
        "game-engine",
        "rendering-engine",
        "physics-engine",
        "2d-game-engine",
        "3d-engine",
    ],
    "mobile_app": [
        "android-app",
        "ios-app",
        "cross-platform",
        "flutter",
        "react-native",
    ],
    "cli_tool": [
        "cli",
        "command-line-tool",
        "terminal",
        "shell",
        "command-line-interface",
    ],
    "networking": [
        "networking",
        "protocol",
        "tcp-ip",
        "http-client",
        "network-library",
    ],
    "embedded": ["embedded", "firmware", "microcontroller", "arduino", "bare-metal"],
    "distributed_system": [
        "distributed-systems",
        "consensus",
        "raft",
        "message-queue",
        "distributed-computing",
    ],
    "container_runtime": ["container", "docker", "oci", "container-runtime", "sandbox"],
    "build_system": ["build-system", "build-tool", "make", "cmake", "package-manager"],
    "testing_framework": [
        "testing",
        "test-framework",
        "unit-testing",
        "mocking",
        "benchmarking",
    ],
    "editor": ["text-editor", "code-editor", "ide", "lsp", "language-server"],
    "crypto": ["cryptography", "blockchain", "zero-knowledge", "tls", "encryption"],
    "data_pipeline": [
        "data-pipeline",
        "etl",
        "stream-processing",
        "data-engineering",
        "workflow",
    ],
    "api_gateway": [
        "api-gateway",
        "reverse-proxy",
        "load-balancer",
        "proxy",
        "ingress",
    ],
    "search_engine": [
        "search-engine",
        "full-text-search",
        "information-retrieval",
        "inverted-index",
    ],
    "debugger": ["debugger", "profiler", "tracing", "observability", "apm"],
}

# Language per domain: (primary, *viable)
DOMAIN_LANGUAGE_MATRIX: dict[str, list[str]] = {
    "operating_system": ["C", "C++", "Rust"],
    "compiler": ["C", "C++", "Rust", "Python", "OCaml"],
    "database": ["C", "C++", "Rust", "Go", "Java"],
    "web_framework": ["Python", "Go", "TypeScript", "Rust", "Java", "Ruby"],
    "ml_framework": ["Python", "C++", "Rust"],
    "game_engine": ["C++", "C#", "Rust"],
    "mobile_app": ["Swift", "Kotlin", "Dart", "TypeScript"],
    "cli_tool": ["Go", "Rust", "Python", "TypeScript"],
    "networking": ["C", "C++", "Go", "Rust", "Python"],
    "embedded": ["C", "C++", "Rust"],
    "distributed_system": ["Go", "Java", "Rust", "C++", "Python"],
    "container_runtime": ["Go", "Rust", "C"],
    "build_system": ["Python", "Go", "Rust", "C++"],
    "testing_framework": ["Python", "Go", "TypeScript", "Java", "Rust"],
    "editor": ["C", "C++", "Rust", "TypeScript"],
    "crypto": ["C", "Rust", "Go", "Python"],
    "data_pipeline": ["Python", "Go", "Java", "Rust"],
    "api_gateway": ["Go", "Rust", "C++", "TypeScript"],
    "search_engine": ["Java", "C++", "Rust", "Go"],
    "debugger": ["C", "C++", "Rust", "Python"],
}

# GitHub language string normalisation
LANGUAGE_ALIASES: dict[str, str] = {
    "TypeScript": "TypeScript",
    "JavaScript": "TypeScript",  # treat JS repos as TS-viable
    "C#": "C#",
    "C++": "C++",
    "C": "C",
    "Rust": "Rust",
    "Go": "Go",
    "Python": "Python",
    "Java": "Java",
    "Kotlin": "Kotlin",
    "Swift": "Swift",
    "Dart": "Dart",
    "Ruby": "Ruby",
    "OCaml": "OCaml",
}


@dataclass
class RepoRecord:
    repo_full_name: str
    domain: str
    language: str
    stars: int
    description: str
    readme_excerpt: str  # first 500 chars of README
    topics: list[str]
    structure: dict[str, list[str]]  # dir → [filenames]
    raw_paths: list[str]
    matched_topic: str

    def to_dict(self) -> dict:
        return asdict(self)


class GitHubCollector:
    def __init__(self, token: str, config: dict):
        self.gh = Github(token, per_page=100)
        self.cfg = config
        self._request_count = 0

    # ── public ────────────────────────────────────────────────────────────────

    def collect_all(self, output_dir: Path, max_per_cell: int = 200) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        cells = [
            (domain, lang)
            for domain, langs in DOMAIN_LANGUAGE_MATRIX.items()
            for lang in langs
        ]

        logger.info(f"Collecting {len(cells)} (domain, language) cells")

        for domain, language in tqdm(cells, desc="cells"):
            out_file = (
                output_dir
                / f"{domain}__{language.lower().replace('+','p').replace('#','sharp')}.jsonl"
            )
            if out_file.exists():
                logger.info(f"  Skipping {domain}/{language} — already collected")
                continue

            records = self._collect_cell(domain, language, max_per_cell)
            self._write_jsonl(records, out_file)
            logger.info(
                f"  {domain}/{language}: {len(records)} repos → {out_file.name}"
            )

    # ── private ───────────────────────────────────────────────────────────────

    def _collect_cell(self, domain: str, language: str, max_n: int) -> list[RepoRecord]:
        records: list[RepoRecord] = []
        topics = DOMAIN_TOPICS.get(domain, [])

        for topic in topics:
            if len(records) >= max_n:
                break

            query = (
                f"topic:{topic} "
                f"language:{language} "
                f"stars:>{self.cfg.get('min_stars', 100)} "
                f"fork:false"
            )

            try:
                logger.info(f"  Searching: {query}")
                results = self.gh.search_repositories(query, sort="stars", order="desc")
                for repo in results:
                    if len(records) >= max_n:
                        break
                    logger.info(f"    Processing {repo.full_name} (★{repo.stargazers_count})")
                    record = self._process_repo(repo, domain, language, topic)
                    if record:
                        records.append(record)
                        logger.info(f"    ✓ Kept {repo.full_name} [{len(records)}/{max_n}]")
                    else:
                        logger.info(f"    ✗ Skipped {repo.full_name}")
                    self._throttle()

            except RateLimitExceededException:
                self._wait_for_rate_limit()
            except GithubException as e:
                logger.warning(f"GitHub error for {topic}/{language}: {e}")
                continue

        return records

    def _process_repo(
        self, repo, domain: str, language: str, matched_topic: str
    ) -> Optional[RepoRecord]:
        try:
            # Fetch file tree
            try:
                tree = repo.get_git_tree(repo.default_branch, recursive=True)
            except GithubException as e:
                logger.debug(f"      git tree failed for {repo.full_name}: {e}")
                return None

            raw_paths = [
                item.path
                for item in tree.tree
                if item.type == "blob"
                and len(item.path.split("/")) <= self.cfg.get("max_depth", 4) + 1
            ]

            if not raw_paths:
                logger.debug(f"      no paths after depth filter: {repo.full_name}")
                return None

            if len(raw_paths) > self.cfg.get("max_files_per_repo", 1500):
                logger.debug(f"      monorepo skip ({len(raw_paths)} files): {repo.full_name}")
                return None  # monorepo — skip

            structure = self._paths_to_structure(raw_paths)
            if not structure:
                return None

            readme = self._get_readme(repo)

            return RepoRecord(
                repo_full_name=repo.full_name,
                domain=domain,
                language=language,
                stars=repo.stargazers_count,
                description=repo.description or "",
                readme_excerpt=readme[:500] if readme else "",
                topics=repo.get_topics(),
                structure=structure,
                raw_paths=raw_paths[:200],  # cap storage
                matched_topic=matched_topic,
            )

        except Exception as e:
            logger.debug(f"Failed to process {repo.full_name}: {e}")
            return None

    def _paths_to_structure(self, paths: list[str]) -> dict[str, list[str]]:
        noise_dirs = set(self.cfg.get("noise_dirs", []))
        noise_exts = set(self.cfg.get("noise_extensions", []))
        max_depth = self.cfg.get("max_depth", 4)

        structure: dict[str, list[str]] = {}

        for path in paths:
            parts = path.split("/")

            # Skip noise directories anywhere in the path
            if any(p in noise_dirs for p in parts):
                continue

            # Skip noise extensions
            if any(path.endswith(ext) for ext in noise_exts):
                continue

            # Enforce depth
            if len(parts) > max_depth + 1:
                continue

            filename = parts[-1]
            dir_key = "/".join(parts[:-1]) if len(parts) > 1 else "."

            structure.setdefault(dir_key, []).append(filename)

        # Sort everything for determinism
        return {k: sorted(v) for k, v in sorted(structure.items())}

    def _get_readme(self, repo) -> str:
        try:
            readme = repo.get_readme()
            content = readme.decoded_content.decode("utf-8", errors="ignore")
            # Strip markdown noise, keep prose
            content = re.sub(r"```[\s\S]*?```", "", content)
            content = re.sub(r"#+\s", "", content)
            content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)
            content = re.sub(r"\s+", " ", content).strip()
            return content
        except Exception:
            return ""

    def _throttle(self) -> None:
        self._request_count += 1
        # GitHub allows ~5000 req/hr for authenticated — be conservative
        if self._request_count % 50 == 0:
            time.sleep(2)

    def _wait_for_rate_limit(self) -> None:
        import datetime
        rate_limit = self.gh.get_rate_limit()
        reset_time = rate_limit.search.reset  # datetime (UTC)
        wait = max((reset_time - datetime.datetime.utcnow()).total_seconds() + 5, 60)
        logger.warning(f"Rate limited — waiting {wait:.0f}s")
        time.sleep(wait)

    @staticmethod
    def _write_jsonl(records: list[RepoRecord], path: Path) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")
