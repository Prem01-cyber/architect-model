"""
evaluator.py
────────────
Evaluates the fine-tuned model on:

1. Validity rate    — % of outputs that parse + pass schema validation
2. Consistency      — same goal produces same fingerprint across N runs
3. Language accuracy — primary language matches expected for known (domain, lang) pairs
4. Structure quality — directory depth, file count distribution, naming conventions
5. Coverage         — across all (domain, language) cells in the matrix
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class EvalResult:
    goal: str
    domain: str
    expected_language: str
    output_valid: bool
    predicted_language: Optional[str]
    language_correct: bool
    structure_fingerprint: Optional[str]
    num_dirs: int
    num_files: int
    error: Optional[str] = None


@dataclass
class EvalSummary:
    total: int
    validity_rate: float
    language_accuracy: float
    consistency_rate: float  # % of repeated goals with same fingerprint
    avg_dirs: float
    avg_files: float
    per_domain: dict[str, dict]  # domain → {validity, lang_acc}
    per_language: dict[str, dict]


# ── Eval probes — (goal, domain, expected_primary_language) ──────────────────

EVAL_PROBES = [
    # Systems
    ("build an operating system kernel", "operating_system", "C"),
    ("create a unix-like os from scratch", "operating_system", "C"),
    ("implement a microkernel in rust", "operating_system", "Rust"),
    ("build a compiler for a c-like language", "compiler", "C++"),
    ("write a bytecode interpreter", "compiler", "C"),
    ("implement a relational database engine", "database", "C++"),
    ("build a key-value storage engine", "database", "C++"),
    ("create an lsm-tree based storage engine", "database", "Rust"),
    # Web
    ("build a web framework with routing and middleware", "web_framework", "Python"),
    ("create a rest api framework in go", "web_framework", "Go"),
    ("build a typescript web server", "web_framework", "TypeScript"),
    # ML
    ("implement a neural network framework from scratch", "ml_framework", "Python"),
    ("build a tensor computation library", "ml_framework", "C++"),
    # Systems tools
    ("build a container runtime", "container_runtime", "Go"),
    ("create a distributed key-value store", "distributed_system", "Go"),
    ("build a build system", "build_system", "Python"),
    # Embedded
    ("write firmware for a microcontroller", "embedded", "C"),
    ("build an rtos scheduler", "embedded", "C"),
    # Games
    ("build a 2d game engine", "game_engine", "C++"),
    ("create a rendering engine with vulkan", "game_engine", "C++"),
    # CLI
    ("build a fast command-line search tool", "cli_tool", "Rust"),
    ("create a cli tool for managing projects", "cli_tool", "Go"),
]

# Goals repeated for consistency testing
CONSISTENCY_GOALS = [
    ("build an operating system", "operating_system", "C"),
    ("build a web framework", "web_framework", "Python"),
    ("build a database", "database", "C++"),
    ("build a compiler", "compiler", "C++"),
    ("build a game engine", "game_engine", "C++"),
]


class ModelEvaluator:

    def __init__(self, model, n_consistency_runs: int = 3):
        """
        model: ArchitectModel instance
        n_consistency_runs: how many times to run each consistency goal
        """
        self.model = model
        self.n_consistency = n_consistency_runs

    def run(self) -> EvalSummary:
        console.print("\n[bold]Running evaluation...[/bold]\n")

        # ── 1. Main eval pass ─────────────────────────────────────────────────
        results: list[EvalResult] = []
        for goal, domain, expected_lang in EVAL_PROBES:
            result = self._eval_one(goal, domain, expected_lang)
            results.append(result)
            status = "[green]OK[/green]" if result.output_valid else "[red]FAIL[/red]"
            console.print(f"  {status} {goal[:60]}")

        # ── 2. Consistency pass ───────────────────────────────────────────────
        consistency_rate = self._eval_consistency()

        # ── 3. Aggregate ──────────────────────────────────────────────────────
        summary = self._aggregate(results, consistency_rate)
        self._print_summary(summary)
        return summary

    # ── private ───────────────────────────────────────────────────────────────

    def _eval_one(self, goal: str, domain: str, expected_lang: str) -> EvalResult:
        try:
            output = self.model.generate(goal, scale="basic", temperature=0.1)
        except Exception as e:
            return EvalResult(
                goal=goal,
                domain=domain,
                expected_language=expected_lang,
                output_valid=False,
                predicted_language=None,
                language_correct=False,
                structure_fingerprint=None,
                num_dirs=0,
                num_files=0,
                error=str(e),
            )

        if output is None:
            return EvalResult(
                goal=goal,
                domain=domain,
                expected_language=expected_lang,
                output_valid=False,
                predicted_language=None,
                language_correct=False,
                structure_fingerprint=None,
                num_dirs=0,
                num_files=0,
                error="None output",
            )

        predicted_lang = output.language_recommendation.primary.language
        structure = output.file_structure.structure
        num_dirs = len(structure)
        num_files = sum(len(v) for v in structure.values())

        lang_correct = predicted_lang.lower() == expected_lang.lower()

        return EvalResult(
            goal=goal,
            domain=domain,
            expected_language=expected_lang,
            output_valid=True,
            predicted_language=predicted_lang,
            language_correct=lang_correct,
            structure_fingerprint=output.fingerprint(),
            num_dirs=num_dirs,
            num_files=num_files,
        )

    def _eval_consistency(self) -> float:
        """
        Runs each consistency goal N times and checks if fingerprints match.
        """
        console.print("\n[bold]Consistency check...[/bold]")
        match_count = 0
        total = 0

        for goal, domain, _ in CONSISTENCY_GOALS:
            fingerprints = []
            for _ in range(self.n_consistency):
                try:
                    out = self.model.generate(goal, scale="basic", temperature=0.1)
                    if out:
                        fingerprints.append(out.fingerprint())
                except Exception:
                    continue

            if len(fingerprints) >= 2:
                all_same = len(set(fingerprints)) == 1
                match_count += int(all_same)
                total += 1
                status = (
                    "[green]consistent[/green]"
                    if all_same
                    else "[yellow]drifted[/yellow]"
                )
                console.print(f"  {status} '{goal}' — fingerprints: {fingerprints}")

        return match_count / total if total > 0 else 0.0

    def _aggregate(
        self, results: list[EvalResult], consistency_rate: float
    ) -> EvalSummary:
        valid = [r for r in results if r.output_valid]
        validity_rate = len(valid) / len(results) if results else 0.0
        lang_acc = (
            sum(1 for r in valid if r.language_correct) / len(valid) if valid else 0.0
        )
        avg_dirs = sum(r.num_dirs for r in valid) / len(valid) if valid else 0.0
        avg_files = sum(r.num_files for r in valid) / len(valid) if valid else 0.0

        per_domain: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "valid": 0, "lang_correct": 0}
        )
        per_language: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

        for r in results:
            per_domain[r.domain]["total"] += 1
            if r.output_valid:
                per_domain[r.domain]["valid"] += 1
            if r.language_correct:
                per_domain[r.domain]["lang_correct"] += 1

            per_language[r.expected_language]["total"] += 1
            if r.language_correct:
                per_language[r.expected_language]["correct"] += 1

        return EvalSummary(
            total=len(results),
            validity_rate=validity_rate,
            language_accuracy=lang_acc,
            consistency_rate=consistency_rate,
            avg_dirs=avg_dirs,
            avg_files=avg_files,
            per_domain=dict(per_domain),
            per_language=dict(per_language),
        )

    def _print_summary(self, s: EvalSummary) -> None:
        console.print("\n")

        # Top-level metrics
        t = Table(title="Overall metrics", show_header=True)
        t.add_column("Metric")
        t.add_column("Value")
        t.add_row("Total probes", str(s.total))
        t.add_row("Validity rate", f"{s.validity_rate:.1%}")
        t.add_row("Language accuracy", f"{s.language_accuracy:.1%}")
        t.add_row("Consistency rate", f"{s.consistency_rate:.1%}")
        t.add_row("Avg dirs", f"{s.avg_dirs:.1f}")
        t.add_row("Avg files", f"{s.avg_files:.1f}")
        console.print(t)

        # Per-domain breakdown
        t2 = Table(title="Per-domain breakdown", show_header=True)
        t2.add_column("Domain")
        t2.add_column("Validity")
        t2.add_column("Lang acc")
        for domain, stats in sorted(s.per_domain.items()):
            total = stats["total"]
            validity = stats["valid"] / total if total else 0
            lang_acc = stats["lang_correct"] / total if total else 0
            t2.add_row(domain, f"{validity:.0%}", f"{lang_acc:.0%}")
        console.print(t2)

    def save(self, summary: EvalSummary, path: Path) -> None:
        data = {
            "total": summary.total,
            "validity_rate": summary.validity_rate,
            "language_accuracy": summary.language_accuracy,
            "consistency_rate": summary.consistency_rate,
            "avg_dirs": summary.avg_dirs,
            "avg_files": summary.avg_files,
            "per_domain": summary.per_domain,
            "per_language": summary.per_language,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"\nEval results saved to {path}")
