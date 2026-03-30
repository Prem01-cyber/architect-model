#!/usr/bin/env python3
"""
05_evaluate.py
──────────────
Evaluates the trained model and optionally runs an interactive demo.

Usage:
    python scripts/05_evaluate.py --model outputs/merged/
    python scripts/05_evaluate.py --model outputs/merged/ --interactive
    python scripts/05_evaluate.py --model outputs/merged/ --goal "build a search engine"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.model import ArchitectModel
from evaluation.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def print_output(output, goal: str) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree

    console = Console()

    if output is None:
        console.print(f"[red]Failed to generate output for: {goal}[/red]")
        return

    rec = output.language_recommendation
    struct = output.file_structure

    # Language recommendation panel
    primary = rec.primary
    lines = [
        f"[bold green]Primary:[/bold green] {primary.language}",
        f"[dim]Tradeoffs:[/dim] {primary.tradeoffs}",
        "",
    ]
    for i, reason in enumerate(primary.reasons, 1):
        lines.append(f"  {i}. {reason}")

    if rec.alternatives:
        lines.append("")
        lines.append("[bold]Alternatives:[/bold]")
        for alt in rec.alternatives:
            lines.append(f"  • [cyan]{alt.language}[/cyan] — {alt.tradeoffs}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Language recommendation for: {goal}",
            border_style="green",
        )
    )

    # File structure tree
    tree = Tree(f"[bold]{struct.language}[/bold] project structure")
    if struct.reference_projects:
        tree.add(f"[dim]Reference: {', '.join(struct.reference_projects)}[/dim]")

    dir_nodes = {}
    for dir_path in sorted(struct.structure.keys()):
        if dir_path == ".":
            node = tree
        else:
            parts = dir_path.split("/")
            parent_path = "/".join(parts[:-1])
            parent_node = dir_nodes.get(parent_path, tree)
            node = parent_node.add(f"[blue]{parts[-1]}/[/blue]")
        dir_nodes[dir_path] = node

        for filename in struct.structure[dir_path]:
            node.add(f"[white]{filename}[/white]")

    console.print(tree)
    console.print(f"\n[dim]Fingerprint: {output.fingerprint()}[/dim]\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="outputs/merged/", help="Path to trained model"
    )
    parser.add_argument(
        "--eval", action="store_true", default=True, help="Run evaluation suite"
    )
    parser.add_argument("--no-eval", dest="eval", action="store_false")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive REPL mode"
    )
    parser.add_argument("--goal", default=None, help="Single goal to run")
    parser.add_argument(
        "--scale", default="basic", choices=["basic", "intermediate", "production"]
    )
    parser.add_argument(
        "--quantize", default="4bit", choices=["4bit", "8bit", "none"],
        help="Quantization mode (default: 4bit)"
    )
    parser.add_argument("--save-eval", default="outputs/eval_results.json")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model}")
    model = ArchitectModel(
        model_path=args.model,
        quantize=None if args.quantize == "none" else args.quantize,
    )

    # ── Single goal mode ──────────────────────────────────────────────────────
    if args.goal:
        output = model.generate(args.goal, scale=args.scale)
        print_output(output, args.goal)
        return

    # ── Eval suite ────────────────────────────────────────────────────────────
    if args.eval:
        evaluator = ModelEvaluator(model, n_consistency_runs=3)
        summary = evaluator.run()
        evaluator.save(summary, Path(args.save_eval))

    # ── Interactive REPL ──────────────────────────────────────────────────────
    if args.interactive:
        from rich.console import Console

        console = Console()
        console.print("\n[bold]Architect Model — Interactive Mode[/bold]")
        console.print("Type a project goal and press Enter. Ctrl+C to quit.\n")

        while True:
            try:
                goal = input("Goal: ").strip()
                if not goal:
                    continue
                scale = (
                    input("Scale (basic/intermediate/production) [basic]: ").strip()
                    or "basic"
                )
                output = model.generate(goal, scale=scale)
                print_output(output, goal)
            except KeyboardInterrupt:
                console.print("\nBye!")
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()
