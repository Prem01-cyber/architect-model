"""
pair_builder.py
───────────────
Converts NormalizedStructure records into training pairs.

Each pair has:
  input:  { goal, scale, constraints }
  output: { language_recommendation, file_structure }

Language reasoning is generated either from a curated knowledge base
or via GPT-4o-mini (when use_openai=True).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Language reasoning knowledge base ─────────────────────────────────────────
# Curated reasons for each (domain, language) pair.
# These become the ground-truth reasoning in training examples.
# The model learns to reproduce reasoning like this for unseen combinations.

LANGUAGE_KNOWLEDGE: dict[str, dict[str, dict]] = {
    "operating_system": {
        "C": {
            "reasons": [
                "dominant language for OS kernels — Linux, XNU, FreeBSD all written in C",
                "direct hardware access with no runtime overhead",
                "full control over memory layout, alignment, and ABI",
            ],
            "tradeoffs": "manual memory management, undefined behaviour risks",
            "reference_projects": ["linux", "xv6", "minix"],
        },
        "C++": {
            "reasons": [
                "used in Windows NT and parts of macOS kernel",
                "zero-cost abstractions useful for driver layer",
                "RAII helps manage hardware resources safely",
            ],
            "tradeoffs": "larger binary, exceptions usually disabled in kernel context",
            "reference_projects": ["serenityos", "haiku"],
        },
        "Rust": {
            "reasons": [
                "memory safety without GC — critical for kernel reliability",
                "Linux 6.1+ accepts Rust drivers, proving viability",
                "ownership model prevents use-after-free bugs common in C kernels",
            ],
            "tradeoffs": "smaller ecosystem, unsafe blocks still needed for hardware access",
            "reference_projects": ["redox-os", "asterinas"],
        },
    },
    "compiler": {
        "C": {
            "reasons": [
                "GCC and early LLVM frontends written in C",
                "bootstrapping is simpler — C compilers exist everywhere",
                "close-to-metal control over code generation passes",
            ],
            "tradeoffs": "no type safety on AST node manipulation",
            "reference_projects": ["tcc", "chibicc"],
        },
        "C++": {
            "reasons": [
                "LLVM/Clang entirely in C++ — de facto standard for production compilers",
                "templates enable type-safe IR visitor patterns",
                "strong ecosystem of compiler infrastructure libraries",
            ],
            "tradeoffs": "complex build system, steep learning curve",
            "reference_projects": ["llvm", "clang", "graal"],
        },
        "Rust": {
            "reasons": [
                "pattern matching makes AST traversal natural and safe",
                "Cranelift JIT backend written in Rust",
                "memory safety prevents common compiler bugs like dangling AST pointers",
            ],
            "tradeoffs": "longer compile times, fewer compiler-specific libraries than C++",
            "reference_projects": ["rustc", "cranelift", "inkwell"],
        },
        "Python": {
            "reasons": [
                "ideal for educational compilers and prototyping",
                "PLY/ANTLR4 Python bindings make parser generation fast",
                "readable AST manipulation without boilerplate",
            ],
            "tradeoffs": "too slow for production use, not self-hostable easily",
            "reference_projects": ["mypy", "pypy-rpython"],
        },
    },
    "database": {
        "C": {
            "reasons": [
                "SQLite, the most deployed database in history, is pure C",
                "maximum portability across embedded and server targets",
                "precise control over page cache and I/O alignment",
            ],
            "tradeoffs": "no RAII for resource management, error handling verbose",
            "reference_projects": ["sqlite", "postgresql-core"],
        },
        "C++": {
            "reasons": [
                "MySQL, RocksDB, DuckDB all written in C++",
                "STL containers map naturally to database internals (B-trees, hash maps)",
                "zero-cost abstractions for hot paths like query execution",
            ],
            "tradeoffs": "memory safety issues common in complex storage engines",
            "reference_projects": ["mysql", "rocksdb", "duckdb"],
        },
        "Rust": {
            "reasons": [
                "TiKV (TiDB storage layer) proves Rust viability at scale",
                "async I/O with Tokio ideal for high-concurrency databases",
                "ownership prevents data races in concurrent transaction managers",
            ],
            "tradeoffs": "ecosystem younger than C++, fewer battle-tested storage libraries",
            "reference_projects": ["tikv", "neon", "risingwave"],
        },
        "Go": {
            "reasons": [
                "CockroachDB and InfluxDB written in Go",
                "goroutines simplify concurrent connection handling",
                "fast compile times accelerate database development iteration",
            ],
            "tradeoffs": "GC pauses problematic for latency-critical paths like WAL writes",
            "reference_projects": ["cockroachdb", "influxdb"],
        },
    },
    "web_framework": {
        "Python": {
            "reasons": [
                "Django and Flask power most Python web services",
                "largest ecosystem of web libraries: ORM, auth, serialisation",
                "async support via FastAPI/Starlette for high-throughput APIs",
            ],
            "tradeoffs": "slower than compiled languages, GIL limits CPU-bound concurrency",
            "reference_projects": ["django", "flask", "fastapi"],
        },
        "Go": {
            "reasons": [
                "net/http in stdlib is production-ready without frameworks",
                "goroutines handle C10K concurrency with minimal memory",
                "single binary deployment, fast startup — ideal for microservices",
            ],
            "tradeoffs": "less expressive than Python, verbose error handling",
            "reference_projects": ["gin", "echo", "fiber"],
        },
        "TypeScript": {
            "reasons": [
                "Next.js and NestJS dominate full-stack and API development",
                "shared types between frontend and backend reduces integration bugs",
                "npm ecosystem has frameworks for every use case",
            ],
            "tradeoffs": "Node.js event loop not suitable for CPU-intensive work",
            "reference_projects": ["nextjs", "nestjs", "fastify"],
        },
        "Rust": {
            "reasons": [
                "Axum and Actix-web among fastest web frameworks in TechEmpower benchmarks",
                "memory safety eliminates common web vulnerabilities",
                "async/await with Tokio provides excellent I/O throughput",
            ],
            "tradeoffs": "longer development time, smaller ecosystem than Python/Node",
            "reference_projects": ["axum", "actix-web"],
        },
    },
    "ml_framework": {
        "Python": {
            "reasons": [
                "PyTorch and TensorFlow — both Python-first APIs",
                "NumPy/SciPy ecosystem provides entire scientific stack",
                "Jupyter notebooks essential for ML experimentation workflow",
            ],
            "tradeoffs": "performance-critical ops must be offloaded to C++/CUDA extensions",
            "reference_projects": ["pytorch", "tensorflow", "jax"],
        },
        "C++": {
            "reasons": [
                "PyTorch's ATen tensor library and CUDA kernels written in C++",
                "libtorch enables production deployment without Python",
                "ONNX Runtime inference engine is C++",
            ],
            "tradeoffs": "slow iteration cycle, not suitable for research experimentation",
            "reference_projects": ["pytorch-core", "onnxruntime"],
        },
    },
    "game_engine": {
        "C++": {
            "reasons": [
                "Unreal Engine, id Tech, Source — all C++",
                "deterministic performance with no GC pauses critical for 60fps",
                "direct access to Vulkan/DirectX/Metal APIs",
            ],
            "tradeoffs": "complex codebase, long compile times",
            "reference_projects": ["godot-cpp", "bgfx", "raylib"],
        },
        "C#": {
            "reasons": [
                "Unity built on C# scripting layer",
                "managed memory simplifies game logic without sacrificing performance",
                "IL2CPP AOT compilation bridges managed and native code",
            ],
            "tradeoffs": "GC spikes can cause frame hitches if not carefully managed",
            "reference_projects": ["unity-runtime", "monogame"],
        },
        "Rust": {
            "reasons": [
                "Bevy ECS is fastest-growing Rust game engine",
                "data-oriented ECS architecture maps well to Rust ownership model",
                "no GC pauses, safe concurrency for job systems",
            ],
            "tradeoffs": "ecosystem much smaller than C++, fewer asset pipelines",
            "reference_projects": ["bevy", "macroquad"],
        },
    },
    "cli_tool": {
        "Go": {
            "reasons": [
                "Docker, kubectl, GitHub CLI all written in Go",
                "single static binary deployment — no runtime dependency",
                "cobra/viper provide battle-tested CLI framework",
            ],
            "tradeoffs": "larger binary size than Rust, GC adds ~few ms startup overhead",
            "reference_projects": ["cobra", "fzf", "gh-cli"],
        },
        "Rust": {
            "reasons": [
                "ripgrep, bat, fd — fastest CLI tools are Rust",
                "clap provides best-in-class argument parsing with derive macros",
                "near-zero startup time, small static binaries",
            ],
            "tradeoffs": "longer development time than Go",
            "reference_projects": ["ripgrep", "bat", "starship"],
        },
        "Python": {
            "reasons": [
                "fastest to prototype — rich, click, typer all excellent",
                "huge library ecosystem for any domain-specific tool",
                "pyinstaller/PyOxidizer can produce single executables",
            ],
            "tradeoffs": "requires Python runtime, slower startup than compiled languages",
            "reference_projects": ["httpie", "black", "poetry"],
        },
    },
    "embedded": {
        "C": {
            "reasons": [
                "virtually every embedded SDK and RTOS is C — FreeRTOS, Zephyr, Arduino",
                "compiler toolchains (arm-none-eabi-gcc) mature and well-tested",
                "predictable code size and timing, essential for real-time systems",
            ],
            "tradeoffs": "no memory safety guarantees, buffer overflows are common bugs",
            "reference_projects": ["freertos", "zephyr", "arduino-core"],
        },
        "C++": {
            "reasons": [
                "mbed OS and ESP-IDF support C++ for higher-level abstractions",
                "RAII useful for peripheral resource management",
                "templates enable zero-overhead abstractions for drivers",
            ],
            "tradeoffs": "dynamic allocation and exceptions usually disabled, limiting C++ utility",
            "reference_projects": ["mbed-os", "esp-idf"],
        },
        "Rust": {
            "reasons": [
                "Embassy async framework for embedded is production-ready",
                "memory safety critical for safety-critical embedded systems",
                "cortex-m HAL ecosystem growing rapidly",
            ],
            "tradeoffs": "toolchain setup more complex, fewer vendor-supplied HALs than C",
            "reference_projects": ["embassy", "rtic"],
        },
    },
}

# Fallback reasons when domain not in knowledge base
DEFAULT_REASONS: dict[str, dict] = {
    "Python": {
        "reasons": [
            "rapid prototyping",
            "extensive library ecosystem",
            "readable syntax",
        ],
        "tradeoffs": "slower than compiled languages",
        "reference_projects": [],
    },
    "Go": {
        "reasons": ["fast compilation", "built-in concurrency", "simple deployment"],
        "tradeoffs": "GC pauses for latency-sensitive code",
        "reference_projects": [],
    },
    "Rust": {
        "reasons": ["memory safety", "zero-cost abstractions", "excellent performance"],
        "tradeoffs": "steep learning curve",
        "reference_projects": [],
    },
    "TypeScript": {
        "reasons": ["type safety", "huge npm ecosystem", "full-stack code sharing"],
        "tradeoffs": "Node.js not ideal for CPU-intensive work",
        "reference_projects": [],
    },
    "Java": {
        "reasons": ["mature ecosystem", "strong typing", "excellent tooling"],
        "tradeoffs": "higher memory usage, JVM startup overhead",
        "reference_projects": [],
    },
    "C++": {
        "reasons": ["maximum performance", "low-level control", "extensive libraries"],
        "tradeoffs": "complex language, manual memory management",
        "reference_projects": [],
    },
    "C": {
        "reasons": [
            "maximum portability",
            "zero overhead",
            "universal toolchain support",
        ],
        "tradeoffs": "no type safety, manual memory management",
        "reference_projects": [],
    },
}


@dataclass
class TrainingPair:
    input: dict
    output: dict
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


class PairBuilder:
    """
    Builds training pairs from NormalizedStructure objects.
    The pair format matches the fine-tuning chat template exactly.
    """

    def __init__(self, domain_language_matrix: dict[str, list[str]]):
        self.matrix = domain_language_matrix

    def build(self, record) -> Optional[TrainingPair]:
        """
        record: NormalizedStructure instance
        """
        language_rec = self._build_language_recommendation(
            record.domain, record.language, record.stars
        )
        if not language_rec:
            return None

        structure_out = self._build_structure_output(record)

        return TrainingPair(
            input={
                "goal": record.goal,
                "scale": record.scale,
                "constraints": [],
            },
            output={
                "language_recommendation": language_rec,
                "file_structure": structure_out,
            },
            metadata={
                "domain": record.domain,
                "source_repo": record.source_repo,
                "stars": record.stars,
                "num_files": record.num_files,
            },
        )

    def build_chat_messages(self, pair: TrainingPair) -> list[dict]:
        """
        Converts a TrainingPair into the chat message format
        used for fine-tuning (system + user + assistant turns).
        """
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
            "intermediate": "This is a functional intermediate-scale implementation.",
            "production": "This is a production-scale implementation.",
        }.get(pair.input["scale"], "")

        user_msg = f"Goal: {pair.input['goal']}\n{scale_hint}".strip()
        assistant_msg = json.dumps(pair.output, indent=2)

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]

    # ── private ───────────────────────────────────────────────────────────────

    def _build_language_recommendation(
        self, domain: str, primary_lang: str, stars: int
    ) -> Optional[dict]:
        domain_kb = LANGUAGE_KNOWLEDGE.get(domain, {})
        primary_info = domain_kb.get(primary_lang) or DEFAULT_REASONS.get(primary_lang)

        if not primary_info:
            return None

        # Build alternatives from same domain, excluding primary
        viable_langs = [l for l in self.matrix.get(domain, []) if l != primary_lang][
            :2
        ]  # max 2 alternatives

        alternatives = []
        for lang in viable_langs:
            info = domain_kb.get(lang) or DEFAULT_REASONS.get(lang)
            if info:
                alternatives.append(
                    {
                        "language": lang,
                        "reasons": info["reasons"][:2],
                        "tradeoffs": info["tradeoffs"],
                    }
                )

        return {
            "primary": {
                "language": primary_lang,
                "reasons": primary_info["reasons"],
                "tradeoffs": primary_info["tradeoffs"],
            },
            "alternatives": alternatives,
        }

    def _build_structure_output(self, record) -> dict:
        domain_kb = LANGUAGE_KNOWLEDGE.get(record.domain, {})
        lang_info = domain_kb.get(record.language, {})
        ref_projects = lang_info.get("reference_projects", [])

        return {
            "language": record.language,
            "reference_projects": ref_projects,
            "structure": record.structure,
        }
