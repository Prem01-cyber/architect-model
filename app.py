"""
app.py
──────
Streamlit UI for the fine-tuned architect model.

Run:
    streamlit run app.py -- --model outputs/final/
    streamlit run app.py -- --model outputs/final/ --quantize none
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Parse CLI args before Streamlit takes over ───────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="outputs/final/")
    parser.add_argument("--quantize", default="4bit", choices=["4bit", "8bit", "none"])
    # Streamlit injects its own args after "--"; strip them
    try:
        idx = sys.argv.index("--")
        args, _ = parser.parse_known_args(sys.argv[idx + 1 :])
    except ValueError:
        args, _ = parser.parse_known_args()
    return args


# ── File tree renderer ────────────────────────────────────────────────────────

def _build_file_tree(structure: dict) -> str:
    """
    Convert flat {dir_path: [files]} into an ASCII tree string.

    Example output:
      .
      ├── README.md
      ├── go.mod
      └── src/
          ├── main.go
          └── handlers/
              └── user.go
    """
    # Build a nested dict: dict value = subdirectory, None value = file
    root: dict = {}
    for dir_path, files in structure.items():
        node = root
        if dir_path != ".":
            for part in dir_path.split("/"):
                if part not in node or node[part] is None:
                    node[part] = {}
                node = node[part]
        for f in sorted(files):
            node.setdefault(f, None)

    lines: list[str] = ["."]

    def _render(node: dict, prefix: str) -> None:
        dirs  = sorted((k, v) for k, v in node.items() if isinstance(v, dict))
        files = sorted((k, v) for k, v in node.items() if v is None)
        children = dirs + files
        for i, (name, child) in enumerate(children):
            is_last   = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            if isinstance(child, dict):
                lines.append(f"{prefix}{connector}{name}/")
                _render(child, prefix + extension)
            else:
                lines.append(f"{prefix}{connector}{name}")

    _render(root, "")
    return "\n".join(lines)


def _tree_stats(structure: dict) -> tuple[int, int, int]:
    """Return (total_files, total_dirs, max_depth)."""
    total_files = sum(len(v) for v in structure.values())
    total_dirs  = sum(1 for k in structure if k != ".")
    max_depth   = max((k.count("/") + 1 for k in structure if k != "."), default=0)
    return total_files, total_dirs, max_depth


cli = _parse_args()
MODEL_PATH = cli.model
QUANTIZE = None if cli.quantize == "none" else cli.quantize

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Architect Model",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ Architect Model")
st.caption(f"Model: `{MODEL_PATH}` · Quantization: `{QUANTIZE or 'none (bf16)'}`")

# ── Load model (cached so it only loads once) ─────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str, quantize):
    from inference.model import ArchitectModel
    return ArchitectModel(model_path=model_path, quantize=quantize)


model = load_model(MODEL_PATH, QUANTIZE)

# ── Sidebar — generation settings ─────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    scale = st.selectbox(
        "Scale",
        ["basic", "intermediate", "production"],
        help="Affects the complexity of the generated file structure.",
    )

    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Higher = more varied output.",
    )

    token_defaults = {"basic": 1200, "intermediate": 2000, "production": 3200}
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=256,
        max_value=4096,
        value=token_defaults[scale],
        step=64,
        help="Auto-set per scale; override if needed.",
    )

    constraints_raw = st.text_area(
        "Constraints (one per line)",
        placeholder="e.g.\nno external dependencies\nmust compile on Windows",
        height=100,
    )
    constraints = [c.strip() for c in constraints_raw.splitlines() if c.strip()]

    st.divider()
    show_raw = st.toggle("Show raw JSON output", value=False)

# ── Main input ────────────────────────────────────────────────────────────────

goal = st.text_input(
    "Project goal",
    placeholder="e.g. build a distributed key-value store",
    label_visibility="visible",
)

run = st.button("Generate", type="primary", disabled=not goal.strip())

# ── Inference & display ───────────────────────────────────────────────────────

if run and goal.strip():
    with st.spinner("Generating architecture…"):
        output = model.generate(
            goal=goal.strip(),
            scale=scale,
            constraints=constraints,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if output is None:
        st.error("Generation failed after all retries. Try a different goal or raise the temperature.")
        st.stop()

    rec = output.language_recommendation
    struct = output.file_structure

    # ── Language recommendation ───────────────────────────────────────────────

    st.subheader("Language Recommendation")

    primary_col, alt_col = st.columns([1, 1], gap="large")

    with primary_col:
        st.markdown(f"#### ✅ Primary: `{rec.primary.language}`")
        for reason in rec.primary.reasons:
            st.markdown(f"- {reason}")
        st.info(f"**Tradeoffs:** {rec.primary.tradeoffs}")

    with alt_col:
        if rec.alternatives:
            st.markdown("#### Alternatives")
            for alt in rec.alternatives:
                with st.expander(f"`{alt.language}`"):
                    for r in alt.reasons:
                        st.markdown(f"- {r}")
                    st.caption(f"Tradeoffs: {alt.tradeoffs}")
        else:
            st.markdown("#### Alternatives")
            st.caption("None suggested.")

    st.divider()

    # ── File structure ────────────────────────────────────────────────────────

    st.subheader("File Structure")

    total_files, total_dirs, max_depth = _tree_stats(struct.structure)

    stat_cols = st.columns(4)
    stat_cols[0].metric("Language", struct.language)
    stat_cols[1].metric("Files", total_files)
    stat_cols[2].metric("Directories", total_dirs)
    stat_cols[3].metric("Max depth", max_depth)

    tree_col, meta_col = st.columns([3, 1], gap="large")

    with tree_col:
        st.code(_build_file_tree(struct.structure), language="text")

    with meta_col:
        st.markdown("**Fingerprint**")
        st.code(output.fingerprint(), language="text")

        if struct.reference_projects:
            st.markdown("**Reference projects**")
            for proj in struct.reference_projects:
                st.markdown(f"- `{proj}`")

    # ── Raw JSON ──────────────────────────────────────────────────────────────

    if show_raw:
        st.divider()
        st.subheader("Raw JSON")
        st.code(json.dumps(output.model_dump(), indent=2), language="json")
