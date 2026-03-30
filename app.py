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

    max_new_tokens = st.slider(
        "Max new tokens", min_value=256, max_value=2048, value=1200, step=64,
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

    meta_col, tree_col = st.columns([1, 2], gap="large")

    with meta_col:
        st.markdown(f"**Language:** `{struct.language}`")
        st.markdown(f"**Fingerprint:** `{output.fingerprint()}`")
        if struct.reference_projects:
            st.markdown("**Reference projects:**")
            for proj in struct.reference_projects:
                st.markdown(f"- `{proj}`")

    with tree_col:
        # Build a nested dict for display
        lines = []
        for dir_path in sorted(struct.structure.keys()):
            if dir_path == ".":
                label = "."
                indent = ""
            else:
                depth = dir_path.count("/")
                indent = "  " * depth
                label = dir_path.split("/")[-1] + "/"
            lines.append(f"{indent}📁 **{label}**")
            for fname in struct.structure[dir_path]:
                file_indent = indent + "  "
                lines.append(f"{file_indent}📄 `{fname}`")

        st.markdown("\n\n".join(lines))

    # ── Raw JSON ──────────────────────────────────────────────────────────────

    if show_raw:
        st.divider()
        st.subheader("Raw JSON")
        st.code(json.dumps(output.model_dump(), indent=2), language="json")
