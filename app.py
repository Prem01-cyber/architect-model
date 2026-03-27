"""
app.py
──────
Streamlit UI for testing the fine-tuned Architect model.

Run with:
    streamlit run app.py
"""

import json
import logging
import sys
from pathlib import Path

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Architect Model",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ Architect Model — Inference UI")
st.caption("Fine-tuned Qwen2.5-Coder-7B-Instruct · Structured project architecture generation")

# ── Model loading (cached) ────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "outputs" / "merged"

sys.path.insert(0, str(Path(__file__).parent))


@st.cache_resource(show_spinner="Loading model — this takes a moment...")
def load_model():
    from inference.model import ArchitectModel

    logging.basicConfig(level=logging.INFO)
    return ArchitectModel(model_path=MODEL_PATH, use_constrained_decoding=True)


# ── Sidebar — generation settings ────────────────────────────────────────────

with st.sidebar:
    st.header("Generation settings")

    scale = st.selectbox(
        "Project scale",
        options=["basic", "intermediate", "production"],
        index=1,
        help="Controls the complexity hint passed to the model.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower = more deterministic output.",
    )

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=256,
        max_value=2048,
        value=1200,
        step=64,
    )

    constrained = st.toggle(
        "Constrained decoding",
        value=True,
        help="Uses lm-format-enforcer to guarantee valid JSON output.",
    )

    st.divider()
    st.markdown("**Model path**")
    st.code(str(MODEL_PATH), language=None)

# ── Main input area ───────────────────────────────────────────────────────────

goal = st.text_area(
    "Project goal",
    placeholder="e.g. Build a real-time collaborative document editor with offline support",
    height=100,
)

constraints_raw = st.text_input(
    "Constraints (comma-separated, optional)",
    placeholder="e.g. must use PostgreSQL, no TypeScript, monorepo",
)

constraints = [c.strip() for c in constraints_raw.split(",") if c.strip()] if constraints_raw else []

run = st.button("Generate architecture", type="primary", disabled=not goal.strip())

# ── Inference & output ────────────────────────────────────────────────────────

if run and goal.strip():
    model = load_model()

    # Reload model with current constrained setting if it differs
    if model.use_constrained != constrained:
        st.cache_resource.clear()
        model = load_model()

    with st.spinner("Generating..."):
        result = model.generate(
            goal=goal.strip(),
            scale=scale,
            constraints=constraints,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if result is None:
        st.error("Model returned an invalid or empty response. Try again or lower the temperature.")
    else:
        st.success(f"Done · fingerprint `{result.fingerprint()}`")

        # ── Language recommendation ───────────────────────────────────────────
        st.subheader("Language recommendation")

        primary = result.language_recommendation.primary
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Primary language", primary.language)
        with col2:
            st.markdown(f"**Tradeoffs:** {primary.tradeoffs}")
            st.markdown("**Reasons:**")
            for r in primary.reasons:
                st.markdown(f"- {r}")

        alts = result.language_recommendation.alternatives
        if alts:
            with st.expander(f"Alternatives ({len(alts)})"):
                for alt in alts:
                    st.markdown(f"### {alt.language}")
                    st.markdown(f"**Tradeoffs:** {alt.tradeoffs}")
                    for r in alt.reasons:
                        st.markdown(f"- {r}")

        st.divider()

        # ── File structure ────────────────────────────────────────────────────
        st.subheader("File structure")

        fs = result.file_structure
        st.markdown(f"**Language:** `{fs.language}`")

        if fs.reference_projects:
            st.markdown("**Reference projects:** " + ", ".join(f"`{p}`" for p in fs.reference_projects))

        # Render as a tree-like view
        tree_lines = []
        dirs = sorted(fs.structure.keys())
        for i, directory in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1
            dir_prefix = "└── " if is_last_dir else "├── "
            tree_lines.append(f"{dir_prefix}{directory}/")
            files = fs.structure[directory]
            for j, fname in enumerate(files):
                is_last_file = j == len(files) - 1
                file_prefix = ("    " if is_last_dir else "│   ") + ("└── " if is_last_file else "├── ")
                tree_lines.append(f"{file_prefix}{fname}")

        st.code("\n".join(tree_lines), language=None)

        st.divider()

        # ── Raw JSON output ───────────────────────────────────────────────────
        with st.expander("Raw JSON"):
            st.json(result.model_dump())

        # ── Download button ───────────────────────────────────────────────────
        st.download_button(
            label="Download JSON",
            data=json.dumps(result.model_dump(), indent=2),
            file_name="architecture.json",
            mime="application/json",
        )
