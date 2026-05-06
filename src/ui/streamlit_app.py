"""
Streamlit Frontend — Rice Disease Detection AI
Usage: streamlit run src/ui/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Disease Detection AI",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Minimal CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Remove default top padding */
.block-container { padding-top: 2rem; }

/* Step card for agent log */
.step-card {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
    color: #15803d;
}

/* Result badge */
.disease-badge {
    display: inline-block;
    background: #16a34a;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 99px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🌾 Rice Disease Detection")
st.caption("Upload a rice leaf image — the AI pipeline will identify the disease and provide a treatment report.")
st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a rice leaf image (JPG, PNG, WebP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if not uploaded_file:
    st.info("👆 Upload an image to get started.")
    st.stop()

# Show preview
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption="Uploaded image", use_container_width=True)

# ── Diagnose button ────────────────────────────────────────────────────────────
if not st.button("🔍 Run Diagnosis", type="primary", use_container_width=True):
    st.stop()

# ── Pipeline ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("⚙️ Agent Pipeline")
log_container = st.container()
logs = []

def update_progress(msg: str):
    logs.append(msg)
    with log_container:
        for m in logs:
            st.markdown(f'<div class="step-card">🤖 {m}</div>', unsafe_allow_html=True)

with st.spinner("Running diagnosis…"):
    from src.core.workflow import run_diagnosis
    results = run_diagnosis(img, progress_callback=update_progress)

st.success("✅ Diagnosis complete!")

# ── Results ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Results")

label = results["classification"]["disease_label"]
conf  = results["classification"]["confidence"]

# Disease label + confidence
st.markdown(f'<div class="disease-badge">{label}</div>', unsafe_allow_html=True)
st.write(f"**Confidence:** {conf:.1f}%")
st.progress(min(int(conf), 100))

# Preprocessed image
if "preprocessed_image" in results and results["preprocessed_image"]:
    with st.expander("🖼️ Preprocessed image (background removed)"):
        st.image(results["preprocessed_image"], use_container_width=True)

st.divider()

# Morphology analysis
st.subheader("🧬 Morphology Analysis")
st.markdown(results["morphology"])

st.divider()

# Final report
st.subheader("📄 Diagnosis Report")
st.markdown(results["final_report"])

# RAG knowledge (collapsed by default)
with st.expander("📚 Scientific Knowledge Base (RAG)"):
    st.markdown(results["knowledge"])

# Download
st.download_button(
    "⬇️ Download report (.txt)",
    data=results["final_report"],
    file_name=f"RiceAI_{label.replace(' ', '_')}.txt",
    mime="text/plain",
    use_container_width=True,
)
