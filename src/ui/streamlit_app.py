"""
Streamlit Frontend — Hệ thống Chẩn Đoán Bệnh Cây Lúa
Usage: streamlit run src/ui/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from PIL import Image

# ── Cấu hình trang ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Disease Detection | Chẩn Đoán Bệnh Cây Lúa",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS tối giản ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container { padding-top: 2rem; }

.step-card {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.9rem;
    color: #15803d;
}

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

# ── Tiêu đề ───────────────────────────────────────────────────────────────────
st.title("🌾 Rice Disease Detection")
st.markdown("**Chẩn Đoán Bệnh Cây Lúa** — Hệ thống AI Đa Tác Nhân")
st.caption("Tải lên ảnh lá lúa — hệ thống AI đa tác nhân sẽ nhận diện bệnh và đưa ra báo cáo điều trị.")
st.divider()

# ── Tải ảnh lên ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Chọn ảnh lá lúa (JPG, PNG, WebP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if not uploaded_file:
    st.info("👆 Tải lên ảnh lá lúa để bắt đầu.")
    st.stop()

# Xem trước ảnh
img = Image.open(uploaded_file).convert("RGB")
st.image(img, caption="Ảnh đã tải lên", use_container_width=True)

# ── Nút chẩn đoán ────────────────────────────────────────────────────────────
if not st.button("🔍 Bắt đầu chẩn đoán", type="primary", use_container_width=True):
    st.stop()

# ── Tiến trình pipeline ───────────────────────────────────────────────────────
st.divider()
st.subheader("⚙️ Tiến trình các Agent")
log_container = st.container()
logs = []

def update_progress(msg: str):
    logs.append(msg)
    with log_container:
        for m in logs:
            st.markdown(f'<div class="step-card">🤖 {m}</div>', unsafe_allow_html=True)

with st.spinner("Đang phân tích, vui lòng chờ…"):
    from src.core.workflow import run_diagnosis
    results = run_diagnosis(img, progress_callback=update_progress)

st.success("✅ Chẩn đoán hoàn tất!")

# ── Kết quả ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Kết quả phân loại")

label = results["classification"]["disease_label"]
conf  = results["classification"]["confidence"]

st.markdown(f'<div class="disease-badge">{label}</div>', unsafe_allow_html=True)
st.write(f"**Độ tin cậy:** {conf:.1f}%")
st.progress(min(int(conf), 100))

st.divider()

# Phân tích hình thái
st.subheader("🧬 Phân tích hình thái lá")
st.markdown(results["morphology"])

st.divider()

# Báo cáo chẩn đoán
st.subheader("📄 Báo cáo chẩn đoán")
st.markdown(results["final_report"])

# Tri thức bệnh học RAG (thu gọn mặc định)
with st.expander("📚 Tri thức bệnh học (ChromaDB RAG)"):
    st.markdown(results["knowledge"])

# Nút tải xuống
st.download_button(
    "⬇️ Tải báo cáo (.txt)",
    data=results["final_report"],
    file_name=f"BaoCao_{label.replace(' ', '_')}.txt",
    mime="text/plain",
    use_container_width=True,
)
