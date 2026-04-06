"""
Streamlit Frontend — Rice Leaf Disease Multi-Agent System
Usage: streamlit run src/ui/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafAI — Chẩn Đoán Bệnh Lúa",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Gradient header */
.hero-header {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(29,85,50,0.3);
}
.hero-header h1 { font-size: 2.8rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero-header p  { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; }

/* Agent step card */
.step-card {
    background: #f8fffe;
    border-left: 4px solid #40916c;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0;
    font-size: 0.95rem;
    color: #1a472a;
    animation: slideIn 0.4s ease-out;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* Result card */
.result-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}
.disease-badge {
    display: inline-block;
    background: linear-gradient(90deg, #1a472a, #40916c);
    color: white;
    border-radius: 24px;
    padding: 0.35rem 1.1rem;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.3px;
}
.confidence-bar-bg {
    background: #e8f5e9;
    border-radius: 8px;
    height: 12px;
    margin-top: 0.4rem;
}
.stProgress > div > div > div > div { background-color: #40916c; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/UConn_Huskies_logo.svg/440px-UConn_Huskies_logo.svg.png", width=120)
    st.markdown("### 🌾 LeafAI — Hướng dẫn")
    st.markdown("""
**Các bước sử dụng:**
1. Tải lên ảnh chụp lá lúa
2. Nhấn **Chẩn đoán ngay**
3. Theo dõi từng Agent hoạt động
4. Đọc báo cáo chẩn đoán chi tiết

**Bệnh hỗ trợ:**
- 🔴 Đạo ôn (Rice Blast)
- 🟡 Bạc lá (Bacterial Blight)
- 🟤 Đốm nâu (Brown Spot)
- 🟠 Vàng lùn (Rice Tungro)
""")
    st.divider()
    st.caption("Phát triển bởi AI Agent System · UConn 2026")

# ── Hero Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🌿 LeafAI — Hệ Thống Chẩn Đoán Bệnh Lúa</h1>
    <p>Ứng dụng Đa Tác Nhân AI · Vision Transformer · Gemini · ChromaDB</p>
</div>
""", unsafe_allow_html=True)

# ── Upload Widget ────────────────────────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### 📸 Tải lên ảnh lá lúa")
    uploaded_file = st.file_uploader(
        "Chọn ảnh JPG hoặc PNG",
        type=["jpg", "jpeg", "png", "webp"],
        help="Nên chụp ảnh rõ nét, ánh sáng đủ, tập trung vào phần lá cần kiểm tra.",
        label_visibility="collapsed"
    )
    
    run_button = st.button("🔍 Chẩn đoán ngay", type="primary", disabled=uploaded_file is None, use_container_width=True)

with col_preview:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh gốc đã tải lên", use_container_width=True)

# ── Diagnosis Pipeline ───────────────────────────────────────────────────────────
if run_button and uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.divider()
    st.markdown("### ⚙️ Tiến trình các Agent")
    
    log_container = st.container()
    logs = []

    def update_progress(msg: str):
        logs.append(msg)
        with log_container:
            for log in logs:
                st.markdown(f'<div class="step-card">{log}</div>', unsafe_allow_html=True)

    with st.spinner("Hệ thống đang khởi động..."):
        # Lazy import so models load only when user clicks diagnose
        from src.core.workflow import run_diagnosis
        result = run_diagnosis(image, progress_callback=update_progress)

    st.success("✅ Chẩn đoán hoàn tất!")
    st.divider()

    # ── Results Layout ────────────────────────────────────────────────────────────
    st.markdown("## 📋 Kết Quả Chẩn Đoán")
    
    r_col1, r_col2 = st.columns([1, 1], gap="large")

    with r_col1:
        # -- Classification Result
        label = result["classification"]["disease_label"]
        conf  = result["classification"]["confidence"]
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("**🔬 Kết quả phân loại (ViT)**")
        st.markdown(f'<span class="disease-badge">🌿 {label}</span>', unsafe_allow_html=True)
        st.markdown(f"**Độ tin cậy:** `{conf}%`")
        st.progress(int(conf))

        st.markdown("**Phân phối nhãn:**")
        probs = result["classification"].get("all_probabilities", {})
        for cls, pct in sorted(probs.items(), key=lambda x: -x[1]):
            st.markdown(f"- `{cls}`: **{pct}%**")
        st.markdown('</div>', unsafe_allow_html=True)

        # -- Preprocessed image
        if "preprocessed_image" in result:
            st.markdown("**🖼️ Ảnh sau khi tách nền:**")
            st.image(result["preprocessed_image"], use_container_width=True)

    with r_col2:
        # -- Morphology
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("**🧬 Phân tích hình thái (Gemini Vision)**")
        st.markdown(result["morphology"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # -- Full Report
    st.markdown("## 📄 Báo Cáo Chẩn Đoán Đầy Đủ")
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(result["final_report"])
    st.markdown('</div>', unsafe_allow_html=True)

    # -- Knowledge
    with st.expander("📚 Thông tin từ cơ sở dữ liệu bệnh học (RAG)", expanded=False):
        st.markdown(result["knowledge"])
    
    # -- Download report
    report_bytes = result["final_report"].encode("utf-8")
    st.download_button(
        label="⬇️ Tải báo cáo (.txt)",
        data=report_bytes,
        file_name=f"chan_doan_{label.replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True
    )
