"""
Streamlit Frontend — Rice Disease Detection AI
Usage: streamlit run src/ui/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from PIL import Image
import io
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rice Disease Detection AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Premium Look ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #2d6a4f;
    --secondary: #40916c;
    --accent: #d8f3dc;
    --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    --glass: rgba(255, 255, 255, 0.7);
    --glass-border: rgba(255, 255, 255, 0.4);
    --text-main: #1e293b;
    --text-muted: #64748b;
}

html, body, [class*="css"] { 
    font-family: 'Outfit', sans-serif; 
    color: var(--text-main);
}

.stApp {
    background: var(--bg-gradient);
}

/* Glassmorphism Card */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.12);
}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 3rem 1rem;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1b4332 0%, #40916c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -2px;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: var(--text-muted);
    font-weight: 400;
}

/* Agent Log Item */
.log-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    margin-bottom: 0.5rem;
    border-left: 4px solid var(--primary);
    animation: fadeIn 0.5s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Badge Styles */
.disease-badge {
    background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-block;
    box-shadow: 0 4px 15px rgba(27, 67, 50, 0.2);
}

/* Buttons */
.stButton > button {
    border-radius: 16px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.2s;
    border: none;
    background: var(--primary) !important;
    color: white !important;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 20px rgba(45, 106, 79, 0.2);
}

/* Sidebar Branding */
.sidebar-logo {
    text-align: center;
    margin-bottom: 2rem;
}

/* Results Grid */
.result-section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/UConn_Huskies_logo.svg/440px-UConn_Huskies_logo.svg.png", width=100)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🌾 RiceAI Assistant")
    st.info("Hệ thống chẩn đoán bệnh lúa thông minh dựa trên AI Đa Tác Nhân.")
    
    with st.expander("📖 Hướng dẫn sử dụng", expanded=True):
        st.markdown("""
        1. **Tải ảnh**: Chọn ảnh lá lúa bị bệnh.
        2. **Phân tích**: Nhấn nút 'Bắt đầu chẩn đoán'.
        3. **Kết quả**: Xem báo cáo chi tiết từ các Agent.
        """)
        
    st.divider()
    st.markdown("#### 🦠 Các bệnh có thể nhận diện")
    st.markdown("- 🌾 **Đạo ôn (Blast)**")
    st.markdown("- 🍂 **Bạc lá (Blight)**")
    st.markdown("- 🌑 **Đốm nâu (Brown Spot)**")
    st.markdown("- 🌀 **Vàng lùn (Tungro)**")
    
    st.divider()
    st.caption("v2.0.0 • Developed by RiceAI Team")

# ── Hero Banner ────────────────────────────────────────────────────────────────
hero_image_path = os.path.join(os.path.dirname(__file__), "assets/hero.png")
if os.path.exists(hero_image_path):
    st.image(hero_image_path, use_container_width=True)
else:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌾 Rice Disease Detection AI</div>
        <div class="hero-subtitle">Advanced Multi-Agent Diagnosis System</div>
    </div>
    """, unsafe_allow_html=True)

# ── Main Content Layout ───────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
col_input, col_info = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📸 Phân tích mẫu bệnh")
    uploaded_file = st.file_uploader(
        "Tải lên ảnh lá lúa (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Mẫu lá đã tải lên", use_container_width=True)
        run_btn = st.button("🚀 Bắt đầu chẩn đoán", use_container_width=True)
    else:
        st.warning("Vui lòng tải lên một hình ảnh để bắt đầu.")
        run_btn = False

with col_info:
    st.markdown("### 🛠️ Cơ chế hoạt động")
    st.markdown("""
    Hệ thống sử dụng **5 Agent chuyên biệt** làm việc cùng nhau:
    - **ViT Agent**: Phân loại bệnh bằng Vision Transformer.
    - **Vision Agent**: Phân tích hình thái qua Gemini Vision.
    - **RAG Agent**: Truy xuất kiến thức từ ChromaDB.
    - **Synthesizer**: Tổng hợp báo cáo chuyên môn.
    - **Preprocess**: Tối ưu hóa ảnh đầu vào.
    """)
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
st.markdown('</div>', unsafe_allow_html=True)

# ── Execution Logic ────────────────────────────────────────────────────────────
if run_btn and uploaded_file:
    st.divider()
    
    col_log, col_res = st.columns([1, 2], gap="large")
    
    with col_log:
        st.markdown('<div class="result-section-title">⚙️ Agent Pipeline</div>', unsafe_allow_html=True)
        log_container = st.empty()
        logs = []
        
        def update_progress(msg: str):
            logs.append(msg)
            with log_container.container():
                for m in logs:
                    st.markdown(f'<div class="log-item"><span>🤖</span> {m}</div>', unsafe_allow_html=True)
        
        with st.spinner("Đang khởi tạo Agent..."):
            from src.core.workflow import run_diagnosis
            results = run_diagnosis(img, progress_callback=update_progress)
            
        st.success("Tất cả Agent đã hoàn tất!")

    with col_res:
        st.markdown('<div class="result-section-title">📊 Kết quả phân tích</div>', unsafe_allow_html=True)
        
        # Classification Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        label = results["classification"]["disease_label"]
        conf = results["classification"]["confidence"]
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f'<div class="disease-badge">{label}</div>', unsafe_allow_html=True)
            st.metric("Độ tin cậy", f"{conf}%")
        with c2:
            if "preprocessed_image" in results:
                st.image(results["preprocessed_image"], caption="Ảnh đã xử lý", width=150)
        
        st.progress(int(conf))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Morphology
        with st.expander("🧬 Phân tích hình thái chi tiết", expanded=True):
            st.markdown(results["morphology"])

    # ── Final Report ──────────────────────────────────────────────────────────
    st.markdown('<div class="result-section-title">📄 Báo cáo chẩn đoán cuối cùng</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(results["final_report"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── RAG Knowledge ─────────────────────────────────────────────────────────
    with st.expander("📚 Kiến thức bệnh học (RAG)", expanded=False):
        st.markdown(results["knowledge"])
        
    # ── Actions ───────────────────────────────────────────────────────────────
    report_text = results["final_report"]
    st.download_button(
        "⬇️ Tải báo cáo PDF/Text",
        data=report_text,
        file_name=f"RiceAI_Report_{label}.txt",
        mime="text/plain",
        use_container_width=True
    )

st.divider()
st.center = st.markdown('<div style="text-align: center; color: var(--text-muted);">© 2026 Rice Disease Detection AI • UConn Agriculture AI Lab</div>', unsafe_allow_html=True)
