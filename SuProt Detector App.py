import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re

# ===== CONFIG =====
st.set_page_config(
    page_title="SuProt Detector",
    layout="wide",
    page_icon="🥛",
    initial_sidebar_state="expanded"
)

# ===== STYLE =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0C10;
    color: #EEEEEE;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding: 2rem 3rem;
    max-width: 1300px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #1F6F5F;
}

/* Header */
.hero-badge {
    display:inline-block;
    background: rgba(111,207,151,0.1);
    border:1px solid rgba(111,207,151,0.4);
    color:#6FCF97;
    padding:5px 12px;
    border-radius:20px;
    font-size:12px;
    margin-bottom:10px;
}

.hero-title {
    font-family:'Syne', sans-serif;
    font-size:42px;
    font-weight:800;
}

.hero-title span {
    color:#6FCF97;
}

.hero-sub {
    color:#888;
    font-size:14px;
}

.hero-divider {
    height:1px;
    background:linear-gradient(to right,#6FCF97,#1F6F5F);
    margin:20px 0;
}

/* Panel */
.img-panel {
    border:1px solid #1F6F5F;
    border-radius:12px;
    padding:10px;
}

/* Result */
.result-box {
    padding:20px;
    border-radius:15px;
    border:1px solid #1F6F5F;
    background:#0D1117;
}

.result-title {
    font-size:30px;
    font-weight:800;
    color:#6FCF97;
}

/* Score */
.score-bar {
    height:6px;
    background:#1F6F5F;
    border-radius:10px;
}

.score-fill {
    height:100%;
    background:#6FCF97;
    border-radius:10px;
}

/* OCR */
.ocr-box {
    border:1px solid #1F6F5F;
    padding:15px;
    border-radius:10px;
    background:#0D1117;
    font-size:13px;
    color:#CCCCCC;
    max-height:200px;
    overflow:auto;
}
</style>
""", unsafe_allow_html=True)

# ===== KEYWORDS =====
keywords = {
    "whey": ["whey", "whey isolate", "whey concentrate", "whey protein"],
    "casein": ["casein", "micellar casein", "calcium caseinate"],
    "plant": ["soy protein", "pea protein", "rice protein", "vegan", "plant protein"],
    "mass_gainer": ["mass gainer", "weight gainer", "maltodextrin", "high calorie", "carbohydrate"]
}

PROTEIN_DESCRIPTIONS = {
    "whey": "Fast-absorbing protein, ideal for post-workout recovery",
    "casein": "Slow-release protein, optimal for overnight muscle repair",
    "plant": "Plant-based vegan protein blend detected",
    "mass_gainer": "High-calorie mass & weight gainer formula"
}

BAR_COLORS = {
    "whey": "bar-whey",
    "casein": "bar-casein",
    "plant": "bar-plant",
    "mass_gainer": "bar-mass_gainer"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def classify_protein(text):
    text = preprocess_text(text)
    score = {k: 0 for k in keywords}
    for category, words in keywords.items():
        for word in words:
            if word in text:
                score[category] += 1
    if score["mass_gainer"] > 0:
        score["mass_gainer"] += 2
    return max(score, key=score.get), score

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<div class="sidebar-section">📂 Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload label komposisi", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">🔧 Geometric Transform</div>', unsafe_allow_html=True)

    operation = st.selectbox("Operasi", ["none", "translasi", "rotasi", "scaling"], label_visibility="collapsed")

    tx, ty, angle, scale = 0, 0, 0, 1.0
    if operation == "translasi":
        tx = st.slider("Geser X", -100, 100, 0)
        ty = st.slider("Geser Y", -100, 100, 0)
    elif operation == "rotasi":
        angle = st.slider("Sudut Rotasi", -180, 180, 0)
    elif operation == "scaling":
        scale = st.slider("Scale Factor", 1.0, 3.0, 1.0)

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">🎨 Image Processing</div>', unsafe_allow_html=True)

    brightness = st.slider("Brightness", -100, 100, 0)
    contrast   = st.slider("Contrast", 1.0, 3.0, 1.0)
    thresh     = st.slider("Threshold", 0, 255, 127)
    blur_k     = st.slider("Blur (Gaussian)", 1, 15, 1)

# ===== HEADER =====
st.markdown("""
<div>
    <div class="hero-badge">🥛 Computer Vision · OCR · NLP</div>
    <h1 class="hero-title">SuProt<span>.</span>Detector</h1>
    <p class="hero-sub">Identifikasi otomatis jenis protein suplemen dari label komposisi menggunakan OCR & Image Processing</p>
</div>
<hr class="hero-divider">
""", unsafe_allow_html=True)

# ===== MAIN =====
if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # ── Process ──
    proc_img = img.copy()
    rows, cols = proc_img.shape[:2]

    if operation == "translasi":
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        proc_img = cv2.warpAffine(proc_img, M, (cols, rows))
    elif operation == "rotasi":
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        proc_img = cv2.warpAffine(proc_img, M, (cols, rows))
    elif operation == "scaling":
        proc_img = cv2.resize(proc_img, None, fx=scale, fy=scale)

    proc = cv2.convertScaleAbs(proc_img, alpha=contrast, beta=brightness)
    gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)

    if blur_k > 1:
        blur_k = blur_k if blur_k % 2 != 0 else blur_k + 1
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary)
    result, score = classify_protein(text)

    # ── Image Columns ──
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<p class="section-label">Original Image</p>', unsafe_allow_html=True)
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">Input</span><span class="img-panel-dot"></span></div><div class="img-panel-body">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # File info chips
        file_size = round(uploaded_file.size / 1024, 1)
        w, h = image.size
        st.markdown(f"""
        <div class="info-strip">
            <div class="info-chip">📐 <b>{w}×{h}</b> px</div>
            <div class="info-chip">💾 <b>{file_size} KB</b></div>
            <div class="info-chip">🔄 <b>{operation.capitalize()}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-label">Processed Image</p>', unsafe_allow_html=True)
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">Binary · OCR Ready</span><span class="img-panel-dot" style="background:#F4A261;box-shadow:0 0 8px #F4A261"></span></div><div class="img-panel-body">', unsafe_allow_html=True)
        st.image(binary, use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        ocr_chars = len(text.strip())
        ocr_words = len(text.split())
        st.markdown(f"""
        <div class="info-strip">
            <div class="info-chip">🔡 <b>{ocr_chars}</b> chars</div>
            <div class="info-chip">📝 <b>{ocr_words}</b> kata</div>
            <div class="info-chip">⚡ Thresh <b>{thresh}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results ──
    res_col, score_col = st.columns([1, 1], gap="large")

    with res_col:
        st.markdown('<p class="section-label">Hasil Klasifikasi</p>', unsafe_allow_html=True)
        desc = PROTEIN_DESCRIPTIONS.get(result, "Protein terdeteksi")
        st.markdown(f"""
        <div class="result-wrapper">
            <div class="result-type-label">Protein Type Detected</div>
            <div class="result-type-value">{result.replace('_', ' ').upper()}</div>
            <div class="result-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with score_col:
        st.markdown('<p class="section-label">Confidence Score</p>', unsafe_allow_html=True)
        max_score = max(score.values()) if max(score.values()) > 0 else 1
        for k, v in score.items():
            bar_pct = int((v / 5) * 100)
            bar_class = BAR_COLORS.get(k, "bar-whey")
            label = k.replace("_", " ").title()
            st.markdown(f"""
            <div class="score-row">
                <div class="score-label">{label}</div>
                <div class="score-bar-track">
                    <div class="score-bar-fill {bar_class}" style="width:{bar_pct}%"></div>
                </div>
                <div class="score-num">{v}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── OCR Output ──
    st.markdown('<p class="section-label">Extracted OCR Text</p>', unsafe_allow_html=True)
    clean_text = text.strip() if text.strip() else "Tidak ada teks terdeteksi. Coba sesuaikan parameter Threshold atau Contrast."
    st.markdown(f'<div class="ocr-box">{clean_text}</div>', unsafe_allow_html=True)

else:
    # ── Empty State ──
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🥛</div>
        <div class="empty-title">Belum ada gambar</div>
        <div class="empty-sub">Upload foto label komposisi suplemen protein dari sidebar untuk memulai analisis OCR otomatis.</div>
    </div>
    """, unsafe_allow_html=True)
