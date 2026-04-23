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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F9FC;
    color: #1F2937;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2.5rem 3rem 4rem 3rem;
    max-width: 1300px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem;
}

.sidebar-section {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin: 1.2rem 0 0.6rem;
}

.sidebar-sep {
    height: 1px;
    background: #E5E7EB;
    margin: 1rem 0;
}

/* INPUT */
[data-testid="stFileUploader"] {
    background: #F9FAFB;
    border: 1.5px dashed #D1D5DB;
    border-radius: 10px;
    padding: 1rem;
}

/* HEADER */
.hero-badge {
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    color: #4F46E5;
    font-size: 0.7rem;
    padding: 0.3rem 0.7rem;
    border-radius: 100px;
    display: inline-block;
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.7rem;
    font-weight: 800;
    color: #111827;
}

.hero-title span {
    color: #4F46E5;
}

.hero-sub {
    color: #6B7280;
    font-size: 0.95rem;
}

.hero-divider {
    height: 1px;
    background: #E5E7EB;
    margin: 1.8rem 0 2rem;
}

/* IMAGE PANEL */
.img-panel {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
}

.img-panel-header {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid #E5E7EB;
}

.img-panel-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6B7280;
}

.img-panel-body {
    padding: 1rem;
}

/* RESULT */
.result-wrapper {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 18px;
    padding: 2rem;
}

.result-type-label {
    font-size: 0.7rem;
    color: #9CA3AF;
}

.result-type-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #4F46E5;
}

.result-desc {
    color: #6B7280;
    font-size: 0.9rem;
}

/* SCORE */
.score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
}

.score-label {
    font-size: 0.75rem;
    color: #6B7280;
    min-width: 110px;
}

.score-bar-track {
    flex: 1;
    height: 6px;
    background: #E5E7EB;
    border-radius: 100px;
}

.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: #4F46E5;
}

.score-num {
    font-size: 0.8rem;
    font-weight: 600;
}

/* OCR */
.ocr-box {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.2rem;
    font-size: 0.85rem;
    color: #374151;
    line-height: 1.6;
}

/* EMPTY */
.empty-state {
    text-align: center;
    padding: 4rem;
    border: 1px dashed #D1D5DB;
    border-radius: 16px;
    background: #FFFFFF;
}

.empty-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #9CA3AF;
}

.empty-sub {
    font-size: 0.9rem;
    color: #9CA3AF;
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
