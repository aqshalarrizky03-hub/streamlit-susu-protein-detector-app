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

# ===== FORCE SIDEBAR OPEN via JavaScript =====
# Inject JS to click the sidebar toggle button if sidebar is collapsed
st.markdown("""
<script>
    // Wait for Streamlit to fully render, then ensure sidebar is expanded
    function expandSidebar() {
        // Try to find and click the sidebar expand button if collapsed
        const buttons = window.parent.document.querySelectorAll('button[kind="header"]');
        buttons.forEach(btn => {
            const ariaLabel = btn.getAttribute('aria-label') || '';
            if (ariaLabel.toLowerCase().includes('open') || ariaLabel.toLowerCase().includes('show')) {
                btn.click();
            }
        });

        // Also try by data-testid
        const collapseBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if (collapseBtn) {
            collapseBtn.click();
        }
    }
    
    // Run after a short delay to ensure DOM is ready
    setTimeout(expandSidebar, 300);
    setTimeout(expandSidebar, 800);
    setTimeout(expandSidebar, 1500);
</script>
""", unsafe_allow_html=True)

# ===== STYLE =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/*
  COLOR PALETTE
  #1F6F5F  deep forest green (primary)
  #6FCF97  fresh mint green  (accent)
  #EEEEEE  cream white       (text)
  #0E1A18  near-black bg
  #132320  dark panel
  #1F3530  border
  #3D6B60  muted
  #A8C4BE  soft text
*/

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0E1A18 !important;
    color: #EEEEEE;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2.5rem 3rem 4rem 3rem !important;
    max-width: 1300px;
}

/* ── SIDEBAR — fully forced open & styled ── */
section[data-testid="stSidebar"] {
    background-color: #0B1614 !important;
    border-right: 1px solid #1F3530 !important;
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
    transform: none !important;
    display: flex !important;
    flex-direction: column !important;
    visibility: visible !important;
    opacity: 1 !important;
    transition: none !important;
}

/* Force the inner wrapper visible */
section[data-testid="stSidebar"] > div:first-child {
    width: 300px !important;
    min-width: 300px !important;
    display: flex !important;
    flex-direction: column !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: none !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem !important;
    width: 100% !important;
    visibility: visible !important;
}

/* All sidebar children must be visible */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] > *,
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > * {
    visibility: visible !important;
    opacity: 1 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([data-testid="collapsedControl"]) {
    color: #A8C4BE !important;
}

/* Sidebar file uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #132320 !important;
    border: 1.5px dashed #1F6F5F !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
    background: rgba(31,111,95,0.08) !important;
    border: 1px dashed #1F3530 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {
    border-color: #6FCF97 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background-color: #1F6F5F !important;
    color: #EEEEEE !important;
    border: none !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] button:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    background-color: #6FCF97 !important;
    color: #0E1A18 !important;
}

/* Sidebar selectbox */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #132320 !important;
    border: 1px solid #1F3530 !important;
    border-radius: 8px !important;
    color: #EEEEEE !important;
}

/* Sidebar sliders */
section[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background-color: #1F6F5F !important;
}
section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    color: #6FCF97 !important;
}

/* Hide the collapse arrow button to prevent accidental collapse */
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ── HERO ── */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(31,111,95,0.2);
    border: 1px solid rgba(111,207,151,0.35);
    color: #6FCF97;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 0.85rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    color: #EEEEEE;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero-title span { color: #6FCF97; }
.hero-sub {
    color: #3D6B60;
    font-size: 0.92rem;
    font-weight: 300;
    margin: 0;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(to right, #1F6F5F 0%, #0E1A18 60%);
    margin: 1.5rem 0 2rem;
    border: none;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3D6B60;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1F3530;
}

/* ── IMAGE PANELS ── */
.img-panel {
    background: #132320;
    border: 1px solid #1F3530;
    border-radius: 16px;
    overflow: hidden;
}
.img-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.2rem;
    border-bottom: 1px solid #1F3530;
}
.img-panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #A8C4BE;
}
.img-panel-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6FCF97;
    box-shadow: 0 0 8px #6FCF97;
}
.img-panel-body { padding: 1rem; }

/* ── RESULT CARD ── */
.result-wrapper {
    background: linear-gradient(135deg, #132320 0%, #1A3530 100%);
    border: 1px solid #1F3530;
    border-radius: 20px;
    padding: 2.5rem;
    position: relative;
    overflow: hidden;
}
.result-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(to right, #1F6F5F, #6FCF97, #EEEEEE);
}
.result-type-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3D6B60;
    margin-bottom: 0.5rem;
}
.result-type-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    background: linear-gradient(135deg, #6FCF97 0%, #EEEEEE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
}
.result-desc {
    color: #A8C4BE;
    font-size: 0.85rem;
    font-weight: 300;
    line-height: 1.5;
}

/* ── SCORE BARS ── */
.score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.85rem;
    padding: 0.7rem 1rem;
    background: #132320;
    border-radius: 10px;
    border: 1px solid #1F3530;
}
.score-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #A8C4BE;
    min-width: 105px;
}
.score-bar-track {
    flex: 1;
    height: 6px;
    background: #1F3530;
    border-radius: 100px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 100px;
}
.score-num {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #6FCF97;
    min-width: 18px;
    text-align: right;
}
.bar-whey        { background: linear-gradient(to right, #1F6F5F, #6FCF97); }
.bar-casein      { background: linear-gradient(to right, #2A8A70, #EEEEEE); }
.bar-plant       { background: linear-gradient(to right, #6FCF97, #EEEEEE); }
.bar-mass_gainer { background: linear-gradient(to right, #1F6F5F, #A8C4BE); }

/* ── OCR BOX ── */
.ocr-box {
    background: #132320;
    border: 1px solid #1F3530;
    border-radius: 14px;
    padding: 1.5rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #A8C4BE;
    line-height: 1.8;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    scrollbar-width: thin;
    scrollbar-color: #1F6F5F transparent;
}

/* ── INFO CHIPS ── */
.info-strip {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 0.85rem;
}
.info-chip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: #0E1A18;
    border: 1px solid #1F3530;
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.76rem;
    color: #3D6B60;
}
.info-chip b { color: #A8C4BE; }

/* ── EMPTY STATE ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 5rem 2rem;
    background: #132320;
    border: 1.5px dashed #1F3530;
    border-radius: 20px;
    gap: 1rem;
}
.empty-icon { font-size: 3.5rem; opacity: 0.4; }
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1F6F5F;
}
.empty-sub {
    color: #3D6B60;
    font-size: 0.85rem;
    max-width: 320px;
    line-height: 1.6;
}

/* ── SIDEBAR SECTION LABELS ── */
.sb-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6FCF97;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1F3530;
}
.sb-sep {
    height: 1px;
    background: #1F3530;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ===== DATA =====
keywords = {
    "whey":        ["whey", "whey isolate", "whey concentrate", "whey protein"],
    "casein":      ["casein", "micellar casein", "calcium caseinate"],
    "plant":       ["soy protein", "pea protein", "rice protein", "vegan", "plant protein"],
    "mass_gainer": ["mass gainer", "weight gainer", "maltodextrin", "high calorie", "carbohydrate"]
}
PROTEIN_DESCRIPTIONS = {
    "whey":        "Fast-absorbing protein, ideal untuk post-workout recovery",
    "casein":      "Slow-release protein, optimal untuk muscle repair saat tidur",
    "plant":       "Plant-based vegan protein blend terdeteksi",
    "mass_gainer": "Formula high-calorie untuk mass & weight gainer"
}
BAR_COLORS = {
    "whey": "bar-whey", "casein": "bar-casein",
    "plant": "bar-plant", "mass_gainer": "bar-mass_gainer"
}

def preprocess_text(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower())

def classify_protein(text):
    text  = preprocess_text(text)
    score = {k: 0 for k in keywords}
    for cat, words in keywords.items():
        for w in words:
            if w in text:
                score[cat] += 1
    if score["mass_gainer"] > 0:
        score["mass_gainer"] += 2
    return max(score, key=score.get), score

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<div class="sb-label">📂 Upload Gambar</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload label komposisi",
        type=["jpg", "png", "jpeg"],
        help="Foto label nutrisi / komposisi produk protein suplemen"
    )

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">🔧 Geometric Transform</div>', unsafe_allow_html=True)
    operation = st.selectbox("Pilih Operasi", ["none", "translasi", "rotasi", "scaling"])
    tx, ty, angle, scale = 0, 0, 0, 1.0
    if operation == "translasi":
        tx = st.slider("Geser X", -100, 100, 0)
        ty = st.slider("Geser Y", -100, 100, 0)
    elif operation == "rotasi":
        angle = st.slider("Sudut Rotasi", -180, 180, 0)
    elif operation == "scaling":
        scale = st.slider("Scale Factor", 1.0, 3.0, 1.0)

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">🎨 Image Processing</div>', unsafe_allow_html=True)
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast   = st.slider("Contrast",   1.0, 3.0, 1.0)
    thresh     = st.slider("Threshold",  0, 255, 127)
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
    img   = np.array(image)

    proc_img   = img.copy()
    rows, cols = proc_img.shape[:2]
    if operation == "translasi":
        M        = np.float32([[1, 0, tx], [0, 1, ty]])
        proc_img = cv2.warpAffine(proc_img, M, (cols, rows))
    elif operation == "rotasi":
        M        = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        proc_img = cv2.warpAffine(proc_img, M, (cols, rows))
    elif operation == "scaling":
        proc_img = cv2.resize(proc_img, None, fx=scale, fy=scale)

    proc = cv2.convertScaleAbs(proc_img, alpha=contrast, beta=brightness)
    gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
    if blur_k > 1:
        bk   = blur_k if blur_k % 2 != 0 else blur_k + 1
        gray = cv2.GaussianBlur(gray, (bk, bk), 0)
    _, binary     = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    text          = pytesseract.image_to_string(binary)
    result, score = classify_protein(text)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<p class="section-label">Original Image</p>', unsafe_allow_html=True)
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">Input</span><span class="img-panel-dot"></span></div><div class="img-panel-body">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        file_size = round(uploaded_file.size / 1024, 1)
        w, h = image.size
        st.markdown(f'<div class="info-strip"><div class="info-chip">📐 <b>{w}×{h}</b> px</div><div class="info-chip">💾 <b>{file_size} KB</b></div><div class="info-chip">🔄 <b>{operation.capitalize()}</b></div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-label">Processed Image</p>', unsafe_allow_html=True)
        st.markdown('<div class="img-panel"><div class="img-panel-header"><span class="img-panel-title">Binary · OCR Ready</span><span class="img-panel-dot" style="background:#1F6F5F;box-shadow:0 0 8px #1F6F5F"></span></div><div class="img-panel-body">', unsafe_allow_html=True)
        st.image(binary, use_column_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        ocr_chars = len(text.strip())
        ocr_words = len(text.split())
        st.markdown(f'<div class="info-strip"><div class="info-chip">🔡 <b>{ocr_chars}</b> chars</div><div class="info-chip">📝 <b>{ocr_words}</b> kata</div><div class="info-chip">⚡ Thresh <b>{thresh}</b></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    res_col, score_col = st.columns([1, 1], gap="large")
    with res_col:
        st.markdown('<p class="section-label">Hasil Klasifikasi</p>', unsafe_allow_html=True)
        desc = PROTEIN_DESCRIPTIONS.get(result, "Protein terdeteksi")
        st.markdown(f'<div class="result-wrapper"><div class="result-type-label">Protein Type Detected</div><div class="result-type-value">{result.replace("_"," ").upper()}</div><div class="result-desc">{desc}</div></div>', unsafe_allow_html=True)

    with score_col:
        st.markdown('<p class="section-label">Confidence Score</p>', unsafe_allow_html=True)
        for k, v in score.items():
            pct   = int((v / 5) * 100)
            cls   = BAR_COLORS.get(k, "bar-whey")
            label = k.replace("_", " ").title()
            st.markdown(f'<div class="score-row"><div class="score-label">{label}</div><div class="score-bar-track"><div class="score-bar-fill {cls}" style="width:{pct}%"></div></div><div class="score-num">{v}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Extracted OCR Text</p>', unsafe_allow_html=True)
    clean_text = text.strip() or "Tidak ada teks terdeteksi. Coba sesuaikan Threshold atau Contrast di sidebar."
    st.markdown(f'<div class="ocr-box">{clean_text}</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🥛</div>
        <div class="empty-title">Belum ada gambar</div>
        <div class="empty-sub">Upload foto label komposisi dari <b style="color:#6FCF97">sidebar kiri ← </b> untuk memulai analisis OCR otomatis.</div>
    </div>
    """, unsafe_allow_html=True)
