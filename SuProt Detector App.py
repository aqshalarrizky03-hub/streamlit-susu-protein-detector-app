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
    page_icon="🥛"
)

# ===== STYLE =====
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #111827;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.small-text {
    color: #9CA3AF;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===== TITLE =====
st.title("🥛 SuProt Detector")
st.caption("Deteksi jenis protein dari label komposisi menggunakan OCR & Image Processing")

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader("Upload Gambar", type=["jpg","png","jpeg"])

operation = st.sidebar.selectbox(
    "Operasi Geometri",
    ["none", "translasi", "rotasi", "scaling"]
)

if operation == "translasi":
    tx = st.sidebar.slider("Geser X", -100, 100, 0)
    ty = st.sidebar.slider("Geser Y", -100, 100, 0)

elif operation == "rotasi":
    angle = st.sidebar.slider("Rotasi", -180, 180, 0)

elif operation == "scaling":
    scale = st.sidebar.slider("Scale", 1.0, 3.0, 1.0)

st.sidebar.markdown("---")

brightness = st.sidebar.slider("Brightness", -100, 100, 0)
contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.0)
thresh = st.sidebar.slider("Threshold", 0, 255, 127)
blur_k = st.sidebar.slider("Blur", 1, 15, 1)

# ===== KEYWORDS =====
keywords = {
    "whey": ["whey", "whey isolate", "whey concentrate", "whey protein"],
    "casein": ["casein", "micellar casein", "calcium caseinate"],
    "plant": ["soy protein", "pea protein", "rice protein", "vegan", "plant protein"],
    "mass_gainer": ["mass gainer", "weight gainer", "maltodextrin", "high calorie", "carbohydrate"]
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def classify_protein(text):
    text = preprocess_text(text)
    score = {k:0 for k in keywords}

    for category, words in keywords.items():
        for word in words:
            if word in text:
                score[category] += 1

    if score["mass_gainer"] > 0:
        score["mass_gainer"] += 2

    return max(score, key=score.get), score

# ===== MAIN =====
if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("📷 Input")
        st.image(image, use_column_width=True)

    # ===== PROCESS =====
    proc_img = img.copy()
    rows, cols = proc_img.shape[:2]

    if operation == "translasi":
        M = np.float32([[1,0,tx],[0,1,ty]])
        proc_img = cv2.warpAffine(proc_img, M, (cols, rows))

    elif operation == "rotasi":
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        proc_img = cv2.warpAffine(proc_img, M, (cols, rows))

    elif operation == "scaling":
        proc_img = cv2.resize(proc_img, None, fx=scale, fy=scale)

    proc = cv2.convertScaleAbs(proc_img, alpha=contrast, beta=brightness)
    gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)

    if blur_k > 1:
        if blur_k % 2 == 0:
            blur_k += 1
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(binary)
    result, score = classify_protein(text)

    with col2:
        st.subheader("⚙️ Processed")
        st.image(binary, use_column_width=True)

    st.markdown("---")

    # ===== HASIL =====
    st.subheader("📊 Hasil Klasifikasi")

    st.markdown(f"""
    <div class="card">
        <h2>{result.upper()}</h2>
        <p class="small-text">Kategori protein terdeteksi berdasarkan komposisi</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔎 Confidence Score")
    for k,v in score.items():
        st.progress(v/5 if v>0 else 0)
        st.write(f"{k} : {v}")

    st.markdown("---")

    st.subheader("📝 Hasil OCR")
    st.text_area("Extracted Text", text, height=200)

else:
    st.info("Silakan upload gambar terlebih dahulu dari sidebar.")
