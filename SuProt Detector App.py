import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re

st.title("Klasifikasi Susu Protein (OCR + Image Processing)")

# ===== UPLOAD =====
uploaded_file = st.file_uploader("Upload Gambar Komposisi", type=["jpg","png","jpeg"])

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

# ===== PROSES =====
if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Gambar Input", use_column_width=True)

    # ===== PILIH OPERASI =====
    operation = st.selectbox("Pilih Operasi Geometri", ["none", "translasi", "rotasi", "scaling"])

    if operation == "translasi":
        tx = st.slider("Geser X", -100, 100, 0)
        ty = st.slider("Geser Y", -100, 100, 0)

    elif operation == "rotasi":
        angle = st.slider("Sudut Rotasi", -180, 180, 0)

    elif operation == "scaling":
        scale = st.slider("Scale", 1.0, 3.0, 1.0)

    # ===== PARAMETER UMUM =====
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", 1.0, 3.0, 1.0)
    thresh = st.slider("Threshold", 0, 255, 127)
    blur_k = st.slider("Blur Kernel", 1, 15, 1)

    # ===== GEOMETRI =====
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

    # ===== BRIGHTNESS & CONTRAST =====
    proc = cv2.convertScaleAbs(proc_img, alpha=contrast, beta=brightness)

    # ===== GRAYSCALE =====
    gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)

    # ===== BLUR =====
    if blur_k > 1:
        if blur_k % 2 == 0:
            blur_k += 1
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # ===== THRESHOLD =====
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # ===== OCR =====
    text = pytesseract.image_to_string(binary)

    # ===== KLASIFIKASI =====
    result, score = classify_protein(text)

    # ===== OUTPUT =====
    st.subheader("Hasil Processing")
    col1, col2 = st.columns(2)

    with col1:
        st.image(proc, caption="Processed Image", use_column_width=True)

    with col2:
        st.image(binary, caption="Binary (OCR Input)", use_column_width=True)

    st.subheader("Hasil OCR")
    st.text(text[:500])

    st.subheader("Hasil Klasifikasi")
    st.write("Kategori:", result)
    st.write("Score:", score)