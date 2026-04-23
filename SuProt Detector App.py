import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re

# ===== CONFIG =====
st.set_page_config(
    page_title="SuProt Detector Pro",
    layout="wide",
    page_icon="🥛",
    initial_sidebar_state="expanded"
)

# ===== MODERN STYLE (CSS) =====
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Card Style */
    .main-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Result Header */
    .result-header {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0;
    }
    
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid #38bdf8;
    }

    /* Sidebar adjustment */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    
    /* Hide default streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===== LOGIC FUNCTIONS =====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

def classify_protein(text):
    keywords = {
        "Whey Protein": ["whey", "isolate", "concentrate", "hydrolyzed"],
        "Casein": ["casein", "micellar", "caseinate"],
        "Plant Based": ["soy", "pea", "rice", "vegan", "hemp", "plant"],
        "Mass Gainer": ["mass gainer", "weight gainer", "maltodextrin", "high calorie"]
    }
    text = preprocess_text(text)
    score = {k:0 for k in keywords}

    for category, words in keywords.items():
        for word in words:
            if word in text:
                score[category] += 1

    # Heuristic boost for mass gainer
    if any(x in text for x in ["maltodextrin", "gainer"]):
        score["Mass Gainer"] += 2

    best_match = max(score, key=score.get)
    if score[best_match] == 0:
        return "Unknown", score
    return best_match, score

# ===== SIDEBAR SETTINGS =====
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050181.png", width=80)
    st.title("Control Panel")
    
    uploaded_file = st.file_uploader("Upload Label Produk", type=["jpg","png","jpeg"])
    
    with st.expander("🛠️ Manipulasi Geometri", expanded=False):
        operation = st.selectbox("Operasi", ["none", "translasi", "rotasi", "scaling"])
        tx, ty, angle, scale = 0, 0, 0, 1.0
        if operation == "translasi":
            tx = st.slider("Geser X", -100, 100, 0)
            ty = st.slider("Geser Y", -100, 100, 0)
        elif operation == "rotasi":
            angle = st.slider("Sudut Rotasi", -180, 180, 0)
        elif operation == "scaling":
            scale = st.slider("Zoom Factor", 1.0, 3.0, 1.2)

    with st.expander("🧪 Image Enhancement", expanded=True):
        brightness = st.slider("Brightness", -100, 100, 0)
        contrast = st.slider("Contrast", 0.5, 3.0, 1.0)
        thresh = st.slider("OCR Threshold", 0, 255, 130)
        blur_k = st.slider("Noise Reduction", 1, 11, 1, step=2)

# ===== MAIN UI =====
st.markdown('<p class="result-header">SuProt Detector Pro</p>', unsafe_allow_html=True)
st.markdown('<span class="status-badge">AI-Powered Protein Classifier</span>', unsafe_allow_html=True)
st.write("---")

if uploaded_file:
    # Load Image
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # Image Processing Logic
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

    # Adjust Brightness/Contrast
    proc = cv2.convertScaleAbs(proc_img, alpha=contrast, beta=brightness)
    gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
    
    if blur_k > 1:
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # OCR Process
    with st.spinner("Menganalisis teks komposisi..."):
        text = pytesseract.image_to_string(binary)
        result, scores = classify_protein(text)

    # LAYOUT COLUMNS
    tab1, tab2 = st.tabs(["📊 Analysis Result", "🔬 Technical Preview"])

    with tab1:
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.image(image, caption="Original Label", use_container_width=True)
        
        with c2:
            st.markdown(f"""
            <div class="main-card">
                <p style="color: #94a3b8; margin-bottom: 5px;">Hasil Deteksi Utama:</p>
                <h1 style="color: #38bdf8; margin: 0;">{result.upper()}</h1>
                <p style="font-size: 0.9rem; opacity: 0.8;">Berdasarkan ekstraksi kata kunci komposisi</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("### 🧬 Confidence Score")
            for cat, val in scores.items():
                col_name, col_bar = st.columns([1, 3])
                col_name.write(cat)
                # Normalize progress for UI (max 5)
                progress_val = min(val / 5, 1.0) if val > 0 else 0
                col_bar.progress(progress_val)

    with tab2:
        col_img, col_txt = st.columns(2)
        with col_img:
            st.write("#### Preprocessed Image (OCR View)")
            st.image(binary, use_container_width=True, clamp=True)
        with col_txt:
            st.write("#### Raw Extracted Text")
            st.text_area("OCR Output", text, height=300)

else:
    # Empty State
    st.markdown("""
    <div style="text-align: center; padding: 100px; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px;">
        <h2 style="color: #64748b;">Belum ada gambar yang diunggah</h2>
        <p style="color: #475569;">Gunakan sidebar sebelah kiri untuk mengunggah foto label komposisi susu protein Anda.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer info
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Built with Streamlit & OpenCV | SuProt Detector v2.0</p>", unsafe_allow_html=True)
