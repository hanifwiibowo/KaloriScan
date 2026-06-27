import streamlit as st
import numpy as np
import random
import os
import time
from PIL import Image

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="KaloriScan — Klasifikasi Makanan & Estimasi Kalori",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load TensorFlow ─────────────────────────────────────────
PRIORITY_MODELS = ["SimpleCNN", "MobileNetV2", "EfficientNetB0"]

@st.cache_resource
def load_models():
    try:
        import tensorflow as tf
        models = {}
        model_files = {
            "SimpleCNN"     : "models/model_SimpleCNN.keras",
            "MobileNetV2"   : "models/model_MobileNetV2.keras",
            "EfficientNetB0": "models/model_EfficientNetB0.keras",
        }
        for name, path in model_files.items():
            if os.path.exists(path):
                models[name] = tf.keras.models.load_model(path)
        return models, tf
    except Exception:
        return {}, None

# ── Konfigurasi kelas & nutrisi ─────────────────────────────
CLASS_NAMES = [
    "ayam_goreng", "burger", "french_fries", "gado_gado",
    "ikan_goreng", "mie_goreng", "nasi_goreng", "nasi_padang",
    "pizza", "rawon", "rendang", "sate", "soto_ayam"
]

FOOD_EMOJI = {
    "ayam_goreng" : "🍗",
    "burger"      : "🍔",
    "french_fries": "🍟",
    "gado_gado"   : "🥗",
    "ikan_goreng" : "🐟",
    "mie_goreng"  : "🍜",
    "nasi_goreng" : "🍳",
    "nasi_padang" : "🍛",
    "pizza"       : "🍕",
    "rawon"       : "🥣",
    "rendang"     : "🥩",
    "sate"        : "🍢",
    "soto_ayam"   : "🍲",
}

DISPLAY_NAMES = {
    "ayam_goreng" : "Ayam Goreng",
    "burger"      : "Burger",
    "french_fries": "French Fries",
    "gado_gado"   : "Gado-Gado",
    "ikan_goreng" : "Ikan Goreng",
    "mie_goreng"  : "Mie Goreng",
    "nasi_goreng" : "Nasi Goreng",
    "nasi_padang" : "Nasi Padang",
    "pizza"       : "Pizza",
    "rawon"       : "Rawon",
    "rendang"     : "Rendang",
    "sate"        : "Sate",
    "soto_ayam"   : "Soto Ayam",
}

CALORIE_DB = {
    "ayam_goreng" : {"porsi_g": 100, "karbo": [0,   5],  "protein": [22, 28], "lemak": [12, 18]},
    "burger"      : {"porsi_g": 150, "karbo": [25,  35], "protein": [12, 18], "lemak": [10, 15]},
    "french_fries": {"porsi_g": 100, "karbo": [37,  45], "protein": [3,   5], "lemak": [12, 18]},
    "gado_gado"   : {"porsi_g": 300, "karbo": [30,  40], "protein": [10, 14], "lemak": [16, 24]},
    "ikan_goreng" : {"porsi_g": 100, "karbo": [0,   4],  "protein": [19, 25], "lemak": [8,  13]},
    "mie_goreng"  : {"porsi_g": 300, "karbo": [54,  66], "protein": [10, 14], "lemak": [12, 18]},
    "nasi_goreng" : {"porsi_g": 300, "karbo": [49,  61], "protein": [8,  12], "lemak": [11, 17]},
    "nasi_padang" : {"porsi_g": 400, "karbo": [62,  78], "protein": [21, 29], "lemak": [25, 35]},
    "pizza"       : {"porsi_g": 150, "karbo": [30,  40], "protein": [10, 14], "lemak": [11, 17]},
    "rawon"       : {"porsi_g": 300, "karbo": [40,  50], "protein": [15, 21], "lemak": [12, 18]},
    "rendang"     : {"porsi_g": 100, "karbo": [3,   7],  "protein": [17, 23], "lemak": [24, 32]},
    "sate"        : {"porsi_g": 150, "karbo": [3,   7],  "protein": [22, 28], "lemak": [15, 21]},
    "soto_ayam"   : {"porsi_g": 300, "karbo": [30,  40], "protein": [12, 18], "lemak": [6,  10]},
}

IMG_SIZE = {"SimpleCNN": (128, 128), "MobileNetV2": (128, 128), "EfficientNetB0": (224, 224)}

# ── Helper functions ─────────────────────────────────────────
def estimate_nutrition(food_class):
    data = CALORIE_DB.get(food_class)
    if not data:
        return None
    karbo   = random.randint(data["karbo"][0],   data["karbo"][1])
    protein = random.randint(data["protein"][0],  data["protein"][1])
    lemak   = random.randint(data["lemak"][0],    data["lemak"][1])
    kalori  = (karbo * 4) + (protein * 4) + (lemak * 9)
    kal_min = data["karbo"][0]*4 + data["protein"][0]*4 + data["lemak"][0]*9
    kal_max = data["karbo"][1]*4 + data["protein"][1]*4 + data["lemak"][1]*9
    return {
        "food": food_class, "porsi_g": data["porsi_g"],
        "karbo": karbo, "protein": protein, "lemak": lemak,
        "kalori": kalori, "kal_min": kal_min, "kal_max": kal_max,
    }

def preprocess_image(img: Image.Image, size: tuple):
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, img_array):
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds[0]))
    return CLASS_NAMES[idx], float(preds[0][idx]), preds[0]

def confidence_badge(conf):
    if conf >= 0.75:
        return "Yakin ✓", "#16a34a", "#dcfce7"
    elif conf >= 0.5:
        return "Cukup Yakin", "#d97706", "#fef3c7"
    else:
        return "Kurang Yakin", "#dc2626", "#fee2e2"

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%);
    border-right: 1px solid #d1fae5;
}
[data-testid="stSidebar"] * { color: #1a3a2a !important; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #15803d !important; }

/* ─── Main background ─── */
.stApp {
    background: linear-gradient(135deg, #f0fdf4 0%, #fefce8 50%, #f0fdf4 100%);
}

/* ─── Hide default Streamlit header ─── */
header[data-testid="stHeader"] { background: transparent; }

/* ─── Hero banner ─── */
.ks-hero {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 40%, #166534 100%);
    border-radius: 24px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(22,163,74,0.25);
}
.ks-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
}
.ks-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(253,224,71,0.12);
}
.ks-hero-logo {
    font-size: 2.2rem;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.ks-hero h1 {
    font-size: 2.4rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 0.2rem 0;
    letter-spacing: -1px;
}
.ks-hero h1 span {
    color: #fde047;
}
.ks-hero-sub {
    color: #bbf7d0;
    font-size: 1rem;
    font-weight: 500;
    margin: 0;
}
.ks-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(253,224,71,0.2);
    border: 1px solid rgba(253,224,71,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #fde047;
    font-weight: 600;
    margin-bottom: 0.9rem;
}

/* ─── Demo/mode notice ─── */
.ks-notice {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #92400e;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ─── Upload zone ─── */
[data-testid="stFileUploader"] {
    border: 2px dashed #86efac !important;
    border-radius: 18px !important;
    background: rgba(240,253,244,0.8) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #16a34a !important;
    background: rgba(220,252,231,0.9) !important;
}

/* ─── Section label ─── */
.ks-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin-bottom: 0.6rem;
}

/* ─── Result card ─── */
.ks-result-card {
    background: #ffffff;
    border: 1px solid #d1fae5;
    border-radius: 20px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 4px 20px rgba(22,163,74,0.08);
    margin-bottom: 1rem;
}
.ks-result-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
}
.ks-conf-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 700;
}
.ks-food-emoji { font-size: 2.8rem; line-height: 1; margin-bottom: 0.3rem; display: block; }
.ks-food-name {
    font-size: 1.8rem;
    font-weight: 800;
    color: #14532d;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.ks-food-sub { font-size: 0.82rem; color: #6b7280; margin-top: 4px; }

/* ─── Calorie card ─── */
.ks-calorie-card {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 6px 24px rgba(22,163,74,0.2);
}
.ks-cal-main { color: #ffffff; }
.ks-cal-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #bbf7d0; }
.ks-cal-number { font-size: 2.8rem; font-weight: 900; line-height: 1; color: #ffffff; }
.ks-cal-unit { font-size: 1rem; font-weight: 600; color: #86efac; margin-left: 3px; }
.ks-cal-range { font-size: 0.75rem; color: #86efac; margin-top: 4px; }
.ks-cal-icon { font-size: 3rem; opacity: 0.3; }

/* ─── Macro pills ─── */
.ks-macro-row { display: flex; gap: 0.65rem; margin-top: 0.8rem; }
.ks-macro-pill {
    flex: 1;
    border-radius: 14px;
    padding: 0.9rem 0.6rem;
    text-align: center;
}
.ks-pill-karbo   { background: #eff6ff; border: 1.5px solid #bfdbfe; }
.ks-pill-protein { background: #f0fdf4; border: 1.5px solid #bbf7d0; }
.ks-pill-lemak   { background: #fffbeb; border: 1.5px solid #fde68a; }
.ks-macro-val { font-size: 1.2rem; font-weight: 800; display: block; }
.ks-macro-name { font-size: 0.7rem; font-weight: 600; margin-top: 1px; text-transform: uppercase; letter-spacing: 0.05em; }
.ks-karbo-val   { color: #1d4ed8; }
.ks-protein-val { color: #15803d; }
.ks-lemak-val   { color: #b45309; }
.ks-karbo-name   { color: #3b82f6; }
.ks-protein-name { color: #16a34a; }
.ks-lemak-name   { color: #d97706; }

/* ─── Alt suggestion ─── */
.ks-alt-hint {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 0 12px 12px 0;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #92400e;
    margin: 0.8rem 0;
}

/* ─── Disclaimer ─── */
.ks-disclaimer {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    font-size: 0.74rem;
    color: #6b7280;
    margin-top: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 6px;
}

/* ─── Empty state ─── */
.ks-empty {
    text-align: center;
    padding: 3rem 1rem;
    color: #9ca3af;
}
.ks-empty-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.ks-empty-title { font-size: 0.9rem; font-weight: 600; color: #6b7280; }
.ks-empty-sub { font-size: 0.78rem; margin-top: 0.3rem; }

/* ─── History card ─── */
.ks-hist-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.9rem;
    transition: border-color 0.2s;
}
.ks-hist-card:hover { border-color: #86efac; }
.ks-hist-emoji { font-size: 1.6rem; }
.ks-hist-food { font-weight: 700; color: #14532d; font-size: 0.92rem; }
.ks-hist-meta { color: #6b7280; font-size: 0.75rem; margin-top: 2px; }
.ks-hist-cal { font-weight: 800; color: #16a34a; font-size: 1rem; margin-left: auto; }

/* ─── Total summary banner ─── */
.ks-total-banner {
    background: linear-gradient(135deg, #ecfdf5, #fefce8);
    border: 1px solid #a7f3d0;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    margin-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ks-total-label { color: #6b7280; font-size: 0.82rem; font-weight: 500; }
.ks-total-count { color: #374151; font-size: 0.82rem; }
.ks-total-kal { color: #15803d; font-weight: 800; font-size: 1.15rem; }

/* ─── Streamlit buttons ─── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 14px rgba(22,163,74,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(22,163,74,0.4) !important;
}
.stButton > button:not([kind="primary"]) {
    border: 1.5px solid #d1fae5 !important;
    border-radius: 10px !important;
    color: #15803d !important;
    font-weight: 600 !important;
    background: #ffffff !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #6b7280;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
}

/* ─── Selectbox ─── */
.stSelectbox label { font-size: 0.82rem; font-weight: 600; color: #374151; }

/* ─── Sidebar items ─── */
.ks-sidebar-logo {
    text-align: center;
    padding: 1rem 0 0.8rem;
}
.ks-sidebar-logo .logo-icon { font-size: 2.4rem; }
.ks-sidebar-logo .logo-name {
    font-size: 1.4rem;
    font-weight: 900;
    color: #15803d !important;
    letter-spacing: -0.5px;
    margin-top: 0.2rem;
}
.ks-sidebar-logo .logo-sub {
    font-size: 0.72rem;
    color: #6b7280 !important;
    font-weight: 500;
}
.ks-step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.4rem 0;
}
.ks-step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #16a34a;
    color: white !important;
    font-size: 0.7rem;
    font-weight: 800;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.ks-step-text { font-size: 0.82rem; color: #374151 !important; line-height: 1.4; }
.ks-food-chip {
    display: inline-block;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.74rem;
    color: #15803d !important;
    margin: 3px 2px;
    font-weight: 500;
}
.ks-src-note {
    font-size: 0.7rem;
    color: #9ca3af !important;
    text-align: center;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='ks-sidebar-logo'>
        <div class='logo-icon'>🥗</div>
        <div class='logo-name'>KaloriScan</div>
        <div class='logo-sub'>Klasifikasi Makanan & Estimasi Kalori</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem;font-weight:700;color:#374151;margin-bottom:0.5rem;'>Cara Pakai</div>", unsafe_allow_html=True)
    steps = [
        ("1", "Upload foto makanan kamu"),
        ("2", "Tekan <b>Analisis Sekarang</b>"),
        ("3", "Lihat estimasi kalori & gizinya"),
        ("4", "Koreksi jika kurang tepat"),
    ]
    for num, text in steps:
        st.markdown(f"""
        <div class='ks-step-item'>
            <div class='ks-step-num'>{num}</div>
            <div class='ks-step-text'>{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🍱 13 Makanan yang Didukung"):
        chips = "".join([f"<span class='ks-food-chip'>{n}</span>" for n in DISPLAY_NAMES.values()])
        st.markdown(f"<div style='line-height:2;'>{chips}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='ks-src-note'>
        Data nutrisi mengacu pada<br>
        <b>TKPI — Kemenkes RI</b><br><br>
        Nilai kalori bersifat <i>estimasi</i>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class='ks-hero'>
    <div class='ks-hero-badge'>✦ Dashboard Klasifikasi Makanan</div>
    <div class='ks-hero-logo'>🥗</div>
    <h1>Kalori<span>Scan</span></h1>
    <p class='ks-hero-sub'>Foto makananmu → kenali jenisnya → estimasi kalori & gizi secara otomatis</p>
</div>
""", unsafe_allow_html=True)

# Load models
models, tf_module = load_models()
active_model = next((m for m in PRIORITY_MODELS if m in models), None)
demo_mode = active_model is None

if demo_mode:
    st.markdown("""
    <div class='ks-notice'>
        🔧 <span><b>Mode Demo Aktif</b> — Model AI belum terhubung. Prediksi bersifat simulasi acak untuk menampilkan tampilan.</span>
    </div>
    """, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "current_upload_id" not in st.session_state:
    st.session_state.current_upload_id = None

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸  Cek Kalori", "📋  Riwayat Hari Ini"])

# ── Tab 1: Cek Kalori ─────────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("<div class='ks-section-label'>Upload Foto Makanan</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Pilih atau drag & drop gambar",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            upload_id = f"{uploaded.name}-{uploaded.size}"
            if st.session_state.current_upload_id != upload_id:
                st.session_state.current_upload_id = upload_id
                st.session_state.result = None

            img = Image.open(uploaded)
            st.image(img, caption="Foto yang kamu upload", use_container_width=True)
            analyze_btn = st.button("🔍 Analisis Sekarang", use_container_width=True, type="primary")
        else:
            st.markdown("""
            <div class='ks-empty' style='border: 2px dashed #86efac; border-radius: 18px; background: rgba(240,253,244,0.5);'>
                <div class='ks-empty-icon'>📷</div>
                <div class='ks-empty-title'>Upload foto makananmu</div>
                <div class='ks-empty-sub'>Format: JPG, PNG, WEBP</div>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

    with col_result:
        st.markdown("<div class='ks-section-label'>Hasil Analisis</div>", unsafe_allow_html=True)

        if uploaded and analyze_btn:
            with st.spinner("Menganalisis foto makanan..."):
                time.sleep(0.4)
                if not demo_mode:
                    size = IMG_SIZE[active_model]
                    arr  = preprocess_image(img, size)
                    food_class, confidence, all_preds = predict(models[active_model], arr)
                    alt_idx   = np.argsort(all_preds)[::-1][1]
                    alt_class = CLASS_NAMES[alt_idx]
                else:
                    food_class = random.choice(CLASS_NAMES)
                    confidence = random.uniform(0.55, 0.97)
                    alt_class  = random.choice([c for c in CLASS_NAMES if c != food_class])

            st.session_state.result = {
                "food": food_class, "confidence": confidence,
                "alt": alt_class, "manual": False,
            }
            nutrition = estimate_nutrition(food_class)
            st.session_state.history.insert(0, {
                "display" : DISPLAY_NAMES[food_class],
                "emoji"   : FOOD_EMOJI.get(food_class, "🍽️"),
                "kalori"  : nutrition["kalori"] if nutrition else "-",
                "kal_min" : nutrition["kal_min"] if nutrition else "-",
                "kal_max" : nutrition["kal_max"] if nutrition else "-",
                "porsi_g" : nutrition["porsi_g"] if nutrition else "-",
            })

        result = st.session_state.result

        if result:
            food_class = result["food"]
            confidence = result["confidence"]
            manual     = result.get("manual", False)
            nutrition  = estimate_nutrition(food_class)
            emoji      = FOOD_EMOJI.get(food_class, "🍽️")

            if manual:
                label, color, bg = "Dipilih Manual", "#374151", "#f3f4f6"
            else:
                label, color, bg = confidence_badge(confidence)

            pct = f"{confidence*100:.0f}%" if not manual else ""

            st.markdown(f"""
            <div class='ks-result-card'>
                <div class='ks-result-top'>
                    <span class='ks-conf-badge' style='color:{color};background:{bg};'>{label} {pct}</span>
                </div>
                <span class='ks-food-emoji'>{emoji}</span>
                <div class='ks-food-name'>{DISPLAY_NAMES[food_class]}</div>
                <div class='ks-food-sub'>Terdeteksi oleh KaloriScan AI</div>
            </div>
            """, unsafe_allow_html=True)

            if nutrition:
                st.markdown(f"""
                <div class='ks-calorie-card'>
                    <div class='ks-cal-main'>
                        <div class='ks-cal-label'>Estimasi Kalori · {nutrition['porsi_g']}g per porsi</div>
                        <div>
                            <span class='ks-cal-number'>~{nutrition['kalori']}</span>
                            <span class='ks-cal-unit'>kkal</span>
                        </div>
                        <div class='ks-cal-range'>Rentang: {nutrition['kal_min']} – {nutrition['kal_max']} kkal</div>
                    </div>
                    <div class='ks-cal-icon'>🔥</div>
                </div>
                <div class='ks-macro-row'>
                    <div class='ks-macro-pill ks-pill-karbo'>
                        <span class='ks-macro-val ks-karbo-val'>~{nutrition['karbo']}g</span>
                        <span class='ks-macro-name ks-karbo-name'>Karbo</span>
                    </div>
                    <div class='ks-macro-pill ks-pill-protein'>
                        <span class='ks-macro-val ks-protein-val'>~{nutrition['protein']}g</span>
                        <span class='ks-macro-name ks-protein-name'>Protein</span>
                    </div>
                    <div class='ks-macro-pill ks-pill-lemak'>
                        <span class='ks-macro-val ks-lemak-val'>~{nutrition['lemak']}g</span>
                        <span class='ks-macro-name ks-lemak-name'>Lemak</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if not manual and result.get("alt") and confidence < 0.55:
                alt_display = DISPLAY_NAMES.get(result["alt"], result["alt"])
                st.markdown(f"""
                <div class='ks-alt-hint'>
                    🤔 Kurang yakin — mungkin juga <b>{alt_display}</b>? Koreksi di bawah jika perlu.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            override_options = ["✅ Sudah benar, lanjutkan"] + [
                n for k, n in DISPLAY_NAMES.items() if k != food_class
            ]
            picked = st.selectbox(
                "Bukan ini? Pilih yang benar:",
                override_options,
                key=f"override_select_{food_class}_{manual}"
            )
            if picked != "✅ Sudah benar, lanjutkan":
                reverse = {v: k for k, v in DISPLAY_NAMES.items()}
                new_class = reverse[picked]
                st.session_state.result = {
                    "food": new_class, "confidence": 1.0,
                    "alt": None, "manual": True,
                }
                if st.session_state.history:
                    new_nutri = estimate_nutrition(new_class)
                    st.session_state.history[0].update({
                        "display": DISPLAY_NAMES[new_class],
                        "emoji"  : FOOD_EMOJI.get(new_class, "🍽️"),
                        "kalori" : new_nutri["kalori"],
                        "kal_min": new_nutri["kal_min"],
                        "kal_max": new_nutri["kal_max"],
                        "porsi_g": new_nutri["porsi_g"],
                    })
                st.rerun()

            st.markdown("""
            <div class='ks-disclaimer'>
                <span>ℹ️</span>
                <span>Nilai kalori & nutrisi adalah <b>estimasi</b> berdasarkan rentang umum per porsi (TKPI Kemenkes RI). Nilai aktual bervariasi tergantung resep, ukuran porsi, dan cara memasak.</span>
            </div>
            """, unsafe_allow_html=True)

        elif uploaded:
            st.markdown("""
            <div class='ks-empty'>
                <div class='ks-empty-icon'>👆</div>
                <div class='ks-empty-title'>Tekan "Analisis Sekarang"</div>
                <div class='ks-empty-sub'>untuk melihat estimasi kalori & gizi</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='ks-empty'>
                <div class='ks-empty-icon'>🥗</div>
                <div class='ks-empty-title'>Hasil analisis muncul di sini</div>
                <div class='ks-empty-sub'>Upload foto makanan terlebih dahulu</div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Riwayat ───────────────────────────────────────────
with tab2:
    if not st.session_state.history:
        st.markdown("""
        <div class='ks-empty' style='padding: 4rem 1rem;'>
            <div class='ks-empty-icon'>📭</div>
            <div class='ks-empty-title'>Riwayat masih kosong</div>
            <div class='ks-empty-sub'>Yuk cek makananmu sekarang!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_title, col_clear = st.columns([4, 1])
        with col_title:
            st.markdown("<div class='ks-section-label'>Semua makanan yang dicek hari ini</div>", unsafe_allow_html=True)
        with col_clear:
            if st.button("🗑️ Hapus Semua", use_container_width=True):
                st.session_state.history = []
                st.session_state.result = None
                st.rerun()

        total_kal = sum(h["kalori"] for h in st.session_state.history if isinstance(h["kalori"], int))
        count = len(st.session_state.history)
        st.markdown(f"""
        <div class='ks-total-banner'>
            <div>
                <div class='ks-total-label'>Total kalori hari ini</div>
                <div class='ks-total-count'>{count} item dianalisis</div>
            </div>
            <div class='ks-total-kal'>~{total_kal} kkal</div>
        </div>
        """, unsafe_allow_html=True)

        for h in st.session_state.history:
            emoji = h.get("emoji", "🍽️")
            st.markdown(f"""
            <div class='ks-hist-card'>
                <div class='ks-hist-emoji'>{emoji}</div>
                <div>
                    <div class='ks-hist-food'>{h['display']}</div>
                    <div class='ks-hist-meta'>Porsi {h['porsi_g']}g &nbsp;·&nbsp; rentang {h['kal_min']}–{h['kal_max']} kkal</div>
                </div>
                <div class='ks-hist-cal'>~{h['kalori']} kkal</div>
            </div>
            """, unsafe_allow_html=True)
