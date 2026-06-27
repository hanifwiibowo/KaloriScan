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

# ── Data ─────────────────────────────────────────────────────
CLASS_NAMES = [
    "ayam_goreng", "burger", "french_fries", "gado_gado",
    "ikan_goreng", "mie_goreng", "nasi_goreng", "nasi_padang",
    "pizza", "rawon", "rendang", "sate", "soto_ayam"
]

FOOD_EMOJI = {
    "ayam_goreng":"🍗","burger":"🍔","french_fries":"🍟","gado_gado":"🥗",
    "ikan_goreng":"🐟","mie_goreng":"🍜","nasi_goreng":"🍳","nasi_padang":"🍛",
    "pizza":"🍕","rawon":"🥣","rendang":"🥩","sate":"🍢","soto_ayam":"🍲",
}

DISPLAY_NAMES = {
    "ayam_goreng":"Ayam Goreng","burger":"Burger","french_fries":"French Fries",
    "gado_gado":"Gado-Gado","ikan_goreng":"Ikan Goreng","mie_goreng":"Mie Goreng",
    "nasi_goreng":"Nasi Goreng","nasi_padang":"Nasi Padang","pizza":"Pizza",
    "rawon":"Rawon","rendang":"Rendang","sate":"Sate","soto_ayam":"Soto Ayam",
}

CALORIE_DB = {
    "ayam_goreng" : {"porsi_g":100,"karbo":[0,5],"protein":[22,28],"lemak":[12,18]},
    "burger"      : {"porsi_g":150,"karbo":[25,35],"protein":[12,18],"lemak":[10,15]},
    "french_fries": {"porsi_g":100,"karbo":[37,45],"protein":[3,5],"lemak":[12,18]},
    "gado_gado"   : {"porsi_g":300,"karbo":[30,40],"protein":[10,14],"lemak":[16,24]},
    "ikan_goreng" : {"porsi_g":100,"karbo":[0,4],"protein":[19,25],"lemak":[8,13]},
    "mie_goreng"  : {"porsi_g":300,"karbo":[54,66],"protein":[10,14],"lemak":[12,18]},
    "nasi_goreng" : {"porsi_g":300,"karbo":[49,61],"protein":[8,12],"lemak":[11,17]},
    "nasi_padang" : {"porsi_g":400,"karbo":[62,78],"protein":[21,29],"lemak":[25,35]},
    "pizza"       : {"porsi_g":150,"karbo":[30,40],"protein":[10,14],"lemak":[11,17]},
    "rawon"       : {"porsi_g":300,"karbo":[40,50],"protein":[15,21],"lemak":[12,18]},
    "rendang"     : {"porsi_g":100,"karbo":[3,7],"protein":[17,23],"lemak":[24,32]},
    "sate"        : {"porsi_g":150,"karbo":[3,7],"protein":[22,28],"lemak":[15,21]},
    "soto_ayam"   : {"porsi_g":300,"karbo":[30,40],"protein":[12,18],"lemak":[6,10]},
}

IMG_SIZE = {"SimpleCNN":(128,128),"MobileNetV2":(128,128),"EfficientNetB0":(224,224)}

# FIX: ambang batas tegas — di bawah ini, gambar dianggap BUKAN makanan
# dan hasil prediksi TIDAK ditampilkan sebagai hasil yang valid.
NON_FOOD_THRESHOLD = 0.35

# ── Helpers ──────────────────────────────────────────────────
def estimate_nutrition(food_class):
    data = CALORIE_DB.get(food_class)
    if not data: return None
    # FIX: pakai titik tengah rentang, BUKAN random.randint.
    # Sebelumnya nilai di-random ulang setiap kali fungsi ini dipanggil,
    # jadi makanan yang sama bisa dapat kalori berbeda tiap analisis.
    karbo   = round(sum(data["karbo"]) / 2)
    protein = round(sum(data["protein"]) / 2)
    lemak   = round(sum(data["lemak"]) / 2)
    kalori  = karbo*4 + protein*4 + lemak*9
    kal_min = data["karbo"][0]*4 + data["protein"][0]*4 + data["lemak"][0]*9
    kal_max = data["karbo"][1]*4 + data["protein"][1]*4 + data["lemak"][1]*9
    return {"food":food_class,"porsi_g":data["porsi_g"],
            "karbo":karbo,"protein":protein,"lemak":lemak,
            "kalori":kalori,"kal_min":kal_min,"kal_max":kal_max}

def preprocess_image(img, size):
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32)/255.0
    return np.expand_dims(arr, axis=0)

def predict(model, img_array):
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds[0]))
    return CLASS_NAMES[idx], float(preds[0][idx]), preds[0]

def confidence_badge(conf):
    if conf >= 0.75: return "Yakin ✓","#16a34a","#dcfce7"
    elif conf >= 0.5: return "Cukup Yakin","#d97706","#fef3c7"
    else: return "Kurang Yakin","#dc2626","#fee2e2"

def hitung_bmr(bb, tb, usia, gender):
    if gender == "Laki-laki":
        return 88.362 + (13.397*bb) + (4.799*tb) - (5.677*usia)
    else:
        return 447.593 + (9.247*bb) + (3.098*tb) - (4.330*usia)

def hitung_tdee(bmr, aktivitas):
    faktor = {"Tidak aktif (kerja kantoran)":1.2,"Ringan (olahraga 1-3x/minggu)":1.375,
              "Sedang (olahraga 3-5x/minggu)":1.55,"Aktif (olahraga 6-7x/minggu)":1.725,
              "Sangat aktif (atlet/kerja fisik berat)":1.9}
    return bmr * faktor.get(aktivitas, 1.2)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#ffffff 0%,#f0fdf4 100%);
    border-right: 1px solid #d1fae5;
}
[data-testid="stSidebar"] * { color: #1a3a2a !important; }
.stApp { background: linear-gradient(135deg,#f0fdf4 0%,#fefce8 50%,#f0fdf4 100%); }
header[data-testid="stHeader"] { background: transparent; }

.ks-hero {
    background: linear-gradient(135deg,#16a34a 0%,#15803d 40%,#166534 100%);
    border-radius: 24px; padding: 2.2rem 2.8rem; margin-bottom: 1.6rem;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(22,163,74,0.25);
}
.ks-hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:200px; height:200px; border-radius:50%;
    background:rgba(255,255,255,0.07);
}
.ks-hero::after {
    content:''; position:absolute; bottom:-40px; left:30%;
    width:140px; height:140px; border-radius:50%;
    background:rgba(253,224,71,0.12);
}
.ks-hero-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(253,224,71,0.2); border:1px solid rgba(253,224,71,0.4);
    border-radius:20px; padding:4px 14px; font-size:0.75rem;
    color:#fde047; font-weight:600; margin-bottom:0.9rem;
}
.ks-hero-logo { font-size:2.2rem; line-height:1; margin-bottom:0.5rem; }
.ks-hero h1 { font-size:2.4rem; font-weight:900; color:#ffffff; margin:0 0 0.2rem 0; letter-spacing:-1px; }
.ks-hero h1 span { color:#fde047; }
.ks-hero-sub { color:#bbf7d0; font-size:1rem; font-weight:500; margin:0; }

.ks-notice {
    background:#fefce8; border:1px solid #fde68a; border-left:4px solid #f59e0b;
    border-radius:10px; padding:0.75rem 1rem; font-size:0.82rem; color:#92400e;
    margin-bottom:1rem; display:flex; align-items:center; gap:0.5rem;
}
.ks-section-label {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#374151; margin-bottom:0.6rem;
}

[data-testid="stFileUploader"] {
    border:2px dashed #86efac !important; border-radius:18px !important;
    background:rgba(240,253,244,0.8) !important;
}
[data-testid="stCameraInput"] { border:2px dashed #86efac !important; border-radius:18px !important; }
[data-testid="stCameraInput"] video { border-radius:14px !important; }

.ks-result-card {
    background:#ffffff; border:1px solid #d1fae5; border-radius:20px;
    padding:1.5rem 1.6rem; box-shadow:0 4px 20px rgba(22,163,74,0.08); margin-bottom:1rem;
}
.ks-conf-badge { display:inline-block; border-radius:20px; padding:4px 12px; font-size:0.72rem; font-weight:700; }
.ks-food-emoji { font-size:2.8rem; line-height:1; margin-bottom:0.3rem; display:block; }
.ks-food-name { font-size:1.8rem; font-weight:800; color:#14532d; letter-spacing:-0.5px; line-height:1.1; }
.ks-food-sub { font-size:0.82rem; color:#4b5563; margin-top:4px; }

.ks-calorie-card {
    background:linear-gradient(135deg,#16a34a 0%,#15803d 100%);
    border-radius:18px; padding:1.4rem 1.6rem; margin:1rem 0;
    display:flex; align-items:center; justify-content:space-between;
    box-shadow:0 6px 24px rgba(22,163,74,0.2);
}
.ks-cal-label { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#bbf7d0; }
.ks-cal-number { font-size:2.8rem; font-weight:900; line-height:1; color:#ffffff; }
.ks-cal-unit { font-size:1rem; font-weight:600; color:#86efac; margin-left:3px; }
.ks-cal-range { font-size:0.75rem; color:#86efac; margin-top:4px; }
.ks-cal-icon { font-size:3rem; opacity:0.3; }

.ks-macro-row { display:flex; gap:0.65rem; margin-top:0.8rem; }
.ks-macro-pill { flex:1; border-radius:14px; padding:0.9rem 0.6rem; text-align:center; }
.ks-pill-karbo   { background:#eff6ff; border:1.5px solid #bfdbfe; }
.ks-pill-protein { background:#f0fdf4; border:1.5px solid #bbf7d0; }
.ks-pill-lemak   { background:#fffbeb; border:1.5px solid #fde68a; }
.ks-macro-val { font-size:1.2rem; font-weight:800; display:block; }
.ks-macro-name { font-size:0.7rem; font-weight:600; margin-top:1px; text-transform:uppercase; letter-spacing:0.05em; }
.ks-karbo-val{color:#1d4ed8;} .ks-protein-val{color:#15803d;} .ks-lemak-val{color:#b45309;}
.ks-karbo-name{color:#3b82f6;} .ks-protein-name{color:#16a34a;} .ks-lemak-name{color:#d97706;}

.ks-alt-hint {
    background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #f59e0b;
    border-radius:0 12px 12px 0; padding:0.65rem 1rem; font-size:0.82rem; color:#92400e; margin:0.8rem 0;
}
.ks-disclaimer {
    background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px;
    padding:0.65rem 0.9rem; font-size:0.74rem; color:#374151; margin-top:1rem;
    display:flex; align-items:flex-start; gap:6px;
}
.ks-empty { text-align:center; padding:3rem 1rem; color:#6b7280; }
.ks-empty-icon { font-size:3rem; margin-bottom:0.6rem; }
.ks-empty-title { font-size:0.9rem; font-weight:600; color:#374151; }
.ks-empty-sub { font-size:0.78rem; margin-top:0.3rem; }

.ks-hist-card {
    background:#ffffff; border:1px solid #e5e7eb; border-radius:14px;
    padding:0.85rem 1rem; margin-bottom:0.5rem;
    display:flex; align-items:center; gap:0.9rem;
}
.ks-hist-emoji { font-size:1.6rem; }
.ks-hist-food { font-weight:700; color:#14532d; font-size:0.92rem; }
.ks-hist-meta { color:#4b5563; font-size:0.75rem; margin-top:2px; }
.ks-hist-cal { font-weight:800; color:#16a34a; font-size:1rem; margin-left:auto; }

.ks-total-banner {
    background:linear-gradient(135deg,#ecfdf5,#fefce8);
    border:1px solid #a7f3d0; border-radius:14px; padding:1rem 1.4rem;
    margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center;
}
.ks-total-label { color:#374151; font-size:0.82rem; font-weight:500; }
.ks-total-count { color:#374151; font-size:0.82rem; }
.ks-total-kal { color:#15803d; font-weight:800; font-size:1.15rem; }

/* Target progress */
.ks-progress-wrap {
    background:#ffffff; border:1px solid #d1fae5; border-radius:16px;
    padding:1.2rem 1.4rem; margin-bottom:1rem;
}
.ks-progress-bar-bg {
    background:#e5e7eb; border-radius:99px; height:14px; overflow:hidden; margin:0.6rem 0;
}
.ks-progress-bar-fill {
    height:100%; border-radius:99px;
    transition:width 0.5s ease;
}

/* BMI card */
.ks-bmi-card {
    border-radius:16px; padding:1.4rem 1.6rem; text-align:center; margin-bottom:1rem;
}
.ks-bmi-number { font-size:3rem; font-weight:900; line-height:1; }
.ks-bmi-label { font-size:0.85rem; font-weight:600; margin-top:0.3rem; }
.ks-bmi-sub { font-size:0.75rem; color:#4b5563; margin-top:0.2rem; }

/* Sidebar */
.ks-sidebar-logo { text-align:center; padding:1rem 0 0.8rem; }
.ks-sidebar-logo .logo-icon { font-size:2.4rem; }
.ks-sidebar-logo .logo-name { font-size:1.4rem; font-weight:900; color:#15803d !important; letter-spacing:-0.5px; margin-top:0.2rem; }
.ks-sidebar-logo .logo-sub { font-size:0.72rem; color:#374151 !important; font-weight:500; }
.ks-step-item { display:flex; align-items:flex-start; gap:0.7rem; padding:0.4rem 0; }
.ks-step-num { width:22px; height:22px; border-radius:50%; background:#16a34a; color:white !important; font-size:0.7rem; font-weight:800; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }
.ks-step-text { font-size:0.82rem; color:#374151 !important; line-height:1.4; }
.ks-food-chip { display:inline-block; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:20px; padding:3px 10px; font-size:0.74rem; color:#15803d !important; margin:3px 2px; font-weight:500; }

/* Streamlit buttons */
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#16a34a,#15803d) !important;
    border:none !important; color:white !important; font-weight:700 !important;
    border-radius:12px !important; padding:0.65rem 1.5rem !important;
    font-size:0.95rem !important; box-shadow:0 4px 14px rgba(22,163,74,0.3) !important;
}
.stButton > button:not([kind="primary"]) {
    border:1.5px solid #d1fae5 !important; border-radius:10px !important;
    color:#15803d !important; font-weight:600 !important; background:#ffffff !important;
}
.stTabs [data-baseweb="tab-list"] {
    background:#ffffff; border-radius:12px; padding:4px; gap:4px;
    border:1px solid #e5e7eb; margin-bottom:1rem;
}
.stTabs [data-baseweb="tab"] { border-radius:9px; font-weight:600; font-size:0.88rem; color:#6b7280; padding:0.5rem 1.2rem; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#16a34a,#15803d) !important; color:white !important; }
.stSelectbox label { font-size:0.82rem; font-weight:600; color:#374151; }

/* ─── Override dark input fields ─── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] > div > div {
    background-color: #ffffff !important;
    color: #1a3a2a !important;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 10px !important;
    outline: none !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: #16a34a !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.15) !important;
}
div[data-baseweb="select"] svg { color: #16a34a !important; }
div[data-baseweb="popover"] ul { background: #ffffff !important; }
div[data-baseweb="popover"] li { color: #1a3a2a !important; }
div[data-baseweb="popover"] li:hover { background: #f0fdf4 !important; }
[data-testid="stNumberInput"] button {
    background: #f0fdf4 !important;
    border-color: #d1fae5 !important;
    color: #16a34a !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] button:hover { background: #dcfce7 !important; }
.stSelectbox > label, [data-testid="stNumberInput"] > label {
    color: #374151 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for key, val in [("history",[]),("result",None),("current_upload_id",None),("input_mode","upload"),("target_kal",2000)]:
    if key not in st.session_state:
        st.session_state[key] = val

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
    for num, text in [("1","Upload atau foto makanan kamu"),("2","Tekan <b>Analisis Sekarang</b>"),("3","Lihat estimasi kalori & gizinya"),("4","Koreksi jika kurang tepat")]:
        st.markdown(f"<div class='ks-step-item'><div class='ks-step-num'>{num}</div><div class='ks-step-text'>{text}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("🍱 13 Makanan yang Didukung"):
        chips = "".join([f"<span class='ks-food-chip'>{n}</span>" for n in DISPLAY_NAMES.values()])
        st.markdown(f"<div style='line-height:2;'>{chips}</div>", unsafe_allow_html=True)

    # Target kalori di sidebar
    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem;font-weight:700;color:#374151;margin-bottom:0.5rem;'>🎯 Target Kalori Harian</div>", unsafe_allow_html=True)
    target_kal = st.number_input(
        "kkal/hari",
        min_value=1000, max_value=5000,
        value=st.session_state["target_kal"],
        step=50,
        label_visibility="collapsed",
    )
    st.session_state["target_kal"] = target_kal
    st.markdown("""
    <div style='background:#fefce8;border:1px solid #fde68a;border-radius:8px;padding:0.55rem 0.75rem;margin-top:0.4rem;font-size:0.74rem;color:#92400e;line-height:1.5;'>
        💡 Belum tahu target kalorimu?<br>
        Hitung dulu di tab <b>⚖️ Kalkulator BMI</b> — hasil TDEE bisa langsung dipakai sebagai target.
    </div>
    """, unsafe_allow_html=True)

    # Footer note pojok kiri bawah sidebar
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#374151; line-height:1.6;'>
        Data nutrisi mengacu pada<br>
        <b style='color:#111827;'>TKPI — Kemenkes RI</b><br>
        Nilai kalori bersifat <i style="color:#374151;">estimasi</i>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class='ks-hero'>
    <div class='ks-hero-badge'>✦ Dashboard Klasifikasi Makanan & Estimasi Kalori</div>
    <div class='ks-hero-logo'>🥗</div>
    <h1>Kalori<span>Scan</span></h1>
    <p class='ks-hero-sub'>Foto makananmu → kenali jenisnya → estimasi kalori & gizi secara otomatis</p>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
models, tf_module = load_models()
active_model = next((m for m in PRIORITY_MODELS if m in models), None)
demo_mode = active_model is None

if demo_mode:
    st.markdown("""
    <div class='ks-notice'>
        🔧 <span><b>Mode Demo Aktif</b> — Model AI belum terhubung. Prediksi bersifat simulasi acak.</span>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸  Cek Kalori", "📋  Riwayat & Grafik", "⚖️  Kalkulator BMI"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Cek Kalori
# ════════════════════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("<div class='ks-section-label'>Foto Makanan</div>", unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📁  Upload File", use_container_width=True,
                         type="primary" if st.session_state.input_mode=="upload" else "secondary"):
                st.session_state.input_mode = "upload"
                st.session_state.result = None
                st.rerun()
        with col_btn2:
            if st.button("📷  Foto Langsung", use_container_width=True,
                         type="primary" if st.session_state.input_mode=="camera" else "secondary"):
                st.session_state.input_mode = "camera"
                st.session_state.result = None
                st.rerun()

        img = None
        if st.session_state.input_mode == "upload":
            uploaded = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
            if uploaded:
                uid = f"{uploaded.name}-{uploaded.size}"
                if st.session_state.current_upload_id != uid:
                    st.session_state.current_upload_id = uid
                    st.session_state.result = None
                img = Image.open(uploaded)
                st.image(img, use_container_width=True)
                st.markdown("<div style='text-align:center;font-size:0.75rem;color:#374151;font-weight:500;margin-top:4px;'>📷 Foto yang kamu upload</div>", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='ks-empty' style='border:2px dashed #86efac;border-radius:18px;background:rgba(240,253,244,0.5);'>
                    <div class='ks-empty-icon'>📁</div>
                    <div class='ks-empty-title'>Upload foto makananmu</div>
                    <div class='ks-empty-sub'>Format: JPG, PNG, WEBP</div></div>""", unsafe_allow_html=True)
        else:
            cam = st.camera_input("Kamera", label_visibility="collapsed")
            if cam:
                cid = f"cam-{len(cam.getvalue())}"
                if st.session_state.current_upload_id != cid:
                    st.session_state.current_upload_id = cid
                    st.session_state.result = None
                img = Image.open(cam)
            else:
                st.markdown("""<div class='ks-empty' style='padding:1rem;'>
                    <div class='ks-empty-icon'>📷</div>
                    <div class='ks-empty-title'>Arahkan kamera ke makanan</div>
                    <div class='ks-empty-sub'>Pastikan pencahayaan cukup</div></div>""", unsafe_allow_html=True)

        analyze_btn = st.button("🔍 Analisis Sekarang", use_container_width=True, type="primary", disabled=(img is None))

    with col_result:
        st.markdown("<div class='ks-section-label'>Hasil Analisis</div>", unsafe_allow_html=True)

        if img and analyze_btn:
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

            nutrition = estimate_nutrition(food_class)

            # FIX: kalau confidence di bawah ambang & bukan demo mode,
            # anggap gambar BUKAN makanan — jangan tampilkan hasil prediksi
            # yang menyesatkan (mis. screenshot kebaca "Rawon" pede 88%).
            is_rejected = (not demo_mode) and (confidence < NON_FOOD_THRESHOLD)

            if is_rejected:
                st.session_state.result = {
                    "food":None, "confidence":confidence, "alt":alt_class,
                    "manual":False, "nutrition":None, "rejected":True,
                }
            else:
                st.session_state.result = {"food":food_class,"confidence":confidence,"alt":alt_class,"manual":False,"nutrition":nutrition,"rejected":False}
                st.session_state.history.insert(0, {
                    "display":DISPLAY_NAMES[food_class], "emoji":FOOD_EMOJI.get(food_class,"🍽️"),
                    "kalori":nutrition["kalori"] if nutrition else "-",
                    "kal_min":nutrition["kal_min"] if nutrition else "-",
                    "kal_max":nutrition["kal_max"] if nutrition else "-",
                    "porsi_g":nutrition["porsi_g"] if nutrition else "-",
                    "karbo":nutrition["karbo"] if nutrition else 0,
                    "protein":nutrition["protein"] if nutrition else 0,
                    "lemak":nutrition["lemak"] if nutrition else 0,
                })

        result = st.session_state.result

        if result:
            food_class = result["food"]
            confidence = result["confidence"]
            manual     = result.get("manual", False)
            rejected   = result.get("rejected", False)

            if rejected:
                st.markdown(f"""
                <div class='ks-result-card' style='text-align:center;'>
                    <div style='font-size:2.4rem;'>🤷‍♂️</div>
                    <div style='font-weight:700;color:#92400e;font-size:1.05rem;margin-top:0.3rem;'>
                        Gambar tidak terdeteksi sebagai makanan
                    </div>
                    <div style='font-size:0.82rem;color:#6b7280;margin-top:0.3rem;'>
                        Confidence model hanya {confidence*100:.0f}% — coba upload foto makanan yang lebih jelas,
                        atau pilih manual di bawah jika ini memang makanan.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                override_options = ["— Pilih makanan secara manual —"] + list(DISPLAY_NAMES.values())
                picked = st.selectbox("Ini sebenarnya makanan apa?", override_options, key="override_rejected")
                if picked != "— Pilih makanan secara manual —":
                    reverse = {v:k for k,v in DISPLAY_NAMES.items()}
                    new_class = reverse[picked]
                    new_nutri = estimate_nutrition(new_class)
                    st.session_state.result = {"food":new_class,"confidence":1.0,"alt":None,"manual":True,"nutrition":new_nutri,"rejected":False}
                    st.session_state.history.insert(0, {
                        "display":DISPLAY_NAMES[new_class], "emoji":FOOD_EMOJI.get(new_class,"🍽️"),
                        "kalori":new_nutri["kalori"], "kal_min":new_nutri["kal_min"], "kal_max":new_nutri["kal_max"],
                        "porsi_g":new_nutri["porsi_g"], "karbo":new_nutri["karbo"], "protein":new_nutri["protein"], "lemak":new_nutri["lemak"],
                    })
                    st.rerun()

            elif food_class:
                nutrition  = result.get("nutrition") or estimate_nutrition(food_class)
                emoji      = FOOD_EMOJI.get(food_class,"🍽️")

                if manual: label,color,bg = "Dipilih Manual","#374151","#f3f4f6"
                else: label,color,bg = confidence_badge(confidence)
                pct = f"{confidence*100:.0f}%" if not manual else ""

                st.markdown(f"""
                <div class='ks-result-card'>
                    <div><span class='ks-conf-badge' style='color:{color};background:{bg};'>{label} {pct}</span></div>
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
                            <div><span class='ks-cal-number'>~{nutrition['kalori']}</span><span class='ks-cal-unit'>kkal</span></div>
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

                    # ── Donut chart makro ──
                    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='ks-section-label'>Distribusi Makronutrien</div>", unsafe_allow_html=True)
                    karbo_kal   = nutrition["karbo"]*4
                    protein_kal = nutrition["protein"]*4
                    lemak_kal   = nutrition["lemak"]*9
                    total_kal_makro = karbo_kal + protein_kal + lemak_kal

                    import plotly.graph_objects as go
                    fig_donut = go.Figure(go.Pie(
                        labels=["Karbohidrat","Protein","Lemak"],
                        values=[karbo_kal, protein_kal, lemak_kal],
                        hole=0.65,
                        marker_colors=["#3b82f6","#16a34a","#f59e0b"],
                        textinfo="label+percent",
                        textfont=dict(size=12, color="#1f2937"),
                        outsidetextfont=dict(size=12, color="#1f2937"),
                        insidetextfont=dict(size=12, color="#ffffff"),
                        hovertemplate="<b>%{label}</b><br>%{value} kkal<br>%{percent}<extra></extra>",
                    ))
                    fig_donut.add_annotation(text=f"<b>{nutrition['kalori']}</b><br>kkal", x=0.5, y=0.5,
                        font_size=16, showarrow=False, font_color="#14532d")
                    fig_donut.update_layout(
                        showlegend=True, margin=dict(t=10,b=80,l=10,r=10),
                        height=300, paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.28,
                            xanchor="center", x=0.5,
                            font=dict(size=13, color="#374151"),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="#d1fae5", borderwidth=1,
                        ),
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                if not manual and result.get("alt") and confidence < 0.55:
                    alt_display = DISPLAY_NAMES.get(result["alt"], result["alt"])
                    st.markdown(f"<div class='ks-alt-hint'>🤔 Kurang yakin — mungkin juga <b>{alt_display}</b>? Koreksi di bawah jika perlu.</div>", unsafe_allow_html=True)

                override_options = ["✅ Sudah benar, lanjutkan"] + [n for k,n in DISPLAY_NAMES.items() if k != food_class]
                picked = st.selectbox("Bukan ini? Pilih yang benar:", override_options, key=f"override_{food_class}_{manual}")
                if picked != "✅ Sudah benar, lanjutkan":
                    reverse = {v:k for k,v in DISPLAY_NAMES.items()}
                    new_class = reverse[picked]
                    new_nutri = estimate_nutrition(new_class)
                    st.session_state.result = {"food":new_class,"confidence":1.0,"alt":None,"manual":True,"nutrition":new_nutri}
                    if st.session_state.history:
                        st.session_state.history[0].update({
                            "display":DISPLAY_NAMES[new_class],"emoji":FOOD_EMOJI.get(new_class,"🍽️"),
                            "kalori":new_nutri["kalori"],"kal_min":new_nutri["kal_min"],"kal_max":new_nutri["kal_max"],
                            "porsi_g":new_nutri["porsi_g"],"karbo":new_nutri["karbo"],"protein":new_nutri["protein"],"lemak":new_nutri["lemak"],
                        })
                    st.rerun()

                st.markdown("<div class='ks-disclaimer'><span>ℹ️</span><span>Nilai kalori & nutrisi adalah <b>estimasi</b> berdasarkan rentang umum per porsi (TKPI Kemenkes RI). Nilai aktual bervariasi tergantung resep, ukuran porsi, dan cara memasak.</span></div>", unsafe_allow_html=True)

        elif img:
            st.markdown("<div class='ks-empty'><div class='ks-empty-icon'>👆</div><div class='ks-empty-title'>Tekan \"Analisis Sekarang\"</div><div class='ks-empty-sub'>untuk melihat estimasi kalori & gizi</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ks-empty'><div class='ks-empty-icon'>🥗</div><div class='ks-empty-title'>Hasil analisis muncul di sini</div><div class='ks-empty-sub'>Upload atau foto makanan terlebih dahulu</div></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — Riwayat & Grafik
# ════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.history:
        st.markdown("<div class='ks-empty' style='padding:4rem 1rem;'><div class='ks-empty-icon'>📭</div><div class='ks-empty-title'>Riwayat masih kosong</div><div class='ks-empty-sub'>Yuk cek makananmu sekarang!</div></div>", unsafe_allow_html=True)
    else:
        target_kal = st.session_state.get("target_kal", 2000)
        total_kal  = sum(h["kalori"] for h in st.session_state.history if isinstance(h["kalori"], int))
        count      = len(st.session_state.history)
        pct_target = min(total_kal / target_kal * 100, 100)
        sisa       = max(target_kal - total_kal, 0)

        if pct_target >= 100:
            bar_color = "#dc2626"
            status_text = "⚠️ Target kalori tercapai!"
        elif pct_target >= 75:
            bar_color = "#f59e0b"
            status_text = f"🔥 Mendekati target — sisa ~{sisa} kkal"
        else:
            bar_color = "#16a34a"
            status_text = f"✅ Masih aman — sisa ~{sisa} kkal"

        # ── Progress target ──
        st.markdown(f"""
        <div class='ks-progress-wrap'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='font-weight:700;color:#14532d;font-size:0.9rem;'>Target Kalori Harian</div>
                    <div style='font-size:0.78rem;color:#374151;margin-top:2px;'>{status_text}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:1.5rem;font-weight:900;color:#16a34a;'>~{total_kal}</div>
                    <div style='font-size:0.72rem;color:#4b5563;'>/ {target_kal} kkal</div>
                </div>
            </div>
            <div class='ks-progress-bar-bg'>
                <div class='ks-progress-bar-fill' style='width:{pct_target:.1f}%;background:{bar_color};'></div>
            </div>
            <div style='font-size:0.72rem;color:#4b5563;text-align:right;'>{pct_target:.0f}% dari target</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Grafik kalori per item ──
        import plotly.graph_objects as go

        labels  = [h["display"] for h in reversed(st.session_state.history)]
        kalvals = [h["kalori"] if isinstance(h["kalori"],int) else 0 for h in reversed(st.session_state.history)]
        emojis  = [h.get("emoji","🍽️") for h in reversed(st.session_state.history)]
        bar_colors = ["#dc2626" if v > target_kal*0.5 else "#16a34a" for v in kalvals]

        fig_bar = go.Figure(go.Bar(
            x=[f"{e} {l}" for e,l in zip(emojis,labels)],
            y=kalvals,
            marker_color=bar_colors,
            text=[f"{v} kkal" for v in kalvals],
            textposition="outside",
            textfont=dict(color="#111827", size=13),
            hovertemplate="<b>%{x}</b><br>%{y} kkal<extra></extra>",
        ))
        fig_bar.update_layout(
            title=dict(text="Kalori per Makanan", font_size=14, font_color="#14532d"),
            xaxis_tickangle=-20,
            margin=dict(t=40,b=60,l=40,r=20), height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="kkal", title_font=dict(color="#374151"), gridcolor="#e5e7eb", tickfont=dict(color="#374151")),
            xaxis=dict(tickfont=dict(color="#374151")),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Grafik makro akumulasi ──
        total_karbo   = sum(h.get("karbo",0) for h in st.session_state.history if isinstance(h.get("karbo"),int))
        total_protein = sum(h.get("protein",0) for h in st.session_state.history if isinstance(h.get("protein"),int))
        total_lemak   = sum(h.get("lemak",0) for h in st.session_state.history if isinstance(h.get("lemak"),int))

        fig_makro = go.Figure(go.Bar(
            x=["Karbohidrat","Protein","Lemak"],
            y=[total_karbo, total_protein, total_lemak],
            marker_color=["#3b82f6","#16a34a","#f59e0b"],
            text=[f"{total_karbo}g",f"{total_protein}g",f"{total_lemak}g"],
            textposition="outside",
            textfont=dict(color="#111827", size=13),
            hovertemplate="<b>%{x}</b><br>%{y}g<extra></extra>",
        ))
        fig_makro.update_layout(
            title=dict(text="Total Makronutrien Hari Ini", font_size=14, font_color="#14532d"),
            margin=dict(t=40,b=20,l=40,r=20), height=260,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="gram", title_font=dict(color="#374151"), gridcolor="#e5e7eb", tickfont=dict(color="#374151")),
            xaxis=dict(tickfont=dict(color="#374151")),
        )
        st.plotly_chart(fig_makro, use_container_width=True)

        # ── Daftar riwayat ──
        st.markdown("---")
        col_title, col_clear = st.columns([4,1])
        with col_title:
            st.markdown("<div class='ks-section-label'>Detail per item</div>", unsafe_allow_html=True)
        with col_clear:
            if st.button("🗑️ Hapus Semua", use_container_width=True):
                st.session_state.history = []
                st.session_state.result  = None
                st.rerun()

        for h in st.session_state.history:
            st.markdown(f"""
            <div class='ks-hist-card'>
                <div class='ks-hist-emoji'>{h.get('emoji','🍽️')}</div>
                <div>
                    <div class='ks-hist-food'>{h['display']}</div>
                    <div class='ks-hist-meta'>Porsi {h['porsi_g']}g &nbsp;·&nbsp; {h.get('karbo','-')}g karbo · {h.get('protein','-')}g protein · {h.get('lemak','-')}g lemak</div>
                </div>
                <div class='ks-hist-cal'>~{h['kalori']} kkal</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — Kalkulator BMI
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='ks-section-label'>Kalkulator BMI & Kebutuhan Kalori Harian</div>", unsafe_allow_html=True)

    col_form, col_bmi = st.columns([1,1], gap="large")

    with col_form:
        gender   = st.selectbox("Jenis Kelamin", ["Laki-laki","Perempuan"])
        usia     = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=22)
        bb       = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=300.0, value=65.0, step=0.5)
        tb       = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
        aktivitas = st.selectbox("Tingkat Aktivitas", [
            "Tidak aktif (kerja kantoran)",
            "Ringan (olahraga 1-3x/minggu)",
            "Sedang (olahraga 3-5x/minggu)",
            "Aktif (olahraga 6-7x/minggu)",
            "Sangat aktif (atlet/kerja fisik berat)",
        ])
        tujuan = st.selectbox("Tujuan", ["Turun berat badan (-500 kkal)","Jaga berat badan","Naik berat badan (+500 kkal)"])
        hitung_btn = st.button("⚖️ Hitung Sekarang", use_container_width=True, type="primary")

    with col_bmi:
        if hitung_btn:
            bmi  = bb / ((tb/100)**2)
            bmr  = hitung_bmr(bb, tb, usia, gender)
            tdee = hitung_tdee(bmr, aktivitas)

            if tujuan.startswith("Turun"):   kebutuhan = tdee - 500
            elif tujuan.startswith("Naik"):  kebutuhan = tdee + 500
            else:                            kebutuhan = tdee

            if bmi < 18.5:   bmi_kat,bmi_color,bmi_bg = "Kurus","#2563eb","#eff6ff"
            elif bmi < 25.0: bmi_kat,bmi_color,bmi_bg = "Normal","#16a34a","#f0fdf4"
            elif bmi < 30.0: bmi_kat,bmi_color,bmi_bg = "Overweight","#d97706","#fffbeb"
            else:             bmi_kat,bmi_color,bmi_bg = "Obesitas","#dc2626","#fef2f2"

            st.markdown(f"""
            <div class='ks-bmi-card' style='background:{bmi_bg};border:2px solid {bmi_color}33;'>
                <div class='ks-bmi-number' style='color:{bmi_color};'>{bmi:.1f}</div>
                <div class='ks-bmi-label' style='color:{bmi_color};'>{bmi_kat}</div>
                <div class='ks-bmi-sub'>Indeks Massa Tubuh (BMI)</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='ks-calorie-card' style='margin-top:0;'>
                <div class='ks-cal-main'>
                    <div class='ks-cal-label'>Kebutuhan Kalori Harian ({tujuan.split('(')[0].strip()})</div>
                    <div><span class='ks-cal-number'>{kebutuhan:.0f}</span><span class='ks-cal-unit'>kkal</span></div>
                    <div class='ks-cal-range'>BMR: {bmr:.0f} kkal &nbsp;·&nbsp; TDEE: {tdee:.0f} kkal</div>
                </div>
                <div class='ks-cal-icon'>🎯</div>
            </div>
            """, unsafe_allow_html=True)

            # Rekomendasi makro
            p_g  = round((kebutuhan * 0.30) / 4)
            k_g  = round((kebutuhan * 0.45) / 4)
            l_g  = round((kebutuhan * 0.25) / 9)
            st.markdown(f"""
            <div style='margin-top:0.5rem;'>
                <div class='ks-section-label'>Rekomendasi Makro Harian</div>
                <div class='ks-macro-row'>
                    <div class='ks-macro-pill ks-pill-karbo'>
                        <span class='ks-macro-val ks-karbo-val'>{k_g}g</span>
                        <span class='ks-macro-name ks-karbo-name'>Karbo</span>
                    </div>
                    <div class='ks-macro-pill ks-pill-protein'>
                        <span class='ks-macro-val ks-protein-val'>{p_g}g</span>
                        <span class='ks-macro-name ks-protein-name'>Protein</span>
                    </div>
                    <div class='ks-macro-pill ks-pill-lemak'>
                        <span class='ks-macro-val ks-lemak-val'>{l_g}g</span>
                        <span class='ks-macro-name ks-lemak-name'>Lemak</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Info insight — user input manual di sidebar
            st.markdown(f"""
            <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                        padding:0.75rem 1rem;margin-top:0.5rem;font-size:0.83rem;color:#14532d;line-height:1.6;'>
                💡 Kebutuhan kalori harianmu sekitar <b>{kebutuhan:.0f} kkal</b>. Masukkan angka ini secara manual ke kolom <b>🎯 Target Kalori Harian</b> di sidebar kiri.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='ks-empty'>
                <div class='ks-empty-icon'>⚖️</div>
                <div class='ks-empty-title'>Isi data di sebelah kiri</div>
                <div class='ks-empty-sub'>lalu tekan Hitung Sekarang</div>
            </div>
            """, unsafe_allow_html=True)
