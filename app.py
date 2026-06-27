import streamlit as st
import numpy as np
import random
import json
import os
import time
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="NutriSnap ID",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load TensorFlow (lazy, agar tidak crash saat belum ada model) ──
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
    except Exception as e:
        return {}, None

# ── Konfigurasi kelas & nutrisi ────────────────────────────
CLASS_NAMES = [
    "ayam_goreng", "burger", "french_fries", "gado_gado",
    "ikan_goreng", "mie_goreng", "nasi_goreng", "nasi_padang",
    "pizza", "rawon", "rendang", "sate", "soto_ayam"
]

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

# Range nutrisi min/max per makanan (dari nutrisi_makanan_indonesia.xlsx + TKPI)
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

# ── Helper functions ────────────────────────────────────────
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

def predict(model, img_array, tf_module):
    start = time.time()
    preds = model.predict(img_array, verbose=0)
    elapsed = (time.time() - start) * 1000
    idx = int(np.argmax(preds[0]))
    return CLASS_NAMES[idx], float(preds[0][idx]), preds[0], elapsed

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
}
[data-testid="stSidebar"] * { color: #e8f4f8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #a8d8ea !important; font-size: 0.82rem; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 { 
    font-size: 2.6rem; font-weight: 800; 
    background: linear-gradient(90deg, #e2b96f, #f7d794, #e2b96f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.5px;
}
.hero p { color: #8aa6c1; font-size: 1rem; margin: 0.5rem 0 0; }

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #2d6a8f !important;
    border-radius: 16px !important;
    background: rgba(45, 106, 143, 0.06) !important;
    padding: 1rem !important;
}

/* Result cards */
.result-card {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border: 1px solid rgba(226, 185, 111, 0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.food-name {
    font-size: 1.9rem; font-weight: 800;
    color: #f7d794; margin: 0; letter-spacing: -0.3px;
}
.confidence-text { color: #8aa6c1; font-size: 0.85rem; margin-top: 0.2rem; }

/* Kalori badge */
.kalori-badge {
    background: linear-gradient(135deg, #e2b96f, #f7d794);
    color: #1a1a2e;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.kalori-badge .number { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.kalori-badge .label  { font-size: 0.78rem; font-weight: 600; opacity: 0.75; margin-top: 2px; }
.kalori-badge .range  { font-size: 0.75rem; opacity: 0.65; margin-top: 4px; font-family: 'DM Mono', monospace; }

/* Macro pills */
.macro-row { display: flex; gap: 0.6rem; margin-top: 0.8rem; }
.macro-pill {
    flex: 1; border-radius: 10px; padding: 0.7rem 0.5rem;
    text-align: center; font-size: 0.78rem;
}
.pill-karbo   { background: rgba(99, 179, 237, 0.15); border: 1px solid rgba(99, 179, 237, 0.3); color: #63b3ed; }
.pill-protein { background: rgba(104, 211, 145, 0.15); border: 1px solid rgba(104, 211, 145, 0.3); color: #68d391; }
.pill-lemak   { background: rgba(252, 129, 74, 0.15);  border: 1px solid rgba(252, 129, 74, 0.3);  color: #fc814a; }
.macro-pill .macro-val { font-size: 1.1rem; font-weight: 700; display: block; }

/* Top-5 bar */
.top5-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.35rem 0; }
.top5-label { font-size: 0.78rem; color: #8aa6c1; width: 110px; flex-shrink: 0; text-align: right; }
.top5-bar-wrap { flex: 1; background: rgba(255,255,255,0.06); border-radius: 6px; height: 10px; overflow: hidden; }
.top5-bar-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #2d6a8f, #e2b96f); }
.top5-pct { font-size: 0.76rem; color: #a0b4c8; font-family: 'DM Mono', monospace; width: 42px; }

/* History card */
.hist-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.hist-food { font-weight: 700; color: #f7d794; font-size: 0.9rem; }
.hist-meta { color: #6b8fa8; font-size: 0.75rem; font-family: 'DM Mono', monospace; }

/* Disclaimer */
.disclaimer {
    background: rgba(226, 185, 111, 0.08);
    border-left: 3px solid #e2b96f;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.75rem; color: #a8956a;
    margin-top: 1rem;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: rgba(45, 106, 143, 0.3);
    border: 1px solid #2d6a8f;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #7ec8e3;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.5rem;
}

/* Demo mode notice */
.demo-notice {
    background: rgba(252, 129, 74, 0.1);
    border: 1px solid rgba(252, 129, 74, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #fc814a;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍛 NutriSnap ID")
    st.markdown("---")
    
    st.markdown("**Pilih Model AI**")
    selected_model = st.radio(
        "", ["SimpleCNN", "MobileNetV2", "EfficientNetB0"],
        index=0,
        help="SimpleCNN: akurasi tertinggi | MobileNetV2: cepat | EfficientNetB0: eksperimen"
    )
    
    model_info = {
        "SimpleCNN"     : ("79.14%", "~101 ms", "10.9 MB"),
        "MobileNetV2"   : ("73.05%", "~117 ms", "22.6 MB"),
        "EfficientNetB0": ("12.24%", "~125 ms", "37.4 MB"),
    }
    acc, inf, size = model_info[selected_model]
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.05);border-radius:10px;padding:0.8rem;margin-top:0.5rem;font-size:0.78rem;'>
        <div>🎯 Akurasi: <b style='color:#f7d794'>{acc}</b></div>
        <div>⚡ Inferensi: <b style='color:#f7d794'>{inf}</b></div>
        <div>💾 Ukuran: <b style='color:#f7d794'>{size}</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**13 Kelas Makanan**")
    for name in DISPLAY_NAMES.values():
        st.markdown(f"<span style='font-size:0.78rem;color:#6b8fa8'>• {name}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<span style='font-size:0.72rem;color:#4a6a7a'>Dikembangkan untuk EAS Machine Learning<br>Data nutrisi: TKPI Kemenkes RI</span>", unsafe_allow_html=True)

# ── Main content ────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🍛 NutriSnap ID</h1>
    <p>Klasifikasi makanan Indonesia & estimasi kalori otomatis berbasis CNN</p>
</div>
""", unsafe_allow_html=True)

# Load models
models, tf_module = load_models()
demo_mode = len(models) == 0

if demo_mode:
    st.markdown("""
    <div class='demo-notice'>
        ⚠️ <b>Mode Demo</b> — File model (.keras) belum ditemukan di folder <code>models/</code>.
        Upload model hasil training notebook ke folder <code>models/</code> untuk prediksi nyata.
        Saat ini aplikasi menggunakan <b>simulasi prediksi</b>.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

# Tabs
tab1, tab2 = st.tabs(["📸 Klasifikasi", "📋 Riwayat Analisis"])

# ── Session state ───────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Tab 1: Klasifikasi ──────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload Foto Makanan")
        uploaded = st.file_uploader(
            "Pilih atau drag & drop gambar",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar yang diupload", use_container_width=True)
            
            analyze_btn = st.button("🔍 Analisis Sekarang", use_container_width=True, type="primary")
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#4a6a7a;'>
                <div style='font-size:3rem'>📷</div>
                <div style='font-size:0.85rem;margin-top:0.5rem'>Upload foto makanan untuk memulai</div>
                <div style='font-size:0.75rem;margin-top:0.3rem;color:#3a5a6a'>JPG, PNG, WEBP didukung</div>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

    with col_result:
        st.markdown("#### Hasil Analisis")

        if uploaded and analyze_btn:
            with st.spinner("Menganalisis gambar..."):
                time.sleep(0.5)  # UX pause

                if not demo_mode and selected_model in models:
                    # Real prediction
                    size = IMG_SIZE[selected_model]
                    arr  = preprocess_image(img, size)
                    food_class, confidence, all_preds, inf_time = predict(
                        models[selected_model], arr, tf_module
                    )
                    top5_idx   = np.argsort(all_preds)[::-1][:5]
                    top5_names = [CLASS_NAMES[i] for i in top5_idx]
                    top5_probs = all_preds[top5_idx]
                else:
                    # Demo mode: simulasi
                    food_class  = random.choice(CLASS_NAMES)
                    confidence  = random.uniform(0.55, 0.99)
                    inf_time    = random.uniform(90, 130)
                    top5_idx    = random.sample(range(13), 5)
                    top5_names  = [CLASS_NAMES[i] for i in top5_idx]
                    remaining   = 1.0
                    top5_probs  = []
                    for i in range(5):
                        v = confidence if i == 0 else random.uniform(0.01, remaining * 0.4)
                        top5_probs.append(v)
                        remaining -= v
                    top5_probs = np.array(top5_probs)
                    top5_names[0] = food_class

                nutrition = estimate_nutrition(food_class)

            # ── Hasil ───────────────────────────────────────
            st.markdown(f"""
            <div class='result-card'>
                <div class='model-badge'>{selected_model} {'(demo)' if demo_mode else ''}</div>
                <div class='food-name'>{DISPLAY_NAMES[food_class]}</div>
                <div class='confidence-text'>Confidence: <b style='color:#f7d794'>{confidence*100:.1f}%</b> &nbsp;·&nbsp; Inferensi: {inf_time:.0f} ms</div>
            </div>
            """, unsafe_allow_html=True)

            if nutrition:
                st.markdown(f"""
                <div class='kalori-badge'>
                    <div class='label'>ESTIMASI KALORI PER PORSI ({nutrition['porsi_g']}g)</div>
                    <div class='number'>~{nutrition['kalori']} kkal</div>
                    <div class='range'>range: {nutrition['kal_min']}–{nutrition['kal_max']} kkal</div>
                </div>
                <div class='macro-row'>
                    <div class='macro-pill pill-karbo'>
                        <span class='macro-val'>~{nutrition['karbo']}g</span>Karbohidrat
                    </div>
                    <div class='macro-pill pill-protein'>
                        <span class='macro-val'>~{nutrition['protein']}g</span>Protein
                    </div>
                    <div class='macro-pill pill-lemak'>
                        <span class='macro-val'>~{nutrition['lemak']}g</span>Lemak
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Top-5
            st.markdown("**Top-5 Prediksi**")
            for name, prob in zip(top5_names, top5_probs):
                pct = float(prob) * 100
                st.markdown(f"""
                <div class='top5-row'>
                    <div class='top5-label'>{DISPLAY_NAMES.get(name, name)}</div>
                    <div class='top5-bar-wrap'>
                        <div class='top5-bar-fill' style='width:{min(pct,100):.1f}%'></div>
                    </div>
                    <div class='top5-pct'>{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class='disclaimer'>
                ℹ️ Nilai kalori & nutrisi adalah <b>estimasi</b> berdasarkan range umum per porsi.
                Nilai aktual bervariasi tergantung resep, ukuran porsi, dan cara memasak.
            </div>
            """, unsafe_allow_html=True)

            # Simpan ke history
            st.session_state.history.insert(0, {
                "food"      : food_class,
                "display"   : DISPLAY_NAMES[food_class],
                "confidence": confidence,
                "kalori"    : nutrition["kalori"] if nutrition else "-",
                "kal_min"   : nutrition["kal_min"] if nutrition else "-",
                "kal_max"   : nutrition["kal_max"] if nutrition else "-",
                "model"     : selected_model,
                "porsi_g"   : nutrition["porsi_g"] if nutrition else "-",
            })

        elif not uploaded:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#4a6a7a;'>
                <div style='font-size:2.5rem'>🔍</div>
                <div style='font-size:0.85rem;margin-top:0.5rem'>Hasil analisis akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Riwayat ──────────────────────────────────────────
with tab2:
    st.markdown("#### 📋 Riwayat Analisis")
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#4a6a7a;'>
            <div style='font-size:2rem'>📭</div>
            <div style='font-size:0.85rem;margin-top:0.5rem'>Belum ada riwayat analisis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_clear = st.columns([4, 1])[1]
        with col_clear:
            if st.button("🗑️ Hapus Semua", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        total_kal = sum(h["kalori"] for h in st.session_state.history if isinstance(h["kalori"], int))
        st.markdown(f"""
        <div style='background:rgba(226,185,111,0.1);border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;'>
            <div style='color:#8aa6c1;font-size:0.85rem'>{len(st.session_state.history)} item dianalisis sesi ini</div>
            <div style='color:#f7d794;font-weight:700;font-size:1.1rem'>~{total_kal} kkal total</div>
        </div>
        """, unsafe_allow_html=True)

        for i, h in enumerate(st.session_state.history):
            st.markdown(f"""
            <div class='hist-card'>
                <div style='font-size:1.4rem'>🍽️</div>
                <div style='flex:1'>
                    <div class='hist-food'>{h['display']}</div>
                    <div class='hist-meta'>
                        ~{h['kalori']} kkal ({h['kal_min']}–{h['kal_max']}) &nbsp;·&nbsp;
                        porsi {h['porsi_g']}g &nbsp;·&nbsp;
                        {h['model']} &nbsp;·&nbsp;
                        confidence {h['confidence']*100:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
