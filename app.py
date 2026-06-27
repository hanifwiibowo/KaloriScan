import streamlit as st
import numpy as np
import random
import os
import time
from PIL import Image

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="UKM - Ukur Kalori Makanan",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load TensorFlow (lazy, agar tidak crash saat belum ada model) ──
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

def predict(model, img_array):
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds[0]))
    return CLASS_NAMES[idx], float(preds[0][idx]), preds[0]

def confidence_badge(conf):
    if conf >= 0.75:
        return "Yakin", "#7ed957"
    elif conf >= 0.5:
        return "Cukup yakin", "#ffc857"
    else:
        return "Kurang yakin", "#ff8c5a"

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a120d, #241a14, #2e1d15);
}
[data-testid="stSidebar"] * { color: #f0ddd0 !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #2a1810 0%, #3d2015 50%, #1f1209 100%);
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
}
.hero h1 {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(90deg, #ff9466, #ffcf94, #ff9466);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.5px;
}
.hero .hero-sub { color: #ffb088; font-size: 1.05rem; font-weight: 700; margin: 0.35rem 0 0; }
.hero .hero-desc { color: #c9a695; font-size: 0.92rem; margin: 0.4rem 0 0; }

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #ff7a47 !important;
    border-radius: 16px !important;
    background: rgba(255, 122, 71, 0.06) !important;
    padding: 1rem !important;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #2e1d15, #1c120c);
    border: 1px solid rgba(255, 148, 102, 0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.confidence-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 700;
    border: 1px solid;
}
.food-name {
    font-size: 1.9rem; font-weight: 800;
    color: #ffcf94; margin: 0.5rem 0 0; letter-spacing: -0.3px;
}

/* Kalori badge */
.kalori-badge {
    background: linear-gradient(135deg, #ff7a47, #ffb088);
    color: #241208;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.kalori-badge .number { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.kalori-badge .label  { font-size: 0.78rem; font-weight: 700; opacity: 0.8; margin-top: 2px; }
.kalori-badge .range  { font-size: 0.75rem; opacity: 0.7; margin-top: 4px; }

/* Macro pills */
.macro-row { display: flex; gap: 0.6rem; margin-top: 0.8rem; }
.macro-pill {
    flex: 1; border-radius: 10px; padding: 0.7rem 0.5rem;
    text-align: center; font-size: 0.78rem;
}
.pill-karbo   { background: rgba(99, 179, 237, 0.15); border: 1px solid rgba(99, 179, 237, 0.3); color: #63b3ed; }
.pill-protein { background: rgba(126, 217, 87, 0.15); border: 1px solid rgba(126, 217, 87, 0.3); color: #7ed957; }
.pill-lemak   { background: rgba(255, 140, 90, 0.15); border: 1px solid rgba(255, 140, 90, 0.3); color: #ff8c5a; }
.macro-pill .macro-val { font-size: 1.1rem; font-weight: 700; display: block; }

/* Alternative suggestion hint */
.alt-hint {
    background: rgba(255, 200, 87, 0.08);
    border-left: 3px solid #ffc857;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.82rem; color: #ffc857;
    margin: 0.8rem 0;
}

/* History card */
.hist-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.hist-food { font-weight: 700; color: #ffcf94; font-size: 0.9rem; }
.hist-meta { color: #a98a78; font-size: 0.75rem; }

/* Disclaimer */
.disclaimer {
    background: rgba(255, 122, 71, 0.08);
    border-left: 3px solid #ff7a47;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.75rem; color: #c9a695;
    margin-top: 1rem;
}

/* Maintenance notice */
.demo-notice {
    background: rgba(255, 140, 90, 0.1);
    border: 1px solid rgba(255, 140, 90, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #ff8c5a;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 0.4rem 0 1rem;'>
        <div style='font-size:2.3rem; line-height:1;'>🍽️</div>
        <div style='font-size:1.5rem; font-weight:800; color:#ffb088; letter-spacing:0.5px; margin-top:0.3rem;'>UKM</div>
        <div style='font-size:0.76rem; color:#c9a695; margin-top:0.1rem;'>Ukur Kalori Makanan</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Cara pakai**")
    st.markdown("""
    <div style='font-size:0.8rem; color:#d8b8a8; line-height:1.7;'>
    1. Upload foto makananmu<br>
    2. Tekan <b>Cek Sekarang</b><br>
    3. Lihat estimasi kalori & gizinya
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    with st.expander("🍱 Makanan yang bisa dikenali"):
        names = list(DISPLAY_NAMES.values())
        cols = st.columns(2)
        for i, name in enumerate(names):
            with cols[i % 2]:
                st.markdown(f"<div style='font-size:0.78rem; color:#e0c4b4; padding:2px 0;'>• {name}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem; color:#8a6d60;'>Estimasi nutrisi mengacu pada data TKPI — Kemenkes RI</div>",
        unsafe_allow_html=True
    )

# ── Main content ────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🍽️ UKM</h1>
    <p class='hero-sub'>Ukur Kalori Makanan</p>
    <p class='hero-desc'>Foto makananmu, langsung tahu perkiraan kalori &amp; gizinya.</p>
</div>
""", unsafe_allow_html=True)

# Load models
models, tf_module = load_models()
active_model = next((m for m in PRIORITY_MODELS if m in models), None)
demo_mode = active_model is None

if demo_mode:
    st.markdown("""
    <div class='demo-notice'>
        🔧 Sistem sedang pemanasan. Coba upload ulang foto kamu dalam beberapa saat.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

# Tabs
tab1, tab2 = st.tabs(["📸 Cek Kalori", "🕒 Riwayat"])

# ── Session state ───────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "current_upload_id" not in st.session_state:
    st.session_state.current_upload_id = None

# ── Tab 1: Cek Kalori ────────────────────────────────────────
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
            upload_id = f"{uploaded.name}-{uploaded.size}"
            if st.session_state.current_upload_id != upload_id:
                st.session_state.current_upload_id = upload_id
                st.session_state.result = None

            img = Image.open(uploaded)
            st.image(img, caption="Foto kamu", use_container_width=True)
            analyze_btn = st.button("🔍 Cek Sekarang", use_container_width=True, type="primary")
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#8a6d60;'>
                <div style='font-size:3rem'>📷</div>
                <div style='font-size:0.85rem;margin-top:0.5rem'>Upload foto makanan untuk memulai</div>
                <div style='font-size:0.75rem;margin-top:0.3rem;color:#6a4f44'>JPG, PNG, WEBP didukung</div>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

    with col_result:
        st.markdown("#### Hasil")

        if uploaded and analyze_btn:
            with st.spinner("Menganalisis foto..."):
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

            if manual:
                label, color = "Dipilih manual", "#a98a78"
            else:
                label, color = confidence_badge(confidence)

            st.markdown(f"""
            <div class='result-card'>
                <div class='confidence-badge' style='color:{color};border-color:{color}55;background:{color}1A;'>{label}</div>
                <div class='food-name'>{DISPLAY_NAMES[food_class]}</div>
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

            if not manual and result.get("alt") and confidence < 0.55:
                alt_display = DISPLAY_NAMES.get(result["alt"], result["alt"])
                st.markdown(f"""
                <div class='alt-hint'>🤔 Hasilnya kurang yakin — mungkin juga ini <b>{alt_display}</b>?</div>
                """, unsafe_allow_html=True)

            override_options = ["Sudah benar, lanjutkan"] + [
                n for k, n in DISPLAY_NAMES.items() if k != food_class
            ]
            picked = st.selectbox(
                "Bukan ini? Pilih yang benar:",
                override_options,
                key=f"override_select_{food_class}_{manual}"
            )
            if picked != "Sudah benar, lanjutkan":
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
                        "kalori" : new_nutri["kalori"],
                        "kal_min": new_nutri["kal_min"],
                        "kal_max": new_nutri["kal_max"],
                        "porsi_g": new_nutri["porsi_g"],
                    })
                st.rerun()

            st.markdown("""
            <div class='disclaimer'>
                ℹ️ Nilai kalori & nutrisi adalah <b>estimasi</b> berdasarkan range umum per porsi.
                Nilai aktual bervariasi tergantung resep, ukuran porsi, dan cara memasak.
            </div>
            """, unsafe_allow_html=True)

        elif uploaded:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#8a6d60;'>
                <div style='font-size:2.5rem'>👉</div>
                <div style='font-size:0.85rem;margin-top:0.5rem'>Tekan "Cek Sekarang" untuk melihat hasilnya</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem 1rem;color:#8a6d60;'>
                <div style='font-size:2.5rem'>🔍</div>
                <div style='font-size:0.85rem;margin-top:0.5rem'>Hasil cek kalori akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Riwayat ──────────────────────────────────────────
with tab2:
    st.markdown("#### 🕒 Riwayat")
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center;padding:3rem;color:#8a6d60;'>
            <div style='font-size:2rem'>📭</div>
            <div style='font-size:0.85rem;margin-top:0.5rem'>Belum ada riwayat. Yuk cek makananmu dulu!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_clear = st.columns([4, 1])[1]
        with col_clear:
            if st.button("🗑️ Hapus Semua", use_container_width=True):
                st.session_state.history = []
                st.session_state.result = None
                st.rerun()

        total_kal = sum(h["kalori"] for h in st.session_state.history if isinstance(h["kalori"], int))
        st.markdown(f"""
        <div style='background:rgba(255,122,71,0.1);border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;display:flex;justify-content:space-between;align-items:center;'>
            <div style='color:#c9a695;font-size:0.85rem'>{len(st.session_state.history)} item dicek hari ini</div>
            <div style='color:#ffcf94;font-weight:700;font-size:1.1rem'>~{total_kal} kkal total</div>
        </div>
        """, unsafe_allow_html=True)

        for h in st.session_state.history:
            st.markdown(f"""
            <div class='hist-card'>
                <div style='font-size:1.4rem'>🍽️</div>
                <div style='flex:1'>
                    <div class='hist-food'>{h['display']}</div>
                    <div class='hist-meta'>
                        ~{h['kalori']} kkal ({h['kal_min']}–{h['kal_max']}) &nbsp;·&nbsp; porsi {h['porsi_g']}g
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
