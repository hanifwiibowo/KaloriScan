# 🍛 NutriSnap ID — Klasifikasi Makanan Indonesia & Estimasi Kalori

Aplikasi web berbasis CNN untuk mengenali makanan tradisional Indonesia dari foto dan memperkirakan kandungan kalorinya secara otomatis.

## 🚀 Demo Langsung
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📋 Fitur
- **Klasifikasi citra** — upload foto makanan, model CNN mengenali jenisnya
- **Estimasi kalori** — perkiraan kalori, karbo, protein, dan lemak per porsi
- **3 model CNN** — SimpleCNN, MobileNetV2, EfficientNetB0
- **Riwayat analisis** — total kalori sesi ini tercatat otomatis
- **Mode demo** — bisa dicoba tanpa model (simulasi prediksi)

## 🍽️ 13 Kelas Makanan
Ayam Goreng · Burger · French Fries · Gado-Gado · Ikan Goreng · Mie Goreng · Nasi Goreng · Nasi Padang · Pizza · Rawon · Rendang · Sate · Soto Ayam

## 📁 Struktur Folder
```
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Tema & konfigurasi
├── models/                 # Taruh file model hasil training di sini
│   ├── model_SimpleCNN.keras
│   ├── model_MobileNetV2.keras
│   └── model_EfficientNetB0.keras
└── README.md
```

## ⚙️ Cara Deploy ke Streamlit Community Cloud

1. **Push repo ini ke GitHub**
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan akun GitHub
4. Klik **New app** → pilih repo ini → `app.py`
5. Klik **Deploy** → dapat link publik otomatis

## 🏋️ Cara Upload Model ke GitHub

Setelah training selesai di Colab, download file `.keras` lalu:
```bash
# Kalau file < 100MB, bisa langsung push
git add models/model_SimpleCNN.keras
git commit -m "add trained model"
git push
```

> ⚠️ Jika file model > 100MB, gunakan [Git LFS](https://git-lfs.com):
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> git add models/
> git commit -m "add models via LFS"
> git push
> ```

## 📊 Performa Model

| Model | Akurasi | Inferensi | Ukuran |
|---|---|---|---|
| SimpleCNN | **79.14%** | ~101 ms | 10.9 MB |
| MobileNetV2 | 73.05% | ~117 ms | 22.6 MB |
| EfficientNetB0 | 12.24% | ~125 ms | 37.4 MB |

## 📌 Catatan
- Nilai kalori & nutrisi adalah **estimasi** berdasarkan range umum per porsi (TKPI Kemenkes RI)
- Nilai aktual bervariasi tergantung resep, ukuran porsi, dan cara memasak
- Dataset: 6.500 gambar, 13 kelas, 500 gambar/kelas
