# 🧠 Deteksi Dini Risiko Kesehatan Mental — SMA Kota Medan

Aplikasi web berbasis **Streamlit** untuk pelatihan deteksi dini risiko kesehatan mental
melalui media sosial, dikembangkan sebagai media **Pengabdian Masyarakat** bagi siswa SMA di Kota Medan.

---

## 📁 Struktur File

```
mental_health_app/
├── app.py                    ← Aplikasi utama Streamlit
├── requirements.txt          ← Daftar library Python
├── .streamlit/
│   └── config.toml           ← Konfigurasi tema Streamlit
└── README.md                 ← Panduan ini
```

---

## 🚀 Cara Deploy ke Streamlit Cloud (GRATIS)

### Langkah 1 — Siapkan GitHub
1. Buat akun di [github.com](https://github.com) (gratis)
2. Buat repository baru, contoh: `mental-health-detection`
3. Upload semua file ke repository tersebut:
   - `app.py`
   - `requirements.txt`
   - folder `.streamlit/` beserta `config.toml`

### Langkah 2 — Deploy ke Streamlit Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **"New app"**
4. Pilih:
   - Repository: `mental-health-detection`
   - Branch: `main`
   - Main file path: `app.py`
5. Klik **"Deploy!"**
6. Tunggu 2-5 menit → aplikasi online!
7. URL akan tersedia: `https://namaapp.streamlit.app`

### Langkah 3 — Bagikan ke Siswa
- Bagikan URL ke seluruh siswa SMA
- Bisa diakses dari HP, tablet, maupun laptop
- Tidak perlu install apapun

---

## 💻 Cara Jalankan Lokal (untuk testing)

```bash
# 1. Install Python 3.9-3.11
# 2. Buka terminal/command prompt
# 3. Install library
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py

# 5. Buka browser → http://localhost:8501
```

---

## 📋 Format Dataset CSV

Dataset harus berformat CSV dengan minimal 2 kolom:

| text | label |
|------|-------|
| Hari ini aku sangat bahagia | Normal |
| Aku merasa sangat sedih | Depresi |
| Gelisah terus memikirkan ujian | Cemas |

**Label yang diterima:**
- `Normal` atau `0`
- `Depresi` atau `1`  
- `Cemas` atau `2`

**Rekomendasi:** minimal 200 data per kelas untuk hasil yang baik.

---

## 🎓 Fitur Aplikasi

| Fitur | Keterangan |
|-------|-----------|
| 🏠 Beranda & Deteksi | Input teks dan lihat hasil deteksi real-time |
| 📊 Training Model | Upload dataset → preprocessing → training BiLSTM + SVM + NB |
| 📚 Edukasi | Materi kesehatan mental untuk siswa SMA |
| 📋 Riwayat | Lihat dan download riwayat deteksi |
| ℹ️ Tentang | Info aplikasi dan disclaimer |

---

## ⚠️ Disclaimer

Aplikasi ini adalah **alat bantu edukasi**, bukan pengganti diagnosis medis.
Jika ada kekhawatiran serius tentang kesehatan mental, segera hubungi:
- Guru BK sekolah
---

© 2026 · Pengabdian Masyarakat · Universitas Bunda Thamrin · Medan
