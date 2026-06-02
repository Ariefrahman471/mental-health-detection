import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import json
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Page config (HARUS paling atas) ────────────────────────────────
st.set_page_config(
    page_title="Deteksi Dini Kesehatan Mental — SMA Medan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Matplotlib ──────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── NLP ─────────────────────────────────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_OK = True
except ImportError:
    SASTRAWI_OK = False

try:
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS_ID = set(stopwords.words('indonesian'))
    NLTK_OK = True
except Exception:
    NLTK_OK = False
    STOPWORDS_ID = {
        'yang', 'dan', 'di', 'dari', 'ini', 'itu', 'untuk',
        'dengan', 'pada', 'ke', 'dalam', 'oleh', 'sebagai',
        'adalah', 'atau', 'karena', 'jika', 'maka',
    }

# Tambahan stopwords media sosial
STOPWORDS_ID.update({
    'rt', 'https', 'tco', 'amp', 'yg', 'udh', 'gak', 'ga',
    'nya', 'sih', 'dong', 'deh', 'yah', 'ya', 'ah', 'oh',
    'eh', 'lah', 'pun', 'kan', 'wkwk', 'wkwkwk',
})

# ─── Deep Learning ───────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, LSTM, Dense,
                                         Dropout, Bidirectional)
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    TF_OK = True
except ImportError:
    TF_OK = False

# ─── Sklearn ─────────────────────────────────────────────────────────
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score,
                              precision_score, recall_score)
from scipy.stats import chi2

# ═══════════════════════════════════════════════════════════════════
# KONSTANTA
# ═══════════════════════════════════════════════════════════════════
LABEL_MAPPING = {
    'normal': 0, 'Normal': 0, 'NORMAL': 0,
    'baik': 0, 'sehat': 0, 'positif': 0, '0': 0, '0.0': 0,
    'depresi': 1, 'Depresi': 1, 'DEPRESI': 1,
    'depression': 1, 'depress': 1, 'sedih': 1, '1': 1, '1.0': 1,
    'cemas': 2, 'Cemas': 2, 'CEMAS': 2,
    'anxiety': 2, 'Anxiety': 2, 'anxious': 2,
    'khawatir': 2, 'gelisah': 2, '2': 2, '2.0': 2,
}
ALL_LABEL_NAMES = ["Normal", "Depresi", "Cemas"]
LABEL_CONFIG = {
    0: {"name": "😊 NORMAL",  "desc": "Kondisi mental baik",
        "color": "#00b894", "bg": "#d4edda", "text": "#155724",
        "advice": "Kondisi mentalmu tampak baik! Pertahankan gaya hidup sehat dan terus jaga hubungan sosial yang positif."},
    1: {"name": "😢 DEPRESI", "desc": "Risiko depresi terdeteksi",
        "color": "#d63031", "bg": "#f8d7da", "text": "#721c24",
        "advice": "Teks ini mengindikasikan risiko depresi. Jika kamu atau temanmu merasa seperti ini, segera bicarakan dengan guru BK, orang tua, atau profesional kesehatan mental."},
    2: {"name": "😰 CEMAS",   "desc": "Risiko kecemasan terdeteksi",
        "color": "#e17055", "bg": "#fff3cd", "text": "#856404",
        "advice": "Teks ini mengindikasikan risiko kecemasan. Cobalah teknik relaksasi, berbagi cerita dengan orang terpercaya, atau konsultasi dengan konselor sekolah."},
}

TEXT_COL_CANDIDATES = [
    'text', 'Text', 'TEXT', 'tweet', 'komentar', 'teks',
    'sentence', 'data', 'post', 'status', 'message', 'content',
]
LABEL_COL_CANDIDATES = [
    'label', 'Label', 'LABEL', 'class', 'Class', 'kategori',
    'category', 'target', 'sentiment', 'kondisi', 'kelas',
]

# Contoh teks untuk demo siswa
DEMO_EXAMPLES = [
    ("😊 Normal", "Hari ini aku sangat bahagia bisa bertemu teman-teman. Semangat belajar!"),
    ("😢 Depresi", "Aku merasa sangat sedih dan tidak bersemangat. Tidak ada yang peduli padaku."),
    ("😰 Cemas", "Gelisah terus, jantung berdebar kencang memikirkan ujian besok. Takut gagal."),
    ("😊 Normal", "Liburan kemarin seru banget, main sama keluarga. Senang sekali!"),
    ("😢 Depresi", "Rasanya ingin menghilang saja. Hidup terasa berat dan tanpa harapan."),
    ("😰 Cemas", "Khawatir banget sama masa depan. Setiap malam susah tidur memikirkannya."),
]

EDUKASI_KONTEN = {
    "Apa itu Kesehatan Mental?": {
        "icon": "🧠",
        "isi": """
Kesehatan mental adalah kondisi kesejahteraan seseorang secara psikologis, emosional, dan sosial. 
Kesehatan mental yang baik memungkinkan kita untuk:
- Menghadapi tekanan kehidupan sehari-hari
- Bekerja dan belajar dengan produktif
- Berkontribusi dalam komunitas dan lingkungan sosial

**Kesehatan mental sama pentingnya dengan kesehatan fisik!**
        """,
    },
    "Mengenal Depresi": {
        "icon": "😢",
        "isi": """
**Depresi** adalah gangguan mood yang menyebabkan perasaan sedih yang terus-menerus dan kehilangan minat.

**Gejala umum:**
- Perasaan sedih, kosong, atau putus asa yang berkepanjangan
- Kehilangan minat pada aktivitas yang biasanya disukai
- Perubahan nafsu makan dan pola tidur
- Kelelahan dan kurang energi
- Sulit berkonsentrasi
- Pikiran negatif tentang diri sendiri

**Jika kamu atau temanmu mengalami ini lebih dari 2 minggu, segera cari bantuan profesional.**
        """,
    },
    "Mengenal Kecemasan": {
        "icon": "😰",
        "isi": """
**Kecemasan (anxiety)** adalah perasaan khawatir, gelisah, atau takut yang berlebihan dan sulit dikendalikan.

**Gejala umum:**
- Khawatir berlebihan tentang berbagai hal
- Jantung berdebar-debar
- Sulit tidur
- Otot tegang
- Sulit berkonsentrasi
- Menghindari situasi yang memicu kecemasan

**Kecemasan ringan adalah normal, tapi jika mengganggu kehidupan sehari-hari, perlu penanganan.**
        """,
    },
    "Cara Menjaga Kesehatan Mental": {
        "icon": "💚",
        "isi": """
**Tips menjaga kesehatan mental untuk remaja:**

1. **Tidur cukup** — 8-9 jam per malam untuk remaja
2. **Olahraga rutin** — minimal 30 menit per hari
3. **Makan bergizi** — jangan skip makan
4. **Batasi media sosial** — max 2-3 jam per hari
5. **Cerita ke orang terpercaya** — teman, orang tua, atau guru BK
6. **Lakukan hobi** — hal yang menyenangkan untuk relaksasi
7. **Meditasi/pernapasan dalam** — bantu tenangkan pikiran
8. **Hindari perbandingan diri** — setiap orang punya perjalanannya sendiri
        """,
    },
    "Kapan Harus Minta Bantuan?": {
        "icon": "🆘",
        "isi": """
**Segera cari bantuan jika kamu atau temanmu:**

- Memiliki pikiran untuk menyakiti diri sendiri
- Merasa tidak ingin hidup lagi
- Tidak bisa menjalankan aktivitas sehari-hari
- Menggunakan alkohol atau obat-obatan untuk mengatasi masalah
- Menarik diri dari semua orang selama berminggu-minggu

**Ke mana mencari bantuan?**
- 🏫 **Guru BK** di sekolahmu
- 👨‍👩‍👧 **Orang tua atau anggota keluarga terpercaya**
- 🏥 **Puskesmas atau klinik terdekat** (ada layanan kesehatan jiwa gratis)
- 📞 **Into The Light Indonesia**: 119 ext 8 (hotline kesehatan jiwa)
- 📞 **Yayasan Pulih**: (021) 788-42580
        """,
    },
}

# ═══════════════════════════════════════════════════════════════════
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    /* ── Global ── */
    .stApp { background-color: #f0f4f8; }
    
    /* ── Header utama ── */
    .main-header {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(108,92,231,0.3);
    }
    .main-header h1 { font-size: 1.8rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 0.95rem; margin: 0.5rem 0 0; opacity: 0.9; }

    /* ── Card ── */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border: 1px solid #e8ecf0;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── Hasil deteksi ── */
    .result-normal {
        background: #d4edda; border-left: 5px solid #00b894;
        padding: 1.2rem 1.5rem; border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .result-depresi {
        background: #f8d7da; border-left: 5px solid #d63031;
        padding: 1.2rem 1.5rem; border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .result-cemas {
        background: #fff3cd; border-left: 5px solid #e17055;
        padding: 1.2rem 1.5rem; border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem;
    }
    .result-desc { font-size: 0.9rem; opacity: 0.8; }
    .result-advice {
        font-size: 0.88rem;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    
    /* ── Badge label ── */
    .badge-normal  { background:#00b894; color:white; padding:3px 12px;
                     border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-depresi { background:#d63031; color:white; padding:3px 12px;
                     border-radius:20px; font-size:0.8rem; font-weight:600; }
    .badge-cemas   { background:#e17055; color:white; padding:3px 12px;
                     border-radius:20px; font-size:0.8rem; font-weight:600; }

    /* ── Metric card ── */
    .metric-card {
        background: white; border-radius: 10px;
        padding: 1rem; text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 1px solid #e8ecf0;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #6c5ce7; }
    .metric-label { font-size: 0.78rem; color: #718096; margin-top: 0.2rem; }

    /* ── Step indicator ── */
    .step-active {
        background: #6c5ce7; color: white;
        padding: 0.4rem 1rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
    }
    .step-done {
        background: #00b894; color: white;
        padding: 0.4rem 1rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
    }
    .step-pending {
        background: #e2e8f0; color: #718096;
        padding: 0.4rem 1rem; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
    }

    /* ── Edukasi ── */
    .edukasi-card {
        background: #f8f9ff; border: 1px solid #e0e4ff;
        border-radius: 12px; padding: 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center; padding: 1.5rem;
        color: #718096; font-size: 0.82rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }

    /* ── Warning box ── */
    .warning-mental {
        background: #fff5f5; border: 1px solid #fc8181;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.88rem; color: #c53030;
    }

    /* ── Sidebar ── */
    .sidebar-title {
        font-weight: 700; font-size: 1rem;
        color: #2d3748; margin-bottom: 0.5rem;
    }

    /* Override Streamlit buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover { transform: translateY(-1px); }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_stemmer():
    if SASTRAWI_OK:
        factory = StemmerFactory()
        return factory.create_stemmer()
    return None

def clean_text(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, stemmer=None, use_stemming=True):
    text   = clean_text(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS_ID and len(w) > 1]
    if use_stemming and stemmer:
        try:
            tokens = [stemmer.stem(w) for w in tokens]
        except Exception:
            pass
    return ' '.join(tokens)

def try_read_csv(file):
    encodings  = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    separators = [',', ';', '\t', '|']
    for enc in encodings:
        for sep in separators:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc, sep=sep,
                                 engine='python', on_bad_lines='skip')
                if df.shape[1] >= 2 and len(df) > 0:
                    return df
            except Exception:
                pass
    raise ValueError("Tidak dapat membaca file CSV.")

def detect_columns(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    text_col = label_col = None
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            text_col = c; break
        if c.lower() in cols_lower:
            text_col = cols_lower[c.lower()]; break
    for c in LABEL_COL_CANDIDATES:
        if c in df.columns:
            label_col = c; break
        if c.lower() in cols_lower:
            label_col = cols_lower[c.lower()]; break
    if text_col is None:
        for c in df.columns:
            if df[c].astype(str).str.len().mean() > 20:
                text_col = c; break
    if label_col is None:
        for c in df.columns:
            if df[c].nunique() <= 10 and c != text_col:
                label_col = c; break
    return text_col, label_col

def convert_labels(series):
    series = series.dropna().reset_index(drop=True)
    def normalize(val):
        s = str(val).strip()
        try:
            return int(float(s))
        except ValueError:
            pass
        return LABEL_MAPPING.get(s, LABEL_MAPPING.get(s.lower(), None))
    mapped = series.apply(normalize)
    if mapped.isna().any():
        unknown = series[mapped.isna()].unique().tolist()
        raise ValueError(f"Label tidak dikenal: {unknown}")
    return mapped.astype(int).to_numpy()

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (np.array(pred_a) == np.array(y_true))
    correct_b = (np.array(pred_b) == np.array(y_true))
    b = np.sum(~correct_a & correct_b)
    c = np.sum(correct_a & ~correct_b)
    denom = b + c
    if denom == 0:
        return 0.0, 1.0, int(b), int(c)
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1 - chi2.cdf(chi2_stat, df=1)
    return float(chi2_stat), float(p_val), int(b), int(c)

def build_bilstm(num_classes, max_words=5000, embedding_dim=64,
                 bilstm_units=64, max_length=50):
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(bilstm_units, dropout=0.2,
                           recurrent_dropout=0.2, return_sequences=False)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, label_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = plt.cm.Blues
    im   = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(label_names)),
           yticks=np.arange(len(label_names)),
           xticklabels=label_names,
           yticklabels=label_names,
           ylabel='Label Sebenarnya',
           xlabel='Prediksi Model',
           title='Confusion Matrix — BiLSTM')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_learning_curve(history_dict):
    acc     = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss    = history_dict['loss']
    val_loss= history_dict['val_loss']
    epochs  = range(1, len(acc) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, acc,     'b-o', markersize=3, label='Train Acc')
    ax1.plot(epochs, val_acc, 'r--o',markersize=3, label='Val Acc')
    ax1.set(title='Akurasi per Epoch', xlabel='Epoch', ylabel='Akurasi')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, loss,     'b-o', markersize=3, label='Train Loss')
    ax2.plot(epochs, val_loss, 'r--o',markersize=3, label='Val Loss')
    ax2.set(title='Loss per Epoch', xlabel='Epoch', ylabel='Loss')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle('BiLSTM Learning Curve', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 Menu Navigasi")
        page = st.radio(
            "Pilih halaman:",
            ["🏠 Beranda & Deteksi",
             "📊 Training Model",
             "📚 Edukasi Kesehatan Mental",
             "📋 Riwayat Deteksi",
             "ℹ️ Tentang Aplikasi"],
            label_visibility="collapsed",
        )
        st.divider()

        # Status model
        st.markdown("### 📌 Status Sistem")
        if st.session_state.get('is_trained'):
            st.success("✅ Model siap digunakan")
            perf = st.session_state.get('model_performance', {})
            if 'BiLSTM' in perf:
                acc = perf['BiLSTM']['accuracy'] * 100
                f1  = perf['BiLSTM']['f1'] * 100
                st.markdown(f"- Akurasi BiLSTM: **{acc:.1f}%**")
                st.markdown(f"- F1-Score: **{f1:.1f}%**")
        elif st.session_state.get('dataset_loaded'):
            st.warning("⚠️ Dataset dimuat, belum training")
        else:
            st.info("💡 Belum ada model. Upload dataset di Training.")

        st.divider()

        # NLP status
        st.markdown("### 🔧 Komponen Sistem")
        st.markdown(f"- Sastrawi: {'✅' if SASTRAWI_OK else '❌ (fallback)'}")
        st.markdown(f"- NLTK: {'✅' if NLTK_OK else '❌ (fallback)'}")
        st.markdown(f"- TensorFlow: {'✅' if TF_OK else '❌'}")

        st.divider()
        st.caption("© 2026 Pengabdian Masyarakat\nUniversitas — SMA Kota Medan")

    return page


# ═══════════════════════════════════════════════════════════════════
# HALAMAN BERANDA & DETEKSI
# ═══════════════════════════════════════════════════════════════════
def page_beranda():
    # Header
    st.markdown("""
    <div class="main-header">
      <h1>🧠 Deteksi Dini Risiko Kesehatan Mental</h1>
      <p>Pelatihan Melalui Media Sosial · Siswa SMA Kota Medan · Berbasis BiLSTM</p>
    </div>
    """, unsafe_allow_html=True)

    # Edukasi singkat di beranda
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:2rem">😊</div>
          <div class="metric-label" style="font-weight:600;color:#00b894;font-size:0.9rem">NORMAL</div>
          <div class="metric-label">Kondisi mental sehat dan baik</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:2rem">😢</div>
          <div class="metric-label" style="font-weight:600;color:#d63031;font-size:0.9rem">DEPRESI</div>
          <div class="metric-label">Risiko depresi terdeteksi</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:2rem">😰</div>
          <div class="metric-label" style="font-weight:600;color:#e17055;font-size:0.9rem">CEMAS</div>
          <div class="metric-label">Risiko kecemasan terdeteksi</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Kolom kiri: input deteksi | kanan: panduan
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Input Teks Media Sosial</div>', unsafe_allow_html=True)

        # Tombol contoh
        st.markdown("**Coba contoh teks:**")
        cols_ex = st.columns(3)
        for i, (lbl, txt) in enumerate(DEMO_EXAMPLES[:3]):
            with cols_ex[i % 3]:
                if st.button(lbl, key=f"ex_{i}", use_container_width=True):
                    st.session_state['input_text'] = txt

        # Baris kedua contoh
        cols_ex2 = st.columns(3)
        for i, (lbl, txt) in enumerate(DEMO_EXAMPLES[3:]):
            with cols_ex2[i % 3]:
                if st.button(lbl, key=f"ex2_{i}", use_container_width=True):
                    st.session_state['input_text'] = txt

        st.divider()

        input_text = st.text_area(
            "Ketik atau paste teks dari media sosial di sini:",
            value=st.session_state.get('input_text', ''),
            height=140,
            placeholder="Contoh: Aku merasa sangat sedih hari ini, tidak ada semangat sama sekali...",
            key="main_input"
        )

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            detect_btn = st.button(
                "🔍 DETEKSI RISIKO MENTAL",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.get('is_trained', False),
            )
        with col_btn2:
            if st.button("🗑️ Hapus", use_container_width=True):
                st.session_state['input_text'] = ''
                st.rerun()

        if not st.session_state.get('is_trained', False):
            st.info("💡 Upload dataset dan latih model terlebih dahulu di menu **Training Model**.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Proses deteksi
        if detect_btn and input_text.strip():
            run_detection(input_text.strip())

    with right:
        st.markdown("""
        <div class="card">
          <div class="card-title">📌 Cara Menggunakan</div>
          <ol style="font-size:0.88rem; color:#4a5568; padding-left:1.2rem; line-height:1.8">
            <li>Kunjungi menu <b>Training Model</b></li>
            <li>Upload dataset CSV (teks + label)</li>
            <li>Klik tombol <b>Latih Model</b></li>
            <li>Kembali ke Beranda</li>
            <li>Ketik atau paste teks media sosial</li>
            <li>Klik <b>Deteksi Risiko Mental</b></li>
            <li>Baca hasil dan saran yang muncul</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-mental">
          <b>⚠️ Penting!</b><br>
          Aplikasi ini adalah alat bantu edukasi, bukan diagnosis medis. 
          Jika kamu atau temanmu membutuhkan bantuan, segera hubungi 
          <b>guru BK</b> atau <b>hotline 119 ext 8</b>.
        </div>
        """, unsafe_allow_html=True)


def run_detection(text):
    stemmer = load_stemmer()
    cleaned = preprocess_text(text, stemmer=stemmer)

    model_key = st.session_state.get('active_model', 'bilstm')
    is_trained = st.session_state.get('is_trained', False)

    if not is_trained:
        st.warning("Model belum dilatih!")
        return

    try:
        tokenizer  = st.session_state['tokenizer']
        max_length = st.session_state.get('max_length', 50)

        if model_key == 'bilstm' and 'bilstm_model' in st.session_state:
            model = st.session_state['bilstm_model']
            seq    = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=max_length, padding='post')
            probs  = model.predict(padded, verbose=0)[0]
            model_name = "BiLSTM"
        elif model_key == 'svm' and 'svm_model' in st.session_state:
            tfidf = st.session_state['tfidf']
            vec   = tfidf.transform([cleaned])
            probs = st.session_state['svm_model'].predict_proba(vec)[0]
            model_name = "SVM"
        elif model_key == 'nb' and 'nb_model' in st.session_state:
            tfidf = st.session_state['tfidf']
            vec   = tfidf.transform([cleaned])
            probs = st.session_state['nb_model'].predict_proba(vec)[0]
            model_name = "Naïve Bayes"
        else:
            model = st.session_state['bilstm_model']
            seq    = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=max_length, padding='post')
            probs  = model.predict(padded, verbose=0)[0]
            model_name = "BiLSTM"

        idx      = int(np.argmax(probs))
        conf     = float(probs[idx])
        n_cls    = st.session_state.get('num_classes', 3)
        lbl_names = st.session_state.get('active_label_names',
                                          ALL_LABEL_NAMES[:n_cls])

        # Map index ke label global
        unique_labels = st.session_state.get('unique_labels', [0,1,2])
        real_label    = unique_labels[idx] if idx < len(unique_labels) else idx
        cfg           = LABEL_CONFIG.get(real_label, LABEL_CONFIG[0])

        # CSS class
        cls_map  = {0: 'normal', 1: 'depresi', 2: 'cemas'}
        res_cls  = cls_map.get(real_label, 'normal')

        st.markdown(f"""
        <div class="result-{res_cls}">
          <div class="result-label">{cfg['name']}</div>
          <div class="result-desc">{cfg['desc']}</div>
          <div style="margin-top:0.6rem; font-size:0.85rem">
            🤖 Model: <b>{model_name}</b> &nbsp;|&nbsp;
            🎯 Keyakinan: <b>{conf*100:.1f}%</b>
          </div>
          <div class="result-advice">💬 {cfg['advice']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Bar probabilitas
        st.markdown("**Distribusi probabilitas:**")
        for i, p in enumerate(probs):
            if i < len(lbl_names):
                lname = lbl_names[i]
                st.progress(float(p), text=f"{lname}: {p*100:.1f}%")

        # Simpan ke riwayat
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].insert(0, {
            'timestamp':  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'teks':       text[:200],
            'hasil':      cfg['name'],
            'keyakinan':  f"{conf*100:.1f}%",
            'model':      model_name,
        })
        # Batasi 50 riwayat
        st.session_state['history'] = st.session_state['history'][:50]

    except Exception as e:
        st.error(f"Error deteksi: {str(e)}")


# ═══════════════════════════════════════════════════════════════════
# HALAMAN TRAINING MODEL
# ═══════════════════════════════════════════════════════════════════
def page_training():
    st.markdown("## 📊 Training Model BiLSTM")
    st.markdown("Upload dataset CSV, lakukan preprocessing, lalu latih model deteksi.")

    # ─── Step indicator ──────────────────────────────────────────
    s1 = "step-done" if st.session_state.get('dataset_loaded') else "step-active"
    s2 = "step-done" if st.session_state.get('preprocessed') else (
         "step-active" if st.session_state.get('dataset_loaded') else "step-pending")
    s3 = "step-done" if st.session_state.get('is_trained') else (
         "step-active" if st.session_state.get('preprocessed') else "step-pending")
    st.markdown(f"""
    <div style="display:flex;gap:1rem;align-items:center;margin-bottom:1.5rem;flex-wrap:wrap">
      <span class="{s1}">1. Upload Dataset</span>
      <span style="color:#a0aec0">→</span>
      <span class="{s2}">2. Preprocessing</span>
      <span style="color:#a0aec0">→</span>
      <span class="{s3}">3. Latih Model</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    # ─── KIRI: Upload + Preprocessing + Hyperparameter ───────────
    with left:

        # ── 1. Upload ──
        with st.expander("📂 1. Upload Dataset", expanded=not st.session_state.get('dataset_loaded')):
            st.markdown("""
            Format CSV dengan dua kolom:
            - **Kolom teks**: teks media sosial (tweet/komentar)
            - **Kolom label**: `Normal`/`Depresi`/`Cemas` atau `0`/`1`/`2`
            """)
            uploaded = st.file_uploader("Pilih file CSV", type=['csv'])
            if uploaded:
                try:
                    df = try_read_csv(uploaded)
                    text_col, label_col = detect_columns(df)
                    if not text_col or not label_col:
                        st.error("Kolom teks atau label tidak ditemukan!")
                    else:
                        df = df.rename(columns={text_col: 'text', label_col: 'label'})
                        df = df.dropna(subset=['label']).reset_index(drop=True)
                        labels = convert_labels(df['label'])
                        df = df.iloc[:len(labels)].copy()
                        df['label'] = labels

                        unique_labels = sorted(df['label'].unique().tolist())
                        n_cls = len(unique_labels)
                        active_names = [ALL_LABEL_NAMES[i] for i in unique_labels
                                        if i < len(ALL_LABEL_NAMES)]

                        st.session_state['dataset']       = df
                        st.session_state['dataset_loaded']= True
                        st.session_state['preprocessed']  = False
                        st.session_state['is_trained']    = False
                        st.session_state['unique_labels'] = unique_labels
                        st.session_state['num_classes']   = n_cls
                        st.session_state['active_label_names'] = active_names
                        st.session_state['text_col']      = text_col
                        st.session_state['label_col']     = label_col

                        st.success(f"✅ {len(df)} data berhasil dimuat | {n_cls} kelas")
                        counts = df['label'].value_counts().sort_index()
                        for lbl in unique_labels:
                            lname = ALL_LABEL_NAMES[lbl] if lbl < 3 else f"Kelas {lbl}"
                            cnt   = counts.get(lbl, 0)
                            pct   = cnt / len(df) * 100
                            st.markdown(f"- **{lname}**: {cnt} data ({pct:.1f}%)")

                        if len(df) < 200:
                            st.warning(f"⚠️ Dataset hanya {len(df)} sampel. Disarankan minimal 200 data.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # ── 2. Preprocessing ──
        with st.expander("⚙️ 2. Preprocessing Teks",
                         expanded=st.session_state.get('dataset_loaded', False)
                                  and not st.session_state.get('preprocessed', False)):
            st.markdown("Pembersihan teks: hapus URL, mention, simbol → tokenisasi → stemming.")
            use_stem = st.checkbox("Gunakan Stemming (Sastrawi)", value=True)
            if st.button("🔄 Mulai Preprocessing", type="primary",
                         disabled=not st.session_state.get('dataset_loaded')):
                df      = st.session_state['dataset'].copy()
                stemmer = load_stemmer()
                bar     = st.progress(0, text="Memulai preprocessing...")
                cleaned = []
                total   = len(df)
                for i, txt in enumerate(df['text']):
                    cleaned.append(preprocess_text(str(txt), stemmer=stemmer,
                                                   use_stemming=use_stem))
                    if i % max(1, total // 20) == 0:
                        bar.progress(i / total,
                                     text=f"Preprocessing... {i}/{total}")
                bar.progress(1.0, text="Selesai!")
                df['cleaned_text'] = cleaned
                df = df[df['cleaned_text'].str.strip().str.len() > 0].reset_index(drop=True)
                st.session_state['dataset']      = df
                st.session_state['preprocessed'] = True
                if len(df) > 0:
                    st.success(f"✅ {len(df)} data siap. Contoh: *{df['cleaned_text'].iloc[0][:60]}...*")

        # ── 3. Hyperparameter ──
        with st.expander("🧠 3. Pengaturan Model",
                         expanded=False):
            st.markdown("**Hyperparameter BiLSTM** (biarkan default jika tidak yakin)")
            c1, c2 = st.columns(2)
            with c1:
                max_words  = st.number_input("Max Vocabulary",  500,  20000, 5000, step=500)
                max_length = st.number_input("Max Panjang Teks", 20,   200,  50,   step=10)
                lstm_units = st.number_input("LSTM Units/arah",  32,   256,  64,   step=32)
            with c2:
                epochs     = st.number_input("Maks Epoch",       5,    100,  30,   step=5)
                batch_size = st.number_input("Batch Size",       8,    128,  32,   step=8)
                patience   = st.number_input("Early Stop Patience", 2, 20,   5,    step=1)

            run_baseline = st.checkbox("Jalankan model baseline (SVM & Naïve Bayes)", value=True)
            st.session_state['hp'] = {
                'max_words': int(max_words),
                'max_length': int(max_length),
                'lstm_units': int(lstm_units),
                'epochs': int(epochs),
                'batch_size': int(batch_size),
                'patience': int(patience),
                'run_baseline': run_baseline,
            }

    # ─── KANAN: Tombol train + Hasil ─────────────────────────────
    with right:
        st.markdown("### 🏋️ Latih Model")

        if not TF_OK:
            st.error("❌ TensorFlow tidak tersedia. Hanya SVM dan Naïve Bayes yang bisa digunakan.")

        can_train = st.session_state.get('preprocessed', False)
        if st.button("🚀 LATIH SEMUA MODEL", type="primary",
                     use_container_width=True, disabled=not can_train):
            run_training()

        if not can_train:
            st.info("Selesaikan langkah 1 & 2 terlebih dahulu.")

        # ── Pilih model aktif ──
        if st.session_state.get('is_trained'):
            st.divider()
            st.markdown("**Model untuk deteksi:**")
            options = ['bilstm'] if TF_OK else []
            options += ['svm', 'nb']
            labels  = {'bilstm': '🧠 BiLSTM (terbaik)',
                       'svm':    '⚙️ SVM', 'nb': '📊 Naïve Bayes'}
            active  = st.radio("Pilih model:",
                               [o for o in options],
                               format_func=lambda x: labels.get(x, x),
                               index=0)
            st.session_state['active_model'] = active

        # ── Tampilkan performa ──
        if st.session_state.get('model_performance'):
            display_performance()


def run_training():
    df      = st.session_state['dataset'].copy()
    hp      = st.session_state.get('hp', {})
    max_words  = hp.get('max_words', 5000)
    max_length = hp.get('max_length', 50)
    lstm_units = hp.get('lstm_units', 64)
    epochs_max = hp.get('epochs', 30)
    batch_size = hp.get('batch_size', 32)
    patience   = hp.get('patience', 5)
    run_base   = hp.get('run_baseline', True)

    X_text = df['cleaned_text'].astype(str).to_numpy()
    y      = df['label'].to_numpy()

    unique_labels = sorted(np.unique(y).tolist())
    n_cls = len(unique_labels)
    label_to_idx  = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_idx = np.array([label_to_idx[l] for l in y])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_text, y_idx, test_size=0.2, random_state=42, stratify=y_idx)

    results = {}
    progress_bar = st.progress(0, text="Memulai training...")

    # ── TF-IDF untuk baseline ──
    tfidf = TfidfVectorizer(max_features=max_words, ngram_range=(1, 2))
    X_tr_tfidf = tfidf.fit_transform(X_train_raw)
    X_te_tfidf = tfidf.transform(X_test_raw)

    # ── Naïve Bayes ──
    if run_base:
        progress_bar.progress(0.1, text="Training Naïve Bayes...")
        t0  = time.time()
        nb  = MultinomialNB()
        nb.fit(X_tr_tfidf, y_train)
        y_pred_nb = nb.predict(X_te_tfidf)
        results['NB'] = {
            'accuracy':  accuracy_score(y_test, y_pred_nb),
            'f1':        f1_score(y_test, y_pred_nb, average='weighted', zero_division=0),
            'precision': precision_score(y_test, y_pred_nb, average='weighted', zero_division=0),
            'recall':    recall_score(y_test, y_pred_nb, average='weighted', zero_division=0),
            'training_time': time.time() - t0,
        }
        st.session_state['nb_model']   = nb
        st.session_state['nb_preds']   = y_pred_nb

        # ── SVM ──
        progress_bar.progress(0.2, text="Training SVM...")
        t0  = time.time()
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_tr_tfidf, y_train)
        y_pred_svm = svm.predict(X_te_tfidf)
        results['SVM'] = {
            'accuracy':  accuracy_score(y_test, y_pred_svm),
            'f1':        f1_score(y_test, y_pred_svm, average='weighted', zero_division=0),
            'precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
            'recall':    recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
            'training_time': time.time() - t0,
        }
        st.session_state['svm_model']  = svm
        st.session_state['svm_preds']  = y_pred_svm
        st.session_state['tfidf']      = tfidf

    # ── BiLSTM ──
    if TF_OK:
        progress_bar.progress(0.35, text="Menyiapkan tokenizer BiLSTM...")
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train_raw)

        X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_raw),
                                 maxlen=max_length, padding='post', truncating='post')
        X_te_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_raw),
                                 maxlen=max_length, padding='post', truncating='post')

        cw_arr  = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw_dict = {i: w for i, w in enumerate(cw_arr)}

        progress_bar.progress(0.4, text="Training BiLSTM (harap tunggu)...")
        t0    = time.time()
        model = build_bilstm(n_cls, max_words, 64, lstm_units, max_length)
        es    = EarlyStopping(monitor='val_loss', patience=patience,
                              restore_best_weights=True, verbose=0)
        hist  = model.fit(X_tr_seq, y_train,
                          epochs=epochs_max, batch_size=batch_size,
                          validation_split=0.2, callbacks=[es],
                          verbose=0, class_weight=cw_dict)
        ep_done = len(hist.history['accuracy'])
        progress_bar.progress(0.85, text="Evaluasi BiLSTM...")

        y_pred_bilstm = np.argmax(model.predict(X_te_seq, verbose=0), axis=1)
        cm = confusion_matrix(y_test, y_pred_bilstm)
        report = classification_report(y_test, y_pred_bilstm,
                                       target_names=st.session_state.get('active_label_names',
                                                                          ALL_LABEL_NAMES[:n_cls]),
                                       output_dict=True, zero_division=0)
        results['BiLSTM'] = {
            'accuracy':     accuracy_score(y_test, y_pred_bilstm),
            'f1':           f1_score(y_test, y_pred_bilstm, average='weighted', zero_division=0),
            'f1_macro':     f1_score(y_test, y_pred_bilstm, average='macro',    zero_division=0),
            'precision':    precision_score(y_test, y_pred_bilstm, average='weighted', zero_division=0),
            'recall':       recall_score(y_test, y_pred_bilstm, average='weighted', zero_division=0),
            'training_time':time.time() - t0,
            'epochs_done':  ep_done,
            'cm':           cm.tolist(),
            'report':       report,
            'history':      {k: [float(v) for v in vals]
                             for k, vals in hist.history.items()},
        }
        st.session_state['bilstm_model']   = model
        st.session_state['bilstm_preds']   = y_pred_bilstm
        st.session_state['tokenizer']      = tokenizer
        st.session_state['max_length']     = max_length

    # McNemar
    if TF_OK and run_base and 'BiLSTM' in results:
        progress_bar.progress(0.92, text="McNemar Test...")
        pb = st.session_state['bilstm_preds']
        ps = st.session_state['svm_preds']
        pn = st.session_state['nb_preds']
        mc = {}
        for name, pa, pbb in [('BiLSTM vs SVM', pb, ps),
                               ('BiLSTM vs NaïveBayes', pb, pn)]:
            c2s, pv, b, c = mcnemar_test(y_test, pa, pbb)
            mc[name] = {'chi2': c2s, 'p_value': pv, 'b': b, 'c': c,
                        'significant': pv < 0.05}
        results['mcnemar'] = mc

    results['y_test'] = y_test.tolist()
    st.session_state['model_performance'] = results
    st.session_state['is_trained']        = True
    st.session_state['unique_labels']     = unique_labels

    progress_bar.progress(1.0, text="✅ Training selesai!")
    st.success("🎉 Semua model berhasil dilatih!")
    st.rerun()


def display_performance():
    r = st.session_state['model_performance']
    st.divider()
    st.markdown("### 📈 Hasil Evaluasi Model")

    # Tabel performa
    rows = []
    for k, lbl in [('BiLSTM', '🧠 BiLSTM'), ('SVM', '⚙️ SVM'), ('NB', '📊 Naïve Bayes')]:
        if k in r:
            v = r[k]
            rows.append({
                'Model':      lbl,
                'Akurasi':    f"{v['accuracy']*100:.2f}%",
                'F1-Score':   f"{v['f1']*100:.2f}%",
                'Presisi':    f"{v['precision']*100:.2f}%",
                'Recall':     f"{v['recall']*100:.2f}%",
                'Waktu (s)':  f"{v.get('training_time',0):.1f}s",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Confusion Matrix
    if 'BiLSTM' in r and 'cm' in r['BiLSTM']:
        cm  = np.array(r['BiLSTM']['cm'])
        lbs = st.session_state.get('active_label_names', ALL_LABEL_NAMES)
        fig = plot_confusion_matrix(cm, lbs)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Learning Curve
    if 'BiLSTM' in r and 'history' in r['BiLSTM']:
        fig2 = plot_learning_curve(r['BiLSTM']['history'])
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Export CSV
    rows_exp = []
    for k, lbl in [('BiLSTM', 'BiLSTM'), ('SVM', 'SVM'), ('NB', 'NaiveBayes')]:
        if k in r:
            v = r[k]
            rows_exp.append({
                'Model':           lbl,
                'Accuracy_%':      round(v['accuracy']*100, 2),
                'F1_Weighted_%':   round(v['f1']*100, 2),
                'Precision_%':     round(v['precision']*100, 2),
                'Recall_%':        round(v['recall']*100, 2),
                'Training_Time_s': round(v.get('training_time',0), 2),
            })
    if rows_exp:
        csv_bytes = pd.DataFrame(rows_exp).to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil (CSV)", csv_bytes,
                           f"hasil_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# HALAMAN EDUKASI
# ═══════════════════════════════════════════════════════════════════
def page_edukasi():
    st.markdown("## 📚 Edukasi Kesehatan Mental")
    st.markdown("Pelajari tentang kesehatan mental dan cara menjaganya.")

    for judul, konten in EDUKASI_KONTEN.items():
        with st.expander(f"{konten['icon']}  {judul}", expanded=False):
            st.markdown(konten['isi'])

    st.divider()
    st.markdown("### 🧪 Cek Pemahaman Kamu")
    st.markdown("Jawab pertanyaan sederhana ini untuk menguji pemahamanmu:")

    q1 = st.radio(
        "1. Apa yang sebaiknya dilakukan jika teman kamu terlihat sedih dan menyendiri terus-menerus?",
        ["Biarkan saja, itu urusan dia sendiri",
         "Dekati dengan empati, ajak bicara, dan sarankan berbicara ke guru BK",
         "Langsung ceritakan ke semua teman",
         "Tidak perlu dilakukan apa-apa"],
        index=None
    )
    if q1:
        if "Dekati" in q1:
            st.success("✅ Tepat sekali! Empati dan dukungan sosial sangat penting dalam kesehatan mental.")
        else:
            st.error("❌ Coba lagi. Teman yang sedang bermasalah membutuhkan dukungan dan perhatian.")

    q2 = st.radio(
        "2. Media sosial yang berlebihan dapat menyebabkan...",
        ["Peningkatan produktivitas belajar",
         "Kecemasan dan depresi akibat perbandingan diri",
         "Tidak ada dampak apapun",
         "Selalu memberikan dampak positif"],
        index=None
    )
    if q2:
        if "Kecemasan" in q2:
            st.success("✅ Benar! Penggunaan media sosial yang berlebihan terbukti berhubungan dengan risiko kecemasan.")
        else:
            st.error("❌ Belum tepat. Penggunaan media sosial berlebihan bisa mempengaruhi kesehatan mental.")


# ═══════════════════════════════════════════════════════════════════
# HALAMAN RIWAYAT
# ═══════════════════════════════════════════════════════════════════
def page_riwayat():
    st.markdown("## 📋 Riwayat Deteksi")

    history = st.session_state.get('history', [])
    if not history:
        st.info("Belum ada riwayat deteksi. Lakukan deteksi di halaman Beranda.")
        return

    st.markdown(f"Total deteksi: **{len(history)}** entri")

    df_hist = pd.DataFrame(history)

    # Statistik
    col1, col2, col3 = st.columns(3)
    counts = df_hist['hasil'].value_counts()
    with col1:
        normal_ct = sum(1 for h in history if 'NORMAL' in h['hasil'].upper())
        st.metric("😊 Normal", normal_ct)
    with col2:
        dep_ct = sum(1 for h in history if 'DEPRESI' in h['hasil'].upper())
        st.metric("😢 Depresi", dep_ct)
    with col3:
        cem_ct = sum(1 for h in history if 'CEMAS' in h['hasil'].upper())
        st.metric("😰 Cemas", cem_ct)

    st.divider()
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    col_dl, col_cl = st.columns([2, 1])
    with col_dl:
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Riwayat (CSV)", csv,
                           f"riwayat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           "text/csv", use_container_width=True)
    with col_cl:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state['history'] = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# HALAMAN TENTANG
# ═══════════════════════════════════════════════════════════════════
def page_tentang():
    st.markdown("## ℹ️ Tentang Aplikasi")

    st.markdown("""
    <div class="card">
      <div class="card-title">🎯 Tujuan Aplikasi</div>
      <p>Aplikasi ini dikembangkan sebagai media pelatihan dalam kegiatan <b>Pengabdian Masyarakat</b> 
      bertema <em>"Pelatihan Deteksi Dini Risiko Kesehatan Mental Melalui Media Sosial bagi Siswa SMA 
      di Kota Medan"</em>.</p>
      <p>Tujuan utama:</p>
      <ul>
        <li>Mengedukasi siswa tentang kesehatan mental</li>
        <li>Mendemonstrasikan bagaimana AI/machine learning dapat membantu deteksi dini</li>
        <li>Meningkatkan literasi digital dan kesehatan mental pada remaja</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
          <div class="card-title">🧠 Teknologi yang Digunakan</div>
          <ul style="font-size:0.88rem;color:#4a5568;line-height:2">
            <li><b>BiLSTM</b> — Bidirectional LSTM (model utama)</li>
            <li><b>SVM</b> — Support Vector Machine (baseline)</li>
            <li><b>Naïve Bayes</b> — model probabilistik (baseline)</li>
            <li><b>Sastrawi</b> — stemming Bahasa Indonesia</li>
            <li><b>NLTK</b> — stopwords removal</li>
            <li><b>Streamlit</b> — framework web interaktif</li>
            <li><b>TensorFlow/Keras</b> — deep learning</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
          <div class="card-title">📐 Arsitektur BiLSTM</div>
          <pre style="font-size:0.78rem;color:#4a5568;background:#f7fafc;
                      padding:0.8rem;border-radius:8px;overflow-x:auto">
Embedding(5000, dim=64)
    ↓
Bidirectional LSTM(64)
  ├─ Forward  LSTM: 64 unit
  └─ Backward LSTM: 64 unit
    → Concat output: 128
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(n_kelas, Softmax)
          </pre>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">⚠️ Disclaimer</div>
      <p style="font-size:0.88rem;color:#4a5568">
      Aplikasi ini adalah <b>alat bantu edukasi</b>, bukan pengganti diagnosis medis atau psikologis profesional. 
      Hasil deteksi harus diinterpretasikan hanya sebagai indikasi awal dan tidak boleh dijadikan satu-satunya 
      dasar pengambilan keputusan terkait kondisi kesehatan mental seseorang. 
      Jika ada kekhawatiran serius tentang kesehatan mental, segera hubungi profesional kesehatan mental.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
      © 2026 · Pengabdian Masyarakat · Deteksi Dini Kesehatan Mental · SMA Kota Medan<br>
      Dikembangkan dengan ❤️ menggunakan Python + Streamlit + TensorFlow
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # Inisialisasi session state
    for key, default in [
        ('is_trained', False), ('dataset_loaded', False),
        ('preprocessed', False), ('history', []),
        ('active_model', 'bilstm'), ('input_text', ''),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    page = render_sidebar()

    if "Beranda" in page:
        page_beranda()
    elif "Training" in page:
        page_training()
    elif "Edukasi" in page:
        page_edukasi()
    elif "Riwayat" in page:
        page_riwayat()
    elif "Tentang" in page:
        page_tentang()


if __name__ == "__main__":
    main()
