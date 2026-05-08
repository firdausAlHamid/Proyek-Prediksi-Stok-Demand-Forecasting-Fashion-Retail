# 📦 Proyek Prediksi Stok & Demand Forecasting — Fashion Retail

Dashboard Machine Learning lengkap untuk memprediksi demand produk retail menggunakan **Linear Regression**, **Random Forest**, dan **KMeans Clustering**, dilengkapi dashboard **Streamlit** interaktif 6 halaman.

---

## 📁 Struktur Folder

```
📁 Proyek Prediksi Stok & Demand Forecasting Fashion Retail/
├── 📄 sales_data.csv                ← Dataset ASLI (76.000 records)
├── 📄 sales_data_enriched.csv       ← Hasil feature_engineering.py (auto-generated)
├── 📄 clustering_result.csv         ← Hasil clustering_mod.py (auto-generated)
├── 🐍 feature_engineering.py        ← Feature Engineering
├── 🐍 LINEAR_mod.py                 ← Model Linear Regression (Clothing)
├── 🐍 FOREST_mod.py                 ← Model Random Forest (Semua Kategori)
├── 🐍 clustering_mod.py             ← KMeans Clustering Produk
├── 🐍 app.py                        ← Dashboard Streamlit (6 halaman)
├── 🐍 run_app.py                    ← Script deployment (optional ngrok)
├── 📄 requirements.txt              ← Dependensi Python
├── 🤖 linear_model.pkl              ← Model tersimpan (auto-generated)
├── 🤖 forest_model.pkl              ← Model tersimpan (auto-generated)
├── 🤖 cluster_model.pkl             ← Model tersimpan (auto-generated)
├── 🤖 cluster_scaler.pkl            ← Scaler tersimpan (auto-generated)
└── 🖼️ *.png                         ← Semua output visualisasi
```

---

## 📊 Dataset

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `Date` | datetime | 2022-01-01 s/d 2024-01-30 |
| `Store ID` | kategorikal | S001–S005 |
| `Product ID` | kategorikal | P0001–P0020 |
| `Category` | kategorikal | Groceries, Clothing, Electronics, Furniture, Toys |
| `Region` | kategorikal | North, South, East, West |
| `Inventory Level` | numerik | Stok tersedia |
| `Units Sold` | numerik | Unit terjual |
| `Price` | numerik | Harga produk |
| `Discount` | numerik | Diskon (%) |
| `Weather Condition` | kategorikal | Cloudy, Rainy, Sunny, Snowy |
| `Promotion` | binary | 0/1 (ada promosi?) |
| `Competitor Pricing` | numerik | Harga kompetitor |
| `Seasonality` | kategorikal | Winter, Spring, Summer, Autumn |
| `Epidemic` | binary | 0/1 (kondisi epidemi?) |
| `Demand` | numerik | **TARGET** — demand aktual |

---

## 🚀 Cara Menjalankan Proyek

### 1. Persiapan Environment

```bash
# Buat virtual environment (opsional tapi disarankan)
python -m venv venv

# Aktifkan (Windows)
venv\Scripts\activate

# Aktifkan (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

> ⚠️ **Catatan:** Jika terjadi konflik versi, coba tanpa pin versi:
> ```bash
> pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib pyngrok openpyxl
> ```

### 3. Feature Engineering

```bash
python feature_engineering.py
```

**Output:** `sales_data_enriched.csv` (dataset + 11 fitur baru)

---

### 4. Training Model Linear Regression

```bash
python LINEAR_mod.py
```

**Output:**
- `linear_model.pkl` — model tersimpan
- `outlier_linear.png` — visualisasi outlier
- `actual_vs_pred_linear.png` — Actual vs Predicted
- `residual_linear.png` — Residual plot
- `residual_dist_linear.png` — Distribusi residuals

---

### 5. Training Model Random Forest

```bash
python FOREST_mod.py
```

> ⏳ **Estimasi waktu: 5–15 menit** (GridSearchCV dengan 36 kombinasi parameter, CV=3)

**Output:**
- `forest_model.pkl` — best model tersimpan
- `outlier_forest.png`, `actual_vs_pred_forest.png`
- `feature_importance_forest.png`, `residual_dist_forest.png`
- `demand_per_category.png`

---

### 6. Clustering Produk

```bash
python clustering_mod.py
```

**Output:**
- `clustering_result.csv` — hasil clustering + interpretasi
- `cluster_model.pkl`, `cluster_scaler.pkl`
- `elbow_method.png`, `silhouette_score.png`, `cluster_scatter.png`

---

### 7. Jalankan Dashboard Streamlit

```bash
streamlit run app.py
```

Buka browser: **http://localhost:8501**

Atau gunakan script deployment:

```bash
python run_app.py
```

---

## 📱 Fitur Dashboard (6 Halaman)

| Halaman | Deskripsi |
|---------|-----------|
| 📊 Overview | Metric cards, bar chart, pie chart distribusi demand |
| 📈 Trend Penjualan | Line chart harian, heatmap bulan×tahun, trend per kategori |
| 🧩 Clustering Produk | Scatter plot cluster, tabel interpretasi, elbow & silhouette chart |
| 📉 Evaluasi Model | Perbandingan R², RMSE, MAE, MAPE — Linear vs Random Forest |
| 🔮 Prediksi Demand | Form input prediksi dengan CI 95% dari kedua model |
| 📦 Rekomendasi Stok | Safety Stock, Reorder Point, Recommended Order Qty per produk |

---

## 📈 Urutan Eksekusi Lengkap

```
1️⃣  pip install -r requirements.txt
2️⃣  python feature_engineering.py
3️⃣  python LINEAR_mod.py
4️⃣  python FOREST_mod.py          ← estimasi 5–15 menit
5️⃣  python clustering_mod.py
6️⃣  streamlit run app.py
```

---

## ⚠️ Catatan Penting

- Semua nama kolom harus **EXACT MATCH** dengan dataset:
  - ✅ `Units Sold`, `Inventory Level`, `Weather Condition`, `Store ID`, `Product ID`
- Model Random Forest menggunakan **semua kategori** (5 kategori)
- Model Linear Regression hanya untuk **Clothing**
- Filter sidebar pada dashboard berlaku **global** untuk semua halaman

---

## 🛠️ Teknologi yang Digunakan

| Library | Versi | Fungsi |
|---------|-------|--------|
| pandas | 2.1.4 | Manipulasi data |
| numpy | 1.26.2 | Komputasi numerik |
| scikit-learn | 1.3.2 | Machine Learning |
| matplotlib | 3.8.2 | Visualisasi |
| seaborn | 0.13.0 | Visualisasi statistik |
| streamlit | 1.31.0 | Dashboard interaktif |
| joblib | 1.3.2 | Menyimpan/load model |

---

## 👨‍💻 Kontributor

Proyek ini dikembangkan untuk keperluan **Tugas Besar Data Mining & Analitik** dengan studi kasus demand forecasting retail fashion.
