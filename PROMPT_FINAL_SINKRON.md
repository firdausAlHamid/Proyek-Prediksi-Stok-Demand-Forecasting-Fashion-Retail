# 🚀 PROMPT LENGKAP: Proyek Prediksi Stok & Demand Forecasting Retail Store
## (Copy-paste seluruh isi file ini ke AI untuk membangun proyek dari NOL sampai DEPLOY)

---

## ⚠️ KONTEKS DATASET YANG DIGUNAKAN

Dataset yang digunakan adalah `sales_data.csv` — SUDAH ADA, TIDAK PERLU DI-GENERATE.

**Profil Dataset:**
- **Jumlah baris:** 76.000 records
- **Rentang tanggal:** 1 Januari 2022 — 30 Januari 2024
- **Missing values:** 0 (semua kolom lengkap)

**16 Kolom dengan nama EXACT:**

| No | Nama Kolom | Tipe | Nilai / Range |
|----|-----------|------|---------------|
| 1 | `Date` | datetime | 2022-01-01 s/d 2024-01-30 |
| 2 | `Store ID` | kategorikal | S001, S002, S003, S004, S005 |
| 3 | `Product ID` | kategorikal | P0001 s/d P0020 (20 produk) |
| 4 | `Category` | kategorikal | Groceries, Clothing, Electronics, Furniture, Toys |
| 5 | `Region` | kategorikal | North, South, East, West |
| 6 | `Inventory Level` | numerik | 0 — 2267 (mean 301) |
| 7 | `Units Sold` | numerik | 0 — 426 (mean 88.8) |
| 8 | `Units Ordered` | numerik | 0 — 1616 (mean 89) |
| 9 | `Price` | numerik | 4.74 — 228.03 (mean 67.7) |
| 10 | `Discount` | numerik | 0 — 25 (mean 9.1) |
| 11 | `Weather Condition` | kategorikal | Cloudy, Rainy, Sunny, Snowy |
| 12 | `Promotion` | numerik (binary) | 0 atau 1 (~33% promo) |
| 13 | `Competitor Pricing` | numerik | 4.29 — 261.22 (mean 69.5) |
| 14 | `Seasonality` | kategorikal | Winter, Spring, Summer, Autumn |
| 15 | `Epidemic` | numerik (binary) | 0 atau 1 (~20%) |
| 16 | `Demand` | numerik (TARGET) | 4 — 430 (mean 104.3) |

**Distribusi Penjualan per Category:**
- Groceries: 3.127.335 units (terbesar)
- Clothing: 1.150.873 units
- Furniture: 880.654 units
- Toys: 834.679 units
- Electronics: 757.335 units

---

## BAGIAN 1: FEATURE ENGINEERING SCRIPT (feature_engineering.py)

```
Buatkan file `feature_engineering.py` yang membaca `sales_data.csv` dan menambahkan fitur-fitur baru yang dibutuhkan oleh model, lalu menyimpannya ke `sales_data_enriched.csv`.

LANGKAH-LANGKAH:

1. LOAD `sales_data.csv` dengan pd.read_csv()
2. Konversi kolom `Date` ke datetime: pd.to_datetime(df['Date'])
3. EXTRACT FITUR WAKTU dari kolom `Date`:
   - `Year` = df['Date'].dt.year
   - `Month` = df['Date'].dt.month
   - `Week` = df['Date'].dt.isocalendar().week.astype(int)
   - `DayOfWeek` = df['Date'].dt.dayofweek  (0=Senin, 6=Minggu)
   - `IsWeekend` = (df['DayOfWeek'] >= 5).astype(int)
   - `Quarter` = df['Date'].dt.quarter

4. HITUNG FITUR TURUNAN:
   - `Revenue` = df['Units Sold'] * df['Price']
   - `Discount_Impact` = df['Discount'] * df['Price'] / 100
   - `Price_Diff` = df['Price'] - df['Competitor Pricing']
   - `Stock_Turnover` = df['Units Sold'] / (df['Inventory Level'] + 1)  # +1 menghindari div by zero
   - `Restock_Needed` = (df['Inventory Level'] < df['Units Sold']).astype(int)

5. PRINT ringkasan:
   - Shape dataset setelah enrichment
   - Nama-nama kolom baru
   - Preview 5 baris pertama

6. SIMPAN ke `sales_data_enriched.csv` (index=False)

Library: pandas, numpy
Semua komentar dan print statement dalam BAHASA INDONESIA.
```

---

## BAGIAN 2: MODEL LINEAR REGRESSION (LINEAR_mod.py)

```
Buatkan file `LINEAR_mod.py` — script standalone untuk model Linear Regression.

PENTING: Gunakan dataset `sales_data_enriched.csv` (hasil dari feature_engineering.py).
Jika file tersebut tidak ditemukan, fallback ke `sales_data.csv` dan lakukan feature engineering di dalam script.

SPESIFIKASI:

1. LOAD DATA dari `sales_data_enriched.csv`
   - Jika tidak ada, load `sales_data.csv` lalu buat fitur waktu (Year, Month, Week, DayOfWeek, IsWeekend, Quarter) langsung di script

2. FILTER hanya untuk 1 kategori spesifik: Category == "Clothing"
   - Print jumlah data setelah filter

3. FEATURE ENGINEERING:
   - Fitur (X): Month, Year, Week, DayOfWeek, IsWeekend, Price, Discount, Promotion, Inventory Level, Competitor Pricing, Epidemic, Seasonality (di-encode), Weather Condition (di-encode), Region (di-encode), Store ID (di-encode)
   - Target (y): Demand

4. OUTLIER REMOVAL menggunakan metode IQR pada kolom `Demand`:
   - Hitung Q1 (25th percentile) dan Q3 (75th percentile)
   - IQR = Q3 - Q1
   - Batas bawah = Q1 - 1.5 * IQR
   - Batas atas = Q3 + 1.5 * IQR
   - Hapus data di luar batas
   - Print jumlah data SEBELUM dan SESUDAH outlier removal
   - Visualisasi boxplot SEBELUM dan SESUDAH IQR removal (2 subplot berdampingan, simpan ke `outlier_linear.png`)

5. PISAHKAN KOLOM berdasarkan tipe:
   - Kolom kategorikal: Seasonality, Weather Condition, Region, Store ID
   - Kolom numerikal: Month, Year, Week, DayOfWeek, IsWeekend, Price, Discount, Promotion, Inventory Level, Competitor Pricing, Epidemic

6. PIPELINE dengan sklearn:
   - ColumnTransformer:
     - OneHotEncoder(handle_unknown='ignore') untuk kolom kategorikal
     - StandardScaler() untuk kolom numerikal
   - LinearRegression() sebagai estimator
   - Gabungkan dalam Pipeline: [('preprocessor', preprocessor), ('model', LinearRegression())]

7. TRAIN-TEST SPLIT: test_size=0.2, random_state=42

8. FIT pipeline pada X_train, y_train

9. EVALUASI pada X_test, y_test:
   - R² Score (Koefisien Determinasi)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error) — hitung manual: np.mean(np.abs((y_test - y_pred) / y_test)) * 100
   - Print SEMUA metrik dengan format rapi dan label BAHASA INDONESIA:
     "========================================"
     "  EVALUASI MODEL LINEAR REGRESSION"
     "  (Kategori: Clothing)"
     "========================================"
     "  R² Score           : 0.XXXX"
     "  MSE                : XXX.XX"
     "  RMSE               : XX.XX"
     "  MAE                : XX.XX"
     "  MAPE               : XX.XX%"
     "========================================"

10. VISUALISASI (simpan semua sebagai file gambar):
    a. Scatter plot: Actual vs Predicted (`actual_vs_pred_linear.png`)
       - Sumbu X = y_test (Actual), Sumbu Y = y_pred (Predicted)
       - Garis diagonal merah (perfect prediction line)
       - Title: "Actual vs Predicted — Linear Regression (Clothing)"
    b. Residual plot (`residual_linear.png`)
       - Sumbu X = y_pred, Sumbu Y = residuals (y_test - y_pred)
       - Garis horizontal di y=0
       - Title: "Residual Plot — Linear Regression"
    c. Distribution of Residuals — histogram (`residual_dist_linear.png`)

11. CONFIDENCE INTERVAL 95%:
    - Hitung standard error dari residuals
    - CI = y_pred ± 1.96 * std_error
    - Print contoh 5 prediksi pertama beserta CI-nya

12. YEAR-ON-YEAR (YoY) Analysis:
    - Hitung rata-rata Demand per Year
    - Print tabel YoY growth dalam persen

13. SAVE MODEL: joblib.dump(pipeline, 'linear_model.pkl')
    - Print konfirmasi: "Model Linear Regression berhasil disimpan ke linear_model.pkl"

Semua komentar dan print statement dalam BAHASA INDONESIA.
Library: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
```

---

## BAGIAN 3: MODEL RANDOM FOREST (FOREST_mod.py)

```
Buatkan file `FOREST_mod.py` — script standalone untuk model Random Forest Regressor.

PENTING: Gunakan dataset `sales_data_enriched.csv` (hasil dari feature_engineering.py).
Jika file tersebut tidak ditemukan, fallback ke `sales_data.csv` dan lakukan feature engineering di dalam script.

SPESIFIKASI:

1. LOAD DATA dari `sales_data_enriched.csv` (SEMUA kategori, TIDAK di-filter)
   - Print shape dataset

2. FEATURE ENGINEERING:
   - Fitur (X): Month, Year, Week, DayOfWeek, IsWeekend, Price, Discount, Promotion, Inventory Level, Competitor Pricing, Epidemic, Seasonality (di-encode), Weather Condition (di-encode), Region (di-encode), Store ID (di-encode), Category (di-encode), Product ID (di-encode)
   - Target (y): Demand

3. OUTLIER REMOVAL menggunakan metode IQR pada kolom `Demand`:
   - Sama seperti LINEAR_mod.py
   - Print jumlah data SEBELUM dan SESUDAH
   - Simpan boxplot ke `outlier_forest.png`

4. PISAHKAN KOLOM berdasarkan tipe:
   - Kolom kategorikal: Seasonality, Weather Condition, Region, Store ID, Category, Product ID
   - Kolom numerikal: Month, Year, Week, DayOfWeek, IsWeekend, Price, Discount, Promotion, Inventory Level, Competitor Pricing, Epidemic

5. PIPELINE dengan sklearn:
   - ColumnTransformer:
     - OneHotEncoder(handle_unknown='ignore') untuk kolom kategorikal
     - StandardScaler() untuk kolom numerikal
   - RandomForestRegressor(random_state=42) sebagai estimator
   - Pipeline: [('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]

6. TRAIN-TEST SPLIT: test_size=0.2, random_state=42

7. HYPERPARAMETER TUNING menggunakan GridSearchCV:
   - param_grid = {
       'model__n_estimators': [100, 200, 300],
       'model__max_depth': [10, 20, None],
       'model__min_samples_split': [2, 5],
       'model__min_samples_leaf': [1, 2]
     }
   - cv=3, scoring='r2', n_jobs=-1, verbose=1
   - Print: "Best Parameters: ..."
   - Print: "Best CV R² Score: ..."
   - Gunakan best_estimator_ untuk evaluasi selanjutnya

8. EVALUASI pada X_test, y_test (pakai best model):
   - R² Score, MSE, RMSE, MAE, MAPE
   - Print format rapi BAHASA INDONESIA (sama format seperti LINEAR_mod.py tapi judul "EVALUASI MODEL RANDOM FOREST")

9. VISUALISASI (simpan semua sebagai file gambar):
   a. Feature Importance — horizontal bar chart top 15 (`feature_importance_forest.png`)
      - Ambil feature names dari preprocessor: pipeline.named_steps['preprocessor'].get_feature_names_out()
      - Sort descending, ambil top 15
      - Title: "Top 15 Feature Importance — Random Forest"
   b. Scatter plot: Actual vs Predicted (`actual_vs_pred_forest.png`)
   c. Residual distribution histogram (`residual_dist_forest.png`)
   d. Perbandingan Demand aktual vs prediksi per Category — grouped bar chart (`demand_per_category.png`)

10. CONFIDENCE INTERVAL 95%:
    - Hitung dari individual tree predictions:
      all_preds = np.array([tree.predict(X_test_transformed) for tree in best_model.named_steps['model'].estimators_])
      mean_pred = all_preds.mean(axis=0)
      std_pred = all_preds.std(axis=0)
      ci_lower = mean_pred - 1.96 * std_pred
      ci_upper = mean_pred + 1.96 * std_pred
    - Print contoh 5 prediksi + CI

11. YoY Analysis: rata-rata Demand per Year + growth %

12. SAVE MODEL: joblib.dump(best_pipeline, 'forest_model.pkl')
    - Print konfirmasi

CATATAN PERFORMA:
- Karena dataset 76K rows dan GridSearchCV dengan banyak kombinasi, tambahkan:
  - print("⏳ Proses training Random Forest + GridSearchCV dimulai...")
  - print("⏳ Estimasi waktu: 5-15 menit tergantung spesifikasi komputer")
  - Setelah selesai: print(f"✅ Training selesai dalam {elapsed_time:.1f} detik")

Semua komentar dan print statement dalam BAHASA INDONESIA.
Library: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, time
```

---

## BAGIAN 4: CLUSTERING PRODUK (clustering_mod.py)

```
Buatkan file `clustering_mod.py` — script standalone untuk clustering produk.

PENTING: Gunakan `sales_data.csv` langsung.

SPESIFIKASI:

1. LOAD DATA dari `sales_data.csv`

2. AGREGASI DATA per Product ID:
   - `avg_demand` = rata-rata kolom `Demand`
   - `avg_price` = rata-rata kolom `Price`
   - `total_units_sold` = sum kolom `Units Sold`
   - `avg_inventory` = rata-rata kolom `Inventory Level`
   - `avg_discount` = rata-rata kolom `Discount`
   - `promo_rate` = rata-rata kolom `Promotion` (persentase promo)
   - Tambahkan kolom `Category` (ambil mode/yang paling sering muncul per Product ID)
   - Hasil: DataFrame 20 baris (1 per produk)

3. FITUR UNTUK CLUSTERING: avg_demand, avg_price, total_units_sold, avg_inventory, avg_discount, promo_rate
   - SCALING dengan StandardScaler

4. ELBOW METHOD:
   - Test k = 2 sampai 10
   - Plot inertia vs k
   - Simpan ke `elbow_method.png`

5. SILHOUETTE SCORE:
   - Hitung silhouette_score untuk setiap k (2-10)
   - Print k dengan silhouette tertinggi → gunakan sebagai k_optimal
   - Plot silhouette vs k
   - Simpan ke `silhouette_score.png`

6. FIT KMEANS dengan k_optimal:
   - KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
   - Tambahkan kolom `Cluster` ke DataFrame agregasi

7. VISUALISASI CLUSTER (`cluster_scatter.png`):
   - Scatter plot: x = avg_price, y = avg_demand
   - Warna = Cluster
   - Setiap titik diberi label `Product ID`
   - Title: "Clustering Produk — Harga vs Demand"
   - Legend cluster

8. INTERPRETASI CLUSTER OTOMATIS:
   - Untuk setiap cluster, hitung median avg_demand dan median avg_price
   - Bandingkan dengan median keseluruhan:
     - demand tinggi + harga rendah → "🟢 Produk Laris Ekonomis"
     - demand tinggi + harga tinggi → "🔵 Produk Premium Laku"
     - demand rendah + harga tinggi → "🟡 Produk Premium Lambat"
     - demand rendah + harga rendah → "🔴 Produk Kurang Diminati"
   - Print tabel interpretasi lengkap
   - Simpan hasil clustering ke `clustering_result.csv`

9. SAVE scaler dan KMeans model:
   - joblib.dump(scaler, 'cluster_scaler.pkl')
   - joblib.dump(kmeans, 'cluster_model.pkl')

Semua komentar dan print statement dalam BAHASA INDONESIA.
Library: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
```

---

## BAGIAN 5: DASHBOARD STREAMLIT (app.py)

```
Buatkan file `app.py` — dashboard Streamlit lengkap untuk proyek Demand Forecasting.

STRUKTUR HALAMAN: Gunakan st.sidebar untuk navigasi dengan radio button.
Menu: ["📊 Overview", "📈 Trend Penjualan", "🧩 Clustering Produk", "📉 Evaluasi Model", "🔮 Prediksi Demand", "📦 Rekomendasi Stok"]

KONFIGURASI AWAL:
- st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide", page_icon="📦")
- Load dataset: sales_data.csv
- Load models: linear_model.pkl, forest_model.pkl, cluster_model.pkl, cluster_scaler.pkl
- Gunakan try-except saat load model, tampilkan st.warning() jika model belum ada

SIDEBAR FILTERS (berlaku global):
- Filter Category: multiselect dari df['Category'].unique()
- Filter Region: multiselect dari df['Region'].unique()
- Filter Date Range: date_input (start, end)
- Filter Store ID: multiselect
- Terapkan filter ke dataframe utama

=== HALAMAN 1: 📊 Overview ===
- st.title("📦 Dashboard Prediksi Stok & Demand Forecasting")
- ROW 1 — 4 kolom metric cards (st.metric):
  Total Transaksi | Total Units Sold | Rata-rata Demand | Total Revenue
- ROW 2 — 2 kolom:
  Kiri: Bar chart Total Units Sold per Category
  Kanan: Pie chart Distribusi Demand per Region
- ROW 3: Tabel ringkasan per Category (count, mean demand, mean price, total units sold)

=== HALAMAN 2: 📈 Trend Penjualan ===
- Line chart: rata-rata Demand harian + rolling average 7 hari
- Multi-line chart: trend per Category
- Heatmap: rata-rata Demand per Month x Year
- Bar chart: rata-rata Demand per Seasonality & per Weather Condition

=== HALAMAN 3: 🧩 Clustering Produk ===
- Load clustering_result.csv (jika ada)
- Scatter plot: avg_price vs avg_demand (warna per cluster, label Product ID)
- Tabel produk per cluster + interpretasi
- Elbow & Silhouette chart

=== HALAMAN 4: 📉 Evaluasi Model ===
- Tabel perbandingan metrik: Linear vs Random Forest (R², RMSE, MAE, MAPE)
- Load visualisasi dari file gambar (.png)
- Rekomendasi model terbaik berdasarkan R²
- Feature importance chart (Random Forest)

=== HALAMAN 5: 🔮 Prediksi Demand ===
- Form input: Category, Product ID, Store ID, Region, Month, Price, Discount, Promotion, Weather, Seasonality, Inventory Level, Competitor Pricing, Epidemic
- Prediksi pakai KEDUA model (linear + forest)
- Tampilkan hasil + Confidence Interval 95%

=== HALAMAN 6: 📦 Rekomendasi Stok ===
- Input: Lead Time (hari), Service Level (90/95/99%)
- Hitung per Product ID: Safety Stock, Reorder Point, Recommended Order Qty
- Tabel output dengan highlight merah jika perlu restock segera
- Bar chart top 10 produk paling butuh restock

STYLING: Custom CSS metric cards, label Bahasa Indonesia, tema konsisten.
```

---

## BAGIAN 6: DEPLOYMENT (run_app.py)

```
Buatkan file `run_app.py`:

import subprocess, threading, time

def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py",
        "--server.port=8501", "--server.headless=true", "--server.address=0.0.0.0"])

if __name__ == "__main__":
    threading.Thread(target=run_streamlit).start()
    time.sleep(5)
    try:
        from pyngrok import ngrok
        url = ngrok.connect(8501)
        print(f"🚀 DASHBOARD ONLINE! URL: {url}")
    except ImportError:
        print("🚀 Streamlit berjalan di http://localhost:8501")
```

---

## BAGIAN 7: REQUIREMENTS (requirements.txt)

```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
joblib==1.3.2
pyngrok==7.0.0
openpyxl==3.1.2
```

---

## BAGIAN 8: STRUKTUR FOLDER FINAL

```
📁 TUBES_DAMIN_2025/
├── 📄 sales_data.csv                  ← Dataset ASLI (SUDAH ADA)
├── 📄 sales_data_enriched.csv         ← Hasil feature_engineering.py
├── 📄 clustering_result.csv           ← Hasil clustering_mod.py
├── 🐍 feature_engineering.py
├── 🐍 LINEAR_mod.py                   ← Linear Regression (Clothing only)
├── 🐍 FOREST_mod.py                   ← Random Forest (semua kategori)
├── 🐍 clustering_mod.py               ← KMeans clustering
├── 🐍 app.py                          ← Dashboard Streamlit (6 halaman)
├── 🐍 run_app.py                      ← Deployment (ngrok)
├── 📄 requirements.txt
├── 🤖 linear_model.pkl / forest_model.pkl / cluster_model.pkl / cluster_scaler.pkl
└── 🖼️ *.png                           ← Semua visualisasi output
```

---

## URUTAN EKSEKUSI

```
1️⃣  pip install -r requirements.txt
2️⃣  python feature_engineering.py
3️⃣  python LINEAR_mod.py
4️⃣  python FOREST_mod.py                 (estimasi 5-15 menit)
5️⃣  python clustering_mod.py
6️⃣  streamlit run app.py   ATAU   python run_app.py
```

---

## ⚠️ CATATAN SINKRONISASI NAMA KOLOM

SEMUA nama kolom di setiap script HARUS EXACT MATCH dengan dataset:
- ✅ `Units Sold` — BUKAN `units_sold`
- ✅ `Inventory Level` — BUKAN `inventory_level`
- ✅ `Weather Condition` — BUKAN `weather`
- ✅ `Competitor Pricing` — BUKAN `competitor_price`
- ✅ `Store ID` — BUKAN `store_id`
- ✅ `Product ID` — BUKAN `product_id`
- ✅ `Demand` sebagai TARGET variable

Struktur proyek ini IDENTIK dengan TUBES_DAMIN_2025 (Salary Analysis Dashboard) — sama struktur, beda domain.

---

## ✅ SELESAI — COPY-PASTE SELURUH ISI FILE INI KE AI UNTUK GENERATE PROYEK LENGKAP
